from __future__ import annotations

import json
import os
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .approval_store import ApprovalRecord, ApprovalStore, default_approver_id
from .config import load_project_env


load_project_env()


def default_approval_api_host() -> str:
    """读取本地审批 API host。"""

    return os.getenv("BEGINNER_AGENT_APPROVAL_API_HOST", "127.0.0.1").strip()


def default_approval_api_port() -> int:
    """读取本地审批 API port。"""

    raw = os.getenv("BEGINNER_AGENT_APPROVAL_API_PORT", "8765").strip()
    try:
        return int(raw)
    except ValueError:
        return 8765


def _record_to_dict(record: ApprovalRecord) -> dict[str, Any]:
    """把 ApprovalRecord 转成 HTTP JSON。"""

    return {
        "approval_id": record.approval_id,
        "thread_id": record.thread_id,
        "task_id": record.task_id,
        "status": record.status,
        "approver_id": record.approver_id,
        "requested_at": record.requested_at,
        "expires_at": record.expires_at,
        "decided_at": record.decided_at,
        "payload": record.payload,
        "reason": record.reason,
        "modified_tool_args": record.modified_tool_args,
    }


class ApprovalHTTPRequestHandler(BaseHTTPRequestHandler):
    """本地审批 HTTP API。

    中文注释：
    这是一个最小可用 API，不引入 FastAPI/uvicorn。
    它适合本地学习和内网工具：

    - GET  /approvals
    - GET  /approvals/{approval_id}
    - GET  /approvals/{approval_id}/audit
    - POST /approvals/{approval_id}/decision

    POST body 示例：

        {
          "approved": true,
          "approver_id": "christopher",
          "reason": "确认可以执行",
          "modified_tool_args": {"path": "graph.py"}
        }
    """

    store: ApprovalStore

    def _json_response(self, status_code: int, body: dict[str, Any] | list[Any]) -> None:
        """返回 JSON 响应。"""

        data = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html_response(self, status_code: int, body: str) -> None:
        """返回极简 HTML 响应。"""

        data = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        """读取 JSON 请求体。"""

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _read_form_body(self) -> dict[str, str]:
        """读取 HTML form 请求体。"""

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(raw)
        return {key: values[0] for key, values in parsed.items()}

    def _render_index(self) -> str:
        """渲染一个够本地使用的审批网页。"""

        records = self.store.list(status="pending", limit=50)
        items: list[str] = []
        for record in records:
            payload = record.payload
            tool_args = json.dumps(payload.get("tool_args", {}), ensure_ascii=False, indent=2)
            items.append(
                f"""
                <section>
                  <h2>{escape(record.approval_id)}</h2>
                  <p>任务：{escape(record.task_id)}</p>
                  <p>工具：{escape(str(payload.get("tool_name", "")))}</p>
                  <p>风险：{escape(str(payload.get("risk_level", "")))}</p>
                  <p>原因：{escape(str(payload.get("reason", "")))}</p>
                  <pre>{escape(tool_args)}</pre>
                  <form method="post" action="/approvals/{escape(record.approval_id)}/decision">
                    <input name="approver_id" placeholder="approver_id" />
                    <input name="reason" placeholder="reason" />
                    <textarea name="modified_tool_args" rows="4"
                      placeholder='{{"path": "graph.py"}}'></textarea>
                    <button name="approved" value="true">Approve</button>
                    <button name="approved" value="false">Deny</button>
                  </form>
                </section>
                """
            )
        body = "\n".join(items) or "<p>暂无 pending 审批。</p>"
        return f"""
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8" />
            <title>Beginner Agent Approvals</title>
            <style>
              body {{ font-family: sans-serif; max-width: 960px; margin: 32px auto; }}
              section {{ border: 1px solid #ddd; padding: 16px; margin: 16px 0; }}
              textarea, input {{ display: block; width: 100%; margin: 8px 0; }}
              button {{ margin-right: 8px; }}
            </style>
          </head>
          <body>
            <h1>Beginner Agent Approvals</h1>
            {body}
          </body>
        </html>
        """

    def do_GET(self) -> None:  # noqa: N802
        """处理查询审批请求。"""

        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self._html_response(200, self._render_index())
            return

        if parsed.path == "/health":
            self._json_response(200, {"status": "ok"})
            return

        if parts == ["approvals"]:
            status = query.get("status", [None])[0]
            limit = int(query.get("limit", ["20"])[0])
            records = self.store.list(status=status, limit=limit)
            self._json_response(200, [_record_to_dict(record) for record in records])
            return

        if len(parts) == 2 and parts[0] == "approvals":
            record = self.store.get(parts[1])
            if record is None:
                self._json_response(404, {"error": "approval not found"})
                return
            self._json_response(200, _record_to_dict(record))
            return

        if len(parts) == 3 and parts[0] == "approvals" and parts[2] == "audit":
            self._json_response(200, self.store.audit_events(parts[1]))
            return

        self._json_response(
            404,
            {
                "error": "not found",
                "routes": [
                    "GET /health",
                    "GET /approvals",
                    "GET /approvals/{approval_id}",
                    "GET /approvals/{approval_id}/audit",
                    "POST /approvals/{approval_id}/decision",
                ],
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        """处理审批决定。"""

        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 3 or parts[0] != "approvals" or parts[2] != "decision":
            self._json_response(404, {"error": "not found"})
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                body = self._read_json_body()
            else:
                form = self._read_form_body()
                modified_raw = form.get("modified_tool_args", "").strip()
                body = {
                    "approved": form.get("approved") == "true",
                    "approver_id": form.get("approver_id", ""),
                    "reason": form.get("reason", ""),
                    "modified_tool_args": json.loads(modified_raw) if modified_raw else {},
                }
            record = self.store.decide(
                parts[1],
                approved=bool(body.get("approved", False)),
                approver_id=str(body.get("approver_id") or default_approver_id()),
                reason=str(body.get("reason", "API 审批。")),
                modified_tool_args=dict(body.get("modified_tool_args") or {}),
            )
        except KeyError as exc:
            self._json_response(404, {"error": str(exc)})
            return
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json_response(400, {"error": str(exc)})
            return

        if "application/json" in self.headers.get("Content-Type", ""):
            self._json_response(200, _record_to_dict(record))
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """降低默认 HTTP server 日志噪音。"""

        return


def run_approval_server(
    *,
    host: str | None = None,
    port: int | None = None,
    store: ApprovalStore | None = None,
) -> None:
    """启动本地审批 API 服务。"""

    server_host = host or default_approval_api_host()
    server_port = port or default_approval_api_port()
    ApprovalHTTPRequestHandler.store = store or ApprovalStore()
    server = ThreadingHTTPServer((server_host, server_port), ApprovalHTTPRequestHandler)
    print(f"Approval API listening on http://{server_host}:{server_port}")
    print("按 Ctrl+C 停止。")
    server.serve_forever()
