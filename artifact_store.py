from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_DIR, load_project_env


load_project_env()


def artifact_root() -> Path:
    """返回 artifact 存储根目录。

    中文注释：
    这是真正的本地 artifact storage，不再只是把 artifact_report 放在 State 里。
    当前先用文件系统目录。
    后续可以替换成 S3、MinIO、Postgres、大厂内部对象存储。
    """

    configured = os.getenv("BEGINNER_AGENT_ARTIFACT_DIR", ".agent_state/artifacts").strip()
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _safe_run_id(run_id: str) -> str:
    """把 run_id 变成安全目录名。"""

    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in run_id)


def write_artifact_manifest(
    *,
    run_id: str,
    report: dict[str, Any],
    state_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """写入 artifact manifest。

    中文注释：
    manifest 是“本轮 agent 产物目录”：
    - 哪些文件被修改。
    - 有多少 patch。
    - 有多少执行尝试。
    - 哪些验证任务产生了结果。

    注意：
    它不是完整 checkpoint，也不保存全部 messages。
    它只保存交付物索引，方便最终报告、审计、后续恢复查看。
    """

    root = artifact_root()
    run_dir = root / _safe_run_id(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    index_path = root / "index.jsonl"
    saved_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "run_id": run_id,
        "saved_at": saved_at,
        "report": report,
        "state_snapshot": state_snapshot,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with index_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "saved_at": saved_at,
                    "manifest_path": manifest_path.as_posix(),
                    "changed_files": report.get("changed_files", []),
                    "patch_count": report.get("patch_count", 0),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return {
        "storage": "filesystem",
        "artifact_root": root.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "index_path": index_path.as_posix(),
        "saved_at": saved_at,
    }
