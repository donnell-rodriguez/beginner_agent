from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .core import REPO_ROOT, STATE_DIR, active_project_root, ensure_state_dirs, json_dumps


RuntimeName = Literal["python", "rust", "generic"]
CwdMode = Literal["repo_root", "active_project"]


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """受控运行环境配置。

    中文注释：
    Executor 本身不应该关心 Python / Rust 具体怎么运行。
    它只执行工具。

    真正的运行环境应该在 runtime 层声明：
    - 用哪个语言 runtime。
    - 默认在哪个目录运行。
    - 需要哪些隔离环境变量。
    - cache / build 产物写到哪里。

    这更接近大厂 code agent 的做法：
    Agent 不直接拼 shell 命令，而是选择一个受控 runtime profile。
    """

    name: RuntimeName
    cwd_mode: CwdMode
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""


RUNTIME_SPECS: dict[RuntimeName, RuntimeSpec] = {
    "python": RuntimeSpec(
        name="python",
        cwd_mode="repo_root",
        env={
            # 中文注释：
            # 防止 Python 在源码目录里到处写 __pycache__。
            "PYTHONDONTWRITEBYTECODE": "1",
            # 中文注释：
            # 把 uv cache 放进 agent 自己的状态目录，而不是写用户全局 ~/.cache/uv。
            "UV_CACHE_DIR": os.getenv(
                "BEGINNER_AGENT_UV_CACHE_DIR",
                (STATE_DIR / "uv-cache").as_posix(),
            ),
        },
        description="Python/uv runtime，用于 py_compile、pytest、ruff、mypy、LangGraph 图构建。",
    ),
    "rust": RuntimeSpec(
        name="rust",
        cwd_mode="active_project",
        env={
            # 中文注释：
            # 把 Cargo home / target dir 放进 agent 状态目录，避免污染用户全局 ~/.cargo。
            "CARGO_HOME": os.getenv(
                "BEGINNER_AGENT_CARGO_HOME",
                (STATE_DIR / "cargo-home").as_posix(),
            ),
            "CARGO_TARGET_DIR": os.getenv(
                "BEGINNER_AGENT_CARGO_TARGET_DIR",
                (STATE_DIR / "cargo-target").as_posix(),
            ),
            "CARGO_TERM_COLOR": "never",
            "RUST_BACKTRACE": "1",
        },
        description="Rust/Cargo runtime，用于 cargo check/test/clippy/fmt。",
    ),
    "generic": RuntimeSpec(
        name="generic",
        cwd_mode="active_project",
        env={},
        description="通用只读/轻量命令 runtime。",
    ),
}


def runtime_cwd(runtime_name: RuntimeName, cwd_override: str | Path | None = None) -> Path:
    """解析 runtime 的工作目录。"""

    if cwd_override == "active_project":
        return active_project_root()
    if cwd_override == "repo_root":
        return REPO_ROOT
    if isinstance(cwd_override, Path):
        return cwd_override
    if isinstance(cwd_override, str) and cwd_override:
        return Path(cwd_override)

    spec = RUNTIME_SPECS[runtime_name]
    return REPO_ROOT if spec.cwd_mode == "repo_root" else active_project_root()


def runtime_env(runtime_name: RuntimeName, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """生成受控 runtime 环境变量。

    中文注释：
    这里会基于 os.environ 拷贝一份环境变量，然后覆盖受控字段。
    这样命令仍然能找到 PATH，但 cache/build 产物会落到 agent 状态目录。
    """

    ensure_state_dirs()
    spec = RUNTIME_SPECS[runtime_name]
    env = {**os.environ, **spec.env}
    if extra_env:
        env.update(extra_env)
    return env


def infer_runtime_name(command_spec: dict[str, Any]) -> RuntimeName:
    """根据命令配置推断 runtime。"""

    configured = str(command_spec.get("runtime", "")).strip()
    if configured in RUNTIME_SPECS:
        return configured  # type: ignore[return-value]
    cmd = list(command_spec.get("cmd", []))
    executable = str(cmd[0]) if cmd else ""
    if executable in {"cargo", "rustc"}:
        return "rust"
    if executable in {"python", "python3", "uv", "pytest", "ruff", "mypy"}:
        return "python"
    return "generic"


def resolve_command_runtime(command_spec: dict[str, Any]) -> dict[str, Any]:
    """把命令 profile 解析成 cwd/env/runtime 元数据。"""

    runtime_name = infer_runtime_name(command_spec)
    cwd = runtime_cwd(runtime_name, command_spec.get("cwd"))
    env = runtime_env(runtime_name, command_spec.get("env"))
    return {
        "runtime": runtime_name,
        "cwd": cwd,
        "env": env,
        "runtime_spec": RUNTIME_SPECS[runtime_name],
    }


def runtime_catalog() -> dict[str, dict[str, Any]]:
    """输出当前支持的 runtime 配置。"""

    return {
        name: {
            "cwd_mode": spec.cwd_mode,
            "env": spec.env,
            "description": spec.description,
        }
        for name, spec in RUNTIME_SPECS.items()
    }


def runtime_catalog_json() -> str:
    """输出当前支持的 runtime 配置，方便手机阅读和工具审计。"""

    return json_dumps(runtime_catalog())
