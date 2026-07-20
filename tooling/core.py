from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# 中文注释：
# core.py 放所有工具共享的安全边界和小工具函数。
# 其他工具文件只通过这里访问路径，避免各自手写路径安全逻辑。
# 你可以把它理解成“工具系统的地基”。
TOOL_ROOT = Path(__file__).resolve().parents[1]
# 中文注释：
# REPO_ROOT 是 langgraph 仓库根目录。
# git_status / git_diff 这类工具要在仓库根目录下执行 git 命令。
REPO_ROOT = TOOL_ROOT.parent
# 中文注释：
# WORKSPACE_ROOT 是本地 aios 工作区根目录。
# 可管理工具平台允许注册的项目根必须在这个目录内。
WORKSPACE_ROOT = REPO_ROOT.parent
# 中文注释：
# STATE_DIR 是 agent 自己保存临时状态的目录。
# 例如 PatchPlan、checkpoint 这类数据会放到这里。
# 注意：这个目录仍然在 beginner_agent 内部，不会写到项目外面。
STATE_DIR = TOOL_ROOT / ".agent_state"
PATCH_PLAN_DIR = STATE_DIR / "patch_plans"
CHECKPOINT_DIR = STATE_DIR / "checkpoints"
PROJECT_ROOTS_FILE = STATE_DIR / "project_roots.json"
ACTIVE_PROJECT_FILE = STATE_DIR / "active_project.json"
DEFAULT_PROJECT_ID = "beginner_agent"

# 中文注释：
# 只允许工具读取/写入这些文本文件类型。
# 不允许二进制文件，是为了避免误读图片、模型文件、数据库等内容。
ALLOWED_SUFFIXES = {".py", ".rs", ".toml", ".md", ".txt", ".json"}

# 中文注释：
# 这些 MAX_* 常量是安全阀。
# Agent 工具不能一次返回无限内容，否则会撑爆 LLM 上下文。
MAX_READ_CHARS = 6000
MAX_SEARCH_RESULTS = 40
MAX_TOOL_OUTPUT_CHARS = 8000
MAX_PATCH_CHARS = 12000

# 中文注释：
# 遍历目录时忽略这些目录。
# 例如 __pycache__ 是 Python 缓存，不是源码；.agent_state 是工具运行状态。
IGNORED_DIRS = {
    "__pycache__",
    ".agent_state",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def ensure_state_dirs() -> None:
    """创建 agent 内部状态目录。"""

    # 中文注释：
    # mkdir(..., parents=True, exist_ok=True) 的意思是：
    # - parents=True：父目录不存在就一起创建。
    # - exist_ok=True：目录已存在也不报错。
    PATCH_PLAN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _default_project_roots() -> dict[str, str]:
    """返回内置项目根。"""

    return {
        DEFAULT_PROJECT_ID: TOOL_ROOT.as_posix(),
    }


def _read_project_roots() -> dict[str, str]:
    """读取已注册项目根。"""

    ensure_state_dirs()
    if not PROJECT_ROOTS_FILE.exists():
        PROJECT_ROOTS_FILE.write_text(json_dumps(_default_project_roots()), encoding="utf-8")
    try:
        data = json.loads(PROJECT_ROOTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = _default_project_roots()
    roots = _default_project_roots()
    if isinstance(data, dict):
        roots.update({str(key): str(value) for key, value in data.items()})
    return roots


def list_project_roots() -> dict[str, str]:
    """列出平台已注册项目根。"""

    return _read_project_roots()


def register_project_root(project_id: str, absolute_path: str) -> dict[str, str]:
    """注册一个受控项目根。

    中文注释：
    这让工具平台可以管理多个项目。
    但为了安全，项目根必须满足：
    - project_id 只能是简单名字。
    - absolute_path 必须已经存在。
    - absolute_path 必须位于 WORKSPACE_ROOT 内部。
    """

    normalized_id = project_id.strip()
    if not normalized_id or not normalized_id.replace("_", "-").replace("-", "").isalnum():
        raise ValueError("project_id 只能包含字母、数字、下划线和中划线。")
    root = Path(absolute_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"项目根不存在或不是目录：{absolute_path}")
    try:
        root.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("项目根必须位于 /Users/christophermanning/Downloads/aios 工作区内。") from exc
    roots = _read_project_roots()
    roots[normalized_id] = root.as_posix()
    PROJECT_ROOTS_FILE.write_text(json_dumps(roots), encoding="utf-8")
    return roots


def active_project_id() -> str:
    """读取当前 active project id。"""

    ensure_state_dirs()
    if not ACTIVE_PROJECT_FILE.exists():
        ACTIVE_PROJECT_FILE.write_text(json_dumps({"project_id": DEFAULT_PROJECT_ID}), encoding="utf-8")
    try:
        data = json.loads(ACTIVE_PROJECT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {"project_id": DEFAULT_PROJECT_ID}
    project_id = str(data.get("project_id") or DEFAULT_PROJECT_ID)
    roots = _read_project_roots()
    if project_id not in roots:
        return DEFAULT_PROJECT_ID
    return project_id


def set_active_project(project_id: str) -> str:
    """切换当前 active project。"""

    roots = _read_project_roots()
    if project_id not in roots:
        raise ValueError(f"未注册项目：{project_id}")
    ACTIVE_PROJECT_FILE.write_text(json_dumps({"project_id": project_id}), encoding="utf-8")
    return project_id


def active_project_root() -> Path:
    """返回当前 active project 的根目录。"""

    roots = _read_project_roots()
    return Path(roots[active_project_id()]).resolve()


def safe_resolve(relative_path: str) -> Path:
    """把相对路径解析到当前 active project 内部。

    中文注释：
    所有工具都必须经过这个函数。
    它会拒绝绝对路径和路径穿越，防止工具访问项目外部文件。
    """

    requested = Path(relative_path)

    # 中文注释：
    # 绝对路径类似：
    #   /Users/xxx/.ssh/id_rsa
    #
    # 这种路径可能访问到用户电脑上的敏感文件，所以直接拒绝。
    if requested.is_absolute():
        raise ValueError("工具拒绝绝对路径，只允许 beginner_agent 内的相对路径。")

    # 中文注释：
    # resolve() 会把路径规范化。
    # 例如：
    #   "a/../b"
    # 会变成真实的 b 路径。
    project_root = active_project_root()
    resolved = (project_root / requested).resolve()
    try:
        # 中文注释：
        # relative_to(project_root) 是核心安全检查。
        # 如果 resolved 不在当前 active project 目录内，这里会抛 ValueError。
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("工具拒绝访问当前 active project 目录之外的路径。") from exc
    return resolved


def safe_text_file(relative_path: str, *, must_exist: bool = True) -> Path:
    """解析并确认目标是允许处理的文本文件。"""

    file_path = safe_resolve(relative_path)

    # 中文注释：
    # must_exist=True 用在 read_file / apply_patch 这类必须操作已有文件的场景。
    if must_exist and not file_path.exists():
        raise ValueError(f"文件不存在：{relative_path}")
    if must_exist and not file_path.is_file():
        raise ValueError(f"不是文件：{relative_path}")
    if file_path.suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"不允许处理这种文件类型：{file_path.suffix}")
    return file_path


def read_text_for_snapshot(relative_path: str) -> str:
    """Executor 用它在写入前后记录快照。"""

    # 中文注释：
    # apply_patch 前后都要读取文件内容。
    # 这样 patch_history 里可以保存 before_content / after_content，
    # rollback 才知道如何恢复。
    return safe_text_file(relative_path).read_text(encoding="utf-8", errors="replace")


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """限制工具输出长度，避免把太多内容塞进 LLM 上下文。"""

    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...工具输出过长，已截断..."


def project_text_files() -> tuple[str, ...]:
    """递归返回当前 active project 中允许读取的文本文件。"""

    files: list[str] = []
    project_root = active_project_root()

    # 中文注释：
    # rglob("*") 会递归遍历当前 active project 下的所有文件和目录。
    # 这里配合 IGNORED_DIRS 和 ALLOWED_SUFFIXES 做过滤。
    for path in sorted(project_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(project_root).parts):
            continue
        if path.suffix in ALLOWED_SUFFIXES:
            files.append(path.relative_to(project_root).as_posix())
    return tuple(files)


def json_dumps(data: Any) -> str:
    """统一 JSON 输出格式，保留中文。"""

    return json.dumps(data, ensure_ascii=False, indent=2)
