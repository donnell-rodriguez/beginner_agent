from __future__ import annotations

from pathlib import Path
from shutil import which

from .core import REPO_ROOT, active_project_root, json_dumps


# 中文注释：
# environment_tools.py 负责“识别项目环境”。
#
# 生产级 code agent 在运行测试前，不能拍脑袋决定用 pytest / npm / cargo。
# 它要先观察项目里有哪些配置文件、有哪些命令可用，再选择验证方式。


def _exists_at_repo(path: str) -> bool:
    """检查 langgraph 仓库根目录下是否存在某个文件。"""

    return (REPO_ROOT / path).exists()


def _exists_at_active_root(path: str) -> bool:
    """检查 active project 目录下是否存在某个文件。"""

    return (active_project_root() / path).exists()


def _repo_matches(pattern: str) -> list[str]:
    """在仓库内按模式查找配置文件，返回相对路径。"""

    return sorted(path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.glob(pattern))


def detect_project_stack_tool() -> str:
    """识别当前项目技术栈和可用工具。"""

    package_managers: list[str] = []
    languages: list[str] = []
    test_frameworks: list[str] = []
    build_systems: list[str] = []

    pyprojects = _repo_matches("**/pyproject.toml")
    uv_locks = _repo_matches("**/uv.lock")
    package_jsons = _repo_matches("**/package.json")
    cargo_tomls = _repo_matches("**/Cargo.toml")
    project_root = active_project_root()
    active_root_cargo = _exists_at_active_root("Cargo.toml")

    if _exists_at_repo("pyproject.toml") or _exists_at_active_root("pyproject.toml") or pyprojects:
        languages.append("python")
        build_systems.append("pyproject")
    if _exists_at_repo("uv.lock") or uv_locks:
        package_managers.append("uv")
    if _exists_at_repo("package.json") or _exists_at_active_root("package.json") or package_jsons:
        languages.append("node")
        package_managers.append("npm")
    if _exists_at_repo("Cargo.toml") or active_root_cargo or cargo_tomls:
        languages.append("rust")
        package_managers.append("cargo")

    if any(Path(REPO_ROOT, item).exists() for item in ("pytest.ini", "tox.ini")):
        test_frameworks.append("pytest")
    if "python" in languages:
        test_frameworks.append("python_compileall")
    if "rust" in languages:
        test_frameworks.append("cargo_test")
    if "node" in languages:
        test_frameworks.append("npm_test")

    executables = {
        "python3": which("python3"),
        "uv": which("uv"),
        "pytest": which("pytest"),
        "ruff": which("ruff"),
        "mypy": which("mypy"),
        "npm": which("npm"),
        "cargo": which("cargo"),
    }

    return json_dumps(
        {
            "languages": sorted(set(languages)),
            "package_managers": sorted(set(package_managers)),
            "test_frameworks": sorted(set(test_frameworks)),
            "build_systems": sorted(set(build_systems)),
            "executables": {name: bool(path) for name, path in executables.items()},
            "repo_root": REPO_ROOT.as_posix(),
            "active_project_root": project_root.as_posix(),
            "active_project_languages": {
                "python": bool(list(project_root.glob("**/*.py"))),
                "rust": active_root_cargo or bool(list(project_root.glob("**/*.rs"))),
            },
            "detected_files": {
                "pyproject": pyprojects[:20],
                "uv_lock": uv_locks[:20],
                "package_json": package_jsons[:20],
                "cargo_toml": cargo_tomls[:20],
            },
        }
    )


def detect_test_framework_tool() -> str:
    """只返回测试框架判断，给 Planner 快速选择测试工具。"""

    stack = detect_project_stack_tool()
    return stack


def detect_package_manager_tool() -> str:
    """只返回包管理器判断。"""

    stack = detect_project_stack_tool()
    return stack


def detect_build_system_tool() -> str:
    """只返回构建系统判断。"""

    stack = detect_project_stack_tool()
    return stack
