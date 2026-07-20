from __future__ import annotations

from .core import (
    active_project_id,
    active_project_root,
    json_dumps,
    list_project_roots,
    register_project_root,
    set_active_project,
)


# 中文注释：
# platform_tools.py 放“可管理工具平台”的管理接口。
#
# 大厂 code agent 不只是有一堆函数。
# 它通常还需要知道：
# - 当前有哪些工具？
# - 每个工具是什么风险？
# - 当前 active project 是谁？
# - 允许管理哪些项目根？
# - 写工具是否需要审批？
#
# 这就是工具从“工具箱”升级成“工具平台”的关键。
#
# 注意：
# 工具目录、工具描述和权限策略报告已经迁移到 registry.py。
# 那里直接读取 ToolSpec，避免再根据工具名猜测元数据。


def list_project_roots_tool() -> str:
    """列出已注册项目根。"""

    return json_dumps(
        {
            "active_project_id": active_project_id(),
            "active_project_root": active_project_root().as_posix(),
            "roots": list_project_roots(),
        }
    )


def register_project_root_tool(project_id: str, path: str) -> str:
    """注册一个项目根。"""

    roots = register_project_root(project_id, path)
    return json_dumps({"status": "success", "project_id": project_id, "roots": roots})


def set_active_project_tool(project_id: str) -> str:
    """切换 active project。"""

    selected = set_active_project(project_id)
    return json_dumps(
        {
            "status": "success",
            "active_project_id": selected,
            "active_project_root": active_project_root().as_posix(),
        }
    )


def get_active_project_tool() -> str:
    """查看当前 active project。"""

    return json_dumps(
        {
            "active_project_id": active_project_id(),
            "active_project_root": active_project_root().as_posix(),
        }
    )
