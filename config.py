from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"

_ENV_LOADED = False


def _strip_env_value(value: str) -> str:
    """清理 .env 里的 value。

    中文注释：
    .env 文件里常见写法是：

        DATABASE_URL=postgresql://...
        OMLX_API_KEY="local-omlx-key"

    这里做最基础的解析：
    - 去掉左右空格。
    - 去掉成对的单引号或双引号。

    这个项目暂时不引入 python-dotenv 依赖，避免为了一个小功能增加包耦合。
    后续如果要支持更复杂的 .env 语法，可以把这里替换成 python-dotenv。
    """

    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def load_project_env(env_path: str | Path | None = None, *, override: bool = False) -> None:
    """加载 beginner_agent 项目的本地 .env 文件。

    中文注释：
    Python 的 os.getenv(...) 只能读取“已经存在的环境变量”，
    它不会自动读取项目目录里的 .env 文件。

    所以我们在这里集中做一件事：

        .env 文件
          -> os.environ
          -> checkpointing.py / memory.py / llm_client.py 等模块读取

    默认不覆盖已经存在的环境变量。
    这符合生产级习惯：
    - shell / Docker / CI 注入的环境变量优先级更高。
    - .env 主要服务本地开发。
    """

    global _ENV_LOADED

    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if _ENV_LOADED and not override:
        return
    if not path.exists():
        _ENV_LOADED = True
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = _strip_env_value(value)

    _ENV_LOADED = True
