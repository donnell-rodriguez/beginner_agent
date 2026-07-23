from __future__ import annotations

import sys

from router_eval import main as router_eval_main


# 中文注释：
# 这是 Router 上线门禁的薄入口。
#
# 为什么不把逻辑重复写一遍？
#   真正逻辑在 scripts/router_eval.py gate 和 routering/regression_gate.py。
#   这个文件只负责提供一个稳定、好记的 CI/pre-push 命令入口。
#
# 用法：
#
#   PYTHONPATH=.. uv run python scripts/router_release_gate.py
#   PYTHONPATH=.. uv run python scripts/router_release_gate.py --dataset router_eval.json
#
# 如果 gate 不通过，它会返回非 0 退出码，
# CI 或 git hook 就可以据此阻止合并/推送/启用新配置。


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    return router_eval_main(["gate", *args])


if __name__ == "__main__":
    raise SystemExit(main())
