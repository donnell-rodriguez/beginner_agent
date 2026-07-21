from __future__ import annotations

import argparse

from beginner_agent.memory_lifecycle_scheduler import memory_lifecycle_scheduled_report_json


def main() -> None:
    """运行 Memory Lifecycle Job。

    中文注释：
    这个文件是后台任务入口。
    现在入口不再直接调用 memory_lifecycle.py 的业务函数，
    而是经过 memory_lifecycle_scheduler.py：
    - 先拿锁。
    - 检查同一个 run_key 是否已经成功。
    - 失败自动重试。
    - 写入运行历史。
    """

    parser = argparse.ArgumentParser(description="Run scheduled memory lifecycle job.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略幂等 run_key 检查，强制执行一次 lifecycle job。",
    )
    args = parser.parse_args()
    print(memory_lifecycle_scheduled_report_json(force=args.force))


if __name__ == "__main__":
    main()
