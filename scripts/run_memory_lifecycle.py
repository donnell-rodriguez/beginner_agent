from __future__ import annotations

from beginner_agent.memory_lifecycle import memory_lifecycle_report_json


def main() -> None:
    """运行 Memory Lifecycle Job。

    中文注释：
    这个文件是后台任务入口。
    本地可以手动运行；生产环境可以交给 cron / Kubernetes CronJob / Prefect。
    """

    print(memory_lifecycle_report_json())


if __name__ == "__main__":
    main()
