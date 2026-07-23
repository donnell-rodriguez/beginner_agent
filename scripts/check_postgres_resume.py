from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.checkpoint_resume_probe import run_checkpoint_resume_probe  # noqa: E402


def main() -> None:
    """验证 Postgres checkpoint 是否支持 interrupt/resume 闭环。"""

    os.environ.setdefault("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    result = run_checkpoint_resume_probe()
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.backend != "postgres":
        raise SystemExit("Expected postgres checkpoint backend.")
    if not result.interrupted:
        raise SystemExit("Expected graph to interrupt before resume.")
    if not result.resumed:
        raise SystemExit("Expected graph to resume and finish.")


if __name__ == "__main__":
    main()

