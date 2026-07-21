"""Beginner LangGraph agent example."""

from beginner_agent.config import load_project_env


# 中文注释：
# 只要导入 beginner_agent 包，就先加载项目本地 .env。
# 这样 checkpointing.py / memory.py / llm_client.py 等模块使用 os.getenv(...)
# 时，可以读到 /Users/christophermanning/Downloads/aios/langgraph/beginner_agent/.env。
load_project_env()
