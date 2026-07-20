# Runtime Environment

中文注释：
这个文件解释 beginner_agent 的“运行环境”在哪里配置。

## 1. Executor 不是运行环境

`executor.py` 的职责是：

```text
接收 tool_name / tool_args
执行经过 Policy 允许的工具
记录 execution_attempt
记录 patch_history
把结果交给 Monitor / Evaluator
```

它不应该自己决定：

```text
Python 用哪个命令运行
Rust 用哪个 cargo 目录
cache 写到哪里
cwd 是 repo root 还是 active project
timeout 是多少
```

这些属于 runtime 配置。

## 2. 当前 runtime 配置位置

代码位置：

```text
tooling/runtime.py
tooling/command_tools.py
```

`tooling/runtime.py` 定义 runtime：

```text
python runtime
  cwd_mode = repo_root
  UV_CACHE_DIR = .agent_state/uv-cache
  PYTHONDONTWRITEBYTECODE = 1

rust runtime
  cwd_mode = active_project
  CARGO_HOME = .agent_state/cargo-home
  CARGO_TARGET_DIR = .agent_state/cargo-target
  RUST_BACKTRACE = 1

generic runtime
  cwd_mode = active_project
```

`tooling/command_tools.py` 定义命令 profile：

```text
python_compileall -> python runtime
pytest_beginner_agent -> python runtime
ruff_check -> python runtime
mypy_beginner_agent -> python runtime

cargo_check -> rust runtime
cargo_test -> rust runtime
cargo_clippy -> rust runtime
cargo_fmt_check -> rust runtime
```

## 3. 为什么 Python 和 Rust 要分开

Python 常见运行环境问题：

```text
需要 uv / venv / pyproject.toml
可能写 __pycache__
可能写 uv cache
测试命令可能是 pytest
类型检查可能是 mypy
lint 可能是 ruff
```

Rust 常见运行环境问题：

```text
需要 Cargo.toml
会写 target 目录
会访问 CARGO_HOME
测试命令是 cargo test
构建检查是 cargo check
lint 是 cargo clippy
格式检查是 cargo fmt --check
```

所以不能让 Executor 用同一套方式运行所有语言。
应该让 Executor 选择工具，工具选择 runtime profile。

## 4. 当前运行链路

```text
Executor
  -> run_tool_model(...)
  -> run_allowed_command_tool(profile)
  -> command_tools._run_profile(profile)
  -> runtime.resolve_command_runtime(...)
  -> subprocess.run(cmd, cwd=..., env=..., timeout=...)
```

也就是说：

```text
Executor 不直接运行 shell。
LLM 不能直接写 shell 命令。
LLM 只能选择白名单工具。
白名单工具只能选择白名单 command profile。
command profile 再绑定 runtime。
```

## 5. 本地如何配置

可以通过 `.env.example` 里的变量覆盖 cache 目录：

```text
BEGINNER_AGENT_UV_CACHE_DIR=.agent_state/uv-cache
BEGINNER_AGENT_CARGO_HOME=.agent_state/cargo-home
BEGINNER_AGENT_CARGO_TARGET_DIR=.agent_state/cargo-target
```

默认情况下，这些 cache 都会写到：

```text
beginner_agent/.agent_state/
```

这样不会污染用户全局目录：

```text
~/.cache/uv
~/.cargo
```

## 6. 大厂通常怎么做

大厂 code agent 一般不会直接在用户主机裸跑命令。
更常见的是：

```text
Runtime Registry
  记录支持哪些 runtime：python、rust、node、go、java

Command Profile
  记录每个 runtime 允许执行哪些命令

Sandbox / Container
  每个任务在隔离容器、VM、sandbox 或 remote worker 中运行

Artifact Store
  测试日志、构建产物、diff、coverage 写到可审计存储

Timeout / Budget
  每个命令有超时、重试、资源预算

Policy Layer
  高风险命令、写文件、依赖安装、网络访问都需要权限控制

Audit
  记录谁执行了什么、在哪个 runtime、输出是什么、改了哪些文件
```

所以大厂式结构大致是：

```text
Planner
  -> Tool Policy
  -> Executor
  -> Runtime Registry
  -> Sandbox Worker
  -> ToolResult
  -> Execution Monitor
  -> Evaluator
```

## 7. 当前项目还差什么

当前 beginner_agent 已经有：

```text
runtime spec
command profile
Python/Rust 分层
cache 隔离
timeout
白名单命令
execution_attempt
watchdog / recovery
```

但还不是完整大厂级运行平台。
后续可以继续升级：

```text
Docker sandbox
remote worker
resource limit
network policy
dependency install approval
artifact storage
test report parser
coverage report
per-project runtime config file
```
