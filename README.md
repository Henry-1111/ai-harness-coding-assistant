# AI Harness 编程助手

AI Harness 是一个面向真实代码仓库的多 Agent 编程执行框架。它把模型的非确定性决策与文件、命令等副作用
隔离开：模型只生成结构化计划和编辑意图，Harness 负责路径边界、工具权限、隔离工作区、验证命令、Reviewer
验收和 Checkpoint 恢复。

## 已实现的闭环

```text
需求输入
  -> TaskRouter / Plan DAG
  -> Repo Map + Forge/Focus/Deepen 渐进式检索
  -> Scout 收集最小上下文
  -> Coder 生成结构化 FileEdit
  -> 隔离工作区应用精确替换
  -> test / lint / type-check / build
  -> Reviewer 验收或有限 Replan
  -> SQLite Checkpoint
  -> 人工确认后 Promote 到源仓库
```

当前版本包括：

- FastAPI 与 CLI 两种入口；
- React + TypeScript + Ant Design 控制台，使用 Monaco 展示待发布 diff；
- Python AST Repo Map，提取 class、function、async function 和 import；
- Git 分支、提交历史、工作区状态与 diff 的只读上下文；
- JSON-RPC LSP Client，可按需补充定义跳转等语义上下文；
- 带服务器 allowlist 和租户/项目身份头的 MCP HTTP Client；
- 基于文件名、符号、依赖与文本命中的渐进式上下文检索；
- 带 Step ID、依赖、输入、输出和验收条件的 Plan DAG；
- LangGraph 驱动的 Scout、Coder、Verify、Reviewer 工作流；
- OpenAI-compatible JSON 模型适配器；
- exact-match 文件编辑、目录穿越防护、文件范围和权限检查；
- allowlist 命令执行，不经过 shell；
- 本地持久化工作区，以及网络隔离、内存/CPU/PID 限制的 Docker Sandbox；
- 测试失败后的有限 Replan，防止无限循环和修改范围扩张；
- SQLite append-only Checkpoint，可查询和恢复任务；
- Policy、Project、User、Session、Scratchpad 五层 SQLite Memory；
- 只有测试通过且 Reviewer 批准后才能 Promote 到源仓库。

## 安全模型

1. 模型不能直接执行命令，只能返回符合 Pydantic Schema 的 JSON。
2. 所有文件路径必须位于仓库根目录和任务 `file_scope` 内。
3. 修改已有文件时，`old_text` 必须唯一匹配，避免模糊替换错误位置。
4. 验证命令通过 `shlex` 拆分并检查可执行文件 allowlist，不使用 `shell=True`。
5. Docker Sandbox 默认 `--network none`、`no-new-privileges`，并限制 CPU、内存和 PID。
6. 修改先保存在任务工作区；源仓库只有在 Reviewer 批准后通过显式 Promote 才会改变。

## 本地开发

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## CLI

设置任意 OpenAI-compatible Provider：

```bash
export AI_HARNESS_MODEL_BASE_URL=https://api.openai.com/v1
export AI_HARNESS_MODEL_API_KEY=your-key
export AI_HARNESS_MODEL=gpt-4.1-mini
```

在隔离工作区执行任务：

```bash
ai-harness run /path/to/repository \
  "修复订单重复扣款，并补充回归测试" \
  --scope src/payments \
  --scope tests/payments \
  --accept "相同幂等键只能扣款一次" \
  --verify "pytest -q"
```

查询、恢复和人工确认发布：

```bash
ai-harness status <task-id>
ai-harness resume <task-id>
ai-harness promote <task-id>
```

## API

```bash
ai-harness serve --host 127.0.0.1 --port 8080
```

```http
POST /v1/tasks
Content-Type: application/json

{
  "repository": "/repository",
  "goal": "修复订单重复扣款，并补充回归测试",
  "file_scope": ["src/payments", "tests/payments"],
  "acceptance_criteria": [
    {"description": "相同幂等键只能扣款一次"}
  ],
  "verification_commands": ["pytest -q"]
}
```

其他接口：

- `GET /health`
- `GET /v1/tasks/{task_id}`
- `GET /v1/tasks/{task_id}/diff`
- `POST /v1/tasks/{task_id}/resume`
- `POST /v1/tasks/{task_id}/promote`

## Web 控制台

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。控制台支持创建任务、查看 Plan DAG 执行状态、验证结果、Reviewer
结论和完整 diff，并且只有任务通过审核后才允许点击 Promote。

## Docker

```bash
export AI_HARNESS_MODEL_API_KEY=your-key
export AI_HARNESS_REPOSITORY=/absolute/path/to/repository
docker compose up --build
```

浏览器访问 `http://127.0.0.1:5173`。Compose 中源仓库以只读方式挂载，任务副本与 Checkpoint
写入独立数据卷；控制台通过容器内反向代理访问 API。

## 项目边界

- 当前 Repo Map 对 Python 提供 AST 级符号解析，其他语言提供文件和文本级索引；LSP Client 已具备协议能力，具体语言服务器由部署方配置。
- 当前 Checkpoint 使用 SQLite，单机部署足够；多实例部署应切换 PostgreSQL 并增加任务租约。
- CLI 默认使用 Docker Sandbox。`AI_HARNESS_SANDBOX_MODE=local` 仅适合测试，或 Harness 本身已经运行在受限容器
  中的场景；不要对不可信请求在宿主机启用本地模式。
- 当前数据不能支持简历中的准确率、召回率、成功率等数字；必须建立固定任务集并真实评测后才能写入。
