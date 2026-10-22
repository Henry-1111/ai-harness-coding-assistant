# AI Harness 编程助手

AI Harness 是一个面向真实代码仓库的多 Agent 编程执行框架。它把模型的非确定性决策与文件、命令等副作用
隔离开：模型只生成结构化计划和编辑意图，Harness 负责路径边界、工具权限、隔离工作区、验证命令、Reviewer
验收和 Checkpoint 恢复。

## 已实现的闭环

```text
需求输入
  -> TaskRouter / Plan DAG
  -> Query Rewrite + Repo Map + Forge/Focus/Deepen 渐进式检索
  -> BM25 / Embedding / AST / Git / LSP 多路召回 + RRF 融合
  -> Evidence Judge -> Query Refine -> 最多 3 轮检索
  -> 全局上下文预算、停止条件与指纹缓存
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
- JSON-RPC LSP Client，并可将定义跳转结果接入 Scout 召回；
- 确定性 Query Rewrite、标准 BM25、可选 OpenAI-compatible Embedding、多路候选召回和
  Reciprocal Rank Fusion；
- 基于字符预算的上下文选择、证据覆盖判断、查询细化、最多三轮检索和仓库指纹缓存；
- 文件级 Recall@K、Precision@K、MRR、NDCG@K、Hit Rate、上下文压缩率、分组指标和
  单轮基线对比；
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
- 节点耗时、结果、检索阶段和 Checkpoint revision 的 Agent Trace；
- 支持按 revision 加载，以及通过二分定位首个异常 Checkpoint；
- Policy、Project、User、Session、Scratchpad 五层 SQLite Memory，Policy 注入 Coder Prompt，
  Project 保存最近检索，Session/Scratchpad 跟踪执行进度；
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

## 任务规范化、动态路由与进度账本

任务进入执行图前会先规范化为任务类型、复杂度、风险等级、派生验收条件、澄清问题和
人工审批要求。路由器据此选择：

- `fast`：简单、低风险修改，执行 `Scout → Code → Verify → Review`；
- `standard` / `complex`：增加 `Plan` 节点，校验 DAG 并为每个步骤创建 Assignment；
- `human_gate`：标记关键风险和人工审批要求，最终仍必须通过显式 Promote 门禁。

任务状态按飞书设计记录 CP1–CP8：目标规范化、上下文确认、Plan 校验、Assignment、写入
前边界、确定性测试、Reviewer、最终验收。每个节点仍会生成 append-only Checkpoint，支持
恢复和历史版本排查。

需要启用 Python 定义跳转时安装可选 LSP 依赖：

```bash
pip install -e '.[dev,lsp]'
export AI_HARNESS_LSP_COMMAND="pyright-langserver --stdio"
```

## CLI

设置任意 OpenAI-compatible Provider：

```bash
export AI_HARNESS_MODEL_BASE_URL=https://api.openai.com/v1
export AI_HARNESS_MODEL_API_KEY=your-key
export AI_HARNESS_MODEL=gpt-4.1-mini
```

可选开启向量召回和语言服务器定义跳转；未配置或外部服务失败时，系统会保留 BM25/AST/Git
检索并记录降级原因：

```bash
export AI_HARNESS_EMBEDDING_MODEL=text-embedding-3-small
export AI_HARNESS_EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
export AI_HARNESS_EMBEDDING_API_KEY='<llmgw-api-key>'
export AI_HARNESS_LSP_COMMAND="pyright-langserver --stdio"
```

通过本项目的 LLM Gateway 调用模型时，使用 Gateway 签发的租户 Key，而不是上游 Provider Key：

```bash
export AI_HARNESS_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export AI_HARNESS_MODEL_API_KEY='<llmgw-api-key>'
export AI_HARNESS_MODEL=gpt-4.1-mini
```

Harness 会调用 `http://127.0.0.1:8000/v1/chat/completions`；限流、Token 配额、Provider
路由、故障转移和调用日志由 Gateway 统一处理。

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
- `GET /v1/tasks/{task_id}/trace`
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

## 检索评测

项目提供带人工相关文件标注的固定任务集：

```bash
ai-harness eval-retrieval . benchmarks/retrieval_cases.json \
  --k 5 \
  --split test \
  --output benchmarks/latest-report.json
```

输出包含：

- `recall_at_k`：标注相关文件被召回的比例；
- `precision_at_k`：前 K 个结果中相关文件的比例；
- `mrr`：第一个相关文件排名的倒数均值；
- `ndcg_at_k`：同时考虑多个相关文件及其排序位置；
- `hit_rate`：至少召回一个相关文件的任务比例；
- `mean_context_reduction_ratio`：相对整个仓库，最终进入模型的上下文减少比例。
- `baseline_recall_at_k` / `recall_delta`：与单轮检索结果对比；
- `baseline_mrr` / `mrr_delta` / `pairwise`：比较单轮与多轮策略逐 Case 的胜、平、负；
- `by_language` / `by_task_type`：按语言和任务类型定位薄弱项；
- `bad_cases`：自动列出漏召回、相关文件未排第一、预算耗尽等失败原因；
- 每个 Case 的 `rounds` 与 `stop_reason`：验证迭代检索是否按证据或预算停止。

`tests/test_retrieval_benchmark.py` 设置了回归下限。当前数据集只有 8 个仓库内任务，只用于验证检索实现和防止
回归，不能作为外部通用准确率；对外写入简历前，应扩展为独立、冻结且不参与调参的任务集。

## Docker

```bash
export AI_HARNESS_MODEL_API_KEY=your-key
export AI_HARNESS_REPOSITORY=/absolute/path/to/repository
docker compose up --build
```

浏览器访问 `http://127.0.0.1:5173`。Compose 中源仓库以只读方式挂载，任务副本与 Checkpoint
写入独立数据卷；控制台通过容器内反向代理访问 API。

## 项目边界

- 当前 Repo Map 对 Python 提供 AST 级符号解析，其他语言提供文件和文本级索引；LSP 定义召回已接入
  Scout，但具体语言服务器仍需部署方通过 `AI_HARNESS_LSP_COMMAND` 配置。
- Embedding 召回通过 OpenAI-compatible `/embeddings` 接口接入；默认关闭。配置
  `AI_HARNESS_EMBEDDING_MODEL`、`AI_HARNESS_EMBEDDING_BASE_URL` 和
  `AI_HARNESS_EMBEDDING_API_KEY` 后，可与聊天请求一样统一经过 LLM Gateway。
- 当前 Checkpoint 使用 SQLite，单机部署足够；多实例部署应切换 PostgreSQL 并增加任务租约。
- CLI 默认使用 Docker Sandbox。`AI_HARNESS_SANDBOX_MODE=local` 仅适合测试，或 Harness 本身已经运行在受限容器
  中的场景；不要对不可信请求在宿主机启用本地模式。
- 当前数据不能支持简历中的准确率、召回率、成功率等数字；必须建立固定任务集并真实评测后才能写入。
