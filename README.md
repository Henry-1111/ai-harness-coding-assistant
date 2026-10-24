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
  -> 证据仍不足时进入受限 ReAct 只读工具子图（可选）
  -> 全局上下文预算、停止条件与指纹缓存
  -> Scout 收集最小上下文
  -> Coder 生成结构化 FileEdit
  -> 隔离工作区应用精确替换
  -> test / lint / type-check / build
  -> Reviewer 验收或有限 Replan
  -> SQLite/PostgreSQL Checkpoint
  -> SQLite/PostgreSQL Job Queue / 有界 Worker / 租约恢复
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
- 受限 ReAct 知识子图：最多 3 次工具调用、Pydantic 参数校验、工具/路径权限过滤、只读 MCP 适配；
- 基于文件名、符号、依赖与文本命中的渐进式上下文检索；
- 带 Step ID、依赖、输入、输出和验收条件的 Plan DAG；
- 复杂任务由 Implementer 与 Test/Integrator 两个 Agent 形成提案、冲突检查和整合决策；
- LangGraph 驱动的 Scout、Coder、Verify、Reviewer 工作流；
- OpenAI-compatible JSON 模型适配器；
- exact-match 文件编辑、目录穿越防护、文件范围和权限检查；
- allowlist 命令执行，不经过 shell；
- 本地持久化工作区，以及网络隔离、内存/CPU/PID 限制的 Docker Sandbox；
- 测试失败后的有限 Replan，防止无限循环和修改范围扩张；
- 结构化 Reviewer Finding、可审计 RepairPlan、重复失败指纹熔断与项目经验沉淀；
- 本地 SQLite、分布式 PostgreSQL append-only Checkpoint，可查询和恢复任务；
- SQLite/PostgreSQL 持久化任务队列、原子领取、Worker 最大并发、租约心跳、失败重试、容量保护与协作式取消；
- Worker 注册表、基础设施失败熔断、半开探测、Readiness 与 Prometheus 指标；
- `Idempotency-Key` 防重复提交，以及与状态变更同事务写入的 Job 审计事件；
- 节点耗时、结果、检索阶段和 Checkpoint revision 的 Agent Trace；
- 支持按 revision 加载，以及通过二分定位首个异常 Checkpoint；
- Policy、Project、User、Session、Scratchpad 五层 SQLite/PostgreSQL Memory，Policy 注入 Coder Prompt，
  Project 保存最近检索，Session/Scratchpad 跟踪执行进度；
- 只有测试通过且 Reviewer 批准后才能 Promote 到源仓库。

## 请求身份、租户边界与 RBAC

HTTP API 可启用 Bearer API Key 与 OIDC JWT 混合身份认证。配置文件只保存 Key 的 SHA-256，不保存明文；
每个身份包含租户、主体、角色和允许访问的仓库根目录。Job 会把 `tenant_id` 与 `created_by` 持久化，
Worker 再把可信租户身份传入 Agent 和 MCP 上下文，避免只在入口鉴权、执行阶段却串用默认租户。

支持的角色：

- `submitter`：提交、恢复和取消本租户任务；
- `reviewer`：批准或拒绝本租户高风险任务，但不能审批自己提交的任务；
- `publisher`：将已通过 Reviewer 的修改发布到源仓库；
- `operator`：读取运行时、队列和 Prometheus 指标；
- `auditor`：读取本租户治理审计事件；
- `admin`：跨租户管理权限，仍会留下审计记录。

先复制 [`config/governance.example.json`](config/governance.example.json)，为每个身份生成独立随机 Key，
再把 SHA-256 写入配置。示例中的 `00/11/22` 哈希只是不可登录的占位符：

```bash
python -c 'import getpass,hashlib; print(hashlib.sha256(getpass.getpass().encode()).hexdigest())'

export AI_HARNESS_AUTH_ENABLED=true
export AI_HARNESS_AUTH_CONFIG_PATH=/absolute/path/to/governance.json
ai-harness serve --host 127.0.0.1 --port 8080
```

调用时发送原始 Key，服务端只计算哈希后进行常量时间比较：

```http
Authorization: Bearer <tenant-api-key>
```

`GET /v1/auth/me` 返回当前主体、租户和角色。控制台中的 Key 只保存在当前标签页的
`sessionStorage`；生产环境若存在不可信前端脚本，应改用 BFF 和 HttpOnly Session，而不是让浏览器直接持有
长期 Key。`GET /v1/audit/events` 返回按租户隔离的 append-only 审计事件，每条事件包含前一条哈希，便于
检测历史记录被改写；`GET /v1/audit/integrity` 会重新计算当前租户的完整哈希链。认证默认关闭，
以兼容本地 CLI；部署环境应显式开启。

生产部署可在同一个治理配置中增加 `oidc`，也可以将 `keys` 设为空、仅接受 JWT：

```json
{
  "keys": [],
  "oidc": {
    "issuer": "https://identity.example.com",
    "audience": "ai-harness",
    "algorithms": ["RS256"],
    "tenant_claim": "tenant_id",
    "roles_claim": "roles",
    "repository_roots_claim": "repository_roots"
  }
}
```

若不填写 `jwks_url`，服务会从 Issuer 的 OpenID Discovery 文档读取 `jwks_uri`。JWT 必须包含 `iss`、
`aud`、`sub`、`iat`、`exp` 以及租户、角色、仓库范围 Claims；算法只接受服务端配置的 RSA 白名单。
JWKS 会按 TTL 缓存，遇到未知 `kid` 时强制刷新一次，因此密钥轮换无需重启。API Key 和 JWT 使用相同的
RBAC、租户隔离、配额与审计链，`GET /v1/auth/me` 的 `authentication_method` 可区分认证来源。

### 租户并发与每日额度

治理配置中的 `tenant_quotas` 为每个租户设置最大在途任务、每日任务和每日 Token 上限。任务提交时会在
SQLite `BEGIN IMMEDIATE` 事务内原子检查并预留 Token；达到任一上限返回 HTTP 429，任务不会进入队列。
同一租户的相同 `Idempotency-Key` 使用确定性 Job ID，因此并发重放只占一份额度；不同租户可安全复用
相同幂等键。

```json
{
  "tenant_id": "tenant-a",
  "max_outstanding_jobs": 4,
  "max_daily_jobs": 100,
  "max_daily_tokens": 1000000,
  "token_reservation": 20000
}
```

### PostgreSQL + Redis 分布式治理

默认本地模式继续使用 SQLite。生产多实例部署同时配置以下两个变量后，Checkpoint、Job、五层 Memory、
资源归属、审计链和额度账本会写入 PostgreSQL，Redis 则作为所有实例共享的实时准入层：

```bash
export AI_HARNESS_GOVERNANCE_DATABASE_URL='postgresql://harness:secret@postgres:5432/ai_harness'
export AI_HARNESS_REDIS_URL='redis://redis:6379/0'
export AI_HARNESS_REDIS_KEY_PREFIX='ai-harness'
export AI_HARNESS_REDIS_QUOTA_TTL_SECONDS=259200
export AI_HARNESS_REDIS_BINDING_CACHE_SECONDS=300
```

额度预留、结算和释放由 Redis Lua 脚本原子完成，因此多个 API/Worker 进程不会分别读取旧值后同时放行。
Redis 准入成功后再写 PostgreSQL 长期账本；若 PostgreSQL 写入失败，服务会删除本次 Redis 预留并恢复计数。
对外额度快照和第二道额度校验以 PostgreSQL 为准，因此 Redis 重启或缓存丢失也不会导致配额超发或展示归零。
资源归属采用 cache-aside：先读 Redis，未命中再读 PostgreSQL 并回填；写入则以 PostgreSQL 为准并刷新缓存。
审计链写入前使用 PostgreSQL transaction-level advisory lock 按租户串行化，避免多个实例同时生成相同
`previous_hash` 导致链分叉。
所有 PostgreSQL Store 在建表和索引初始化前还会争用统一的 transaction-level advisory lock，确保多个
API/Worker 副本首次同时启动或滚动扩容时不会并发修改系统目录。Checkpoint、Job 和 Governance Schema
分别在 `ai_harness_schema_migrations` 记录连续版本；迁移在同一事务中执行，旧版本程序若遇到更高版本数据库
会拒绝启动，避免用不兼容结构继续读写。

Compose 已启用 PostgreSQL 持久卷和 Redis AOF。当前 Lua 脚本面向单 Redis 主节点或 Sentinel；若部署
Redis Cluster，需要把同一次脚本访问的 Key 重新设计为相同 hash slot。

Worker 会把 OpenAI-compatible Provider 返回的 `usage.prompt_tokens` 和 `usage.completion_tokens` 聚合到
`TaskResult.model_usage`，终态后用实际 Token 替换预留量并释放在途名额。若 Provider 没有返回 usage，
系统保守地按预留量结算，避免缺失统计导致配额超发；排队取消按 0 Token 结算。高风险任务停在审批门时
继续占用在途名额，审批完成或拒绝后再结算。`GET /v1/quota` 返回当前租户的实时额度快照。
后台对账器会周期扫描仍为 active、但 Job 已进入终态的持久化预留，在结算回调遭遇短暂数据库或 Redis
故障时自动重试；重复结算是幂等的。对账次数、成功数和失败数暴露为 Prometheus 指标，扫描间隔可通过
`AI_HARNESS_QUOTA_RECONCILIATION_INTERVAL_SECONDS` 配置。

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

## 自动化质量门禁

仓库的 GitHub Actions 会在每次 push 和 Pull Request 中并行执行三组检查：

- 后端：Python 编译、Ruff、全项目 Pyright，以及连接真实 PostgreSQL 16 和 Redis 7.4 的完整测试；
- 前端：固定依赖安装、TypeScript 检查和 Vite 生产构建；
- 交付物：分别构建 API 与控制台容器镜像，并验证 Compose 配置可以解析。

本地提交前可运行同等的核心检查：

```bash
python -m compileall -q src
ruff check .
pyright --pythonpath "$(command -v python)" src
pytest -ra
cd frontend && npm ci && npm run build
```

## 任务规范化、动态路由与进度账本

任务进入执行图前会先规范化为任务类型、复杂度、风险等级、派生验收条件、澄清问题和
人工审批要求。路由器据此选择：

- `fast`：简单、低风险修改，执行 `Scout → Code → Verify → Review`；
- `standard` / `complex`：增加 `Plan` 节点，校验 DAG 并为每个步骤创建 Assignment；
- `human_gate`：高风险任务在模型调用和工具执行前进入 `needs_input`，必须记录审批人和决定；
  Reviewer 通过后仍需单独执行 Promote 才能修改源仓库。

任务状态按飞书设计记录 CP1–CP8：目标规范化、上下文确认、Plan 校验、Assignment、写入
前边界、确定性测试、Reviewer、最终验收。每个节点仍会生成 append-only Checkpoint，支持
恢复和历史版本排查。

## 结果评审、自愈与经验沉淀

Reviewer 会把无修改、验证失败、超时和越界修改转换为结构化 Finding。失败任务生成带失败指纹、
失败命令和修复指令的 RepairPlan，并在下一轮显式注入 Coder。相同确定性失败第二次出现时立即停止
自动修复，避免无效循环耗尽模型额度；`max_replans` 仍是总重试上限。

终态任务会把结果、验证状态、修复次数和失败指纹写入项目级 `experiences` 记忆。记录按 task ID
幂等写入并最多保留 20 条；后续 Coder 只读取最近 5 条结构化经验，不保存模型思维链或 Provider 密钥。

## 多 Agent 协作

`complex` 和 `human_gate` 路由会把 Code 阶段展开为 `implementation → test_design → code`：
Implementer 先生成候选修改，Test/Integrator 再依据验收条件、测试缺口和候选冲突生成唯一的最终决策。
候选提案不会直接写文件；只有经过整合且不存在同一源文本竞争替换的最终编辑，才能进入权限受控的
`apply_edit`。快速和标准任务仍使用单 Coder 路径，避免额外模型调用。

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

常规渐进检索仍无法形成充分证据时，可启用受限 ReAct。它只开放仓库搜索和限定行读取，
默认关闭；已获得充分证据的任务不会进入该子图，也不会产生额外模型调用：

```bash
export AI_HARNESS_RESTRICTED_REACT_ENABLED=true
export AI_HARNESS_RESTRICTED_REACT_MAX_TOOL_CALLS=3
```

受限 ReAct 拒绝写工具、Shell、SQL 和 Redis，工具参数必须通过 Schema 校验，返回的文件来源
还要通过任务 `file_scope` 过滤。MCP 工具只能通过 `MCPReadOnlyTool` 注册，并同时满足工具 allowlist、
MCP Server allowlist 以及租户/项目身份透传。

### 配置外部 MCP 工具

外部工具目录必须由部署方在本地固定，Agent 不能从远端动态扩大自己的权限。参考
[`config/mcp-tools.example.json`](config/mcp-tools.example.json) 配置 Server、只读工具、输入 Schema
和租户/项目 allowlist，然后设置：

```bash
export AI_HARNESS_MCP_CATALOG_PATH=/absolute/path/to/mcp-tools.json
export AI_HARNESS_TENANT_ID=local
export AI_HARNESS_MCP_AUTHORIZATION='Bearer replace-me'
```

配置中的 `headers_from_env` 只保存环境变量名称，不保存密钥。启动时会校验工具名唯一性、Server
引用、租户授权包含关系、只读标记以及受支持的 JSON Schema；未知字段会被拒绝。MCP 返回文本按
单次调用 12,000 字符封顶，并作为带来源标识的只读上下文进入 Coder，不会获得文件写权限。

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
ai-harness approve <task-id> --actor security-reviewer --note "scope reviewed"
ai-harness reject <task-id> --actor repository-owner --note "unsafe request"
ai-harness promote <task-id>
```

## API

```bash
ai-harness serve --host 127.0.0.1 --port 8080
```

```http
POST /v1/jobs
Content-Type: application/json
Idempotency-Key: client-request-123

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

异步入口立即返回 `job_id` 和 `pending` 状态。Worker 在后台领取任务，客户端通过
`GET /v1/jobs/{job_id}` 查询 `pending → running → needs_input/succeeded/failed`。高风险任务仍会
停在人工审批门，审批结果会回写到原 Job。`pending` 和 `needs_input` Job 会立即取消；审批门取消会同时
写入 Task Checkpoint，迟到的审批请求不能重新激活任务。`running` Job 会记录停止请求，
并在 Agent 节点、模型返回、文件写入和验证命令之间的安全边界协作式停止，不会伪装成已立即终止。

调用方可以为 `POST /v1/jobs` 提供最长 200 字符的 `Idempotency-Key`。相同 Key 与相同请求会返回
第一次创建的 Job，不消耗第二份队列容量；相同 Key 携带不同请求时返回 HTTP 409，防止误把不同任务合并。
`GET /v1/jobs/{job_id}/events` 按顺序返回 `queued`、`claimed`、`retry_scheduled`、取消和完成等事件。
事件与对应 Job 状态在同一个存储事务中落库，因此问题排查不依赖易丢失的进程日志。

本地队列使用 SQLite 事务，分布式队列使用 PostgreSQL 行锁与 `SKIP LOCKED`，保证同一任务只有一个
Worker 领取，并通过心跳续租。服务异常退出后，租约过期的
`running` Job 会在重试预算内重新领取，因此执行语义是 at-least-once；每次执行都在新的隔离工作区内，且
不会自动 Promote 到源仓库，避免重试产生不可逆的重复副作用。

并发与恢复参数：

```bash
export AI_HARNESS_JOB_WORKERS=2
export AI_HARNESS_JOB_POLL_INTERVAL_SECONDS=0.2
export AI_HARNESS_JOB_LEASE_SECONDS=300
export AI_HARNESS_JOB_QUEUE_CAPACITY=100
export AI_HARNESS_JOB_TIMEOUT_SECONDS=1800
export AI_HARNESS_CIRCUIT_FAILURE_THRESHOLD=3
export AI_HARNESS_CIRCUIT_RECOVERY_SECONDS=30
```

容量上限统计 `pending/running/needs_input` 三类未完成任务，达到上限时新建 Job 返回 HTTP 429，防止
模型调用和本地执行资源被无限排队拖垮。总执行时限覆盖完整 Agent 流程；单次模型请求和 Sandbox 命令
仍保留各自更短的超时。`GET /v1/jobs/metrics` 返回各状态数量、重试任务数、取消请求数、最老排队时间和
当前容量，用于控制台或监控系统采集。

Worker 会区分业务验证失败与基础设施失败。测试不通过、Reviewer 拒绝等业务结果不会触发熔断；连接超时、
Provider HTTP 错误、SQLite 运行错误和任务总超时会进入重试并累计连续失败。达到阈值后执行熔断打开，暂停
领取新任务；恢复窗口结束后只允许一个 Worker 做半开探测，探测成功才恢复并发。`GET /ready` 在熔断打开，
或 Checkpoint、Job Store、治理存储任一依赖不可用时返回 HTTP 503；PostgreSQL 探测连接最长等待 5 秒。
`GET /v1/runtime` 返回 Worker、队列和熔断快照，`GET /metrics` 输出 Prometheus 文本，并分别暴露各依赖的
ready gauge。HTTP 指标按方法、路由模板和状态码统计请求数及延迟直方图；`/v1/jobs/{job_id}` 这类动态
路径只使用模板作为标签，不会因真实 Job ID 持续制造高基数时间序列。
请求可携带最长 64 字符、仅包含字母数字和 `._-` 的 `X-Request-ID`；服务会在响应头原样返回，并输出不含
请求正文、Authorization 或 API Key 的 JSON 访问日志。非法或缺失的 ID 会替换为随机值，未知 HTTP Method
统一聚合为 `OTHER`，避免日志注入和 Prometheus 标签基数被恶意请求放大。异步任务还会把规范化后的
`request_id` 持久化到 Job 及首个 `queued` 事件中，使入口日志、任务状态和 Worker 事件可跨进程关联；
幂等重放始终返回首次创建任务的关联 ID，不覆盖原始审计链路。Worker 在领取和结算任务时也会输出带
`request_id/job_id/worker_id` 的 JSON 日志；若租约已被其他 Worker 接管，迟到的结算会记录为
`job_settlement_rejected`，且不计入当前 Worker 的完成统计。
API 响应默认携带 `nosniff`、禁止嵌入、禁止 Referrer 和 `no-store` 安全头。生产控制台域名必须通过
`AI_HARNESS_CORS_ALLOWED_ORIGINS` 以逗号分隔显式配置；通配符、带用户信息或带路径的 Origin 会在启动时
被拒绝，避免误把带 Bearer Token 的跨域 API 暴露给任意站点。

原有同步 `POST /v1/tasks` 保留，便于 CLI、测试和兼容调用方使用。

其他接口：

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/runtime`
- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/events`
- `GET /v1/jobs/metrics`
- `POST /v1/jobs/{job_id}/cancel`
- `POST /v1/tasks`（同步兼容入口）
- `GET /v1/tasks/{task_id}`
- `GET /v1/tasks/{task_id}/diff`
- `GET /v1/tasks/{task_id}/trace`
- `POST /v1/tasks/{task_id}/resume`
- `POST /v1/tasks/{task_id}/approval`
- `POST /v1/tasks/{task_id}/promote`

## Web 控制台

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。控制台使用异步 Job API 创建任务，展示排队/执行状态、尝试次数，
并允许安全取消任务；顶部监控卡每 3 秒刷新队列、活跃 Worker、Readiness 和熔断状态。任务完成后可查看
Plan DAG、验证结果、Reviewer 结论和完整 diff，并且只有任务通过审核后才允许点击 Promote。

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

## 单 Agent / 多 Agent A/B 评测

项目提供成对评测命令：同一模型、同一 Provider、同一冻结任务集分别走单 Coder 和
Implementer + Test/Integrator 两条路径。每个 Case 都使用全新的隔离工作区，且运行顺序交替，避免
前后顺序持续偏向某一组。评测不会执行 Promote，因此不会修改被测源仓库。

仓库内附带 3 个确定性缺陷作为冒烟样例。使用宿主机本地执行器只适用于这组可信 Fixture：

```bash
export AI_HARNESS_SANDBOX_MODE=local
export AI_HARNESS_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export AI_HARNESS_MODEL_API_KEY='<llmgw-api-key>'
export AI_HARNESS_MODEL=gpt-4.1-mini

ai-harness eval-agents \
  benchmarks/fixtures/agent_eval \
  benchmarks/agent_cases.json \
  --split test \
  --output benchmarks/agent-report.json
```

报告包含两组成功率、耗时、模型调用次数、请求字符数、Provider Token、修复轮数、协作轮数、冲突数、
预期修改文件的 Recall/Precision，以及逐 Case 的胜/平/负。只有上游 OpenAI-compatible 接口返回
`usage` 时才统计 Token；否则字段为 `null`，不会用字符数伪造 Token。报告同时写入冻结数据集的
SHA-256，便于确认不同实验使用的是同一版本。

这 3 个 Case 只验证评测链路，不能作为简历结论。要形成可信数据，应扩展到至少 30 个未参与提示词
调优的独立任务，固定模型与 Gateway 配置，多次重复运行，并同时报告成功率提升和 Token/时延成本。
首轮真实 DeepSeek 冒烟结果见 [`benchmarks/AGENT_RESULTS.md`](benchmarks/AGENT_RESULTS.md)，原始机器可读
报告见 [`benchmarks/agent-report.json`](benchmarks/agent-report.json)。

## 系统级验证与运行时基准

自动化测试中的 `tests/test_system_e2e.py` 会启动一个本地 OpenAI-compatible HTTP Stub，验证完整链路：

```text
POST /v1/jobs -> SQLite Queue -> Worker -> LLM Gateway HTTP
-> 结构化编辑 -> 隔离测试 -> Reviewer -> 显式 Promote
```

测试还会注入第一次 HTTP 500、第二次成功的故障，确认任务发生一次 `retry_scheduled` 后恢复，并且
`queued → claimed → retry_scheduled → claimed → completed` 审计事件顺序可查。Stub 测试不访问外网，
也不会消耗真实模型额度。

运行时基准使用合成任务测量 SQLite Queue、Worker 并发和幂等语义：

```bash
ai-harness eval-runtime \
  --jobs 100 \
  --workers 4 \
  --synthetic-work-ms 10 \
  --output benchmarks/runtime-report.json
```

当前机器的一次实测结果见 [`benchmarks/RUNTIME_RESULTS.md`](benchmarks/RUNTIME_RESULTS.md)。该命令不会
调用模型，`jobs_per_second` 只代表本机任务运行时上限，不能写成 DeepSeek 或 LLM Gateway 的请求吞吐。

需要验证真实 AI Harness → LLM Gateway → DeepSeek 链路时，必须显式开启额度开关：

```bash
export AI_HARNESS_LIVE_E2E=1
export AI_HARNESS_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export AI_HARNESS_MODEL_API_KEY='<llmgw-api-key>'
export AI_HARNESS_MODEL=deepseek-chat

PYTHONPATH=src python scripts/live_gateway_smoke.py
```

烟测只在临时仓库工作，不会自动 Promote，也不会打印 API Key。输出包含最终状态、修改文件、Provider
返回的 Token 使用量，以及源仓库是否保持不变；未设置 `AI_HARNESS_LIVE_E2E=1` 时脚本直接拒绝运行。

## Docker

```bash
export AI_HARNESS_MODEL_API_KEY=your-key
export AI_HARNESS_REPOSITORY=/absolute/path/to/repository
docker compose up --build
```

浏览器访问 `http://127.0.0.1:5173`。Compose 中源仓库以只读方式挂载，任务副本与 Checkpoint
写入独立数据卷；控制台通过容器内反向代理访问 API。API 容器会使用 `/ready` 同时检查 Worker、熔断器、
Checkpoint、队列、PostgreSQL 和 Redis，只有健康后才启动控制台；停止时保留 30 秒供 Worker 结束当前
安全边界与释放资源，未完成任务仍可在租约过期后由其他实例接管。

异步 `POST /v1/jobs` 在入队前会把授权后的源仓库复制到
`AI_HARNESS_WORKSPACE_SNAPSHOT_DIR`。快照先写临时目录再原子发布，忽略 `.git`、虚拟环境、依赖目录和缓存，
并保留符号链接而不跟随读取仓库边界外的内容。Job 只保存不可由客户端指定的快照 ID；Worker 从快照创建
自己的隔离工作区，因此不再依赖 API 主机上的原始绝对路径。多主机部署需要把该目录以相同路径挂载到
所有 API/Worker 实例；未配置时保持原来的本机路径模式。任务进入审批等待或被数据库接受为终态后，当前
工作区会以 `task-{task_id}` 原子更新到共享目录；其他实例处理审批恢复或 Diff 时会自动重建本地工作区。
租约已经失效的 Worker 不会发布任务快照，避免迟到结果覆盖当前所有者的产物。

共享快照总容量由 `AI_HARNESS_WORKSPACE_SNAPSHOT_MAX_BYTES` 控制（默认 10 GiB）。API 实例使用跨进程文件锁
串行完成“统计、校验、原子发布”，不会因多个实例同时入队而超卖磁盘预算；超过上限的新任务返回 HTTP 413，
更新任务快照失败时保留上一份可恢复版本。`/metrics` 暴露快照数量、已用字节和容量上限，Prometheus 在连续
10 分钟超过 85% 时告警，Grafana 看板展示实时占用比例。

### Prometheus 与 Grafana

监控栈使用可选 Compose profile，不会增加默认开发环境的启动负担：

```bash
docker compose --profile monitoring up --build
```

Prometheus 位于 `http://127.0.0.1:9090`，Grafana 位于 `http://127.0.0.1:3000`。Grafana 会自动加载
Prometheus 数据源和 `AI Harness Overview` 看板，展示 Readiness、依赖健康、队列积压、Worker、熔断状态、
HTTP RPS/P95 延迟、任务状态、快照容量与额度对账失败。Prometheus 同时预置依赖不可用、熔断打开、队列阻塞、
无 Worker、快照容量过高、额度对账失败和 HTTP 5xx 比例过高等告警规则。

该 profile 默认采用只读匿名看板，适用于本机或受保护的内部网络；对外部署前应关闭 Grafana 匿名访问，接入
统一身份认证，并为 Prometheus 抓取 `/metrics` 配置具有 `operator` 角色的凭据。

## 项目边界

- 当前 Repo Map 对 Python 提供 AST 级符号解析，其他语言提供文件和文本级索引；LSP 定义召回已接入
  Scout，但具体语言服务器仍需部署方通过 `AI_HARNESS_LSP_COMMAND` 配置。
- Embedding 召回通过 OpenAI-compatible `/embeddings` 接口接入；默认关闭。配置
  `AI_HARNESS_EMBEDDING_MODEL`、`AI_HARNESS_EMBEDDING_BASE_URL` 和
  `AI_HARNESS_EMBEDDING_API_KEY` 后，可与聊天请求一样统一经过 LLM Gateway。
- 配置 PostgreSQL + Redis 后，Checkpoint、任务队列、Worker 租约、治理、额度和资源归属均使用共享后端；
  未配置时继续使用 SQLite 本地模式。多实例 Worker 通过 PostgreSQL `FOR UPDATE SKIP LOCKED` 抢占任务，
  通过带所有者校验的租约心跳、完成和失败更新避免旧 Worker 覆盖已经被接管的任务。
- PostgreSQL 模式也会共享五层 Memory，并用按记忆键加锁的事务避免多个 Worker 同时追加经验时丢数据。
  Project Memory 的身份由 `tenant_id + repository` 共同派生，同一路径被多个租户授权时也不会互相读取经验；
  User Memory 使用认证主体作为 owner。
  异步任务可通过共享 POSIX 快照目录跨主机启动，审批恢复和 Diff 可从任务快照自动恢复。原始仓库路径只继续
  用于授权、项目记忆身份和显式 Promote；Promote 仍必须在能访问原仓库的实例执行。当前 Compose 是单主机
  部署，尚未提供 S3/对象存储适配器以及跨主机 Promote 协调。
- CLI 默认使用 Docker Sandbox。`AI_HARNESS_SANDBOX_MODE=local` 仅适合测试，或 Harness 本身已经运行在受限容器
  中的场景；不要对不可信请求在宿主机启用本地模式。
- 当前数据不能支持简历中的准确率、召回率、成功率等数字；必须建立固定任务集并真实评测后才能写入。
