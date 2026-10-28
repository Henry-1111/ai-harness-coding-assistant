# AI Harness 编程助手

## 项目介绍

AI Harness 是一个帮助开发者在代码仓库中完成修改任务的编程助手。你可以告诉它要解决什么问题、允许修改
哪些文件、怎样验证结果；它会查找相关代码、提出修改、运行测试，并把结果和差异展示给你检查。

例如，提交“修复订单重复扣款，并补充回归测试”后，你可以查看它的计划、改动和测试结果。改动先保存在
独立工作区，不会直接覆盖原仓库；只有通过检查并由人明确确认后，才会发布。任务失败时，也可以查看原因，
在限制范围内尝试修复或从记录的进度继续。

项目目前支持命令行、HTTP API 和网页控制台。简单任务可以由单个编程 Agent 完成；符合协作条件的任务
可以由实现与测试角色分别提出方案，再整合结果。它还提供任务队列、权限与额度管理、审计记录和运行监控，
便于管理多个任务。模型可通过兼容 OpenAI API 的服务接入，也可以连接本项目的 LLM Gateway。

想动手试用，可从[本地开发](#本地开发)安装，按 [CLI 示例](#cli)运行一个任务，或启动
[Web 控制台](#web-控制台)。下面的章节提供实现细节、配置方式和评测记录。

当前默认采用本地 SQLite 模式；多实例所需的 PostgreSQL、Redis、共享存储等需要另行配置，现有
Docker Compose 主要面向单主机。检索和任务成功率来自仓库内固定测试集，不代表在任意真实项目中都有
相同表现。

## 技术实现概览

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
- Python AST Repo Map，以及基于 Tree-sitter 的 JavaScript/TypeScript（含 JSX、TSX）符号与静态导入解析；
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
- 复杂任务由 Implementer 与 Test Specialist 并行形成提案，本地合并并在冲突时调用 Arbiter；
- LangGraph 驱动的 Scout、Coder、Verify、Reviewer 工作流；
- OpenAI-compatible JSON 模型适配器；
- 整批编辑预检、唯一锚点重定位、目录穿越防护、文件范围和权限检查；
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
3. 修改已有文件时优先要求 `old_text` 唯一匹配；无法唯一定位时，可使用带相似度保护的 1-based
   `start_line/end_line` 范围重定位。Python 范围必须与完整 AST 语句或同级连续语句对齐，修改后还会重新
   解析语法；范围越界、内容不相似、跨 AST 层级或语法损坏均会拒绝。一个决策中的全部编辑先完成预检，
   再统一写入，避免半应用补丁。
4. 验证命令通过 `shlex` 拆分并检查可执行文件 allowlist，不使用 `shell=True`。
5. Docker Sandbox 默认 `--network none`、`no-new-privileges`，并限制 CPU、内存和 PID。
6. 修改先保存在任务工作区；源仓库只有在 Reviewer 批准后通过显式 Promote 才会改变。

## 本地开发

真实 API Key 只放在本机环境变量、未跟踪的 `.env` 或 Secret 文件中；不要提交明文密钥或运行时数据库。

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

## 密钥文件与 Secret 挂载

敏感配置既可以使用原有环境变量，也可以使用同名的 `_FILE` 变量读取 Docker/Kubernetes Secret 挂载文件。
两种方式不能同时提供，Secret 文件必须非空且不超过 64 KiB，读取失败时服务会直接拒绝启动。例如：

```bash
export AI_HARNESS_MODEL_API_KEY_FILE=/run/secrets/model-api-key
export AI_HARNESS_GOVERNANCE_DATABASE_URL_FILE=/run/secrets/postgres-url
export AI_HARNESS_REDIS_URL_FILE=/run/secrets/redis-url
export AI_HARNESS_PROMOTION_GITHUB_TOKEN_FILE=/run/secrets/github-token
```

同样支持 Embedding API Key、S3 Access Key ID/Secret Access Key、Git 远程 URL 和 GitLab Token。
Compose 已透传上述敏感配置的 `_FILE` 变量；部署时仍需通过 Compose override、
Docker Secret、Kubernetes Secret 或平台密钥管理器把对应只读文件挂载到容器。未设置模型 Key 或 Key 文件时，
应用会在启动阶段失败，不会带空凭据运行。生产环境优先使用短期身份、Workload Identity 或 Secret 挂载，
避免把凭据写入 Compose 文件、镜像层和命令行。

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

Reviewer 会把无修改、补丁预检失败、验证失败、超时和越界修改转换为结构化 Finding。补丁无法安全
应用时不会留下部分写入；系统重新读取目标文件，将失败代码、路径、原因和最新文件片段写入
RepairPlan，并在下一轮显式注入 Coder。相同确定性失败第二次出现时立即停止自动修复，避免无效循环
耗尽模型额度；`max_replans` 仍是总重试上限。

终态任务会把结果、验证状态、修复次数和失败指纹写入项目级 `experiences` 记忆。记录按 task ID
幂等写入并最多保留 20 条；后续 Coder 只读取最近 5 条结构化经验，不保存模型思维链或 Provider 密钥。

## 多 Agent 协作

多 Agent 不再仅由 `complex` 标签直接触发。`auto` 策略会综合跨模块目标、多路径范围、迁移、高风险和
未限定范围的复杂任务；没有这些协作收益信号时，即使是单文件重构也走单 Coder，避免无意义地增加 Token
与尾延迟。`forced` 和 `disabled` 模式仍可用于 A/B 评测。

启用后，Code 阶段展开为 `implementation_* ∥ test_design → integration → code`：系统按显式文件范围动态
生成最多 3 个 Implementer 子任务和 1 个 Test Specialist，并基于同一冻结上下文并行生成相互独立的提案；
没有竞争修改时由本地确定性合并，只有同一锚点或重叠行范围出现不同替换时才调用 Arbiter。协作策略会持久化
启用原因、最大并行 Agent 数、最大轮数和全局模型调用预算；默认最多两轮、每轮最多 4 个并行 Agent，预算
耗尽后，后续自愈自动降级为单 Coder。
候选提案不会直接写文件；只有经过整合且不存在竞争替换的最终编辑，才能进入 `apply_edits` 整批预检。

每轮还会把任务范围拆成生产代码与测试代码两份 Assignment：Implementer 只能修改生产范围，Test
Specialist 只能修改显式分配的测试路径；任务未提供测试路径时，Test Specialist 自动降为只读建议角色。
每个提案在整合前都经过路径边界校验，越权或结构化输出失败时只重试失败角色一次，已经成功的并行提案不会
重复调用。角色范围、每个角色的尝试次数和是否触发仲裁都会写入 Checkpoint、执行 Trace，并显示在控制台。
每个动态子任务有独立的模型调用额度，默认允许首次调用和一次隔离重试；整轮额外预留一次条件仲裁。因此
单轮最坏情况为 4 次首次调用、4 次隔离重试和 1 次仲裁，全局策略预算按两轮最多 18 次模型调用预留。

实验性的同子任务多方案竞争通过 `AI_HARNESS_MULTI_PROPOSAL_ENABLED=true` 开启（默认关闭，避免无意增加模型
费用）。当并行名额和本轮额度足够时，第一个可编辑 Implementer 会额外独立生成一个候选；本地 Judge 先拒绝
越权、结构错误与内部冲突，再按编辑数、新文件数和变更字符数给出可解释的静态风险/成本分数，同分保留原方案。
原始提案、通过校验的备选提案、分数、拒绝原因与最终选择均写入 Checkpoint；评分和选择也写入 Trace。评分
**不证明功能正确**，不能替代后续
补丁预检、隔离验证与 Reviewer；应先在冻结任务集 A/B 评测成功率和 Token 成本，再考虑默认开启。

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
export AI_HARNESS_API_MAX_REQUEST_BODY_BYTES=1048576
```

容量上限统计 `pending/running/needs_input` 三类未完成任务，达到上限时新建 Job 返回 HTTP 429，防止
模型调用和本地执行资源被无限排队拖垮。总执行时限覆盖完整 Agent 流程；单次模型请求和 Sandbox 命令
仍保留各自更短的超时。`GET /v1/jobs/metrics` 返回各状态数量、重试任务数、取消请求数、最老排队时间和
当前容量，用于控制台或监控系统采集。

API 请求体默认限制为 1 MiB，可通过 `AI_HARNESS_API_MAX_REQUEST_BODY_BYTES` 调整，最大允许配置为 64 MiB。
服务会先拒绝超限的 `Content-Length`，并在没有长度头的分块请求中逐块累计，超过上限立即返回 HTTP 413，
避免攻击者通过省略长度头绕过保护。非法或负数 `Content-Length` 返回 HTTP 400；拒绝结果同样携带请求关联 ID、
安全响应头并进入按状态码聚合的 HTTP 指标。

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

项目提供 40 个带人工相关文件标注的仓库内冻结任务。数据文件带版本元数据，加载时会拒绝重复 ID、
重复 Query、越界路径、缺失标注文件和直接泄漏答案路径的 Query；回归测试固定规范化数据集的
SHA-256，标签发生变化时必须显式审查并更新指纹。

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
- `dataset_sha256`：规范化 Case 集指纹，用于确认不同实验使用相同标签。

`tests/test_retrieval_benchmark.py` 同时设置数量、覆盖面、数据指纹和指标回归下限。仓库附带的历史
40 Case 报告记录 Recall@5 `97.5%`、MRR `85.08%`、NDCG@5 `87.95%`、Hit Rate `97.5%`，原始逐 Case
结果见 [`benchmarks/latest-report.json`](benchmarks/latest-report.json)。代码和索引文件变化后，指标需要重新
运行上面的命令核对；这些数字只描述本仓库的 Python 文件级检索，不能表述成跨仓库或跨语言的通用准确率。

## 路由与 Plan DAG 评测

项目提供 30 个冻结任务，人工标注 Task Type、复杂度、风险等级和预期 Route，并对每个生成 Plan 验证：

- 必须包含 Scout、Code、Verify 和 Review；
- Standard/Complex/Human Gate 必须包含 Plan，Fast Route 不应产生多余 Plan；
- 具有跨模块、多路径、迁移或高风险信号的任务必须包含并行 Implementer、Test Specialist 与 Integration 步骤；
- DAG 必须无环、依赖顺序正确，且每个 Step 都有且只有一个 Assignment。

```bash
ai-harness eval-planning benchmarks/planning_cases.json \
  --split test \
  --output benchmarks/planning-report.json
```

当前冻结集实测 Route Accuracy 为 `96.67%`，Planning Decision Accuracy 为 `100%`，Plan 结构合法率为
`100%`。报告保留 `migrate` 复杂度误判 Case，没有通过删除失败样本或测试后修改标签制造 100% 路由准确率。
原始结果见 [`benchmarks/planning-report.json`](benchmarks/planning-report.json)。这些数字衡量确定性规则在本项目
冻结任务上的表现，不代表开放领域自然语言路由准确率。

## Checkpoint 故障恢复评测

Checkpoint 基准会并发写入多任务、多修订状态，销毁原 Store 实例后由新实例重新打开持久化数据库，验证
最新状态、完整历史修订和连续 Revision；同时注入损坏 JSON 记录，确认坏任务读取失败时不会影响健康任务。

```bash
ai-harness eval-checkpoints \
  --tasks 200 \
  --revisions 8 \
  --writers 8 \
  --corruption-trials 20 \
  --output benchmarks/checkpoint-report.json
```

当前本机 SQLite 基准共写入 1,600 个 Checkpoint，新进程恢复率 `100%`、历史修订完整率 `100%`、
损坏记录隔离率 `100%`，恢复延迟 P50/P95/P99 为 `0.086/0.167/0.421 ms`。原始报告见
[`benchmarks/checkpoint-report.json`](benchmarks/checkpoint-report.json)。这些数字支持本地持久化恢复能力，
不等同于机器宕机后的磁盘耐久性，也不能替代 PostgreSQL 跨节点故障演练。

## 安全攻击评测

冻结安全集包含 48 个攻击 Case，覆盖仓库路径穿越、任务 Scope 越权写入、非白名单命令、非法 Snapshot ID、
Tar 路径穿越、恶意符号链接以及 Docker 隔离参数缺失。每次文件攻击都在临时目录中运行，并检查仓库外
哨兵文件和 Scope 外文件是否发生副作用。

```bash
ai-harness eval-security benchmarks/security_cases.json \
  --output benchmarks/security-report.json
```

当前结果为 48/48 阻断，阻断率 `100%`，观察到的越权写入和宿主路径逃逸均为 `0`。原始逐 Case 结果见
[`benchmarks/security-report.json`](benchmarks/security-report.json)。其中 Docker Case 验证启动参数包含
`--network none`、只读根文件系统、资源限制、`no-new-privileges` 和删除 Linux capabilities；该结果不是
专业容器逃逸红队认证，也不表示系统能够防御未知内核漏洞。

## 单 Agent / 多 Agent A/B 评测

项目提供成对评测命令：同一模型、同一 Provider、同一冻结任务集分别走单 Coder 和
Implementer + Test/Integrator 两条路径。每个 Case 都使用全新的隔离工作区，且运行顺序交替，避免
前后顺序持续偏向某一组。评测不会执行 Promote，因此不会修改被测源仓库。

仓库内附带 30 个冻结的确定性缺陷任务，覆盖算术与边界、集合处理、解析规范化、状态策略、重试、
配额和配置优先级。每个 Case 都有独立的失败验证命令；数据集回归测试会确认所有缺陷在基线 Fixture
中确实失败，避免把已经通过的无效任务计入成功率。使用宿主机本地执行器只适用于这组可信 Fixture：

```bash
export AI_HARNESS_SANDBOX_MODE=local
export AI_HARNESS_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export AI_HARNESS_MODEL_API_KEY='<llmgw-api-key>'
export AI_HARNESS_MODEL=gpt-4.1-mini

ai-harness eval-agents \
  benchmarks/fixtures/agent_eval \
  benchmarks/agent_cases.json \
  --split test \
  --repetitions 3 \
  --approve-trusted-fixtures \
  --output benchmarks/agent-report.json
```

报告包含两组成功率、耗时、模型调用次数、请求字符数、Provider Token、修复轮数、协作轮数、冲突数、
预期修改文件的 Recall/Precision，以及逐 Case 的胜/平/负。只有上游 OpenAI-compatible 接口返回
`usage` 时才统计 Token；否则字段为 `null`，不会用字符数伪造 Token。报告同时写入冻结数据集的
SHA-256，便于确认不同实验使用的是同一版本。`--repetitions` 会对每个 Case 重复运行，并在相邻轮次
交换 single/multi 的先后顺序；报告记录 `run_index`、重复次数和每个 Variant 的观测数量。

`--approve-trusted-fixtures` 只适用于仓库内这组经过审查的本地 Fixture。它不会关闭审批门：任务仍会先进入
`needs_input` 并写入审批记录，然后由评测执行器以 `benchmark-runner` 身份显式批准。评测外的任务以及未提供
该开关的评测仍必须由正常治理接口审批。

这 30 个任务仍是仓库内可控 Fixture，不等于外部真实项目泛化能力。引用时应同时报告成功率、
Token、时延和失败类型，而不能只挑最佳轮次。补丁前 30 Case、3 次重复的 DeepSeek 基线结果见
[`benchmarks/AGENT_RESULTS.md`](benchmarks/AGENT_RESULTS.md)，对应的 180 条原始观测见
[`benchmarks/agent-report.json`](benchmarks/agent-report.json)。本次实测成功率约为 48%；报告同时保留了
精确编辑失败、验证失败、审批暂停和上游超时的分布。早期 3 Case 冒烟测试不代表完整任务集表现。

针对历史编辑失败 Case 的补丁回归结果见
[`benchmarks/AGENT_REGRESSION_RESULTS.md`](benchmarks/AGENT_REGRESSION_RESULTS.md)：10 Case 单轮相对同一
子集历史基线有明显提升，因此继续触发了 30 Case 单轮验证；后者因 DeepSeek 余额耗尽、上游返回 402 而中止，
受污染报告保留为 `agent-report-single-pass-aborted.json`，不能作为整体成功率引用。补充余额后从头完成的
30 Case 单轮结果为单 Agent 83.33%、多 Agent 63.33%，原始观测见 `agent-report-single-pass.json`。

补丁后三轮正式结果见 [`benchmarks/AGENT_THREE_RUN_RESULTS.md`](benchmarks/AGENT_THREE_RUN_RESULTS.md)，聚合 JSON
见 `benchmarks/agent-report-three-runs.json`：每个变体共 90 个观测，单 Agent 原始成功率 74.44%、Provider 调整后
诊断成功率 87.01%、P95 66.5 秒、平均已知 Token 3,601；多 Agent 原始成功率 65.56%，成本和尾延迟均更高。
这些数字仅适用于固定 30 Case 本地冻结任务集和对应的三轮运行配置。

针对 AST/行范围安全重定位的同一 10 Case 单轮回归见
[`benchmarks/AGENT_LINE_RANGE_RESULTS.md`](benchmarks/AGENT_LINE_RANGE_RESULTS.md)。多 Agent 原始成功率从 20%
升至 30%，编辑应用失败从 5 个降至 1 个；单 Agent 原始成功率仍为 30%，结果并不支持宣称整体成功率已稳定
提升。后续评测报告会额外记录精确锚点、规范化锚点、行范围和新文件策略的实际次数，以及具体拒绝码。

新版并行提案多 Agent 经本地 LLM Gateway 的完整 30 Case 单轮测试见
[`benchmarks/AGENT_GATEWAY_FULLFLOW_2026-10-27.md`](benchmarks/AGENT_GATEWAY_FULLFLOW_2026-10-27.md)：
单 Agent 成功 20/30，多 Agent 成功 23/30，成对比较为多 Agent 4 胜、单 Agent 1 胜、25 平；
但多 Agent 的 P95 耗时和模型调用次数更高，且 Gateway 有 5 次上游请求超时。这个单轮结果只是正向信号，
不能与历史三轮结果直接合并，也不能证明稳定提升；仍需在相同配置下补足重复实验。

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

多主机不具备共享 POSIX 文件系统时，可把同一接口切换到 S3 或 MinIO：

```bash
export AI_HARNESS_WORKSPACE_SNAPSHOT_S3_BUCKET=ai-harness-snapshots
export AI_HARNESS_WORKSPACE_SNAPSHOT_S3_PREFIX=production/workspaces
# MinIO 等 S3-compatible 服务才需要自定义 endpoint；AWS S3 留空。
export AI_HARNESS_WORKSPACE_SNAPSHOT_S3_ENDPOINT=http://minio:9000
export AI_HARNESS_WORKSPACE_SNAPSHOT_S3_REGION=us-east-1
export AI_HARNESS_WORKSPACE_SNAPSHOT_S3_ACCESS_KEY_ID='<access-key>'
export AI_HARNESS_WORKSPACE_SNAPSHOT_S3_SECRET_ACCESS_KEY='<secret-key>'
```

配置 Bucket 后，对象存储成为权威副本，`AI_HARNESS_WORKSPACE_SNAPSHOT_DIR` 变为每个实例的有界本地缓存；
未显式配置访问密钥时使用 AWS SDK 默认凭据链，生产环境优先采用实例角色或 Workload Identity。上传使用
条件写避免同一 Job 快照被覆盖，更新后的 ETag 会使其他主机自动刷新缓存；下载校验对象元数据中的 SHA-256，
并采用拒绝路径穿越、越界符号链接、设备文件及硬链接的安全解包流程。本地缓存达到上限时按最近最少使用顺序淘汰旧条目，
不会删除 S3 权威副本；快照容量监控反映本机缓存使用率。对象存储的生命周期与版本策略仍应在 Bucket 侧配置。

Promote 会为每个首次修改的文件持久化 SHA-256 基线。正式写回前先检查全部目标文件：源文件仍等于基线才
允许发布，已经等于任务结果则按幂等成功处理，其他情况返回冲突且不写入任何文件，避免覆盖任务执行期间的
人工或其他 Agent 修改。写回使用同目录临时文件与原子替换，进程中断后可安全重试；共享 Git 仓库还通过
`.git/ai-harness-promote.lock` 协调多实例 Promote。

当 Worker 无法挂载原始仓库时，可以把 Promote 切换成远程 Git 分支发布：

```bash
export AI_HARNESS_PROMOTION_MODE=git_branch
export AI_HARNESS_PROMOTION_GIT_REMOTE_URL=git@github.com:example/project.git
export AI_HARNESS_PROMOTION_GIT_BASE_REF=main
export AI_HARNESS_PROMOTION_GIT_BRANCH_PREFIX=ai-harness
# 可选：分支发布后自动创建或复用 GitHub Draft Pull Request。
export AI_HARNESS_PROMOTION_GITHUB_REPOSITORY=example/project
export AI_HARNESS_PROMOTION_GITHUB_TOKEN='<github-app-or-fine-grained-token>'
export AI_HARNESS_PROMOTION_GITHUB_DRAFT=true
# 可选：GitHub API 超时、最大尝试次数与指数退避基数。
export AI_HARNESS_PROMOTION_GITHUB_TIMEOUT_SECONDS=30
export AI_HARNESS_PROMOTION_GITHUB_MAX_ATTEMPTS=3
export AI_HARNESS_PROMOTION_GITHUB_RETRY_BACKOFF_SECONDS=0.25
```

GitLab.com 或自建 GitLab 可以改用 Merge Request 集成；GitHub 与 GitLab 配置互斥：

```bash
export AI_HARNESS_PROMOTION_GITLAB_PROJECT=example/platform/project
export AI_HARNESS_PROMOTION_GITLAB_TOKEN='<project-or-personal-access-token>'
# 自建 GitLab 时替换为 https://gitlab.example.com/api/v4
export AI_HARNESS_PROMOTION_GITLAB_API_URL=https://gitlab.com/api/v4
export AI_HARNESS_PROMOTION_GITLAB_DRAFT=true
export AI_HARNESS_PROMOTION_GITLAB_TIMEOUT_SECONDS=30
export AI_HARNESS_PROMOTION_GITLAB_MAX_ATTEMPTS=3
export AI_HARNESS_PROMOTION_GITLAB_RETRY_BACKOFF_SECONDS=0.25
```

Publisher 只会创建 `ai-harness/{task_id}` 独立分支，不直接写主分支；凭据由 SSH Agent、Git Credential
Helper 或运行环境提供，配置和日志不保存 Token。相同任务重复发布且内容一致时返回同一 Commit；多个
API 实例并发发布同一结果时，先写入者创建分支，其他实例校验远程内容后按幂等重放成功处理；若远程
分支已存在但内容不同则拒绝覆盖。使用 SSH 时还应以只读方式挂载预置的 `known_hosts`，不要关闭主机校验。
API 与控制台会返回分支名、Commit 和代码评审请求信息。启用 GitHub 集成后，Publisher 会先查询同一
`head + base` 的开放 PR：存在则幂等复用，不存在才创建；并发创建返回 422 时会再次查询并接纳胜出者。
请求使用版本化 GitHub REST API，Token 只进入 Authorization Header，不写入任务状态、回执或错误日志。
网络异常、429 与暂时性 5xx 会在有界次数内指数退避重试；服务端返回 `Retry-After` 时优先采用并限制单次等待，
避免代码托管平台短暂抖动直接使已完成任务发布失败。即使创建 PR 的响应丢失，重试得到 422 后也会再次查询并
复用已经创建的 PR，保持调用幂等。
推荐使用仅具有目标仓库 Pull requests 读写权限的 GitHub App Installation Token 或
Fine-grained Token。未配置任何代码托管 API 变量时仍保持纯分支发布，可由 CI 创建评审请求。
GitLab 集成采用 URL 编码后的项目路径调用 v4 Merge Requests API，使用 `PRIVATE-TOKEN` Header；它同样会
先按 source/target branch 查询开放 MR，创建冲突或响应丢失时再次查询复用，并对网络异常、429 和暂时性
5xx 执行有界重试。建议使用只具有目标项目 API 权限的 Project Access Token。

### Prometheus 与 Grafana

监控栈使用可选 Compose profile，不会增加默认开发环境的启动负担：

```bash
docker compose --profile monitoring up --build
```

Prometheus 位于 `http://127.0.0.1:9090`，Grafana 位于 `http://127.0.0.1:3000`。Grafana 会自动加载
Prometheus 数据源和 `AI Harness Overview` 看板，展示 Readiness、依赖健康、队列积压、Worker、熔断状态、
HTTP RPS/P95 延迟、任务状态、快照容量与额度对账失败。Promote 另外记录本地/远程模式、成功、幂等重放、
冲突和系统错误，以及发布耗时直方图；失败尝试会进入审计链。Prometheus 同时预置依赖不可用、熔断打开、
队列阻塞、无 Worker、快照容量过高、发布失败、额度对账失败和 HTTP 5xx 比例过高等告警规则。

该 profile 默认采用只读匿名看板，适用于本机或受保护的内部网络；对外部署前应关闭 Grafana 匿名访问，接入
统一身份认证，并为 Prometheus 抓取 `/metrics` 配置具有 `operator` 角色的凭据。

## 项目边界

- 当前 Repo Map 对 Python 提供 AST 级符号解析，对 JavaScript/TypeScript（含 JSX、TSX、
  `.mjs/.cjs/.mts/.cts`）提供 Tree-sitter 符号与静态导入解析。Go、Java 等其他语言仍主要依赖
  文件和文本级索引；LSP 定义召回已接入 Scout，但具体语言服务器仍需部署方通过
  `AI_HARNESS_LSP_COMMAND` 配置。动态导入和计算属性名不计入静态结构索引。
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
  用于授权和项目记忆身份。共享 POSIX Git 仓库可通过文件锁协调原地 Promote；不能挂载原仓库的 Worker
  可发布到远程 Git 独立分支，并选择性创建 GitHub Pull Request 或 GitLab Merge Request。任务快照已支持
  S3/MinIO，但当前 Compose 编排仍是单主机。
- CLI 默认使用 Docker Sandbox。`AI_HARNESS_SANDBOX_MODE=local` 仅适合测试，或 Harness 本身已经运行在受限容器
  中的场景；不要对不可信请求在宿主机启用本地模式。
- 固定任务集已有可复核的检索与 Agent 评测数据，但仅支持注明数据集、运行轮数与 Provider 故障口径的有限结论；
  不可将仓库内 Fixture 的结果泛化为真实外部项目，也不可把新版多 Agent 的单轮成绩写成稳定成功率。

## 许可证

本项目代码采用 [MIT License](LICENSE)。
