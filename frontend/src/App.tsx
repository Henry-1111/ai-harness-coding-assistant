import { useEffect, useMemo, useState } from "react";
import {
  CheckCircleFilled,
  CodeOutlined,
  DeploymentUnitOutlined,
  PlayCircleFilled,
  SafetyCertificateOutlined,
} from "@ant-design/icons";
import Editor from "@monaco-editor/react";
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
  Row,
  Space,
  Statistic,
  Steps,
  Tag,
  Typography,
  message,
} from "antd";
import {
  cancelJob,
  createJob,
  decideApproval,
  getIdentity,
  getStoredApiKey,
  getDiff,
  getJob,
  getJobEvents,
  getQuota,
  getRuntime,
  promoteTask,
  setApiKey,
} from "./api";
import type {
  CreateTaskInput,
  JobEvent,
  PrincipalIdentity,
  RuntimeSnapshot,
  TaskJob,
  TaskResult,
  TenantQuotaSnapshot,
} from "./types";

const { Text, Title, Paragraph } = Typography;
const { TextArea } = Input;

interface FormValues {
  repository: string;
  goal: string;
  scope: string;
  acceptance: string;
  verification: string;
  maxReplans: number;
}

const splitLines = (value: string): string[] =>
  value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);

const wait = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

const collaborationReasonLabel = (reason: string): string => ({
  cross_module_goal: "跨模块任务",
  multi_path_scope: "涉及多个路径",
  broad_unbounded_scope: "复杂任务尚未限定范围",
  high_risk: "高风险变更",
  migration: "迁移任务",
  "collaboration explicitly forced": "已强制启用",
  "collaboration explicitly disabled": "已手动关闭",
  "single-agent route: no cross-module, multi-path, migration or high-risk signal":
    "未发现需要协作的复杂信号",
})[reason] ?? reason;

export default function App() {
  const [form] = Form.useForm<FormValues>();
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<TaskResult>();
  const [job, setJob] = useState<TaskJob>();
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [diff, setDiff] = useState("");
  const [error, setError] = useState("");
  const [promoted, setPromoted] = useState(false);
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [approvalActor, setApprovalActor] = useState("");
  const [approvalNote, setApprovalNote] = useState("");
  const [runtime, setRuntime] = useState<RuntimeSnapshot>();
  const [apiKey, setApiKeyValue] = useState(getStoredApiKey());
  const [authRevision, setAuthRevision] = useState(0);
  const [identity, setIdentity] = useState<PrincipalIdentity>();
  const [quota, setQuota] = useState<TenantQuotaSnapshot>();
  const [messageApi, contextHolder] = message.useMessage();

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [snapshot, currentIdentity, currentQuota] = await Promise.all([
          getRuntime().catch(() => undefined),
          getIdentity(),
          getQuota(),
        ]);
        if (active) {
          setRuntime(snapshot);
          setIdentity(currentIdentity);
          setQuota(currentQuota);
        }
      } catch {
        if (active) {
          setRuntime(undefined);
          setIdentity(undefined);
          setQuota(undefined);
        }
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [authRevision]);

  const applyApiKey = () => {
    setApiKey(apiKey);
    setAuthRevision((value) => value + 1);
  };

  const stepState = useMemo(() => {
    if (!result) return 0;
    if (result.status === "needs_input") return 0;
    const approvalOffset = result.approval ? 1 : 0;
    const planningOffset = result.route?.requires_planning ? 1 : 0;
    if (result.status === "succeeded") return 4 + planningOffset + approvalOffset;
    if (result.status === "failed") return 3 + planningOffset + approvalOffset;
    return 2 + planningOffset + approvalOffset;
  }, [result]);

  const executionSteps = useMemo(() => {
    const steps: Array<{ title: string; description: string }> = [];
    if (result?.approval) {
      steps.push({ title: "Approval", description: "Human risk gate" });
    }
    steps.push({ title: "Scout", description: "Repo Map / Context" });
    if (result?.route?.requires_planning) {
      steps.push({ title: "Plan", description: "DAG / Assignment" });
    }
    steps.push(
      { title: "Coder", description: "Structured edits" },
      { title: "Verify", description: "Tests / Lint / Build" },
      { title: "Reviewer", description: "Scope / Evidence" },
      { title: "Ready", description: "Promote gate" },
    );
    return steps;
  }, [result?.route?.requires_planning]);

  const submit = async (values: FormValues) => {
    setRunning(true);
    setError("");
    setResult(undefined);
    setJob(undefined);
    setJobEvents([]);
    setDiff("");
    setPromoted(false);
    setApprovalActor("");
    setApprovalNote("");
    const input: CreateTaskInput = {
      repository: values.repository,
      goal: values.goal,
      file_scope: splitLines(values.scope ?? ""),
      acceptance_criteria: splitLines(values.acceptance ?? "").map((description) => ({
        description,
      })),
      verification_commands: splitLines(values.verification),
      max_replans: values.maxReplans,
    };
    try {
      const idempotencyKey = window.crypto.randomUUID();
      let current = await createJob(input, idempotencyKey);
      setJob(current);
      setJobEvents(await getJobEvents(current.job_id));
      while (current.status === "pending" || current.status === "running") {
        await wait(500);
        const [nextJob, events] = await Promise.all([
          getJob(current.job_id),
          getJobEvents(current.job_id),
        ]);
        current = nextJob;
        setJob(current);
        setJobEvents(events);
      }
      if (current.result) {
        setResult(current.result);
        const taskDiff = await getDiff(current.result.task_id);
        setDiff(taskDiff.diff);
      } else if (current.status === "cancelled") {
        messageApi.info("排队任务已取消");
      } else {
        setError(current.error ?? "后台任务执行失败");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "任务执行失败");
    } finally {
      setRunning(false);
    }
  };

  const cancelActiveJob = async () => {
    if (!job) return;
    try {
      const cancelled = await cancelJob(job.job_id);
      setJob(cancelled);
      messageApi.info(
        cancelled.status === "cancelled" ? "排队任务已取消" : "已提交安全停止请求",
      );
    } catch (caught) {
      messageApi.warning(caught instanceof Error ? caught.message : "取消失败");
    }
  };

  const promote = async () => {
    if (!result) return;
    try {
      const response = await promoteTask(result.task_id);
      setPromoted(true);
      messageApi.success(
        response.promotion?.pull_request_url
          ? `已发布 ${response.promoted_files.length} 个文件并创建代码评审请求 #${response.promotion.pull_request_number}`
          : response.promotion
          ? `已发布 ${response.promoted_files.length} 个文件到分支 ${response.promotion.branch}`
          : `已发布 ${response.promoted_files.length} 个文件`,
      );
    } catch (caught) {
      messageApi.error(caught instanceof Error ? caught.message : "发布失败");
    }
  };

  const submitApproval = async (approved: boolean) => {
    if (!result || !approvalActor.trim()) {
      messageApi.warning("请填写审批人");
      return;
    }
    setApprovalBusy(true);
    try {
      const task = await decideApproval(result.task_id, {
        approved,
        actor: approvalActor.trim(),
        note: approvalNote.trim(),
      });
      setResult(task);
      if (job) {
        setJob(await getJob(job.job_id));
        setJobEvents(await getJobEvents(job.job_id));
      }
      const taskDiff = await getDiff(task.task_id);
      setDiff(taskDiff.diff);
      if (approved) {
        messageApi.success("审批通过，任务已继续执行");
      } else {
        messageApi.warning("任务已拒绝");
      }
    } catch (caught) {
      messageApi.error(caught instanceof Error ? caught.message : "审批失败");
    } finally {
      setApprovalBusy(false);
    }
  };

  return (
    <main className="app-shell">
      {contextHolder}
      <header className="hero">
        <div>
          <div className="eyebrow"><DeploymentUnitOutlined /> MULTI-AGENT CODING RUNTIME</div>
          <Title>AI Harness</Title>
          <Paragraph className="hero-copy">
            让 Agent 在真实代码仓库里规划、修改、验证，并且把每一个副作用限制在可审计边界内。
          </Paragraph>
        </div>
        <Space className="security-pill">
          <SafetyCertificateOutlined /> Docker Sandbox · Checkpoint · Review Gate
        </Space>
      </header>

      <Card
        className="panel identity-panel"
        title="请求身份"
        extra={identity ? <Tag color="green">{identity.tenant_id}</Tag> : <Tag>未认证</Tag>}
      >
        <Row gutter={[16, 12]} align="middle">
          <Col xs={24} md={10}>
            <Space.Compact block>
              <Input.Password
                value={apiKey}
                onChange={(event) => setApiKeyValue(event.target.value)}
                onPressEnter={applyApiKey}
                placeholder="API Key 或 OIDC JWT（仅保存在当前浏览器会话）"
              />
              <Button onClick={applyApiKey}>应用身份</Button>
            </Space.Compact>
          </Col>
          <Col xs={24} md={14}>
            {identity ? (
              <Space wrap>
                <Text>{identity.subject}</Text>
                {identity.roles.map((role) => <Tag key={role}>{role}</Tag>)}
                {quota && (
                  <Tag color="blue">
                    在途 {quota.outstanding_jobs}/{quota.policy.max_outstanding_jobs} · 今日任务
                    {" "}{quota.daily_jobs}/{quota.policy.max_daily_jobs} · Token
                    {" "}{quota.charged_tokens}/{quota.policy.max_daily_tokens}
                  </Tag>
                )}
              </Space>
            ) : (
              <Text type="secondary">认证开启时，请输入租户 API Key 或 OIDC JWT。</Text>
            )}
          </Col>
        </Row>
      </Card>

      <Card
        className="panel runtime-panel"
        title="运行时监控"
        extra={(
          <Tag color={runtime?.ready ? "green" : "red"}>
            {runtime?.ready ? "READY" : "NOT READY"}
          </Tag>
        )}
      >
        <Row gutter={[16, 16]}>
          <Col xs={12} md={6}>
            <Statistic
              title="排队任务"
              value={runtime?.queue.counts.pending ?? 0}
              suffix={`/ ${runtime?.queue.queue_capacity ?? 0}`}
            />
          </Col>
          <Col xs={12} md={6}>
            <Statistic
              title="运行任务"
              value={runtime?.queue.counts.running ?? 0}
            />
          </Col>
          <Col xs={12} md={6}>
            <Statistic
              title="活跃 Worker"
              value={runtime?.workers.filter((worker) => worker.state === "running").length ?? 0}
              suffix={`/ ${runtime?.workers.length ?? 0}`}
            />
          </Col>
          <Col xs={12} md={6}>
            <Statistic
              title="执行熔断"
              value={(runtime?.circuit.state ?? "unknown").toUpperCase()}
              valueStyle={{
                color: runtime?.circuit.state === "closed" ? "#52c41a" : "#ff4d4f",
              }}
            />
          </Col>
        </Row>
        {runtime?.circuit.state === "open" && (
          <Alert
            className="result-alert"
            type="error"
            showIcon
            message="任务执行熔断已打开"
            description={`连续基础设施失败 ${runtime.circuit.consecutive_failures} 次，约 ${Math.ceil(
              runtime.circuit.retry_after_seconds,
            )} 秒后进行半开探测。`}
          />
        )}
      </Card>

      <Row gutter={[20, 20]}>
        <Col xs={24} xl={9}>
          <Card className="panel" title="新建开发任务" extra={<CodeOutlined />}>
            <Form
              form={form}
              layout="vertical"
              onFinish={submit}
              initialValues={{
                verification: "pytest -q",
                maxReplans: 1,
                scope: "",
                acceptance: "",
              }}
            >
              <Form.Item label="代码仓库" name="repository" rules={[{ required: true }]}>
                <Input placeholder="/absolute/path/to/repository" />
              </Form.Item>
              <Form.Item label="开发目标" name="goal" rules={[{ required: true, min: 5 }]}>
                <TextArea rows={5} placeholder="描述要实现或修复的行为，以及不能破坏的边界。" />
              </Form.Item>
              <Form.Item label="验收条件（每行一条）" name="acceptance">
                <TextArea rows={3} placeholder="重复请求不会产生第二次扣款" />
              </Form.Item>
              <Form.Item label="允许修改的路径（每行一条）" name="scope">
                <TextArea rows={3} placeholder={"src/payments\ntests/payments"} />
              </Form.Item>
              <Form.Item label="验证命令（每行一条）" name="verification">
                <TextArea rows={3} />
              </Form.Item>
              <Form.Item label="最大 Replan 次数" name="maxReplans">
                <InputNumber min={0} max={3} />
              </Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={running}
                block
                size="large"
                icon={<PlayCircleFilled />}
              >
                在隔离工作区执行
              </Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={15}>
          <Card className="panel execution-panel" title="执行与验收">
            <Steps
              current={stepState}
              status={
                result?.status === "failed" || result?.status === "cancelled"
                  ? "error"
                  : "process"
              }
              items={executionSteps}
            />

            {error && <Alert className="result-alert" type="error" showIcon message={error} />}
            {!result && !running && (
              <div className="empty-state">
                <CodeOutlined />
                <Title level={4}>等待任务</Title>
                <Text type="secondary">Agent 的修改不会直接进入源仓库。</Text>
              </div>
            )}
            {running && (
              <div className="empty-state pulse">
                <DeploymentUnitOutlined />
                <Title level={4}>
                  {job?.status === "pending" ? "任务正在排队" : "正在编排任务"}
                </Title>
                <Space direction="vertical">
                  <Text type="secondary">
                    {job
                      ? `Job ${job.job_id.slice(0, 8)} · 第 ${job.attempts}/${job.max_attempts} 次执行`
                      : "正在创建持久化任务"}
                  </Text>
                  <Text type="secondary">Scout → Coder → Verify → Reviewer</Text>
                  {(job?.status === "pending" || job?.status === "running") && (
                    <Button
                      danger
                      disabled={job.cancel_requested}
                      onClick={cancelActiveJob}
                    >
                      {job.cancel_requested ? "正在安全停止" : "取消任务"}
                    </Button>
                  )}
                </Space>
              </div>
            )}
            {result && (
              <>
                <Divider />
                {job && (
                  <Space wrap className="file-tags">
                    <Tag color="geekblue">Job {job.job_id.slice(0, 8)}</Tag>
                    <Tag color={job.status === "succeeded" ? "green" : "orange"}>
                      {job.status}
                    </Tag>
                    <Tag>attempt {job.attempts}/{job.max_attempts}</Tag>
                  </Space>
                )}
                {jobEvents.length > 0 && (
                  <div className="routing-summary">
                    <Text strong>Job 审计时间线</Text>
                    <Space wrap>
                      {jobEvents.map((event) => (
                        <Tag key={event.sequence} color="geekblue">
                          #{event.sequence} {event.event_type} · {event.status}
                        </Tag>
                      ))}
                    </Space>
                  </div>
                )}
                {result.status === "cancelled" && (
                  <Alert
                    className="result-alert"
                    type="warning"
                    showIcon
                    message="任务已在安全边界停止"
                    description={result.error}
                  />
                )}
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic title="状态" value={result.status.toUpperCase()} />
                  </Col>
                  <Col span={6}>
                    <Statistic title="修改文件" value={result.changed_files.length} />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="验证通过"
                      value={result.verification.filter((item) => item.exit_code === 0).length}
                      suffix={`/ ${result.verification.length}`}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="上下文压缩"
                      value={Math.round((result.retrieval_trace?.context_reduction_ratio ?? 0) * 100)}
                      suffix="%"
                    />
                  </Col>
                </Row>
                {result.normalized && result.route && (
                  <div className="routing-summary">
                    <Text strong>任务规范化与路由</Text>
                    <Space wrap>
                      <Tag color="blue">{result.normalized.task_type}</Tag>
                      <Tag color="purple">{result.normalized.complexity}</Tag>
                      <Tag color={result.normalized.risk_level === "low" ? "green" : "orange"}>
                        risk: {result.normalized.risk_level}
                      </Tag>
                      <Tag color="cyan">route: {result.route.route}</Tag>
                      {result.normalized.requires_human_approval && (
                        <Tag color="red">需要人工审批</Tag>
                      )}
                    </Space>
                  </div>
                )}
                {result.collaboration_policy && (
                  <div className="routing-summary">
                    <Text strong>Agent 协作策略</Text>
                    <Space wrap>
                      <Tag color={result.collaboration_policy.enabled ? "cyan" : "green"}>
                        {result.collaboration_policy.enabled
                          ? "Implementer + Reviewer"
                          : "单 Agent"}
                      </Tag>
                      {result.collaboration_policy.enabled && (
                        <>
                          <Tag>最多 {result.collaboration_policy.max_rounds} 轮</Tag>
                          <Tag>调用预算 {result.collaboration_policy.model_call_budget}</Tag>
                        </>
                      )}
                      {result.collaboration_policy.reasons.map((reason) => (
                        <Tag key={reason}>{collaborationReasonLabel(reason)}</Tag>
                      ))}
                    </Space>
                  </div>
                )}
                {result.approval?.status === "pending" && (
                  <Alert
                    className="result-alert"
                    type="warning"
                    showIcon
                    message="高风险任务等待人工审批"
                    description={(
                      <Space direction="vertical" style={{ width: "100%" }}>
                        <Text>{result.approval.reason}</Text>
                        <Input
                          value={approvalActor}
                          onChange={(event) => setApprovalActor(event.target.value)}
                          placeholder="审批人（必填）"
                        />
                        <Input
                          value={approvalNote}
                          onChange={(event) => setApprovalNote(event.target.value)}
                          placeholder="审批说明（可选）"
                        />
                        <Space>
                          <Button
                            type="primary"
                            loading={approvalBusy}
                            onClick={() => submitApproval(true)}
                          >
                            批准并继续
                          </Button>
                          <Button
                            danger
                            disabled={approvalBusy}
                            onClick={() => submitApproval(false)}
                          >
                            拒绝任务
                          </Button>
                        </Space>
                      </Space>
                    )}
                  />
                )}
                {result.approval && result.approval.status !== "pending" && (
                  <Alert
                    className="result-alert"
                    type={result.approval.status === "approved" ? "success" : "error"}
                    showIcon
                    message={
                      result.approval.status === "approved" ? "人工审批已通过" : "人工审批已拒绝"
                    }
                    description={`审批人：${result.approval.decided_by ?? "未知"}${
                      result.approval.note ? `；说明：${result.approval.note}` : ""
                    }`}
                  />
                )}
                {result.milestones.length > 0 && (
                  <div className="checkpoint-ledger">
                    <Text strong>CP1–CP8 任务进度账本</Text>
                    <Steps
                      size="small"
                      responsive
                      items={result.milestones.map((milestone) => ({
                        title: milestone.code,
                        description: milestone.title,
                        status: milestone.status === "completed" ? "finish" : "error",
                      }))}
                    />
                  </div>
                )}
                <Space wrap className="file-tags">
                  {result.changed_files.map((path) => <Tag key={path}>{path}</Tag>)}
                </Space>
                {result.review && (
                  <Alert
                    className="result-alert"
                    type={result.review.approved ? "success" : "warning"}
                    showIcon
                    message={result.review.summary}
                    description={result.review.issues.join("；") || "验收证据完整，可以人工发布。"}
                  />
                )}
                <div className="routing-summary">
                  <Text strong>模型用量</Text>
                  <Space wrap>
                    <Tag>调用 {result.model_usage.calls}</Tag>
                    <Tag>Prompt {result.model_usage.prompt_tokens ?? "unknown"}</Tag>
                    <Tag>Completion {result.model_usage.completion_tokens ?? "unknown"}</Tag>
                    <Tag color="blue">Total {result.model_usage.total_tokens ?? "reserved"}</Tag>
                  </Space>
                </div>
                {result.repair_history.length > 0 && (
                  <div className="routing-summary">
                    <Text strong>自愈记录</Text>
                    <Space wrap>
                      {result.repair_history.map((repair) => (
                        <Tag
                          key={`${repair.attempt}-${repair.failure_fingerprint}`}
                          color={repair.can_retry ? "orange" : "red"}
                        >
                          第 {repair.attempt} 次：
                          {repair.can_retry ? "已生成修复计划" : "重复失败，已熔断"}
                        </Tag>
                      ))}
                    </Space>
                  </div>
                )}
                {result.collaboration_history.length > 0 && (
                  <div className="routing-summary">
                    <Text strong>多 Agent 协作</Text>
                    <Space wrap>
                      {result.collaboration_history.map((turn) => (
                        <Tag key={turn.round} color={turn.detected_conflicts.length ? "orange" : "cyan"}>
                          第 {turn.round} 轮：实现提案 {turn.implementation_proposal.edits.length} 项，
                          整合后 {turn.integrated_decision.edits.length} 项
                        </Tag>
                      ))}
                    </Space>
                  </div>
                )}
                <div className="diff-header">
                  <Text strong>Workspace Diff</Text>
                  <Button
                    type="primary"
                    icon={<CheckCircleFilled />}
                    disabled={result.status !== "succeeded" || promoted}
                    onClick={promote}
                  >
                    {promoted ? "已发布" : "确认并发布"}
                  </Button>
                </div>
                <div className="editor-frame">
                  <Editor
                    height="390px"
                    defaultLanguage="diff"
                    theme="vs-dark"
                    value={diff || "No diff generated."}
                    options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13 }}
                  />
                </div>
              </>
            )}
          </Card>
        </Col>
      </Row>
    </main>
  );
}
