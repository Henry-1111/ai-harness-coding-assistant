import { useMemo, useState } from "react";
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
import { createTask, getDiff, promoteTask } from "./api";
import type { CreateTaskInput, TaskResult } from "./types";

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

export default function App() {
  const [form] = Form.useForm<FormValues>();
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<TaskResult>();
  const [diff, setDiff] = useState("");
  const [error, setError] = useState("");
  const [promoted, setPromoted] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();

  const stepState = useMemo(() => {
    if (!result) return 0;
    if (result.status === "succeeded") return 4;
    if (result.status === "failed") return 3;
    return 2;
  }, [result]);

  const submit = async (values: FormValues) => {
    setRunning(true);
    setError("");
    setResult(undefined);
    setDiff("");
    setPromoted(false);
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
      const task = await createTask(input);
      setResult(task);
      const taskDiff = await getDiff(task.task_id);
      setDiff(taskDiff.diff);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "任务执行失败");
    } finally {
      setRunning(false);
    }
  };

  const promote = async () => {
    if (!result) return;
    try {
      const response = await promoteTask(result.task_id);
      setPromoted(true);
      messageApi.success(`已发布 ${response.promoted_files.length} 个文件`);
    } catch (caught) {
      messageApi.error(caught instanceof Error ? caught.message : "发布失败");
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
              status={result?.status === "failed" ? "error" : "process"}
              items={[
                { title: "Scout", description: "Repo Map / Context" },
                { title: "Coder", description: "Structured edits" },
                { title: "Verify", description: "Tests / Lint / Build" },
                { title: "Reviewer", description: "Scope / Evidence" },
                { title: "Ready", description: "Promote gate" },
              ]}
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
                <Title level={4}>正在编排任务</Title>
                <Text type="secondary">Scout → Coder → Verify → Reviewer</Text>
              </div>
            )}
            {result && (
              <>
                <Divider />
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
                <div className="diff-header">
                  <Text strong>Workspace Diff</Text>
                  <Button
                    type="primary"
                    icon={<CheckCircleFilled />}
                    disabled={result.status !== "succeeded" || promoted}
                    onClick={promote}
                  >
                    {promoted ? "已发布" : "确认并发布到源仓库"}
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
