export type TaskStatus =
  | "pending"
  | "running"
  | "needs_input"
  | "succeeded"
  | "failed"
  | "cancelled";
export type JobStatus = TaskStatus;

export interface PrincipalIdentity {
  key_id: string;
  tenant_id: string;
  subject: string;
  roles: string[];
  authentication_enabled: boolean;
}

export interface ModelUsage {
  calls: number;
  request_chars: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface TenantQuotaSnapshot {
  tenant_id: string;
  usage_date: string;
  outstanding_jobs: number;
  daily_jobs: number;
  charged_tokens: number;
  reserved_tokens: number;
  actual_tokens: number;
  policy: {
    max_outstanding_jobs: number;
    max_daily_jobs: number;
    max_daily_tokens: number;
    token_reservation: number;
  };
}

export interface CommandResult {
  command: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
  timed_out: boolean;
}

export interface ReviewDecision {
  approved: boolean;
  summary: string;
  issues: string[];
  findings: ReviewFinding[];
}

export interface ReviewFinding {
  category: "no_changes" | "verification" | "timeout" | "scope" | "repeated_failure";
  severity: "warning" | "error";
  message: string;
  command?: string;
  path?: string;
  evidence: string;
}

export interface RepairPlan {
  attempt: number;
  failure_fingerprint: string;
  summary: string;
  instructions: string[];
  failed_commands: string[];
  repeat_count: number;
  can_retry: boolean;
  created_at: string;
}

export interface CollaborationTurn {
  round: number;
  implementation_proposal: {
    summary: string;
    edits: Array<{ path: string; old_text: string; new_text: string }>;
  };
  integrated_decision: {
    summary: string;
    edits: Array<{ path: string; old_text: string; new_text: string }>;
  };
  detected_conflicts: string[];
  participants: Array<"implementer" | "test_integrator">;
  created_at: string;
}

export interface ApprovalRecord {
  status: "pending" | "approved" | "rejected";
  reason: string;
  requested_at: string;
  decided_at?: string;
  decided_by?: string;
  note?: string;
}

export interface RetrievalTrace {
  queries: string[];
  cache_hit: boolean;
  query_coverage: number;
  context_chars: number;
  repository_chars: number;
  context_reduction_ratio: number;
  stop_reason: string;
}

export interface AgentTraceEvent {
  node: string;
  outcome: "succeeded" | "failed" | "replan";
  duration_ms: number;
  details: Record<string, unknown>;
  created_at: string;
}

export interface NormalizedTask {
  normalized_goal: string;
  task_type: "bugfix" | "feature" | "refactor" | "migration" | "test" | "general";
  complexity: "simple" | "standard" | "complex";
  risk_level: "low" | "medium" | "high" | "critical";
  derived_acceptance: string[];
  requires_clarification: boolean;
  clarification_questions: string[];
  requires_human_approval: boolean;
}

export interface RouteDecision {
  route: "fast" | "standard" | "complex" | "human_gate";
  reasons: string[];
  requires_planning: boolean;
  required_roles: Array<"scout" | "planner" | "coder" | "reviewer">;
}

export interface TaskMilestone {
  code: `CP${1 | 2 | 3 | 4 | 5 | 6 | 7 | 8}`;
  title: string;
  status: "completed" | "failed";
  details: Record<string, unknown>;
  created_at: string;
}

export interface TaskResult {
  task_id: string;
  status: TaskStatus;
  normalized?: NormalizedTask;
  route?: RouteDecision;
  milestones: TaskMilestone[];
  changed_files: string[];
  verification: CommandResult[];
  review?: ReviewDecision;
  repair_history: RepairPlan[];
  collaboration_history: CollaborationTurn[];
  approval?: ApprovalRecord;
  model_usage: ModelUsage;
  retrieval_trace?: RetrievalTrace;
  trace: AgentTraceEvent[];
  error?: string;
}

export interface CreateTaskInput {
  repository: string;
  goal: string;
  file_scope: string[];
  acceptance_criteria: Array<{ description: string }>;
  verification_commands: string[];
  max_replans: number;
}

export interface TaskJob {
  job_id: string;
  status: JobStatus;
  request: CreateTaskInput;
  task_id?: string;
  result?: TaskResult;
  error?: string;
  attempts: number;
  max_attempts: number;
  cancel_requested: boolean;
  lease_owner?: string;
  lease_expires_at?: string;
  created_at: string;
  updated_at: string;
  idempotency_key?: string;
  tenant_id: string;
  created_by: string;
}

export interface JobEvent {
  job_id: string;
  sequence: number;
  event_type: string;
  status: JobStatus;
  worker_id?: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface RuntimeSnapshot {
  ready: boolean;
  queue: {
    counts: Record<string, number>;
    outstanding_jobs: number;
    queue_capacity: number;
    retried_jobs: number;
    cancellation_requests: number;
    oldest_pending_age_seconds: number;
  };
  circuit: {
    state: "closed" | "open" | "half_open";
    consecutive_failures: number;
    failure_threshold: number;
    recovery_seconds: number;
    retry_after_seconds: number;
  };
  workers: Array<{
    worker_id: string;
    state: string;
    current_job_id?: string;
    completed_jobs: number;
    failed_jobs: number;
    last_error?: string;
    started_at: string;
    last_seen_at: string;
  }>;
}
