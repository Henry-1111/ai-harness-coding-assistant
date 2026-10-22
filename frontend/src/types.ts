export type TaskStatus = "pending" | "running" | "needs_input" | "succeeded" | "failed";

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
