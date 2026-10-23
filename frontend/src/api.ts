import type {
  CreateTaskInput,
  JobEvent,
  PrincipalIdentity,
  RuntimeSnapshot,
  TenantQuotaSnapshot,
  TaskJob,
  TaskResult,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
let apiKey = window.sessionStorage.getItem("ai-harness-api-key") ?? "";

export function setApiKey(value: string): void {
  apiKey = value.trim();
  if (apiKey) {
    window.sessionStorage.setItem("ai-harness-api-key", apiKey);
  } else {
    window.sessionStorage.removeItem("ai-harness-api-key");
  }
}

export function getStoredApiKey(): string {
  return apiKey;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getIdentity(): Promise<PrincipalIdentity> {
  return request("/v1/auth/me");
}

export function getQuota(): Promise<TenantQuotaSnapshot> {
  return request("/v1/quota");
}

export function createTask(input: CreateTaskInput): Promise<TaskResult> {
  return request("/v1/tasks", { method: "POST", body: JSON.stringify(input) });
}

export function createJob(input: CreateTaskInput, idempotencyKey: string): Promise<TaskJob> {
  return request("/v1/jobs", {
    method: "POST",
    headers: { "Idempotency-Key": idempotencyKey },
    body: JSON.stringify(input),
  });
}

export function getJob(jobId: string): Promise<TaskJob> {
  return request(`/v1/jobs/${jobId}`);
}

export function getJobEvents(jobId: string): Promise<JobEvent[]> {
  return request(`/v1/jobs/${jobId}/events`);
}

export function cancelJob(jobId: string): Promise<TaskJob> {
  return request(`/v1/jobs/${jobId}/cancel`, { method: "POST" });
}

export function getRuntime(): Promise<RuntimeSnapshot> {
  return request("/v1/runtime");
}

export function getDiff(taskId: string): Promise<{ task_id: string; diff: string }> {
  return request(`/v1/tasks/${taskId}/diff`);
}

export function promoteTask(taskId: string): Promise<{ promoted_files: string[] }> {
  return request(`/v1/tasks/${taskId}/promote`, { method: "POST" });
}

export function decideApproval(
  taskId: string,
  decision: { approved: boolean; actor: string; note: string },
): Promise<TaskResult> {
  return request(`/v1/tasks/${taskId}/approval`, {
    method: "POST",
    body: JSON.stringify(decision),
  });
}
