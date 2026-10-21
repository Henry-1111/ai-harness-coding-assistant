import type { CreateTaskInput, TaskResult } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function createTask(input: CreateTaskInput): Promise<TaskResult> {
  return request("/v1/tasks", { method: "POST", body: JSON.stringify(input) });
}

export function getDiff(taskId: string): Promise<{ task_id: string; diff: string }> {
  return request(`/v1/tasks/${taskId}/diff`);
}

export function promoteTask(taskId: string): Promise<{ promoted_files: string[] }> {
  return request(`/v1/tasks/${taskId}/promote`, { method: "POST" });
}
