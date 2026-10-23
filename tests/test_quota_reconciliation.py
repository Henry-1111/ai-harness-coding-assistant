from pathlib import Path

from ai_harness.governance import GovernanceStore, TenantQuotaPolicy
from ai_harness.jobs import TaskJobStore
from ai_harness.models import TaskRequest, TaskResult, TaskStatus
from ai_harness.quota_reconciliation import QuotaReconciler


def test_reconciler_settles_terminal_jobs_and_leaves_active_jobs_reserved(
    tmp_path: Path,
) -> None:
    governance = GovernanceStore(tmp_path / "governance.sqlite3")
    jobs = TaskJobStore(tmp_path / "jobs.sqlite3")
    policy = TenantQuotaPolicy(
        tenant_id="tenant-a",
        max_outstanding_jobs=4,
        max_daily_jobs=10,
        max_daily_tokens=10_000,
        token_reservation=1_000,
    )
    request = TaskRequest(repository=str(tmp_path), goal="reconcile quota")
    completed = jobs.enqueue(request, job_id="job-completed", tenant_id="tenant-a")
    pending = jobs.enqueue(request, job_id="job-pending", tenant_id="tenant-a")
    governance.reserve_quota(job_id=completed.job_id, tenant_id="tenant-a", policy=policy)
    governance.reserve_quota(job_id=pending.job_id, tenant_id="tenant-a", policy=policy)
    assert jobs.claim("worker-a", lease_seconds=30) is not None
    assert jobs.complete(
        completed.job_id,
        "worker-a",
        TaskResult(
            task_id="task-completed",
            status=TaskStatus.SUCCEEDED,
            changed_files=[],
            verification=[],
        ),
    )

    reconciler = QuotaReconciler(
        governance,
        jobs,
        lambda job: governance.settle_quota(job.job_id, 42),
    )

    assert reconciler.reconcile_once() == 1
    assert governance.active_quota_job_ids() == [pending.job_id]
    snapshot = reconciler.snapshot()
    assert snapshot.runs == 1
    assert snapshot.inspected_reservations == 2
    assert snapshot.settled_reservations == 1
    assert snapshot.failures == 0


def test_reconciler_retries_a_transient_settlement_failure(tmp_path: Path) -> None:
    governance = GovernanceStore(tmp_path / "governance.sqlite3")
    jobs = TaskJobStore(tmp_path / "jobs.sqlite3")
    policy = TenantQuotaPolicy(
        tenant_id="tenant-a",
        max_outstanding_jobs=2,
        max_daily_jobs=10,
        max_daily_tokens=10_000,
        token_reservation=1_000,
    )
    queued = jobs.enqueue(
        TaskRequest(repository=str(tmp_path), goal="retry settlement"),
        job_id="job-retry",
        tenant_id="tenant-a",
    )
    governance.reserve_quota(job_id=queued.job_id, tenant_id="tenant-a", policy=policy)
    assert jobs.claim("worker-a", lease_seconds=30) is not None
    assert jobs.complete(
        queued.job_id,
        "worker-a",
        TaskResult(
            task_id="task-retry",
            status=TaskStatus.SUCCEEDED,
            changed_files=[],
            verification=[],
        ),
    )
    attempts = 0

    def flaky_settlement(job) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary quota backend outage")
        return governance.settle_quota(job.job_id, 17)

    reconciler = QuotaReconciler(governance, jobs, flaky_settlement)

    assert reconciler.reconcile_once() == 0
    assert governance.active_quota_job_ids() == [queued.job_id]
    assert reconciler.reconcile_once() == 1
    assert governance.active_quota_job_ids() == []
    snapshot = reconciler.snapshot()
    assert snapshot.runs == 2
    assert snapshot.failures == 1
    assert snapshot.settled_reservations == 1
