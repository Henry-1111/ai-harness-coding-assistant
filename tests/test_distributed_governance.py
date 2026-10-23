from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from ai_harness.distributed_governance import DistributedGovernanceStore
from ai_harness.governance import (
    Principal,
    QuotaExceededError,
    Role,
    TenantQuotaPolicy,
    TenantQuotaSnapshot,
)


def test_postgres_failure_compensates_redis_reservation() -> None:
    class FakeQuota:
        released: list[str] = []

        def reserve_quota(self, **_: Any) -> TenantQuotaSnapshot:
            return TenantQuotaSnapshot(
                tenant_id="tenant-a",
                usage_date="2026-01-01",
                outstanding_jobs=1,
                daily_jobs=1,
                charged_tokens=10,
                reserved_tokens=10,
                actual_tokens=0,
                policy=TenantQuotaPolicy(tenant_id="tenant-a", token_reservation=10),
            )

        def release_quota(self, job_id: str, *, forget: bool = False) -> bool:
            self.released.append(job_id)
            assert forget is True
            return True

    class FailingPostgres:
        def reserve_quota(self, **_: Any) -> None:
            raise RuntimeError("postgres unavailable")

    store = object.__new__(DistributedGovernanceStore)
    store.quota = FakeQuota()
    store.postgres = FailingPostgres()

    with pytest.raises(RuntimeError, match="postgres unavailable"):
        store.reserve_quota(
            job_id="job-1",
            tenant_id="tenant-a",
            policy=TenantQuotaPolicy(tenant_id="tenant-a", token_reservation=10),
        )

    assert store.quota.released == ["job-1"]


POSTGRES_URL = os.getenv("AI_HARNESS_TEST_POSTGRES_URL", "")
REDIS_URL = os.getenv("AI_HARNESS_TEST_REDIS_URL", "")
requires_services = pytest.mark.skipif(
    not POSTGRES_URL or not REDIS_URL,
    reason="distributed governance integration services are not configured",
)


@requires_services
def test_two_instances_share_atomic_quota_cache_and_audit_chain() -> None:
    suffix = uuid.uuid4().hex
    tenant_id = f"tenant-{suffix}"
    prefix = f"ai-harness-test:{suffix}"
    first = DistributedGovernanceStore.from_urls(
        POSTGRES_URL, REDIS_URL, key_prefix=prefix, binding_cache_seconds=60
    )
    second = DistributedGovernanceStore.from_urls(
        POSTGRES_URL, REDIS_URL, key_prefix=prefix, binding_cache_seconds=60
    )
    policy = TenantQuotaPolicy(
        tenant_id=tenant_id,
        max_outstanding_jobs=3,
        max_daily_jobs=100,
        max_daily_tokens=1000,
        token_reservation=10,
    )

    def reserve(index: int) -> str:
        store = first if index % 2 == 0 else second
        try:
            store.reserve_quota(
                job_id=f"job-{suffix}-{index}", tenant_id=tenant_id, policy=policy
            )
            return "accepted"
        except QuotaExceededError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(reserve, range(12)))

    assert outcomes.count("accepted") == 3
    assert outcomes.count("rejected") == 9
    snapshot = second.quota_snapshot(tenant_id, policy)
    assert snapshot.outstanding_jobs == 3
    assert snapshot.charged_tokens == 30

    accepted_jobs = [
        f"job-{suffix}-{index}"
        for index, outcome in enumerate(outcomes)
        if outcome == "accepted"
    ]
    for job_id in accepted_jobs:
        first.settle_quota(job_id, 5)
    settled = second.quota_snapshot(tenant_id, policy)
    assert settled.outstanding_jobs == 0
    assert settled.actual_tokens == 15
    assert settled.charged_tokens == 15

    principal = Principal(
        key_id="integration",
        tenant_id=tenant_id,
        subject="integration-test",
        roles=frozenset({Role.SUBMITTER, Role.AUDITOR}),
        repository_roots=("*",),
    )
    resource_id = f"resource-{suffix}"
    binding = first.bind("job", resource_id, principal, repository="/tmp/repository")
    assert second.binding("job", resource_id) == binding
    assert second.redis.exists(second._binding_cache_key("job", resource_id)) == 1

    def audit(index: int) -> None:
        store = first if index % 2 == 0 else second
        store.audit(
            actor=principal.subject,
            tenant_id=tenant_id,
            action=f"integration.{index}",
            outcome="accepted",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(audit, range(20)))

    assert second.verify_integrity(principal) == (True, 20)
