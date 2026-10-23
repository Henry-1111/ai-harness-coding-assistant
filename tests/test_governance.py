from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_harness.governance import (
    APIKeyAuthenticator,
    APIKeyDefinition,
    AuthenticationError,
    GovernanceConfig,
    GovernanceStore,
    Principal,
    Role,
)


def secret_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def test_api_keys_are_hashed_and_repository_scoped(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    authenticator = APIKeyAuthenticator(
        enabled=True,
        config=GovernanceConfig(
            keys=[
                APIKeyDefinition(
                    key_id="tenant-a-submitter",
                    secret_sha256=secret_hash("secret-a"),
                    tenant_id="tenant-a",
                    subject="alice",
                    roles=frozenset({Role.SUBMITTER}),
                    repository_roots=[str(allowed)],
                )
            ]
        ),
    )

    principal = authenticator.authenticate("Bearer secret-a")

    assert principal.subject == "alice"
    assert principal.permits_repository(str(allowed / "repository"))
    assert not principal.permits_repository(str(tmp_path / "outside"))
    with pytest.raises(AuthenticationError, match="invalid API key"):
        authenticator.authenticate("Bearer wrong")
    with pytest.raises(AuthenticationError, match="required"):
        authenticator.authenticate(None)


def test_resource_ownership_and_audit_hash_chain(tmp_path: Path) -> None:
    store = GovernanceStore(tmp_path / "governance.sqlite3")
    principal = Principal(
        key_id="key-a",
        tenant_id="tenant-a",
        subject="alice",
        roles=frozenset({Role.SUBMITTER, Role.AUDITOR}),
        repository_roots=("*",),
    )
    binding = store.bind("job", "job-1", principal, repository=str(tmp_path))
    first = store.audit(
        actor=principal.subject,
        tenant_id=principal.tenant_id,
        action="job.create",
        resource_type="job",
        resource_id="job-1",
        outcome="accepted",
    )
    second = store.audit(
        actor=principal.subject,
        tenant_id=principal.tenant_id,
        action="job.read",
        resource_type="job",
        resource_id="job-1",
        outcome="accepted",
    )

    assert store.authorize(principal, "job", "job-1") == binding
    assert second.previous_hash == first.event_hash
    assert [event.action for event in store.events(principal)] == ["job.read", "job.create"]
    assert store.verify_integrity(principal) == (True, 2)
