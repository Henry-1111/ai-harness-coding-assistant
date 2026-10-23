from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import utc_now


class AuthenticationError(RuntimeError):
    pass


class AuthorizationError(RuntimeError):
    pass


class Role(str, Enum):
    SUBMITTER = "submitter"
    REVIEWER = "reviewer"
    PUBLISHER = "publisher"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    ADMIN = "admin"


class APIKeyDefinition(BaseModel):
    key_id: str = Field(min_length=1, max_length=100)
    secret_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_id: str = Field(min_length=1, max_length=100)
    subject: str = Field(min_length=1, max_length=200)
    roles: frozenset[Role] = Field(min_length=1)
    repository_roots: list[str] = Field(min_length=1)
    enabled: bool = True

    @field_validator("repository_roots")
    @classmethod
    def validate_repository_roots(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            if value == "*":
                normalized.append(value)
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise ValueError("repository_roots must be absolute paths or '*'")
            normalized.append(str(path.resolve()))
        return normalized


class GovernanceConfig(BaseModel):
    keys: list[APIKeyDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> "GovernanceConfig":
        key_ids = [key.key_id for key in self.keys]
        secret_hashes = [key.secret_sha256 for key in self.keys]
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("governance key_id values must be unique")
        if len(secret_hashes) != len(set(secret_hashes)):
            raise ValueError("governance API key hashes must be unique")
        return self


@dataclass(frozen=True)
class Principal:
    key_id: str
    tenant_id: str
    subject: str
    roles: frozenset[Role]
    repository_roots: tuple[str, ...]
    authentication_enabled: bool = True

    def has_any_role(self, *roles: Role) -> bool:
        return Role.ADMIN in self.roles or bool(self.roles.intersection(roles))

    def permits_repository(self, repository: str) -> bool:
        if Role.ADMIN in self.roles or "*" in self.repository_roots:
            return True
        candidate = Path(repository).expanduser().resolve()
        return any(
            candidate == Path(root) or Path(root) in candidate.parents
            for root in self.repository_roots
        )


class APIKeyAuthenticator:
    def __init__(self, *, enabled: bool, config: Optional[GovernanceConfig] = None) -> None:
        if enabled and config is None:
            raise ValueError("governance config is required when authentication is enabled")
        self.enabled = enabled
        self.config = config

    @classmethod
    def disabled(cls) -> "APIKeyAuthenticator":
        return cls(enabled=False)

    @classmethod
    def from_file(cls, path: Path) -> "APIKeyAuthenticator":
        return cls(
            enabled=True,
            config=GovernanceConfig.model_validate_json(path.read_text(encoding="utf-8")),
        )

    def authenticate(self, authorization: Optional[str]) -> Principal:
        if not self.enabled:
            return Principal(
                key_id="local-development",
                tenant_id="local",
                subject="local-development",
                roles=frozenset(Role),
                repository_roots=("*",),
                authentication_enabled=False,
            )
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("a Bearer API key is required")
        secret = authorization.removeprefix("Bearer ").strip()
        if not secret:
            raise AuthenticationError("a Bearer API key is required")
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        assert self.config is not None
        for definition in self.config.keys:
            if definition.enabled and hmac.compare_digest(digest, definition.secret_sha256):
                return Principal(
                    key_id=definition.key_id,
                    tenant_id=definition.tenant_id,
                    subject=definition.subject,
                    roles=definition.roles,
                    repository_roots=tuple(definition.repository_roots),
                )
        raise AuthenticationError("invalid API key")


class ResourceBinding(BaseModel):
    resource_type: str
    resource_id: str
    tenant_id: str
    created_by: str
    repository: Optional[str] = None
    created_at: str


class AuditEvent(BaseModel):
    event_id: str
    sequence: int
    tenant_id: Optional[str] = None
    actor: str
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    outcome: str
    details: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    event_hash: str
    created_at: str


class GovernanceStore:
    """Tenant resource ownership and append-only, hash-chained audit events."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS resource_bindings (
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    repository TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (resource_type, resource_id)
                );
                CREATE TABLE IF NOT EXISTS governance_audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    outcome TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_governance_audit_tenant_sequence
                    ON governance_audit_events(tenant_id, sequence);
                """
            )

    def bind(
        self,
        resource_type: str,
        resource_id: str,
        principal: Principal,
        *,
        repository: Optional[str] = None,
    ) -> ResourceBinding:
        created_at = utc_now().isoformat()
        normalized_repository = str(Path(repository).resolve()) if repository else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM resource_bindings WHERE resource_type = ? AND resource_id = ?",
                (resource_type, resource_id),
            ).fetchone()
            if existing is not None:
                binding = self._decode_binding(existing)
                if binding.tenant_id != principal.tenant_id:
                    connection.execute("ROLLBACK")
                    raise AuthorizationError("resource belongs to another tenant")
                connection.execute("COMMIT")
                return binding
            connection.execute(
                """
                INSERT INTO resource_bindings (
                    resource_type, resource_id, tenant_id, created_by, repository, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    resource_type,
                    resource_id,
                    principal.tenant_id,
                    principal.subject,
                    normalized_repository,
                    created_at,
                ),
            )
            connection.execute("COMMIT")
        return ResourceBinding(
            resource_type=resource_type,
            resource_id=resource_id,
            tenant_id=principal.tenant_id,
            created_by=principal.subject,
            repository=normalized_repository,
            created_at=created_at,
        )

    def bind_from(
        self,
        resource_type: str,
        resource_id: str,
        owner: ResourceBinding,
    ) -> ResourceBinding:
        principal = Principal(
            key_id="resource-inheritance",
            tenant_id=owner.tenant_id,
            subject=owner.created_by,
            roles=frozenset({Role.ADMIN}),
            repository_roots=("*",),
        )
        return self.bind(
            resource_type,
            resource_id,
            principal,
            repository=owner.repository,
        )

    def binding(self, resource_type: str, resource_id: str) -> Optional[ResourceBinding]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM resource_bindings WHERE resource_type = ? AND resource_id = ?",
                (resource_type, resource_id),
            ).fetchone()
        return self._decode_binding(row) if row else None

    def authorize(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: str,
    ) -> ResourceBinding:
        binding = self.binding(resource_type, resource_id)
        if binding is None:
            raise AuthorizationError("resource has no tenant ownership record")
        if Role.ADMIN not in principal.roles and binding.tenant_id != principal.tenant_id:
            raise AuthorizationError("resource belongs to another tenant")
        return binding

    def audit(
        self,
        *,
        actor: str,
        action: str,
        outcome: str,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        event_id = str(uuid.uuid4())
        created_at = utc_now().isoformat()
        details_json = json.dumps(
            details or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                """
                SELECT event_hash FROM governance_audit_events
                WHERE tenant_id IS ? ORDER BY sequence DESC LIMIT 1
                """,
                (tenant_id,),
            ).fetchone()
            previous_hash = str(previous["event_hash"]) if previous else "0" * 64
            event_hash = self._event_hash(
                previous_hash=previous_hash,
                event_id=event_id,
                tenant_id=tenant_id,
                actor=actor,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                outcome=outcome,
                details_json=details_json,
                created_at=created_at,
            )
            cursor = connection.execute(
                """
                INSERT INTO governance_audit_events (
                    event_id, tenant_id, actor, action, resource_type, resource_id,
                    outcome, details_json, previous_hash, event_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    tenant_id,
                    actor,
                    action,
                    resource_type,
                    resource_id,
                    outcome,
                    details_json,
                    previous_hash,
                    event_hash,
                    created_at,
                ),
            )
            sequence = int(cursor.lastrowid)
            connection.execute("COMMIT")
        return AuditEvent(
            event_id=event_id,
            sequence=sequence,
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            details=json.loads(details_json),
            previous_hash=previous_hash,
            event_hash=event_hash,
            created_at=created_at,
        )

    def events(self, principal: Principal, *, limit: int = 100) -> list[AuditEvent]:
        if not 1 <= limit <= 1000:
            raise ValueError("audit event limit must be within 1..1000")
        with self._connect() as connection:
            if Role.ADMIN in principal.roles:
                rows = connection.execute(
                    "SELECT * FROM governance_audit_events ORDER BY sequence DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM governance_audit_events
                    WHERE tenant_id = ? ORDER BY sequence DESC LIMIT ?
                    """,
                    (principal.tenant_id, limit),
                ).fetchall()
        return [self._decode_event(row) for row in rows]

    def verify_integrity(self, principal: Principal) -> tuple[bool, int]:
        with self._connect() as connection:
            if Role.ADMIN in principal.roles:
                rows = connection.execute(
                    "SELECT * FROM governance_audit_events ORDER BY tenant_id, sequence"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM governance_audit_events
                    WHERE tenant_id = ? ORDER BY sequence
                    """,
                    (principal.tenant_id,),
                ).fetchall()
        previous_by_tenant: dict[Optional[str], str] = {}
        for row in rows:
            tenant_id = row["tenant_id"]
            expected_previous = previous_by_tenant.get(tenant_id, "0" * 64)
            if row["previous_hash"] != expected_previous:
                return False, len(rows)
            calculated = self._event_hash(
                previous_hash=row["previous_hash"],
                event_id=row["event_id"],
                tenant_id=tenant_id,
                actor=row["actor"],
                action=row["action"],
                resource_type=row["resource_type"],
                resource_id=row["resource_id"],
                outcome=row["outcome"],
                details_json=row["details_json"],
                created_at=row["created_at"],
            )
            if not hmac.compare_digest(calculated, row["event_hash"]):
                return False, len(rows)
            previous_by_tenant[tenant_id] = row["event_hash"]
        return True, len(rows)

    @staticmethod
    def _event_hash(
        *,
        previous_hash: str,
        event_id: str,
        tenant_id: Optional[str],
        actor: str,
        action: str,
        resource_type: Optional[str],
        resource_id: Optional[str],
        outcome: str,
        details_json: str,
        created_at: str,
    ) -> str:
        canonical = "\n".join(
            [
                previous_hash,
                event_id,
                tenant_id or "",
                actor,
                action,
                resource_type or "",
                resource_id or "",
                outcome,
                details_json,
                created_at,
            ]
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode_binding(row: sqlite3.Row) -> ResourceBinding:
        return ResourceBinding(
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            tenant_id=row["tenant_id"],
            created_by=row["created_by"],
            repository=row["repository"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> AuditEvent:
        return AuditEvent(
            event_id=row["event_id"],
            sequence=int(row["sequence"]),
            tenant_id=row["tenant_id"],
            actor=row["actor"],
            action=row["action"],
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            outcome=row["outcome"],
            details=json.loads(row["details_json"]),
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
            created_at=row["created_at"],
        )
