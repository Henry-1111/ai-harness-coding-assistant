from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import httpx
import jwt
from pydantic import BaseModel, Field, field_validator, model_validator

from .models import utc_now


class AuthenticationError(RuntimeError):
    pass


class AuthorizationError(RuntimeError):
    pass


class QuotaExceededError(RuntimeError):
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


class TenantQuotaPolicy(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100)
    max_outstanding_jobs: int = Field(default=10, ge=1, le=100_000)
    max_daily_jobs: int = Field(default=100, ge=1, le=1_000_000)
    max_daily_tokens: int = Field(default=1_000_000, ge=1, le=10_000_000_000)
    token_reservation: int = Field(default=20_000, ge=1, le=10_000_000_000)

    @model_validator(mode="after")
    def validate_token_reservation(self) -> "TenantQuotaPolicy":
        if self.token_reservation > self.max_daily_tokens:
            raise ValueError("token_reservation cannot exceed max_daily_tokens")
        return self


class OIDCConfig(BaseModel):
    issuer: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    jwks_url: Optional[str] = None
    algorithms: tuple[str, ...] = ("RS256",)
    tenant_claim: str = "tenant_id"
    roles_claim: str = "roles"
    repository_roots_claim: str = "repository_roots"
    subject_claim: str = "sub"
    clock_skew_seconds: int = Field(default=30, ge=0, le=300)
    jwks_cache_seconds: int = Field(default=300, ge=1, le=86_400)
    request_timeout_seconds: float = Field(default=5, gt=0, le=30)

    @field_validator("issuer")
    @classmethod
    def normalize_issuer(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {"RS256", "RS384", "RS512"}
        if not values or any(value not in allowed for value in values):
            raise ValueError("OIDC algorithms must be one or more of RS256, RS384, RS512")
        return values


class GovernanceConfig(BaseModel):
    keys: list[APIKeyDefinition] = Field(default_factory=list)
    oidc: Optional[OIDCConfig] = None
    tenant_quotas: list[TenantQuotaPolicy] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> "GovernanceConfig":
        if not self.keys and self.oidc is None:
            raise ValueError("at least one API key or OIDC configuration is required")
        key_ids = [key.key_id for key in self.keys]
        secret_hashes = [key.secret_sha256 for key in self.keys]
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("governance key_id values must be unique")
        if len(secret_hashes) != len(set(secret_hashes)):
            raise ValueError("governance API key hashes must be unique")
        quota_tenants = [quota.tenant_id for quota in self.tenant_quotas]
        if len(quota_tenants) != len(set(quota_tenants)):
            raise ValueError("tenant quota policies must be unique")
        return self


@dataclass(frozen=True)
class Principal:
    key_id: str
    tenant_id: str
    subject: str
    roles: frozenset[Role]
    repository_roots: tuple[str, ...]
    authentication_enabled: bool = True
    authentication_method: str = "api_key"

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
    def __init__(
        self,
        *,
        enabled: bool,
        config: Optional[GovernanceConfig] = None,
        oidc_fetcher: Optional[Callable[[str], dict[str, Any]]] = None,
    ) -> None:
        if enabled and config is None:
            raise ValueError("governance config is required when authentication is enabled")
        self.enabled = enabled
        self.config = config
        self._oidc_validator = (
            OIDCJWTValidator(config.oidc, fetcher=oidc_fetcher)
            if enabled and config is not None and config.oidc is not None
            else None
        )

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
                authentication_method="disabled",
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
        if self._oidc_validator is not None and secret.count(".") == 2:
            return self._oidc_validator.authenticate(secret)
        raise AuthenticationError("invalid API key")

    def quota_for(self, tenant_id: str) -> TenantQuotaPolicy:
        if self.config is not None:
            for quota in self.config.tenant_quotas:
                if quota.tenant_id == tenant_id:
                    return quota
        return TenantQuotaPolicy(tenant_id=tenant_id)


class OIDCJWTValidator:
    """Validate asymmetric OIDC JWTs and cache discovery/JWKS documents."""

    def __init__(
        self,
        config: OIDCConfig,
        *,
        fetcher: Optional[Callable[[str], dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self._fetcher = fetcher or self._fetch_json
        self._jwks_url = config.jwks_url
        self._jwks: dict[str, Any] = {}
        self._cache_expires_at = 0.0
        self._lock = threading.Lock()

    def _fetch_json(self, url: str) -> dict[str, Any]:
        response = httpx.get(url, timeout=self.config.request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("OIDC endpoint returned a non-object response")
        return payload

    def _resolve_jwks_url(self) -> str:
        if self._jwks_url:
            return self._jwks_url
        discovery = self._fetcher(
            f"{self.config.issuer}/.well-known/openid-configuration"
        )
        if discovery.get("issuer", "").rstrip("/") != self.config.issuer:
            raise AuthenticationError("OIDC discovery issuer mismatch")
        jwks_url = discovery.get("jwks_uri")
        if not isinstance(jwks_url, str) or not jwks_url:
            raise AuthenticationError("OIDC discovery document has no jwks_uri")
        self._jwks_url = jwks_url
        return jwks_url

    def _load_jwks(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if not force and self._jwks and now < self._cache_expires_at:
                return self._jwks
            payload = self._fetcher(self._resolve_jwks_url())
            keys = payload.get("keys")
            if not isinstance(keys, list):
                raise AuthenticationError("OIDC JWKS response has no keys array")
            self._jwks = payload
            self._cache_expires_at = now + self.config.jwks_cache_seconds
            return payload

    def _signing_key(self, kid: str, algorithm: str) -> Any:
        for force in (False, True):
            for key in self._load_jwks(force=force).get("keys", []):
                if key.get("kid") == kid and key.get("kty") == "RSA":
                    try:
                        return jwt.PyJWK.from_dict(key, algorithm=algorithm).key
                    except (jwt.PyJWTError, ValueError, TypeError) as exc:
                        raise AuthenticationError("OIDC signing key is invalid") from exc
        raise AuthenticationError("OIDC signing key was not found")

    @staticmethod
    def _string_set(value: Any, claim_name: str) -> frozenset[str]:
        if isinstance(value, str):
            values = value.split()
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            values = value
        else:
            raise AuthenticationError(f"OIDC {claim_name} claim must contain strings")
        if not values:
            raise AuthenticationError(f"OIDC {claim_name} claim cannot be empty")
        return frozenset(values)

    def authenticate(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")
            kid = header.get("kid")
            if algorithm not in self.config.algorithms:
                raise AuthenticationError("OIDC token uses a disallowed algorithm")
            if not isinstance(kid, str) or not kid:
                raise AuthenticationError("OIDC token has no key id")
            claims = jwt.decode(
                token,
                self._signing_key(kid, algorithm),
                algorithms=list(self.config.algorithms),
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.clock_skew_seconds,
                options={
                    "require": [
                        "exp",
                        "iat",
                        "iss",
                        "aud",
                        self.config.subject_claim,
                        self.config.tenant_claim,
                        self.config.roles_claim,
                        self.config.repository_roots_claim,
                    ]
                },
            )
        except AuthenticationError:
            raise
        except (jwt.PyJWTError, httpx.HTTPError, ValueError, TypeError) as exc:
            raise AuthenticationError("invalid OIDC token") from exc

        subject = claims.get(self.config.subject_claim)
        tenant_id = claims.get(self.config.tenant_claim)
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("OIDC subject claim must be a non-empty string")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise AuthenticationError("OIDC tenant claim must be a non-empty string")
        try:
            roles = frozenset(
                Role(value)
                for value in self._string_set(
                    claims.get(self.config.roles_claim), self.config.roles_claim
                )
            )
        except ValueError as exc:
            raise AuthenticationError("OIDC token contains an unknown role") from exc
        try:
            repository_roots = APIKeyDefinition.validate_repository_roots(
                list(
                    self._string_set(
                        claims.get(self.config.repository_roots_claim),
                        self.config.repository_roots_claim,
                    )
                )
            )
        except ValueError as exc:
            raise AuthenticationError("OIDC repository roots claim is invalid") from exc
        return Principal(
            key_id=f"oidc:{kid}",
            tenant_id=tenant_id,
            subject=subject,
            roles=roles,
            repository_roots=tuple(repository_roots),
            authentication_method="oidc",
        )


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


class TenantQuotaSnapshot(BaseModel):
    tenant_id: str
    usage_date: str
    outstanding_jobs: int
    daily_jobs: int
    charged_tokens: int
    reserved_tokens: int
    actual_tokens: int
    policy: TenantQuotaPolicy


class GovernanceStoreProtocol(Protocol):
    """Persistence contract shared by local and distributed governance stores."""

    def reserve_quota(
        self,
        *,
        job_id: str,
        tenant_id: str,
        policy: TenantQuotaPolicy,
    ) -> TenantQuotaSnapshot: ...

    def settle_quota(self, job_id: str, actual_tokens: Optional[int]) -> bool: ...

    def release_quota(self, job_id: str) -> bool: ...

    def quota_snapshot(
        self,
        tenant_id: str,
        policy: TenantQuotaPolicy,
    ) -> TenantQuotaSnapshot: ...

    def bind(
        self,
        resource_type: str,
        resource_id: str,
        principal: Principal,
        *,
        repository: Optional[str] = None,
    ) -> ResourceBinding: ...

    def bind_from(
        self,
        resource_type: str,
        resource_id: str,
        owner: ResourceBinding,
    ) -> ResourceBinding: ...

    def binding(self, resource_type: str, resource_id: str) -> Optional[ResourceBinding]: ...

    def authorize(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: str,
    ) -> ResourceBinding: ...

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
    ) -> AuditEvent: ...

    def events(self, principal: Principal, *, limit: int = 100) -> list[AuditEvent]: ...

    def verify_integrity(self, principal: Principal) -> tuple[bool, int]: ...


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
                CREATE TABLE IF NOT EXISTS tenant_quota_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    usage_date TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    actual_tokens INTEGER,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    settled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_tenant_quota_usage
                    ON tenant_quota_reservations(tenant_id, usage_date, status);
                """
            )

    def reserve_quota(
        self,
        *,
        job_id: str,
        tenant_id: str,
        policy: TenantQuotaPolicy,
    ) -> TenantQuotaSnapshot:
        now = utc_now()
        usage_date = now.date().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT reservation_id FROM tenant_quota_reservations WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return self.quota_snapshot(tenant_id, policy)
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS outstanding,
                    SUM(CASE WHEN status != 'released' THEN 1 ELSE 0 END) AS daily_jobs,
                    SUM(
                        CASE
                            WHEN status = 'released' THEN 0
                            WHEN status = 'settled' THEN COALESCE(actual_tokens, reserved_tokens)
                            ELSE reserved_tokens
                        END
                    ) AS charged_tokens
                FROM tenant_quota_reservations
                WHERE tenant_id = ? AND usage_date = ?
                """,
                (tenant_id, usage_date),
            ).fetchone()
            outstanding = int(row["outstanding"] or 0)
            daily_jobs = int(row["daily_jobs"] or 0)
            charged_tokens = int(row["charged_tokens"] or 0)
            if outstanding >= policy.max_outstanding_jobs:
                connection.execute("ROLLBACK")
                raise QuotaExceededError("tenant outstanding job limit has been reached")
            if daily_jobs >= policy.max_daily_jobs:
                connection.execute("ROLLBACK")
                raise QuotaExceededError("tenant daily job quota has been exhausted")
            if charged_tokens + policy.token_reservation > policy.max_daily_tokens:
                connection.execute("ROLLBACK")
                raise QuotaExceededError("tenant daily token quota has been exhausted")
            connection.execute(
                """
                INSERT INTO tenant_quota_reservations (
                    reservation_id, job_id, tenant_id, usage_date, reserved_tokens,
                    actual_tokens, status, created_at, settled_at
                ) VALUES (?, ?, ?, ?, ?, NULL, 'active', ?, NULL)
                """,
                (
                    str(uuid.uuid4()),
                    job_id,
                    tenant_id,
                    usage_date,
                    policy.token_reservation,
                    now.isoformat(),
                ),
            )
            connection.execute("COMMIT")
        return self.quota_snapshot(tenant_id, policy)

    def settle_quota(self, job_id: str, actual_tokens: Optional[int]) -> bool:
        if actual_tokens is not None and actual_tokens < 0:
            raise ValueError("actual_tokens cannot be negative")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tenant_quota_reservations
                SET status = 'settled', actual_tokens = ?, settled_at = ?
                WHERE job_id = ? AND status = 'active'
                """,
                (actual_tokens, utc_now().isoformat(), job_id),
            )
        return cursor.rowcount == 1

    def release_quota(self, job_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tenant_quota_reservations
                SET status = 'released', settled_at = ?
                WHERE job_id = ? AND status = 'active'
                """,
                (utc_now().isoformat(), job_id),
            )
        return cursor.rowcount == 1

    def quota_snapshot(
        self,
        tenant_id: str,
        policy: TenantQuotaPolicy,
    ) -> TenantQuotaSnapshot:
        usage_date = utc_now().date().isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS outstanding,
                    SUM(CASE WHEN status != 'released' THEN 1 ELSE 0 END) AS daily_jobs,
                    SUM(
                        CASE
                            WHEN status = 'released' THEN 0
                            WHEN status = 'settled' THEN COALESCE(actual_tokens, reserved_tokens)
                            ELSE reserved_tokens
                        END
                    ) AS charged_tokens,
                    SUM(CASE WHEN status = 'active' THEN reserved_tokens ELSE 0 END)
                        AS reserved_tokens,
                    SUM(
                        CASE WHEN status = 'settled' THEN COALESCE(actual_tokens, 0) ELSE 0 END
                    ) AS actual_tokens
                FROM tenant_quota_reservations
                WHERE tenant_id = ? AND usage_date = ?
                """,
                (tenant_id, usage_date),
            ).fetchone()
        return TenantQuotaSnapshot(
            tenant_id=tenant_id,
            usage_date=usage_date,
            outstanding_jobs=int(row["outstanding"] or 0),
            daily_jobs=int(row["daily_jobs"] or 0),
            charged_tokens=int(row["charged_tokens"] or 0),
            reserved_tokens=int(row["reserved_tokens"] or 0),
            actual_tokens=int(row["actual_tokens"] or 0),
            policy=policy,
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
