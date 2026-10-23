from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import psycopg
import redis
from psycopg.rows import dict_row

from .governance import (
    AuditEvent,
    AuthorizationError,
    GovernanceStore,
    Principal,
    QuotaExceededError,
    ResourceBinding,
    Role,
    TenantQuotaPolicy,
    TenantQuotaSnapshot,
)
from .models import utc_now


class PostgresGovernanceStore:
    """Durable governance source of truth shared by all API instances."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._initialize()

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return cast(
            psycopg.Connection[dict[str, Any]],
            psycopg.connect(self.database_url, row_factory=cast(Any, dict_row)),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
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
                    sequence BIGSERIAL PRIMARY KEY,
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
                    reserved_tokens BIGINT NOT NULL,
                    actual_tokens BIGINT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    settled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_tenant_quota_usage
                    ON tenant_quota_reservations(tenant_id, usage_date, status);
                """
            )

    def reserve_quota(
        self, *, job_id: str, tenant_id: str, policy: TenantQuotaPolicy
    ) -> TenantQuotaSnapshot:
        now = utc_now()
        usage_date = now.date().isoformat()
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"quota:{tenant_id}:{usage_date}",),
            )
            existing = connection.execute(
                "SELECT 1 FROM tenant_quota_reservations WHERE job_id = %s", (job_id,)
            ).fetchone()
            if existing is None:
                row = self._quota_row(connection, tenant_id, usage_date)
                self._enforce_quota(row, policy)
                connection.execute(
                    """
                    INSERT INTO tenant_quota_reservations (
                        reservation_id, job_id, tenant_id, usage_date, reserved_tokens,
                        actual_tokens, status, created_at, settled_at
                    ) VALUES (%s, %s, %s, %s, %s, NULL, 'active', %s, NULL)
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
        return self.quota_snapshot(tenant_id, policy)

    @staticmethod
    def _quota_row(
        connection: psycopg.Connection[dict[str, Any]], tenant_id: str, usage_date: str
    ) -> Mapping[str, Any]:
        row = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'active') AS outstanding,
                COUNT(*) FILTER (WHERE status != 'released') AS daily_jobs,
                COALESCE(SUM(
                    CASE
                        WHEN status = 'released' THEN 0
                        WHEN status = 'settled' THEN COALESCE(actual_tokens, reserved_tokens)
                        ELSE reserved_tokens
                    END
                ), 0) AS charged_tokens,
                COALESCE(SUM(reserved_tokens) FILTER (WHERE status = 'active'), 0)
                    AS reserved_tokens,
                COALESCE(SUM(COALESCE(actual_tokens, 0)) FILTER (WHERE status = 'settled'), 0)
                    AS actual_tokens
            FROM tenant_quota_reservations
            WHERE tenant_id = %s AND usage_date = %s
            """,
            (tenant_id, usage_date),
        ).fetchone()
        assert row is not None
        return row

    @staticmethod
    def _enforce_quota(row: Mapping[str, Any], policy: TenantQuotaPolicy) -> None:
        if int(row["outstanding"]) >= policy.max_outstanding_jobs:
            raise QuotaExceededError("tenant outstanding job limit has been reached")
        if int(row["daily_jobs"]) >= policy.max_daily_jobs:
            raise QuotaExceededError("tenant daily job quota has been exhausted")
        if int(row["charged_tokens"]) + policy.token_reservation > policy.max_daily_tokens:
            raise QuotaExceededError("tenant daily token quota has been exhausted")

    def settle_quota(self, job_id: str, actual_tokens: Optional[int]) -> bool:
        if actual_tokens is not None and actual_tokens < 0:
            raise ValueError("actual_tokens cannot be negative")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tenant_quota_reservations
                SET status = 'settled', actual_tokens = %s, settled_at = %s
                WHERE job_id = %s AND status = 'active'
                """,
                (actual_tokens, utc_now().isoformat(), job_id),
            )
            return cursor.rowcount == 1

    def release_quota(self, job_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tenant_quota_reservations
                SET status = 'released', settled_at = %s
                WHERE job_id = %s AND status = 'active'
                """,
                (utc_now().isoformat(), job_id),
            )
            return cursor.rowcount == 1

    def quota_snapshot(
        self, tenant_id: str, policy: TenantQuotaPolicy
    ) -> TenantQuotaSnapshot:
        usage_date = utc_now().date().isoformat()
        with self._connect() as connection:
            row = self._quota_row(connection, tenant_id, usage_date)
        return TenantQuotaSnapshot(
            tenant_id=tenant_id,
            usage_date=usage_date,
            outstanding_jobs=int(row["outstanding"]),
            daily_jobs=int(row["daily_jobs"]),
            charged_tokens=int(row["charged_tokens"]),
            reserved_tokens=int(row["reserved_tokens"]),
            actual_tokens=int(row["actual_tokens"]),
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
            connection.execute(
                """
                INSERT INTO resource_bindings (
                    resource_type, resource_id, tenant_id, created_by, repository, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_type, resource_id) DO NOTHING
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
            row = connection.execute(
                "SELECT * FROM resource_bindings WHERE resource_type = %s AND resource_id = %s",
                (resource_type, resource_id),
            ).fetchone()
            assert row is not None
            binding = self._decode_binding(row)
            if binding.tenant_id != principal.tenant_id:
                raise AuthorizationError("resource belongs to another tenant")
            return binding

    def bind_from(
        self, resource_type: str, resource_id: str, owner: ResourceBinding
    ) -> ResourceBinding:
        return self.bind(
            resource_type,
            resource_id,
            Principal(
                key_id="resource-inheritance",
                tenant_id=owner.tenant_id,
                subject=owner.created_by,
                roles=frozenset({Role.ADMIN}),
                repository_roots=("*",),
            ),
            repository=owner.repository,
        )

    def binding(self, resource_type: str, resource_id: str) -> Optional[ResourceBinding]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM resource_bindings WHERE resource_type = %s AND resource_id = %s",
                (resource_type, resource_id),
            ).fetchone()
        return self._decode_binding(row) if row else None

    def authorize(
        self, principal: Principal, resource_type: str, resource_id: str
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
            details or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"audit:{tenant_id or ''}",),
            )
            previous = connection.execute(
                """
                SELECT event_hash FROM governance_audit_events
                WHERE tenant_id IS NOT DISTINCT FROM %s ORDER BY sequence DESC LIMIT 1
                """,
                (tenant_id,),
            ).fetchone()
            previous_hash = str(previous["event_hash"]) if previous else "0" * 64
            event_hash = GovernanceStore._event_hash(
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
            row = connection.execute(
                """
                INSERT INTO governance_audit_events (
                    event_id, tenant_id, actor, action, resource_type, resource_id,
                    outcome, details_json, previous_hash, event_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING sequence
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
            ).fetchone()
            assert row is not None
        return AuditEvent(
            event_id=event_id,
            sequence=int(row["sequence"]),
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
                    "SELECT * FROM governance_audit_events ORDER BY sequence DESC LIMIT %s",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM governance_audit_events
                    WHERE tenant_id = %s ORDER BY sequence DESC LIMIT %s
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
                    WHERE tenant_id = %s ORDER BY sequence
                    """,
                    (principal.tenant_id,),
                ).fetchall()
        previous_by_tenant: dict[Optional[str], str] = {}
        for row in rows:
            tenant_id = row["tenant_id"]
            expected_previous = previous_by_tenant.get(tenant_id, "0" * 64)
            if row["previous_hash"] != expected_previous:
                return False, len(rows)
            calculated = GovernanceStore._event_hash(
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
            if calculated != row["event_hash"]:
                return False, len(rows)
            previous_by_tenant[tenant_id] = row["event_hash"]
        return True, len(rows)

    @staticmethod
    def _decode_binding(row: Mapping[str, Any]) -> ResourceBinding:
        return ResourceBinding(**{key: row[key] for key in ResourceBinding.model_fields})

    @staticmethod
    def _decode_event(row: Mapping[str, Any]) -> AuditEvent:
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


RESERVE_QUOTA_LUA = """
local existing = redis.call('HGET', KEYS[2], ARGV[1])
if existing then
  return {0, redis.call('HGET', KEYS[1], 'outstanding') or 0,
    redis.call('HGET', KEYS[1], 'daily_jobs') or 0,
    redis.call('HGET', KEYS[1], 'charged_tokens') or 0,
    redis.call('HGET', KEYS[1], 'reserved_tokens') or 0,
    redis.call('HGET', KEYS[1], 'actual_tokens') or 0}
end
local outstanding = tonumber(redis.call('HGET', KEYS[1], 'outstanding') or '0')
local daily_jobs = tonumber(redis.call('HGET', KEYS[1], 'daily_jobs') or '0')
local charged = tonumber(redis.call('HGET', KEYS[1], 'charged_tokens') or '0')
if outstanding >= tonumber(ARGV[3]) then return {-1} end
if daily_jobs >= tonumber(ARGV[4]) then return {-2} end
if charged + tonumber(ARGV[2]) > tonumber(ARGV[5]) then return {-3} end
outstanding = redis.call('HINCRBY', KEYS[1], 'outstanding', 1)
daily_jobs = redis.call('HINCRBY', KEYS[1], 'daily_jobs', 1)
charged = redis.call('HINCRBY', KEYS[1], 'charged_tokens', ARGV[2])
local reserved = redis.call('HINCRBY', KEYS[1], 'reserved_tokens', ARGV[2])
redis.call('HSET', KEYS[2], ARGV[1], 'active:' .. ARGV[2])
redis.call('SET', KEYS[3], ARGV[6] .. '|' .. ARGV[7], 'EX', ARGV[8])
redis.call('EXPIRE', KEYS[1], ARGV[8])
redis.call('EXPIRE', KEYS[2], ARGV[8])
return {1, outstanding, daily_jobs, charged, reserved,
  redis.call('HGET', KEYS[1], 'actual_tokens') or 0}
"""

SETTLE_QUOTA_LUA = """
local reservation = redis.call('HGET', KEYS[2], ARGV[1])
if not reservation or string.sub(reservation, 1, 7) ~= 'active:' then return {0} end
local reserved = tonumber(string.sub(reservation, 8))
local actual = ARGV[2] == '' and reserved or tonumber(ARGV[2])
redis.call('HSET', KEYS[2], ARGV[1], 'settled:' .. actual)
redis.call('HINCRBY', KEYS[1], 'outstanding', -1)
redis.call('HINCRBY', KEYS[1], 'reserved_tokens', -reserved)
redis.call('HINCRBY', KEYS[1], 'charged_tokens', actual - reserved)
redis.call('HINCRBY', KEYS[1], 'actual_tokens', actual)
redis.call('DEL', KEYS[3])
return {1}
"""

RELEASE_QUOTA_LUA = """
local reservation = redis.call('HGET', KEYS[2], ARGV[1])
if not reservation or string.sub(reservation, 1, 7) ~= 'active:' then return {0} end
local reserved = tonumber(string.sub(reservation, 8))
if ARGV[2] == '1' then
  redis.call('HDEL', KEYS[2], ARGV[1])
else
  redis.call('HSET', KEYS[2], ARGV[1], 'released')
end
redis.call('HINCRBY', KEYS[1], 'outstanding', -1)
redis.call('HINCRBY', KEYS[1], 'daily_jobs', -1)
redis.call('HINCRBY', KEYS[1], 'reserved_tokens', -reserved)
redis.call('HINCRBY', KEYS[1], 'charged_tokens', -reserved)
redis.call('DEL', KEYS[3])
return {1}
"""


class RedisQuotaStore:
    """Atomic, cross-process quota gate backed by Redis Lua scripts."""

    def __init__(
        self, client: redis.Redis, *, key_prefix: str, ttl_seconds: int
    ) -> None:
        self.client = client
        self.key_prefix = key_prefix.rstrip(":")
        self.ttl_seconds = ttl_seconds
        self._reserve_script = client.register_script(RESERVE_QUOTA_LUA)
        self._settle_script = client.register_script(SETTLE_QUOTA_LUA)
        self._release_script = client.register_script(RELEASE_QUOTA_LUA)

    def _keys(self, tenant_id: str, usage_date: str, job_id: str) -> tuple[str, str, str]:
        base = f"{self.key_prefix}:quota:{tenant_id}:{usage_date}"
        return f"{base}:state", f"{base}:reservations", f"{self.key_prefix}:job:{job_id}"

    def reserve_quota(
        self, *, job_id: str, tenant_id: str, policy: TenantQuotaPolicy
    ) -> TenantQuotaSnapshot:
        usage_date = utc_now().date().isoformat()
        keys = self._keys(tenant_id, usage_date, job_id)
        result = cast(
            list[Any],
            self._reserve_script(
                keys=list(keys),
                args=[
                    job_id,
                    policy.token_reservation,
                    policy.max_outstanding_jobs,
                    policy.max_daily_jobs,
                    policy.max_daily_tokens,
                    tenant_id,
                    usage_date,
                    self.ttl_seconds,
                ],
            ),
        )
        code = int(result[0])
        if code == -1:
            raise QuotaExceededError("tenant outstanding job limit has been reached")
        if code == -2:
            raise QuotaExceededError("tenant daily job quota has been exhausted")
        if code == -3:
            raise QuotaExceededError("tenant daily token quota has been exhausted")
        return self._snapshot_from_values(tenant_id, usage_date, policy, result[1:])

    def _reservation_keys(self, job_id: str) -> Optional[tuple[str, str, str]]:
        index_key = f"{self.key_prefix}:job:{job_id}"
        value = cast(Optional[str], self.client.get(index_key))
        if not value:
            return None
        tenant_id, usage_date = str(value).split("|", 1)
        return self._keys(tenant_id, usage_date, job_id)

    def settle_quota(self, job_id: str, actual_tokens: Optional[int]) -> bool:
        if actual_tokens is not None and actual_tokens < 0:
            raise ValueError("actual_tokens cannot be negative")
        keys = self._reservation_keys(job_id)
        if keys is None:
            return False
        result = cast(
            list[Any],
            self._settle_script(
                keys=list(keys), args=[job_id, "" if actual_tokens is None else actual_tokens]
            ),
        )
        return int(result[0]) == 1

    def release_quota(self, job_id: str, *, forget: bool = False) -> bool:
        keys = self._reservation_keys(job_id)
        if keys is None:
            return False
        result = cast(
            list[Any],
            self._release_script(keys=list(keys), args=[job_id, 1 if forget else 0]),
        )
        return int(result[0]) == 1

    def quota_snapshot(
        self, tenant_id: str, policy: TenantQuotaPolicy
    ) -> TenantQuotaSnapshot:
        usage_date = utc_now().date().isoformat()
        state_key = self._keys(tenant_id, usage_date, "snapshot")[0]
        values = cast(
            list[Any],
            self.client.hmget(
                state_key,
                [
                    "outstanding",
                    "daily_jobs",
                    "charged_tokens",
                    "reserved_tokens",
                    "actual_tokens",
                ],
            ),
        )
        return self._snapshot_from_values(tenant_id, usage_date, policy, values)

    @staticmethod
    def _snapshot_from_values(
        tenant_id: str,
        usage_date: str,
        policy: TenantQuotaPolicy,
        values: list[Any],
    ) -> TenantQuotaSnapshot:
        counters = [int(value or 0) for value in values]
        return TenantQuotaSnapshot(
            tenant_id=tenant_id,
            usage_date=usage_date,
            outstanding_jobs=counters[0],
            daily_jobs=counters[1],
            charged_tokens=counters[2],
            reserved_tokens=counters[3],
            actual_tokens=counters[4],
            policy=policy,
        )


class DistributedGovernanceStore:
    """PostgreSQL source of truth with Redis admission control and read cache."""

    def __init__(
        self,
        postgres: PostgresGovernanceStore,
        redis_client: redis.Redis,
        *,
        key_prefix: str = "ai-harness",
        quota_ttl_seconds: int = 259_200,
        binding_cache_seconds: int = 300,
    ) -> None:
        self.postgres = postgres
        self.redis = redis_client
        self.key_prefix = key_prefix.rstrip(":")
        self.binding_cache_seconds = binding_cache_seconds
        self.quota = RedisQuotaStore(
            redis_client, key_prefix=self.key_prefix, ttl_seconds=quota_ttl_seconds
        )

    @classmethod
    def from_urls(
        cls,
        database_url: str,
        redis_url: str,
        *,
        key_prefix: str = "ai-harness",
        quota_ttl_seconds: int = 259_200,
        binding_cache_seconds: int = 300,
    ) -> "DistributedGovernanceStore":
        client = cast(
            redis.Redis,
            redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            ),
        )
        client.ping()
        return cls(
            PostgresGovernanceStore(database_url),
            client,
            key_prefix=key_prefix,
            quota_ttl_seconds=quota_ttl_seconds,
            binding_cache_seconds=binding_cache_seconds,
        )

    def reserve_quota(
        self, *, job_id: str, tenant_id: str, policy: TenantQuotaPolicy
    ) -> TenantQuotaSnapshot:
        self.quota.reserve_quota(
            job_id=job_id, tenant_id=tenant_id, policy=policy
        )
        try:
            return self.postgres.reserve_quota(
                job_id=job_id, tenant_id=tenant_id, policy=policy
            )
        except Exception:
            self.quota.release_quota(job_id, forget=True)
            raise

    def settle_quota(self, job_id: str, actual_tokens: Optional[int]) -> bool:
        durable = self.postgres.settle_quota(job_id, actual_tokens)
        operational = self.quota.settle_quota(job_id, actual_tokens)
        return durable or operational

    def release_quota(self, job_id: str) -> bool:
        durable = self.postgres.release_quota(job_id)
        operational = self.quota.release_quota(job_id)
        return durable or operational

    def quota_snapshot(
        self, tenant_id: str, policy: TenantQuotaPolicy
    ) -> TenantQuotaSnapshot:
        return self.postgres.quota_snapshot(tenant_id, policy)

    def _binding_cache_key(self, resource_type: str, resource_id: str) -> str:
        return f"{self.key_prefix}:binding:{resource_type}:{resource_id}"

    def bind(
        self,
        resource_type: str,
        resource_id: str,
        principal: Principal,
        *,
        repository: Optional[str] = None,
    ) -> ResourceBinding:
        binding = self.postgres.bind(
            resource_type, resource_id, principal, repository=repository
        )
        self._cache_binding(binding)
        return binding

    def bind_from(
        self, resource_type: str, resource_id: str, owner: ResourceBinding
    ) -> ResourceBinding:
        binding = self.postgres.bind_from(resource_type, resource_id, owner)
        self._cache_binding(binding)
        return binding

    def _cache_binding(self, binding: ResourceBinding) -> None:
        self.redis.setex(
            self._binding_cache_key(binding.resource_type, binding.resource_id),
            self.binding_cache_seconds,
            binding.model_dump_json(),
        )

    def binding(self, resource_type: str, resource_id: str) -> Optional[ResourceBinding]:
        key = self._binding_cache_key(resource_type, resource_id)
        cached = cast(Optional[str], self.redis.get(key))
        if cached:
            return ResourceBinding.model_validate_json(cached)
        binding = self.postgres.binding(resource_type, resource_id)
        if binding is not None:
            self._cache_binding(binding)
        return binding

    def authorize(
        self, principal: Principal, resource_type: str, resource_id: str
    ) -> ResourceBinding:
        binding = self.binding(resource_type, resource_id)
        if binding is None:
            raise AuthorizationError("resource has no tenant ownership record")
        if Role.ADMIN not in principal.roles and binding.tenant_id != principal.tenant_id:
            raise AuthorizationError("resource belongs to another tenant")
        return binding

    def audit(self, **kwargs: Any) -> AuditEvent:
        return self.postgres.audit(**kwargs)

    def events(self, principal: Principal, *, limit: int = 100) -> list[AuditEvent]:
        return self.postgres.events(principal, limit=limit)

    def verify_integrity(self, principal: Principal) -> tuple[bool, int]:
        return self.postgres.verify_integrity(principal)
