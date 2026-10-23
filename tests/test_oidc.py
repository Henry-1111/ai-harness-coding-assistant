from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from ai_harness.governance import (
    APIKeyAuthenticator,
    APIKeyDefinition,
    AuthenticationError,
    GovernanceConfig,
    OIDCConfig,
    Role,
)

ISSUER = "https://identity.example.test"
AUDIENCE = "ai-harness"


def make_key(kid: str) -> tuple[Any, dict[str, Any]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return private_key, jwk


def make_token(
    private_key: Any,
    kid: str,
    *,
    audience: str = AUDIENCE,
    subject: str = "alice",
    expires_in_seconds: int = 300,
) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "iss": ISSUER,
            "aud": audience,
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(seconds=expires_in_seconds),
            "tenant_id": "tenant-a",
            "roles": ["submitter", "reviewer"],
            "repository_roots": ["/repositories/tenant-a"],
        },
        private_key,
        algorithm="RS256",
        headers={"kid": kid},
    )


def test_oidc_token_maps_claims_to_principal_and_uses_discovery() -> None:
    private_key, public_jwk = make_key("key-1")
    requests: list[str] = []

    def fetcher(url: str) -> dict[str, Any]:
        requests.append(url)
        if url.endswith("openid-configuration"):
            return {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        return {"keys": [public_jwk]}

    authenticator = APIKeyAuthenticator(
        enabled=True,
        config=GovernanceConfig(oidc=OIDCConfig(issuer=ISSUER, audience=AUDIENCE)),
        oidc_fetcher=fetcher,
    )

    principal = authenticator.authenticate(f"Bearer {make_token(private_key, 'key-1')}")

    assert principal.authentication_method == "oidc"
    assert principal.key_id == "oidc:key-1"
    assert principal.tenant_id == "tenant-a"
    assert principal.subject == "alice"
    assert principal.roles == frozenset({Role.SUBMITTER, Role.REVIEWER})
    assert principal.repository_roots == (str(Path("/repositories/tenant-a").resolve()),)
    assert requests == [f"{ISSUER}/.well-known/openid-configuration", f"{ISSUER}/jwks"]


def test_oidc_rejects_wrong_audience_and_disallowed_algorithm() -> None:
    private_key, public_jwk = make_key("key-1")
    authenticator = APIKeyAuthenticator(
        enabled=True,
        config=GovernanceConfig(
            oidc=OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks_url=f"{ISSUER}/jwks")
        ),
        oidc_fetcher=lambda _: {"keys": [public_jwk]},
    )

    with pytest.raises(AuthenticationError, match="invalid OIDC token"):
        authenticator.authenticate(
            f"Bearer {make_token(private_key, 'key-1', audience='another-service')}"
        )
    with pytest.raises(AuthenticationError, match="invalid OIDC token"):
        authenticator.authenticate(
            f"Bearer {make_token(private_key, 'key-1', expires_in_seconds=-300)}"
        )

    now = datetime.now(timezone.utc)
    hs_token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "alice",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        "not-a-production-secret-that-is-long-enough",
        algorithm="HS256",
        headers={"kid": "key-1"},
    )
    with pytest.raises(AuthenticationError, match="disallowed algorithm"):
        authenticator.authenticate(f"Bearer {hs_token}")


def test_unknown_kid_forces_jwks_refresh_for_key_rotation() -> None:
    first_private_key, first_public_jwk = make_key("key-1")
    second_private_key, second_public_jwk = make_key("key-2")
    jwks_requests = 0

    def fetcher(_: str) -> dict[str, Any]:
        nonlocal jwks_requests
        jwks_requests += 1
        if jwks_requests == 1:
            return {"keys": [first_public_jwk]}
        return {"keys": [first_public_jwk, second_public_jwk]}

    authenticator = APIKeyAuthenticator(
        enabled=True,
        config=GovernanceConfig(
            oidc=OIDCConfig(
                issuer=ISSUER,
                audience=AUDIENCE,
                jwks_url=f"{ISSUER}/jwks",
                jwks_cache_seconds=3600,
            )
        ),
        oidc_fetcher=fetcher,
    )

    authenticator.authenticate(f"Bearer {make_token(first_private_key, 'key-1')}")
    authenticator.authenticate(f"Bearer {make_token(first_private_key, 'key-1')}")
    rotated = authenticator.authenticate(f"Bearer {make_token(second_private_key, 'key-2')}")

    assert rotated.key_id == "oidc:key-2"
    assert jwks_requests == 2


def test_api_key_and_oidc_can_coexist() -> None:
    secret = "legacy-secret"
    authenticator = APIKeyAuthenticator(
        enabled=True,
        config=GovernanceConfig(
            keys=[
                APIKeyDefinition(
                    key_id="legacy",
                    secret_sha256=hashlib.sha256(secret.encode()).hexdigest(),
                    tenant_id="tenant-a",
                    subject="legacy-client",
                    roles=frozenset({Role.SUBMITTER}),
                    repository_roots=["*"],
                )
            ],
            oidc=OIDCConfig(
                issuer=ISSUER,
                audience=AUDIENCE,
                jwks_url=f"{ISSUER}/jwks",
            ),
        ),
        oidc_fetcher=lambda _: pytest.fail("API key authentication must not fetch JWKS"),
    )

    principal = authenticator.authenticate(f"Bearer {secret}")

    assert principal.subject == "legacy-client"
    assert principal.authentication_method == "api_key"
