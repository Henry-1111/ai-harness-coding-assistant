from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field

HTTP_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
PROMOTION_LATENCY_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
HTTP_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})
PROMOTION_MODES = frozenset({"git_branch", "local"})
PROMOTION_OUTCOMES = frozenset({"conflict", "error", "not_found", "replay", "success"})
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def normalize_http_method(method: str) -> str:
    normalized = method.upper()
    return normalized if normalized in HTTP_METHODS else "OTHER"


def normalize_request_id(value: str | None) -> str:
    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid.uuid4().hex


def format_http_access_log(
    *,
    request_id: str,
    method: str,
    route: str,
    status_code: int,
    duration: float,
) -> str:
    return json.dumps(
        {
            "duration_ms": round(max(0.0, duration) * 1000, 3),
            "event": "http_request",
            "method": normalize_http_method(method),
            "request_id": request_id,
            "route": route,
            "status_code": status_code,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def format_worker_event(
    *,
    event: str,
    request_id: str | None,
    job_id: str,
    worker_id: str,
    status: str,
    attempt: int,
) -> str:
    return json.dumps(
        {
            "attempt": attempt,
            "event": event,
            "job_id": job_id,
            "request_id": request_id,
            "status": status,
            "worker_id": worker_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass
class _HTTPSeries:
    count: int = 0
    duration_sum: float = 0.0
    buckets: list[int] = field(
        default_factory=lambda: [0 for _ in HTTP_LATENCY_BUCKETS]
    )


@dataclass
class _PromotionSeries:
    count: int = 0
    duration_sum: float = 0.0
    buckets: list[int] = field(
        default_factory=lambda: [0 for _ in PROMOTION_LATENCY_BUCKETS]
    )


class HTTPMetrics:
    """Thread-safe, bounded-label HTTP counters and latency histograms."""

    def __init__(self) -> None:
        self._series: dict[tuple[str, str, int], _HTTPSeries] = {}
        self._lock = threading.Lock()

    def observe(self, method: str, route: str, status_code: int, duration: float) -> None:
        key = (normalize_http_method(method), route, status_code)
        with self._lock:
            series = self._series.setdefault(key, _HTTPSeries())
            series.count += 1
            series.duration_sum += max(0.0, duration)
            for index, upper_bound in enumerate(HTTP_LATENCY_BUCKETS):
                if duration <= upper_bound:
                    series.buckets[index] += 1

    def render_prometheus(self) -> list[str]:
        lines = [
            "# HELP ai_harness_http_requests_total HTTP requests by route and status.",
            "# TYPE ai_harness_http_requests_total counter",
            "# HELP ai_harness_http_request_duration_seconds HTTP request latency.",
            "# TYPE ai_harness_http_request_duration_seconds histogram",
        ]
        with self._lock:
            snapshot = [
                (key, value.count, value.duration_sum, list(value.buckets))
                for key, value in sorted(self._series.items())
            ]
        for (method, route, status_code), count, duration_sum, buckets in snapshot:
            labels = (
                f'method="{self._escape(method)}",'
                f'route="{self._escape(route)}",'
                f'status="{status_code}"'
            )
            lines.append(f"ai_harness_http_requests_total{{{labels}}} {count}")
            for upper_bound, bucket_count in zip(HTTP_LATENCY_BUCKETS, buckets):
                lines.append(
                    "ai_harness_http_request_duration_seconds_bucket"
                    f'{{{labels},le="{upper_bound:g}"}} {bucket_count}'
                )
            lines.append(
                "ai_harness_http_request_duration_seconds_bucket"
                f'{{{labels},le="+Inf"}} {count}'
            )
            lines.append(
                f"ai_harness_http_request_duration_seconds_sum{{{labels}}} "
                f"{duration_sum:.9f}"
            )
            lines.append(
                f"ai_harness_http_request_duration_seconds_count{{{labels}}} {count}"
            )
        return lines

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class PromotionMetrics:
    """Thread-safe promotion outcomes with fixed-cardinality labels."""

    def __init__(self) -> None:
        self._series: dict[tuple[str, str], _PromotionSeries] = {}
        self._lock = threading.Lock()

    def observe(self, mode: str, outcome: str, duration: float) -> None:
        normalized_mode = mode if mode in PROMOTION_MODES else "other"
        normalized_outcome = outcome if outcome in PROMOTION_OUTCOMES else "error"
        key = (normalized_mode, normalized_outcome)
        duration = max(0.0, duration)
        with self._lock:
            series = self._series.setdefault(key, _PromotionSeries())
            series.count += 1
            series.duration_sum += duration
            for index, upper_bound in enumerate(PROMOTION_LATENCY_BUCKETS):
                if duration <= upper_bound:
                    series.buckets[index] += 1

    def render_prometheus(self) -> list[str]:
        lines = [
            "# HELP ai_harness_promotions_total Promotion attempts by mode and outcome.",
            "# TYPE ai_harness_promotions_total counter",
            "# HELP ai_harness_promotion_duration_seconds Promotion latency.",
            "# TYPE ai_harness_promotion_duration_seconds histogram",
        ]
        with self._lock:
            snapshot = [
                (key, value.count, value.duration_sum, list(value.buckets))
                for key, value in sorted(self._series.items())
            ]
        for (mode, outcome), count, duration_sum, buckets in snapshot:
            labels = f'mode="{mode}",outcome="{outcome}"'
            lines.append(f"ai_harness_promotions_total{{{labels}}} {count}")
            for upper_bound, bucket_count in zip(PROMOTION_LATENCY_BUCKETS, buckets):
                lines.append(
                    "ai_harness_promotion_duration_seconds_bucket"
                    f'{{{labels},le="{upper_bound:g}"}} {bucket_count}'
                )
            lines.append(
                "ai_harness_promotion_duration_seconds_bucket"
                f'{{{labels},le="+Inf"}} {count}'
            )
            lines.append(
                f"ai_harness_promotion_duration_seconds_sum{{{labels}}} "
                f"{duration_sum:.9f}"
            )
            lines.append(
                f"ai_harness_promotion_duration_seconds_count{{{labels}}} {count}"
            )
        return lines
