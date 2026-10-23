from __future__ import annotations

import threading
from dataclasses import dataclass, field

HTTP_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class _HTTPSeries:
    count: int = 0
    duration_sum: float = 0.0
    buckets: list[int] = field(
        default_factory=lambda: [0 for _ in HTTP_LATENCY_BUCKETS]
    )


class HTTPMetrics:
    """Thread-safe, bounded-label HTTP counters and latency histograms."""

    def __init__(self) -> None:
        self._series: dict[tuple[str, str, int], _HTTPSeries] = {}
        self._lock = threading.Lock()

    def observe(self, method: str, route: str, status_code: int, duration: float) -> None:
        key = (method.upper(), route, status_code)
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
