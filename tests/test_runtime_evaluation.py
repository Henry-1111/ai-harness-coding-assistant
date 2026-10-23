from __future__ import annotations

import pytest

from ai_harness.runtime_evaluation import percentile, run_runtime_benchmark


def test_percentile_interpolates_sorted_values() -> None:
    assert percentile([], 0.95) == 0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_runtime_benchmark_measures_queue_and_idempotency() -> None:
    report = run_runtime_benchmark(jobs=12, workers=3, synthetic_work_ms=10)

    assert report.submitted_jobs == 12
    assert report.completed_jobs == 12
    assert report.failed_jobs == 0
    assert report.idempotency_replays == 12
    assert report.duplicate_executions == 0
    assert 1 < report.peak_concurrency <= 3
    assert report.jobs_per_second > 0
    assert report.submission_latency_ms.p50 <= report.submission_latency_ms.p95
    assert report.submission_latency_ms.p95 <= report.submission_latency_ms.p99


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"jobs": 0}, "jobs"),
        ({"workers": 0}, "workers"),
        ({"synthetic_work_ms": -1}, "synthetic_work_ms"),
    ],
)
def test_runtime_benchmark_rejects_invalid_bounds(
    arguments: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_runtime_benchmark(**arguments)
