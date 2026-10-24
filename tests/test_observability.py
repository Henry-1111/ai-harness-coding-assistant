import json

from ai_harness.observability import (
    HTTPMetrics,
    PromotionMetrics,
    format_http_access_log,
    format_worker_event,
    normalize_request_id,
)


def test_http_metrics_render_cumulative_histogram_with_bounded_route_labels() -> None:
    metrics = HTTPMetrics()
    metrics.observe("get", "/v1/jobs/{job_id}", 200, 0.02)
    metrics.observe("get", "/v1/jobs/{job_id}", 200, 0.2)

    rendered = "\n".join(metrics.render_prometheus())

    assert (
        'ai_harness_http_requests_total{method="GET",'
        'route="/v1/jobs/{job_id}",status="200"} 2'
    ) in rendered
    assert (
        'ai_harness_http_request_duration_seconds_bucket{method="GET",'
        'route="/v1/jobs/{job_id}",status="200",le="0.01"} 0'
    ) in rendered
    assert (
        'ai_harness_http_request_duration_seconds_bucket{method="GET",'
        'route="/v1/jobs/{job_id}",status="200",le="0.05"} 1'
    ) in rendered
    assert (
        'ai_harness_http_request_duration_seconds_bucket{method="GET",'
        'route="/v1/jobs/{job_id}",status="200",le="0.25"} 2'
    ) in rendered
    assert (
        'ai_harness_http_request_duration_seconds_count{method="GET",'
        'route="/v1/jobs/{job_id}",status="200"} 2'
    ) in rendered


def test_http_metrics_collapse_unknown_methods_to_bounded_label() -> None:
    metrics = HTTPMetrics()
    metrics.observe("CUSTOM-ONE", "unmatched", 404, 0.01)
    metrics.observe("CUSTOM-TWO", "unmatched", 404, 0.01)

    rendered = "\n".join(metrics.render_prometheus())

    assert 'method="OTHER",route="unmatched",status="404"} 2' in rendered
    assert "CUSTOM-ONE" not in rendered
    assert "CUSTOM-TWO" not in rendered


def test_promotion_metrics_render_outcomes_and_bound_unknown_labels() -> None:
    metrics = PromotionMetrics()
    metrics.observe("git_branch", "success", 0.4)
    metrics.observe("untrusted-mode", "untrusted-outcome", 2.0)

    rendered = "\n".join(metrics.render_prometheus())

    assert (
        'ai_harness_promotions_total{mode="git_branch",outcome="success"} 1'
        in rendered
    )
    assert (
        'ai_harness_promotion_duration_seconds_bucket{mode="git_branch",'
        'outcome="success",le="0.5"} 1'
    ) in rendered
    assert 'ai_harness_promotions_total{mode="other",outcome="error"} 1' in rendered
    assert "untrusted-mode" not in rendered
    assert "untrusted-outcome" not in rendered


def test_request_ids_and_access_logs_are_safe_and_structured() -> None:
    assert normalize_request_id("client-request_123") == "client-request_123"
    generated = normalize_request_id("unsafe request\nforged=true")
    assert len(generated) == 32
    assert generated.isalnum()

    payload = json.loads(
        format_http_access_log(
            request_id=generated,
            method="custom",
            route="unmatched",
            status_code=404,
            duration=0.01234,
        )
    )
    assert payload == {
        "duration_ms": 12.34,
        "event": "http_request",
        "method": "OTHER",
        "request_id": generated,
        "route": "unmatched",
        "status_code": 404,
    }


def test_worker_events_are_structured_and_correlation_safe() -> None:
    payload = json.loads(
        format_worker_event(
            event="job_started",
            request_id="trace-123",
            job_id="job-456",
            worker_id="worker-789",
            status="running",
            attempt=2,
        )
    )

    assert payload == {
        "attempt": 2,
        "event": "job_started",
        "job_id": "job-456",
        "request_id": "trace-123",
        "status": "running",
        "worker_id": "worker-789",
    }
