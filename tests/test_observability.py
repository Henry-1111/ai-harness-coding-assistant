import json

from ai_harness.observability import (
    HTTPMetrics,
    format_http_access_log,
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
