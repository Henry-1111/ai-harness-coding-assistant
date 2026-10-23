from ai_harness.observability import HTTPMetrics


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
