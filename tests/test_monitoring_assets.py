from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_prometheus_config_and_alerts_reference_runtime_metrics() -> None:
    config = yaml.safe_load(
        (ROOT / "monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8")
    )
    alerts = yaml.safe_load(
        (ROOT / "monitoring/prometheus/alerts.yml").read_text(encoding="utf-8")
    )

    scrape = config["scrape_configs"][0]
    assert scrape["metrics_path"] == "/metrics"
    assert scrape["static_configs"][0]["targets"] == ["ai-harness:8080"]

    rules = alerts["groups"][0]["rules"]
    names = {rule["alert"] for rule in rules}
    assert len(names) == len(rules)
    assert {
        "AIHarnessNotReady",
        "AIHarnessDependencyUnavailable",
        "AIHarnessCircuitOpen",
        "AIHarnessQueueStalled",
        "AIHarnessQuotaReconciliationFailure",
        "AIHarnessWorkspaceSnapshotCapacityHigh",
        "AIHarnessPromotionFailure",
    }.issubset(names)


def test_grafana_dashboard_has_stable_uid_and_core_operational_panels() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/ai-harness-overview.json").read_text(
            encoding="utf-8"
        )
    )

    assert dashboard["uid"] == "ai-harness-overview"
    assert dashboard["refresh"] == "15s"
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Runtime readiness",
        "Outstanding jobs",
        "HTTP P95 latency",
        "Jobs by state",
        "Dependency health",
        "Workspace snapshot capacity",
        "Promotion outcomes / 1h",
        "Promotion P95 latency",
    }.issubset(titles)
    assert len({panel["id"] for panel in dashboard["panels"]}) == len(
        dashboard["panels"]
    )
