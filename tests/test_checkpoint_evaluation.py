import pytest

from ai_harness.checkpoint_evaluation import run_checkpoint_benchmark


def test_checkpoint_benchmark_recovers_revisions_after_restart() -> None:
    report = run_checkpoint_benchmark(
        tasks=12,
        revisions_per_task=4,
        writers=3,
        corruption_trials=3,
    )

    assert report.total_checkpoints == 48
    assert report.recovered_tasks == 12
    assert report.recovery_rate == 1.0
    assert report.revision_integrity_rate == 1.0
    assert report.corruption_isolation_rate == 1.0
    assert report.failed_task_ids == []
    assert report.recovery_latency_ms.p99 >= 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"tasks": 0}, "tasks"),
        ({"revisions_per_task": 0}, "revisions_per_task"),
        ({"writers": 0}, "writers"),
        ({"corruption_trials": 101}, "corruption_trials"),
    ],
)
def test_checkpoint_benchmark_rejects_invalid_bounds(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_checkpoint_benchmark(**arguments)
