from ai_harness.agents import ReviewerAgent
from ai_harness.models import CommandResult, TaskRequest
from ai_harness.self_healing import SelfHealingPlanner


def test_reviewer_creates_structured_timeout_and_verification_findings():
    review = ReviewerAgent().review(
        TaskRequest(repository="/repo", goal="Fix deterministic verification failure"),
        ["app.py"],
        [
            CommandResult(command="pytest -q", exit_code=1, stderr="assert 2 == 3"),
            CommandResult(command="npm test", exit_code=124, timed_out=True),
        ],
    )

    assert review.approved is False
    assert [finding.category for finding in review.findings] == ["verification", "timeout"]
    assert review.findings[0].command == "pytest -q"


def test_self_healing_stops_when_normalized_failure_repeats():
    planner = SelfHealingPlanner()
    review = ReviewerAgent().review(
        TaskRequest(repository="/repo", goal="Fix deterministic verification failure"),
        ["app.py"],
        [CommandResult(command="pytest -q", exit_code=1, stderr="failed in 0.12s")],
    )
    first = planner.build(
        review,
        [CommandResult(command="pytest -q", exit_code=1, stderr="failed in 0.12s")],
        [],
    )
    second = planner.build(
        review,
        [CommandResult(command="pytest -q", exit_code=1, stderr="failed in 0.89s")],
        [first],
    )

    assert first.can_retry is True
    assert second.failure_fingerprint == first.failure_fingerprint
    assert second.repeat_count == 1
    assert second.can_retry is False
