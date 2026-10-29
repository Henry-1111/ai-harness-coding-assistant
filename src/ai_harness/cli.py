from __future__ import annotations

import argparse
import json
import shlex
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

from .agent_evaluation import (
    AgentBenchmarkCase,
    AgentBenchmarkEvaluator,
    AgentBenchmarkObservation,
    AgentBenchmarkReport,
    aggregate_agent_reports,
    changed_path_metrics,
    dataset_sha256,
    load_agent_cases,
    render_agent_aggregate_markdown,
)
from .agents import OpenAICompatibleModel
from .api import create_app
from .checkpoint_evaluation import run_checkpoint_benchmark
from .config import Settings
from .distributed_governance import DistributedGovernanceStore
from .distributed_memory import PostgresLayeredMemoryStore
from .distributed_runtime import PostgresCheckpointStore, PostgresTaskJobStore
from .embeddings import OpenAICompatibleEmbeddingProvider
from .evaluation import RetrievalEvaluator, compare_structure_index, load_cases
from .governance import APIKeyAuthenticator, GovernanceStore
from .lsp import LSPDefinitionEnricher
from .mcp import MCPToolCatalog
from .models import AcceptanceCriterion, TaskRequest, TaskStatus
from .object_workspaces import S3WorkspaceSnapshotStore
from .orchestrator import HarnessOrchestrator
from .planning_evaluation import PlanningEvaluator, load_planning_cases
from .promotion import (
    GitBranchPublisher,
    GitHubPullRequestClient,
    GitHubPullRequestPublisher,
    GitLabMergeRequestClient,
    GitLabMergeRequestPublisher,
    PromotionPublisher,
)
from .repository import DefinitionProvider, Repository
from .runtime_evaluation import run_runtime_benchmark
from .sandbox import CommandRunner, DockerSandbox, LocalCommandSandbox
from .security_evaluation import SecurityEvaluator, load_security_cases
from .workspaces import FilesystemWorkspaceSnapshotStore, WorkspaceSnapshotStore


def build_workspace_store(settings: Settings) -> Optional[WorkspaceSnapshotStore]:
    if settings.workspace_snapshot_s3_bucket:
        cache_root = (
            Path(settings.workspace_snapshot_directory)
            if settings.workspace_snapshot_directory
            else settings.data_directory / "workspace-snapshot-cache"
        )
        return S3WorkspaceSnapshotStore(
            cache_root,
            bucket=settings.workspace_snapshot_s3_bucket,
            prefix=settings.workspace_snapshot_s3_prefix,
            endpoint_url=settings.workspace_snapshot_s3_endpoint,
            region=settings.workspace_snapshot_s3_region,
            access_key_id=settings.workspace_snapshot_s3_access_key_id,
            secret_access_key=settings.workspace_snapshot_s3_secret_access_key,
            max_bytes=settings.workspace_snapshot_max_bytes,
        )
    if settings.workspace_snapshot_directory:
        return FilesystemWorkspaceSnapshotStore(
            Path(settings.workspace_snapshot_directory),
            max_bytes=settings.workspace_snapshot_max_bytes,
        )
    return None


def build_promotion_publisher(settings: Settings) -> Optional[PromotionPublisher]:
    if settings.promotion_mode == "local":
        return None
    if settings.promotion_mode != "git_branch":
        raise SystemExit("AI_HARNESS_PROMOTION_MODE must be 'local' or 'git_branch'")
    if not settings.promotion_git_remote_url:
        raise SystemExit(
            "AI_HARNESS_PROMOTION_GIT_REMOTE_URL is required for git_branch promotion"
        )
    branch_publisher = GitBranchPublisher(
        settings.promotion_git_remote_url,
        settings.data_directory / "promotion-staging",
        base_ref=settings.promotion_git_base_ref,
        branch_prefix=settings.promotion_git_branch_prefix,
        author_name=settings.promotion_git_author_name,
        author_email=settings.promotion_git_author_email,
    )
    github_configured = bool(
        settings.promotion_github_repository or settings.promotion_github_token
    )
    gitlab_configured = bool(
        settings.promotion_gitlab_project or settings.promotion_gitlab_token
    )
    if github_configured and gitlab_configured:
        raise SystemExit("configure only one of GitHub or GitLab promotion")
    if not github_configured and not gitlab_configured:
        return branch_publisher
    if github_configured:
        if not settings.promotion_github_repository or not settings.promotion_github_token:
            raise SystemExit(
                "AI_HARNESS_PROMOTION_GITHUB_REPOSITORY and "
                "AI_HARNESS_PROMOTION_GITHUB_TOKEN must be configured together"
            )
        return GitHubPullRequestPublisher(
            branch_publisher,
            GitHubPullRequestClient(
                settings.promotion_github_repository,
                settings.promotion_github_token,
                api_url=settings.promotion_github_api_url,
                api_version=settings.promotion_github_api_version,
                timeout_seconds=settings.promotion_github_timeout_seconds,
                max_attempts=settings.promotion_github_max_attempts,
                retry_backoff_seconds=settings.promotion_github_retry_backoff_seconds,
                draft=settings.promotion_github_draft,
            ),
        )
    if not settings.promotion_gitlab_project or not settings.promotion_gitlab_token:
        raise SystemExit(
            "AI_HARNESS_PROMOTION_GITLAB_PROJECT and "
            "AI_HARNESS_PROMOTION_GITLAB_TOKEN must be configured together"
        )
    return GitLabMergeRequestPublisher(
        branch_publisher,
        GitLabMergeRequestClient(
            settings.promotion_gitlab_project,
            settings.promotion_gitlab_token,
            api_url=settings.promotion_gitlab_api_url,
            timeout_seconds=settings.promotion_gitlab_timeout_seconds,
            max_attempts=settings.promotion_gitlab_max_attempts,
            retry_backoff_seconds=settings.promotion_gitlab_retry_backoff_seconds,
            draft=settings.promotion_gitlab_draft,
        ),
    )


def build_orchestrator(
    *,
    settings: Optional[Settings] = None,
    model: Optional[OpenAICompatibleModel] = None,
    data_directory: Optional[Path] = None,
    collaboration_mode: Literal["auto", "disabled", "forced"] = "auto",
) -> HarnessOrchestrator:
    settings = settings or Settings.from_environment()
    if not settings.model_api_key:
        raise SystemExit("AI_HARNESS_MODEL_API_KEY is required")
    model = model or OpenAICompatibleModel(
        settings.model_base_url,
        settings.model_api_key,
        settings.model_name,
        settings.model_timeout_seconds,
    )
    runner_factory: Callable[[Path], CommandRunner]
    if settings.sandbox_mode == "docker":

        def docker_runner_factory(workspace: Path) -> CommandRunner:
            return DockerSandbox(workspace, image=settings.sandbox_image)

        runner_factory = docker_runner_factory

    elif settings.sandbox_mode == "local":
        runner_factory = LocalCommandSandbox
    else:
        raise SystemExit("AI_HARNESS_SANDBOX_MODE must be 'docker' or 'local'")
    embedding_provider = (
        OpenAICompatibleEmbeddingProvider(
            settings.embedding_base_url,
            settings.embedding_api_key,
            settings.embedding_model,
            settings.model_timeout_seconds,
        )
        if settings.embedding_model
        else None
    )
    definition_provider_factory: Optional[Callable[[Path], DefinitionProvider]] = None
    if settings.lsp_command:
        lsp_command = shlex.split(settings.lsp_command)

        def lsp_definition_provider_factory(repository: Path) -> DefinitionProvider:
            return LSPDefinitionEnricher(lsp_command, repository)

        definition_provider_factory = lsp_definition_provider_factory

    mcp_tool_catalog = (
        MCPToolCatalog.load(Path(settings.mcp_catalog_path))
        if settings.mcp_catalog_path
        else None
    )
    checkpoint_store = (
        PostgresCheckpointStore(settings.governance_database_url)
        if settings.governance_database_url
        else None
    )
    memory_store = (
        PostgresLayeredMemoryStore(settings.governance_database_url)
        if settings.governance_database_url
        else None
    )
    workspace_store = build_workspace_store(settings)
    promotion_publisher = build_promotion_publisher(settings)
    return HarnessOrchestrator(
        model,
        data_directory or settings.data_directory,
        checkpoint_store=checkpoint_store,
        memory_store=memory_store,
        runner_factory=runner_factory,
        embedding_provider=embedding_provider,
        definition_provider_factory=definition_provider_factory,
        restricted_react_enabled=settings.restricted_react_enabled,
        restricted_react_max_tool_calls=settings.restricted_react_max_tool_calls,
        mcp_tool_catalog=mcp_tool_catalog,
        tenant_id=settings.tenant_id,
        collaboration_mode=collaboration_mode,
        multi_proposal_enabled=settings.multi_proposal_enabled,
        workspace_store=workspace_store,
        promotion_publisher=promotion_publisher,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a coding task in an isolated workspace")
    run.add_argument("repository")
    run.add_argument("goal")
    run.add_argument("--accept", action="append", default=[])
    run.add_argument("--scope", action="append", default=[])
    run.add_argument("--verify", action="append", default=[])

    status = subparsers.add_parser("status", help="Read the latest task checkpoint")
    status.add_argument("task_id")

    resume = subparsers.add_parser("resume", help="Resume from the latest checkpoint")
    resume.add_argument("task_id")

    approve = subparsers.add_parser(
        "approve", help="Approve a task that is waiting at the human gate"
    )
    approve.add_argument("task_id")
    approve.add_argument("--actor", required=True)
    approve.add_argument("--note", default="")

    reject = subparsers.add_parser(
        "reject", help="Reject a task that is waiting at the human gate"
    )
    reject.add_argument("task_id")
    reject.add_argument("--actor", required=True)
    reject.add_argument("--note", default="")

    promote = subparsers.add_parser("promote", help="Copy an approved patch to the source repo")
    promote.add_argument("task_id")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    evaluate = subparsers.add_parser(
        "eval-retrieval", help="Measure repository retrieval against a labelled dataset"
    )
    evaluate.add_argument("repository")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--split", choices=["train", "validation", "test"])
    evaluate.add_argument("--output", help="Write the JSON report to this path")
    evaluate.add_argument(
        "--compare-structure",
        action="store_true",
        help="Compare the full index against the same retriever without symbols/imports",
    )

    agent_evaluate = subparsers.add_parser(
        "eval-agents",
        help="Compare single-agent and multi-agent execution on the same task dataset",
    )
    agent_evaluate.add_argument("repository")
    agent_evaluate.add_argument("dataset")
    agent_evaluate.add_argument("--split", choices=["train", "validation", "test"])
    agent_evaluate.add_argument("--repetitions", type=int, default=1)
    agent_evaluate.add_argument(
        "--approve-trusted-fixtures",
        action="store_true",
        help=(
            "Record and approve governance gates during an explicitly trusted local "
            "fixture benchmark"
        ),
    )
    agent_evaluate.add_argument("--output", help="Write the JSON report to this path")

    agent_summary = subparsers.add_parser(
        "summarize-agent-reports",
        help="Aggregate compatible agent benchmark JSON reports",
    )
    agent_summary.add_argument("reports", nargs="+")
    agent_summary.add_argument("--output", help="Write the aggregate JSON report")
    agent_summary.add_argument(
        "--markdown-output", help="Write a human-readable Markdown report"
    )

    runtime_evaluate = subparsers.add_parser(
        "eval-runtime",
        help="Benchmark durable queue latency, throughput, concurrency and idempotency",
    )
    runtime_evaluate.add_argument("--jobs", type=int, default=100)
    runtime_evaluate.add_argument("--workers", type=int, default=4)
    runtime_evaluate.add_argument("--synthetic-work-ms", type=float, default=10)
    runtime_evaluate.add_argument("--output", help="Write the JSON report to this path")

    planning_evaluate = subparsers.add_parser(
        "eval-planning",
        help="Measure task normalization, routing and generated Plan integrity",
    )
    planning_evaluate.add_argument("dataset")
    planning_evaluate.add_argument("--split", choices=["train", "validation", "test"])
    planning_evaluate.add_argument("--output", help="Write the JSON report to this path")

    checkpoint_evaluate = subparsers.add_parser(
        "eval-checkpoints",
        help="Benchmark durable checkpoint recovery after process restart",
    )
    checkpoint_evaluate.add_argument("--tasks", type=int, default=200)
    checkpoint_evaluate.add_argument("--revisions", type=int, default=8)
    checkpoint_evaluate.add_argument("--writers", type=int, default=8)
    checkpoint_evaluate.add_argument("--corruption-trials", type=int, default=20)
    checkpoint_evaluate.add_argument("--output", help="Write the JSON report to this path")

    security_evaluate = subparsers.add_parser(
        "eval-security",
        help="Run frozen boundary, permission, command and archive attacks",
    )
    security_evaluate.add_argument("dataset")
    security_evaluate.add_argument("--output", help="Write the JSON report to this path")
    return parser


def run_agent_benchmark(args: argparse.Namespace) -> str:
    settings = Settings.from_environment()
    if not settings.model_api_key:
        raise SystemExit("AI_HARNESS_MODEL_API_KEY is required")
    repository = str(Path(args.repository).resolve())
    dataset_path = Path(args.dataset)
    cases = load_agent_cases(dataset_path)

    with tempfile.TemporaryDirectory(prefix="ai-harness-agent-eval-") as temp_directory:
        data_root = Path(temp_directory)

        def runner(
            case: AgentBenchmarkCase,
            variant: Literal["single", "multi"],
            run_index: int,
        ) -> AgentBenchmarkObservation:
            model = OpenAICompatibleModel(
                settings.model_base_url,
                settings.model_api_key,
                settings.model_name,
                settings.model_timeout_seconds,
            )
            case_key = sha256(case.id.encode("utf-8")).hexdigest()[:12]
            orchestrator = build_orchestrator(
                settings=settings,
                model=model,
                data_directory=data_root / case_key / f"run-{run_index}" / variant,
                collaboration_mode="disabled" if variant == "single" else "forced",
            )
            request = TaskRequest(
                repository=repository,
                goal=case.goal,
                acceptance_criteria=[
                    AcceptanceCriterion(description=item)
                    for item in case.acceptance_criteria
                ],
                file_scope=case.file_scope,
                verification_commands=case.verification_commands,
                max_replans=case.max_replans,
            )
            started = time.monotonic()
            try:
                result = orchestrator.run(request)
                if (
                    args.approve_trusted_fixtures
                    and result.status == TaskStatus.NEEDS_INPUT
                    and result.approval is not None
                    and result.approval.status == "pending"
                ):
                    result = orchestrator.decide_approval(
                        result.task_id,
                        approved=True,
                        actor="benchmark-runner",
                        note="Explicitly trusted frozen local evaluation fixture",
                    )
                duration_ms = int((time.monotonic() - started) * 1000)
                usage = model.usage_snapshot()
                recall, precision = changed_path_metrics(
                    result.changed_files,
                    case.expected_changed_paths,
                )
                persisted_result = orchestrator.get(result.task_id)
                edit_metrics = (
                    persisted_result.scratchpad.get("edit_application_metrics", {})
                    if persisted_result is not None
                    else {}
                )
                if not isinstance(edit_metrics, dict):
                    edit_metrics = {}
                edit_strategies = edit_metrics.get("strategies", {})
                edit_failure_codes = edit_metrics.get("failure_codes", {})
                return AgentBenchmarkObservation(
                    case_id=case.id,
                    variant=variant,
                    run_index=run_index,
                    status=result.status,
                    duration_ms=duration_ms,
                    model_calls=usage.calls,
                    request_chars=usage.request_chars,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    repair_rounds=len(result.repair_history),
                    collaboration_rounds=len(result.collaboration_history),
                    detected_conflicts=sum(
                        len(turn.detected_conflicts)
                        for turn in result.collaboration_history
                    ),
                    changed_paths=result.changed_files,
                    changed_path_recall=recall,
                    changed_path_precision=precision,
                    edit_strategies=(
                        dict(edit_strategies)
                        if isinstance(edit_strategies, dict)
                        else {}
                    ),
                    edit_failure_codes=(
                        dict(edit_failure_codes)
                        if isinstance(edit_failure_codes, dict)
                        else {}
                    ),
                    error=result.error,
                )
            except Exception as exc:
                usage = model.usage_snapshot()
                return AgentBenchmarkObservation(
                    case_id=case.id,
                    variant=variant,
                    run_index=run_index,
                    status=TaskStatus.FAILED,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    model_calls=usage.calls,
                    request_chars=usage.request_chars,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    repair_rounds=0,
                    collaboration_rounds=0,
                    detected_conflicts=0,
                    error=f"{type(exc).__name__}: {exc}",
                )

        evaluation = AgentBenchmarkEvaluator().evaluate(
            cases,
            runner,
            dataset_hash=dataset_sha256(dataset_path),
            split=args.split,
            repetitions=args.repetitions,
        )
    return evaluation.model_dump_json(indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "eval-retrieval":
        repository_root = Path(args.repository).resolve()
        repository = Repository(repository_root)
        cases = load_cases(Path(args.dataset), repository_root=repository_root)
        if args.compare_structure:
            evaluation = compare_structure_index(
                repository, cases, k=args.k, split=args.split
            )
        else:
            evaluation = RetrievalEvaluator(repository).evaluate(
                cases, k=args.k, split=args.split
            )
        report = evaluation.model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    if args.command == "eval-agents":
        report = run_agent_benchmark(args)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    if args.command == "summarize-agent-reports":
        reports = [
            AgentBenchmarkReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in args.reports
        ]
        aggregate = aggregate_agent_reports(reports)
        report = aggregate.model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        if args.markdown_output:
            Path(args.markdown_output).write_text(
                render_agent_aggregate_markdown(aggregate), encoding="utf-8"
            )
        print(report)
        return
    if args.command == "eval-runtime":
        report = run_runtime_benchmark(
            jobs=args.jobs,
            workers=args.workers,
            synthetic_work_ms=args.synthetic_work_ms,
        ).model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    if args.command == "eval-planning":
        report = PlanningEvaluator().evaluate(
            load_planning_cases(Path(args.dataset)), split=args.split
        ).model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    if args.command == "eval-checkpoints":
        report = run_checkpoint_benchmark(
            tasks=args.tasks,
            revisions_per_task=args.revisions,
            writers=args.writers,
            corruption_trials=args.corruption_trials,
        ).model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    if args.command == "eval-security":
        report = SecurityEvaluator().evaluate(
            load_security_cases(Path(args.dataset))
        ).model_dump_json(indent=2)
        if args.output:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    orchestrator = build_orchestrator()
    if args.command == "run":
        request = TaskRequest(
            repository=str(Path(args.repository).resolve()),
            goal=args.goal,
            acceptance_criteria=[AcceptanceCriterion(description=item) for item in args.accept],
            file_scope=args.scope,
            verification_commands=args.verify or ["pytest -q"],
        )
        print(orchestrator.run(request).model_dump_json(indent=2))
    elif args.command == "status":
        task = orchestrator.get(args.task_id)
        if task is None:
            raise SystemExit("task not found")
        print(task.model_dump_json(indent=2))
    elif args.command == "resume":
        print(orchestrator.resume(args.task_id).model_dump_json(indent=2))
    elif args.command in {"approve", "reject"}:
        print(
            orchestrator.decide_approval(
                args.task_id,
                approved=args.command == "approve",
                actor=args.actor,
                note=args.note,
            ).model_dump_json(indent=2)
        )
    elif args.command == "promote":
        print(json.dumps({"promoted_files": orchestrator.promote(args.task_id)}, indent=2))
    elif args.command == "serve":
        import uvicorn

        settings = Settings.from_environment()
        if settings.auth_enabled and not settings.auth_config_path:
            raise SystemExit(
                "AI_HARNESS_AUTH_CONFIG_PATH is required when AI_HARNESS_AUTH_ENABLED=true"
            )
        authenticator = (
            APIKeyAuthenticator.from_file(Path(settings.auth_config_path))
            if settings.auth_enabled
            else APIKeyAuthenticator.disabled()
        )
        if bool(settings.governance_database_url) != bool(settings.redis_url):
            raise SystemExit(
                "AI_HARNESS_GOVERNANCE_DATABASE_URL and AI_HARNESS_REDIS_URL "
                "must be configured together"
            )
        governance_store = (
            DistributedGovernanceStore.from_urls(
                settings.governance_database_url,
                settings.redis_url,
                key_prefix=settings.redis_key_prefix,
                quota_ttl_seconds=settings.redis_quota_ttl_seconds,
                binding_cache_seconds=settings.redis_binding_cache_seconds,
            )
            if settings.governance_database_url
            else GovernanceStore(settings.data_directory / "governance.sqlite3")
        )
        job_store = (
            PostgresTaskJobStore(
                settings.governance_database_url,
                max_outstanding_jobs=settings.job_queue_capacity,
            )
            if settings.governance_database_url
            else None
        )
        workspace_store = build_workspace_store(settings)
        uvicorn.run(
            create_app(
                orchestrator,
                job_store=job_store,
                authenticator=authenticator,
                governance_store=governance_store,
                max_job_workers=settings.job_workers,
                job_poll_interval_seconds=settings.job_poll_interval_seconds,
                job_lease_seconds=settings.job_lease_seconds,
                job_queue_capacity=settings.job_queue_capacity,
                job_timeout_seconds=settings.job_timeout_seconds,
                circuit_failure_threshold=settings.circuit_failure_threshold,
                circuit_recovery_seconds=settings.circuit_recovery_seconds,
                quota_reconciliation_interval_seconds=(
                    settings.quota_reconciliation_interval_seconds
                ),
                max_request_body_bytes=settings.api_max_request_body_bytes,
                cors_allowed_origins=settings.cors_allowed_origins,
                workspace_store=workspace_store,
            ),
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
