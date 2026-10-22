from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Optional, Sequence

from .agents import OpenAICompatibleModel
from .api import create_app
from .config import Settings
from .embeddings import OpenAICompatibleEmbeddingProvider
from .evaluation import RetrievalEvaluator, load_cases
from .lsp import LSPDefinitionEnricher
from .models import AcceptanceCriterion, TaskRequest
from .orchestrator import HarnessOrchestrator
from .repository import Repository
from .sandbox import DockerSandbox, LocalCommandSandbox


def build_orchestrator() -> HarnessOrchestrator:
    settings = Settings.from_environment()
    if not settings.model_api_key:
        raise SystemExit("AI_HARNESS_MODEL_API_KEY is required")
    model = OpenAICompatibleModel(
        settings.model_base_url,
        settings.model_api_key,
        settings.model_name,
        settings.model_timeout_seconds,
    )
    if settings.sandbox_mode == "docker":
        def runner_factory(workspace):
            return DockerSandbox(workspace, image=settings.sandbox_image)

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
    definition_provider_factory = None
    if settings.lsp_command:
        lsp_command = shlex.split(settings.lsp_command)

        def definition_provider_factory(repository):
            return LSPDefinitionEnricher(lsp_command, repository)
    return HarnessOrchestrator(
        model,
        settings.data_directory,
        runner_factory=runner_factory,
        embedding_provider=embedding_provider,
        definition_provider_factory=definition_provider_factory,
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "eval-retrieval":
        evaluation = RetrievalEvaluator(Repository(Path(args.repository))).evaluate(
            load_cases(Path(args.dataset)),
            k=args.k,
            split=args.split,
        )
        report = evaluation.model_dump_json(indent=2)
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
    elif args.command == "promote":
        print(json.dumps({"promoted_files": orchestrator.promote(args.task_id)}, indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(orchestrator), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
