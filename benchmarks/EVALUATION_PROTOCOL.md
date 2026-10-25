# Frozen evaluation protocol

## Retrieval benchmark

- Dataset: `retrieval_cases.json`, 40 repository-internal file-level tasks.
- Freeze identifier: `ai-harness-retrieval-2026-10-v1`.
- Split: `test`; labels must not be changed while tuning retrieval behavior.
- Command: `ai-harness eval-retrieval . benchmarks/retrieval_cases.json --k 5 --split test --output benchmarks/latest-report.json`.
- Report Recall@5, Precision@5, MRR, NDCG@5, Hit Rate, context reduction, slices, and every bad case.
- The regression test locks the normalized dataset hash. Any label change requires a new dataset version and reviewed hash update.

This benchmark measures retrieval inside AI_Harness only. It is not evidence of cross-repository or cross-language accuracy.

## Planning and routing benchmark

- Dataset: `planning_cases.json`, 30 frozen task descriptions with manually assigned task type, complexity, risk, and route labels.
- Freeze identifier: `ai-harness-planning-2026-10-v1`.
- Command: `ai-harness eval-planning benchmarks/planning_cases.json --split test --output benchmarks/planning-report.json`.
- Report task type, complexity, risk, route and planning-decision accuracy separately.
- Plan validity requires an acyclic topological order, the expected core/collaboration steps, and a one-to-one assignment list.
- Keep every mismatch in the report. Do not relabel or delete failed cases after observing results.

The frozen set measures deterministic routing rules in this project; it is not an open-domain intent classification benchmark.

## Checkpoint recovery benchmark

- Command: `ai-harness eval-checkpoints --tasks 200 --revisions 8 --writers 8 --corruption-trials 20 --output benchmarks/checkpoint-report.json`.
- Concurrent writers create multiple revisions per task in a temporary SQLite database.
- A fresh store instance simulates process restart and must recover the latest state plus every historical revision.
- Invalid JSON rows are injected under separate task IDs to verify that corrupt records fail closed without blocking healthy tasks.
- Report recovery rate, revision integrity, corruption isolation and P50/P95/P99 load latency.

This benchmark covers process restart and record isolation on local SQLite. It does not claim host-crash durability or PostgreSQL cross-node recovery.

## Security attack benchmark

- Dataset: `security_cases.json`, 48 frozen boundary and policy attacks.
- Command: `ai-harness eval-security benchmarks/security_cases.json --output benchmarks/security-report.json`.
- Categories cover repository traversal, scope violations, command allow-list bypass, snapshot identifiers, archive paths, archive symlinks, and Docker policy construction.
- File attacks run only inside a temporary directory and check both out-of-repository and out-of-scope sentinels for side effects.
- Report every attack, rejection error, category block rate, unauthorized write, and observed host-path escape.

Docker cases assert the generated isolation policy; they do not constitute a container-runtime penetration test or protection against unknown kernel vulnerabilities.

## Agent A/B benchmark

- Dataset: `agent_cases.json`, 30 deterministic defect tasks.
- Run the same frozen dataset against single-agent and multi-agent variants with the same model, Gateway, temperature, and infrastructure.
- Use at least three repetitions. Variant order is counterbalanced between repetitions.
- Preserve failed runs and null token observations; do not replace missing provider usage with estimated tokens.
- Report the dataset SHA-256, model and Gateway configuration, repetition count, success rate, latency, model calls, provider tokens, repair rounds, and changed-path Recall/Precision.
- Do not edit the fixtures or prompts after looking at test results. Material changes require a new dataset version and a fresh report.

The fixture suite is intentionally bounded and safe for local execution. External-repository generalization requires a separate held-out benchmark.
