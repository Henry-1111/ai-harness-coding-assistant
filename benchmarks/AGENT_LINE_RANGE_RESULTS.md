# Agent benchmark aggregate

Dataset SHA-256: `c4ae5c097670fce72928795b8a39719f20daef3c7e8556c9a1d0c833f3cd9d0e`  
Cases: 10  
Runs: 1  
Observations per variant: 10

| Metric | Single agent | Multi agent |
| --- | ---: | ---: |
| Raw success | 3/10 (30.00%) | 3/10 (30.00%) |
| Provider-adjusted success | 30.00% | 33.33% |
| P50 duration | 25,197 ms | 48,998 ms |
| P95 duration | 52,274 ms | 89,605 ms |
| Mean model calls | 1.900 | 3.700 |
| Mean known tokens | 7,444 | 13,435 |
| Provider failures | 0 | 1 |
| Agent failures | 7 | 6 |

## Per-run raw success

| Run | Single agent | Multi agent | Single Provider failures | Multi Provider failures |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30.00% | 30.00% | 0 | 1 |

## Failure classes

### Single agent

- `edit_application`: 6
- `repeated_verification`: 1

### Multi agent

- `edit_application`: 1
- `provider_failure`: 1
- `repeated_verification`: 5

## Edit application

### Single agent

- Applied strategies: not recorded
- Rejected edits: none recorded

### Multi agent

- Applied strategies: not recorded
- Rejected edits: none recorded

Provider-adjusted success is diagnostic only; raw success remains the end-to-end metric.

## Comparison with the previous 10-case regression

The dataset and single-run protocol match `agent-regression-10-report.json`, but Provider
availability differed between runs. Raw and Provider-adjusted rates are therefore both shown;
neither run is treated as a statistically stable estimate.

| Metric | Previous single | Current single | Previous multi | Current multi |
| --- | ---: | ---: | ---: | ---: |
| Raw success | 30.00% | 30.00% | 20.00% | 30.00% |
| Provider-adjusted success | 42.86% | 30.00% | 22.22% | 33.33% |
| P95 duration | 85,013 ms | 52,274 ms | 84,537 ms | 89,605 ms |
| Mean known tokens | 9,651 | 7,444 | 12,769 | 13,435 |
| Edit-application failures | 2 | 6 | 5 | 1 |
| Provider failures | 3 | 0 | 1 | 1 |

Multi-agent edit-application failures fell from 5 to 1 and raw success rose by one case. Single
agent raw success was unchanged, while six failures still ended at edit application. This is a
small, mixed result: it justifies retaining the guarded line-range fallback, but does not justify
claiming a general success-rate improvement. This report predates per-edit strategy telemetry, so
it cannot prove how often the model actually supplied or used a line range. Subsequent benchmark
reports record exact-anchor, normalized-anchor, line-range and new-file applications together
with concrete edit rejection codes.

## Resume-safe result

On the frozen 10-case dataset across 1 runs, single-agent execution achieved 30.00% end-to-end success with P95 latency of 52,274 ms and mean observed usage of 7,444 tokens.

The Provider-adjusted diagnostic success rate was 30.00%; this excludes 0 explicit Provider failures and must not replace the end-to-end rate.
