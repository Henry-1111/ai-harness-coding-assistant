# Agent benchmark aggregate

Dataset SHA-256: `6f9d43944a941efd78d2ae6534f62eed5a4a5e794820687c5c79ae1b6c72e167`  
Cases: 30  
Runs: 3  
Observations per variant: 90

| Metric | Single agent | Multi agent |
| --- | ---: | ---: |
| Raw success | 67/90 (74.44%) | 59/90 (65.56%) |
| Provider-adjusted success | 87.01% | 72.84% |
| P50 duration | 10,845 ms | 26,793 ms |
| P95 duration | 66,457 ms | 100,782 ms |
| Mean model calls | 1.367 | 2.878 |
| Mean known tokens | 3,601 | 7,090 |
| Provider failures | 13 | 9 |
| Agent failures | 10 | 22 |

## Per-run raw success

| Run | Single agent | Multi agent | Single Provider failures | Multi Provider failures |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.33% | 63.33% | 3 | 5 |
| 2 | 70.00% | 70.00% | 5 | 2 |
| 3 | 70.00% | 63.33% | 5 | 2 |

## Failure classes

### Single agent

- `edit_application`: 8
- `provider_failure`: 13
- `repeated_verification`: 1
- `verification`: 1

### Multi agent

- `edit_application`: 16
- `provider_failure`: 9
- `repeated_verification`: 6

Provider-adjusted success is diagnostic only; raw success remains the end-to-end metric.

## Resume-safe result

On the frozen 30-case dataset across 3 runs, single-agent execution achieved 74.44% end-to-end success with P95 latency of 66,457 ms and mean observed usage of 3,601 tokens.

The Provider-adjusted diagnostic success rate was 87.01%; this excludes 13 explicit Provider failures and must not replace the end-to-end rate.
