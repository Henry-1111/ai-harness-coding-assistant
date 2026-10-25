# Single-agent vs multi-agent benchmark

Run date: 2026-10-25

This run used `deepseek-flash` through the local LLM Gateway and the frozen 30-case
dataset whose SHA-256 is
`6f9d43944a941efd78d2ae6534f62eed5a4a5e794820687c5c79ae1b6c72e167`.
Every case ran three times against both variants, producing 90 observations per
variant. Variant order was counterbalanced between repetitions. Raw observations,
including every failure, are stored in `agent-report.json`.

| Metric | Single agent | Multi agent | Multi minus single |
| --- | ---: | ---: | ---: |
| Success rate | 48.89% | 47.78% | -1.11 percentage points |
| Mean duration | 19,259 ms | 26,936 ms | +7,677 ms |
| P50 duration | 11,751 ms | 19,326 ms | — |
| P95 duration | 63,889 ms | 87,259 ms | — |
| Mean model calls | 1.222 | 2.667 | +1.445 |
| Mean observed tokens | 3,415 | 6,464 | — |
| Token observations | 82/90 | 88/90 | — |
| Mean repair rounds | 0.356 | 0.556 | +0.200 |
| Mean collaboration rounds | 0.000 | 1.322 | +1.322 |
| Changed-path recall | 71.11% | 80.00% | +8.89 percentage points |
| Changed-path precision | 71.11% | 80.00% | +8.89 percentage points |

The strict `mean_total_tokens` fields in the raw report are `null`: 10 observations
lost Provider usage when their request timed out, and the evaluator does not estimate
missing Token values. The observed-only means above are labelled separately and use
82 single-agent and 88 multi-agent observations. Known totals were 280,025 and
568,861 tokens respectively.

Pairwise success produced 6 multi-agent wins, 7 single-agent wins, and 77 ties. The
three repetition success counts were 14/30, 14/30, and 16/30 for single-agent, and
15/30, 15/30, and 13/30 for multi-agent. On this fixture, multi-agent execution did
not improve success rate and used substantially more time, model calls, and observed
tokens.

## Failure analysis

| Failure class | Single agent | Multi agent |
| --- | ---: | ---: |
| Exact edit match failure | 25 | 24 |
| Empty edit anchor | 1 | 2 |
| Verification failure after editing | 6 | 13 |
| Approval gate (`needs_input`) | 6 | 6 |
| Upstream timeout | 7 | 2 |
| Upstream connection error | 1 | 0 |

The dominant weakness is edit reliability: the model often emitted an `old_text`
anchor that did not occur exactly once in the target file. Multi-agent review improved
changed-path selection but did not correct enough malformed edits or verification
failures to improve end-to-end success. Two quota-labelled cases were classified as
high risk and paused at the real approval gate in every repetition. A future automated
benchmark should explicitly approve these trusted fixture tasks after recording the
gate, rather than counting them as model failures.

This is a reproducible local fixture benchmark, not evidence of external-repository
generalization. These measured results must not be replaced with the 75%–80% target
from the project description. The earlier three-case smoke run reached 100%, but was
too small and easy for a defensible quality claim.
