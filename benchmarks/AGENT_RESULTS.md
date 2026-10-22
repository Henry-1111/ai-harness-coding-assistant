# Single-agent vs multi-agent smoke benchmark

Run date: 2026-10-22

This run used `deepseek-flash` through the local LLM Gateway and the frozen dataset whose
SHA-256 is `be0582c60612d4a1970adac9ca099aaf7e19f2b6b02c189d391ff3007b5cd35e`.
The benchmark used the trusted local sandbox fixture, one paired run per case, and three test
cases. Raw observations are stored in `agent-report.json`.

| Metric | Single agent | Multi agent | Multi minus single |
| --- | ---: | ---: | ---: |
| Success rate | 100% | 100% | 0 percentage points |
| Mean duration | 2,380 ms | 5,751 ms | +3,371 ms |
| Mean model calls | 1.0 | 2.0 | +1.0 |
| Mean total tokens | 481 | 1,467 | +985 |
| Mean repair rounds | 0.0 | 0.0 | 0.0 |
| Changed-path recall | 100% | 100% | 0 percentage points |
| Changed-path precision | 100% | 100% | 0 percentage points |

Pairwise success was three ties, with no win for either variant. No collaboration conflict was
detected. On these small, isolated defects, multi-agent execution added cost without improving
the measured outcome.

This is a pipeline smoke benchmark, not a resume-grade quality claim. The dataset is too small
and too easy, and each case ran only once. A defensible comparison requires a larger held-out
set with complex cross-file tasks, repeated runs, fixed model/Gateway configuration, confidence
intervals, and simultaneous reporting of quality, latency, and token cost.
