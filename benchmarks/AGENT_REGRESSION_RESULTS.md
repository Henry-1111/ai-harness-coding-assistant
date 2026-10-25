# Targeted agent edit regression

Run date: 2026-10-25

This regression selected ten historically failing cases from the frozen 30-case
agent dataset. The derived dataset SHA-256 is
`c4ae5c097670fce72928795b8a39719f20daef3c7e8556c9a1d0c833f3cd9d0e`.
It ran once against both variants after exact-snippet relocation, atomic edit-batch
preflight, and bounded edit self-healing were implemented. The raw 20 observations
are stored in `agent-regression-10-report.json`.

| Metric | Historical subset baseline | Regression | Change |
| --- | ---: | ---: | ---: |
| Single-agent success | 1/30 (3.33%) | 3/10 (30.00%) | +26.67 pp |
| Multi-agent success | 0/30 (0.00%) | 2/10 (20.00%) | +20.00 pp |

The historical baseline contains three repetitions of these exact ten cases from
`agent-report.json`; the new regression contains one repetition, so the comparison is
directional rather than a confidence interval. The lift is large enough to justify a
30-case single-pass validation, but the absolute result remains too low to claim the
edit reliability problem is solved. Failures still included seven unsafe edit batches,
four repeated deterministic verification failures, and four observations affected
by upstream timeouts.

## 30-case single-pass attempt

The follow-up run used the unchanged frozen 30-case dataset, but DeepSeek exhausted
its account balance partway through and returned `402 Payment Required`. The local
Gateway normalized these upstream responses to 502. Of the 60 planned observations,
35 were therefore invalid before receiving a model response. The aborted raw report
is retained as `agent-report-single-pass-aborted.json`; its aggregate success rates
must not be reported as agent-quality metrics.

Before the upstream failure, 25 observations had valid responses: single agent passed
11/13 and multi agent passed 9/12. This prefix is biased toward earlier, easier cases
and is not a substitute for the complete run. The full 30-case single pass must be
restarted from the beginning after Provider balance is restored.
