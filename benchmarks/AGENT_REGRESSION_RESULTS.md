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

## Completed 30-case single pass

After the Provider balance was restored, the full dataset was restarted from the
beginning and completed. The raw 60 observations are stored in
`agent-report-single-pass.json`.

| Metric | Single agent | Multi agent | Multi minus single |
| --- | ---: | ---: | ---: |
| Raw success rate | 25/30 (83.33%) | 19/30 (63.33%) | -20.00 pp |
| Provider-adjusted success rate | 25/27 (92.59%) | 19/25 (76.00%) | -16.59 pp |
| Mean duration | 17,919 ms | 35,825 ms | +17,907 ms |
| P50 duration | 10,845 ms | 23,619 ms | +12,774 ms |
| P95 duration | 41,164 ms | 90,308 ms | +49,144 ms |
| Mean model calls | 1.400 | 2.700 | +1.300 |
| Mean known tokens | 3,543 | 6,248 | +2,705 |
| Changed-path recall | 93.33% | 100.00% | +6.67 pp |
| Changed-path precision | 93.33% | 100.00% | +6.67 pp |

Provider-adjusted rates exclude three single-agent and five multi-agent observations
that failed because of upstream timeout or connection errors. They are diagnostic
figures, not replacements for the raw end-to-end success rates. Strict mean token
fields remain `null` because Provider failures do not report usage; the table labels
means over the 27 and 25 observations with known token counts.

Remaining non-Provider failures were two single-agent edit-application failures,
three multi-agent edit-application failures, and three multi-agent repeated
verification failures. The single-agent result exceeds the 75%–80% target in this
one frozen-fixture run, but it must be repeated before being presented as a stable
project-wide success rate. Multi-agent execution improved path selection but reduced
success, roughly doubled latency, and used about 76% more observed tokens per valid
observation.

The subsequent three-run aggregate is the authoritative post-fix result and is stored
in `AGENT_THREE_RUN_RESULTS.md` and `agent-report-three-runs.json`. The single-pass
figures above remain useful for auditability but must not replace the three-run result.
