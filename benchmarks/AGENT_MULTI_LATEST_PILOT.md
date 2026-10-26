# Latest multi-agent A/B pilot — inconclusive

Run date: 2026-10-26. Code revision: `e29a06c`. This pilot used the 10-case
historical-regression subset (SHA-256
`c4ae5c097670fce72928795b8a39719f20daef3c7e8556c9a1d0c833f3cd9d0e`),
one repetition, `deepseek-flash`, and the same direct DeepSeek endpoint for both
variants. The multi-agent arm enabled competing proposals. The single-agent arm
disabled collaboration. Variant order alternated by case. Raw observations are in
`agent-multi-latest-pilot.json`; the aggregate is in
`agent-multi-latest-pilot-aggregate.json`.

| Metric | Single agent | Multi agent |
| --- | ---: | ---: |
| End-to-end success | 0/10 | 0/10 |
| Provider failures | 7/10 | 7/10 |
| Agent failures | 3/10 | 3/10 |
| P95 duration | 62,244 ms | 70,021 ms |
| Mean model calls | 1.3 | 4.7 |

The 0% figures **must not be used as an agent-quality or resume claim**: TLS
connections to the Provider failed in seven paired cases. A subsequent three-call
Provider probe succeeded, consistent with an intermittent connectivity issue. The
three remaining pairs were all unsuccessful: single-agent failures included one
edit-application and two repeated-verification failures; multi-agent failures were
three edit-application failures. This small subset is also intentionally enriched
for historical failures and is not representative of the full frozen 30-case set.

The benchmark classifier was updated to count wrapped TLS EOF errors as Provider
failures. Next, stabilize the model endpoint, then rerun the unchanged 30-case set
for at least three repetitions. Report raw success, Provider-adjusted diagnostic
success, Token usage, latency and failure types without dropping failed observations.
