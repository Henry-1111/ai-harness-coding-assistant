# Gateway → AI Harness full-flow A/B (single run)

Run date: 2026-10-27. AI Harness revision: `83b311b`. Frozen dataset:
`agent_cases.json`, SHA-256
`6f9d43944a941efd78d2ae6534f62eed5a4a5e794820687c5c79ae1b6c72e167`.
All 30 cases ran once against both variants (60 task observations) with
counterbalanced order. Both used `deepseek-flash` through the local LLM Gateway
at `http://127.0.0.1:8000/v1`, temperature 0, and trusted local fixtures.
The multi-agent arm enabled competing proposals; the single-agent arm disabled
collaboration. Gateway `/health` and `/ready` passed, and an authenticated smoke
request returned HTTP 200, usage, and `X-Request-ID` before the benchmark.

| Metric | Single agent | Multi agent |
| --- | ---: | ---: |
| End-to-end success | 20/30 (66.67%) | 23/30 (76.67%) |
| Provider-adjusted success (diagnostic) | 20/27 (74.07%) | 23/29 (79.31%) |
| Provider-failed tasks | 3 | 1 |
| Agent-failed tasks | 7 | 6 |
| P50 task duration | 11,927 ms | 15,383 ms |
| P95 task duration | 68,716 ms | 130,411 ms |
| Mean model calls per task | 1.433 | 4.533 |
| Tasks with complete Token usage | 27/30 | 23/30 |
| Tokens across complete-usage tasks only | 121,853 | 201,615 |

Pairwise outcome: multi-agent 4 wins, single-agent 1 win, 25 ties. The raw
single-run gain is 10 percentage points, but this is **directional, not evidence
of a stable improvement**. It should not be compared as a formal lift over the
earlier three-run report; at least two more controlled repetitions are needed.

Failure classes were: single-agent 5 edit-application, 2 repeated-verification,
3 Provider; multi-agent 3 edit-application, 2 repeated-verification, 1 verification,
1 Provider. Gateway recorded 179 benchmark model requests: 174 succeeded and 5
failed with `upstream_timeout`. The separate smoke request succeeded. Successful
benchmark requests consumed **579,425 Provider-reported Tokens** in total. The
Harness per-variant Token totals above exclude tasks whose usage was incomplete,
so they cannot be added or used as exact variant-cost totals. The Gateway total
is the complete known successful-request cost for this run; failed requests may
still have unreported upstream work.

The temporary Gateway API Key was revoked after the run. Local rate limiting and
quota controls were disabled in the Gateway configuration, so this run validates
the authenticated Gateway → Provider → Harness task path, not those controls.
Raw task observations: `agent-gateway-fullflow-2026-10-27.json`. Aggregate:
`agent-gateway-fullflow-2026-10-27-aggregate.json`.
