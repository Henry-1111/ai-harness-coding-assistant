# AI Harness 运行时基准

## 2026-10-23 本机结果

运行命令：

```bash
ai-harness eval-runtime \
  --jobs 100 \
  --workers 4 \
  --synthetic-work-ms 10 \
  --output benchmarks/runtime-report.json
```

结果：

| 指标 | 数值 |
| --- | ---: |
| 提交任务 | 100 |
| 成功 / 失败 | 100 / 0 |
| Worker | 4 |
| 合成工作时长 | 10 ms / job |
| 总耗时 | 378.926 ms |
| 吞吐 | 263.904 jobs/s |
| 提交延迟 P50 / P95 / P99 | 0.258 / 0.885 / 1.068 ms |
| 峰值并发 | 4 |
| 幂等重放 | 100 |
| 重复执行 | 0 |

原始机器可读报告见 [`runtime-report.json`](runtime-report.json)。

## 结果边界

这是单机 SQLite Queue + Worker 的合成运行时基准，每个任务只休眠 10 ms，不进行模型调用、代码检索、
容器启动或真实测试。因此它可以证明本次运行中 Worker 并发达到 4、100 次同 Key 重放没有产生重复任务，
但不能代表真实模型 RPS、端到端编码任务耗时，或多机部署能力。

真实链路应另行固定 Gateway、模型、任务集和网络环境，多次运行后分别报告端到端成功率、P50/P95
完成时延、Token 成本、故障恢复率和 Provider 切换时间。
