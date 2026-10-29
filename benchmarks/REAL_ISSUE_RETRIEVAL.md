# 未见过的真实 issue／PR 文件定位测试（2026-10-29）

在上一轮排序代码定稿后，选取四个公开仓库中**此前未用于调参**的实际故障。先从修复 PR 记录固定修复前的 base commit 与被修改的生产源码文件，再将 issue／PR 描述改写成不包含目标路径的查询；清单先提交为 `6f5e2da`，之后才运行检索。原始标签、版本及来源链接见 [`real_issue_retrieval_cases.json`](real_issue_retrieval_cases.json)，逐 Case 排名见 [`real_issue_retrieval_report.json`](real_issue_retrieval_report.json)。

| 语言与来源 | 修复目标 | 结构索引名次 | 去符号/导入名次 |
| --- | --- | ---: | ---: |
| JavaScript：[Express PR #7459](https://github.com/expressjs/express/pull/7459) | `lib/response.js` | 1 | 1 |
| TypeScript：[p-queue issue #219](https://github.com/sindresorhus/p-queue/issues/219)，[修复 PR #220](https://github.com/sindresorhus/p-queue/pull/220) | `source/index.ts` | 1 | 1 |
| Go：[chi PR #1220](https://github.com/go-chi/chi/pull/1220) | `middleware/supress_notfound.go` | 3 | 未进入前 3 |
| Java：[JUnit 4 PR #1517](https://github.com/junit-team/junit4/pull/1517) | `src/main/java/org/junit/internal/runners/statements/FailOnTimeout.java` | 1 | 1 |

合计：结构索引 Recall@3 **4/4**、MRR **0.833**；去符号/导入 Recall@3 **3/4**、MRR **0.750**。chi 的目标文件仅排第 3，说明它还处于上下文预算的边缘。任务来源中只有 p-queue 使用独立 issue 正文，其余使用修复 PR 的故障描述；这不是四个严格同质的 issue 样本。

本次只测**修复前代码库中的文件定位**，不调用模型、不执行外部代码、不生成补丁。修复 PR 的改动文件是标注依据，不作为查询文本直接喂给检索器。四题是人工选取的小样本，且问题描述是改写版，不能推断通用准确率或 Agent 修复成功率。下一步如果测端到端修复，需要在这些 base commit 的隔离环境中运行 Agent、执行修复前后测试，并统计 Token、时延和 Provider 失败。

复现：按清单的四个固定 `commit` 下载源码，解压到 `express`、`pqueue`、`chi`、`junit4` 目录后运行：

```bash
ai-harness eval-real-repos benchmarks/real_issue_retrieval_cases.json \
  --repository express=/path/to/express \
  --repository pqueue=/path/to/pqueue \
  --repository chi=/path/to/chi \
  --repository junit4=/path/to/junit4 \
  --k 3 --output benchmarks/real_issue_retrieval_report.json
```

评测器核对清单中的源码 SHA-256；版本或内容不一致时会拒绝比较。
