# 真实开源仓库多语言检索评测（2026-10-29）

这是一组**文件级检索**任务，不调用模型、不执行下载仓库的代码，也不衡量 Agent 修改代码的成功率。人工阅读固定提交中的实现后，为每个仓库标注 3 个维护任务和目标文件；JUnit 4 的超时任务标注了配置规则和实际执行超时的两个文件。查询不包含目标文件路径。数据见 [`real_repo_retrieval_cases.json`](real_repo_retrieval_cases.json)，逐 Case 排名见 [`real_repo_retrieval_report.json`](real_repo_retrieval_report.json)。

| 语言 | 仓库 | 固定版本 | 任务数 | 结构 Recall@3 | 去符号/导入 Recall@3 |
| --- | --- | --- | ---: | ---: | ---: |
| JavaScript | [Express](https://github.com/expressjs/express) | `1c6f1b0fe400` | 3 | 100% | 66.7% |
| TypeScript | [p-queue](https://github.com/sindresorhus/p-queue) | `7f272ab5ff22` | 3 | 66.7% | 66.7% |
| Go | [chi](https://github.com/go-chi/chi) | `9f8adb1aabaa` | 3 | 100% | 100% |
| Java | [JUnit 4](https://github.com/junit-team/junit4) | `890f3c972647` | 3 | 66.7% | 0% |
| **合计** | 4 个仓库 | | **12** | **83.3%** | **58.3%** |

总体 MRR 分别为 0.736 和 0.486。按首个相关文件的名次成对比较：结构索引胜 5、持平 5、负 2。这里的「去符号/导入」仍保留路径、全文、BM25、查询改写和相同检索参数，因此是结构字段的消融，不是“没有检索”的对照。

值得保留的失败：p-queue 的稳定优先级任务从去结构时的第 1 降为结构索引下的第 2；二分插入辅助函数在结构索引开启时未进入 Top 3，但去结构后排第 2。JUnit 4 的超时任务两组均未找到配置规则与执行实现。测试文件和语义相近的调用处容易挤占目标实现文件的位置。结果说明结构索引在这些样本上有正向信号，也暴露了排序回退；**12 个非随机任务不足以推断跨仓库通用提升，更不能替代真实 Agent 任务成功率**。

复现时下载清单中的四个固定提交，分别放到 `express`、`pqueue`、`chi`、`junit4` 目录。可使用 GitHub 的固定版本压缩包，例如 `https://codeload.github.com/expressjs/express/tar.gz/1c6f1b0fe4000977b9a7b4f34097e4d527dd9824`。评测器核对每个仓库的已索引文件内容 SHA-256；若目录内容与本次实验不一致，将拒绝给出可比结果。然后运行：

```bash
ai-harness eval-real-repos benchmarks/real_repo_retrieval_cases.json \
  --repository express=/path/to/express \
  --repository pqueue=/path/to/pqueue \
  --repository chi=/path/to/chi \
  --repository junit4=/path/to/junit4 \
  --k 3 --output benchmarks/real_repo_retrieval_report.json
```

本地下载失败时可重试单个固定版本压缩包；不要把缺失仓库、不同提交或不完整下载产生的分数与本报告混用。下一轮应扩大独立标注任务，覆盖跨文件定位、同名符号、长期维护中的真实 issue，并预先冻结标签后再比较算法。
