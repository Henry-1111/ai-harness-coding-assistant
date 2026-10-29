# 多语言检索小样本评测（2026-10-29）

本次只评估文件级检索，不调用 LLM，也不评估 Agent 修改代码的成功率。数据集为仓库内人工编写的 8 个固定样例：JavaScript、TypeScript、Go、Java 各 2 个。每个查询标注一个目标文件，按 Recall@1 / MRR 比较。

对照实验保留相同的文件路径、全文内容、查询、检索参数和排序代码，只移除 Repo Map 的符号与静态导入字段；两组分别使用独立缓存。标签数据集 SHA-256：`267274715224d469541f84b586b588d7bf0bd015ac16bbb26761478aa8abee45`；样例文件内容 SHA-256：`aa8e3b3eaeac43565c4c42d1c981b6afe4212e557143a32d34271588618f3fdc`。原始逐 Case 结果见 [`multilang_retrieval_report.json`](multilang_retrieval_report.json)。

| 方案 | Recall@1 | MRR | 命中数 |
| --- | ---: | ---: | ---: |
| 结构索引开启 | 100% | 1.00 | 8/8 |
| 去除符号/导入 | 100% | 1.00 | 8/8 |

成对比较：结构索引胜 0、持平 8、去除结构索引胜 0。**这组简单样例未测出结构索引对最终排序的增益**；全文检索已足以找到目标文件。`bad_cases` 中若出现 `no_new_evidence`，表示渐进检索停止原因，不代表文件检索失败。

复现：

```bash
ai-harness eval-retrieval benchmarks/fixtures/multilang_retrieval \
  benchmarks/multilang_retrieval_cases.json --k 1 --compare-structure \
  --output benchmarks/multilang_retrieval_report.json
```

限制：仅 8 个短小的仓库内合成文件，没有跨模块复杂依赖、真实仓库噪声或跨语言编辑任务；不能将 100% 写作多语言通用 Recall，更不能据此声称结构索引显著提升。下一轮应使用独立标注的真实项目任务和更难的同名符号/长文件场景。
