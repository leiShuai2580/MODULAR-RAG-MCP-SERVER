# QA 测试进度（中文版本）

> 对应原文件：`QA_TEST_PROGRESS.md`
> 原文件包含大量英文工具输出、命令名、指标名和日志片段。本文件提供中文阅读版，保留必要的命令、文件名、工具名和指标原文。

## 总览

- 生成时间：2026-02-25 23:02
- 测试用例总数：150
- 通过：138
- 失败：0
- 跳过：12
- 需要修复后复测：0
- 待执行：0

## 状态图例

| 标记 | 中文含义 |
|------|----------|
| ✅ Pass | 测试通过 |
| ❌ Fail | 测试失败 |
| ⏭️ Skip | 跳过执行 |
| 🔧 Fix | 已修复，需重新测试 |
| ⬜ Pending | 待执行 |

## 分组进度

| 分组 | 中文说明 | 当前结果 |
|------|----------|----------|
| A. Overview 页面 | 验证系统总览、组件配置、集合统计和 Trace 统计。 | 全部通过 |
| B. Data Browser 页面 | 验证集合选择、文档列表、Chunk 展开、Metadata、图片预览和清空数据确认流程。 | 全部通过 |
| C. Ingestion Manager 页面 | 验证上传摄取、图片 PDF、集合写入、幂等性、删除和阶段进度。 | 全部通过 |
| D. Ingestion Traces 页面 | 验证摄取 Trace 列表、阶段详情、瀑布耗时、失败 Trace 和倒序排序。 | 全部通过 |
| E. Query Traces 页面 | 验证查询 Trace、关键词过滤、Dense/Sparse/Fusion/Rerank 展示和 Ragas 按钮。 | 全部通过 |
| F. Evaluation Panel 页面 | 验证 Ragas 评估、每条查询详情、无效 Golden Path、历史记录和空知识库评估。 | 全部通过 |
| G. CLI 摄取 | 验证 `ingest.py` 的单文件、目录、集合、dry-run、force、verbose 和异常路径处理。 | 全部通过 |
| H. CLI 查询 | 验证 `query.py` 的中文查询、Top-K、集合隔离、verbose、空查询、长查询和 Trace 记录。 | 全部通过 |
| I. CLI 评估 | 验证 `evaluate.py` 的默认评估、自定义测试集、JSON 输出、no-search 和缺失文件错误。 | 全部通过 |
| J. MCP Server 协议 | 验证服务启动、工具列表、工具调用、多模态内容、无效参数和引用透明性。 | 全部通过 |
| K. DeepSeek Provider 切换 | 因没有 DeepSeek API Key，DeepSeek 相关 12 个用例跳过。 | 全部跳过 |
| L. Reranker 模式 | 验证 None、Cross-Encoder、LLM Reranker、Top-K 和失败回退。 | 全部通过 |
| M. 配置变更与容错 | 验证无效 Key、无效 Endpoint、YAML 错误、缺失字段、Trace 损坏和参数调整。 | 全部通过 |
| N. 数据生命周期闭环 | 验证摄取、查询、删除、重新摄取、多集合隔离和清空恢复。 | 全部通过 |
| O. 文档替换与多场景验证 | 验证中文技术文档、表格图表、长文档、代码块、批量摄取和文档替换评估。 | 全部通过 |

## 关键结论

- Dashboard 相关页面通过 AppTest 完成无头渲染验证，页面加载、组件显示、交互按钮、空状态和错误提示均可正常工作。
- CLI 摄取、查询和评估脚本均能在成功路径和异常路径下返回明确结果。
- MCP Server 能按 JSON-RPC 协议完成初始化、工具发现和工具调用，并返回可追溯引用信息。
- Reranker 支持关闭、Cross-Encoder 和 LLM 三种模式；当 LLM Reranker 失败时，系统可回退到 RRF 融合结果。
- DeepSeek 相关用例因缺少 API Key 被跳过，不计为失败。
- 多集合隔离、删除恢复、清空数据后重建等数据生命周期场景均已通过。

## 术语说明

| 英文原文 | 中文含义 |
|----------|----------|
| AppTest | Streamlit 的无头应用测试能力 |
| CLI | 命令行工具 |
| exit code | 命令退出码 |
| Dense returned | 稠密检索返回数量 |
| Sparse returned | 稀疏检索返回数量 |
| Fusion returned | 融合排序返回数量 |
| Rerank results | 重排序结果 |
| pytest | Python 测试框架 |
| VERDICT=PASS | 多步骤验证最终通过 |
| Skip | 由于前置条件不足而跳过 |

