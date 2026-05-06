# QA 专项测试计划（中文版本）

> 对应原文件：`QA_TEST_PLAN.md`
> 原文件主体已经是中文，本文件作为中文阅读版，统一补充英文页面名、命令名、状态名和测试分组的中文解释。CLI 命令、配置键、工具名和页面标题保留英文原文，避免影响执行含义。

## 测试目标

验证 Modular RAG MCP Server 在以下方面是否稳定可用：

- Dashboard 六个页面的加载、交互、空状态、数据展示与错误提示。
- CLI 脚本的摄取、查询、评估、参数处理与异常处理。
- MCP Server 的 JSON-RPC 协议交互、工具列表、工具调用和引用返回。
- Provider 切换、Reranker 切换、无效密钥、无效配置、目录缺失等容错场景。
- 数据生命周期闭环，包括摄取、查询、删除、重新摄取、多集合隔离和清空恢复。
- 多种文档场景，包括中文技术文档、表格图表、多页长文档、代码块文档和批量摄取。

## 状态定义

| 原状态 | 中文含义 |
|--------|----------|
| `Empty` | 空数据状态：没有集合数据，也没有 Trace。 |
| `Baseline` | 基准状态：包含默认集合、测试集合和可用于验证的 Trace。 |
| `DeepSeek` | DeepSeek 文本 LLM 配置状态，同时关闭 Vision 能力。 |
| `Rerank_LLM` | 启用 LLM Reranker 的基准状态。 |
| `NoVision` | 关闭 Vision LLM 的基准状态。 |
| `InvalidKey` | LLM API Key 无效的故障状态。 |
| `InvalidEmbedKey` | Embedding API Key 无效的故障状态。 |
| `Any` | 任意系统状态均可执行。 |

## 测试分组

| 分组 | 中文说明 |
|------|----------|
| A. Dashboard — Overview | 系统总览页面，验证组件配置、数据资产和 Trace 统计。 |
| B. Dashboard — Data Browser | 数据浏览页面，验证集合、文档、Chunk、Metadata、图片预览和清空数据流程。 |
| C. Dashboard — Ingestion Manager | 摄取管理页面，验证上传、摄取、删除、重复摄取和进度展示。 |
| D. Dashboard — Ingestion Traces | 摄取追踪页面，验证阶段耗时、各阶段详情、失败 Trace 和排序。 |
| E. Dashboard — Query Traces | 查询追踪页面，验证关键词过滤、Dense/Sparse/Fusion/Rerank 阶段展示和 Ragas 评估。 |
| F. Dashboard — Evaluation Panel | 评估面板，验证 Ragas、自定义集合、无效 Golden Path 和历史记录。 |
| G. CLI — ingest.py | 命令行摄取，验证单文件、目录、集合、dry-run、force、verbose 和异常路径。 |
| H. CLI — query.py | 命令行查询，验证中文查询、Top-K、集合隔离、verbose、空查询和 Trace 记录。 |
| I. CLI — evaluate.py | 命令行评估，验证默认评估、自定义测试集、JSON 输出和 no-search 模式。 |
| J. MCP Server | MCP 协议交互，验证启动、tools/list、tools/call、多模态结果和参数错误。 |
| K. Provider 切换 — DeepSeek | DeepSeek LLM 切换及相关摄取、查询、评估和回退验证。 |
| L. Provider 切换 — Reranker | None、Cross-Encoder、LLM Reranker 的切换、首次加载和失败回退。 |
| M. 配置变更与容错 | API Key、Endpoint、YAML、缺失字段、目录缺失、Trace 损坏和参数调整。 |
| N. 数据生命周期闭环 | 摄取、查询、删除、恢复、多集合隔离和同文件多集合验证。 |
| O. 文档替换与多场景验证 | 不同文档类型、长文档、图表、代码块、批量摄取和替换评估。 |

## 英文术语对照

| 英文原文 | 中文含义 |
|----------|----------|
| Dashboard | 可视化管理面板 |
| Overview | 系统总览 |
| Data Browser | 数据浏览器 |
| Ingestion Manager | 摄取管理器 |
| Ingestion Traces | 摄取追踪 |
| Query Traces | 查询追踪 |
| Evaluation Panel | 评估面板 |
| CLI | 命令行工具 |
| Provider | 模型或服务提供方 |
| Reranker | 重排序器 |
| Trace | 链路追踪记录 |
| Golden Test Set | 黄金测试集 |
| Dense Retrieval | 稠密向量检索 |
| Sparse Retrieval | 稀疏关键词检索 |
| Fusion | 融合排序 |
| Ragas Evaluate | Ragas 指标评估 |

## 执行说明

原计划中的命令、路径、配置键和 UI 按钮文案按原样保留。例如 `python scripts/ingest.py`、`settings.yaml`、`query_knowledge_hub`、`top-k` 等都属于可执行或可定位标识，不建议翻译成纯中文。

