# 项目排期（中文阅读版）

> 对应原文件：`06-schedule.md`
> 原文件主体已经是中文，英文主要是任务编号、模块名、类名、命令名和测试指标。本文件补充中文说明，便于快速阅读。

## 中文概览

排期文档按 A 到 I 九个阶段推进项目落地：

- 阶段 A：工程骨架与测试基座。
- 阶段 B：Libs 可插拔层，包括工厂、抽象接口和默认实现。
- 阶段 C：Ingestion Pipeline，将文档转为 Chunk、Embedding 并写入存储。
- 阶段 D：Retrieval 查询链路，包括 Dense、Sparse、RRF 和可选 Rerank。
- 阶段 E：MCP Server 层和工具能力落地。
- 阶段 F：Trace 基础设施和链路打点。
- 阶段 G：Streamlit Dashboard 管理平台。
- 阶段 H：评估体系。
- 阶段 I：端到端验收和文档收口。

## 英文标识对照

| 英文原文 | 中文含义 |
|----------|----------|
| Progress Tracking | 进度跟踪 |
| MVP | 最小可用版本 |
| TDD | 测试驱动开发 |
| Mock | 模拟对象或模拟调用 |
| Fake | 测试用替身实现 |
| Factory | 工厂模式 |
| BaseLLM / BaseEmbedding / BaseSplitter | LLM、Embedding、Splitter 的抽象基类 |
| VectorStore | 向量存储 |
| Reranker | 重排序器 |
| Evaluator | 评估器 |
| OpenAI-Compatible | OpenAI 兼容接口 |
| Ollama | 本地模型服务 |
| Cross-Encoder | 交叉编码重排序模型 |
| Vision LLM | 视觉语言模型 |
| TraceContext | 链路追踪上下文 |
| Dashboard | 可视化管理面板 |
| E2E | 端到端测试 |

