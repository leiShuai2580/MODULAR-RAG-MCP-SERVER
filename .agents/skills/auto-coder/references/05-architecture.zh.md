# 系统架构与模块设计（中文阅读版）

> 对应原文件：`05-architecture.md`
> 原文件主体已经是中文，夹带的英文主要是协议名、模块名、类名、目录名和技术术语。本文件用于说明这些英文标识的中文含义；源码标识不翻译，以保持与代码一致。

## 中文概览

该文档描述 Modular RAG MCP Server 的分层架构：

- MCP Clients：外部客户端层，包括 GitHub Copilot、Claude Desktop 和其他 MCP Agent。
- MCP Server 层：协议接口层，负责处理 `tools/list`、`tools/call`、`resources/*` 等 MCP 请求。
- Core 层：核心业务逻辑层，包括查询预处理、混合检索、重排序、响应构建和链路追踪。
- Storage 层：存储层，包括 Chroma 向量库、BM25 索引、图片文件和完整性记录。
- Ingestion 层：离线摄取层，负责加载文档、切分 Chunk、元数据增强、图片理解、Embedding 编码和写入存储。
- Libs 层：可插拔能力层，包括 LLM、Embedding、Vector Store、Splitter、Reranker、Evaluator 和 Vision LLM 的抽象与默认实现。
- Dashboard 层：Streamlit 可视化管理平台，用于数据浏览、摄取管理、Trace 查看和评估。

## 英文标识对照

| 英文原文 | 中文含义 |
|----------|----------|
| MCP Clients | MCP 客户端 |
| JSON-RPC 2.0 | JSON-RPC 2.0 协议 |
| Stdio Transport | 标准输入输出传输 |
| Protocol Handler | 协议处理器 |
| Query Engine | 查询引擎 |
| Query Processor | 查询预处理器 |
| Hybrid Search Engine | 混合检索引擎 |
| Dense Route | 稠密向量检索路径 |
| Sparse Route | 稀疏关键词检索路径 |
| Fusion | 多路召回融合 |
| RRF | Reciprocal Rank Fusion，倒数排序融合 |
| Reranker | 重排序模块 |
| Response Builder | 响应构建器 |
| Citation | 引用信息 |
| Trace Collector | 链路追踪收集器 |
| Vector Store | 向量存储 |
| Chroma DB | Chroma 向量数据库 |
| Ingestion Pipeline | 数据摄取流水线 |
| Loader | 文档加载器 |
| Splitter | 文档切分器 |
| Transform | 转换增强阶段 |
| Embedding | 向量编码 |
| Upsert | 写入或更新存储 |
| Factory | 工厂类 |
| Base Interface | 基础抽象接口 |
| Streamlit Dashboard | Streamlit 可视化面板 |

