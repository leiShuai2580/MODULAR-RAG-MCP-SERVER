# 系统架构与模块设计（中文阅读版）

> 对应原文件：`05-architecture.md`
> 原文件主体已经是中文，夹带的英文主要是协议名、模块名、类名、目录名和技术术语。本文件用于说明这些英文标识的中文含义；源码标识不翻译，以保持与代码一致。

## 中文概览

该文档描述 Modular RAG MCP Server 的分层架构：MCP 客户端层、MCP Server 接口层、Core 核心业务层、Storage 存储层、Ingestion 摄取层、Libs 可插拔能力层和 Streamlit Dashboard 管理层。

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
| RRF | 倒数排序融合 |
| Reranker | 重排序模块 |
| Response Builder | 响应构建器 |
| Citation | 引用信息 |
| Trace Collector | 链路追踪收集器 |
| Vector Store | 向量存储 |
| Ingestion Pipeline | 数据摄取流水线 |
| Loader / Splitter / Transform | 加载器、切分器、转换增强阶段 |
| Embedding / Upsert | 向量编码、写入或更新存储 |
| Factory | 工厂类 |
| Streamlit Dashboard | Streamlit 可视化面板 |

