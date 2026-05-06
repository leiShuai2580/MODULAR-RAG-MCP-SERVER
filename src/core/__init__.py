"""
Core Layer - Core business logic. / 核心层 - 核心业务逻辑。

This package contains the core business logic including: / 本包包含核心业务逻辑，包括：
- Configuration management (settings.py) / 配置管理（settings.py）
- Core data types (types.py) - shared contracts for all pipeline stages / 核心数据类型（types.py）- 所有流水线阶段的共享契约
- Query engine / 查询引擎
- Response building / 响应构建
- Trace collection / 追踪收集
"""

from src.core.types import Document, Chunk, ChunkRecord, Metadata, Vector, SparseVector

__all__ = [
    "Document",
    "Chunk", 
    "ChunkRecord",
    "Metadata",
    "Vector",
    "SparseVector"
]
