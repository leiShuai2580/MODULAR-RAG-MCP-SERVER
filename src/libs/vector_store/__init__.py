"""
Vector Store Module. / 向量存储模块。

This package contains vector store abstractions and implementations: / 此包包含向量存储抽象和实现：
- Base vector store class / 基础向量存储类
- Vector store factory / 向量存储工厂
- Implementations (Chroma, etc.) / 实现（Chroma 等）
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# Auto-register ChromaStore provider / 自动注册 ChromaStore provider
try:
    from src.libs.vector_store.chroma_store import ChromaStore
    VectorStoreFactory.register_provider('chroma', ChromaStore)
except ImportError:
    # ChromaDB not installed, skip registration / 未安装 ChromaDB，跳过注册
    pass

__all__ = [
    'BaseVectorStore',
    'VectorStoreFactory',
    'ChromaStore',
]
