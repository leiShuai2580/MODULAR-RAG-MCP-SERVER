"""
Embedding Module. / 嵌入模块。

This package contains embedding service abstractions and implementations: / 此包包含嵌入服务抽象和实现：
- Base embedding class / 基础嵌入类
- Embedding factory / 嵌入工厂
- Provider implementations (OpenAI, Azure, Ollama) / Provider 实现（OpenAI、Azure、Ollama）
"""

from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "AzureEmbedding",
    "OllamaEmbedding",
]
