"""
Embedding Module. / 嵌入模块。

This package contains embedding components: / 此包包含嵌入组件：
- Dense encoder / 稠密编码器
- Sparse encoder (BM25) / 稀疏编码器（BM25）
- Batch processor / 批处理器
"""

from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchResult

__all__ = ["DenseEncoder", "SparseEncoder", "BatchProcessor", "BatchResult"]
