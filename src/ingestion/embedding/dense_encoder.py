"""Dense Encoder for generating embeddings from text chunks. / 用于从文本块生成嵌入的稠密编码器。

This module implements the Dense Encoder component of the Ingestion Pipeline, / 此模块实现 Ingestion Pipeline 的 Dense Encoder 组件，
responsible for converting text chunks into dense vector representations using / 负责使用可配置的嵌入 provider
configurable embedding providers. / 将文本块转换为稠密向量表示。

Design Principles: / 设计原则：
- Config-Driven: Uses factory pattern to obtain embedding provider from settings / 配置驱动：使用工厂模式从配置获取嵌入 provider
- Batch Processing: Optimizes API calls through batching / 批处理：通过分批优化 API 调用
- Observable: Accepts TraceContext for future observability integration / 可观测：接收 TraceContext 以便未来集成可观测能力
- Error Handling: Individual failures shouldn't crash entire batch / 错误处理：单个失败不应导致整个批次崩溃
- Deterministic: Same inputs produce same outputs / 确定性：相同输入产生相同输出
"""

from typing import List, Optional, Any
from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """Encodes text chunks into dense vectors using BaseEmbedding provider. / 使用 BaseEmbedding provider 将文本块编码为稠密向量。
    
    This encoder acts as a bridge between the ingestion pipeline and the / 此编码器充当摄取流水线和
    pluggable embedding layer. It handles batching, error recovery, and / 可插拔嵌入层之间的桥梁。它处理批处理、错误恢复，
    maintains alignment between input chunks and output vectors. / 并保持输入块与输出向量之间的对齐。
    
    Design: / 设计：
    - Dependency Injection: Receives BaseEmbedding instance (no direct factory call) / 依赖注入：接收 BaseEmbedding 实例（不直接调用工厂）
    - Batch-First: Processes all chunks in configurable batch sizes / 批处理优先：按可配置批大小处理所有块
    - Stateless: No internal state between encode() calls / 无状态：encode() 调用之间不保留内部状态
    
    Example: / 示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> encoder = DenseEncoder(embedding, batch_size=32)
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world", metadata={})]
        >>> vectors = encoder.encode(chunks)
        >>> print(len(vectors))  # 1
        >>> print(len(vectors[0]))  # dimension (e.g., 1536)
    """
    
    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
    ):
        """Initialize DenseEncoder. / 初始化 DenseEncoder。
        
        Args: / 参数：
            embedding: Embedding provider instance (from EmbeddingFactory) / 嵌入 provider 实例（来自 EmbeddingFactory）
            batch_size: Number of chunks to process per API call (default: 100) / 每次 API 调用处理的块数量（默认：100）
        
        Raises: / 异常：
            ValueError: If batch_size <= 0 / 如果 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.embedding = embedding
        self.batch_size = batch_size
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """Encode chunks into dense vectors. / 将块编码为稠密向量。
        
        This method: / 此方法：
        1. Extracts text from each chunk / 从每个块提取文本
        2. Batches texts according to batch_size / 按 batch_size 对文本分批
        3. Calls embedding.embed() for each batch / 对每个批次调用 embedding.embed()
        4. Concatenates results maintaining chunk order / 拼接结果并保持块顺序
        
        Args: / 参数：
            chunks: List of Chunk objects to encode / 要编码的 Chunk 对象列表
            trace: Optional TraceContext for observability (reserved for Stage F) / 用于可观测性的可选 TraceContext（为 Stage F 预留）
        
        Returns: / 返回：
            List of dense vectors (one per chunk, in same order). / 稠密向量列表（每个块一个，顺序相同）。
            Each vector is a list of floats with dimension matching the embedding model. / 每个向量是浮点数列表，维度与嵌入模型匹配。
        
        Raises: / 异常：
            ValueError: If chunks list is empty / 如果 chunks 列表为空
            RuntimeError: If embedding provider fails for all batches / 如果嵌入 provider 对所有批次都失败
        
        Example: / 示例：
            >>> chunks = [
            ...     Chunk(id="1", text="First chunk", metadata={}),
            ...     Chunk(id="2", text="Second chunk", metadata={})
            ... ]
            >>> vectors = encoder.encode(chunks)
            >>> len(vectors) == len(chunks)  # True
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        # Extract text from chunks / 从块中提取文本
        texts = [chunk.text for chunk in chunks]
        
        # Validate that all texts are non-empty / 校验所有文本都非空
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text"
                )
        
        # Process in batches / 分批处理
        all_vectors: List[List[float]] = []
        
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # Call embedding provider / 调用嵌入 provider
                batch_vectors = self.embedding.embed(
                    texts=batch_texts,
                    trace=trace,
                )
                
                # Validate output shape / 校验输出形状
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding provider returned {len(batch_vectors)} vectors "
                        f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                    )
                
                all_vectors.extend(batch_vectors)
                
            except Exception as e:
                # Re-raise with context about which batch failed / 带上失败批次上下文后重新抛出
                raise RuntimeError(
                    f"Failed to encode batch {batch_start}-{batch_end}: {str(e)}"
                ) from e
        
        # Final validation / 最终校验
        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Vector count mismatch: got {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks"
            )
        
        # Validate vector dimensions are consistent / 校验向量维度一致
        if all_vectors:
            expected_dim = len(all_vectors[0])
            for i, vec in enumerate(all_vectors):
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"Inconsistent vector dimensions: vector {i} has "
                        f"{len(vec)} dimensions, expected {expected_dim}"
                    )
        
        return all_vectors
    
    def get_batch_count(self, num_chunks: int) -> int:
        """Calculate number of batches needed for given chunk count. / 计算给定块数量所需的批次数。
        
        Utility method for logging/progress tracking. / 用于日志/进度追踪的工具方法。
        
        Args: / 参数：
            num_chunks: Number of chunks to encode / 要编码的块数量
        
        Returns: / 返回：
            Number of batches required / 所需批次数
        """
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size
