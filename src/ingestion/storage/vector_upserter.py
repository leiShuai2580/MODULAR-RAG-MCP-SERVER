"""Vector Upserter for writing chunks to vector database. / 用于将分块写入向量数据库的向量 Upserter。

This module implements the VectorUpserter component, responsible for: / 本模块实现 VectorUpserter 组件，负责：
- Generating deterministic chunk IDs from content / 根据内容生成确定性的分块 ID
- Transforming chunks and vectors into storage records / 将分块和向量转换为存储记录
- Calling VectorStore for idempotent writes / 调用 VectorStore 进行幂等写入
- Supporting batch operations with consistent ordering / 支持保持顺序一致的批量操作

Design Principles: / 设计原则：
- Idempotent: Same content produces same ID, repeated writes safe / 幂等：相同内容生成相同 ID，重复写入安全
- Observable: Accepts TraceContext for future integration / 可观测：接收 TraceContext 以便未来集成
- Config-Driven: Uses VectorStoreFactory from settings / 配置驱动：基于 settings 使用 VectorStoreFactory
- Deterministic: Stable hash-based ID generation / 确定性：基于哈希生成稳定 ID
- Type-Safe: Full type hints and validation / 类型安全：完整类型提示和校验
"""

import hashlib
from typing import List, Dict, Any, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """Write chunks and vectors to vector database with idempotent guarantees. / 将分块和向量写入向量数据库，并提供幂等保证。
    
    This upserter receives chunks and their dense vectors from DenseEncoder, / 该 upserter 从 DenseEncoder 接收分块及其稠密向量，
    generates stable chunk IDs, and writes them to the configured vector store. / 生成稳定的分块 ID，并写入配置好的向量存储。
    
    Chunk ID Format: / 分块 ID 格式：
        {source_path_hash}_{chunk_index:04d}_{content_hash}
        
        Where: / 其中：
        - source_path_hash = first 8 chars of SHA256(source_path) / source_path_hash = SHA256(source_path) 的前 8 个字符
        - chunk_index = zero-padded 4-digit index / chunk_index = 左侧补零的 4 位索引
        - content_hash = first 8 chars of SHA256(chunk.text) / content_hash = SHA256(chunk.text) 的前 8 个字符
        
    This ensures: / 这可以保证：
        - Same content → same ID (idempotent) / 相同内容 → 相同 ID（幂等）
        - Content change → different ID (versioning) / 内容变化 → 不同 ID（版本化）
        - Human-readable with source traceability / 人类可读，并具备来源可追溯性
    
    Example: / 示例：
        >>> upserter = VectorUpserter(settings)
        >>> 
        >>> chunks = [
        ...     Chunk(id="temp1", text="Hello world", metadata={"source_path": "doc.pdf", "chunk_index": 0}),
        ...     Chunk(id="temp2", text="Python rocks", metadata={"source_path": "doc.pdf", "chunk_index": 1})
        ... ]
        >>> vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> 
        >>> upserter.upsert(chunks, vectors)
        >>> # Chunks written with stable IDs like: "a1b2c3d4_0000_e5f6g7h8" / 分块会以类似 "a1b2c3d4_0000_e5f6g7h8" 的稳定 ID 写入
    """
    
    def __init__(self, settings: Settings, collection_name: Optional[str] = None):
        """Initialize VectorUpserter with configured vector store. / 使用配置好的向量存储初始化 VectorUpserter。
        
        Args: / 参数：
            settings: Application settings containing vector_store configuration. / settings：包含 vector_store 配置的应用配置。
            collection_name: Optional collection name to override settings default. / collection_name：可选集合名称，用于覆盖 settings 默认值。
        
        Raises: / 异常：
            ValueError: If settings are invalid or vector store cannot be created. / ValueError：当 settings 无效或无法创建向量存储时抛出。
        """
        self.settings = settings
        kwargs = {}
        if collection_name:
            kwargs['collection_name'] = collection_name
        self.vector_store = VectorStoreFactory.create(settings, **kwargs)
    
    def upsert(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """Upsert chunks with their vectors to vector store. / 将分块及其向量 upsert 到向量存储。
        
        Args: / 参数：
            chunks: List of Chunk objects to store. / chunks：要存储的 Chunk 对象列表。
            vectors: List of embedding vectors (same order and length as chunks). / vectors：嵌入向量列表（顺序和长度与 chunks 相同）。
            trace: Optional TraceContext for observability (reserved for Stage F). / trace：用于可观测性的可选 TraceContext（为阶段 F 预留）。
        
        Returns: / 返回：
            List of generated chunk IDs (same order as input chunks). / 生成的分块 ID 列表（顺序与输入 chunks 相同）。
        
        Raises: / 异常：
            ValueError: If chunks and vectors lengths don't match, or if required / ValueError：当 chunks 和 vectors 长度不匹配，或必需
                       metadata fields are missing. / 元数据字段缺失时抛出。
            RuntimeError: If vector store upsert operation fails. / RuntimeError：当向量存储 upsert 操作失败时抛出。
        
        Example:
            >>> chunks = [Chunk(...), Chunk(...)]
            >>> vectors = [[0.1, 0.2], [0.3, 0.4]]
            >>> chunk_ids = upserter.upsert(chunks, vectors)
            >>> len(chunk_ids) == len(chunks)  # True
        """
        # Validate input lengths match / 校验输入长度是否匹配
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match vector count ({len(vectors)})"
            )
        
        if not chunks:
            raise ValueError("Cannot upsert empty chunks list")
        
        # Generate stable chunk IDs and build records / 生成稳定分块 ID 并构建记录
        records = []
        chunk_ids = []
        
        for chunk, vector in zip(chunks, vectors):
            # Generate deterministic chunk ID / 生成确定性分块 ID
            chunk_id = self._generate_chunk_id(chunk)
            chunk_ids.append(chunk_id)
            
            # Build storage record / 构建存储记录
            record = {
                "id": chunk_id,
                "vector": vector,
                "metadata": {
                    **chunk.metadata,  # Preserve all original metadata / 保留所有原始元数据
                    "text": chunk.text,  # Store text for retrieval / 存储文本以便检索
                    "chunk_id": chunk_id,  # Redundant but useful for queries / 冗余字段，但对查询有用
                },
            }
            records.append(record)
        
        # Perform idempotent upsert / 执行幂等 upsert
        try:
            self.vector_store.upsert(records, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Vector store upsert failed: {str(e)}"
            ) from e
        
        return chunk_ids
    
    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """Generate deterministic chunk ID from content. / 根据内容生成确定性的分块 ID。
        
        Args: / 参数：
            chunk: Chunk object to generate ID for. / chunk：要为其生成 ID 的 Chunk 对象。
        
        Returns: / 返回：
            Stable chunk ID string. / 稳定的分块 ID 字符串。
        
        Raises: / 异常：
            ValueError: If required metadata fields are missing. / ValueError：当必需元数据字段缺失时抛出。
        """
        # Validate required metadata / 校验必需元数据
        if "source_path" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")
        if "chunk_index" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'chunk_index'")
        
        source_path = chunk.metadata["source_path"]
        chunk_index = chunk.metadata["chunk_index"]
        
        # Compute stable hashes / 计算稳定哈希
        source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
        
        # Format: {source_hash}_{index:04d}_{content_hash} / 格式：{source_hash}_{index:04d}_{content_hash}
        chunk_id = f"{source_hash}_{chunk_index:04d}_{content_hash}"
        
        return chunk_id
    
    def upsert_batch(
        self,
        batches: List[tuple[List[Chunk], List[List[float]]]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """Upsert multiple batches of chunks and vectors. / Upsert 多批分块和向量。
        
        This is a convenience method for processing outputs from BatchProcessor. / 这是用于处理 BatchProcessor 输出的便捷方法。
        All batches are flattened and processed in a single upsert operation / 所有批次会被展平，并在一次 upsert 操作中处理，
        to maintain ordering and reduce vector store round trips. / 以保持顺序并减少向量存储往返。
        
        Args: / 参数：
            batches: List of (chunks, vectors) tuples from batch processing. / batches：来自批处理的 (chunks, vectors) 元组列表。
            trace: Optional TraceContext for observability. / trace：用于可观测性的可选 TraceContext。
        
        Returns: / 返回：
            List of all generated chunk IDs in order. / 按顺序排列的所有生成分块 ID 列表。
        
        Example:
            >>> batch1 = ([chunk1, chunk2], [[0.1, 0.2], [0.3, 0.4]])
            >>> batch2 = ([chunk3], [[0.5, 0.6]])
            >>> chunk_ids = upserter.upsert_batch([batch1, batch2])
            >>> len(chunk_ids)  # 3
        """
        # Flatten all batches / 展平所有批次
        all_chunks = []
        all_vectors = []
        
        for chunks, vectors in batches:
            all_chunks.extend(chunks)
            all_vectors.extend(vectors)
        
        # Single upsert operation / 单次 upsert 操作
        return self.upsert(all_chunks, all_vectors, trace=trace)
