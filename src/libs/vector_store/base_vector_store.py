"""Abstract base class for VectorStore providers. / VectorStore provider 的抽象基类。

This module defines the pluggable interface for VectorStore providers, / 此模块定义 VectorStore provider 的可插拔接口，
enabling seamless switching between different backends (Chroma, Qdrant, Milvus, etc.) / 支持在不同后端（Chroma、Qdrant、Milvus 等）之间无缝切换，
through configuration-driven instantiation. / 并通过配置驱动的实例化完成选择。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """Abstract base class for VectorStore providers. / VectorStore provider 的抽象基类。
    
    All VectorStore implementations must inherit from this class and implement / 所有 VectorStore 实现都必须继承此类并实现
    the upsert() and query() methods. This ensures consistent interface across / upsert() 和 query() 方法。这确保不同
    different providers (Chroma, Qdrant, Milvus, etc.). / provider（Chroma、Qdrant、Milvus 等）之间接口一致。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Subclasses can be swapped without changing upstream code. / 可插拔：无需修改上游代码即可替换子类。
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选 TraceContext 以集成可观测能力。
    - Config-Driven: Instances are created via factory based on settings. / 配置驱动：基于 settings 通过工厂创建实例。
    - Idempotent: upsert() operations should be safely repeatable. / 幂等：upsert() 操作应可安全重复执行。
    """
    
    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update records in the vector store. / 在向量存储中插入或更新记录。
        
        Args: / 参数：
            records: List of records to upsert. Each record is a dict with: / 要 upsert 的记录列表。每条记录都是包含以下字段的字典：
                - 'id': Unique identifier (str) / 'id'：唯一标识符（str）
                - 'vector': Embedding vector (List[float]) / 'vector'：嵌入向量（List[float]）
                - 'metadata': Optional metadata dict (source, chunk_index, etc.) / 'metadata'：可选元数据字典（source、chunk_index 等）
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Raises: / 异常：
            ValueError: If records list is empty or contains invalid entries. / 如果 records 列表为空或包含无效条目。
            RuntimeError: If the vector store operation fails. / 如果向量存储操作失败。
        
        Example: / 示例：
            >>> records = [
            ...     {
            ...         'id': 'doc1_chunk0',
            ...         'vector': [0.1, 0.2, ..., 0.5],
            ...         'metadata': {'source': 'doc1.pdf', 'page': 1}
            ...     }
            ... ]
            >>> vector_store.upsert(records)
        
        Notes: / 说明：
            - This operation should be idempotent: upserting the same record / 此操作应是幂等的：多次 upsert 同一记录
              multiple times should produce the same final state. / 应产生相同最终状态。
            - Implementations should handle batch operations efficiently. / 实现应高效处理批量操作。
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Query the vector store for similar vectors. / 在向量存储中查询相似向量。
        
        Args: / 参数：
            vector: Query vector (embedding) to search for. / 用于搜索的查询向量（嵌入）。
            top_k: Maximum number of results to return. / 返回结果最大数量。
            filters: Optional metadata filters (e.g., {'source': 'doc1.pdf'}). / 可选元数据过滤器（例如 {'source': 'doc1.pdf'}）。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Returns: / 返回：
            List of matching records, sorted by similarity (descending). / 匹配记录列表，按相似度降序排序。
            Each record is a dict with: / 每条记录是包含以下字段的字典：
                - 'id': Record identifier / 'id'：记录标识符
                - 'score': Similarity score (higher = more similar) / 'score'：相似度分数（越高越相似）
                - 'metadata': Associated metadata / 'metadata'：关联元数据
                - 'vector': Optional, the stored vector (provider-dependent) / 'vector'：可选，存储的向量（取决于 provider）
        
        Raises: / 异常：
            ValueError: If vector is empty or top_k is invalid. / 如果 vector 为空或 top_k 无效。
            RuntimeError: If the vector store query fails. / 如果向量存储查询失败。
        
        Example: / 示例：
            >>> query_vector = [0.1, 0.2, ..., 0.5]
            >>> results = vector_store.query(query_vector, top_k=5)
            >>> for result in results:
            ...     print(f"ID: {result['id']}, Score: {result['score']}")
        """
        pass
    
    def validate_records(self, records: List[Dict[str, Any]]) -> None:
        """Validate records before upsert. / upsert 前校验记录。
        
        Args: / 参数：
            records: List of records to validate. / 要校验的记录列表。
        
        Raises: / 异常：
            ValueError: If records list is empty or contains invalid entries. / 如果 records 列表为空或包含无效条目。
        """
        if not records:
            raise ValueError("Records list cannot be empty")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at index {i} is not a dict (type: {type(record).__name__})"
                )
            
            # Validate required fields / 校验必需字段
            if 'id' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'id'")
            if 'vector' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'vector'")
            
            # Validate vector format / 校验向量格式
            vector = record['vector']
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    f"Record at index {i} has invalid vector type: {type(vector).__name__}. "
                    "Expected list or tuple of floats."
                )
            
            if not vector:
                raise ValueError(f"Record at index {i} has empty vector")
    
    def validate_query_vector(self, vector: List[float], top_k: int) -> None:
        """Validate query parameters. / 校验查询参数。
        
        Args: / 参数：
            vector: Query vector to validate. / 要校验的查询向量。
            top_k: Number of results to validate. / 要校验的结果数量。
        
        Raises: / 异常：
            ValueError: If parameters are invalid. / 如果参数无效。
        """
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Query vector must be a list or tuple, got {type(vector).__name__}"
            )
        
        if not vector:
            raise ValueError("Query vector cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Delete records from the vector store by IDs. / 按 ID 从向量存储删除记录。
        
        Args: / 参数：
            ids: List of record IDs to delete. / 要删除的记录 ID 列表。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Raises: / 异常：
            ValueError: If ids list is empty. / 如果 ids 列表为空。
            RuntimeError: If the delete operation fails. / 如果删除操作失败。
            NotImplementedError: If the provider doesn't support deletion. / 如果 provider 不支持删除。
        
        Notes: / 说明：
            This is an optional operation. Providers that don't support / 这是可选操作。不支持
            deletion should raise NotImplementedError with a clear message. / 删除的 provider 应抛出带清晰信息的 NotImplementedError。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement delete() method. "
            "This operation is optional and provider-dependent."
        )
    
    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Clear all records from the vector store or a specific collection. / 清空向量存储或特定 collection 中的所有记录。
        
        Args: / 参数：
            collection_name: Optional collection name to clear. If None, clears default collection. / 要清空的可选 collection 名称。为 None 时清空默认 collection。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Raises: / 异常：
            RuntimeError: If the clear operation fails. / 如果清空操作失败。
            NotImplementedError: If the provider doesn't support clearing. / 如果 provider 不支持清空。
        
        Notes: / 说明：
            This is primarily for testing and development. Use with caution in production. / 这主要用于测试和开发。生产环境中请谨慎使用。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear() method. "
            "This operation is optional and primarily for testing."
        )
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve records by their IDs. / 按 ID 检索记录。
        
        This method is used by SparseRetriever to fetch text and metadata / SparseRetriever 使用此方法为 BM25 搜索匹配到的块
        for chunks that were matched by BM25 search (which only returns IDs and scores). / 获取文本和元数据（BM25 只返回 ID 和分数）。
        
        Args: / 参数：
            ids: List of record IDs to retrieve. / 要检索的记录 ID 列表。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Returns: / 返回：
            List of records in the same order as input ids. / 与输入 ids 顺序相同的记录列表。
            Each record is a dict with: / 每条记录是包含以下字段的字典：
                - 'id': Record identifier / 'id'：记录标识符
                - 'text': The stored text content / 'text'：存储的文本内容
                - 'metadata': Associated metadata / 'metadata'：关联元数据
            If an ID is not found, an empty dict is returned for that position. / 如果某个 ID 未找到，则该位置返回空字典。
        
        Raises: / 异常：
            ValueError: If ids list is empty. / 如果 ids 列表为空。
            RuntimeError: If the retrieval operation fails. / 如果检索操作失败。
            NotImplementedError: If the provider doesn't support this operation. / 如果 provider 不支持此操作。
        
        Example: / 示例：
            >>> ids = ["chunk_001", "chunk_002", "chunk_003"]
            >>> records = vector_store.get_by_ids(ids)
            >>> for record in records:
            ...     print(f"ID: {record['id']}, Text: {record['text'][:50]}...")
        
        Notes: / 说明：
            This operation is essential for hybrid search where BM25 returns / 此操作对混合搜索至关重要，因为 BM25 返回的
            chunk IDs that need to be enriched with text and metadata from / chunk ID 需要从向量存储中补充
            the vector store. / 文本和元数据。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_by_ids() method. "
            "This operation is required for SparseRetriever support."
        )
