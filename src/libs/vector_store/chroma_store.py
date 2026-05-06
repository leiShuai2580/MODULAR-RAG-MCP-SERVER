"""ChromaDB VectorStore implementation. / ChromaDB VectorStore 实现。

This module provides a concrete implementation of BaseVectorStore using ChromaDB, / 此模块提供使用 ChromaDB 的 BaseVectorStore 具体实现，
a lightweight, open-source embedding database designed for local-first deployment. / ChromaDB 是为本地优先部署设计的轻量开源嵌入数据库。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """ChromaDB implementation of VectorStore. / VectorStore 的 ChromaDB 实现。
    
    This class provides local-first, persistent vector storage using ChromaDB. / 此类使用 ChromaDB 提供本地优先的持久化向量存储。
    It supports upsert, query, and metadata filtering operations. / 它支持 upsert、query 和元数据过滤操作。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Implements BaseVectorStore interface, swappable with other providers. / 可插拔：实现 BaseVectorStore 接口，可与其他 provider 替换。
    - Config-Driven: All settings (persist_directory, collection_name) from settings.yaml. / 配置驱动：所有设置（persist_directory、collection_name）来自 settings.yaml。
    - Idempotent: upsert operations with same ID overwrite existing records. / 幂等：相同 ID 的 upsert 操作会覆盖现有记录。
    - Observable: Accepts optional TraceContext (reserved for Stage F). / 可观测：接收可选 TraceContext（为 Stage F 预留）。
    - Fail-Fast: Validates dependencies and configuration on initialization. / 快速失败：初始化时校验依赖和配置。
    
    Attributes: / 属性：
        client: ChromaDB client instance. / ChromaDB 客户端实例。
        collection: ChromaDB collection for storing vectors. / 用于存储向量的 ChromaDB collection。
        collection_name: Name of the collection. / collection 名称。
        persist_directory: Directory path for persistent storage. / 持久化存储目录路径。
    
    Example: / 示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> store = ChromaStore(settings=settings)
        >>> records = [
        ...     {
        ...         'id': 'doc1_chunk0',
        ...         'vector': [0.1, 0.2, 0.3],
        ...         'metadata': {'source': 'doc1.pdf'}
        ...     }
        ... ]
        >>> store.upsert(records)
        >>> results = store.query([0.1, 0.2, 0.3], top_k=5)
    """
    
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialize ChromaStore with configuration. / 使用配置初始化 ChromaStore。
        
        Args: / 参数：
            settings: Application settings containing vector_store configuration. / 包含 vector_store 配置的应用设置。
            **kwargs: Optional overrides for collection_name or persist_directory. / collection_name 或 persist_directory 的可选覆盖值。
        
        Raises: / 异常：
            ImportError: If chromadb package is not installed. / 如果未安装 chromadb 包。
            ValueError: If required configuration is missing. / 如果缺少必需配置。
            RuntimeError: If ChromaDB client initialization fails. / 如果 ChromaDB 客户端初始化失败。
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for ChromaStore. "
                "Install it with: pip install chromadb"
            )
        
        # Extract configuration / 提取配置
        try:
            vector_store_config = settings.vector_store
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Please ensure 'vector_store' section exists in settings.yaml"
            ) from e
        
        # Collection name (allow override) / Collection 名称（允许覆盖）
        self.collection_name = kwargs.get(
            'collection_name',
            getattr(vector_store_config, 'collection_name', 'knowledge_hub')
        )
        
        # Persist directory (allow override) / 持久化目录（允许覆盖）
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
        self.persist_directory = resolve_path(persist_dir_str)
        
        # Ensure persist directory exists / 确保持久化目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initializing ChromaStore: collection='{self.collection_name}', "
            f"persist_directory='{self.persist_directory}'"
        )
        
        # Initialize ChromaDB client with persistent storage / 使用持久化存储初始化 ChromaDB 客户端
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}"
            ) from e
        
        # Get or create collection / 获取或创建 collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity / 使用余弦相似度
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{self.collection_name}': {e}"
            ) from e
        
        logger.info(
            f"ChromaStore initialized successfully. "
            f"Collection count: {self.collection.count()}"
        )
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update records in ChromaDB. / 在 ChromaDB 中插入或更新记录。
        
        Args: / 参数：
            records: List of records to upsert. Each record must have: / 要 upsert 的记录列表。每条记录必须包含：
                - 'id': Unique identifier (str) / 'id'：唯一标识符（str）
                - 'vector': Embedding vector (List[float]) / 'vector'：嵌入向量（List[float]）
                - 'metadata': Optional metadata dict / 'metadata'：可选元数据字典
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters (unused for Chroma). / provider 特有参数（Chroma 中未使用）。
        
        Raises: / 异常：
            ValueError: If records list is empty or contains invalid entries. / 如果 records 列表为空或包含无效条目。
            RuntimeError: If the upsert operation fails. / 如果 upsert 操作失败。
        """
        # Validate records / 校验记录
        self.validate_records(records)
        
        # Prepare data for ChromaDB / 准备 ChromaDB 数据
        ids = []
        embeddings = []
        metadatas = []
        documents = []  # ChromaDB requires documents field / ChromaDB 要求 documents 字段
        
        for record in records:
            ids.append(str(record['id']))
            embeddings.append(record['vector'])
            
            # Metadata: extract or default to empty dict / Metadata：提取或默认为空字典
            metadata = record.get('metadata', {})
            # Ensure all metadata values are JSON-serializable / 确保所有 metadata 值都可 JSON 序列化
            # ChromaDB requires string, int, float, or bool values / ChromaDB 要求值为 string、int、float 或 bool
            sanitized_metadata = self._sanitize_metadata(metadata)
            
            # ChromaDB requires non-empty metadata dict / ChromaDB 要求非空 metadata 字典
            if not sanitized_metadata:
                sanitized_metadata = {'_placeholder': 'true'}
            
            metadatas.append(sanitized_metadata)
            
            # Document: use metadata.text if available, otherwise use id / Document：如果可用则使用 metadata.text，否则使用 id
            document = metadata.get('text', record['id'])
            documents.append(str(document))
        
        # Perform upsert (ChromaDB's add() is idempotent with same IDs) / 执行 upsert（ChromaDB 对相同 ID 的 add() 是幂等的）
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug(f"Successfully upserted {len(records)} records to ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records to ChromaDB: {e}"
            ) from e
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for similar vectors. / 在 ChromaDB 中查询相似向量。
        
        Args: / 参数：
            vector: Query vector (embedding) to search for. / 用于搜索的查询向量（嵌入）。
            top_k: Maximum number of results to return. / 返回结果最大数量。
            filters: Optional metadata filters (e.g., {'source': 'doc1.pdf'}). / 可选元数据过滤器（例如 {'source': 'doc1.pdf'}）。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters (unused for Chroma). / provider 特有参数（Chroma 中未使用）。
        
        Returns: / 返回：
            List of matching records, sorted by similarity (descending). / 匹配记录列表，按相似度降序排序。
            Each record contains: / 每条记录包含：
                - 'id': Record identifier / 'id'：记录标识符
                - 'score': Similarity score (1.0 = identical, 0.0 = orthogonal) / 'score'：相似度分数（1.0 = 完全相同，0.0 = 正交）
                - 'metadata': Associated metadata / 'metadata'：关联元数据
        
        Raises: / 异常：
            ValueError: If vector is empty or top_k is invalid. / 如果 vector 为空或 top_k 无效。
            RuntimeError: If the query operation fails. / 如果查询操作失败。
        """
        # Validate query parameters / 校验查询参数
        self.validate_query_vector(vector, top_k)
        
        # Build ChromaDB where clause from filters / 根据 filters 构建 ChromaDB where 子句
        where_clause = self._build_where_clause(filters) if filters else None
        
        # Perform query / 执行查询
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query ChromaDB with top_k={top_k}: {e}"
            ) from e
        
        # Transform results to standard format / 将结果转换为标准格式
        # ChromaDB returns nested lists: [[id1, id2, ...]] / ChromaDB 返回嵌套列表：[[id1, id2, ...]]
        output = []
        
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(ids)
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(ids)
            documents = results['documents'][0] if 'documents' in results else [''] * len(ids)
            
            for i, record_id in enumerate(ids):
                # Convert distance to similarity score / 将距离转换为相似度分数
                # ChromaDB returns cosine distance (0=identical, 2=opposite) / ChromaDB 返回余弦距离（0=相同，2=相反）
                # Convert to similarity: score = 1 - (distance / 2) / 转为相似度：score = 1 - (distance / 2)
                distance = distances[i]
                score = 1.0 - (distance / 2.0)
                
                output.append({
                    'id': record_id,
                    'score': max(0.0, score),  # Clamp to [0, 1] / 限制到 [0, 1]
                    'text': documents[i] if documents[i] else '',  # Include text from documents / 包含 documents 中的文本
                    'metadata': metadatas[i] if metadatas[i] else {}
                })
        
        logger.debug(f"Query returned {len(output)} results")
        return output
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Delete records from ChromaDB by IDs. / 按 ID 从 ChromaDB 删除记录。
        
        Args: / 参数：
            ids: List of record IDs to delete. / 要删除的记录 ID 列表。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Raises: / 异常：
            ValueError: If ids list is empty. / 如果 ids 列表为空。
            RuntimeError: If the delete operation fails. / 如果删除操作失败。
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        try:
            self.collection.delete(ids=[str(id_) for id_ in ids])
            logger.debug(f"Successfully deleted {len(ids)} records from ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {len(ids)} records from ChromaDB: {e}"
            ) from e
    
    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Clear all records from the ChromaDB collection. / 清空 ChromaDB collection 中的所有记录。
        
        Args: / 参数：
            collection_name: Optional collection name to clear. If None, clears current collection. / 要清空的可选 collection 名称。为 None 时清空当前 collection。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Provider-specific parameters. / provider 特有参数。
        
        Raises: / 异常：
            RuntimeError: If the clear operation fails. / 如果清空操作失败。
        """
        try:
            target_collection = collection_name or self.collection_name
            
            # Delete and recreate collection (most efficient way to clear in Chroma) / 删除并重建 collection（Chroma 中最有效的清空方式）
            self.client.delete_collection(name=target_collection)
            self.collection = self.client.get_or_create_collection(
                name=target_collection,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully cleared collection '{target_collection}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to clear collection '{collection_name or self.collection_name}': {e}"
            ) from e

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        """Delete records matching a metadata filter. / 删除匹配元数据过滤器的记录。

        Args: / 参数：
            filter_dict: Metadata key/value pairs to match / 要匹配的元数据键/值对
                (e.g. ``{"source_hash": "abc123"}``). / （例如 ``{"source_hash": "abc123"}``）。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。

        Returns: / 返回：
            Number of records deleted. / 已删除记录数量。

        Raises: / 异常：
            ValueError: If *filter_dict* is empty. / 如果 *filter_dict* 为空。
            RuntimeError: If the operation fails. / 如果操作失败。
        """
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_where_clause(filter_dict)
            # Query matching IDs first / 先查询匹配的 ID
            results = self.collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(f"delete_by_metadata: no records matched {filter_dict}")
                return 0

            self.collection.delete(ids=matching_ids)
            logger.info(
                f"delete_by_metadata: deleted {len(matching_ids)} records "
                f"matching {filter_dict}"
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure ChromaDB compatibility. / 清理元数据以确保与 ChromaDB 兼容。
        
        ChromaDB requires metadata values to be str, int, float, or bool. / ChromaDB 要求元数据值为 str、int、float 或 bool。
        This method converts or filters out incompatible types. / 此方法会转换或过滤不兼容类型。
        
        Args: / 参数：
            metadata: Raw metadata dict. / 原始元数据字典。
        
        Returns: / 返回：
            Sanitized metadata dict. / 清理后的元数据字典。
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                # Skip None values / 跳过 None 值
                continue
            elif isinstance(value, (list, tuple)):
                # Convert to comma-separated string / 转换为逗号分隔字符串
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                # Convert to string as fallback / 回退为转换成字符串
                sanitized[key] = str(value)
        
        return sanitized
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters. / 根据 filters 构建 ChromaDB where 子句。
        
        Converts standard filter dict to ChromaDB's query format. / 将标准过滤字典转换为 ChromaDB 查询格式。
        
        Args: / 参数：
            filters: Standard filter dict (e.g., {'source': 'doc1.pdf'}). / 标准过滤字典（例如 {'source': 'doc1.pdf'}）。
        
        Returns: / 返回：
            ChromaDB where clause dict. / ChromaDB where 子句字典。
        
        Note: / 说明：
            ChromaDB supports operators like $eq, $ne, $gt, $lt, $in, etc. / ChromaDB 支持 $eq、$ne、$gt、$lt、$in 等操作符。
            For simplicity, we currently support only exact equality matches. / 为简单起见，目前只支持精确相等匹配。
            Future enhancement: support complex filters. / 未来增强：支持复杂过滤器。
        """
        # Simple implementation: exact equality matches only / 简单实现：仅精确相等匹配
        # For complex filters (e.g., {'score': {'$gt': 0.5}}), extend this method / 如需复杂过滤器（例如 {'score': {'$gt': 0.5}}），可扩展此方法
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in ChromaDB operator format (e.g., {'$eq': 'value'}) / 已是 ChromaDB 操作符格式（例如 {'$eq': 'value'}）
                where[key] = value
            else:
                # Simple equality / 简单相等
                where[key] = value
        
        return where
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection. / 获取当前 collection 的统计信息。
        
        Returns: / 返回：
            Dict containing collection statistics: / 包含 collection 统计信息的字典：
                - count: Number of records in collection / count：collection 中的记录数量
                - name: Collection name / name：collection 名称
                - metadata: Collection metadata / metadata：collection 元数据
        """
        return {
            'count': self.collection.count(),
            'name': self.collection_name,
            'metadata': self.collection.metadata
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve records by their IDs from ChromaDB. / 按 ID 从 ChromaDB 检索记录。
        
        This method is used by SparseRetriever to fetch text and metadata / SparseRetriever 使用此方法为 BM25 搜索匹配到的块
        for chunks that were matched by BM25 search. / 获取文本和元数据。
        
        Args: / 参数：
            ids: List of record IDs to retrieve. / 要检索的记录 ID 列表。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Provider-specific parameters (unused for Chroma). / provider 特有参数（Chroma 中未使用）。
        
        Returns: / 返回：
            List of records in the same order as input ids. / 与输入 ids 顺序相同的记录列表。
            Each record contains: / 每条记录包含：
                - 'id': Record identifier / 'id'：记录标识符
                - 'text': The stored text content / 'text'：存储的文本内容
                - 'metadata': Associated metadata / 'metadata'：关联元数据
            If an ID is not found, an empty dict is returned for that position. / 如果某个 ID 未找到，则该位置返回空字典。
        
        Raises: / 异常：
            ValueError: If ids list is empty. / 如果 ids 列表为空。
            RuntimeError: If the retrieval operation fails. / 如果检索操作失败。
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        # Ensure all IDs are strings / 确保所有 ID 都是字符串
        str_ids = [str(id_) for id_ in ids]
        
        try:
            # ChromaDB's get method retrieves records by IDs / ChromaDB 的 get 方法按 ID 检索记录
            results = self.collection.get(
                ids=str_ids,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by IDs from ChromaDB: {e}"
            ) from e
        
        # Build a mapping from ID to result for O(1) lookup / 构建 ID 到结果的映射，用于 O(1) 查找
        id_to_result: Dict[str, Dict[str, Any]] = {}
        
        if results and results.get('ids'):
            result_ids = results['ids']
            documents = results.get('documents', [None] * len(result_ids))
            metadatas = results.get('metadatas', [{}] * len(result_ids))
            
            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    'id': record_id,
                    'text': documents[i] if documents and documents[i] else '',
                    'metadata': metadatas[i] if metadatas and metadatas[i] else {}
                }
        
        # Return results in the same order as input ids / 按输入 ids 的相同顺序返回结果
        output = []
        for id_ in str_ids:
            if id_ in id_to_result:
                output.append(id_to_result[id_])
            else:
                # ID not found, return empty dict / ID 未找到，返回空字典
                output.append({})
        
        logger.debug(f"Retrieved {len([r for r in output if r])} of {len(ids)} records by IDs")
        return output
