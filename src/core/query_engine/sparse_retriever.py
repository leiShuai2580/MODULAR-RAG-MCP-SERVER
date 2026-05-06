"""Sparse Retriever for keyword-based search using BM25. / 使用 BM25 进行关键词搜索的稀疏检索器。

This module implements the SparseRetriever component that performs keyword-based / 本模块实现 SparseRetriever 组件，用于基于关键词
search using BM25 inverted indexes. It forms the Sparse route in the Hybrid / 使用 BM25 倒排索引进行搜索。它构成混合
Search Engine, complementing the DenseRetriever's semantic search. / 搜索引擎中的稀疏检索路径，用于补充 DenseRetriever 的语义搜索。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Sparse retriever using BM25 keyword-based search. / 使用 BM25 关键词搜索的稀疏检索器。
    
    This class performs keyword-based retrieval by: / 该类通过以下步骤执行基于关键词的检索：
    1. Querying the BM25 index with keywords to get matching chunk IDs and scores / 1. 使用关键词查询 BM25 索引，获取匹配的分块 ID 和分数
    2. Fetching text and metadata from the vector store using get_by_ids() / 2. 使用 get_by_ids() 从向量存储中获取文本和元数据
    3. Returning normalized RetrievalResult objects / 3. 返回标准化的 RetrievalResult 对象
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Accepts bm25_indexer and vector_store via dependency injection. / 可插拔：通过依赖注入接收 bm25_indexer 和 vector_store。
    - Config-Driven: Default top_k and collection read from settings. / 配置驱动：默认 top_k 和 collection 从 settings 读取。
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选的 TraceContext 用于可观测性集成。
    - Fail-Fast: Validates inputs early with clear error messages. / 快速失败：尽早校验输入并给出清晰错误信息。
    - Type-Safe: Returns standardized RetrievalResult objects (same as DenseRetriever). / 类型安全：返回标准化的 RetrievalResult 对象（与 DenseRetriever 相同）。
    
    Attributes: / 属性：
        bm25_indexer: The BM25 indexer for keyword search. / bm25_indexer：用于关键词搜索的 BM25 索引器。
        vector_store: The vector store for fetching text and metadata. / vector_store：用于获取文本和元数据的向量存储。
        default_top_k: Default number of results to return. / default_top_k：默认返回结果数量。
        default_collection: Default BM25 index collection to query. / default_collection：默认查询的 BM25 索引集合。
    
    Example: / 示例：
        >>> from src.ingestion.storage.bm25_indexer import BM25Indexer
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>> 
        >>> settings = Settings.load('config/settings.yaml')
        >>> bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> bm25_indexer.load("default")
        >>> vector_store = VectorStoreFactory.create(settings)
        >>> 
        >>> retriever = SparseRetriever(
        ...     settings=settings,
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve(["RAG", "retrieval"], top_k=5)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
        default_collection: str = "default",
    ) -> None:
        """Initialize SparseRetriever with dependencies. / 使用依赖项初始化 SparseRetriever。
        
        Args: / 参数：
            settings: Application settings. Used to extract default_top_k if not provided. / settings：应用配置。如果未提供 default_top_k，则用于提取默认值。
            bm25_indexer: BM25 indexer for keyword search. / bm25_indexer：用于关键词搜索的 BM25 索引器。
                          Required for actual retrieval operations. / 实际执行检索操作时必需。
            vector_store: Vector store for fetching text and metadata. / vector_store：用于获取文本和元数据的向量存储。
                          Required for actual retrieval operations. / 实际执行检索操作时必需。
            default_top_k: Default number of results to return (default: 10). / default_top_k：默认返回结果数量（默认值：10）。
                           Can be overridden from settings.retrieval.sparse_top_k. / 可被 settings.retrieval.sparse_top_k 覆盖。
            default_collection: Default BM25 index collection name (default: "default"). / default_collection：默认 BM25 索引集合名称（默认值："default"）。
        
        Note: / 说明：
            Dependencies can be injected for testing (with mocks) or for / 可以注入依赖项用于测试（使用 mock），也可以用于
            production use (with real implementations from factories). / 生产环境（使用工厂创建的真实实现）。
        """
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.default_collection = default_collection
        
        # Extract default_top_k from settings if available / 如果配置可用，则从配置中提取 default_top_k
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'sparse_top_k', default_top_k
                )
        
        logger.info(
            f"SparseRetriever initialized with default_top_k={self.default_top_k}, "
            f"default_collection='{self.default_collection}'"
        )
    
    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Retrieve chunks matching the given keywords using BM25. / 使用 BM25 检索匹配给定关键词的分块。
        
        Args: / 参数：
            keywords: List of keywords to search for (typically from QueryProcessor). / keywords：要搜索的关键词列表（通常来自 QueryProcessor）。
            top_k: Maximum number of results to return. If None, uses default_top_k. / top_k：最大返回结果数量。如果为 None，则使用 default_top_k。
            collection: BM25 index collection to query. If None, uses default_collection. / collection：要查询的 BM25 索引集合。如果为 None，则使用 default_collection。
            trace: Optional TraceContext for observability (reserved for Stage F). / trace：用于可观测性的可选 TraceContext（为阶段 F 预留）。
        
        Returns: / 返回：
            List of RetrievalResult objects, sorted by BM25 score (descending). / RetrievalResult 对象列表，按 BM25 分数降序排序。
            Each result contains chunk_id, score, text, and metadata. / 每个结果包含 chunk_id、score、text 和 metadata。
        
        Raises: / 异常：
            ValueError: If keywords list is empty. / ValueError：当关键词列表为空时抛出。
            RuntimeError: If bm25_indexer or vector_store is not configured, / RuntimeError：当 bm25_indexer 或 vector_store 未配置，
                          or if the retrieval operation fails. / 或检索操作失败时抛出。
        
        Example: / 示例：
            >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"])
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # Validate inputs / 校验输入
        self._validate_keywords(keywords)
        self._validate_dependencies()
        
        # Use defaults if not specified / 如果未指定，则使用默认值
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_collection = collection if collection is not None else self.default_collection
        
        logger.debug(
            f"Retrieving for keywords={keywords[:5]}{'...' if len(keywords) > 5 else ''}, "
            f"top_k={effective_top_k}, collection='{effective_collection}'"
        )
        
        # Step 1: Ensure index is loaded / 第 1 步：确保索引已加载
        if not self._ensure_index_loaded(effective_collection):
            logger.warning(
                f"BM25 index for collection '{effective_collection}' not available. "
                "Returning empty results."
            )
            return []
        
        # Step 2: Query BM25 index / 第 2 步：查询 BM25 索引
        try:
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                top_k=effective_top_k,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query BM25 index: {e}. "
                "Check index availability and query terms."
            ) from e
        
        # Early return if no matches / 如果没有匹配结果则提前返回
        if not bm25_results:
            logger.debug("BM25 query returned no results")
            return []
        
        # Step 3: Fetch text and metadata from vector store / 第 3 步：从向量存储中获取文本和元数据
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        try:
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch records from vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e
        
        # Step 4: Merge BM25 scores with text/metadata / 第 4 步：将 BM25 分数与文本和元数据合并
        results = self._merge_results(bm25_results, records)
        
        logger.debug(f"Retrieved {len(results)} results for keywords")
        return results
    
    def _validate_keywords(self, keywords: List[str]) -> None:
        """Validate the keywords list. / 校验关键词列表。
        
        Args: / 参数：
            keywords: Keywords list to validate. / keywords：待校验的关键词列表。
        
        Raises: / 异常：
            ValueError: If keywords is empty or not a list. / ValueError：当 keywords 为空或不是列表时抛出。
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        # Filter out empty strings but allow the call to proceed / 过滤空字符串，但允许调用继续执行
        # (empty strings will simply not match anything) / （空字符串只是什么都匹配不到）
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured. / 校验必需依赖是否已配置。
        
        Raises: / 异常：
            RuntimeError: If bm25_indexer or vector_store is None. / RuntimeError：当 bm25_indexer 或 vector_store 为 None 时抛出。
        """
        if self.bm25_indexer is None:
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
    
    def _ensure_index_loaded(self, collection: str) -> bool:
        """Ensure the BM25 index is loaded for the given collection. / 确保给定集合的 BM25 索引已加载。
        
        Always reloads from disk because the index may have been updated / 总是从磁盘重新加载，因为索引可能已被
        by another process (e.g., dashboard ingestion).  The load is / 其他进程更新（例如 dashboard ingestion）。与整体查询相比，
        fast (a single JSON file read) compared to the overall query. / 加载很快（只读取单个 JSON 文件）。
        
        Args: / 参数：
            collection: The collection name to load. / collection：要加载的集合名称。
        
        Returns: / 返回：
            True if index is loaded and ready, False otherwise. / 如果索引已加载并就绪则返回 True，否则返回 False。
        """
        try:
            loaded = self.bm25_indexer.load(collection=collection)
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for collection '{collection}': {e}")
            return False
    
    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Merge BM25 scores with text and metadata from vector store. / 将 BM25 分数与向量存储中的文本和元数据合并。
        
        Args: / 参数：
            bm25_results: Results from BM25 query, each with 'chunk_id' and 'score'. / bm25_results：BM25 查询结果，每项包含 'chunk_id' 和 'score'。
            records: Records from vector store, each with 'id', 'text', 'metadata'. / records：向量存储记录，每项包含 'id'、'text'、'metadata'。
        
        Returns: / 返回：
            List of RetrievalResult objects with complete information. / 包含完整信息的 RetrievalResult 对象列表。
        """
        results = []
        
        for bm25_result, record in zip(bm25_results, records):
            chunk_id = bm25_result["chunk_id"]
            score = bm25_result["score"]
            
            # Handle case where record was not found / 处理未找到记录的情况
            if not record:
                logger.warning(
                    f"No record found in vector store for chunk_id='{chunk_id}'. "
                    "Skipping this result."
                )
                continue
            
            # Validate record has expected fields / 校验记录是否包含预期字段
            text = record.get('text', '')
            metadata = record.get('metadata', {})
            
            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=str(text),
                    metadata=metadata,
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to create RetrievalResult for chunk_id='{chunk_id}': {e}. "
                    "Skipping this result."
                )
                continue
        
        return results


def create_sparse_retriever(
    settings: Settings,
    bm25_indexer: Optional[BM25Indexer] = None,
    vector_store: Optional[BaseVectorStore] = None,
    index_dir: str = "data/db/bm25",
) -> SparseRetriever:
    """Factory function to create a SparseRetriever with optional dependency injection. / 用于创建 SparseRetriever 的工厂函数，支持可选依赖注入。
    
    This function simplifies SparseRetriever creation by automatically creating / 该函数通过在未提供依赖时自动从工厂创建依赖，
    dependencies from factories if not provided. / 简化 SparseRetriever 的创建过程。
    
    Args: / 参数：
        settings: Application settings. / settings：应用配置。
        bm25_indexer: Optional pre-configured BM25 indexer. / bm25_indexer：可选的预配置 BM25 索引器。
                      If None, created with default index_dir. / 如果为 None，则使用默认 index_dir 创建。
        vector_store: Optional pre-configured vector store. / vector_store：可选的预配置向量存储。
                      If None, created from VectorStoreFactory. / 如果为 None，则从 VectorStoreFactory 创建。
        index_dir: Directory for BM25 index files (default: "data/db/bm25"). / index_dir：BM25 索引文件目录（默认值："data/db/bm25"）。
    
    Returns: / 返回：
        Configured SparseRetriever instance. / 配置完成的 SparseRetriever 实例。
    
    Example: / 示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_sparse_retriever(settings)
    """
    # Lazy import to avoid circular dependencies / 延迟导入以避免循环依赖
    if bm25_indexer is None:
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        bm25_indexer = BM25Indexer(index_dir=index_dir)
    
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    
    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
