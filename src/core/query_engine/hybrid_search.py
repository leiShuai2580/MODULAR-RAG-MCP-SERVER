"""Hybrid Search Engine orchestrating Dense + Sparse retrieval with RRF Fusion. / 编排稠密检索和稀疏检索并使用 RRF 融合的混合搜索引擎。

This module implements the HybridSearch class that combines: / 本模块实现 HybridSearch 类，用于组合：
1. QueryProcessor: Preprocess queries and extract keywords/filters / 1. QueryProcessor：预处理查询并提取关键词和过滤条件
2. DenseRetriever: Semantic search using embeddings / 2. DenseRetriever：使用嵌入进行语义搜索
3. SparseRetriever: Keyword search using BM25 / 3. SparseRetriever：使用 BM25 进行关键词搜索
4. RRFFusion: Combine results using Reciprocal Rank Fusion / 4. RRFFusion：使用倒数排名融合组合结果

Design Principles: / 设计原则：
- Graceful Degradation: If one retrieval path fails, fall back to the other / 优雅降级：一个检索路径失败时回退到另一个路径
- Pluggable: All components injected via constructor for testability / 可插拔：所有组件通过构造函数注入以便测试
- Observable: TraceContext integration for debugging and monitoring / 可观测：集成 TraceContext 便于调试和监控
- Config-Driven: Top-k and other parameters read from settings / 配置驱动：top-k 和其他参数从 settings 读取
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.types import ProcessedQuery, RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


def _snapshot_results(
    results: Optional[List[RetrievalResult]],
) -> List[Dict[str, Any]]:
    """Create a serialisable snapshot of retrieval results for trace storage. / 创建可序列化的检索结果快照，用于追踪存储。

    Args: / 参数：
        results: List of RetrievalResult objects. / results：RetrievalResult 对象列表。

    Returns: / 返回：
        List of dicts with chunk_id, score, full text, source. / 包含 chunk_id、score、完整文本和 source 的字典列表。
    """
    if not results:
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "text": r.text or "",
            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
        }
        for r in results
    ]


@dataclass
class HybridSearchConfig:
    """Configuration for HybridSearch. / HybridSearch 的配置。
    
    Attributes: / 属性：
        dense_top_k: Number of results from dense retrieval / dense_top_k：稠密检索返回结果数量
        sparse_top_k: Number of results from sparse retrieval / sparse_top_k：稀疏检索返回结果数量
        fusion_top_k: Final number of results after fusion / fusion_top_k：融合后的最终结果数量
        enable_dense: Whether to use dense retrieval / enable_dense：是否使用稠密检索
        enable_sparse: Whether to use sparse retrieval / enable_sparse：是否使用稀疏检索
        parallel_retrieval: Whether to run retrievals in parallel / parallel_retrieval：是否并行运行检索
        metadata_filter_post: Apply metadata filters after fusion (fallback) / metadata_filter_post：融合后应用元数据过滤（回退机制）
    """
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True


@dataclass
class HybridSearchResult:
    """Result of a hybrid search operation. / 混合搜索操作的结果。
    
    Attributes: / 属性：
        results: Final ranked list of RetrievalResults / results：最终排序后的 RetrievalResult 列表
        dense_results: Results from dense retrieval (for debugging) / dense_results：稠密检索结果（用于调试）
        sparse_results: Results from sparse retrieval (for debugging) / sparse_results：稀疏检索结果（用于调试）
        dense_error: Error message if dense retrieval failed / dense_error：稠密检索失败时的错误信息
        sparse_error: Error message if sparse retrieval failed / sparse_error：稀疏检索失败时的错误信息
        used_fallback: Whether fallback mode was used / used_fallback：是否使用了回退模式
        processed_query: The processed query (for debugging) / processed_query：处理后的查询（用于调试）
    """
    results: List[RetrievalResult] = field(default_factory=list)
    dense_results: Optional[List[RetrievalResult]] = None
    sparse_results: Optional[List[RetrievalResult]] = None
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    used_fallback: bool = False
    processed_query: Optional[ProcessedQuery] = None


class HybridSearch:
    """Hybrid Search Engine combining Dense and Sparse retrieval. / 组合稠密和稀疏检索的混合搜索引擎。
    
    This class orchestrates the complete hybrid search flow: / 该类编排完整的混合搜索流程：
    1. Query Processing: Extract keywords and filters from raw query / 1. 查询处理：从原始查询中提取关键词和过滤条件
    2. Parallel Retrieval: Run Dense and Sparse retrievers concurrently / 2. 并行检索：并发运行稠密和稀疏检索器
    3. Fusion: Combine results using RRF algorithm / 3. 融合：使用 RRF 算法组合结果
    4. Post-Filtering: Apply metadata filters if specified / 4. 后置过滤：如有指定则应用元数据过滤
    
    Design Principles Applied: / 应用的设计原则：
    - Graceful Degradation: If one path fails, use results from the other / 优雅降级：一个路径失败时使用另一个路径的结果
    - Pluggable: All components via dependency injection / 可插拔：所有组件通过依赖注入提供
    - Observable: TraceContext support for debugging / 可观测：支持 TraceContext 进行调试
    - Config-Driven: All parameters from settings / 配置驱动：所有参数来自 settings
    
    Example: / 示例：
        >>> # Initialize components / 初始化组件
        >>> query_processor = QueryProcessor()
        >>> dense_retriever = DenseRetriever(settings, embedding_client, vector_store)
        >>> sparse_retriever = SparseRetriever(settings, bm25_indexer, vector_store)
        >>> fusion = RRFFusion(k=60)
        >>> 
        >>> # Create HybridSearch / 创建 HybridSearch
        >>> hybrid = HybridSearch(
        ...     settings=settings,
        ...     query_processor=query_processor,
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ...     fusion=fusion
        ... )
        >>> 
        >>> # Search / 搜索
        >>> results = hybrid.search("如何配置 Azure OpenAI？", top_k=10)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize HybridSearch with components. / 使用组件初始化 HybridSearch。
        
        Args: / 参数：
            settings: Application settings for extracting configuration. / settings：用于提取配置的应用配置。
            query_processor: QueryProcessor for preprocessing queries. / query_processor：用于预处理查询的 QueryProcessor。
            dense_retriever: DenseRetriever for semantic search. / dense_retriever：用于语义搜索的 DenseRetriever。
            sparse_retriever: SparseRetriever for keyword search. / sparse_retriever：用于关键词搜索的 SparseRetriever。
            fusion: RRFFusion for combining results. / fusion：用于组合结果的 RRFFusion。
            config: Optional HybridSearchConfig. If not provided, extracted from settings. / config：可选 HybridSearchConfig。未提供时从 settings 提取。
        
        Note: / 说明：
            At least one of dense_retriever or sparse_retriever must be provided / 为了让搜索可用，至少必须提供 dense_retriever 或 sparse_retriever
            for search to function. The search will gracefully degrade if one / 其中之一。如果某个检索器不可用或失败，
            is unavailable or fails. / 搜索会优雅降级。
        """
        self.query_processor = query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        
        # Extract config from settings or use provided/default / 从 settings 提取配置，或使用已提供/默认配置
        self.config = config or self._extract_config(settings)
        
        logger.info(
            f"HybridSearch initialized: dense={self.dense_retriever is not None}, "
            f"sparse={self.sparse_retriever is not None}, "
            f"config={self.config}"
        )
    
    def _extract_config(self, settings: Optional[Settings]) -> HybridSearchConfig:
        """Extract HybridSearchConfig from Settings. / 从 Settings 中提取 HybridSearchConfig。
        
        Args: / 参数：
            settings: Application settings object. / settings：应用配置对象。
            
        Returns: / 返回：
            HybridSearchConfig with values from settings or defaults. / 包含 settings 取值或默认值的 HybridSearchConfig。
        """
        if settings is None:
            return HybridSearchConfig()
        
        retrieval_config = getattr(settings, 'retrieval', None)
        if retrieval_config is None:
            return HybridSearchConfig()
        
        return HybridSearchConfig(
            dense_top_k=getattr(retrieval_config, 'dense_top_k', 20),
            sparse_top_k=getattr(retrieval_config, 'sparse_top_k', 20),
            fusion_top_k=getattr(retrieval_config, 'fusion_top_k', 10),
            enable_dense=True,
            enable_sparse=True,
            parallel_retrieval=True,
            metadata_filter_post=True,
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        return_details: bool = False,
    ) -> List[RetrievalResult] | HybridSearchResult:
        """Perform hybrid search combining Dense and Sparse retrieval. / 执行组合稠密和稀疏检索的混合搜索。
        
        Args: / 参数：
            query: The search query string. / query：搜索查询字符串。
            top_k: Maximum number of results to return. If None, uses config.fusion_top_k. / top_k：最大返回结果数量。为 None 时使用 config.fusion_top_k。
            filters: Optional metadata filters (e.g., {"collection": "docs"}). / filters：可选元数据过滤条件（例如 {"collection": "docs"}）。
            trace: Optional TraceContext for observability. / trace：用于可观测性的可选 TraceContext。
            return_details: If True, return HybridSearchResult with debug info. / return_details：如果为 True，则返回包含调试信息的 HybridSearchResult。
        
        Returns: / 返回：
            If return_details=False: List of RetrievalResult sorted by relevance. / 如果 return_details=False：返回按相关性排序的 RetrievalResult 列表。
            If return_details=True: HybridSearchResult with full details. / 如果 return_details=True：返回包含完整详情的 HybridSearchResult。
        
        Raises: / 异常：
            ValueError: If query is empty or invalid. / ValueError：当 query 为空或无效时抛出。
            RuntimeError: If both retrievers fail or are unavailable. / RuntimeError：当两个检索器都失败或不可用时抛出。
        
        Example: / 示例：
            >>> results = hybrid.search("Azure configuration", top_k=5)
            >>> for r in results:
            ...     print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:50]}...")
        """
        # Validate query / 校验查询
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        
        effective_top_k = top_k if top_k is not None else self.config.fusion_top_k
        
        logger.debug(f"HybridSearch: query='{query[:50]}...', top_k={effective_top_k}")
        
        # Step 1: Process query / 第 1 步：处理查询
        _t0 = time.monotonic()
        processed_query = self._process_query(query)
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("query_processing", {
                "method": "query_processor",
                "original_query": query,
                "keywords": processed_query.keywords,
            }, elapsed_ms=_elapsed)
        
        # Merge explicit filters with query-extracted filters / 合并显式过滤条件和从查询中提取的过滤条件
        merged_filters = self._merge_filters(processed_query.filters, filters)
        
        # Step 2: Run retrievals / 第 2 步：运行检索
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            filters=merged_filters,
            trace=trace,
        )
        
        # Step 3: Handle fallback scenarios / 第 3 步：处理回退场景
        used_fallback = False
        if dense_error and sparse_error:
            # Both failed - raise error / 两者都失败，则抛出错误
            raise RuntimeError(
                f"Both retrieval paths failed. "
                f"Dense error: {dense_error}. Sparse error: {sparse_error}"
            )
        elif dense_error:
            # Dense failed, use sparse only / 稠密检索失败，仅使用稀疏检索
            logger.warning(f"Dense retrieval failed, using sparse only: {dense_error}")
            used_fallback = True
            fused_results = sparse_results or []
        elif sparse_error:
            # Sparse failed, use dense only / 稀疏检索失败，仅使用稠密检索
            logger.warning(f"Sparse retrieval failed, using dense only: {sparse_error}")
            used_fallback = True
            fused_results = dense_results or []
        elif not dense_results and not sparse_results:
            # Both succeeded but returned empty / 两者都成功但返回为空
            fused_results = []
        else:
            # Step 4: Fuse results / 第 4 步：融合结果
            fused_results = self._fuse_results(
                dense_results=dense_results or [],
                sparse_results=sparse_results or [],
                top_k=effective_top_k,
                trace=trace,
            )
        
        # Step 5: Apply post-fusion metadata filters (if any) / 第 5 步：应用融合后的元数据过滤（如果有）
        if merged_filters and self.config.metadata_filter_post:
            fused_results = self._apply_metadata_filters(fused_results, merged_filters)
        
        # Step 6: Limit to top_k / 第 6 步：限制为 top_k
        final_results = fused_results[:effective_top_k]
        
        logger.debug(f"HybridSearch: returning {len(final_results)} results")
        
        if return_details:
            return HybridSearchResult(
                results=final_results,
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_error=dense_error,
                sparse_error=sparse_error,
                used_fallback=used_fallback,
                processed_query=processed_query,
            )
        
        return final_results
    
    def _process_query(self, query: str) -> ProcessedQuery:
        """Process raw query using QueryProcessor. / 使用 QueryProcessor 处理原始查询。
        
        Args: / 参数：
            query: Raw query string. / query：原始查询字符串。
            
        Returns: / 返回：
            ProcessedQuery with keywords and filters. / 包含关键词和过滤条件的 ProcessedQuery。
        """
        if self.query_processor is None:
            # Fallback: create basic ProcessedQuery / 回退：创建基础 ProcessedQuery
            logger.warning("No QueryProcessor configured, using basic tokenization")
            keywords = query.split()
            return ProcessedQuery(
                original_query=query,
                keywords=keywords,
                filters={},
            )
        
        return self.query_processor.process(query)
    
    def _merge_filters(
        self,
        query_filters: Dict[str, Any],
        explicit_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge query-extracted filters with explicit filters. / 合并从查询中提取的过滤条件和显式过滤条件。
        
        Explicit filters take precedence over query-extracted filters. / 显式过滤条件优先于从查询中提取的过滤条件。
        
        Args: / 参数：
            query_filters: Filters extracted from query by QueryProcessor. / query_filters：QueryProcessor 从查询中提取的过滤条件。
            explicit_filters: Filters passed explicitly to search(). / explicit_filters：显式传递给 search() 的过滤条件。
            
        Returns: / 返回：
            Merged filter dictionary. / 合并后的过滤条件字典。
        """
        merged = query_filters.copy() if query_filters else {}
        if explicit_filters:
            merged.update(explicit_filters)
        return merged
    
    def _run_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals. / 运行稠密和稀疏检索。
        
        Runs in parallel if configured, otherwise sequentially. / 如果已配置则并行运行，否则顺序运行。
        
        Args: / 参数：
            processed_query: The processed query with keywords. / processed_query：包含关键词的已处理查询。
            filters: Merged filters to apply. / filters：要应用的合并过滤条件。
            trace: Optional TraceContext. / trace：可选 TraceContext。
            
        Returns: / 返回：
            Tuple of (dense_results, sparse_results, dense_error, sparse_error). / 四元组：（dense_results、sparse_results、dense_error、sparse_error）。
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None
        
        # Determine what to run / 确定要运行的检索路径
        run_dense = (
            self.config.enable_dense 
            and self.dense_retriever is not None
        )
        run_sparse = (
            self.config.enable_sparse 
            and self.sparse_retriever is not None
            and processed_query.keywords  # Need keywords for sparse / 稀疏检索需要关键词
        )
        
        if not run_dense and not run_sparse:
            # Nothing to run / 没有可运行的检索路径
            if self.dense_retriever is None and self.sparse_retriever is None:
                dense_error = "No retriever configured"
                sparse_error = "No retriever configured"
            return dense_results, sparse_results, dense_error, sparse_error
        
        if self.config.parallel_retrieval and run_dense and run_sparse:
            # Run in parallel / 并行运行
            dense_results, sparse_results, dense_error, sparse_error = (
                self._run_parallel_retrievals(processed_query, filters, trace)
            )
        else:
            # Run sequentially / 顺序运行
            if run_dense:
                dense_results, dense_error = self._run_dense_retrieval(
                    processed_query.original_query, filters, trace
                )
            
            if run_sparse:
                sparse_results, sparse_error = self._run_sparse_retrieval(
                    processed_query.keywords, filters, trace
                )
        
        return dense_results, sparse_results, dense_error, sparse_error
    
    def _run_parallel_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals in parallel using ThreadPoolExecutor. / 使用 ThreadPoolExecutor 并行运行稠密和稀疏检索。
        
        Args: / 参数：
            processed_query: The processed query. / processed_query：已处理查询。
            filters: Filters to apply. / filters：要应用的过滤条件。
            trace: Optional TraceContext. / trace：可选 TraceContext。
            
        Returns: / 返回：
            Tuple of (dense_results, sparse_results, dense_error, sparse_error). / 四元组：（dense_results、sparse_results、dense_error、sparse_error）。
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # Submit dense retrieval / 提交稠密检索
            futures['dense'] = executor.submit(
                self._run_dense_retrieval,
                processed_query.original_query,
                filters,
                trace,
            )
            
            # Submit sparse retrieval / 提交稀疏检索
            futures['sparse'] = executor.submit(
                self._run_sparse_retrieval,
                processed_query.keywords,
                filters,
                trace,
            )
            
            # Collect results / 收集结果
            for name, future in futures.items():
                try:
                    results, error = future.result(timeout=30)
                    if name == 'dense':
                        dense_results = results
                        dense_error = error
                    else:
                        sparse_results = results
                        sparse_error = error
                except Exception as e:
                    error_msg = f"{name} retrieval failed with exception: {e}"
                    logger.error(error_msg)
                    if name == 'dense':
                        dense_error = error_msg
                    else:
                        sparse_error = error_msg
        
        return dense_results, sparse_results, dense_error, sparse_error
    
    def _run_dense_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run dense retrieval with error handling. / 运行带错误处理的稠密检索。
        
        Args: / 参数：
            query: Original query string. / query：原始查询字符串。
            filters: Filters to apply. / filters：要应用的过滤条件。
            trace: Optional TraceContext. / trace：可选 TraceContext。
            
        Returns: / 返回：
            Tuple of (results, error). If successful, error is None. / 二元组：（results、error）。成功时 error 为 None。
        """
        if self.dense_retriever is None:
            return None, "Dense retriever not configured"
        
        try:
            _t0 = time.monotonic()
            results = self.dense_retriever.retrieve(
                query=query,
                top_k=self.config.dense_top_k,
                filters=filters,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "provider": getattr(self.dense_retriever, 'provider_name', 'unknown'),
                    "top_k": self.config.dense_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Dense retrieval error: {e}"
            logger.error(error_msg)
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "error": error_msg,
                    "result_count": 0,
                })
            return None, error_msg
    
    def _run_sparse_retrieval(
        self,
        keywords: List[str],
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run sparse retrieval with error handling. / 运行带错误处理的稀疏检索。
        
        Args: / 参数：
            keywords: List of keywords from QueryProcessor. / keywords：来自 QueryProcessor 的关键词列表。
            filters: Filters to apply. / filters：要应用的过滤条件。
            trace: Optional TraceContext. / trace：可选 TraceContext。
            
        Returns: / 返回：
            Tuple of (results, error). If successful, error is None. / 二元组：（results、error）。成功时 error 为 None。
        """
        if self.sparse_retriever is None:
            return None, "Sparse retriever not configured"
        
        if not keywords:
            return [], None  # No keywords, return empty (not an error) / 没有关键词，返回空列表（不是错误）
        
        try:
            # Extract collection from filters if present / 如果过滤条件中存在 collection，则提取它
            collection = filters.get('collection') if filters else None
            
            _t0 = time.monotonic()
            results = self.sparse_retriever.retrieve(
                keywords=keywords,
                top_k=self.config.sparse_top_k,
                collection=collection,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("sparse_retrieval", {
                    "method": "bm25",
                    "keyword_count": len(keywords),
                    "top_k": self.config.sparse_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Sparse retrieval error: {e}"
            logger.error(error_msg)
            return None, error_msg
    
    def _fuse_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        """Fuse Dense and Sparse results using RRF. / 使用 RRF 融合稠密和稀疏结果。
        
        Args: / 参数：
            dense_results: Results from dense retrieval. / dense_results：稠密检索结果。
            sparse_results: Results from sparse retrieval. / sparse_results：稀疏检索结果。
            top_k: Number of results to return after fusion. / top_k：融合后返回的结果数量。
            trace: Optional TraceContext. / trace：可选 TraceContext。
            
        Returns: / 返回：
            Fused and ranked list of RetrievalResults. / 融合并排序后的 RetrievalResult 列表。
        """
        if self.fusion is None:
            # Fallback: interleave results (simple round-robin) / 回退：交错合并结果（简单轮询）
            logger.warning("No fusion configured, using simple interleave")
            return self._interleave_results(dense_results, sparse_results, top_k)
        
        # Build ranking lists for RRF / 为 RRF 构建排名列表
        ranking_lists = []
        if dense_results:
            ranking_lists.append(dense_results)
        if sparse_results:
            ranking_lists.append(sparse_results)
        
        if not ranking_lists:
            return []
        
        if len(ranking_lists) == 1:
            # Only one source, no fusion needed / 只有一个来源，无需融合
            return ranking_lists[0][:top_k]
        
        _t0 = time.monotonic()
        fused = self.fusion.fuse(
            ranking_lists=ranking_lists,
            top_k=top_k,
            trace=trace,
        )
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("fusion", {
                "method": "rrf",
                "input_lists": len(ranking_lists),
                "top_k": top_k,
                "result_count": len(fused),
                "chunks": _snapshot_results(fused),
            }, elapsed_ms=_elapsed)
        return fused
    
    def _interleave_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Simple interleave fallback when no fusion is configured. / 未配置融合器时的简单交错回退。
        
        Args: / 参数：
            dense_results: Results from dense retrieval. / dense_results：稠密检索结果。
            sparse_results: Results from sparse retrieval. / sparse_results：稀疏检索结果。
            top_k: Maximum results to return. / top_k：最大返回结果数量。
            
        Returns: / 返回：
            Interleaved results, deduped by chunk_id. / 交错后的结果，并按 chunk_id 去重。
        """
        seen_ids = set()
        interleaved = []
        
        d_idx, s_idx = 0, 0
        while len(interleaved) < top_k and (d_idx < len(dense_results) or s_idx < len(sparse_results)):
            # Alternate between dense and sparse / 在稠密和稀疏结果之间交替选择
            if d_idx < len(dense_results):
                r = dense_results[d_idx]
                d_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)
            
            if len(interleaved) >= top_k:
                break
            
            if s_idx < len(sparse_results):
                r = sparse_results[s_idx]
                s_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)
        
        return interleaved
    
    def _apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results (post-fusion fallback). / 对结果应用元数据过滤（融合后回退机制）。
        
        This is a backup filter mechanism for cases where the underlying / 这是一个备用过滤机制，用于底层
        storage doesn't fully support the filter syntax. / 存储未完全支持过滤语法的情况。
        
        Args: / 参数：
            results: Results to filter. / results：要过滤的结果。
            filters: Filter conditions to apply. / filters：要应用的过滤条件。
            
        Returns: / 返回：
            Filtered results. / 过滤后的结果。
        """
        if not filters:
            return results
        
        filtered = []
        for result in results:
            if self._matches_filters(result.metadata, filters):
                filtered.append(result)
        
        return filtered
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """Check if metadata matches all filter conditions. / 检查元数据是否匹配所有过滤条件。
        
        Args: / 参数：
            metadata: Result metadata. / metadata：结果元数据。
            filters: Filter conditions. / filters：过滤条件。
            
        Returns: / 返回：
            True if all filters match, False otherwise. / 如果所有过滤条件都匹配则返回 True，否则返回 False。
        """
        for key, value in filters.items():
            if key == "collection":
                # Collection might be in different metadata keys / collection 可能存在于不同的元数据键中
                meta_collection = (
                    metadata.get("collection") 
                    or metadata.get("source_collection")
                )
                if meta_collection != value:
                    return False
            elif key == "doc_type":
                if metadata.get("doc_type") != value:
                    return False
            elif key == "tags":
                # Tags is a list - check intersection / tags 是列表，检查交集
                meta_tags = metadata.get("tags", [])
                if not isinstance(value, list):
                    value = [value]
                if not set(meta_tags) & set(value):
                    return False
            elif key == "source_path":
                # Partial match for path / 对路径进行部分匹配
                source = metadata.get("source_path", "")
                if value not in source:
                    return False
            else:
                # Generic exact match / 通用精确匹配
                if metadata.get(key) != value:
                    return False
        
        return True


def create_hybrid_search(
    settings: Optional[Settings] = None,
    query_processor: Optional[QueryProcessor] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    fusion: Optional[RRFFusion] = None,
) -> HybridSearch:
    """Factory function to create HybridSearch with default components. / 使用默认组件创建 HybridSearch 的工厂函数。
    
    This is a convenience function that creates a HybridSearch with / 这是一个便捷函数，如果未提供融合器，
    default RRFFusion if not provided. / 会使用默认 RRFFusion 创建 HybridSearch。
    
    Args: / 参数：
        settings: Application settings. / settings：应用配置。
        query_processor: QueryProcessor instance. / query_processor：QueryProcessor 实例。
        dense_retriever: DenseRetriever instance. / dense_retriever：DenseRetriever 实例。
        sparse_retriever: SparseRetriever instance. / sparse_retriever：SparseRetriever 实例。
        fusion: RRFFusion instance. If None, creates default with k=60. / fusion：RRFFusion 实例。如果为 None，则创建 k=60 的默认实例。
        
    Returns: / 返回：
        Configured HybridSearch instance. / 配置完成的 HybridSearch 实例。
    
    Example: / 示例：
        >>> hybrid = create_hybrid_search(
        ...     settings=settings,
        ...     query_processor=QueryProcessor(),
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ... )
    """
    # Create default fusion if not provided / 如果未提供，则创建默认融合器
    if fusion is None:
        from src.core.query_engine.fusion import RRFFusion
        rrf_k = 60
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                rrf_k = getattr(retrieval_config, 'rrf_k', 60)
        fusion = RRFFusion(k=rrf_k)
    
    return HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )
