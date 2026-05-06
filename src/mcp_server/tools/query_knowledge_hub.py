"""MCP Tool: query_knowledge_hub / MCP 工具：query_knowledge_hub

This tool provides knowledge retrieval capabilities through the MCP protocol. / 该工具通过 MCP 协议提供知识检索能力。
It combines HybridSearch (Dense + Sparse + RRF Fusion) with optional Reranking / 它结合 HybridSearch（稠密 + 稀疏 + RRF 融合）和可选重排，
to find relevant documents and return formatted results with citations. / 查找相关文档并返回带引用的格式化结果。

Usage via MCP: / MCP 使用方式：
    Tool name: query_knowledge_hub / 工具名称：query_knowledge_hub
    Input schema: / 输入 schema：
        - query (string, required): The search query / query（字符串，必填）：搜索查询
        - top_k (integer, optional): Number of results to return (default: 5) / top_k（整数，可选）：返回结果数量（默认 5）
        - collection (string, optional): Limit search to specific collection / collection（字符串，可选）：限制搜索到指定集合
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from src.core.response.response_builder import ResponseBuilder, MCPToolResponse
from src.core.settings import load_settings, resolve_path, Settings
from src.core.trace import TraceContext, TraceCollector
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker

logger = logging.getLogger(__name__)


# Tool metadata / 工具元数据
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = """Search the knowledge base for relevant documents.

This tool uses hybrid search (semantic + keyword) to find the most relevant 
documents matching your query. Results include source citations for reference.

Parameters:
- query: Your search question or keywords
- top_k: Maximum number of results (default: 5)
- collection: Limit search to a specific document collection
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question to find relevant documents for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit the search scope.",
        },
    },
    "required": ["query"],
}


@dataclass
class QueryKnowledgeHubConfig:
    """Configuration for query_knowledge_hub tool. / query_knowledge_hub 工具配置。
    
    Attributes: / 属性：
        default_top_k: Default number of results if not specified / 未指定时的默认结果数量
        max_top_k: Maximum allowed top_k value / 允许的最大 top_k 值
        default_collection: Default collection if not specified / 未指定时的默认集合
        enable_rerank: Whether to apply reranking / 是否应用重排
    """
    default_top_k: int = 5
    max_top_k: int = 20
    default_collection: str = "default"
    enable_rerank: bool = True


class QueryKnowledgeHubTool:
    """MCP Tool for knowledge base queries. / 用于知识库查询的 MCP 工具。
    
    This class encapsulates the query_knowledge_hub tool logic, / 该类封装 query_knowledge_hub 工具逻辑，
    coordinating HybridSearch and Reranker to produce formatted results. / 协调 HybridSearch 和 Reranker 生成格式化结果。
    
    Design Principles: / 设计原则：
    - Lazy initialization: Components created on first use / 延迟初始化：组件在首次使用时创建
    - Error resilience: Graceful handling of search/rerank failures / 错误韧性：优雅处理搜索或重排失败
    - Configurable: All parameters from settings.yaml / 可配置：所有参数来自 settings.yaml
    
    Example:
        >>> tool = QueryKnowledgeHubTool(settings)
        >>> result = await tool.execute(query="Azure 配置", top_k=5)
        >>> print(result.content)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[QueryKnowledgeHubConfig] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[CoreReranker] = None,
        response_builder: Optional[ResponseBuilder] = None,
    ) -> None:
        """Initialize QueryKnowledgeHubTool. / 初始化 QueryKnowledgeHubTool。
        
        Args: / 参数：
            settings: Application settings. If None, loaded from default path. / 应用设置；如果为 None，则从默认路径加载。
            config: Tool configuration. If None, uses defaults. / 工具配置；如果为 None，则使用默认值。
            hybrid_search: Optional pre-configured HybridSearch instance. / 可选的预配置 HybridSearch 实例。
            reranker: Optional pre-configured CoreReranker instance. / 可选的预配置 CoreReranker 实例。
            response_builder: Optional pre-configured ResponseBuilder instance. / 可选的预配置 ResponseBuilder 实例。
        """
        self._settings = settings
        self.config = config or QueryKnowledgeHubConfig()
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._embedding_client = None
        self._response_builder = response_builder or ResponseBuilder()
        
        # Track initialization state / 跟踪初始化状态
        self._initialized = False
        self._current_collection: Optional[str] = None
    
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary. / 获取设置，必要时加载。"""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings
    
    def _ensure_initialized(self, collection: str) -> None:
        """Ensure search components are initialized for the given collection. / 确保给定集合的搜索组件已初始化。
        
        Caching strategy (balances speed vs freshness): / 缓存策略（平衡速度与新鲜度）：
        - **Fully cached** (stateless, never go stale): embedding client, / **完全缓存**（无状态，不会过期）：embedding 客户端、
          reranker, query processor, settings. / reranker、query processor、settings。
        - **Cached until collection changes**: vector store (ChromaDB / **缓存到集合变化为止**：vector store（ChromaDB
          PersistentClient reads from SQLite — sees data written by other / PersistentClient 从 SQLite 读取，可看到其他进程写入的
          processes), dense retriever, hybrid search. / 数据）、dense retriever、hybrid search。
        - **Auto-refreshes on every query**: BM25 sparse index — the / **每次查询自动刷新**：BM25 稀疏索引 -
          ``SparseRetriever._ensure_index_loaded()`` always reloads from / ``SparseRetriever._ensure_index_loaded()`` 总是从
          disk, so the cached SparseRetriever object is fine. / 磁盘重新加载，因此缓存 SparseRetriever 对象本身没有问题。
        
        Only when *collection* changes do we tear down and rebuild. / 只有当 *collection* 变化时，才会拆除并重建。
        
        Args: / 参数：
            collection: Target collection name. / 目标集合名称。
        """
        # Always rebuild vector_store and retriever components so that / 始终重建 vector_store 和 retriever 组件，以便
        # data ingested by other processes (e.g. Dashboard) is visible / 其他进程（例如 Dashboard）写入的数据可以
        # immediately without requiring an MCP Server restart. / 无需重启 MCP Server 便立即可见。
        
        logger.info(f"Initializing query components for collection: {collection}")
        
        # Import here to avoid circular imports and allow lazy loading / 在这里导入以避免循环导入并支持延迟加载
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        
        # === Fully cached components (stateless, never go stale) === / === 完全缓存组件（无状态，不会过期）===
        if self._embedding_client is None:
            self._embedding_client = EmbeddingFactory.create(self.settings)
        
        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)
        
        # === Rebuild for new collection === / === 为新集合重建 ===
        # ChromaDB PersistentClient uses SQLite under the hood — / ChromaDB PersistentClient 底层使用 SQLite -
        # concurrent readers see committed writes from other processes / 并发读取者可以看到其他进程已提交的写入
        # (dashboard ingestion), so caching the client is safe. / （例如 dashboard ingestion），因此缓存客户端是安全的。
        vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        
        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )
        
        # BM25Indexer just holds the index dir path; the SparseRetriever / BM25Indexer 只保存索引目录路径；SparseRetriever
        # calls _ensure_index_loaded() on every search, which always / 每次搜索都会调用 _ensure_index_loaded()，它总是
        # reloads from disk — so it picks up dashboard-written data. / 从磁盘重新加载，因此能获取 dashboard 写入的数据。
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        
        query_processor = QueryProcessor()
        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        
        self._current_collection = collection
        self._initialized = True
        logger.info(f"Query components initialized for collection: {collection}")
    
    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> MCPToolResponse:
        """Execute the query_knowledge_hub tool. / 执行 query_knowledge_hub 工具。
        
        Args: / 参数：
            query: Search query string. / 搜索查询字符串。
            top_k: Maximum results to return. / 要返回的最大结果数。
            collection: Target collection name. / 目标集合名称。
            
        Returns: / 返回：
            MCPToolResponse with formatted content and citations. / 包含格式化内容和引用的 MCPToolResponse。
            
        Raises: / 抛出：
            ValueError: If query is empty or invalid. / 如果查询为空或无效。
        """
        # Validate query / 校验查询
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Apply defaults / 应用默认值
        effective_top_k = min(
            top_k or self.config.default_top_k,
            self.config.max_top_k
        )
        effective_collection = collection or self.config.default_collection
        
        logger.info(
            f"Executing query_knowledge_hub: query='{query[:50]}...', "
            f"top_k={effective_top_k}, collection={effective_collection}"
        )
        
        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["top_k"] = effective_top_k
        trace.metadata["collection"] = effective_collection
        trace.metadata["source"] = "mcp"

        try:
            # Initialize components for collection / 初始化集合相关组件
            # Run blocking I/O (embedding API, ChromaDB, BM25) in a thread / 在线程中运行阻塞 I/O（embedding API、ChromaDB、BM25）
            # to avoid blocking the async event loop / MCP stdio transport / 避免阻塞异步事件循环或 MCP stdio 传输
            import time as _time
            _init_t0 = _time.monotonic()
            await asyncio.to_thread(self._ensure_initialized, effective_collection)
            _init_elapsed = (_time.monotonic() - _init_t0) * 1000.0
            trace.record_stage("initialization", {
                "collection": effective_collection,
                "cold_start": _init_elapsed > 500,  # >500ms ≈ cold
            }, elapsed_ms=_init_elapsed)
            
            # Perform hybrid search (blocking: embedding API + DB queries) / 执行混合搜索（阻塞：embedding API + 数据库查询）
            results = await asyncio.to_thread(
                self._perform_search, query, effective_top_k, trace,
            )
            
            # Apply reranking if enabled (may call LLM API) / 如果启用则应用重排（可能调用 LLM API）
            if self.config.enable_rerank and results:
                results = await asyncio.to_thread(
                    self._apply_rerank, query, results, effective_top_k, trace,
                )
            
            # Build response / 构建响应
            response = self._response_builder.build(
                results=results,
                query=query,
                collection=effective_collection,
            )
            
            # Store final results in trace for dashboard display / 将最终结果存入 trace 供 dashboard 展示
            trace.metadata["final_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.text or "",
                    "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                    "title": r.metadata.get("title", ""),
                }
                for r in results
            ]

            logger.info(
                f"query_knowledge_hub completed: {len(results)} results, "
                f"is_empty={response.is_empty}"
            )
            
            TraceCollector().collect(trace)
            return response
            
        except Exception as e:
            logger.exception(f"query_knowledge_hub failed: {e}")
            TraceCollector().collect(trace)
            # Return error response / 返回错误响应
            return self._build_error_response(query, effective_collection, str(e))
    
    def _perform_search(
        self,
        query: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search. / 执行混合搜索。
        
        Args: / 参数：
            query: Search query. / 搜索查询。
            top_k: Maximum results. / 最大结果数。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            
        Returns: / 返回：
            List of RetrievalResult. / RetrievalResult 列表。
        """
        if self._hybrid_search is None:
            raise RuntimeError("HybridSearch not initialized")
        
        # Use a larger initial retrieval for reranking / 为重排使用更大的初始检索数量
        initial_top_k = top_k * 2 if self.config.enable_rerank else top_k
        
        try:
            results = self._hybrid_search.search(
                query=query,
                top_k=initial_top_k,
                filters=None,
                trace=trace,
                return_details=False,
            )
            return results if isinstance(results, list) else results.results
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []
    
    def _apply_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Apply reranking to search results. / 对搜索结果应用重排。
        
        Args: / 参数：
            query: Original query. / 原始查询。
            results: Search results to rerank. / 要重排的搜索结果。
            top_k: Final number of results. / 最终结果数量。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            
        Returns: / 返回：
            Reranked results (or original if reranking fails). / 重排后的结果（如果重排失败则返回原始结果）。
        """
        if self._reranker is None or not self._reranker.is_enabled:
            return results[:top_k]
        
        try:
            rerank_result = self._reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                trace=trace,
            )
            
            if rerank_result.used_fallback:
                logger.warning(
                    f"Reranker fallback: {rerank_result.fallback_reason}"
                )
            
            return rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:top_k]
    
    def _build_error_response(
        self,
        query: str,
        collection: str,
        error_message: str,
    ) -> MCPToolResponse:
        """Build error response. / 构建错误响应。
        
        Args: / 参数：
            query: Original query. / 原始查询。
            collection: Target collection. / 目标集合。
            error_message: Error description. / 错误描述。
            
        Returns: / 返回：
            MCPToolResponse indicating error. / 表示错误的 MCPToolResponse。
        """
        content = f"## 查询失败\n\n"
        content += f"查询: **{query}**\n"
        content += f"集合: `{collection}`\n\n"
        content += f"**错误信息:** {error_message}\n\n"
        content += "请检查:\n"
        content += "- 数据库连接是否正常\n"
        content += "- 集合是否已创建并包含数据\n"
        content += "- 配置文件是否正确\n"
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "query": query,
                "collection": collection,
                "error": error_message,
            },
            is_empty=True,
        )


# Module-level tool instance (lazy-initialized) / 模块级工具实例（延迟初始化）
_tool_instance: Optional[QueryKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> QueryKnowledgeHubTool:
    """Get or create the tool instance. / 获取或创建工具实例。
    
    Args: / 参数：
        settings: Optional settings to use for initialization. / 用于初始化的可选设置。
        
    Returns: / 返回：
        QueryKnowledgeHubTool instance. / QueryKnowledgeHubTool 实例。
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = QueryKnowledgeHubTool(settings=settings)
    return _tool_instance


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    """Handler function for MCP tool registration. / MCP 工具注册的处理函数。
    
    This function is registered with the ProtocolHandler and called / 该函数注册到 ProtocolHandler，
    when the MCP client invokes the query_knowledge_hub tool. / 并在 MCP 客户端调用 query_knowledge_hub 工具时执行。
    
    Supports multimodal responses - if search results contain images, / 支持多模态响应 - 如果搜索结果包含图片，
    the response will include ImageContent blocks alongside TextContent. / 响应会在 TextContent 之外包含 ImageContent 块。
    
    Args: / 参数：
        query: Search query string. / 搜索查询字符串。
        top_k: Maximum number of results. / 最大结果数量。
        collection: Optional collection name. / 可选集合名称。
        
    Returns: / 返回：
        MCP CallToolResult with content blocks (text and optionally images). / 包含内容块（文本和可选图片）的 MCP CallToolResult。
    """
    tool = get_tool_instance()
    
    try:
        response = await tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
        )
        
        # Use to_mcp_content() which handles multimodal (text + images) / 使用可处理多模态（文本 + 图片）的 to_mcp_content()
        content_blocks = response.to_mcp_content()
        
        return types.CallToolResult(
            content=content_blocks,
            isError=response.is_empty and "error" in response.metadata,
        )
        
    except ValueError as e:
        # Invalid parameters / 无效参数
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"参数错误: {e}",
                )
            ],
            isError=True,
        )
    except Exception as e:
        # Internal error / 内部错误
        logger.exception(f"query_knowledge_hub handler error: {e}")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"内部错误: 查询处理失败",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register query_knowledge_hub tool with the protocol handler. / 将 query_knowledge_hub 工具注册到协议处理器。
    
    Args: / 参数：
        protocol_handler: ProtocolHandler instance to register with. / 要注册到的 ProtocolHandler 实例。
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,
    )
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
