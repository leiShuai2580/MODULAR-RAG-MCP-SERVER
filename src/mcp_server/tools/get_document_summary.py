"""MCP Tool: get_document_summary / MCP 工具：get_document_summary

This tool provides document summary retrieval capabilities through the MCP protocol. / 该工具通过 MCP 协议提供文档摘要检索能力。
It returns title, summary, and tags for a specific document identified by doc_id. / 它会返回由 doc_id 标识的特定文档的标题、摘要和标签。

Usage via MCP: / MCP 使用方式：
    Tool name: get_document_summary / 工具名称：get_document_summary
    Input schema: / 输入 schema：
        - doc_id (string, required): The document ID to retrieve summary for / doc_id（字符串，必填）：要检索摘要的文档 ID
        - collection (string, optional): Collection name to search in / collection（字符串，可选）：要搜索的集合名称
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata / 工具元数据
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = """Get summary and metadata for a specific document.

Returns structured information about a document including:
- Title (extracted or inferred from content)
- Summary (first chunk preview or metadata summary)
- Tags (document-level tags/categories)
- Source path
- Chunk count

Use this tool after list_collections to get details about specific documents.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "The document ID to retrieve summary for. Can be full doc_id (e.g., 'doc_abc123') or the hash portion.",
        },
        "collection": {
            "type": "string",
            "description": "Collection name to search in. If not specified, searches the default collection.",
        },
    },
    "required": ["doc_id"],
}


@dataclass
class DocumentSummary:
    """Summary information for a document. / 文档摘要信息。
    
    Attributes: / 属性：
        doc_id: Document identifier / 文档标识符
        title: Document title (from metadata or inferred) / 文档标题（来自元数据或推断）
        summary: Brief summary or preview of document content / 文档内容的简要摘要或预览
        tags: List of tags/categories associated with the document / 与文档关联的标签或类别列表
        source_path: Original file path / 原始文件路径
        chunk_count: Number of chunks for this document / 该文档的分块数量
        metadata: Additional document metadata / 额外文档元数据
    """
    doc_id: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation. / 转换为字典表示。"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source_path": self.source_path,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
        }


@dataclass
class GetDocumentSummaryConfig:
    """Configuration for get_document_summary tool. / get_document_summary 工具配置。
    
    Attributes: / 属性：
        persist_directory: Path to ChromaDB storage directory / ChromaDB 存储目录路径
        default_collection: Default collection name if not specified / 未指定时的默认集合名称
        summary_max_length: Maximum characters for summary preview / 摘要预览的最大字符数
    """
    persist_directory: str = "./data/db/chroma"
    default_collection: str = "knowledge_hub"
    summary_max_length: int = 500


class DocumentNotFoundError(Exception):
    """Raised when a document with the specified ID is not found. / 找不到指定 ID 的文档时抛出。"""
    
    def __init__(self, doc_id: str, collection: Optional[str] = None):
        self.doc_id = doc_id
        self.collection = collection
        message = f"Document '{doc_id}' not found"
        if collection:
            message += f" in collection '{collection}'"
        super().__init__(message)


class GetDocumentSummaryTool:
    """MCP Tool for retrieving document summaries. / 用于检索文档摘要的 MCP 工具。
    
    This class encapsulates the get_document_summary tool logic, / 该类封装 get_document_summary 工具逻辑，
    querying the vector store to retrieve document metadata and content preview. / 通过查询向量存储获取文档元数据和内容预览。
    
    Design Principles: / 设计原则：
    - Config-Driven: Paths from settings.yaml / 配置驱动：路径来自 settings.yaml
    - Error Resilience: Clear error messages for missing documents / 错误韧性：为缺失文档提供清晰错误信息
    - Observable: Logging for debugging / 可观测：记录日志便于调试
    - Lazy Init: ChromaDB client created on first use / 延迟初始化：ChromaDB 客户端在首次使用时创建
    
    Example:
        >>> tool = GetDocumentSummaryTool(settings)
        >>> result = await tool.execute(doc_id="doc_abc123")
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[GetDocumentSummaryConfig] = None,
    ) -> None:
        """Initialize GetDocumentSummaryTool. / 初始化 GetDocumentSummaryTool。
        
        Args: / 参数：
            settings: Application settings. If None, loaded from default path. / 应用设置；如果为 None，则从默认路径加载。
            config: Tool configuration. If None, derived from settings. / 工具配置；如果为 None，则从设置推导。
        """
        self._settings = settings
        self._config = config
        self._chroma_client = None
        
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary. / 获取设置，必要时加载。"""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings
    
    @property
    def config(self) -> GetDocumentSummaryConfig:
        """Get configuration, deriving from settings if necessary. / 获取配置，必要时从设置推导。"""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
                default_collection = getattr(
                    self.settings.vector_store,
                    'collection_name',
                    'knowledge_hub'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
                default_collection = 'knowledge_hub'
            
            self._config = GetDocumentSummaryConfig(
                persist_directory=persist_dir,
                default_collection=default_collection,
            )
        return self._config
    
    def _get_chroma_client(self) -> Any:
        """Get or create ChromaDB client. / 获取或创建 ChromaDB 客户端。
        
        Returns: / 返回：
            ChromaDB PersistentClient instance. / ChromaDB PersistentClient 实例。
            
        Raises: / 抛出：
            ImportError: If chromadb is not installed. / 如果未安装 chromadb。
            RuntimeError: If client creation fails. / 如果客户端创建失败。
        """
        if self._chroma_client is not None:
            return self._chroma_client
        
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for get_document_summary. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return self._chroma_client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e
    
    def _get_collection(self, collection_name: Optional[str] = None) -> Any:
        """Get ChromaDB collection. / 获取 ChromaDB 集合。
        
        Args: / 参数：
            collection_name: Collection name. Uses default if not specified. / 集合名称；未指定时使用默认值。
            
        Returns: / 返回：
            ChromaDB collection instance. / ChromaDB 集合实例。
            
        Raises: / 抛出：
            ValueError: If collection does not exist. / 如果集合不存在。
        """
        client = self._get_chroma_client()
        name = collection_name or self.config.default_collection
        
        try:
            # Try to get existing collection / 尝试获取现有集合
            collection = client.get_collection(name=name)
            return collection
        except Exception as e:
            raise ValueError(
                f"Collection '{name}' does not exist: {e}"
            ) from e
    
    def _find_document_chunks(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find all chunks belonging to a document. / 查找属于某个文档的所有分块。
        
        Searches for chunks where source_ref matches the doc_id. / 搜索 source_ref 与 doc_id 匹配的分块。
        Falls back to partial matching on chunk IDs if source_ref is not available. / 如果 source_ref 不可用，则回退到对 chunk ID 进行部分匹配。
        
        Args: / 参数：
            doc_id: Document ID to search for. / 要搜索的文档 ID。
            collection_name: Collection to search in. / 要搜索的集合。
            
        Returns: / 返回：
            List of chunk data with metadata. / 包含元数据的分块数据列表。
        """
        collection = self._get_collection(collection_name)
        
        # Strategy 1: Search by source_ref metadata / 策略 1：按 source_ref 元数据搜索
        # Chunks should have source_ref pointing to parent document / 分块应有指向父文档的 source_ref
        try:
            results = collection.get(
                where={"source_ref": doc_id},
                include=["metadatas", "documents"]
            )
            
            if results and results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        'id': chunk_id,
                        'text': results['documents'][i] if results.get('documents') else '',
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {}
                    })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"source_ref search failed: {e}")
        
        # Strategy 2: Search by doc_id in chunk ID prefix / 策略 2：在 chunk ID 前缀中按 doc_id 搜索
        # Chunk IDs follow format: {doc_id}_{index:04d}_{hash} / chunk ID 遵循格式：{doc_id}_{index:04d}_{hash}
        try:
            # Get all chunks and filter by ID prefix / 获取所有分块并按 ID 前缀过滤
            all_results = collection.get(include=["metadatas", "documents"])
            
            if all_results and all_results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(all_results['ids']):
                    # Check if chunk_id starts with doc_id / 检查 chunk_id 是否以 doc_id 开头
                    if chunk_id.startswith(doc_id) or doc_id in chunk_id:
                        chunks.append({
                            'id': chunk_id,
                            'text': all_results['documents'][i] if all_results.get('documents') else '',
                            'metadata': all_results['metadatas'][i] if all_results.get('metadatas') else {}
                        })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"ID prefix search failed: {e}")
        
        # No chunks found / 未找到分块
        return []
    
    def get_document_summary(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> DocumentSummary:
        """Get summary for a specific document. / 获取特定文档的摘要。
        
        Args: / 参数：
            doc_id: Document ID to retrieve. / 要检索的文档 ID。
            collection: Collection name to search in. / 要搜索的集合名称。
            
        Returns: / 返回：
            DocumentSummary with title, summary, tags, etc. / 包含标题、摘要、标签等信息的 DocumentSummary。
            
        Raises: / 抛出：
            DocumentNotFoundError: If document is not found. / 如果未找到文档。
        """
        chunks = self._find_document_chunks(doc_id, collection)
        
        if not chunks:
            raise DocumentNotFoundError(doc_id, collection)
        
        # Sort chunks by chunk_index if available / 如果存在 chunk_index，则按其排序分块
        chunks.sort(key=lambda c: c.get('metadata', {}).get('chunk_index', 0))
        
        # Extract document-level info from first chunk's metadata / 从第一个分块的元数据中提取文档级信息
        first_chunk = chunks[0]
        metadata = first_chunk.get('metadata', {})
        
        # Extract title / 提取标题
        title = self._extract_title(metadata, first_chunk.get('text', ''))
        
        # Extract or generate summary / 提取或生成摘要
        summary = self._extract_summary(chunks)
        
        # Extract tags / 提取标签
        tags = self._extract_tags(metadata)
        
        # Extract source path / 提取源路径
        source_path = metadata.get('source_path', metadata.get('source', None))
        
        # Collect additional metadata (excluding internal fields) / 收集额外元数据（排除内部字段）
        additional_metadata = self._filter_metadata(metadata)
        
        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            summary=summary,
            tags=tags,
            source_path=source_path,
            chunk_count=len(chunks),
            metadata=additional_metadata,
        )
    
    def _extract_title(self, metadata: Dict[str, Any], first_text: str) -> str:
        """Extract document title from metadata or content. / 从元数据或内容中提取文档标题。
        
        Priority: / 优先级：
        1. metadata['title'] / metadata['title']
        2. First heading from content / 内容中的第一个标题
        3. Filename from source_path / source_path 中的文件名
        4. "Untitled Document" / “Untitled Document”
        
        Args: / 参数：
            metadata: Chunk metadata. / 分块元数据。
            first_text: First chunk text content. / 第一个分块的文本内容。
            
        Returns: / 返回：
            Extracted or inferred title. / 提取或推断出的标题。
        """
        # Priority 1: Explicit title in metadata / 优先级 1：元数据中的显式标题
        if metadata.get('title'):
            return str(metadata['title'])
        
        # Priority 2: First markdown heading / 优先级 2：第一个 Markdown 标题
        if first_text:
            lines = first_text.split('\n')
            for line in lines[:10]:  # Check first 10 lines / 检查前 10 行
                line = line.strip()
                if line.startswith('# '):
                    return line[2:].strip()
        
        # Priority 3: Filename from source_path / 优先级 3：source_path 中的文件名
        source_path = metadata.get('source_path', metadata.get('source'))
        if source_path:
            filename = Path(source_path).stem
            # Convert snake_case/kebab-case to Title Case / 将 snake_case/kebab-case 转换为 Title Case
            title = filename.replace('_', ' ').replace('-', ' ').title()
            return title
        
        # Priority 4: Default / 优先级 4：默认值
        return "Untitled Document"
    
    def _extract_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Extract or generate document summary. / 提取或生成文档摘要。
        
        Priority: / 优先级：
        1. metadata['summary'] from any chunk / 任意分块中的 metadata['summary']
        2. First N characters from first chunk text / 第一个分块文本的前 N 个字符
        
        Args: / 参数：
            chunks: List of document chunks. / 文档分块列表。
            
        Returns: / 返回：
            Summary text. / 摘要文本。
        """
        # Priority 1: Explicit summary in metadata / 优先级 1：元数据中的显式摘要
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('summary'):
                return str(metadata['summary'])
        
        # Priority 2: Preview from first chunk / 优先级 2：来自第一个分块的预览
        first_text = chunks[0].get('text', '') if chunks else ''
        if first_text:
            # Clean up and truncate / 清理并截断
            summary = first_text.strip()
            
            # Skip markdown headers for preview / 预览时跳过 Markdown 标题
            lines = summary.split('\n')
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    content_lines.append(line)
                    if len(' '.join(content_lines)) > self.config.summary_max_length:
                        break
            
            summary = ' '.join(content_lines)
            
            if len(summary) > self.config.summary_max_length:
                summary = summary[:self.config.summary_max_length - 3] + "..."
            
            return summary if summary else "No content preview available."
        
        return "No summary available."
    
    def _extract_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract tags from metadata. / 从元数据中提取标签。
        
        Args: / 参数：
            metadata: Chunk metadata. / 分块元数据。
            
        Returns: / 返回：
            List of tags. / 标签列表。
        """
        tags = []
        
        # Check for explicit tags field / 检查显式 tags 字段
        if 'tags' in metadata:
            tag_value = metadata['tags']
            if isinstance(tag_value, list):
                tags.extend(str(t) for t in tag_value)
            elif isinstance(tag_value, str):
                # Could be comma-separated / 可能是逗号分隔
                tags.extend(t.strip() for t in tag_value.split(',') if t.strip())
        
        # Add doc_type as a tag if available / 如果存在 doc_type，则作为标签添加
        if metadata.get('doc_type'):
            doc_type = str(metadata['doc_type']).upper()
            if doc_type not in tags:
                tags.append(doc_type)
        
        return tags
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metadata to exclude internal fields. / 过滤元数据以排除内部字段。
        
        Args: / 参数：
            metadata: Raw metadata dict. / 原始元数据字典。
            
        Returns: / 返回：
            Filtered metadata with user-relevant fields only. / 仅包含用户相关字段的过滤后元数据。
        """
        # Fields to exclude from additional metadata / 要从额外元数据中排除的字段
        exclude_fields = {
            'source_ref', 'chunk_index', 'start_offset', 'end_offset',
            '_placeholder', 'text', 'title', 'summary', 'tags',
            'source_path', 'source'
        }
        
        return {
            k: v for k, v in metadata.items()
            if k not in exclude_fields and not k.startswith('_')
        }
    
    def format_response(self, summary: DocumentSummary) -> str:
        """Format document summary as a readable string. / 将文档摘要格式化为可读字符串。
        
        Args: / 参数：
            summary: DocumentSummary object. / DocumentSummary 对象。
            
        Returns: / 返回：
            Formatted string suitable for MCP response. / 适合 MCP 响应的格式化字符串。
        """
        lines = [
            f"## Document: {summary.title}",
            "",
            f"**Document ID:** `{summary.doc_id}`",
        ]
        
        if summary.source_path:
            lines.append(f"**Source:** {summary.source_path}")
        
        lines.append(f"**Chunks:** {summary.chunk_count}")
        
        if summary.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in summary.tags)
            lines.append(f"**Tags:** {tags_str}")
        
        lines.extend([
            "",
            "### Summary",
            "",
            summary.summary,
        ])
        
        if summary.metadata:
            lines.extend([
                "",
                "### Additional Metadata",
                "",
            ])
            for key, value in summary.metadata.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def format_error(self, error: Exception) -> str:
        """Format error as a readable string. / 将错误格式化为可读字符串。
        
        Args: / 参数：
            error: Exception that occurred. / 发生的异常。
            
        Returns: / 返回：
            Formatted error message. / 格式化后的错误消息。
        """
        if isinstance(error, DocumentNotFoundError):
            return f"## Document Not Found\n\n{str(error)}\n\nPlease verify the document ID and collection name."
        elif isinstance(error, ValueError):
            return f"## Invalid Request\n\n{str(error)}"
        else:
            return f"## Error\n\nAn error occurred: {str(error)}"
    
    async def execute(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """Execute the get_document_summary tool. / 执行 get_document_summary 工具。
        
        Args: / 参数：
            doc_id: Document ID to retrieve summary for. / 要检索摘要的文档 ID。
            collection: Optional collection name. / 可选集合名称。
            
        Returns: / 返回：
            CallToolResult with formatted document summary or error. / 包含格式化文档摘要或错误的 CallToolResult。
        """
        logger.info(f"Executing get_document_summary (doc_id={doc_id}, collection={collection})")
        
        try:
            # Run blocking ChromaDB I/O in a thread to avoid blocking / 在线程中运行阻塞式 ChromaDB I/O，避免阻塞
            # the async event loop / MCP stdio transport / 异步事件循环或 MCP stdio 传输
            summary = await asyncio.to_thread(
                self.get_document_summary, doc_id, collection,
            )
            response_text = self.format_response(summary)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except DocumentNotFoundError as e:
            logger.warning(f"Document not found: {e}")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )
            
        except Exception as e:
            logger.exception("Error executing get_document_summary")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the get_document_summary tool with the protocol handler. / 将 get_document_summary 工具注册到协议处理器。
    
    This function is called by _register_default_tools() in protocol_handler.py / 该函数由 protocol_handler.py 中的 _register_default_tools() 调用，
    to register this tool when the MCP server starts. / 在 MCP 服务器启动时注册该工具。
    
    Args: / 参数：
        protocol_handler: ProtocolHandler instance to register with. / 要注册到的 ProtocolHandler 实例。
    """
    tool = GetDocumentSummaryTool()
    
    async def handler(
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """Handler function for MCP tool calls. / MCP 工具调用的处理函数。"""
        return await tool.execute(doc_id=doc_id, collection=collection)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
