"""MCP Tool: list_collections / MCP 工具：list_collections

This tool provides collection listing capabilities through the MCP protocol. / 该工具通过 MCP 协议提供集合列表能力。
It lists all available collections in the vector store with statistics. / 它会列出向量存储中的所有可用集合及其统计信息。

Usage via MCP: / MCP 使用方式：
    Tool name: list_collections / 工具名称：list_collections
    Input schema: / 输入 schema：
        - include_stats (boolean, optional): Include statistics for each collection / include_stats（布尔值，可选）：包含每个集合的统计信息
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata / 工具元数据
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = """List all available document collections in the knowledge base.

Returns information about each collection including:
- Collection name
- Document count (if include_stats=true)
- Collection metadata

Use this tool to discover available collections before querying.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "Whether to include statistics (document count) for each collection.",
            "default": True,
        },
    },
    "required": [],
}


@dataclass
class CollectionInfo:
    """Information about a single collection. / 单个集合的信息。
    
    Attributes: / 属性：
        name: Collection name / 集合名称
        count: Number of documents/chunks in the collection (optional) / 集合中的文档或分块数量（可选）
        metadata: Collection metadata dictionary / 集合元数据字典
    """
    name: str
    count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation. / 转换为字典表示。"""
        result: Dict[str, Any] = {"name": self.name}
        if self.count is not None:
            result["count"] = self.count
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ListCollectionsConfig:
    """Configuration for list_collections tool. / list_collections 工具配置。
    
    Attributes: / 属性：
        persist_directory: Path to ChromaDB storage directory / ChromaDB 存储目录路径
        include_stats_default: Default value for include_stats parameter / include_stats 参数的默认值
    """
    persist_directory: str = "./data/db/chroma"
    include_stats_default: bool = True


class ListCollectionsTool:
    """MCP Tool for listing knowledge base collections. / 用于列出知识库集合的 MCP 工具。
    
    This class encapsulates the list_collections tool logic, / 该类封装 list_collections 工具逻辑，
    querying the vector store to enumerate available collections. / 通过查询向量存储枚举可用集合。
    
    Design Principles: / 设计原则：
    - Config-Driven: Paths from settings.yaml / 配置驱动：路径来自 settings.yaml
    - Error Resilience: Graceful handling of missing directories / 错误韧性：优雅处理目录缺失
    - Observable: Logging for debugging / 可观测：记录日志便于调试
    
    Example:
        >>> tool = ListCollectionsTool(settings)
        >>> result = await tool.execute(include_stats=True)
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[ListCollectionsConfig] = None,
    ) -> None:
        """Initialize ListCollectionsTool. / 初始化 ListCollectionsTool。
        
        Args: / 参数：
            settings: Application settings. If None, loaded from default path. / 应用设置；如果为 None，则从默认路径加载。
            config: Tool configuration. If None, derived from settings. / 工具配置；如果为 None，则从设置推导。
        """
        self._settings = settings
        self._config = config
        
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary. / 获取设置，必要时加载。"""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings
    
    @property
    def config(self) -> ListCollectionsConfig:
        """Get configuration, deriving from settings if necessary. / 获取配置，必要时从设置推导。"""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
            
            self._config = ListCollectionsConfig(
                persist_directory=persist_dir
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
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for list_collections. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            # Return client anyway - it will just have no collections / 仍然返回客户端 - 它只会没有任何集合
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e
    
    def list_collections(
        self,
        include_stats: bool = True
    ) -> List[CollectionInfo]:
        """List all available collections. / 列出所有可用集合。
        
        Args: / 参数：
            include_stats: Whether to include document counts. / 是否包含文档数量。
            
        Returns: / 返回：
            List of CollectionInfo objects. / CollectionInfo 对象列表。
        """
        try:
            client = self._get_chroma_client()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get ChromaDB client: {e}")
            return []
        
        collections_info: List[CollectionInfo] = []
        
        try:
            # Get all collections from ChromaDB / 从 ChromaDB 获取所有集合
            collections = client.list_collections()
            
            for collection in collections:
                info = CollectionInfo(
                    name=collection.name,
                    metadata=collection.metadata
                )
                
                if include_stats:
                    try:
                        info.count = collection.count()
                    except Exception as e:
                        logger.warning(
                            f"Failed to get count for collection '{collection.name}': {e}"
                        )
                        info.count = None
                
                collections_info.append(info)
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
        
        logger.info(f"Found {len(collections_info)} collections")
        return collections_info
    
    def format_response(
        self,
        collections: List[CollectionInfo]
    ) -> str:
        """Format collections list as a readable string. / 将集合列表格式化为可读字符串。
        
        Args: / 参数：
            collections: List of CollectionInfo objects. / CollectionInfo 对象列表。
            
        Returns: / 返回：
            Formatted string suitable for MCP response. / 适合 MCP 响应的格式化字符串。
        """
        if not collections:
            return "No collections found in the knowledge base."
        
        lines = [
            f"## Available Collections ({len(collections)} total)\n"
        ]
        
        for i, coll in enumerate(collections, 1):
            line = f"{i}. **{coll.name}**"
            
            if coll.count is not None:
                line += f" - {coll.count} documents"
            
            if coll.metadata:
                # Filter out internal metadata / 过滤内部元数据
                user_metadata = {
                    k: v for k, v in coll.metadata.items()
                    if not k.startswith('_') and not k.startswith('hnsw:')
                }
                if user_metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in user_metadata.items())
                    line += f" ({meta_str})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    async def execute(
        self,
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """Execute the list_collections tool. / 执行 list_collections 工具。
        
        Args: / 参数：
            include_stats: Whether to include statistics for each collection. / 是否包含每个集合的统计信息。
            
        Returns: / 返回：
            CallToolResult with formatted collection list. / 包含格式化集合列表的 CallToolResult。
        """
        logger.info(f"Executing list_collections (include_stats={include_stats})")
        
        try:
            # Run blocking ChromaDB I/O in a thread to avoid blocking / 在线程中运行阻塞式 ChromaDB I/O，避免阻塞
            # the async event loop / MCP stdio transport / 异步事件循环或 MCP stdio 传输
            collections = await asyncio.to_thread(
                self.list_collections, include_stats,
            )
            response_text = self.format_response(collections)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except Exception as e:
            logger.exception("Error executing list_collections")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error listing collections: {str(e)}",
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the list_collections tool with the protocol handler. / 将 list_collections 工具注册到协议处理器。
    
    This function is called by _register_default_tools() in protocol_handler.py / 该函数由 protocol_handler.py 中的 _register_default_tools() 调用，
    to register this tool when the MCP server starts. / 在 MCP 服务器启动时注册该工具。
    
    Args: / 参数：
        protocol_handler: ProtocolHandler instance to register with. / 要注册到的 ProtocolHandler 实例。
    """
    tool = ListCollectionsTool()
    
    async def handler(
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """Handler function for MCP tool calls. / MCP 工具调用的处理函数。"""
        return await tool.execute(include_stats=include_stats)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
