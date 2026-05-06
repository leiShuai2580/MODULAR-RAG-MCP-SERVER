"""MCP Protocol Handler for JSON-RPC 2.0 message handling. / 用于处理 JSON-RPC 2.0 消息的 MCP 协议处理器。

This module provides the ProtocolHandler class that encapsulates: / 该模块提供 ProtocolHandler 类，封装以下能力：
- Tool registration and schema management / 工具注册和 schema 管理
- JSON-RPC error code handling / JSON-RPC 错误码处理
- Capability negotiation during initialize / 初始化期间的能力协商
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mcp import types
from mcp.server.lowlevel import Server

from src.observability.logger import get_logger


# JSON-RPC 2.0 Error Codes / JSON-RPC 2.0 错误码
class JSONRPCErrorCodes:
    """Standard JSON-RPC 2.0 error codes. / 标准 JSON-RPC 2.0 错误码。"""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class ToolDefinition:
    """Definition of an MCP tool. / MCP 工具定义。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ProtocolHandler:
    """Handles MCP protocol operations including tool registration and execution. / 处理 MCP 协议操作，包括工具注册和执行。

    This class encapsulates: / 该类封装以下能力：
    - Tool registration with schema validation / 带 schema 校验的工具注册
    - Tool execution with error handling / 带错误处理的工具执行
    - Capability declaration for initialize response / 初始化响应的能力声明

    Attributes: / 属性：
        server_name: Name of the MCP server. / MCP 服务器名称。
        server_version: Version string of the server. / 服务器版本字符串。
        tools: Registry of available tools. / 可用工具注册表。
    """

    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize logger after dataclass initialization. / 在 dataclass 初始化后初始化日志器。"""
        self._logger = get_logger(log_level="INFO")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool with the protocol handler. / 将工具注册到协议处理器。

        Args: / 参数：
            name: Unique name for the tool. / 工具的唯一名称。
            description: Human-readable description of what the tool does. / 描述工具功能的可读说明。
            input_schema: JSON Schema for the tool's input parameters. / 工具输入参数的 JSON Schema。
            handler: Async function that executes the tool logic. / 执行工具逻辑的异步函数。

        Raises: / 抛出：
            ValueError: If a tool with the same name is already registered. / 如果同名工具已经注册。
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self._logger.info("Registered tool: %s", name)

    def get_tool_schemas(self) -> List[types.Tool]:
        """Get list of tool schemas for tools/list response. / 获取 tools/list 响应的工具 schema 列表。

        Returns: / 返回：
            List of Tool objects with name, description, and inputSchema. / 包含 name、description 和 inputSchema 的 Tool 对象列表。
        """
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in self.tools.values()
        ]

    async def execute_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Execute a registered tool by name. / 按名称执行已注册工具。

        Args: / 参数：
            name: Name of the tool to execute. / 要执行的工具名称。
            arguments: Arguments to pass to the tool handler. / 传给工具处理器的参数。

        Returns: / 返回：
            CallToolResult with content blocks or error indication. / 包含内容块或错误标记的 CallToolResult。

        Raises: / 抛出：
            ValueError: If tool is not found. / 如果未找到工具。
        """
        if name not in self.tools:
            self._logger.warning("Tool not found: %s", name)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Tool '{name}' not found",
                    )
                ],
                isError=True,
            )

        tool = self.tools[name]
        try:
            self._logger.info("Executing tool: %s", name)
            result = await tool.handler(**arguments)

            # Handle different return types / 处理不同的返回类型
            if isinstance(result, types.CallToolResult):
                return result
            if isinstance(result, str):
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=result)],
                    isError=False,
                )
            if isinstance(result, list):
                return types.CallToolResult(content=result, isError=False)
            # Default: convert to string / 默认转换为字符串
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
                isError=False,
            )

        except TypeError as e:
            # Invalid parameters / 无效参数
            self._logger.error("Invalid params for tool %s: %s", name, e)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Invalid parameters - {e}",
                    )
                ],
                isError=True,
            )
        except Exception as e:
            # Internal error - don't leak stack trace / 内部错误 - 不泄露堆栈跟踪
            self._logger.exception("Internal error executing tool %s", name)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Internal server error while executing '{name}'",
                    )
                ],
                isError=True,
            )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for initialize response. / 获取初始化响应的服务器能力。

        Returns: / 返回：
            Dictionary of server capabilities. / 服务器能力字典。
        """
        return {
            "tools": {} if self.tools else {},
        }


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    """Register all default MCP tools with the protocol handler. / 将所有默认 MCP 工具注册到协议处理器。

    Args: / 参数：
        protocol_handler: ProtocolHandler instance to register tools with. / 用于注册工具的 ProtocolHandler 实例。
    """
    # Import and register query_knowledge_hub tool / 导入并注册 query_knowledge_hub 工具
    from src.mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool
    register_query_tool(protocol_handler)
    
    # Import and register list_collections tool / 导入并注册 list_collections 工具
    from src.mcp_server.tools.list_collections import register_tool as register_list_tool
    register_list_tool(protocol_handler)
    
    # Import and register get_document_summary tool / 导入并注册 get_document_summary 工具
    from src.mcp_server.tools.get_document_summary import register_tool as register_summary_tool
    register_summary_tool(protocol_handler)


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
) -> Server:
    """Create and configure an MCP server with the protocol handler. / 创建并配置带协议处理器的 MCP 服务器。

    This factory function creates a low-level MCP Server instance and / 该工厂函数创建底层 MCP Server 实例，
    registers the necessary handlers for tools/list and tools/call. / 并注册 tools/list 和 tools/call 所需的处理器。

    Args: / 参数：
        server_name: Name of the server. / 服务器名称。
        server_version: Version string. / 版本字符串。
        protocol_handler: Optional pre-configured protocol handler. / 可选的预配置协议处理器。
            If None, a new one will be created. / 如果为 None，则创建新的实例。
        register_tools: Whether to register default tools (default: True). / 是否注册默认工具（默认 True）。

    Returns: / 返回：
        Configured Server instance ready to run. / 已配置且可运行的 Server 实例。
    """
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )

    # Register default tools if requested / 如果需要则注册默认工具
    if register_tools:
        _register_default_tools(protocol_handler)

    # Create low-level server / 创建底层服务器
    server = Server(server_name)

    # Register tools/list handler / 注册 tools/list 处理器
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """Handle tools/list request. / 处理 tools/list 请求。"""
        return protocol_handler.get_tool_schemas()

    # Register tools/call handler / 注册 tools/call 处理器
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Handle tools/call request. / 处理 tools/call 请求。"""
        return await protocol_handler.execute_tool(name, arguments)

    # Store protocol handler on server for access / 将协议处理器存到服务器对象上以便访问
    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]

    return server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    """Get the protocol handler from a server instance.

    Args:
        server: Server instance created by create_mcp_server.

    Returns:
        The ProtocolHandler associated with the server.

    Raises:
        AttributeError: If server was not created with create_mcp_server.
    """
    return server._protocol_handler  # type: ignore[attr-defined]
