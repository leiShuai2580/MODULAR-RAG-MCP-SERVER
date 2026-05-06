"""MCP Server entry point using official MCP SDK. / 使用官方 MCP SDK 的 MCP 服务器入口。

This module implements the MCP server using the official Python MCP SDK / 该模块使用官方 Python MCP SDK 实现 MCP 服务器，
with stdio transport. It ensures stdout only contains protocol messages / 并使用标准输入输出传输。它确保 stdout 只包含协议消息，
while all logs go to stderr. / 所有日志都输出到 stderr。
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr() -> None:
    """Redirect all root logger handlers to stderr. / 将所有根日志处理器重定向到 stderr。

    MCP stdio transport reserves stdout for JSON-RPC messages. / MCP stdio 传输将 stdout 保留给 JSON-RPC 消息。
    Any logging to stdout corrupts the protocol stream. / 任何输出到 stdout 的日志都会破坏协议流。
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # Replace any existing stream handlers that might point to stdout / 替换任何可能指向 stdout 的现有流处理器
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


def _preload_heavy_imports() -> None:
    """Eagerly import heavy third-party modules in the **main thread**. / 在**主线程**中提前导入较重的第三方模块。

    MCP SDK uses anyio + background threads for stdin/stdout I/O. / MCP SDK 使用 anyio 和后台线程处理 stdin/stdout I/O。
    When a tool handler runs ``asyncio.to_thread(fn)``, *fn* executes in / 当工具处理器运行 ``asyncio.to_thread(fn)`` 时，*fn* 会在
    a new worker thread.  If it tries to ``import chromadb`` (which / 新的工作线程中执行。如果它尝试 ``import chromadb``（该导入会
    transitively pulls in onnxruntime, numpy, sqlite3 C extensions …), / 间接拉入 onnxruntime、numpy、sqlite3 C 扩展等），
    that import can deadlock with the stdin-reader thread because both / 该导入可能与 stdin 读取线程发生死锁，因为二者都会
    compete for Python's global *import lock*. / 竞争 Python 的全局 *import lock*。

    Pre-importing here – before anyio spins up its I/O threads – avoids / 在 anyio 启动 I/O 线程之前在这里预导入，可以完全避免
    the deadlock entirely: subsequent ``import`` statements in worker / 这个死锁：后续工作线程中的 ``import`` 语句只会命中
    threads simply hit ``sys.modules`` and return immediately. / ``sys.modules`` 并立即返回。
    """
    # chromadb is the heaviest culprit (onnxruntime, numpy, …) / chromadb 是最重的触发源（onnxruntime、numpy 等）
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401
    except ImportError:
        pass  # optional at install time / 安装时可选

    # Internal modules that tools lazy-import inside asyncio.to_thread / 工具在 asyncio.to_thread 内部延迟导入的内部模块
    try:
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.dense_retriever  # noqa: F401
        import src.core.query_engine.sparse_retriever  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
        import src.ingestion.storage.bm25_indexer  # noqa: F401
        import src.libs.embedding.embedding_factory  # noqa: F401
        import src.libs.vector_store.vector_store_factory  # noqa: F401
    except ImportError:
        pass


async def run_stdio_server_async() -> int:
    """Run MCP server over stdio asynchronously. / 通过 stdio 异步运行 MCP 服务器。

    Returns: / 返回：
        Exit code. / 退出码。
    """
    # Import here to avoid import errors if mcp not installed / 在这里导入以避免 mcp 未安装时的导入错误
    import mcp.server.stdio

    # Ensure ALL logging goes to stderr (stdout is reserved for JSON-RPC) / 确保所有日志输出到 stderr（stdout 保留给 JSON-RPC）
    _redirect_all_loggers_to_stderr()

    # Pre-load heavy deps in main thread to prevent import-lock deadlocks / 在主线程中预加载重依赖，避免导入锁死锁
    # when tool handlers later call asyncio.to_thread(). / 当工具处理器之后调用 asyncio.to_thread() 时。
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    # Create server with protocol handler / 创建带协议处理器的服务器
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    # Run with stdio transport / 使用 stdio 传输运行
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """Run MCP server over stdio (synchronous wrapper). / 通过 stdio 运行 MCP 服务器（同步包装器）。

    Returns: / 返回：
        Exit code. / 退出码。
    """
    return asyncio.run(run_stdio_server_async())


def main() -> int:
    """Entry point for stdio MCP server. / stdio MCP 服务器入口点。"""
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
