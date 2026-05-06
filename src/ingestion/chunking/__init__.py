"""Chunking module - document splitting adapter layer. / 分块模块 - 文档拆分适配层。

This module provides the business adapter for text splitting, transforming / 此模块为文本拆分提供业务适配器，
Document objects into Chunk objects with proper metadata and traceability. / 将 Document 对象转换为带有适当元数据和可追溯性的 Chunk 对象。
"""

from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
