"""Abstract base class for document loaders. / 文档加载器的抽象基类。

This module defines the pluggable interface for document loaders, / 此模块定义文档加载器的可插拔接口，
enabling seamless loading of different document formats (PDF, Markdown, etc.) / 支持无缝加载不同文档格式（PDF、Markdown 等），
with unified output structure. / 并提供统一输出结构。

Design Principles: / 设计原则：
- Single Responsibility: Loaders only handle format unification + structure extraction / 单一职责：加载器只处理格式统一 + 结构提取
- Type Safety: Return standardized Document type from core.types / 类型安全：返回 core.types 中标准化的 Document 类型
- No Splitting: Loaders don't chunk documents, only parse and normalize / 不做拆分：加载器不对文档分块，只解析并规范化
- Graceful Degradation: Failures in optional features (e.g., image extraction) shouldn't block text parsing / 优雅降级：可选功能失败（如图片提取）不应阻塞文本解析
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.types import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders. / 文档加载器的抽象基类。
    
    All loaders must implement the load() method to parse a file / 所有加载器都必须实现 load() 方法来解析文件，
    and return a standardized Document object with: / 并返回包含以下内容的标准化 Document 对象：
    - text: Normalized content (preferably Markdown format) / text：规范化内容（最好是 Markdown 格式）
    - metadata: At minimum must contain 'source_path' / metadata：至少必须包含 'source_path'
    
    Loaders should handle: / 加载器应处理：
    - Format-specific parsing logic / 格式特定解析逻辑
    - Metadata extraction (title, page count, etc.) / 元数据提取（标题、页数等）
    - Structure normalization (to Markdown when possible) / 结构规范化（尽可能转为 Markdown）
    - Optional: Image extraction and placeholder insertion / 可选：图片提取和占位符插入
    """
    
    @abstractmethod
    def load(self, file_path: str | Path) -> Document:
        """Load and parse a document file. / 加载并解析文档文件。
        
        Args: / 参数：
            file_path: Path to the document file to load. / 要加载的文档文件路径。
            
        Returns: / 返回：
            Document object with parsed content and metadata. / 包含解析后内容和元数据的 Document 对象。
            metadata MUST contain at least 'source_path'. / metadata 必须至少包含 'source_path'。
            
        Raises: / 异常：
            FileNotFoundError: If the file doesn't exist. / 如果文件不存在。
            ValueError: If the file format is invalid or unsupported. / 如果文件格式无效或不受支持。
            RuntimeError: If parsing fails critically. / 如果解析发生严重失败。
            
        Example: / 示例：
            >>> loader = PdfLoader()
            >>> doc = loader.load("data/documents/report.pdf")
            >>> assert "source_path" in doc.metadata
            >>> assert doc.text  # Non-empty text
        """
        pass
    
    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """Validate that file exists and is readable. / 校验文件存在且可读。
        
        Args: / 参数：
            file_path: Path to validate. / 要校验的路径。
            
        Returns: / 返回：
            Resolved Path object. / 解析后的 Path 对象。
            
        Raises: / 异常：
            FileNotFoundError: If file doesn't exist. / 如果文件不存在。
            PermissionError: If file is not readable. / 如果文件不可读。
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
