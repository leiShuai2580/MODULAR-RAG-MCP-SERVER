"""Recursive Splitter implementation using LangChain. / 使用 LangChain 的 Recursive Splitter 实现。

This module provides a recursive character-based text splitting strategy / 此模块提供一种基于递归字符的文本拆分策略，
that respects document structure (headers, code blocks) and splits text / 它尊重文档结构（标题、代码块）并以层级方式拆分文本，
hierarchically to maintain semantic coherence. / 以保持语义连贯性。
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Recursive character-based text splitter. / 基于递归字符的文本拆分器。
    
    This splitter uses LangChain's RecursiveCharacterTextSplitter to split text / 此拆分器使用 LangChain 的 RecursiveCharacterTextSplitter 拆分文本，
    by trying different separators in order (paragraphs, sentences, words) while / 按顺序尝试不同分隔符（段落、句子、词），同时
    respecting Markdown structure elements like headers and code blocks. / 尊重标题和代码块等 Markdown 结构元素。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Implements BaseSplitter interface for factory instantiation. / 可插拔：实现 BaseSplitter 接口以支持工厂实例化。
    - Config-Driven: Reads chunk_size and chunk_overlap from settings. / 配置驱动：从 settings 读取 chunk_size 和 chunk_overlap。
    - Fail-Fast: Raises ImportError if langchain-text-splitters is not installed. / 快速失败：如果未安装 langchain-text-splitters 则抛出 ImportError。
    - Graceful Degradation: Validates inputs and provides clear error messages. / 优雅降级：校验输入并提供清晰错误信息。
    
    Attributes: / 属性：
        chunk_size: Maximum size of each chunk in characters. / 每个块的最大字符数。
        chunk_overlap: Number of overlapping characters between chunks. / 块之间重叠的字符数。
        separators: List of separators to try in order (defaults to Markdown-aware). / 按顺序尝试的分隔符列表（默认感知 Markdown）。
        
    Raises: / 异常：
        ImportError: If langchain-text-splitters package is not installed. / 如果未安装 langchain-text-splitters 包。
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",  # Double newline (paragraphs) / 双换行（段落）
        "\n",    # Single newline / 单换行
        ". ",    # Sentence endings / 句子结尾
        "! ",
        "? ",
        "; ",
        ", ",
        " ",     # Spaces / 空格
        "",      # Characters / 字符
    ]
    
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RecursiveSplitter. / 初始化 RecursiveSplitter。
        
        Args: / 参数：
            settings: Application settings containing ingestion configuration. / 包含 ingestion 配置的应用设置。
            chunk_size: Optional override for chunk size (defaults to settings.ingestion.chunk_size). / 可选块大小覆盖值（默认使用 settings.ingestion.chunk_size）。
            chunk_overlap: Optional override for overlap (defaults to settings.ingestion.chunk_overlap). / 可选重叠大小覆盖值（默认使用 settings.ingestion.chunk_overlap）。
            separators: Optional list of separator strings (defaults to Markdown-aware separators). / 可选分隔符字符串列表（默认使用感知 Markdown 的分隔符）。
            **kwargs: Additional parameters passed to LangChain splitter. / 传给 LangChain 拆分器的额外参数。
        
        Raises: / 异常：
            ImportError: If langchain-text-splitters is not installed. / 如果未安装 langchain-text-splitters。
            ValueError: If chunk_size or chunk_overlap are invalid. / 如果 chunk_size 或 chunk_overlap 无效。
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )
        
        self.settings = settings
        
        # Extract configuration from settings with overrides / 从 settings 中提取配置，并支持覆盖值
        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e
        
        # Validate configuration / 校验配置
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        
        # Initialize LangChain splitter / 初始化 LangChain 拆分器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Split text into chunks recursively. / 递归地将文本拆分为块。
        
        This method splits text by trying different separators hierarchically, / 此方法通过层级尝试不同分隔符来拆分文本，
        preserving document structure like Markdown headers and code blocks. / 保留 Markdown 标题和代码块等文档结构。
        
        Args: / 参数：
            text: Input text to split. Must be a non-empty string. / 要拆分的输入文本，必须是非空字符串。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Additional parameters (currently unused, reserved for future extensions). / 额外参数（当前未使用，为未来扩展预留）。
        
        Returns: / 返回：
            A list of text chunks. Each chunk respects the configured chunk_size / 文本块列表。每个块遵守配置的 chunk_size
            and chunk_overlap. Order preserves the original text sequence. / 和 chunk_overlap。顺序保持原始文本顺序。
        
        Raises: / 异常：
            ValueError: If input text is invalid (empty, wrong type). / 如果输入文本无效（为空、类型错误）。
            RuntimeError: If splitting fails unexpectedly. / 如果拆分意外失败。
        
        Example: / 示例：
            >>> splitter = RecursiveSplitter(settings)
            >>> chunks = splitter.split_text("# Header\\n\\nParagraph 1.\\n\\nParagraph 2.")
            >>> len(chunks)
            1  # If text fits in chunk_size
        """
        # Validate input / 校验输入
        self.validate_text(text)
        
        try:
            # Perform splitting / 执行拆分
            chunks = self._splitter.split_text(text)
            
            # Handle edge case: LangChain may return empty list for very short text / 处理边界情况：LangChain 对很短文本可能返回空列表
            if not chunks:
                chunks = [text]
            
            # Validate output / 校验输出
            self.validate_chunks(chunks)
            
            return chunks
            
        except Exception as e:
            # Catch any LangChain errors and provide context / 捕获所有 LangChain 错误并提供上下文
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e
