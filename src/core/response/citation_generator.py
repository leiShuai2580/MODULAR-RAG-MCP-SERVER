"""Citation Generator for generating structured reference information. / 用于生成结构化引用信息的引用生成器。

This module generates citation information from retrieval results, / 本模块根据检索结果生成引用信息，
enabling MCP tools to return properly formatted references that / 使 MCP 工具能够返回格式正确的引用，
can be used by AI assistants for source attribution. / 供 AI 助手进行来源归因。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult


@dataclass
class Citation:
    """Represents a single citation/reference. / 表示单条引用/参考信息。
    
    Attributes: / 属性：
        index: Citation index number (1-based, for display as [1], [2], etc.) / index：引用序号（从 1 开始，用于显示为 [1]、[2] 等）
        chunk_id: Unique identifier for the source chunk / chunk_id：来源分块的唯一标识
        source: Source file path or document name / source：来源文件路径或文档名称
        page: Page number in source document (if applicable) / page：来源文档中的页码（如适用）
        score: Relevance score from retrieval / score：检索得到的相关性分数
        text_snippet: Short excerpt from the referenced content / text_snippet：引用内容的简短摘录
        metadata: Additional metadata (title, section, etc.) / metadata：额外元数据（title、section 等）
    """
    index: int
    chunk_id: str
    source: str
    score: float
    text_snippet: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization. / 转换为字典以便 JSON 序列化。"""
        result = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "score": round(self.score, 4),
            "text_snippet": self.text_snippet,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class CitationGenerator:
    """Generates citation information from retrieval results. / 根据检索结果生成引用信息。
    
    This class transforms RetrievalResult objects into Citation objects / 该类将 RetrievalResult 对象转换为 Citation 对象，
    with proper indexing and metadata extraction. / 并处理正确的编号和元数据提取。
    
    Example: / 示例：
        >>> generator = CitationGenerator()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> citations = generator.generate(results)
        >>> print(citations[0].index)  # 1 / 1
        >>> print(citations[0].source)  # "docs/guide.pdf" / "docs/guide.pdf"
    """
    
    def __init__(
        self,
        snippet_max_length: int = 200,
        include_metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """Initialize CitationGenerator. / 初始化 CitationGenerator。
        
        Args: / 参数：
            snippet_max_length: Maximum characters for text_snippet (default: 200) / snippet_max_length：text_snippet 的最大字符数（默认值：200）
            include_metadata_fields: Optional list of metadata fields to include. / include_metadata_fields：可选的待包含元数据字段列表。
                If None, includes 'title', 'section', 'chunk_index'. / 如果为 None，则包含 'title'、'section'、'chunk_index'。
        """
        self.snippet_max_length = snippet_max_length
        self.include_metadata_fields = include_metadata_fields or [
            "title", "section", "chunk_index", "doc_type"
        ]
    
    def generate(self, results: List[RetrievalResult]) -> List[Citation]:
        """Generate citations from retrieval results. / 根据检索结果生成引用。
        
        Args: / 参数：
            results: List of RetrievalResult objects from search. / results：搜索返回的 RetrievalResult 对象列表。
            
        Returns: / 返回：
            List of Citation objects with 1-based indexing. / 从 1 开始编号的 Citation 对象列表。
        """
        citations = []
        
        for idx, result in enumerate(results, start=1):
            citation = self._create_citation(idx, result)
            citations.append(citation)
        
        return citations
    
    def _create_citation(self, index: int, result: RetrievalResult) -> Citation:
        """Create a Citation from a single RetrievalResult. / 从单个 RetrievalResult 创建 Citation。
        
        Args: / 参数：
            index: 1-based citation index. / index：从 1 开始的引用序号。
            result: RetrievalResult to convert. / result：要转换的 RetrievalResult。
            
        Returns: / 返回：
            Citation object with extracted information. / 包含提取信息的 Citation 对象。
        """
        metadata = result.metadata or {}
        
        # Extract source path / 提取来源路径
        source = metadata.get("source_path", "unknown")
        
        # Extract page number (may be int or string) / 提取页码（可能是 int 或 string）
        page = metadata.get("page") or metadata.get("page_num")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None
        
        # Generate text snippet / 生成文本片段
        text_snippet = self._generate_snippet(result.text)
        
        # Extract selected metadata fields / 提取选定的元数据字段
        extra_metadata = {}
        for field_name in self.include_metadata_fields:
            if field_name in metadata and field_name not in ("source_path", "page", "page_num"):
                extra_metadata[field_name] = metadata[field_name]
        
        return Citation(
            index=index,
            chunk_id=result.chunk_id,
            source=source,
            score=result.score,
            text_snippet=text_snippet,
            page=page,
            metadata=extra_metadata,
        )
    
    def _generate_snippet(self, text: str) -> str:
        """Generate a truncated snippet from text. / 从文本生成截断片段。
        
        Args: / 参数：
            text: Full text content. / text：完整文本内容。
            
        Returns: / 返回：
            Truncated text with ellipsis if needed. / 如有需要，返回带省略号的截断文本。
        """
        if not text:
            return ""
        
        # Clean up whitespace / 清理空白
        cleaned = " ".join(text.split())
        
        if len(cleaned) <= self.snippet_max_length:
            return cleaned
        
        # Truncate and add ellipsis / 截断并添加省略号
        truncated = cleaned[:self.snippet_max_length].rsplit(" ", 1)[0]
        return truncated + "..."
    
    def format_citation_marker(self, index: int) -> str:
        """Format a citation marker for inline use. / 格式化用于行内显示的引用标记。
        
        Args: / 参数：
            index: 1-based citation index. / index：从 1 开始的引用序号。
            
        Returns: / 返回：
            Formatted marker like "[1]". / 类似 "[1]" 的格式化标记。
        """
        return f"[{index}]"
