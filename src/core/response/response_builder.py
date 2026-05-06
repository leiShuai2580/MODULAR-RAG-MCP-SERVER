"""Response Builder for constructing MCP-formatted responses. / 用于构建 MCP 格式响应的响应构建器。

This module builds structured responses for MCP tools, combining: / 本模块为 MCP 工具构建结构化响应，组合：
- Human-readable Markdown content with citation markers / 带引用标记的人类可读 Markdown 内容
- Structured citation data for machine consumption / 供机器消费的结构化引用数据
- Multimodal content (text + images) support / 多模态内容（文本 + 图片）支持
- Proper handling of empty results and error cases / 对空结果和错误情况的正确处理
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.types import RetrievalResult


@dataclass
class MCPToolResponse:
    """Structured response for MCP tools. / MCP 工具的结构化响应。
    
    Attributes: / 属性：
        content: Human-readable Markdown content with citation markers [1], [2], etc. / content：带引用标记 [1]、[2] 等的人类可读 Markdown 内容。
        citations: List of structured citations for reference / citations：用于参考的结构化引用列表
        metadata: Additional response metadata (query, result_count, etc.) / metadata：额外响应元数据（query、result_count 等）
        is_empty: Whether the search returned no results / is_empty：搜索是否未返回结果
        image_contents: List of MCP ImageContent blocks for multimodal responses / image_contents：多模态响应中的 MCP ImageContent 块列表
    """
    content: str
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_empty: bool = False
    image_contents: List[types.ImageContent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP protocol. / 转换为 MCP 协议使用的字典。
        
        Returns: / 返回：
            Dictionary with 'content' and 'structuredContent' fields. / 包含 'content' 和 'structuredContent' 字段的字典。
        """
        return {
            "content": self.content,
            "structuredContent": {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "isEmpty": self.is_empty,
            }
        }
    
    def to_mcp_content(self) -> List[Union[types.TextContent, types.ImageContent]]:
        """Convert to MCP content blocks format. / 转换为 MCP 内容块格式。
        
        Returns: / 返回：
            List of content blocks for MCP CallToolResult. / MCP CallToolResult 的内容块列表。
            Includes TextContent and optionally ImageContent blocks. / 包含 TextContent，并可选包含 ImageContent 块。
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = [
            types.TextContent(
                type="text",
                text=self.content,
            )
        ]
        
        # Add image blocks if present (multimodal response) / 如果存在图片块则添加（多模态响应）
        if self.image_contents:
            blocks.extend(self.image_contents)
        
        # Add structured data as a separate text block (JSON format) / 将结构化数据作为独立文本块添加（JSON 格式）
        if self.citations or self.metadata:
            import json
            structured = {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "has_images": len(self.image_contents) > 0,
                "image_count": len(self.image_contents),
            }
            blocks.append(
                types.TextContent(
                    type="text",
                    text=f"\n---\n**References (JSON):**\n```json\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n```",
                )
            )
        
        return blocks
    
    @property
    def has_images(self) -> bool:
        """Check if response contains images. / 检查响应是否包含图片。
        
        Returns: / 返回：
            True if response has image content, False otherwise. / 如果响应包含图片内容则返回 True，否则返回 False。
        """
        return len(self.image_contents) > 0


class ResponseBuilder:
    """Builds MCP-formatted responses from retrieval results. / 根据检索结果构建 MCP 格式响应。
    
    This class transforms retrieval results into structured MCP responses, / 该类将检索结果转换为结构化 MCP 响应，
    including human-readable Markdown with inline citations and structured / 包括带行内引用的人类可读 Markdown，以及
    citation data for machine consumption. / 供机器消费的结构化引用数据。
    
    Supports multimodal responses with images when results contain image / 当结果元数据中包含图片引用时，
    references in their metadata. / 支持带图片的多模态响应。
    
    Example: / 示例：
        >>> builder = ResponseBuilder()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> response = builder.build(results, "What is Azure OpenAI?")
        >>> print(response.content)  # Markdown with [1], [2] markers / 带 [1]、[2] 标记的 Markdown
        >>> print(response.citations[0].source)  # "docs/guide.pdf" / "docs/guide.pdf"
        >>> print(response.has_images)  # True if images found / 如果找到图片则为 True
    """
    
    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional["MultimodalAssembler"] = None,
        max_results_in_content: int = 5,
        snippet_max_length: int = 300,
        enable_multimodal: bool = True,
    ) -> None:
        """Initialize ResponseBuilder. / 初始化 ResponseBuilder。
        
        Args: / 参数：
            citation_generator: Optional CitationGenerator instance. / citation_generator：可选 CitationGenerator 实例。
                If None, creates a default one. / 如果为 None，则创建默认实例。
            multimodal_assembler: Optional MultimodalAssembler for image handling. / multimodal_assembler：用于图片处理的可选 MultimodalAssembler。
                If None and enable_multimodal=True, creates a default one. / 如果为 None 且 enable_multimodal=True，则创建默认实例。
            max_results_in_content: Maximum results to show in Markdown content. / max_results_in_content：Markdown 内容中最多显示的结果数量。
            snippet_max_length: Maximum characters per result snippet in content. / snippet_max_length：内容中每条结果片段的最大字符数。
            enable_multimodal: Whether to include images in response (default: True). / enable_multimodal：是否在响应中包含图片（默认值：True）。
        """
        self.citation_generator = citation_generator or CitationGenerator()
        self.max_results_in_content = max_results_in_content
        self.snippet_max_length = snippet_max_length
        self.enable_multimodal = enable_multimodal
        
        # Lazy-load multimodal assembler to avoid circular imports / 延迟加载多模态组装器以避免循环导入
        self._multimodal_assembler = multimodal_assembler
    
    @property
    def multimodal_assembler(self) -> "MultimodalAssembler":
        """Get or create MultimodalAssembler instance. / 获取或创建 MultimodalAssembler 实例。"""
        if self._multimodal_assembler is None:
            from src.core.response.multimodal_assembler import MultimodalAssembler
            self._multimodal_assembler = MultimodalAssembler()
        return self._multimodal_assembler
    
    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        collection: Optional[str] = None,
        include_images: bool = True,
    ) -> MCPToolResponse:
        """Build MCP response from retrieval results. / 根据检索结果构建 MCP 响应。
        
        Args: / 参数：
            results: List of RetrievalResult from search. / results：搜索得到的 RetrievalResult 列表。
            query: Original user query. / query：原始用户查询。
            collection: Optional collection name. / collection：可选集合名称。
            include_images: Whether to include images in response (default: True). / include_images：是否在响应中包含图片（默认值：True）。
            
        Returns: / 返回：
            MCPToolResponse with formatted content, citations, and optional images. / 包含格式化内容、引用和可选图片的 MCPToolResponse。
        """
        # Handle empty results / 处理空结果
        if not results:
            return self._build_empty_response(query, collection)
        
        # Generate citations / 生成引用
        citations = self.citation_generator.generate(results)
        
        # Build Markdown content / 构建 Markdown 内容
        content = self._build_markdown_content(results, citations, query)
        
        # Build metadata / 构建元数据
        metadata = self._build_metadata(query, collection, len(results))
        
        # Assemble image content if enabled / 如果启用，则组装图片内容
        image_contents: List[types.ImageContent] = []
        if self.enable_multimodal and include_images:
            image_blocks = self.multimodal_assembler.assemble(results, collection)
            # Filter to only ImageContent blocks / 只过滤出 ImageContent 块
            image_contents = [
                block for block in image_blocks
                if isinstance(block, types.ImageContent)
            ]
            if image_contents:
                metadata["has_images"] = True
                metadata["image_count"] = len(image_contents)
        
        return MCPToolResponse(
            content=content,
            citations=citations,
            metadata=metadata,
            is_empty=False,
            image_contents=image_contents,
        )
    
    def _build_empty_response(
        self,
        query: str,
        collection: Optional[str] = None,
    ) -> MCPToolResponse:
        """Build response for empty results. / 为无结果场景构建响应。
        
        Args: / 参数：
            query: Original user query. / query：原始用户查询。
            collection: Optional collection name. / collection：可选集合名称。
            
        Returns: / 返回：
            MCPToolResponse indicating no results found. / 表示未找到结果的 MCPToolResponse。
        """
        content = f"## 未找到相关结果\n\n"
        content += f"查询: **{query}**\n\n"
        
        if collection:
            content += f"在集合 `{collection}` 中未找到与查询相关的文档。\n\n"
        else:
            content += "未找到与查询相关的文档。\n\n"
        
        content += "**建议:**\n"
        content += "- 尝试使用不同的关键词\n"
        content += "- 检查是否已摄取相关文档\n"
        content += "- 扩大搜索范围（如不指定 collection）\n"
        
        metadata = self._build_metadata(query, collection, 0)
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata=metadata,
            is_empty=True,
        )
    
    def _build_markdown_content(
        self,
        results: List[RetrievalResult],
        citations: List[Citation],
        query: str,
    ) -> str:
        """Build Markdown content with inline citations. / 构建带行内引用的 Markdown 内容。
        
        Args: / 参数：
            results: List of RetrievalResult. / results：RetrievalResult 列表。
            citations: List of Citation objects. / citations：Citation 对象列表。
            query: Original query string. / query：原始查询字符串。
            
        Returns: / 返回：
            Formatted Markdown string. / 格式化后的 Markdown 字符串。
        """
        lines = []
        
        # Header / 标题
        lines.append(f"## 检索结果\n")
        lines.append(f"针对查询 **\"{query}\"** 找到 {len(results)} 条相关结果:\n")
        
        # Results section / 结果部分
        display_count = min(len(results), self.max_results_in_content)
        
        for i, (result, citation) in enumerate(zip(results[:display_count], citations[:display_count])):
            marker = self.citation_generator.format_citation_marker(citation.index)
            
            # Format single result / 格式化单条结果
            lines.append(f"### {marker} 结果 {citation.index}")
            lines.append(f"**相关度:** {citation.score:.2%}")
            lines.append(f"**来源:** `{citation.source}`")
            
            if citation.page is not None:
                lines.append(f"**页码:** {citation.page}")
            
            # Content snippet / 内容片段
            snippet = self._truncate_text(result.text, self.snippet_max_length)
            lines.append(f"\n> {snippet}\n")
        
        # Additional results indicator / 额外结果提示
        if len(results) > display_count:
            remaining = len(results) - display_count
            lines.append(f"\n*...还有 {remaining} 条结果未显示*\n")
        
        # References section / 引用来源部分
        lines.append("\n---\n")
        lines.append("## 引用来源\n")
        
        for citation in citations:
            source_info = f"`{citation.source}`"
            if citation.page is not None:
                source_info += f" (p.{citation.page})"
            lines.append(f"- [{citation.index}] {source_info}")
        
        return "\n".join(lines)
    
    def _build_metadata(
        self,
        query: str,
        collection: Optional[str],
        result_count: int,
    ) -> Dict[str, Any]:
        """Build response metadata. / 构建响应元数据。
        
        Args: / 参数：
            query: Original query. / query：原始查询。
            collection: Collection name. / collection：集合名称。
            result_count: Number of results. / result_count：结果数量。
            
        Returns: / 返回：
            Metadata dictionary. / 元数据字典。
        """
        metadata = {
            "query": query,
            "result_count": result_count,
        }
        if collection:
            metadata["collection"] = collection
        return metadata
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length. / 将文本截断到最大长度。
        
        Args: / 参数：
            text: Text to truncate. / text：要截断的文本。
            max_length: Maximum characters. / max_length：最大字符数。
            
        Returns: / 返回：
            Truncated text with ellipsis if needed. / 如有需要，返回带省略号的截断文本。
        """
        if not text:
            return ""
        
        # Clean whitespace / 清理空白
        cleaned = " ".join(text.split())
        
        if len(cleaned) <= max_length:
            return cleaned
        
        # Truncate at word boundary / 在单词边界处截断
        truncated = cleaned[:max_length].rsplit(" ", 1)[0]
        return truncated + "..."
