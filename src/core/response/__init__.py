"""
Response Module. / 响应模块。

This package contains response building components: / 本包包含响应构建组件：
- Response builder / 响应构建器
- Citation generator / 引用生成器
- Multimodal assembler / 多模态组装器
"""

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.multimodal_assembler import (
    ImageContent,
    ImageReference,
    MultimodalAssembler,
)
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder

__all__ = [
    "Citation",
    "CitationGenerator",
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
