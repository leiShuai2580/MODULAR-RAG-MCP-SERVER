"""Unit tests for ResponseBuilder and CitationGenerator. / ResponseBuilder 和 CitationGenerator 的单元测试。

This module tests the response building components used by MCP tools / 本模块测试 MCP 工具使用的响应构建组件，
to generate formatted output with citations. / 用于生成带 citations 的格式化输出。
"""

import pytest
from typing import Dict, Any, List

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.response_builder import ResponseBuilder, MCPToolResponse
from src.core.types import RetrievalResult


# =============================================================================
# Test Fixtures / 测试 Fixture
# =============================================================================

@pytest.fixture
def sample_retrieval_results() -> List[RetrievalResult]:
    """Create sample retrieval results for testing. / 创建测试用示例 retrieval results。"""
    return [
        RetrievalResult(
            chunk_id="doc1_chunk_001",
            score=0.95,
            text="Azure OpenAI 是微软提供的人工智能服务，提供 GPT-4 和 DALL-E 等模型的访问能力。它与 Azure 云服务深度集成，支持企业级安全和合规性需求。",
            metadata={
                "source_path": "docs/azure-guide.pdf",
                "page": 5,
                "chunk_index": 0,
                "title": "Azure OpenAI 简介",
                "doc_type": "pdf",
            },
        ),
        RetrievalResult(
            chunk_id="doc1_chunk_002",
            score=0.87,
            text="配置 Azure OpenAI 需要以下步骤：1. 创建 Azure 订阅；2. 在 Azure Portal 创建 OpenAI 资源；3. 获取 API 密钥和端点地址。",
            metadata={
                "source_path": "docs/azure-guide.pdf",
                "page": 10,
                "chunk_index": 5,
                "title": "配置步骤",
            },
        ),
        RetrievalResult(
            chunk_id="doc2_chunk_003",
            score=0.72,
            text="GPT-4 模型支持多轮对话、代码生成、文本分析等多种任务。在 Azure 平台上，您可以通过 REST API 或 SDK 进行调用。",
            metadata={
                "source_path": "docs/gpt4-usage.md",
                "chunk_index": 2,
                "title": "GPT-4 使用指南",
            },
        ),
    ]


@pytest.fixture
def empty_retrieval_results() -> List[RetrievalResult]:
    """Empty results list. / 空结果列表。"""
    return []


@pytest.fixture
def citation_generator() -> CitationGenerator:
    """Create CitationGenerator instance. / 创建 CitationGenerator 实例。"""
    return CitationGenerator()


@pytest.fixture
def response_builder() -> ResponseBuilder:
    """Create ResponseBuilder instance. / 创建 ResponseBuilder 实例。"""
    return ResponseBuilder()


# =============================================================================
# CitationGenerator Tests / CitationGenerator 测试
# =============================================================================

class TestCitationGenerator:
    """Tests for CitationGenerator class. / CitationGenerator 类测试。"""
    
    def test_generate_citations_basic(
        self,
        citation_generator: CitationGenerator,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test basic citation generation. / 测试基础 citation 生成。"""
        citations = citation_generator.generate(sample_retrieval_results)
        
        assert len(citations) == 3
        
        # Check first citation / 检查第一个 citation
        assert citations[0].index == 1
        assert citations[0].chunk_id == "doc1_chunk_001"
        assert citations[0].source == "docs/azure-guide.pdf"
        assert citations[0].score == 0.95
        assert citations[0].page == 5
        
        # Check indexing is 1-based / 检查索引从 1 开始
        assert citations[1].index == 2
        assert citations[2].index == 3
    
    def test_generate_citations_empty_results(
        self,
        citation_generator: CitationGenerator,
        empty_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test citation generation with empty results. / 测试空结果的 citation 生成。"""
        citations = citation_generator.generate(empty_retrieval_results)
        assert citations == []
    
    def test_citation_to_dict(
        self,
        citation_generator: CitationGenerator,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test Citation.to_dict() serialization. / 测试 Citation.to_dict() 序列化。"""
        citations = citation_generator.generate(sample_retrieval_results)
        
        citation_dict = citations[0].to_dict()
        
        assert citation_dict["index"] == 1
        assert citation_dict["chunk_id"] == "doc1_chunk_001"
        assert citation_dict["source"] == "docs/azure-guide.pdf"
        assert citation_dict["score"] == 0.95
        assert citation_dict["page"] == 5
        assert "text_snippet" in citation_dict
    
    def test_text_snippet_truncation(self) -> None:
        """Test that long text is truncated in snippets. / 测试长文本在 snippets 中被截断。"""
        generator = CitationGenerator(snippet_max_length=50)
        
        result = RetrievalResult(
            chunk_id="test_001",
            score=0.9,
            text="这是一段非常长的文本，用于测试文本截断功能是否正常工作。" * 10,
            metadata={"source_path": "test.pdf"},
        )
        
        citations = generator.generate([result])
        
        # Snippet should be truncated / snippet 应被截断
        assert len(citations[0].text_snippet) <= 60  # 50 + "..."
        assert citations[0].text_snippet.endswith("...")
    
    def test_metadata_extraction(
        self,
        citation_generator: CitationGenerator,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test that metadata fields are correctly extracted. / 测试 metadata 字段被正确提取。"""
        citations = citation_generator.generate(sample_retrieval_results)
        
        # First citation should have title in metadata / 第一个 citation 的 metadata 中应有 title
        assert "title" in citations[0].metadata
        assert citations[0].metadata["title"] == "Azure OpenAI 简介"
        
        # Check chunk_index is included / 检查包含 chunk_index
        assert "chunk_index" in citations[0].metadata
        assert citations[0].metadata["chunk_index"] == 0
    
    def test_page_number_extraction(self) -> None:
        """Test page number extraction from different metadata formats. / 测试从不同 metadata 格式提取页码。"""
        generator = CitationGenerator()
        
        # Test with 'page' key / 测试使用 'page' key
        result1 = RetrievalResult(
            chunk_id="test_001",
            score=0.9,
            text="Test content",
            metadata={"source_path": "test.pdf", "page": 5},
        )
        
        # Test with 'page_num' key / 测试使用 'page_num' key
        result2 = RetrievalResult(
            chunk_id="test_002",
            score=0.8,
            text="Test content",
            metadata={"source_path": "test.pdf", "page_num": 10},
        )
        
        # Test without page info / 测试无 page 信息
        result3 = RetrievalResult(
            chunk_id="test_003",
            score=0.7,
            text="Test content",
            metadata={"source_path": "test.md"},
        )
        
        citations = generator.generate([result1, result2, result3])
        
        assert citations[0].page == 5
        assert citations[1].page == 10
        assert citations[2].page is None
    
    def test_format_citation_marker(
        self,
        citation_generator: CitationGenerator,
    ) -> None:
        """Test citation marker formatting. / 测试 citation marker 格式化。"""
        assert citation_generator.format_citation_marker(1) == "[1]"
        assert citation_generator.format_citation_marker(10) == "[10]"
        assert citation_generator.format_citation_marker(99) == "[99]"


# =============================================================================
# ResponseBuilder Tests / ResponseBuilder 测试
# =============================================================================

class TestResponseBuilder:
    """Tests for ResponseBuilder class. / ResponseBuilder 类测试。"""
    
    def test_build_basic_response(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test basic response building. / 测试基础响应构建。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure OpenAI 配置",
            collection="docs",
        )
        
        assert isinstance(response, MCPToolResponse)
        assert response.is_empty is False
        assert len(response.citations) == 3
        assert response.metadata["query"] == "Azure OpenAI 配置"
        assert response.metadata["collection"] == "docs"
        assert response.metadata["result_count"] == 3
    
    def test_build_empty_response(
        self,
        response_builder: ResponseBuilder,
        empty_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test response building with empty results. / 测试空结果的响应构建。"""
        response = response_builder.build(
            results=empty_retrieval_results,
            query="不存在的查询",
            collection="docs",
        )
        
        assert response.is_empty is True
        assert len(response.citations) == 0
        assert "未找到相关结果" in response.content
        assert "建议" in response.content
    
    def test_content_contains_citation_markers(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test that content contains citation markers. / 测试 content 包含 citation markers。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure 配置",
        )
        
        # Should contain citation markers / 应包含 citation markers
        assert "[1]" in response.content
        assert "[2]" in response.content
        assert "[3]" in response.content
    
    def test_content_contains_source_info(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test that content contains source information. / 测试 content 包含 source 信息。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        # Should contain source paths / 应包含 source paths
        assert "azure-guide.pdf" in response.content
        assert "gpt4-usage.md" in response.content
    
    def test_content_contains_scores(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test that content contains relevance scores. / 测试 content 包含相关性分数。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        # Should contain score indicators / 应包含分数标识
        assert "相关度" in response.content or "95%" in response.content
    
    def test_response_to_dict(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test MCPToolResponse.to_dict() serialization. / 测试 MCPToolResponse.to_dict() 序列化。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        response_dict = response.to_dict()
        
        assert "content" in response_dict
        assert "structuredContent" in response_dict
        assert "citations" in response_dict["structuredContent"]
        assert len(response_dict["structuredContent"]["citations"]) == 3
    
    def test_response_to_mcp_content(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test MCPToolResponse.to_mcp_content() for MCP protocol. / 测试 MCP 协议下的 MCPToolResponse.to_mcp_content()。"""
        from mcp import types
        
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        content_blocks = response.to_mcp_content()
        
        assert len(content_blocks) >= 1
        # Now returns TextContent objects instead of dicts / 现在返回 TextContent 对象而不是 dict
        assert isinstance(content_blocks[0], types.TextContent)
        assert content_blocks[0].type == "text"
        assert "检索结果" in content_blocks[0].text
    
    def test_max_results_in_content(self) -> None:
        """Test that max_results_in_content is respected. / 测试遵守 max_results_in_content。"""
        builder = ResponseBuilder(max_results_in_content=2)
        
        results = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                score=0.9 - i * 0.1,
                text=f"Content {i}",
                metadata={"source_path": f"doc{i}.pdf"},
            )
            for i in range(5)
        ]
        
        response = builder.build(results=results, query="test")
        
        # Content should mention that more results exist / content 应提到还有更多结果
        assert "还有" in response.content or "未显示" in response.content
        
        # But all citations should still be included / 但所有 citations 仍应包含
        assert len(response.citations) == 5
    
    def test_collection_in_metadata(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test that collection is included in metadata when provided. / 测试提供 collection 时 metadata 包含 collection。"""
        # With collection / 带 collection
        response1 = response_builder.build(
            results=sample_retrieval_results,
            query="test",
            collection="my_collection",
        )
        assert response1.metadata["collection"] == "my_collection"
        
        # Without collection / 不带 collection
        response2 = response_builder.build(
            results=sample_retrieval_results,
            query="test",
        )
        assert "collection" not in response2.metadata
    
    def test_empty_response_with_collection(
        self,
        response_builder: ResponseBuilder,
    ) -> None:
        """Test empty response mentions collection name. / 测试空响应会提到 collection 名称。"""
        response = response_builder.build(
            results=[],
            query="test",
            collection="my_docs",
        )
        
        assert "my_docs" in response.content


# =============================================================================
# Citation Dataclass Tests / Citation Dataclass 测试
# =============================================================================

class TestCitationDataclass:
    """Tests for Citation dataclass. / Citation dataclass 测试。"""
    
    def test_citation_creation(self) -> None:
        """Test Citation object creation. / 测试 Citation 对象创建。"""
        citation = Citation(
            index=1,
            chunk_id="test_001",
            source="test.pdf",
            score=0.95,
            text_snippet="Test content...",
            page=5,
            metadata={"title": "Test"},
        )
        
        assert citation.index == 1
        assert citation.chunk_id == "test_001"
        assert citation.source == "test.pdf"
        assert citation.score == 0.95
        assert citation.page == 5
    
    def test_citation_to_dict_without_page(self) -> None:
        """Test Citation.to_dict() when page is None. / 测试 page 为 None 时的 Citation.to_dict()。"""
        citation = Citation(
            index=1,
            chunk_id="test_001",
            source="test.md",
            score=0.8,
            text_snippet="Test",
        )
        
        citation_dict = citation.to_dict()
        
        assert "page" not in citation_dict
    
    def test_citation_score_rounding(self) -> None:
        """Test that scores are rounded in to_dict(). / 测试 to_dict() 中分数会被四舍五入。"""
        citation = Citation(
            index=1,
            chunk_id="test_001",
            source="test.pdf",
            score=0.123456789,
            text_snippet="Test",
        )
        
        citation_dict = citation.to_dict()
        
        # Score should be rounded to 4 decimal places / 分数应四舍五入到 4 位小数
        assert citation_dict["score"] == 0.1235


# =============================================================================
# MCPToolResponse Dataclass Tests / MCPToolResponse Dataclass 测试
# =============================================================================

class TestMCPToolResponseDataclass:
    """Tests for MCPToolResponse dataclass. / MCPToolResponse dataclass 测试。"""
    
    def test_response_creation(self) -> None:
        """Test MCPToolResponse object creation. / 测试 MCPToolResponse 对象创建。"""
        response = MCPToolResponse(
            content="Test content",
            citations=[],
            metadata={"query": "test"},
            is_empty=False,
        )
        
        assert response.content == "Test content"
        assert response.citations == []
        assert response.metadata["query"] == "test"
        assert response.is_empty is False
    
    def test_response_default_values(self) -> None:
        """Test MCPToolResponse default values. / 测试 MCPToolResponse 默认值。"""
        response = MCPToolResponse(content="Test")
        
        assert response.citations == []
        assert response.metadata == {}
        assert response.is_empty is False


# =============================================================================
# Integration Tests / 集成测试
# =============================================================================

class TestResponseBuilderIntegration:
    """Integration tests for ResponseBuilder with CitationGenerator. / ResponseBuilder 与 CitationGenerator 的集成测试。"""
    
    def test_full_response_generation_flow(
        self,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test complete response generation flow. / 测试完整响应生成流程。"""
        builder = ResponseBuilder()
        
        response = builder.build(
            results=sample_retrieval_results,
            query="如何配置 Azure OpenAI？",
            collection="technical_docs",
        )
        
        # Verify response structure / 验证响应结构
        assert isinstance(response, MCPToolResponse)
        assert not response.is_empty
        
        # Verify content / 验证 content
        assert "检索结果" in response.content
        assert "Azure" in response.content
        
        # Verify citations / 验证 citations
        assert len(response.citations) == 3
        assert all(isinstance(c, Citation) for c in response.citations)
        
        # Verify metadata / 验证 metadata
        assert response.metadata["query"] == "如何配置 Azure OpenAI？"
        assert response.metadata["collection"] == "technical_docs"
        assert response.metadata["result_count"] == 3
        
        # Verify serialization / 验证序列化
        response_dict = response.to_dict()
        assert "content" in response_dict
        assert "structuredContent" in response_dict
    
    def test_custom_citation_generator(
        self,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """Test ResponseBuilder with custom CitationGenerator. / 测试带自定义 CitationGenerator 的 ResponseBuilder。"""
        custom_generator = CitationGenerator(
            snippet_max_length=50,
            include_metadata_fields=["title"],
        )
        
        builder = ResponseBuilder(citation_generator=custom_generator)
        response = builder.build(
            results=sample_retrieval_results,
            query="test",
        )
        
        # Snippets should be shorter / snippets 应更短
        for citation in response.citations:
            assert len(citation.text_snippet) <= 60  # 50 + "..."
        
        # Only 'title' should be in metadata / metadata 中应只有 'title'
        for citation in response.citations:
            if citation.metadata:
                assert "chunk_index" not in citation.metadata
