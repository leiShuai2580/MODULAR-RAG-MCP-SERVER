"""Contract tests for MetadataEnricher transform. / MetadataEnricher transform 的契约测试。"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.core.types import Chunk
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import Message


# ============================================================================
# Test Fixtures / 测试 Fixtures
# ============================================================================

@pytest.fixture
def mock_settings_llm_disabled():
    """Settings with LLM disabled. / 禁用 LLM 的 settings。"""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {'use_llm': False}
    return settings


@pytest.fixture
def mock_settings_llm_enabled():
    """Settings with LLM enabled. / 启用 LLM 的 settings。"""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {'use_llm': True}
    settings.llm = Mock()
    settings.llm.provider = 'openai'
    return settings


@pytest.fixture
def sample_chunk_simple():
    """Simple text chunk. / 简单文本 chunk。"""
    return Chunk(
        id="chunk_001",
        text="This is a test document about Python programming. It covers basic concepts.",
        metadata={"source_path": "test.pdf"},
        source_ref="test.pdf#page1"
    )


@pytest.fixture
def sample_chunk_with_heading():
    """Chunk with markdown heading. / 带 markdown heading 的 chunk。"""
    return Chunk(
        id="chunk_002",
        text="# Introduction to Machine Learning\n\nMachine learning is a subset of artificial intelligence. It enables systems to learn from data.",
        metadata={"source_path": "ml.pdf"},
        source_ref="ml.pdf#page1"
    )


@pytest.fixture
def sample_chunk_with_code():
    """Chunk with code identifiers. / 带代码 identifiers 的 chunk。"""
    return Chunk(
        id="chunk_003",
        text="The getUserById function retrieves user data. It uses async_fetch and handles errors gracefully.",
        metadata={"source_path": "code.md"},
        source_ref="code.md#section2"
    )


@pytest.fixture
def temp_prompt_file(tmp_path):
    """Create temporary prompt file. / 创建临时 prompt 文件。"""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text(
        "Analyze this text:\n{chunk_text}\n\n"
        "Title: <title>\nSummary: <summary>\nTags: <tags>"
    )
    return str(prompt_file)


# ============================================================================
# Rule-Based Enrichment Tests / 基于规则的 enrichment 测试
# ============================================================================

class TestRuleBasedEnrichment:
    """Test rule-based metadata extraction. / 测试基于规则的 metadata 提取。"""
    
    def test_extract_title_from_heading(self, mock_settings_llm_disabled, sample_chunk_with_heading):
        """Should extract title from markdown heading. / 应从 markdown heading 提取 title。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_with_heading])
        
        assert len(result) == 1
        assert result[0].metadata['title'] == "Introduction to Machine Learning"
        assert result[0].metadata['enriched_by'] == "rule"
    
    def test_extract_title_from_first_line(self, mock_settings_llm_disabled):
        """Should use first line as title if short and appropriate. / 如果第一行较短且合适，应将其作为 title。"""
        chunk = Chunk(
            id="chunk_004",
            text="Quick Start Guide\n\nFollow these steps to get started.",
            metadata={"source_path": "guide.md"},
            source_ref="guide.md"
        )
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([chunk])
        
        assert result[0].metadata['title'] == "Quick Start Guide"
    
    def test_extract_title_from_sentence(self, mock_settings_llm_disabled, sample_chunk_simple):
        """Should extract first sentence as title. / 应提取第一句作为 title。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_simple])
        
        # Title should be first sentence without trailing period / title 应为不带句尾句号的第一句
        assert result[0].metadata['title'] == "This is a test document about Python programming"
    
    def test_extract_summary_from_text(self, mock_settings_llm_disabled, sample_chunk_with_heading):
        """Should generate summary from first few sentences. / 应从前几句生成 summary。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_with_heading])
        
        assert result[0].metadata['summary']
        assert len(result[0].metadata['summary']) > 0
        assert "Machine learning" in result[0].metadata['summary']
    
    def test_extract_tags_from_capitalized_words(self, mock_settings_llm_disabled, sample_chunk_with_heading):
        """Should extract capitalized words as tags. / 应提取大写开头单词作为 tags。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_with_heading])
        
        assert 'tags' in result[0].metadata
        tags = result[0].metadata['tags']
        assert isinstance(tags, list)
        # Should find "Machine" or "Learning" or "Introduction" / 应找到 "Machine"、"Learning" 或 "Introduction"
        assert any(tag in ['Machine', 'Learning', 'Introduction'] for tag in tags)
    
    def test_extract_tags_from_code_identifiers(self, mock_settings_llm_disabled, sample_chunk_with_code):
        """Should extract code identifiers as tags. / 应提取代码 identifiers 作为 tags。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_with_code])
        
        tags = result[0].metadata['tags']
        # Should find async_fetch / 应找到 async_fetch
        assert 'async_fetch' in tags or 'getUserById' in tags
    
    def test_metadata_preserved(self, mock_settings_llm_disabled, sample_chunk_simple):
        """Should preserve existing metadata. / 应保留已有 metadata。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_simple])
        
        assert result[0].metadata['source_path'] == "test.pdf"
        assert 'title' in result[0].metadata
        assert 'summary' in result[0].metadata
        assert 'tags' in result[0].metadata
    
    def test_empty_text_handling(self, mock_settings_llm_disabled):
        """Should handle empty text gracefully. / 应优雅处理空文本。"""
        chunk = Chunk(
            id="chunk_empty",
            text="",
            metadata={"source_path": "empty.txt"},
            source_ref="empty.txt"
        )
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([chunk])
        
        assert len(result) == 1
        assert result[0].metadata['title'] == "Untitled"
        assert result[0].metadata['summary'] == ""
        assert result[0].metadata['tags'] == []


# ============================================================================
# Transform Pipeline Tests / Transform Pipeline 测试
# ============================================================================

class TestTransformPipeline:
    """Test the complete transform pipeline. / 测试完整 transform pipeline。"""
    
    def test_transform_single_chunk(self, mock_settings_llm_disabled, sample_chunk_simple):
        """Should enrich a single chunk. / 应 enrich 单个 chunk。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([sample_chunk_simple])
        
        assert len(result) == 1
        assert result[0].id == sample_chunk_simple.id
        assert result[0].text == sample_chunk_simple.text  # Text unchanged / 文本不变
        assert 'title' in result[0].metadata
        assert 'summary' in result[0].metadata
        assert 'tags' in result[0].metadata
        assert result[0].metadata['enriched_by'] == "rule"
    
    def test_transform_multiple_chunks(self, mock_settings_llm_disabled):
        """Should enrich multiple chunks independently. / 应独立 enrich 多个 chunks。"""
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Content {i}", metadata={"source_path": f"doc{i}.txt"}, source_ref=f"doc{i}.txt")
            for i in range(5)
        ]
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform(chunks)
        
        assert len(result) == 5
        for i, chunk in enumerate(result):
            assert chunk.id == f"chunk_{i}"
            assert 'title' in chunk.metadata
            assert 'summary' in chunk.metadata
            assert 'tags' in chunk.metadata
    
    def test_transform_empty_list(self, mock_settings_llm_disabled):
        """Should handle empty chunk list. / 应处理空 chunk 列表。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        result = enricher.transform([])
        
        assert result == []
    
    def test_trace_recording(self, mock_settings_llm_disabled, sample_chunk_simple):
        """Should record processing info in trace context. / 应在 trace context 中记录处理信息。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        trace = TraceContext(trace_id="test_trace")
        
        enricher.transform([sample_chunk_simple], trace=trace)
        
        # Check trace was recorded / 检查 trace 已记录
        assert len(trace.stages) > 0
        stage_data = trace.get_stage_data('metadata_enricher')
        assert stage_data is not None
        assert stage_data['total_chunks'] == 1
        assert stage_data['success_count'] == 1


# ============================================================================
# LLM Enhancement Tests / LLM Enhancement 测试
# ============================================================================

class TestLLMEnhancement:
    """Test LLM-based metadata enrichment. / 测试基于 LLM 的 metadata enrichment。"""
    
    def test_llm_success_path(self, mock_settings_llm_enabled, sample_chunk_simple, temp_prompt_file):
        """Should use LLM when enabled and successful. / 启用且成功时应使用 LLM。"""
        mock_llm = Mock()
        mock_llm.chat.return_value = (
            "Title: Python Programming Guide\n"
            "Summary: A comprehensive guide to Python programming covering basic concepts and syntax.\n"
            "Tags: Python, programming, basics, tutorial"
        )
        
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            llm=mock_llm,
            prompt_path=temp_prompt_file
        )
        
        result = enricher.transform([sample_chunk_simple])
        
        assert len(result) == 1
        assert result[0].metadata['title'] == "Python Programming Guide"
        assert "comprehensive guide" in result[0].metadata['summary']
        assert "Python" in result[0].metadata['tags']
        assert result[0].metadata['enriched_by'] == "llm"
        mock_llm.chat.assert_called_once()
    
    def test_llm_fallback_on_failure(self, mock_settings_llm_enabled, sample_chunk_simple, temp_prompt_file):
        """Should fallback to rule-based on LLM failure. / LLM 失败时应回退到基于规则。"""
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("API Error")
        
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            llm=mock_llm,
            prompt_path=temp_prompt_file
        )
        
        result = enricher.transform([sample_chunk_simple])
        
        assert len(result) == 1
        assert result[0].metadata['enriched_by'] == "rule"
        assert 'enrich_fallback_reason' in result[0].metadata
        assert result[0].metadata['enrich_fallback_reason'] == "llm_failed"
        # Should still have valid metadata from rule-based / 仍应拥有来自规则方式的有效 metadata
        assert result[0].metadata['title']
        assert result[0].metadata['summary']
    
    def test_llm_fallback_on_empty_response(self, mock_settings_llm_enabled, sample_chunk_simple, temp_prompt_file):
        """Should fallback to rule-based when LLM returns empty response. / LLM 返回空响应时应回退到基于规则。"""
        mock_llm = Mock()
        mock_llm.chat.return_value = ""
        
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            llm=mock_llm,
            prompt_path=temp_prompt_file
        )
        
        result = enricher.transform([sample_chunk_simple])
        
        assert result[0].metadata['enriched_by'] == "rule"
        assert 'enrich_fallback_reason' in result[0].metadata
    
    def test_llm_trace_recording(self, mock_settings_llm_enabled, sample_chunk_simple, temp_prompt_file):
        """Should record LLM calls in trace context. / 应在 trace context 中记录 LLM 调用。"""
        mock_llm = Mock()
        mock_llm.chat.return_value = (
            "Title: Test\nSummary: Test summary\nTags: test"
        )
        
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            llm=mock_llm,
            prompt_path=temp_prompt_file
        )
        trace = TraceContext(trace_id="test_llm_trace")
        
        enricher.transform([sample_chunk_simple], trace=trace)
        
        # Should have both llm_enrich and metadata_enricher stages / 应同时有 llm_enrich 和 metadata_enricher stages
        assert trace.get_stage_data('llm_enrich') is not None
        assert trace.get_stage_data('metadata_enricher') is not None


# ============================================================================
# Prompt Loading Tests / Prompt 加载测试
# ============================================================================

class TestPromptLoading:
    """Test prompt template loading. / 测试 prompt template 加载。"""
    
    def test_load_existing_prompt(self, mock_settings_llm_enabled, temp_prompt_file):
        """Should load prompt from file. / 应从文件加载 prompt。"""
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            prompt_path=temp_prompt_file
        )
        
        prompt = enricher._load_prompt()
        
        assert prompt
        assert "{chunk_text}" in prompt
    
    def test_prompt_cached_after_first_load(self, mock_settings_llm_enabled, temp_prompt_file):
        """Should cache prompt after first load. / 首次加载后应缓存 prompt。"""
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            prompt_path=temp_prompt_file
        )
        
        prompt1 = enricher._load_prompt()
        prompt2 = enricher._load_prompt()
        
        assert prompt1 is prompt2  # Same object (cached) / 同一个对象（已缓存）
    
    def test_missing_prompt_file(self, mock_settings_llm_enabled):
        """Should raise FileNotFoundError for missing prompt. / prompt 缺失时应抛出 FileNotFoundError。"""
        enricher = MetadataEnricher(
            mock_settings_llm_enabled,
            prompt_path="/nonexistent/prompt.txt"
        )
        
        with pytest.raises(FileNotFoundError):
            enricher._load_prompt()


# ============================================================================
# Atomic Processing Tests / 原子处理测试
# ============================================================================

class TestAtomicProcessing:
    """Test that failures in one chunk don't affect others. / 测试单个 chunk 失败不会影响其他 chunks。"""
    
    def test_exception_in_one_chunk_preserves_minimal_metadata(self, mock_settings_llm_disabled):
        """Should provide minimal metadata for failed chunks. / 应为失败 chunks 提供最小 metadata。"""
        chunks = [
            Chunk(id="chunk_good", text="Valid content", metadata={"source_path": "good.txt"}, source_ref="good.txt"),
            Chunk(id="chunk_bad", text=None, metadata={"source_path": "bad.txt"}, source_ref="bad.txt"),  # Will cause error / 会导致错误
            Chunk(id="chunk_good2", text="More valid content", metadata={"source_path": "good2.txt"}, source_ref="good2.txt"),
        ]
        
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        # Should not raise exception / 不应抛出异常
        result = enricher.transform(chunks)
        
        assert len(result) == 3
        # First and third should be enriched normally / 第一个和第三个应正常 enrich
        assert result[0].metadata['enriched_by'] == "rule"
        assert result[2].metadata['enriched_by'] == "rule"
        # Second should have error metadata / 第二个应有 error metadata
        assert result[1].metadata['enriched_by'] == "error"
        assert 'enrich_error' in result[1].metadata
        assert result[1].metadata['title'] == 'Untitled'


# ============================================================================
# Configuration Tests / 配置测试
# ============================================================================

class TestConfiguration:
    """Test configuration handling. / 测试配置处理。"""
    
    def test_use_llm_disabled_by_default(self):
        """Should default to rule-based when LLM not configured. / 未配置 LLM 时应默认使用基于规则。"""
        settings = Mock(spec=Settings)
        settings.ingestion = Mock()
        settings.ingestion.metadata_enricher = {}
        
        enricher = MetadataEnricher(settings)
        
        assert enricher.use_llm is False
    
    def test_lazy_llm_initialization(self, mock_settings_llm_enabled):
        """Should initialize LLM lazily when first accessed. / 首次访问时应延迟初始化 LLM。"""
        with patch('src.ingestion.transform.metadata_enricher.LLMFactory.create') as mock_create:
            mock_llm = Mock()
            mock_create.return_value = mock_llm
            
            enricher = MetadataEnricher(mock_settings_llm_enabled)
            
            # Should not create LLM yet / 此时尚不应创建 LLM
            mock_create.assert_not_called()
            
            # Access llm property / 访问 llm 属性
            _ = enricher.llm
            
            # Now should create / 现在应创建
            mock_create.assert_called_once()
    
    def test_llm_init_failure_disables_llm(self, mock_settings_llm_enabled):
        """Should disable LLM on initialization failure. / 初始化失败时应禁用 LLM。"""
        with patch('src.ingestion.transform.metadata_enricher.LLMFactory.create') as mock_create:
            mock_create.side_effect = Exception("Init failed")
            
            enricher = MetadataEnricher(mock_settings_llm_enabled)
            
            # Access llm property (will trigger init) / 访问 llm 属性（将触发初始化）
            _ = enricher.llm
            
            # Should disable LLM after failure / 失败后应禁用 LLM
            assert enricher.use_llm is False


# ============================================================================
# Response Parsing Tests / 响应解析测试
# ============================================================================

class TestResponseParsing:
    """Test LLM response parsing. / 测试 LLM 响应解析。"""
    
    def test_parse_well_formatted_response(self, mock_settings_llm_disabled):
        """Should parse properly formatted LLM response. / 应解析格式正确的 LLM 响应。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        response = (
            "Title: Data Science Fundamentals\n"
            "Summary: An introduction to data science covering statistics, machine learning, and data visualization.\n"
            "Tags: data science, statistics, machine learning, visualization"
        )
        
        metadata = enricher._parse_llm_response(response)
        
        assert metadata['title'] == "Data Science Fundamentals"
        assert "introduction to data science" in metadata['summary']
        assert "data science" in metadata['tags']
        assert len(metadata['tags']) == 4
    
    def test_parse_missing_fields(self, mock_settings_llm_disabled):
        """Should handle missing fields gracefully. / 应优雅处理字段缺失。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        response = "Title: Just a Title"
        
        metadata = enricher._parse_llm_response(response)
        
        assert metadata['title'] == "Just a Title"
        assert metadata['summary']  # Should have fallback / 应有 fallback
        assert isinstance(metadata['tags'], list)
    
    def test_parse_malformed_response(self, mock_settings_llm_disabled):
        """Should handle malformed response. / 应处理格式错误的响应。"""
        enricher = MetadataEnricher(mock_settings_llm_disabled)
        
        response = "This is not formatted correctly at all"
        
        metadata = enricher._parse_llm_response(response)
        
        assert metadata['title'] == "Untitled"  # Fallback / fallback
        assert metadata['summary']  # Should use raw response as summary / 应使用原始响应作为 summary
        assert isinstance(metadata['tags'], list)
