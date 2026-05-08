"""Unit tests for ChunkRefiner transform. / ChunkRefiner transform 的单元测试。"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.llm.base_llm import BaseLLM


# Fixtures / Fixture

@pytest.fixture
def noisy_chunks_data():
    """Load test data from fixtures. / 从 fixtures 加载测试数据。"""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"
    with open(fixture_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def mock_settings():
    """Create mock settings without LLM enabled. / 创建未启用 LLM 的 mock settings。"""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': False}
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """Create mock settings with LLM enabled. / 创建启用 LLM 的 mock settings。"""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    return settings


@pytest.fixture
def mock_llm():
    """Create mock LLM. / 创建 mock LLM。"""
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = "LLM refined text"
    return llm


@pytest.fixture
def sample_chunk():
    """Create a sample chunk. / 创建示例 chunk。"""
    return Chunk(
        id="test_chunk_001",
        text="Sample text with  extra   spaces\n\n\n\nand newlines",
        metadata={"source": "test.pdf", "source_path": "test.pdf"},
        source_ref="test_doc"
    )


# Test Rule-Based Refinement / 基于规则的精炼测试

class TestRuleBasedRefinement:
    """Test rule-based cleaning without LLM. / 测试不使用 LLM 的规则清洗。"""
    
    def test_remove_excessive_whitespace(self, mock_settings, noisy_chunks_data):
        """Test removal of excessive spaces and newlines. / 测试移除多余空格和换行。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['excessive_whitespace']['input']
        expected = noisy_chunks_data['excessive_whitespace']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_remove_page_headers_footers(self, mock_settings, noisy_chunks_data):
        """Test removal of page headers and footers with separator lines. / 测试移除带分隔线的页眉页脚。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['page_header_footer']['input']
        expected = noisy_chunks_data['page_header_footer']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_remove_html_tags_and_comments(self, mock_settings, noisy_chunks_data):
        """Test removal of HTML tags and comments while preserving Markdown. / 测试在保留 Markdown 的同时移除 HTML 标签和注释。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['format_markers']['input']
        expected = noisy_chunks_data['format_markers']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_preserve_code_blocks(self, mock_settings, noisy_chunks_data):
        """Test that code blocks internal formatting is preserved. / 测试代码块内部格式会被保留。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['code_blocks']['input']
        expected = noisy_chunks_data['code_blocks']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_not_overclean_good_text(self, mock_settings, noisy_chunks_data):
        """Test that clean text is not over-processed. / 测试干净文本不会被过度处理。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['clean_text']['input']
        expected = noisy_chunks_data['clean_text']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_mixed_noise_scenario(self, mock_settings, noisy_chunks_data):
        """Test comprehensive noise removal with mixed issues. / 测试混合问题下的综合噪声移除。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['mixed_noise']['input']
        expected = noisy_chunks_data['mixed_noise']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_empty_string_handling(self, mock_settings):
        """Test handling of empty strings. / 测试空字符串处理。"""
        refiner = ChunkRefiner(mock_settings)
        
        assert refiner._rule_based_refine("") == ""
        # Whitespace-only strings return empty after strip / 仅空白字符串 strip 后返回空
        result = refiner._rule_based_refine("   \n\n  ")
        assert result.strip() == ""
    
    def test_none_handling(self, mock_settings):
        """Test handling of None input. / 测试 None 输入处理。"""
        refiner = ChunkRefiner(mock_settings)
        result = refiner._rule_based_refine(None)
        assert result is None


# Test Transform Pipeline (Rule-Only Mode) / Transform Pipeline 测试（仅规则模式）

class TestTransformPipelineRuleOnly:
    """Test full transform pipeline without LLM. / 测试不使用 LLM 的完整 transform pipeline。"""
    
    def test_transform_single_chunk(self, mock_settings, sample_chunk):
        """Test transforming a single chunk. / 测试转换单个 chunk。"""
        refiner = ChunkRefiner(mock_settings)
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert result[0].id == sample_chunk.id
        assert result[0].text == "Sample text with extra spaces\n\nand newlines"
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_transform_multiple_chunks(self, mock_settings):
        """Test transforming multiple chunks. / 测试转换多个 chunks。"""
        refiner = ChunkRefiner(mock_settings)
        
        chunks = [
            Chunk(id="c1", text="Text  with   spaces", metadata={"source_path": "test1.pdf"}),
            Chunk(id="c2", text="Another\n\n\nchunk", metadata={"source_path": "test2.pdf"}),
            Chunk(id="c3", text="Clean text", metadata={"source_path": "test3.pdf"})
        ]
        
        result = refiner.transform(chunks)
        
        assert len(result) == 3
        assert result[0].text == "Text with spaces"
        assert result[1].text == "Another\n\nchunk"
        assert result[2].text == "Clean text"
        assert all(r.metadata['refined_by'] == 'rule' for r in result)
    
    def test_transform_empty_list(self, mock_settings):
        """Test transforming empty chunk list. / 测试转换空 chunk 列表。"""
        refiner = ChunkRefiner(mock_settings)
        result = refiner.transform([])
        assert result == []
    
    def test_metadata_preserved(self, mock_settings, sample_chunk):
        """Test that original metadata is preserved. / 测试原始 metadata 会被保留。"""
        refiner = ChunkRefiner(mock_settings)
        
        result = refiner.transform([sample_chunk])
        
        assert result[0].metadata['source'] == 'test.pdf'
        assert 'refined_by' in result[0].metadata
    
    def test_trace_recording(self, mock_settings, sample_chunk):
        """Test that trace context records processing info. / 测试 trace context 会记录处理信息。"""
        refiner = ChunkRefiner(mock_settings)
        trace = TraceContext()
        
        refiner.transform([sample_chunk], trace=trace)
        
        stage_data = trace.get_stage_data('chunk_refiner')
        assert stage_data is not None
        assert stage_data['total_chunks'] == 1
        assert stage_data['success_count'] == 1
        assert stage_data['use_llm'] is False


# Test LLM Enhancement Mode / LLM 增强模式测试

class TestLLMEnhancement:
    """Test LLM-based refinement. / 测试基于 LLM 的精炼。"""
    
    def test_llm_success_path(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test successful LLM refinement. / 测试 LLM 精炼成功路径。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = "Beautifully refined text by LLM"
        
        # Mock prompt loading / Mock prompt 加载
        refiner._prompt_template = "Refine this: {text}"
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert "Beautifully refined" in result[0].text
        assert result[0].metadata['refined_by'] == 'llm'
        mock_llm.chat.assert_called_once()
    
    def test_llm_fallback_on_failure(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test fallback to rule-based when LLM fails. / 测试 LLM 失败时回退到规则模式。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.side_effect = Exception("LLM API error")
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_llm_fallback_on_empty_response(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test fallback when LLM returns empty string. / 测试 LLM 返回空字符串时回退。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = ""
        refiner._prompt_template = "Refine: {text}"
        
        result = refiner.transform([sample_chunk])
        
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_llm_trace_recording(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """Test trace records LLM enhancement count. / 测试 trace 会记录 LLM 增强数量。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = "LLM result"
        refiner._prompt_template = "Refine: {text}"
        trace = TraceContext()
        
        refiner.transform([sample_chunk], trace=trace)
        
        stage_data = trace.get_stage_data('chunk_refiner')
        assert stage_data is not None
        assert stage_data['llm_enhanced_count'] == 1
        assert stage_data['fallback_count'] == 0


# Test Prompt Loading / Prompt 加载测试

class TestPromptLoading:
    """Test prompt template loading. / 测试 prompt 模板加载。"""
    
    def test_load_existing_prompt(self, mock_settings):
        """Test loading existing prompt file. / 测试加载已存在的 prompt 文件。"""
        refiner = ChunkRefiner(mock_settings)
        
        # Use real prompt file / 使用真实 prompt 文件
        prompt = refiner._load_prompt()
        
        assert prompt is not None
        assert '{text}' in prompt
    
    def test_prompt_cached_after_first_load(self, mock_settings):
        """Test that prompt is cached after first load. / 测试首次加载后 prompt 会被缓存。"""
        refiner = ChunkRefiner(mock_settings)
        
        prompt1 = refiner._load_prompt()
        prompt2 = refiner._load_prompt()
        
        assert prompt1 is prompt2  # Same object reference / 相同对象引用
    
    def test_missing_prompt_file(self, mock_settings):
        """Test handling of missing prompt file. / 测试缺失 prompt 文件的处理。"""
        refiner = ChunkRefiner(mock_settings, prompt_path="nonexistent.txt")
        
        prompt = refiner._load_prompt()
        
        assert prompt is None
    
    def test_prompt_missing_placeholder(self, mock_settings, mock_llm, sample_chunk):
        """Test LLM refinement fails gracefully when prompt lacks {text}. / 测试 prompt 缺少 {text} 时 LLM 精炼会优雅失败。"""
        refiner = ChunkRefiner(mock_settings, llm=mock_llm)
        refiner._prompt_template = "This prompt has no placeholder"
        refiner.use_llm = True
        
        result = refiner._llm_refine("test text")
        
        assert result is None


# Test Atomic Processing / 原子处理测试

class TestAtomicProcessing:
    """Test that individual chunk failures don't affect others. / 测试单个 chunk 失败不会影响其他 chunk。"""
    
    def test_exception_in_one_chunk_preserves_original(self, mock_settings):
        """Test that exception in processing one chunk preserves its original. / 测试处理某个 chunk 异常时会保留其原始内容。"""
        refiner = ChunkRefiner(mock_settings)
        
        # Mock _rule_based_refine to fail on specific chunk / Mock _rule_based_refine，使其在特定 chunk 上失败
        original_method = refiner._rule_based_refine
        
        def failing_refine(text):
            if "fail" in text:
                raise ValueError("Intentional test error")
            return original_method(text)
        
        refiner._rule_based_refine = failing_refine
        
        chunks = [
            Chunk(id="c1", text="Normal text", metadata={"source_path": "test1.pdf"}),
            Chunk(id="c2", text="This should fail processing", metadata={"source_path": "test2.pdf"}),
            Chunk(id="c3", text="Another normal text", metadata={"source_path": "test3.pdf"})
        ]
        
        result = refiner.transform(chunks)
        
        # All chunks returned / 所有 chunks 都会返回
        assert len(result) == 3
        # Failed chunk preserved as-is / 失败的 chunk 原样保留
        assert result[1].text == "This should fail processing"
        # Other chunks processed normally / 其他 chunks 正常处理
        assert result[0].metadata.get('refined_by') == 'rule'
        assert result[2].metadata.get('refined_by') == 'rule'


# Test Configuration / 配置测试

class TestConfiguration:
    """Test configuration handling. / 测试配置处理。"""
    
    def test_use_llm_disabled_by_default(self):
        """Test that LLM is disabled when config missing. / 测试配置缺失时 LLM 默认禁用。"""
        settings = Mock(spec=Settings)
        settings.ingestion = None
        
        refiner = ChunkRefiner(settings)
        
        assert refiner.use_llm is False
    
    def test_lazy_llm_initialization(self, mock_settings_with_llm):
        """Test that LLM is only initialized when needed. / 测试 LLM 只在需要时初始化。"""
        with patch('src.ingestion.transform.chunk_refiner.LLMFactory.create') as mock_factory:
            refiner = ChunkRefiner(mock_settings_with_llm)
            
            # LLM not initialized yet / LLM 尚未初始化
            assert refiner._llm is None
            mock_factory.assert_not_called()
            
            # Access llm property triggers initialization / 访问 llm 属性会触发初始化
            _ = refiner.llm
            mock_factory.assert_called_once()
    
    def test_llm_init_failure_disables_llm(self, mock_settings_with_llm):
        """Test that LLM initialization failure disables LLM mode. / 测试 LLM 初始化失败会禁用 LLM 模式。"""
        with patch('src.ingestion.transform.chunk_refiner.LLMFactory.create', side_effect=Exception("Init failed")):
            refiner = ChunkRefiner(mock_settings_with_llm)
            
            # Try to access LLM / 尝试访问 LLM
            llm = refiner.llm
            
            # LLM mode disabled after failure / 失败后 LLM 模式被禁用
            assert llm is None
            assert refiner.use_llm is False
