"""Integration tests for ChunkRefiner with real LLM providers. / 使用真实 LLM provider 的 ChunkRefiner 集成测试。

These tests require actual API keys and will make real API calls. / 这些测试需要真实 API key，并会发起真实 API 调用。
Run with: pytest tests/integration/test_chunk_refiner_llm.py -v -s / 运行方式：pytest tests/integration/test_chunk_refiner_llm.py -v -s

Required environment variables: / 必需环境变量：
    - OPENAI_API_KEY: For OpenAI tests / OPENAI_API_KEY：用于 OpenAI 测试
    - AZURE_OPENAI_API_KEY: For Azure tests / AZURE_OPENAI_API_KEY：用于 Azure 测试
    - OLLAMA_BASE_URL: For Ollama tests (default: http://localhost:11434) / OLLAMA_BASE_URL：用于 Ollama 测试（默认：http://localhost:11434）
"""

import os
import pytest
from unittest.mock import Mock

from src.core.settings import Settings, load_settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner


# Test data: Realistic noisy chunk from PDF extraction / 测试数据：从 PDF 提取出的真实风格噪声 chunk
NOISY_PDF_CHUNK = """
────────────────────────────
Page 42 | Technical Documentation
────────────────────────────


Chapter 5: System Architecture

The   microservices   architecture  consists  of  several  key  components.

<!-- Internal note: Update diagram -->

<div class="important">
Each service communicates via REST API or message queues.
</div>




The main   components   are:
- Gateway   Service  
- Authentication  Service  
- Data   Processing   Service


────────────────────────────
Footer: © 2024 Company | Confidential
────────────────────────────
"""

EXPECTED_CLEAN_RESULT_KEYWORDS = [
    "Chapter 5",
    "System Architecture",
    "microservices",
    "REST API",
    "message queues",
    "Gateway Service",
    "Authentication Service"
]


# Fixtures / Fixture

@pytest.fixture
def sample_noisy_chunk():
    """Create a sample noisy chunk for testing. / 创建用于测试的噪声 chunk 示例。"""
    return Chunk(
        id="test_pdf_chunk_001",
        text=NOISY_PDF_CHUNK,
        metadata={"source": "technical_doc.pdf", "source_path": "technical_doc.pdf", "page": 42},
        source_ref="doc_technical_2024"
    )


def create_settings_for_provider(provider: str) -> Settings:
    """Create settings object for specific provider. / 为指定 provider 创建 settings 对象。
    
    For Azure provider, loads actual settings from settings.yaml. / 对 Azure provider，从 settings.yaml 加载真实 settings。
    For other providers, uses environment variables. / 对其他 provider，使用环境变量。
    
    Args: / 参数：
        provider: One of 'openai', 'azure', 'ollama' / provider：'openai'、'azure'、'ollama' 之一
    
    Returns: / 返回：
        Settings object configured for the provider / 为该 provider 配置好的 Settings 对象
    """
    if provider == 'azure':
        # Load real settings from settings.yaml for Azure / 为 Azure 从 settings.yaml 加载真实 settings
        try:
            import yaml
            with open("config/settings.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # Inject into environment variables for AzureLLM / 注入环境变量供 AzureLLM 使用
            if 'llm' in config and config['llm'].get('provider') == 'azure':
                os.environ["AZURE_OPENAI_API_KEY"] = config['llm'].get('api_key', '')
                os.environ["AZURE_OPENAI_ENDPOINT"] = config['llm'].get('azure_endpoint', '')
                os.environ["ENDPOINT"] = config['llm'].get('azure_endpoint', '')
                
                os.environ["AZURE_OPENAI_API_VERSION"] = config['llm'].get('api_version', '')
                os.environ["OPENAI_API_VERSION"] = config['llm'].get('api_version', '') 
            
            real_settings = load_settings("config/settings.yaml")

            # Create a Mock Settings object to allow modification (real settings are frozen) / 创建 Mock Settings 对象以便修改（真实 settings 是冻结的）
            settings = Mock(spec=Settings)
            # Copy necessary frozen configs / 复制必要的冻结配置
            settings.llm = real_settings.llm
            settings.ingestion = Mock()
            # Enable LLM for chunk refiner / 为 chunk refiner 启用 LLM
            settings.ingestion.chunk_refiner = {'use_llm': True}
            return settings
        except Exception as e:
            pytest.skip(f"Failed to load settings.yaml or configure Azure: {e}")
    
    # For non-Azure providers, use environment variables / 对非 Azure provider，使用环境变量
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    
    # LLM configuration / LLM 配置
    settings.llm = Mock()
    settings.llm.provider = provider
    
    if provider == 'openai':
        settings.llm.model = "gpt-3.5-turbo"
        settings.llm.api_key = os.getenv('OPENAI_API_KEY')
        settings.llm.temperature = 0.3
        settings.llm.max_tokens = 1000
        
    elif provider == 'ollama':
        settings.llm.model = "llama2"
        settings.llm.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        settings.llm.temperature = 0.3
        
    return settings


# Helper function to check provider availability / 检查 provider 可用性的辅助函数

def is_provider_available(provider: str) -> tuple[bool, str]:
    """Check if provider credentials are available. / 检查 provider 凭据是否可用。
    
    Returns: / 返回：
        (is_available, env_var_name) / （是否可用，环境变量名）
    """
    if provider == 'azure':
        # For Azure, check if settings.yaml exists and has LLM config / 对 Azure，检查 settings.yaml 是否存在且包含 LLM 配置
        try:
            import yaml
            with open("config/settings.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if 'llm' in config and config['llm'].get('provider') == 'azure':
                return True, 'settings.yaml'
        except:
            pass
        return False, 'settings.yaml'
        
    if provider == 'openai':
        env_var = 'OPENAI_API_KEY'
        return os.getenv(env_var) is not None, env_var
        
    elif provider == 'ollama':
        # Ollama assumed available if base_url is set or default / 如果设置了 base_url 或使用默认值，则认为 Ollama 可用
        return True, 'OLLAMA_BASE_URL'
        
    return False, ''


# Test Cases / 测试用例

@pytest.mark.integration
@pytest.mark.parametrize("provider,env_var", [
    ("openai", "OPENAI_API_KEY"),
    ("azure", "AZURE_OPENAI_API_KEY"),
    ("ollama", "OLLAMA_BASE_URL"),
])
def test_multiple_providers_if_available(provider, env_var, sample_noisy_chunk):
    """Test refinement with multiple LLM providers (if configured). / 测试使用多个 LLM provider 执行精炼（如果已配置）。
    
    This test will be skipped for providers without credentials. / 对没有凭据的 provider 会跳过该测试。
    """
    available, env_name = is_provider_available(provider)
    
    if not available:
        pytest.skip(f"Skipping {provider} test: {env_name} not set")
    
    # Create settings for provider / 为 provider 创建 settings
    settings = create_settings_for_provider(provider)
    
    # Create refiner / 创建 refiner
    refiner = ChunkRefiner(settings)
    trace = TraceContext()
    
    # Perform refinement / 执行精炼
    result = refiner.transform([sample_noisy_chunk], trace=trace)
    
    # Assertions / 断言
    assert len(result) == 1
    refined_chunk = result[0]
    
    # Should be marked as LLM-refined / 应标记为 LLM 精炼
    assert refined_chunk.metadata['refined_by'] == 'llm', \
        f"Expected LLM refinement for {provider}, got {refined_chunk.metadata.get('refined_by')}"
    
    # Refined text should be cleaner (no separator lines, no HTML) / 精炼后的文本应更干净（无分隔线、无 HTML）
    assert '────────────' not in refined_chunk.text
    assert '<!-- ' not in refined_chunk.text
    assert '<div' not in refined_chunk.text
    assert 'Footer:' not in refined_chunk.text.lower()
    
    # Should preserve key content / 应保留关键内容
    for keyword in EXPECTED_CLEAN_RESULT_KEYWORDS:
        assert keyword in refined_chunk.text, \
            f"Expected keyword '{keyword}' not found in refined text"
    
    # Trace should record LLM usage / Trace 应记录 LLM 使用情况
    stage_data = trace.get_stage_data('chunk_refiner')
    assert stage_data is not None
    assert stage_data['data']['llm_enhanced_count'] == 1
    assert stage_data['data']['fallback_count'] == 0
    
    # Print for manual review / 打印用于人工审查
    print(f"\n{'='*60}")
    print(f"Provider: {provider}")
    print(f"{'='*60}")
    print("ORIGINAL TEXT (first 200 chars):")
    print(sample_noisy_chunk.text[:200])
    print(f"\n{'-'*60}")
    print("REFINED TEXT:")
    print(refined_chunk.text)
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_graceful_fallback_with_invalid_model(sample_noisy_chunk):
    """Test that refiner falls back to rule-based when LLM fails. / 测试 LLM 失败时 refiner 会回退到基于规则。"""
    # Create settings with intentionally invalid model / 创建故意使用无效模型的 settings
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    settings.llm = Mock()
    settings.llm.provider = 'openai'
    settings.llm.model = 'nonexistent-model-xyz'
    settings.llm.api_key = os.getenv('OPENAI_API_KEY', 'fake-key')
    
    refiner = ChunkRefiner(settings)
    trace = TraceContext()
    
    # Should not crash, should fallback / 不应崩溃，应回退
    result = refiner.transform([sample_noisy_chunk], trace=trace)
    
    assert len(result) == 1
    # Should fallback to rule-based / 应回退到基于规则
    assert result[0].metadata['refined_by'] == 'rule'
    assert 'refine_fallback_reason' in result[0].metadata
    
    # Should still apply rule-based cleaning / 仍应应用基于规则的清理
    assert '────────────' not in result[0].text
    assert 'Footer:' not in result[0].text


@pytest.mark.integration
@pytest.mark.skipif(not is_provider_available('openai')[0], reason="OPENAI_API_KEY not set")
def test_refinement_quality_comparison(sample_noisy_chunk):
    """Compare rule-based vs LLM refinement quality. / 比较基于规则与 LLM 精炼质量。
    
    This test provides visual comparison for manual quality assessment. / 该测试为人工质量评估提供可视化对比。
    """
    # Rule-based only / 仅基于规则
    settings_rule = Mock(spec=Settings)
    settings_rule.ingestion = Mock()
    settings_rule.ingestion.chunk_refiner = {'use_llm': False}
    
    refiner_rule = ChunkRefiner(settings_rule)
    result_rule = refiner_rule.transform([sample_noisy_chunk])
    
    # LLM-enhanced / LLM 增强
    settings_llm = create_settings_for_provider('openai')
    refiner_llm = ChunkRefiner(settings_llm)
    result_llm = refiner_llm.transform([sample_noisy_chunk])
    
    # Print comparison / 打印对比
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print(f"{'='*60}")
    print("\nORIGINAL (first 300 chars):")
    print(sample_noisy_chunk.text[:300])
    print(f"\n{'-'*60}")
    print("RULE-BASED REFINEMENT:")
    print(result_rule[0].text)
    print(f"\n{'-'*60}")
    print("LLM-ENHANCED REFINEMENT:")
    print(result_llm[0].text)
    print(f"{'='*60}\n")
    
    # Basic assertions / 基础断言
    assert len(result_rule[0].text) > 0
    assert len(result_llm[0].text) > 0
    assert result_rule[0].metadata['refined_by'] == 'rule'
    assert result_llm[0].metadata['refined_by'] == 'llm'


@pytest.mark.integration
@pytest.mark.skipif(not is_provider_available('openai')[0], reason="OPENAI_API_KEY not set")
def test_batch_refinement_performance(sample_noisy_chunk):
    """Test refining multiple chunks in a batch. / 测试批量精炼多个 chunk。"""
    settings = create_settings_for_provider('openai')
    refiner = ChunkRefiner(settings)
    trace = TraceContext()
    
    # Create 3 test chunks / 创建 3 个测试 chunk
    chunks = [
        sample_noisy_chunk,
        Chunk(
            id="chunk_002",
            text="Another  chunk   with   noise\n\n\n\nand issues.",
            metadata={"source": "test2.pdf", "source_path": "test2.pdf"}
        ),
        Chunk(
            id="chunk_003",
            text="Clean chunk without much noise.",
            metadata={"source": "test3.pdf", "source_path": "test3.pdf"}
        )
    ]
    
    # Refine all / 全部精炼
    import time
    start_time = time.time()
    result = refiner.transform(chunks, trace=trace)
    elapsed_time = time.time() - start_time
    
    # Assertions / 断言
    assert len(result) == 3
    assert all(r.metadata['refined_by'] == 'llm' for r in result)
    
    # Trace / Trace
    stage_data = trace.get_stage_data('chunk_refiner')
    assert stage_data['data']['llm_enhanced_count'] == 3
    
    print(f"\n{'='*60}")
    print(f"Refined {len(chunks)} chunks in {elapsed_time:.2f} seconds")
    print(f"Average: {elapsed_time/len(chunks):.2f} seconds per chunk")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
