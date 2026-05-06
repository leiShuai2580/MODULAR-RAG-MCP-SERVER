"""Integration tests for DenseEncoder with real Azure Embedding. / 使用真实 Azure Embedding 的 DenseEncoder 集成测试。

This test suite validates that DenseEncoder correctly integrates with / 该测试套件验证 DenseEncoder 能正确集成
the Azure Embedding provider using real API calls. These tests verify: / 使用真实 API 调用的 Azure Embedding provider。这些测试验证：
- Real embedding API connectivity / 真实 embedding API 连通性
- Correct vector dimensions / 正确的向量维度
- Batch processing with real provider / 使用真实 provider 的批处理
- Error handling in production-like scenarios / 类生产场景中的错误处理

⚠️ WARNING: These tests make REAL API calls and incur costs. / ⚠️ 警告：这些测试会发起真实 API 调用并产生费用。
Run only when validating Azure Embedding configuration. / 仅在验证 Azure Embedding 配置时运行。
"""

import pytest
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.core.types import Chunk
from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory


@pytest.fixture(scope="module")
def settings():
    """Load settings from config file. / 从配置文件加载 settings。"""
    return load_settings("config/settings.yaml")


@pytest.fixture(scope="module")
def azure_embedding(settings):
    """Create Azure Embedding instance from settings. / 从 settings 创建 Azure Embedding 实例。
    
    This fixture verifies that: / 该 fixture 验证：
    1. Settings contain valid Azure Embedding configuration / Settings 包含有效的 Azure Embedding 配置
    2. EmbeddingFactory can create Azure provider from settings.yaml / EmbeddingFactory 可以从 settings.yaml 创建 Azure provider
    3. The provider is ready for use / provider 已准备好使用
    
    All configuration (endpoint, api_key, model) comes from settings.yaml. / 所有配置（endpoint、api_key、model）都来自 settings.yaml。
    No hardcoded values or environment variable overrides. / 没有硬编码值或环境变量覆盖。
    """
    # Validate Azure configuration is present / 验证 Azure 配置存在
    assert settings.embedding.provider == "azure", \
        "Integration test requires Azure embedding provider in settings"
    
    assert settings.embedding.azure_endpoint, \
        "Azure endpoint must be configured in settings.yaml"
    
    assert settings.embedding.api_key, \
        "Azure API key must be configured in settings.yaml"
    
    # Create embedding provider via factory (reads all config from settings) / 通过 factory 创建 embedding provider（从 settings 读取全部配置）
    embedding = EmbeddingFactory.create(settings)
    
    return embedding


@pytest.fixture
def encoder(azure_embedding):
    """Create DenseEncoder with Azure Embedding provider. / 使用 Azure Embedding provider 创建 DenseEncoder。"""
    return DenseEncoder(azure_embedding, batch_size=10)


# ============================================================================
# Real API Call Tests / 真实 API 调用测试
# ============================================================================

def test_encode_single_chunk_with_azure(encoder):
    """Test encoding a single chunk with real Azure Embedding API. / 测试使用真实 Azure Embedding API 编码单个 chunk。
    
    This test verifies: / 该测试验证：
    - API connectivity / API 连通性
    - Correct response format / 正确的响应格式
    - Expected vector dimensions (1536 for text-embedding-ada-002) / 预期向量维度（text-embedding-ada-002 为 1536）
    """
    chunks = [
        Chunk(
            id="test_1",
            text="This is a test chunk for Azure Embedding integration.",
            metadata={'source_path': 'integration_test'}
        )
    ]
    
    vectors = encoder.encode(chunks)
    
    # Verify output structure / 验证输出结构
    assert len(vectors) == 1, "Should return exactly 1 vector"
    assert len(vectors[0]) == 1536, \
        "text-embedding-ada-002 should return 1536-dimensional vectors"
    
    # Verify all values are floats / 验证所有值都是 float
    assert all(isinstance(v, float) for v in vectors[0]), \
        "All vector components should be floats"
    
    # Verify non-zero vector (sanity check) / 验证非零向量（健全性检查）
    assert any(v != 0.0 for v in vectors[0]), \
        "Vector should contain non-zero values"


def test_encode_multiple_chunks_with_azure(encoder):
    """Test encoding multiple chunks in single batch. / 测试在单个批次中编码多个 chunk。
    
    This test verifies: / 该测试验证：
    - Batch processing works with real API / 批处理可与真实 API 配合工作
    - All chunks are processed / 所有 chunk 都被处理
    - Vectors are semantically distinct / 向量在语义上可区分
    """
    chunks = [
        Chunk(id="1", text="Artificial intelligence is transforming technology.", metadata={'source_path': 'integration_test'}),
        Chunk(id="2", text="Machine learning enables computers to learn from data.", metadata={'source_path': 'integration_test'}),
        Chunk(id="3", text="Python is a popular programming language.", metadata={'source_path': 'integration_test'}),
    ]
    
    vectors = encoder.encode(chunks)
    
    # Verify correct number of vectors / 验证向量数量正确
    assert len(vectors) == 3, "Should return 3 vectors for 3 chunks"
    
    # Verify all vectors have correct dimension / 验证所有向量维度正确
    assert all(len(v) == 1536 for v in vectors), \
        "All vectors should have 1536 dimensions"
    
    # Verify vectors are distinct (semantic difference) / 验证向量不同（语义差异）
    # Chunks 1 and 2 are related (AI/ML), chunk 3 is different (Python) / chunk 1 和 2 相关（AI/ML），chunk 3 不同（Python）
    # We don't assert exact similarity values, just that vectors differ / 不断言精确相似度值，只断言向量不同
    assert vectors[0] != vectors[1] != vectors[2], \
        "Vectors should be distinct for different texts"


def test_encode_with_batching(azure_embedding):
    """Test encoding with multiple batches using real API. / 测试使用真实 API 进行多批次编码。
    
    This test verifies: / 该测试验证：
    - Multi-batch processing works correctly / 多批次处理工作正确
    - Batch boundaries don't cause issues / 批次边界不会引发问题
    - All chunks are processed across batches / 所有 chunk 都会跨批次处理
    """
    encoder = DenseEncoder(azure_embedding, batch_size=2)
    
    chunks = [
        Chunk(id=f"chunk_{i}", text=f"Test chunk number {i} with unique content.", metadata={'source_path': 'integration_test'})
        for i in range(5)
    ]
    
    vectors = encoder.encode(chunks)
    
    # Should process in 3 batches: [0:2], [2:4], [4:5] / 应处理为 3 个批次：[0:2]、[2:4]、[4:5]
    assert len(vectors) == 5, "All 5 chunks should be processed"
    assert all(len(v) == 1536 for v in vectors), "All vectors should have correct dimension"
    
    # Verify batch processing didn't corrupt ordering / 验证批处理没有破坏顺序
    # (We can't verify exact order without knowing vector content, / （不了解向量内容时无法验证精确顺序，
    #  but we can verify count and dimensions) /  但可以验证数量和维度）
    assert encoder.get_batch_count(5) == 3, "Should calculate 3 batches"


def test_encode_realistic_text_lengths(encoder):
    """Test encoding chunks of varying realistic lengths. / 测试编码不同真实长度的 chunk。
    
    This test verifies: / 该测试验证：
    - Short chunks (titles/headers) / 短 chunk（标题/页眉）
    - Medium chunks (paragraphs) / 中等 chunk（段落）
    - Long chunks (full sections) / 长 chunk（完整章节）
    """
    chunks = [
        Chunk(
            id="short",
            text="Introduction",
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="medium",
            text=(
                "This is a medium-length chunk representing a typical paragraph. "
                "It contains multiple sentences with varying content and structure. "
                "The embedding should capture the semantic meaning effectively."
            ),
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="long",
            text=(
                "This is a longer chunk that might represent a full section of a document. "
                "It contains multiple paragraphs with different topics and concepts. "
                "The first paragraph introduces the main theme and sets the context. "
                "The second paragraph provides detailed explanations and examples. "
                "The third paragraph summarizes key points and conclusions. "
                "This tests the embedding model's ability to capture meaning from longer texts."
            ),
            metadata={'source_path': 'integration_test'}
        ),
    ]
    
    vectors = encoder.encode(chunks)
    
    assert len(vectors) == 3, "Should process all 3 chunks"
    assert all(len(v) == 1536 for v in vectors), "All vectors should have correct dimension"


def test_encode_special_characters(encoder):
    """Test encoding text with special characters and formatting. / 测试编码包含特殊字符和格式的文本。
    
    This test verifies: / 该测试验证：
    - Unicode handling / Unicode 处理
    - Special characters / 特殊字符
    - Code snippets / 代码片段
    - Mathematical symbols / 数学符号
    """
    chunks = [
        Chunk(
            id="unicode",
            text="多语言文本支持：中文、日本語、한국어、العربية",
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="code",
            text='def hello_world():\n    print("Hello, World!")\n    return True',
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="math",
            text="Mathematical formula: E = mc² and α + β = γ",
            metadata={'source_path': 'integration_test'}
        ),
    ]
    
    vectors = encoder.encode(chunks)
    
    assert len(vectors) == 3, "Should handle special characters correctly"
    assert all(len(v) == 1536 for v in vectors), "Vector dimensions should be consistent"


# ============================================================================
# Configuration Validation Tests / 配置校验测试
# ============================================================================

def test_azure_configuration_is_valid(settings):
    """Verify that Azure Embedding configuration in settings.yaml is correct. / 验证 settings.yaml 中的 Azure Embedding 配置正确。
    
    This test validates: / 该测试校验：
    - Provider is set to 'azure' / Provider 设置为 'azure'
    - Model name is specified / 已指定模型名称
    - Dimensions match model (1536 for ada-002) / 维度匹配模型（ada-002 为 1536）
    - Credentials can be provided via env vars / 凭据可通过环境变量提供
    """
    assert settings.embedding.provider == "azure", \
        "Provider should be 'azure' for integration tests"
    
    assert settings.embedding.model == "text-embedding-ada-002", \
        "Model should be text-embedding-ada-002 for this test suite"
    
    assert settings.embedding.dimensions == 1536, \
        "Dimensions should be 1536 for text-embedding-ada-002"


def test_factory_creates_azure_embedding(settings):
    """Verify that EmbeddingFactory correctly creates Azure provider. / 验证 EmbeddingFactory 正确创建 Azure provider。"""
    embedding = EmbeddingFactory.create(settings)
    
    # Verify it's the correct type (AzureEmbedding) / 验证它是正确类型（AzureEmbedding）
    assert embedding.__class__.__name__ == "AzureEmbedding", \
        "Factory should create AzureEmbedding instance"


# ============================================================================
# Error Handling Tests (Graceful Degradation) / 错误处理测试（优雅降级）
# ============================================================================

def test_encode_handles_empty_text_gracefully(encoder):
    """Test that empty/invalid chunks are caught before API call. / 测试空/无效 chunk 会在 API 调用前被捕获。
    
    This prevents wasting API quota on invalid requests. / 这会避免在无效请求上浪费 API 配额。
    """
    chunks = [
        Chunk(id="empty", text="", metadata={'source_path': 'integration_test'})
    ]
    
    with pytest.raises(ValueError, match="has empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_validates_before_api_call(encoder):
    """Test that validation happens before making expensive API calls. / 测试校验会在发起昂贵 API 调用前执行。"""
    chunks = [
        Chunk(id="valid", text="Valid chunk", metadata={'source_path': 'integration_test'}),
        Chunk(id="invalid", text="   \n  ", metadata={'source_path': 'integration_test'}),  # Whitespace only / 仅空白字符
    ]
    
    # Should fail validation before any API call is made / 应在发起任何 API 调用前校验失败
    with pytest.raises(ValueError, match="whitespace-only"):
        encoder.encode(chunks)


# ============================================================================
# Performance and Quality Observations / 性能和质量观察
# ============================================================================

def test_encode_performance_observation(encoder):
    """Observe encoding performance with real API. / 观察真实 API 的编码性能。
    
    This test doesn't assert performance, but logs timing information / 该测试不断言性能，只记录耗时信息
    for manual observation and optimization opportunities. / 用于人工观察和发现优化机会。
    """
    import time
    
    chunks = [
        Chunk(id=f"perf_{i}", text=f"Performance test chunk {i}", metadata={'source_path': 'integration_test'})
        for i in range(10)
    ]
    
    start_time = time.time()
    vectors = encoder.encode(chunks)
    elapsed = time.time() - start_time
    
    assert len(vectors) == 10
    
    # Log for observation (not assertion) / 记录用于观察（非断言）
    print(f"\n⏱️  Encoded 10 chunks in {elapsed:.2f}s ({elapsed/10:.2f}s per chunk)")


def test_semantic_similarity_observation(encoder):
    """Observe semantic similarity of related vs unrelated chunks. / 观察相关与不相关 chunk 的语义相似度。
    
    This test computes cosine similarity to validate that the embeddings / 该测试计算余弦相似度以验证 embeddings
    capture semantic relationships correctly. / 能正确捕获语义关系。
    """
    import math
    
    chunks = [
        Chunk(id="1", text="Dogs are loyal pets and great companions.", metadata={'source_path': 'integration_test'}),
        Chunk(id="2", text="Cats are independent animals that enjoy solitude.", metadata={'source_path': 'integration_test'}),
        Chunk(id="3", text="Canines make excellent guard animals for homes.", metadata={'source_path': 'integration_test'}),
    ]
    
    vectors = encoder.encode(chunks)
    
    # Helper function to compute cosine similarity / 用于计算余弦相似度的辅助函数
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        return dot_product / (magnitude1 * magnitude2)
    
    # Compute similarities / 计算相似度
    sim_dogs_cats = cosine_similarity(vectors[0], vectors[1])  # Different: dogs vs cats / 不同：dogs vs cats
    sim_dogs_canines = cosine_similarity(vectors[0], vectors[2])  # Similar: dogs vs canines / 相似：dogs vs canines
    
    # Log observations / 记录观察结果
    print(f"\n📊 Semantic Similarity:")
    print(f"   Dogs vs Cats: {sim_dogs_cats:.4f}")
    print(f"   Dogs vs Canines: {sim_dogs_canines:.4f}")
    
    # Expectation: Dogs should be more similar to Canines than to Cats / 预期：Dogs 与 Canines 的相似度应高于 Dogs 与 Cats
    # (This is a weak assertion - real thresholds depend on the model) / （这是弱断言，真实阈值取决于模型）
    assert sim_dogs_canines > sim_dogs_cats, \
        "Semantically related chunks should have higher similarity"
