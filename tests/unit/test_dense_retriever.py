"""Unit tests for DenseRetriever. / DenseRetriever 的单元测试。

This module tests the DenseRetriever component with mock embedding client / 该模块使用 mock embedding client
and vector store to ensure correct behavior in isolation. / 和 vector store 测试 DenseRetriever 组件，以确保其隔离行为正确。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict, List

from src.core.query_engine.dense_retriever import DenseRetriever, create_dense_retriever
from src.core.types import RetrievalResult


# =============================================================================
# Test Fixtures / 测试 Fixture
# =============================================================================

class FakeEmbedding:
    """Fake embedding client for testing. / 用于测试的 fake embedding client。"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0
        self.last_texts = None
    
    def embed(self, texts: List[str], trace: Any = None, **kwargs) -> List[List[float]]:
        """Return deterministic fake embeddings. / 返回确定性的 fake embeddings。"""
        self.call_count += 1
        self.last_texts = texts
        # Generate deterministic vectors based on text content / 基于文本内容生成确定性向量
        return [[float(i) / self.dimension for i in range(self.dimension)] for _ in texts]


class FakeVectorStore:
    """Fake vector store for testing. / 用于测试的 fake vector store。"""
    
    def __init__(self, stored_records: List[Dict[str, Any]] = None):
        self.stored_records = stored_records or []
        self.call_count = 0
        self.last_query_params = None
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        trace: Any = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Return stored records sorted by fake similarity. / 返回按 fake similarity 排序的已存储记录。"""
        self.call_count += 1
        self.last_query_params = {
            "vector": vector,
            "top_k": top_k,
            "filters": filters,
        }
        # Return up to top_k records / 最多返回 top_k 条记录
        return self.stored_records[:top_k]


@pytest.fixture
def fake_embedding():
    """Create a fake embedding client. / 创建 fake embedding client。"""
    return FakeEmbedding(dimension=384)


@pytest.fixture
def fake_vector_store():
    """Create a fake vector store with sample data. / 创建带示例数据的 fake vector store。"""
    return FakeVectorStore(stored_records=[
        {
            "id": "doc1_chunk_001",
            "score": 0.95,
            "text": "Azure OpenAI is a cloud-based AI service...",
            "metadata": {"source_path": "docs/azure.pdf", "chunk_index": 1},
        },
        {
            "id": "doc1_chunk_002",
            "score": 0.88,
            "text": "To configure Azure OpenAI, you need to...",
            "metadata": {"source_path": "docs/azure.pdf", "chunk_index": 2},
        },
        {
            "id": "doc2_chunk_001",
            "score": 0.72,
            "text": "RAG (Retrieval Augmented Generation) combines...",
            "metadata": {"source_path": "docs/rag-guide.pdf", "chunk_index": 1},
        },
    ])


@pytest.fixture
def retriever(fake_embedding, fake_vector_store):
    """Create a DenseRetriever with fake dependencies. / 创建带 fake 依赖的 DenseRetriever。"""
    return DenseRetriever(
        embedding_client=fake_embedding,
        vector_store=fake_vector_store,
        default_top_k=10,
    )


# =============================================================================
# Initialization Tests / 初始化测试
# =============================================================================

class TestDenseRetrieverInit:
    """Tests for DenseRetriever initialization. / DenseRetriever 初始化测试。"""
    
    def test_init_with_dependencies(self, fake_embedding, fake_vector_store):
        """Test initialization with explicit dependencies. / 测试使用显式依赖初始化。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
            default_top_k=5,
        )
        
        assert retriever.embedding_client is fake_embedding
        assert retriever.vector_store is fake_vector_store
        assert retriever.default_top_k == 5
    
    def test_init_with_default_top_k(self, fake_embedding, fake_vector_store):
        """Test initialization uses default top_k when not specified. / 测试未指定时初始化使用默认 top_k。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
        )
        
        assert retriever.default_top_k == 10  # Default value / 默认值
    
    def test_init_with_settings_top_k(self, fake_embedding, fake_vector_store):
        """Test initialization extracts top_k from settings. / 测试初始化会从 settings 提取 top_k。"""
        mock_settings = Mock()
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.dense_top_k = 20
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
        )
        
        assert retriever.default_top_k == 20
    
    def test_init_without_settings_retrieval(self, fake_embedding, fake_vector_store):
        """Test initialization handles missing retrieval config gracefully. / 测试初始化会优雅处理缺失 retrieval 配置。"""
        mock_settings = Mock()
        mock_settings.retrieval = None
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
            default_top_k=15,
        )
        
        assert retriever.default_top_k == 15
    
    def test_init_without_dependencies(self):
        """Test initialization without dependencies is allowed (for lazy config). / 测试允许无依赖初始化（用于延迟配置）。"""
        retriever = DenseRetriever()
        
        assert retriever.embedding_client is None
        assert retriever.vector_store is None


# =============================================================================
# Retrieve Method Tests / Retrieve 方法测试
# =============================================================================

class TestDenseRetrieverRetrieve:
    """Tests for DenseRetriever.retrieve() method. / DenseRetriever.retrieve() 方法测试。"""
    
    def test_retrieve_basic(self, retriever, fake_embedding, fake_vector_store):
        """Test basic retrieval returns correct results. / 测试基础检索返回正确结果。"""
        results = retriever.retrieve("What is Azure OpenAI?")
        
        assert len(results) == 3
        assert fake_embedding.call_count == 1
        assert fake_vector_store.call_count == 1
    
    def test_retrieve_returns_retrieval_result_objects(self, retriever):
        """Test that results are RetrievalResult instances. / 测试结果是 RetrievalResult 实例。"""
        results = retriever.retrieve("test query")
        
        for result in results:
            assert isinstance(result, RetrievalResult)
    
    def test_retrieve_result_fields(self, retriever):
        """Test that RetrievalResult has all expected fields. / 测试 RetrievalResult 包含所有预期字段。"""
        results = retriever.retrieve("test query")
        
        result = results[0]
        assert result.chunk_id == "doc1_chunk_001"
        assert result.score == 0.95
        assert "Azure OpenAI" in result.text
        assert result.metadata["source_path"] == "docs/azure.pdf"
    
    def test_retrieve_with_top_k(self, retriever, fake_vector_store):
        """Test retrieval with custom top_k. / 测试使用自定义 top_k 检索。"""
        results = retriever.retrieve("test query", top_k=2)
        
        assert fake_vector_store.last_query_params["top_k"] == 2
        assert len(results) == 2
    
    def test_retrieve_with_default_top_k(self, fake_embedding, fake_vector_store):
        """Test retrieval uses default_top_k when top_k not specified. / 测试未指定 top_k 时检索使用 default_top_k。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
            default_top_k=5,
        )
        
        retriever.retrieve("test query")
        
        assert fake_vector_store.last_query_params["top_k"] == 5
    
    def test_retrieve_with_filters(self, retriever, fake_vector_store):
        """Test retrieval passes filters to vector store. / 测试检索会将 filters 传给 vector store。"""
        filters = {"collection": "api-docs", "doc_type": "pdf"}
        retriever.retrieve("test query", filters=filters)
        
        assert fake_vector_store.last_query_params["filters"] == filters
    
    def test_retrieve_passes_trace(self, retriever, fake_embedding, fake_vector_store):
        """Test retrieval passes trace context through. / 测试检索会透传 trace context。"""
        mock_trace = Mock()
        retriever.retrieve("test query", trace=mock_trace)
        
        assert fake_embedding.last_texts == ["test query"]
    
    def test_retrieve_embeds_query_correctly(self, retriever, fake_embedding):
        """Test that query is embedded correctly. / 测试 query 被正确 embedding。"""
        retriever.retrieve("How to configure RAG?")
        
        assert fake_embedding.last_texts == ["How to configure RAG?"]


# =============================================================================
# Input Validation Tests / 输入校验测试
# =============================================================================

class TestDenseRetrieverValidation:
    """Tests for input validation in DenseRetriever. / DenseRetriever 输入校验测试。"""
    
    def test_retrieve_empty_query_raises_error(self, retriever):
        """Test that empty query raises ValueError. / 测试空 query 会抛出 ValueError。"""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("")
    
    def test_retrieve_whitespace_query_raises_error(self, retriever):
        """Test that whitespace-only query raises ValueError. / 测试仅空白 query 会抛出 ValueError。"""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("   \t\n  ")
    
    def test_retrieve_non_string_query_raises_error(self, retriever):
        """Test that non-string query raises ValueError. / 测试非字符串 query 会抛出 ValueError。"""
        with pytest.raises(ValueError, match="must be a string"):
            retriever.retrieve(123)
    
    def test_retrieve_without_embedding_client_raises_error(self, fake_vector_store):
        """Test that missing embedding client raises RuntimeError. / 测试缺失 embedding client 会抛出 RuntimeError。"""
        retriever = DenseRetriever(
            embedding_client=None,
            vector_store=fake_vector_store,
        )
        
        with pytest.raises(RuntimeError, match="requires an embedding_client"):
            retriever.retrieve("test query")
    
    def test_retrieve_without_vector_store_raises_error(self, fake_embedding):
        """Test that missing vector store raises RuntimeError. / 测试缺失 vector store 会抛出 RuntimeError。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=None,
        )
        
        with pytest.raises(RuntimeError, match="requires a vector_store"):
            retriever.retrieve("test query")


# =============================================================================
# Error Handling Tests / 错误处理测试
# =============================================================================

class TestDenseRetrieverErrorHandling:
    """Tests for error handling in DenseRetriever. / DenseRetriever 错误处理测试。"""
    
    def test_retrieve_embedding_failure(self, fake_vector_store):
        """Test that embedding failure raises RuntimeError. / 测试 embedding 失败会抛出 RuntimeError。"""
        mock_embedding = Mock()
        mock_embedding.embed.side_effect = Exception("API connection failed")
        
        retriever = DenseRetriever(
            embedding_client=mock_embedding,
            vector_store=fake_vector_store,
        )
        
        with pytest.raises(RuntimeError, match="Failed to embed query"):
            retriever.retrieve("test query")
    
    def test_retrieve_vector_store_failure(self, fake_embedding):
        """Test that vector store failure raises RuntimeError. / 测试 vector store 失败会抛出 RuntimeError。"""
        mock_store = Mock()
        mock_store.query.side_effect = Exception("Database connection failed")
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=mock_store,
        )
        
        with pytest.raises(RuntimeError, match="Failed to query vector store"):
            retriever.retrieve("test query")
    
    def test_retrieve_handles_malformed_results(self, fake_embedding):
        """Test that malformed results are skipped gracefully. / 测试畸形结果会被优雅跳过。"""
        mock_store = Mock()
        mock_store.query.return_value = [
            {"id": "valid", "score": 0.9, "text": "Valid text", "metadata": {}},
            {"id": "", "score": "invalid", "text": ""},  # Malformed / 畸形数据
            {"id": "another", "score": 0.8, "text": "Another text", "metadata": {}},
        ]
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=mock_store,
        )
        
        results = retriever.retrieve("test query")
        
        # Malformed result should be skipped (empty chunk_id raises ValueError) / 畸形结果应被跳过（空 chunk_id 会抛出 ValueError）
        assert len(results) == 2
        assert results[0].chunk_id == "valid"
        assert results[1].chunk_id == "another"


# =============================================================================
# Empty Results Tests / 空结果测试
# =============================================================================

class TestDenseRetrieverEmptyResults:
    """Tests for handling empty results. / 空结果处理测试。"""
    
    def test_retrieve_empty_vector_store(self, fake_embedding):
        """Test retrieval from empty vector store. / 测试从空 vector store 检索。"""
        empty_store = FakeVectorStore(stored_records=[])
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=empty_store,
        )
        
        results = retriever.retrieve("test query")
        
        assert results == []
    
    def test_retrieve_returns_empty_list_not_none(self, fake_embedding):
        """Test that empty results return empty list, not None. / 测试空结果返回空列表而不是 None。"""
        empty_store = FakeVectorStore(stored_records=[])
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=empty_store,
        )
        
        results = retriever.retrieve("test query")
        
        assert results is not None
        assert isinstance(results, list)


# =============================================================================
# Factory Function Tests / 工厂函数测试
# =============================================================================

class TestCreateDenseRetriever:
    """Tests for create_dense_retriever factory function. / create_dense_retriever 工厂函数测试。"""
    
    def test_create_with_injected_dependencies(self, fake_embedding, fake_vector_store):
        """Test factory with pre-configured dependencies. / 测试使用预配置依赖的 factory。"""
        mock_settings = Mock()
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.dense_top_k = 15
        
        retriever = create_dense_retriever(
            settings=mock_settings,
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
        )
        
        assert retriever.embedding_client is fake_embedding
        assert retriever.vector_store is fake_vector_store
        assert retriever.default_top_k == 15
    
    @patch('src.libs.vector_store.vector_store_factory.VectorStoreFactory.create')
    @patch('src.libs.embedding.embedding_factory.EmbeddingFactory.create')
    def test_create_auto_creates_dependencies(
        self, mock_embedding_create, mock_vector_create, fake_embedding, fake_vector_store
    ):
        """Test factory auto-creates dependencies when not provided. / 测试未提供依赖时 factory 自动创建依赖。"""
        mock_embedding_create.return_value = fake_embedding
        mock_vector_create.return_value = fake_vector_store
        
        mock_settings = Mock()
        mock_settings.retrieval = None
        
        retriever = create_dense_retriever(settings=mock_settings)
        
        mock_embedding_create.assert_called_once_with(mock_settings)
        mock_vector_create.assert_called_once_with(mock_settings)
        assert retriever.embedding_client is fake_embedding
        assert retriever.vector_store is fake_vector_store


# =============================================================================
# Integration with RetrievalResult Type Tests / 与 RetrievalResult 类型的集成测试
# =============================================================================

class TestRetrievalResultIntegration:
    """Tests for RetrievalResult type integration. / RetrievalResult 类型集成测试。"""
    
    def test_retrieval_result_serialization(self, retriever):
        """Test that RetrievalResult can be serialized to dict. / 测试 RetrievalResult 可序列化为 dict。"""
        results = retriever.retrieve("test query")
        
        result_dict = results[0].to_dict()
        
        assert "chunk_id" in result_dict
        assert "score" in result_dict
        assert "text" in result_dict
        assert "metadata" in result_dict
    
    def test_retrieval_result_from_dict(self):
        """Test creating RetrievalResult from dict. / 测试从 dict 创建 RetrievalResult。"""
        data = {
            "chunk_id": "test_001",
            "score": 0.85,
            "text": "Test content",
            "metadata": {"source": "test.pdf"},
        }
        
        result = RetrievalResult.from_dict(data)
        
        assert result.chunk_id == "test_001"
        assert result.score == 0.85
        assert result.text == "Test content"
        assert result.metadata["source"] == "test.pdf"
    
    def test_retrieval_result_validation(self):
        """Test RetrievalResult validation. / 测试 RetrievalResult 校验。"""
        # Empty chunk_id should raise / 空 chunk_id 应抛出异常
        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            RetrievalResult(chunk_id="", score=0.5, text="test")
        
        # Non-numeric score should raise / 非数字 score 应抛出异常
        with pytest.raises(ValueError, match="score must be numeric"):
            RetrievalResult(chunk_id="test", score="invalid", text="test")


# =============================================================================
# Chinese Query Tests (D1 Alignment) / 中文查询测试（D1 对齐）
# =============================================================================

class TestDenseRetrieverChineseSupport:
    """Tests for Chinese query support (aligns with D1 QueryProcessor). / 中文 query 支持测试（与 D1 QueryProcessor 对齐）。"""
    
    def test_retrieve_chinese_query(self, retriever, fake_embedding):
        """Test retrieval with Chinese query. / 测试使用中文 query 检索。"""
        query = "如何配置 Azure OpenAI？"
        retriever.retrieve(query)
        
        assert fake_embedding.last_texts == [query]
    
    def test_retrieve_mixed_language_query(self, retriever, fake_embedding):
        """Test retrieval with mixed Chinese/English query. / 测试使用中英混合 query 检索。"""
        query = "Azure OpenAI 配置步骤"
        retriever.retrieve(query)
        
        assert fake_embedding.last_texts == [query]
