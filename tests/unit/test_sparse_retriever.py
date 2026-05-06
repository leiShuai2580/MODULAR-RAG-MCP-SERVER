"""Unit tests for SparseRetriever. / SparseRetriever 的单元测试。

Tests cover: / 测试覆盖：
- Initialization and configuration / 初始化和配置
- Keyword validation / 关键词校验
- Dependency validation / 依赖校验
- BM25 query + vector store get_by_ids integration / BM25 query + vector store get_by_ids 集成
- Result merging and transformation / 结果合并和转换
- Error handling and edge cases / 错误处理和边界情况
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.core.query_engine.sparse_retriever import SparseRetriever, create_sparse_retriever
from src.core.types import RetrievalResult


# ============================================================================
# Mock Classes / Mock 类
# ============================================================================

class MockBM25Indexer:
    """Mock BM25 indexer for testing. / 用于测试的 Mock BM25 indexer。"""
    
    def __init__(self, index_dir: str = "data/db/bm25"):
        self.index_dir = index_dir
        self._index = {}
        self._metadata = {}
        self._loaded_collection = None
        
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        """Simulate loading an index. / 模拟加载索引。"""
        self._loaded_collection = collection
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Return mock BM25 results. / 返回 mock BM25 结果。"""
        # Return predetermined results based on query terms / 基于 query terms 返回预设结果
        results = [
            {"chunk_id": "chunk_001", "score": 2.5},
            {"chunk_id": "chunk_002", "score": 1.8},
            {"chunk_id": "chunk_003", "score": 1.2},
        ]
        return results[:top_k]


class MockBM25IndexerEmpty:
    """Mock BM25 indexer that returns empty results. / 返回空结果的 Mock BM25 indexer。"""
    
    def __init__(self):
        self._metadata = {}
    
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return []


class MockBM25IndexerFailing:
    """Mock BM25 indexer that fails on load or query. / load 或 query 时失败的 Mock BM25 indexer。"""
    
    def __init__(self, fail_on_load: bool = False, fail_on_query: bool = False):
        self.fail_on_load = fail_on_load
        self.fail_on_query = fail_on_query
        self._metadata = {}
    
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        if self.fail_on_load:
            raise RuntimeError("Simulated load failure")
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if self.fail_on_query:
            raise RuntimeError("Simulated query failure")
        return []


class MockVectorStore:
    """Mock vector store for testing. / 用于测试的 Mock vector store。"""
    
    def __init__(self):
        self._records = {
            "chunk_001": {
                "id": "chunk_001",
                "text": "This is the first chunk about machine learning.",
                "metadata": {"source_path": "doc1.pdf", "chunk_index": 0}
            },
            "chunk_002": {
                "id": "chunk_002",
                "text": "This is the second chunk about neural networks.",
                "metadata": {"source_path": "doc1.pdf", "chunk_index": 1}
            },
            "chunk_003": {
                "id": "chunk_003",
                "text": "This is the third chunk about deep learning.",
                "metadata": {"source_path": "doc2.pdf", "chunk_index": 0}
            },
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return mock records by IDs. / 按 IDs 返回 mock records。"""
        results = []
        for id_ in ids:
            if id_ in self._records:
                results.append(self._records[id_])
            else:
                results.append({})  # Not found / 未找到
        return results


class MockVectorStoreFailing:
    """Mock vector store that fails on get_by_ids. / get_by_ids 时失败的 Mock vector store。"""
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise RuntimeError("Simulated vector store failure")


class MockSettings:
    """Mock settings for testing. / 用于测试的 Mock settings。"""
    
    def __init__(self, sparse_top_k: int = 15):
        self.retrieval = MagicMock()
        self.retrieval.sparse_top_k = sparse_top_k


# ============================================================================
# Test: Initialization / 测试：初始化
# ============================================================================

class TestSparseRetrieverInit:
    """Tests for SparseRetriever initialization. / SparseRetriever 初始化测试。"""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters. / 测试使用默认参数初始化。"""
        retriever = SparseRetriever()
        
        assert retriever.bm25_indexer is None
        assert retriever.vector_store is None
        assert retriever.default_top_k == 10
        assert retriever.default_collection == "default"
    
    def test_init_with_custom_top_k(self):
        """Test initialization with custom default_top_k. / 测试使用自定义 default_top_k 初始化。"""
        retriever = SparseRetriever(default_top_k=20)
        
        assert retriever.default_top_k == 20
    
    def test_init_with_settings(self):
        """Test initialization extracts top_k from settings. / 测试初始化会从 settings 提取 top_k。"""
        settings = MockSettings(sparse_top_k=25)
        retriever = SparseRetriever(settings=settings)
        
        assert retriever.default_top_k == 25
    
    def test_init_with_dependencies(self):
        """Test initialization with injected dependencies. / 测试使用注入依赖初始化。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        
        retriever = SparseRetriever(
            bm25_indexer=bm25,
            vector_store=vs,
            default_collection="my_collection"
        )
        
        assert retriever.bm25_indexer is bm25
        assert retriever.vector_store is vs
        assert retriever.default_collection == "my_collection"


# ============================================================================
# Test: Input Validation / 测试：输入校验
# ============================================================================

class TestSparseRetrieverValidation:
    """Tests for input validation. / 输入校验测试。"""
    
    def test_retrieve_raises_on_empty_keywords(self):
        """Test that empty keywords raises ValueError. / 测试空 keywords 会抛出 ValueError。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve([])
    
    def test_retrieve_raises_on_non_list_keywords(self):
        """Test that non-list keywords raises ValueError. / 测试非列表 keywords 会抛出 ValueError。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(ValueError, match="must be a list"):
            retriever.retrieve("not a list")  # type: ignore
    
    def test_retrieve_raises_without_bm25_indexer(self):
        """Test that missing bm25_indexer raises RuntimeError. / 测试缺失 bm25_indexer 会抛出 RuntimeError。"""
        vs = MockVectorStore()
        retriever = SparseRetriever(vector_store=vs)
        
        with pytest.raises(RuntimeError, match="requires a bm25_indexer"):
            retriever.retrieve(["keyword"])
    
    def test_retrieve_raises_without_vector_store(self):
        """Test that missing vector_store raises RuntimeError. / 测试缺失 vector_store 会抛出 RuntimeError。"""
        bm25 = MockBM25Indexer()
        retriever = SparseRetriever(bm25_indexer=bm25)
        
        with pytest.raises(RuntimeError, match="requires a vector_store"):
            retriever.retrieve(["keyword"])


# ============================================================================
# Test: Basic Retrieval / 测试：基础检索
# ============================================================================

class TestSparseRetrieverRetrieve:
    """Tests for retrieve() method. / retrieve() 方法测试。"""
    
    def test_retrieve_returns_results(self):
        """Test basic retrieval returns RetrievalResult objects. / 测试基础检索返回 RetrievalResult 对象。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["machine", "learning"])
        
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_retrieve_results_have_correct_fields(self):
        """Test that results have all required fields. / 测试结果包含所有必需字段。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        first = results[0]
        assert first.chunk_id == "chunk_001"
        assert first.score == 2.5
        assert "machine learning" in first.text
        assert first.metadata["source_path"] == "doc1.pdf"
    
    def test_retrieve_respects_top_k(self):
        """Test that top_k parameter limits results. / 测试 top_k 参数会限制结果数量。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=2)
        
        assert len(results) == 2
    
    def test_retrieve_uses_default_top_k(self):
        """Test that default_top_k is used when top_k not specified. / 测试未指定 top_k 时使用 default_top_k。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(
            bm25_indexer=bm25, 
            vector_store=vs, 
            default_top_k=2
        )
        
        results = retriever.retrieve(["test"])
        
        assert len(results) == 2
    
    def test_retrieve_with_custom_collection(self):
        """Test retrieval with custom collection parameter. / 测试使用自定义 collection 参数检索。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # Should not raise, collection is passed to bm25_indexer / 不应抛出异常，collection 会传给 bm25_indexer
        results = retriever.retrieve(["test"], collection="custom_collection")
        
        assert len(results) > 0
    
    def test_retrieve_preserves_result_order(self):
        """Test that results maintain BM25 score ordering. / 测试结果保持 BM25 分数排序。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        # Results should be in descending score order (from BM25) / 结果应按分数降序排列（来自 BM25）
        assert results[0].score > results[1].score > results[2].score


# ============================================================================
# Test: Empty Results / 测试：空结果
# ============================================================================

class TestSparseRetrieverEmptyResults:
    """Tests for empty result handling. / 空结果处理测试。"""
    
    def test_retrieve_returns_empty_when_no_matches(self):
        """Test that no BM25 matches returns empty list. / 测试无 BM25 匹配时返回空列表。"""
        bm25 = MockBM25IndexerEmpty()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["nonexistent"])
        
        assert results == []
    
    def test_retrieve_returns_empty_when_index_not_loaded(self):
        """Test graceful handling when index cannot be loaded. / 测试索引无法加载时的优雅处理。"""
        bm25 = MockBM25IndexerFailing(fail_on_load=True)
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # Should return empty, not raise / 应返回空结果而不是抛出异常
        results = retriever.retrieve(["test"])
        
        assert results == []


# ============================================================================
# Test: Error Handling / 测试：错误处理
# ============================================================================

class TestSparseRetrieverErrorHandling:
    """Tests for error handling. / 错误处理测试。"""
    
    def test_retrieve_raises_on_bm25_query_failure(self):
        """Test that BM25 query failure raises RuntimeError. / 测试 BM25 query 失败会抛出 RuntimeError。"""
        bm25 = MockBM25IndexerFailing(fail_on_query=True)
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(RuntimeError, match="Failed to query BM25"):
            retriever.retrieve(["test"])
    
    def test_retrieve_raises_on_vector_store_failure(self):
        """Test that vector store failure raises RuntimeError. / 测试 vector store 失败会抛出 RuntimeError。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStoreFailing()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(RuntimeError, match="Failed to fetch records"):
            retriever.retrieve(["test"])


# ============================================================================
# Test: Missing Records / 测试：缺失 Records
# ============================================================================

class TestSparseRetrieverMissingRecords:
    """Tests for handling missing records. / 缺失 records 处理测试。"""
    
    def test_retrieve_skips_missing_records(self):
        """Test that missing records in vector store are skipped. / 测试 vector store 中缺失的 records 会被跳过。"""
        bm25 = MockBM25Indexer()
        
        # Create vector store with only partial records / 创建只包含部分 records 的 vector store
        vs = MockVectorStore()
        vs._records = {
            "chunk_001": {
                "id": "chunk_001",
                "text": "First chunk.",
                "metadata": {"source_path": "doc.pdf"}
            },
            # chunk_002 and chunk_003 are missing / chunk_002 和 chunk_003 缺失
        }
        
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])
        
        # Should only return the one found record / 应只返回找到的那条 record
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"


# ============================================================================
# Test: Result Transformation / 测试：结果转换
# ============================================================================

class TestSparseRetrieverResultTransformation:
    """Tests for result transformation. / 结果转换测试。"""
    
    def test_retrieve_result_is_serializable(self):
        """Test that results can be serialized to dict. / 测试结果可序列化为 dict。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=1)
        
        result_dict = results[0].to_dict()
        assert "chunk_id" in result_dict
        assert "score" in result_dict
        assert "text" in result_dict
        assert "metadata" in result_dict
    
    def test_retrieve_result_score_is_float(self):
        """Test that score is converted to float. / 测试 score 会转换为 float。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        for result in results:
            assert isinstance(result.score, float)


# ============================================================================
# Test: Factory Function / 测试：工厂函数
# ============================================================================

class TestCreateSparseRetriever:
    """Tests for create_sparse_retriever factory function. / create_sparse_retriever 工厂函数测试。"""
    
    def test_create_with_injected_dependencies(self):
        """Test factory with injected dependencies. / 测试使用注入依赖的 factory。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        settings = MockSettings()
        
        retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25,
            vector_store=vs,
        )
        
        assert retriever.bm25_indexer is bm25
        assert retriever.vector_store is vs


# ============================================================================
# Test: Index Loading / 测试：索引加载
# ============================================================================

class TestSparseRetrieverIndexLoading:
    """Tests for BM25 index loading behavior. / BM25 索引加载行为测试。"""
    
    def test_index_loaded_once_per_collection(self):
        """Test that index is only loaded once per collection. / 测试每个 collection 只加载一次索引。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # First call should load the index / 第一次调用应加载索引
        retriever.retrieve(["test"], collection="test_col")
        assert bm25._loaded_collection == "test_col"
        
        # Second call with same collection should not reload / 同一 collection 的第二次调用不应重新加载
        bm25._loaded_collection = "test_col"  # Mark as loaded / 标记为已加载
        retriever.retrieve(["test"], collection="test_col")
        assert bm25._loaded_collection == "test_col"
    
    def test_index_reloaded_for_different_collection(self):
        """Test that index is reloaded for different collection. / 测试不同 collection 会重新加载索引。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # Load first collection / 加载第一个 collection
        retriever.retrieve(["test"], collection="col_a")
        assert bm25._metadata.get("collection") == "col_a"
        
        # Load different collection / 加载不同 collection
        retriever.retrieve(["test"], collection="col_b")
        assert bm25._metadata.get("collection") == "col_b"


# ============================================================================
# Test: Integration with Real Types / 测试：与真实类型集成
# ============================================================================

class TestSparseRetrieverTypeIntegration:
    """Tests for integration with core types. / 与 core types 的集成测试。"""
    
    def test_retrieve_result_type_matches_dense_retriever(self):
        """Test that SparseRetriever returns same type as DenseRetriever. / 测试 SparseRetriever 返回与 DenseRetriever 相同的类型。"""
        from src.core.types import RetrievalResult as TypedRetrievalResult
        
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        for result in results:
            assert isinstance(result, TypedRetrievalResult)
    
    def test_retrieve_result_compatible_with_from_dict(self):
        """Test results can be recreated from dict. / 测试结果可从 dict 重建。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=1)
        result_dict = results[0].to_dict()
        
        # Recreate from dict / 从 dict 重建
        recreated = RetrievalResult.from_dict(result_dict)
        
        assert recreated.chunk_id == results[0].chunk_id
        assert recreated.score == results[0].score
        assert recreated.text == results[0].text
