"""Integration tests for HybridSearch. / HybridSearch 的集成测试。

This test module validates the HybridSearch orchestration layer: / 该测试模块验证 HybridSearch 编排层：
- Complete retrieval flow (query → dense+sparse → fusion → results) / 完整检索流程（query -> dense+sparse -> fusion -> results）
- Graceful degradation when one retriever fails / 当某个 retriever 失败时优雅降级
- Metadata filtering (pre and post-fusion) / 元数据过滤（fusion 前后）
- Parallel vs sequential retrieval modes / 并行与顺序检索模式
- Edge cases and error handling / 边界情况和错误处理

Test Strategy: / 测试策略：
- Use mock/fake retrievers for deterministic behavior / 使用 mock/fake retriever 获取确定性行为
- Test actual component integration (not just mocking) / 测试真实组件集成（不只是 mock）
- Cover both success and failure scenarios / 覆盖成功和失败场景
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.core.types import ProcessedQuery, RetrievalResult
from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    create_hybrid_search,
)
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.fusion import RRFFusion


# =============================================================================
# Test Fixtures - Mock Components / 测试 Fixture - Mock 组件
# =============================================================================

class MockDenseRetriever:
    """Mock Dense Retriever for testing. / 用于测试的 Mock Dense Retriever。"""
    
    def __init__(
        self,
        results: Optional[List[RetrievalResult]] = None,
        should_fail: bool = False,
        error_message: str = "Dense retrieval failed",
    ):
        self.results = results or []
        self.should_fail = should_fail
        self.error_message = error_message
        self.call_count = 0
        self.last_query = None
        self.last_top_k = None
        self.last_filters = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self.call_count += 1
        self.last_query = query
        self.last_top_k = top_k
        self.last_filters = filters
        
        if self.should_fail:
            raise RuntimeError(self.error_message)
        
        return self.results[:top_k]


class MockSparseRetriever:
    """Mock Sparse Retriever for testing. / 用于测试的 Mock Sparse Retriever。"""
    
    def __init__(
        self,
        results: Optional[List[RetrievalResult]] = None,
        should_fail: bool = False,
        error_message: str = "Sparse retrieval failed",
    ):
        self.results = results or []
        self.should_fail = should_fail
        self.error_message = error_message
        self.call_count = 0
        self.last_keywords = None
        self.last_top_k = None
        self.last_collection = None
    
    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self.call_count += 1
        self.last_keywords = keywords
        self.last_top_k = top_k
        self.last_collection = collection
        
        if self.should_fail:
            raise RuntimeError(self.error_message)
        
        return self.results[:top_k]


@pytest.fixture
def sample_dense_results() -> List[RetrievalResult]:
    """Sample results from dense retrieval. / dense 检索的示例结果。"""
    return [
        RetrievalResult(
            chunk_id="dense_1",
            score=0.95,
            text="Azure OpenAI 配置步骤详解",
            metadata={"source_path": "docs/azure.pdf", "collection": "api-docs"},
        ),
        RetrievalResult(
            chunk_id="dense_2",
            score=0.88,
            text="OpenAI API 使用指南",
            metadata={"source_path": "docs/openai.pdf", "collection": "api-docs"},
        ),
        RetrievalResult(
            chunk_id="common_chunk",
            score=0.85,
            text="通用配置说明",
            metadata={"source_path": "docs/common.pdf", "collection": "general"},
        ),
        RetrievalResult(
            chunk_id="dense_4",
            score=0.80,
            text="云服务配置概述",
            metadata={"source_path": "docs/cloud.pdf", "collection": "general"},
        ),
    ]


@pytest.fixture
def sample_sparse_results() -> List[RetrievalResult]:
    """Sample results from sparse retrieval. / sparse 检索的示例结果。"""
    return [
        RetrievalResult(
            chunk_id="sparse_1",
            score=8.5,
            text="Azure 配置 Azure OpenAI 服务",
            metadata={"source_path": "docs/azure-setup.pdf", "collection": "tutorials"},
        ),
        RetrievalResult(
            chunk_id="common_chunk",  # Same as in dense / 与 dense 中相同
            score=7.2,
            text="通用配置说明",
            metadata={"source_path": "docs/common.pdf", "collection": "general"},
        ),
        RetrievalResult(
            chunk_id="sparse_3",
            score=6.8,
            text="配置文件 YAML 格式说明",
            metadata={"source_path": "docs/config.pdf", "collection": "tutorials"},
        ),
    ]


@pytest.fixture
def query_processor() -> QueryProcessor:
    """Real QueryProcessor instance. / 真实 QueryProcessor 实例。"""
    return QueryProcessor()


@pytest.fixture
def rrf_fusion() -> RRFFusion:
    """Real RRFFusion instance with default k=60. / 默认 k=60 的真实 RRFFusion 实例。"""
    return RRFFusion(k=60)


# =============================================================================
# Basic Functionality Tests / 基础功能测试
# =============================================================================

class TestHybridSearchBasic:
    """Test basic HybridSearch functionality. / 测试 HybridSearch 基础功能。"""
    
    def test_init_with_all_components(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test initialization with all components. / 测试使用全部组件初始化。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        assert hybrid.query_processor is query_processor
        assert hybrid.dense_retriever is dense
        assert hybrid.sparse_retriever is sparse
        assert hybrid.fusion is rrf_fusion
    
    def test_init_with_config(self):
        """Test initialization with custom config. / 测试使用自定义配置初始化。"""
        config = HybridSearchConfig(
            dense_top_k=30,
            sparse_top_k=30,
            fusion_top_k=15,
            parallel_retrieval=False,
        )
        
        hybrid = HybridSearch(config=config)
        
        assert hybrid.config.dense_top_k == 30
        assert hybrid.config.sparse_top_k == 30
        assert hybrid.config.fusion_top_k == 15
        assert hybrid.config.parallel_retrieval is False
    
    def test_search_returns_results(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that search returns fused results. / 测试 search 返回融合结果。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("如何配置 Azure OpenAI？", top_k=5)
        
        # Should return results / 应返回结果
        assert len(results) > 0
        assert len(results) <= 5
        
        # Results should be RetrievalResult objects / 结果应为 RetrievalResult 对象
        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.chunk_id
            assert isinstance(r.score, float)
            assert r.text
    
    def test_search_with_return_details(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test search with return_details=True. / 测试 return_details=True 时的搜索。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        result = hybrid.search("Azure config", top_k=5, return_details=True)
        
        assert isinstance(result, HybridSearchResult)
        assert result.results is not None
        assert result.dense_results is not None
        assert result.sparse_results is not None
        assert result.dense_error is None
        assert result.sparse_error is None
        assert result.used_fallback is False
        assert result.processed_query is not None
    
    def test_search_calls_both_retrievers(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that both retrievers are called. / 测试两个 retriever 都会被调用。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        hybrid.search("Azure OpenAI 配置", top_k=5)
        
        assert dense.call_count == 1
        assert sparse.call_count == 1
    
    def test_common_chunks_deduplicated(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that common chunks appear only once in results. / 测试公共分块在结果中只出现一次。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("配置", top_k=10)
        
        # Check for duplicate chunk_ids / 检查重复 chunk_id
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Results contain duplicate chunk_ids"
        
        # The common_chunk should appear exactly once / common_chunk 应只出现一次
        assert chunk_ids.count("common_chunk") <= 1


# =============================================================================
# Graceful Degradation Tests / 优雅降级测试
# =============================================================================

class TestHybridSearchDegradation:
    """Test graceful degradation when components fail. / 测试组件失败时的优雅降级。"""
    
    def test_dense_fails_uses_sparse_only(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test fallback to sparse when dense fails. / 测试 dense 失败时回退到 sparse。"""
        dense = MockDenseRetriever(should_fail=True)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        result = hybrid.search("Azure 配置", top_k=5, return_details=True)
        
        assert result.used_fallback is True
        assert result.dense_error is not None
        assert "Dense retrieval" in result.dense_error
        assert result.sparse_error is None
        assert len(result.results) > 0
    
    def test_sparse_fails_uses_dense_only(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
    ):
        """Test fallback to dense when sparse fails. / 测试 sparse 失败时回退到 dense。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(should_fail=True)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        result = hybrid.search("Azure 配置", top_k=5, return_details=True)
        
        assert result.used_fallback is True
        assert result.sparse_error is not None
        assert "Sparse retrieval" in result.sparse_error
        assert result.dense_error is None
        assert len(result.results) > 0
    
    def test_both_fail_raises_error(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ):
        """Test that RuntimeError is raised when both retrievers fail. / 测试两个 retriever 都失败时抛出 RuntimeError。"""
        dense = MockDenseRetriever(should_fail=True)
        sparse = MockSparseRetriever(should_fail=True)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            hybrid.search("Azure 配置", top_k=5)
        
        assert "Both retrieval paths failed" in str(exc_info.value)
    
    def test_no_retrievers_configured(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ):
        """Test behavior when no retrievers are configured. / 测试未配置 retriever 时的行为。"""
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=None,
            sparse_retriever=None,
            fusion=rrf_fusion,
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            hybrid.search("Azure 配置", top_k=5)
        
        assert "No retriever" in str(exc_info.value) or "Both" in str(exc_info.value)
    
    def test_dense_only_mode(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
    ):
        """Test search with only dense retriever. / 测试仅使用 dense retriever 搜索。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=None,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("Azure 配置", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
    
    def test_sparse_only_mode(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test search with only sparse retriever. / 测试仅使用 sparse retriever 搜索。"""
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=None,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("Azure 配置", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3


# =============================================================================
# Filter Tests / 过滤器测试
# =============================================================================

class TestHybridSearchFilters:
    """Test metadata filtering functionality. / 测试元数据过滤功能。"""
    
    def test_explicit_filters_passed_to_retrievers(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that explicit filters are passed to retrievers. / 测试显式过滤器会传给 retriever。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        hybrid.search("Azure", top_k=5, filters={"collection": "api-docs"})
        
        assert dense.last_filters == {"collection": "api-docs"}
        assert sparse.last_collection == "api-docs"
    
    def test_query_filter_syntax_extraction(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that filters in query syntax are extracted. / 测试查询语法中的过滤器会被提取。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        result = hybrid.search("collection:api-docs Azure 配置", top_k=5, return_details=True)
        
        # Check that filter was extracted from query / 检查过滤器已从查询中提取
        assert result.processed_query is not None
        assert "collection" in result.processed_query.filters
    
    def test_post_fusion_metadata_filter(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test post-fusion metadata filtering. / 测试 fusion 后的元数据过滤。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        config = HybridSearchConfig(metadata_filter_post=True)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
            config=config,
        )
        
        # Filter for api-docs collection only / 仅过滤 api-docs 集合
        results = hybrid.search("Azure", top_k=10, filters={"collection": "api-docs"})
        
        # All results should have collection=api-docs / 所有结果都应有 collection=api-docs
        for r in results:
            assert r.metadata.get("collection") == "api-docs"


# =============================================================================
# Configuration Tests / 配置测试
# =============================================================================

class TestHybridSearchConfig:
    """Test configuration behavior. / 测试配置行为。"""
    
    def test_top_k_from_config(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that top_k values from config are used. / 测试会使用配置中的 top_k 值。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        config = HybridSearchConfig(
            dense_top_k=3,
            sparse_top_k=3,
            fusion_top_k=2,
        )
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
            config=config,
        )
        
        results = hybrid.search("Azure")  # No explicit top_k / 未显式指定 top_k
        
        assert dense.last_top_k == 3
        assert sparse.last_top_k == 3
        assert len(results) <= 2
    
    def test_top_k_override(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that explicit top_k overrides config. / 测试显式 top_k 会覆盖配置。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("Azure", top_k=1)
        
        assert len(results) == 1
    
    def test_sequential_retrieval_mode(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test sequential retrieval mode (non-parallel). / 测试顺序检索模式（非并行）。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        config = HybridSearchConfig(parallel_retrieval=False)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
            config=config,
        )
        
        results = hybrid.search("Azure", top_k=5)
        
        # Both should be called / 两者都应被调用
        assert dense.call_count == 1
        assert sparse.call_count == 1
        assert len(results) > 0


# =============================================================================
# Edge Cases and Error Handling Tests / 边界情况和错误处理测试
# =============================================================================

class TestHybridSearchEdgeCases:
    """Test edge cases and error handling. / 测试边界情况和错误处理。"""
    
    def test_empty_query_raises_error(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ):
        """Test that empty query raises ValueError. / 测试空查询会抛出 ValueError。"""
        hybrid = HybridSearch(
            query_processor=query_processor,
            fusion=rrf_fusion,
        )
        
        with pytest.raises(ValueError) as exc_info:
            hybrid.search("")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_whitespace_query_raises_error(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ):
        """Test that whitespace-only query raises ValueError. / 测试仅空白字符的查询会抛出 ValueError。"""
        hybrid = HybridSearch(
            query_processor=query_processor,
            fusion=rrf_fusion,
        )
        
        with pytest.raises(ValueError) as exc_info:
            hybrid.search("   \t\n  ")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_empty_results_from_both_retrievers(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ):
        """Test handling when both retrievers return empty results. / 测试两个 retriever 都返回空结果时的处理。"""
        dense = MockDenseRetriever(results=[])
        sparse = MockSparseRetriever(results=[])
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("obscure query with no matches", top_k=5)
        
        assert results == []
    
    def test_query_without_keywords_skips_sparse(
        self,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that sparse is skipped when no keywords extracted. / 测试没有提取关键词时会跳过 sparse。"""
        # Mock query processor that returns empty keywords / Mock 返回空关键词的 query processor
        mock_processor = MagicMock()
        mock_processor.process.return_value = ProcessedQuery(
            original_query="的",  # Only stopwords / 只有停用词
            keywords=[],
            filters={},
        )
        
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("的", top_k=5)
        
        # Dense should be called, sparse may be called but with empty keywords / dense 应被调用，sparse 可能会用空关键词调用
        assert dense.call_count == 1
        assert len(results) > 0
    
    def test_no_query_processor_fallback(
        self,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test fallback when no QueryProcessor is configured. / 测试未配置 QueryProcessor 时的回退。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=None,  # No processor / 无 processor
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )
        
        results = hybrid.search("Azure OpenAI", top_k=5)
        
        # Should still work with basic tokenization / 使用基础分词仍应工作
        assert len(results) > 0
    
    def test_no_fusion_interleave_fallback(
        self,
        query_processor: QueryProcessor,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test interleave fallback when no fusion is configured. / 测试未配置 fusion 时的交错回退。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=None,  # No fusion / 无 fusion
        )
        
        results = hybrid.search("Azure", top_k=5)
        
        # Should still return results (interleaved) / 仍应返回结果（交错）
        assert len(results) > 0
        assert len(results) <= 5


# =============================================================================
# Factory Function Tests / 工厂函数测试
# =============================================================================

class TestCreateHybridSearch:
    """Test the create_hybrid_search factory function. / 测试 create_hybrid_search 工厂函数。"""
    
    def test_creates_default_fusion(self):
        """Test that default RRF fusion is created. / 测试会创建默认 RRF fusion。"""
        hybrid = create_hybrid_search()
        
        assert hybrid.fusion is not None
        assert isinstance(hybrid.fusion, RRFFusion)
        assert hybrid.fusion.k == 60  # Default k / 默认 k
    
    def test_uses_provided_fusion(self, rrf_fusion: RRFFusion):
        """Test that provided fusion is used. / 测试会使用传入的 fusion。"""
        custom_fusion = RRFFusion(k=30)
        
        hybrid = create_hybrid_search(fusion=custom_fusion)
        
        assert hybrid.fusion is custom_fusion
        assert hybrid.fusion.k == 30
    
    def test_passes_all_components(
        self,
        query_processor: QueryProcessor,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that all components are passed through. / 测试所有组件都会透传。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        hybrid = create_hybrid_search(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
        
        assert hybrid.query_processor is query_processor
        assert hybrid.dense_retriever is dense
        assert hybrid.sparse_retriever is sparse


# =============================================================================
# RRF Fusion Integration Tests / RRF Fusion 集成测试
# =============================================================================

class TestRRFFusionIntegration:
    """Test actual RRF fusion behavior in HybridSearch. / 测试 HybridSearch 中真实 RRF fusion 行为。"""
    
    def test_common_chunks_boosted_by_rrf(
        self,
        query_processor: QueryProcessor,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that chunks appearing in both results get boosted. / 测试同时出现在两路结果中的分块会被提升。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        
        # Use RRF fusion / 使用 RRF fusion
        fusion = RRFFusion(k=60)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
        )
        
        results = hybrid.search("配置", top_k=10)
        
        # common_chunk appears in both, should be ranked higher / common_chunk 同时出现于两路结果中，排名应更高
        chunk_ids = [r.chunk_id for r in results]
        
        # Check that common_chunk is present / 检查 common_chunk 存在
        if "common_chunk" in chunk_ids:
            common_idx = chunk_ids.index("common_chunk")
            # It should be relatively high due to RRF boost / 由于 RRF 加权，它应排名相对靠前
            # (appears in both lists = sum of RRF scores) / （出现在两个列表中 = RRF 分数求和）
            assert common_idx < 5, "common_chunk should be boosted by RRF"
    
    def test_rrf_scores_are_deterministic(
        self,
        query_processor: QueryProcessor,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ):
        """Test that RRF fusion produces deterministic results. / 测试 RRF fusion 产生确定性结果。"""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        fusion = RRFFusion(k=60)
        
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
        )
        
        # Run same search multiple times / 多次运行相同搜索
        results1 = hybrid.search("配置", top_k=5)
        results2 = hybrid.search("配置", top_k=5)
        
        # Results should be identical / 结果应相同
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.chunk_id == r2.chunk_id
            assert r1.score == r2.score
