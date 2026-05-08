"""Unit tests for CoreReranker fallback behavior. / CoreReranker fallback 行为的单元测试。

This module tests the CoreReranker class with focus on: / 本模块测试 CoreReranker 类，重点关注：
1. Normal reranking flow with mocked backend / 使用 mocked backend 的正常重排序流程
2. Fallback behavior when backend fails / backend 失败时的 fallback 行为
3. Configuration handling / 配置处理
4. Type conversion between RetrievalResult and reranker format / RetrievalResult 与 reranker 格式之间的类型转换
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.core.query_engine.reranker import (
    CoreReranker,
    RerankConfig,
    RerankError,
    RerankResult,
    create_core_reranker,
)
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


# =============================================================================
# Fixtures / Fixture
# =============================================================================

class MockSettings:
    """Mock Settings class for testing. / 用于测试的 Mock Settings 类。"""
    
    def __init__(
        self,
        enabled: bool = True,
        provider: str = "llm",
        top_k: int = 5,
    ):
        self.rerank = MagicMock()
        self.rerank.enabled = enabled
        self.rerank.provider = provider
        self.rerank.top_k = top_k


class FakeReranker(BaseReranker):
    """Fake reranker that returns candidates in reverse order. / 返回反向 candidates 顺序的 fake reranker。"""
    
    def __init__(self, settings: Any = None, **kwargs: Any):
        self.call_count = 0
        self.last_query = None
        self.last_candidates = None
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_query = query
        self.last_candidates = candidates
        
        # Reverse order and add rerank_score / 反转顺序并添加 rerank_score
        reranked = []
        for i, candidate in enumerate(reversed(candidates)):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = 3.0 - i * 0.5  # Descending scores / 降序分数
            reranked.append(candidate_copy)
        
        return reranked


class FailingReranker(BaseReranker):
    """Reranker that always fails. / 总是失败的 Reranker。"""
    
    def __init__(self, settings: Any = None, error_message: str = "Reranker failed", **kwargs: Any):
        self.error_message = error_message
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise RuntimeError(self.error_message)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing. / 创建用于测试的 mock settings。"""
    return MockSettings(enabled=True, provider="llm", top_k=5)


@pytest.fixture
def sample_results():
    """Create sample RetrievalResults for testing. / 创建用于测试的示例 RetrievalResults。"""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            score=0.9,
            text="First chunk about machine learning.",
            metadata={"source_path": "doc1.pdf", "page": 1},
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            score=0.8,
            text="Second chunk about deep learning.",
            metadata={"source_path": "doc1.pdf", "page": 2},
        ),
        RetrievalResult(
            chunk_id="chunk_003",
            score=0.7,
            text="Third chunk about neural networks.",
            metadata={"source_path": "doc2.pdf", "page": 1},
        ),
    ]


# =============================================================================
# Test: Initialization / 测试：初始化
# =============================================================================

class TestCoreRerankerInit:
    """Test CoreReranker initialization. / 测试 CoreReranker 初始化。"""
    
    def test_init_with_custom_reranker(self, mock_settings):
        """Test initialization with custom reranker backend. / 测试使用自定义 reranker backend 初始化。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        assert reranker._reranker is fake_reranker
        assert reranker.reranker_type == "fakereranker"
    
    def test_init_with_disabled_config(self):
        """Test initialization with disabled reranking. / 测试使用禁用 reranking 的配置初始化。"""
        settings = MockSettings(enabled=False, provider="none")
        reranker = CoreReranker(settings=settings)
        
        assert isinstance(reranker._reranker, NoneReranker)
        assert reranker.reranker_type == "none"
        assert not reranker.is_enabled
    
    def test_init_with_custom_config(self, mock_settings):
        """Test initialization with custom RerankConfig. / 测试使用自定义 RerankConfig 初始化。"""
        config = RerankConfig(enabled=True, top_k=10, timeout=60.0)
        fake_reranker = FakeReranker()
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        assert reranker.config.top_k == 10
        assert reranker.config.timeout == 60.0
    
    def test_init_extracts_config_from_settings(self, mock_settings):
        """Test that config is extracted from settings when not provided. / 测试未提供 config 时从 settings 提取配置。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        assert reranker.config.enabled is True
        assert reranker.config.top_k == 5


# =============================================================================
# Test: Normal Reranking Flow / 测试：正常重排序流程
# =============================================================================

class TestCoreRerankerNormalFlow:
    """Test normal reranking flow. / 测试正常重排序流程。"""
    
    def test_rerank_calls_backend(self, mock_settings, sample_results):
        """Test that reranking calls the backend reranker. / 测试 reranking 会调用 backend reranker。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert fake_reranker.call_count == 1
        assert fake_reranker.last_query == "test query"
        assert len(fake_reranker.last_candidates) == 3
    
    def test_rerank_returns_reranked_results(self, mock_settings, sample_results):
        """Test that reranked results are returned correctly. / 测试正确返回 reranked results。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert isinstance(result, RerankResult)
        assert len(result.results) == 3
        assert result.used_fallback is False
        assert result.reranker_type == "fakereranker"
    
    def test_rerank_preserves_original_data(self, mock_settings, sample_results):
        """Test that original result data is preserved. / 测试原始 result 数据会被保留。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        # FakeReranker reverses order, so first result should be chunk_003 / FakeReranker 会反转顺序，因此第一个结果应为 chunk_003
        first = result.results[0]
        assert first.chunk_id == "chunk_003"
        assert first.text == "Third chunk about neural networks."
        assert first.metadata["source_path"] == "doc2.pdf"
    
    def test_rerank_adds_score_metadata(self, mock_settings, sample_results):
        """Test that rerank scores are added to metadata. / 测试 rerank scores 会添加到 metadata。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        first = result.results[0]
        assert "original_score" in first.metadata
        assert "rerank_score" in first.metadata
        assert first.metadata["reranked"] is True
    
    def test_rerank_respects_top_k(self, mock_settings, sample_results):
        """Test that top_k limits results. / 测试 top_k 限制结果。"""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True, top_k=2)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        result = reranker.rerank("test query", sample_results)
        
        assert len(result.results) == 2
    
    def test_rerank_explicit_top_k_overrides_config(self, mock_settings, sample_results):
        """Test that explicit top_k parameter overrides config. / 测试显式 top_k 参数会覆盖 config。"""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True, top_k=5)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        result = reranker.rerank("test query", sample_results, top_k=1)
        
        assert len(result.results) == 1


# =============================================================================
# Test: Fallback Behavior / 测试：Fallback 行为
# =============================================================================

class TestCoreRerankerFallback:
    """Test fallback behavior when backend fails. / 测试 backend 失败时的 fallback 行为。"""
    
    def test_fallback_on_backend_error(self, mock_settings, sample_results):
        """Test that fallback returns original order on error. / 测试出错时 fallback 返回原始顺序。"""
        failing_reranker = FailingReranker(error_message="Backend crashed")
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.used_fallback is True
        assert "Backend crashed" in result.fallback_reason
        assert len(result.results) == 3
    
    def test_fallback_preserves_original_order(self, mock_settings, sample_results):
        """Test that fallback preserves original order. / 测试 fallback 保留原始顺序。"""
        failing_reranker = FailingReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.results[0].chunk_id == "chunk_001"
        assert result.results[1].chunk_id == "chunk_002"
        assert result.results[2].chunk_id == "chunk_003"
    
    def test_fallback_marks_results(self, mock_settings, sample_results):
        """Test that fallback results are marked appropriately. / 测试 fallback 结果会被适当标记。"""
        failing_reranker = FailingReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        for r in result.results:
            assert r.metadata.get("reranked") is False
            assert r.metadata.get("rerank_fallback") is True
    
    def test_fallback_respects_top_k(self, mock_settings, sample_results):
        """Test that fallback respects top_k. / 测试 fallback 遵守 top_k。"""
        failing_reranker = FailingReranker()
        config = RerankConfig(enabled=True, top_k=2, fallback_on_error=True)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=failing_reranker,
            config=config,
        )
        
        result = reranker.rerank("test query", sample_results)
        
        assert len(result.results) == 2
    
    def test_fallback_disabled_raises_error(self, mock_settings, sample_results):
        """Test that error is raised when fallback is disabled. / 测试禁用 fallback 时会抛出错误。"""
        failing_reranker = FailingReranker()
        config = RerankConfig(enabled=True, top_k=5, fallback_on_error=False)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=failing_reranker,
            config=config,
        )
        
        with pytest.raises(RerankError) as exc_info:
            reranker.rerank("test query", sample_results)
        
        assert "fallback disabled" in str(exc_info.value)
    
    def test_stores_original_order_on_fallback(self, mock_settings, sample_results):
        """Test that original order is stored in result. / 测试原始顺序会存入 result。"""
        failing_reranker = FailingReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=failing_reranker)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.original_order is not None
        assert len(result.original_order) == 3


# =============================================================================
# Test: Edge Cases / 测试：边界情况
# =============================================================================

class TestCoreRerankerEdgeCases:
    """Test edge cases. / 测试边界情况。"""
    
    def test_empty_results(self, mock_settings):
        """Test handling of empty results. / 测试处理空 results。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        result = reranker.rerank("test query", [])
        
        assert result.results == []
        assert result.used_fallback is False
        assert fake_reranker.call_count == 0  # Should not call backend / 不应调用 backend
    
    def test_single_result(self, mock_settings):
        """Test handling of single result. / 测试处理单个 result。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        single_result = [RetrievalResult(
            chunk_id="only_one",
            score=0.9,
            text="Single result",
            metadata={"source_path": "test.pdf"},
        )]
        
        result = reranker.rerank("test query", single_result)
        
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "only_one"
        assert fake_reranker.call_count == 0  # Should not call backend for single result / 单个 result 不应调用 backend
    
    def test_disabled_reranking(self, sample_results):
        """Test that disabled reranking returns original order. / 测试禁用 reranking 时返回原始顺序。"""
        settings = MockSettings(enabled=False, provider="none")
        reranker = CoreReranker(settings=settings)
        
        result = reranker.rerank("test query", sample_results)
        
        assert result.reranker_type == "none"
        assert result.results[0].chunk_id == "chunk_001"
        assert result.used_fallback is False
    
    def test_none_reranker_returns_top_k(self, sample_results):
        """Test that NoneReranker returns top_k results. / 测试 NoneReranker 返回 top_k 结果。"""
        settings = MockSettings(enabled=False, provider="none", top_k=2)
        reranker = CoreReranker(settings=settings)
        
        result = reranker.rerank("test query", sample_results)
        
        assert len(result.results) == 2


# =============================================================================
# Test: Type Conversion / 测试：类型转换
# =============================================================================

class TestTypeConversion:
    """Test type conversion between RetrievalResult and reranker format. / 测试 RetrievalResult 和 reranker 格式之间的类型转换。"""
    
    def test_results_to_candidates(self, mock_settings, sample_results):
        """Test conversion from RetrievalResult to candidate dict. / 测试从 RetrievalResult 转换为 candidate dict。"""
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=FakeReranker(),
        )
        
        candidates = reranker._results_to_candidates(sample_results)
        
        assert len(candidates) == 3
        assert candidates[0]["id"] == "chunk_001"
        assert candidates[0]["text"] == "First chunk about machine learning."
        assert candidates[0]["score"] == 0.9
        assert "source_path" in candidates[0]["metadata"]
    
    def test_candidates_to_results(self, mock_settings, sample_results):
        """Test conversion from candidate dict back to RetrievalResult. / 测试从 candidate dict 转回 RetrievalResult。"""
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=FakeReranker(),
        )
        
        candidates = [
            {"id": "chunk_001", "text": "Updated text", "rerank_score": 2.5},
            {"id": "chunk_002", "text": "Another text", "rerank_score": 2.0},
        ]
        
        results = reranker._candidates_to_results(candidates, sample_results)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score == 2.5
        assert results[0].metadata.get("original_score") == 0.9


# =============================================================================
# Test: Factory Function / 测试：工厂函数
# =============================================================================

class TestCreateCoreReranker:
    """Test create_core_reranker factory function. / 测试 create_core_reranker 工厂函数。"""
    
    def test_creates_reranker(self, mock_settings):
        """Test that factory creates CoreReranker. / 测试工厂创建 CoreReranker。"""
        fake_reranker = FakeReranker()
        reranker = create_core_reranker(mock_settings, reranker=fake_reranker)
        
        assert isinstance(reranker, CoreReranker)
        assert reranker._reranker is fake_reranker


# =============================================================================
# Test: Properties / 测试：属性
# =============================================================================

class TestCoreRerankerProperties:
    """Test CoreReranker properties. / 测试 CoreReranker 属性。"""
    
    def test_reranker_type_property(self, mock_settings):
        """Test reranker_type property. / 测试 reranker_type 属性。"""
        fake_reranker = FakeReranker()
        reranker = CoreReranker(settings=mock_settings, reranker=fake_reranker)
        
        assert reranker.reranker_type == "fakereranker"
    
    def test_is_enabled_true(self, mock_settings):
        """Test is_enabled returns True when enabled. / 测试启用时 is_enabled 返回 True。"""
        fake_reranker = FakeReranker()
        config = RerankConfig(enabled=True)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=fake_reranker,
            config=config,
        )
        
        assert reranker.is_enabled is True
    
    def test_is_enabled_false_when_disabled(self):
        """Test is_enabled returns False when disabled. / 测试禁用时 is_enabled 返回 False。"""
        settings = MockSettings(enabled=False)
        reranker = CoreReranker(settings=settings)
        
        assert reranker.is_enabled is False
    
    def test_is_enabled_false_for_none_reranker(self, mock_settings):
        """Test is_enabled returns False for NoneReranker. / 测试 NoneReranker 时 is_enabled 返回 False。"""
        none_reranker = NoneReranker(settings=mock_settings)
        config = RerankConfig(enabled=True)  # Config says enabled / 配置表示启用
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=none_reranker,  # But using NoneReranker / 但使用 NoneReranker
            config=config,
        )
        
        assert reranker.is_enabled is False
