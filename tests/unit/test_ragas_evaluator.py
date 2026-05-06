"""Unit tests for RagasEvaluator. / RagasEvaluator 的单元测试。

Tests verify: / 测试验证：
- Initialization with valid/invalid metrics / 使用有效/无效指标初始化
- ImportError handling when ragas is not installed / 未安装 ragas 时的 ImportError 处理
- Input validation (missing answer, empty query, etc.) / 输入校验（缺失答案、空查询等）
- Metric extraction from settings / 从 settings 提取指标
- Text extraction from various chunk formats / 从多种 chunk 格式提取文本

Note: Actual Ragas evaluation (LLM calls) is mocked to keep unit tests fast / 注意：真实 Ragas 评估（LLM 调用）已 mock，以保证单元测试快速
and deterministic. / 且确定。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict

import pytest


class TestRagasEvaluatorInit:
    """Tests for RagasEvaluator initialisation. / RagasEvaluator 初始化测试。"""

    def test_init_default_metrics(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator()
        assert set(evaluator._metric_names) == {
            "answer_relevancy",
            "context_precision",
            "faithfulness",
        }

    def test_init_custom_metrics(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        assert evaluator._metric_names == ["faithfulness"]

    def test_init_unsupported_metric_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        with pytest.raises(ValueError, match="Unsupported ragas metrics"):
            RagasEvaluator(metrics=["hit_rate"])

    def test_init_reads_metrics_from_settings(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        settings = MagicMock()
        settings.evaluation.metrics = ["faithfulness", "answer_relevancy", "hit_rate"]

        evaluator = RagasEvaluator(settings=settings)
        # hit_rate is not a ragas metric, should be filtered out / hit_rate 不是 ragas 指标，应被过滤掉
        assert "hit_rate" not in evaluator._metric_names
        assert "faithfulness" in evaluator._metric_names
        assert "answer_relevancy" in evaluator._metric_names

    def test_init_no_settings_defaults_to_all(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(settings=None, metrics=None)
        assert len(evaluator._metric_names) == 3


class TestRagasImportCheck:
    """Tests for ragas import validation. / ragas 导入校验测试。"""

    def test_import_error_when_ragas_missing(self) -> None:
        from src.observability.evaluation.ragas_evaluator import _import_ragas

        with patch.dict("sys.modules", {"ragas": None}):
            with pytest.raises(ImportError, match="ragas"):
                _import_ragas()


class TestRagasEvaluatorValidation:
    """Tests for input validation in evaluate(). / evaluate() 输入校验测试。"""

    def test_empty_query_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        with pytest.raises(ValueError, match="Query cannot be empty"):
            evaluator.evaluate("  ", [{"text": "ctx"}], generated_answer="ans")

    def test_empty_chunks_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
            evaluator.evaluate("query", [], generated_answer="ans")

    def test_missing_generated_answer_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        with pytest.raises(ValueError, match="generated_answer"):
            evaluator.evaluate("query", [{"text": "ctx"}], generated_answer=None)

    def test_empty_generated_answer_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        with pytest.raises(ValueError, match="generated_answer"):
            evaluator.evaluate("query", [{"text": "ctx"}], generated_answer="   ")


class TestRagasEvaluatorTextExtraction:
    """Tests for _extract_texts helper. / _extract_texts 辅助函数测试。"""

    def test_extract_from_dicts(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        result = evaluator._extract_texts([
            {"text": "chunk1"},
            {"content": "chunk2"},
            {"page_content": "chunk3"},
        ])
        assert result == ["chunk1", "chunk2", "chunk3"]

    def test_extract_from_strings(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        result = evaluator._extract_texts(["chunk1", "chunk2"])
        assert result == ["chunk1", "chunk2"]

    def test_extract_from_objects(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])

        class Chunk:
            def __init__(self, text: str) -> None:
                self.text = text

        result = evaluator._extract_texts([Chunk("hello"), Chunk("world")])
        assert result == ["hello", "world"]


class TestRagasEvaluatorEvaluate:
    """Tests for evaluate() with mocked Ragas backend. / 使用 mock Ragas backend 的 evaluate() 测试。"""

    def _make_mock_ragas_result(self, scores: Dict[str, float]) -> MagicMock:
        """Create a mock ragas evaluation result. / 创建 mock ragas 评估结果。"""
        import pandas as pd

        df = pd.DataFrame([scores])
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df
        return mock_result

    def test_evaluate_returns_metrics_dict(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness", "context_precision"])

        expected = {"faithfulness": 0.92, "context_precision": 0.85}
        evaluator._run_ragas = MagicMock(return_value=expected)  # type: ignore[method-assign]

        result = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=["RAG is retrieval augmented generation."],
            generated_answer="RAG stands for Retrieval Augmented Generation.",
        )

        assert result == expected

    def test_evaluate_with_mocked_run_ragas(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness", "answer_relevancy"])

        expected_scores = {"faithfulness": 0.95, "answer_relevancy": 0.88}
        evaluator._run_ragas = MagicMock(return_value=expected_scores)  # type: ignore[method-assign]

        result = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"text": "RAG is Retrieval Augmented Generation"}],
            generated_answer="RAG stands for Retrieval Augmented Generation.",
        )

        assert result == expected_scores
        evaluator._run_ragas.assert_called_once()

    def test_evaluate_runtime_error_on_ragas_failure(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        evaluator._run_ragas = MagicMock(  # type: ignore[method-assign]
            side_effect=Exception("LLM call failed"),
        )

        with pytest.raises(RuntimeError, match="Ragas evaluation failed"):
            evaluator.evaluate(
                query="test",
                retrieved_chunks=[{"text": "ctx"}],
                generated_answer="answer",
            )

    def test_ground_truth_is_ignored(self) -> None:
        """Ragas should work fine even when ground_truth is provided. / 即使提供 ground_truth，Ragas 也应正常工作。"""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness"])
        evaluator._run_ragas = MagicMock(return_value={"faithfulness": 0.9})  # type: ignore[method-assign]

        result = evaluator.evaluate(
            query="test",
            retrieved_chunks=[{"text": "ctx"}],
            generated_answer="answer",
            ground_truth=["chunk_001"],  # should be ignored / 应被忽略
        )

        assert "faithfulness" in result


class TestRagasEvaluatorFactory:
    """Tests for factory integration. / factory 集成测试。"""

    def test_factory_creates_ragas_evaluator(self) -> None:
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        settings = MagicMock()
        settings.evaluation.enabled = True
        settings.evaluation.provider = "ragas"
        settings.evaluation.metrics = ["faithfulness"]

        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, RagasEvaluator)

    def test_factory_lists_ragas(self) -> None:
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        providers = EvaluatorFactory.list_providers()
        assert "custom" in providers
        # ragas may be in _PROVIDERS after first create or in _LAZY_PROVIDERS / ragas 可能在首次创建后位于 _PROVIDERS，或位于 _LAZY_PROVIDERS
