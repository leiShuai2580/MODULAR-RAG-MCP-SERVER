"""Composite evaluator that combines multiple evaluators. / 组合多个评估器的复合评估器。

This evaluator implements the Composite pattern: it holds a list of / 该评估器实现组合模式：它持有一组
BaseEvaluator instances, runs them all, and merges their metric / BaseEvaluator 实例，全部运行后将它们的指标
dictionaries into a single result. / 字典合并为一个结果。

Design Principles: / 设计原则：
- Pluggable: Any BaseEvaluator can be composed. / 可插拔：任意 BaseEvaluator 都可组合。
- Config-Driven: `evaluation.backends: [ragas, custom]` drives composition. / 配置驱动：由 `evaluation.backends: [ragas, custom]` 驱动组合。
- Observable: Logs individual evaluator successes/failures. / 可观测：记录各个评估器的成功或失败。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """Evaluator that composes multiple evaluators and merges metrics. / 组合多个评估器并合并指标的评估器。

    Each sub-evaluator is invoked with the same arguments.  Results are / 每个子评估器都会用相同参数调用。结果会
    merged into a single metrics dictionary.  If two evaluators produce / 合并为单个指标字典。如果两个评估器产生
    the same metric key, the later one's value wins (with a warning). / 相同指标键，则后者覆盖前者（并记录警告）。

    Partial failure is tolerated: if one sub-evaluator fails, its error / 允许部分失败：如果某个子评估器失败，会记录错误，
    is logged and the rest still execute. / 其余评估器仍会继续执行。

    Example::

        composite = CompositeEvaluator(evaluators=[
            CustomEvaluator(metrics=["hit_rate", "mrr"]),
            RagasEvaluator(metrics=["faithfulness"]),
        ])
        metrics = composite.evaluate(
            query="test", retrieved_chunks=[...],
            generated_answer="...", ground_truth=[...]
        )
        # metrics == {"hit_rate": 1.0, "mrr": 0.5, "faithfulness": 0.92} / metrics 等于 {"hit_rate": 1.0, "mrr": 0.5, "faithfulness": 0.92}
    """

    def __init__(
        self,
        evaluators: Optional[Sequence[BaseEvaluator]] = None,
        settings: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CompositeEvaluator. / 初始化 CompositeEvaluator。

        Args: / 参数：
            evaluators: Pre-built evaluator instances. If None, built from settings. / 预构建评估器实例；如果为 None，则从设置构建。
            settings: Application settings (used for config-driven composition). / 应用设置（用于配置驱动组合）。
            **kwargs: Additional parameters forwarded to sub-evaluators. / 转发给子评估器的额外参数。

        Raises: / 抛出：
            ValueError: If no evaluators are provided and settings don't / 如果没有提供评估器且 settings 未
                specify backends. / 指定后端。
        """
        self.settings = settings
        self.kwargs = kwargs

        if evaluators is not None:
            self._evaluators: List[BaseEvaluator] = list(evaluators)
        else:
            self._evaluators = self._build_from_settings(settings, **kwargs)

        if not self._evaluators:
            raise ValueError(
                "CompositeEvaluator requires at least one sub-evaluator. "
                "Provide evaluators directly or configure "
                "'evaluation.backends' in settings.yaml."
            )

        logger.info(
            "CompositeEvaluator initialised with %d evaluator(s): %s",
            len(self._evaluators),
            [type(e).__name__ for e in self._evaluators],
        )

    @property
    def evaluators(self) -> List[BaseEvaluator]:
        """Return the list of composed evaluators. / 返回已组合的评估器列表。"""
        return list(self._evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run all sub-evaluators and merge their metrics. / 运行所有子评估器并合并它们的指标。

        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            retrieved_chunks: Retrieved chunks or records. / 检索到的分块或记录。
            generated_answer: Optional generated answer text. / 可选的生成答案文本。
            ground_truth: Optional ground truth data. / 可选的真实标注数据。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Additional parameters. / 额外参数。

        Returns: / 返回：
            Merged dictionary of all metric names to float values. / 所有指标名称到浮点值的合并字典。

        Raises: / 抛出：
            RuntimeError: If ALL sub-evaluators fail. / 如果所有子评估器都失败。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        merged: Dict[str, float] = {}
        errors: List[str] = []

        for evaluator in self._evaluators:
            name = type(evaluator).__name__
            try:
                metrics = evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    trace=trace,
                    **kwargs,
                )
                for key, value in metrics.items():
                    if key in merged:
                        logger.warning(
                            "Metric '%s' produced by multiple evaluators; "
                            "overwriting with value from %s",
                            key,
                            name,
                        )
                    merged[key] = value

                logger.debug(
                    "%s produced %d metric(s): %s",
                    name,
                    len(metrics),
                    list(metrics.keys()),
                )

            except Exception as exc:
                msg = f"{name} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        if not merged and errors:
            raise RuntimeError(
                "All sub-evaluators failed:\n" + "\n".join(errors)
            )

        return merged

    # ── config-driven builder ──────────────────────────────────── / ── 配置驱动构建器 ───────────────────────────────────

    @staticmethod
    def _build_from_settings(
        settings: Any,
        **kwargs: Any,
    ) -> List[BaseEvaluator]:
        """Build sub-evaluators from settings.evaluation.backends. / 从 settings.evaluation.backends 构建子评估器。

        Expected config::

            evaluation:
              enabled: true
              provider: composite
              backends:
                - ragas
                - custom
              metrics:
                - faithfulness
                - hit_rate
                - mrr

        Args: / 参数：
            settings: Application settings. / 应用设置。
            **kwargs: Forwarded to each sub-evaluator constructor. / 转发给每个子评估器构造函数。

        Returns: / 返回：
            List of BaseEvaluator instances. / BaseEvaluator 实例列表。
        """
        if settings is None:
            return []

        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []

        backends = getattr(evaluation, "backends", None)
        if not backends:
            return []

        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        evaluators: List[BaseEvaluator] = []
        for backend_name in backends:
            backend_name = str(backend_name).strip().lower()
            if backend_name in {"composite", "none", "disabled"}:
                continue  # avoid infinite recursion / no-ops / 避免无限递归或无操作

            try:
                # Create a mock settings with provider overridden / 创建覆盖 provider 的模拟 settings
                from unittest.mock import MagicMock

                sub_settings = MagicMock(wraps=settings)
                sub_eval = MagicMock()
                sub_eval.enabled = True
                sub_eval.provider = backend_name
                sub_eval.metrics = getattr(evaluation, "metrics", [])
                sub_eval.backends = []  # prevent recursion / 防止递归
                sub_settings.evaluation = sub_eval

                evaluator = EvaluatorFactory.create(sub_settings, **kwargs)
                evaluators.append(evaluator)
                logger.info("CompositeEvaluator: loaded backend '%s'", backend_name)
            except Exception as exc:
                logger.warning(
                    "CompositeEvaluator: failed to load backend '%s': %s",
                    backend_name,
                    exc,
                )

        return evaluators
