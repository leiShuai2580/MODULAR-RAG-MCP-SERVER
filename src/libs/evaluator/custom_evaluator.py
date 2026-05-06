"""Custom evaluator implementation for lightweight metrics. / 轻量指标的自定义 evaluator 实现。

This evaluator computes simple, deterministic metrics such as hit rate and MRR. / 此 evaluator 计算 hit rate 和 MRR 等简单、确定性指标。
It is designed for fast regression checks and sanity validation. / 它用于快速回归检查和基本合理性校验。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics (hit_rate, mrr). / 用于轻量指标（hit_rate、mrr）的自定义 evaluator。

    The evaluator expects retrieved chunks to contain an identifier field. / 此 evaluator 期望已检索块包含标识符字段。
    Supported id fields: id, chunk_id, document_id, doc_id. / 支持的 id 字段：id、chunk_id、document_id、doc_id。
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute requested metrics for the given retrieval results. / 为给定检索结果计算请求的指标。

        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            retrieved_chunks: Retrieved chunks or records. / 已检索块或记录。
            generated_answer: Optional generated answer (unused). / 可选生成答案（未使用）。
            ground_truth: Ground truth ids or structure. / 真实标注 ID 或结构。
            trace: Optional TraceContext (unused). / 可选 TraceContext（未使用）。
            **kwargs: Additional parameters (unused). / 额外参数（未使用）。

        Returns: / 返回：
            Dictionary of metric name to float value. / 指标名称到浮点值的字典。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
        ground_truth_ids = self._extract_ground_truth_ids(ground_truth)

        results: Dict[str, float] = {}

        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(retrieved_ids, ground_truth_ids)
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_ids, ground_truth_ids)

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available. / 如果可用，从 settings 中提取指标列表。"""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract ground truth ids from various input shapes. / 从各种输入形状中提取真实标注 ID。"""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            return self._extract_ids([ground_truth], label="ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")

        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract ids from a list of items. / 从条目列表中提取 ID。"""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue

            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type "
                f"{type(item).__name__}"
            )

        return ids

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute hit rate (binary). / 计算命中率（二元）。"""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR). / 计算平均倒数排名（MRR）。"""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0
