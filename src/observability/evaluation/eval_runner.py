"""Evaluation runner for batch quality assessment. / 用于批量质量评估的评估运行器。

EvalRunner reads a golden test set, runs HybridSearch for each test case, / EvalRunner 读取黄金测试集，为每个测试用例运行 HybridSearch，
optionally generates answers, then invokes the configured Evaluator(s) to / 可选地生成答案，然后调用已配置的 Evaluator，
produce a structured evaluation report. / 生成结构化评估报告。

Design Principles: / 设计原则：
- Config-Driven: Evaluator selected via settings.yaml. / 配置驱动：通过 settings.yaml 选择评估器。
- Observable: Produces EvalReport with per-query details. / 可观测：生成包含逐查询详情的 EvalReport。
- Decoupled: Works with any BaseEvaluator implementation. / 解耦：可与任何 BaseEvaluator 实现配合使用。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase:
    """A single evaluation test case from the golden test set. / 黄金测试集中的单个评估测试用例。

    Attributes: / 属性：
        query: The test query string. / 测试查询字符串。
        expected_chunk_ids: Ground-truth chunk IDs for IR metrics. / 用于 IR 指标的真实分块 ID。
        expected_sources: Ground-truth source file names (optional). / 真实来源文件名（可选）。
        reference_answer: Reference answer text for LLM-as-Judge (optional). / 用于 LLM-as-Judge 的参考答案文本（可选）。
    """

    query: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenTestCase:
        return cls(
            query=data["query"],
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            expected_sources=data.get("expected_sources", []),
            reference_answer=data.get("reference_answer"),
        )


@dataclass
class QueryResult:
    """Result of evaluating a single test case. / 单个测试用例的评估结果。

    Attributes: / 属性：
        query: The test query. / 测试查询。
        retrieved_chunk_ids: IDs of chunks actually retrieved. / 实际检索到的分块 ID。
        generated_answer: The generated answer (if applicable). / 生成的答案（如果适用）。
        metrics: Evaluation metrics for this query. / 该查询的评估指标。
        elapsed_ms: Time taken for retrieval + evaluation. / 检索和评估耗时。
    """

    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation report across all test cases. / 所有测试用例的聚合评估报告。

    Attributes: / 属性：
        query_results: Per-query evaluation results. / 逐查询评估结果。
        aggregate_metrics: Averaged metrics across all queries. / 所有查询的平均指标。
        total_elapsed_ms: Total time for the entire evaluation. / 整个评估总耗时。
        evaluator_name: Name of the evaluator used. / 所用评估器名称。
        test_set_path: Path to the golden test set file. / 黄金测试集文件路径。
    """

    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise report to dictionary. / 将报告序列化为字典。"""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {
                k: round(v, 4) for k, v in self.aggregate_metrics.items()
            },
            "query_count": len(self.query_results),
            "query_results": [
                {
                    "query": qr.query,
                    "retrieved_chunk_ids": qr.retrieved_chunk_ids,
                    "generated_answer": qr.generated_answer,
                    "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
                    "elapsed_ms": round(qr.elapsed_ms, 1),
                }
                for qr in self.query_results
            ],
        }


def load_test_set(path: str | Path) -> List[GoldenTestCase]:
    """Load golden test set from a JSON file. / 从 JSON 文件加载黄金测试集。

    Args: / 参数：
        path: Path to the golden test set JSON file. / 黄金测试集 JSON 文件路径。

    Returns: / 返回：
        List of TestCase instances. / TestCase 实例列表。

    Raises: / 抛出：
        FileNotFoundError: If the file does not exist. / 如果文件不存在。
        ValueError: If the file format is invalid. / 如果文件格式无效。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "test_cases" not in data:
        raise ValueError(
            "Invalid golden test set format: missing 'test_cases' key."
        )

    return [GoldenTestCase.from_dict(tc) for tc in data["test_cases"]]


class EvalRunner:
    """Runs batch evaluation against a golden test set. / 基于黄金测试集运行批量评估。

    This class orchestrates: / 该类编排以下流程：
    1. Loading the golden test set / 加载黄金测试集
    2. Running HybridSearch for each query / 为每个查询运行 HybridSearch
    3. Optionally generating answers / 可选生成答案
    4. Invoking the evaluator to score each result / 调用评估器为每个结果打分
    5. Aggregating metrics into an EvalReport / 将指标聚合到 EvalReport

    Example::

        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
        )
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: Optional[BaseEvaluator] = None,
        answer_generator: Any = None,
        answer_overrides: Optional[Dict[int, str]] = None,
        reranker: Any = None,
    ) -> None:
        """Initialize EvalRunner. / 初始化 EvalRunner。

        Args: / 参数：
            settings: Application settings. / 应用设置。
            hybrid_search: HybridSearch instance for retrieval. / 用于检索的 HybridSearch 实例。
            evaluator: BaseEvaluator instance for scoring. / 用于评分的 BaseEvaluator 实例。
            answer_generator: Optional callable(query, chunks) -> str / 可选 callable(query, chunks) -> str，
                for generating answers. If None, a simple concatenation / 用于生成答案。如果为 None，则使用简单拼接
                is used as a placeholder. / 作为占位实现。
            answer_overrides: Optional dict mapping test case index (0-based) / 可选字典，将测试用例索引（从 0 开始）
                to a user-provided answer string. When present, the override / 映射到用户提供的答案字符串。存在时，该覆盖
                answer is used instead of auto-generation for that test case. / 答案会替代该测试用例的自动生成答案。
            reranker: Optional CoreReranker instance for reranking results. / 用于重排结果的可选 CoreReranker 实例。
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.answer_overrides = answer_overrides or {}
        self.reranker = reranker

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """Run evaluation on the golden test set. / 在黄金测试集上运行评估。

        Args: / 参数：
            test_set_path: Path to golden_test_set.json. / golden_test_set.json 路径。
            top_k: Number of chunks to retrieve per query. / 每个查询要检索的分块数量。
            collection: Optional collection name filter. / 可选集合名称过滤器。

        Returns: / 返回：
            EvalReport with per-query and aggregate metrics. / 包含逐查询指标和聚合指标的 EvalReport。

        Raises: / 抛出：
            FileNotFoundError: If test set file doesn't exist. / 如果测试集文件不存在。
            ValueError: If evaluator or hybrid_search is not set. / 如果未设置 evaluator 或 hybrid_search。
        """
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")

        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        logger.info(
            "Starting evaluation: %d test cases, evaluator=%s",
            len(test_cases),
            type(self.evaluator).__name__,
        )

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__,
            test_set_path=str(test_set_path),
        )

        t0 = time.monotonic()

        for idx, tc in enumerate(test_cases):
            logger.info("Evaluating [%d/%d]: %s", idx + 1, len(test_cases), tc.query[:60])
            # Use user-provided answer override if available for this index / 如果该索引存在用户提供的答案覆盖，则使用它
            answer_override = self.answer_overrides.get(idx)
            qr = self._evaluate_single(
                tc, top_k=top_k, collection=collection,
                answer_override=answer_override,
            )
            report.query_results.append(qr)

        report.total_elapsed_ms = (time.monotonic() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    def _evaluate_single(
        self,
        test_case: GoldenTestCase,
        top_k: int = 10,
        collection: Optional[str] = None,
        answer_override: Optional[str] = None,
    ) -> QueryResult:
        """Evaluate a single test case. / 评估单个测试用例。

        Args: / 参数：
            test_case: The test case to evaluate. / 要评估的测试用例。
            top_k: Number of results to retrieve. / 要检索的结果数量。
            collection: Optional collection filter. / 可选集合过滤器。
            answer_override: User-provided answer text. When set, used / 用户提供的答案文本。设置时，
                instead of auto-generated answer from chunks. / 替代从分块自动生成的答案。

        Returns: / 返回：
            QueryResult with metrics for this test case. / 包含该测试用例指标的 QueryResult。
        """
        t0 = time.monotonic()
        qr = QueryResult(query=test_case.query)

        # Step 1: Retrieve chunks / 步骤 1：检索分块
        retrieved_chunks = self._retrieve(test_case.query, top_k, collection)
        qr.retrieved_chunk_ids = [
            self._get_chunk_id(c) for c in retrieved_chunks
        ]

        # Step 2: Generate answer — prefer user override, then generator, then fallback / 步骤 2：生成答案 - 优先用户覆盖，其次生成器，最后回退
        if answer_override:
            answer = answer_override
        else:
            answer = self._generate_answer(test_case.query, retrieved_chunks)
        qr.generated_answer = answer

        # Step 3: Build ground truth / 步骤 3：构建真实标注
        ground_truth = (
            {"ids": test_case.expected_chunk_ids}
            if test_case.expected_chunk_ids
            else None
        )

        # Step 4: Evaluate / 步骤 4：评估
        try:
            metrics = self.evaluator.evaluate(  # type: ignore[union-attr]
                query=test_case.query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
            )
            qr.metrics = metrics
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", test_case.query[:40], exc)
            qr.metrics = {}

        qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return qr

    def _retrieve(
        self,
        query: str,
        top_k: int,
        collection: Optional[str],
    ) -> List[Any]:
        """Retrieve chunks using HybridSearch + optional Reranking. / 使用 HybridSearch 和可选重排检索分块。

        Falls back to an empty list if search is not configured. / 如果未配置搜索，则回退为空列表。
        """
        if self.hybrid_search is None:
            logger.warning("No HybridSearch configured; returning empty results.")
            return []

        try:
            # Retrieve more candidates if reranker is enabled / 如果启用重排器，则检索更多候选
            has_reranker = self.reranker is not None and getattr(self.reranker, 'is_enabled', False)
            initial_top_k = top_k * 2 if has_reranker else top_k

            results = self.hybrid_search.search(
                query=query,
                top_k=initial_top_k,
            )
            results = results if isinstance(results, list) else results.results

            # Apply reranking if enabled / 如果启用，则应用重排
            if has_reranker and results:
                rerank_result = self.reranker.rerank(query=query, results=results, top_k=top_k)
                results = rerank_result.results

            return results
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", query[:40], exc)
            return []

    def _generate_answer(self, query: str, chunks: List[Any]) -> str:
        """Generate an answer from retrieved chunks. / 根据检索分块生成答案。

        If a custom answer_generator is provided, use it. / 如果提供了自定义 answer_generator，则使用它。
        Otherwise, concatenate chunk texts as a simple placeholder. / 否则，将分块文本简单拼接作为占位。
        """
        if self.answer_generator is not None:
            try:
                return self.answer_generator(query, chunks)
            except Exception as exc:
                logger.warning("Answer generation failed: %s", exc)

        # Fallback: concatenate chunk texts / 回退：拼接分块文本
        texts = []
        for c in chunks:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            elif hasattr(c, "text"):
                texts.append(str(getattr(c, "text")))
            else:
                texts.append(str(c))

        return " ".join(texts[:5])  # first 5 chunks / 前 5 个分块

    def _get_chunk_id(self, chunk: Any) -> str:
        """Extract chunk ID from various representations. / 从各种表示中提取分块 ID。"""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("id", "chunk_id"):
                if key in chunk:
                    return str(chunk[key])
            return str(chunk)
        if hasattr(chunk, "chunk_id"):
            return str(getattr(chunk, "chunk_id"))
        if hasattr(chunk, "id"):
            return str(getattr(chunk, "id"))
        return str(chunk)

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """Compute average metrics across all query results. / 计算所有查询结果的平均指标。

        Args: / 参数：
            results: List of QueryResult with per-query metrics. / 包含逐查询指标的 QueryResult 列表。

        Returns: / 返回：
            Dictionary of average metric values. / 平均指标值字典。
        """
        if not results:
            return {}

        # Collect all metric keys / 收集所有指标键
        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        # Average each metric / 对每个指标求平均值
        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [qr.metrics[key] for qr in results if key in qr.metrics]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages
