"""Ragas-based evaluator for RAG quality assessment. / 用于 RAG 质量评估的 Ragas 评估器。

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics: / 该评估器封装 Ragas 框架来计算 LLM-as-Judge 指标：
- Faithfulness: Does the answer stick to the retrieved context? / 忠实度：答案是否严格基于检索上下文？
- Answer Relevancy: Is the answer relevant to the query? / 答案相关性：答案是否与查询相关？
- Context Precision: Are the retrieved chunks relevant and well-ordered? / 上下文精确率：检索到的分块是否相关且排序良好？

Design Principles: / 设计原则：
- Pluggable: Implements BaseEvaluator interface, swappable via factory. / 可插拔：实现 BaseEvaluator 接口，可通过工厂替换。
- Config-Driven: LLM/Embedding backend read from settings.yaml. / 配置驱动：从 settings.yaml 读取 LLM/Embedding 后端。
- Graceful Degradation: Clear ImportError if ragas not installed. / 优雅降级：未安装 ragas 时给出清晰 ImportError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants / 指标名称常量
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not. / 校验 ragas 是否可导入，否则抛出清晰错误。"""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics. / 使用 Ragas 框架计算 LLM-as-Judge 指标的评估器。

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge / Ragas 不需要真实标签。它使用 LLM 根据检索上下文
    the quality of the generated answer against the retrieved context. / 判断生成答案的质量。

    Supported metrics: / 支持的指标：
        - faithfulness: Measures factual consistency with context. / 衡量与上下文的事实一致性。
        - answer_relevancy: Measures how relevant the answer is to the query. / 衡量答案与查询的相关程度。
        - context_precision: Measures relevance/ordering of retrieved chunks. / 衡量检索分块的相关性和排序质量。

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...} / metrics 等于 {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator. / 初始化 RagasEvaluator。

        Args: / 参数：
            settings: Application settings (used to configure LLM backend). / 应用设置（用于配置 LLM 后端）。
            metrics: Metric names to compute. Defaults to all supported. / 要计算的指标名称；默认计算所有支持指标。
            **kwargs: Additional parameters (reserved). / 额外参数（预留）。

        Raises: / 抛出：
            ImportError: If ragas is not installed. / 如果未安装 ragas。
            ValueError: If unsupported metric names are requested. / 如果请求了不支持的指标名称。
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ──────────────────────────────────────────────── / ── 公共 API ───────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics. / 使用 Ragas LLM-as-Judge 指标评估 RAG 质量。

        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings). / 检索到的分块（带 'text' 键的字典或字符串）。
            generated_answer: The generated answer text. Required for Ragas. / 生成答案文本；Ragas 必需。
            ground_truth: Ignored by Ragas (not needed for LLM-as-Judge). / Ragas 会忽略（LLM-as-Judge 不需要）。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Additional parameters. / 额外参数。

        Returns: / 返回：
            Dictionary mapping metric names to float scores (0.0 – 1.0). / 指标名称到浮点分数（0.0 - 1.0）的字典。

        Raises: / 抛出：
            ValueError: If query/chunks are invalid or generated_answer is missing. / 如果 query/chunks 无效或缺少 generated_answer。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ─────────────────────────────────────────── / ── 私有辅助方法 ───────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics and return normalised scores. / 执行 Ragas collections 指标并返回归一化分数。

        Ragas 0.4+ collections metrics use per-metric ``score()`` instead of / Ragas 0.4+ collections 指标使用每个指标自己的 ``score()``，
        the legacy ``evaluate()`` pipeline.  Each metric has its own signature: / 而不是旧版 ``evaluate()`` 流水线。每个指标都有自己的签名：
        - Faithfulness / ContextPrecision: (user_input, response, retrieved_contexts) / Faithfulness / ContextPrecision：(user_input, response, retrieved_contexts)
        - AnswerRelevancy: (user_input, response) / AnswerRelevancy：(user_input, response)
        """
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        # Build LLM / Embedding wrappers from settings / 根据设置构建 LLM / Embedding 包装器
        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = m.score(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings. / 根据项目设置构建 Ragas LLM 和 Embedding 包装器。

        Uses Ragas 0.4+ native API (InstructorLLM + OpenAIEmbeddings) / 使用 Ragas 0.4+ 原生 API（InstructorLLM + OpenAIEmbeddings），
        instead of deprecated LangchainLLMWrapper. / 而不是已废弃的 LangchainLLMWrapper。

        Returns: / 返回：
            Tuple of (llm_wrapper, embeddings_wrapper). / (llm_wrapper, embeddings_wrapper) 元组。
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        # ── LLM ── / ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        # Azure-compatible mode: if azure_endpoint is configured, use Azure / Azure 兼容模式：如果配置了 azure_endpoint，则使用 Azure
        # client even when provider is "openai" (matches project convention). / 客户端，即使 provider 是 "openai"（符合项目约定）。
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif provider == "openai":
            llm_client = AsyncOpenAI(api_key=llm_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )

        llm = llm_factory(llm_cfg.model, client=llm_client, max_tokens=8192)

        # ── Embeddings ── / ── Embeddings ──
        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        emb_azure_endpoint = getattr(emb_cfg, "azure_endpoint", None)

        # Same Azure-compatible mode detection for embeddings / 对 embeddings 使用相同的 Azure 兼容模式检测
        use_azure_emb = (
            emb_provider == "azure"
            or (emb_provider == "openai" and emb_azure_endpoint)
        )

        if use_azure_emb:
            emb_client = AsyncAzureOpenAI(
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_azure_endpoint or emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif emb_provider == "openai":
            emb_client = AsyncOpenAI(api_key=emb_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai"
            )

        embeddings = OpenAIEmbeddings(model=emb_cfg.model, client=emb_client)

        return llm, embeddings

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations. / 从各种分块表示中提取文本字符串。

        Args: / 参数：
            chunks: List of chunk dicts, strings, or objects with .text. / 分块字典、字符串或带 .text 的对象列表。

        Returns: / 返回：
            List of text strings. / 文本字符串列表。
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available. / 如果可用，则从设置中提取指标列表。"""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics / 只保留 ragas 支持的指标
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
