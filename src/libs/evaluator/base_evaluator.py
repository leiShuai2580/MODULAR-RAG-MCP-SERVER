"""Abstract base class for Evaluator providers. / Evaluator provider 的抽象基类。

This module defines the pluggable interface for evaluation providers, / 此模块定义 evaluation provider 的可插拔接口，
enabling seamless switching between different evaluation backends / 支持在不同评估后端之间无缝切换，
through configuration-driven instantiation. / 并通过配置驱动的实例化完成选择。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """Abstract base class for evaluation providers. / 评估 provider 的抽象基类。

    All evaluator implementations must inherit from this class and implement / 所有 evaluator 实现都必须继承此类并实现
    the evaluate() method. This ensures a consistent interface across different / evaluate() 方法。这确保不同
    evaluation backends. / 评估后端之间接口一致。

    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Subclasses can be swapped without changing upstream code. / 可插拔：无需修改上游代码即可替换子类。
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选 TraceContext 以集成可观测能力。
    - Config-Driven: Instances are created via factory based on settings. / 配置驱动：基于 settings 通过工厂创建实例。
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate retrieval and generation quality. / 评估检索和生成质量。

        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            retrieved_chunks: Retrieved chunks or records to evaluate. / 要评估的已检索块或记录。
            generated_answer: Optional generated answer text. / 可选生成答案文本。
            ground_truth: Optional ground truth data (ids or answers). / 可选真实标注数据（ID 或答案）。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters. / provider 特有参数。

        Returns: / 返回：
            Dictionary of metric names to float values. / 指标名称到浮点值的字典。

        Raises: / 异常：
            ValueError: If inputs are invalid. / 如果输入无效。
            RuntimeError: If evaluation fails unexpectedly. / 如果评估意外失败。
        """
        pass

    def validate_query(self, query: str) -> None:
        """Validate query string. / 校验查询字符串。

        Args: / 参数：
            query: Query string to validate. / 要校验的查询字符串。

        Raises: / 异常：
            ValueError: If query is invalid. / 如果查询无效。
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_retrieved_chunks(self, retrieved_chunks: List[Any]) -> None:
        """Validate retrieved chunks structure. / 校验已检索块结构。

        Args: / 参数：
            retrieved_chunks: Retrieved chunks list to validate. / 要校验的已检索块列表。

        Raises: / 异常：
            ValueError: If retrieved_chunks is invalid. / 如果 retrieved_chunks 无效。
        """
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")


class NoneEvaluator(BaseEvaluator):
    """No-op evaluator that returns empty metrics. / 返回空指标的 no-op evaluator。

    This implementation is used when evaluation is disabled. / 禁用评估时使用此实现。
    """

    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}
