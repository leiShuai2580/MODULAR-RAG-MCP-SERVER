"""Factory for creating Evaluator provider instances. / 用于创建 Evaluator provider 实例的工厂。

This module implements the Factory Pattern to instantiate the appropriate / 此模块实现工厂模式，用于基于配置
Evaluator provider based on configuration. / 实例化合适的 Evaluator provider。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings


def _get_ragas_evaluator() -> type[BaseEvaluator]:
    """Lazy import RagasEvaluator to avoid hard dependency on ragas. / 延迟导入 RagasEvaluator，以避免对 ragas 的硬依赖。"""
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    return RagasEvaluator


def _get_composite_evaluator() -> type[BaseEvaluator]:
    """Lazy import CompositeEvaluator. / 延迟导入 CompositeEvaluator。"""
    from src.observability.evaluation.composite_evaluator import CompositeEvaluator
    return CompositeEvaluator


class EvaluatorFactory:
    """Factory for creating Evaluator provider instances. / 用于创建 Evaluator provider 实例的工厂。

    Design Principles Applied: / 应用的设计原则：
    - Factory Pattern: Centralizes object creation logic. / 工厂模式：集中对象创建逻辑。
    - Config-Driven: Provider selection based on settings.yaml. / 配置驱动：基于 settings.yaml 选择 provider。
    - Fallback: Disabled evaluation returns NoneEvaluator. / 回退：禁用评估时返回 NoneEvaluator。
    - Fail-Fast: Raises clear errors for unknown providers. / 快速失败：对未知 provider 抛出清晰错误。
    """

    _PROVIDERS: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    # Lazy-loaded providers (import on demand to avoid hard dependencies) / 延迟加载的 provider（按需导入以避免硬依赖）
    _LAZY_PROVIDERS: dict[str, Any] = {
        "ragas": _get_ragas_evaluator,
        "composite": _get_composite_evaluator,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEvaluator]) -> None:
        """Register a new Evaluator provider implementation. / 注册新的 Evaluator provider 实现。

        Args: / 参数：
            name: The provider identifier (e.g., 'ragas', 'custom'). / provider 标识符（例如 'ragas'、'custom'）。
            provider_class: The BaseEvaluator subclass implementing the provider. / 实现 provider 的 BaseEvaluator 子类。

        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseEvaluator. / 如果 provider_class 未继承 BaseEvaluator。
        """
        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEvaluator"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEvaluator:
        """Create an Evaluator instance based on configuration. / 基于配置创建 Evaluator 实例。

        Args: / 参数：
            settings: The application settings containing evaluation config. / 包含 evaluation 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。

        Returns: / 返回：
            An instance of the configured Evaluator provider. / 已配置 Evaluator provider 的实例。

        Raises: / 异常：
            ValueError: If the configured provider is not supported or missing. / 如果配置的 provider 不受支持或缺失。
            RuntimeError: If provider initialization fails. / 如果 provider 初始化失败。
        """
        try:
            # Accept either full Settings (with .evaluation attr) or / 既接受完整 Settings（带 .evaluation 属性），也
            # a bare EvaluationSettings object directly. / 直接接受裸 EvaluationSettings 对象。
            if hasattr(settings, "evaluation"):
                evaluation_settings = settings.evaluation
            elif hasattr(settings, "provider") and hasattr(settings, "enabled"):
                evaluation_settings = settings
            else:
                raise AttributeError("settings has no 'evaluation' attribute")
            if evaluation_settings is None:
                raise AttributeError("settings.evaluation is None")
            provider_name = evaluation_settings.provider.lower()
            enabled = bool(evaluation_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.evaluation.provider. "
                "Please ensure 'evaluation.provider' is specified in settings.yaml"
            ) from e

        if not enabled or provider_name in {"none", "disabled"}:
            return NoneEvaluator(settings=settings, **override_kwargs)

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None and provider_name in cls._LAZY_PROVIDERS:
            try:
                provider_class = cls._LAZY_PROVIDERS[provider_name]()
                cls._PROVIDERS[provider_name] = provider_class  # cache for next call / 缓存供下次调用
            except ImportError as e:
                raise ValueError(
                    f"Provider '{provider_name}' requires additional dependencies: {e}"
                ) from e
        if provider_class is None:
            all_providers = sorted(set(cls._PROVIDERS.keys()) | set(cls._LAZY_PROVIDERS.keys()))
            available = ", ".join(all_providers) if all_providers else "none"
            raise ValueError(
                f"Unsupported Evaluator provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Evaluator provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names. / 列出所有已注册的 provider 名称。

        Returns: / 返回：
            Sorted list of available provider identifiers. / 可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())
