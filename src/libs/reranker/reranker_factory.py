"""Factory for creating Reranker provider instances. / 用于创建 Reranker provider 实例的工厂。

This module implements the Factory Pattern to instantiate the appropriate / 此模块实现工厂模式，用于实例化合适的
Reranker provider based on configuration, enabling configuration-driven selection / Reranker provider，基于配置实现不同后端的选择，
of different backends without code changes. / 无需修改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from src.core.settings import Settings


def _lazy_import_llm_reranker():
    """Lazy import to avoid circular dependencies. / 延迟导入以避免循环依赖。"""
    from src.libs.reranker.llm_reranker import LLMReranker
    return LLMReranker


def _lazy_import_cross_encoder_reranker():
    """Lazy import to avoid circular dependencies. / 延迟导入以避免循环依赖。"""
    from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
    return CrossEncoderReranker


class RerankerFactory:
    """Factory for creating Reranker provider instances. / 用于创建 Reranker provider 实例的工厂。
    
    This factory reads the rerank configuration from settings and instantiates / 此工厂从 settings 读取 rerank 配置并实例化
    the corresponding Reranker implementation. Provider implementations will be / 对应的 Reranker 实现。Provider 实现会在
    added in subsequent tasks (B7.7, B7.8). / 后续任务（B7.7、B7.8）中添加。
    
    Design Principles Applied: / 应用的设计原则：
    - Factory Pattern: Centralizes object creation logic. / 工厂模式：集中对象创建逻辑。
    - Config-Driven: Provider selection based on settings.yaml. / 配置驱动：基于 settings.yaml 选择 provider。
    - Fallback: Disabled or 'none' provider returns NoneReranker. / 回退：禁用或 'none' provider 返回 NoneReranker。
    - Fail-Fast: Raises clear errors for unknown providers. / 快速失败：对未知 provider 抛出清晰错误。
    """
    
    _PROVIDERS: dict[str, type[BaseReranker]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseReranker]) -> None:
        """Register a new Reranker provider implementation. / 注册新的 Reranker provider 实现。
        
        Args: / 参数：
            name: The provider identifier (e.g., 'cross_encoder', 'llm'). / provider 标识符（例如 'cross_encoder'、'llm'）。
            provider_class: The BaseReranker subclass implementing the provider. / 实现 provider 的 BaseReranker 子类。
        
        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseReranker. / 如果 provider_class 未继承 BaseReranker。
        """
        if not issubclass(provider_class, BaseReranker):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseReranker"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseReranker:
        """Create a Reranker instance based on configuration. / 基于配置创建 Reranker 实例。
        
        Args: / 参数：
            settings: The application settings containing rerank configuration. / 包含 rerank 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。
        
        Returns: / 返回：
            An instance of the configured Reranker provider. / 已配置 Reranker provider 的实例。
        
        Raises: / 异常：
            ValueError: If the configured provider is not supported or missing. / 如果配置的 provider 不受支持或缺失。
            RuntimeError: If provider initialization fails. / 如果 provider 初始化失败。
        """
        # Lazy register LLM reranker if not already registered / 如果尚未注册，则延迟注册 LLM reranker
        if "llm" not in cls._PROVIDERS:
            LLMReranker = _lazy_import_llm_reranker()
            cls.register_provider("llm", LLMReranker)
        
        # Lazy register Cross-Encoder reranker if not already registered / 如果尚未注册，则延迟注册 Cross-Encoder reranker
        if "cross_encoder" not in cls._PROVIDERS:
            CrossEncoderReranker = _lazy_import_cross_encoder_reranker()
            cls.register_provider("cross_encoder", CrossEncoderReranker)
        
        try:
            rerank_settings = settings.rerank
            if rerank_settings is None:
                raise AttributeError("settings.rerank is None")
            provider_name = rerank_settings.provider.lower()
            enabled = bool(rerank_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.rerank.provider. "
                "Please ensure 'rerank.provider' is specified in settings.yaml"
            ) from e
        
        if not enabled or provider_name == "none":
            return NoneReranker(settings=settings, **override_kwargs)
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Reranker provider: '{provider_name}'. "
                f"Available providers: {available}."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Reranker provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names. / 列出所有已注册的 provider 名称。
        
        Returns: / 返回：
            Sorted list of available provider identifiers. / 可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())
