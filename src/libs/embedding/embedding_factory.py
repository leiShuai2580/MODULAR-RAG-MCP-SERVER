"""Factory for creating Embedding provider instances. / 用于创建 Embedding provider 实例的工厂。

This module implements the Factory Pattern to instantiate the appropriate / 此模块实现工厂模式，用于实例化合适的
Embedding provider based on configuration, enabling configuration-driven selection / Embedding provider，基于配置实现不同后端的选择，
of different backends without code changes. / 无需修改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class EmbeddingFactory:
    """Factory for creating Embedding provider instances. / 用于创建 Embedding provider 实例的工厂。
    
    This factory reads the provider configuration from settings and instantiates / 此工厂从 settings 读取 provider 配置并实例化
    the corresponding Embedding implementation. Supported providers: OpenAI, Azure, / 对应的 Embedding 实现。支持的 provider：OpenAI、Azure，
    and more to be added in subsequent tasks. / 后续任务会添加更多。
    
    Design Principles Applied: / 应用的设计原则：
    - Factory Pattern: Centralizes object creation logic. / 工厂模式：集中对象创建逻辑
    - Config-Driven: Provider selection based on settings.yaml. / 配置驱动：基于 settings.yaml 选择 provider
    - Fail-Fast: Raises clear errors for unknown providers. / 快速失败：对未知 provider 抛出清晰错误
    """
    
    # Registry of supported providers / 支持的 provider 注册表
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEmbedding]) -> None:
        """Register a new Embedding provider implementation. / 注册新的 Embedding provider 实现。
        
        This method allows provider implementations to register themselves / 此方法允许 provider 实现将自身注册到
        with the factory, supporting extensibility. / 工厂中，以支持可扩展性。
        
        Args: / 参数：
            name: The provider identifier (e.g., 'openai', 'azure', 'local'). / provider 标识符（例如 'openai'、'azure'、'local'）。
            provider_class: The BaseEmbedding subclass implementing the provider. / 实现 provider 的 BaseEmbedding 子类。
        
        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseEmbedding. / 如果 provider_class 未继承 BaseEmbedding。
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEmbedding"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """Create an Embedding instance based on configuration. / 基于配置创建 Embedding 实例。
        
        Args: / 参数：
            settings: The application settings containing Embedding configuration. / 包含 Embedding 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。
        
        Returns: / 返回：
            An instance of the configured Embedding provider. / 已配置 Embedding provider 的实例。
        
        Raises: / 异常：
            ValueError: If the configured provider is not supported. / 如果配置的 provider 不受支持。
            AttributeError: If required configuration fields are missing. / 如果缺少必需配置字段。
        
        Example: / 示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> embedding = EmbeddingFactory.create(settings)
            >>> vectors = embedding.embed(["hello world", "test"])
        """
        # Extract provider name from settings / 从 settings 中提取 provider 名称
        try:
            provider_name = settings.embedding.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.embedding.provider. "
                "Please ensure 'embedding.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in registry / 在注册表中查找 provider 类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Embedding provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        
        # Instantiate the provider / 实例化 provider
        # Provider classes should accept settings and optional kwargs / Provider 类应接收 settings 和可选 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Embedding provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names. / 列出所有已注册的 provider 名称。
        
        Returns: / 返回：
            Sorted list of available provider identifiers. / 可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())


# Auto-register providers on module import / 模块导入时自动注册 provider
def _register_builtin_providers() -> None:
    """Register built-in Embedding providers with the factory. / 将内置 Embedding provider 注册到工厂。"""
    try:
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
    except ImportError:
        pass  # OpenAI provider not available / OpenAI provider 不可用
    
    try:
        from src.libs.embedding.azure_embedding import AzureEmbedding
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
    except ImportError:
        pass  # Azure provider not available / Azure provider 不可用
    
    try:
        from src.libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
    except ImportError:
        pass  # Ollama provider not available / Ollama provider 不可用


# Register providers when module is imported / 模块导入时注册 provider
_register_builtin_providers()
