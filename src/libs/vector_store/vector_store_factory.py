"""Factory for creating VectorStore provider instances. / 用于创建 VectorStore provider 实例的工厂。

This module implements the Factory Pattern to instantiate the appropriate / 此模块实现工厂模式，用于实例化合适的
VectorStore provider based on configuration, enabling configuration-driven selection / VectorStore provider，基于配置实现不同后端的选择，
of different backends without code changes. / 无需修改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorStoreFactory:
    """Factory for creating VectorStore provider instances. / 用于创建 VectorStore provider 实例的工厂。
    
    This factory reads the provider configuration from settings and instantiates / 此工厂从 settings 读取 provider 配置并实例化
    the corresponding VectorStore implementation. Supported providers will be added / 对应的 VectorStore 实现。支持的 provider 会在
    in subsequent tasks (B7.6 and beyond). / 后续任务（B7.6 及以后）中添加。
    
    Design Principles Applied: / 应用的设计原则：
    - Factory Pattern: Centralizes object creation logic. / 工厂模式：集中对象创建逻辑。
    - Config-Driven: Provider selection based on settings.yaml. / 配置驱动：基于 settings.yaml 选择 provider。
    - Fail-Fast: Raises clear errors for unknown providers. / 快速失败：对未知 provider 抛出清晰错误。
    """
    
    # Registry of supported providers (to be populated in B7.x tasks) / 支持的 provider 注册表（将在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseVectorStore]) -> None:
        """Register a new VectorStore provider implementation. / 注册新的 VectorStore provider 实现。
        
        This method allows provider implementations to register themselves / 此方法允许 provider 实现将自身注册到
        with the factory, supporting extensibility. / 工厂中，以支持可扩展性。
        
        Args: / 参数：
            name: The provider identifier (e.g., 'chroma', 'qdrant', 'milvus'). / provider 标识符（例如 'chroma'、'qdrant'、'milvus'）。
            provider_class: The BaseVectorStore subclass implementing the provider. / 实现 provider 的 BaseVectorStore 子类。
        
        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseVectorStore. / 如果 provider_class 未继承 BaseVectorStore。
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVectorStore"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """Create a VectorStore instance based on configuration. / 基于配置创建 VectorStore 实例。
        
        Args: / 参数：
            settings: The application settings containing VectorStore configuration. / 包含 VectorStore 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。
        
        Returns: / 返回：
            An instance of the configured VectorStore provider. / 已配置 VectorStore provider 的实例。
        
        Raises: / 异常：
            ValueError: If the configured provider is not supported. / 如果配置的 provider 不受支持。
            AttributeError: If required configuration fields are missing. / 如果缺少必需配置字段。
        
        Example: / 示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> vector_store = VectorStoreFactory.create(settings)
            >>> vector_store.upsert([{'id': 'doc1', 'vector': [0.1, 0.2]}])
        """
        # Extract provider name from settings / 从 settings 中提取 provider 名称
        try:
            provider_name = settings.vector_store.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store.provider. "
                "Please ensure 'vector_store.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in registry / 在注册表中查找 provider 类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in task B7.6 and beyond."
            )
        
        # Instantiate the provider / 实例化 provider
        # Provider classes should accept settings and optional kwargs / Provider 类应接收 settings 和可选 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate VectorStore provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names. / 列出所有已注册的 provider 名称。
        
        Returns: / 返回：
            Sorted list of provider names. / provider 名称的排序列表。
        
        Example: / 示例：
            >>> VectorStoreFactory.list_providers()
            ['chroma', 'milvus', 'qdrant']
        """
        return sorted(cls._PROVIDERS.keys())
