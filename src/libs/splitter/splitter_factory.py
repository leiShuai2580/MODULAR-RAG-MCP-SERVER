"""Factory for creating Splitter instances. / 用于创建 Splitter 实例的工厂。

This module implements the Factory Pattern to instantiate the appropriate / 此模块实现工厂模式，用于实例化合适的
Splitter provider based on configuration, enabling configuration-driven selection / Splitter provider，基于配置实现不同拆分策略的选择，
of different splitting strategies without code changes. / 无需修改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from src.core.settings import Settings


def _register_builtin_providers() -> None:
    """Register built-in splitter providers. / 注册内置 splitter provider。
    
    This function is called automatically when the module is imported. / 此函数在模块导入时自动调用。
    It registers all available splitter implementations with the factory. / 它会将所有可用 splitter 实现注册到工厂。
    """
    # Import here to avoid circular imports and handle missing dependencies gracefully / 在这里导入以避免循环导入，并优雅处理缺失依赖
    try:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
        SplitterFactory.register_provider("recursive", RecursiveSplitter)
    except ImportError:
        pass  # RecursiveSplitter not available (missing langchain dependency) / RecursiveSplitter 不可用（缺少 langchain 依赖）


class SplitterFactory:
    """Factory for creating Splitter provider instances. / 用于创建 Splitter provider 实例的工厂。
    
    This factory reads the splitter configuration from settings and instantiates / 此工厂从 settings 读取 splitter 配置并实例化
    the corresponding Splitter implementation. Supported providers will be added / 对应的 Splitter 实现。支持的 provider 会在
    in subsequent tasks (B7.5). / 后续任务（B7.5）中添加。
    
    Design Principles Applied: / 应用的设计原则：
    - Factory Pattern: Centralizes object creation logic. / 工厂模式：集中对象创建逻辑。
    - Config-Driven: Provider selection based on settings.yaml. / 配置驱动：基于 settings.yaml 选择 provider。
    - Fail-Fast: Raises clear errors for unknown providers. / 快速失败：对未知 provider 抛出清晰错误。
    """
    
    _PROVIDERS: dict[str, type[BaseSplitter]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseSplitter]) -> None:
        """Register a new Splitter provider implementation. / 注册新的 Splitter provider 实现。
        
        Args: / 参数：
            name: The provider identifier (e.g., 'recursive', 'semantic', 'fixed'). / provider 标识符（例如 'recursive'、'semantic'、'fixed'）。
            provider_class: The BaseSplitter subclass implementing the provider. / 实现 provider 的 BaseSplitter 子类。
        
        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseSplitter. / 如果 provider_class 未继承 BaseSplitter。
        """
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseSplitter"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseSplitter:
        """Create a Splitter instance based on configuration. / 基于配置创建 Splitter 实例。
        
        Args: / 参数：
            settings: The application settings containing ingestion configuration. / 包含 ingestion 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。
        
        Returns: / 返回：
            An instance of the configured Splitter provider. / 已配置 Splitter provider 的实例。
        
        Raises: / 异常：
            ValueError: If the configured provider is not supported or missing. / 如果配置的 provider 不受支持或缺失。
        """
        try:
            ingestion_settings = settings.ingestion
            if ingestion_settings is None:
                raise AttributeError("settings.ingestion is None")
            provider_name = ingestion_settings.splitter.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.ingestion.splitter. "
                "Please ensure 'ingestion.splitter' is specified in settings.yaml"
            ) from e
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{provider_name}'. "
                f"Available providers: {available}. "
                "Provider implementations will be added in task B7.5."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Splitter provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names. / 列出所有已注册的 provider 名称。
        
        Returns: / 返回：
            Sorted list of available provider identifiers. / 可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())


# Auto-register built-in providers when module is imported / 模块导入时自动注册内置 provider
_register_builtin_providers()
