"""Factory for creating LLM provider instances. / 用于创建 LLM provider 实例的工厂。

This module implements the Factory Pattern to instantiate the appropriate / 此模块实现工厂模式，用于实例化合适的
LLM provider based on configuration, enabling configuration-driven selection / LLM provider，基于配置实现不同后端的选择，
of different backends without code changes. / 无需修改代码。

Supports both text-only LLMs and Vision LLMs (multimodal). / 同时支持纯文本 LLM 和 Vision LLM（多模态）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


# Import and register Vision LLM providers at module load time / 模块加载时导入并注册 Vision LLM provider
def _register_vision_providers() -> None:
    """Register all Vision LLM provider implementations. / 注册所有 Vision LLM provider 实现。
    
    This function is called at module import time to populate the / 此函数在模块导入时调用，用于填充
    Vision LLM provider registry. Add new providers here as they / Vision LLM provider 注册表。实现新的 provider 后，
    are implemented. / 在这里添加。
    """
    try:
        from src.libs.llm.azure_vision_llm import AzureVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("azure", AzureVisionLLM)
    except ImportError:
        # Provider not yet implemented, skip registration / Provider 尚未实现，跳过注册
        pass
    
    try:
        from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("openai", OpenAIVisionLLM)
    except ImportError:
        pass


class LLMFactory:
    """Factory for creating LLM provider instances. / 用于创建 LLM provider 实例的工厂。
    
    This factory reads the provider configuration from settings and instantiates / 此工厂从 settings 读取 provider 配置并实例化
    the corresponding LLM implementation. Supports both text-only LLMs and / 对应的 LLM 实现。同时支持纯文本 LLM 和
    Vision LLMs (multimodal). / Vision LLM（多模态）。
    
    Design Principles Applied: / 应用的设计原则：
    - Factory Pattern: Centralizes object creation logic. / 工厂模式：集中对象创建逻辑
    - Config-Driven: Provider selection based on settings.yaml. / 配置驱动：基于 settings.yaml 选择 provider
    - Fail-Fast: Raises clear errors for unknown providers. / 快速失败：对未知 provider 抛出清晰错误
    - Separation: Text and Vision LLM registries are separate. / 分离：文本和 Vision LLM 注册表相互独立
    """
    
    # Registry of supported text-only LLM providers (to be populated in B7.x tasks) / 支持的纯文本 LLM provider 注册表（将在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseLLM]] = {}
    
    # Registry of supported Vision LLM providers (to be populated in B9+ tasks) / 支持的 Vision LLM provider 注册表（将在 B9+ 任务中填充）
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new LLM provider implementation. / 注册新的 LLM provider 实现。
        
        This method allows provider implementations to register themselves / 此方法允许 provider 实现将自身注册到
        with the factory, supporting extensibility. / 工厂中，以支持可扩展性。
        
        Args: / 参数：
            name: The provider identifier (e.g., 'openai', 'azure', 'ollama'). / provider 标识符（例如 'openai'、'azure'、'ollama'）。
            provider_class: The BaseLLM subclass implementing the provider. / 实现 provider 的 BaseLLM 子类。
        
        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseLLM. / 如果 provider_class 未继承 BaseLLM。
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """Create an LLM instance based on configuration. / 基于配置创建 LLM 实例。
        
        Args: / 参数：
            settings: The application settings containing LLM configuration. / 包含 LLM 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。
        
        Returns: / 返回：
            An instance of the configured LLM provider. / 已配置 LLM provider 的实例。
        
        Raises: / 异常：
            ValueError: If the configured provider is not supported. / 如果配置的 provider 不受支持。
            AttributeError: If required configuration fields are missing. / 如果缺少必需配置字段。
        
        Example: / 示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        # Extract provider name from settings / 从 settings 中提取 provider 名称
        try:
            provider_name = settings.llm.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.llm.provider. "
                "Please ensure 'llm.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in registry / 在注册表中查找 provider 类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in tasks B7.1-B7.2."
            )
        
        # Instantiate the provider / 实例化 provider
        # Provider classes should accept settings and optional kwargs / Provider 类应接收 settings 和可选 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names. / 列出所有已注册的 provider 名称。
        
        Returns: / 返回：
            Sorted list of available provider identifiers. / 可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())
    
    @classmethod
    def register_vision_provider(
        cls,
        name: str,
        provider_class: type[BaseVisionLLM]
    ) -> None:
        """Register a new Vision LLM provider implementation. / 注册新的 Vision LLM provider 实现。
        
        This method allows Vision LLM provider implementations to register / 此方法允许 Vision LLM provider 实现将自身注册到
        themselves with the factory, supporting extensibility. / 工厂中，以支持可扩展性。
        
        Args: / 参数：
            name: The provider identifier (e.g., 'azure', 'ollama'). / provider 标识符（例如 'azure'、'ollama'）。
            provider_class: The BaseVisionLLM subclass implementing the provider. / 实现 provider 的 BaseVisionLLM 子类。
        
        Raises: / 异常：
            ValueError: If provider_class doesn't inherit from BaseVisionLLM. / 如果 provider_class 未继承 BaseVisionLLM。
        """
        if not issubclass(provider_class, BaseVisionLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVisionLLM"
            )
        cls._VISION_PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **override_kwargs: Any
    ) -> BaseVisionLLM:
        """Create a Vision LLM instance based on configuration. / 基于配置创建 Vision LLM 实例。
        
        Vision LLMs support multimodal input (text + image) and are used for / Vision LLM 支持多模态输入（文本 + 图片），用于
        tasks like image captioning, visual question answering, and document / 图片说明、视觉问答和带嵌入图片的文档
        understanding with embedded images. / 理解等任务。
        
        Args: / 参数：
            settings: The application settings containing Vision LLM configuration. / 包含 Vision LLM 配置的应用设置。
            **override_kwargs: Optional parameters to override config values. / 用于覆盖配置值的可选参数。
        
        Returns: / 返回：
            An instance of the configured Vision LLM provider. / 已配置 Vision LLM provider 的实例。
        
        Raises: / 异常：
            ValueError: If the configured provider is not supported or configuration is missing. / 如果配置的 provider 不受支持或缺少配置。
            RuntimeError: If provider instantiation fails. / 如果 provider 实例化失败。
        
        Example: / 示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image("Describe this", image)
        """
        # Extract provider name from settings / 从 settings 中提取 provider 名称
        # Vision LLM config may be nested under settings.vision_llm or settings.llm / Vision LLM 配置可能嵌套在 settings.vision_llm 或 settings.llm 下
        try:
            # Try vision_llm section first / 先尝试 vision_llm 配置段
            if hasattr(settings, 'vision_llm') and hasattr(settings.vision_llm, 'provider'):
                provider_name = settings.vision_llm.provider.lower()
            # Fallback to llm.provider (some providers support both text and vision) / 回退到 llm.provider（部分 provider 同时支持文本和视觉）
            elif hasattr(settings, 'llm') and hasattr(settings.llm, 'provider'):
                provider_name = settings.llm.provider.lower()
            else:
                raise AttributeError("No vision_llm or llm provider configuration found")
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vision_llm.provider or settings.llm.provider. "
                "Please ensure 'vision_llm.provider' or 'llm.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in vision registry / 在 vision 注册表中查找 provider 类
        provider_class = cls._VISION_PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._VISION_PROVIDERS.keys())) if cls._VISION_PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available Vision LLM providers: {available}. "
                f"Vision LLM implementations will be added in tasks B9+."
            )
        
        # Instantiate the provider / 实例化 provider
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """List all registered Vision LLM provider names. / 列出所有已注册的 Vision LLM provider 名称。
        
        Returns: / 返回：
            Sorted list of available Vision LLM provider identifiers. / 可用 Vision LLM provider 标识符的排序列表。
        """
        return sorted(cls._VISION_PROVIDERS.keys())


# Register Vision LLM providers at module load time / 模块加载时注册 Vision LLM provider
_register_vision_providers()
