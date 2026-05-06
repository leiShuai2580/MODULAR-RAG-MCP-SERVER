"""
LLM Module. / LLM 模块。

This package contains LLM client abstractions and implementations: / 此包包含 LLM 客户端抽象和实现：
- Base LLM class (text-only) / 基础 LLM 类（纯文本）
- Base Vision LLM class (multimodal: text + image) / 基础 Vision LLM 类（多模态：文本 + 图片）
- LLM factory / LLM 工厂
- Provider implementations (OpenAI, Azure, Ollama, DeepSeek) / Provider 实现（OpenAI、Azure、Ollama、DeepSeek）
"""

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError
from src.libs.llm.azure_llm import AzureLLM, AzureLLMError
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError

# Register text-only LLM providers with factory / 将纯文本 LLM provider 注册到工厂
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)

# Note: Vision LLM providers will be registered in task B9+ / 注意：Vision LLM provider 会在任务 B9+ 中注册

__all__ = [
    # Base classes / 基类
    "BaseLLM",
    "BaseVisionLLM",
    # Data types / 数据类型
    "ChatResponse",
    "Message",
    "ImageInput",
    # Factory / 工厂
    "LLMFactory",
    # Text-only LLM implementations / 纯文本 LLM 实现
    "OpenAILLM",
    "OpenAILLMError",
    "AzureLLM",
    "AzureLLMError",
    "DeepSeekLLM",
    "DeepSeekLLMError",
    "OllamaLLM",
    "OllamaLLMError",
    # Vision LLM implementations / Vision LLM 实现
    "OpenAIVisionLLM",
    "OpenAIVisionLLMError",
]
