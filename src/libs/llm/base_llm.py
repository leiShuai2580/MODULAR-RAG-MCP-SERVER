"""Abstract base class for LLM providers. / LLM provider 的抽象基类。

This module defines the pluggable interface for Language Model providers, / 此模块定义语言模型 provider 的可插拔接口，
enabling seamless switching between different backends (OpenAI, Azure, Ollama, etc.) / 支持在不同后端（OpenAI、Azure、Ollama 等）之间无缝切换，
through configuration-driven instantiation. / 并通过配置驱动的实例化完成选择。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Represents a single message in a chat conversation. / 表示聊天会话中的单条消息。
    
    Attributes: / 属性：
        role: The role of the message sender ('system', 'user', or 'assistant'). / 消息发送方角色（'system'、'user' 或 'assistant'）。
        content: The text content of the message. / 消息的文本内容。
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM chat completion. / 来自 LLM 聊天补全的响应。
    
    Attributes: / 属性：
        content: The generated text response. / 生成的文本响应。
        model: The model identifier that generated the response. / 生成响应的模型标识符。
        usage: Optional token usage statistics (prompt_tokens, completion_tokens, total_tokens). / 可选 token 使用统计（prompt_tokens、completion_tokens、total_tokens）。
        raw_response: Optional raw response from the provider for debugging. / 可选的 provider 原始响应用于调试。
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers. / LLM provider 的抽象基类。
    
    All LLM implementations must inherit from this class and implement / 所有 LLM 实现都必须继承此类并实现
    the chat() method. This ensures consistent interface across different / chat() 方法。这确保不同
    providers (OpenAI, Azure, DeepSeek, Ollama, etc.). / provider（OpenAI、Azure、DeepSeek、Ollama 等）之间接口一致。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Subclasses can be swapped without changing upstream code. / 可插拔：无需修改上游代码即可替换子类
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选 TraceContext 以集成可观测能力
    - Config-Driven: Instances are created via factory based on settings. / 配置驱动：基于 settings 通过工厂创建实例
    """
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion response. / 生成聊天补全响应。
        
        Args: / 参数：
            messages: List of conversation messages (role + content). / 会话消息列表（角色 + 内容）。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.). / provider 特有参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse containing the generated text and metadata. / 包含生成文本和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If messages list is empty or malformed. / 如果 messages 列表为空或格式错误。
            RuntimeError: If the LLM provider call fails. / 如果 LLM provider 调用失败。
        """
        pass
    
    def validate_messages(self, messages: List[Message]) -> None:
        """Validate message list structure. / 校验消息列表结构。
        
        Args: / 参数：
            messages: List of messages to validate. / 要校验的消息列表。
        
        Raises: / 异常：
            ValueError: If messages list is empty or contains invalid roles. / 如果 messages 列表为空或包含无效角色。
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message at index {i} is not a Message instance")
            if msg.role not in valid_roles:
                raise ValueError(
                    f"Message at index {i} has invalid role '{msg.role}'. "
                    f"Must be one of: {valid_roles}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message at index {i} has empty content")
