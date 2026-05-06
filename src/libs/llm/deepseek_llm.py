"""DeepSeek LLM implementation. / DeepSeek LLM 实现。

This module provides the DeepSeek LLM implementation that works with / 此模块提供可配合
DeepSeek's API. DeepSeek uses an OpenAI-compatible API format but with / DeepSeek API 使用的 DeepSeek LLM 实现。DeepSeek 使用 OpenAI 兼容 API 格式，
its own endpoint and authentication. / 但有自己的端点和认证方式。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class DeepSeekLLMError(RuntimeError):
    """Raised when DeepSeek API call fails. / DeepSeek API 调用失败时抛出。"""


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM provider implementation. / DeepSeek LLM provider 实现。
    
    This class implements the BaseLLM interface for DeepSeek's chat API. / 此类为 DeepSeek 的聊天 API 实现 BaseLLM 接口。
    DeepSeek provides an OpenAI-compatible API with its own endpoint. / DeepSeek 提供带自有端点的 OpenAI 兼容 API。
    
    Attributes: / 属性：
        api_key: The API key for authentication. / 用于认证的 API key。
        base_url: The base URL for the API. / API 基础 URL。
        model: The model identifier to use. / 要使用的模型标识符。
        default_temperature: Default temperature for generation. / 生成时的默认 temperature。
        default_max_tokens: Default max tokens for generation. / 生成时的默认最大 token 数。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = DeepSeekLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepSeek LLM provider. / 初始化 DeepSeek LLM provider。
        
        Args: / 参数：
            settings: Application settings containing LLM configuration. / 包含 LLM 配置的应用设置。
            api_key: Optional API key override (falls back to env var DEEPSEEK_API_KEY). / 可选 API key 覆盖值（回退到环境变量 DEEPSEEK_API_KEY）。
            base_url: Optional base URL override. / 可选基础 URL 覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If API key is not provided and not found in environment. / 如果未提供 API key 且环境中也找不到。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # API key: explicit > env var / API key：显式参数 > 环境变量
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Base URL: explicit > default / 基础 URL：显式参数 > 默认值
        self.base_url = base_url or self.DEFAULT_BASE_URL
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using DeepSeek API. / 使用 DeepSeek API 生成聊天补全。
        
        Args: / 参数：
            messages: List of conversation messages. / 会话消息列表。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (temperature, max_tokens, etc.). / 覆盖参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse with generated content and metadata. / 包含生成内容和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If messages are invalid. / 如果 messages 无效。
            DeepSeekLLMError: If API call fails. / 如果 API 调用失败。
        """
        # Validate input / 校验输入
        self.validate_messages(messages)
        
        # Prepare request parameters / 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        
        # Convert messages to API format / 将消息转换为 API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Make API call / 发起 API 调用
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Parse response / 解析响应
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise DeepSeekLLMError(
                f"[DeepSeek] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, DeepSeekLLMError):
                raise
            raise DeepSeekLLMError(
                f"[DeepSeek] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Make the actual API call to DeepSeek. / 发起实际的 DeepSeek API 调用。
        
        This method is separated to allow easy mocking in tests. / 此方法被拆出，便于测试中 mock。
        
        Args: / 参数：
            messages: Messages in API format. / API 格式的消息。
            model: Model identifier. / 模型标识符。
            temperature: Generation temperature. / 生成 temperature。
            max_tokens: Maximum tokens to generate. / 要生成的最大 token 数。
        
        Returns: / 返回：
            Raw API response as dictionary. / 字典形式的原始 API 响应。
        
        Raises: / 异常：
            DeepSeekLLMError: If the API call fails. / 如果 API 调用失败。
        """
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise DeepSeekLLMError(
                        f"[DeepSeek] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise DeepSeekLLMError(
                f"[DeepSeek] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise DeepSeekLLMError(
                f"[DeepSeek] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response. / 从 API 响应中解析错误详情。
        
        Args: / 参数：
            response: The HTTP response object. / HTTP 响应对象。
        
        Returns: / 返回：
            Human-readable error message. / 人类可读的错误信息。
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return response.text
        except Exception:
            return response.text or "Unknown error"
