"""OpenAI-compatible LLM implementation. / OpenAI 兼容 LLM 实现。

This module provides the OpenAI LLM implementation that works with / 此模块提供可配合
the standard OpenAI API. It can also be used with other OpenAI-compatible / 标准 OpenAI API 使用的 OpenAI LLM 实现。也可以通过配置 base_url，
endpoints by configuring the base_url. / 与其他 OpenAI 兼容端点配合使用。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OpenAILLMError(RuntimeError):
    """Raised when OpenAI API call fails. / OpenAI API 调用失败时抛出。"""


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation. / OpenAI LLM provider 实现。
    
    This class implements the BaseLLM interface for OpenAI's chat completion API. / 此类为 OpenAI 的聊天补全 API 实现 BaseLLM 接口。
    It supports the standard OpenAI API and any OpenAI-compatible endpoints. / 它支持标准 OpenAI API 和任何 OpenAI 兼容端点。
    
    Attributes: / 属性：
        api_key: The API key for authentication. / 用于认证的 API key。
        base_url: The base URL for the API (default: OpenAI's endpoint). / API 基础 URL（默认：OpenAI 端点）。
        model: The model identifier to use. / 要使用的模型标识符。
        default_temperature: Default temperature for generation. / 生成时的默认 temperature。
        default_max_tokens: Default max tokens for generation. / 生成时的默认最大 token 数。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OpenAILLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI LLM provider. / 初始化 OpenAI LLM provider。
        
        Args: / 参数：
            settings: Application settings containing LLM configuration. / 包含 LLM 配置的应用设置。
            api_key: Optional API key override (falls back to settings.llm.api_key or env var). / 可选 API key 覆盖值（回退到 settings.llm.api_key 或环境变量）。
            base_url: Optional base URL override. / 可选基础 URL 覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If API key is not provided and not found in environment. / 如果未提供 API key 且环境中也找不到。
        
        Note: / 说明：
            When azure_endpoint is present in settings, the provider automatically / 当 settings 中存在 azure_endpoint 时，provider 会自动
            constructs the Azure-compatible OpenAI URL and uses api-key auth header. / 构造兼容 Azure 的 OpenAI URL，并使用 api-key 认证头。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # API key: explicit > settings > env var / API key：显式参数 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure-compatible mode detection / Azure 兼容模式检测
        azure_endpoint = getattr(settings.llm, 'azure_endpoint', None)
        self.api_version = getattr(settings.llm, 'api_version', None)
        
        if base_url:
            self.base_url = base_url
            self._use_azure_auth = False
        elif azure_endpoint:
            # Azure-compatible mode: construct deployment-based URL / Azure 兼容模式：构造基于 deployment 的 URL
            deployment = getattr(settings.llm, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            self.base_url = self.DEFAULT_BASE_URL
            self._use_azure_auth = False
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using OpenAI API. / 使用 OpenAI API 生成聊天补全。
        
        Args: / 参数：
            messages: List of conversation messages. / 会话消息列表。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (temperature, max_tokens, etc.). / 覆盖参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse with generated content and metadata. / 包含生成内容和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If messages are invalid. / 如果 messages 无效。
            OpenAILLMError: If API call fails. / 如果 API 调用失败。
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
            raise OpenAILLMError(
                f"[OpenAI] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OpenAILLMError):
                raise
            raise OpenAILLMError(
                f"[OpenAI] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Make the actual API call to OpenAI. / 发起实际的 OpenAI API 调用。
        
        This method is separated to allow easy mocking in tests. / 此方法被拆出，便于测试中 mock。
        
        Args: / 参数：
            messages: Messages in API format. / API 格式的消息。
            model: Model identifier. / 模型标识符。
            temperature: Generation temperature. / 生成 temperature。
            max_tokens: Maximum tokens to generate. / 要生成的最大 token 数。
        
        Returns: / 返回：
            Raw API response as dictionary. / 字典形式的原始 API 响应。
        
        Raises: / 异常：
            OpenAILLMError: If the API call fails. / 如果 API 调用失败。
        """
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if self.api_version:
            url += f"?api-version={self.api_version}"
        
        if self._use_azure_auth:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }
        else:
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
                    raise OpenAILLMError(
                        f"[OpenAI] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise OpenAILLMError(
                f"[OpenAI] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise OpenAILLMError(
                f"[OpenAI] Connection failed: {type(e).__name__}: {e}"
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
