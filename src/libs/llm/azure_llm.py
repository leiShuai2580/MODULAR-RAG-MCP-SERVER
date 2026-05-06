"""Azure OpenAI LLM implementation. / Azure OpenAI LLM 实现。

This module provides the Azure OpenAI LLM implementation that works with / 此模块提供可配合
Azure's OpenAI Service API. It handles the Azure-specific authentication / Azure OpenAI Service API 使用的 Azure OpenAI LLM 实现。它处理 Azure 特有认证
and endpoint configuration. / 和端点配置。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class AzureLLMError(RuntimeError):
    """Raised when Azure OpenAI API call fails. / Azure OpenAI API 调用失败时抛出。"""


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM provider implementation. / Azure OpenAI LLM provider 实现。
    
    This class implements the BaseLLM interface for Azure's OpenAI Service. / 此类为 Azure OpenAI Service 实现 BaseLLM 接口。
    Azure uses a different authentication method (API key in header) and / 相比标准 OpenAI，Azure 使用不同的认证方式（请求头中的 API key）
    endpoint structure compared to standard OpenAI. / 和端点结构。
    
    Attributes: / 属性：
        api_key: The Azure API key for authentication. / 用于认证的 Azure API key。
        endpoint: The Azure OpenAI endpoint URL. / Azure OpenAI 端点 URL。
        deployment_name: The deployment name for the model. / 模型的 deployment 名称。
        api_version: The API version to use. / 要使用的 API 版本。
        default_temperature: Default temperature for generation. / 生成时的默认 temperature。
        default_max_tokens: Default max tokens for generation. / 生成时的默认最大 token 数。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = AzureLLM(settings, endpoint='https://my-resource.openai.azure.com')
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_API_VERSION = "2024-02-15-preview"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI LLM provider. / 初始化 Azure OpenAI LLM provider。
        
        Args: / 参数：
            settings: Application settings containing LLM configuration. / 包含 LLM 配置的应用设置。
            api_key: Optional API key override (falls back to settings.llm.api_key or env var). / 可选 API key 覆盖值（回退到 settings.llm.api_key 或环境变量）。
            endpoint: Optional endpoint override (falls back to settings.llm.azure_endpoint or env var). / 可选端点覆盖值（回退到 settings.llm.azure_endpoint 或环境变量）。
            deployment_name: Optional deployment name (defaults to settings.llm.deployment_name or model). / 可选 deployment 名称（默认使用 settings.llm.deployment_name 或 model）。
            api_version: Optional API version override. / 可选 API 版本覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If required configuration is missing. / 如果缺少必需配置。
        """
        # Deployment name: explicit > settings.deployment_name > settings.model / Deployment 名称：显式参数 > settings.deployment_name > settings.model
        self.deployment_name = (
            deployment_name 
            or getattr(settings.llm, 'deployment_name', None) 
            or settings.llm.model
        )
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # API key: explicit > settings > env var / API key：显式参数 > settings > 环境变量
        self.api_key = (
            api_key 
            or getattr(settings.llm, 'api_key', None) 
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Endpoint: explicit > settings > env var / 端点：显式参数 > settings > 环境变量
        self.endpoint = (
            endpoint 
            or getattr(settings.llm, 'azure_endpoint', None) 
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not provided. Set in settings.yaml (llm.azure_endpoint), "
                "AZURE_OPENAI_ENDPOINT environment variable, or pass endpoint parameter."
            )
        
        # API version: explicit > settings > default / API 版本：显式参数 > settings > 默认值
        self.api_version = (
            api_version 
            or getattr(settings.llm, 'api_version', None) 
            or self.DEFAULT_API_VERSION
        )
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using Azure OpenAI API. / 使用 Azure OpenAI API 生成聊天补全。
        
        Args: / 参数：
            messages: List of conversation messages. / 会话消息列表。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (temperature, max_tokens, etc.). / 覆盖参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse with generated content and metadata. / 包含生成内容和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If messages are invalid. / 如果 messages 无效。
            AzureLLMError: If API call fails. / 如果 API 调用失败。
        """
        # Validate input / 校验输入
        self.validate_messages(messages)
        
        # Prepare request parameters / 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        deployment = kwargs.get("deployment_name", self.deployment_name)
        
        # Convert messages to API format / 将消息转换为 API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Make API call / 发起 API 调用
        try:
            response_data = self._call_api(
                messages=api_messages,
                deployment=deployment,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Parse response / 解析响应
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", deployment),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise AzureLLMError(
                f"[Azure] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, AzureLLMError):
                raise
            raise AzureLLMError(
                f"[Azure] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Make the actual API call to Azure OpenAI. / 发起实际的 Azure OpenAI API 调用。
        
        This method is separated to allow easy mocking in tests. / 此方法被拆出，便于测试中 mock。
        
        Args: / 参数：
            messages: Messages in API format. / API 格式的消息。
            deployment: Deployment name. / Deployment 名称。
            temperature: Generation temperature. / 生成 temperature。
            max_tokens: Maximum tokens to generate. / 要生成的最大 token 数。
        
        Returns: / 返回：
            Raw API response as dictionary. / 字典形式的原始 API 响应。
        
        Raises: / 异常：
            AzureLLMError: If the API call fails. / 如果 API 调用失败。
        """
        import httpx
        
        # Azure endpoint format: / Azure 端点格式：
        # {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version} / {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
        url = (
            f"{self.endpoint.rstrip('/')}/openai/deployments/{deployment}/"
            f"chat/completions?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise AzureLLMError(
                        f"[Azure] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise AzureLLMError(
                f"[Azure] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise AzureLLMError(
                f"[Azure] Connection failed: {type(e).__name__}: {e}"
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
