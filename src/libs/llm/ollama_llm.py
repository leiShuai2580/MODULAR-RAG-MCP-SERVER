"""Ollama LLM implementation for local model inference. / 面向本地模型推理的 Ollama LLM 实现。

This module provides the Ollama LLM implementation that works with / 此模块提供与本地运行的
locally running Ollama instances. Ollama enables running LLMs like / Ollama 实例配合使用的 Ollama LLM 实现。Ollama 支持在本地硬件上运行
Llama, Mistral, CodeLlama, etc. on local hardware. / Llama、Mistral、CodeLlama 等 LLM。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OllamaLLMError(RuntimeError):
    """Raised when Ollama API call fails. / Ollama API 调用失败时抛出。
    
    This exception provides clear error messages without exposing / 此异常提供清晰错误信息，同时不暴露
    sensitive configuration details like internal URLs or credentials. / 内部 URL 或凭证等敏感配置细节。
    """


class OllamaLLM(BaseLLM):
    """Ollama LLM provider implementation for local inference. / 面向本地推理的 Ollama LLM provider 实现。
    
    This class implements the BaseLLM interface for Ollama's chat API, / 此类为 Ollama 的聊天 API 实现 BaseLLM 接口，
    enabling local LLM inference without cloud dependencies. / 支持无需云依赖的本地 LLM 推理。
    
    Attributes: / 属性：
        base_url: The base URL for the Ollama server (default: http://localhost:11434). / Ollama 服务器基础 URL（默认：http://localhost:11434）。
        model: The model identifier to use (e.g., 'llama3', 'mistral'). / 要使用的模型标识符（例如 'llama3'、'mistral'）。
        default_temperature: Default temperature for generation. / 生成时的默认 temperature。
        default_max_tokens: Default max tokens for generation (num_predict in Ollama). / 生成时的默认最大 token 数（Ollama 中为 num_predict）。
        timeout: Request timeout in seconds. / 请求超时时间（秒）。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OllamaLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # Longer timeout for local inference / 本地推理使用更长超时
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama LLM provider. / 初始化 Ollama LLM provider。
        
        Args: / 参数：
            settings: Application settings containing LLM configuration. / 包含 LLM 配置的应用设置。
            base_url: Optional base URL override (falls back to env var OLLAMA_BASE_URL). / 可选基础 URL 覆盖值（回退到环境变量 OLLAMA_BASE_URL）。
            timeout: Optional timeout override for requests. / 可选请求超时覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If required configuration is missing. / 如果缺少必需配置。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # Base URL: explicit > env var > default / 基础 URL：显式参数 > 环境变量 > 默认值
        self.base_url = (
            base_url 
            or os.environ.get("OLLAMA_BASE_URL") 
            or self.DEFAULT_BASE_URL
        )
        
        # Timeout: explicit > default / 超时：显式参数 > 默认值
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using Ollama API. / 使用 Ollama API 生成聊天补全。
        
        Args: / 参数：
            messages: List of conversation messages. / 会话消息列表。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (temperature, max_tokens, etc.). / 覆盖参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse with generated content and metadata. / 包含生成内容和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If messages are invalid. / 如果 messages 无效。
            OllamaLLMError: If API call fails. / 如果 API 调用失败。
        """
        # Validate input / 校验输入
        self.validate_messages(messages)
        
        # Prepare request parameters / 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        
        # Convert messages to Ollama API format / 将消息转换为 Ollama API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Make API call / 发起 API 调用
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Parse response - Ollama returns different format than OpenAI / 解析响应 - Ollama 返回格式不同于 OpenAI
            # Handle both /api/chat (streaming disabled) response format / 同时处理 /api/chat（禁用流式）响应格式
            if "message" in response_data:
                # Standard chat response / 标准聊天响应
                content = response_data["message"]["content"]
            elif "response" in response_data:
                # Legacy generate endpoint response (fallback) / 旧 generate 端点响应（回退）
                content = response_data["response"]
            else:
                raise OllamaLLMError(
                    "[Ollama] Unexpected response format: missing 'message' or 'response' key"
                )
            
            # Build usage stats if available / 如果可用则构建 usage 统计
            usage = None
            if "eval_count" in response_data or "prompt_eval_count" in response_data:
                usage = {
                    "prompt_tokens": response_data.get("prompt_eval_count", 0),
                    "completion_tokens": response_data.get("eval_count", 0),
                    "total_tokens": (
                        response_data.get("prompt_eval_count", 0) +
                        response_data.get("eval_count", 0)
                    ),
                }
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise OllamaLLMError(
                f"[Ollama] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OllamaLLMError):
                raise
            raise OllamaLLMError(
                f"[Ollama] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Make the actual API call to Ollama. / 发起实际的 Ollama API 调用。
        
        This method is separated to allow easy mocking in tests. / 此方法被拆出，便于测试中 mock。
        
        Args: / 参数：
            messages: Messages in API format. / API 格式的消息。
            model: Model identifier. / 模型标识符。
            temperature: Generation temperature. / 生成 temperature。
            max_tokens: Maximum tokens to generate (num_predict in Ollama). / 要生成的最大 token 数（Ollama 中为 num_predict）。
        
        Returns: / 返回：
            Raw API response as dictionary. / 字典形式的原始 API 响应。
        
        Raises: / 异常：
            OllamaLLMError: If the API call fails. / 如果 API 调用失败。
        """
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/api/chat"
        headers = {
            "Content-Type": "application/json",
        }
        
        # Ollama uses 'num_predict' instead of 'max_tokens' / Ollama 使用 'num_predict' 而不是 'max_tokens'
        # and 'options' object for model parameters / 并使用 'options' 对象传递模型参数
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,  # Disable streaming for synchronous response / 禁用流式以获得同步响应
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise OllamaLLMError(
                        f"[Ollama] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise OllamaLLMError(
                f"[Ollama] Request timed out after {self.timeout} seconds. "
                "Consider increasing timeout for larger models or longer responses."
            ) from e
        except httpx.ConnectError as e:
            raise OllamaLLMError(
                "[Ollama] Connection failed. Ensure Ollama is running locally. "
                "Start it with 'ollama serve' command."
            ) from e
        except httpx.RequestError as e:
            raise OllamaLLMError(
                f"[Ollama] Request failed: {type(e).__name__}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response. / 从 API 响应中解析错误详情。
        
        Args: / 参数：
            response: The HTTP response object. / HTTP 响应对象。
        
        Returns: / 返回：
            Human-readable error message without exposing sensitive details. / 不暴露敏感详情的人类可读错误信息。
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                return str(error_data["error"])
            return response.text[:200] if response.text else "Unknown error"
        except Exception:
            return response.text[:200] if response.text else "Unknown error"
