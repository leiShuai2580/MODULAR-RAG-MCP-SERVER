"""OpenAI-compatible Vision LLM implementation. / OpenAI 兼容 Vision LLM 实现。

This module provides an OpenAI-compatible Vision LLM implementation for multimodal / 此模块提供面向多模态
interactions (text + image). Supports GPT-4o and similar vision-capable models. / 交互（文本 + 图片）的 OpenAI 兼容 Vision LLM 实现。支持 GPT-4o 和类似的视觉能力模型。

When azure_endpoint is present in settings, the provider automatically constructs / 当 settings 中存在 azure_endpoint 时，provider 会自动构造
the Azure-compatible URL and uses api-key auth header, allowing the same OpenAI / Azure 兼容 URL 并使用 api-key 认证头，从而让同一 OpenAI
standard protocol to work against Azure OpenAI endpoints. / 标准协议可用于 Azure OpenAI 端点。
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class OpenAIVisionLLMError(RuntimeError):
    """Raised when OpenAI Vision API call fails. / OpenAI Vision API 调用失败时抛出。"""


class OpenAIVisionLLM(BaseVisionLLM):
    """OpenAI-compatible Vision LLM provider implementation. / OpenAI 兼容 Vision LLM provider 实现。
    
    This class implements the BaseVisionLLM interface using the OpenAI standard / 此类使用 OpenAI 标准
    protocol. It supports both standard OpenAI endpoints and Azure OpenAI / 协议实现 BaseVisionLLM 接口。它同时支持标准 OpenAI 端点和
    endpoints in compatibility mode. / 兼容模式下的 Azure OpenAI 端点。
    
    When azure_endpoint is detected in settings.vision_llm, it automatically: / 当在 settings.vision_llm 中检测到 azure_endpoint 时，它会自动：
    - Constructs the deployment-based URL / 构造基于 deployment 的 URL
    - Uses api-key header for authentication / 使用 api-key 请求头认证
    - Appends api-version query parameter / 追加 api-version 查询参数
    
    Attributes: / 属性：
        api_key: The API key for authentication. / 用于认证的 API key。
        base_url: The base URL for the API. / API 基础 URL。
        model: The model identifier / deployment name. / 模型标识符 / deployment 名称。
        api_version: Optional API version (for Azure compat). / 可选 API 版本（用于 Azure 兼容）。
        max_image_size: Maximum image dimension in pixels (default 2048). / 最大图片尺寸，单位像素（默认 2048）。
        default_temperature: Default temperature for generation. / 生成时的默认 temperature。
        default_max_tokens: Default max tokens for generation. / 生成时的默认最大 token 数。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = OpenAIVisionLLM(settings)
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image("Describe this", image)
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MAX_IMAGE_SIZE = 2048  # pixels / 像素
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI Vision LLM provider. / 初始化 OpenAI Vision LLM provider。
        
        Args: / 参数：
            settings: Application settings containing vision_llm configuration. / 包含 vision_llm 配置的应用设置。
            api_key: Optional API key override. / 可选 API key 覆盖值。
            base_url: Optional base URL override. / 可选基础 URL 覆盖值。
            max_image_size: Maximum image dimension in pixels for auto-compression. / 自动压缩使用的最大图片尺寸，单位像素。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If required configuration is missing. / 如果缺少必需配置。
        """
        # Get vision settings section / 获取 vision 配置段
        vision_settings = getattr(settings, "vision_llm", None)
        
        # Temperature / max_tokens: vision_llm section > llm section defaults / Temperature / max_tokens：vision_llm 配置段 > llm 配置段默认值
        self.default_temperature = getattr(settings.llm, 'temperature', 0.0)
        self.default_max_tokens = getattr(settings.llm, 'max_tokens', 4096)
        
        # Model / deployment name / 模型 / deployment 名称
        vision_model = getattr(vision_settings, 'model', None) if vision_settings else None
        vision_dep = getattr(vision_settings, 'deployment_name', None) if vision_settings else None
        self.model = vision_dep or vision_model or settings.llm.model
        
        # Max image size / 最大图片尺寸
        vision_max_size = getattr(vision_settings, 'max_image_size', None) if vision_settings else None
        self.max_image_size = max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE
        
        # API key: explicit > vision_settings > llm settings > env var / API key：显式参数 > vision_settings > llm settings > 环境变量
        self.api_key = api_key
        if not self.api_key and vision_settings:
            self.api_key = getattr(vision_settings, 'api_key', None)
        if not self.api_key:
            self.api_key = getattr(settings.llm, 'api_key', None)
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (vision_llm.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure-compatible mode detection / Azure 兼容模式检测
        azure_endpoint = None
        if vision_settings:
            azure_endpoint = getattr(vision_settings, 'azure_endpoint', None)
        if not azure_endpoint:
            azure_endpoint = getattr(settings.llm, 'azure_endpoint', None)
        
        self.api_version = None
        if vision_settings:
            self.api_version = getattr(vision_settings, 'api_version', None)
        if not self.api_version:
            self.api_version = getattr(settings.llm, 'api_version', None)
        
        self._use_azure_auth = False
        
        if base_url:
            self.base_url = base_url
        elif azure_endpoint:
            # Azure-compatible mode / Azure 兼容模式
            self.base_url = (
                f"{azure_endpoint.rstrip('/')}/openai/deployments/{self.model}"
            )
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            self.base_url = self.DEFAULT_BASE_URL
        
        self._extra_config = kwargs
    
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response based on text prompt and image input. / 基于文本提示和图片输入生成响应。
        
        Args: / 参数：
            text: The text prompt or question about the image. / 关于图片的文本提示或问题。
            image: The image input (path, bytes, or base64). / 图片输入（路径、字节或 base64）。
            messages: Optional conversation history for context. / 可选会话历史上下文。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Override parameters (temperature, max_tokens, etc.). / 覆盖参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse containing the generated text and metadata. / 包含生成文本和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If text or image input is invalid. / 如果文本或图片输入无效。
            OpenAIVisionLLMError: If API call fails. / 如果 API 调用失败。
        """
        # Validate inputs / 校验输入
        self.validate_text(text)
        self.validate_image(image)
        
        # Preprocess image (compress if needed) / 预处理图片（需要时压缩）
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )
        
        # Convert image to base64 / 将图片转换为 base64
        image_base64 = self._get_image_base64(processed_image)
        
        # Prepare request parameters / 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        
        # Build message list / 构建消息列表
        api_messages = []
        if messages:
            api_messages.extend([{"role": m.role, "content": m.content} for m in messages])
        
        # Add current text + image message / 添加当前文本 + 图片消息
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{processed_image.mime_type};base64,{image_base64}"
                    }
                }
            ]
        }
        api_messages.append(current_message)
        
        # Make API call / 发起 API 调用
        try:
            response_data = self._call_api(
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OpenAIVisionLLMError):
                raise
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """Preprocess image before sending to Vision API. / 发送给 Vision API 前预处理图片。
        
        Compresses image if it exceeds max_size to reduce payload size. / 如果图片超过 max_size，则压缩以减少 payload 大小。
        
        Args: / 参数：
            image: The input image to preprocess. / 要预处理的输入图片。
            max_size: Maximum dimensions (width, height) in pixels. / 最大尺寸（宽、高），单位为像素。
        
        Returns: / 返回：
            Preprocessed ImageInput with compressed data if needed. / 需要时包含压缩数据的预处理后 ImageInput。
        """
        if not max_size:
            return image
        
        try:
            from PIL import Image
        except ImportError:
            return image
        
        # Get image bytes / 获取图片字节
        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            return image
        else:
            return image
        
        # Load image and check size / 加载图片并检查尺寸
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        max_width, max_height = max_size
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate new size maintaining aspect ratio / 计算保持宽高比的新尺寸
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Resize image / 调整图片大小
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes / 转换为字节
        buffer = io.BytesIO()
        img_format = img.format or "PNG"
        img_resized.save(buffer, format=img_format)
        compressed_bytes = buffer.getvalue()
        
        return ImageInput(
            data=compressed_bytes,
            mime_type=image.mime_type
        )
    
    def _get_image_base64(self, image: ImageInput) -> str:
        """Convert ImageInput to base64 string. / 将 ImageInput 转换为 base64 字符串。"""
        try:
            if image.base64:
                return image.base64
            elif image.data:
                return base64.b64encode(image.data).decode("utf-8")
            elif image.path:
                image_bytes = Path(image.path).read_bytes()
                return base64.b64encode(image_bytes).decode("utf-8")
            else:
                raise ValueError("ImageInput has no valid data source")
        except Exception as e:
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] Failed to encode image: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Make HTTP request to the Vision API. / 向 Vision API 发起 HTTP 请求。
        
        Args: / 参数：
            messages: List of API-formatted messages. / API 格式的消息列表。
            temperature: Generation temperature. / 生成 temperature。
            max_tokens: Maximum tokens to generate. / 要生成的最大 token 数。
        
        Returns: / 返回：
            API response as dictionary. / 字典形式的 API 响应。
        
        Raises: / 异常：
            OpenAIVisionLLMError: If API call fails. / 如果 API 调用失败。
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
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise OpenAIVisionLLMError(
                        f"[OpenAI Vision] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise OpenAIVisionLLMError(
                "[OpenAI Vision] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response. / 从 API 响应中解析错误详情。"""
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
