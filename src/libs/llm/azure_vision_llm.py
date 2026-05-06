"""Azure OpenAI Vision LLM implementation. / Azure OpenAI Vision LLM 实现。

This module provides Azure OpenAI Vision LLM implementation for multimodal / 此模块提供面向多模态
interactions (text + image). Supports GPT-4o and GPT-4-Vision-Preview models / 交互（文本 + 图片）的 Azure OpenAI Vision LLM 实现。支持 GPT-4o 和 GPT-4-Vision-Preview 模型，
for image understanding tasks like image captioning, visual question answering, / 用于图片说明、视觉问答
and document analysis. / 和文档分析等图片理解任务。
"""

from __future__ import annotations

import base64
import io
import os
import requests
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class AzureVisionLLMError(RuntimeError):
    """Raised when Azure Vision API call fails. / Azure Vision API 调用失败时抛出。"""


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM provider implementation. / Azure OpenAI Vision LLM provider 实现。
    
    This class implements the BaseVisionLLM interface for Azure's OpenAI Vision / 此类为 Azure OpenAI Vision
    Service, supporting GPT-4o and GPT-4-Vision-Preview models. It handles / Service 实现 BaseVisionLLM 接口，支持 GPT-4o 和 GPT-4-Vision-Preview 模型。它处理
    Azure-specific authentication, endpoint configuration, and image preprocessing. / Azure 特有认证、端点配置和图片预处理。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Implements BaseVisionLLM for seamless provider switching / 可插拔：实现 BaseVisionLLM 以便无缝切换 provider
    - Config-Driven: Uses settings.yaml for all configuration / 配置驱动：所有配置使用 settings.yaml
    - Observable: Accepts TraceContext parameter (reserved for Stage F) / 可观测：接收 TraceContext 参数（为 Stage F 预留）
    - Graceful Errors: Provides clear, actionable error messages / 优雅错误：提供清晰、可操作的错误信息
    - Image Preprocessing: Auto-compresses images exceeding max_image_size / 图片预处理：自动压缩超过 max_image_size 的图片
    
    Attributes: / 属性：
        api_key: The Azure API key for authentication. / 用于认证的 Azure API key。
        endpoint: The Azure OpenAI endpoint URL. / Azure OpenAI 端点 URL。
        deployment_name: The deployment name for the Vision model. / Vision 模型的 deployment 名称。
        api_version: The API version to use. / 要使用的 API 版本。
        max_image_size: Maximum image dimension in pixels (default 2048). / 最大图片尺寸，单位像素（默认 2048）。
        default_temperature: Default temperature for generation. / 生成时的默认 temperature。
        default_max_tokens: Default max tokens for generation. / 生成时的默认最大 token 数。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = AzureVisionLLM(
        ...     settings,
        ...     endpoint='https://my-resource.openai.azure.com',
        ...     deployment_name='gpt-4o'
        ... )
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image(
        ...     text="Describe this diagram",
        ...     image=image
        ... )
    """
    
    DEFAULT_API_VERSION = "2024-02-15-preview"
    DEFAULT_MAX_IMAGE_SIZE = 2048  # pixels / 像素
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI Vision LLM provider. / 初始化 Azure OpenAI Vision LLM provider。
        
        Args: / 参数：
            settings: Application settings containing LLM configuration. / 包含 LLM 配置的应用设置。
            api_key: Optional API key override (falls back to env var). / 可选 API key 覆盖值（回退到环境变量）。
            endpoint: Optional endpoint override (falls back to env var). / 可选端点覆盖值（回退到环境变量）。
            deployment_name: Optional deployment name (defaults to settings.llm.model). / 可选 deployment 名称（默认使用 settings.llm.model）。
            api_version: Optional API version override. / 可选 API 版本覆盖值。
            max_image_size: Maximum image dimension in pixels for auto-compression. / 自动压缩使用的最大图片尺寸，单位像素。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If required configuration is missing. / 如果缺少必需配置。
        """
        self.deployment_name = deployment_name or settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # Check source of vision settings / 检查 vision 配置来源
        vision_settings = getattr(settings, "vision_llm", None)
        
        # Resolve Deployment Name / 解析 Deployment 名称
        # Priority: arg > vision_settings.deployment > vision_settings.model > settings.llm.model / 优先级：参数 > vision_settings.deployment > vision_settings.model > settings.llm.model
        vision_dep = getattr(vision_settings, "deployment_name", None)
        vision_model = getattr(vision_settings, "model", None)
        self.deployment_name = deployment_name or vision_dep or vision_model or settings.llm.model
        
        # Resolve Max Image Size / 解析最大图片尺寸
        # Priority: arg > vision_settings.max_image_size > DEFAULT / 优先级：参数 > vision_settings.max_image_size > 默认值
        vision_max_size = getattr(vision_settings, "max_image_size", None)
        self.max_image_size = max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE

            
        # API Key Resolution Order: / API Key 解析顺序：
        # 1. Constructor arg / 1. 构造函数参数
        # 2. Vision settings (settings.vision_llm.api_key) / 2. Vision 配置（settings.vision_llm.api_key）
        # 3. Env var (AZURE_OPENAI_API_KEY) / 3. 环境变量（AZURE_OPENAI_API_KEY）
        # 4. LLM settings fallback? (Usually not for API keys in settings object, but let's check) / 4. LLM 配置回退？（settings 对象中通常不用 API key，但这里检查一下）
        
        self.api_key = api_key 
        if not self.api_key and vision_settings and vision_settings.api_key:
             self.api_key = vision_settings.api_key
        if not self.api_key:
             self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
             
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY "
                "environment variable or pass api_key parameter."
            )
            
        # Endpoint Resolution Order / 端点解析顺序
        self.endpoint = endpoint
        if not self.endpoint and vision_settings and vision_settings.azure_endpoint:
             self.endpoint = vision_settings.azure_endpoint
        if not self.endpoint:
             self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or getattr(settings.llm, "azure_endpoint", None)
             
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided.")
            
        # API Version Resolution Order / API 版本解析顺序
        self.api_version = api_version
        if not self.api_version and vision_settings and vision_settings.api_version:
             self.api_version = vision_settings.api_version
        if not self.api_version:
             self.api_version = getattr(settings.llm, "api_version", None) or self.DEFAULT_API_VERSION
        
        # Validate initialized client / 校验已初始化的客户端配置
        # We don't initialize a client object here because we use request/httpx per call or custom logic / 这里不初始化客户端对象，因为每次调用使用 request/httpx 或自定义逻辑
        # But we ensures keys are present. / 但会确保 key 存在。

        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
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
        
        This method sends the text and image to Azure OpenAI Vision API / 此方法将文本和图片发送到 Azure OpenAI Vision API
        (GPT-4o or GPT-4-Vision-Preview) and returns the generated response. / （GPT-4o 或 GPT-4-Vision-Preview），并返回生成响应。
        
        Args: / 参数：
            text: The text prompt or question about the image. / 关于图片的文本提示或问题。
            image: The image input (path, bytes, or base64). / 图片输入（路径、字节或 base64）。
            messages: Optional conversation history for context. / 可选会话历史上下文。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (temperature, max_tokens, etc.). / 覆盖参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse containing the generated text and metadata. / 包含生成文本和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If text or image input is invalid. / 如果文本或图片输入无效。
            AzureVisionLLMError: If API call fails. / 如果 API 调用失败。
        
        Example: / 示例：
            >>> image = ImageInput(path="chart.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="What does this chart show?",
            ...     image=image
            ... )
        """
        # Validate inputs / 校验输入
        self.validate_text(text)
        self.validate_image(image)
        
        # Preprocess image (compress if needed) / 预处理图片（需要时压缩）
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )
        
        # Convert image to base64 if needed / 需要时将图片转换为 base64
        image_base64 = self._get_image_base64(processed_image)
        
        # Prepare request parameters / 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        deployment = kwargs.get("deployment_name", self.deployment_name)
        
        # Build message list / 构建消息列表
        api_messages = []
        if messages:
            # Add conversation history / 添加会话历史
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
            raise AzureVisionLLMError(
                f"[Azure Vision] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, AzureVisionLLMError):
                raise
            raise AzureVisionLLMError(
                f"[Azure Vision] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """Preprocess image before sending to Azure Vision API. / 发送给 Azure Vision API 前预处理图片。
        
        Compresses image if it exceeds max_size to reduce API payload size / 如果图片超过 max_size，则压缩以减少 API payload 大小，
        and meet Azure's size limits. Uses PIL for image manipulation. / 并满足 Azure 的大小限制。使用 PIL 处理图片。
        
        This operation is idempotent - calling it multiple times with the / 此操作是幂等的：对同一输入多次调用
        same input produces the same output. / 会产生相同输出。
        
        Args: / 参数：
            image: The input image to preprocess. / 要预处理的输入图片。
            max_size: Maximum dimensions (width, height) in pixels. / 最大尺寸（宽、高），单位为像素。
        
        Returns: / 返回：
            Preprocessed ImageInput with compressed data if needed. / 需要时包含压缩数据的预处理后 ImageInput。
        
        Note: / 说明：
            If the image is already within size limits, returns it unchanged. / 如果图片已经在尺寸限制内，则原样返回。
        """
        if not max_size:
            return image
        
        try:
            from PIL import Image
        except ImportError:
            # If PIL not available, skip preprocessing / 如果 PIL 不可用，则跳过预处理
            return image
        
        # Get image bytes / 获取图片字节
        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            # Already encoded, skip preprocessing / 已编码，跳过预处理
            return image
        else:
            return image
        
        # Load image and check size / 加载图片并检查尺寸
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        # Check if compression needed / 检查是否需要压缩
        max_width, max_height = max_size
        if width <= max_width and height <= max_height:
            # No compression needed / 不需要压缩
            return image
        
        # Calculate new size maintaining aspect ratio / 计算保持宽高比的新尺寸
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Resize image / 调整图片大小
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes / 转换为字节
        buffer = io.BytesIO()
        # Preserve original format if possible / 尽可能保留原始格式
        img_format = img.format or "PNG"
        img_resized.save(buffer, format=img_format)
        compressed_bytes = buffer.getvalue()
        
        # Return new ImageInput with compressed data / 返回带压缩数据的新 ImageInput
        return ImageInput(
            data=compressed_bytes,
            mime_type=image.mime_type
        )
    
    def _get_image_base64(self, image: ImageInput) -> str:
        """Convert ImageInput to base64 string. / 将 ImageInput 转换为 base64 字符串。
        
        Args: / 参数：
            image: The image to convert. / 要转换的图片。
        
        Returns: / 返回：
            Base64-encoded image string. / Base64 编码的图片字符串。
        
        Raises: / 异常：
            AzureVisionLLMError: If image cannot be encoded. / 如果图片无法编码。
        """
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
            raise AzureVisionLLMError(
                f"[Azure Vision] Failed to encode image: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: list[dict],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Make HTTP request to Azure OpenAI Vision API. / 向 Azure OpenAI Vision API 发起 HTTP 请求。
        
        Args: / 参数：
            messages: List of API-formatted messages. / API 格式的消息列表。
            deployment: Deployment name to use. / 要使用的 deployment 名称。
            temperature: Generation temperature. / 生成 temperature。
            max_tokens: Maximum tokens to generate. / 要生成的最大 token 数。
        
        Returns: / 返回：
            API response as dictionary. / 字典形式的 API 响应。
        
        Raises: / 异常：
            AzureVisionLLMError: If API call fails. / 如果 API 调用失败。
        """
        # Clean endpoint trail slash / 清理端点末尾斜杠
        endpoint = self.endpoint.rstrip("/")
        
        # Build API URL / 构建 API URL
        url = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        
        # Prepare request payload / 准备请求 payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
             error_details = e.response.text if e.response else str(e)
             raise AzureVisionLLMError(
                 f"[Azure Vision] HTTP Error: {e} - Response: {error_details}"
             ) from e
        except requests.exceptions.RequestException as e:
             raise AzureVisionLLMError(
                 f"[Azure Vision] Request failed: {e}"
             ) from e
