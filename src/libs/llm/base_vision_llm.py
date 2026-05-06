"""Abstract base class for Vision LLM providers. / Vision LLM provider 的抽象基类。

This module defines the pluggable interface for Vision Language Model providers, / 此模块定义 Vision Language Model provider 的可插拔接口，
enabling multimodal interactions (text + image) with seamless switching between / 支持多模态交互（文本 + 图片），并可通过配置
different backends (Azure Vision, Ollama Vision, etc.) through configuration. / 在不同后端（Azure Vision、Ollama Vision 等）之间无缝切换。

Vision LLMs extend standard LLMs by accepting image inputs alongside text prompts, / Vision LLM 通过在文本提示之外接收图片输入来扩展标准 LLM，
enabling tasks like image captioning, visual question answering, and document / 支持图片说明、视觉问答和带嵌入图片的文档
understanding with embedded images. / 理解等任务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """Represents an image input for Vision LLM. / 表示 Vision LLM 的图片输入。
    
    Supports multiple input formats: / 支持多种输入格式：
    - File path: Local image file to be read and encoded / 文件路径：要读取并编码的本地图片文件
    - Bytes: Raw image bytes (already loaded) / 字节：原始图片字节（已加载）
    - Base64: Already encoded image string / Base64：已编码的图片字符串
    
    Attributes: / 属性：
        path: Path to the image file (if loading from disk). / 图片文件路径（如果从磁盘加载）。
        data: Raw image bytes (if already loaded). / 原始图片字节（如果已加载）。
        base64: Base64-encoded image string (if already encoded). / Base64 编码图片字符串（如果已编码）。
        mime_type: MIME type of the image (e.g., 'image/png', 'image/jpeg'). / 图片 MIME 类型（例如 'image/png'、'image/jpeg'）。
    """
    path: Optional[Union[str, Path]] = None
    data: Optional[bytes] = None
    base64: Optional[str] = None
    mime_type: str = "image/png"
    
    def __post_init__(self) -> None:
        """Validate that exactly one input format is provided. / 校验恰好提供一种输入格式。"""
        provided_inputs = sum([
            self.path is not None,
            self.data is not None,
            self.base64 is not None,
        ])
        if provided_inputs == 0:
            raise ValueError("Must provide one of: path, data, or base64")
        if provided_inputs > 1:
            raise ValueError("Must provide exactly one of: path, data, or base64")


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers. / Vision LLM provider 的抽象基类。
    
    Vision LLMs accept both text and image inputs, enabling multimodal / Vision LLM 同时接收文本和图片输入，支持多模态
    understanding tasks such as image captioning, visual question answering, / 理解任务，例如图片说明、视觉问答，
    and document analysis with embedded images. / 以及带嵌入图片的文档分析。
    
    All Vision LLM implementations must inherit from this class and implement / 所有 Vision LLM 实现都必须继承此类并实现
    the chat_with_image() method. This ensures consistent interface across / chat_with_image() 方法。这确保不同
    different providers (Azure Vision, Ollama Vision, etc.). / provider（Azure Vision、Ollama Vision 等）之间接口一致。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Subclasses can be swapped without changing upstream code. / 可插拔：无需修改上游代码即可替换子类
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选 TraceContext 以集成可观测能力
    - Config-Driven: Instances are created via factory based on settings. / 配置驱动：基于 settings 通过工厂创建实例
    - Interface Segregation: Minimal interface focused on multimodal input. / 接口隔离：最小接口聚焦于多模态输入
    - Extension Point: Image preprocessing (compression, format conversion) can be / 扩展点：图片预处理（压缩、格式转换）可以
      added in subclasses without changing the base interface. / 在子类中添加，而无需修改基础接口。
    """
    
    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response based on text prompt and image input. / 基于文本提示和图片输入生成响应。
        
        This method enables multimodal interactions where the model can "see" / 此方法支持多模态交互，使模型可以“看到”
        the image and respond to questions or generate descriptions about it. / 图片并回答问题或生成关于它的描述。
        
        Args: / 参数：
            text: The text prompt or question about the image. / 关于图片的文本提示或问题。
            image: The image input (path, bytes, or base64). / 图片输入（路径、字节或 base64）。
            messages: Optional conversation history for context. If provided, / 可选会话历史上下文。如果提供，
                the text + image will be appended as the latest user message. / 文本 + 图片会作为最新 user 消息追加。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.). / provider 特有参数（temperature、max_tokens 等）。
        
        Returns: / 返回：
            ChatResponse containing the generated text and metadata. / 包含生成文本和元数据的 ChatResponse。
        
        Raises: / 异常：
            ValueError: If text is empty or image input is invalid. / 如果文本为空或图片输入无效。
            RuntimeError: If the Vision LLM provider call fails. / 如果 Vision LLM provider 调用失败。
        
        Example: / 示例：
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="Describe this diagram",
            ...     image=image
            ... )
            >>> print(response.content)
            "This diagram shows a system architecture with..."
        """
        pass
    
    def validate_text(self, text: str) -> None:
        """Validate text prompt. / 校验文本提示。
        
        Args: / 参数：
            text: Text prompt to validate. / 要校验的文本提示。
        
        Raises: / 异常：
            ValueError: If text is empty or not a string. / 如果文本为空或不是字符串。
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        if not text or not text.strip():
            raise ValueError("Text prompt cannot be empty")
    
    def validate_image(self, image: ImageInput) -> None:
        """Validate image input. / 校验图片输入。
        
        Args: / 参数：
            image: Image input to validate. / 要校验的图片输入。
        
        Raises: / 异常：
            ValueError: If image is not an ImageInput instance. / 如果 image 不是 ImageInput 实例。
        """
        if not isinstance(image, ImageInput):
            raise ValueError(
                f"Image must be an ImageInput instance, got {type(image).__name__}"
            )
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """Preprocess image before sending to Vision LLM. / 发送给 Vision LLM 前预处理图片。
        
        This method provides an extension point for image preprocessing such as: / 此方法为图片预处理提供扩展点，例如：
        - Resizing to meet provider size limits / 调整大小以满足 provider 尺寸限制
        - Format conversion (e.g., PNG to JPEG) / 格式转换（例如 PNG 转 JPEG）
        - Compression to reduce payload size / 压缩以减少 payload 大小
        
        Default implementation returns the image unchanged. Subclasses can / 默认实现原样返回图片。子类可以
        override to add provider-specific preprocessing. / 重写以添加 provider 特有预处理。
        
        Args: / 参数：
            image: The input image to preprocess. / 要预处理的输入图片。
            max_size: Optional maximum dimensions (width, height) in pixels. / 可选最大尺寸（宽、高），单位为像素。
        
        Returns: / 返回：
            Preprocessed ImageInput (may be the same instance if no changes needed). / 预处理后的 ImageInput（如果无需改变，可能是同一实例）。
        
        Note: / 说明：
            Preprocessing should be idempotent - calling it multiple times / 预处理应是幂等的：对相同输入多次调用
            with the same input should produce the same output. / 应产生相同输出。
        """
        # Default: no preprocessing / 默认：不做预处理
        return image
