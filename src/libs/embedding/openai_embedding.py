"""OpenAI Embedding implementation. / OpenAI Embedding 实现。

This module provides the OpenAI Embedding implementation that works with / 此模块提供可配合
the standard OpenAI Embeddings API. / 标准 OpenAI Embeddings API 使用的 OpenAI Embedding 实现。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """Raised when OpenAI Embeddings API call fails. / OpenAI Embeddings API 调用失败时抛出。"""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding provider implementation. / OpenAI Embedding provider 实现。
    
    This class implements the BaseEmbedding interface for OpenAI's Embeddings API. / 此类为 OpenAI 的 Embeddings API 实现 BaseEmbedding 接口。
    It supports text-embedding-3-small, text-embedding-3-large, and older models / 它支持 text-embedding-3-small、text-embedding-3-large 以及旧模型，
    like text-embedding-ada-002. / 例如 text-embedding-ada-002。
    
    Attributes: / 属性：
        api_key: The API key for authentication. / 用于认证的 API key。
        model: The model identifier to use. / 要使用的模型标识符。
        dimensions: Optional dimension reduction (only for text-embedding-3-*). / 可选降维（仅适用于 text-embedding-3-*）。
        base_url: The base URL for the API (default: OpenAI's endpoint). / API 的基础 URL（默认：OpenAI 端点）。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OpenAIEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI Embedding provider. / 初始化 OpenAI Embedding provider。
        
        Args: / 参数：
            settings: Application settings containing Embedding configuration. / 包含 Embedding 配置的应用设置。
            api_key: Optional API key override (falls back to settings.embedding.api_key or env var). / 可选 API key 覆盖值（回退到 settings.embedding.api_key 或环境变量）。
            base_url: Optional base URL override. / 可选基础 URL 覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If API key is not provided and not found in environment. / 如果未提供 API key 且环境中也找不到。
        
        Note: / 说明：
            When azure_endpoint is present in settings, the provider automatically / 当 settings 中存在 azure_endpoint 时，provider 会自动
            constructs the Azure-compatible OpenAI URL and uses api-key auth. / 构造兼容 Azure 的 OpenAI URL，并使用 api-key 认证。
        """
        self.model = settings.embedding.model
        
        # Extract optional dimensions setting / 提取可选 dimensions 配置
        self.dimensions = getattr(settings.embedding, 'dimensions', None)
        
        # API key: explicit > settings > env var / API key：显式参数 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (embedding.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure-compatible mode detection / Azure 兼容模式检测
        azure_endpoint = getattr(settings.embedding, 'azure_endpoint', None)
        self.api_version = getattr(settings.embedding, 'api_version', None)
        self._use_azure_auth = False
        
        if base_url:
            self.base_url = base_url
        elif azure_endpoint:
            # Azure-compatible mode: construct deployment-based URL / Azure 兼容模式：构造基于 deployment 的 URL
            deployment = getattr(settings.embedding, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            settings_base_url = getattr(settings.embedding, 'base_url', None)
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using OpenAI API. / 使用 OpenAI API 为一批文本生成嵌入。
        
        Args: / 参数：
            texts: List of text strings to embed. Must not be empty. / 要嵌入的文本字符串列表，不能为空。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (dimensions, etc.). / 覆盖参数（dimensions 等）。
        
        Returns: / 返回：
            List of embedding vectors, where each vector is a list of floats. / 嵌入向量列表，其中每个向量都是浮点数列表。
            The length of the outer list matches len(texts). / 外层列表长度与 len(texts) 一致。
        
        Raises: / 异常：
            ValueError: If texts list is empty or contains invalid entries. / 如果 texts 列表为空或包含无效条目。
            OpenAIEmbeddingError: If API call fails. / 如果 API 调用失败。
        """
        # Validate input / 校验输入
        self.validate_texts(texts)
        
        # Import OpenAI client (lazy import to avoid dependency at module level) / 导入 OpenAI 客户端（延迟导入以避免模块级依赖）
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e
        
        # Initialize OpenAI client / 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        # Azure-compatible mode: add api-version query param and api-key header / Azure 兼容模式：添加 api-version 查询参数和 api-key 请求头
        if self._use_azure_auth and self.api_version:
            client_kwargs["default_query"] = {"api-version": self.api_version}
            client_kwargs["default_headers"] = {"api-key": self.api_key}
        
        client = OpenAI(**client_kwargs)
        
        # Prepare API call parameters / 准备 API 调用参数
        api_params = {
            "input": texts,
            "model": self.model,
        }
        
        # Add dimensions if specified (only for text-embedding-3-* models) / 如果指定则添加 dimensions（仅适用于 text-embedding-3-* 模型）
        # text-embedding-ada-002 does NOT support the dimensions parameter / text-embedding-ada-002 不支持 dimensions 参数
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and self.model.startswith("text-embedding-3"):
            api_params["dimensions"] = dimensions
        
        # Call OpenAI API / 调用 OpenAI API
        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"OpenAI Embeddings API call failed: {e}"
            ) from e
        
        # Extract embeddings from response / 从响应中提取嵌入
        # Response format: response.data is a list of objects with .embedding attribute / 响应格式：response.data 是带 .embedding 属性的对象列表
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise OpenAIEmbeddingError(
                f"Failed to parse OpenAI Embeddings API response: {e}"
            ) from e
        
        # Verify output matches input length / 校验输出与输入长度一致
        if len(embeddings) != len(texts):
            raise OpenAIEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        
        return embeddings
    
    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension for the configured model. / 获取已配置模型的嵌入维度。
        
        Returns: / 返回：
            The embedding dimension, or None if not deterministic. / 嵌入维度；如果无法确定则返回 None。
        
        Note: / 说明：
            For text-embedding-3-* models with custom dimensions, returns / 对带自定义 dimensions 的 text-embedding-3-* 模型，返回
            the configured dimension. For other models, returns their default. / 已配置维度。对其他模型，返回其默认值。
        """
        # If dimensions explicitly configured, return it / 如果显式配置了 dimensions，则返回它
        if self.dimensions is not None:
            return self.dimensions
        
        # Model-specific defaults / 模型特定默认值
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        return model_dimensions.get(self.model)
