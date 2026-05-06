"""Azure OpenAI Embedding implementation. / Azure OpenAI Embedding 实现。

This module provides the Azure OpenAI Embedding implementation, which handles / 此模块提供 Azure OpenAI Embedding 实现，用于处理
Azure-specific configuration (endpoint, api-version, deployment names) while / Azure 特有配置（endpoint、api-version、deployment names），同时
reusing the core OpenAI embedding logic. / 复用核心 OpenAI embedding 逻辑。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class AzureEmbeddingError(RuntimeError):
    """Raised when Azure OpenAI Embeddings API call fails. / Azure OpenAI Embeddings API 调用失败时抛出。"""


class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding provider implementation. / Azure OpenAI Embedding provider 实现。
    
    This class implements the BaseEmbedding interface for Azure OpenAI's Embeddings API. / 此类为 Azure OpenAI 的 Embeddings API 实现 BaseEmbedding 接口。
    It handles Azure-specific configuration like endpoint, api-version, and deployment names. / 它处理 endpoint、api-version、deployment names 等 Azure 特有配置。
    
    Attributes: / 属性：
        api_key: The Azure API key for authentication. / 用于认证的 Azure API key。
        azure_endpoint: The Azure OpenAI endpoint URL. / Azure OpenAI 端点 URL。
        api_version: The API version to use. / 要使用的 API 版本。
        deployment_name: The deployment name (replaces 'model' in standard OpenAI). / deployment 名称（替代标准 OpenAI 中的 'model'）。
        dimensions: Optional dimension reduction (only for text-embedding-3-*). / 可选降维（仅适用于 text-embedding-3-*）。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = AzureEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_API_VERSION = "2024-02-01"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI Embedding provider. / 初始化 Azure OpenAI Embedding provider。
        
        Args: / 参数：
            settings: Application settings containing Embedding configuration. / 包含 Embedding 配置的应用设置。
            api_key: Optional API key override (falls back to env var AZURE_OPENAI_API_KEY). / 可选 API key 覆盖值（回退到环境变量 AZURE_OPENAI_API_KEY）。
            azure_endpoint: Optional endpoint override (falls back to env var AZURE_OPENAI_ENDPOINT). / 可选端点覆盖值（回退到环境变量 AZURE_OPENAI_ENDPOINT）。
            api_version: Optional API version override. / 可选 API 版本覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If required Azure-specific configuration is missing. / 如果缺少必需的 Azure 特有配置。
        """
        # Azure uses 'deployment_name' instead of 'model' / Azure 使用 'deployment_name' 而不是 'model'
        # Try settings.embedding.deployment_name first, fallback to model / 先尝试 settings.embedding.deployment_name，再回退到 model
        self.deployment_name = (
            getattr(settings.embedding, 'deployment_name', None) or 
            settings.embedding.model
        )
        
        # Extract optional dimensions setting / 提取可选 dimensions 配置
        self.dimensions = getattr(settings.embedding, 'dimensions', None)
        
        # API key: explicit parameter > settings.yaml > env var (fallback for backward compatibility) / API key：显式参数 > settings.yaml > 环境变量（用于向后兼容）
        self.api_key = (
            api_key or 
            getattr(settings.embedding, 'api_key', None) or
            os.environ.get("AZURE_OPENAI_API_KEY") or
            os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Configure 'api_key' in settings.yaml, "
                "set AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure endpoint: explicit parameter > settings.yaml > env var (fallback) / Azure 端点：显式参数 > settings.yaml > 环境变量（回退）
        self.azure_endpoint = (
            azure_endpoint or
            getattr(settings.embedding, 'azure_endpoint', None) or
            os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not provided. Configure 'azure_endpoint' in settings.yaml, "
                "set AZURE_OPENAI_ENDPOINT environment variable, or pass azure_endpoint parameter."
            )
        
        # API version: explicit > settings > default / API 版本：显式参数 > settings > 默认值
        self.api_version = (
            api_version or
            getattr(settings.embedding, 'api_version', None) or
            self.DEFAULT_API_VERSION
        )
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Azure OpenAI API. / 使用 Azure OpenAI API 为一批文本生成嵌入。
        
        Args: / 参数：
            texts: List of text strings to embed. Must not be empty. / 要嵌入的文本字符串列表，不能为空。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Override parameters (dimensions, etc.). / 覆盖参数（dimensions 等）。
        
        Returns: / 返回：
            List of embedding vectors, where each vector is a list of floats. / 嵌入向量列表，其中每个向量都是浮点数列表。
            The length of the outer list matches len(texts). / 外层列表长度与 len(texts) 一致。
        
        Raises: / 异常：
            ValueError: If texts list is empty or contains invalid entries. / 如果 texts 列表为空或包含无效条目。
            AzureEmbeddingError: If API call fails. / 如果 API 调用失败。
        """
        # Validate input / 校验输入
        self.validate_texts(texts)
        
        # Import Azure OpenAI client (lazy import to avoid dependency at module level) / 导入 Azure OpenAI 客户端（延迟导入以避免模块级依赖）
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e
        
        # Initialize Azure OpenAI client / 初始化 Azure OpenAI 客户端
        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        
        # Prepare API call parameters / 准备 API 调用参数
        # Azure uses 'model' parameter but expects deployment name / Azure 使用 'model' 参数，但期望传入 deployment 名称
        api_params = {
            "input": texts,
            "model": self.deployment_name,
        }
        
        # Add dimensions if specified (only for text-embedding-3-* models) / 如果指定则添加 dimensions（仅适用于 text-embedding-3-* 模型）
        # text-embedding-ada-002 does NOT support dimensions parameter / text-embedding-ada-002 不支持 dimensions 参数
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and "text-embedding-3" in self.deployment_name.lower():
            api_params["dimensions"] = dimensions
        
        # Call Azure OpenAI API / 调用 Azure OpenAI API
        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise AzureEmbeddingError(
                f"Azure OpenAI Embeddings API call failed: {e}"
            ) from e
        
        # Extract embeddings from response / 从响应中提取嵌入
        # Response format is identical to OpenAI / 响应格式与 OpenAI 相同
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise AzureEmbeddingError(
                f"Failed to parse Azure OpenAI Embeddings API response: {e}"
            ) from e
        
        # Verify output matches input length / 校验输出与输入长度一致
        if len(embeddings) != len(texts):
            raise AzureEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        
        return embeddings
    
    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension for the configured deployment. / 获取已配置 deployment 的嵌入维度。
        
        Returns: / 返回：
            The embedding dimension, or None if not deterministic. / 嵌入维度；如果无法确定则返回 None。
        
        Note: / 说明：
            For text-embedding-3-* deployments with custom dimensions, returns / 对带自定义 dimensions 的 text-embedding-3-* deployment，返回
            the configured dimension. For other deployments, returns their default. / 已配置维度。对其他 deployment，返回其默认值。
        """
        # If dimensions explicitly configured, return it / 如果显式配置了 dimensions，则返回它
        if self.dimensions is not None:
            return self.dimensions
        
        # Common Azure deployment defaults / 常见 Azure deployment 默认值
        # Note: deployment names are user-defined, but often follow these patterns / 注意：deployment 名称由用户定义，但通常遵循这些模式
        deployment_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        # Try exact match first / 先尝试精确匹配
        if self.deployment_name in deployment_dimensions:
            return deployment_dimensions[self.deployment_name]
        
        # Check for partial matches (e.g., "my-embedding-3-large-prod" contains "embedding-3-large") / 检查部分匹配（例如 "my-embedding-3-large-prod" 包含 "embedding-3-large"）
        # Try longer patterns first to avoid false matches / 先尝试更长模式以避免误匹配
        for model_key in sorted(deployment_dimensions.keys(), key=len, reverse=True):
            if model_key in self.deployment_name:
                return deployment_dimensions[model_key]
        
        # Cannot determine dimension / 无法确定维度
        return None
