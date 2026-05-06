"""Ollama Embedding implementation for local embedding models. / 面向本地嵌入模型的 Ollama Embedding 实现。

This module provides the Ollama Embedding implementation that works with / 此模块提供与本地运行的
locally running Ollama instances. Ollama enables running embedding models like / Ollama 实例配合使用的 Ollama Embedding 实现。Ollama 支持在本地硬件上运行
nomic-embed-text, mxbai-embed-large, etc. on local hardware. / nomic-embed-text、mxbai-embed-large 等嵌入模型。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama Embeddings API call fails. / Ollama Embeddings API 调用失败时抛出。
    
    This exception provides clear error messages without exposing / 此异常提供清晰错误信息，同时不暴露
    sensitive configuration details like internal URLs. / 内部 URL 等敏感配置细节。
    """


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider implementation for local embedding. / 面向本地嵌入的 Ollama Embedding provider 实现。
    
    This class implements the BaseEmbedding interface for Ollama's embeddings API, / 此类为 Ollama 的 embeddings API 实现 BaseEmbedding 接口，
    enabling local embedding generation without cloud dependencies. / 支持无需云依赖的本地嵌入生成。
    
    Attributes: / 属性：
        base_url: The base URL for the Ollama server (default: http://localhost:11434). / Ollama 服务器基础 URL（默认：http://localhost:11434）。
        model: The model identifier to use (e.g., 'nomic-embed-text', 'mxbai-embed-large'). / 要使用的模型标识符（例如 'nomic-embed-text'、'mxbai-embed-large'）。
        timeout: Request timeout in seconds. / 请求超时时间（秒）。
        dimension: The dimensionality of embeddings produced by this model. / 此模型生成的嵌入维度。
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OllamaEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # Longer timeout for local inference / 本地推理使用更长超时
    DEFAULT_DIMENSION = 768  # Common dimension for local embedding models / 本地嵌入模型的常见维度
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama Embedding provider. / 初始化 Ollama Embedding provider。
        
        Args: / 参数：
            settings: Application settings containing Embedding configuration. / 包含 Embedding 配置的应用设置。
            base_url: Optional base URL override (falls back to env var OLLAMA_BASE_URL). / 可选基础 URL 覆盖值（回退到环境变量 OLLAMA_BASE_URL）。
            timeout: Optional timeout override for requests. / 可选请求超时覆盖值。
            **kwargs: Additional configuration overrides. / 额外配置覆盖项。
        
        Raises: / 异常：
            ValueError: If required configuration is missing. / 如果缺少必需配置。
        """
        self.model = settings.embedding.model
        
        # Base URL: explicit > env var > default / 基础 URL：显式参数 > 环境变量 > 默认值
        self.base_url = (
            base_url 
            or os.environ.get("OLLAMA_BASE_URL") 
            or self.DEFAULT_BASE_URL
        )
        
        # Timeout: explicit > default / 超时：显式参数 > 默认值
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        
        # Dimension: settings > default / 维度：settings > 默认值
        self.dimension = getattr(settings.embedding, 'dimensions', self.DEFAULT_DIMENSION)
        
        # Store any additional kwargs for future use / 存储额外 kwargs 以备未来使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Ollama API. / 使用 Ollama API 为一批文本生成嵌入。
        
        Args: / 参数：
            texts: List of text strings to embed. Must not be empty. / 要嵌入的文本字符串列表，不能为空。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Additional parameters (currently unused, reserved for future). / 额外参数（当前未使用，为未来预留）。
        
        Returns: / 返回：
            List of embedding vectors, where each vector is a list of floats. / 嵌入向量列表，其中每个向量都是浮点数列表。
        
        Raises: / 异常：
            ValueError: If texts list is empty or contains invalid entries. / 如果 texts 列表为空或包含无效条目。
            OllamaEmbeddingError: If API call fails. / 如果 API 调用失败。
        
        Example: / 示例：
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 vectors
            >>> len(embeddings[0])  # dimension (e.g., 768)
        """
        # Validate input / 校验输入
        self.validate_texts(texts)
        
        try:
            import httpx
        except ImportError as e:
            raise OllamaEmbeddingError(
                "httpx library is required for Ollama Embedding. "
                "Install with: pip install httpx"
            ) from e
        
        # Prepare API request / 准备 API 请求
        url = f"{self.base_url}/api/embeddings"
        
        embeddings: List[List[float]] = []
        
        # Process each text individually (Ollama API expects single prompt) / 逐条处理文本（Ollama API 期望单个 prompt）
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Extract embedding from response / 从响应中提取嵌入
                    if "embedding" not in result:
                        raise OllamaEmbeddingError(
                            f"Unexpected response format from Ollama API. "
                            f"Expected 'embedding' field but got: {list(result.keys())}"
                        )
                    
                    embeddings.append(result["embedding"])
                    
            except httpx.HTTPStatusError as e:
                # HTTP error (4xx, 5xx) / HTTP 错误（4xx、5xx）
                raise OllamaEmbeddingError(
                    f"Ollama API request failed with status {e.response.status_code}. "
                    f"Ensure Ollama is running and model '{self.model}' is available."
                ) from e
            except httpx.ConnectError as e:
                # Connection error (server not reachable) / 连接错误（服务器不可达）
                raise OllamaEmbeddingError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    f"Ensure Ollama is running (try: ollama serve)"
                ) from e
            except httpx.TimeoutException as e:
                # Request timeout / 请求超时
                raise OllamaEmbeddingError(
                    f"Ollama API request timed out after {self.timeout}s. "
                    f"The model may be loading or the request is too large."
                ) from e
            except httpx.RequestError as e:
                # Other request errors / 其他请求错误
                raise OllamaEmbeddingError(
                    f"Ollama API request failed: {str(e)}"
                ) from e
            except (KeyError, ValueError, TypeError) as e:
                # JSON parsing or data extraction error / JSON 解析或数据提取错误
                raise OllamaEmbeddingError(
                    f"Failed to parse Ollama API response: {str(e)}"
                ) from e
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider. / 获取此 provider 生成的嵌入维度。
        
        Returns: / 返回：
            The vector dimension configured for this instance. / 此实例配置的向量维度。
        
        Note: / 说明：
            The actual dimension may vary by model. Common dimensions: / 实际维度可能因模型而异。常见维度：
            - nomic-embed-text: 768 / nomic-embed-text：768
            - mxbai-embed-large: 1024 / mxbai-embed-large：1024
            Configure via settings.embedding.dimensions or accepts default 768. / 通过 settings.embedding.dimensions 配置，或接受默认值 768。
        """
        return self.dimension
