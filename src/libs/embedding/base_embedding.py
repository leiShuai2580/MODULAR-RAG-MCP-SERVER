"""Abstract base class for Embedding providers. / Embedding provider 的抽象基类。

This module defines the pluggable interface for Embedding service providers, / 此模块定义 Embedding 服务 provider 的可插拔接口，
enabling seamless switching between different backends (OpenAI, Local/BGE, etc.) / 支持在不同后端（OpenAI、Local/BGE 等）之间无缝切换，
through configuration-driven instantiation. / 并通过配置驱动的实例化完成选择。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseEmbedding(ABC):
    """Abstract base class for Embedding providers. / Embedding provider 的抽象基类。
    
    All Embedding implementations must inherit from this class and implement / 所有 Embedding 实现都必须继承此类并实现
    the embed() method. This ensures consistent interface across different / embed() 方法。这确保不同
    providers (OpenAI, Local/BGE, Ollama, etc.). / provider（OpenAI、Local/BGE、Ollama 等）之间接口一致。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Subclasses can be swapped without changing upstream code. / 可插拔：无需修改上游代码即可替换子类
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选 TraceContext 以集成可观测能力
    - Config-Driven: Instances are created via factory based on settings. / 配置驱动：基于 settings 通过工厂创建实例
    - Batch-First: Designed for batch processing to maximize efficiency. / 批处理优先：为批处理设计以最大化效率
    """
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts. / 为一批文本生成嵌入。
        
        Args: / 参数：
            texts: List of text strings to embed. Must not be empty. / 要嵌入的文本字符串列表，不能为空。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Provider-specific parameters (batch_size, dimensions, etc.). / provider 特有参数（batch_size、dimensions 等）。
        
        Returns: / 返回：
            List of embedding vectors, where each vector is a list of floats. / 嵌入向量列表，其中每个向量都是浮点数列表。
            The length of the outer list matches len(texts). / 外层列表长度与 len(texts) 一致。
            The length of each inner list (vector dimension) is provider-dependent. / 每个内层列表长度（向量维度）取决于 provider。
        
        Raises: / 异常：
            ValueError: If texts list is empty or contains invalid entries. / 如果 texts 列表为空或包含无效条目。
            RuntimeError: If the embedding provider call fails. / 如果嵌入 provider 调用失败。
        
        Example: / 示例：
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 vectors
            >>> len(embeddings[0])  # dimension (e.g., 1536 for OpenAI)
        """
        pass
    
    def validate_texts(self, texts: List[str]) -> None:
        """Validate input text list. / 校验输入文本列表。
        
        Args: / 参数：
            texts: List of texts to validate. / 要校验的文本列表。
        
        Raises: / 异常：
            ValueError: If texts list is empty or contains invalid entries. / 如果 texts 列表为空或包含无效条目。
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} is not a string (type: {type(text).__name__})"
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only. "
                    "Embedding providers typically reject empty strings."
                )
    
    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider. / 获取此 provider 生成的嵌入维度。
        
        Returns: / 返回：
            The vector dimension (e.g., 1536 for OpenAI text-embedding-3-small). / 向量维度（例如 OpenAI text-embedding-3-small 为 1536）。
        
        Raises: / 异常：
            NotImplementedError: If the subclass doesn't override this method. / 如果子类没有重写此方法。
        
        Note: / 说明：
            Subclasses should override this method to return their specific dimension. / 子类应重写此方法以返回自己的维度。
            This is useful for validation and storage configuration. / 这对校验和存储配置有用。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_dimension() method"
        )
