"""Abstract base class for text splitters. / 文本拆分器的抽象基类。

This module defines the pluggable interface for text splitter providers, / 此模块定义文本拆分器 provider 的可插拔接口，
enabling seamless switching between different splitting strategies / 支持在不同拆分策略之间无缝切换，
(Recursive, Semantic, FixedLength, etc.) through configuration-driven instantiation. / （Recursive、Semantic、FixedLength 等）并通过配置驱动的实例化完成选择。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseSplitter(ABC):
    """Abstract base class for text splitters. / 文本拆分器的抽象基类。
    
    All splitter implementations must inherit from this class and implement / 所有 splitter 实现都必须继承此类并实现
    the split_text() method. This ensures a consistent interface across / split_text() 方法。这确保不同
    different strategies. / 策略之间接口一致。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Subclasses can be swapped without changing upstream code. / 可插拔：无需修改上游代码即可替换子类。
    - Observable: Accepts optional TraceContext for observability integration. / 可观测：接收可选 TraceContext 以集成可观测能力。
    - Config-Driven: Instances are created via factory based on settings. / 配置驱动：基于 settings 通过工厂创建实例。
    """
    
    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Split input text into a list of chunks. / 将输入文本拆分为块列表。
        
        Args: / 参数：
            text: Input text to split. Must be a non-empty string. / 要拆分的输入文本，必须是非空字符串。
            trace: Optional TraceContext for observability (reserved for Stage F). / 用于可观测性的可选 TraceContext（为 Stage F 预留）。
            **kwargs: Strategy-specific parameters (chunk_size, overlap, etc.). / 策略特有参数（chunk_size、overlap 等）。
        
        Returns: / 返回：
            A list of text chunks. Order must preserve the original text sequence. / 文本块列表。顺序必须保持原始文本顺序。
        
        Raises: / 异常：
            ValueError: If input text is invalid. / 如果输入文本无效。
            RuntimeError: If the splitter fails unexpectedly. / 如果拆分器意外失败。
        """
        pass
    
    def validate_text(self, text: str) -> None:
        """Validate input text. / 校验输入文本。
        
        Args: / 参数：
            text: Input text to validate. / 要校验的输入文本。
        
        Raises: / 异常：
            ValueError: If text is not a non-empty string. / 如果文本不是非空字符串。
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")
    
    def validate_chunks(self, chunks: List[str]) -> None:
        """Validate output chunks. / 校验输出块。
        
        Args: / 参数：
            chunks: List of chunk strings to validate. / 要校验的块字符串列表。
        
        Raises: / 异常：
            ValueError: If chunks are empty or contain invalid entries. / 如果 chunks 为空或包含无效条目。
        """
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list of strings")
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise ValueError(
                    f"Chunk at index {i} is not a string (type: {type(chunk).__name__})"
                )
            if not chunk.strip():
                raise ValueError(
                    f"Chunk at index {i} is empty or whitespace-only"
                )
