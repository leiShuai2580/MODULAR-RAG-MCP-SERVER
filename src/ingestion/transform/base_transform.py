"""Base class for chunk transform operations. / 块转换操作的基类。"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Abstract base class for chunk transformation operations. / 块转换操作的抽象基类。
    
    Transform operations process chunks to enhance their quality, add metadata, / 转换操作会处理块，以提升质量、添加元数据，
    or prepare them for downstream processing (embedding, indexing). / 或为下游处理（嵌入、索引）做准备。
    
    Design Principles: / 设计原则：
        - Single Responsibility: Each transform does ONE type of enhancement / 单一职责：每个转换只做一种增强
        - Atomic Operations: Failure in one chunk doesn't affect others / 原子操作：一个块失败不会影响其他块
        - Observable: Records processing info in TraceContext / 可观测：在 TraceContext 中记录处理信息
        - Graceful Degradation: Returns original chunk on unrecoverable errors / 优雅降级：遇到不可恢复错误时返回原始块
    """
    
    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform a list of chunks. / 转换块列表。
        
        Args: / 参数：
            chunks: List of chunks to transform / 要转换的块列表
            trace: Optional trace context for observability / 用于可观测性的可选追踪上下文
            
        Returns: / 返回：
            List of transformed chunks (same length as input) / 转换后的块列表（长度与输入相同）
            
        Raises: / 异常：
            ValueError: If input validation fails / 如果输入校验失败
        """
        pass
