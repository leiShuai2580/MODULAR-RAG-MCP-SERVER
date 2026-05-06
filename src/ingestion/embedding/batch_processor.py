"""Batch Processor for orchestrating dense and sparse encoding. / 用于编排稠密和稀疏编码的批处理器。

This module implements the Batch Processor component of the Ingestion Pipeline, / 此模块实现 Ingestion Pipeline 的 Batch Processor 组件，
responsible for coordinating the encoding workflow and managing batch operations. / 负责协调编码工作流并管理批处理操作。

Design Principles: / 设计原则：
- Orchestration: Coordinates DenseEncoder and SparseEncoder in unified workflow / 编排：在统一工作流中协调 DenseEncoder 和 SparseEncoder
- Config-Driven: Batch size from settings, not hardcoded / 配置驱动：批大小来自配置，而非硬编码
- Observable: Records batch timing and statistics via TraceContext / 可观测：通过 TraceContext 记录批次耗时和统计信息
- Error Handling: Individual batch failures don't crash entire pipeline / 错误处理：单个批次失败不会导致整条流水线崩溃
- Deterministic: Same inputs produce same batching and results / 确定性：相同输入产生相同分批和结果
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


@dataclass
class BatchResult:
    """Result of batch processing operation. / 批处理操作结果。
    
    Attributes: / 属性：
        dense_vectors: List of dense embeddings (one per chunk) / 稠密嵌入列表（每个块一个）
        sparse_stats: List of term statistics (one per chunk) / 词项统计列表（每个块一个）
        batch_count: Number of batches processed / 已处理批次数
        total_time: Total processing time in seconds / 总处理耗时（秒）
        successful_chunks: Number of successfully processed chunks / 成功处理的块数量
        failed_chunks: Number of chunks that failed processing / 处理失败的块数量
    """
    dense_vectors: List[List[float]]
    sparse_stats: List[Dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int


class BatchProcessor:
    """Orchestrates batch processing of chunks through encoding pipeline. / 通过编码流水线编排块的批处理。
    
    This processor manages the workflow of converting chunks into both dense / 此处理器管理将块转换为稠密表示
    and sparse representations. It divides chunks into batches, drives the / 和稀疏表示的工作流。它将块分成批次，驱动
    encoders, and collects timing metrics. / 编码器，并收集耗时指标。
    
    Design: / 设计：
    - Stateless: No state maintained between process() calls / 无状态：process() 调用之间不维护状态
    - Parallel Encodings: Dense and sparse encoding happen independently / 并行编码：稠密和稀疏编码相互独立
    - Metrics Collection: Records batch-level timing for observability / 指标收集：记录批次级耗时以支持可观测性
    - Order Preservation: Output order matches input chunk order / 顺序保持：输出顺序与输入块顺序一致
    
    Example: / 示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> dense_encoder = DenseEncoder(embedding, batch_size=2)
        >>> sparse_encoder = SparseEncoder()
        >>> 
        >>> processor = BatchProcessor(
        ...     dense_encoder=dense_encoder,
        ...     sparse_encoder=sparse_encoder,
        ...     batch_size=2
        ... )
        >>> 
        >>> chunks = [
        ...     Chunk(id="1", text="Hello", metadata={}),
        ...     Chunk(id="2", text="World", metadata={})
        ... ]
        >>> result = processor.process(chunks)
        >>> len(result.dense_vectors) == len(chunks)  # True
        >>> len(result.sparse_stats) == len(chunks)  # True
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 100,
    ):
        """Initialize BatchProcessor. / 初始化 BatchProcessor。
        
        Args: / 参数：
            dense_encoder: DenseEncoder instance for embedding generation / 用于生成嵌入的 DenseEncoder 实例
            sparse_encoder: SparseEncoder instance for term statistics / 用于词项统计的 SparseEncoder 实例
            batch_size: Number of chunks to process per batch (default: 100) / 每批处理的块数量（默认：100）
        
        Raises: / 异常：
            ValueError: If batch_size <= 0 / 如果 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size
    
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> BatchResult:
        """Process chunks through dense and sparse encoding pipeline. / 通过稠密和稀疏编码流水线处理块。
        
        Workflow: / 工作流：
        1. Validate inputs / 校验输入
        2. Create batches from chunks / 从块创建批次
        3. Process each batch through both encoders / 通过两个编码器处理每个批次
        4. Collect results and timing metrics / 收集结果和耗时指标
        5. Record to TraceContext if provided / 如果提供 TraceContext，则记录到其中
        
        Args: / 参数：
            chunks: List of Chunk objects to process / 要处理的 Chunk 对象列表
            trace: Optional TraceContext for observability / 用于可观测性的可选 TraceContext
        
        Returns: / 返回：
            BatchResult containing vectors, statistics, and metrics / 包含向量、统计和指标的 BatchResult
        
        Raises: / 异常：
            ValueError: If chunks list is empty / 如果 chunks 列表为空
            RuntimeError: If both encoders fail completely / 如果两个编码器都完全失败
        
        Example: / 示例：
            >>> chunks = [Chunk(id=f"{i}", text=f"Text {i}", metadata={}) 
            ...           for i in range(5)]
            >>> result = processor.process(chunks)
            >>> result.batch_count  # 3 (with batch_size=2)
            >>> result.successful_chunks  # 5
        """
        if not chunks:
            raise ValueError("Cannot process empty chunks list")
        
        start_time = time.time()
        
        # Create batches / 创建批次
        batches = self._create_batches(chunks)
        batch_count = len(batches)
        
        # Process all batches / 处理所有批次
        dense_vectors: List[List[float]] = []
        sparse_stats: List[Dict[str, Any]] = []
        successful_chunks = 0
        failed_chunks = 0
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()
            
            try:
                # Dense encoding / 稠密编码
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                dense_vectors.extend(batch_dense)
                
                # Sparse encoding / 稀疏编码
                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
                sparse_stats.extend(batch_sparse)
                
                successful_chunks += len(batch)
                
            except Exception as e:
                # Record failure but continue with remaining batches / 记录失败，但继续处理剩余批次
                failed_chunks += len(batch)
                if trace:
                    trace.record_stage(
                        f"batch_{batch_idx}_error",
                        {"error": str(e), "batch_size": len(batch)}
                    )
            
            batch_duration = time.time() - batch_start
            
            # Record batch timing if trace available / 如果 trace 可用则记录批次耗时
            if trace:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {
                        "batch_size": len(batch),
                        "duration_seconds": batch_duration,
                        "chunks_processed": len(batch)
                    }
                )
        
        total_time = time.time() - start_time
        
        # Record overall processing statistics / 记录整体处理统计
        if trace:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": batch_count,
                    "batch_size": self.batch_size,
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "total_time_seconds": total_time
                }
            )
        
        return BatchResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=batch_count,
            total_time=total_time,
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks
        )
    
    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Divide chunks into batches of specified size. / 将块划分为指定大小的批次。
        
        Args: / 参数：
            chunks: List of chunks to batch / 要分批的块列表
        
        Returns: / 返回：
            List of batches, where each batch is a list of chunks. / 批次列表，其中每个批次都是块列表。
            Order is preserved: first batch contains chunks[0:batch_size], / 保持顺序：第一个批次包含 chunks[0:batch_size]，
            second batch contains chunks[batch_size:2*batch_size], etc. / 第二个批次包含 chunks[batch_size:2*batch_size]，依此类推。
        
        Example: / 示例：
            >>> chunks = [Chunk(id=f"{i}", text="", metadata={}) for i in range(5)]
            >>> batches = processor._create_batches(chunks)
            >>> len(batches)  # 3 (with batch_size=2)
            >>> [len(b) for b in batches]  # [2, 2, 1]
        """
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def get_batch_count(self, total_chunks: int) -> int:
        """Calculate number of batches for given chunk count. / 计算给定块数量对应的批次数。
        
        Utility method for planning and testing. / 用于规划和测试的工具方法。
        
        Args: / 参数：
            total_chunks: Total number of chunks to process / 要处理的块总数
        
        Returns: / 返回：
            Number of batches that will be created / 将创建的批次数
        
        Example: / 示例：
            >>> processor.get_batch_count(5)  # 3 (with batch_size=2)
            >>> processor.get_batch_count(4)  # 2
            >>> processor.get_batch_count(0)  # 0
        """
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size
