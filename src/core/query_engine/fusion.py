"""Reciprocal Rank Fusion (RRF) for combining multiple retrieval results. / 用于组合多个检索结果的倒数排名融合（RRF）。

This module implements the RRF fusion algorithm that combines ranking lists from / 本模块实现 RRF 融合算法，用于将来自
Dense and Sparse retrievers into a unified ranking. RRF is a simple yet effective / 稠密和稀疏检索器的排名列表合并为统一排名。RRF 是一种简单而有效的
rank aggregation method that doesn't require score normalization. / 排名聚合方法，不需要进行分数归一化。

Reference: / 参考：
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR '09.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class RRFFusion:
    """Reciprocal Rank Fusion (RRF) for combining multiple ranking lists. / 用于组合多个排名列表的倒数排名融合（RRF）。
    
    RRF combines rankings from multiple sources using the formula: / RRF 使用如下公式组合多个来源的排名：
        RRF_score(d) = Σ 1 / (k + rank(d))
    
    where: / 其中：
        - d is a document (chunk) / d 是一个文档（分块）
        - k is a smoothing constant (typically 60) / k 是平滑常数（通常为 60）
        - rank(d) is the 1-based rank of document d in a ranking list / rank(d) 是文档 d 在排名列表中的从 1 开始的名次
    
    Key Properties: / 关键特性：
    - Deterministic: Same inputs always produce same output ordering / 确定性：相同输入总是产生相同输出顺序
    - Score-agnostic: Uses only rank positions, not raw scores / 与分数无关：只使用排名位置，不使用原始分数
    - No normalization needed: Works with heterogeneous score scales / 无需归一化：可处理不同尺度的分数
    - Handles missing documents: Documents in only one list still contribute / 处理缺失文档：只出现在一个列表中的文档仍会贡献分数
    
    Design Principles Applied: / 应用的设计原则：
    - Config-Driven: k parameter configurable (default: 60) / 配置驱动：k 参数可配置（默认值：60）
    - Type-Safe: Returns standardized RetrievalResult objects / 类型安全：返回标准化的 RetrievalResult 对象
    - Deterministic: Stable sorting with tie-breaking on chunk_id / 确定性：稳定排序，并使用 chunk_id 打破平局
    - Observable: Logging for debugging fusion process / 可观测：记录日志以调试融合过程
    
    Attributes: / 属性：
        k: Smoothing constant for RRF formula (default: 60). / k：RRF 公式的平滑常数（默认值：60）。
           Higher k gives more weight to lower-ranked documents. / k 越大，排名较低的文档权重越高。
    
    Example: / 示例：
        >>> fusion = RRFFusion(k=60)
        >>> dense_results = [
        ...     RetrievalResult(chunk_id="a", score=0.9, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="b", score=0.8, text="...", metadata={}),
        ... ]
        >>> sparse_results = [
        ...     RetrievalResult(chunk_id="b", score=5.2, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="c", score=4.1, text="...", metadata={}),
        ... ]
        >>> fused = fusion.fuse([dense_results, sparse_results], top_k=5)
    """
    
    # Default smoothing constant as recommended in the original RRF paper / 原始 RRF 论文推荐的默认平滑常数
    DEFAULT_K = 60
    
    def __init__(self, k: int = DEFAULT_K) -> None:
        """Initialize RRF fusion with configurable smoothing constant. / 使用可配置的平滑常数初始化 RRF 融合器。
        
        Args: / 参数：
            k: Smoothing constant for RRF formula (default: 60). / k：RRF 公式的平滑常数（默认值：60）。
               - Must be a positive integer / 必须是正整数
               - Higher values reduce the importance of rank differences / 值越大，排名差异的重要性越低
               - Common values: 60 (original paper), 20, 100 / 常见值：60（原论文）、20、100
        
        Raises: / 异常：
            ValueError: If k is not a positive integer. / ValueError：当 k 不是正整数时抛出。
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        self.k = k
        logger.info(f"RRFFusion initialized with k={k}")
    
    def fuse(
        self,
        ranking_lists: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Fuse multiple ranking lists using Reciprocal Rank Fusion. / 使用倒数排名融合合并多个排名列表。
        
        Args: / 参数：
            ranking_lists: List of ranking lists, each containing RetrievalResult / ranking_lists：排名列表的列表，每个列表包含
                           objects sorted by relevance (descending). / 按相关性降序排列的 RetrievalResult 对象。
                           Typically [dense_results, sparse_results]. / 通常为 [dense_results, sparse_results]。
            top_k: Maximum number of results to return. If None, returns all. / top_k：最大返回结果数量。为 None 时返回全部。
            trace: Optional TraceContext for observability (reserved for Stage F). / trace：用于可观测性的可选 TraceContext（为阶段 F 预留）。
        
        Returns: / 返回：
            List of RetrievalResult objects, sorted by fused RRF score (descending). / RetrievalResult 对象列表，按融合后的 RRF 分数降序排序。
            The score field contains the RRF score, not the original retrieval score. / score 字段包含 RRF 分数，而不是原始检索分数。
            Text and metadata are preserved from the first occurrence of each chunk. / text 和 metadata 保留每个分块首次出现时的数据。
        
        Raises: / 异常：
            ValueError: If ranking_lists is empty. / ValueError：当 ranking_lists 为空时抛出。
        
        Note: / 说明：
            - Documents appearing in multiple lists get contributions from all / 出现在多个列表中的文档会获得所有列表的贡献
            - Documents appearing in only one list still receive RRF score / 只出现在一个列表中的文档仍会获得 RRF 分数
            - Tie-breaking: When RRF scores are equal, sort by chunk_id for stability / 平局处理：RRF 分数相等时按 chunk_id 排序以保持稳定
        
        Example: / 示例：
            >>> fusion = RRFFusion(k=60)
            >>> fused = fusion.fuse([dense_results, sparse_results], top_k=10)
            >>> for r in fused:
            ...     print(f"[RRF={r.score:.4f}] {r.chunk_id}")
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")
        
        # Filter out empty lists / 过滤空列表
        non_empty_lists = [lst for lst in ranking_lists if lst]
        
        if not non_empty_lists:
            logger.debug("All ranking lists are empty, returning empty result")
            return []
        
        logger.debug(
            f"Fusing {len(non_empty_lists)} ranking lists with "
            f"sizes {[len(lst) for lst in non_empty_lists]}"
        )
        
        # Step 1: Calculate RRF scores for each unique chunk / 第 1 步：为每个唯一分块计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}  # Preserve text/metadata / 保留文本和元数据
        
        for list_idx, ranking_list in enumerate(non_empty_lists):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id
                
                # Calculate RRF contribution: 1 / (k + rank) / 计算 RRF 贡献：1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)
                
                # Accumulate scores / 累积分数
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    # Store first occurrence's data (text, metadata) / 存储首次出现的数据（文本、元数据）
                    chunk_data[chunk_id] = result
                
                rrf_scores[chunk_id] += rrf_contribution
        
        logger.debug(f"Computed RRF scores for {len(rrf_scores)} unique chunks")
        
        # Step 2: Create fused results with RRF scores / 第 2 步：创建包含 RRF 分数的融合结果
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            original = chunk_data[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_score,
                    text=original.text,
                    metadata=original.metadata.copy(),
                )
            )
        
        # Step 3: Sort by RRF score (descending), then by chunk_id for stability / 第 3 步：按 RRF 分数降序排序，再按 chunk_id 保持稳定
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))
        
        # Step 4: Apply top_k limit if specified / 第 4 步：如果指定 top_k，则应用数量限制
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]
        
        logger.debug(
            f"Fusion complete: {len(fused_results)} results "
            f"(top_k={top_k if top_k else 'all'})"
        )
        
        return fused_results
    
    def fuse_with_weights(
        self,
        ranking_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Fuse multiple ranking lists with optional per-list weights. / 使用可选的列表权重融合多个排名列表。
        
        This is an extended version of fuse() that allows weighting different / 这是 fuse() 的扩展版本，允许为不同的
        ranking sources. For example, giving more weight to dense retrieval / 排名来源设置权重。例如，为语义查询赋予稠密检索更高权重，
        for semantic queries, or more weight to sparse retrieval for keyword queries. / 或为关键词查询赋予稀疏检索更高权重。
        
        Args: / 参数：
            ranking_lists: List of ranking lists, each containing RetrievalResult objects. / ranking_lists：排名列表的列表，每个列表包含 RetrievalResult 对象。
            weights: Optional list of weights for each ranking list (default: uniform). / weights：每个排名列表的可选权重列表（默认：均匀权重）。
                     Must have same length as ranking_lists if provided. / 如果提供，长度必须与 ranking_lists 相同。
                     Weights are multiplied with RRF contributions. / 权重会与 RRF 贡献相乘。
            top_k: Maximum number of results to return. If None, returns all. / top_k：最大返回结果数量。为 None 时返回全部。
            trace: Optional TraceContext for observability (reserved for Stage F). / trace：用于可观测性的可选 TraceContext（为阶段 F 预留）。
        
        Returns: / 返回：
            List of RetrievalResult objects, sorted by weighted RRF score (descending). / RetrievalResult 对象列表，按加权 RRF 分数降序排序。
        
        Raises: / 异常：
            ValueError: If ranking_lists is empty or weights length doesn't match. / ValueError：当 ranking_lists 为空或 weights 长度不匹配时抛出。
        
        Example: / 示例：
            >>> fusion = RRFFusion(k=60)
            >>> # Give 1.5x weight to dense results / 给稠密结果 1.5 倍权重
            >>> fused = fusion.fuse_with_weights(
            ...     [dense_results, sparse_results],
            ...     weights=[1.5, 1.0],
            ...     top_k=10
            ... )
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")
        
        # Default to uniform weights / 默认使用均匀权重
        if weights is None:
            weights = [1.0] * len(ranking_lists)
        
        if len(weights) != len(ranking_lists):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"ranking_lists length ({len(ranking_lists)})"
            )
        
        # Validate weights / 校验权重
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"Weight at index {i} must be non-negative, got {w}")
        
        # Filter out empty lists (keep their weights aligned) / 过滤空列表（保持其权重对齐）
        filtered = [
            (lst, w) for lst, w in zip(ranking_lists, weights) if lst
        ]
        
        if not filtered:
            logger.debug("All ranking lists are empty, returning empty result")
            return []
        
        non_empty_lists, filtered_weights = zip(*filtered)
        
        logger.debug(
            f"Fusing {len(non_empty_lists)} ranking lists with "
            f"weights={list(filtered_weights)}"
        )
        
        # Calculate weighted RRF scores / 计算加权 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}
        
        for list_idx, (ranking_list, weight) in enumerate(zip(non_empty_lists, filtered_weights)):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id
                
                # Weighted RRF contribution / 加权 RRF 贡献
                rrf_contribution = weight * (1.0 / (self.k + rank))
                
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = result
                
                rrf_scores[chunk_id] += rrf_contribution
        
        # Create and sort results / 创建并排序结果
        fused_results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score,
                text=chunk_data[chunk_id].text,
                metadata=chunk_data[chunk_id].metadata.copy(),
            )
            for chunk_id, rrf_score in rrf_scores.items()
        ]
        
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))
        
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]
        
        return fused_results


def rrf_score(rank: int, k: int = RRFFusion.DEFAULT_K) -> float:
    """Calculate RRF score contribution for a single rank position. / 计算单个排名位置的 RRF 分数贡献。
    
    This is a utility function for calculating individual RRF contributions. / 这是用于计算单个 RRF 贡献的工具函数。
    
    Args: / 参数：
        rank: 1-based rank position (1 = highest rank) / rank：从 1 开始的排名位置（1 表示最高排名）
        k: Smoothing constant (default: 60) / k：平滑常数（默认值：60）
    
    Returns: / 返回：
        RRF score contribution: 1 / (k + rank) / RRF 分数贡献：1 / (k + rank)
    
    Raises: / 异常：
        ValueError: If rank is not a positive integer or k is not positive. / ValueError：当 rank 不是正整数或 k 不是正数时抛出。
    
    Example: / 示例：
        >>> rrf_score(1, k=60)  # Top-ranked document / 排名第一的文档
        0.01639344262295082
        >>> rrf_score(10, k=60)  # 10th-ranked document / 排名第 10 的文档
        0.014285714285714285
    """
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    
    return 1.0 / (k + rank)
