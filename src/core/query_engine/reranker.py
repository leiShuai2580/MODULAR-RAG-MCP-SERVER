"""Core layer Reranker orchestrating libs.reranker backends with fallback support. / 核心层重排器，用于编排 libs.reranker 后端并支持回退。

This module implements the CoreReranker class that: / 本模块实现 CoreReranker 类，其功能包括：
1. Integrates with libs.reranker (LLM, CrossEncoder, None) via RerankerFactory / 1. 通过 RerankerFactory 集成 libs.reranker（LLM、CrossEncoder、None）
2. Provides graceful fallback when backend fails or times out / 2. 在后端失败或超时时提供优雅回退
3. Converts RetrievalResult to/from reranker input/output format / 3. 在 RetrievalResult 与重排器输入/输出格式之间转换
4. Supports TraceContext for observability / 4. 支持用于可观测性的 TraceContext

Design Principles: / 设计原则：
- Pluggable: Uses RerankerFactory to instantiate configured backend / 可插拔：使用 RerankerFactory 实例化配置的后端
- Config-Driven: Reads rerank settings from settings.yaml / 配置驱动：从 settings.yaml 读取重排配置
- Graceful Fallback: Returns original order on backend failure / 优雅回退：后端失败时返回原始顺序
- Observable: TraceContext integration for debugging / 可观测：集成 TraceContext 便于调试
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """Raised when reranking fails. / 重排失败时抛出。"""


@dataclass
class RerankConfig:
    """Configuration for CoreReranker. / CoreReranker 的配置。
    
    Attributes: / 属性：
        enabled: Whether reranking is enabled / enabled：是否启用重排
        top_k: Number of results to return after reranking / top_k：重排后返回的结果数量
        timeout: Timeout for reranker backend (seconds) / timeout：重排后端超时时间（秒）
        fallback_on_error: Whether to return original order on error / fallback_on_error：出错时是否返回原始顺序
    """
    enabled: bool = True
    top_k: int = 5
    timeout: float = 30.0
    fallback_on_error: bool = True


@dataclass
class RerankResult:
    """Result of a rerank operation. / 重排操作的结果。
    
    Attributes: / 属性：
        results: Reranked list of RetrievalResults / results：重排后的 RetrievalResult 列表
        used_fallback: Whether fallback was used due to backend failure / used_fallback：是否因后端失败使用了回退
        fallback_reason: Reason for fallback (if applicable) / fallback_reason：回退原因（如适用）
        reranker_type: Type of reranker used ('llm', 'cross_encoder', 'none') / reranker_type：使用的重排器类型（'llm'、'cross_encoder'、'none'）
        original_order: Original results before reranking (for debugging) / original_order：重排前的原始结果（用于调试）
    """
    results: List[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    reranker_type: str = "none"
    original_order: Optional[List[RetrievalResult]] = None


class CoreReranker:
    """Core layer Reranker with fallback support. / 支持回退的核心层重排器。
    
    This class wraps libs.reranker implementations and provides: / 该类封装 libs.reranker 实现，并提供：
    1. Type conversion between RetrievalResult and reranker dict format / 1. RetrievalResult 与重排器字典格式之间的类型转换
    2. Graceful fallback when backend fails / 2. 后端失败时的优雅回退
    3. Configuration-driven backend selection / 3. 配置驱动的后端选择
    4. TraceContext integration / 4. TraceContext 集成
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Backend via RerankerFactory / 可插拔：通过 RerankerFactory 选择后端
    - Config-Driven: All parameters from settings / 配置驱动：所有参数来自 settings
    - Fallback: Returns original order on failure / 回退：失败时返回原始顺序
    - Observable: TraceContext support / 可观测：支持 TraceContext
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> reranker = CoreReranker(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.8, text="...", metadata={})]
        >>> reranked = reranker.rerank("query", results)
        >>> print(reranked.results)
    """
    
    def __init__(
        self,
        settings: Settings,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ) -> None:
        """Initialize CoreReranker. / 初始化 CoreReranker。
        
        Args: / 参数：
            settings: Application settings containing rerank configuration. / settings：包含重排配置的应用配置。
            reranker: Optional reranker backend. If None, creates via RerankerFactory. / reranker：可选重排后端。如果为 None，则通过 RerankerFactory 创建。
            config: Optional RerankConfig. If None, extracts from settings. / config：可选 RerankConfig。如果为 None，则从 settings 提取。
        """
        self.settings = settings
        
        # Extract config from settings or use provided / 从 settings 提取配置或使用已提供配置
        if config is not None:
            self.config = config
        else:
            self.config = self._extract_config(settings)
        
        # Initialize reranker backend / 初始化重排器后端
        if reranker is not None:
            self._reranker = reranker
        elif not self.config.enabled:
            self._reranker = NoneReranker(settings=settings)
        else:
            try:
                self._reranker = RerankerFactory.create(settings)
            except Exception as e:
                logger.warning(f"Failed to create reranker, using NoneReranker: {e}")
                self._reranker = NoneReranker(settings=settings)
        
        # Determine reranker type for result reporting / 确定用于结果报告的重排器类型
        self._reranker_type = self._get_reranker_type()
    
    def _extract_config(self, settings: Settings) -> RerankConfig:
        """Extract RerankConfig from settings. / 从 settings 中提取 RerankConfig。
        
        Args: / 参数：
            settings: Application settings. / settings：应用配置。
            
        Returns: / 返回：
            RerankConfig with values from settings. / 包含 settings 中取值的 RerankConfig。
        """
        try:
            rerank_settings = settings.rerank
            return RerankConfig(
                enabled=bool(rerank_settings.enabled) if rerank_settings else False,
                top_k=int(rerank_settings.top_k) if rerank_settings and hasattr(rerank_settings, 'top_k') else 5,
                timeout=float(getattr(rerank_settings, 'timeout', 30.0)) if rerank_settings else 30.0,
                fallback_on_error=True,
            )
        except AttributeError:
            logger.warning("Missing rerank configuration, using defaults (disabled)")
            return RerankConfig(enabled=False)
    
    def _get_reranker_type(self) -> str:
        """Get the type name of the current reranker backend. / 获取当前重排器后端的类型名称。
        
        Returns: / 返回：
            String identifier for the reranker type. / 重排器类型的字符串标识。
        """
        class_name = self._reranker.__class__.__name__
        if "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()
    
    def _results_to_candidates(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Convert RetrievalResults to reranker candidate format. / 将 RetrievalResult 转换为重排器候选格式。
        
        Args: / 参数：
            results: List of RetrievalResult objects. / results：RetrievalResult 对象列表。
            
        Returns: / 返回：
            List of dicts suitable for reranker input. / 适合作为重排器输入的字典列表。
        """
        candidates = []
        for result in results:
            candidates.append({
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata.copy(),
            })
        return candidates
    
    def _candidates_to_results(
        self,
        candidates: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Convert reranked candidates back to RetrievalResults. / 将重排后的候选项转换回 RetrievalResult。
        
        Args: / 参数：
            candidates: Reranked candidates from reranker. / candidates：来自重排器的重排候选项。
            original_results: Original results for reference. / original_results：用于参考的原始结果。
            
        Returns: / 返回：
            List of RetrievalResult in reranked order. / 按重排顺序排列的 RetrievalResult 列表。
        """
        # Build lookup from original results / 从原始结果构建查找表
        id_to_original = {r.chunk_id: r for r in original_results}
        
        results = []
        for candidate in candidates:
            chunk_id = candidate["id"]
            
            # Get original result or build new one / 获取原始结果或构建新结果
            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                # Create new result with updated score / 使用更新后的分数创建新结果
                rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
                results.append(RetrievalResult(
                    chunk_id=original.chunk_id,
                    score=rerank_score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "original_score": original.score,
                        "rerank_score": rerank_score,
                        "reranked": True,
                    },
                ))
            else:
                # Candidate not in original - build from candidate data / 候选项不在原始结果中，则根据候选数据构建
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=candidate.get("rerank_score", candidate.get("score", 0.0)),
                    text=candidate.get("text", ""),
                    metadata=candidate.get("metadata", {}),
                ))
        
        return results
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank retrieval results using configured backend. / 使用配置的后端重排检索结果。
        
        Args: / 参数：
            query: The user query string. / query：用户查询字符串。
            results: List of RetrievalResult objects to rerank. / results：待重排的 RetrievalResult 对象列表。
            top_k: Number of results to return. If None, uses config.top_k. / top_k：返回结果数量。如果为 None，则使用 config.top_k。
            trace: Optional TraceContext for observability. / trace：用于可观测性的可选 TraceContext。
            **kwargs: Additional parameters passed to reranker backend. / **kwargs：传递给重排器后端的额外参数。
            
        Returns: / 返回：
            RerankResult containing reranked results and metadata. / 包含重排结果和元数据的 RerankResult。
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k
        
        # Early return for empty or single results / 对空结果或单个结果提前返回
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        # If reranking disabled, return top_k results in original order / 如果重排被禁用，则按原始顺序返回 top_k 个结果
        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                used_fallback=False,
                reranker_type="none",
                original_order=results[:],
            )
        
        # Convert to reranker input format / 转换为重排器输入格式
        candidates = self._results_to_candidates(results)
        
        # Attempt reranking / 尝试重排
        try:
            logger.debug(f"Reranking {len(candidates)} candidates with {self._reranker_type}")
            _t0 = time.monotonic()
            reranked_candidates = self._reranker.rerank(
                query=query,
                candidates=candidates,
                trace=trace,
                **kwargs,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            # Convert back to RetrievalResult / 转换回 RetrievalResult
            reranked_results = self._candidates_to_results(reranked_candidates, results)
            
            # Apply top_k limit / 应用 top_k 限制
            final_results = reranked_results[:effective_top_k]
            
            logger.info(f"Reranking complete: {len(final_results)} results returned")
            
            if trace is not None:
                trace.record_stage("rerank", {
                    "method": self._reranker_type,
                    "provider": self._reranker_type,
                    "input_count": len(candidates),
                    "output_count": len(final_results),
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": round(r.score, 4),
                            "text": r.text or "",
                            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                        }
                        for r in final_results
                    ],
                }, elapsed_ms=_elapsed)
            
            return RerankResult(
                results=final_results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=results[:],
            )
            
        except Exception as e:
            logger.warning(f"Reranking failed, using fallback: {e}")
            
            if self.config.fallback_on_error:
                # Return original order as fallback / 将原始顺序作为回退结果返回
                fallback_results = []
                for result in results[:effective_top_k]:
                    fallback_results.append(RetrievalResult(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        text=result.text,
                        metadata={
                            **result.metadata,
                            "reranked": False,
                            "rerank_fallback": True,
                        },
                    ))
                
                return RerankResult(
                    results=fallback_results,
                    used_fallback=True,
                    fallback_reason=str(e),
                    reranker_type=self._reranker_type,
                    original_order=results[:],
                )
            else:
                raise RerankError(f"Reranking failed and fallback disabled: {e}") from e
    
    @property
    def reranker_type(self) -> str:
        """Get the type of the current reranker backend. / 获取当前重排器后端的类型。"""
        return self._reranker_type
    
    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled. / 检查重排是否启用。"""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """Factory function to create a CoreReranker instance. / 创建 CoreReranker 实例的工厂函数。
    
    Args: / 参数：
        settings: Application settings. / settings：应用配置。
        reranker: Optional reranker backend override. / reranker：可选的重排器后端覆盖项。
        
    Returns: / 返回：
        Configured CoreReranker instance. / 配置完成的 CoreReranker 实例。
    """
    return CoreReranker(settings=settings, reranker=reranker)
