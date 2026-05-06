"""Cross-Encoder based Reranker implementation. / 基于 Cross-Encoder 的 Reranker 实现。

This module implements reranking using Cross-Encoder models that directly score / 此模块使用直接为
(query, passage) pairs. Supports both local models via sentence-transformers / (query, passage) 对打分的 Cross-Encoder 模型实现重排序。支持通过 sentence-transformers 使用本地模型
and API-based endpoints. / 以及基于 API 的端点。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderRerankError(RuntimeError):
    """Raised when Cross-Encoder reranking fails. / Cross-Encoder 重排序失败时抛出。"""


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder based reranker for scoring query-passage pairs. / 用于为 query-passage 对打分的 Cross-Encoder 重排序器。
    
    This implementation uses Cross-Encoder models (e.g., ms-marco-MiniLM) / 此实现使用 Cross-Encoder 模型（例如 ms-marco-MiniLM），
    that directly encode and score (query, passage) pairs, providing more / 直接编码并打分 (query, passage) 对，相比 bi-encoder 方法提供
    accurate relevance scores than bi-encoder approaches at the cost of / 更准确的相关性分数，但代价是
    higher computational requirements. / 更高的计算需求。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Can be swapped with other reranker implementations via factory. / 可插拔：可通过工厂替换为其他 reranker 实现。
    - Config-Driven: Model name and parameters come from settings.yaml. / 配置驱动：模型名称和参数来自 settings.yaml。
    - Observable: Supports TraceContext for monitoring (Stage F integration). / 可观测：支持 TraceContext 进行监控（Stage F 集成）。
    - Fallback-Aware: Provides timeout/failure signals for upstream fallback. / 感知回退：为上游回退提供超时/失败信号。
    - Deterministic Testing: Supports mock scorer injection for testing. / 确定性测试：支持注入 mock scorer 用于测试。
    """
    
    def __init__(
        self,
        settings: Any,
        model: Optional[Any] = None,
        timeout: float = 10.0,
        **kwargs: Any
    ) -> None:
        """Initialize the Cross-Encoder Reranker. / 初始化 Cross-Encoder Reranker。
        
        Args: / 参数：
            settings: Application settings containing rerank configuration. / 包含 rerank 配置的应用设置。
            model: Optional pre-initialized CrossEncoder model. If None, creates / 可选的预初始化 CrossEncoder 模型。如果为 None，则
                from settings.rerank.model. Used for testing to inject mock models. / 从 settings.rerank.model 创建。用于测试时注入 mock 模型。
            timeout: Maximum time (seconds) to wait for reranking. Default 10s. / 等待重排序的最长时间（秒）。默认 10 秒。
                Used to enable fallback strategies when reranking takes too long. / 当重排序耗时过长时，用于启用回退策略。
            **kwargs: Additional provider-specific parameters. / 额外 provider 特有参数。
        """
        self.settings = settings
        self.timeout = timeout
        self.kwargs = kwargs
        
        # Initialize or inject model / 初始化或注入模型
        if model is not None:
            self.model = model
        else:
            try:
                model_name = self._get_model_name_from_settings(settings)
                self.model = self._load_cross_encoder_model(model_name)
            except Exception as e:
                raise CrossEncoderRerankError(
                    f"Failed to initialize Cross-Encoder model: {e}"
                ) from e
    
    def _get_model_name_from_settings(self, settings: Any) -> str:
        """Extract model name from settings. / 从 settings 中提取模型名称。
        
        Args: / 参数：
            settings: Application settings. / 应用设置。
        
        Returns: / 返回：
            Model name string. / 模型名称字符串。
        
        Raises: / 异常：
            AttributeError: If rerank.model is not configured. / 如果未配置 rerank.model。
        """
        try:
            model_name = settings.rerank.model
            if not model_name or not isinstance(model_name, str):
                raise ValueError("Model name must be a non-empty string")
            return model_name
        except AttributeError as e:
            raise AttributeError(
                "Missing configuration: settings.rerank.model. "
                "Please specify 'rerank.model' in settings.yaml"
            ) from e
    
    def _load_cross_encoder_model(self, model_name: str) -> Any:
        """Load the Cross-Encoder model. / 加载 Cross-Encoder 模型。
        
        Args: / 参数：
            model_name: Name or path of the Cross-Encoder model. / Cross-Encoder 模型名称或路径。
        
        Returns: / 返回：
            Initialized CrossEncoder instance. / 已初始化的 CrossEncoder 实例。
        
        Raises: / 异常：
            ImportError: If sentence-transformers is not installed. / 如果未安装 sentence-transformers。
            RuntimeError: If model loading fails. / 如果模型加载失败。
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e
        
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            model = CrossEncoder(model_name)
            logger.info(f"Cross-Encoder model loaded successfully: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Cross-Encoder model '{model_name}': {e}"
            ) from e
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using Cross-Encoder scoring. / 使用 Cross-Encoder 打分重排候选。
        
        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            candidates: List of candidate records to rerank. Each must contain / 要重排的候选记录列表。每个候选都必须包含
                either 'text' or 'content' field for scoring. / 'text' 或 'content' 字段用于打分。
            trace: Optional TraceContext for observability (Stage F integration). / 用于可观测性的可选 TraceContext（Stage F 集成）。
            **kwargs: Additional parameters (top_k to limit output, etc.). / 额外参数（top_k 用于限制输出等）。
        
        Returns: / 返回：
            Reranked list of candidates ordered by relevance score (descending). / 按相关性分数降序排列的重排候选列表。
            Each candidate includes a 'rerank_score' field with the model's score. / 每个候选都包含带模型分数的 'rerank_score' 字段。
        
        Raises: / 异常：
            ValueError: If query or candidates are invalid. / 如果 query 或 candidates 无效。
            CrossEncoderRerankError: If scoring fails or times out. / 如果打分失败或超时。
        """
        # Validate inputs / 校验输入
        self.validate_query(query)
        self.validate_candidates(candidates)
        
        # Extract top_k parameter / 提取 top_k 参数
        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
        try:
            # Prepare (query, passage) pairs for scoring / 准备用于打分的 (query, passage) 对
            pairs = self._prepare_pairs(query, candidates)
            
            # Score pairs using the model / 使用模型为文本对打分
            scores = self._score_pairs(pairs, trace=trace)
            
            # Attach scores to candidates and sort / 将分数附加到候选并排序
            reranked = self._attach_scores_and_sort(candidates, scores, top_k)
            
            if trace:
                self._log_trace(trace, query, len(candidates), len(reranked))
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-Encoder reranking failed: {e}", exc_info=True)
            # Signal failure for upstream fallback logic / 为上游回退逻辑发送失败信号
            raise CrossEncoderRerankError(
                f"Cross-Encoder reranking failed: {e}"
            ) from e
    
    def _prepare_pairs(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """Prepare (query, passage) pairs for scoring. / 准备用于打分的 (query, passage) 对。
        
        Args: / 参数：
            query: The user query. / 用户查询。
            candidates: List of candidate records. / 候选记录列表。
        
        Returns: / 返回：
            List of (query, passage_text) tuples. / (query, passage_text) 元组列表。
        """
        pairs = []
        for candidate in candidates:
            # Extract text from candidate (support both 'text' and 'content' keys) / 从候选中提取文本（同时支持 'text' 和 'content' 键）
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            pairs.append((query, text))
        return pairs
    
    def _score_pairs(
        self,
        pairs: List[tuple[str, str]],
        trace: Optional[Any] = None
    ) -> List[float]:
        """Score (query, passage) pairs using the Cross-Encoder model. / 使用 Cross-Encoder 模型为 (query, passage) 对打分。
        
        Args: / 参数：
            pairs: List of (query, passage) tuples. / (query, passage) 元组列表。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
        
        Returns: / 返回：
            List of relevance scores (one per pair). / 相关性分数列表（每个文本对一个）。
        
        Raises: / 异常：
            CrossEncoderRerankError: If scoring fails or times out. / 如果打分失败或超时。
        """
        try:
            # Use model.predict() to score all pairs in batch / 使用 model.predict() 批量为所有文本对打分
            scores = self.model.predict(pairs)
            
            # Convert numpy array to list if needed / 需要时将 numpy 数组转换为列表
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            return scores
            
        except Exception as e:
            raise CrossEncoderRerankError(
                f"Failed to score pairs with Cross-Encoder: {e}"
            ) from e
    
    def _attach_scores_and_sort(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Attach scores to candidates and sort by relevance. / 将分数附加到候选并按相关性排序。
        
        Args: / 参数：
            candidates: Original candidate list. / 原始候选列表。
            scores: Relevance scores from the model. / 模型给出的相关性分数。
            top_k: Number of top candidates to return. / 要返回的 top 候选数量。
        
        Returns: / 返回：
            Sorted list of top_k candidates with 'rerank_score' field added. / 已添加 'rerank_score' 字段的 top_k 候选排序列表。
        """
        # Attach scores to candidates / 将分数附加到候选
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            # Create a copy to avoid modifying original / 创建副本以避免修改原始对象
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)
        
        # Sort by score (descending) and take top_k / 按分数降序排序并取 top_k
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        return sorted_candidates[:top_k]
    
    def _log_trace(
        self,
        trace: Any,
        query: str,
        input_count: int,
        output_count: int
    ) -> None:
        """Log reranking operation to trace context. / 将重排序操作记录到追踪上下文。
        
        Args: / 参数：
            trace: TraceContext instance. / TraceContext 实例。
            query: The query string. / 查询字符串。
            input_count: Number of input candidates. / 输入候选数量。
            output_count: Number of output candidates. / 输出候选数量。
        """
        # Placeholder for Stage F integration / Stage F 集成占位
        # Future: trace.log_rerank_step(...) / 未来：trace.log_rerank_step(...)
        logger.debug(
            f"Cross-Encoder rerank: query='{query[:50]}...', "
            f"input={input_count}, output={output_count}"
        )
