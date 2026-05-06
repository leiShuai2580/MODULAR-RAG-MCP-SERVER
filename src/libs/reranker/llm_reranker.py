"""LLM-based Reranker implementation. / 基于 LLM 的 Reranker 实现。

This module implements reranking using Large Language Models to evaluate / 此模块使用大语言模型评估
the relevance of candidate passages to a given query. It reads prompts from / 候选段落与给定查询的相关性来实现重排序。它从
config/prompts/rerank.txt and structures LLM outputs for downstream processing. / config/prompts/rerank.txt 读取提示词，并将 LLM 输出结构化供下游处理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import BaseReranker


class LLMRerankError(RuntimeError):
    """Raised when LLM reranking fails. / LLM 重排序失败时抛出。"""


class LLMReranker(BaseReranker):
    """LLM-based reranker using structured prompts. / 使用结构化提示词的 LLM 重排序器。
    
    This implementation leverages an LLM to score and rerank candidate passages / 此实现使用 LLM 为候选段落打分并重排，
    based on their relevance to a query. It reads the reranking prompt from / 基于它们与查询的相关性。它从
    a configurable file and expects structured JSON output from the LLM. / 可配置文件读取重排序提示词，并期望 LLM 输出结构化 JSON。
    
    Design Principles Applied: / 应用的设计原则：
    - Pluggable: Can be swapped with other reranker implementations via factory. / 可插拔：可通过工厂替换为其他 reranker 实现。
    - Config-Driven: Prompt file path and LLM settings come from configuration. / 配置驱动：提示词文件路径和 LLM 设置来自配置。
    - Observable: Supports TraceContext for monitoring (Stage F integration). / 可观测：支持 TraceContext 进行监控（Stage F 集成）。
    - Fallback-Aware: Returns signal on failure for upstream fallback logic. / 感知回退：失败时为上游回退逻辑返回信号。
    - Structured Output: Validates LLM output against expected schema. / 结构化输出：根据预期 schema 校验 LLM 输出。
    """
    
    def __init__(
        self,
        settings: Any,
        prompt_path: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the LLM Reranker. / 初始化 LLM Reranker。
        
        Args: / 参数：
            settings: Application settings containing LLM and rerank configuration. / 包含 LLM 和 rerank 配置的应用设置。
            prompt_path: Optional path to rerank prompt file. If None, uses default / 可选的 rerank 提示词文件路径。如果为 None，则使用默认
                'config/prompts/rerank.txt'. Used for testing to inject custom prompts. / 'config/prompts/rerank.txt'。用于测试时注入自定义提示词。
            llm: Optional LLM instance. If None, creates via LLMFactory from settings. / 可选 LLM 实例。如果为 None，则通过 LLMFactory 从 settings 创建。
                Used for testing to inject mock LLMs. / 用于测试时注入 mock LLM。
            **kwargs: Additional provider-specific parameters. / 额外 provider 特有参数。
        """
        self.settings = settings
        self.prompt_path = prompt_path or str(resolve_path("config/prompts/rerank.txt"))
        self.llm = llm or LLMFactory.create(settings)
        self.kwargs = kwargs
        
        # Load prompt template / 加载提示词模板
        try:
            self.prompt_template = self._load_prompt_template(self.prompt_path)
        except Exception as e:
            raise LLMRerankError(f"Failed to load rerank prompt from {self.prompt_path}: {e}") from e
    
    def _load_prompt_template(self, path: str) -> str:
        """Load the rerank prompt template from file. / 从文件加载 rerank 提示词模板。
        
        Args: / 参数：
            path: Path to the prompt template file. / 提示词模板文件路径。
        
        Returns: / 返回：
            The prompt template as a string. / 字符串形式的提示词模板。
        
        Raises: / 异常：
            FileNotFoundError: If prompt file doesn't exist. / 如果提示词文件不存在。
            IOError: If file can't be read. / 如果文件无法读取。
        """
        prompt_file = Path(path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Rerank prompt file not found: {path}")
        
        return prompt_file.read_text(encoding="utf-8")
    
    def _build_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Build the reranking prompt with query and candidates. / 使用查询和候选构建重排序提示词。
        
        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            candidates: List of candidate records to rerank. / 要重排的候选记录列表。
        
        Returns: / 返回：
            Formatted prompt string ready for LLM. / 可供 LLM 使用的格式化提示词字符串。
        """
        # Format candidates for the prompt / 为提示词格式化候选
        candidates_text = []
        for i, candidate in enumerate(candidates):
            passage_id = candidate.get("id", f"passage_{i}")
            text = candidate.get("text", candidate.get("content", ""))
            candidates_text.append(f"Passage ID: {passage_id}\nText: {text}\n")
        
        candidates_str = "\n".join(candidates_text)
        
        # Construct full prompt / 构造完整提示词
        full_prompt = f"{self.prompt_template}\n\nQuery: {query}\n\nPassages:\n{candidates_str}\n\nOutput your response as a JSON array of objects, one per passage."
        
        return full_prompt
    
    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse and validate LLM response. / 解析并校验 LLM 响应。
        
        Args: / 参数：
            response_text: Raw text response from LLM. / LLM 的原始文本响应。
        
        Returns: / 返回：
            Parsed ranking records with passage_id and score. / 带 passage_id 和 score 的已解析排序记录。
        
        Raises: / 异常：
            LLMRerankError: If response doesn't match expected schema. / 如果响应不符合预期 schema。
        """
        # Try to extract JSON from response (LLM might wrap in markdown) / 尝试从响应中提取 JSON（LLM 可能包在 markdown 中）
        text = response_text.strip()
        
        # Remove markdown code blocks if present / 如果存在 markdown 代码块则移除
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON / 解析 JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMRerankError(
                f"LLM response is not valid JSON: {e}\nResponse: {response_text[:200]}"
            ) from e
        
        # Validate structure / 校验结构
        if not isinstance(parsed, list):
            raise LLMRerankError(
                f"Expected JSON array, got {type(parsed).__name__}\nResponse: {response_text[:200]}"
            )
        
        # Validate each item / 校验每个条目
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise LLMRerankError(
                    f"Item {i} is not a dict (type: {type(item).__name__})"
                )
            if "passage_id" not in item:
                raise LLMRerankError(f"Item {i} missing required field 'passage_id'")
            if "score" not in item:
                raise LLMRerankError(f"Item {i} missing required field 'score'")
            
            # Validate score is numeric / 校验 score 是数字
            score = item["score"]
            if not isinstance(score, (int, float)):
                raise LLMRerankError(
                    f"Item {i} score must be numeric, got {type(score).__name__}: {score}"
                )
        
        return parsed
    
    def _map_results_to_candidates(
        self,
        parsed_results: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map LLM ranking results back to original candidate objects. / 将 LLM 排序结果映射回原始候选对象。
        
        Args: / 参数：
            parsed_results: Parsed LLM output with passage_id and score. / 带 passage_id 和 score 的已解析 LLM 输出。
            candidates: Original candidate list. / 原始候选列表。
        
        Returns: / 返回：
            Reranked list of candidate objects, sorted by score descending. / 按分数降序排序的重排候选对象列表。
        """
        # Create mapping from id to candidate / 创建 id 到候选的映射
        id_to_candidate = {}
        for i, candidate in enumerate(candidates):
            candidate_id = candidate.get("id", f"passage_{i}")
            id_to_candidate[candidate_id] = candidate
        
        # Build reranked list / 构建重排列表
        reranked = []
        for result in parsed_results:
            passage_id = result["passage_id"]
            score = result["score"]
            
            if passage_id in id_to_candidate:
                candidate = id_to_candidate[passage_id].copy()
                # Update score with LLM rerank score / 使用 LLM 重排分数更新 score
                candidate["rerank_score"] = float(score)
                reranked.append(candidate)
        
        # Sort by rerank score descending / 按 rerank 分数降序排序
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using LLM-based relevance scoring. / 使用基于 LLM 的相关性打分重排候选。
        
        Args: / 参数：
            query: The user query string. / 用户查询字符串。
            candidates: List of candidate records to rerank. / 要重排的候选记录列表。
            trace: Optional TraceContext for observability. / 用于可观测性的可选 TraceContext。
            **kwargs: Additional parameters (timeout, temperature, etc.). / 额外参数（timeout、temperature 等）。
        
        Returns: / 返回：
            Reranked list of candidates sorted by LLM-assigned relevance score. / 按 LLM 分配的相关性分数排序后的候选列表。
            On success, each candidate will have 'rerank_score' in metadata. / 成功时，每个候选都会在元数据中带有 'rerank_score'。
        
        Raises: / 异常：
            ValueError: If query or candidates are invalid. / 如果 query 或 candidates 无效。
            LLMRerankError: If LLM call fails or response is malformed. / 如果 LLM 调用失败或响应格式错误。
        """
        # Validate inputs / 校验输入
        self.validate_query(query)
        self.validate_candidates(candidates)
        
        # If only one candidate, no need to rerank / 如果只有一个候选，则无需重排
        if len(candidates) == 1:
            return candidates
        
        # Build prompt / 构建提示词
        try:
            prompt = self._build_rerank_prompt(query, candidates)
        except Exception as e:
            raise LLMRerankError(f"Failed to build rerank prompt: {e}") from e
        
        # Call LLM / 调用 LLM
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace, **kwargs)
            response_text = response.content
        except Exception as e:
            # Return fallback signal - let upstream decide how to handle / 返回回退信号，由上游决定如何处理
            raise LLMRerankError(f"LLM call failed during reranking: {e}") from e
        
        # Parse response / 解析响应
        try:
            parsed_results = self._parse_llm_response(response_text)
        except LLMRerankError:
            raise
        except Exception as e:
            raise LLMRerankError(f"Failed to parse LLM rerank response: {e}") from e
        
        # Map back to candidates and sort / 映射回候选并排序
        try:
            reranked = self._map_results_to_candidates(parsed_results, candidates)
        except Exception as e:
            raise LLMRerankError(f"Failed to map LLM results to candidates: {e}") from e
        
        return reranked
