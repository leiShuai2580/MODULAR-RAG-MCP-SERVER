"""Dashboard configuration reading service. / 仪表盘配置读取服务。

Wraps :class:`Settings` to provide formatted component information / 封装 :class:`Settings`，为 Overview 页面
for the Overview page. / 提供格式化后的组件信息。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import Settings, load_settings


@dataclass
class ComponentInfo:
    """Summary of a single configured component. / 单个已配置组件的摘要。"""

    name: str
    provider: str
    model: str
    extra: Dict[str, Any]


class ConfigService:
    """Read-only service that exposes application configuration. / 暴露应用配置的只读服务。

    Args: / 参数：
        settings_path: Path to ``settings.yaml``. / ``settings.yaml`` 路径。
    """

    def __init__(self, settings_path: Optional[str] = None) -> None:
        self._settings_path = settings_path
        self._settings: Optional[Settings] = None

    # ── lazy load ──────────────────────────────────────────────────── / ── 延迟加载 ─────────────────────────────────────

    def _load(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self._settings_path)
        return self._settings

    def reload(self) -> None:
        """Force reload of settings from disk. / 强制从磁盘重新加载设置。"""
        self._settings = None

    @property
    def settings(self) -> Settings:
        return self._load()

    # ── component cards ────────────────────────────────────────────── / ── 组件卡片 ─────────────────────────────────────

    def get_component_cards(self) -> List[ComponentInfo]:
        """Return a list of component summaries for the Overview page. / 返回 Overview 页面的组件摘要列表。"""
        s = self._load()
        cards: List[ComponentInfo] = []

        # LLM / LLM
        cards.append(ComponentInfo(
            name="LLM",
            provider=s.llm.provider,
            model=s.llm.model,
            extra={"temperature": s.llm.temperature, "max_tokens": s.llm.max_tokens},
        ))

        # Embedding / Embedding
        cards.append(ComponentInfo(
            name="Embedding",
            provider=s.embedding.provider,
            model=s.embedding.model,
            extra={"dimensions": s.embedding.dimensions},
        ))

        # VectorStore / 向量存储
        cards.append(ComponentInfo(
            name="Vector Store",
            provider=s.vector_store.provider,
            model=s.vector_store.collection_name,
            extra={"persist_directory": s.vector_store.persist_directory},
        ))

        # Retrieval / 检索
        cards.append(ComponentInfo(
            name="Retrieval",
            provider="hybrid",
            model="dense + sparse + RRF",
            extra={
                "dense_top_k": s.retrieval.dense_top_k,
                "sparse_top_k": s.retrieval.sparse_top_k,
                "fusion_top_k": s.retrieval.fusion_top_k,
            },
        ))

        # Rerank / 重排
        cards.append(ComponentInfo(
            name="Reranker",
            provider=s.rerank.provider if s.rerank.enabled else "disabled",
            model=s.rerank.model if s.rerank.enabled else "-",
            extra={"enabled": s.rerank.enabled, "top_k": s.rerank.top_k},
        ))

        # Vision LLM / 视觉 LLM
        if s.vision_llm and s.vision_llm.enabled:
            cards.append(ComponentInfo(
                name="Vision LLM",
                provider=s.vision_llm.provider,
                model=s.vision_llm.model,
                extra={"max_image_size": s.vision_llm.max_image_size},
            ))

        # Ingestion / 摄入
        if s.ingestion:
            cards.append(ComponentInfo(
                name="Ingestion",
                provider=s.ingestion.splitter,
                model="-",
                extra={
                    "chunk_size": s.ingestion.chunk_size,
                    "chunk_overlap": s.ingestion.chunk_overlap,
                    "batch_size": s.ingestion.batch_size,
                },
            ))

        return cards
