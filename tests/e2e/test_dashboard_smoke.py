"""E2E smoke tests for the Streamlit Dashboard pages. / Streamlit Dashboard 页面的 E2E 冒烟测试。

Uses Streamlit's ``AppTest`` framework to render each page's ``render()`` / 使用 Streamlit 的 ``AppTest`` 框架以 headless 模式
function in headless mode and verify that: / 渲染每个页面的 ``render()`` 函数，并验证：

1. No Python exception is raised during render. / 渲染期间不会抛出 Python 异常。
2. Each page produces at least one expected UI element (header / info / metric). / 每个页面至少产生一个预期 UI 元素（header / info / metric）。

These tests do **not** require live data – they should pass on a fresh / 这些测试**不**需要实时数据，在向量存储为空的
checkout where the vector store is empty. / 全新检出环境中也应通过。

Usage:: / 用法：

    pytest tests/e2e/test_dashboard_smoke.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────── / ── 辅助方法 ───────────────────────────────────


def _mock_settings() -> MagicMock:
    """Return a minimal mock Settings that satisfies all dashboard pages. / 返回满足所有 dashboard 页面需求的最小 mock Settings。"""
    s = MagicMock()
    s.llm.provider = "azure"
    s.llm.model = "gpt-4o"
    s.llm.temperature = 0.0
    s.llm.max_tokens = 4096

    s.embedding.provider = "azure"
    s.embedding.model = "text-embedding-ada-002"
    s.embedding.dimensions = 1536

    s.vector_store.provider = "chroma"
    s.vector_store.collection_name = "default"
    s.vector_store.persist_directory = "./data/db/chroma"

    s.retrieval.dense_top_k = 20
    s.retrieval.sparse_top_k = 20
    s.retrieval.fusion_top_k = 10

    s.rerank.enabled = False
    s.rerank.provider = "none"
    s.rerank.model = ""
    s.rerank.top_k = 5

    s.vision_llm.enabled = False
    s.vision_llm.provider = "azure"
    s.vision_llm.model = "gpt-4o"
    s.vision_llm.max_image_size = 2048

    s.observability.log_level = "INFO"
    s.observability.trace_enabled = True
    s.observability.trace_file = "./logs/traces.jsonl"
    s.observability.structured_logging = True

    s.ingestion.chunk_size = 1000
    s.ingestion.chunk_overlap = 200
    s.ingestion.splitter = "recursive"
    s.ingestion.batch_size = 100
    return s


def _collect_text(at: Any) -> str:
    """Collect all rendered text from an AppTest run for assertion. / 收集 AppTest 运行中渲染的所有文本用于断言。"""
    parts: List[str] = []
    for attr in ("markdown", "header", "subheader", "info", "error", "title", "text", "success", "warning"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


# ── Tests ───────────────────────────────────────────────────────────── / ── 测试 ───────────────────────────────────────


class TestDashboardSmoke:
    """Smoke tests: each page renders without uncaught exceptions. / 冒烟测试：每个页面渲染时没有未捕获异常。"""

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # 1. Overview page / 1. Overview 页面
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_overview_page_renders(self) -> None:
        """Overview page loads and shows system overview header. / Overview 页面加载并显示系统概览标题。"""
        from streamlit.testing.v1 import AppTest

        def page_script():
            from src.observability.dashboard.pages.overview import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.services.config_service.load_settings",
            return_value=_mock_settings(),
        ):
            at.run()

        assert not at.exception, (
            f"Overview page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "overview" in text.lower() or "system" in text.lower()

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # 2. Data Browser page / 2. Data Browser 页面
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_data_browser_page_renders(self) -> None:
        """Data Browser page loads (may show 'no documents' info). / Data Browser 页面可加载（可能显示 'no documents' 信息）。"""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_documents.return_value = []

        def page_script():
            from src.observability.dashboard.pages.data_browser import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.data_browser.DataService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Data Browser page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "data" in text.lower() or "browser" in text.lower() or "document" in text.lower()

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # 3. Ingestion Manager page / 3. Ingestion Manager 页面
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_ingestion_manager_page_renders(self) -> None:
        """Ingestion Manager page loads without errors. / Ingestion Manager 页面无错误加载。"""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_documents.return_value = []

        def page_script():
            from src.observability.dashboard.pages.ingestion_manager import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.ingestion_manager.DataService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Ingestion Manager page raised an exception: {at.exception}"
        )

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # 4. Ingestion Traces page / 4. Ingestion Traces 页面
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_ingestion_traces_page_renders(self) -> None:
        """Ingestion Traces page loads (empty trace list is OK). / Ingestion Traces 页面可加载（空 trace 列表也可以）。"""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_traces.return_value = []

        def page_script():
            from src.observability.dashboard.pages.ingestion_traces import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.ingestion_traces.TraceService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Ingestion Traces page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "trace" in text.lower() or "ingestion" in text.lower()

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # 5. Query Traces page / 5. Query Traces 页面
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_query_traces_page_renders(self) -> None:
        """Query Traces page loads (empty trace list is OK). / Query Traces 页面可加载（空 trace 列表也可以）。"""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_traces.return_value = []

        def page_script():
            from src.observability.dashboard.pages.query_traces import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.query_traces.TraceService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Query Traces page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "query" in text.lower() or "trace" in text.lower()

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # 6. Evaluation Panel page / 6. Evaluation Panel 页面
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_evaluation_panel_page_renders(self) -> None:
        """Evaluation Panel page loads without errors. / Evaluation Panel 页面无错误加载。"""
        from streamlit.testing.v1 import AppTest

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        at.run()

        assert not at.exception, (
            f"Evaluation Panel page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "evaluation" in text.lower() or "panel" in text.lower()
