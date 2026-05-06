"""Modular RAG Dashboard – multi-page Streamlit application. / Modular RAG Dashboard - 多页面 Streamlit 应用。

Entry-point: ``streamlit run src/observability/dashboard/app.py`` / 入口点：``streamlit run src/observability/dashboard/app.py``

Pages are registered via ``st.navigation()`` and rendered by their / 页面通过 ``st.navigation()`` 注册，并由 ``pages/`` 下
respective modules under ``pages/``.  Pages not yet implemented show / 各自的模块渲染。尚未实现的页面会显示
a placeholder message. / 占位消息。
"""

from __future__ import annotations

import streamlit as st


# ── Page definitions ───────────────────────────────────────────────── / ── 页面定义 ─────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


# ── Navigation ─────────────────────────────────────────────────────── / ── 导航 ───────────────────────────────────────

pages = [
    st.Page(_page_overview, title="Overview", icon="📊", default=True),
    st.Page(_page_data_browser, title="Data Browser", icon="🔍"),
    st.Page(_page_ingestion_manager, title="Ingestion Manager", icon="📥"),
    st.Page(_page_ingestion_traces, title="Ingestion Traces", icon="🔬"),
    st.Page(_page_query_traces, title="Query Traces", icon="🔎"),
    st.Page(_page_evaluation_panel, title="Evaluation Panel", icon="📏"),
]


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    # When run directly via `streamlit run app.py` / 当通过 `streamlit run app.py` 直接运行时
    main()
