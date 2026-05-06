"""Data Browser page – browse ingested documents, chunks, and images. / Data Browser 页面 - 浏览已摄入文档、分块和图片。

Layout: / 布局：
1. Collection selector (sidebar) / 集合选择器（侧边栏）
2. Document list with chunk counts / 带分块数量的文档列表
3. Expandable document detail → chunk cards with text + metadata / 可展开文档详情 -> 带文本和元数据的分块卡片
4. Image preview gallery / 图片预览画廊
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """Render the Data Browser page. / 渲染 Data Browser 页面。"""
    st.header("🔍 Data Browser")

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    # ── Collection selector ──────────────────────────────────────── / ── 集合选择器 ───────────────────────────────────
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        "Collection",
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    # ── Danger zone: clear all data ──────────────────────────────── / ── 危险区：清空所有数据 ─────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.warning(
            "This will **permanently delete** all data: "
            "ChromaDB collections, BM25 indexes, images, ingestion history, and trace logs."
        )
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button("🗑️ Clear All Data", type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("Are you sure? This action cannot be undone!")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ Yes, delete everything", key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"Cleared with {len(result['errors'])} error(s): "
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"All data cleared! "
                            f"{result['collections_deleted']} collection(s) deleted."
                        )
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # ── Document list ────────────────────────────────────────────── / ── 文档列表 ─────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents found in this collection.** "
            "Use the Ingestion Manager page to upload and ingest files, "
            "or select a different collection from the dropdown above."
        )
        return

    st.subheader(f"📄 Documents ({len(docs)})")

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = f"📑 {source_name}  —  {doc['chunk_count']} chunks · {doc['image_count']} images"
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── Document metadata ────────────────────────────────── / ── 文档元数据 ───────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunks", doc["chunk_count"])
            col_b.metric("Images", doc["image_count"])
            col_c.metric("Collection", doc.get("collection", "—"))
            st.caption(
                f"**Source:** {doc['source_path']}  ·  "
                f"**Hash:** `{doc['source_hash'][:16]}…`  ·  "
                f"**Processed:** {doc.get('processed_at', '—')}"
            )

            st.divider()

            # ── Chunk cards ──────────────────────────────────────── / ── 分块卡片 ─────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(f"### 📦 Chunks ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # Title from metadata or first line / 标题来自元数据或第一行
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            f"**Chunk {cidx + 1}** · `{chunk_id[-16:]}` · "
                            f"{len(text)} chars"
                        )
                        # Show the actual chunk text (scrollable) / 显示实际分块文本（可滚动）
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            "Content",
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # Expandable metadata / 可展开元数据
                        with st.expander("📋 Metadata", expanded=False):
                            st.json(meta)
            else:
                st.caption("No chunks found in vector store for this document.")

            # ── Image preview ────────────────────────────────────── / ── 图片预览 ─────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(f"### 🖼️ Images ({len(images)})")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(f"{img['image_id']} (file missing)")
