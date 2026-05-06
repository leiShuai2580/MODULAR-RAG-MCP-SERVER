"""DataService – read-only facade for browsing ingested data. / DataService - 用于浏览已摄入数据的只读门面。

Wraps ``DocumentManager``, ``ChromaStore``, and ``ImageStorage`` to / 封装 ``DocumentManager``、``ChromaStore`` 和 ``ImageStorage``，
provide the data the Data Browser page needs, without coupling the / 提供 Data Browser 页面所需数据，同时避免
UI to storage internals. / UI 与存储内部实现耦合。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataService:
    """Provide read-only access to ingested documents, chunks, and images. / 提供对已摄入文档、分块和图片的只读访问。

    Lazily instantiates the heavy storage objects on first call so that / 首次调用时才延迟实例化较重的存储对象，
    importing the module alone has zero cost. / 因此单纯导入该模块没有额外成本。
    """

    def __init__(self) -> None:
        self._manager: Any = None
        self._chroma: Any = None
        self._images: Any = None
        self._current_collection: str = ""

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # Lazy initialisation / 延迟初始化
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    def _ensure_stores(self, collection: Optional[str] = None) -> None:
        """Create storage objects on first use. / 首次使用时创建存储对象。

        Args: / 参数：
            collection: Optional collection name. The ChromaStore will be / 可选集合名称。如果请求的集合不同于
                        re-created if the requested collection differs from / 当前已加载集合，则会重新创建
                        the currently loaded one. / ChromaStore。
        """
        target_collection = collection or "default"

        # Re-create chroma if collection changed / 如果集合变化，则重新创建 chroma
        if (
            self._manager is not None
            and self._current_collection == target_collection
        ):
            return

        from src.core.settings import load_settings, resolve_path
        from src.ingestion.document_manager import DocumentManager
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import ImageStorage
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings()

        chroma = VectorStoreFactory.create(
            settings, collection_name=target_collection
        )
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{target_collection}")))
        images = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images")),
        )
        integrity = SQLiteIntegrityChecker(
            db_path=str(resolve_path("data/db/ingestion_history.db"))
        )

        self._chroma = chroma
        self._images = images
        self._manager = DocumentManager(chroma, bm25, images, integrity)
        self._current_collection = target_collection

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # Public API / 公共 API
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """Return all available ChromaDB collection names. / 返回所有可用的 ChromaDB 集合名称。"""
        try:
            from src.core.settings import load_settings, resolve_path
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            settings = load_settings()
            persist_dir = str(
                resolve_path(settings.vector_store.persist_directory)
            )
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            return sorted(c.name for c in client.list_collections())
        except Exception as exc:
            logger.warning("Failed to list collections: %s", exc)
            return ["default"]

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return ingested documents as plain dicts (UI-friendly). / 将已摄入文档作为普通字典返回（便于 UI 使用）。

        Each dict has keys: source_path, source_hash, collection, / 每个字典包含以下键：source_path、source_hash、collection、
        chunk_count, image_count, processed_at. / chunk_count、image_count、processed_at。
        """
        self._ensure_stores(collection)
        from dataclasses import asdict

        docs = self._manager.list_documents(collection)
        return [asdict(d) for d in docs]

    def get_document_detail(
        self, doc_id: str, collection: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return document detail as a plain dict, or None. / 将文档详情作为普通字典返回，或返回 None。"""
        self._ensure_stores(collection)
        from dataclasses import asdict

        detail = self._manager.get_document_detail(doc_id)
        if detail is None:
            return None
        return asdict(detail)

    def get_chunks(
        self, source_hash: str, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return chunk records from ChromaDB matching *source_hash*. / 从 ChromaDB 返回匹配 *source_hash* 的分块记录。

        Each dict has keys: id, text, metadata. / 每个字典包含 id、text、metadata 键。
        """
        self._ensure_stores(collection)
        try:
            results = self._chroma.collection.get(
                where={"doc_hash": source_hash},
                include=["documents", "metadatas"],
            )
            chunks: List[Dict[str, Any]] = []
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            for i, cid in enumerate(ids):
                chunks.append(
                    {
                        "id": cid,
                        "text": docs[i] if docs else "",
                        "metadata": metas[i] if metas else {},
                    }
                )
            return chunks
        except Exception as exc:
            logger.warning("Failed to get chunks for %s: %s", source_hash, exc)
            return []

    def get_images(
        self, source_hash: str, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return image records for a document. / 返回某个文档的图片记录。"""
        self._ensure_stores(collection)
        try:
            return self._images.list_images(doc_hash=source_hash)
        except Exception as exc:
            logger.warning("Failed to get images for %s: %s", source_hash, exc)
            return []

    def delete_document(
        self,
        source_path: str,
        collection: Optional[str] = None,
        source_hash: Optional[str] = None,
    ) -> Any:
        """Delete a document via the underlying DocumentManager. / 通过底层 DocumentManager 删除文档。

        Returns a ``DeleteResult`` dataclass. / 返回 ``DeleteResult`` dataclass。
        """
        self._ensure_stores(collection)
        return self._manager.delete_document(
            source_path,
            collection or "default",
            source_hash=source_hash,
        )

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return aggregate stats as a plain dict. / 将聚合统计作为普通字典返回。"""
        self._ensure_stores(collection)
        from dataclasses import asdict

        stats = self._manager.get_collection_stats(collection)
        return asdict(stats)

    def reset_all(self) -> Dict[str, Any]:
        """Delete ALL data: ChromaDB collections, BM25 indexes, images, integrity DB, and trace logs. / 删除所有数据：ChromaDB 集合、BM25 索引、图片、完整性 DB 和 trace 日志。

        Returns a summary dict with counts of what was deleted. / 返回包含删除数量的摘要字典。
        """
        import shutil
        from src.core.settings import load_settings, resolve_path
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        summary: Dict[str, Any] = {
            "collections_deleted": 0,
            "bm25_cleared": False,
            "images_cleared": False,
            "integrity_cleared": False,
            "traces_cleared": False,
            "errors": [],
        }

        settings = load_settings()

        # 1. Delete all ChromaDB collections / 1. 删除所有 ChromaDB 集合
        try:
            persist_dir = str(resolve_path(settings.vector_store.persist_directory))
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            colls = client.list_collections()
            for c in colls:
                client.delete_collection(c.name)
                summary["collections_deleted"] += 1
        except Exception as exc:
            summary["errors"].append(f"ChromaDB: {exc}")

        # 2. Clear BM25 indexes (remove entire bm25 directory) / 2. 清空 BM25 索引（删除整个 bm25 目录）
        try:
            bm25_dir = resolve_path("data/db/bm25")
            if bm25_dir.exists():
                shutil.rmtree(bm25_dir)
                bm25_dir.mkdir(parents=True, exist_ok=True)
            summary["bm25_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"BM25: {exc}")

        # 3. Clear image storage (SQLite DB + image files) / 3. 清空图片存储（SQLite DB + 图片文件）
        try:
            img_db = resolve_path("data/db/image_index.db")
            if img_db.exists():
                img_db.unlink()
            img_dir = resolve_path("data/images")
            if img_dir.exists():
                shutil.rmtree(img_dir)
                img_dir.mkdir(parents=True, exist_ok=True)
            summary["images_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"Images: {exc}")

        # 4. Clear file integrity database / 4. 清空文件完整性数据库
        try:
            integrity_db = resolve_path("data/db/ingestion_history.db")
            if integrity_db.exists():
                integrity_db.unlink()
            summary["integrity_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"Integrity: {exc}")

        # 5. Clear trace logs / 5. 清空 trace 日志
        try:
            traces_file = resolve_path("logs/traces.jsonl")
            if traces_file.exists():
                traces_file.unlink()
            summary["traces_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"Traces: {exc}")

        # Reset internal state so next call re-initializes / 重置内部状态，以便下次调用重新初始化
        self._manager = None
        self._chroma = None
        self._images = None
        self._current_collection = ""

        return summary
