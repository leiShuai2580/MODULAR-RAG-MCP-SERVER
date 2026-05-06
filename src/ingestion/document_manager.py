"""Cross-store document lifecycle management. / 跨存储的文档生命周期管理。

This module provides a single entry-point for listing, inspecting, and / 本模块提供统一入口，用于跨四类存储后端
deleting documents across all four storage backends (ChromaDB, BM25, / （ChromaDB、BM25、
ImageStorage, FileIntegrityChecker). / ImageStorage、FileIntegrityChecker）列出、查看和删除文档。

Design Principles: / 设计原则：
- Coordinated: one call cascades into all relevant stores. / 协同：一次调用级联到所有相关存储。
- Fail-safe: partial failures are reported but do not abort remaining stores. / 故障安全：报告局部失败，但不中止剩余存储的处理。
- Read-only safe: list / stats / detail methods never mutate data. / 只读安全：list / stats / detail 方法永不修改数据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data-classes / 结果数据类
# ---------------------------------------------------------------------------

@dataclass
class DocumentInfo:
    """Summary information about an ingested document. / 已摄取文档的摘要信息。"""

    source_path: str
    source_hash: str
    collection: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0
    processed_at: Optional[str] = None


@dataclass
class DocumentDetail(DocumentInfo):
    """Extended document info including chunk IDs and image IDs. / 包含分块 ID 和图片 ID 的扩展文档信息。"""

    chunk_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    """Outcome of a delete_document operation. / delete_document 操作的结果。"""

    success: bool
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """Aggregate statistics for a collection. / 集合的聚合统计信息。"""

    collection: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


# ---------------------------------------------------------------------------
# DocumentManager / 文档管理器
# ---------------------------------------------------------------------------

class DocumentManager:
    """Coordinate document lifecycle across all storage backends. / 协调所有存储后端中的文档生命周期。

    Args: / 参数：
        chroma_store: ChromaStore instance (vector store). / chroma_store：ChromaStore 实例（向量存储）。
        bm25_indexer: BM25Indexer instance (sparse index). / bm25_indexer：BM25Indexer 实例（稀疏索引）。
        image_storage: ImageStorage instance (image files + SQLite index). / image_storage：ImageStorage 实例（图片文件 + SQLite 索引）。
        file_integrity: SQLiteIntegrityChecker instance (ingestion history). / file_integrity：SQLiteIntegrityChecker 实例（摄取历史）。
    """

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
    ) -> None:
        self.chroma = chroma_store
        self.bm25 = bm25_indexer
        self.images = image_storage
        self.integrity = file_integrity

    # ------------------------------------------------------------------
    # list_documents / 列出文档
    # ------------------------------------------------------------------

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[DocumentInfo]:
        """Return a list of ingested documents. / 返回已摄取文档列表。

        Combines information from the integrity checker (source_path, / 将完整性检查器中的信息（source_path、
        hash, processed_at) with counts from ChromaDB and ImageStorage. / hash、processed_at）与 ChromaDB 和 ImageStorage 中的计数合并。

        Args: / 参数：
            collection: Optional collection filter. / collection：可选集合过滤条件。

        Returns: / 返回：
            List of ``DocumentInfo`` objects. / ``DocumentInfo`` 对象列表。
        """
        records = self.integrity.list_processed(collection)

        docs: List[DocumentInfo] = []
        for rec in records:
            source_hash = rec["file_hash"]
            source_path = rec["file_path"]
            coll = rec.get("collection")

            # Count chunks in Chroma / 统计 Chroma 中的分块数量
            chunk_count = self._count_chunks(source_hash)

            # Count images / 统计图片数量
            image_count = self._count_images(source_hash)

            docs.append(
                DocumentInfo(
                    source_path=source_path,
                    source_hash=source_hash,
                    collection=coll,
                    chunk_count=chunk_count,
                    image_count=image_count,
                    processed_at=rec.get("processed_at"),
                )
            )

        return docs

    # ------------------------------------------------------------------
    # get_document_detail / 获取文档详情
    # ------------------------------------------------------------------

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetail]:
        """Get detailed information about a single document. / 获取单个文档的详细信息。

        *doc_id* is matched against the ``source_hash`` stored in the / *doc_id* 会与完整性检查器中存储的
        integrity checker. / ``source_hash`` 进行匹配。

        Args: / 参数：
            doc_id: The document's source_hash. / doc_id：文档的 source_hash。

        Returns: / 返回：
            ``DocumentDetail`` with chunk/image IDs, or *None* if not found. / 包含分块/图片 ID 的 ``DocumentDetail``；未找到时返回 *None*。
        """
        # Look up integrity record / 查找完整性记录
        all_records = self.integrity.list_processed()
        record = None
        for rec in all_records:
            if rec["file_hash"] == doc_id:
                record = rec
                break

        if record is None:
            return None

        source_hash = record["file_hash"]

        # Collect chunk IDs from Chroma / 从 Chroma 收集分块 ID
        chunk_ids = self._get_chunk_ids(source_hash)

        # Collect image IDs / 收集图片 ID
        image_ids = self._get_image_ids(source_hash)

        return DocumentDetail(
            source_path=record["file_path"],
            source_hash=source_hash,
            collection=record.get("collection"),
            chunk_count=len(chunk_ids),
            image_count=len(image_ids),
            processed_at=record.get("processed_at"),
            chunk_ids=chunk_ids,
            image_ids=image_ids,
        )

    # ------------------------------------------------------------------
    # delete_document / 删除文档
    # ------------------------------------------------------------------

    def delete_document(
        self,
        source_path: str,
        collection: str = "default",
        source_hash: Optional[str] = None,
    ) -> DeleteResult:
        """Delete a document from all storage backends. / 从所有存储后端删除文档。

        Coordinates deletion across ChromaDB, BM25, ImageStorage, and / 协调 ChromaDB、BM25、ImageStorage 和
        FileIntegrity.  Partial failures are captured in / FileIntegrity 中的删除操作。局部失败会记录在
        ``DeleteResult.errors`` but do not prevent remaining stores / ``DeleteResult.errors`` 中，但不会阻止剩余存储
        from being cleaned. / 被清理。

        The document is identified by its *source_hash*.  When the hash / 文档通过其 *source_hash* 标识。当未提供哈希时，
        is not supplied the method tries to compute it from the file; / 该方法会尝试从文件计算；
        if the file no longer exists it falls back to looking up the / 如果文件已不存在，则回退为根据路径
        hash from the integrity records by path. / 从完整性记录中查找哈希。

        Args: / 参数：
            source_path: Original filesystem path of the document. / source_path：文档的原始文件系统路径。
            collection: Collection the document belongs to. / collection：文档所属集合。
            source_hash: Pre-computed SHA-256 hash.  When provided the / source_hash：预先计算的 SHA-256 哈希。提供后，
                method will not attempt to read the source file. / 该方法不会尝试读取源文件。

        Returns: / 返回：
            ``DeleteResult`` summarising what was cleaned. / 汇总清理内容的 ``DeleteResult``。
        """
        result = DeleteResult(success=True)

        # Resolve hash – prefer caller-supplied, then file, then DB lookup / 解析哈希：优先使用调用方提供值，其次文件计算，最后查数据库
        if source_hash is None:
            try:
                source_hash = self.integrity.compute_sha256(source_path)
            except Exception as e:
                source_hash = self._hash_from_path(source_path)
                if source_hash is None:
                    result.success = False
                    result.errors.append(f"Cannot identify document: {e}")
                    return result

        # 1. ChromaDB – delete chunks matching source_hash / 1. ChromaDB：删除匹配 source_hash 的分块
        try:
            count = self.chroma.delete_by_metadata(
                {"doc_hash": source_hash}
            )
            result.chunks_deleted = count
        except Exception as e:
            result.errors.append(f"ChromaDB delete failed: {e}")

        # 2. BM25 – remove postings for this document / 2. BM25：移除该文档的倒排记录
        try:
            result.bm25_removed = self.bm25.remove_document(
                source_hash, collection
            )
        except Exception as e:
            result.errors.append(f"BM25 remove failed: {e}")

        # 3. ImageStorage – delete images by doc_hash / 3. ImageStorage：按 doc_hash 删除图片
        try:
            images = self.images.list_images(doc_hash=source_hash)
            deleted_imgs = 0
            for img in images:
                if self.images.delete_image(img["image_id"]):
                    deleted_imgs += 1
            result.images_deleted = deleted_imgs
        except Exception as e:
            result.errors.append(f"ImageStorage delete failed: {e}")

        # 4. FileIntegrity – remove the ingestion record / 4. FileIntegrity：移除摄取记录
        try:
            result.integrity_removed = self.integrity.remove_record(
                source_hash
            )
        except Exception as e:
            result.errors.append(f"FileIntegrity remove failed: {e}")

        if result.errors:
            result.success = False

        return result

    # ------------------------------------------------------------------
    # get_collection_stats / 获取集合统计
    # ------------------------------------------------------------------

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> CollectionStats:
        """Return aggregate statistics for a collection. / 返回集合的聚合统计信息。

        Args: / 参数：
            collection: Collection name.  When *None*, stats span / collection：集合名称。当为 *None* 时，统计范围为
                all collections. / 所有集合。

        Returns: / 返回：
            ``CollectionStats`` dataclass. / ``CollectionStats`` 数据类。
        """
        docs = self.list_documents(collection)
        chunk_total = sum(d.chunk_count for d in docs)
        image_total = sum(d.image_count for d in docs)

        return CollectionStats(
            collection=collection,
            document_count=len(docs),
            chunk_count=chunk_total,
            image_count=image_total,
        )

    # ------------------------------------------------------------------
    # Private helpers / 私有辅助方法
    # ------------------------------------------------------------------

    def _count_chunks(self, source_hash: str) -> int:
        """Count chunks in Chroma that belong to *source_hash*. / 统计 Chroma 中属于 *source_hash* 的分块数量。"""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return len(results.get("ids", []))
        except Exception:
            return 0

    def _get_chunk_ids(self, source_hash: str) -> List[str]:
        """Return chunk IDs from Chroma matching *source_hash*. / 返回 Chroma 中匹配 *source_hash* 的分块 ID。"""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return results.get("ids", [])
        except Exception:
            return []

    def _count_images(self, source_hash: str) -> int:
        """Count images belonging to *source_hash*. / 统计属于 *source_hash* 的图片数量。"""
        try:
            return len(self.images.list_images(doc_hash=source_hash))
        except Exception:
            return 0

    def _get_image_ids(self, source_hash: str) -> List[str]:
        """Return image IDs belonging to *source_hash*. / 返回属于 *source_hash* 的图片 ID。"""
        try:
            imgs = self.images.list_images(doc_hash=source_hash)
            return [img["image_id"] for img in imgs]
        except Exception:
            return []

    def _hash_from_path(self, source_path: str) -> Optional[str]:
        """Try to find a source_hash from integrity records by path. / 尝试根据路径从完整性记录中查找 source_hash。"""
        try:
            for rec in self.integrity.list_processed():
                if rec["file_path"] == source_path:
                    return rec["file_hash"]
        except Exception:
            pass
        return None
