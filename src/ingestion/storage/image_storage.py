"""Image storage with SQLite indexing for multimodal RAG. / 面向多模态 RAG、带 SQLite 索引的图片存储。

This module provides persistent image storage with SQLite-backed indexing, / 本模块提供带 SQLite 索引的持久化图片存储，
enabling efficient image retrieval and management across collections. / 支持跨集合高效检索和管理图片。

Design Principles: / 设计原则：
- Persistent: Images stored on filesystem, metadata in SQLite / 持久化：图片存储在文件系统中，元数据存储在 SQLite 中
- Concurrent: WAL mode enables concurrent read/write operations / 并发：WAL 模式支持并发读写操作
- Idempotent: Re-saving same image_id updates metadata safely / 幂等：重复保存相同 image_id 会安全更新元数据
- Organized: Images grouped by collection for namespace isolation / 组织化：图片按集合分组，实现命名空间隔离
"""

import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Union


class ImageStorage:
    """SQLite-backed image storage manager. / 基于 SQLite 的图片存储管理器。
    
    Stores image files in organized directory structure and maintains / 将图片文件存储在有组织的目录结构中，并维护
    a SQLite index for efficient lookup and querying. / SQLite 索引用于高效查找和查询。
    
    Directory Structure: / 目录结构：
        data/images/{collection}/{image_id}.png
    
    Database Schema: / 数据库结构：
        image_index (
            image_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            collection TEXT,
            doc_hash TEXT,
            page_num INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        
        INDEX idx_collection ON image_index(collection)
        INDEX idx_doc_hash ON image_index(doc_hash)
    
    Args: / 参数：
        db_path: Path to SQLite database file (default: data/db/image_index.db). / db_path：SQLite 数据库文件路径（默认：data/db/image_index.db）。
        images_root: Root directory for image storage (default: data/images). / images_root：图片存储根目录（默认：data/images）。
    
    Example: / 示例：
        >>> storage = ImageStorage()
        >>> 
        >>> # Save an image / 保存图片
        >>> with open("sample.png", "rb") as f:
        >>>     image_data = f.read()
        >>> path = storage.save_image(
        ...     image_id="doc123_p1_img0",
        ...     image_data=image_data,
        ...     collection="contracts",
        ...     doc_hash="abc123",
        ...     page_num=1
        ... )
        >>> print(path)  # data/images/contracts/doc123_p1_img0.png / data/images/contracts/doc123_p1_img0.png
        >>> 
        >>> # Retrieve image path / 获取图片路径
        >>> path = storage.get_image_path("doc123_p1_img0")
        >>> print(path)  # data/images/contracts/doc123_p1_img0.png / data/images/contracts/doc123_p1_img0.png
        >>> 
        >>> # List images in collection / 列出集合中的图片
        >>> images = storage.list_images("contracts")
        >>> print(len(images))  # 1 / 1
    """
    
    def __init__(
        self,
        db_path: str = "data/db/image_index.db",
        images_root: str = "data/images"
    ):
        """Initialize image storage and create database if needed. / 初始化图片存储，并在需要时创建数据库。
        
        Args: / 参数：
            db_path: Path to SQLite database file. / db_path：SQLite 数据库文件路径。
            images_root: Root directory for storing image files. / images_root：图片文件存储根目录。
        """
        self.db_path = db_path
        self.images_root = Path(images_root)
        self._conn = None
        self._ensure_database()
    
    def close(self) -> None:
        """Close database connection if open. / 如果数据库连接已打开，则关闭它。"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __del__(self):
        """Cleanup: close connection on deletion. / 清理：对象删除时关闭连接。"""
        self.close()
    
    def _ensure_database(self) -> None:
        """Create database file and schema if they don't exist. / 如果数据库文件和结构不存在，则创建它们。"""
        # Create parent directories / 创建父目录
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create images root directory / 创建图片根目录
        self.images_root.mkdir(parents=True, exist_ok=True)
        
        # Connect and initialize schema / 连接并初始化数据库结构
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for concurrent access / 启用 WAL 模式以支持并发访问
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Create table if not exists / 如果表不存在则创建
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for efficient queries / 创建索引以提高查询效率
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection 
                ON image_index(collection)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash 
                ON image_index(doc_hash)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def save_image(
        self,
        image_id: str,
        image_data: Union[bytes, Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        extension: str = "png"
    ) -> str:
        """Save image to filesystem and register in database. / 将图片保存到文件系统并注册到数据库。
        
        This operation is idempotent - re-saving with same image_id / 该操作是幂等的：使用相同 image_id 再次保存
        will update the metadata and overwrite the file. / 会更新元数据并覆盖文件。
        
        Args: / 参数：
            image_id: Unique identifier for the image. / image_id：图片唯一标识。
            image_data: Image data as bytes, or path to source image file. / image_data：图片字节数据，或源图片文件路径。
            collection: Optional collection/namespace for organization. / collection：用于组织的可选集合/命名空间。
            doc_hash: Optional document hash for traceability. / doc_hash：用于追溯的可选文档哈希。
            page_num: Optional page number if from paginated document. / page_num：如果来自分页文档，则为可选页码。
            extension: File extension without dot (default: "png"). / extension：不带点的文件扩展名（默认："png"）。
        
        Returns: / 返回：
            Relative path where image was saved. / 图片保存位置的相对路径。
            
        Raises: / 异常：
            ValueError: If image_id is empty or invalid. / ValueError：当 image_id 为空或无效时抛出。
            IOError: If image file cannot be saved. / IOError：当图片文件无法保存时抛出。
            RuntimeError: If database operation fails. / RuntimeError：当数据库操作失败时抛出。
            
        Example:
            >>> # Save from bytes / 从字节保存
            >>> path = storage.save_image("img1", b"PNG_DATA", "docs")
            >>> 
            >>> # Save from file / 从文件保存
            >>> path = storage.save_image("img2", Path("source.png"), "docs")
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")
        
        # Determine collection directory / 确定集合目录
        if collection:
            collection_dir = self.images_root / collection
        else:
            collection_dir = self.images_root / "default"
        
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        # Build image file path / 构建图片文件路径
        image_filename = f"{image_id}.{extension}"
        image_path = collection_dir / image_filename
        
        # Save image file / 保存图片文件
        try:
            if isinstance(image_data, bytes):
                # Write bytes directly / 直接写入字节
                image_path.write_bytes(image_data)
            elif isinstance(image_data, (Path, str)):
                # Copy from source file / 从源文件复制
                source_path = Path(image_data)
                if not source_path.exists():
                    raise FileNotFoundError(f"Source image not found: {source_path}")
                shutil.copy2(source_path, image_path)
            else:
                raise ValueError(f"Unsupported image_data type: {type(image_data)}")
        except Exception as e:
            raise IOError(f"Failed to save image {image_id}: {e}")
        
        # Store absolute path for reliable retrieval / 存储绝对路径以保证可靠检索
        # (relative paths would fail with temp directories in tests) / （测试中使用临时目录时，相对路径会失效）
        stored_path = str(image_path.resolve())
        
        # Register in database / 注册到数据库
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Use INSERT OR REPLACE for idempotent operation / 使用 INSERT OR REPLACE 实现幂等操作
            conn.execute("""
                INSERT OR REPLACE INTO image_index 
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, stored_path, collection, doc_hash, page_num, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to register image {image_id}: {e}")
        finally:
            conn.close()
        
        return stored_path
    
    def register_image(
        self,
        image_id: str,
        file_path: Union[Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> str:
        """Register an existing image file in the database index. / 将已有图片文件注册到数据库索引。
        
        Unlike save_image(), this method does NOT copy or move the file. / 与 save_image() 不同，该方法不会复制或移动文件。
        It only creates a database entry pointing to the existing file. / 它只创建一条指向已有文件的数据库记录。
        Use this when the image has already been saved by another component / 当图片已由其他组件保存
        (e.g., PdfLoader) and you just need to index it. / （例如 PdfLoader），且只需建立索引时使用。
        
        Args:
            image_id: Unique identifier for the image. / image_id：图片唯一标识。
            file_path: Path to the existing image file. / file_path：已有图片文件路径。
            collection: Optional collection/namespace for organization. / collection：用于组织的可选集合/命名空间。
            doc_hash: Optional document hash for traceability. / doc_hash：用于追溯的可选文档哈希。
            page_num: Optional page number if from paginated document. / page_num：如果来自分页文档，则为可选页码。
        
        Returns:
            Absolute path to the registered image. / 已注册图片的绝对路径。
            
        Raises:
            ValueError: If image_id is empty or invalid. / ValueError：当 image_id 为空或无效时抛出。
            FileNotFoundError: If the image file does not exist. / FileNotFoundError：当图片文件不存在时抛出。
            RuntimeError: If database operation fails. / RuntimeError：当数据库操作失败时抛出。
            
        Example:
            >>> # Register an image that was saved by PdfLoader / 注册由 PdfLoader 保存的图片
            >>> path = storage.register_image(
            ...     image_id="doc123_p1_img0",
            ...     file_path="data/images/tech_docs/abc123/doc123_p1_img0.png",
            ...     collection="tech_docs",
            ...     doc_hash="abc123",
            ...     page_num=1
            ... )
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")
        
        # Verify file exists / 验证文件存在
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Store absolute path for reliable retrieval / 存储绝对路径以保证可靠检索
        stored_path = str(path.resolve())
        
        # Register in database / 注册到数据库
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Use INSERT OR REPLACE for idempotent operation / 使用 INSERT OR REPLACE 实现幂等操作
            conn.execute("""
                INSERT OR REPLACE INTO image_index 
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, stored_path, collection, doc_hash, page_num, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to register image {image_id}: {e}")
        finally:
            conn.close()
        
        return stored_path
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """Get filesystem path for an image by ID. / 根据 ID 获取图片的文件系统路径。
        
        Args:
            image_id: Unique identifier for the image. / image_id：图片唯一标识。
            
        Returns:
            Relative file path if image exists, None otherwise. / 如果图片存在则返回相对文件路径，否则返回 None。
            
        Example:
            >>> path = storage.get_image_path("img1")
            >>> if path:
            ...     with open(path, "rb") as f:
            ...         image_data = f.read()
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def image_exists(self, image_id: str) -> bool:
        """Check if image exists in database. / 检查图片是否存在于数据库中。
        
        Args:
            image_id: Unique identifier for the image. / image_id：图片唯一标识。
            
        Returns:
            True if image is registered, False otherwise. / 如果图片已注册则返回 True，否则返回 False。
        """
        return self.get_image_path(image_id) is not None
    
    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """List images with optional filtering. / 使用可选过滤条件列出图片。
        
        Args:
            collection: Optional collection filter. / collection：可选集合过滤条件。
            doc_hash: Optional document hash filter. / doc_hash：可选文档哈希过滤条件。
            
        Returns:
            List of image metadata dictionaries with keys: / 图片元数据字典列表，包含以下键：
            - image_id: Image identifier / image_id：图片标识
            - file_path: Filesystem path / file_path：文件系统路径
            - collection: Collection name / collection：集合名称
            - doc_hash: Document hash / doc_hash：文档哈希
            - page_num: Page number (if applicable) / page_num：页码（如适用）
            - created_at: Creation timestamp / created_at：创建时间戳
            
        Example:
            >>> # List all images in a collection / 列出集合中的所有图片
            >>> images = storage.list_images(collection="contracts")
            >>> for img in images:
            ...     print(img["image_id"], img["file_path"])
            
            >>> # List images from a specific document / 列出指定文档的图片
            >>> images = storage.list_images(doc_hash="abc123")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access / 启用类似字典的访问
        
        try:
            # Build query with optional filters / 使用可选过滤条件构建查询
            query = "SELECT * FROM image_index WHERE 1=1"
            params = []
            
            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)
            
            if doc_hash is not None:
                query += " AND doc_hash = ?"
                params.append(doc_hash)
            
            query += " ORDER BY created_at ASC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries / 将行转换为字典
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def delete_image(self, image_id: str, remove_file: bool = True) -> bool:
        """Delete image from database and optionally from filesystem. / 从数据库删除图片，并可选从文件系统删除。
        
        Args:
            image_id: Unique identifier for the image. / image_id：图片唯一标识。
            remove_file: If True, also delete the image file (default: True). / remove_file：如果为 True，也删除图片文件（默认：True）。
            
        Returns:
            True if image was deleted, False if not found. / 如果图片已删除则返回 True，未找到则返回 False。
            
        Example:
            >>> # Delete image and file / 删除图片和文件
            >>> deleted = storage.delete_image("img1")
            >>> 
            >>> # Remove from database only, keep file / 仅从数据库移除，保留文件
            >>> deleted = storage.delete_image("img2", remove_file=False)
        """
        # Get file path before deleting from database / 从数据库删除前先获取文件路径
        file_path = self.get_image_path(image_id)
        
        if file_path is None:
            return False
        
        # Delete from database / 从数据库删除
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        except sqlite3.Error:
            return False
        finally:
            conn.close()
        
        # Optionally delete file / 可选删除文件
        if remove_file and deleted:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                # Log but don't fail if file deletion fails / 文件删除失败时仅记录但不使操作失败
                pass
        
        return deleted
    
    def get_collection_stats(self, collection: str) -> Dict[str, any]:
        """Get statistics for a collection. / 获取集合统计信息。
        
        Args:
            collection: Collection name. / collection：集合名称。
            
        Returns:
            Dictionary with statistics: / 包含统计信息的字典：
            - total_images: Number of images in collection / total_images：集合中的图片数量
            - total_size_bytes: Total file size (if files exist) / total_size_bytes：总文件大小（如果文件存在）
            
        Example:
            >>> stats = storage.get_collection_stats("contracts")
            >>> print(f"Total images: {stats['total_images']}")
        """
        images = self.list_images(collection=collection)
        
        total_size = 0
        for img in images:
            try:
                file_path = Path(img["file_path"])
                if file_path.exists():
                    total_size += file_path.stat().st_size
            except Exception:
                pass
        
        return {
            "total_images": len(images),
            "total_size_bytes": total_size
        }
