"""File integrity checker for incremental ingestion. / 用于增量摄取的文件完整性检查器。

This module provides SHA256-based file integrity tracking to enable incremental / 此模块提供基于 SHA256 的文件完整性追踪，以支持增量
ingestion. Files that have been successfully processed can be skipped on / 摄取。已成功处理的文件可以在
subsequent ingestion runs. / 后续摄取运行中跳过。

Design Principles: / 设计原则：
- Idempotent: Multiple ingestion runs of the same file are safe / 幂等：同一文件多次摄取运行是安全的
- Persistent: SQLite-backed storage survives process restarts / 持久化：SQLite 存储可跨进程重启保留
- Concurrent: WAL mode enables concurrent read/write operations / 并发：WAL 模式支持并发读写操作
- Graceful: Failed ingestions are tracked but don't block retries / 优雅：失败摄取会被追踪，但不会阻止重试
"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileIntegrityChecker(ABC):
    """Abstract base class for file integrity checking. / 文件完整性检查的抽象基类。
    
    Implementations track which files have been successfully processed / 实现类会追踪哪些文件已成功处理，
    to enable incremental ingestion. / 以支持增量摄取。
    """
    
    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of file. / 计算文件的 SHA256 哈希。
        
        Args: / 参数：
            file_path: Path to the file to hash. / 要计算哈希的文件路径。
            
        Returns: / 返回：
            Hexadecimal SHA256 hash string (64 characters). / 十六进制 SHA256 哈希字符串（64 个字符）。
            
        Raises: / 异常：
            FileNotFoundError: If file does not exist. / 如果文件不存在。
            IOError: If path is not a file or cannot be read. / 如果路径不是文件或无法读取。
        """
        pass
    
    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Check if file should be skipped based on hash. / 基于哈希检查是否应跳过文件。
        
        Args: / 参数：
            file_hash: SHA256 hash of the file. / 文件的 SHA256 哈希。
            
        Returns: / 返回：
            True if file has been successfully processed before, False otherwise. / 如果文件此前已成功处理则为 True，否则为 False。
        """
        pass
    
    @abstractmethod
    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """Mark file as successfully processed. / 将文件标记为已成功处理。
        
        Args: / 参数：
            file_hash: SHA256 hash of the file. / 文件的 SHA256 哈希。
            file_path: Original file path (for tracking). / 原始文件路径（用于追踪）。
            collection: Optional collection/namespace identifier. / 可选 collection/namespace 标识符。
            
        Raises: / 异常：
            RuntimeError: If database operation fails. / 如果数据库操作失败。
        """
        pass
    
    @abstractmethod
    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """Mark file processing as failed. / 将文件处理标记为失败。
        
        Failed files are tracked but not skipped on subsequent runs, / 失败文件会被追踪，但不会在后续运行中跳过，
        allowing retries. / 从而允许重试。
        
        Args: / 参数：
            file_hash: SHA256 hash of the file. / 文件的 SHA256 哈希。
            file_path: Original file path (for tracking). / 原始文件路径（用于追踪）。
            error_msg: Error message describing the failure. / 描述失败原因的错误信息。
            
        Raises: / 异常：
            RuntimeError: If database operation fails. / 如果数据库操作失败。
        """
        pass

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:
        """Remove an ingestion record by its file hash. / 按文件哈希删除摄取记录。

        Args: / 参数：
            file_hash: SHA256 hash identifying the record. / 标识记录的 SHA256 哈希。

        Returns: / 返回：
            True if a record was deleted, False if not found. / 如果删除了记录则为 True，未找到则为 False。
        """
        pass

    @abstractmethod
    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List successfully processed files. / 列出已成功处理的文件。

        Args: / 参数：
            collection: Optional collection filter.  When *None* all / 可选 collection 过滤器。当为 *None* 时，
                successful records are returned. / 返回所有成功记录。

        Returns: / 返回：
            List of dicts with keys: file_hash, file_path, collection, / 字典列表，包含键：file_hash、file_path、collection、
            processed_at, updated_at. / processed_at、updated_at。
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed file integrity checker. / 基于 SQLite 的文件完整性检查器。
    
    Stores ingestion history in a SQLite database with WAL mode for / 使用 WAL 模式在 SQLite 数据库中存储摄取历史，
    concurrent access. / 以支持并发访问。
    
    Database Schema: / 数据库结构：
        ingestion_history (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,  -- 'success' or 'failed'
            collection TEXT,
            error_msg TEXT,
            processed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    
    Args: / 参数：
        db_path: Path to SQLite database file (will be created if needed). / SQLite 数据库文件路径（需要时会创建）。
    
    Raises: / 异常：
        sqlite3.DatabaseError: If database file is corrupted. / 如果数据库文件损坏。
    """
    
    def __init__(self, db_path: str):
        """Initialize checker and create database if needed. / 初始化检查器，并在需要时创建数据库。
        
        Args: / 参数：
            db_path: Path to SQLite database file. / SQLite 数据库文件路径。
        """
        self.db_path = db_path
        self._conn = None
        self._ensure_database()
    
    def close(self) -> None:
        """Close database connection if open. / 如果数据库连接已打开则关闭。"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __del__(self):
        """Cleanup: close connection on deletion. / 清理：删除对象时关闭连接。"""
        self.close()
    
    def _ensure_database(self) -> None:
        """Create database file and schema if they don't exist. / 如果数据库文件和结构不存在则创建。"""
        # Create parent directories if needed / 需要时创建父目录
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and initialize schema / 连接并初始化结构
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for concurrent access / 启用 WAL 模式以支持并发访问
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Create table if not exists / 如果表不存在则创建
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection TEXT,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create index on status for faster queries / 在 status 上创建索引以加速查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON ingestion_history(status)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of file using chunked reading. / 使用分块读取计算文件 SHA256 哈希。
        
        Uses 64KB chunks to handle large files without loading entire / 使用 64KB 块处理大文件，
        file into memory. / 无需将整个文件加载到内存。
        
        Args: / 参数：
            file_path: Path to the file to hash. / 要计算哈希的文件路径。
            
        Returns: / 返回：
            Hexadecimal SHA256 hash string (64 characters). / 十六进制 SHA256 哈希字符串（64 个字符）。
            
        Raises: / 异常：
            FileNotFoundError: If file does not exist. / 如果文件不存在。
            IOError: If path is not a file or cannot be read. / 如果路径不是文件或无法读取。
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")
        
        # Compute hash using chunked reading / 使用分块读取计算哈希
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read in 64KB chunks / 以 64KB 块读取
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")
        
        return sha256_hash.hexdigest()
    
    def should_skip(self, file_hash: str) -> bool:
        """Check if file should be skipped. / 检查是否应跳过文件。
        
        Only files with status='success' are skipped. Failed files / 只有 status='success' 的文件会被跳过。失败文件
        can be retried. / 可以重试。
        
        Args: / 参数：
            file_hash: SHA256 hash of the file. / 文件的 SHA256 哈希。
            
        Returns: / 返回：
            True if file has status='success', False otherwise. / 如果文件状态为 'success' 则为 True，否则为 False。
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result is None:
                return False
            
            return result[0] == "success"
        finally:
            conn.close()
    
    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """Mark file as successfully processed. / 将文件标记为已成功处理。
        
        Uses INSERT OR REPLACE for idempotent operation. / 使用 INSERT OR REPLACE 保证操作幂等。
        
        Args: / 参数：
            file_hash: SHA256 hash of the file. / 文件的 SHA256 哈希。
            file_path: Original file path (for tracking). / 原始文件路径（用于追踪）。
            collection: Optional collection/namespace identifier. / 可选 collection/namespace 标识符。
            
        Raises: / 异常：
            RuntimeError: If database operation fails. / 如果数据库操作失败。
        """
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if record exists to preserve processed_at / 检查记录是否存在以保留 processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing record / 更新现有记录
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'success',
                        collection = ?,
                        error_msg = NULL,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, collection, now, file_hash))
            else:
                # Insert new record / 插入新记录
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'success', ?, NULL, ?, ?)
                """, (file_hash, file_path, collection, now, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark success for {file_path}: {e}")
        finally:
            conn.close()
    
    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """Mark file processing as failed. / 将文件处理标记为失败。
        
        Failed files are not skipped, allowing retries. / 失败文件不会被跳过，以允许重试。
        
        Args: / 参数：
            file_hash: SHA256 hash of the file. / 文件的 SHA256 哈希。
            file_path: Original file path (for tracking). / 原始文件路径（用于追踪）。
            error_msg: Error message describing the failure. / 描述失败原因的错误信息。
            
        Raises: / 异常：
            RuntimeError: If database operation fails. / 如果数据库操作失败。
        """
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if record exists to preserve processed_at / 检查记录是否存在以保留 processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing record / 更新现有记录
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'failed',
                        error_msg = ?,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, error_msg, now, file_hash))
            else:
                # Insert new record / 插入新记录
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'failed', NULL, ?, ?, ?)
                """, (file_hash, file_path, error_msg, now, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark failure for {file_path}: {e}")
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> bool:
        """Remove an ingestion record by its file hash. / 按文件哈希删除摄取记录。

        Args: / 参数：
            file_hash: SHA256 hash identifying the record. / 标识记录的 SHA256 哈希。

        Returns: / 返回：
            True if a record was deleted, False if not found. / 如果删除了记录则为 True，未找到则为 False。
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove record {file_hash}: {e}")
        finally:
            conn.close()

    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List successfully processed files. / 列出已成功处理的文件。

        Args: / 参数：
            collection: Optional collection filter. / 可选 collection 过滤器。

        Returns: / 返回：
            List of dicts with keys: file_hash, file_path, collection, / 字典列表，包含键：file_hash、file_path、collection、
            processed_at, updated_at. / processed_at、updated_at。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = (
                "SELECT file_hash, file_path, collection, processed_at, updated_at "
                "FROM ingestion_history WHERE status = 'success'"
            )
            params: list[str] = []
            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)
            query += " ORDER BY processed_at ASC"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
