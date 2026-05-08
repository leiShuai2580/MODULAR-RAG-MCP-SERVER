"""Unit tests for ImageStorage module. / ImageStorage 模块的单元测试。"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def temp_storage():
    """Create ImageStorage with temporary directories. / 使用临时目录创建 ImageStorage。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "image_index.db")
        images_root = str(Path(tmpdir) / "images")
        storage = ImageStorage(db_path=db_path, images_root=images_root)
        yield storage
        storage.close()


@pytest.fixture
def sample_image_data():
    """Generate sample PNG-like binary data. / 生成类似 PNG 的示例二进制数据。"""
    # Simple PNG header (not a valid PNG, but sufficient for testing) / 简单 PNG 头（不是有效 PNG，但足以测试）
    return b'\x89PNG\r\n\x1a\n' + b'\x00' * 100


class TestImageStorageInitialization:
    """Test ImageStorage initialization and setup. / 测试 ImageStorage 初始化和设置。"""
    
    def test_creates_database_file(self):
        """Database file should be created on initialization. / 初始化时应创建数据库文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            storage = ImageStorage(db_path=db_path)
            
            assert Path(db_path).exists()
            storage.close()
    
    def test_creates_images_directory(self):
        """Images root directory should be created. / 应创建图片根目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_root = str(Path(tmpdir) / "images")
            storage = ImageStorage(images_root=images_root)
            
            assert Path(images_root).exists()
            storage.close()
    
    def test_creates_nested_directories(self):
        """Should create nested parent directories. / 应创建嵌套父目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nested" / "path" / "db.db")
            storage = ImageStorage(db_path=db_path)
            
            assert Path(db_path).exists()
            storage.close()
    
    def test_database_schema_created(self, temp_storage):
        """Database should have correct schema. / 数据库应具有正确 schema。"""
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='image_index'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_database_indexes_created(self, temp_storage):
        """Database should have collection and doc_hash indexes. / 数据库应具有 collection 和 doc_hash 索引。"""
        conn = sqlite3.connect(temp_storage.db_path)
        
        # Check for idx_collection / 检查 idx_collection
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_collection'"
        )
        assert cursor.fetchone() is not None
        
        # Check for idx_doc_hash / 检查 idx_doc_hash
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_doc_hash'"
        )
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_wal_mode_enabled(self, temp_storage):
        """Database should use WAL mode for concurrency. / 数据库应使用 WAL 模式支持并发。"""
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()
        
        assert mode.lower() == "wal"


class TestSaveImage:
    """Test image saving functionality. / 测试图片保存功能。"""
    
    def test_save_image_from_bytes(self, temp_storage, sample_image_data):
        """Should save image from bytes data. / 应从 bytes 数据保存图片。"""
        path = temp_storage.save_image(
            image_id="test_img_1",
            image_data=sample_image_data,
            collection="test_collection"
        )
        
        assert path is not None
        assert Path(path).exists()
        assert Path(path).read_bytes() == sample_image_data
    
    def test_save_image_from_file(self, temp_storage):
        """Should save image by copying from source file. / 应通过复制源文件保存图片。"""
        # Create source file / 创建源文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(b"test_image_data")
            source_path = tmp.name
        
        try:
            path = temp_storage.save_image(
                image_id="test_img_2",
                image_data=source_path,
                collection="test_collection"
            )
            
            assert path is not None
            assert Path(path).exists()
            assert Path(path).read_bytes() == b"test_image_data"
        finally:
            Path(source_path).unlink(missing_ok=True)
    
    def test_save_image_creates_collection_directory(self, temp_storage, sample_image_data):
        """Should create collection directory if it doesn't exist. / collection 目录不存在时应创建。"""
        path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="new_collection"
        )
        
        assert "new_collection" in path
        collection_dir = Path(temp_storage.images_root) / "new_collection"
        assert collection_dir.exists()
    
    def test_save_image_with_custom_extension(self, temp_storage, sample_image_data):
        """Should respect custom file extension. / 应遵守自定义文件扩展名。"""
        path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="test",
            extension="jpg"
        )
        
        assert path.endswith(".jpg")
    
    def test_save_image_registers_in_database(self, temp_storage, sample_image_data):
        """Should register image metadata in database. / 应在数据库中注册图片 metadata。"""
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="test",
            doc_hash="abc123",
            page_num=5
        )
        
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute(
            "SELECT image_id, collection, doc_hash, page_num FROM image_index WHERE image_id = ?",
            ("img1",)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "img1"
        assert row[1] == "test"
        assert row[2] == "abc123"
        assert row[3] == 5
    
    def test_save_image_idempotent(self, temp_storage, sample_image_data):
        """Re-saving same image_id should update, not duplicate. / 重新保存相同 image_id 应更新而不是重复。"""
        # Save first time / 第一次保存
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="coll1"
        )
        
        # Save again with different collection / 使用不同 collection 再次保存
        new_data = b"updated_data"
        temp_storage.save_image(
            image_id="img1",
            image_data=new_data,
            collection="coll2"
        )
        
        # Should have only one record / 应只有一条记录
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM image_index WHERE image_id = ?", ("img1",))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
        
        # File should be updated / 文件应被更新
        path = temp_storage.get_image_path("img1")
        assert Path(path).read_bytes() == new_data
    
    def test_save_image_empty_id_raises_error(self, temp_storage, sample_image_data):
        """Empty image_id should raise ValueError. / 空 image_id 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="image_id cannot be empty"):
            temp_storage.save_image("", sample_image_data)
        
        with pytest.raises(ValueError, match="image_id cannot be empty"):
            temp_storage.save_image("   ", sample_image_data)
    
    def test_save_image_missing_source_file_raises_error(self, temp_storage):
        """Non-existent source file should raise error. / 不存在的源文件应抛出错误。"""
        with pytest.raises(IOError, match="Failed to save image"):
            temp_storage.save_image(
                image_id="img1",
                image_data="/nonexistent/path.png"
            )
    
    def test_save_image_uses_default_collection(self, temp_storage, sample_image_data):
        """Should use 'default' collection when none specified. / 未指定时应使用 'default' collection。"""
        path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data
        )
        
        assert "default" in path


class TestGetImagePath:
    """Test image path retrieval. / 测试图片路径获取。"""
    
    def test_get_existing_image_path(self, temp_storage, sample_image_data):
        """Should return path for existing image. / 对已存在图片应返回路径。"""
        original_path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data
        )
        
        retrieved_path = temp_storage.get_image_path("img1")
        
        assert retrieved_path == original_path
    
    def test_get_nonexistent_image_path_returns_none(self, temp_storage):
        """Should return None for non-existent image. / 对不存在图片应返回 None。"""
        path = temp_storage.get_image_path("nonexistent_id")
        
        assert path is None


class TestImageExists:
    """Test image existence checking. / 测试图片存在性检查。"""
    
    def test_image_exists_returns_true(self, temp_storage, sample_image_data):
        """Should return True for existing image. / 对已存在图片应返回 True。"""
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data
        )
        
        assert temp_storage.image_exists("img1") is True
    
    def test_image_exists_returns_false(self, temp_storage):
        """Should return False for non-existent image. / 对不存在图片应返回 False。"""
        assert temp_storage.image_exists("nonexistent") is False


class TestListImages:
    """Test image listing functionality. / 测试图片列表功能。"""
    
    def test_list_all_images(self, temp_storage, sample_image_data):
        """Should list all images when no filter. / 无过滤条件时应列出所有图片。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll2")
        temp_storage.save_image("img3", sample_image_data, "coll1")
        
        images = temp_storage.list_images()
        
        assert len(images) == 3
        image_ids = {img["image_id"] for img in images}
        assert image_ids == {"img1", "img2", "img3"}
    
    def test_list_images_by_collection(self, temp_storage, sample_image_data):
        """Should filter images by collection. / 应按 collection 过滤图片。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll2")
        temp_storage.save_image("img3", sample_image_data, "coll1")
        
        images = temp_storage.list_images(collection="coll1")
        
        assert len(images) == 2
        image_ids = {img["image_id"] for img in images}
        assert image_ids == {"img1", "img3"}
    
    def test_list_images_by_doc_hash(self, temp_storage, sample_image_data):
        """Should filter images by document hash. / 应按 document hash 过滤图片。"""
        temp_storage.save_image("img1", sample_image_data, "coll1", doc_hash="hash1")
        temp_storage.save_image("img2", sample_image_data, "coll1", doc_hash="hash2")
        temp_storage.save_image("img3", sample_image_data, "coll1", doc_hash="hash1")
        
        images = temp_storage.list_images(doc_hash="hash1")
        
        assert len(images) == 2
        image_ids = {img["image_id"] for img in images}
        assert image_ids == {"img1", "img3"}
    
    def test_list_images_with_both_filters(self, temp_storage, sample_image_data):
        """Should combine collection and doc_hash filters. / 应组合 collection 和 doc_hash 过滤器。"""
        temp_storage.save_image("img1", sample_image_data, "coll1", doc_hash="hash1")
        temp_storage.save_image("img2", sample_image_data, "coll2", doc_hash="hash1")
        temp_storage.save_image("img3", sample_image_data, "coll1", doc_hash="hash2")
        
        images = temp_storage.list_images(collection="coll1", doc_hash="hash1")
        
        assert len(images) == 1
        assert images[0]["image_id"] == "img1"
    
    def test_list_images_returns_complete_metadata(self, temp_storage, sample_image_data):
        """Should return complete metadata for each image. / 应返回每张图片的完整 metadata。"""
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="test",
            doc_hash="abc123",
            page_num=5
        )
        
        images = temp_storage.list_images()
        
        assert len(images) == 1
        img = images[0]
        assert img["image_id"] == "img1"
        assert img["collection"] == "test"
        assert img["doc_hash"] == "abc123"
        assert img["page_num"] == 5
        assert "file_path" in img
        assert "created_at" in img
    
    def test_list_images_empty_when_no_matches(self, temp_storage):
        """Should return empty list when no images match. / 没有图片匹配时应返回空列表。"""
        images = temp_storage.list_images(collection="nonexistent")
        
        assert images == []


class TestDeleteImage:
    """Test image deletion. / 测试图片删除。"""
    
    def test_delete_image_removes_from_database(self, temp_storage, sample_image_data):
        """Should remove image from database. / 应从数据库删除图片。"""
        temp_storage.save_image("img1", sample_image_data)
        
        deleted = temp_storage.delete_image("img1")
        
        assert deleted is True
        assert temp_storage.image_exists("img1") is False
    
    def test_delete_image_removes_file(self, temp_storage, sample_image_data):
        """Should delete image file by default. / 默认应删除图片文件。"""
        path = temp_storage.save_image("img1", sample_image_data)
        
        temp_storage.delete_image("img1", remove_file=True)
        
        assert not Path(path).exists()
    
    def test_delete_image_keeps_file_when_requested(self, temp_storage, sample_image_data):
        """Should keep file when remove_file=False. / remove_file=False 时应保留文件。"""
        path = temp_storage.save_image("img1", sample_image_data)
        
        temp_storage.delete_image("img1", remove_file=False)
        
        assert Path(path).exists()  # File still exists / 文件仍存在
        assert temp_storage.image_exists("img1") is False  # But not in database / 但不在数据库中
    
    def test_delete_nonexistent_image_returns_false(self, temp_storage):
        """Should return False when deleting non-existent image. / 删除不存在图片时应返回 False。"""
        deleted = temp_storage.delete_image("nonexistent")
        
        assert deleted is False


class TestGetCollectionStats:
    """Test collection statistics. / 测试 collection 统计。"""
    
    def test_get_collection_stats_counts_images(self, temp_storage, sample_image_data):
        """Should count total images in collection. / 应统计 collection 中的图片总数。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll1")
        temp_storage.save_image("img3", sample_image_data, "coll2")
        
        stats = temp_storage.get_collection_stats("coll1")
        
        assert stats["total_images"] == 2
    
    def test_get_collection_stats_calculates_size(self, temp_storage, sample_image_data):
        """Should calculate total size of images. / 应计算图片总大小。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll1")
        
        stats = temp_storage.get_collection_stats("coll1")
        
        # Should be approximately 2 * len(sample_image_data) / 应约等于 2 * len(sample_image_data)
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_bytes"] == 2 * len(sample_image_data)
    
    def test_get_collection_stats_empty_collection(self, temp_storage):
        """Should handle empty collection. / 应处理空 collection。"""
        stats = temp_storage.get_collection_stats("empty")
        
        assert stats["total_images"] == 0
        assert stats["total_size_bytes"] == 0


class TestConcurrency:
    """Test concurrent access patterns. / 测试并发访问模式。"""
    
    def test_multiple_connections_can_read(self, temp_storage, sample_image_data):
        """WAL mode should allow concurrent reads. / WAL 模式应允许并发读取。"""
        temp_storage.save_image("img1", sample_image_data)
        
        # Simulate multiple readers / 模拟多个读取者
        path1 = temp_storage.get_image_path("img1")
        path2 = temp_storage.get_image_path("img1")
        
        assert path1 == path2
    
    def test_can_write_while_reading(self, temp_storage, sample_image_data):
        """WAL mode should allow writes during reads. / WAL 模式应允许读取期间写入。"""
        temp_storage.save_image("img1", sample_image_data)
        
        # Read existing / 读取已有数据
        path1 = temp_storage.get_image_path("img1")
        
        # Write new / 写入新数据
        temp_storage.save_image("img2", sample_image_data)
        
        # Read should still work / 读取仍应正常工作
        path2 = temp_storage.get_image_path("img2")
        
        assert path1 is not None
        assert path2 is not None


class TestEdgeCases:
    """Test edge cases and error handling. / 测试边界情况和错误处理。"""
    
    def test_special_characters_in_image_id(self, temp_storage, sample_image_data):
        """Should handle special characters in image_id. / 应处理 image_id 中的特殊字符。"""
        # Note: Some characters might be problematic for filesystems / 注意：某些字符可能对文件系统有问题
        # Using safe special characters / 使用安全特殊字符
        image_id = "doc_123-page_5.img_0"
        
        path = temp_storage.save_image(image_id, sample_image_data)
        
        assert temp_storage.image_exists(image_id)
        retrieved_path = temp_storage.get_image_path(image_id)
        assert retrieved_path == path
    
    def test_unicode_in_collection_name(self, temp_storage, sample_image_data):
        """Should handle Unicode in collection names. / 应处理 collection 名称中的 Unicode。"""
        collection = "文档集合"
        
        path = temp_storage.save_image(
            "img1",
            sample_image_data,
            collection=collection
        )
        
        assert collection in path
        images = temp_storage.list_images(collection=collection)
        assert len(images) == 1
    
    def test_close_and_reopen(self, sample_image_data):
        """Should persist data across close/reopen. / 关闭并重新打开后应持久化数据。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            images_root = str(Path(tmpdir) / "images")
            
            # Create and save / 创建并保存
            storage1 = ImageStorage(db_path=db_path, images_root=images_root)
            storage1.save_image("img1", sample_image_data, "coll1")
            storage1.close()
            
            # Reopen and verify / 重新打开并验证
            storage2 = ImageStorage(db_path=db_path, images_root=images_root)
            assert storage2.image_exists("img1")
            images = storage2.list_images()
            assert len(images) == 1
            storage2.close()
