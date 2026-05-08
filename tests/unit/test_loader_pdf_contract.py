"""Unit tests for PDF Loader contract and behavior. / PDF Loader 契约和行为的单元测试。

Tests verify: / 测试验证：
- BaseLoader abstract interface / BaseLoader 抽象接口
- PdfLoader initialization and configuration / PdfLoader 初始化与配置
- Helper methods (hash computation, title extraction, etc.) / 辅助方法（hash 计算、标题提取等）
- Error handling for invalid inputs / 无效输入的错误处理
- Core PDF conversion functionality using real test files / 使用真实测试文件验证核心 PDF 转换功能

Note: Additional integration tests are in tests/integration/test_pdf_loader_integration.py / 注意：额外集成测试位于 tests/integration/test_pdf_loader_integration.py
"""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader


# Test fixtures paths / 测试 fixtures 路径
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


class TestBaseLoader:
    """Tests for BaseLoader abstract interface. / BaseLoader 抽象接口测试。"""
    
    def test_cannot_instantiate_abstract_class(self):
        """BaseLoader cannot be instantiated directly. / BaseLoader 不能直接实例化。"""
        with pytest.raises(TypeError, match="abstract"):
            BaseLoader()
    
    def test_validate_file_existing_file(self, tmp_path):
        """_validate_file returns Path for existing files. / _validate_file 对已存在文件返回 Path。"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")
        
        # Access static method via subclass / 通过子类访问静态方法
        validated = PdfLoader._validate_file(test_file)
        assert validated.exists()
        assert validated.is_file()
    
    def test_validate_file_nonexistent(self):
        """_validate_file raises FileNotFoundError for missing files. / _validate_file 对缺失文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            PdfLoader._validate_file("nonexistent_file.pdf")
    
    def test_validate_file_directory(self, tmp_path):
        """_validate_file raises ValueError for directories. / _validate_file 对目录抛出 ValueError。"""
        with pytest.raises(ValueError, match="not a file"):
            PdfLoader._validate_file(tmp_path)


class TestPdfLoaderInitialization:
    """Tests for PdfLoader initialization. / PdfLoader 初始化测试。"""
    
    def test_default_initialization(self):
        """PdfLoader can be initialized with defaults. / PdfLoader 可使用默认值初始化。"""
        loader = PdfLoader()
        assert loader.extract_images is True
        assert loader.image_storage_dir == Path("data/images")
    
    def test_custom_initialization(self):
        """PdfLoader respects custom configuration. / PdfLoader 遵守自定义配置。"""
        loader = PdfLoader(
            extract_images=False,
            image_storage_dir="custom/path"
        )
        assert loader.extract_images is False
        assert loader.image_storage_dir == Path("custom/path")
    
    def test_markitdown_available(self):
        """PdfLoader requires MarkItDown to be available. / PdfLoader 要求 MarkItDown 可用。"""
        loader = PdfLoader()
        assert loader._markitdown is not None


class TestPdfLoaderValidation:
    """Tests for input validation. / 输入校验测试。"""
    
    def test_load_requires_pdf_extension(self, tmp_path):
        """load() raises ValueError for non-PDF files. / load() 对非 PDF 文件抛出 ValueError。"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        
        loader = PdfLoader()
        with pytest.raises(ValueError, match="not a PDF"):
            loader.load(txt_file)
    
    def test_load_nonexistent_file(self):
        """load() raises FileNotFoundError for missing files. / load() 对缺失文件抛出 FileNotFoundError。"""
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.pdf")


class TestPdfLoaderHelperMethods:
    """Tests for helper methods. / 辅助方法测试。"""
    
    def test_compute_file_hash_consistency(self, tmp_path):
        """_compute_file_hash returns consistent hash for same content. / _compute_file_hash 对相同内容返回一致 hash。"""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"consistent content")
        
        loader = PdfLoader()
        hash1 = loader._compute_file_hash(test_file)
        hash2 = loader._compute_file_hash(test_file)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length / SHA256 十六进制长度
    
    def test_compute_file_hash_differs_for_different_content(self, tmp_path):
        """_compute_file_hash returns different hashes for different content. / _compute_file_hash 对不同内容返回不同 hashes。"""
        file1 = tmp_path / "file1.pdf"
        file2 = tmp_path / "file2.pdf"
        file1.write_bytes(b"content 1")
        file2.write_bytes(b"content 2")
        
        loader = PdfLoader()
        hash1 = loader._compute_file_hash(file1)
        hash2 = loader._compute_file_hash(file2)
        
        assert hash1 != hash2
    
    def test_extract_title_from_markdown_heading(self):
        """_extract_title finds first Markdown heading. / _extract_title 查找第一个 Markdown 标题。"""
        loader = PdfLoader()
        
        text = "# Main Title\n\nParagraph text\n\n## Subtitle"
        title = loader._extract_title(text)
        assert title == "Main Title"
    
    def test_extract_title_from_first_line(self):
        """_extract_title uses first non-empty line as fallback. / _extract_title 使用第一个非空行作为 fallback。"""
        loader = PdfLoader()
        
        text = "First Line Title\n\nParagraph text"
        title = loader._extract_title(text)
        assert title == "First Line Title"
    
    def test_extract_title_handles_empty_text(self):
        """_extract_title returns None for empty text. / _extract_title 对空文本返回 None。"""
        loader = PdfLoader()
        
        text = ""
        title = loader._extract_title(text)
        assert title is None
    
    def test_generate_image_id_format(self):
        """_generate_image_id creates consistent ID format. / _generate_image_id 创建一致的 ID 格式。"""
        image_id = PdfLoader._generate_image_id("abc123def456", 2, 0)
        assert image_id == "abc123de_2_0"


class TestPdfConversionCore:
    """Tests for core PDF conversion functionality using real PDF files. / 使用真实 PDF 文件测试核心 PDF 转换功能。"""
    
    def test_convert_simple_pdf_to_text(self):
        """Convert simple PDF to text - verifies core Markdown conversion. / 将简单 PDF 转成文本，验证核心 Markdown 转换。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # Verify Document structure / 验证 Document 结构
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        
        # Verify text content is extracted / 验证文本内容已提取
        assert len(doc.text) > 0
        assert isinstance(doc.text, str)
        
        # Verify expected content is present (from our generated PDF) / 验证预期内容存在（来自我们生成的 PDF）
        text_lower = doc.text.lower()
        assert "sample" in text_lower or "document" in text_lower
        assert "test" in text_lower or "pdf" in text_lower
        
        # Verify metadata / 验证 metadata
        assert doc.metadata["source_path"] == str(SIMPLE_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata
    
    def test_extract_title_from_pdf(self):
        """Verify title extraction from real PDF. / 验证从真实 PDF 提取标题。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # Should extract title (either from heading or first line) / 应提取标题（来自 heading 或第一行）
        assert "title" in doc.metadata
        assert doc.metadata["title"] is not None
        assert len(doc.metadata["title"]) > 0
        
        # Title should contain relevant keywords / 标题应包含相关关键词
        title_lower = doc.metadata["title"].lower()
        assert "sample" in title_lower or "document" in title_lower
    
    def test_pdf_with_images_structure(self):
        """Verify PDF with images is processed correctly with image extraction. / 验证带图片 PDF 在图片提取开启时可正确处理。"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")
        
        loader = PdfLoader(extract_images=True)
        doc = loader.load(IMAGES_PDF)
        
        # Verify basic structure / 验证基础结构
        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        
        # Verify images were extracted / 验证图片已提取
        assert "images" in doc.metadata
        assert isinstance(doc.metadata["images"], list)
        
        if len(doc.metadata["images"]) > 0:
            # Verify image metadata structure / 验证图片 metadata 结构
            for img in doc.metadata["images"]:
                assert "id" in img
                assert "path" in img
                assert "page" in img
                assert "text_offset" in img
                assert "text_length" in img
                assert "position" in img
                
                # Verify image file exists / 验证图片文件存在
                img_path = Path(img["path"])
                assert img_path.exists(), f"Image file should exist: {img_path}"
                
                # Verify placeholder exists in text / 验证 placeholder 存在于文本中
                placeholder = f"[IMAGE: {img['id']}]"
                assert placeholder in doc.text, f"Placeholder {placeholder} should be in text"
    
    def test_image_extraction_disabled(self):
        """Verify image extraction can be disabled. / 验证图片提取可被禁用。"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")
        
        loader = PdfLoader(extract_images=False)
        doc = loader.load(IMAGES_PDF)
        
        # Should still extract text / 仍应提取文本
        assert len(doc.text) > 0
        
        # Should not have images metadata / 不应包含 images metadata
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
    
    def test_document_hash_consistency(self):
        """Verify same PDF produces same document hash. / 验证同一 PDF 产生相同 document hash。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        
        # Load same file twice / 加载同一文件两次
        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(SIMPLE_PDF)
        
        # Should produce identical hashes (for idempotency) / 应产生相同 hashes（用于幂等性）
        assert doc1.metadata["doc_hash"] == doc2.metadata["doc_hash"]
        assert doc1.id == doc2.id
    
    def test_document_serialization(self):
        """Verify loaded document can be serialized. / 验证加载后的 document 可序列化。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # Serialize to dict / 序列化为 dict
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "text" in doc_dict
        assert "metadata" in doc_dict
        
        # Verify metadata is complete / 验证 metadata 完整
        assert "source_path" in doc_dict["metadata"]
        assert "doc_type" in doc_dict["metadata"]
        assert doc_dict["metadata"]["doc_type"] == "pdf"
        
        # Verify can recreate from dict / 验证可从 dict 重建
        doc_recreated = Document.from_dict(doc_dict)
        assert doc_recreated.id == doc.id
        assert doc_recreated.text == doc.text
        assert doc_recreated.metadata == doc.metadata
