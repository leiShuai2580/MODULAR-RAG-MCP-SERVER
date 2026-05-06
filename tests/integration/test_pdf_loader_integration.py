"""Integration tests for PDF Loader using real PDF files. / 使用真实 PDF 文件的 PDF Loader 集成测试。

These tests use actual PDF files from fixtures/sample_documents to verify / 这些测试使用 fixtures/sample_documents 中的真实 PDF 文件，
end-to-end functionality of the PDF loader. / 验证 PDF loader 的端到端功能。
"""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.pdf_loader import PdfLoader


# Fixture paths / Fixture 路径
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


class TestPdfLoaderWithRealFiles:
    """Integration tests using real PDF files. / 使用真实 PDF 文件的集成测试。"""
    
    def test_simple_pdf_exists(self):
        """Verify test fixture exists. / 验证测试 fixture 存在。"""
        assert SIMPLE_PDF.exists(), f"Test fixture not found: {SIMPLE_PDF}"
    
    def test_images_pdf_exists(self):
        """Verify test fixture with images exists. / 验证带图片的测试 fixture 存在。"""
        assert IMAGES_PDF.exists(), f"Test fixture not found: {IMAGES_PDF}"
    
    def test_load_simple_pdf(self):
        """Load a simple text-only PDF and verify Document structure. / 加载简单纯文本 PDF 并验证 Document 结构。"""
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # Verify Document structure / 验证 Document 结构
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        assert len(doc.text) > 0
        
        # Verify required metadata / 验证必需元数据
        assert doc.metadata["source_path"] == str(SIMPLE_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata
        
        # Verify title extraction (the PDF has "Sample Document" as title) / 验证标题提取（该 PDF 标题为 "Sample Document"）
        assert "title" in doc.metadata
        assert len(doc.metadata["title"]) > 0
        assert "sample" in doc.metadata["title"].lower() or "document" in doc.metadata["title"].lower()
        
        # Verify text content contains expected keywords / 验证文本内容包含预期关键词
        assert "sample" in doc.text.lower() or "test" in doc.text.lower()
    
    def test_load_pdf_with_images(self):
        """Load a PDF with images and verify Document structure. / 加载带图片的 PDF 并验证 Document 结构。"""
        loader = PdfLoader(extract_images=True)
        doc = loader.load(IMAGES_PDF)
        
        # Verify Document structure / 验证 Document 结构
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        assert len(doc.text) > 0
        
        # Verify required metadata / 验证必需元数据
        assert doc.metadata["source_path"] == str(IMAGES_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata
        
        # Note: Current implementation returns empty images list (stub) / 注意：当前实现返回空图片列表（stub）
        # Full image extraction will be implemented later / 完整图片提取将在之后实现
        if "images" in doc.metadata:
            assert isinstance(doc.metadata["images"], list)
    
    def test_load_simple_pdf_without_image_extraction(self):
        """Load PDF with image extraction disabled. / 在禁用图片提取时加载 PDF。"""
        loader = PdfLoader(extract_images=False)
        doc = loader.load(SIMPLE_PDF)
        
        assert isinstance(doc, Document)
        assert doc.metadata["doc_type"] == "pdf"
        # Should not have images metadata when extraction is disabled / 禁用提取时不应包含 images 元数据
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
    
    def test_document_is_serializable(self):
        """Verify loaded Document can be serialized to dict/JSON. / 验证加载的 Document 可序列化为 dict/JSON。"""
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "text" in doc_dict
        assert "metadata" in doc_dict
        
        # Verify can recreate from dict / 验证可从字典重建
        doc_recreated = Document.from_dict(doc_dict)
        assert doc_recreated.id == doc.id
        assert doc_recreated.text == doc.text
    
    def test_file_hash_consistency(self):
        """Verify same file produces same hash. / 验证同一文件产生相同哈希。"""
        loader = PdfLoader()
        
        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(SIMPLE_PDF)
        
        # Same file should produce same doc_hash / 同一文件应产生相同 doc_hash
        assert doc1.metadata["doc_hash"] == doc2.metadata["doc_hash"]
    
    def test_different_files_different_hash(self):
        """Verify different files produce different hashes. / 验证不同文件产生不同哈希。"""
        loader = PdfLoader()
        
        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(IMAGES_PDF)
        
        # Different files should produce different hashes / 不同文件应产生不同哈希
        assert doc1.metadata["doc_hash"] != doc2.metadata["doc_hash"]
    
    def test_custom_image_storage_dir(self):
        """Verify custom image storage directory is respected. / 验证自定义图片存储目录会被遵守。"""
        custom_dir = "custom/images"
        loader = PdfLoader(extract_images=True, image_storage_dir=custom_dir)
        
        assert loader.image_storage_dir == Path(custom_dir)
