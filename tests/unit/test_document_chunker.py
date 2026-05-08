"""Unit tests for DocumentChunker - Document to Chunk adapter. / DocumentChunker（Document 到 Chunk 适配器）的单元测试。

This test suite validates the DocumentChunker's five core value-add features: / 本测试套件验证 DocumentChunker 的五个核心增值特性：
1. Unique and deterministic chunk ID generation / 唯一且确定性的 chunk ID 生成
2. Complete metadata inheritance from Document to Chunk / 从 Document 到 Chunk 的完整 metadata 继承
3. chunk_index tracking for sequential position / 用 chunk_index 跟踪顺序位置
4. source_ref establishing parent-child traceability / 用 source_ref 建立父子可追溯关系
5. Type contract compliance (core.types.Chunk) / 类型契约合规性（core.types.Chunk）

All tests use FakeSplitter to isolate DocumentChunker's business logic from / 所有测试使用 FakeSplitter 将 DocumentChunker 的业务逻辑与
the underlying text splitting implementation, ensuring no external dependencies. / 底层文本切分实现隔离，确保没有外部依赖。
"""

import pytest
from unittest.mock import Mock

from src.core.types import Document, Chunk
from src.core.settings import Settings
from src.ingestion.chunking import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """Fake splitter for testing - returns predictable chunks. / 用于测试的 fake splitter，会返回可预测 chunks。
    
    This fake implementation allows us to test DocumentChunker's business logic / 该 fake 实现允许我们测试 DocumentChunker 的业务逻辑，
    without depending on real text splitting algorithms or external libraries. / 而不依赖真实文本切分算法或外部库。
    """
    
    def __init__(self, chunk_size: int = 100, overlap: int = 0, **kwargs):
        """Initialize with ignored parameters for compatibility. / 使用被忽略的参数初始化以保持兼容性。"""
        pass
    
    def split_text(self, text: str) -> list[str]:
        """Split text by double newlines (paragraph-based splitting). / 按双换行切分文本（基于段落切分）。"""
        # Simple splitting for testing: split on double newlines / 测试用简单切分：按双换行切分
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


@pytest.fixture
def fake_settings():
    """Fixture providing minimal settings for testing. / 提供测试所需最小 settings 的 fixture。"""
    settings = Mock(spec=Settings)
    settings.splitter = Mock()
    settings.splitter.provider = "fake"
    settings.splitter.chunk_size = 100
    settings.splitter.overlap = 0
    return settings


@pytest.fixture
def chunker(fake_settings, monkeypatch):
    """Fixture providing DocumentChunker with FakeSplitter injected. / 提供注入 FakeSplitter 的 DocumentChunker fixture。"""
    # Monkey-patch SplitterFactory to return our FakeSplitter / monkey-patch SplitterFactory，使其返回我们的 FakeSplitter
    from src.libs.splitter import splitter_factory
    
    original_create = splitter_factory.SplitterFactory.create
    
    def mock_create(settings):
        return FakeSplitter()
    
    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create)
    
    return DocumentChunker(fake_settings)


@pytest.fixture
def sample_document():
    """Fixture providing a sample document for testing. / 提供测试用示例 document 的 fixture。"""
    return Document(
        id="doc_sample_001",
        text="First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content.",
        metadata={
            "source_path": "data/documents/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample Document",
            "page_count": 3
        }
    )


# =============================================================================
# Test 1: Chunk ID Generation - Uniqueness and Determinism / 测试 1：Chunk ID 生成 - 唯一性和确定性
# =============================================================================

def test_chunk_ids_are_unique(chunker, sample_document):
    """Test that each chunk gets a unique ID. / 测试每个 chunk 都获得唯一 ID。"""
    chunks = chunker.split_document(sample_document)
    
    # Extract all chunk IDs / 提取所有 chunk IDs
    chunk_ids = [chunk.id for chunk in chunks]
    
    # Verify uniqueness: no duplicates / 验证唯一性：没有重复
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"


def test_chunk_ids_are_deterministic(chunker, sample_document):
    """Test that splitting the same document twice produces identical IDs. / 测试同一 document 切分两次会产生相同 IDs。"""
    # Split twice / 切分两次
    chunks_first = chunker.split_document(sample_document)
    chunks_second = chunker.split_document(sample_document)
    
    # Extract IDs / 提取 IDs
    ids_first = [c.id for c in chunks_first]
    ids_second = [c.id for c in chunks_second]
    
    # Verify determinism: IDs match across runs / 验证确定性：多次运行的 IDs 一致
    assert ids_first == ids_second, "Chunk IDs must be deterministic"


def test_chunk_id_format(chunker, sample_document):
    """Test that chunk IDs follow expected format: {doc_id}_{index:04d}_{hash}. / 测试 chunk IDs 符合预期格式：{doc_id}_{index:04d}_{hash}。"""
    chunks = chunker.split_document(sample_document)
    
    for i, chunk in enumerate(chunks):
        # Expected format: doc_sample_001_0000_{hash}, doc_sample_001_0001_{hash}, etc. / 预期格式：doc_sample_001_0000_{hash}、doc_sample_001_0001_{hash} 等
        assert chunk.id.startswith(f"doc_sample_001_{i:04d}_"), \
            f"Chunk ID should start with 'doc_sample_001_{i:04d}_', got: {chunk.id}"
        
        # Hash portion should be 8 characters / hash 部分应为 8 个字符
        hash_part = chunk.id.split("_")[-1]
        assert len(hash_part) == 8, f"Hash should be 8 chars, got: {hash_part}"


def test_chunk_id_changes_with_content(chunker):
    """Test that chunk ID changes when content changes. / 测试内容变化时 chunk ID 会变化。"""
    doc1 = Document(
        id="doc_001",
        text="Content A",
        metadata={"source_path": "file.pdf"}
    )
    doc2 = Document(
        id="doc_001",  # Same doc_id / 相同 doc_id
        text="Content B",  # Different content / 不同内容
        metadata={"source_path": "file.pdf"}
    )
    
    chunks1 = chunker.split_document(doc1)
    chunks2 = chunker.split_document(doc2)
    
    # IDs should differ due to content hash / 由于内容 hash 不同，IDs 应不同
    assert chunks1[0].id != chunks2[0].id, \
        "Chunk ID should change when content changes"


# =============================================================================
# Test 2: Metadata Inheritance - Complete Propagation / 测试 2：Metadata 继承 - 完整传播
# =============================================================================

def test_metadata_inheritance(chunker, sample_document):
    """Test that all document metadata is inherited by chunks. / 测试所有 document metadata 都会被 chunks 继承。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        # All document metadata should be present / 所有 document metadata 都应存在
        assert chunk.metadata["source_path"] == "data/documents/sample.pdf"
        assert chunk.metadata["doc_type"] == "pdf"
        assert chunk.metadata["title"] == "Sample Document"
        assert chunk.metadata["page_count"] == 3


def test_metadata_independence(chunker, sample_document):
    """Test that each chunk gets its own metadata dict (not shared reference). / 测试每个 chunk 都获得独立 metadata dict（非共享引用）。"""
    chunks = chunker.split_document(sample_document)
    
    # Modify first chunk's metadata / 修改第一个 chunk 的 metadata
    chunks[0].metadata["custom_field"] = "test_value"
    
    # Other chunks should not be affected / 其他 chunks 不应受影响
    assert "custom_field" not in chunks[1].metadata, \
        "Chunks should have independent metadata dicts"


def test_metadata_with_empty_document_metadata(chunker):
    """Test chunking when document has minimal metadata. / 测试 document 只有最小 metadata 时的 chunking。"""
    doc = Document(
        id="doc_minimal",
        text="Paragraph 1.\n\nParagraph 2.",
        metadata={"source_path": "minimal.txt"}  # Only required field / 只有必填字段
    )
    
    chunks = chunker.split_document(doc)
    
    # Should still work with minimal metadata / 使用最小 metadata 仍应正常工作
    assert len(chunks) == 2
    assert chunks[0].metadata["source_path"] == "minimal.txt"


# =============================================================================
# Test 3: chunk_index - Sequential Position Tracking / 测试 3：chunk_index - 顺序位置跟踪
# =============================================================================

def test_chunk_index_sequential(chunker, sample_document):
    """Test that chunk_index starts at 0 and increments sequentially. / 测试 chunk_index 从 0 开始并按顺序递增。"""
    chunks = chunker.split_document(sample_document)
    
    # Verify sequential indices: 0, 1, 2, ... / 验证顺序索引：0、1、2、...
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i, \
            f"Chunk at position {i} should have chunk_index={i}"


def test_chunk_index_added_to_all_chunks(chunker, sample_document):
    """Test that every chunk has chunk_index field. / 测试每个 chunk 都有 chunk_index 字段。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata, \
            "All chunks must have chunk_index field"
        assert isinstance(chunk.metadata["chunk_index"], int), \
            "chunk_index must be an integer"


# =============================================================================
# Test 4: source_ref - Parent-Child Traceability / 测试 4：source_ref - 父子可追溯性
# =============================================================================

def test_source_ref_points_to_document(chunker, sample_document):
    """Test that source_ref correctly references parent document ID. / 测试 source_ref 正确引用父 document ID。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        assert chunk.metadata["source_ref"] == sample_document.id, \
            f"source_ref should point to document ID '{sample_document.id}'"


def test_source_ref_added_to_all_chunks(chunker, sample_document):
    """Test that every chunk has source_ref field. / 测试每个 chunk 都有 source_ref 字段。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        assert "source_ref" in chunk.metadata, \
            "All chunks must have source_ref field"


# =============================================================================
# Test 5: Type Contract - core.types.Chunk Compliance / 测试 5：类型契约 - 符合 core.types.Chunk
# =============================================================================

def test_chunks_are_chunk_type(chunker, sample_document):
    """Test that output items are Chunk objects. / 测试输出项都是 Chunk 对象。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        assert isinstance(chunk, Chunk), \
            f"Output should be Chunk objects, got: {type(chunk)}"


def test_chunk_serialization(chunker, sample_document):
    """Test that chunks can be serialized to dict. / 测试 chunks 可序列化为 dict。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        chunk_dict = chunk.to_dict()
        
        # Verify dict structure / 验证 dict 结构
        assert "id" in chunk_dict
        assert "text" in chunk_dict
        assert "metadata" in chunk_dict
        assert isinstance(chunk_dict["metadata"], dict)


def test_chunk_fields_complete(chunker, sample_document):
    """Test that chunks have all required Chunk fields. / 测试 chunks 包含所有必需 Chunk 字段。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        # Required Chunk fields / 必需 Chunk 字段
        assert hasattr(chunk, "id") and chunk.id
        assert hasattr(chunk, "text") and chunk.text
        assert hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict)


# =============================================================================
# Test 6: Configuration-Driven Behavior / 测试 6：配置驱动行为
# =============================================================================

def test_different_splitter_config_produces_different_chunks(fake_settings, monkeypatch):
    """Test that changing splitter config affects chunk output. / 测试修改 splitter 配置会影响 chunk 输出。"""
    # This test verifies that DocumentChunker respects splitter configuration / 本测试验证 DocumentChunker 遵守 splitter 配置
    
    from src.libs.splitter import splitter_factory
    
    # Create two different fake splitters with different behaviors / 创建两个具有不同行为的 fake splitters
    class SmallChunkSplitter(BaseSplitter):
        def split_text(self, text: str) -> list[str]:
            # Split by sentence (period) / 按句子（句号）切分
            return [s.strip() + "." for s in text.split(".") if s.strip()]
    
    class LargeChunkSplitter(BaseSplitter):
        def split_text(self, text: str) -> list[str]:
            # Return entire text as one chunk / 将整段文本作为一个 chunk 返回
            return [text]
    
    document = Document(
        id="doc_test",
        text="First sentence. Second sentence. Third sentence.",
        metadata={"source_path": "test.txt"}
    )
    
    # Test with small chunks / 测试小 chunks
    def mock_create_small(settings):
        return SmallChunkSplitter()
    
    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create_small)
    chunker_small = DocumentChunker(fake_settings)
    chunks_small = chunker_small.split_document(document)
    
    # Test with large chunks / 测试大 chunks
    def mock_create_large(settings):
        return LargeChunkSplitter()
    
    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create_large)
    chunker_large = DocumentChunker(fake_settings)
    chunks_large = chunker_large.split_document(document)
    
    # Different configs should produce different number of chunks / 不同配置应产生不同数量的 chunks
    assert len(chunks_small) != len(chunks_large), \
        "Different splitter configs should produce different chunk counts"


# =============================================================================
# Test 7: Edge Cases and Error Handling / 测试 7：边界情况与错误处理
# =============================================================================

def test_empty_document_raises_error(chunker):
    """Test that empty document raises clear error. / 测试空 document 会抛出清晰错误。"""
    doc = Document(
        id="doc_empty",
        text="",
        metadata={"source_path": "empty.txt"}
    )
    
    with pytest.raises(ValueError, match="has no text content"):
        chunker.split_document(doc)


def test_whitespace_only_document_raises_error(chunker):
    """Test that whitespace-only document raises error. / 测试仅空白 document 会抛出错误。"""
    doc = Document(
        id="doc_whitespace",
        text="   \n\n   \t  ",
        metadata={"source_path": "whitespace.txt"}
    )
    
    with pytest.raises(ValueError, match="has no text content"):
        chunker.split_document(doc)


def test_splitter_returns_empty_list_raises_error(chunker, monkeypatch):
    """Test that if splitter returns no chunks, a clear error is raised. / 测试 splitter 未返回 chunks 时会抛出清晰错误。"""
    from src.libs.splitter import splitter_factory
    
    class EmptySplitter(BaseSplitter):
        def split_text(self, text: str) -> list[str]:
            return []  # Return no chunks / 不返回 chunks
    
    def mock_create(settings):
        return EmptySplitter()
    
    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create)
    
    doc = Document(
        id="doc_test",
        text="Some content",
        metadata={"source_path": "test.txt"}
    )
    
    chunker_empty = DocumentChunker(chunker._settings)
    
    with pytest.raises(ValueError, match="Splitter returned no chunks"):
        chunker_empty.split_document(doc)


# =============================================================================
# Test 8: Integration Smoke Test / 测试 8：集成冒烟测试
# =============================================================================

def test_end_to_end_smoke(chunker, sample_document):
    """Smoke test verifying complete end-to-end chunking flow. / 验证完整端到端 chunking 流程的冒烟测试。"""
    # Execute full split / 执行完整切分
    chunks = chunker.split_document(sample_document)
    
    # Basic sanity checks / 基础合理性检查
    assert len(chunks) > 0, "Should produce at least one chunk"
    
    # Verify each chunk passes all requirements / 验证每个 chunk 都满足所有要求
    for i, chunk in enumerate(chunks):
        # ID requirements / ID 要求
        assert chunk.id
        assert chunk.id.startswith(sample_document.id)
        
        # Text requirements / 文本要求
        assert chunk.text
        assert chunk.text.strip()
        
        # Metadata requirements / Metadata 要求
        assert chunk.metadata["source_path"] == sample_document.metadata["source_path"]
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source_ref"] == sample_document.id
        
        # Type requirements / 类型要求
        assert isinstance(chunk, Chunk)
