"""Unit tests for SparseEncoder. / SparseEncoder 的单元测试。

Tests cover: / 测试覆盖：
- Constructor validation / 构造函数校验
- Basic encoding functionality / 基础编码功能
- Tokenization behavior / 分词行为
- Edge cases (empty text, special characters) / 边界情况（空文本、特殊字符）
- Corpus statistics calculation / 语料统计计算
- Deterministic behavior / 确定性行为
"""

import pytest
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.core.types import Chunk


# ============================================================================
# Constructor Tests / 构造函数测试
# ============================================================================

def test_constructor_default():
    """Test default constructor. / 测试默认构造函数。"""
    encoder = SparseEncoder()
    assert encoder.min_term_length == 2
    assert encoder.lowercase is True


def test_constructor_custom_min_term_length():
    """Test custom min_term_length. / 测试自定义 min_term_length。"""
    encoder = SparseEncoder(min_term_length=3)
    assert encoder.min_term_length == 3


def test_constructor_custom_lowercase():
    """Test custom lowercase setting. / 测试自定义 lowercase 设置。"""
    encoder = SparseEncoder(lowercase=False)
    assert encoder.lowercase is False


def test_constructor_rejects_zero_min_term_length():
    """Test that min_term_length=0 is rejected. / 测试 min_term_length=0 会被拒绝。"""
    with pytest.raises(ValueError, match="min_term_length must be >= 1"):
        SparseEncoder(min_term_length=0)


def test_constructor_rejects_negative_min_term_length():
    """Test that negative min_term_length is rejected. / 测试负数 min_term_length 会被拒绝。"""
    with pytest.raises(ValueError, match="min_term_length must be >= 1"):
        SparseEncoder(min_term_length=-1)


# ============================================================================
# Basic Encoding Tests / 基础编码测试
# ============================================================================

def test_encode_single_chunk():
    """Test encoding a single chunk. / 测试编码单个 chunk。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "1"
    assert results[0]["term_frequencies"]["hello"] == 1
    assert results[0]["term_frequencies"]["world"] == 1
    assert results[0]["doc_length"] == 2
    assert results[0]["unique_terms"] == 2


def test_encode_multiple_chunks():
    """Test encoding multiple chunks. / 测试编码多个 chunk。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="deep learning networks", metadata={"source_path": "test.txt"}),
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "1"
    assert results[1]["chunk_id"] == "2"


def test_encode_with_repeated_terms():
    """Test that term frequencies count correctly. / 测试 term frequencies 计数正确。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world hello hello", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert results[0]["term_frequencies"]["hello"] == 3
    assert results[0]["term_frequencies"]["world"] == 1
    assert results[0]["doc_length"] == 4
    assert results[0]["unique_terms"] == 2


# ============================================================================
# Tokenization Tests / 分词测试
# ============================================================================

def test_tokenize_lowercases_by_default():
    """Test that terms are lowercased by default. / 测试默认会将 terms 转为小写。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Hello World HELLO", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]
    assert results[0]["term_frequencies"]["hello"] == 2


def test_tokenize_preserves_case_when_configured():
    """Test that case is preserved when lowercase=False. / 测试 lowercase=False 时保留大小写。"""
    encoder = SparseEncoder(lowercase=False)
    chunks = [
        Chunk(id="1", text="Hello World", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert "Hello" in results[0]["term_frequencies"]
    assert "World" in results[0]["term_frequencies"]
    assert "hello" not in results[0]["term_frequencies"]


def test_tokenize_filters_by_min_term_length():
    """Test that short terms are filtered out. / 测试短 terms 会被过滤。"""
    encoder = SparseEncoder(min_term_length=3)
    chunks = [
        Chunk(id="1", text="I am learning Python AI", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Should filter out "I", "am", and "AI" (length < 3) / 应过滤 "I"、"am" 和 "AI"（长度 < 3）
    assert "learning" in results[0]["term_frequencies"]
    assert "python" in results[0]["term_frequencies"]
    assert results[0]["unique_terms"] == 2  # learning, python / learning、python


def test_tokenize_handles_punctuation():
    """Test that punctuation is handled correctly. / 测试标点符号处理正确。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Hello, world! How are you?", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Punctuation should be removed / 标点符号应被移除
    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]
    assert "how" in results[0]["term_frequencies"]
    assert "," not in results[0]["term_frequencies"]
    assert "!" not in results[0]["term_frequencies"]


def test_tokenize_handles_hyphens_and_underscores():
    """Test that hyphens and underscores are preserved. / 测试连字符和下划线会被保留。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine-learning deep_learning", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert "machine-learning" in results[0]["term_frequencies"]
    assert "deep_learning" in results[0]["term_frequencies"]


def test_tokenize_handles_numbers():
    """Test that numbers are tokenized. / 测试数字会被分词。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Python 3.11 and GPT-4", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Numbers should be preserved as alphanumeric tokens / 数字应作为字母数字 token 保留
    assert "python" in results[0]["term_frequencies"]
    # "3.11" may be split into "3" and "11" depending on tokenizer / "3.11" 可能根据 tokenizer 被拆为 "3" 和 "11"
    # "gpt-4" should be preserved as hyphenated term / "gpt-4" 应作为连字符 term 保留
    assert "gpt-4" in results[0]["term_frequencies"]


# ============================================================================
# Edge Cases / 边界情况
# ============================================================================

def test_encode_rejects_empty_chunks_list():
    """Test that empty chunks list is rejected. / 测试空 chunks 列表会被拒绝。"""
    encoder = SparseEncoder()
    
    with pytest.raises(ValueError, match="Cannot encode empty chunks list"):
        encoder.encode([])


def test_encode_rejects_chunk_with_empty_text():
    """Test that chunk with empty text is rejected. / 测试空文本 chunk 会被拒绝。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="", metadata={"source_path": "test.txt"})
    ]
    
    with pytest.raises(ValueError, match="empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_rejects_chunk_with_whitespace_only_text():
    """Test that chunk with whitespace-only text is rejected. / 测试仅空白文本的 chunk 会被拒绝。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="   \n\t  ", metadata={"source_path": "test.txt"})
    ]
    
    with pytest.raises(ValueError, match="empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_handles_special_characters():
    """Test encoding text with special characters. / 测试编码包含特殊字符的文本。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="C++ and C# programming @2024", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Should extract alphanumeric terms / 应提取字母数字 terms
    assert "programming" in results[0]["term_frequencies"]
    assert "2024" in results[0]["term_frequencies"]


def test_encode_handles_unicode():
    """Test encoding text with unicode characters. / 测试编码包含 unicode 字符的文本。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="café résumé naïve", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Unicode characters should be handled / Unicode 字符应被处理
    assert results[0]["doc_length"] > 0
    assert results[0]["unique_terms"] > 0


# ============================================================================
# Determinism Tests / 确定性测试
# ============================================================================

def test_encode_is_deterministic():
    """Test that encoding is deterministic. / 测试编码是确定性的。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning deep learning", metadata={"source_path": "test.txt"})
    ]
    
    results1 = encoder.encode(chunks)
    results2 = encoder.encode(chunks)
    
    assert results1 == results2


def test_encode_preserves_chunk_order():
    """Test that output order matches input order. / 测试输出顺序匹配输入顺序。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="first chunk", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="second chunk", metadata={"source_path": "test.txt"}),
        Chunk(id="3", text="third chunk", metadata={"source_path": "test.txt"}),
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 3
    assert results[0]["chunk_id"] == "1"
    assert results[1]["chunk_id"] == "2"
    assert results[2]["chunk_id"] == "3"


# ============================================================================
# Corpus Statistics Tests / 语料统计测试
# ============================================================================

def test_get_corpus_stats_single_document():
    """Test corpus stats for single document. / 测试单文档的语料统计。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world", metadata={"source_path": "test.txt"})
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    assert stats["num_docs"] == 1
    assert stats["avg_doc_length"] == 2.0
    assert stats["document_frequency"]["hello"] == 1
    assert stats["document_frequency"]["world"] == 1


def test_get_corpus_stats_multiple_documents():
    """Test corpus stats for multiple documents. / 测试多文档的语料统计。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="deep learning networks", metadata={"source_path": "test.txt"}),
        Chunk(id="3", text="machine learning algorithms", metadata={"source_path": "test.txt"}),
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    assert stats["num_docs"] == 3
    assert stats["avg_doc_length"] == (2 + 3 + 3) / 3
    # "learning" appears in all 3 docs / "learning" 出现在全部 3 个文档中
    assert stats["document_frequency"]["learning"] == 3
    # "machine" appears in 2 docs / "machine" 出现在 2 个文档中
    assert stats["document_frequency"]["machine"] == 2
    # "deep" appears in 1 doc / "deep" 出现在 1 个文档中
    assert stats["document_frequency"]["deep"] == 1


def test_get_corpus_stats_calculates_average_doc_length():
    """Test that average document length is calculated correctly. / 测试平均文档长度计算正确。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="short", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="this is a longer document", metadata={"source_path": "test.txt"}),
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    # First doc: 1 term ("short"), Second doc: 4 terms ("this", "is", "longer", "document" - "a" filtered), avg = 2.5 / 第一个文档：1 个 term（"short"），第二个文档：4 个 term（"this"、"is"、"longer"、"document"，"a" 被过滤），平均值 = 2.5
    assert stats["avg_doc_length"] == 2.5


def test_get_corpus_stats_handles_empty_list():
    """Test corpus stats with empty encoded chunks list. / 测试空 encoded chunks 列表的语料统计。"""
    encoder = SparseEncoder()
    
    stats = encoder.get_corpus_stats([])
    
    assert stats["num_docs"] == 0
    assert stats["avg_doc_length"] == 0.0
    assert stats["document_frequency"] == {}


# ============================================================================
# Integration Test / 集成测试
# ============================================================================

def test_realistic_encoding_scenario():
    """Test realistic encoding scenario with varied content. / 测试包含多样内容的真实风格编码场景。"""
    encoder = SparseEncoder()
    
    chunks = [
        Chunk(
            id="doc1_chunk0",
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source_path": "textbook.pdf"}
        ),
        Chunk(
            id="doc1_chunk1",
            text="Deep learning uses neural networks with multiple layers.",
            metadata={"source_path": "textbook.pdf"}
        ),
        Chunk(
            id="doc2_chunk0",
            text="Natural language processing (NLP) enables machines to understand text.",
            metadata={"source_path": "paper.pdf"}
        ),
    ]
    
    results = encoder.encode(chunks)
    
    # Validate structure / 验证结构
    assert len(results) == 3
    for i, result in enumerate(results):
        assert "chunk_id" in result
        assert "term_frequencies" in result
        assert "doc_length" in result
        assert "unique_terms" in result
        assert result["chunk_id"] == chunks[i].id
        assert result["doc_length"] > 0
        assert result["unique_terms"] > 0
        assert len(result["term_frequencies"]) == result["unique_terms"]
    
    # Get corpus stats / 获取语料统计
    corpus_stats = encoder.get_corpus_stats(results)
    assert corpus_stats["num_docs"] == 3
    assert corpus_stats["avg_doc_length"] > 0
    assert len(corpus_stats["document_frequency"]) > 0

