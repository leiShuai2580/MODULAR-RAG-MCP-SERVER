"""Unit tests for BM25Indexer. / BM25Indexer 的单元测试。

Tests cover: / 测试覆盖：
- Index building from term statistics / 从词项统计构建索引
- IDF calculation accuracy / IDF 计算准确性
- Query functionality with BM25 scoring / 使用 BM25 打分的查询功能
- Index persistence (save/load roundtrip) / 索引持久化（保存/加载往返）
- Rebuild and incremental update scenarios / 重建与增量更新场景
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from src.ingestion.storage.bm25_indexer import BM25Indexer


class TestBM25IndexerBasics:
    """Test basic BM25Indexer functionality. / 测试 BM25Indexer 基础功能。"""
    
    def test_indexer_initialization_default(self):
        """Test default initialization. / 测试默认初始化。"""
        indexer = BM25Indexer()
        
        assert indexer.k1 == 1.5
        assert indexer.b == 0.75
        assert os.path.normpath(str(indexer.index_dir)) == os.path.normpath("data/db/bm25")
    
    def test_indexer_initialization_custom(self):
        """Test initialization with custom parameters. / 测试使用自定义参数初始化。"""
        indexer = BM25Indexer(
            index_dir="custom/path",
            k1=2.0,
            b=0.5
        )
        
        assert indexer.k1 == 2.0
        assert indexer.b == 0.5
        assert os.path.normpath(str(indexer.index_dir)) == os.path.normpath("custom/path")
    
    def test_indexer_initialization_invalid_k1(self):
        """Test that invalid k1 raises ValueError. / 测试无效 k1 会抛出 ValueError。"""
        with pytest.raises(ValueError, match="k1 must be > 0"):
            BM25Indexer(k1=0)
        
        with pytest.raises(ValueError, match="k1 must be > 0"):
            BM25Indexer(k1=-1.0)
    
    def test_indexer_initialization_invalid_b(self):
        """Test that invalid b raises ValueError. / 测试无效 b 会抛出 ValueError。"""
        with pytest.raises(ValueError, match="b must be in"):
            BM25Indexer(b=-0.1)
        
        with pytest.raises(ValueError, match="b must be in"):
            BM25Indexer(b=1.5)


class TestBM25IndexBuilding:
    """Test index building functionality. / 测试索引构建功能。"""
    
    def test_build_simple_index(self, tmp_path):
        """Test building a simple index from term statistics. / 测试从词项统计构建简单索引。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {
                "chunk_id": "doc1",
                "term_frequencies": {"hello": 2, "world": 1},
                "doc_length": 3
            },
            {
                "chunk_id": "doc2",
                "term_frequencies": {"hello": 1, "python": 1},
                "doc_length": 2
            }
        ]
        
        indexer.build(term_stats, collection="test")
        
        # Check metadata / 检查 metadata
        assert indexer._metadata["num_docs"] == 2
        assert indexer._metadata["avg_doc_length"] == 2.5
        assert indexer._metadata["total_terms"] == 3  # hello, world, python / hello、world、python
        
        # Check index structure / 检查索引结构
        assert "hello" in indexer._index
        assert "world" in indexer._index
        assert "python" in indexer._index
        
        # Check hello posting list (appears in both docs) / 检查 hello 的 posting list（出现在两个文档中）
        hello_data = indexer._index["hello"]
        assert hello_data["df"] == 2
        assert len(hello_data["postings"]) == 2
    
    def test_build_empty_term_stats_raises(self, tmp_path):
        """Test that building with empty term_stats raises ValueError. / 测试使用空 term_stats 构建会抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        with pytest.raises(ValueError, match="Cannot build index from empty"):
            indexer.build([])
    
    def test_build_invalid_structure_raises(self, tmp_path):
        """Test that invalid term_stats structure raises ValueError. / 测试无效 term_stats 结构会抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # Missing chunk_id / 缺失 chunk_id
        with pytest.raises(ValueError, match="missing required field: chunk_id"):
            indexer.build([{"term_frequencies": {}, "doc_length": 0}])
        
        # Missing term_frequencies / 缺失 term_frequencies
        with pytest.raises(ValueError, match="missing required field: term_frequencies"):
            indexer.build([{"chunk_id": "1", "doc_length": 0}])
        
        # Missing doc_length / 缺失 doc_length
        with pytest.raises(ValueError, match="missing required field: doc_length"):
            indexer.build([{"chunk_id": "1", "term_frequencies": {}}])
    
    def test_build_invalid_types_raises(self, tmp_path):
        """Test that invalid field types raise ValueError. / 测试无效字段类型会抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # term_frequencies not a dict / term_frequencies 不是 dict
        with pytest.raises(ValueError, match="term_frequencies.*must be dict"):
            indexer.build([
                {"chunk_id": "1", "term_frequencies": "invalid", "doc_length": 0}
            ])
        
        # doc_length not an int / doc_length 不是 int
        with pytest.raises(ValueError, match="doc_length.*must be non-negative int"):
            indexer.build([
                {"chunk_id": "1", "term_frequencies": {}, "doc_length": "invalid"}
            ])
        
        # Negative doc_length / 负数 doc_length
        with pytest.raises(ValueError, match="doc_length.*must be non-negative int"):
            indexer.build([
                {"chunk_id": "1", "term_frequencies": {}, "doc_length": -1}
            ])


class TestIDFCalculation:
    """Test IDF calculation accuracy. / 测试 IDF 计算准确性。"""
    
    def test_idf_calculation_formula(self, tmp_path):
        """Test that IDF is calculated correctly using BM25 formula. / 测试 IDF 会按 BM25 公式正确计算。
        
        Formula: IDF(term) = log((N - df + 0.5) / (df + 0.5)) / 公式：IDF(term) = log((N - df + 0.5) / (df + 0.5))
        """
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # N=3 docs, df=1 (term appears in 1 doc) / N=3 个文档，df=1（词项出现在 1 个文档中）
        # Expected IDF = log((3 - 1 + 0.5) / (1 + 0.5)) = log(2.5 / 1.5) = log(1.6667) / 期望 IDF = log((3 - 1 + 0.5) / (1 + 0.5)) = log(2.5 / 1.5) = log(1.6667)
        import math
        expected_idf = math.log((3 - 1 + 0.5) / (1 + 0.5))
        
        actual_idf = indexer._calculate_idf(num_docs=3, df=1)
        
        assert abs(actual_idf - expected_idf) < 0.0001
    
    def test_idf_rare_vs_common_terms(self, tmp_path):
        """Test that rare terms have higher IDF than common terms. / 测试稀有词项比常见词项具有更高 IDF。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "1", "term_frequencies": {"rare": 1, "common": 1}, "doc_length": 2},
            {"chunk_id": "2", "term_frequencies": {"common": 1}, "doc_length": 1},
            {"chunk_id": "3", "term_frequencies": {"common": 1}, "doc_length": 1},
        ]
        
        indexer.build(term_stats, collection="test")
        
        # "rare" appears in 1/3 docs, "common" appears in 3/3 docs / "rare" 出现在 1/3 文档，"common" 出现在 3/3 文档
        rare_idf = indexer._index["rare"]["idf"]
        common_idf = indexer._index["common"]["idf"]
        
        # Rare terms should have higher IDF / 稀有词项应具有更高 IDF
        assert rare_idf > common_idf


class TestBM25Querying:
    """Test BM25 query functionality. / 测试 BM25 查询功能。"""
    
    def test_query_returns_sorted_results(self, tmp_path):
        """Test that query returns results sorted by BM25 score. / 测试 query 返回按 BM25 分数排序的结果。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # Use a clearer scenario: one doc with query terms, others without / 使用更清晰的场景：一个文档包含 query terms，其他文档不包含
        term_stats = [
            {"chunk_id": "relevant", "term_frequencies": {"machine": 3, "learning": 2}, "doc_length": 5},
            {"chunk_id": "partial", "term_frequencies": {"machine": 1}, "doc_length": 3},
            {"chunk_id": "irrelevant", "term_frequencies": {"python": 2}, "doc_length": 2},
        ]
        
        indexer.build(term_stats, collection="test")
        
        results = indexer.query(["machine", "learning"], top_k=3)
        
        # Should return 2 results (relevant and partial contain query terms) / 应返回 2 条结果（relevant 和 partial 包含 query terms）
        assert len(results) == 2
        
        # Results should be sorted by score descending / 结果应按分数降序排序
        assert results[0]["score"] >= results[1]["score"]
        
        # "relevant" contains both terms, should rank first / "relevant" 包含两个词项，应排第一
        assert results[0]["chunk_id"] == "relevant"
        # "partial" contains only one term, should rank second / "partial" 只包含一个词项，应排第二
        assert results[1]["chunk_id"] == "partial"
    
    def test_query_respects_top_k(self, tmp_path):
        """Test that query respects top_k parameter. / 测试 query 遵守 top_k 参数。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": f"doc{i}", "term_frequencies": {"test": 1}, "doc_length": 1}
            for i in range(10)
        ]
        
        indexer.build(term_stats, collection="test")
        
        results = indexer.query(["test"], top_k=3)
        
        assert len(results) == 3
    
    def test_query_term_not_in_corpus(self, tmp_path):
        """Test querying for term not in corpus returns empty. / 测试查询语料中不存在的词项会返回空。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"hello": 1}, "doc_length": 1}
        ]
        
        indexer.build(term_stats, collection="test")
        
        results = indexer.query(["nonexistent"], top_k=10)
        
        assert len(results) == 0
    
    def test_query_before_build_raises(self):
        """Test that querying before build raises ValueError. / 测试构建前查询会抛出 ValueError。"""
        indexer = BM25Indexer()
        
        with pytest.raises(ValueError, match="Index not loaded"):
            indexer.query(["test"])
    
    def test_query_empty_terms_raises(self, tmp_path):
        """Test that querying with empty terms raises ValueError. / 测试使用空 terms 查询会抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        indexer.build(term_stats, collection="test")
        
        with pytest.raises(ValueError, match="query_terms cannot be empty"):
            indexer.query([])


class TestIndexPersistence:
    """Test index save/load functionality. / 测试索引保存/加载功能。"""
    
    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saved index can be loaded and produces same results. / 测试保存的索引可加载且产生相同结果。"""
        indexer1 = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"hello": 2, "world": 1}, "doc_length": 3},
            {"chunk_id": "doc2", "term_frequencies": {"hello": 1, "python": 1}, "doc_length": 2}
        ]
        
        indexer1.build(term_stats, collection="test")
        
        # Query with first indexer / 使用第一个 indexer 查询
        results1 = indexer1.query(["hello"], top_k=2)
        
        # Create new indexer and load / 创建新 indexer 并加载
        indexer2 = BM25Indexer(index_dir=str(tmp_path))
        loaded = indexer2.load(collection="test")
        
        assert loaded is True
        
        # Query with loaded indexer / 使用已加载的 indexer 查询
        results2 = indexer2.query(["hello"], top_k=2)
        
        # Results should be identical / 结果应相同
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["chunk_id"] == r2["chunk_id"]
            assert abs(r1["score"] - r2["score"]) < 0.0001
    
    def test_load_nonexistent_index(self, tmp_path):
        """Test loading non-existent index returns False. / 测试加载不存在的索引会返回 False。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        loaded = indexer.load(collection="nonexistent")
        
        assert loaded is False
    
    def test_load_corrupted_index_raises(self, tmp_path):
        """Test loading corrupted index raises ValueError. / 测试加载损坏索引会抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # Create corrupted index file / 创建损坏的索引文件
        index_path = tmp_path / "corrupted_bm25.json"
        index_path.write_text("not valid json{")
        
        with pytest.raises(ValueError, match="Corrupted index file"):
            indexer.load(collection="corrupted")
    
    def test_load_invalid_structure_raises(self, tmp_path):
        """Test loading index with invalid structure raises ValueError. / 测试加载结构无效的索引会抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # Create index file with missing fields / 创建缺失字段的索引文件
        index_path = tmp_path / "invalid_bm25.json"
        index_path.write_text(json.dumps({"metadata": {}}))  # Missing "index" / 缺少 "index"
        
        with pytest.raises(ValueError, match="Invalid index file structure"):
            indexer.load(collection="invalid")
    
    def test_index_file_created_in_correct_location(self, tmp_path):
        """Test that index file is created in the correct directory. / 测试索引文件会创建在正确目录中。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        
        indexer.build(term_stats, collection="my_collection")
        
        expected_path = tmp_path / "my_collection_bm25.json"
        assert expected_path.exists()
        
        # Verify it's valid JSON / 验证它是有效 JSON
        with open(expected_path) as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "index" in data


class TestRebuildFunctionality:
    """Test rebuild and update scenarios. / 测试重建与更新场景。"""
    
    def test_rebuild_replaces_old_index(self, tmp_path):
        """Test that rebuild completely replaces old index. / 测试 rebuild 会完全替换旧索引。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # Build initial index / 构建初始索引
        term_stats1 = [
            {"chunk_id": "doc1", "term_frequencies": {"old": 1}, "doc_length": 1}
        ]
        indexer.build(term_stats1, collection="test")
        
        # Rebuild with new data / 使用新数据重建
        term_stats2 = [
            {"chunk_id": "doc2", "term_frequencies": {"new": 1}, "doc_length": 1}
        ]
        indexer.rebuild(term_stats2, collection="test")
        
        # Old term should not exist / 旧词项不应存在
        results_old = indexer.query(["old"], top_k=10)
        assert len(results_old) == 0
        
        # New term should exist / 新词项应存在
        results_new = indexer.query(["new"], top_k=10)
        assert len(results_new) == 1
        assert results_new[0]["chunk_id"] == "doc2"
    
    def test_rebuild_is_deterministic(self, tmp_path):
        """Test that rebuilding same data produces same index. / 测试重建相同数据会产生相同索引。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"test": 2}, "doc_length": 2},
            {"chunk_id": "doc2", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        
        # Build twice / 构建两次
        indexer.build(term_stats, collection="test1")
        results1 = indexer.query(["test"], top_k=2)
        
        indexer.rebuild(term_stats, collection="test2")
        results2 = indexer.query(["test"], top_k=2)
        
        # Results should be identical / 结果应相同
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["chunk_id"] == r2["chunk_id"]
            assert abs(r1["score"] - r2["score"]) < 0.0001


class TestEdgeCases:
    """Test edge cases and boundary conditions. / 测试边界情况和边界条件。"""
    
    def test_single_document_corpus(self, tmp_path):
        """Test indexing and querying a single-document corpus. / 测试单文档语料的索引和查询。
        
        Note: In a single-document corpus where the term appears in all documents, / 注意：在单文档语料中，当词项出现在所有文档里时，
        IDF can be negative per BM25 formula: log((N - df + 0.5) / (df + 0.5)) / 按 BM25 公式 IDF 可能为负：log((N - df + 0.5) / (df + 0.5))
        When N=1, df=1: log((1-1+0.5)/(1+0.5)) = log(0.5/1.5) = log(0.33) < 0 / 当 N=1、df=1：log((1-1+0.5)/(1+0.5)) = log(0.5/1.5) = log(0.33) < 0
        This is expected behavior for BM25. / 这是 BM25 的预期行为。
        """
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "only_doc", "term_frequencies": {"hello": 1}, "doc_length": 1}
        ]
        
        indexer.build(term_stats, collection="test")
        results = indexer.query(["hello"], top_k=1)
        
        assert len(results) == 1
        assert results[0]["chunk_id"] == "only_doc"
        # Score can be negative for single-document corpus (expected BM25 behavior) / 单文档语料中分数可能为负（BM25 的预期行为）
        assert isinstance(results[0]["score"], float)
    
    def test_empty_document(self, tmp_path):
        """Test handling document with zero terms (edge case). / 测试处理零词项文档（边界情况）。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "empty", "term_frequencies": {}, "doc_length": 0},
            {"chunk_id": "normal", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        
        # Should not raise / 不应抛出异常
        indexer.build(term_stats, collection="test")
        
        # Query should still work / 查询仍应正常工作
        results = indexer.query(["test"], top_k=10)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "normal"
    
    def test_very_long_posting_list(self, tmp_path):
        """Test handling term that appears in many documents. / 测试处理出现在大量文档中的词项。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        # Create 100 docs all containing "common" / 创建 100 个都包含 "common" 的文档
        term_stats = [
            {"chunk_id": f"doc{i}", "term_frequencies": {"common": 1}, "doc_length": 1}
            for i in range(100)
        ]
        
        indexer.build(term_stats, collection="test")
        
        # Should handle large posting list / 应能处理大型 posting list
        assert indexer._index["common"]["df"] == 100
        assert len(indexer._index["common"]["postings"]) == 100
        
        # Query should work / 查询应正常工作
        results = indexer.query(["common"], top_k=10)
        assert len(results) == 10  # Respects top_k / 遵守 top_k
    
    def test_special_characters_in_terms(self, tmp_path):
        """Test handling terms with special characters. / 测试处理包含特殊字符的词项。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {
                "chunk_id": "doc1",
                "term_frequencies": {"hello-world": 1, "test_case": 1},
                "doc_length": 2
            }
        ]
        
        indexer.build(term_stats, collection="test")
        
        # Should be able to query with special characters / 应能使用特殊字符查询
        results = indexer.query(["hello-world"], top_k=1)
        assert len(results) == 1
        
        results = indexer.query(["test_case"], top_k=1)
        assert len(results) == 1
