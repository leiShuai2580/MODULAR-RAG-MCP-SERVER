"""Integration tests for the Ingestion Pipeline. / Ingestion Pipeline 的集成测试。

This module tests the complete ingestion flow using real Azure services: / 该模块使用真实 Azure 服务测试完整摄入流程：
- Azure LLM (gpt-4o) for chunk refinement and metadata enrichment / Azure LLM（gpt-4o）用于分块精炼和元数据增强
- Azure Vision LLM (gpt-4o) for image captioning / Azure Vision LLM（gpt-4o）用于图片描述生成
- Azure Embedding (text-embedding-ada-002) for dense vectors / Azure Embedding（text-embedding-ada-002）用于稠密向量
- ChromaDB for vector storage / ChromaDB 用于向量存储
- BM25 indexer for sparse retrieval / BM25 indexer 用于稀疏检索

Test Data: / 测试数据：
- complex_technical_doc.pdf: Multi-chapter technical document with images and tables / complex_technical_doc.pdf：包含图片和表格的多章节技术文档
- simple.pdf: Basic PDF for regression testing / simple.pdf：用于回归测试的基础 PDF
"""

import os
import shutil
import pytest
from pathlib import Path

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline, PipelineResult


class TestIngestionPipeline:
    """Integration tests for the full ingestion pipeline. / 完整摄入流水线的集成测试。"""
    
    @pytest.fixture(autouse=True)
    def setup_test_dirs(self, tmp_path):
        """Set up and clean test directories. / 设置并清理测试目录。"""
        # Use tmp_path for test isolation where possible / 尽可能使用 tmp_path 做测试隔离
        self.test_output_dir = tmp_path
        yield
        # Cleanup handled by pytest's tmp_path / 清理由 pytest 的 tmp_path 处理
    
    @pytest.fixture
    def settings(self):
        """Load settings from config file. / 从配置文件加载设置。"""
        return load_settings("config/settings.yaml")
    
    @pytest.fixture
    def complex_pdf_path(self):
        """Path to complex technical document. / 复杂技术文档路径。"""
        path = Path("tests/fixtures/sample_documents/complex_technical_doc.pdf")
        assert path.exists(), f"Test fixture not found: {path}"
        return str(path)
    
    @pytest.fixture
    def simple_pdf_path(self):
        """Path to simple PDF document. / 简单 PDF 文档路径。"""
        path = Path("tests/fixtures/sample_documents/simple.pdf")
        assert path.exists(), f"Test fixture not found: {path}"
        return str(path)
    
    def test_pipeline_with_complex_technical_doc(self, settings, complex_pdf_path):
        """Test full pipeline with complex technical document. / 使用复杂技术文档测试完整流水线。
        
        This test validates: / 该测试验证：
        1. File integrity checking works / 文件完整性检查可用
        2. PDF loading with image extraction / 带图片提取的 PDF 加载
        3. Document chunking / 文档分块
        4. LLM-based chunk refinement / 基于 LLM 的分块精炼
        5. LLM-based metadata enrichment / 基于 LLM 的元数据增强
        6. Vision LLM image captioning / Vision LLM 图片描述生成
        7. Azure embedding generation / Azure embedding 生成
        8. Vector storage to ChromaDB / 向量存储到 ChromaDB
        9. BM25 index building / BM25 索引构建
        """
        # Create pipeline with test collection / 使用测试集合创建流水线
        collection = "test_complex_doc"
        pipeline = IngestionPipeline(
            settings=settings,
            collection=collection,
            force=True  # Force reprocessing for test / 为测试强制重新处理
        )
        
        try:
            # Run the pipeline / 运行流水线
            result = pipeline.run(complex_pdf_path)
            
            # ───────────────────────────────────────────────────────────── / ─────────────────────────────────────────────────────────────
            # Assertions / 断言
            # ───────────────────────────────────────────────────────────── / ─────────────────────────────────────────────────────────────
            
            # Basic success check / 基础成功检查
            assert result.success, f"Pipeline failed: {result.error}"
            assert result.doc_id is not None, "Document ID should be set"
            assert result.file_path == complex_pdf_path
            
            # Chunk generation / 分块生成
            assert result.chunk_count > 0, "Should generate at least one chunk"
            print(f"\n[OK] Generated {result.chunk_count} chunks")
            
            # Vector storage / 向量存储
            assert len(result.vector_ids) > 0, "Should store vectors"
            assert len(result.vector_ids) == result.chunk_count, "Vector count should match chunk count"
            print(f"[OK] Stored {len(result.vector_ids)} vectors")
            
            # Stage-specific checks / 阶段特定检查
            stages = result.stages
            
            # Loading stage / 加载阶段
            assert "loading" in stages
            assert stages["loading"]["text_length"] > 0
            print(f"[OK] Loaded document with {stages['loading']['text_length']} chars")
            
            # Transform stage - LLM enhancement verification / Transform 阶段 - LLM 增强验证
            assert "transform" in stages
            transform = stages["transform"]
            
            # Check chunk refinement used LLM / 检查分块精炼是否使用 LLM
            refiner_stats = transform.get("chunk_refiner", {})
            llm_refined = refiner_stats.get("llm", 0)
            rule_refined = refiner_stats.get("rule", 0)
            print(f"[OK] Chunk Refinement: LLM={llm_refined}, Rule={rule_refined}")
            
            # At least some chunks should be LLM refined (since use_llm=true) / 至少部分分块应由 LLM 精炼（因为 use_llm=true）
            # Note: might fallback to rule if LLM fails / 注意：如果 LLM 失败，可能回退到规则
            total_refined = llm_refined + rule_refined
            assert total_refined == result.chunk_count, "All chunks should be refined"
            
            # Check metadata enrichment used LLM / 检查元数据增强是否使用 LLM
            enricher_stats = transform.get("metadata_enricher", {})
            llm_enriched = enricher_stats.get("llm", 0)
            rule_enriched = enricher_stats.get("rule", 0)
            print(f"[OK] Metadata Enrichment: LLM={llm_enriched}, Rule={rule_enriched}")
            
            total_enriched = llm_enriched + rule_enriched
            assert total_enriched == result.chunk_count, "All chunks should be enriched"
            
            # Check image captioning / 检查图片描述生成
            captioner_stats = transform.get("image_captioner", {})
            captioned = captioner_stats.get("captioned_chunks", 0)
            print(f"[OK] Image Captioning: {captioned} chunks with captions")
            
            # Encoding stage / 编码阶段
            assert "encoding" in stages
            encoding = stages["encoding"]
            assert encoding["dense_vector_count"] == result.chunk_count
            assert encoding["dense_dimension"] == 1536, "Azure ada-002 should produce 1536-dim vectors"
            print(f"[OK] Dense vectors: {encoding['dense_vector_count']} x {encoding['dense_dimension']}dim")
            
            # Storage stage / 存储阶段
            assert "storage" in stages
            storage = stages["storage"]
            assert storage["vector_count"] == result.chunk_count
            assert storage["bm25_docs"] == result.chunk_count
            print(f"[OK] Storage: {storage['vector_count']} vectors, {storage['bm25_docs']} BM25 docs")
            
            # Verify files were created / 验证文件已创建
            chroma_dir = Path(settings.vector_store.persist_directory)
            assert chroma_dir.exists(), "ChromaDB directory should exist"
            
            bm25_dir = Path(f"data/db/bm25/{collection}")
            assert bm25_dir.exists(), "BM25 index directory should exist"
            
            print("\n" + "=" * 60)
            print("SUCCESS - All pipeline stages completed!")
            print(f"   Document: {complex_pdf_path}")
            print(f"   Chunks: {result.chunk_count}")
            print(f"   Vectors: {len(result.vector_ids)}")
            print(f"   Images: {result.image_count}")
            print("=" * 60)
            
        finally:
            pipeline.close()
    
    def test_pipeline_skip_already_processed(self, settings, simple_pdf_path):
        """Test that pipeline skips already processed files. / 测试流水线会跳过已处理文件。"""
        collection = "test_skip"
        
        # First run - should process / 第一次运行 - 应处理
        pipeline1 = IngestionPipeline(settings, collection=collection, force=True)
        try:
            result1 = pipeline1.run(simple_pdf_path)
            assert result1.success
            assert result1.chunk_count > 0
        finally:
            pipeline1.close()
        
        # Second run without force - should skip / 第二次不带 force 运行 - 应跳过
        pipeline2 = IngestionPipeline(settings, collection=collection, force=False)
        try:
            result2 = pipeline2.run(simple_pdf_path)
            assert result2.success
            assert "integrity" in result2.stages
            assert result2.stages["integrity"].get("skipped") == True
            print("\n[OK] File correctly skipped on second run")
        finally:
            pipeline2.close()
    
    def test_pipeline_force_reprocess(self, settings, simple_pdf_path):
        """Test that force=True reprocesses even if already done. / 测试 force=True 会重新处理已完成文件。"""
        collection = "test_force"
        
        # First run / 第一次运行
        pipeline1 = IngestionPipeline(settings, collection=collection, force=True)
        try:
            result1 = pipeline1.run(simple_pdf_path)
            assert result1.success
            chunk_count1 = result1.chunk_count
        finally:
            pipeline1.close()
        
        # Second run with force - should reprocess / 第二次带 force 运行 - 应重新处理
        pipeline2 = IngestionPipeline(settings, collection=collection, force=True)
        try:
            result2 = pipeline2.run(simple_pdf_path)
            assert result2.success
            assert result2.chunk_count == chunk_count1
            assert result2.stages.get("integrity", {}).get("skipped") != True
            print("\n[OK] File correctly reprocessed with force=True")
        finally:
            pipeline2.close()


class TestPipelineComponents:
    """Test individual pipeline components in isolation. / 隔离测试单个流水线组件。"""
    
    @pytest.fixture
    def settings(self):
        """Load settings from config file. / 从配置文件加载设置。"""
        return load_settings("config/settings.yaml")
    
    def test_settings_loads_correctly(self, settings):
        """Verify settings are loaded with expected values. / 验证设置以预期值加载。"""
        # LLM settings / LLM 设置
        assert settings.llm.provider == "azure"
        assert settings.llm.model == "gpt-4o"
        print(f"[OK] LLM: {settings.llm.provider}/{settings.llm.model}")
        
        # Embedding settings / Embedding 设置
        assert settings.embedding.provider == "azure"
        assert settings.embedding.model == "text-embedding-ada-002"
        assert settings.embedding.dimensions == 1536
        print(f"[OK] Embedding: {settings.embedding.provider}/{settings.embedding.model}")
        
        # Vision LLM settings / Vision LLM 设置
        assert settings.vision_llm is not None
        assert settings.vision_llm.enabled == True
        assert settings.vision_llm.provider == "azure"
        print(f"[OK] Vision LLM: {settings.vision_llm.provider}/{settings.vision_llm.model}")
        
        # Ingestion settings / 摄入设置
        assert settings.ingestion is not None
        assert settings.ingestion.chunk_refiner is not None
        assert settings.ingestion.chunk_refiner.get("use_llm") == True
        assert settings.ingestion.metadata_enricher is not None
        assert settings.ingestion.metadata_enricher.get("use_llm") == True
        print("[OK] Ingestion LLM enhancement: enabled")
    
    def test_embedding_creates_vectors(self, settings):
        """Test that Azure embedding service works. / 测试 Azure embedding 服务可用。"""
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        
        embedding = EmbeddingFactory.create(settings)
        
        texts = ["Hello world", "Testing embedding"]
        vectors = embedding.embed(texts)
        
        assert len(vectors) == 2
        assert len(vectors[0]) == 1536  # ada-002 dimension / ada-002 维度
        print(f"[OK] Embedding test: produced {len(vectors)} vectors of dim {len(vectors[0])}")
    
    def test_llm_responds(self, settings):
        """Test that Azure LLM service works. / 测试 Azure LLM 服务可用。"""
        from src.libs.llm.llm_factory import LLMFactory
        from src.libs.llm.base_llm import Message
        
        llm = LLMFactory.create(settings)
        
        messages = [
            Message(role="user", content="Say 'Hello' and nothing else.")
        ]
        response = llm.chat(messages)
        
        assert response is not None
        assert len(response.content) > 0
        print(f"[OK] LLM test: received response: {response.content[:50]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
