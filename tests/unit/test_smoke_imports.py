"""Smoke tests for package imports. / 包导入的冒烟测试。

This module verifies that all key packages can be imported successfully. / 该模块验证所有关键包都能成功导入。
It serves as a basic sanity check for the project structure. / 它作为项目结构的基础健全性检查。
"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """Smoke tests to verify all key packages are importable. / 验证所有关键包可导入的冒烟测试。"""

    def test_import_src_package(self) -> None:
        """Test that the src package can be imported. / 测试 src 包可导入。"""
        import src
        assert src is not None

    def test_import_mcp_server(self) -> None:
        """Test that the mcp_server package can be imported. / 测试 mcp_server 包可导入。"""
        from src import mcp_server
        assert mcp_server is not None

    def test_import_mcp_server_tools(self) -> None:
        """Test that the mcp_server.tools subpackage can be imported. / 测试 mcp_server.tools 子包可导入。"""
        from src.mcp_server import tools
        assert tools is not None

    def test_import_core(self) -> None:
        """Test that the core package can be imported. / 测试 core 包可导入。"""
        from src import core
        assert core is not None

    def test_import_core_query_engine(self) -> None:
        """Test that the core.query_engine subpackage can be imported. / 测试 core.query_engine 子包可导入。"""
        from src.core import query_engine
        assert query_engine is not None

    def test_import_core_response(self) -> None:
        """Test that the core.response subpackage can be imported. / 测试 core.response 子包可导入。"""
        from src.core import response
        assert response is not None

    def test_import_core_trace(self) -> None:
        """Test that the core.trace subpackage can be imported. / 测试 core.trace 子包可导入。"""
        from src.core import trace
        assert trace is not None

    def test_import_ingestion(self) -> None:
        """Test that the ingestion package can be imported. / 测试 ingestion 包可导入。"""
        from src import ingestion
        assert ingestion is not None

    def test_import_ingestion_embedding(self) -> None:
        """Test that the ingestion.embedding subpackage can be imported. / 测试 ingestion.embedding 子包可导入。"""
        from src.ingestion import embedding
        assert embedding is not None

    def test_import_ingestion_storage(self) -> None:
        """Test that the ingestion.storage subpackage can be imported. / 测试 ingestion.storage 子包可导入。"""
        from src.ingestion import storage
        assert storage is not None

    def test_import_ingestion_transform(self) -> None:
        """Test that the ingestion.transform subpackage can be imported. / 测试 ingestion.transform 子包可导入。"""
        from src.ingestion import transform
        assert transform is not None

    def test_import_libs(self) -> None:
        """Test that the libs package can be imported. / 测试 libs 包可导入。"""
        from src import libs
        assert libs is not None

    def test_import_libs_embedding(self) -> None:
        """Test that the libs.embedding subpackage can be imported. / 测试 libs.embedding 子包可导入。"""
        from src.libs import embedding
        assert embedding is not None

    def test_import_libs_evaluator(self) -> None:
        """Test that the libs.evaluator subpackage can be imported. / 测试 libs.evaluator 子包可导入。"""
        from src.libs import evaluator
        assert evaluator is not None

    def test_import_libs_llm(self) -> None:
        """Test that the libs.llm subpackage can be imported. / 测试 libs.llm 子包可导入。"""
        from src.libs import llm
        assert llm is not None

    def test_import_libs_loader(self) -> None:
        """Test that the libs.loader subpackage can be imported. / 测试 libs.loader 子包可导入。"""
        from src.libs import loader
        assert loader is not None

    def test_import_libs_reranker(self) -> None:
        """Test that the libs.reranker subpackage can be imported. / 测试 libs.reranker 子包可导入。"""
        from src.libs import reranker
        assert reranker is not None

    def test_import_libs_splitter(self) -> None:
        """Test that the libs.splitter subpackage can be imported. / 测试 libs.splitter 子包可导入。"""
        from src.libs import splitter
        assert splitter is not None

    def test_import_libs_vector_store(self) -> None:
        """Test that the libs.vector_store subpackage can be imported. / 测试 libs.vector_store 子包可导入。"""
        from src.libs import vector_store
        assert vector_store is not None

    def test_import_observability(self) -> None:
        """Test that the observability package can be imported. / 测试 observability 包可导入。"""
        from src import observability
        assert observability is not None

    def test_import_observability_dashboard(self) -> None:
        """Test that the observability.dashboard subpackage can be imported. / 测试 observability.dashboard 子包可导入。"""
        from src.observability import dashboard
        assert dashboard is not None

    def test_import_observability_evaluation(self) -> None:
        """Test that the observability.evaluation subpackage can be imported. / 测试 observability.evaluation 子包可导入。"""
        from src.observability import evaluation
        assert evaluation is not None
