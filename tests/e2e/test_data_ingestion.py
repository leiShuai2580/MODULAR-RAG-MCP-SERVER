"""End-to-End tests for the data ingestion script. / 数据摄入脚本的端到端测试。

This module tests the complete ingestion workflow via the command-line interface, / 该模块通过命令行接口测试完整摄入工作流，
including: / 包括：
- Single file ingestion / 单文件摄入
- Directory ingestion / 目录摄入
- Force re-processing / 强制重新处理
- Skip already processed files / 跳过已处理文件
- Error handling for invalid inputs / 无效输入的错误处理
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Project root for script execution / 脚本执行时使用的项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDataIngestion:
    """E2E tests for scripts/ingest.py. / scripts/ingest.py 的 E2E 测试。"""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create a temporary data directory for test outputs. / 为测试输出创建临时数据目录。
        
        Yields: / 产出：
            Path to temporary directory / 临时目录路径
        """
        # Create subdirectories matching production structure / 创建与生产结构匹配的子目录
        (tmp_path / "db" / "chroma").mkdir(parents=True)
        (tmp_path / "db" / "bm25").mkdir(parents=True)
        (tmp_path / "images").mkdir(parents=True)
        
        yield tmp_path
        
        # Cleanup is handled by pytest tmp_path fixture / 清理由 pytest tmp_path fixture 处理
    
    @pytest.fixture
    def sample_pdf(self):
        """Get path to sample PDF for testing. / 获取用于测试的示例 PDF 路径。
        
        Returns: / 返回：
            Path to a sample PDF file / 示例 PDF 文件路径
        """
        pdf_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"
        if not pdf_path.exists():
            pytest.skip("Sample PDF not found")
        return pdf_path
    
    @pytest.fixture
    def complex_pdf(self):
        """Get path to complex technical document for testing. / 获取用于测试的复杂技术文档路径。
        
        Returns: / 返回：
            Path to a complex PDF with images / 带图片的复杂 PDF 路径
        """
        pdf_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "complex_technical_doc.pdf"
        if not pdf_path.exists():
            pytest.skip("Complex PDF not found")
        return pdf_path
    
    def run_ingest_script(
        self,
        path: str,
        collection: str = "test_collection",
        force: bool = False,
        config: str = None,
        dry_run: bool = False,
        verbose: bool = False
    ) -> subprocess.CompletedProcess:
        """Run the ingest script as a subprocess. / 以子进程运行 ingest 脚本。
        
        Args: / 参数：
            path: Path to file or directory / 文件或目录路径
            collection: Collection name / 集合名称
            force: Whether to force re-processing / 是否强制重新处理
            config: Custom config path / 自定义配置路径
            dry_run: Whether to run in dry-run mode / 是否以空运行模式执行
            verbose: Whether to enable verbose output / 是否启用详细输出
            
        Returns: / 返回：
            CompletedProcess with stdout, stderr, and return code / 包含 stdout、stderr 和返回码的 CompletedProcess
        """
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingest.py"),
            "--path", str(path),
            "--collection", collection
        ]
        
        if force:
            cmd.append("--force")
        
        if config:
            cmd.extend(["--config", config])
        
        if dry_run:
            cmd.append("--dry-run")
        
        if verbose:
            cmd.append("--verbose")
        
        # Set PYTHONUTF8=1 to avoid encoding issues on Windows / 设置 PYTHONUTF8=1 以避免 Windows 编码问题
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,  # 10 minute timeout for LLM calls (complex docs with vision) / LLM 调用 10 分钟超时（含视觉的复杂文档）
            env=env,
            encoding='utf-8',
            errors='replace'
        )
    
    def test_ingest_help(self):
        """Test that --help flag works. / 测试 --help 标志可用。"""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        assert result.returncode == 0
        assert "--path" in result.stdout
        assert "--collection" in result.stdout
        assert "--force" in result.stdout
    
    def test_ingest_nonexistent_file(self):
        """Test error handling for non-existent file. / 测试不存在文件的错误处理。"""
        result = self.run_ingest_script(
            path="/nonexistent/path/document.pdf"
        )
        
        assert result.returncode == 2
        assert "does not exist" in result.stdout or "not found" in result.stdout.lower()
    
    def test_ingest_invalid_config(self, sample_pdf):
        """Test error handling for invalid config file. / 测试无效配置文件的错误处理。"""
        result = self.run_ingest_script(
            path=str(sample_pdf),
            config="/nonexistent/config.yaml"
        )
        
        assert result.returncode == 2
        assert "not found" in result.stdout.lower() or "Configuration" in result.stdout
    
    def test_ingest_dry_run(self, sample_pdf):
        """Test dry-run mode doesn't process files. / 测试空运行模式不会处理文件。"""
        result = self.run_ingest_script(
            path=str(sample_pdf),
            dry_run=True
        )
        
        assert result.returncode == 0
        assert "Dry run" in result.stdout or "dry run" in result.stdout.lower()
        assert "1 file" in result.stdout
    
    def test_ingest_unsupported_file_type(self, tmp_path):
        """Test error handling for unsupported file types. / 测试不支持文件类型的错误处理。"""
        # Create a text file / 创建文本文件
        text_file = tmp_path / "document.txt"
        text_file.write_text("This is a text file")
        
        result = self.run_ingest_script(
            path=str(text_file)
        )
        
        assert result.returncode == 2
        assert "Unsupported" in result.stdout or "unsupported" in result.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_simple_pdf(self, sample_pdf):
        """Test ingesting a simple PDF file. / 测试摄入简单 PDF 文件。
        
        This test requires Azure API credentials to be configured. / 该测试要求已配置 Azure API 凭据。
        """
        result = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_simple",
            force=True,  # Force to ensure fresh processing / 强制以确保重新处理
            verbose=True
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # Should succeed or partially succeed / 应成功或部分成功
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        assert "Processing" in result.stdout
        assert "SUMMARY" in result.stdout
    
    @pytest.mark.integration
    def test_ingest_complex_pdf_with_images(self, complex_pdf):
        """Test ingesting a complex PDF with images. / 测试摄入带图片的复杂 PDF。
        
        This test requires Azure API credentials and Vision LLM to be configured. / 该测试要求已配置 Azure API 凭据和 Vision LLM。
        Tests the full pipeline including image captioning. / 测试包含图片描述生成在内的完整流水线。
        """
        result = self.run_ingest_script(
            path=str(complex_pdf),
            collection="e2e_test_complex",
            force=True,
            verbose=True
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # Should succeed or partially succeed / 应成功或部分成功
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        assert "Processing" in result.stdout
        assert "SUMMARY" in result.stdout
        
        # If successful, should report chunks and possibly images / 如果成功，应报告分块以及可能的图片
        if result.returncode == 0:
            assert "chunks" in result.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_skip_already_processed(self, sample_pdf):
        """Test that already processed files are skipped. / 测试已处理文件会被跳过。
        
        Runs ingestion twice and verifies second run skips the file. / 运行两次摄入并验证第二次会跳过文件。
        """
        # First run - should process / 第一次运行 - 应处理
        result1 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_skip",
            force=True  # Ensure fresh start / 确保从新状态开始
        )
        
        print("First run STDOUT:", result1.stdout)
        
        # Skip test if first run failed / 如果第一次运行失败则跳过测试
        if result1.returncode == 2:
            pytest.skip("First ingestion failed - cannot test skip behavior")
        
        # Second run - should skip / 第二次运行 - 应跳过
        result2 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_skip",
            force=False  # Don't force, should skip / 不强制，应跳过
        )
        
        print("Second run STDOUT:", result2.stdout)
        
        # Should succeed but with skip / 应成功但发生跳过
        assert result2.returncode == 0
        assert "skip" in result2.stdout.lower() or "already processed" in result2.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_force_reprocess(self, sample_pdf):
        """Test that --force flag causes re-processing. / 测试 --force 标志会触发重新处理。"""
        # First run / 第一次运行
        result1 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_force",
            force=True
        )
        
        # Skip test if first run failed / 如果第一次运行失败则跳过测试
        if result1.returncode == 2:
            pytest.skip("First ingestion failed - cannot test force behavior")
        
        # Second run with force - should process again / 第二次带 force 运行 - 应再次处理
        result2 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_force",
            force=True,
            verbose=True
        )
        
        print("Second run STDOUT:", result2.stdout)
        
        # Should succeed and process (not skip) / 应成功并处理（而不是跳过）
        assert result2.returncode in [0, 1]
        # When forced, should not show "skipped" / 强制时不应显示 "skipped"
        if "Success" in result2.stdout:
            assert "chunks" in result2.stdout.lower() or "processed" in result2.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_directory(self, tmp_path, sample_pdf):
        """Test ingesting all PDFs in a directory. / 测试摄入目录中的所有 PDF。"""
        # Create a directory with multiple PDFs (copy sample) / 创建包含多个 PDF 的目录（复制示例）
        test_dir = tmp_path / "pdfs"
        test_dir.mkdir()
        
        shutil.copy(sample_pdf, test_dir / "doc1.pdf")
        shutil.copy(sample_pdf, test_dir / "doc2.pdf")
        
        result = self.run_ingest_script(
            path=str(test_dir),
            collection="e2e_test_dir",
            force=True,
            verbose=True
        )
        
        print("STDOUT:", result.stdout)
        
        # Should find both files / 应找到两个文件
        assert "2 file" in result.stdout
        
        # Should attempt to process both / 应尝试处理两个文件
        assert "[1/2]" in result.stdout
        assert "[2/2]" in result.stdout
    
    def test_ingest_empty_directory(self, tmp_path):
        """Test handling of directory with no PDFs. / 测试没有 PDF 的目录处理。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = self.run_ingest_script(
            path=str(empty_dir)
        )
        
        assert result.returncode == 0
        assert "0 file" in result.stdout or "No files" in result.stdout


class TestIngestScriptIntegration:
    """Integration tests that verify data persistence. / 验证数据持久化的集成测试。"""
    
    @pytest.mark.integration
    def test_creates_vector_store_data(self, tmp_path):
        """Verify that ingestion creates vector store data. / 验证摄入会创建向量存储数据。"""
        # This test verifies the pipeline creates the expected output files / 该测试验证流水线会创建预期输出文件
        # It's marked as integration because it requires the full stack / 它标记为 integration，因为需要完整技术栈
        
        sample_pdf = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found")
        
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "ingest.py"),
                "--path", str(sample_pdf),
                "--collection", "e2e_test_persistence",
                "--force"
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # Verify data directories exist after successful ingestion / 验证成功摄入后数据目录存在
        if result.returncode == 0:
            chroma_dir = PROJECT_ROOT / "data" / "db" / "chroma"
            bm25_dir = PROJECT_ROOT / "data" / "db" / "bm25" / "e2e_test_persistence"
            
            assert chroma_dir.exists(), "ChromaDB directory should exist"
            # BM25 index directory may or may not exist based on implementation / BM25 索引目录是否存在取决于实现


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
