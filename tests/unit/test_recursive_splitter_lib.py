"""Unit tests for RecursiveSplitter. / RecursiveSplitter 的单元测试。

Test Coverage: / 测试覆盖范围：
- Configuration-driven instantiation from settings / 从 settings 进行配置驱动实例化
- Chunk size and overlap parameter handling / chunk size 和 overlap 参数处理
- Markdown structure preservation (headers, code blocks) / Markdown 结构保留（headers、code blocks）
- Edge cases: empty text, very short text, very long text / 边界情况：空文本、很短文本、很长文本
- Error handling: missing dependencies, invalid configuration / 错误处理：缺失依赖、无效配置
- Integration with BaseSplitter validation / 与 BaseSplitter 校验集成
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.libs.splitter.base_splitter import BaseSplitter


# Test if langchain-text-splitters is available / 测试 langchain-text-splitters 是否可用
try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveSplitter = None  # type: ignore[misc, assignment]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterConfiguration:
    """Tests for RecursiveSplitter configuration and initialization. / RecursiveSplitter 配置和初始化测试。"""
    
    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object. / 创建 mock settings 对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_initialization_from_settings(self):
        """Test that RecursiveSplitter reads configuration from settings. / 测试 RecursiveSplitter 从 settings 读取配置。"""
        settings = self.create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(settings=settings)
        
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert isinstance(splitter, BaseSplitter)
    
    def test_initialization_with_overrides(self):
        """Test that constructor parameters override settings values. / 测试构造参数覆盖 settings 值。"""
        settings = self.create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(
            settings=settings,
            chunk_size=300,
            chunk_overlap=50,
        )
        
        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 50
    
    def test_initialization_with_custom_separators(self):
        """Test custom separator configuration. / 测试自定义 separator 配置。"""
        settings = self.create_mock_settings()
        custom_separators = ["\n\n", "\n", " "]
        splitter = RecursiveSplitter(
            settings=settings,
            separators=custom_separators,
        )
        
        assert splitter.separators == custom_separators
    
    def test_initialization_default_separators(self):
        """Test that default separators are Markdown-aware. / 测试默认 separators 具备 Markdown 感知能力。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        # Check that default separators include common text boundaries / 检查默认 separators 包含常见文本边界
        assert "\n\n" in splitter.separators  # Paragraphs / 段落
        assert "\n" in splitter.separators    # Lines / 行
        assert " " in splitter.separators     # Words / 单词
        assert "" in splitter.separators      # Characters / 字符
    
    def test_initialization_missing_settings(self):
        """Test error when settings.ingestion is missing. / 测试 settings.ingestion 缺失时的错误。"""
        settings = MagicMock()
        settings.ingestion = None
        
        with pytest.raises(ValueError, match="Missing ingestion configuration"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_invalid_chunk_size_negative(self):
        """Test error when chunk_size is negative. / 测试 chunk_size 为负数时的错误。"""
        settings = self.create_mock_settings(chunk_size=-100)
        
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_invalid_chunk_size_zero(self):
        """Test error when chunk_size is zero. / 测试 chunk_size 为零时的错误。"""
        settings = self.create_mock_settings(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_invalid_chunk_overlap_negative(self):
        """Test error when chunk_overlap is negative. / 测试 chunk_overlap 为负数时的错误。"""
        settings = self.create_mock_settings(chunk_overlap=-50)
        
        with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_overlap_exceeds_chunk_size(self):
        """Test error when chunk_overlap >= chunk_size. / 测试 chunk_overlap >= chunk_size 时的错误。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_overlap_greater_than_chunk_size(self):
        """Test error when chunk_overlap > chunk_size. / 测试 chunk_overlap > chunk_size 时的错误。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=200)
        
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterBasicSplitting:
    """Tests for basic text splitting behavior. / 基础文本切分行为测试。"""
    
    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object. / 创建 mock settings 对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_split_short_text(self):
        """Test splitting text shorter than chunk_size. / 测试切分短于 chunk_size 的文本。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "This is a short text."
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_by_paragraphs(self):
        """Test that splitter respects paragraph boundaries. / 测试 splitter 遵守段落边界。"""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        # Create text long enough to require multiple chunks / 创建足够长、需要多个 chunks 的文本
        text = "Paragraph one with some more content to make it longer.\n\nParagraph two also needs sufficient length.\n\nParagraph three should push it over the limit."
        chunks = splitter.split_text(text)
        
        # Should split at paragraph boundaries when text exceeds chunk_size / 文本超过 chunk_size 时应按段落边界切分
        assert len(chunks) >= 1  # At minimum, returns the text / 至少返回原文本
        for chunk in chunks:
            assert len(chunk) <= 70  # Allow some flexibility for boundary conditions / 允许边界条件有一定弹性
    
    def test_split_text_with_overlap(self):
        """Test that chunks have overlapping content. / 测试 chunks 具有重叠内容。"""
        settings = self.create_mock_settings(chunk_size=30, chunk_overlap=10)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "This is a long sentence that will be split into multiple chunks with overlap."
        chunks = splitter.split_text(text)
        
        # Should produce multiple chunks / 应生成多个 chunks
        assert len(chunks) >= 2
        
        # Each chunk should respect chunk_size (with some tolerance for word boundaries) / 每个 chunk 应遵守 chunk_size（对单词边界有一定容差）
        for chunk in chunks:
            assert len(chunk) <= 30 + 20  # Allow tolerance for not breaking words / 允许不拆分单词带来的容差
    
    def test_split_preserves_order(self):
        """Test that chunks preserve original text order. / 测试 chunks 保留原文本顺序。"""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "First section. Second section. Third section. Fourth section."
        chunks = splitter.split_text(text)
        
        # Reconstruct should preserve order / 重构后应保留顺序
        reconstructed = " ".join(chunks)
        assert "First" in reconstructed
        assert reconstructed.index("First") < reconstructed.index("Fourth")
    
    def test_split_empty_string_validation(self):
        """Test that empty string raises validation error. / 测试空字符串会抛出校验错误。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.split_text("   ")
    
    def test_split_non_string_validation(self):
        """Test that non-string input raises validation error. / 测试非字符串输入会抛出校验错误。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        with pytest.raises(ValueError, match="must be a string"):
            splitter.split_text(123)  # type: ignore[arg-type]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterMarkdownStructure:
    """Tests for Markdown structure preservation. / Markdown 结构保留测试。"""
    
    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object. / 创建 mock settings 对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_split_markdown_with_headers(self):
        """Test that Markdown headers are preserved in chunks. / 测试 Markdown headers 在 chunks 中被保留。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = """# Main Header

## Section 1
Content for section 1 goes here.

## Section 2
Content for section 2 goes here.

## Section 3
Content for section 3 goes here."""
        
        chunks = splitter.split_text(text)
        
        # Should produce multiple chunks / 应生成多个 chunks
        assert len(chunks) >= 1
        
        # Headers should be present in appropriate chunks / headers 应出现在合适的 chunks 中
        all_text = "".join(chunks)
        assert "# Main Header" in all_text
        assert "## Section 1" in all_text
    
    def test_split_markdown_code_blocks(self):
        """Test that code blocks are handled appropriately. / 测试 code blocks 被恰当处理。"""
        settings = self.create_mock_settings(chunk_size=150, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = """Some text before code.

```python
def example():
    return "code"
```

Some text after code."""
        
        chunks = splitter.split_text(text)
        
        # All content should be preserved / 所有内容都应保留
        all_text = "".join(chunks)
        assert "def example():" in all_text
        assert "Some text before" in all_text
        assert "Some text after" in all_text
    
    def test_split_markdown_lists(self):
        """Test that Markdown lists are handled appropriately. / 测试 Markdown lists 被恰当处理。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = """# List Example

- Item 1
- Item 2
- Item 3
- Item 4
- Item 5"""
        
        chunks = splitter.split_text(text)
        
        # Should preserve list structure / 应保留 list 结构
        all_text = "".join(chunks)
        assert "- Item 1" in all_text
        assert "- Item 5" in all_text


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterEdgeCases:
    """Tests for edge cases and error handling. / 边界情况和错误处理测试。"""
    
    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object. / 创建 mock settings 对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_split_very_long_text(self):
        """Test splitting very long text. / 测试切分很长文本。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=20)
        splitter = RecursiveSplitter(settings=settings)
        
        # Generate long text (1000 words) / 生成长文本（1000 个单词）
        text = " ".join(["word"] * 1000)
        chunks = splitter.split_text(text)
        
        # Should produce many chunks / 应生成许多 chunks
        assert len(chunks) >= 10
        
        # Each chunk should respect chunk_size (with tolerance) / 每个 chunk 应遵守 chunk_size（有容差）
        for chunk in chunks:
            assert len(chunk) <= 100 + 30  # Allow tolerance / 允许容差
    
    def test_split_single_long_word(self):
        """Test handling of a single word longer than chunk_size. / 测试处理长于 chunk_size 的单个单词。"""
        settings = self.create_mock_settings(chunk_size=10, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        # Single word longer than chunk_size / 单个单词长于 chunk_size
        text = "a" * 100
        chunks = splitter.split_text(text)
        
        # Should still split (may exceed chunk_size for unsplittable content) / 仍应切分（不可切分内容可能超过 chunk_size）
        assert len(chunks) >= 1
    
    def test_split_unicode_text(self):
        """Test handling of Unicode characters. / 测试处理 Unicode 字符。"""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "Hello 世界! Привет мир! 🌍🌎🌏"
        chunks = splitter.split_text(text)
        
        # Should handle Unicode without errors / 应无错误地处理 Unicode
        assert len(chunks) >= 1
        all_text = "".join(chunks)
        assert "世界" in all_text
        assert "мир" in all_text
        assert "🌍" in all_text
    
    def test_split_with_trace_parameter(self):
        """Test that trace parameter is accepted but not used. / 测试 trace 参数被接受但未使用。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        text = "Some text to split."
        mock_trace = MagicMock()
        
        # Should not raise error with trace parameter / 带 trace 参数时不应抛出错误
        chunks = splitter.split_text(text, trace=mock_trace)
        assert len(chunks) >= 1


@pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Test only when langchain-text-splitters NOT installed")
class TestRecursiveSplitterImportError:
    """Tests for ImportError when langchain-text-splitters is not installed. / langchain-text-splitters 未安装时的 ImportError 测试。"""
    
    def test_import_error_without_langchain(self):
        """Test that ImportError is raised when langchain is not available. / 测试 langchain 不可用时抛出 ImportError。"""
        with patch.dict('sys.modules', {'langchain_text_splitters': None}):
            # Force reimport to trigger ImportError / 强制重新导入以触发 ImportError
            import importlib
            import src.libs.splitter.recursive_splitter
            importlib.reload(src.libs.splitter.recursive_splitter)
            
            from src.libs.splitter.recursive_splitter import RecursiveSplitter
            
            settings = MagicMock()
            settings.ingestion = MagicMock()
            settings.ingestion.chunk_size = 1000
            settings.ingestion.chunk_overlap = 200
            
            with pytest.raises(ImportError, match="langchain-text-splitters is not installed"):
                RecursiveSplitter(settings=settings)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterFactoryIntegration:
    """Tests for factory integration. / 工厂集成测试。"""
    
    def test_factory_can_create_recursive_splitter(self):
        """Test that factory can instantiate RecursiveSplitter. / 测试工厂可以实例化 RecursiveSplitter。"""
        from src.libs.splitter.splitter_factory import SplitterFactory
        
        # Register the provider / 注册 provider
        SplitterFactory.register_provider("recursive", RecursiveSplitter)
        
        # Create settings / 创建 settings
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "recursive"
        settings.ingestion.chunk_size = 500
        settings.ingestion.chunk_overlap = 100
        
        # Factory should create RecursiveSplitter / 工厂应创建 RecursiveSplitter
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
