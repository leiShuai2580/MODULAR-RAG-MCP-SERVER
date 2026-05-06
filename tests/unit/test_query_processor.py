"""Unit tests for QueryProcessor. / QueryProcessor 的单元测试。

Tests cover: / 测试覆盖：
- Basic keyword extraction / 基础关键词提取
- Chinese and English stopword filtering / 中英文停用词过滤
- Filter syntax parsing (collection:xxx, type:xxx) / 过滤器语法解析（collection:xxx、type:xxx）
- Edge cases (empty query, special characters) / 边界情况（空 query、特殊字符）
- Configuration options / 配置选项
"""

import pytest
from src.core.query_engine.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    create_query_processor,
    DEFAULT_STOPWORDS,
    CHINESE_STOPWORDS,
    ENGLISH_STOPWORDS,
)
from src.core.types import ProcessedQuery


class TestQueryProcessorBasic:
    """Test basic QueryProcessor functionality. / 测试 QueryProcessor 基础功能。"""
    
    def test_simple_english_query(self):
        """Test simple English query keyword extraction. / 测试简单英文 query 的关键词提取。"""
        processor = QueryProcessor()
        result = processor.process("Azure OpenAI configuration")
        
        assert result.original_query == "Azure OpenAI configuration"
        assert "Azure" in result.keywords
        assert "OpenAI" in result.keywords
        assert "configuration" in result.keywords
        assert isinstance(result.filters, dict)
    
    def test_simple_chinese_query(self):
        """Test simple Chinese query keyword extraction. / 测试简单中文 query 的关键词提取。"""
        processor = QueryProcessor()
        result = processor.process("配置 Azure OpenAI")
        
        assert result.original_query == "配置 Azure OpenAI"
        assert "配置" in result.keywords
        assert "Azure" in result.keywords
        assert "OpenAI" in result.keywords
    
    def test_mixed_language_query(self):
        """Test mixed Chinese-English query. / 测试中英混合 query。"""
        processor = QueryProcessor()
        result = processor.process("如何配置 Azure OpenAI embedding 模型")
        
        # Note: Simple tokenizer treats continuous Chinese as single token / 注意：简单 tokenizer 会将连续中文视作单个 token
        # So "如何配置" is one token, not split into "如何" + "配置" / 因此 "如何配置" 是一个 token，不会拆分为 "如何" + "配置"
        assert "Azure" in result.keywords
        assert "OpenAI" in result.keywords
        assert "embedding" in result.keywords
        assert "模型" in result.keywords
        # Keywords should be non-empty (acceptance criteria) / keywords 应非空（验收标准）
        assert len(result.keywords) > 0


class TestStopwordFiltering:
    """Test stopword filtering. / 测试停用词过滤。"""
    
    def test_chinese_stopwords_filtered(self):
        """Test that Chinese stopwords are filtered. / 测试中文停用词会被过滤。"""
        processor = QueryProcessor()
        # Use space-separated Chinese words for proper tokenization / 使用空格分隔中文词以便正确分词
        result = processor.process("如何 在 配置 系统")
        
        # Individual stopwords should be filtered / 单独的停用词应被过滤
        assert "如何" not in result.keywords
        assert "在" not in result.keywords
        # Content words should remain / 内容词应保留
        assert "配置" in result.keywords
        assert "系统" in result.keywords
    
    def test_english_stopwords_filtered(self):
        """Test that English stopwords are filtered. / 测试英文停用词会被过滤。"""
        processor = QueryProcessor()
        result = processor.process("how to configure the Azure API")
        
        # Stopwords should be filtered / 停用词应被过滤
        assert "how" not in result.keywords
        assert "to" not in result.keywords
        assert "the" not in result.keywords
        # Content words should remain / 内容词应保留
        assert "configure" in result.keywords
        assert "Azure" in result.keywords
        assert "API" in result.keywords
    
    def test_custom_stopwords(self):
        """Test custom stopwords configuration. / 测试自定义停用词配置。"""
        custom_stopwords = {"custom", "word"}
        config = QueryProcessorConfig(stopwords=custom_stopwords)
        processor = QueryProcessor(config)
        
        result = processor.process("custom word test")
        
        assert "custom" not in result.keywords
        assert "word" not in result.keywords
        assert "test" in result.keywords
    
    def test_add_stopwords(self):
        """Test adding stopwords dynamically. / 测试动态添加停用词。"""
        processor = QueryProcessor()
        processor.add_stopwords({"newstop"})
        
        result = processor.process("newstop important")
        
        assert "newstop" not in result.keywords
        assert "important" in result.keywords
    
    def test_remove_stopwords(self):
        """Test removing stopwords dynamically. / 测试动态移除停用词。"""
        processor = QueryProcessor()
        processor.remove_stopwords({"如何"})
        
        # Use space-separated input for proper tokenization / 使用空格分隔输入以便正确分词
        result = processor.process("如何 配置")
        
        assert "如何" in result.keywords
        assert "配置" in result.keywords


class TestFilterParsing:
    """Test filter syntax parsing. / 测试过滤器语法解析。"""
    
    def test_collection_filter(self):
        """Test collection filter parsing. / 测试 collection 过滤器解析。"""
        processor = QueryProcessor()
        result = processor.process("collection:api-docs Azure configuration")
        
        assert result.filters.get("collection") == "api-docs"
        assert "Azure" in result.keywords
        assert "configuration" in result.keywords
        # Filter syntax should not appear in keywords / 过滤器语法不应出现在 keywords 中
        assert "collection" not in result.keywords
        assert "api-docs" not in result.keywords
    
    def test_collection_short_syntax(self):
        """Test collection short syntax (col:). / 测试 collection 简写语法（col:）。"""
        processor = QueryProcessor()
        result = processor.process("col:docs Azure")
        
        assert result.filters.get("collection") == "docs"
        assert "Azure" in result.keywords
    
    def test_type_filter(self):
        """Test doc_type filter parsing. / 测试 doc_type 过滤器解析。"""
        processor = QueryProcessor()
        result = processor.process("type:pdf search query")
        
        assert result.filters.get("doc_type") == "pdf"
        assert "search" in result.keywords
        assert "query" in result.keywords
    
    def test_source_filter(self):
        """Test source path filter parsing. / 测试 source path 过滤器解析。"""
        processor = QueryProcessor()
        result = processor.process("source:readme.md content")
        
        assert result.filters.get("source_path") == "readme.md"
        assert "content" in result.keywords
    
    def test_tag_filter(self):
        """Test tag filter parsing. / 测试 tag 过滤器解析。"""
        processor = QueryProcessor()
        result = processor.process("tag:important,urgent search")
        
        assert "tags" in result.filters
        assert "important" in result.filters["tags"]
        assert "urgent" in result.filters["tags"]
    
    def test_multiple_filters(self):
        """Test multiple filters in one query. / 测试一个 query 中的多个过滤器。"""
        processor = QueryProcessor()
        result = processor.process("collection:docs type:pdf Azure configuration")
        
        assert result.filters.get("collection") == "docs"
        assert result.filters.get("doc_type") == "pdf"
        assert "Azure" in result.keywords
        assert "configuration" in result.keywords
    
    def test_generic_filter(self):
        """Test generic filter key:value syntax. / 测试通用过滤器 key:value 语法。"""
        processor = QueryProcessor()
        result = processor.process("custom_field:custom_value search")
        
        assert result.filters.get("custom_field") == "custom_value"
        assert "search" in result.keywords
    
    def test_disable_filter_parsing(self):
        """Test disabling filter parsing. / 测试禁用过滤器解析。"""
        config = QueryProcessorConfig(enable_filter_parsing=False)
        processor = QueryProcessor(config)
        
        result = processor.process("collection:docs Azure")
        
        assert len(result.filters) == 0
        # collection:docs should be treated as text / collection:docs 应被当作文本处理
        assert "collection" in result.keywords or "docs" in result.keywords


class TestEdgeCases:
    """Test edge cases and error handling. / 测试边界情况和错误处理。"""
    
    def test_empty_query(self):
        """Test empty query handling. / 测试空 query 处理。"""
        processor = QueryProcessor()
        result = processor.process("")
        
        assert result.original_query == ""
        assert result.keywords == []
        assert result.filters == {}
    
    def test_none_query(self):
        """Test None query handling. / 测试 None query 处理。"""
        processor = QueryProcessor()
        result = processor.process(None)
        
        assert result.original_query == ""
        assert result.keywords == []
        assert result.filters == {}
    
    def test_whitespace_only_query(self):
        """Test whitespace-only query. / 测试仅空白字符的 query。"""
        processor = QueryProcessor()
        result = processor.process("   \t\n  ")
        
        assert result.keywords == []
    
    def test_special_characters(self):
        """Test query with special characters. / 测试包含特殊字符的 query。"""
        processor = QueryProcessor()
        result = processor.process("Azure-OpenAI API_key configuration")
        
        # Hyphenated and underscored words should be handled / 应处理连字符和下划线单词
        assert any("Azure" in kw or "OpenAI" in kw for kw in result.keywords)
        assert any("API" in kw or "key" in kw for kw in result.keywords)
        assert "configuration" in result.keywords
    
    def test_numbers_in_query(self):
        """Test query with numbers. / 测试包含数字的 query。"""
        processor = QueryProcessor()
        result = processor.process("GPT4 text-embedding-3-small")
        
        assert "GPT4" in result.keywords
        # Handle hyphenated model names / 处理带连字符的模型名
        assert any("embedding" in kw.lower() for kw in result.keywords)
    
    def test_duplicate_keywords(self):
        """Test duplicate keyword handling. / 测试重复关键词处理。"""
        processor = QueryProcessor()
        result = processor.process("Azure Azure azure AZURE")
        
        # Should deduplicate (case-insensitive) / 应去重（大小写不敏感）
        azure_count = sum(1 for kw in result.keywords if kw.lower() == "azure")
        assert azure_count == 1
    
    def test_very_long_query(self):
        """Test very long query with max_keywords limit. / 测试带 max_keywords 限制的超长 query。"""
        config = QueryProcessorConfig(max_keywords=5)
        processor = QueryProcessor(config)
        
        # Query with many keywords / 包含许多 keywords 的 query
        query = " ".join([f"keyword{i}" for i in range(20)])
        result = processor.process(query)
        
        assert len(result.keywords) <= 5
    
    def test_min_keyword_length(self):
        """Test minimum keyword length constraint. / 测试最小关键词长度约束。"""
        config = QueryProcessorConfig(min_keyword_length=3)
        processor = QueryProcessor(config)
        
        result = processor.process("a ab abc abcd")
        
        assert "a" not in result.keywords
        assert "ab" not in result.keywords
        assert "abc" in result.keywords
        assert "abcd" in result.keywords


class TestProcessedQueryContract:
    """Test ProcessedQuery data contract. / 测试 ProcessedQuery 数据契约。"""
    
    def test_processed_query_structure(self):
        """Test ProcessedQuery has expected structure. / 测试 ProcessedQuery 具有预期结构。"""
        processor = QueryProcessor()
        result = processor.process("test query")
        
        assert isinstance(result, ProcessedQuery)
        assert hasattr(result, "original_query")
        assert hasattr(result, "keywords")
        assert hasattr(result, "filters")
        assert hasattr(result, "expanded_terms")
    
    def test_processed_query_serialization(self):
        """Test ProcessedQuery can be serialized to dict. / 测试 ProcessedQuery 可序列化为 dict。"""
        processor = QueryProcessor()
        result = processor.process("collection:docs Azure test")
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data["original_query"] == "collection:docs Azure test"
        assert "Azure" in data["keywords"]
        assert "test" in data["keywords"]
        assert data["filters"]["collection"] == "docs"
    
    def test_processed_query_from_dict(self):
        """Test ProcessedQuery can be created from dict. / 测试 ProcessedQuery 可从 dict 创建。"""
        data = {
            "original_query": "test query",
            "keywords": ["test", "query"],
            "filters": {"collection": "docs"},
            "expanded_terms": []
        }
        
        result = ProcessedQuery.from_dict(data)
        
        assert result.original_query == "test query"
        assert result.keywords == ["test", "query"]
        assert result.filters == {"collection": "docs"}


class TestFactoryFunction:
    """Test create_query_processor factory function. / 测试 create_query_processor 工厂函数。"""
    
    def test_default_factory(self):
        """Test factory with default settings. / 测试使用默认 settings 的 factory。"""
        processor = create_query_processor()
        result = processor.process("test Azure")
        
        assert isinstance(processor, QueryProcessor)
        assert "Azure" in result.keywords
    
    def test_factory_with_custom_stopwords(self):
        """Test factory with custom stopwords. / 测试使用自定义停用词的 factory。"""
        processor = create_query_processor(stopwords={"custom"})
        result = processor.process("custom test")
        
        assert "custom" not in result.keywords
        assert "test" in result.keywords
        # Default stopwords should not apply / 默认停用词不应生效
        assert "how" not in processor.config.stopwords
    
    def test_factory_with_min_length(self):
        """Test factory with custom min_keyword_length. / 测试使用自定义 min_keyword_length 的 factory。"""
        processor = create_query_processor(min_keyword_length=4)
        result = processor.process("a ab abc abcd abcde")
        
        assert "a" not in result.keywords
        assert "abc" not in result.keywords
        assert "abcd" in result.keywords
    
    def test_factory_with_max_keywords(self):
        """Test factory with custom max_keywords. / 测试使用自定义 max_keywords 的 factory。"""
        processor = create_query_processor(max_keywords=2)
        result = processor.process("one two three four five")
        
        assert len(result.keywords) <= 2
    
    def test_factory_disable_filters(self):
        """Test factory with filter parsing disabled. / 测试禁用过滤器解析的 factory。"""
        processor = create_query_processor(enable_filter_parsing=False)
        result = processor.process("collection:docs test")
        
        assert len(result.filters) == 0


class TestChineseTextProcessing:
    """Test Chinese text processing specifics. / 测试中文文本处理细节。"""
    
    def test_chinese_only_query(self):
        """Test pure Chinese query. / 测试纯中文 query。"""
        processor = QueryProcessor()
        result = processor.process("向量数据库配置指南")
        
        assert len(result.keywords) > 0
        # Should extract meaningful Chinese words/phrases / 应提取有意义的中文词/短语
        assert any("向量" in kw or "数据库" in kw or "配置" in kw for kw in result.keywords)
    
    def test_chinese_with_punctuation(self):
        """Test Chinese query with punctuation. / 测试带标点的中文 query。"""
        processor = QueryProcessor()
        # Use space-separated for proper tokenization / 使用空格分隔以便正确分词
        result = processor.process("配置 问题 ？ 帮助 ！")
        
        # Punctuation should not affect extraction, keywords extracted / 标点不应影响提取，应提取 keywords
        assert "配置" in result.keywords
        assert "问题" in result.keywords
        # Keywords should be non-empty / keywords 应非空
        assert len(result.keywords) > 0


class TestKeywordsNonEmpty:
    """Test that keywords are non-empty for valid queries (acceptance criteria). / 测试有效 query 的 keywords 非空（验收标准）。"""
    
    def test_keywords_non_empty_english(self):
        """Test keywords non-empty for English query. / 测试英文 query 的 keywords 非空。"""
        processor = QueryProcessor()
        result = processor.process("configure Azure API")
        
        # Per acceptance criteria: keywords should be non-empty / 根据验收标准：keywords 应非空
        assert len(result.keywords) > 0
    
    def test_keywords_non_empty_chinese(self):
        """Test keywords non-empty for Chinese query. / 测试中文 query 的 keywords 非空。"""
        processor = QueryProcessor()
        result = processor.process("配置数据库")
        
        assert len(result.keywords) > 0
    
    def test_keywords_non_empty_mixed(self):
        """Test keywords non-empty for mixed query. / 测试混合 query 的 keywords 非空。"""
        processor = QueryProcessor()
        result = processor.process("Azure 配置")
        
        assert len(result.keywords) > 0
    
    def test_filters_is_dict(self):
        """Test filters is always a dict (acceptance criteria). / 测试 filters 始终为 dict（验收标准）。"""
        processor = QueryProcessor()
        
        # Without filters / 无 filters
        result1 = processor.process("simple query")
        assert isinstance(result1.filters, dict)
        
        # With filters / 有 filters
        result2 = processor.process("collection:docs query")
        assert isinstance(result2.filters, dict)
