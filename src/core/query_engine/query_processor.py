"""Query Processor for preprocessing user queries. / 用于预处理用户查询的查询处理器。

This module provides query preprocessing functionality including: / 本模块提供查询预处理功能，包括：
- Keyword extraction using rule-based tokenization / 使用基于规则的分词提取关键词
- Stopword filtering for Chinese and English / 针对中文和英文进行停用词过滤
- Filter parsing from query syntax (e.g., "collection:docs") / 从查询语法中解析过滤条件（例如 "collection:docs"）
- Query normalization and cleaning / 查询规范化和清理

Design Principles: / 设计原则：
- Rule-based first: Use simple, deterministic rules for reliability / 规则优先：使用简单、确定性的规则保证可靠性
- Language-aware: Support both Chinese and English queries / 语言感知：同时支持中文和英文查询
- Extensible: Easy to add synonym expansion or LLM-based processing later / 易扩展：后续易于加入同义词扩展或基于 LLM 的处理
- Configuration-driven: Stopwords and patterns configurable via settings / 配置驱动：停用词和模式可通过配置调整
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set

import jieba

from src.core.types import ProcessedQuery


# Default stopwords for Chinese / 中文默认停用词
CHINESE_STOPWORDS: Set[str] = {
    # 疑问词
    "如何", "怎么", "怎样", "什么", "哪个", "哪些", "为什么", "为何",
    "谁", "多少", "几", "是否", "能否", "可否",
    # 助词
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀",
    # 介词/连词
    "在", "于", "和", "与", "或", "及", "并", "而", "但", "但是",
    "因为", "所以", "如果", "那么", "虽然", "然而",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # 副词
    "很", "非常", "特别", "更", "最", "都", "也", "还", "又", "再",
    "已", "已经", "正在", "将", "会", "能", "可以", "应该", "必须",
    # 动词(通用)
    "是", "有", "做", "进行", "使用", "通过",
    # 量词
    "个", "种", "类",
    # 标点等
    "？", "。", "！", "，", "、",
}

# Default stopwords for English / 英文默认停用词
ENGLISH_STOPWORDS: Set[str] = {
    # Articles / 冠词
    "a", "an", "the",
    # Prepositions / 介词
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "through", "between", "after", "before",
    # Conjunctions / 连词
    "and", "or", "but", "if", "then", "because", "while", "although",
    # Pronouns / 代词
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    # Auxiliary verbs / 助动词
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can",
    # Common verbs / 常见动词
    "get", "use", "make",
    # Question words / 疑问词
    "how", "why", "when", "where",
    # Others / 其他
    "not", "no", "yes", "so", "very", "just", "also", "too",
}

# Combined default stopwords / 合并后的默认停用词
DEFAULT_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ENGLISH_STOPWORDS

# Pattern for filter syntax: key:value / 过滤语法模式：key:value
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')


@dataclass
class QueryProcessorConfig:
    """Configuration for QueryProcessor. / QueryProcessor 的配置。
    
    Attributes: / 属性：
        stopwords: Set of words to filter out / stopwords：需要过滤掉的词集合
        min_keyword_length: Minimum length for a keyword to be included / min_keyword_length：关键词被保留所需的最小长度
        max_keywords: Maximum number of keywords to extract / max_keywords：最多提取的关键词数量
        enable_filter_parsing: Whether to parse filter syntax from query / enable_filter_parsing：是否从查询中解析过滤语法
    """
    stopwords: Set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True


class QueryProcessor:
    """Preprocesses user queries for retrieval. / 为检索预处理用户查询。
    
    Extracts keywords, filters stopwords, and parses filter syntax / 提取关键词、过滤停用词并解析过滤语法，
    to prepare queries for Dense and Sparse retrievers. / 为稠密和稀疏检索器准备查询。
    
    Example: / 示例：
        >>> processor = QueryProcessor()
        >>> result = processor.process("如何配置 Azure OpenAI？")
        >>> print(result.keywords)
        ['配置', 'Azure', 'OpenAI']
    """
    
    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        """Initialize QueryProcessor. / 初始化 QueryProcessor。
        
        Args: / 参数：
            config: Optional configuration. Uses defaults if not provided. / config：可选配置。未提供时使用默认配置。
        """
        self.config = config or QueryProcessorConfig()
    
    def process(self, query: str) -> ProcessedQuery:
        """Process a user query into structured format. / 将用户查询处理为结构化格式。
        
        Args: / 参数：
            query: Raw user query string / query：原始用户查询字符串
            
        Returns: / 返回：
            ProcessedQuery with extracted keywords and filters / 包含提取关键词和过滤条件的 ProcessedQuery
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query or "",
                keywords=[],
                filters={}
            )
        
        # Normalize query / 规范化查询
        normalized = self._normalize(query)
        
        # Extract filters from query (e.g., "collection:docs") / 从查询中提取过滤条件（例如 "collection:docs"）
        filters, query_without_filters = self._extract_filters(normalized)
        
        # Tokenize and extract keywords / 分词并提取关键词
        tokens = self._tokenize(query_without_filters)
        
        # Filter stopwords and apply constraints / 过滤停用词并应用约束
        keywords = self._filter_keywords(tokens)
        
        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=filters
        )
    
    def _normalize(self, query: str) -> str:
        """Normalize query string. / 规范化查询字符串。
        
        - Strip whitespace / 去除空白
        - Normalize unicode / 规范化 Unicode
        - Convert to consistent format / 转换为一致格式
        
        Args: / 参数：
            query: Raw query string / query：原始查询字符串
            
        Returns: / 返回：
            Normalized query string / 规范化后的查询字符串
        """
        # Strip and normalize whitespace / 去除并规范化空白
        normalized = " ".join(query.split())
        return normalized
    
    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """Extract filter syntax from query. / 从查询中提取过滤语法。
        
        Supports syntax like: "collection:api-docs keyword1 keyword2" / 支持类似 "collection:api-docs keyword1 keyword2" 的语法
        
        Args: / 参数：
            query: Normalized query string / query：规范化后的查询字符串
            
        Returns: / 返回：
            Tuple of (filters dict, query without filter syntax) / 二元组：（过滤条件字典，移除过滤语法后的查询）
        """
        if not self.config.enable_filter_parsing:
            return {}, query
        
        filters: Dict[str, Any] = {}
        
        # Find all filter patterns / 查找所有过滤模式
        matches = FILTER_PATTERN.findall(query)
        for key, value in matches:
            # Support common filter keys / 支持常见过滤键
            key_lower = key.lower()
            if key_lower in ("collection", "col", "c"):
                filters["collection"] = value
            elif key_lower in ("type", "doc_type", "t"):
                filters["doc_type"] = value
            elif key_lower in ("source", "src", "s"):
                filters["source_path"] = value
            elif key_lower in ("tag", "tags"):
                # Tags can be comma-separated / 标签可以用逗号分隔
                if "tags" not in filters:
                    filters["tags"] = []
                filters["tags"].extend(value.split(","))
            else:
                # Generic filter / 通用过滤条件
                filters[key] = value
        
        # Remove filter patterns from query / 从查询中移除过滤模式
        query_without_filters = FILTER_PATTERN.sub("", query).strip()
        query_without_filters = " ".join(query_without_filters.split())
        
        return filters, query_without_filters
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/terms. / 将文本切分为词或术语。
        
        Uses jieba for Chinese text segmentation, consistent with the / 使用 jieba 进行中文分词，并与索引侧分词器
        index-side tokenizer (SparseEncoder) so BM25 matching works. / （SparseEncoder）保持一致，以保证 BM25 匹配可用。
        English text is handled natively by jieba (preserved as-is). / 英文文本由 jieba 原生处理（保持原样）。
        
        Args: / 参数：
            text: Text to tokenize / text：待分词文本
            
        Returns: / 返回：
            List of tokens / token 列表
        """
        tokens: List[str] = []

        # Use jieba to segment (handles Chinese + keeps English intact) / 使用 jieba 分词（处理中文并保持英文完整）
        raw_tokens = jieba.lcut(text)

        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # Skip pure punctuation / whitespace / 跳过纯标点或空白
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                continue
            tokens.append(token)
        
        return tokens
    
    def _filter_keywords(self, tokens: List[str]) -> List[str]:
        """Filter tokens to get meaningful keywords. / 过滤 token 以获得有意义的关键词。
        
        - Remove stopwords / 移除停用词
        - Apply minimum length constraint / 应用最小长度约束
        - Deduplicate while preserving order / 在保留顺序的同时去重
        - Apply maximum count limit / 应用最大数量限制
        
        Args: / 参数：
            tokens: List of tokens / tokens：token 列表
            
        Returns: / 返回：
            List of filtered keywords / 过滤后的关键词列表
        """
        seen: Set[str] = set()
        keywords: List[str] = []
        
        for token in tokens:
            # Normalize for comparison / 规范化以便比较
            token_lower = token.lower()
            
            # Skip if already seen (case-insensitive dedup) / 如果已出现则跳过（大小写不敏感去重）
            if token_lower in seen:
                continue
            
            # Skip stopwords (check both original and lowercase) / 跳过停用词（同时检查原词和小写形式）
            if token in self.config.stopwords or token_lower in self.config.stopwords:
                continue
            
            # Skip if too short / 如果过短则跳过
            if len(token) < self.config.min_keyword_length:
                continue
            
            # Add keyword (preserve original case) / 添加关键词（保留原始大小写）
            seen.add(token_lower)
            keywords.append(token)
            
            # Stop if we have enough / 如果数量足够则停止
            if len(keywords) >= self.config.max_keywords:
                break
        
        return keywords
    
    def add_stopwords(self, words: Set[str]) -> None:
        """Add words to stopword set. / 向停用词集合添加词。
        
        Args: / 参数：
            words: Set of words to add / words：要添加的词集合
        """
        self.config.stopwords.update(words)
    
    def remove_stopwords(self, words: Set[str]) -> None:
        """Remove words from stopword set. / 从停用词集合中移除词。
        
        Args: / 参数：
            words: Set of words to remove / words：要移除的词集合
        """
        self.config.stopwords -= words


def create_query_processor(
    stopwords: Optional[Set[str]] = None,
    min_keyword_length: int = 1,
    max_keywords: int = 20,
    enable_filter_parsing: bool = True
) -> QueryProcessor:
    """Factory function to create QueryProcessor. / 创建 QueryProcessor 的工厂函数。
    
    Args: / 参数：
        stopwords: Custom stopwords set. Uses default if None. / stopwords：自定义停用词集合。为 None 时使用默认值。
        min_keyword_length: Minimum keyword length / min_keyword_length：最小关键词长度
        max_keywords: Maximum keywords to extract / max_keywords：最多提取的关键词数量
        enable_filter_parsing: Whether to parse filter syntax / enable_filter_parsing：是否解析过滤语法
        
    Returns: / 返回：
        Configured QueryProcessor instance / 配置完成的 QueryProcessor 实例
    """
    config = QueryProcessorConfig(
        stopwords=stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy(),
        min_keyword_length=min_keyword_length,
        max_keywords=max_keywords,
        enable_filter_parsing=enable_filter_parsing
    )
    return QueryProcessor(config)
