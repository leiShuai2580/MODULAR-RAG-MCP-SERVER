"""Sparse Encoder for generating BM25 term statistics from text chunks. / 用于从文本块生成 BM25 词项统计的稀疏编码器。

This module implements the Sparse Encoder component of the Ingestion Pipeline, / 此模块实现 Ingestion Pipeline 的 Sparse Encoder 组件，
responsible for extracting term statistics needed for BM25 indexing. / 负责提取 BM25 索引所需的词项统计。

Design Principles: / 设计原则：
- Stateless Processing: No internal state between encode() calls / 无状态处理：encode() 调用之间不保留内部状态
- Observable: Accepts TraceContext for future observability integration / 可观测：接收 TraceContext 以便未来集成可观测能力
- Deterministic: Same inputs produce same term statistics / 确定性：相同输入产生相同词项统计
- Clear Contracts: Well-defined output structure for downstream BM25Indexer / 清晰契约：为下游 BM25Indexer 定义清楚的输出结构
"""

from typing import List, Dict, Optional, Any
from collections import Counter
import re

import jieba

from src.core.types import Chunk


class SparseEncoder:
    """Encodes text chunks into BM25 term statistics. / 将文本块编码为 BM25 词项统计。
    
    This encoder prepares term-level statistics needed for BM25 indexing. / 此编码器准备 BM25 索引所需的词项级统计。
    The actual index construction is handled by BM25Indexer (C12). / 实际索引构建由 BM25Indexer（C12）处理。
    
    Output Structure: / 输出结构：
        For each chunk, produces: / 对每个块，生成：
        {
            "chunk_id": str,
            "term_frequencies": Dict[str, int],  # term -> count in this chunk / 词项 -> 此块中的计数
            "doc_length": int,                    # number of terms in chunk / 块中的词项数量
            "unique_terms": int                   # vocabulary size in chunk / 块中的词表大小
        }
    
    Design: / 设计：
    - Tokenization: Simple whitespace + lowercasing (can be enhanced later) / 分词：简单空白切分 + 小写化（后续可增强）
    - Stop Words: None by default (can add in future iterations) / 停用词：默认无（可在未来迭代添加）
    - Deterministic: Same chunk text always produces same statistics / 确定性：相同块文本总是生成相同统计
    
    Example: / 示例：
        >>> from src.core.types import Chunk
        >>> encoder = SparseEncoder()
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world hello", metadata={})]
        >>> stats = encoder.encode(chunks)
        >>> stats[0]["term_frequencies"]["hello"]  # 2
        >>> stats[0]["doc_length"]  # 3
    """
    
    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """Initialize SparseEncoder. / 初始化 SparseEncoder。
        
        Args: / 参数：
            min_term_length: Minimum character length for a term (default: 2) / 词项的最小字符长度（默认：2）
            lowercase: Whether to convert terms to lowercase (default: True) / 是否将词项转为小写（默认：True）
        
        Raises: / 异常：
            ValueError: If min_term_length < 1 / 如果 min_term_length < 1
        """
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")
        
        self.min_term_length = min_term_length
        self.lowercase = lowercase
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Encode chunks into BM25 term statistics. / 将块编码为 BM25 词项统计。
        
        For each chunk, extracts: / 对每个块提取：
        - Term frequencies (term -> count) / 词频（词项 -> 计数）
        - Document length (total terms) / 文档长度（总词项数）
        - Unique terms count / 唯一词项数
        
        Args: / 参数：
            chunks: List of Chunk objects to encode / 要编码的 Chunk 对象列表
            trace: Optional TraceContext for observability (reserved for Stage F) / 用于可观测性的可选 TraceContext（为 Stage F 预留）
        
        Returns: / 返回：
            List of statistics dictionaries (one per chunk, in same order). / 统计字典列表（每个块一个，顺序相同）。
            Each dict contains: chunk_id, term_frequencies, doc_length, unique_terms / 每个字典包含：chunk_id、term_frequencies、doc_length、unique_terms
        
        Raises: / 异常：
            ValueError: If chunks list is empty / 如果 chunks 列表为空
            ValueError: If any chunk has empty text / 如果任一块文本为空
        
        Example: / 示例：
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            # Validate chunk text / 校验块文本
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )
            
            # Tokenize and count terms / 分词并统计词项
            terms = self._tokenize(chunk.text)
            term_frequencies = Counter(terms)
            
            # Build statistics dict / 构建统计字典
            stat_dict = {
                "chunk_id": chunk.id,
                "term_frequencies": dict(term_frequencies),  # Convert Counter to dict / 将 Counter 转为字典
                "doc_length": len(terms),
                "unique_terms": len(term_frequencies),
            }
            
            results.append(stat_dict)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms. / 将文本分词为词项。
        
        Uses jieba for Chinese text segmentation and regex for English. / 使用 jieba 进行中文分词，并用正则处理英文。
        This ensures consistent tokenization with the query-side / 这确保与查询侧
        (QueryProcessor), which is required for BM25 matching. / (QueryProcessor) 的分词一致，这是 BM25 匹配所必需的。
        
        Args: / 参数：
            text: Input text to tokenize / 要分词的输入文本
        
        Returns: / 返回：
            List of valid terms / 有效词项列表
        """
        tokens: List[str] = []

        # Use jieba to segment the text (handles both Chinese and English) / 使用 jieba 对文本分词（同时处理中文和英文）
        raw_tokens = jieba.lcut(text)

        # Clean tokens: keep only alphanumeric and Chinese characters / 清理 token：仅保留字母数字和中文字符
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # Skip pure punctuation / whitespace / 跳过纯标点/空白
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                continue
            tokens.append(token)
        
        # Apply lowercase if configured / 如果已配置则应用小写化
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        
        # Filter by minimum length / 按最小长度过滤
        terms = [t for t in tokens if len(t) >= self.min_term_length]
        
        return terms
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate corpus-level statistics from encoded chunks. / 从已编码块中计算语料级统计。
        
        Utility method for BM25Indexer to compute: / 供 BM25Indexer 计算以下内容的工具方法：
        - Average document length / 平均文档长度
        - Document frequency (how many docs contain each term) / 文档频率（包含每个词项的文档数）
        - Total number of documents / 文档总数
        
        Args: / 参数：
            encoded_chunks: List of statistics dicts from encode() / 来自 encode() 的统计字典列表
        
        Returns: / 返回：
            Dictionary with corpus-level statistics: / 包含语料级统计的字典：
            {
                "num_docs": int,
                "avg_doc_length": float,
                "document_frequency": Dict[str, int]  # term -> # docs containing it / 词项 -> 包含它的文档数
            }
        """
        if not encoded_chunks:
            return {
                "num_docs": 0,
                "avg_doc_length": 0.0,
                "document_frequency": {}
            }
        
        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # Calculate document frequency (DF) for each term / 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            # Each unique term in this chunk contributes 1 to DF / 此块中的每个唯一词项都会为 DF 贡献 1
            for term in chunk_stats["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }
