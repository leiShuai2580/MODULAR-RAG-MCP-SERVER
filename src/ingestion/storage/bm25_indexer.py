"""BM25 Indexer for building and querying inverted indexes. / 用于构建和查询倒排索引的 BM25 索引器。

This module implements the BM25 indexing component, responsible for: / 本模块实现 BM25 索引组件，负责：
- Computing IDF (Inverse Document Frequency) scores / 计算 IDF（逆文档频率）分数
- Building inverted index structures / 构建倒排索引结构
- Persisting and loading indexes from disk / 将索引持久化到磁盘并从磁盘加载
- Supporting incremental updates / 支持增量更新

Design Principles: / 设计原则：
- Idempotent: Rebuild produces same results for same input / 幂等：相同输入重建会产生相同结果
- Observable: Accepts TraceContext for future integration / 可观测：接收 TraceContext 以便未来集成
- Persistent: Indexes saved to data/db/bm25/ directory / 持久化：索引保存到 data/db/bm25/ 目录
- Deterministic: Same corpus produces same IDF scores / 确定性：相同语料产生相同 IDF 分数
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class BM25Indexer:
    """Build and query BM25 inverted indexes. / 构建和查询 BM25 倒排索引。
    
    This indexer receives term statistics from SparseEncoder and constructs / 该索引器接收 SparseEncoder 输出的词项统计，
    a queryable BM25 index with IDF scores and posting lists. / 构建包含 IDF 分数和倒排列表的可查询 BM25 索引。
    
    Index Structure: / 索引结构：
        {
            "metadata": {
                "num_docs": int,
                "avg_doc_length": float,
                "total_terms": int
            },
            "index": {
                "term": {
                    "idf": float,
                    "df": int,  # document frequency / 文档频率
                    "postings": [
                        {"chunk_id": str, "tf": int, "doc_length": int},
                        ...
                    ]
                },
                ...
            }
        }
    
    BM25 IDF Formula: / BM25 IDF 公式：
        IDF(term) = log((N - df + 0.5) / (df + 0.5))
        
        Where: / 其中：
        - N = total number of documents / N = 文档总数
        - df = document frequency (number of docs containing term) / df = 文档频率（包含该词项的文档数量）
    
    Example: / 示例：
        >>> indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> 
        >>> # Build index from SparseEncoder output / 从 SparseEncoder 输出构建索引
        >>> term_stats = [
        ...     {"chunk_id": "1", "term_frequencies": {"hello": 2, "world": 1}, "doc_length": 3},
        ...     {"chunk_id": "2", "term_frequencies": {"hello": 1, "python": 1}, "doc_length": 2}
        ... ]
        >>> indexer.build(term_stats)
        >>> 
        >>> # Query the index / 查询索引
        >>> results = indexer.query(["hello"], top_k=2)
        >>> len(results) <= 2  # True
    """
    
    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25Indexer. / 初始化 BM25Indexer。
        
        Args: / 参数：
            index_dir: Directory to store index files (default: data/db/bm25) / index_dir：索引文件存储目录（默认：data/db/bm25）
            k1: BM25 term frequency saturation parameter (default: 1.5) / k1：BM25 词频饱和参数（默认：1.5）
            b: BM25 length normalization parameter (default: 0.75) / b：BM25 长度归一化参数（默认：0.75）
        
        Raises: / 异常：
            ValueError: If k1 or b are out of valid ranges / ValueError：当 k1 或 b 超出有效范围时抛出
        """
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")
        
        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b
        
        # In-memory index structure / 内存中的索引结构
        self._index: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}
        
    def build(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        """Build BM25 index from term statistics. / 根据词项统计构建 BM25 索引。
        
        This method: / 该方法：
        1. Calculates corpus-level statistics (N, avg_doc_length, DF) / 1. 计算语料级统计（N、avg_doc_length、DF）
        2. Computes IDF for each term / 2. 为每个词项计算 IDF
        3. Builds inverted index with posting lists / 3. 构建带倒排列表的倒排索引
        4. Persists to disk / 4. 持久化到磁盘
        
        Args: / 参数：
            term_stats: List of statistics from SparseEncoder.encode() / term_stats：来自 SparseEncoder.encode() 的统计信息列表
                Each item should have: chunk_id, term_frequencies, doc_length / 每项应包含：chunk_id、term_frequencies、doc_length
            collection: Collection name for organizing indexes (default: "default") / collection：用于组织索引的集合名称（默认："default"）
            trace: Optional TraceContext for observability / trace：用于可观测性的可选 TraceContext
        
        Raises: / 异常：
            ValueError: If term_stats is empty or has invalid structure / ValueError：当 term_stats 为空或结构无效时抛出
        
        Example:
            >>> term_stats = [
            ...     {
            ...         "chunk_id": "doc1_chunk0",
            ...         "term_frequencies": {"machine": 2, "learning": 1},
            ...         "doc_length": 3
            ...     }
            ... ]
            >>> indexer.build(term_stats, collection="my_docs")
        """
        if not term_stats:
            raise ValueError("Cannot build index from empty term_stats")
        
        # Validate structure / 校验结构
        self._validate_term_stats(term_stats)
        
        # Step 1: Calculate corpus-level statistics / 第 1 步：计算语料级统计
        num_docs = len(term_stats)
        total_length = sum(stat["doc_length"] for stat in term_stats)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # Calculate document frequency (DF) for each term / 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for stat in term_stats:
            for term in stat["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        # Step 2: Build inverted index with IDF / 第 2 步：构建带 IDF 的倒排索引
        index: Dict[str, Dict[str, Any]] = {}
        
        for term, df in doc_freq.items():
            # Calculate IDF using BM25 formula / 使用 BM25 公式计算 IDF
            idf = self._calculate_idf(num_docs, df)
            
            # Build posting list for this term / 为该词项构建倒排列表
            postings = []
            for stat in term_stats:
                tf = stat["term_frequencies"].get(term, 0)
                if tf > 0:  # Only include docs that contain this term / 只包含含有该词项的文档
                    postings.append({
                        "chunk_id": stat["chunk_id"],
                        "tf": tf,
                        "doc_length": stat["doc_length"]
                    })
            
            index[term] = {
                "idf": idf,
                "df": df,
                "postings": postings
            }
        
        # Step 3: Store metadata / 第 3 步：存储元数据
        self._metadata = {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "total_terms": len(index),
            "collection": collection,
        }
        
        self._index = index
        
        # Step 4: Persist to disk / 第 4 步：持久化到磁盘
        self._save(collection)
    
    def load(
        self,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> bool:
        """Load index from disk. / 从磁盘加载索引。
        
        Args:
            collection: Collection name to load / collection：要加载的集合名称
            trace: Optional TraceContext for observability / trace：用于可观测性的可选 TraceContext
        
        Returns:
            True if index loaded successfully, False if not found / 索引成功加载时返回 True，未找到时返回 False
        
        Raises:
            ValueError: If index file is corrupted / ValueError：当索引文件损坏时抛出
        """
        index_path = self._get_index_path(collection)
        
        if not index_path.exists():
            return False
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure / 校验结构
            if "metadata" not in data or "index" not in data:
                raise ValueError(f"Invalid index file structure: missing metadata or index")
            
            self._metadata = data["metadata"]
            self._index = data["index"]
            
            return True
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted index file at {index_path}: {e}")
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Query the index using BM25 scoring. / 使用 BM25 评分查询索引。
        
        Args:
            query_terms: List of terms to search for / query_terms：要搜索的词项列表
            top_k: Maximum number of results to return / top_k：最大返回结果数量
            trace: Optional TraceContext for observability / trace：用于可观测性的可选 TraceContext
        
        Returns:
            List of results sorted by BM25 score (descending). / 按 BM25 分数降序排序的结果列表。
            Each result: {"chunk_id": str, "score": float} / 每个结果：{"chunk_id": str, "score": float}
        
        Raises:
            ValueError: If index not loaded or query_terms empty / ValueError：当索引未加载或 query_terms 为空时抛出
        
        Example:
            >>> indexer.load("my_docs")
            >>> results = indexer.query(["machine", "learning"], top_k=5)
            >>> results[0]["score"] > 0  # True if matches found
        """
        if not self._index:
            raise ValueError("Index not loaded. Call load() or build() first.")
        
        if not query_terms:
            raise ValueError("query_terms cannot be empty")
        
        # Lowercase query terms to match index (SparseEncoder lowercases during build) / 将查询词项小写化以匹配索引（SparseEncoder 构建时会小写化）
        query_terms = [t.lower() for t in query_terms]
        
        # Calculate BM25 scores for all documents / 为所有文档计算 BM25 分数
        scores: Dict[str, float] = {}
        
        for term in query_terms:
            if term not in self._index:
                continue  # Term not in corpus, skip / 词项不在语料中，跳过
            
            term_data = self._index[term]
            idf = term_data["idf"]
            
            for posting in term_data["postings"]:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]
                doc_length = posting["doc_length"]
                
                # BM25 score contribution from this term / 该词项的 BM25 分数贡献
                term_score = self._calculate_bm25_score(
                    tf=tf,
                    doc_length=doc_length,
                    avg_doc_length=self._metadata["avg_doc_length"],
                    idf=idf
                )
                
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score
        
        # Sort by score descending and return top_k / 按分数降序排序并返回 top_k
        sorted_results = sorted(
            [{"chunk_id": cid, "score": score} for cid, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def rebuild(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        """Rebuild index from scratch (alias for build with clear intent). / 从零重建索引（语义更清晰的 build 别名）。
        
        This is a convenience method that makes the intent clear when / 这是一个便捷方法，在替换已有索引时
        replacing an existing index. / 可以更清楚地表达意图。
        
        Args:
            term_stats: List of statistics from SparseEncoder / term_stats：来自 SparseEncoder 的统计信息列表
            collection: Collection name / collection：集合名称
            trace: Optional TraceContext for observability / trace：用于可观测性的可选 TraceContext
        """
        self.build(term_stats, collection, trace)

    def add_documents(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        doc_id: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> None:
        """Incrementally add documents to the BM25 index. / 向 BM25 索引中增量添加文档。

        Loads the existing index (if any), optionally removes old postings / 加载已有索引（如有），可选移除给定 *doc_id*
        for the given *doc_id* (to support re-ingestion), merges the new / 的旧倒排记录（以支持重新摄取），合并新的
        term stats, recomputes IDF scores, and saves. / 词项统计，重新计算 IDF 分数并保存。

        Args:
            term_stats: New term statistics from SparseEncoder.encode(). / term_stats：来自 SparseEncoder.encode() 的新词项统计。
            collection: Collection name. / collection：集合名称。
            doc_id: If provided, remove existing postings whose chunk_id / doc_id：如果提供，则在添加新记录前移除 chunk_id
                starts with this prefix before adding new ones (idempotent / 以此前缀开头的已有倒排记录（幂等
                re-ingestion). / 重新摄取）。
            trace: Optional TraceContext. / trace：可选 TraceContext。
        """
        if not term_stats:
            return

        self._validate_term_stats(term_stats)

        # Load existing index (ignore if missing – will start fresh) / 加载已有索引（缺失则忽略，将从空索引开始）
        if not self._index:
            self.load(collection)

        # Remove stale postings for this document (re-ingest case) / 移除该文档的过期倒排记录（重新摄取场景）
        if doc_id and self._index:
            self.remove_document(doc_id, collection)

        # Reconstruct existing term_stats from current index postings / 从当前索引倒排记录中重建已有 term_stats
        existing_stats: Dict[str, Dict[str, Any]] = {}  # chunk_id -> stat / chunk_id -> 统计
        for term, term_data in self._index.items():
            for posting in term_data["postings"]:
                cid = posting["chunk_id"]
                if cid not in existing_stats:
                    existing_stats[cid] = {
                        "chunk_id": cid,
                        "term_frequencies": {},
                        "doc_length": posting["doc_length"],
                    }
                existing_stats[cid]["term_frequencies"][term] = posting["tf"]

        # Merge: existing + new / 合并：已有 + 新增
        combined = list(existing_stats.values()) + list(term_stats)

        # Rebuild full index from combined stats / 根据合并后的统计重建完整索引
        self.build(combined, collection, trace)

    def remove_document(
        self,
        doc_id: str,
        collection: str = "default",
    ) -> bool:
        """Remove all postings for a document from the BM25 index. / 从 BM25 索引中移除某文档的所有倒排记录。

        Loads the index (if not already loaded), removes any postings / 加载索引（如果尚未加载），移除
        whose ``chunk_id`` starts with *doc_id*, recalculates statistics, / ``chunk_id`` 以 *doc_id* 开头的所有倒排记录，重新计算统计信息，
        and re-saves the index. / 并重新保存索引。

        Args:
            doc_id: Document identifier (or prefix).  All postings whose / doc_id：文档标识（或前缀）。所有
                ``chunk_id`` starts with this value are removed. / ``chunk_id`` 以该值开头的倒排记录都会被移除。
            collection: Collection name. / collection：集合名称。

        Returns:
            ``True`` if any postings were removed, ``False`` otherwise. / 如果移除了任意倒排记录则返回 ``True``，否则返回 ``False``。
        """
        if not self._index:
            if not self.load(collection):
                return False

        removed_any = False
        terms_to_delete: list[str] = []

        for term, term_data in self._index.items():
            original_len = len(term_data["postings"])
            term_data["postings"] = [
                p for p in term_data["postings"]
                if not p["chunk_id"].startswith(doc_id)
            ]
            if len(term_data["postings"]) < original_len:
                removed_any = True
            # Mark empty terms for cleanup / 标记空词项以便清理
            if not term_data["postings"]:
                terms_to_delete.append(term)
            else:
                term_data["df"] = len(term_data["postings"])

        # Remove empty terms / 移除空词项
        for term in terms_to_delete:
            del self._index[term]

        if removed_any:
            # Recalculate global metadata / 重新计算全局元数据
            all_chunk_ids: set[str] = set()
            total_length = 0
            for td in self._index.values():
                for p in td["postings"]:
                    all_chunk_ids.add(p["chunk_id"])
                    total_length += p["doc_length"]

            num_docs = len(all_chunk_ids)
            avg_doc_length = total_length / num_docs if num_docs else 0.0

            # Recalculate IDF values / 重新计算 IDF 值
            for td in self._index.values():
                td["idf"] = self._calculate_idf(num_docs, td["df"])

            self._metadata = {
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "total_terms": len(self._index),
                "collection": collection,
            }
            self._save(collection)

        return removed_any
    
    # ===== Private Helper Methods ===== / ===== 私有辅助方法 =====
    
    def _calculate_idf(self, num_docs: int, df: int) -> float:
        """Calculate IDF using BM25 formula. / 使用 BM25 公式计算 IDF。
        
        Formula: IDF(term) = log((N - df + 0.5) / (df + 0.5)) / 公式：IDF(term) = log((N - df + 0.5) / (df + 0.5))
        
        Args:
            num_docs: Total number of documents in corpus / num_docs：语料中的文档总数
            df: Document frequency (number of docs containing term) / df：文档频率（包含该词项的文档数量）
        
        Returns:
            IDF score (can be negative for very common terms) / IDF 分数（非常常见的词项可能为负）
        """
        return math.log((num_docs - df + 0.5) / (df + 0.5))
    
    def _calculate_bm25_score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float
    ) -> float:
        """Calculate BM25 score for a single term in a document. / 计算文档中单个词项的 BM25 分数。
        
        Formula: score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))) / 公式：score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        
        Args:
            tf: Term frequency in document / tf：词项在文档中的频率
            doc_length: Length of document (number of terms) / doc_length：文档长度（词项数量）
            avg_doc_length: Average document length in corpus / avg_doc_length：语料中的平均文档长度
            idf: IDF score for this term / idf：该词项的 IDF 分数
        
        Returns:
            BM25 score contribution / BM25 分数贡献
        """
        # Avoid division by zero / 避免除零
        if avg_doc_length == 0:
            avg_doc_length = 1.0
        
        # BM25 formula / BM25 公式
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def _validate_term_stats(self, term_stats: List[Dict[str, Any]]) -> None:
        """Validate term_stats structure. / 校验 term_stats 结构。
        
        Raises:
            ValueError: If structure is invalid / ValueError：当结构无效时抛出
        """
        for i, stat in enumerate(term_stats):
            if not isinstance(stat, dict):
                raise ValueError(f"term_stats[{i}] must be a dict, got {type(stat)}")
            
            required_fields = ["chunk_id", "term_frequencies", "doc_length"]
            for field in required_fields:
                if field not in stat:
                    raise ValueError(f"term_stats[{i}] missing required field: {field}")
            
            if not isinstance(stat["term_frequencies"], dict):
                raise ValueError(
                    f"term_stats[{i}]['term_frequencies'] must be dict, "
                    f"got {type(stat['term_frequencies'])}"
                )
            
            if not isinstance(stat["doc_length"], int) or stat["doc_length"] < 0:
                raise ValueError(
                    f"term_stats[{i}]['doc_length'] must be non-negative int, "
                    f"got {stat['doc_length']}"
                )
    
    def _get_index_path(self, collection: str) -> Path:
        """Get file path for index file. / 获取索引文件路径。
        
        Args:
            collection: Collection name / collection：集合名称
        
        Returns:
            Path to index file / 索引文件路径
        """
        return self.index_dir / f"{collection}_bm25.json"
    
    def _save(self, collection: str) -> None:
        """Save index to disk. / 将索引保存到磁盘。
        
        Args:
            collection: Collection name / collection：集合名称
        """
        # Ensure directory exists / 确保目录存在
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = self._get_index_path(collection)
        
        # Prepare data / 准备数据
        data = {
            "metadata": self._metadata,
            "index": self._index
        }
        
        # Write atomically (write to temp file, then rename) / 原子写入（先写临时文件，再重命名）
        temp_path = index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename / 原子重命名
            temp_path.replace(index_path)
            
        except Exception as e:
            # Clean up temp file if write failed / 写入失败时清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise
