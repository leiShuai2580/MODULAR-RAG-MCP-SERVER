"""Core data types and contracts for the entire pipeline. / 整条流水线的核心数据类型和契约。

This module defines the fundamental data structures used across all pipeline stages: / 本模块定义所有流水线阶段使用的基础数据结构：
- ingestion (loaders, transforms, embedding, storage) / 摄取（加载器、转换、嵌入、存储）
- retrieval (query engine, search, reranking) / 检索（查询引擎、搜索、重排）
- mcp_server (tools, response formatting) / MCP 服务（工具、响应格式化）

Design Principles: / 设计原则：
- Centralized contracts: All stages use these types to avoid coupling / 集中契约：所有阶段使用这些类型以避免耦合
- Serializable: All types support dict/JSON conversion / 可序列化：所有类型都支持 dict/JSON 转换
- Extensible metadata: Minimum required fields with flexible extension / 可扩展元数据：最小必需字段加灵活扩展
- Type-safe: Full type hints for static analysis / 类型安全：完整类型提示支持静态分析
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """Represents a raw document loaded from source. / 表示从来源加载的原始文档。
    
    This is the output of Loaders (e.g., PDF Loader) before splitting. / 这是切分前加载器（例如 PDF Loader）的输出。
    
    Attributes: / 属性：
        id: Unique identifier for the document (e.g., file hash or path-based ID) / id：文档唯一标识（例如文件哈希或基于路径的 ID）
        text: Document content in standardized Markdown format. / text：标准化 Markdown 格式的文档内容。
              Images are represented as placeholders: [IMAGE: {image_id}] / 图片表示为占位符：[IMAGE: {image_id}]
        metadata: Document-level metadata including: / metadata：文档级元数据，包括：
            - source_path (required): Original file path / source_path（必需）：原始文件路径
            - doc_type: Document type (e.g., 'pdf', 'markdown') / doc_type：文档类型（例如 'pdf'、'markdown'）
            - title: Document title extracted or inferred / title：提取或推断出的文档标题
            - page_count: Total pages (if applicable) / page_count：总页数（如适用）
            - images: List of image references (see Images Field Specification below) / images：图片引用列表（见下方图片字段规范）
            - Any other custom metadata / 任意其他自定义元数据
    
    Images Field Specification (metadata.images): / 图片字段规范（metadata.images）：
        Structure: List[{"id": str, "path": str, "page": int, "text_offset": int, / 结构：List[{"id": str, "path": str, "page": int, "text_offset": int,
                        "text_length": int, "position": dict}] / "text_length": int, "position": dict}]
        Fields: / 字段：
            - id: Unique image identifier (format: {doc_hash}_{page}_{seq}) / id：图片唯一标识（格式：{doc_hash}_{page}_{seq}）
            - path: Image file storage path (convention: data/images/{collection}/{image_id}.png) / path：图片文件存储路径（约定：data/images/{collection}/{image_id}.png）
            - page: Page number in original document (optional, for paginated docs like PDF) / page：原始文档中的页码（可选，用于 PDF 等分页文档）
            - text_offset: Starting character position of placeholder in Document.text (0-based) / text_offset：占位符在 Document.text 中的起始字符位置（从 0 开始）
            - text_length: Length of placeholder string (typically len("[IMAGE: {image_id}]")) / text_length：占位符字符串长度（通常为 len("[IMAGE: {image_id}]")）
            - position: Physical position info in original doc (optional, e.g., PDF coords, pixel position) / position：原始文档中的物理位置信息（可选，例如 PDF 坐标、像素位置）
        Note: text_offset and text_length enable precise placeholder location, / 说明：text_offset 和 text_length 支持精确定位占位符，
              supporting scenarios where the same image appears multiple times / 适用于同一图片多次出现的场景
    
    Example: / 示例：
        >>> doc = Document(
        ...     id="doc_abc123",
        ...     text="# Title\\n\\nContent...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "doc_type": "pdf",
        ...         "title": "Annual Report 2025"
        ...     }
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required metadata fields. / 校验必需的元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization. / 转换为字典以便序列化。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary. / 从字典创建 Document。"""
        return cls(**data)


@dataclass
class Chunk:
    """Represents a text chunk after splitting a Document. / 表示 Document 切分后的文本分块。
    
    This is the output of Splitters and input to Transform pipeline. / 这是切分器的输出，也是转换流水线的输入。
    Each chunk maintains traceability to its source document. / 每个分块都保持对源文档的可追溯性。
    
    Attributes: / 属性：
        id: Unique chunk identifier (e.g., hash-based or sequential) / id：分块唯一标识（例如基于哈希或顺序生成）
        text: Chunk content (subset of original document text). / text：分块内容（原始文档文本的子集）。
              Images are represented as placeholders: [IMAGE: {image_id}] / 图片表示为占位符：[IMAGE: {image_id}]
        metadata: Chunk-level metadata inherited and extended from Document: / metadata：从 Document 继承并扩展的分块级元数据：
            - source_path (required): Original file path / source_path（必需）：原始文件路径
            - chunk_index: Sequential position in document (0-based) / chunk_index：在文档中的顺序位置（从 0 开始）
            - start_offset: Character offset in original document (optional) / start_offset：原始文档中的字符偏移（可选）
            - end_offset: Character offset in original document (optional) / end_offset：原始文档中的结束字符偏移（可选）
            - source_ref: Reference to parent document ID (optional) / source_ref：父文档 ID 引用（可选）
            - images: Subset of Document.images that fall within this chunk (optional) / images：落在该分块范围内的 Document.images 子集（可选）
            - Any document-level metadata propagated from Document / 从 Document 传播来的任意文档级元数据
        start_offset: Starting character position in original document (optional) / start_offset：原始文档中的起始字符位置（可选）
        end_offset: Ending character position in original document (optional) / end_offset：原始文档中的结束字符位置（可选）
        source_ref: Reference to parent Document.id (optional) / source_ref：父 Document.id 引用（可选）
    
    Note: If chunk contains image placeholders, metadata.images should contain / 说明：如果分块包含图片占位符，metadata.images 应只包含
          only the image references relevant to this chunk's text range. / 与该分块文本范围相关的图片引用。
    
    Example: / 示例：
        >>> chunk = Chunk(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "page": 1
        ...     },
        ...     start_offset=0,
        ...     end_offset=150
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_ref: Optional[str] = None
    
    def __post_init__(self):
        """Validate required metadata fields. / 校验必需的元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization. / 转换为字典以便序列化。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary. / 从字典创建 Chunk。"""
        return cls(**data)


@dataclass
class ChunkRecord:
    """Represents a fully processed chunk ready for storage and retrieval. / 表示已完全处理、可用于存储和检索的分块。
    
    This is the output of the embedding pipeline and the data structure / 这是嵌入流水线的输出，也是存储在向量数据库中的
    stored in vector databases. It extends Chunk with vector representations. / 数据结构。它在 Chunk 基础上扩展了向量表示。
    
    Attributes: / 属性：
        id: Unique chunk identifier (must be stable for idempotent upsert) / id：分块唯一标识（必须稳定以支持幂等 upsert）
        text: Chunk content (same as Chunk.text). / text：分块内容（与 Chunk.text 相同）。
              Images are represented as placeholders: [IMAGE: {image_id}] / 图片表示为占位符：[IMAGE: {image_id}]
        metadata: Extended metadata including: / metadata：扩展元数据，包括：
            - source_path (required): Original file path / source_path（必需）：原始文件路径
            - chunk_index: Sequential position / chunk_index：顺序位置
            - All metadata from Chunk / Chunk 中的全部元数据
            - images: Image references from Chunk (see Document.images specification) / images：来自 Chunk 的图片引用（见 Document.images 规范）
            - Any enrichment from Transform pipeline (title, summary, tags) / 转换流水线产生的任意增强信息（title、summary、tags）
            - image_captions: Dict[image_id, caption_text] if multimodal enrichment applied / image_captions：如果应用了多模态增强，则为 Dict[image_id, caption_text]
        dense_vector: Dense embedding vector (e.g., from OpenAI, BGE) / dense_vector：稠密嵌入向量（例如来自 OpenAI、BGE）
        sparse_vector: Sparse vector for BM25/keyword matching (optional) / sparse_vector：用于 BM25/关键词匹配的稀疏向量（可选）
    
    Note: Image captions generated by ImageCaptioner are stored in metadata.image_captions / 说明：ImageCaptioner 生成的图片描述存储在 metadata.image_captions 中，
          as a dictionary mapping image_id to generated caption text. / 形式为 image_id 到生成描述文本的字典映射。
    
    Example: / 示例：
        >>> record = ChunkRecord(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "title": "Introduction",
        ...         "summary": "Overview of project goals"
        ...     },
        ...     dense_vector=[0.1, 0.2, ..., 0.3],
        ...     sparse_vector={"word1": 0.5, "word2": 0.3}
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate required metadata fields. / 校验必需的元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization. / 转换为字典以便序列化。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Create ChunkRecord from dictionary. / 从字典创建 ChunkRecord。"""
        return cls(**data)
    
    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> "ChunkRecord":
        """Create ChunkRecord from a Chunk with vectors. / 使用 Chunk 和向量创建 ChunkRecord。
        
        Args: / 参数：
            chunk: Source Chunk object / chunk：源 Chunk 对象
            dense_vector: Dense embedding vector / dense_vector：稠密嵌入向量
            sparse_vector: Sparse vector representation / sparse_vector：稀疏向量表示
            
        Returns: / 返回：
            ChunkRecord with all fields populated from chunk / 使用 chunk 填充所有字段后的 ChunkRecord
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# Type aliases for convenience / 便捷类型别名
Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]


@dataclass
class ProcessedQuery:
    """Represents a processed query ready for retrieval. / 表示已处理、可用于检索的查询。
    
    This is the output of QueryProcessor, containing extracted keywords / 这是 QueryProcessor 的输出，包含提取出的关键词
    and parsed filters for downstream Dense/Sparse retrievers. / 和解析出的过滤条件，供下游稠密/稀疏检索器使用。
    
    Attributes: / 属性：
        original_query: The raw user query string / original_query：原始用户查询字符串
        keywords: List of extracted keywords after stopword removal / keywords：移除停用词后提取出的关键词列表
        filters: Dictionary of filter conditions (e.g., {"collection": "api-docs"}) / filters：过滤条件字典（例如 {"collection": "api-docs"}）
        expanded_terms: Optional list of synonyms/expanded terms (for future use) / expanded_terms：可选的同义词/扩展词列表（供未来使用）
    
    Example: / 示例：
        >>> pq = ProcessedQuery(
        ...     original_query="如何配置 Azure OpenAI？",
        ...     keywords=["配置", "Azure", "OpenAI"],
        ...     filters={"collection": "docs"}
        ... )
    """
    
    original_query: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization. / 转换为字典以便序列化。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedQuery":
        """Create ProcessedQuery from dictionary. / 从字典创建 ProcessedQuery。"""
        return cls(**data)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result from Dense/Sparse retrievers. / 表示来自稠密/稀疏检索器的单条检索结果。
    
    This is the output of DenseRetriever, SparseRetriever, and HybridSearch, / 这是 DenseRetriever、SparseRetriever 和 HybridSearch 的输出，
    providing a unified contract for retrieval results across all search methods. / 为所有搜索方法提供统一的检索结果契约。
    
    Attributes: / 属性：
        chunk_id: Unique identifier for the retrieved chunk / chunk_id：被检索分块的唯一标识
        score: Relevance score (higher = more relevant, normalized to [0, 1]) / score：相关性分数（越高越相关，归一化到 [0, 1]）
        text: The actual text content of the retrieved chunk / text：被检索分块的实际文本内容
        metadata: Associated metadata (source_path, chunk_index, title, etc.) / metadata：关联元数据（source_path、chunk_index、title 等）
    
    Example: / 示例：
        >>> result = RetrievalResult(
        ...     chunk_id="doc1_chunk_003",
        ...     score=0.85,
        ...     text="Azure OpenAI 配置步骤如下...",
        ...     metadata={
        ...         "source_path": "docs/azure-guide.pdf",
        ...         "chunk_index": 3,
        ...         "title": "Azure Configuration"
        ...     }
        ... )
    """
    
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fields after initialization. / 初始化后校验字段。"""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be numeric, got {type(self.score).__name__}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization. / 转换为字典以便序列化。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create RetrievalResult from dictionary. / 从字典创建 RetrievalResult。"""
        return cls(**data)
