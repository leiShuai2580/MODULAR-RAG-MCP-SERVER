"""Document chunking module - adapts libs.splitter for business layer. / 文档分块模块 - 为业务层适配 libs.splitter。

This module serves as the adapter layer between libs.splitter (pure text splitting) / 此模块作为 libs.splitter（纯文本拆分）与 Ingestion Pipeline（业务对象转换）之间的适配层，
and Ingestion Pipeline (business object transformation). It transforms Document / 并将 Document
objects into Chunk objects with proper ID generation, metadata inheritance, and / 对象转换为 Chunk 对象，同时生成合适的 ID、继承元数据，
traceability. / 并保留可追溯性。

Core Value-Add (vs libs.splitter): / 核心增值（相较于 libs.splitter）：
1. Chunk ID Generation: Deterministic and unique IDs for each chunk / Chunk ID 生成：为每个块生成确定且唯一的 ID
2. Metadata Inheritance: Propagates Document metadata to all chunks / 元数据继承：将 Document 元数据传播到所有块
3. chunk_index: Records sequential position within document / chunk_index：记录在文档中的顺序位置
4. source_ref: Establishes parent-child traceability / source_ref：建立父子可追溯关系
5. Type Conversion: str → Chunk object (core.types contract) / 类型转换：str → Chunk 对象（core.types 契约）

Design Principles: / 设计原则：
- Adapter Pattern: Bridges text splitter tool with business objects / 适配器模式：连接文本拆分工具与业务对象
- Config-Driven: Uses SplitterFactory for configuration-based strategy selection / 配置驱动：使用 SplitterFactory 基于配置选择策略
- Deterministic: Same Document produces same Chunk IDs on repeat splits / 确定性：同一 Document 重复拆分会生成相同 Chunk ID
- Type-Safe: Enforces core.types.Chunk contract / 类型安全：强制遵循 core.types.Chunk 契约
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """Converts Documents into Chunks with business-level enrichment. / 将 Documents 转换为带业务级增强的 Chunks。
    
    This class wraps a text splitter (from libs) and adds business logic: / 此类包装文本拆分器（来自 libs），并添加业务逻辑：
    - Generates stable chunk IDs / 生成稳定的块 ID
    - Inherits and extends metadata / 继承并扩展元数据
    - Maintains document traceability / 维护文档可追溯性
    
    Attributes: / 属性：
        _splitter: The underlying text splitter from libs layer / libs 层底层文本拆分器
        _settings: Configuration settings for chunking behavior / 分块行为的配置设置
    
    Example: / 示例：
        >>> from src.core.settings import load_settings
        >>> from src.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="Long document content...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"First chunk ID: {chunks[0].id}")
        >>> print(f"First chunk index: {chunks[0].metadata['chunk_index']}")
    """
    
    def __init__(self, settings: Settings):
        """Initialize DocumentChunker with configuration. / 使用配置初始化 DocumentChunker。
        
        Args: / 参数：
            settings: Configuration settings containing splitter configuration. / 包含拆分器配置的配置对象。
                     The splitter config is expected at settings.splitter.* / 拆分器配置预期位于 settings.splitter.*
        
        Raises: / 异常：
            ValueError: If splitter configuration is invalid or provider unknown / 如果拆分器配置无效或 provider 未知
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
    
    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with full business enrichment. / 将 Document 拆分为带完整业务增强的 Chunks。
        
        This is the main entry point that orchestrates the transformation: / 这是编排转换的主入口：
        1. Uses underlying splitter to get text fragments / 使用底层拆分器获取文本片段
        2. Generates deterministic IDs for each chunk / 为每个块生成确定性 ID
        3. Inherits and extends metadata from document / 从文档继承并扩展元数据
        4. Creates Chunk objects conforming to core.types contract / 创建符合 core.types 契约的 Chunk 对象
        
        Args: / 参数：
            document: Source document to split into chunks / 要拆分为块的源文档
        
        Returns: / 返回：
            List of Chunk objects with: / 具备以下内容的 Chunk 对象列表：
            - Unique, deterministic IDs / 唯一且确定性的 ID
            - Inherited metadata + chunk_index + source_ref / 继承的元数据 + chunk_index + source_ref
            - Proper type contract (core.types.Chunk) / 正确的类型契约（core.types.Chunk）
        
        Raises: / 异常：
            ValueError: If document has no text or invalid structure / 如果文档没有文本或结构无效
        
        Example: / 示例：
            >>> doc = Document(
            ...     id="doc_abc",
            ...     text="Section 1 content.\\n\\nSection 2 content.",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> chunker = DocumentChunker(settings)
            >>> chunks = chunker.split_document(doc)
            >>> len(chunks) >= 1
            True
            >>> chunks[0].metadata["source_path"]
            'file.pdf'
            >>> chunks[0].metadata["chunk_index"]
            0
            >>> chunks[0].metadata["source_ref"]
            'doc_abc'
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")
        
        # Step 1: Use underlying splitter to get text fragments / 步骤 1：使用底层拆分器获取文本片段
        text_fragments = self._splitter.split_text(document.text)
        
        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )
        
        # Step 2: Transform text fragments into Chunk objects with enrichment / 步骤 2：将文本片段转换为带增强的 Chunk 对象
        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            
            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique and deterministic chunk ID. / 生成唯一且确定性的块 ID。
        
        ID format: {doc_id}_{index:04d}_{content_hash} / ID 格式：{doc_id}_{index:04d}_{content_hash}
        - doc_id: Parent document identifier / doc_id：父文档标识符
        - index: Sequential position (zero-padded to 4 digits) / index：顺序位置（补零到 4 位）
        - content_hash: First 8 chars of text SHA256 hash / content_hash：文本 SHA256 哈希的前 8 个字符
        
        This ensures: / 这确保：
        - Uniqueness: Combination of doc_id + index + content_hash / 唯一性：doc_id + index + content_hash 的组合
        - Determinism: Same input always produces same ID / 确定性：相同输入始终生成相同 ID
        - Debuggability: Human-readable structure / 可调试性：结构可读
        
        Args: / 参数：
            doc_id: Parent document ID / 父文档 ID
            index: Sequential position of chunk (0-based) / 块的顺序位置（从 0 开始）
            text: Chunk text content / 块文本内容
        
        Returns: / 返回：
            Unique chunk ID string / 唯一块 ID 字符串
        
        Example: / 示例：
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # Compute content hash for uniqueness / 计算内容哈希以保证唯一性
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        
        # Format: {doc_id}_{index:04d}_{hash_8chars} / 格式：{doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        """Inherit metadata from document and add chunk-specific fields. / 从文档继承元数据并添加块特有字段。
        
        This creates a new metadata dict containing: / 这会创建一个新的元数据字典，包含：
        - All fields from document.metadata (copied, not referenced) / document.metadata 中的所有字段（复制而非引用）
        - chunk_index: Sequential position (0-based) / chunk_index：顺序位置（从 0 开始）
        - source_ref: Reference to parent document ID / source_ref：对父文档 ID 的引用
        - image_refs: List of image IDs referenced in this chunk (extracted from placeholders) / image_refs：此块中引用的图片 ID 列表（从占位符提取）
        
        Note: The document-level 'images' field is intentionally excluded from chunk / 注意：文档级的 'images' 字段会被有意从块元数据中排除，
        metadata as it would be redundant. Instead, chunk-specific 'image_refs' is / 因为它是冗余的。相反，块特有的 'image_refs'
        populated based on [IMAGE: xxx] placeholders found in the chunk text. / 会基于块文本中的 [IMAGE: xxx] 占位符填充。
        
        Args: / 参数：
            document: Source document whose metadata to inherit / 要继承其元数据的源文档
            chunk_index: Sequential position of this chunk / 此块的顺序位置
            chunk_text: The text content of this chunk (used to extract image_refs) / 此块的文本内容（用于提取 image_refs）
        
        Returns: / 返回：
            Metadata dict with inherited and chunk-specific fields / 包含继承字段和块特有字段的元数据字典
        
        Example: / 示例：
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="Content",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "See [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        import re
        
        # Copy all document metadata (shallow copy is sufficient for primitives) / 复制所有文档元数据（对基础类型浅拷贝已足够）
        chunk_metadata = document.metadata.copy()
        
        # Get document-level images for lookup / 获取文档级图片以便查找
        doc_images = document.metadata.get("images", [])
        
        # Remove document-level 'images' field - we'll add chunk-specific images below / 移除文档级 'images' 字段，下面会添加块特有图片
        chunk_metadata.pop("images", None)
        
        # Add chunk-specific fields / 添加块特有字段
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id
        
        # Extract image_refs from chunk text by finding [IMAGE: xxx] placeholders / 通过查找 [IMAGE: xxx] 占位符从块文本提取 image_refs
        image_refs = []
        if chunk_text:
            # Pattern matches [IMAGE: image_id] placeholders / 模式匹配 [IMAGE: image_id] 占位符
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]
        
        chunk_metadata["image_refs"] = image_refs
        
        # Build chunk-specific 'images' list with full metadata for referenced images / 为引用的图片构建带完整元数据的块特有 'images' 列表
        # This is needed by ImageCaptioner to access image paths for Vision API calls / ImageCaptioner 需要它来访问图片路径以调用 Vision API
        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])
        
        if chunk_images:
            chunk_metadata["images"] = chunk_images
        
        # Try to determine page_num from the first referenced image / 尝试从第一张引用图片确定 page_num
        if chunk_images:
            chunk_metadata["page_num"] = chunk_images[0].get("page")
        
        return chunk_metadata
