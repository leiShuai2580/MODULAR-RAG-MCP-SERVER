"""PDF Loader implementation using MarkItDown. / 使用 MarkItDown 的 PDF Loader 实现。

This module implements PDF parsing with image extraction support, / 此模块实现支持图片提取的 PDF 解析，
converting PDFs to standardized Markdown format with image placeholders. / 将 PDF 转换为带图片占位符的标准化 Markdown 格式。

Features: / 特性：
- Text extraction and Markdown conversion via MarkItDown / 通过 MarkItDown 提取文本并转换为 Markdown
- Image extraction and storage / 图片提取与存储
- Image placeholder insertion with metadata tracking / 插入图片占位符并追踪元数据
- Graceful degradation if image extraction fails / 图片提取失败时优雅降级
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from PIL import Image
import io

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """PDF Loader using MarkItDown for text extraction and Markdown conversion. / 使用 MarkItDown 进行文本提取和 Markdown 转换的 PDF Loader。
    
    This loader: / 此加载器：
    1. Extracts text from PDF and converts to Markdown / 从 PDF 提取文本并转换为 Markdown
    2. Extracts images and saves to data/images/{doc_hash}/ / 提取图片并保存到 data/images/{doc_hash}/
    3. Inserts image placeholders in the format [IMAGE: {image_id}] / 插入格式为 [IMAGE: {image_id}] 的图片占位符
    4. Records image metadata in Document.metadata.images / 在 Document.metadata.images 中记录图片元数据
    
    Configuration: / 配置：
        extract_images: Enable/disable image extraction (default: True) / 启用/禁用图片提取（默认：True）
        image_storage_dir: Base directory for image storage (default: data/images) / 图片存储基础目录（默认：data/images）
    
    Graceful Degradation: / 优雅降级：
        If image extraction fails, logs warning and continues with text-only parsing. / 如果图片提取失败，记录 warning 并继续纯文本解析。
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images"
    ):
        """Initialize PDF Loader. / 初始化 PDF Loader。
        
        Args: / 参数：
            extract_images: Whether to extract images from PDFs. / 是否从 PDF 中提取图片。
            image_storage_dir: Base directory for storing extracted images. / 存储提取图片的基础目录。
        """
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )
        
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()
    
    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file. / 加载并解析 PDF 文件。
        
        Args: / 参数：
            file_path: Path to the PDF file. / PDF 文件路径。
            
        Returns: / 返回：
            Document with Markdown text and metadata. / 包含 Markdown 文本和元数据的 Document。
            
        Raises: / 异常：
            FileNotFoundError: If the PDF file doesn't exist. / 如果 PDF 文件不存在。
            ValueError: If the file is not a valid PDF. / 如果文件不是有效 PDF。
            RuntimeError: If parsing fails critically. / 如果解析发生严重失败。
        """
        # Validate file / 校验文件
        path = self._validate_file(file_path)
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {path}")
        
        # Compute document hash for unique ID and image directory / 计算文档哈希，用于唯一 ID 和图片目录
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        
        # Parse PDF with MarkItDown / 使用 MarkItDown 解析 PDF
        try:
            result = self._markitdown.convert(str(path))
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e
        
        # Initialize metadata / 初始化元数据
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        
        # Extract title from first heading if available / 如果可用，从第一个标题提取标题
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title
        
        # Handle image extraction (with graceful degradation) / 处理图片提取（带优雅降级）
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )
        
        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content. / 计算文件内容的 SHA256 哈希。
        
        Args: / 参数：
            file_path: Path to file. / 文件路径。
            
        Returns: / 返回：
            Hex string of SHA256 hash. / SHA256 哈希的十六进制字符串。
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line. / 从第一个 Markdown 标题或第一个非空行提取标题。
        
        Args: / 参数：
            text: Markdown text content. / Markdown 文本内容。
            
        Returns: / 返回：
            If found, title string; otherwise None. / 找到则返回标题字符串，否则返回 None。
        """
        lines = text.split('\n')
        
        # First try to find a markdown heading / 先尝试查找 Markdown 标题
        for line in lines[:20]:  # Check first 20 lines / 检查前 20 行
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback: use first non-empty line as title / 回退：使用第一个非空行作为标题
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line
        
        return None
    
    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Extract images from PDF and insert placeholders. / 从 PDF 中提取图片并插入占位符。
        
        Uses PyMuPDF to extract images, save them to disk, and insert / 使用 PyMuPDF 提取图片、保存到磁盘，并在
        placeholders in the text content. / 文本内容中插入占位符。
        
        Args: / 参数：
            pdf_path: Path to PDF file. / PDF 文件路径。
            text_content: Extracted text content. / 已提取的文本内容。
            doc_hash: Document hash for image directory. / 用于图片目录的文档哈希。
            
        Returns: / 返回：
            Tuple of (modified_text, images_metadata_list) / (修改后的文本、图片元数据列表) 元组
        """
        if not self.extract_images:
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []
        
        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []
        
        images_metadata = []
        modified_text = text_content
        
        try:
            # Create image storage directory / 创建图片存储目录
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # Open PDF with PyMuPDF / 使用 PyMuPDF 打开 PDF
            doc = fitz.open(pdf_path)
            
            # Track text offset for placeholder insertion / 追踪文本偏移以插入占位符
            text_offset = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # Extract image / 提取图片
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Generate image ID and filename / 生成图片 ID 和文件名
                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename
                        
                        # Save image / 保存图片
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Get image dimensions / 获取图片尺寸
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0
                        
                        # Create placeholder / 创建占位符
                        placeholder = f"[IMAGE: {image_id}]"
                        
                        # Insert placeholder at end of current page's content / 在当前页面内容末尾插入占位符
                        # (simplified - in production, you'd parse page boundaries) / （简化处理 - 生产环境中应解析页面边界）
                        insert_position = len(modified_text)
                        modified_text += f"\n{placeholder}\n"
                        
                        # Convert path to be relative to project root or absolute / 将路径转换为相对于项目根目录的路径或绝对路径
                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            # If not in cwd, use absolute path / 如果不在 cwd 中，则使用绝对路径
                            relative_path = image_path.absolute()
                        
                        # Record metadata / 记录元数据
                        image_metadata = {
                            "id": image_id,
                            "path": str(relative_path),
                            "page": page_num + 1,
                            "text_offset": insert_position + 1,  # +1 for newline
                            "text_length": len(placeholder),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index
                            }
                        }
                        images_metadata.append(image_metadata)
                        
                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")
            
            return modified_text, images_metadata
            
        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            # Graceful degradation: return original text without images / 优雅降级：返回不含图片的原始文本
            return text_content, []
    
    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate unique image ID. / 生成唯一图片 ID。
        
        Args: / 参数：
            doc_hash: Document hash. / 文档哈希。
            page: Page number (0-based). / 页码（从 0 开始）。
            sequence: Image sequence on page (0-based). / 页面中的图片序号（从 0 开始）。
            
        Returns: / 返回：
            Unique image ID string. / 唯一图片 ID 字符串。
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
