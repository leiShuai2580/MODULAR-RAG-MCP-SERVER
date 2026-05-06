"""Image Captioner transform for enriching chunks with image descriptions.  / 用图片描述增强块的 Image Captioner 转换。

Performance Optimizations:  / 性能优化：
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)  / 只处理块文本中实际引用的图片（通过 [IMAGE: id] 占位符）
2. Uses caption cache to avoid redundant Vision API calls for the same image  / 使用 caption 缓存避免对同一图片重复调用 Vision API
3. Skips chunks without image references entirely  / 完全跳过没有图片引用的块
4. Parallel processing of unique images with thread-safe caching  / 借助线程安全缓存并行处理唯一图片
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Regex to find image placeholders: [IMAGE: some_id]  / 用于查找图片占位符的正则：[IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Default max parallel workers for Vision API calls  / Vision API 调用的默认最大并行工作线程数
DEFAULT_MAX_WORKERS = 3  # Lower than text LLM due to higher cost/latency  / 由于成本/延迟更高，低于文本 LLM


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.  / 使用 Vision LLM 为块中引用的图片生成说明。
    
    This transform identifies chunks containing image references, uses a Vision LLM  / 此转换会识别包含图片引用的块，使用 Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with  / 生成描述性说明，并用这些说明增强块文本/元数据，
    these captions to improve retrieval for visual content.  / 以提升视觉内容检索效果。
    
    Key Features:  / 核心特性：
    - Only processes images actually referenced in chunk text (not all images in metadata)  / 只处理块文本中实际引用的图片（不是元数据中的所有图片）
    - Caches captions to avoid redundant Vision API calls  / 缓存图片说明，避免重复调用 Vision API
    - Thread-safe caption cache for potential future parallelization  / 线程安全的说明缓存，便于未来并行化
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None
    ):
        self.settings = settings
        self.llm = None
        # Caption cache: image_id -> caption string (thread-safe with lock)  / 说明缓存：image_id -> caption 字符串（通过锁保证线程安全）
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        
        # Check if vision LLM is enabled in settings  / 检查配置中是否启用了 vision LLM
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # We don't raise here to allow pipeline to continue without captioning  / 这里不抛出异常，以允许流水线在没有图片说明的情况下继续
                 # effectively falling back to no-op for this transform  / 实际上会让此转换回退为 no-op
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration.  / 从配置中加载图片说明提示词。"""
        # Assuming standard relative path. In production, logic might be robust.  / 假设使用标准相对路径；生产环境中逻辑可能需要更健壮。
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.  / 提取块文本中实际引用的图片 ID。
        
        Args:  / 参数：
            text: Chunk text content  / 块文本内容
            
        Returns:  / 返回：
            List of image IDs found in [IMAGE: id] placeholders  / 在 [IMAGE: id] 占位符中找到的图片 ID 列表
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Get caption for an image, using cache if available. Thread-safe.  / 获取图片说明；如果可用则使用缓存。线程安全。
        
        Args:  / 参数：
            img_id: Image identifier  / 图片标识符
            img_path: Path to image file  / 图片文件路径
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            Caption string or None if failed  / 图片说明字符串；失败时返回 None
        """
        # Check cache first (thread-safe read)  / 先检查缓存（线程安全读取）
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]
        
        # Validate path  / 校验路径
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace
            )
            caption = response.content
            
            # Cache the result (thread-safe write)  / 缓存结果（线程安全写入）
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images.  / 处理块并为引用的图片添加说明。
        
        Only processes images that are actually referenced in chunk text  / 只处理块文本中实际引用的图片
        via [IMAGE: id] placeholders. Uses caching to avoid redundant API calls.  / 通过 [IMAGE: id] 占位符识别。使用缓存避免重复 API 调用。
        Parallel processing for unique images.  / 对唯一图片进行并行处理。
        """
        if not self.llm:
            return chunks
        
        # Build image lookup from all chunks' metadata  / 从所有块的元数据构建图片查找表
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # Clear cache for new document processing  / 为新文档处理清空缓存
        with self._cache_lock:
            self._caption_cache.clear()
        
        # First pass: collect all unique image IDs that need captioning  / 第一遍：收集所有需要生成说明的唯一图片 ID
        images_to_caption: Dict[str, str] = {}  # img_id -> img_path  / 图片 ID -> 图片路径
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        images_to_caption[img_id] = img_meta.get("path")
        
        # Parallel caption generation for all unique images  / 为所有唯一图片并行生成说明
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)
        
        # Second pass: apply captions to chunks  / 第二遍：将说明应用到块
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []
            
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                
                # Get caption from cache (already populated by parallel processing)  / 从缓存获取说明（已由并行处理填充）
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)
                
                if caption:
                    captions.append({"id": img_id_stripped, "caption": caption})
                    
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            chunk.text = new_text
            
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Generate captions for multiple images in parallel.  / 并行为多张图片生成说明。
        
        Args:  / 参数：
            images_to_caption: Dict of img_id -> img_path  / img_id -> img_path 的字典
            trace: Optional trace context  / 可选的追踪上下文
        """
        if not images_to_caption:
            return
        
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
