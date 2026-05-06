"""Metadata enrichment transform: rule-based + optional LLM enhancement.  / 元数据增强转换：基于规则 + 可选的 LLM 增强。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Default max parallel workers for LLM calls  / 默认的 LLM 调用最大并行工作线程数
DEFAULT_MAX_WORKERS = 5


class MetadataEnricher(BaseTransform):
    """Enriches chunk metadata with title, summary, and tags.  / 使用标题、摘要和标签增强块元数据。
    
    Processing Pipeline:  / 处理流程：
        1. Rule-based enrichment: Extract basic metadata from content  / 基于规则增强：从内容中提取基础元数据
        2. (Optional) LLM enrichment: Generate semantic-rich metadata  / （可选）LLM 增强：生成语义更丰富的元数据
        3. On LLM failure: Gracefully fallback to rule-based metadata  / LLM 失败时：优雅降级到基于规则的元数据
    
    Output Metadata:  / 输出元数据：
        - title: Brief title/heading for the chunk  / 块的简短标题/标题行
        - summary: Concise summary of the content  / 内容的简要摘要
        - tags: List of relevant keywords/topics  / 相关关键词/主题列表
        - enriched_by: "rule" or "llm"  / “rule” 或 “llm”
    
    Configuration (via settings.yaml):  / 配置（通过 settings.yaml）：
        - ingestion.metadata_enricher.use_llm: bool - Enable LLM enhancement  / 启用 LLM 增强
        - ingestion.metadata_enricher.prompt_path: str - Custom prompt file path  / 自定义提示词文件路径
    
    Design Principles:  / 设计原则：
        - Graceful Degradation: LLM errors don't block ingestion  / 优雅降级：LLM 错误不会阻塞摄取
        - Atomic Processing: Each chunk processed independently  / 原子处理：每个块独立处理
        - Observable: Records enriched_by in metadata  / 可观测：在元数据中记录 enriched_by
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """Initialize MetadataEnricher.  / 初始化 MetadataEnricher。
        
        Args:  / 参数：
            settings: Application settings  / 应用配置
            llm: Optional LLM instance (for testing; auto-created if None)  / 可选的 LLM 实例（用于测试；为 None 时自动创建）
            prompt_path: Optional custom prompt file path  / 可选的自定义提示词文件路径
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/metadata_enrichment.txt"))
        
        # Determine if LLM should be used  / 判断是否应使用 LLM
        enricher_config = {}
        if hasattr(settings, 'ingestion') and settings.ingestion is not None:
            ingestion_config = settings.ingestion
            # Check if ingestion has metadata_enricher attribute (dataclass) or dict  / 检查 ingestion 是否包含 metadata_enricher 属性（dataclass）或字典项
            if hasattr(ingestion_config, 'metadata_enricher') and ingestion_config.metadata_enricher:
                enricher_config = ingestion_config.metadata_enricher
            elif isinstance(ingestion_config, dict):
                enricher_config = ingestion_config.get('metadata_enricher', {})
        
        self.use_llm = enricher_config.get('use_llm', False) if enricher_config else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance.  / 延迟加载 LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for metadata enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform chunks by enriching their metadata.  / 通过增强元数据来转换块。
        
        Args:  / 参数：
            chunks: List of chunks to enrich  / 要增强的块列表
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            List of enriched chunks (same length as input)  / 增强后的块列表（长度与输入相同）
        """
        if not chunks:
            return []
        
        # Process chunks in parallel if LLM is enabled  / 如果启用了 LLM，则并行处理块
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _enrich_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """Enrich a single chunk. Thread-safe.  / 增强单个块。线程安全。
        
        Args:  / 参数：
            chunk: Chunk to enrich  / 要增强的块
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            Tuple of (enriched_chunk, enriched_by, error_message)  / (增强后的块、增强方式、错误信息) 元组
        """
        try:
            # Step 1: Rule-based enrichment  / 步骤 1：基于规则增强
            rule_metadata = self._rule_based_enrich(chunk.text)
            
            # Step 2: LLM enhancement  / 步骤 2：LLM 增强
            if self.use_llm and self.llm:
                llm_metadata = self._llm_enrich(chunk.text, trace)
                
                if llm_metadata:
                    enriched_metadata = llm_metadata
                    enriched_by = "llm"
                else:
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                    enriched_metadata['enrich_fallback_reason'] = "llm_failed"
            else:
                enriched_metadata = rule_metadata
                enriched_by = "rule"
            
            final_metadata = {
                **(chunk.metadata or {}),
                **enriched_metadata,
                'enriched_by': enriched_by
            }
            
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=final_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, enriched_by, None)
            
        except Exception as e:
            logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
            text_preview = ""
            if chunk.text:
                text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
            minimal_metadata = {
                **(chunk.metadata or {}),
                'title': 'Untitled',
                'summary': text_preview,
                'tags': [],
                'enriched_by': 'error',
                'enrich_error': str(e)
            }
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text or "",
                metadata=minimal_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks in parallel using ThreadPoolExecutor.  / 使用 ThreadPoolExecutor 并行处理块。"""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        enriched_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._enrich_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enriched_chunk, enriched_by, error = future.result()
                    enriched_chunks[idx] = enriched_chunk
                    
                    if enriched_by == "llm":
                        llm_enhanced_count += 1
                    elif enriched_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel enrichment: {e}")
                    enriched_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in enriched_chunks if c is not None)
        
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks sequentially (fallback when LLM disabled).  / 顺序处理块（LLM 禁用时的回退方式）。"""
        enriched_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # Step 1: Rule-based enrichment (always performed)  / 步骤 1：基于规则增强（始终执行）
                rule_metadata = self._rule_based_enrich(chunk.text)
                
                # Step 2: Optional LLM enhancement  / 步骤 2：可选的 LLM 增强
                if self.use_llm and self.llm:
                    llm_metadata = self._llm_enrich(chunk.text, trace)
                    
                    if llm_metadata:
                        # LLM success  / LLM 成功
                        enriched_metadata = llm_metadata
                        enriched_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM failed, fallback to rule-based  / LLM 失败，回退到基于规则的结果
                        enriched_metadata = rule_metadata
                        enriched_by = "rule"
                        fallback_count += 1
                        enriched_metadata['enrich_fallback_reason'] = "llm_failed"
                else:
                    # LLM disabled, use rule-based  / LLM 已禁用，使用基于规则的结果
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                
                # Merge enriched metadata with existing metadata  / 将增强后的元数据与现有元数据合并
                final_metadata = {
                    **(chunk.metadata or {}),
                    **enriched_metadata,
                    'enriched_by': enriched_by
                }
                
                # Create enriched chunk  / 创建增强后的块
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=final_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
                success_count += 1
                
            except Exception as e:
                # Atomic failure: log and preserve original with minimal metadata  / 原子失败：记录日志并用最小元数据保留原始内容
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # Handle None text case  / 处理文本为 None 的情况
                text_preview = ""
                if chunk.text:
                    text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
                minimal_metadata = {
                    **(chunk.metadata or {}),
                    'title': 'Untitled',
                    'summary': text_preview,
                    'tags': [],
                    'enriched_by': 'error',
                    'enrich_error': str(e)
                }
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text or "",  # Ensure text is not None  / 确保 text 不是 None
                    metadata=minimal_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
        
        # Record trace  / 记录追踪信息
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """Extract metadata using rule-based heuristics.  / 使用基于规则的启发式方法提取元数据。
        
        Args:  / 参数：
            text: Chunk text content  / 块文本内容
            
        Returns:  / 返回：
            Dictionary with title, summary, tags  / 包含标题、摘要和标签的字典
            
        Raises:  / 异常：
            TypeError: If text is None  / 如果 text 为 None
        """
        if text is None:
            raise TypeError("Chunk text cannot be None")
        
        # Extract title from first heading or first line  / 从第一个标题或第一行提取标题
        title = self._extract_title(text)
        
        # Generate summary from first sentences  / 从前几个句子生成摘要
        summary = self._extract_summary(text)
        
        # Extract tags from common patterns  / 从常见模式提取标签
        tags = self._extract_tags(text)
        
        return {
            'title': title,
            'summary': summary,
            'tags': tags
        }
    
    def _extract_title(self, text: str) -> str:
        """Extract title from text using heuristics.  / 使用启发式方法从文本中提取标题。
        
        Priority:  / 优先级：
            1. Markdown heading (# Title)  / Markdown 标题（# Title）
            2. First line if short enough  / 如果第一行足够短则使用第一行
            3. First sentence  / 第一句
            4. First N characters  / 前 N 个字符
        """
        if not text:
            return "Untitled"
        
        # Check for markdown heading  / 检查 Markdown 标题
        heading_match = re.match(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        # Use first line if it's short and looks like a title  / 如果第一行较短且看起来像标题，则使用第一行
        first_line = text.split('\n')[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith(('.', ',', ';')):
            return first_line
        
        # Use first sentence (without trailing punctuation)  / 使用第一句（不含末尾标点）
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0]:
            title = sentences[0].strip()
            # Remove trailing punctuation if present  / 如果存在末尾标点则移除
            title = re.sub(r'[.!?]+$', '', title)
            if len(title) <= 150:
                return title
            return title[:147] + "..."
        
        # Fallback: first 100 chars  / 回退：前 100 个字符
        return text[:100].strip() + ("..." if len(text) > 100 else "")
    
    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """Extract summary from text using first N sentences.  / 使用前 N 个句子从文本中提取摘要。
        
        Args:  / 参数：
            text: Source text  / 源文本
            max_sentences: Maximum number of sentences to include  / 要包含的最大句子数
            
        Returns:  / 返回：
            Summary text  / 摘要文本
        """
        if not text:
            return ""
        
        # Split into sentences  / 拆分为句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Take first N sentences  / 取前 N 个句子
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences).strip()
        
        # Limit length  / 限制长度
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def _extract_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """Extract tags using keyword extraction heuristics.  / 使用关键词提取启发式方法提取标签。
        
        Args:  / 参数：
            text: Source text  / 源文本
            max_tags: Maximum number of tags to extract  / 要提取的最大标签数
            
        Returns:  / 返回：
            List of tag strings  / 标签字符串列表
        """
        if not text:
            return []
        
        tags = set()
        
        # Extract capitalized words (potential proper nouns)  / 提取首字母大写词（可能是专有名词）
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        tags.update(capitalized[:5])
        
        # Extract code identifiers (camelCase, snake_case)  / 提取代码标识符（camelCase、snake_case）
        identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b', text)
        tags.update(identifiers[:5])
        
        # Extract markdown bold/italic terms (potential keywords)  / 提取 Markdown 加粗/斜体词条（可能是关键词）
        markdown_keywords = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', text)
        for match in markdown_keywords[:5]:
            for group in match:
                if group:
                    tags.add(group.strip())
        
        # Convert to list and limit  / 转换为列表并限制数量
        tag_list = sorted(list(tags))[:max_tags]
        
        return tag_list
    
    def _llm_enrich(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """Enrich metadata using LLM.  / 使用 LLM 增强元数据。
        
        Args:  / 参数：
            text: Chunk text content  / 块文本内容
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            Dictionary with title, summary, tags, or None on failure  / 包含标题、摘要和标签的字典；失败时返回 None
        """
        if not self.llm:
            return None
        
        try:
            # Load prompt template  / 加载提示词模板
            prompt = self._load_prompt()
            
            # Build prompt with text  / 使用文本构建提示词
            formatted_prompt = prompt.replace("{chunk_text}", text[:2000])  # Limit text length  / 限制文本长度
            
            # Call LLM  / 调用 LLM
            messages = [Message(role="user", content=formatted_prompt)]
            response = self.llm.chat(messages)
            
            if not response:
                logger.warning("LLM returned empty response for metadata enrichment")
                return None
            
            # Extract text from response (handle both string and ChatResponse object)  / 从响应中提取文本（同时处理字符串和 ChatResponse 对象）
            response_text = response
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif not isinstance(response, str):
                response_text = str(response)
            
            # Parse LLM response  / 解析 LLM 响应
            metadata = self._parse_llm_response(response_text)
            
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": True,
                    "response_length": len(response_text)
                })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": False,
                    "error": str(e)
                })
            return None
    
    def _load_prompt(self) -> str:
        """Load prompt template from file.  / 从文件加载提示词模板。
        
        Returns:  / 返回：
            Prompt template string  / 提示词模板字符串
            
        Raises:  / 异常：
            FileNotFoundError: If prompt file doesn't exist  / 如果提示词文件不存在
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        prompt_path = Path(self._prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self._prompt_path}")
        
        self._prompt_template = prompt_path.read_text(encoding='utf-8')
        logger.info(f"Loaded metadata enrichment prompt from {self._prompt_path}")
        
        return self._prompt_template
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured metadata.  / 将 LLM 响应解析为结构化元数据。
        
        Expected format:  / 预期格式：
            Title: <title>  / 标题：<标题>
            Summary: <summary>  / 摘要：<摘要>
            Tags: <tag1>, <tag2>, <tag3>  / 标签：<标签1>, <标签2>, <标签3>
        
        Args:  / 参数：
            response: LLM response text  / LLM 响应文本
            
        Returns:  / 返回：
            Dictionary with title, summary, tags  / 包含标题、摘要和标签的字典
        """
        metadata = {
            'title': '',
            'summary': '',
            'tags': []
        }
        
        # Extract title  / 提取标题
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract summary  / 提取摘要
        summary_match = re.search(r'Summary:\s*(.+?)(?:\n(?:Tags:|$))', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            metadata['summary'] = summary_match.group(1).strip()
        
        # Extract tags  / 提取标签
        tags_match = re.search(r'Tags:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if tags_match:
            tags_text = tags_match.group(1).strip()
            # Split by comma and clean  / 按逗号拆分并清理
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            metadata['tags'] = tags
        
        # Validation: ensure non-empty values  / 校验：确保值非空
        if not metadata['title']:
            metadata['title'] = 'Untitled'
        if not metadata['summary']:
            metadata['summary'] = response[:500]  # Fallback to raw response  / 回退为原始响应
        
        return metadata
