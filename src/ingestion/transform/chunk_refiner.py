"""Chunk refinement transform: rule-based cleaning + optional LLM enhancement.  / 块精炼转换：基于规则的清洗 + 可选的 LLM 增强。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

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


class ChunkRefiner(BaseTransform):
    """Refines chunks through rule-based cleaning and optional LLM enhancement.  / 通过基于规则的清洗和可选的 LLM 增强来精炼块。
    
    Processing Pipeline:  / 处理流程：
        1. Rule-based refine: Remove noise (whitespace, headers/footers, HTML)  / 基于规则精炼：移除噪声（空白、页眉/页脚、HTML）
        2. (Optional) LLM refine: Intelligent content improvement  / （可选）LLM 精炼：智能改进内容
        3. On LLM failure: Gracefully fallback to rule-based result  / LLM 失败时：优雅降级到基于规则的结果
    
    Configuration (via settings.yaml):  / 配置（通过 settings.yaml）：
        - ingestion.chunk_refiner.use_llm: bool - Enable LLM enhancement  / 启用 LLM 增强
        - ingestion.chunk_refiner.prompt_path: str - Custom prompt file path  / 自定义提示词文件路径
    
    Design Principles:  / 设计原则：
        - Graceful Degradation: LLM errors don't block ingestion  / 优雅降级：LLM 错误不会阻塞摄取
        - Atomic Processing: Each chunk processed independently  / 原子处理：每个块独立处理
        - Observable: Records refined_by in metadata  / 可观测：在元数据中记录 refined_by
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """Initialize ChunkRefiner.  / 初始化 ChunkRefiner。
        
        Args:  / 参数：
            settings: Application settings  / 应用配置
            llm: Optional LLM instance (for testing; auto-created if None)  / 可选的 LLM 实例（用于测试；为 None 时自动创建）
            prompt_path: Optional custom prompt file path  / 可选的自定义提示词文件路径
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/chunk_refinement.txt"))
        
        # Determine if LLM should be used  / 判断是否应使用 LLM
        self.use_llm = getattr(
            getattr(settings, 'ingestion', None), 
            'chunk_refiner', 
            {}
        ).get('use_llm', False) if hasattr(settings, 'ingestion') else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance.  / 延迟加载 LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for chunk refinement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform chunks through refinement pipeline.  / 通过精炼流水线转换块。
        
        Args:  / 参数：
            chunks: List of chunks to refine  / 要精炼的块列表
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            List of refined chunks (same length as input)  / 精炼后的块列表（长度与输入相同）
        """
        if not chunks:
            return []
        
        # Process chunks in parallel if LLM is enabled  / 如果启用了 LLM，则并行处理块
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _refine_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """Refine a single chunk. Thread-safe.  / 精炼单个块。线程安全。
        
        Args:  / 参数：
            chunk: Chunk to refine  / 要精炼的块
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            Tuple of (refined_chunk, refined_by, error_message)  / (精炼后的块、精炼方式、错误信息) 元组
        """
        try:
            # Step 1: Rule-based refinement  / 步骤 1：基于规则精炼
            rule_refined_text = self._rule_based_refine(chunk.text)
            
            # Step 2: LLM enhancement  / 步骤 2：LLM 增强
            if self.use_llm and self.llm:
                llm_refined_text = self._llm_refine(rule_refined_text, trace)
                
                if llm_refined_text:
                    refined_text = llm_refined_text
                    refined_by = "llm"
                else:
                    refined_text = rule_refined_text
                    refined_by = "rule"
            else:
                refined_text = rule_refined_text
                refined_by = "rule"
            
            refined_chunk = Chunk(
                id=chunk.id,
                text=refined_text,
                metadata={
                    **(chunk.metadata or {}),
                    'refined_by': refined_by
                },
                source_ref=chunk.source_ref
            )
            return (refined_chunk, refined_by, None)
            
        except Exception as e:
            logger.error(f"Failed to refine chunk {chunk.id}: {e}")
            return (chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks in parallel using ThreadPoolExecutor.  / 使用 ThreadPoolExecutor 并行处理块。"""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        refined_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks  / 提交所有任务
            future_to_idx = {
                executor.submit(self._refine_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            # Collect results  / 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    refined_chunk, refined_by, error = future.result()
                    refined_chunks[idx] = refined_chunk
                    
                    if refined_by == "llm":
                        llm_enhanced_count += 1
                    elif refined_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel refinement: {e}")
                    refined_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in refined_chunks if c is not None)
        
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks sequentially (fallback when LLM disabled).  / 顺序处理块（LLM 禁用时的回退方式）。"""
        refined_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # Step 1: Rule-based refinement (always performed)  / 步骤 1：基于规则精炼（始终执行）
                rule_refined_text = self._rule_based_refine(chunk.text)
                
                # Step 2: Optional LLM enhancement  / 步骤 2：可选的 LLM 增强
                if self.use_llm and self.llm:
                    llm_refined_text = self._llm_refine(rule_refined_text, trace)
                    
                    if llm_refined_text:
                        # LLM success  / LLM 成功
                        refined_text = llm_refined_text
                        refined_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM failed, fallback to rule-based  / LLM 失败，回退到基于规则的结果
                        refined_text = rule_refined_text
                        refined_by = "rule"
                        fallback_count += 1
                        if chunk.metadata:
                            chunk.metadata['refine_fallback_reason'] = "llm_failed"
                else:
                    # LLM disabled, use rule-based  / LLM 已禁用，使用基于规则的结果
                    refined_text = rule_refined_text
                    refined_by = "rule"
                
                # Create refined chunk  / 创建精炼后的块
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text,
                    metadata={
                        **(chunk.metadata or {}),
                        'refined_by': refined_by
                    },
                    source_ref=chunk.source_ref
                )
                refined_chunks.append(refined_chunk)
                success_count += 1
                
            except Exception as e:
                # Atomic failure: log and preserve original  / 原子失败：记录日志并保留原始块
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                refined_chunks.append(chunk)
        
        # Record trace  / 记录追踪信息
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based text cleaning.  / 应用基于规则的文本清洗。
        
        Cleaning operations:  / 清洗操作：
            1. Remove page headers/footers (separator lines + metadata)  / 移除页眉/页脚（分隔线 + 元数据）
            2. Remove HTML comments  / 移除 HTML 注释
            3. Remove HTML tags (preserve content)  / 移除 HTML 标签（保留内容）
            4. Normalize excessive whitespace  / 规范化过多空白
            5. Preserve code blocks and Markdown formatting  / 保留代码块和 Markdown 格式
        
        Args:  / 参数：
            text: Raw chunk text  / 原始块文本
            
        Returns:  / 返回：
            Cleaned text  / 清洗后的文本
        """
        if not text:
            return text
        
        # Early return if only whitespace  / 如果只有空白则提前返回
        if not text.strip():
            return ""
        
        # Preserve code blocks (extract and restore later)  / 保留代码块（先提取，稍后恢复）
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'
        
        def extract_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        text = re.sub(code_block_pattern, extract_code_block, text)
        
        # 1. Remove separator lines with page numbers/footers  / 1. 移除带页码/页脚的分隔线
        # Pattern: ────────────────  / 模式：────────────────
        # Followed by: Page XX, Footer text, etc.  / 后接：Page XX、页脚文本等
        text = re.sub(
            r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(r'─{10,}', '', text)  # Remove remaining separator lines  / 移除剩余分隔线
        
        # 2. Remove HTML comments  / 2. 移除 HTML 注释
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # 3. Remove HTML tags (but preserve content)  / 3. 移除 HTML 标签（但保留内容）
        text = re.sub(r'<[^>]+>', '', text)
        
        # 4. Normalize whitespace  / 4. 规范化空白
        # - Collapse multiple spaces to single space  / - 将多个空格折叠为单个空格
        text = re.sub(r' {2,}', ' ', text)
        
        # - Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)  / - 将连续 3 个及以上换行折叠为 2 个（保留段落分隔）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 5. Remove leading/trailing whitespace from each line  / 5. 移除每行首尾空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 6. Restore code blocks  / 6. 恢复代码块
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)
        
        # Final cleanup  / 最终清理
        text = text.strip()
        
        return text
    
    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Apply LLM-based intelligent refinement.  / 应用基于 LLM 的智能精炼。
        
        Args:  / 参数：
            text: Rule-refined text  / 基于规则精炼后的文本
            trace: Optional trace context  / 可选的追踪上下文
            
        Returns:  / 返回：
            LLM-refined text, or None if refinement failed  / LLM 精炼后的文本；精炼失败时返回 None
        """
        if not text or not text.strip():
            return text
        
        try:
            # Load prompt template  / 加载提示词模板
            prompt_template = self._load_prompt()
            if not prompt_template:
                logger.warning("Prompt template not found, skipping LLM refinement")
                return None
            
            # Fill prompt  / 填充提示词
            if '{text}' not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None
            
            prompt = prompt_template.replace('{text}', text)
            
            # Call LLM with Message objects  / 使用 Message 对象调用 LLM
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)
            
            # Extract text from ChatResponse  / 从 ChatResponse 中提取文本
            if isinstance(response, str):
                refined_text = response
            else:
                # response is ChatResponse object  / response 是 ChatResponse 对象
                refined_text = response.content
            
            if refined_text and refined_text.strip():
                return refined_text.strip()
            else:
                logger.warning("LLM returned empty result")
                return None
                
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
    
    def _load_prompt(self) -> Optional[str]:
        """Load prompt template from file.  / 从文件加载提示词模板。
        
        Returns:  / 返回：
            Prompt template string, or None if file not found  / 提示词模板字符串；文件不存在时返回 None
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        try:
            prompt_path = Path(self._prompt_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {self._prompt_path}")
                return None
            
            self._prompt_template = prompt_path.read_text(encoding='utf-8')
            logger.debug(f"Loaded prompt template from {self._prompt_path}")
            return self._prompt_template
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None
