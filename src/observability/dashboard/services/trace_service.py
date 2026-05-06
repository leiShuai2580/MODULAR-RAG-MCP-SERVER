"""TraceService – read and parse traces from logs/traces.jsonl. / TraceService - 从 logs/traces.jsonl 读取并解析 trace。

Provides a typed, filterable interface over the raw JSONL trace log. / 在原始 JSONL trace 日志之上提供带类型、可过滤的接口。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent) / trace 文件默认路径（绝对路径，与 CWD 无关）
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceService:
    """Read-only service for querying recorded traces. / 用于查询已记录 trace 的只读服务。

    Args: / 参数：
        traces_path: Path to the JSONL file.  Defaults to / JSONL 文件路径，默认是
            ``logs/traces.jsonl``. / ``logs/traces.jsonl``。
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # Public API / 公共 API
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order. / 按时间倒序返回 trace。

        Args: / 参数：
            trace_type: Filter by ``trace_type`` field (e.g. / 按 ``trace_type`` 字段过滤（例如
                ``"ingestion"`` or ``"query"``).  ``None`` = all. / ``"ingestion"`` 或 ``"query"``）。``None`` 表示全部。
            limit: Maximum number of traces to return. / 要返回的最大 trace 数量。

        Returns: / 返回：
            List of trace dicts (newest first). / trace 字典列表（最新在前）。
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # Newest first / 最新在前
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``. / 按 ``trace_id`` 检索单个 trace。

        Returns: / 返回：
            Trace dict, or ``None`` if not found. / trace 字典；如果未找到则返回 ``None``。
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace. / 从 trace 中提取阶段耗时。

        Returns: / 返回：
            List of dicts with keys: stage_name, elapsed_ms, data. / 包含 stage_name、elapsed_ms、data 键的字典列表。
            Ordered by appearance. / 按出现顺序排列。
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms / 原始 stage 字典包含 stage、timestamp、data（字典）、elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening / 直接提取内部 'data' 字典，而不是扁平化
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    # ------------------------------------------------------------------ / ------------------------------------------------------------------
    # Internal helpers / 内部辅助方法
    # ------------------------------------------------------------------ / ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every line in the JSONL file. / 解析 JSONL 文件中的每一行。

        Silently skips malformed lines. / 静默跳过格式错误的行。
        """
        if not self.traces_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces
