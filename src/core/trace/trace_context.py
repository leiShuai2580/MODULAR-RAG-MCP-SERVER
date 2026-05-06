"""Trace context for observability across pipeline stages. / 用于跨流水线阶段可观测性的追踪上下文。

Provides trace_id, trace_type (query/ingestion), per-stage timing, / 提供 trace_id、trace_type（query/ingestion）、逐阶段耗时、
finish() lifecycle, and to_dict() serialisation for JSON Lines output. / finish() 生命周期，以及用于 JSON Lines 输出的 to_dict() 序列化。
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TraceContext:
    """Request-scoped trace context that records pipeline stages and timing. / 记录流水线阶段和耗时的请求级追踪上下文。

    Attributes: / 属性：
        trace_id: Unique identifier for this trace. / trace_id：该追踪的唯一标识。
        trace_type: Either ``"query"`` or ``"ingestion"``. / trace_type：``"query"`` 或 ``"ingestion"``。
        started_at: ISO-8601 timestamp when the trace was created. / started_at：追踪创建时的 ISO-8601 时间戳。
        finished_at: ISO-8601 timestamp when ``finish()`` was called, or None. / finished_at：调用 ``finish()`` 时的 ISO-8601 时间戳，或 None。
        stages: Ordered list of recorded stage dicts. / stages：已记录阶段字典的有序列表。
        metadata: Arbitrary key/value pairs attached to the trace. / metadata：附加到追踪上的任意键值对。
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # internal monotonic clock for accurate elapsed calculation / 用于精确计算耗时的内部单调时钟
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    # ---- recording --------------------------------------------------- / ---- 记录 ---------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """Record data from a pipeline stage. / 记录流水线阶段的数据。

        Args: / 参数：
            stage_name: Name of the stage (e.g. ``"dense_retrieval"``). / stage_name：阶段名称（例如 ``"dense_retrieval"``）。
            data: Stage-specific payload (method, provider, details …). / data：阶段专属载荷（method、provider、details 等）。
            elapsed_ms: Pre-computed elapsed time in ms.  If *None* the / elapsed_ms：预先计算的耗时（毫秒）。如果为 *None*，
                caller should measure externally, or leave it to the / 调用方应在外部测量，或交给
                ``stage_timer`` context-manager. / ``stage_timer`` 上下文管理器处理。
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- lifecycle ---------------------------------------------------- / ---- 生命周期 ----------------------------------------------------

    def finish(self) -> None:
        """Mark the trace as finished and record wall-clock end time. / 将追踪标记为完成并记录墙钟结束时间。"""
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- timing helpers ----------------------------------------------- / ---- 耗时辅助方法 -----------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """Return elapsed time in milliseconds. / 返回毫秒级耗时。

        Args: / 参数：
            stage_name: If given, return the elapsed time recorded for / stage_name：如果提供，则返回该阶段记录的耗时。
                that stage.  If *None*, return the total trace elapsed / 如果为 *None*，则返回整个追踪的总耗时
                time (start → finish, or start → now if not yet / （start → finish；如果尚未完成则为 start → now）。
                finished).

        Returns: / 返回：
            Elapsed milliseconds. / 毫秒级耗时。

        Raises: / 异常：
            KeyError: If *stage_name* was provided but not found. / KeyError：当提供了 *stage_name* 但未找到时抛出。
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    # ---- serialisation ------------------------------------------------ / ---- 序列化 ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the trace to a plain dict suitable for ``json.dumps``. / 将追踪序列化为适合 ``json.dumps`` 的普通字典。

        Returns: / 返回：
            Dictionary with all trace data. / 包含所有追踪数据的字典。
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ---- backwards-compat helper used in C5 / C6 ----------------------- / ---- C5 / C6 使用的向后兼容辅助方法 -----------------------

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve recorded data for a specific stage. / 获取指定阶段的已记录数据。

        Searches stages list (last-write-wins for duplicate names). / 搜索 stages 列表（重复名称采用最后写入优先）。

        Args: / 参数：
            stage_name: Name of the stage to retrieve. / stage_name：要获取的阶段名称。

        Returns: / 返回：
            The ``data`` dict of the matching stage, or *None*. / 匹配阶段的 ``data`` 字典，或 *None*。
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
