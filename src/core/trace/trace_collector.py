"""Trace collector – receives finished TraceContext and persists them. / 追踪收集器：接收已完成的 TraceContext 并持久化。

The collector is the bridge between in-memory TraceContext objects and / 收集器是在内存中的 TraceContext 对象与
the on-disk JSON Lines log used by the Dashboard.  It is intentionally / Dashboard 使用的磁盘 JSON Lines 日志之间的桥梁。它有意
decoupled from the logging module so that trace persistence remains / 与 logging 模块解耦，使追踪持久化保持
predictable and testable. / 可预测且易测试。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Default absolute path for traces file (CWD-independent) / 追踪文件的默认绝对路径（与当前工作目录无关）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """Collects finished traces and appends them to a JSON Lines file. / 收集已完成追踪并追加到 JSON Lines 文件。

    Args: / 参数：
        traces_path: File path for the ``traces.jsonl`` output. / traces_path：``traces.jsonl`` 输出文件路径。
            Parent directories are created automatically. / 父目录会自动创建。
    """

    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._path = Path(traces_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """Persist a single trace as one JSON line. / 将单个追踪持久化为一行 JSON。

        If the trace has not been finished yet, ``finish()`` is called / 如果追踪尚未完成，则会自动调用 ``finish()``，
        automatically so the output always contains timing data. / 因而输出始终包含耗时数据。

        Args: / 参数：
            trace: A populated :class:`TraceContext`. / trace：已填充的 :class:`TraceContext`。
        """
        if trace.finished_at is None:
            trace.finish()

        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("Failed to write trace %s", trace.trace_id)

    @property
    def path(self) -> Path:
        """Return the resolved path of the traces file. / 返回追踪文件解析后的路径。"""
        return self._path
