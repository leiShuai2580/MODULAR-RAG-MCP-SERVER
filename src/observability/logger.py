"""Observability logger utilities. / 可观测性日志工具。

Provides: / 提供：
- ``get_logger``: standard human-readable logger (unchanged from C-phase). / 标准可读日志器（与 C 阶段保持一致）。
- ``JSONFormatter``: custom :class:`logging.Formatter` that emits JSON. / 输出 JSON 的自定义 :class:`logging.Formatter`。
- ``get_trace_logger``: returns a logger backed by a JSON Lines file handler. / 返回由 JSON Lines 文件处理器支撑的日志器。
- ``write_trace``: convenience function to append a trace dict to / 追加 trace 字典的便捷函数，
  ``logs/traces.jsonl``. / 写入 ``logs/traces.jsonl``。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import resolve_path

# Default path for traces file (absolute, CWD-independent) / trace 文件默认路径（绝对路径，与 CWD 无关）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


# ── Human-readable logger (existing) ──────────────────────────────── / ── 可读日志器（已有）────────────────────────────────


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger. / 获取配置好的日志器。

    Args: / 参数：
        name: Logger name. / 日志器名称。
        log_level: Optional log level string (e.g., "INFO"). / 可选日志级别字符串（例如 "INFO"）。

    Returns: / 返回：
        Configured logger instance. / 配置好的日志器实例。
    """

    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # Suppress httpx logs (contains sensitive endpoint URLs) / 抑制 httpx 日志（包含敏感端点 URL）
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(name)


# ── JSON Lines formatter ──────────────────────────────────────────── / ── JSON Lines 格式化器 ───────────────────────────────


class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs one JSON object per line. / 每行输出一个 JSON 对象的日志格式化器。

    Each log record is serialised to a dict containing at least: / 每条日志记录会序列化为至少包含以下字段的字典：
    ``timestamp``, ``level``, ``logger``, ``message``.  If the record / ``timestamp``、``level``、``logger``、``message``。如果记录
    carries an ``exc_info`` tuple the traceback is included as / 携带 ``exc_info`` 元组，则会把 traceback 作为
    ``exception``. / ``exception`` 包含进去。

    Extra attributes attached via *extra=* on the logger call are / 通过日志调用中的 *extra=* 附加的额外属性会
    merged into the top-level dict (except internal Python fields). / 合并到顶层字典中（Python 内部字段除外）。
    """

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Return the log record as a single-line JSON string. / 将日志记录作为单行 JSON 字符串返回。"""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # merge extra fields the caller attached / 合并调用方附加的额外字段
        for key, val in record.__dict__.items():
            if key not in self._INTERNAL_ATTRS and key not in payload:
                try:
                    json.dumps(val)  # cheap serialisability test / 低成本可序列化性测试
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── Trace logger ──────────────────────────────────────────────────── / ── Trace 日志器 ─────────────────────────────────────


def get_trace_logger(
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    name: str = "modular-rag.trace",
) -> logging.Logger:
    """Return a logger that writes JSON Lines to *traces_path*. / 返回一个将 JSON Lines 写入 *traces_path* 的日志器。

    The logger uses :class:`JSONFormatter` and a :class:`FileHandler` / 该日志器使用 :class:`JSONFormatter` 和配置为追加模式的
    configured to append.  Repeated calls with the same *name* return / :class:`FileHandler`。使用相同 *name* 重复调用会返回
    the same logger (standard :mod:`logging` semantics). / 同一个日志器（标准 :mod:`logging` 语义）。

    Args: / 参数：
        traces_path: File path for the JSONL output.  Parent directories / JSONL 输出文件路径；父目录
            are created automatically. / 会自动创建。
        name: Logger name. / 日志器名称。

    Returns: / 返回：
        A :class:`logging.Logger` ready for JSON Lines output. / 可用于 JSON Lines 输出的 :class:`logging.Logger`。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated calls / 避免重复调用时添加重复处理器
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # don't echo to console / 不回显到控制台

    return logger


# ── Convenience writer for trace dicts ────────────────────────────── / ── trace 字典便捷写入器 ─────────────────────────────


def write_trace(
    trace_dict: Dict[str, Any],
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
) -> None:
    """Append a single trace dictionary as one JSON line. / 将单个 trace 字典追加为一行 JSON。

    This is a thin wrapper that writes directly — no logging / 这是一个直接写入的轻量包装器，不涉及日志
    framework involved — so the output is identical to what / 框架，因此输出与
    :class:`~src.core.trace.trace_collector.TraceCollector` produces. / :class:`~src.core.trace.trace_collector.TraceCollector` 生成的内容一致。

    Args: / 参数：
        trace_dict: A JSON-serialisable dictionary (typically from / 可 JSON 序列化的字典（通常来自
            ``TraceContext.to_dict()``). / ``TraceContext.to_dict()``）。
        traces_path: Output file path; parent directories are created / 输出文件路径；父目录
            automatically. / 会自动创建。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(trace_dict, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
