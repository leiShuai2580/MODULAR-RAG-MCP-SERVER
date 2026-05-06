"""
Splitter Module. / Splitter 模块。

This package contains text splitter abstractions and implementations: / 此包包含文本拆分器抽象和实现：
- Base splitter class / 基础 splitter 类
- Splitter factory / Splitter 工厂
- Implementations (Recursive, Semantic, FixedLength) / 实现（Recursive、Semantic、FixedLength）
"""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

# Import concrete implementations (they auto-register with factory) / 导入具体实现（它们会自动注册到工厂）
try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter
except ImportError:
    RecursiveSplitter = None  # type: ignore[assignment, misc]

__all__ = [
    "BaseSplitter",
    "SplitterFactory",
    "RecursiveSplitter",
]
