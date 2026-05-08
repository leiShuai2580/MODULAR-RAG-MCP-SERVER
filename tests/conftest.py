"""Pytest configuration and shared fixtures. / Pytest 配置和共享 fixtures。

This module contains pytest configuration and fixtures that are shared / 本模块包含所有测试模块共享的 pytest 配置和 fixtures，
across all test modules. / 跨全部测试模块使用。
"""

import sys
from pathlib import Path

import pytest

# Add the project root to the Python path / 将项目根目录加入 Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory path. / 返回项目根目录路径。
    
    Returns: / 返回：
        Path to the project root directory. / 项目根目录路径。
    """
    return PROJECT_ROOT


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """Return the sample documents directory path. / 返回示例 documents 目录路径。
    
    Args: / 参数：
        project_root: The project root directory path. / 项目根目录路径。
        
    Returns: / 返回：
        Path to the sample documents directory. / 示例 documents 目录路径。
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path. / 返回 config 目录路径。
    
    Args: / 参数：
        project_root: The project root directory path. / 项目根目录路径。
        
    Returns: / 返回：
        Path to the config directory. / config 目录路径。
    """
    return project_root / "config"
