"""Unit tests for Embedding Factory and Base Embedding. / Embedding Factory 和 Base Embedding 的单元测试。

Test Coverage: / 测试覆盖：
- Factory pattern: provider registration, creation, and routing / Factory 模式：provider 注册、创建和路由
- Configuration-driven instantiation / 配置驱动的实例化
- Error handling for unknown/missing providers / 未知/缺失 provider 的错误处理
- Validation logic in BaseEmbedding / BaseEmbedding 中的校验逻辑
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Fake embedding provider for testing. / 用于测试的 fake embedding provider。
    
    Returns deterministic fake vectors for reproducible testing. / 返回确定性的 fake vectors，便于可复现测试。
    """
    
    def __init__(self, settings: Any = None, dimension: int = 384, **kwargs: Any):
        """Initialize fake embedding provider. / 初始化 fake embedding provider。
        
        Args: / 参数：
            settings: Optional settings (unused in fake). / 可选 settings（fake 中不使用）。
            dimension: Vector dimension to return. / 要返回的向量维度。
            **kwargs: Additional parameters (unused). / 额外参数（不使用）。
        """
        self.settings = settings
        self.dimension = dimension
        self.call_count = 0
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate fake embeddings. / 生成 fake embeddings。"""
        self.validate_texts(texts)
        self.call_count += 1
        
        # Return deterministic fake vectors / 返回确定性的 fake vectors
        return [[float(i + j) for j in range(self.dimension)] for i in range(len(texts))]
    
    def get_dimension(self) -> int:
        """Return configured dimension. / 返回配置的维度。"""
        return self.dimension


class TestBaseEmbedding:
    """Tests for BaseEmbedding abstract class. / BaseEmbedding 抽象类测试。"""
    
    def test_validate_texts_success(self):
        """Valid text list should pass validation. / 有效文本列表应通过校验。"""
        embedding = FakeEmbedding()
        # Should not raise / 不应抛出异常
        embedding.validate_texts(["hello", "world"])
    
    def test_validate_texts_empty_list(self):
        """Empty list should raise ValueError. / 空列表应抛出 ValueError。"""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding.validate_texts([])
    
    def test_validate_texts_non_string(self):
        """Non-string entries should raise ValueError. / 非字符串条目应抛出 ValueError。"""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="not a string"):
            embedding.validate_texts(["valid", 123, "text"])  # type: ignore
    
    def test_validate_texts_empty_string(self):
        """Empty or whitespace-only strings should raise ValueError. / 空字符串或仅空白字符串应抛出 ValueError。"""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.validate_texts(["valid", "   ", "text"])
    
    def test_get_dimension_implemented(self):
        """FakeEmbedding should return configured dimension. / FakeEmbedding 应返回配置的维度。"""
        embedding = FakeEmbedding(dimension=512)
        assert embedding.get_dimension() == 512
    
    def test_get_dimension_not_implemented(self):
        """BaseEmbedding without override should raise NotImplementedError. / 未覆盖方法的 BaseEmbedding 应抛出 NotImplementedError。"""
        
        class IncompleteEmbedding(BaseEmbedding):
            def embed(self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[List[float]]:
                return [[0.0]]
        
        incomplete = IncompleteEmbedding()
        with pytest.raises(NotImplementedError, match="must implement get_dimension"):
            incomplete.get_dimension()


class TestFakeEmbedding:
    """Tests for FakeEmbedding provider implementation. / FakeEmbedding provider 实现测试。"""
    
    def test_embed_single_text(self):
        """Embedding single text should return one vector. / 对单个文本 embedding 应返回一个向量。"""
        embedding = FakeEmbedding(dimension=3)
        result = embedding.embed(["hello"])
        
        assert len(result) == 1
        assert len(result[0]) == 3
        assert result[0] == [0.0, 1.0, 2.0]
    
    def test_embed_multiple_texts(self):
        """Embedding multiple texts should return matching number of vectors. / 对多个文本 embedding 应返回数量匹配的向量。"""
        embedding = FakeEmbedding(dimension=2)
        result = embedding.embed(["hello", "world", "test"])
        
        assert len(result) == 3
        assert result[0] == [0.0, 1.0]
        assert result[1] == [1.0, 2.0]
        assert result[2] == [2.0, 3.0]
    
    def test_embed_increments_call_count(self):
        """Each embed call should increment the counter. / 每次 embed 调用都应递增计数器。"""
        embedding = FakeEmbedding()
        assert embedding.call_count == 0
        
        embedding.embed(["test1"])
        assert embedding.call_count == 1
        
        embedding.embed(["test2", "test3"])
        assert embedding.call_count == 2
    
    def test_embed_validates_input(self):
        """embed() should call validate_texts and raise on invalid input. / embed() 应调用 validate_texts 并在输入无效时抛错。"""
        embedding = FakeEmbedding()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding.embed([])
        
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.embed(["  "])


class TestEmbeddingFactory:
    """Tests for EmbeddingFactory. / EmbeddingFactory 测试。"""
    
    def setup_method(self):
        """Reset factory registry before each test. / 每个测试前重置 factory 注册表。"""
        EmbeddingFactory._PROVIDERS.clear()
    
    def test_register_provider_success(self):
        """Registering valid provider should succeed. / 注册有效 provider 应成功。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        assert "fake" in EmbeddingFactory._PROVIDERS
        assert EmbeddingFactory._PROVIDERS["fake"] == FakeEmbedding
    
    def test_register_provider_case_insensitive(self):
        """Provider names should be normalized to lowercase. / Provider 名称应规范化为小写。"""
        EmbeddingFactory.register_provider("OpenAI", FakeEmbedding)
        assert "openai" in EmbeddingFactory._PROVIDERS
    
    def test_register_provider_invalid_class(self):
        """Registering non-BaseEmbedding class should raise ValueError. / 注册非 BaseEmbedding 类应抛出 ValueError。"""
        
        class NotAnEmbedding:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseEmbedding"):
            EmbeddingFactory.register_provider("invalid", NotAnEmbedding)  # type: ignore
    
    def test_list_providers_empty(self):
        """list_providers should return empty list when no providers registered. / 未注册 provider 时 list_providers 应返回空列表。"""
        assert EmbeddingFactory.list_providers() == []
    
    def test_list_providers_sorted(self):
        """list_providers should return sorted provider names. / list_providers 应返回排序后的 provider 名称。"""
        EmbeddingFactory.register_provider("zebra", FakeEmbedding)
        EmbeddingFactory.register_provider("alpha", FakeEmbedding)
        EmbeddingFactory.register_provider("beta", FakeEmbedding)
        
        providers = EmbeddingFactory.list_providers()
        assert providers == ["alpha", "beta", "zebra"]
    
    def test_create_success(self):
        """Creating registered provider should succeed. / 创建已注册 provider 应成功。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "fake"
        
        embedding = EmbeddingFactory.create(settings)
        
        assert isinstance(embedding, FakeEmbedding)
        assert embedding.settings == settings
    
    def test_create_case_insensitive(self):
        """Provider lookup should be case-insensitive. / Provider 查找应大小写不敏感。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "FAKE"
        
        embedding = EmbeddingFactory.create(settings)
        assert isinstance(embedding, FakeEmbedding)
    
    def test_create_with_overrides(self):
        """Factory should pass override kwargs to provider constructor. / Factory 应将覆盖 kwargs 传给 provider 构造函数。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "fake"
        
        embedding = EmbeddingFactory.create(settings, dimension=1024)
        assert embedding.dimension == 1024
    
    def test_create_unknown_provider(self):
        """Creating unregistered provider should raise clear error. / 创建未注册 provider 应抛出清晰错误。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "unknown"
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Unsupported Embedding provider: 'unknown'" in error_message
        assert "Available providers:" in error_message
    
    def test_create_missing_provider_config(self):
        """Missing provider in settings should raise clear error. / settings 中缺失 provider 应抛出清晰错误。"""
        settings = MagicMock()
        del settings.embedding  # Simulate missing config / 模拟缺失配置
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Missing required configuration" in error_message
        assert "settings.embedding.provider" in error_message
        assert "settings.yaml" in error_message
    
    def test_create_provider_instantiation_failure(self):
        """Provider constructor errors should be wrapped in RuntimeError. / Provider 构造函数错误应包装为 RuntimeError。"""
        
        class BrokenEmbedding(BaseEmbedding):
            def __init__(self, settings: Any, **kwargs: Any):
                raise ValueError("Intentional init error")
            
            def embed(self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[List[float]]:
                return [[0.0]]
        
        EmbeddingFactory.register_provider("broken", BrokenEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "broken"
        
        with pytest.raises(RuntimeError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Failed to instantiate Embedding provider 'broken'" in error_message
        assert "Intentional init error" in error_message
    
    def test_create_no_providers_registered(self):
        """Creating provider when registry is empty should show helpful message. / 注册表为空时创建 provider 应显示有帮助的消息。"""
        settings = MagicMock()
        settings.embedding.provider = "openai"
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Unsupported Embedding provider: 'openai'" in error_message
        assert "Available providers: none" in error_message
