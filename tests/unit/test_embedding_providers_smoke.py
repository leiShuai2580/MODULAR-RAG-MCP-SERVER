"""Unit tests for OpenAI and Azure Embedding provider implementations. / OpenAI 和 Azure Embedding provider 实现的单元测试。

This test suite validates the OpenAI and Azure Embedding implementations / 该测试套件使用 mock HTTP 响应验证 OpenAI 和 Azure Embedding 实现，
using mocked HTTP responses to ensure reliable, fast, and offline testing. / 以确保测试可靠、快速且可离线运行。
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.libs.embedding.azure_embedding import AzureEmbedding, AzureEmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError


# =============================================================================
# Test Fixtures / 测试 Fixture
# =============================================================================

@pytest.fixture
def mock_settings_openai() -> Any:
    """Create mock settings for OpenAI embedding. / 创建 OpenAI embedding 的 mock settings。"""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "openai"
    settings.embedding.model = "text-embedding-3-small"
    settings.embedding.dimensions = 1536
    settings.embedding.base_url = None  # No base_url in settings by default / settings 默认没有 base_url
    return settings


@pytest.fixture
def mock_settings_azure() -> Any:
    """Create mock settings for Azure embedding. / 创建 Azure embedding 的 mock settings。"""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "azure"
    settings.embedding.model = "text-embedding-ada-002"
    settings.embedding.deployment_name = "my-embedding-deployment"
    settings.embedding.azure_endpoint = "https://my-resource.openai.azure.com/"
    settings.embedding.api_version = "2024-02-01"
    settings.embedding.dimensions = None
    return settings


@pytest.fixture
def mock_openai_response() -> Mock:
    """Create a mock OpenAI embeddings response. / 创建 mock OpenAI embeddings 响应。"""
    response = Mock()
    response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]
    return response


# =============================================================================
# OpenAI Embedding Tests / OpenAI Embedding 测试
# =============================================================================

class TestOpenAIEmbedding:
    """Test suite for OpenAIEmbedding implementation. / OpenAIEmbedding 实现测试套件。"""
    
    def test_initialization_with_api_key(self, mock_settings_openai: Any) -> None:
        """Test successful initialization with API key from parameter. / 测试使用参数中的 API key 成功初始化。"""
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        
        assert embedding.api_key == "test-key"
        assert embedding.model == "text-embedding-3-small"
        assert embedding.dimensions == 1536
        assert embedding.base_url == OpenAIEmbedding.DEFAULT_BASE_URL
    
    def test_initialization_with_env_var(
        self, mock_settings_openai: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization with API key from environment variable. / 测试使用环境变量中的 API key 初始化。"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        
        embedding = OpenAIEmbedding(mock_settings_openai)
        assert embedding.api_key == "env-key"
    
    def test_initialization_missing_api_key(self, mock_settings_openai: Any) -> None:
        """Test that initialization fails when API key is missing. / 测试缺失 API key 时初始化失败。"""
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            OpenAIEmbedding(mock_settings_openai)
    
    def test_initialization_with_custom_base_url(self, mock_settings_openai: Any) -> None:
        """Test initialization with custom base URL. / 测试使用自定义 base URL 初始化。"""
        embedding = OpenAIEmbedding(
            mock_settings_openai,
            api_key="test-key",
            base_url="https://custom.api.com/v1"
        )
        
        assert embedding.base_url == "https://custom.api.com/v1"
    
    @patch('openai.OpenAI')
    def test_embed_success(
        self, mock_openai_class: Mock, mock_settings_openai: Any, mock_openai_response: Mock
    ) -> None:
        """Test successful embedding generation. / 测试成功生成 embedding。"""
        # Mock OpenAI client / Mock OpenAI client
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        # Create embedding instance and call embed / 创建 embedding 实例并调用 embed
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        result = embedding.embed(["hello", "world"])
        
        # Verify result / 验证结果
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        # Verify API call / 验证 API 调用
        mock_client.embeddings.create.assert_called_once_with(
            input=["hello", "world"],
            model="text-embedding-3-small",
            dimensions=1536,
        )
    
    @patch('openai.OpenAI')
    def test_embed_without_dimensions(
        self, mock_openai_class: Mock, mock_settings_openai: Any
    ) -> None:
        """Test embedding without dimensions parameter. / 测试不带 dimensions 参数的 embedding。"""
        mock_settings_openai.embedding.dimensions = None
        
        # Create a mock response with just 1 embedding for 1 text / 创建只含 1 个文本对应 1 个 embedding 的 mock 响应
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        embedding.embed(["test"])
        
        # Verify dimensions not passed when None / 验证 dimensions 为 None 时不会传递
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert "dimensions" not in call_kwargs
    
    def test_embed_empty_list_raises(
        self, mock_settings_openai: Any
    ) -> None:
        """Test that empty text list raises ValueError. / 测试空文本列表会抛出 ValueError。"""
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embedding.embed([])
    
    @patch('openai.OpenAI')
    def test_embed_api_error(
        self, mock_openai_class: Mock, mock_settings_openai: Any
    ) -> None:
        """Test handling of API errors. / 测试 API 错误处理。"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        
        with pytest.raises(OpenAIEmbeddingError, match="OpenAI Embeddings API call failed"):
            embedding.embed(["test"])
    
    @patch('openai.OpenAI')
    def test_embed_length_mismatch(
        self, mock_openai_class: Mock, mock_settings_openai: Any
    ) -> None:
        """Test handling of response length mismatch. / 测试响应长度不匹配的处理。"""
        # Return only 1 embedding for 2 inputs / 对 2 个输入只返回 1 个 embedding
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        
        with pytest.raises(OpenAIEmbeddingError, match="Output length mismatch"):
            embedding.embed(["test1", "test2"])
    
    def test_get_dimension_with_configured_value(self, mock_settings_openai: Any) -> None:
        """Test get_dimension returns configured dimension. / 测试 get_dimension 返回配置的维度。"""
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        assert embedding.get_dimension() == 1536
    
    def test_get_dimension_model_defaults(self, mock_settings_openai: Any) -> None:
        """Test get_dimension returns model-specific defaults. / 测试 get_dimension 返回模型特定默认值。"""
        mock_settings_openai.embedding.dimensions = None
        mock_settings_openai.embedding.model = "text-embedding-3-large"
        
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        assert embedding.get_dimension() == 3072
    
    def test_get_dimension_unknown_model(self, mock_settings_openai: Any) -> None:
        """Test get_dimension returns None for unknown models. / 测试未知模型时 get_dimension 返回 None。"""
        mock_settings_openai.embedding.dimensions = None
        mock_settings_openai.embedding.model = "unknown-model"
        
        embedding = OpenAIEmbedding(mock_settings_openai, api_key="test-key")
        assert embedding.get_dimension() is None


# =============================================================================
# Azure Embedding Tests / Azure Embedding 测试
# =============================================================================

class TestAzureEmbedding:
    """Test suite for AzureEmbedding implementation. / AzureEmbedding 实现测试套件。"""
    
    def test_initialization_with_all_params(self, mock_settings_azure: Any) -> None:
        """Test successful initialization with all Azure-specific parameters. / 测试使用所有 Azure 专用参数成功初始化。"""
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01"
        )
        
        assert embedding.api_key == "test-key"
        assert embedding.azure_endpoint == "https://test.openai.azure.com/"
        assert embedding.api_version == "2024-02-01"
        assert embedding.deployment_name == "my-embedding-deployment"
    
    def test_initialization_with_env_vars(
        self, mock_settings_azure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization with Azure environment variables. / 测试使用 Azure 环境变量初始化。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com/")
        
        embedding = AzureEmbedding(mock_settings_azure)
        
        assert embedding.api_key == "azure-env-key"
        assert embedding.azure_endpoint == "https://env.openai.azure.com/"
    
    def test_initialization_fallback_to_openai_env_var(
        self, mock_settings_azure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that Azure falls back to OPENAI_API_KEY if Azure key not set. / 测试未设置 Azure key 时 Azure 会回退到 OPENAI_API_KEY。"""
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        
        embedding = AzureEmbedding(mock_settings_azure)
        assert embedding.api_key == "openai-key"
    
    def test_initialization_missing_api_key(self, mock_settings_azure: Any) -> None:
        """Test that initialization fails when API key is missing. / 测试缺失 API key 时初始化失败。"""
        with pytest.raises(ValueError, match="Azure OpenAI API key not provided"):
            AzureEmbedding(mock_settings_azure)
    
    def test_initialization_missing_endpoint(
        self, mock_settings_azure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that initialization fails when endpoint is missing. / 测试缺失 endpoint 时初始化失败。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        mock_settings_azure.embedding.azure_endpoint = None
        
        with pytest.raises(ValueError, match="Azure OpenAI endpoint not provided"):
            AzureEmbedding(mock_settings_azure)
    
    def test_deployment_name_fallback_to_model(self, mock_settings_azure: Any) -> None:
        """Test that deployment_name falls back to model if not specified. / 测试未指定 deployment_name 时会回退到 model。"""
        mock_settings_azure.embedding.deployment_name = None
        
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        
        assert embedding.deployment_name == "text-embedding-ada-002"
    
    @patch('openai.AzureOpenAI')
    def test_embed_success(
        self, mock_azure_class: Mock, mock_settings_azure: Any, mock_openai_response: Mock
    ) -> None:
        """Test successful embedding generation with Azure. / 测试使用 Azure 成功生成 embedding。"""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_response
        mock_azure_class.return_value = mock_client
        
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        result = embedding.embed(["hello", "world"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        # Verify API call uses deployment name / 验证 API 调用使用 deployment name
        mock_client.embeddings.create.assert_called_once_with(
            input=["hello", "world"],
            model="my-embedding-deployment",
        )
    
    @patch('openai.AzureOpenAI')
    def test_embed_api_error(
        self, mock_azure_class: Mock, mock_settings_azure: Any
    ) -> None:
        """Test handling of Azure API errors. / 测试 Azure API 错误处理。"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Azure API Error")
        mock_azure_class.return_value = mock_client
        
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        
        with pytest.raises(AzureEmbeddingError, match="Azure OpenAI Embeddings API call failed"):
            embedding.embed(["test"])
    
    def test_get_dimension_exact_match(self, mock_settings_azure: Any) -> None:
        """Test get_dimension with exact deployment name match. / 测试 deployment name 精确匹配时的 get_dimension。"""
        mock_settings_azure.embedding.deployment_name = "text-embedding-3-small"
        
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        
        assert embedding.get_dimension() == 1536
    
    def test_get_dimension_partial_match(self, mock_settings_azure: Any) -> None:
        """Test get_dimension with partial deployment name match. / 测试 deployment name 部分匹配时的 get_dimension。"""
        mock_settings_azure.embedding.deployment_name = "my-text-embedding-3-large-prod"
        
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        
        assert embedding.get_dimension() == 3072
    
    def test_get_dimension_configured_value(self, mock_settings_azure: Any) -> None:
        """Test get_dimension returns configured dimension when set. / 测试已设置维度时 get_dimension 返回配置的维度。"""
        mock_settings_azure.embedding.dimensions = 768
        
        embedding = AzureEmbedding(
            mock_settings_azure,
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        
        assert embedding.get_dimension() == 768


# =============================================================================
# Factory Integration Tests / Factory 集成测试
# =============================================================================

class TestEmbeddingFactoryRegistration:
    """Test suite for factory registration of OpenAI and Azure providers. / OpenAI 和 Azure providers 的 factory 注册测试套件。"""
    
    def test_openai_registered(self) -> None:
        """Test that OpenAI provider is registered with factory. / 测试 OpenAI provider 已注册到 factory。"""
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
        
        providers = EmbeddingFactory.list_providers()
        assert "openai" in providers
    
    def test_azure_registered(self) -> None:
        """Test that Azure provider is registered with factory. / 测试 Azure provider 已注册到 factory。"""
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
        
        providers = EmbeddingFactory.list_providers()
        assert "azure" in providers
    
    def test_factory_creates_openai(
        self, mock_settings_openai: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test factory creates OpenAI embedding instance. / 测试 factory 创建 OpenAI embedding 实例。"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
        
        embedding = EmbeddingFactory.create(mock_settings_openai)
        
        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.model == "text-embedding-3-small"
    
    def test_factory_creates_azure(
        self, mock_settings_azure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test factory creates Azure embedding instance. / 测试 factory 创建 Azure embedding 实例。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
        
        embedding = EmbeddingFactory.create(mock_settings_azure)
        
        assert isinstance(embedding, AzureEmbedding)
        assert embedding.deployment_name == "my-embedding-deployment"
