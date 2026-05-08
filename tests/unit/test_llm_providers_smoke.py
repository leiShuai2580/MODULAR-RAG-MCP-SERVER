"""Smoke tests for LLM provider implementations. / LLM provider 实现的 smoke tests。

This module tests the OpenAI, Azure, and DeepSeek LLM providers / 本模块测试 OpenAI、Azure 和 DeepSeek LLM providers，
using mocked HTTP responses to avoid real API calls. / 使用 mock HTTP 响应以避免真实 API 调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm import (
    AzureLLM,
    AzureLLMError,
    BaseLLM,
    DeepSeekLLM,
    DeepSeekLLMError,
    LLMFactory,
    Message,
    OpenAILLM,
    OpenAILLMError,
)


# -----------------------------------------------------------------------------
# Module-level Setup: Ensure providers are registered / 模块级设置：确保 providers 已注册
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ensure_providers_registered():
    """Ensure all LLM providers are registered before each test. / 确保每个测试前所有 LLM providers 都已注册。
    
    This is needed because other tests (e.g., test_llm_factory.py) may / 这是必需的，因为其他测试（例如 test_llm_factory.py）可能
    clear the provider registry during their setup. / 在 setup 期间清空 provider 注册表。
    """
    # Re-register providers if they've been cleared / 如果 providers 已被清空，则重新注册
    if "openai" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("openai", OpenAILLM)
    if "azure" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("azure", AzureLLM)
    if "deepseek" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("deepseek", DeepSeekLLM)
    yield


# -----------------------------------------------------------------------------
# Test Fixtures / 测试 Fixtures
# -----------------------------------------------------------------------------


@dataclass
class MockLLMSettings:
    """Mock settings for LLM testing. / 用于 LLM 测试的 mock settings。"""
    
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class MockSettings:
    """Mock application settings. / Mock 应用 settings。"""
    
    llm: MockLLMSettings = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = MockLLMSettings()


def make_mock_response(
    content: str = "Hello! How can I help you?",
    model: str = "gpt-4o-mini",
    status_code: int = 200,
) -> MagicMock:
    """Create a mock HTTP response. / 创建 mock HTTP 响应。"""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1704067200,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    return response


def make_error_response(
    status_code: int = 400,
    error_message: str = "Invalid request",
) -> MagicMock:
    """Create a mock error HTTP response. / 创建 mock 错误 HTTP 响应。"""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "error": {
            "message": error_message,
            "type": "invalid_request_error",
        }
    }
    response.text = f"Error: {error_message}"
    return response


# -----------------------------------------------------------------------------
# Factory Registration Tests / 工厂注册测试
# -----------------------------------------------------------------------------


class TestLLMFactoryRegistration:
    """Tests for LLM factory provider registration. / LLM 工厂 provider 注册测试。"""
    
    def test_openai_registered(self):
        """OpenAI provider should be registered. / OpenAI provider 应已注册。"""
        assert "openai" in LLMFactory.list_providers()
    
    def test_azure_registered(self):
        """Azure provider should be registered. / Azure provider 应已注册。"""
        assert "azure" in LLMFactory.list_providers()
    
    def test_deepseek_registered(self):
        """DeepSeek provider should be registered. / DeepSeek provider 应已注册。"""
        assert "deepseek" in LLMFactory.list_providers()
    
    def test_factory_creates_openai(self):
        """Factory should create OpenAI instance when provider=openai. / provider=openai 时工厂应创建 OpenAI 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="openai"))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, OpenAILLM)
    
    def test_factory_creates_azure(self):
        """Factory should create Azure instance when provider=azure. / provider=azure 时工厂应创建 Azure 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="azure"))
        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env_vars):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, AzureLLM)
    
    def test_factory_creates_deepseek(self):
        """Factory should create DeepSeek instance when provider=deepseek. / provider=deepseek 时工厂应创建 DeepSeek 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="deepseek"))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, DeepSeekLLM)
    
    def test_factory_unknown_provider_error(self):
        """Factory should raise error for unknown provider. / 未知 provider 时工厂应抛出错误。"""
        settings = MockSettings(llm=MockLLMSettings(provider="unknown"))
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create(settings)


# -----------------------------------------------------------------------------
# OpenAI LLM Tests / OpenAI LLM 测试
# -----------------------------------------------------------------------------


class TestOpenAILLM:
    """Tests for OpenAI LLM implementation. / OpenAI LLM 实现测试。"""
    
    def test_init_with_api_key(self):
        """Should initialize with provided API key. / 应使用提供的 API key 初始化。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        assert llm.api_key == "test-key"
        assert llm.model == "gpt-4o-mini"
    
    def test_init_with_env_var(self):
        """Should initialize with API key from environment. / 应使用环境变量中的 API key 初始化。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            llm = OpenAILLM(settings)
            assert llm.api_key == "env-key"
    
    def test_init_missing_api_key(self):
        """Should raise error when API key is missing. / API key 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {}, clear=True):
            # Ensure OPENAI_API_KEY is not in environment / 确保 OPENAI_API_KEY 不在环境变量中
            import os
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            with pytest.raises(ValueError, match="API key not provided"):
                OpenAILLM(settings)
    
    def test_custom_base_url(self):
        """Should use custom base URL when provided. / 提供自定义 base URL 时应使用它。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key", base_url="https://custom.api.com")
        assert llm.base_url == "https://custom.api.com"
    
    def test_chat_success(self):
        """Should return ChatResponse on successful API call. / API 调用成功时应返回 ChatResponse。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_mock_response("Test response", "gpt-4o-mini")
            )
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "Test response"
            assert response.model == "gpt-4o-mini"
            assert response.usage["total_tokens"] == 30
    
    def test_chat_empty_messages_error(self):
        """Should raise ValueError for empty messages list. / messages 列表为空时应抛出 ValueError。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            llm.chat([])
    
    def test_chat_invalid_role_error(self):
        """Should raise ValueError for invalid message role. / message role 无效时应抛出 ValueError。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with pytest.raises(ValueError, match="invalid role"):
            llm.chat([Message(role="invalid", content="Hello")])
    
    def test_chat_api_error(self):
        """Should raise OpenAILLMError on API error. / API 错误时应抛出 OpenAILLMError。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_error_response(400, "Bad request")
            )
            
            with pytest.raises(OpenAILLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


# -----------------------------------------------------------------------------
# Azure LLM Tests / Azure LLM 测试
# -----------------------------------------------------------------------------


class TestAzureLLM:
    """Tests for Azure OpenAI LLM implementation. / Azure OpenAI LLM 实现测试。"""
    
    def test_init_with_credentials(self):
        """Should initialize with provided credentials. / 应使用提供的 credentials 初始化。"""
        settings = MockSettings()
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )
        assert llm.api_key == "test-key"
        assert llm.endpoint == "https://test.openai.azure.com"
    
    def test_init_with_env_vars(self):
        """Should initialize with credentials from environment. / 应使用环境变量中的 credentials 初始化。"""
        settings = MockSettings()
        env_vars = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
        }
        with patch.dict("os.environ", env_vars):
            llm = AzureLLM(settings)
            assert llm.api_key == "env-key"
            assert llm.endpoint == "https://env.openai.azure.com"
    
    def test_init_missing_api_key(self):
        """Should raise error when API key is missing. / API key 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.com"}):
            with pytest.raises(ValueError, match="API key not provided"):
                AzureLLM(settings)
    
    def test_init_missing_endpoint(self):
        """Should raise error when endpoint is missing. / endpoint 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="endpoint not provided"):
                AzureLLM(settings)
    
    def test_chat_success(self):
        """Should return ChatResponse on successful API call. / API 调用成功时应返回 ChatResponse。"""
        settings = MockSettings()
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_mock_response("Azure response", "gpt-4o-mini")
            )
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "Azure response"
            assert response.usage["total_tokens"] == 30
    
    def test_chat_api_error(self):
        """Should raise AzureLLMError on API error. / API 错误时应抛出 AzureLLMError。"""
        settings = MockSettings()
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_error_response(401, "Unauthorized")
            )
            
            with pytest.raises(AzureLLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


# -----------------------------------------------------------------------------
# DeepSeek LLM Tests / DeepSeek LLM 测试
# -----------------------------------------------------------------------------


class TestDeepSeekLLM:
    """Tests for DeepSeek LLM implementation. / DeepSeek LLM 实现测试。"""
    
    def test_init_with_api_key(self):
        """Should initialize with provided API key. / 应使用提供的 API key 初始化。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key")
        assert llm.api_key == "test-key"
        assert llm.base_url == "https://api.deepseek.com"
    
    def test_init_with_env_var(self):
        """Should initialize with API key from environment. / 应使用环境变量中的 API key 初始化。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}):
            llm = DeepSeekLLM(settings)
            assert llm.api_key == "env-key"
    
    def test_init_missing_api_key(self):
        """Should raise error when API key is missing. / API key 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {}, clear=True):
            import os
            if "DEEPSEEK_API_KEY" in os.environ:
                del os.environ["DEEPSEEK_API_KEY"]
            with pytest.raises(ValueError, match="API key not provided"):
                DeepSeekLLM(settings)
    
    def test_custom_base_url(self):
        """Should use custom base URL when provided. / 提供自定义 base URL 时应使用它。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key", base_url="https://custom.deepseek.com")
        assert llm.base_url == "https://custom.deepseek.com"
    
    def test_chat_success(self):
        """Should return ChatResponse on successful API call. / API 调用成功时应返回 ChatResponse。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_mock_response("DeepSeek response", "deepseek-chat")
            )
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "DeepSeek response"
            assert response.model == "deepseek-chat"
    
    def test_chat_api_error(self):
        """Should raise DeepSeekLLMError on API error. / API 错误时应抛出 DeepSeekLLMError。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_error_response(500, "Internal server error")
            )
            
            with pytest.raises(DeepSeekLLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


# -----------------------------------------------------------------------------
# Message Validation Tests / Message 校验测试
# -----------------------------------------------------------------------------


class TestMessageValidation:
    """Tests for message validation across all providers. / 所有 providers 的 message 校验测试。"""
    
    @pytest.mark.parametrize("llm_class,api_key_env", [
        (OpenAILLM, "OPENAI_API_KEY"),
        (DeepSeekLLM, "DEEPSEEK_API_KEY"),
    ])
    def test_empty_content_validation(self, llm_class, api_key_env):
        """Should reject messages with empty content. / 应拒绝 content 为空的 messages。"""
        settings = MockSettings()
        with patch.dict("os.environ", {api_key_env: "test-key"}):
            llm = llm_class(settings)
            with pytest.raises(ValueError, match="empty content"):
                llm.chat([Message(role="user", content="")])
    
    @pytest.mark.parametrize("llm_class,api_key_env", [
        (OpenAILLM, "OPENAI_API_KEY"),
        (DeepSeekLLM, "DEEPSEEK_API_KEY"),
    ])
    def test_valid_roles_accepted(self, llm_class, api_key_env):
        """Should accept valid roles: system, user, assistant. / 应接受有效 roles：system、user、assistant。"""
        settings = MockSettings()
        with patch.dict("os.environ", {api_key_env: "test-key"}):
            llm = llm_class(settings)
            
            # These should not raise validation errors / 这些不应抛出校验错误
            messages = [
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ]
            
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    make_mock_response()
                )
                # Should not raise / 不应抛出异常
                llm.chat(messages)


# -----------------------------------------------------------------------------
# Integration-Style Tests (Still Mocked) / 集成风格测试（仍使用 mock）
# -----------------------------------------------------------------------------


class TestLLMIntegration:
    """Integration-style tests using the factory pattern. / 使用工厂模式的集成风格测试。"""
    
    def test_factory_to_chat_flow_openai(self):
        """Test complete flow: factory -> create -> chat for OpenAI. / 测试 OpenAI 的完整流程：factory -> create -> chat。"""
        settings = MockSettings(llm=MockLLMSettings(provider="openai"))
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    make_mock_response("Integration test response")
                )
                
                response = llm.chat([Message(role="user", content="Test")])
                assert response.content == "Integration test response"
    
    def test_factory_to_chat_flow_deepseek(self):
        """Test complete flow: factory -> create -> chat for DeepSeek. / 测试 DeepSeek 的完整流程：factory -> create -> chat。"""
        settings = MockSettings(llm=MockLLMSettings(provider="deepseek"))
        
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    make_mock_response("DeepSeek integration response")
                )
                
                response = llm.chat([Message(role="user", content="Test")])
                assert response.content == "DeepSeek integration response"
