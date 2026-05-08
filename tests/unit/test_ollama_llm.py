"""Unit tests for Ollama LLM implementation. / Ollama LLM 实现的单元测试。

This module tests the Ollama LLM provider using mocked HTTP responses / 本模块使用 mocked HTTP responses 测试 Ollama LLM provider，
to avoid requiring a running Ollama instance. / 以避免要求运行中的 Ollama 实例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm import LLMFactory, Message, OllamaLLM, OllamaLLMError


# -----------------------------------------------------------------------------
# Test Fixtures / 测试 Fixture
# -----------------------------------------------------------------------------


@dataclass
class MockLLMSettings:
    """Mock settings for LLM testing. / 用于 LLM 测试的 mock settings。"""
    
    provider: str = "ollama"
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class MockSettings:
    """Mock application settings. / Mock 应用 settings。"""
    
    llm: MockLLMSettings = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = MockLLMSettings()


def make_ollama_response(
    content: str = "Hello! I'm running locally via Ollama.",
    model: str = "llama3",
    status_code: int = 200,
    prompt_eval_count: int = 15,
    eval_count: int = 25,
) -> MagicMock:
    """Create a mock Ollama HTTP response. / 创建 mock Ollama HTTP 响应。
    
    Ollama's /api/chat endpoint returns a different format than OpenAI: / Ollama 的 /api/chat endpoint 返回与 OpenAI 不同的格式：
    - Uses 'message' object with 'role' and 'content' / 使用带 'role' 和 'content' 的 'message' 对象
    - Uses 'eval_count' and 'prompt_eval_count' for token counts / 使用 'eval_count' 和 'prompt_eval_count' 表示 token 数量
    """
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "model": model,
        "created_at": "2026-01-28T12:00:00.000000Z",
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1234567890,
        "load_duration": 12345678,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": 123456789,
        "eval_count": eval_count,
        "eval_duration": 987654321,
    }
    return response


def make_error_response(
    status_code: int = 400,
    error_message: str = "model not found",
) -> MagicMock:
    """Create a mock error HTTP response. / 创建 mock 错误 HTTP 响应。"""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"error": error_message}
    response.text = f"Error: {error_message}"
    return response


# -----------------------------------------------------------------------------
# Factory Registration Tests / 工厂注册测试
# -----------------------------------------------------------------------------


class TestOllamaFactoryRegistration:
    """Tests for Ollama provider factory registration. / Ollama provider 工厂注册测试。"""
    
    @pytest.fixture(autouse=True)
    def ensure_provider_registered(self):
        """Ensure Ollama provider is registered before each test. / 确保每个测试前 Ollama provider 已注册。"""
        if "ollama" not in LLMFactory._PROVIDERS:
            LLMFactory.register_provider("ollama", OllamaLLM)
        yield
    
    def test_ollama_registered(self):
        """Ollama provider should be registered with factory. / Ollama provider 应注册到工厂。"""
        assert "ollama" in LLMFactory.list_providers()
    
    def test_factory_creates_ollama(self):
        """Factory should create OllamaLLM instance when provider=ollama. / provider=ollama 时工厂应创建 OllamaLLM 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="ollama"))
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OllamaLLM)
    
    def test_factory_creates_ollama_case_insensitive(self):
        """Factory should handle case-insensitive provider name. / 工厂应处理大小写不敏感的 provider 名称。"""
        settings = MockSettings(llm=MockLLMSettings(provider="OLLAMA"))
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OllamaLLM)


# -----------------------------------------------------------------------------
# Initialization Tests / 初始化测试
# -----------------------------------------------------------------------------


class TestOllamaInit:
    """Tests for OllamaLLM initialization. / OllamaLLM 初始化测试。"""
    
    def test_init_default_base_url(self):
        """Should use default localhost URL when none specified. / 未指定时应使用默认 localhost URL。"""
        settings = MockSettings()
        llm = OllamaLLM(settings)
        assert llm.base_url == "http://localhost:11434"
    
    def test_init_custom_base_url(self):
        """Should use provided base URL. / 应使用提供的 base URL。"""
        settings = MockSettings()
        llm = OllamaLLM(settings, base_url="http://192.168.1.100:11434")
        assert llm.base_url == "http://192.168.1.100:11434"
    
    def test_init_base_url_from_env(self):
        """Should read base URL from environment variable. / 应从环境变量读取 base URL。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://remote:11434"}):
            llm = OllamaLLM(settings)
            assert llm.base_url == "http://remote:11434"
    
    def test_init_explicit_base_url_overrides_env(self):
        """Explicit base URL should override environment variable. / 显式 base URL 应覆盖环境变量。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env:11434"}):
            llm = OllamaLLM(settings, base_url="http://explicit:11434")
            assert llm.base_url == "http://explicit:11434"
    
    def test_init_model_from_settings(self):
        """Should read model from settings. / 应从 settings 读取 model。"""
        settings = MockSettings(llm=MockLLMSettings(model="mistral"))
        llm = OllamaLLM(settings)
        assert llm.model == "mistral"
    
    def test_init_temperature_from_settings(self):
        """Should read temperature from settings. / 应从 settings 读取 temperature。"""
        settings = MockSettings(llm=MockLLMSettings(temperature=0.5))
        llm = OllamaLLM(settings)
        assert llm.default_temperature == 0.5
    
    def test_init_max_tokens_from_settings(self):
        """Should read max_tokens from settings. / 应从 settings 读取 max_tokens。"""
        settings = MockSettings(llm=MockLLMSettings(max_tokens=4096))
        llm = OllamaLLM(settings)
        assert llm.default_max_tokens == 4096
    
    def test_init_custom_timeout(self):
        """Should accept custom timeout. / 应接受自定义 timeout。"""
        settings = MockSettings()
        llm = OllamaLLM(settings, timeout=300.0)
        assert llm.timeout == 300.0
    
    def test_init_default_timeout(self):
        """Should use default timeout when none specified. / 未指定时应使用默认 timeout。"""
        settings = MockSettings()
        llm = OllamaLLM(settings)
        assert llm.timeout == 120.0  # Longer default for local inference / 本地推理使用更长默认值


# -----------------------------------------------------------------------------
# Chat Tests with Mocked HTTP / 使用 Mocked HTTP 的 Chat 测试
# -----------------------------------------------------------------------------


class TestOllamaChat:
    """Tests for OllamaLLM chat functionality with mocked HTTP. / 使用 mocked HTTP 测试 OllamaLLM chat 功能。"""
    
    @pytest.fixture
    def llm(self):
        """Create an OllamaLLM instance for testing. / 创建用于测试的 OllamaLLM 实例。"""
        settings = MockSettings()
        return OllamaLLM(settings)
    
    def test_chat_success(self, llm):
        """Should successfully process a chat request. / 应成功处理 chat 请求。"""
        mock_response = make_ollama_response(content="Hello from Ollama!")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "Hello from Ollama!"
            assert response.model == "llama3"
    
    def test_chat_returns_usage_stats(self, llm):
        """Should include usage statistics in response. / 响应中应包含 usage 统计。"""
        mock_response = make_ollama_response(
            prompt_eval_count=20,
            eval_count=50,
        )
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.usage is not None
            assert response.usage["prompt_tokens"] == 20
            assert response.usage["completion_tokens"] == 50
            assert response.usage["total_tokens"] == 70
    
    def test_chat_preserves_raw_response(self, llm):
        """Should preserve raw response for debugging. / 应保留 raw response 以便调试。"""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.raw_response is not None
            assert "model" in response.raw_response
    
    def test_chat_with_system_message(self, llm):
        """Should handle system messages correctly. / 应正确处理 system messages。"""
        mock_response = make_ollama_response(content="I understand the context.")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
            response = llm.chat(messages)
            
            assert response.content == "I understand the context."
            
            # Verify API was called with both messages / 验证 API 使用两条 messages 调用
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
    
    def test_chat_with_conversation(self, llm):
        """Should handle multi-turn conversations. / 应处理多轮对话。"""
        mock_response = make_ollama_response(content="The capital is Paris.")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            messages = [
                Message(role="user", content="What is the capital of France?"),
                Message(role="assistant", content="Paris is the capital of France."),
                Message(role="user", content="Tell me more about it."),
            ]
            response = llm.chat(messages)
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert len(payload["messages"]) == 3
    
    def test_chat_with_temperature_override(self, llm):
        """Should allow temperature override in chat call. / chat 调用中应允许覆盖 temperature。"""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")], temperature=0.1)
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["temperature"] == 0.1
    
    def test_chat_with_max_tokens_override(self, llm):
        """Should allow max_tokens override in chat call. / chat 调用中应允许覆盖 max_tokens。"""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")], max_tokens=500)
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["num_predict"] == 500  # Ollama uses num_predict / Ollama 使用 num_predict
    
    def test_chat_uses_stream_false(self, llm):
        """Should disable streaming for synchronous response. / 同步响应应禁用 streaming。"""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")])
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["stream"] is False


# -----------------------------------------------------------------------------
# Validation Tests / 校验测试
# -----------------------------------------------------------------------------


class TestOllamaValidation:
    """Tests for input validation. / 输入校验测试。"""
    
    @pytest.fixture
    def llm(self):
        """Create an OllamaLLM instance for testing. / 创建用于测试的 OllamaLLM 实例。"""
        settings = MockSettings()
        return OllamaLLM(settings)
    
    def test_chat_empty_messages_raises(self, llm):
        """Should raise ValueError for empty messages list. / 空 messages 列表应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            llm.chat([])
    
    def test_chat_invalid_role_raises(self, llm):
        """Should raise ValueError for invalid message role. / 无效 message role 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="invalid role"):
            llm.chat([Message(role="invalid", content="Hello")])
    
    def test_chat_empty_content_raises(self, llm):
        """Should raise ValueError for empty message content. / 空 message content 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="empty content"):
            llm.chat([Message(role="user", content="")])


# -----------------------------------------------------------------------------
# Error Handling Tests / 错误处理测试
# -----------------------------------------------------------------------------


class TestOllamaErrorHandling:
    """Tests for error handling scenarios. / 错误处理场景测试。"""
    
    @pytest.fixture
    def llm(self):
        """Create an OllamaLLM instance for testing. / 创建用于测试的 OllamaLLM 实例。"""
        settings = MockSettings()
        return OllamaLLM(settings)
    
    def test_api_error_response(self, llm):
        """Should raise OllamaLLMError on API error response. / API 错误响应时应抛出 OllamaLLMError。"""
        mock_response = make_error_response(status_code=404, error_message="model not found")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "HTTP 404" in str(exc_info.value)
            assert "model not found" in str(exc_info.value)
    
    def test_timeout_error(self, llm):
        """Should raise OllamaLLMError on timeout with helpful message. / 超时时应抛出带帮助信息的 OllamaLLMError。"""
        import httpx
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timeout")
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "timed out" in str(exc_info.value).lower()
            assert "120" in str(exc_info.value)  # Default timeout value / 默认 timeout 值
    
    def test_connection_error(self, llm):
        """Should raise OllamaLLMError on connection failure with helpful message. / 连接失败时应抛出带帮助信息的 OllamaLLMError。"""
        import httpx
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError("Connection refused")
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            error_msg = str(exc_info.value)
            assert "Connection failed" in error_msg
            assert "ollama serve" in error_msg.lower()  # Helpful hint / 有用提示
    
    def test_request_error(self, llm):
        """Should raise OllamaLLMError on general request failure. / 通用请求失败时应抛出 OllamaLLMError。"""
        import httpx
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.RequestError("Network error")
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "[Ollama]" in str(exc_info.value)
    
    def test_error_does_not_leak_sensitive_info(self, llm):
        """Error messages should not expose internal URLs or config details. / 错误消息不应暴露内部 URL 或配置细节。"""
        mock_response = make_error_response(status_code=500, error_message="internal error")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            error_msg = str(exc_info.value)
            # Should not contain the full internal URL / 不应包含完整内部 URL
            assert "localhost:11434" not in error_msg
    
    def test_unexpected_response_format(self, llm):
        """Should handle unexpected response format gracefully. / 应优雅处理非预期响应格式。"""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"unexpected": "format"}
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = response
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "Unexpected response format" in str(exc_info.value)


# -----------------------------------------------------------------------------
# API URL Construction Tests / API URL 构造测试
# -----------------------------------------------------------------------------


class TestOllamaAPIEndpoint:
    """Tests for API endpoint URL construction. / API endpoint URL 构造测试。"""
    
    def test_api_url_construction(self):
        """Should construct correct API URL. / 应构造正确的 API URL。"""
        settings = MockSettings()
        llm = OllamaLLM(settings, base_url="http://localhost:11434")
        
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")])
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.args[0] == "http://localhost:11434/api/chat"
    
    def test_api_url_strips_trailing_slash(self):
        """Should handle base URL with trailing slash. / 应处理带尾部斜杠的 base URL。"""
        settings = MockSettings()
        llm = OllamaLLM(settings, base_url="http://localhost:11434/")
        
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")])
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.args[0] == "http://localhost:11434/api/chat"


# -----------------------------------------------------------------------------
# Model Override Tests / Model 覆盖测试
# -----------------------------------------------------------------------------


class TestOllamaModelOverride:
    """Tests for model override functionality. / model 覆盖功能测试。"""
    
    def test_model_override_in_chat(self):
        """Should allow model override per chat call. / 每次 chat 调用应允许覆盖 model。"""
        settings = MockSettings(llm=MockLLMSettings(model="llama3"))
        llm = OllamaLLM(settings)
        
        mock_response = make_ollama_response(model="mistral")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat(
                [Message(role="user", content="Hello")],
                model="mistral",
            )
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["model"] == "mistral"
            assert response.model == "mistral"
