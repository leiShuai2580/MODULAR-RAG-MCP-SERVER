"""Unit tests for Azure Vision LLM implementation. / Azure Vision LLM 实现的单元测试。

This module tests the AzureVisionLLM provider implementation, including: / 本模块测试 AzureVisionLLM provider 实现，包括：
- Configuration validation and initialization / 配置校验与初始化
- Image preprocessing and compression / 图片预处理与压缩
- API call structure and error handling / API 调用结构与错误处理
- Factory registration and creation / 工厂注册与创建
- Mock tests covering various scenarios / 覆盖多种场景的 Mock 测试
"""

import base64
import io
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory


# ================================
# Helper Functions / 辅助函数
# ================================

def _has_pil() -> bool:
    """Check if PIL is available. / 检查 PIL 是否可用。"""
    try:
        import PIL
        return True
    except ImportError:
        return False


# ================================
# Test Fixtures / 测试 Fixture
# ================================

class MockSettings:
    """Mock settings object for testing. / 用于测试的 Mock settings 对象。"""
    
    class LLMSettings:
        provider = "azure"
        model = "gpt-4o"
        temperature = 0.7
        max_tokens = 1000
    
    def __init__(self):
        self.llm = self.LLMSettings()
        self.vision_llm = None


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing. / 提供用于测试的 mock settings。"""
    return MockSettings()


@pytest.fixture
def test_image_bytes():
    """Create a small test image as bytes. / 创建小型测试图片字节。"""
    try:
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    except ImportError:
        # Return mock bytes if PIL not available / 如果 PIL 不可用则返回 mock 字节
        return b"fake_image_data"


@pytest.fixture
def test_image_base64(test_image_bytes):
    """Create base64-encoded test image. / 创建 base64 编码的测试图片。"""
    return base64.b64encode(test_image_bytes).decode('utf-8')


# ================================
# Test Initialization / 初始化测试
# ================================

class TestAzureVisionLLMInit:
    """Test Azure Vision LLM initialization and configuration. / 测试 Azure Vision LLM 初始化与配置。"""
    
    def test_init_with_env_vars(self, mock_settings, monkeypatch):
        """Initialize with environment variables. / 使用环境变量初始化。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        
        assert llm.api_key == "test-key"
        assert llm.endpoint == "https://test.openai.azure.com"
        assert llm.deployment_name == "gpt-4o"
        assert llm.api_version == AzureVisionLLM.DEFAULT_API_VERSION
        assert llm.max_image_size == AzureVisionLLM.DEFAULT_MAX_IMAGE_SIZE
    
    def test_init_with_explicit_params(self, mock_settings):
        """Initialize with explicit parameters. / 使用显式参数初始化。"""
        llm = AzureVisionLLM(
            mock_settings,
            api_key="explicit-key",
            endpoint="https://explicit.openai.azure.com",
            deployment_name="gpt-4-vision-preview",
            api_version="2024-03-01-preview",
            max_image_size=1024
        )
        
        assert llm.api_key == "explicit-key"
        assert llm.endpoint == "https://explicit.openai.azure.com"
        assert llm.deployment_name == "gpt-4-vision-preview"
        assert llm.api_version == "2024-03-01-preview"
        assert llm.max_image_size == 1024
    
    def test_init_explicit_overrides_env(self, mock_settings, monkeypatch):
        """Explicit parameters override environment variables. / 显式参数会覆盖环境变量。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        
        llm = AzureVisionLLM(
            mock_settings,
            api_key="explicit-key",
            endpoint="https://explicit.openai.azure.com"
        )
        
        assert llm.api_key == "explicit-key"
        assert llm.endpoint == "https://explicit.openai.azure.com"
    
    def test_init_missing_api_key(self, mock_settings, monkeypatch):
        """Raise error if API key is missing. / 缺失 API key 时抛出错误。"""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        with pytest.raises(ValueError, match="Azure OpenAI API key not provided"):
            AzureVisionLLM(mock_settings)
    
    def test_init_missing_endpoint(self, mock_settings, monkeypatch):
        """Raise error if endpoint is missing. / 缺失 endpoint 时抛出错误。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        
        with pytest.raises(ValueError, match="Azure OpenAI endpoint not provided"):
            AzureVisionLLM(mock_settings)


# ================================
# Test Image Preprocessing / 图片预处理测试
# ================================

class TestImagePreprocessing:
    """Test image preprocessing and compression. / 测试图片预处理与压缩。"""
    
    def test_get_image_base64_from_bytes(self, mock_settings, monkeypatch, test_image_bytes):
        """Convert image bytes to base64. / 将图片字节转换为 base64。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        result = llm._get_image_base64(image)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid base64 / 验证它是有效的 base64
        decoded = base64.b64decode(result)
        assert decoded == test_image_bytes
    
    def test_get_image_base64_from_base64(self, mock_settings, monkeypatch, test_image_base64):
        """Return base64 string unchanged if already encoded. / 如果已经编码则原样返回 base64 字符串。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(base64=test_image_base64)
        
        result = llm._get_image_base64(image)
        
        assert result == test_image_base64
    
    def test_get_image_base64_from_path(self, mock_settings, monkeypatch, test_image_bytes, tmp_path):
        """Convert image file path to base64. / 将图片文件路径转换为 base64。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        # Create temporary image file / 创建临时图片文件
        image_path = tmp_path / "test.png"
        image_path.write_bytes(test_image_bytes)
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(path=str(image_path))
        
        result = llm._get_image_base64(image)
        
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded == test_image_bytes
    
    @pytest.mark.skipif(
        not _has_pil(),
        reason="PIL not available"
    )
    def test_preprocess_image_no_compression_needed(self, mock_settings, monkeypatch, test_image_bytes):
        """Image within size limits is not compressed. / 尺寸限制内的图片不会被压缩。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        # Image is 100x100, max_size is 2048x2048 - no compression needed / 图片为 100x100，max_size 为 2048x2048，无需压缩
        result = llm.preprocess_image(image, max_size=(2048, 2048))
        
        # Should return same image / 应返回相同图片
        assert result.data == test_image_bytes
    
    @pytest.mark.skipif(
        not _has_pil(),
        reason="PIL not available"
    )
    def test_preprocess_image_compression_needed(self, mock_settings, monkeypatch):
        """Large image is compressed to fit max_size. / 大图片会被压缩以适配 max_size。"""
        from PIL import Image
        
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        # Create large image (3000x3000) / 创建大图片（3000x3000）
        large_img = Image.new('RGB', (3000, 3000), color='blue')
        buffer = io.BytesIO()
        large_img.save(buffer, format='PNG')
        large_bytes = buffer.getvalue()
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=large_bytes)
        
        # Compress to max 1024x1024 / 压缩到最大 1024x1024
        result = llm.preprocess_image(image, max_size=(1024, 1024))
        
        # Should return compressed image / 应返回压缩后的图片
        assert result.data != large_bytes
        assert len(result.data) < len(large_bytes)
        
        # Verify compressed image dimensions / 验证压缩后的图片尺寸
        compressed_img = Image.open(io.BytesIO(result.data))
        width, height = compressed_img.size
        assert width <= 1024
        assert height <= 1024
    
    def test_preprocess_image_preserves_aspect_ratio(self, mock_settings, monkeypatch):
        """Compression preserves aspect ratio. / 压缩会保留宽高比。"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")
        
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        # Create rectangular image (2000x1000) / 创建矩形图片（2000x1000）
        rect_img = Image.new('RGB', (2000, 1000), color='green')
        buffer = io.BytesIO()
        rect_img.save(buffer, format='PNG')
        rect_bytes = buffer.getvalue()
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=rect_bytes)
        
        # Compress to max 800x800 / 压缩到最大 800x800
        result = llm.preprocess_image(image, max_size=(800, 800))
        
        # Verify aspect ratio preserved (2:1) / 验证宽高比保持为 2:1
        compressed_img = Image.open(io.BytesIO(result.data))
        width, height = compressed_img.size
        assert abs(width / height - 2.0) < 0.01  # Allow small floating point error / 允许很小的浮点误差
    
    def test_preprocess_image_already_base64(self, mock_settings, monkeypatch, test_image_base64):
        """Base64 images are not preprocessed. / base64 图片不会被预处理。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(base64=test_image_base64)
        
        result = llm.preprocess_image(image, max_size=(100, 100))
        
        # Should return same image unchanged / 应原样返回相同图片
        assert result.base64 == test_image_base64


# ================================
# Test chat_with_image / chat_with_image 测试
# ================================

class TestChatWithImage:
    """Test chat_with_image method. / 测试 chat_with_image 方法。"""
    
    def test_chat_with_image_basic(self, mock_settings, monkeypatch, test_image_bytes):
        """Basic chat with image call. / 基础图片聊天调用。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        # Mock the API call / Mock API 调用
        mock_response = {
            "choices": [{
                "message": {
                    "content": "This is a red square image."
                }
            }],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60
            }
        }
        
        with patch.object(llm, '_call_api', return_value=mock_response):
            response = llm.chat_with_image(
                text="What is in this image?",
                image=image
            )
        
        assert isinstance(response, ChatResponse)
        assert response.content == "This is a red square image."
        assert response.model == "gpt-4o"
        assert response.usage["total_tokens"] == 60
    
    def test_chat_with_image_validates_text(self, mock_settings, monkeypatch, test_image_bytes):
        """Validate text input. / 校验文本输入。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            llm.chat_with_image(text="", image=image)
        
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            llm.chat_with_image(text="   ", image=image)
    
    def test_chat_with_image_validates_image(self, mock_settings, monkeypatch):
        """Validate image input. / 校验图片输入。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        
        with pytest.raises(ValueError, match="Image must be an ImageInput instance"):
            llm.chat_with_image(text="Test", image="not_an_image")  # type: ignore
    
    def test_chat_with_image_with_conversation_history(self, mock_settings, monkeypatch, test_image_bytes):
        """Chat with image includes conversation history. / 图片聊天会包含对话历史。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        
        mock_response = {
            "choices": [{"message": {"content": "Response"}}],
            "model": "gpt-4o"
        }
        
        with patch.object(llm, '_call_api', return_value=mock_response) as mock_call:
            llm.chat_with_image(
                text="What about this image?",
                image=image,
                messages=messages
            )
            
            # Verify history was included / 验证历史消息已被包含
            call_args = mock_call.call_args
            api_messages = call_args.kwargs['messages']
            
            assert len(api_messages) == 3  # 2 history + 1 current / 2 条历史 + 1 条当前消息
            assert api_messages[0]["role"] == "user"
            assert api_messages[0]["content"] == "Hello"
            assert api_messages[1]["role"] == "assistant"
            assert api_messages[1]["content"] == "Hi there!"
    
    def test_chat_with_image_kwargs_override(self, mock_settings, monkeypatch, test_image_bytes):
        """Override parameters via kwargs. / 通过 kwargs 覆盖参数。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        mock_response = {
            "choices": [{"message": {"content": "Response"}}],
            "model": "custom-deployment"
        }
        
        with patch.object(llm, '_call_api', return_value=mock_response) as mock_call:
            llm.chat_with_image(
                text="Test",
                image=image,
                temperature=0.9,
                max_tokens=500,
                deployment_name="custom-deployment"
            )
            
            # Verify overrides were passed to API / 验证覆盖参数已传给 API
            call_args = mock_call.call_args
            assert call_args.kwargs['temperature'] == 0.9
            assert call_args.kwargs['max_tokens'] == 500
            assert call_args.kwargs['deployment'] == "custom-deployment"
    
    def test_chat_with_image_api_error(self, mock_settings, monkeypatch, test_image_bytes):
        """Handle API errors gracefully. / 优雅处理 API 错误。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        # Mock API failure / Mock API 失败
        with patch.object(llm, '_call_api', side_effect=Exception("Connection timeout")):
            with pytest.raises(AzureVisionLLMError, match="API call failed"):
                llm.chat_with_image(text="Test", image=image)
    
    def test_chat_with_image_malformed_response(self, mock_settings, monkeypatch, test_image_bytes):
        """Handle malformed API response. / 处理格式异常的 API 响应。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        llm = AzureVisionLLM(mock_settings)
        image = ImageInput(data=test_image_bytes)
        
        # Mock response missing required keys / Mock 缺失必需 key 的响应
        bad_response = {"choices": []}  # Missing message content / 缺少 message content
        
        with patch.object(llm, '_call_api', return_value=bad_response):
            with pytest.raises(AzureVisionLLMError, match="API call failed.*IndexError"):
                llm.chat_with_image(text="Test", image=image)


# ================================
# Test Factory Integration / 工厂集成测试
# ================================

class TestFactoryIntegration:
    """Test Azure Vision LLM factory registration and creation. / 测试 Azure Vision LLM 工厂注册与创建。"""
    
    def test_azure_registered_in_factory(self):
        """Azure Vision LLM is registered in factory. / Azure Vision LLM 已注册到工厂。"""
        providers = LLMFactory.list_vision_providers()
        assert "azure" in providers
    
    def test_create_vision_llm_from_factory(self, mock_settings, monkeypatch):
        """Create Azure Vision LLM via factory. / 通过工厂创建 Azure Vision LLM。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        
        # Modify settings to have vision_llm config / 修改 settings 使其包含 vision_llm 配置
        class VisionLLMSettings:
            provider = "azure"
            model = "gpt-4o"
            deployment_name = "gpt-4o"
            max_image_size = 2048
            api_key = None
            azure_endpoint = None
            api_version = None
        
        mock_settings.vision_llm = VisionLLMSettings()
        
        llm = LLMFactory.create_vision_llm(mock_settings)
        
        assert isinstance(llm, AzureVisionLLM)
        assert llm.api_key == "test-key"
    
    def test_create_vision_llm_with_override_kwargs(self, mock_settings, monkeypatch):
        """Create Vision LLM with override kwargs. / 使用覆盖 kwargs 创建 Vision LLM。"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        
        class VisionLLMSettings:
            provider = "azure"
            model = "gpt-4o"
            deployment_name = None
            max_image_size = None
            api_key = None
            azure_endpoint = None
            api_version = None
        
        mock_settings.vision_llm = VisionLLMSettings()
        
        llm = LLMFactory.create_vision_llm(
            mock_settings,
            endpoint="https://override.openai.azure.com",
            max_image_size=1024
        )
        
        assert isinstance(llm, AzureVisionLLM)
        assert llm.endpoint == "https://override.openai.azure.com"
        assert llm.max_image_size == 1024

