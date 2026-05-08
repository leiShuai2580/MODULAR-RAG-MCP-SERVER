"""Unit tests for Vision LLM factory and base interface. / Vision LLM 工厂和基础接口的单元测试。

This module tests the Vision LLM factory pattern, provider registration, / 本模块使用 fake 实现测试 Vision LLM 工厂模式、provider 注册，
and the BaseVisionLLM abstract interface using a fake implementation. / 以及 BaseVisionLLM 抽象接口。
"""

import pytest
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory


# ================================
# Fake Implementation for Testing / 用于测试的 Fake 实现
# ================================

class FakeVisionLLM(BaseVisionLLM):
    """Fake Vision LLM implementation for testing. / 用于测试的 Fake Vision LLM 实现。
    
    This implementation returns deterministic responses for testing / 该实现为测试返回确定性响应，
    and tracks call counts for verification. / 并跟踪调用次数以便验证。
    """
    
    def __init__(
        self,
        settings: Any = None,
        response_template: str = "I see: {text} | Image: {image_type}",
        **kwargs: Any
    ):
        """Initialize fake Vision LLM. / 初始化 fake Vision LLM。
        
        Args: / 参数：
            settings: Optional settings object (unused in fake). / 可选 settings 对象（fake 中未使用）。
            response_template: Template for generating responses. / 用于生成响应的模板。
            **kwargs: Additional parameters (unused). / 额外参数（未使用）。
        """
        self.settings = settings
        self.response_template = response_template
        self.call_count = 0
        self.last_text = None
        self.last_image = None
        self.last_messages = None
    
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate fake response based on inputs. / 基于输入生成 fake 响应。"""
        # Validate inputs using base class methods / 使用 base class 方法校验输入
        self.validate_text(text)
        self.validate_image(image)
        
        # Track call / 跟踪调用
        self.call_count += 1
        self.last_text = text
        self.last_image = image
        self.last_messages = messages
        
        # Determine image type for response / 确定响应中的图片类型
        if image.path:
            image_type = f"path({image.path})"
        elif image.data:
            image_type = f"bytes({len(image.data)} bytes)"
        elif image.base64:
            image_type = f"base64({len(image.base64)} chars)"
        else:
            image_type = "unknown"
        
        # Generate deterministic response / 生成确定性响应
        content = self.response_template.format(
            text=text,
            image_type=image_type
        )
        
        return ChatResponse(
            content=content,
            model="fake-vision-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )


# =========================
# Test BaseVisionLLM / BaseVisionLLM 测试
# =========================

class TestBaseVisionLLM:
    """Test the BaseVisionLLM abstract interface. / 测试 BaseVisionLLM 抽象接口。"""
    
    def test_abstract_cannot_instantiate(self):
        """BaseVisionLLM cannot be instantiated directly. / BaseVisionLLM 不能直接实例化。"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseVisionLLM()
    
    def test_validate_text_success(self):
        """validate_text accepts valid text. / validate_text 接受有效文本。"""
        fake = FakeVisionLLM()
        fake.validate_text("valid text")  # Should not raise / 不应抛出异常
    
    def test_validate_text_empty(self):
        """validate_text rejects empty text. / validate_text 拒绝空文本。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            fake.validate_text("")
    
    def test_validate_text_whitespace_only(self):
        """validate_text rejects whitespace-only text. / validate_text 拒绝仅空白文本。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            fake.validate_text("   \n\t  ")
    
    def test_validate_text_non_string(self):
        """validate_text rejects non-string input. / validate_text 拒绝非字符串输入。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Text must be a string"):
            fake.validate_text(123)  # type: ignore
    
    def test_validate_image_success(self):
        """validate_image accepts valid ImageInput. / validate_image 接受有效 ImageInput。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="test.png")
        fake.validate_image(image)  # Should not raise / 不应抛出异常
    
    def test_validate_image_invalid_type(self):
        """validate_image rejects non-ImageInput. / validate_image 拒绝非 ImageInput。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Image must be an ImageInput instance"):
            fake.validate_image("not_an_image_input")  # type: ignore
    
    def test_preprocess_image_default(self):
        """preprocess_image returns image unchanged by default. / preprocess_image 默认原样返回图片。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="test.png")
        result = fake.preprocess_image(image)
        assert result is image  # Same instance / 相同实例


# =========================
# Test ImageInput / ImageInput 测试
# =========================

class TestImageInput:
    """Test the ImageInput dataclass. / 测试 ImageInput dataclass。"""
    
    def test_image_input_path(self):
        """ImageInput can be created with path. / ImageInput 可使用 path 创建。"""
        image = ImageInput(path="test.png")
        assert image.path == "test.png"
        assert image.data is None
        assert image.base64 is None
        assert image.mime_type == "image/png"
    
    def test_image_input_path_as_pathlib(self):
        """ImageInput accepts pathlib.Path. / ImageInput 接受 pathlib.Path。"""
        path = Path("test.png")
        image = ImageInput(path=path)
        assert image.path == path
    
    def test_image_input_data(self):
        """ImageInput can be created with bytes data. / ImageInput 可使用 bytes 数据创建。"""
        data = b"fake_image_bytes"
        image = ImageInput(data=data, mime_type="image/jpeg")
        assert image.data == data
        assert image.path is None
        assert image.base64 is None
        assert image.mime_type == "image/jpeg"
    
    def test_image_input_base64(self):
        """ImageInput can be created with base64 string. / ImageInput 可使用 base64 字符串创建。"""
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAUA"
        image = ImageInput(base64=base64_str)
        assert image.base64 == base64_str
        assert image.path is None
        assert image.data is None
    
    def test_image_input_no_input(self):
        """ImageInput raises error if no input provided. / 未提供输入时 ImageInput 抛出错误。"""
        with pytest.raises(ValueError, match="Must provide one of: path, data, or base64"):
            ImageInput()
    
    def test_image_input_multiple_inputs(self):
        """ImageInput raises error if multiple inputs provided. / 提供多个输入时 ImageInput 抛出错误。"""
        with pytest.raises(ValueError, match="Must provide exactly one of"):
            ImageInput(path="test.png", data=b"bytes")


# =========================
# Test FakeVisionLLM / FakeVisionLLM 测试
# =========================

class TestFakeVisionLLM:
    """Test the FakeVisionLLM implementation. / 测试 FakeVisionLLM 实现。"""
    
    def test_chat_with_image_path(self):
        """FakeVisionLLM generates response for path-based image. / FakeVisionLLM 为基于 path 的图片生成响应。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="diagram.png")
        
        response = fake.chat_with_image(
            text="Describe this diagram",
            image=image
        )
        
        assert response.content == "I see: Describe this diagram | Image: path(diagram.png)"
        assert response.model == "fake-vision-model"
        assert fake.call_count == 1
        assert fake.last_text == "Describe this diagram"
        assert fake.last_image == image
    
    def test_chat_with_image_bytes(self):
        """FakeVisionLLM generates response for bytes-based image. / FakeVisionLLM 为基于 bytes 的图片生成响应。"""
        fake = FakeVisionLLM()
        data = b"fake_image_data_12345"
        image = ImageInput(data=data)
        
        response = fake.chat_with_image(
            text="What is this?",
            image=image
        )
        
        assert "bytes(21 bytes)" in response.content
        assert fake.call_count == 1
    
    def test_chat_with_image_base64(self):
        """FakeVisionLLM generates response for base64-based image. / FakeVisionLLM 为基于 base64 的图片生成响应。"""
        fake = FakeVisionLLM()
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAUA"
        image = ImageInput(base64=base64_str)
        
        response = fake.chat_with_image(
            text="Analyze this",
            image=image
        )
        
        assert f"base64({len(base64_str)} chars)" in response.content
        assert fake.call_count == 1
    
    def test_chat_with_image_with_messages(self):
        """FakeVisionLLM accepts conversation history. / FakeVisionLLM 接受对话历史。"""
        fake = FakeVisionLLM()
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Previous question")
        ]
        image = ImageInput(path="test.png")
        
        response = fake.chat_with_image(
            text="New question",
            image=image,
            messages=messages
        )
        
        assert response.content is not None
        assert fake.last_messages == messages
    
    def test_chat_with_image_validates_text(self):
        """FakeVisionLLM validates text input. / FakeVisionLLM 校验文本输入。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="test.png")
        
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            fake.chat_with_image(text="", image=image)
    
    def test_chat_with_image_validates_image(self):
        """FakeVisionLLM validates image input. / FakeVisionLLM 校验图片输入。"""
        fake = FakeVisionLLM()
        
        with pytest.raises(ValueError, match="Image must be an ImageInput instance"):
            fake.chat_with_image(text="test", image="not_an_image")  # type: ignore
    
    def test_custom_response_template(self):
        """FakeVisionLLM accepts custom response template. / FakeVisionLLM 接受自定义响应模板。"""
        fake = FakeVisionLLM(response_template="Custom: {text}")
        image = ImageInput(path="test.png")
        
        response = fake.chat_with_image(text="Hello", image=image)
        
        assert response.content == "Custom: Hello"


# ================================
# Test Vision LLM Factory / Vision LLM 工厂测试
# ================================

class TestVisionLLMFactory:
    """Test the Vision LLM factory pattern. / 测试 Vision LLM 工厂模式。"""
    
    def setup_method(self):
        """Clean up registry before each test. / 每个测试前清理注册表。"""
        LLMFactory._VISION_PROVIDERS.clear()
    
    def test_register_vision_provider_success(self):
        """register_vision_provider registers valid provider. / register_vision_provider 注册有效 provider。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        assert "fake" in LLMFactory._VISION_PROVIDERS
        assert LLMFactory._VISION_PROVIDERS["fake"] == FakeVisionLLM
    
    def test_register_vision_provider_case_insensitive(self):
        """register_vision_provider normalizes provider name to lowercase. / register_vision_provider 将 provider 名称规范化为小写。"""
        LLMFactory.register_vision_provider("FakeVision", FakeVisionLLM)
        
        assert "fakevision" in LLMFactory._VISION_PROVIDERS
        assert "FakeVision" not in LLMFactory._VISION_PROVIDERS
    
    def test_register_vision_provider_invalid_class(self):
        """register_vision_provider rejects non-BaseVisionLLM class. / register_vision_provider 拒绝非 BaseVisionLLM 类。"""
        class NotAVisionLLM:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseVisionLLM"):
            LLMFactory.register_vision_provider("invalid", NotAVisionLLM)  # type: ignore
    
    def test_list_vision_providers_empty(self):
        """list_vision_providers returns empty list when no providers registered. / 未注册 providers 时 list_vision_providers 返回空列表。"""
        assert LLMFactory.list_vision_providers() == []
    
    def test_list_vision_providers_sorted(self):
        """list_vision_providers returns sorted provider names. / list_vision_providers 返回排序后的 provider 名称。"""
        LLMFactory.register_vision_provider("zebra", FakeVisionLLM)
        LLMFactory.register_vision_provider("alpha", FakeVisionLLM)
        LLMFactory.register_vision_provider("beta", FakeVisionLLM)
        
        providers = LLMFactory.list_vision_providers()
        assert providers == ["alpha", "beta", "zebra"]
    
    def test_create_vision_llm_success(self):
        """create_vision_llm creates instance from vision_llm config. / create_vision_llm 从 vision_llm 配置创建实例。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        # Mock settings with vision_llm section / Mock 带 vision_llm section 的 settings
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        assert isinstance(vision_llm, FakeVisionLLM)
        assert vision_llm.settings == settings
    
    def test_create_vision_llm_fallback_to_llm_config(self):
        """create_vision_llm falls back to llm.provider if vision_llm not present. / vision_llm 不存在时 create_vision_llm 回退到 llm.provider。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        # Mock settings with only llm section (no vision_llm) / Mock 只有 llm section 的 settings（无 vision_llm）
        class FakeSettings:
            class LLM:
                provider = "fake"
            llm = LLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        assert isinstance(vision_llm, FakeVisionLLM)
    
    def test_create_vision_llm_case_insensitive(self):
        """create_vision_llm handles case-insensitive provider names. / create_vision_llm 处理大小写不敏感的 provider 名称。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        class FakeSettings:
            class VisionLLM:
                provider = "FAKE"  # Uppercase / 大写
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        assert isinstance(vision_llm, FakeVisionLLM)
    
    def test_create_vision_llm_unknown_provider(self):
        """create_vision_llm raises error for unknown provider. / 未知 provider 时 create_vision_llm 抛出错误。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        class FakeSettings:
            class VisionLLM:
                provider = "unknown"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        
        with pytest.raises(ValueError, match="Unsupported Vision LLM provider: 'unknown'"):
            LLMFactory.create_vision_llm(settings)
    
    def test_create_vision_llm_missing_config(self):
        """create_vision_llm raises error if config is missing. / config 缺失时 create_vision_llm 抛出错误。"""
        class FakeSettings:
            pass  # No vision_llm or llm section / 没有 vision_llm 或 llm section
        
        settings = FakeSettings()
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            LLMFactory.create_vision_llm(settings)
    
    def test_create_vision_llm_no_providers_registered(self):
        """create_vision_llm shows available: none when registry is empty. / 注册表为空时 create_vision_llm 显示 available: none。"""
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        
        with pytest.raises(ValueError, match="Available Vision LLM providers: none"):
            LLMFactory.create_vision_llm(settings)
    
    def test_create_vision_llm_with_overrides(self):
        """create_vision_llm passes override kwargs to provider. / create_vision_llm 将覆盖 kwargs 传给 provider。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(
            settings,
            response_template="Override: {text}"
        )
        
        assert isinstance(vision_llm, FakeVisionLLM)
        assert vision_llm.response_template == "Override: {text}"
    
    def test_create_vision_llm_provider_instantiation_failure(self):
        """create_vision_llm handles provider instantiation errors. / create_vision_llm 处理 provider 实例化错误。"""
        class BrokenVisionLLM(BaseVisionLLM):
            def __init__(self, settings, **kwargs):
                raise RuntimeError("Intentional failure")
            
            def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
                pass
        
        LLMFactory.register_vision_provider("broken", BrokenVisionLLM)
        
        class FakeSettings:
            class VisionLLM:
                provider = "broken"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        
        with pytest.raises(RuntimeError, match="Failed to instantiate Vision LLM provider 'broken'"):
            LLMFactory.create_vision_llm(settings)


# ================================
# Integration Tests / 集成测试
# ================================

class TestVisionLLMIntegration:
    """Integration tests combining factory and implementation. / 组合工厂和实现的集成测试。"""
    
    def setup_method(self):
        """Clean up registry and register fake provider. / 清理注册表并注册 fake provider。"""
        LLMFactory._VISION_PROVIDERS.clear()
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
    
    def test_end_to_end_vision_workflow(self):
        """Full workflow: create from factory -> call vision method. / 完整流程：从工厂创建 -> 调用 vision 方法。"""
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        
        # Create Vision LLM from factory / 从工厂创建 Vision LLM
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        # Use it for image captioning / 将其用于图片 captioning
        image = ImageInput(path="document.pdf.page1.png")
        response = vision_llm.chat_with_image(
            text="Describe the main content of this page",
            image=image
        )
        
        assert "Describe the main content" in response.content
        assert "document.pdf.page1.png" in response.content
        assert response.model == "fake-vision-model"
