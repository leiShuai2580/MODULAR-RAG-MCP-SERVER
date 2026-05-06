import pytest
from pathlib import Path
from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner

@pytest.mark.integration
def test_image_captioner_azure_integration():
    """Integration test for ImageCaptioner using real Azure OpenAI Vision LLM. / 使用真实 Azure OpenAI Vision LLM 的 ImageCaptioner 集成测试。
    
    Requires valid credentials in config/settings.yaml (configured by user). / 需要 config/settings.yaml 中存在有效凭据（由用户配置）。
    """
    # 1. Load Settings / 1. 加载 Settings
    settings = load_settings("config/settings.yaml")
    
    # Skip if vision not enabled or provider not configured / 如果未启用 vision 或未配置 provider，则跳过
    if not settings.vision_llm or not settings.vision_llm.enabled:
        pytest.skip("Vision LLM not enabled in settings")
        
    if settings.vision_llm.provider != "azure":
        pytest.skip("Test specific for Azure provider (as requested)")

    # 2. Check Test Image / 2. 检查测试图片
    image_path = Path("tests/fixtures/sample_documents/test_vision_llm.jpg")
    if not image_path.exists():
        pytest.fail(f"Test image not found at {image_path}")
        
    # 3. Create Sample Chunk / 3. 创建示例 Chunk
    # Emulate a chunk that came from a loader / 模拟来自 loader 的 chunk
    chunk = Chunk(
        id="chunk_test_001",
        text="Here is an image: [IMAGE: img_001]",
        metadata={
            "source_path": str(image_path),
            "images": [
                {
                    "id": "img_001",
                    "path": str(image_path),
                    "page": 1
                }
            ]
        }
    )
    
    # 4. Initialize ImageCaptioner / 4. 初始化 ImageCaptioner
    captioner = ImageCaptioner(settings=settings)
    
    # 5. Run Transform / 5. 运行 Transform
    # This calls the real API / 这里会调用真实 API
    processed_chunks = captioner.transform([chunk])
    
    # 6. Verify Results / 6. 验证结果
    assert len(processed_chunks) == 1
    processed_chunk = processed_chunks[0]
    
    # Check text modification / 检查文本修改
    print(f"\nOriginal Text: 'Here is an image: [IMAGE: img_001]'")
    print(f"New Text: '{processed_chunk.text}'")
    
    assert "[IMAGE: img_001]" in processed_chunk.text
    assert "(Description:" in processed_chunk.text
    assert "image_captions" in processed_chunk.metadata
    assert len(processed_chunk.metadata["image_captions"]) == 1
    
    caption = processed_chunk.metadata["image_captions"][0]["caption"]
    print(f"Generated Caption: {caption}")
    assert len(caption) > 10
