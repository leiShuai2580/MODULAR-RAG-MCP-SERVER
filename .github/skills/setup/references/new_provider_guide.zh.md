# 新 Provider 脚手架指南

当用户选择尚未实现的 provider 时，按本流程自动生成代码脚手架。本项目使用插件式架构，新增 provider 通常是机械化操作。

## 快速检查：Provider 是否内置

内置 LLM provider：`openai`、`azure`、`deepseek`、`ollama`
内置 Embedding provider：`openai`、`azure`、`ollama`
内置 Vision provider：`openai`、`azure`

如果 provider 不在以上列表中，则继续执行脚手架流程。

## 关键判断：OpenAI 兼容 Provider

许多现代 provider（Qwen/DashScope、Gemini、Groq、Mistral、Together AI 等）提供 OpenAI 兼容 API。对这类 provider，实现通常很简单：继承现有 `OpenAILLM` / `OpenAIEmbedding`，覆盖 `base_url` 和 API Key 解析即可。

## 步骤 1：创建 LLM Provider

创建 `src/libs/llm/{name}_llm.py`。

OpenAI 兼容 provider 推荐：
- 继承 `OpenAILLM`
- 设置 `DEFAULT_BASE_URL`
- 从参数、`settings.llm.api_key` 或环境变量读取 API Key
- 缺少 API Key 时抛出清晰错误

非 OpenAI 兼容 provider：
- 直接继承 `BaseLLM`
- 实现 `chat()`
- 可参考 `deepseek_llm.py` 或 `ollama_llm.py`

## 步骤 2：创建 Embedding Provider

如需要 embedding，创建 `src/libs/embedding/{name}_embedding.py`。

OpenAI 兼容 provider 推荐：
- 继承 `OpenAIEmbedding`
- 设置 `DEFAULT_BASE_URL`
- 从参数、`settings.embedding.api_key` 或环境变量读取 API Key
- 缺少 API Key 时抛出清晰错误

## 步骤 3：创建 Vision LLM Provider

如需要视觉模型，创建 `src/libs/llm/{name}_vision_llm.py`，按与 LLM provider 相同的模式继承 `OpenAIVisionLLM`。

## 步骤 4：注册 Provider

LLM：
- 在 `src/libs/llm/__init__.py` 中导入 provider
- 调用 `LLMFactory.register_provider("{name}", ProviderClass)`

Embedding：
- 在 `src/libs/embedding/embedding_factory.py` 的 `_register_builtin_providers()` 中导入并注册
- 同步更新 `src/libs/embedding/__init__.py` 和 `__all__`

Vision：
- 在 `src/libs/llm/llm_factory.py` 的 `_register_vision_providers()` 中导入并注册

## 步骤 5：确认 Settings 支持 base_url

检查 `src/core/settings.py`。`LLMSettings` 和 `EmbeddingSettings` 已包含 `base_url: Optional[str] = None`，通常无需修改。

## 步骤 6：安装 SDK

多数 OpenAI 兼容 provider 只需要 `openai` 包。

非兼容模式可能需要：
- Qwen：`dashscope`
- Gemini：`google-generativeai`

但优先推荐 OpenAI 兼容模式。

## Provider 参考

| Provider | LLM Base URL | Embedding Base URL | API Key 环境变量 | OpenAI 兼容 |
|---|---|---|---|---|
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | same | `DASHSCOPE_API_KEY` | 是 |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | same | `GEMINI_API_KEY` | 是 |
| Groq | `https://api.groq.com/openai/v1` | N/A | `GROQ_API_KEY` | 是 |
| Mistral | `https://api.mistral.ai/v1` | same | `MISTRAL_API_KEY` | 是 |
| Together | `https://api.together.xyz/v1` | same | `TOGETHER_API_KEY` | 是 |

## 校验

脚手架生成后，验证 provider 是否出现在列表中：

```python
python -c "
from src.libs.llm import LLMFactory
from src.libs.embedding import EmbeddingFactory
print('LLM providers:', LLMFactory.list_providers())
print('Embedding providers:', EmbeddingFactory.list_providers())
print('Vision providers:', LLMFactory.list_vision_providers())
"
```

新 provider 出现在列表中后，再继续 setup 的配置生成步骤。

