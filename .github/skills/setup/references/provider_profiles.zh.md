# Provider 配置参考

本文件是支持 provider 配置的速查表。

## LLM Providers

### OpenAI

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "<OPENAI_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```

必填字段：`provider`、`model`、`api_key`
应移除或留空：`azure_endpoint`、`deployment_name`、`api_version`

### Azure OpenAI

```yaml
llm:
  provider: "azure"
  model: "gpt-4o"
  deployment_name: "<YOUR_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```

以上字段均必填。

### DeepSeek

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "<DEEPSEEK_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```

### Ollama 本地模型

```yaml
llm:
  provider: "ollama"
  model: "llama3"
  base_url: "http://localhost:11434"
  temperature: 0.0
  max_tokens: 4096
```

不需要 API Key。

## Embedding Providers

### OpenAI

```yaml
embedding:
  provider: "openai"
  model: "text-embedding-ada-002"
  dimensions: 1536
  api_key: "<OPENAI_API_KEY>"
```

### Azure OpenAI

```yaml
embedding:
  provider: "azure"
  model: "text-embedding-ada-002"
  dimensions: 1536
  deployment_name: "<YOUR_EMBEDDING_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
```

### Ollama

```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  dimensions: 768
  base_url: "http://localhost:11434"
```

## 可自动脚手架的 Provider

Qwen、Gemini、Groq、Mistral、Together AI 等通常使用 OpenAI 兼容 API。选择时 setup skill 可自动生成 provider 代码。

### Qwen

```yaml
llm:
  provider: "qwen"
  model: "qwen-turbo"
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### Gemini

```yaml
llm:
  provider: "gemini"
  model: "gemini-2.0-flash"
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
```

## 模型维度速查

| 模型 | 维度 |
|---|---:|
| text-embedding-ada-002 | 1536 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| nomic-embed-text (Ollama) | 768 |
| mxbai-embed-large (Ollama) | 1024 |
| text-embedding-v3 (Qwen) | 1024 |
| text-embedding-004 (Gemini) | 768 |

## Vision LLM Providers

Vision 使用独立配置段 `vision_llm`。

推荐模型：
- OpenAI：`gpt-4o`、`gpt-4o-mini`
- Azure：Azure 部署的 `gpt-4o`
- Ollama：`llava`、`llava:13b`
- Qwen：`qwen-vl-max`、`qwen-vl-plus`
- Gemini：`gemini-2.0-flash`、`gemini-1.5-pro`
- DeepSeek：不支持 Vision，需要选择其他 provider

## Rerank Providers

禁用：

```yaml
rerank:
  enabled: false
  provider: "none"
  model: ""
  top_k: 5
```

LLM-based rerank 推荐优先于尚未充分测试的 Cross-Encoder。

