---
name: setup
description: 交互式项目初始化向导。引导用户选择 Provider、配置 API Key、安装依赖、生成配置并启动 Dashboard，必要时自动诊断和修复启动失败。
---

# Setup 中文版

该技能用于从干净代码库完成项目初始化和首次启动。

## 流程

```text
预检查 -> 询问用户 -> 生成配置 -> 安装依赖 -> 校验 -> 启动 -> 使用说明
```

## 预检查

- 检查 Python 版本
- 检查或创建 `.venv`
- 激活虚拟环境
- 检查依赖安装状态

## 配置内容

引导用户选择 OpenAI、Azure、DeepSeek、Ollama、Qwen、Gemini 等 Provider，并写入 `config/settings.yaml`。

## 自动修复

若某一步失败，自动诊断、修复并最多重试 3 轮。

