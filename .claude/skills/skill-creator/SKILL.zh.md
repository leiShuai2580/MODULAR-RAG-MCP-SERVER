---
name: skill-creator
description: 创建或更新 Agent Skill 的指南，用于把专业知识、工作流、脚本和参考资料封装成可复用能力包。
---

# Skill Creator 中文版

Skill 是一种模块化能力包，用于让通用 Agent 在特定领域拥有专门工作流、工具使用规则和上下文资料。

## Skill 能提供什么

1. 专门工作流
2. 工具集成
3. 领域知识
4. 绑定资源

## 核心原则

- 保持简洁，只提供 Agent 真正需要的信息。
- 根据任务脆弱程度控制自由度。
- 重复、易错、要求确定性的操作优先封装为脚本。
- 大文档放入 `references/`，按需读取。

## 目录结构

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
└── assets/
```

`SKILL.md` 必须包含 `name` 和 `description`，用于技能触发；正文说明具体工作流和注意事项。

