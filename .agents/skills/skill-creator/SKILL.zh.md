---
name: skill-creator
description: 创建或更新 Agent Skill 的指南，用于把专业知识、工作流、脚本和参考资料封装成可复用能力包。
---

# Skill Creator 中文版

Skill 是一种模块化能力包，用于让通用 Agent 在特定领域拥有专门工作流、工具使用规则和上下文资料。

## Skill 能提供什么

1. 专门工作流：针对某类任务的多步骤过程。
2. 工具集成：处理特定文件格式、API 或命令的说明。
3. 领域知识：项目、公司或业务专属知识。
4. 绑定资源：脚本、参考文档、模板或素材。

## 核心原则

- 保持简洁：只提供 Agent 真正需要的信息。
- 控制自由度：任务越脆弱，步骤越要具体；任务越开放，规则越可概括。
- 优先使用脚本：重复、易错、要求确定性的操作应封装为脚本。
- 按需加载参考资料：大文档放入 `references/`，只在需要时读取。

## 目录结构

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
└── assets/
```

`SKILL.md` 必须包含 YAML frontmatter，其中 `name` 和 `description` 用于触发技能；正文用于说明工作流和注意事项。

