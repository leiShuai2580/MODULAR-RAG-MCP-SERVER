---
name: skill-creator
description: 创建和维护 Agent Skill 的指南，帮助把可复用工作流、专业知识、脚本和参考资料封装为技能包。
---

# Skill Creator 中文版

Skill 是 Agent 的能力扩展包，用于把特定领域的操作流程、工具规则和知识资产交给 Agent 使用。

## 核心原则

- 简洁优先，避免占用不必要上下文。
- 根据任务风险控制指令粒度。
- 高重复、高风险操作优先脚本化。
- 大资料放入 `references/`，按需读取。

## 目录结构

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
└── assets/
```

## 创建流程

1. 明确触发场景。
2. 编写精确的 `description`。
3. 设计最小可用工作流。
4. 添加必要脚本和参考资料。
5. 快速校验技能是否可触发、可执行、可维护。

