# 输出模式

当技能需要产出稳定、高质量结果时，可以使用模板和示例。

## 模板模式

严格场景应给出固定模板，例如报告结构：

```markdown
# [分析标题]

## 执行摘要
[关键发现概述]

## 关键发现
- 发现 1
- 发现 2
- 发现 3

## 建议
1. 具体建议
2. 具体建议
```

灵活场景可以给默认结构，并允许 Agent 根据上下文调整。

## 示例模式

如果输出风格很重要，应提供输入/输出示例。例如 commit message：

```markdown
输入：Added user authentication with JWT tokens
输出：
feat(auth): implement JWT-based authentication

Add login endpoint and token validation middleware
```

示例比抽象描述更容易让 Agent 对齐风格、语气和详细程度。

