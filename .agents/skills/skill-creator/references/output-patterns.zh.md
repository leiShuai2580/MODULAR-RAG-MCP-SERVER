# 输出模式

当技能需要稳定地产出高质量结果时，可以使用以下模式。

## 模板模式

为输出格式提供模板。模板严格程度应与任务需求匹配。

**适用于严格要求（例如 API 响应或数据格式）：**

```markdown
## 报告结构

始终使用以下精确模板结构：

# [分析标题]

## 执行摘要
[用一个段落概述关键发现]

## 关键发现
- 发现 1，并附支持数据
- 发现 2，并附支持数据
- 发现 3，并附支持数据

## 建议
1. 具体可执行建议
2. 具体可执行建议
```

**适用于灵活指导（需要根据上下文调整时）：**

```markdown
## 报告结构

下面是一个合理的默认格式，但请结合实际情况判断：

# [分析标题]

## 执行摘要
[概述]

## 关键发现
[根据实际发现调整章节]

## 建议
[结合具体上下文定制]

可根据具体分析类型调整章节。
```

## 示例模式

如果输出质量依赖示例，应提供输入/输出对：

```markdown
## Commit message 格式

按以下示例生成 commit message：

**示例 1：**
输入：Added user authentication with JWT tokens
输出：
feat(auth): implement JWT-based authentication

Add login endpoint and token validation middleware

**示例 2：**
输入：Fixed bug where dates displayed incorrectly in reports
输出：
fix(reports): correct date formatting in timezone conversion

Use UTC timestamps consistently across report generation

遵循该风格：type(scope): 简短描述，然后给出详细说明。
```

相比单纯描述，示例能更清楚地帮助 Agent 理解期望风格和详细程度。

