---
name: auto-coder
description: 自动化规格驱动开发 Agent。同步 DEV_SPEC.md，查找任务，按架构实现代码，运行测试并自动修复，最后持久化进度。
---

# Auto Coder 中文版

该技能用于从规格文档自动推进开发任务。

## 流程

```text
同步规格 -> 查找任务 -> 实现 -> 测试（最多 3 轮修复）-> 持久化
```

## 参考资料

`.github/skills/auto-coder/references/` 下包含：
- `01-overview.md`：项目概览
- `02-features.md`：功能规格
- `03-tech-stack.md`：技术栈
- `04-testing.md`：测试约定
- `05-architecture.md`：架构设计
- `06-schedule.md`：排期状态
- `07-future.md`：后续规划

## 执行要点

优先读取排期，选择进行中或未开始任务；实现前抽取验收标准和文件范围；实现后运行测试；失败时最多自动修复 3 轮；完成后更新进度。

