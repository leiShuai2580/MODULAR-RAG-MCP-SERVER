---
name: qa-tester
description: 全自动 QA 测试 Agent。串行执行 QA 测试计划，验证 CLI、Dashboard、MCP、Provider 切换和数据生命周期，记录进度并自动修复失败项。
---

# QA Tester 中文版

该技能用于按 QA 测试计划自动执行项目测试。

## 核心规则

1. 严格串行，一次只执行一个测试。
2. 通过必须来自本会话真实终端输出。
3. 禁止交叉引用其他测试结果。
4. 禁止推断式通过。
5. 每个章节结束后运行校验脚本。

## 流程

```text
选择待测项 -> 准备状态 -> 运行命令 -> 对照预期 -> 修复 -> 记录进度 -> 下一个
```

## 测试类型

- Dashboard UI
- CLI
- MCP JSON-RPC
- Provider 切换
- 故障容错
- 数据生命周期

