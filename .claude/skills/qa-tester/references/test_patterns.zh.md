# Dashboard 与 MCP 测试模式参考

执行 Dashboard（A-F）或 MCP（J）测试时参考本文件。

## Dashboard 测试

使用 Streamlit `AppTest` 在 headless 模式下渲染页面。基本流程：

1. 导入页面 `render()`。
2. 使用 `AppTest.from_function()` 创建测试对象。
3. 调用 `at.run()`。
4. 断言 `at.exception` 为空。
5. 打印 header、markdown、metric、selectbox、button 等元素用于验证。

页面模块包括：`overview`、`data_browser`、`ingestion_manager`、`ingestion_traces`、`query_traces`、`evaluation_panel`。

## 常见交互

- 下拉框：`at.selectbox[0].select("test_col")`
- 按钮：`at.button[0].click()`
- 文本输入：`at.text_input[0].input("search term")`
- 交互后都需要重新 `at.run()`

## 文件上传

`AppTest` 不能模拟 `st.file_uploader`，因此使用 CLI 先摄入文件，再通过页面渲染验证文档数量或 chunk 列表。

## MCP 测试

可直接运行已有测试：

```text
pytest tests/e2e/test_mcp_client.py -v
pytest tests/integration/test_mcp_server.py -v
```

也可以通过子进程启动 `src.mcp_server.server`，用 stdin/stdout 发送 JSON-RPC 请求。

## MCP 断言要点

- `initialize` 响应包含 `serverInfo` 和 `capabilities`
- `tools/list` 返回 `query_knowledge_hub`、`list_collections`、`get_document_summary`
- `tools/call` 返回文本 content
- 多模态响应包含 `type: "image"`
- 错误参数应优雅返回 error，不应崩溃
- 连续多次调用应稳定成功

