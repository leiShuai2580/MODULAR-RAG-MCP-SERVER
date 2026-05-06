# Dashboard 与 MCP 测试模式参考

执行 Dashboard（A-F）或 MCP（J）测试时读取本文件。

## Dashboard 测试：Streamlit AppTest

Streamlit `AppTest` 可以在无浏览器的 headless 模式下渲染页面。每个测试都可以编写并执行一段内联 Python 脚本。

### 基础模板

```python
from streamlit.testing.v1 import AppTest

def page_script():
    from src.observability.dashboard.pages.<PAGE_MODULE> import render
    render()

at = AppTest.from_function(page_script, default_timeout=30)
at.run()
assert not at.exception, f"Exception: {at.exception}"

for attr in ("header", "subheader", "info", "error", "warning", "markdown"):
    for el in getattr(at, attr, []):
        print(f"{attr}: {getattr(el, 'value', '')}")
for m in at.metric:
    print(f"metric: label={m.label}, value={m.value}")
for s in at.selectbox:
    print(f"selectbox: options={s.options}, value={s.value}")
print(f"button count: {len(at.button)}")
print(f"expander count: {len(at.expander)}")
```

页面模块包括：`overview`、`data_browser`、`ingestion_manager`、`ingestion_traces`、`query_traces`、`evaluation_panel`。

### 交互模式

**选择下拉框并重新渲染：**
```python
at.selectbox[0].select("test_col")
at.run()
```

**点击按钮并重新渲染：**
```python
at.button[0].click()
at.run()
```

**文本输入：**
```python
at.text_input[0].input("search term")
at.run()
```

### 文件上传替代方案

`AppTest` 无法模拟 `st.file_uploader`。使用两步法：
1. 通过 CLI 摄入：`python scripts/ingest.py --path <file> [--collection <name>]`
2. 通过 AppTest 验证：渲染页面并检查文档数量、chunk 列表等

### 数据变更（清空全部 / 删除文档）

直接调用 service，然后通过 AppTest 或 CLI 验证：

```python
from src.observability.dashboard.services.data_service import DataService
svc = DataService()
svc.delete_document(source_path, collection, source_hash)  # 或 svc.reset_all()
```

### 参考冒烟测试

```text
pytest tests/e2e/test_dashboard_smoke.py -v
```

## MCP 测试：子进程 JSON-RPC

### 方法 1：已有 pytest 测试

```text
pytest tests/e2e/test_mcp_client.py -v
pytest tests/integration/test_mcp_server.py -v
```

### 方法 2：内联 JSON-RPC 脚本

启动 `src.mcp_server.server` 子进程，通过 stdin/stdout 发送 JSON-RPC 请求并收集响应。核心断言是：响应中包含 `id`，且包含 `result` 或 `error`。

### MCP 断言要点

| 测试 ID | 验证内容 | 关键断言 |
|---|---|---|
| J-01 | Server startup | `initialize` 响应包含 `serverInfo`、`capabilities` |
| J-02 | tools/list | 3 个工具：`query_knowledge_hub`、`list_collections`、`get_document_summary` |
| J-03 | query_knowledge_hub | `content` 数组包含文本块 |
| J-04 | list_collections | 响应包含 collection 名称 |
| J-05 | get_document_summary | 响应包含 title/summary |
| J-06 | Multimodal | 响应包含 `type: "image"` 块 |
| J-07 | Bad collection | 优雅报错，不崩溃 |
| J-08 | Invalid params | JSON-RPC error 或 `isError: true` |
| J-09 | Stability | 连续 5 次调用均成功 |
| J-10 | Citations | 响应包含 `source_file`、`page`、`score` |

