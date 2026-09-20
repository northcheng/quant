# tests 使用说明

本目录存放 quant 项目的回归测试（mock 测试）。当前包含 `test_bc_data_io_robustness.py`：针对 `bc_data_io.py` 健壮性的 14 个行为验证用例。

## 运行方式

```powershell
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_robustness.py
```

**必须用 venv 的 python**（系统 python 缺 pandas/akshare 等依赖）。

预期输出：每行均为 `PASS`，末尾 `ALL 14 TESTS PASSED`，退出码 0。任何一行 `FAIL` 即退出码 1，可直接在 PowerShell 里用 `$LASTEXITCODE` 判断。

## 什么时候跑

- 修改了 `bc_data_io.py` 或 `bc_trader.py` 之后，提交/部署前跑一遍
- 环境升级（pandas / akshare 版本变化）后跑一遍

## 测试覆盖

| 组 | 用例数 | 验证内容 |
|---|---|---|
| config 原子写与防覆写 | 6 | `modify/add/remove/create_config` 用临时文件+`os.replace` 原子落盘、无 `.tmp` 残留、损坏 JSON 拒绝覆写 |
| 本地数据文件加载 | 3 | `load_stock_data` 区分两类失败：处理失败（列缺失）保留原文件，文件真坏（不可读）才删除待重下 |
| 邮件报告健壮性 | 1 | `send_result_by_email` 在 portfolio.json 缺平台键、`updated=None` 时不崩溃 |
| ak 下载容错 | 4 | code map 为 None 干净中止；ak 全部抛错回退 today；正常路径返回数据；美股保存路径无 NameError |

## 设计要点

- **不碰真实环境**：所有文件读写发生在 `tempfile.TemporaryDirectory()`，结束自动清理
- **不联网**：外部依赖（`get_data_from_ak`、`get_code_map_from_ak`）用 `unittest.mock.patch.object` 替换
  - `side_effect=RuntimeError(...)` 模拟网络失败
  - `return_value=None` 模拟空结果
  - `return_value=构造的DataFrame` 模拟成功（须经过 `post_process_download_data` 处理，与真实 `get_data_from_ak` 的返回契约一致）
- **不影响真实持仓/组合数据**：`send_result_by_email` 用 `test=True` 模式，config 指向临时目录

## 如何新增用例

仿照现有模式：

```python
with mock.patch.object(io, '依赖函数', return_value=构造数据):
  ret = io.被测函数(...)
  check('组名: 一句话行为描述', 断言)
```

命名建议：`<组名>: <一句话行为描述>`（如 `加载: 处理失败时保留原文件`），描述预期行为而非问题编号。新用例加入后，末尾的 `ALL N TESTS PASSED` 中的 N 会自动更新。
