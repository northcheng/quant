# tests 使用说明

本目录存放 quant 项目的回归测试与自检工具。**先读「一键自检」，再按需看下面的细则。**

---

## 一键自检（推荐入口）

2026-09 有一批修复（见 `quant/CODE_REVIEW_2026-09.md` 附录 B），改动涉及 8 个文件。
自检脚本会依次做：环境体检 → 32 个模块的导入冒烟 → 自动跑下面三个测试文件。

```powershell
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tools\preflight.py
```

只做环境和导入冒烟（不跑测试，最快）：

```powershell
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tools\preflight.py --skip-tests
```

输出里的状态含义：

| 状态 | 含义 | 要做的事 |
|---|---|---|
| `FAIL` | 需要排查。若异常是 `AttributeError` / `NameError` / `SyntaxError` / `IndentationError`，**几乎一定是代码问题** | 把那一行原样发出来 |
| `WARN` | 第三方依赖缺失导致跑不了（例如 `No module named 'akshare'`） | 按提示 `pip install` |
| `SKIP` | 可选重依赖（lightgbm / torch / futu / tigeropen）缺失，该模块本次无法冒烟 | 需要该能力时再装 |

> 脚本本身的定位是"分流"：把「我改坏了代码」和「环境没装包」分开。这两件事在 `.venv` 缺依赖时长得一模一样。

---

## 测试文件

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `test_bc_data_io_robustness.py` | 14 | `bc_data_io` 健壮性（config 原子写 / 本地加载 / 邮件 / ak 容错） |
| `test_bc_data_io_incremental.py` | 30（含可选 `trade_ok` 组共 34） | 增量拼接一致性 / 分红拆股真值保护 / CSV 原子写 / 下载失败判定 / 安全判空 / 下单成功判定 / 交易所后缀 / **窗口单一真源** |
| `test_bc_data_io_eod_new_data.py` | 87 | `bc_data_io` 的 `#-- EOD New Data --#` 区段（9 个私有助手 + 44 个 `get_xxx_from_eod` 新方法）：envelope 展平 / 分页 / filter 键与路径 / intraday unix 秒 / 错误路径 |

单独运行：

```powershell
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_robustness.py
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_incremental.py
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_eod_new_data.py
```

预期输出：每行 `PASS`（`trade_ok` 组在缺 `futu`/`tigeropen` 时会打印一行 `[SKIP]`），
末尾 `ALL N TESTS PASSED`，退出码 0。任何一行 `FAIL` 即退出码 1，可用 `$LASTEXITCODE` 判断。

> ⚠️ **robustness / incremental 两个测试都还没有在任何环境里运行过**（编写时所在沙箱禁止执行 python）。
> 若断言失败，**有可能是断言本身写错了**而非代码错了 —— 把输出贴出来即可。
> （`test_bc_data_io_eod_new_data.py` 已在 venv 中跑过全绿：`ALL 87 TESTS PASSED`。）

> ⚠️ **`"必须用 venv 的 python"` 这条约定当前不成立**：`.venv\Lib\site-packages` 下
> 缺 `akshare`（也缺 `pytest`/`futu`/`tigeropen`）。跑测试前需要先装依赖：
> `pip install -r quant/requirements.txt`。装不上就先跑 `preflight.py --skip-tests` 看导入是否干净。

---

## 什么时候跑

- 修改了 `bc_data_io.py` / `bc_trader.py` / `factor/*` / `ml/*` 之后，提交/部署前
- 环境升级（pandas / akshare 版本变化）后
- **重跑 pkl 之前**：复权索引修复、6 处指标修复、窗口口径统一都会改变数值，
  跑一遍能先确认代码没坏，再去看数值变化
- 改 `.gitignore` / 建仓库之前（确认没把测试产物带进去）

## 设计要点

- **不碰真实环境**：所有文件读写发生在 `tempfile.TemporaryDirectory()`，结束自动清理
- **不联网**：外部依赖用 `unittest.mock.patch.object` 替换（`side_effect=RuntimeError(...)` 模拟网络失败、`return_value=None` 模拟空结果、构造 DataFrame 模拟成功）
- **不影响真实持仓/组合数据**：`send_result_by_email` 用 `test=True`，config 指向临时目录
- **零第三方测试框架**：脚本式（`check(name, cond)` + 末尾哨兵 + `sys.exit(1)`），双击即可跑。
  刻意不做 pytest 化 —— 在跑不了 pytest 的环境里改造是净负收益（会在你能验证之前先破坏可运行性）

## 如何新增用例

仿照现有模式：

```python
with mock.patch.object(io, '依赖函数', return_value=构造数据):
  ret = io.被测函数(...)
  check('组名: 一句话行为描述', 断言)
```

命名建议：`<组名>: <一句话行为描述>`（如 `加载: 处理失败时保留原文件`），
描述**预期行为**而非问题编号。新用例加入后，末尾的 `ALL N TESTS PASSED` 中的 N 会自动更新。

### 优先补哪几类（按"出错代价"排）

1. **增量拼接/复权**：旧 CSV 3 行 + mock 新窗口含 2 行重叠且价格不同 → 断言告警或全量重下（已有基础覆盖，可继续加边界）
2. **下单安全**：资金不足必须拒绝、重复调用不得重复下单、持仓未知时不得卖出
3. **`factor.backtest` 引擎**：`verify_backtest` 的 oracle/shuffle/投毒不变量已经是现成的强断言，包成用例即可
4. **`_sanitize_prices`**：构造 k≥3 的常数垃圾段，断言被整段剔除（当前掩码方向与注释相反，段内部会漏网）
