# automatic_trader.py 完整流程文档

> 对象: `c:\Users\northcheng\automatic_trader.py`（778 行，2026-09 版本）
> 所有行号均对应该版本源码。美股相关时间以**夏令时（DST）为基准**，冬令时在原时间上 **+1 小时**；A 股时间固定不变。

---

## 〇、两个核心问题的直接答案

### 1. 什么时候会运行 signal_bridge.py？

**每个交易日的「盘前段」，即 `now ∈ [16:00, 21:30)`（冬令时 `[17:00, 22:30)`）内执行一次**，位置在 [automatic_trader.py L423-435](file:///c:/Users/northcheng/automatic_trader.py#L423-L435)：

- 触发条件：主循环睡到 `pre_open_time + 5min`（16:05）醒来后进入盘前段，在完成 company_300 的 eod 数据更新之后调用；
- 调用命令固定为 `signal_bridge.py --pool company_300 --trade-date <D>`，其中 `D = main_trader.trade_time['open_time'].date()`（**当日美股开盘日**）；
- **幂等设计**：先检查 `result_path/signal_bridge_{D}.xlsx` 是否存在，存在则直接 skip（当日不重算）；不存在才运行，`retry=1, timeout=600s`；
- 若桥因数据不齐报错退出，主循环**不会中断**：当日 03:45 交易时回退读 legacy 信号文件（见下）；若盘前段内抛异常被外层 `except` 捕获（L770-772），循环 `continue` 后仍在盘前时间窗内，会重入盘前段再次尝试。

### 2. 什么时候会依据信号进行交易？

**每个交易日的「收盘前 15 分钟」，即 `now ∈ [03:45, 04:00)`（冬令时 `[04:45, 05:00)`）执行**，位置在 [automatic_trader.py L544-573](file:///c:/Users/northcheng/automatic_trader.py#L544-L573)：

- 前置动作：先跑一遍 universe 池的 realtime 更新+计算（L529-538，10 分钟超时，用于刷新最新价格）；
- 信号文件选择：**优先**读盘前生成的 `signal_bridge_{D}.xlsx`（按 `open_time` 口径定位 D 日，因为 03:45 时 `now.date()` 已是 D+1）；bridge 不存在则**回退**读 legacy 文件 `result_path/{now.date()}.xlsx`（technical_analyst 实时产出，now 口径）；
- 读 sheet `signal` → 列 `signal` 重命名为 `action`（值 `b`/`s`），按 `symbol` 建索引，合并 `latest_price`；
- 订单类型：`now < close_time`（04:00）→ `market`；已达收盘 → `limit`；
- 最终交易入口：**遍历全部 traders**（默认 tiger simu + tiger real + futu real），逐账户调用

```
tmp_trader.signal_trade(signal, money_per_sec=init_cash[plfm][acnt],
                        pool=selected_sec_list[trade.pool[plfm][acnt]], order_type)
```

- 同一段内还有**条件交易**入口（L581-613）：读 `trader_path/trade_condition.json`，过滤出 `date == D` 且非 TEMPLATE 的条目，按 plfm/acnt 匹配后调用 `condition_trade`。两者相互独立、顺序执行。

一句话总结：**信号在盘前 16:05~21:30 之间算好落盘（桥），订单在收盘前 15 分钟（03:45~04:00）统一发出**；此外睡眠期间还持续运行与信号无关的止盈止损。

---

## 一、时间口径与「注释 vs 实现」差异

`main_trader.trade_time` 字典由 `bc_trader` 的市场状态方法维护（初始化时 L127 调用 `update_market_status`，每日收盘后段 L689 调用 `update_trade_time` 刷新到下一交易日），关键键值（DST）：

| 键 | 含义 | 夏令时 | 冬令时 |
|---|---|---|---|
| `pre_open_time` | 美股盘前交易开始 | 16:00 | 17:00 |
| `open_time` | 美股开盘（信号日 D 的锚点） | 21:30 | 22:30 |
| `close_time` | 美股收盘 | 04:00 (D+1) | 05:00 (D+1) |
| `post_close_time` | 美股盘后交易结束 | 08:00 (D+1) | 09:00 (D+1) |
| `a_open_time` / `a_close_time` | A 股开/收盘 | 09:30 / 15:00 | 不变 |

**注意：文件头部 L1-36 的注释是旧版设计稿，与实际代码有出入，以代码为准**：

| 项目 | 头部注释 | 代码实际值 |
|---|---|---|
| 盘中睡眠分界「午夜」 | 23:15 | **23:30**（L478：`open + 2h + (30-分钟数)`，恒落在开盘后 2 小时的 :30） |
| 收盘前交易触发 | 03:50 | **03:45**（L479：`close - 15min`） |
| A 股近收盘分界 | 14:30 / 14:15 | **14:40**（L273：`a_close - 20min`） |
| A 股午休 | 12:00~13:00 | **11:30~13:00**（L274-275：`a_open+120/210min`，即真实 A 股午休） |
| `check_frequency = 3600` 注释 | "= 30min" | 实为 60 分钟 |

---

## 二、初始化段（L38-136）

1. **路径**（L48-51）：`home_path = Path.home()`，`git_path = home/git` 并加入 `sys.path`（因此 `from quant import ...` 可用）。
2. **导入**（L64-68）：`bc_util`（日志/脚本运行）、`bc_data_io`（邮件/行情简报）、`bc_trader`（交易器）、`bc_technical_analysis`（配置加载）。
3. **配置**（L74）：`ta_util.load_config(root_paths)` → `config`，含 `result_path`、`home_path`、`log_path`、`trader_path`、`config_path`、`api_key`、`trade`（init_cash / pool / 止盈止损阈值）、`selected_sec_list` 等。
4. **日志**（L81）：统一走 `util.setup_logging`，前缀 `automatic_trade_log`，库层日志同样落盘。
5. **命令行参数**（L88-108）：
   - `--source`：TA 脚本数据源，默认 `eod`；
   - `--tiger`：`<none>/<simu>/<real>/<both>`，默认 `both`；**必须非 none**（否则报错退出，L98-102，因为主交易器必须是 tiger）；
   - `--futu`：默认 `real`。
6. **交易器创建**（L115-124）：按 `trader_info` 展开平台×账户组，`eval` 构造 Trader 实例存入 `traders` 字典（键如 `tiger_simu`/`tiger_real`/`futu_real`）；**`main_trader` = 第一个 tiger 交易器**（默认即 `tiger_simu`）。`main_trader` 只负责市场状态/时间/睡眠调度，**不负责下单**；下单是遍历整个 `traders` 字典。
7. **市场状态初始化**（L127）：`update_market_status(return_str=True)` 填充 `trade_time`，打印美股/A 股时间串。

---

## 三、主循环骨架（L143-772）

```
while main_trader is not None:
  try:
    ① 若设了 target_time: 打印横幅 → 内层睡眠循环 → 醒后仓位检查
    ② counter+1 / now / update_market_status()
    ③ 按 now 落入五段时间窗之一, 执行该段动作并设置下一个 target_time
  except: 记日志, continue          # L770-772, 循环永不因异常退出
```

- 唯一正常退出路径：时间状态无法归类（L766-768）→ `break` → 打印 `[stop]`（L774）。
- 每段结尾都是「设置 `target_time`/`target_dscr`/`check_frequency` → `continue`」，回到循环顶进入睡眠。

### 睡眠循环（L169-220）——第一层止盈止损

睡到 target_time 之前，**每 `check_frequency` 秒醒来一轮**（常规 3600s，临近关键时点 300s）：

1. 遍历全部 traders：`update_position(get_briefs=True)` → 打印持仓；
2. 读 `config['trade']` 四档阈值（`stop_loss` / `stop_profit` / `stop_loss_inday` / `stop_profit_inday`，按 plfm×acnt）；
3. `cash_out(...)` 执行止盈止损（**tiger 模拟盘在此循环内跳过**，L201-202）；
4. `update_portfolio_record(...)` 落盘持仓记录；
5. 计算剩余睡眠时长，`time.sleep(min(剩余, check_frequency))`。

### 醒后仓位检查（L224-261）——第二层止盈止损

到达 target_time 后、进入下一段时间窗之前，再对所有 traders 执行一遍 `update_position → cash_out → update_portfolio_record`（此处 tiger 模拟盘**不**跳过——L231-233 的 skip 代码已被注释）。

> 所以止盈止损与信号交易完全独立：信号每天只在 03:45 窗口交易一次，而止盈止损在**整个睡眠周期内持续巡检**（约 5~60 分钟一轮）。

---

## 四、五段时间窗详解

### 段 1：未开盘 `now < pre_open_time`（L268-389）

窗口覆盖前一天 08:05 之后 ~ 当日 16:00，**A 股逻辑全部嵌在这一段里**。先计算三个 A 股时点（L272-275）：午休开始 11:30、午休结束 13:00、近收盘 14:40。

| 子分支 | 时间（A 股口径） | 动作 | 睡至 |
|---|---|---|---|
| 1.1 | < 09:30 | 发持仓邮件（L283-288） | 11:30（若当日 A 股先行开盘；正常走此分支）或 16:05 |
| 1.2a | 09:30 ~ 11:30 | 无 | 11:30 |
| 1.2b | 11:30 ~ 14:40 | 跑 `technical_analyst_parallel --pool hs300 --update_mode both --source eod`（L322，timeout 120min，retry 1） | 14:40，freq 300 |
| 1.2c | 14:40 ~ 15:00 | 跑 `--pool a --required_date <A开盘日> --update_mode both --source eod`（L342，timeout 25min，retry 5）→ 发 A 股信号邮件（pool='a'） | 15:05，freq 300 |
| 1.3 | ≥ 15:00 | 无 | 当日美股开盘 → 16:05；否则次日 00:05（L372：`a_close+545min`） |

### 段 2：盘前 `[16:00, 21:30)`（L391-465）★ signal_bridge 调用段

进入段内的**固定执行顺序**（16:05 醒来后逐个跑，全部阻塞式 `util.run_script`）：

1. **company_300 纯数据更新**（L399-407，仅当 `company_star_update_time < today`）：`--pool company_300 --update_mode eod --skip_calculation --skip_visualization --skip_postprocess`，timeout 3h，retry 5；
2. **company_300 完整更新**（L409-418，同条件）：`--update_mode eod --pool company_300`（含计算+可视化+company_star 股票池更新），timeout 3h，retry 5；
3. **★ signal_bridge**（L423-435）：见「〇、答案 1」。`--pool company_300 --trade-date <D>`，retry 1，timeout 10min，产出 `result_path/signal_bridge_{D}.xlsx`（sheet `signal`：代码/symbol/action=b|s 名单；桥内部自带 company_300 组合预设 `C_tmqmom:1 + F_er20:1, top_k 12`，调用方参数不变）；
4. **us 池数据更新**（L437-446）：`--pool us --update_mode eod --skip_*`，timeout 30min，retry 5；
5. **hs300 更新**（L448-458，仅当 `a_company_update_time < today`）：`--pool hs300 --update_mode eod`，timeout 3h，retry 5；
6. 睡至 **21:35**（开盘后 5 分钟），freq 3600。

### 段 3：盘中 `[21:30, 04:00)`（L467-620）★ 交易段

先计算两个内部时点（L475-479）：`midnight = 23:30`、`before_close = 03:45`。

| 子分支 | 时间 | 动作 | 睡至 |
|---|---|---|---|
| 3.1 | 21:30 ~ 23:30 | 无（睡眠，期间止盈止损巡检 freq 300） | 23:30 |
| 3.2 | 23:30 ~ 03:45 | `--pool us --update_mode realtime`（timeout 30min，retry 5）→ 发邮件（signal_file_date=now 日期） | 03:45，freq 300 |
| 3.3 | 03:45 ~ 04:00 | 见下方「收盘前 15 分钟完整动作」 | 06:00，freq 3600 |

**3.3 收盘前 15 分钟的完整动作**（本程序唯一的自动下单窗口）：

1. `--pool universe --update_mode realtime --required_date <D> --skip_visualization`（L531，timeout 10min，retry 5）——刷新全 universe 最新价；
2. **信号交易**（L544-579）：
   - 定位信号文件：`bridge = result_path/signal_bridge_{D}.xlsx`（D 按 `open_time` 口径）；`legacy = result_path/{now.date()}.xlsx`；`bridge 存在 ? bridge : legacy`；
   - `pd.read_excel(file, sheet_name='signal', dtype={'代码': str})` → `signal` 列改名 `action` → 以 `symbol` 为索引 → `get_stock_briefs` 合并 `latest_price` → 打印信号表；
   - `order_type`：now < 04:00 → `market`；≥ 04:00 → `limit`；
   - 遍历 `traders`，逐账户 `signal_trade(signal, money_per_sec=init_cash[plfm][acnt], pool=selected_sec_list[trade.pool[plfm][acnt]], order_type)`；
   - 文件不存在 → 日志 `[erro]`；信号为空 → `[skip]`（均不中断循环）；
3. **条件交易**（L581-613）：`pd.read_json(trader_path/trade_condition.json).T` → 过滤 `index != 'TEMPLATE'` 且 `date == D` → 合并最新价 → 遍历 traders，按 plfm/acnt 过滤出属于自己的条目 → `condition_trade(tmp_signal)`。无有效条目或文件不存在仅记日志；
4. 睡至 **06:00**（收盘后 2 小时），freq 3600。

### 段 4：盘后 `[04:00, 08:00)`（L622-654）

- `--pool us --update_mode eod --required_date <D>`（timeout 30min，retry 5）：把当日美股价正式落成 eod 并计算+可视化；
- 发邮件（`log_file_date = D`、`signal_file_date = D+1`）；
- 睡至 **08:05**，freq 3600。

### 段 5：收盘后 `≥ 08:00`（L656-762）

1. **git 同步**（L671-683）：在 `config_path` 仓库 `pull` → `add selected_sec_list.json + portfolio.json` → commit（"update portfolio and selected_sec_list"）→ `push`；失败仅记日志；
2. **切换到新交易日**（L685-702）：`update_trade_time()` → `update_market_status()` → `rotate_log_file`（日志文件轮转到下一交易日）→ 重置 `counter/target_time/check_frequency`；
3. **周六 refresh**（L704-760）：`today.weekday() == 5` 时以 `--update_mode refresh` 全量重刷五个池：**company_300 → etf_3x → global → a_etf → hs300**（每个 timeout 3h，retry 5）；
4. `continue` → 循环顶 `target_time = None`，直接跳过睡眠段 → `now` 落入新交易日的「未开盘段」，回到段 1，周而复始。

---

## 五、信号 → 订单 全链路汇总

```
[D-1 及更早的 eod 数据]
        │  盘前 16:05~21:30（段 2 第 3 步）
        ▼
signal_bridge.py --pool company_300 --trade-date <D>
        │  幂等: signal_bridge_{D}.xlsx 已存在则跳过
        │  产出: result_path/signal_bridge_{D}.xlsx (sheet 'signal': b/s 名单)
        ▼
[睡眠 · 周期性止盈止损 cash_out]  ←—— 与信号无关, 独立运行
        │  03:45 (段 3.3)
        ▼
universe realtime 刷新最新价
        ▼
读信号文件: bridge(D) 优先 ──不存在──▶ legacy {now.date()}.xlsx
        ▼
signal_trade()  遍历 tiger_simu / tiger_real / futu_real
        │  money_per_sec = init_cash[plfm][acnt]
        │  pool = selected_sec_list[trade.pool[plfm][acnt]]
        │  order_type = market (<04:00) / limit (≥04:00)
        ▼
[实盘订单]   (同窗口并行: condition_trade ← trade_condition.json)
```

---

## 六、每日动作时间表（夏令时口径）

| 时刻 | 动作 | 代码位置 |
|---|---|---|
| ~09:30 前 | 持仓邮件 → 睡至 11:30 | L280-301 |
| 11:30~14:40 | hs300 池更新+计算 | L319-336 |
| 14:40~15:00 | a 池实时计算 + 邮件 → 睡至 15:05 | L339-365 |
| 16:05 | 进入盘前段 | L395 |
| 16:05~21:30 | c300 eod 更新(×2) → **signal_bridge** → us eod → hs300 | L399-458 |
| 21:35~23:30 | 盘中睡眠（止盈止损巡检 freq 300） | L483-490 |
| 23:30~03:45 | us realtime + 邮件 | L495-523 |
| 03:45~04:00 | universe realtime → **signal_trade + condition_trade** → 睡至 06:00 | L528-620 |
| 06:00~08:00 | us eod 正式化 + 邮件 | L626-654 |
| 08:05 | git 同步、换交易日、日志轮转（周六加五池 refresh） | L656-762 |

（冬令时美股各时点 +1h；A 股时点不变。）

---

## 七、注意事项与已知特性

1. **头注释陈旧**：见「一」的差异表，读代码时不要被 L1-36 误导。
2. **桥失败不阻断**：signal_bridge 返回非 0 只记日志；当天信号交易自动回退 legacy 文件（technical_analyst 的 universe 实时口径，即旧四信号体系）。想确认当天用的是哪套信号，看日志 `[signal]: using signal bridge file ...` 有没有出现。
3. **桥幂等的副作用**：盘前段若中途重启程序，只要 `signal_bridge_{D}.xlsx` 已存在就不再重算——手工删除该文件可强制重跑。
4. **信号只交易一次**：03:45 窗口之外，程序不会因信号文件变化再下单（止盈止损除外）。
5. **tiger 模拟盘**：睡眠巡检中跳过 `cash_out`（L201-202），醒后检查段不跳过（L231-233 已注释）；信号交易则所有账户都执行。
6. **`--source` 参数**：默认 eod，会透传给盘前/盘中的 TA 脚本与 `get_stock_briefs` 的价格源。
7. **周六 refresh** 顺带完成 etf_3x/global/a_etf 池的全量重算（周一开盘前盘前段的 company_300 也会因 `company_star_update_time` 条件重跑）。
8. **异常自愈**：主循环任何段抛异常都 `continue`，最坏情况是当晚跳过该段动作后继续按时间窗运转；只有「时间无法归类」才会终止程序。
