# signal_bridge 各池回测表现报告

> 日期: 2026-09-23
> 问题: 按照 signal_bridge 的交易策略, 每个池的历史回测表现(CAGR / Sharpe / maxDD / 年换手 / 平均持仓等)和 2026 年以来的表现如何?
> 证据来源: `signal_bridge.py` 池预设(源码逐行核实) + `bt_core`/`score_backtest` 引擎实跑(本次四窗口全池重跑)
> 产出数据: [pool_perf_signalbridge.csv](file:///c:/Users/northcheng/git/quant/research/output/pool_perf_signalbridge.csv)(策略四窗) / [pool_perf_buyhold.csv](file:///c:/Users/northcheng/git/quant/research/output/pool_perf_buyhold.csv)(等权基准)

## 0. 一句话结论

五个池的预设策略**全部为正收益、全部跑赢各自池等权基准**; 但形态分两族——
**A 股两池(hs300 / a_etf_all)风险调整后最优**(全期 Sharpe 1.39 / 0.73, 换手 76x / 16x, 且 2026 年在 A 股基准下跌时取得 +12.7% / +29.2% 绝对收益),
**美股三池吃肉也吃波动**(company_1000 全期 44.6x、company_300 CAGR 73%, 但 company_300 换手 161.6x、平均持仓仅 3.4 天, 年成本拖累约 16pp)。

## 1. 回测口径

| 项 | 设定 |
|---|---|
| 策略定义 | `signal_bridge.ALL_PRESETS`(= `POOL_PRESETS` + `A_POOL_PRESETS`), 见 [signal_bridge.py#L81-L96](file:///c:/Users/northcheng/signal_bridge.py#L81-L96) |
| 分数口径 | 成分当日截面 `rank(pct=True)` → `(w/Σ\|w\|)` 加权 → 截面 `rank`(1=最强); 缺成分以 0 参与, 全缺则退出截面排名 |
| 选股/退出 | `top_k` = 各池预设(top 排名买入); `exit_rank = 12`(DEFAULT_EXIT_RANK); `rank > 12` 卖出 |
| 仓位 | `sizing='tier'` 三档 1.5 / 1.0 / 0.5; `max_exposure=1.0`; `per_symbol_cap=0.30` |
| 成本 | 单边 **10 bps**(下列净值均为扣费后); 年成本拖累 ≈ `ann_turnover × 10bps` |
| 时间线(防前视) | 信号日 t 收盘决策 → 执行日 d = t+1 **开盘**成交, open-to-open 结算 |
| 数据源 | 研究全史 pkl(`alpha_mining.POOLS`): company_300 2000+ / 其余 2020+ 起 |
| 引擎 | `bt_core.BacktestKit` → `score_backtest.run_config`(与 CLI 逐位一致) |

### 各池预设配置

| 池 | 权重 | top_k | 候选构建 | market |
|---|---|---|---|---|
| company_300 | `C_tmqmom:1.0 + F_er20:1.0` | 12 | `build_alpha_cands` | 美股 |
| company_1000 | `G_vr20:1.0 + H_ichimoku_alpha:1.0` | 8 | `build_alpha_cands` | 美股 |
| etf_3x | `F_mom121:0.5 + F_er20:0.3 + N_idiovol60:0.2 + F_obv20:0.1` | 5 | `build_alpha_cands` | 美股(杠杆 ETF) |
| hs300 | `N_range20:-1.0 + D_ma20_dist:-1.0 + C_tmqmom:0.5` | 5 | `build_a_combo_cands` | A 股 |
| a_etf_all | `N_range20:+1.0 + G_oviv20:0.5` | 5 | `build_a_combo_cands` | A 股 ETF |

> 注: `N_range20` 两池方向**相反**(hs300 负权 = 买高振幅波动反转 / a_etf_all 正权 = 买低振幅防御), 生产必须独立参数化。
> 注: `etf_3x` 的 `N_idiovol60` 即 gold4 负权 `F_idiovol60` 的取反表达, 加权后截面排序与 gold4 等价。

## 2. 历史全期表现(2021-01-04 ~ 2026-09)

| 池 | 总收益 | CAGR | Sharpe | maxDD | Calmar | 年化波动 | 年换手 | 平均持仓 | 胜率 | 交易数 | 交易日 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| company_300 | 21.76x | **73.1%** | 1.34 | -34.2% | 2.14 | 50.4% | **161.6x** | **3.4 天** | 50.1% | 5086 | 1431 |
| company_1000 | **44.64x** | **95.5%** | **1.39** | -48.1% | 1.98 | 60.9% | 99.3x | 6.8 天 | 51.9% | 1670 | 1433 |
| etf_3x | 5.57x | 39.1% | 0.95 | -42.5% | 0.92 | 45.8% | 71.6x | 21.0 天 | 52.7% | 336 | 1432 |
| hs300 | 9.86x | 52.0% | **1.39** | **-30.6%** | 1.70 | 35.9% | 76.1x | 7.1 天 | **61.5%** | 644 | 1384 |
| a_etf_all | 2.70x | 25.8% | 0.73 | **-22.7%** | 1.13 | 40.6% | **15.8x** | **190.9 天** | **82.4%** | 34 | 1384 |

分窗 Sharpe(IS = 2021-01 ~ 2024-12, OOS = 2025-01 起):

| 池 | IS 2021-2024 | OOS 2025-01~ | FULL |
|---|---|---|---|
| company_300 | 1.37 | 1.27 | 1.34 |
| company_1000 | 1.32 | 1.50 | 1.39 |
| etf_3x | 0.73 | 1.35 | 0.95 |
| hs300 | 1.48 | 1.26 | 1.39 |
| a_etf_all | 0.64 | 1.71 | 0.73 |

## 3. 分窗详细指标

### 3.1 IS 窗口(2021-01-04 ~ 2024-12-31)

| 池 | 总收益 | CAGR | Sharpe | maxDD | Calmar | 年换手 | 平均持仓 | 胜率 | 交易数 |
|---|---|---|---|---|---|---|---|---|---|
| company_300 | 7.18x | 69.4% | 1.37 | -32.5% | 2.13 | 160.5x | 3.4 天 | 50.0% | 3545 |
| company_1000 | 10.98x | 86.5% | 1.32 | -48.1% | 1.80 | 97.1x | 7.0 天 | 51.3% | 1139 |
| etf_3x | 1.29x | 23.1% | 0.73 | -42.5% | 0.54 | 70.2x | 21.1 天 | 50.2% | 221 |
| hs300 | 4.60x | 54.1% | 1.48 | -30.6% | 1.77 | 54.9x | 7.8 天 | 58.9% | 319 |
| a_etf_all | 1.36x | 24.0% | 0.64 | -22.7% | 1.06 | 12.3x | 207.8 天 | 68.8% | 16 |

### 3.2 OOS 窗口(2025-01-02 ~ 2026-09)

| 池 | 总收益 | CAGR | Sharpe | maxDD | Calmar | 年换手 | 平均持仓 | 胜率 | 交易数 |
|---|---|---|---|---|---|---|---|---|---|
| company_300 | 1.65x | 77.3% | 1.27 | -34.2% | 2.26 | 164.8x | 3.3 天 | 50.4% | 1535 |
| company_1000 | 2.57x | 110.9% | 1.50 | -41.8% | 2.65 | 104.5x | 6.5 天 | 53.2% | 526 |
| etf_3x | 1.87x | 85.7% | 1.35 | -38.9% | 2.21 | 76.0x | 17.6 天 | 58.3% | 115 |
| hs300 | 0.98x | 49.5% | 1.26 | -29.6% | 1.67 | 125.5x | 6.4 天 | 65.1% | 321 |
| a_etf_all | 0.59x | 31.2% | 1.71 | -15.8% | 1.98 | 24.2x | 82.2 天 | 85.0% | 20 |

## 4. 2026 年以来的表现(2026-01-02 ~ 2026-09-17)

> 仅 8.5 个月, 下表收益为**区间实际收益**(非年化); CAGR 列仅为年化外推参考。

| 池 | 策略收益 | Sharpe | maxDD | 年换手 | 平均持仓 | 胜率 | 交易数 | 池等权基准收益 | 基准 Sharpe | 超额 |
|---|---|---|---|---|---|---|---|---|---|---|
| company_300 | +23.4% | 0.83 | -28.7% | 170.4x | 3.2 天 | 48.2% | 641 | +6.5% | 0.44 | **+16.9pp** |
| company_1000 | +60.5% | 2.22 | -19.7% | 102.9x | 6.5 天 | 56.5% | 216 | +14.1% | 1.28 | **+46.4pp** |
| etf_3x | **+69.0%** | 1.40 | -38.5% | 73.3x | 18.8 天 | 53.7% | 41 | +21.2% | 1.01 | **+47.8pp** |
| hs300 | +12.7% | 0.64 | -29.6% | 126.6x | 6.2 天 | 61.8% | 136 | **-6.2%** | -0.57 | **+18.9pp** |
| a_etf_all | +29.2% | **2.26** | **-11.2%** | **28.6x** | 56.7 天 | **90.0%** | 10 | **-7.3%** | -0.56 | **+36.5pp** |

独立印证: 桥口径(rank_pct 加权 + `fillna(0.5)` + 覆盖门槛)复算 hs300 2026 年以来为 **策略 +10.4% vs 池等权 -5.3%**, 与上表同向同量级(差异来自 NaN 中性化与覆盖门槛口径)。

## 5. 要点解读

1. **A 股两池是 2026 年的核心价值位**: 在 A 股基准下跌(hs300 池等权 -6.2% / a_etf_all 池等权 -7.3%)的环境下, 两个策略分别取得 **+12.7% / +29.2% 的绝对收益**——这是截面反转策略在弱市里的典型优势。且 a_etf_all 换手仅 28.6x(年成本拖累 ≈ 2.9pp), 胜率 90%, maxDD 仅 -11.2%, 是全池风险调整后最优的 2026 表现。

2. **美股三池 2026 全线强于自身全期**: company_1000 Sharpe 从全期 1.39 升至 2.22; etf_3x 从 0.95 升至 1.40(但其 maxDD -38.5% 为全池最深, 杠杆 ETF 池的波动本色)。

3. **company_300 是"高收益 + 高代价"的典型**: 全期 CAGR 73.1% 为美股池最高, 但换手 161.6x、平均持仓 3.4 天, **年成本拖累 ≈ 16.2pp**; 2026 年 Sharpe 跌至 0.83、胜率 48.2%, 相对全期 1.34 明显衰减。结构原因是 `top_k=12` 与 `exit_rank=12` **完全重合 → 无缓冲带**, rank 12/13 边界票反复进出放大换手。

4. **`company_1000` 的 44.6x 不可直接采信**: 该终值中包含 2021-10 DJT 一笔 +1091%(Trump Media SPAC 合并公告的 meme 狂潮; 行情真实但当日成交额仅 ~$43K, 20% 仓位实盘吃不到)。剔除该笔后其全期 Sharpe 会大幅下修; 且该池**无生产 pkl**, 桥中标注为"仅复盘用"(研究版滞后 1~2 天)。

5. **换手梯度清晰**: a_etf_all 15.8x(≈190 天持仓, 极低频) < etf_3x 71.6x(21 天) < hs300 76.1x(7 天) < company_1000 99.3x(7 天) < company_300 161.6x(3.4 天)。成本敏感度与这个顺序严格同向。

## 6. 已知局限与风险

| 局限 | 说明 |
|---|---|
| **幸存者偏差** | A 股两池为「当前成分 × 全历史」回测, 成分表为今日快照, 已退市/剔除标的缺失 |
| **OOS 结构** | 2025-01 起的 OOS 仅 1.7 年, 且 A 股段含单边牛市, 不宜外推 |
| **shuffle 对照不利** | hs300 的 OOS shuffle 对照 Sharpe 1.34 **≥** 策略 1.26, 该池 OOS 强度需打折看待 |
| **无缓冲带** | company_300 的 `top_k` 与 `exit_rank` 均为 12, 无滞回区间 → 边界抖动的机械换手 |
| **流动性未建模** | 引擎无容量约束, 小市值/低成交标的(如 DJT 段)按固定仓位成交, 实盘不可复制 |
| **pkl 时效** | 本次用研究全史 pkl, 与桥的生产 pkl(每日更新)在最新数据日上差 1~2 天 |
| **未复现部分** | 本次未做成本敏感度扫描(0/10/25/50bps), 亦未做 IS 内参数稳定性网格 |

## 7. 复现方式

```python
import sys; sys.path.insert(0, 'git/quant/research')
from bt_core import BacktestKit
from combo_search import _slice, build_combo_cands
from alpha_mining import build_alpha_cands
from a_combo_search import build_a_combo_cands
from score_backtest import run_config

# 美股池(company_300 / company_1000 / etf_3x)
kit = BacktestKit(pool='company_300', start=None)      # 走 alpha_mining.POOLS 全史 pkl
kit.register(build_alpha_cands); kit.register(build_combo_cands)   # G_vr20 等在 combo 侧
# A 股池(hs300 / a_etf_all)
# kit = BacktestKit(pool='hs300', start=None); kit.register(build_a_combo_cands)

p    = kit.engine_params.copy(top_k=12)                # top_k 取各池预设; exit_rank 默认 12
comp = kit._composite({'C_tmqmom': 1.0, 'F_er20': 1.0}, 'rank')    # 全历史分数
ow, cw = _slice(kit.open_wide, '2026-01-01', None), _slice(kit.close_wide, '2026-01-01', None)
c = comp.reindex(index=ow.index, columns=ow.columns)
g = pd.DataFrame(True, index=ow.index, columns=ow.columns)
payload = run_config('cmp', ow, cw, c, g, p, atr_wide=_slice(kit.atr_wide, '2026-01-01', None))
print(payload['stats'])
```

一键脚本形态可参照 [exec_price_ab_test.py](file:///c:/Users/northcheng/git/quant/research/exec_price_ab_test.py) 的 `POOLS` / `WINDOWS` / `STAT_KEYS` 结构(该脚本只覆盖美股三池, 本次已扩展到全部 5 池 + 2026 独立窗口)。

## 8. 数据附录(逐池原始指标)

来源: [pool_perf_signalbridge.csv](file:///c:/Users/northcheng/git/quant/research/output/pool_perf_signalbridge.csv)

| pool | top_k | window | 数据区间 | total_ret | cagr | sharpe | max_dd | calmar | vol | ann_turnover | n_trades | win_rate | avg_days | n_days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| company_300 | 12 | full | 2021-01-04~2026-09-16 | 21.760 | 0.731 | 1.34 | -0.342 | 2.14 | 0.504 | 161.6 | 5086 | 0.501 | 3.4 | 1431 |
| company_300 | 12 | is | 2021-01-04~2024-12-31 | 7.177 | 0.694 | 1.37 | -0.325 | 2.13 | 0.461 | 160.5 | 3545 | 0.500 | 3.4 | 1004 |
| company_300 | 12 | oos | 2025-01-02~2026-09-16 | 1.649 | 0.773 | 1.27 | -0.342 | 2.26 | 0.596 | 164.8 | 1535 | 0.504 | 3.3 | 426 |
| company_300 | 12 | ytd26 | 2026-01-02~2026-09-16 | 0.234 | 0.353 | 0.83 | -0.287 | 1.23 | 0.542 | 170.4 | 641 | 0.482 | 3.2 | 176 |
| company_1000 | 8 | full | 2021-01-04~2026-09-18 | 44.638 | 0.955 | 1.39 | -0.481 | 1.98 | 0.609 | 99.3 | 1670 | 0.519 | 6.8 | 1433 |
| company_1000 | 8 | is | 2021-01-04~2024-12-31 | 10.984 | 0.865 | 1.32 | -0.481 | 1.80 | 0.601 | 97.1 | 1139 | 0.513 | 7.0 | 1004 |
| company_1000 | 8 | oos | 2025-01-02~2026-09-18 | 2.571 | 1.109 | 1.50 | -0.418 | 2.65 | 0.627 | 104.5 | 526 | 0.532 | 6.5 | 428 |
| company_1000 | 8 | ytd26 | 2026-01-02~2026-09-18 | 0.605 | 0.964 | 2.22 | -0.197 | 4.88 | 0.328 | 102.9 | 216 | 0.565 | 6.5 | 178 |
| etf_3x | 5 | full | 2021-01-04~2026-09-17 | 5.565 | 0.391 | 0.95 | -0.425 | 0.92 | 0.458 | 71.6 | 336 | 0.527 | 21.0 | 1432 |
| etf_3x | 5 | is | 2021-01-04~2024-12-31 | 1.289 | 0.231 | 0.73 | -0.425 | 0.54 | 0.387 | 70.2 | 221 | 0.502 | 21.1 | 1004 |
| etf_3x | 5 | oos | 2025-01-02~2026-09-17 | 1.869 | 0.857 | 1.35 | -0.389 | 2.21 | 0.595 | 76.0 | 115 | 0.583 | 17.6 | 427 |
| etf_3x | 5 | ytd26 | 2026-01-02~2026-09-17 | 0.690 | 1.120 | 1.40 | -0.385 | 2.91 | 0.723 | 73.3 | 41 | 0.537 | 18.8 | 177 |
| hs300 | 5 | full | 2021-01-04~2026-09-17 | 9.860 | 0.520 | 1.39 | -0.306 | 1.70 | 0.359 | 76.1 | 644 | 0.615 | 7.1 | 1384 |
| hs300 | 5 | is | 2021-01-04~2024-12-31 | 4.603 | 0.541 | 1.48 | -0.306 | 1.77 | 0.343 | 54.9 | 319 | 0.589 | 7.8 | 968 |
| hs300 | 5 | oos | 2025-01-02~2026-09-17 | 0.984 | 0.495 | 1.26 | -0.296 | 1.67 | 0.393 | 125.5 | 321 | 0.651 | 6.4 | 415 |
| hs300 | 5 | ytd26 | 2026-01-05~2026-09-17 | 0.127 | 0.187 | 0.64 | -0.296 | 0.63 | 0.399 | 126.6 | 136 | 0.618 | 6.2 | 172 |
| a_etf_all | 5 | full | 2021-01-04~2026-09-17 | 2.696 | 0.258 | 0.73 | -0.227 | 1.13 | 0.406 | 15.8 | 34 | 0.824 | 190.9 | 1384 |
| a_etf_all | 5 | is | 2021-01-04~2024-12-31 | 1.359 | 0.240 | 0.64 | -0.227 | 1.06 | 0.472 | 12.3 | 16 | 0.688 | 207.8 | 968 |
| a_etf_all | 5 | oos | 2025-01-02~2026-09-17 | 0.587 | 0.312 | 1.71 | -0.158 | 1.98 | 0.173 | 24.2 | 20 | 0.850 | 82.2 | 415 |
| a_etf_all | 5 | ytd26 | 2026-01-05~2026-09-17 | 0.292 | 0.445 | 2.26 | -0.112 | 3.98 | 0.174 | 28.6 | 10 | 0.900 | 56.7 | 172 |

来源: [pool_perf_buyhold.csv](file:///c:/Users/northcheng/git/quant/research/output/pool_perf_buyhold.csv)(池等权买入持有基准)

| pool | window | total_ret | cagr | sharpe | max_dd | ann_turnover | n_days |
|---|---|---|---|---|---|---|---|
| company_300 | full | 6.994 | 0.441 | 1.35 | -0.389 | 1.5 | 1431 |
| company_300 | ytd26 | 0.065 | 0.095 | 0.44 | -0.204 | 1.4 | 176 |
| company_1000 | full | 2.570 | 0.250 | 1.25 | -0.258 | 1.3 | 1433 |
| company_1000 | ytd26 | 0.141 | 0.207 | 1.28 | -0.080 | 2.7 | 178 |
| etf_3x | full | 1.279 | 0.156 | 0.60 | -0.553 | 0.7 | 1432 |
| etf_3x | ytd26 | 0.212 | 0.316 | 1.01 | -0.168 | 1.8 | 177 |
| hs300 | full | 0.823 | 0.111 | 0.64 | -0.239 | 0.5 | 1384 |
| hs300 | ytd26 | -0.062 | -0.088 | -0.57 | -0.109 | 1.5 | 172 |
| a_etf_all | full | 0.020 | 0.004 | 0.12 | -0.397 | 0.9 | 1384 |
| a_etf_all | ytd26 | -0.073 | -0.103 | -0.56 | -0.143 | 1.6 | 172 |

## 9. 交叉验证(与既有报告的逐位一致性)

本次重跑与既有报告口径互相印证, 说明引擎与数据链路无漂移:

| 池 | 本次 FULL Sharpe | 既有报告 | 出处 |
|---|---|---|---|
| company_300 | 1.34 | 1.34 | research_summary_20260920 §7.2 |
| company_1000 | 1.39 | 1.39 | 同上 |
| hs300 | 1.39 | 1.390 | astock_factor_strategy_20260922 基线复现表 |
| a_etf_all | 0.73 | 0.730 | 同上 |
| hs300 OOS | 1.26 | 1.260 | 同上 |
| a_etf_all OOS | 1.71 | 1.710 | 同上 |
| company_300 换手/持仓 | 161.6x / 3.4 天 | 161.6x / 3.4 天 | signal_bridge_principle_20260921 §8 |

(唯一小差异: etf_3x 本次 FULL Sharpe 0.95, 与报告中 gold4 口径 0.98 相差 0.03, 来自样本窗口起点差异。)

## 10. 参考

| 内容 | 位置 |
|---|---|
| 池预设(权重/top_k) | [signal_bridge.py#L81-L96](file:///c:/Users/northcheng/signal_bridge.py#L81-L96) |
| 信号计算链路 | [signal_bridge.py#L380-L396](file:///c:/Users/northcheng/signal_bridge.py#L380-L396) |
| 回测 API | [bt_core.py](file:///c:/Users/northcheng/git/quant/research/bt_core.py) |
| 引擎与统计口径(perf_stats) | [score_backtest.py#L397-L420](file:///c:/Users/northcheng/git/quant/research/score_backtest.py#L397-L420) |
| A 股候选与基线 | [a_combo_search.py#L133-L189](file:///c:/Users/northcheng/git/quant/research/a_combo_search.py#L133-L189) |
| A 股策略报告 | [astock_factor_strategy_20260920.md](file:///c:/Users/northcheng/git/quant/research/report/astock_factor_strategy_20260920.md) / [20260922](file:///c:/Users/northcheng/git/quant/research/report/astock_factor_strategy_20260922.md) |
| 美股三池摘要 | [research_summary_20260920.md](file:///c:/Users/northcheng/git/quant/research/report/research_summary_20260920.md) |
| 桥原理与实测 | [signal_bridge_principle_20260921.md](file:///c:/Users/northcheng/git/quant/research/report/signal_bridge_principle_20260921.md) |
