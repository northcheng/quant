# A 股因子/组合/策略再挖掘报告（hs300 / a_etf_all）

日期: 2026-09-22 | 数据: `~/quant/{hs300,a_etf_all}_day_ta_data_research.pkl`（2021-01 ~ 2026-09 全史）
承接: `astock_factor_strategy_20260920.md` §6.1 的待办（"给组合层加 `--objective min(is,oos)`"）
本次工具: `a_robust_combo.py`（组合层双半窗重估 + 从基线出发的改进）、`a_robust_probe.py`（冗余度/稳健性探针）
产出目录:
- `output/a_mine_20260922_191603/`（R/I/W 残差动量·信息离散度·回撤时间族）
- `output/k_mine_20260922_191917/`（K 条件层族）
- `output/x_mine_20260922_192012/`（X 自建 A 股专属族）
- `output/a_robust_20260922_193204/`（组合层重估：矩阵 + 三路终验 + 六窗）

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| 有更好的新因子吗？ | **没有**。新挖 R/I/W、K、X 共 4 族（约 145 个候选组合）**全部无 A/B 级、无 FDR 显著**；唯一 B 级 `X_ovnshare60` 与在用的 `G_oviv20` 相关系数 0.61~0.81，是同一因子的变形 |
| 有更好的因子组合吗？ | **没有**。以 `min(IS1,IS2)` 为目标从零贪心，两池都止步于单信号；从基线出发做"删除/加入/调幅"，hs300 的唯一"改进"在 OOS 由 1.26 崩到 0.56，被否决 |
| 有更好的交易策略吗？ | **没有确定的**。hs300 维持现行三因子预设（三项各有角色，任一删除都在某窗口塌陷）；a_etf_all 存在**可选简化**（去掉 `G_oviv20`，六窗全部 ≥ 基线），但差距在噪声带内，建议只做小仓位 A/B 而非直接替换 |
| 生产要不要改？ | **不改**。新增族进"观察名单"不上线 |

---

## 2. 因子挖掘：新增四族全部未通过

沿用 0920 报告口径（多 horizon 5/20/60 可交易口径 → NW-HAC t 检验 → BH-FDR → 分级）：

| 族 | 模块 | 分级计数 | 头部候选 | 判定 |
|---|---|---|---|---|
| R/I/W 残差动量·信息离散度·回撤时间 | `a_stat_mining.py` | C弱 16 / S静态 12 / D无效 8 | `W_martin60 h60` score 0.669（q 0.381） | 无 A/B，**无 FDR 显著**（`q_max` 最小 0.38） |
| K 条件层（状态依赖） | `conditional_mining.py` | C弱 43 / D无效 17 / S静态 3 | — | 无 A/B，无 FDR 显著 |
| X 自建 A 股专属（隔夜结构/涨跌停/彩票极值/位置/流动性） | `a_share_mining.py` | C弱 28 / D无效 22 / S静态 9 / **B中 1** | `X_ovnshare60 h60` score 0.645 | 唯一 B 级**无动态增量**（dyn_mean −0.020） |

**关键交叉验证**（`a_robust_probe.py` A 段，日截面秩相关 rho）：

| 因子对 | hs300 FULL | a_etf_all FULL | 含义 |
|---|---|---|---|
| `G_oviv20` ~ `X_ovnshare60` | **+0.620** | **+0.789** | 隔夜/日内波动比的两种写法，**同族冗余** |
| `N_range20` ~ `G_oviv20` | −0.053 | +0.311 | hs300 正交；ETF 池同向同族（低波动暴露） |
| `N_range20` ~ `F_beta60` | −0.570 | −0.703 | 低振幅 vs 低 beta，强负相关（两池一致） |

- `X_ovnshare60` 单因子实测（a_etf_all）：IS1 0.57 / IS2 2.11 / OOS **1.15** / FULL 0.70，**弱于** 已在用的 `G_oviv20`（0.57/1.89/1.13/0.83 及组合口径）；hs300 方向为负且 OOS −0.67。→ **不引入**。
- 观察名单（未上线）：`X_low52dist h60`（q 0.122 为全表最低，dyn_mean +0.054 正）、`X_limit20 h20`、`X_hi52prox h20`、`W_martin60 h60`、`I_id20 h20`。

**小结：A 股两池的可用 alpha 已被现有 F/H/N/C/D/G 族覆盖；本轮新增 4 族未带来增量信息。**

---

## 3. 组合层口径升级：目标函数 = min(IS1, IS2)

- 窗口：`IS1 = 2021-01~2022-12`、`IS2 = 2023-01~2024-12`（两种 regime）、`OOS = 2025-01~`；
  OOS 再拆 `OOSA(25H1)/OOSB(25H2)` 看稳定性。**OOS 只复验，不参与选参**。
- 候选池：`a_combo_search.CAND_SIGNALS + FILTER_CANDS`（14）+ 补充探针 `I_id20 / X_ovnshare60` = **16 个 × 双向**。
- 组合口径与生产完全一致：`Σ(wᵢ/Σ|wᵢ|)×当日截面 rank_pct`，NaN 记 0.5，`top_k=5 / exit_rank=12`，单边 10bps。
- 基线复现（与 0920 报告逐位一致）：

| 池 | 预设 | FULL | IS1 | IS2 | OOS | buyhold(FULL) |
|---|---|---|---|---|---|---|
| hs300 | `N_range20:-1, D_ma20_dist:-1, C_tmqmom:0.5` | 1.390 | 1.630 | 1.520 | 1.260 | 0.640 |
| a_etf_all | `N_range20:1, G_oviv20:0.5` | 0.730 | 0.560 | 2.060 | 1.710 | 0.120 |

---

## 4. 候选稳健性矩阵（16 × 双向，目标 min(IS1,IS2)）

hs300 前 8：

| 信号 | 方向 | obj_min | IS1 | IS2 | OOS | FULL |
|---|---|---|---|---|---|---|
| `F_beta60` | neg | **1.53** | 1.53 | 1.97 | **−0.22** | 1.26 |
| `N_range20` | neg | 1.40 | 1.40 | 1.67 | 1.23 | 1.36 |
| `H_adxstr_alpha` | neg | 1.31 | 1.45 | 1.31 | −0.39 | 0.83 |
| `X_ovnshare60` | neg | 1.23 | 1.68 | 1.23 | −0.67 | 0.95 |
| `G_oviv20` | neg | 1.09 | 1.09 | 1.72 | −1.01 | 0.84 |
| `G_chop20` | pos | 1.07 | 1.07 | 1.51 | −0.39 | 0.81 |
| `D_ma20_dist` | neg | 0.95 | 1.61 | 0.95 | **1.49** | 1.26 |
| `C_tmqmom` | neg | 0.36 | 0.75 | 0.36 | −1.13 | 0.37 |

a_etf_all 全体（后半为负值，说明 ETF 池有效信号极少）：

| 信号 | 方向 | obj_min | IS1 | IS2 | OOS | FULL |
|---|---|---|---|---|---|---|
| `N_range20` | pos | **0.58** | 0.58 | 2.64 | 2.49 | 0.83 |
| `G_oviv20` | pos | 0.57 | 0.57 | 1.89 | 1.13 | 0.83 |
| `X_ovnshare60` | pos | 0.57 | 0.57 | 2.11 | 1.15 | 0.70 |
| `G_vr20` | pos | 0.14 | 0.14 | 0.32 | 0.33 | 0.22 |
| 其余 12 个 | pos | ≤ −0.03 | | | | |

**两个关键读法**：

1. **单信号支配现象**：两池各自的 obj 榜首都是单信号（hs300 `F_beta60:-1`、a_etf_all `N_range20:+1`），且贪心在第 1 轮就停止（无新增能让 obj 提升 > margin 0.1）。→ A 股两池**没有多因子协同**，多因子基线的价值不来自"组合 alpha"，而来自"regime 对冲"（见 §5）。
2. **obj 高 ≠ 可上线**：`F_beta60:-1` obj 1.53 最高，但 OOS −0.22（基线 1.26）；`H_adxstr_alpha / X_ovnshare60 / G_oviv20`（hs300 向）obj 均 >1 而 OOS 全为负。→ IS 双半窗能防"单窗过拟合"，但**防不了"只在熊市有效"**，OOS 复核不可省。

---

## 5. 从基线出发的改进（核心章节）

### 5.1 hs300：基线不可改进（维持）

单成分删除扫描（top_k=5）：

| 组合 | IS1 | IS2 | OOS | FULL |
|---|---|---|---|---|
| 基线 `N:-1, D:-1, C:0.5` | 1.63 | **1.52** | **1.26** | **1.39** |
| 去 `N_range20` | 1.63 | 1.58 | 0.67 | 1.23 |
| 去 `D_ma20_dist` | 1.63 | 1.37 | 1.42 | 1.33 |
| 去 `C_tmqmom` | 1.62 | 1.31 | 0.56 | 1.05 |

- 算法按 obj 选出的唯一"改进"= 去 `N_range20` 并把 `D_ma20_dist` 调到 −0.5（obj 1.520 → **1.630**），**但 OOS 从 1.26 崩到 0.56**，且 top_k=8/12 也不稳（obj 0.75/1.02）→ **否决**。
- 三项各司其职：`D_ma20_dist` 主 IS1、`N_range20` 主 IS2 与 OOS、`C_tmqmom` 单独很弱（obj 0.36，OOS −1.13）却在组合内把 OOS 从 0.56 抬回 1.26 —— **组合收益不是成分 alpha 的线性叠加**，而是"低波/超卖打分 + 动量质量再排序"的交互。
- 因此：**hs300 维持 `{N_range20:-1, D_ma20_dist:-1, C_tmqmom:0.5}`，top_k=5**。

### 5.2 a_etf_all：存在可选简化，但差距在噪声带

`G_oviv20` 权重扫描（`N_range20:1` 固定，top_k=5）：

| 权重 | IS1 | IS2 | OOS | FULL |
|---|---|---|---|---|
| 0.0（=单因子） | 0.58 | **2.64** | **2.49** | **0.83** |
| 0.25 | 0.56 | 2.68 | 1.69 | 0.76 |
| 0.50（=现行基线） | 0.56 | 2.06 | 1.71 | 0.73 |
| 1.00 | 0.54 | 2.14 | 1.46 | 0.71 |

- `G_oviv20` 权重越大，OOS/FULL **单调变差**；IS1 只在 0.54~0.58 抖动（噪声级）。
- rho(`N_range20`, `G_oviv20`) = +0.23~0.38 → 两者是**同族（低波动）暴露**，叠加不产生新信息，只稀释。
- top_k 敏感性一致：单因子在 top_k=5/8/12 分别为 obj 0.58/0.44/0.15，均 ≥ 基线 0.56/0.38/0.03。
- **判定**：ETF 池的 alpha 基本由 `N_range20` 单因子承担，`G_oviv20` 无 OOS 贡献。差异（OOS 2.49 vs 1.71）虽一致但幅度受单一牛市窗影响，**建议按"可选简化"处理**：保持现基线不动，或在 ETF 池做小仓位 A/B 验证后再替换。

### 5.3 三路终验（六窗 + 对照）

| 池 | 路径 | 权重 | IS1 | IS2 | OOS | OOSA | OOSB | FULL | FULL_shuffle |
|---|---|---|---|---|---|---|---|---|---|
| hs300 | 零贪心 | `F_beta60:-1` | 1.53 | 1.97 | −0.22 | 1.22 | −0.78 | 1.26 | −0.64 |
| hs300 | 基线改进 | `D_ma20_dist:-0.5, C_tmqmom:0.5` | 1.63 | 1.75 | 0.56 | 0.86 | 0.38 | 1.27 | −0.76 |
| hs300 | **基线（采用）** | `N_range20:-1, D_ma20_dist:-1, C_tmqmom:0.5` | 1.63 | 1.52 | **1.26** | 1.67 | 1.05 | **1.39** | **−0.48** |
| a_etf_all | 零贪心 | `N_range20:1` | 0.58 | 2.64 | 2.49 | 2.41 | 2.65 | 0.83 | 0.37 |
| a_etf_all | 基线改进（=基线） | `N_range20:1, G_oviv20:0.5` | 0.56 | 2.06 | 1.71 | 1.20 | 2.22 | 0.73 | 0.39 |

hs300 基线的 `FULL_shuffle = −0.48`（远低于 1.39）是全表最强的"分数含真实信息"证据；ETF 池 shuffle 为正（0.39）是池小 + 低波结构性 beta 的已知现象（见 0920 §7）。

---

## 6. 反向控制与 shuffle 分窗复核（重要限定）

**反向控制全部通过**（符号翻转后收益应显著变差）：

| 控制 | 反向表现 | 原方向 |
|---|---|---|
| hs300 `N_range20:+1` | OOS −0.16 / FULL 0.50 | OOS 1.23 / FULL 1.36 |
| hs300 `F_beta60:+1` | IS1 −0.09 / FULL 0.60 | IS1 1.53 / FULL 1.26 |
| a_etf_all `N_range20:−1` | OOS −0.17 / FULL −0.09 | OOS 2.49 / FULL 0.83 |

**shuffle（时序置换：保留截面分布、打乱日间对齐）分窗结果需更正 0920 报告的一处乐观表述**：

| 池 / 组合 | IS1 shuffle vs 原 | IS2 shuffle vs 原 | OOS shuffle vs 原 |
|---|---|---|---|
| hs300 基线 | 0.23 vs 1.63 | 0.81 vs 1.52 | **1.34 vs 1.26** |
| a_etf_all 基线 | 0.56 vs 0.56 | 1.41 vs 2.06 | 0.52 vs 1.71 |
| a_etf_all `N_range20:1` | 0.48 vs 0.58 | 1.06 vs 2.64 | **−0.13 vs 2.49** |

- IS 段两池 shuffle 均远低于原值 → **时序信息真实存在**（0920 报告的结论在这一段成立）。
- 但 hs300 **OOS 段 shuffle（1.34）≥ 原策略（1.26）** → 2025 段 hs300 的收益几乎全部来自**持久的截面 tilt（静态身份）**，时序打分没有贡献。结合"池 = 当前指数成分（幸存者偏差）"，**hs300 的 OOS 1.26 需要打折看待**。
- a_etf_all 相反：单因子 `N_range20` OOS shuffle −0.13 << 2.49，**时序信息纯净**；但 ETF 池截面仅 97 只（早期 26 只），低截面期噪声大。

---

## 7. 方法论沉淀（供后续复用）

1. **目标函数用 `min(IS1,IS2)` 而非 `min(IS,OOS)`**：OOS 一旦进目标函数就不再是样本外；用 IS 内部两个 regime 代替"防过拟合"，OOS 只做终审。
2. **该目标函数会漏掉"只在 OOS 有效"的成分**：`C_tmqmom:+1` 就是典型（obj ≤0.36 却贡献 OOS +0.70）。→ 必须配"**从基线出发的删除/加入/调幅**"路径，而不是只看排行榜。
3. **IS 双窗仍不足以选参**：算法选出的 hs300"改进"在 OOS 崩掉。→ **选参用 IS，采纳必须过 OOS 复核**（本次即用此规则否决）。
4. **新因子必须先做冗余度体检**：`X_ovnshare60` 的 FDR/q 数字看着可用（q 0.216），但 rho 0.62~0.79 与在用因子同族 → 直接判死。冗余度应作为分级的前置门槛，而非事后说明。
5. **shuffle 必须分窗看**：全窗 shuffle 为负会掩盖"某个子窗全靠静态 tilt"的问题（本次 hs300 OOS 即如此）。

---

## 8. 落地建议

1. **生产不改**：hs300 `{N_range20:-1, D_ma20_dist:-1, C_tmqmom:0.5}`、a_etf_all `{N_range20:1, G_oviv20:0.5}`，均 top_k=5 / exit_rank=12 / 单边 10bps 保持现状。
2. **可选实验（低优先）**：ETF 池试 `N_range20:1` 单因子（或把 `G_oviv20` 降到 0.25），小仓位 A/B 观察，别直接替换。
3. **观察名单（不上线）**：`X_low52dist h60`、`X_limit20 h20`、`X_hi52prox h20`、`W_martin60 h60`、`I_id20 h20`。
4. **不要做的事**：`F_beta60` 单信号（OOS −0.22）、`X_ovnshare60`（与 `G_oviv20` 冗余）、任何"IS 好看但 OOS 塌陷"的删减版基线。

## 9. 局限与后续

- **幸存者偏差**：两池均为"当前成分 × 全历史"，`OOS` 与静态 tilt 相关结论偏乐观；可用"历史指数成分"重跑验证。
- **OOS 仅 1.7 年且为单边牛市**（2025+），`OOSA/OOSB` 也只是把同一牛市切两半，不足以覆盖熊市。
- 尚未做严格 **walk-forward**（滚动重选权重）与 **参数扰动**（margin/top_k/成本）联合压力测试；下一步建议在 `a_robust_combo.py` 上加 `--walk-forward` 与成本 20/30bps 情景。
- ETF 池早期截面仅 26 只，`MIN_CS≥10` 保护有限，低截面期结果不宜单独解读。

## 10. 复现

```bash
cd ~/git/quant/research
# a_stat_mining.py / conditional_mining.py 的 POOLS 仅内置 etf_3x/company_300/company_1000，
# 跑 hs300/a_etf_all 必须用 --pkl "池=路径" 覆盖（分号分隔，注意文件名含 _day_）。
# 该覆盖分支不做 expanduser，故 --pkl 内必须写 $HOME/... （字面 ~ 会被 os.path.exists 判为不存在）
PKL="hs300=$HOME/quant/hs300_day_ta_data_research.pkl;a_etf_all=$HOME/quant/a_etf_all_day_ta_data_research.pkl"
~/.venv/Scripts/python.exe -u a_stat_mining.py --pools hs300,a_etf_all --pkl "$PKL"  # R/I/W 族
~/.venv/Scripts/python.exe -u a_share_mining.py --pools hs300,a_etf_all  # X 族（POOLS 默认即 hs300/a_etf_all）
~/.venv/Scripts/python.exe -u conditional_mining.py --pools hs300,a_etf_all --pkl "$PKL"  # K 族
~/.venv/Scripts/python.exe -u a_robust_combo.py    # 组合层重估(约 340s, 输出 a_robust_*)
~/.venv/Scripts/python.exe -u a_robust_probe.py    # 冗余度 + 敏感性/反向控制探针
```

生产接入（与 `signal_bridge.py` 的 `A_POOL_PRESETS` 一致，未变更）：

```python
from bt_core import BacktestKit
from a_combo_search import build_a_combo_cands
kit = BacktestKit(pool='hs300', start='2021-01-01'); kit.register(build_a_combo_cands)
res = kit.run({'N_range20': -1, 'D_ma20_dist': -1, 'C_tmqmom': 0.5})   # ETF 池: {'N_range20': 1, 'G_oviv20': 0.5}
```
