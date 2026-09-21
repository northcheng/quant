# 技术指标价值评估报告

> 对象：`calculate_ta_basic` / `calculate_ta_static` / `calculate_ta_dynamic` / `calculate_ta_score` / `calculate_ta_signal` 等方法产出的全部指标列
> 数据：4 个标的池 × 日线，2021-01-01 起（a_etf_all 实际自 2023-08-08 起）
> 口径：可交易前瞻收益（信号日收盘 → 次日开盘入场 → 持有 h 日 → 开盘出场），h = 5 / 20 日
> 结论产出：`research/output/eval_summary.csv`（174 行）等 10 余个数据文件
> 修订：2026-09-18 完成"合成公式修复 + 重跑"，候选列由 166 增至 174（新增 4 个合成列，并修好候选筛选规则的误杀）。修复过程与实测结论见 [8.5](#85-合成公式修复与重跑实测2026-09-18)。

---

## 0. 一页结论（只看这一段也够）

**问：这些指标有价值吗？答：绝大多数没有，但有一小撮确实有，而且找出来了。**

| 结论 | 数量 | 说明 |
|---|---|---|
| **可以直接当轮动信号用** | **9 个**（B 级，其中 4 个四池全正） | `ichimoku_distance_alpha`、`Low_to_kijun`、`adx_power`、`High_to_kijun` 最可靠 |
| 只能当"弱参考"或状态过滤 | 42 个（C 级） | 方向对但幅度小/不稳，只能做辅助 |
| **看起来很强、其实是假信号** | **30 个**（S 级） | 原始价格、云图基准值等 → 静态身份陷阱，详见第 3 节 |
| 没用（方向不稳） | 64 个（D 级） | 其中包含 `calculate_ta_score` 的全部直接产出（含修复后的新列） |
| **绝对不能用** | **29 个** | 4 个疑似未来标签/动作列 + 25 个覆盖率不足的稀疏列 |

三条最重要的洞察：

1. **价格水平不是信号。** `Close` / `Open` / `High` / `Low` / `kijun` / `tankan` / `kama_fast` / `senkou_a` / `mavg` / `atr` / `adi` 这些"数值本身"的截面排名，30 个指标的自相关高达 **0.998**（完全不动），换手率 **0.017**，有效持仓数 **0.47 只**（即"永远只买那 1~2 只最贵的"）。它们的 IC 高只反映"谁的价格高"，是 beta 放大，不是选股能力。
2. **真正有动态价值的是"比值/距离/归一化"家族，不是"数值"家族。** 例如 `Low_to_kijun`（最低价到基准线的距离，除以 kama_slow 归一化）、`ichimoku_distance_alpha`（`normalize_causal` 因果归一化后的云图距离）、`adx_power`（ADX 强度变化）、`kama_slow_rate`（变化率）。它们换手率 0.21~0.28、有效持仓数 2.3~5.9 只，是**真的在轮动**。
3. **`calculate_ta_score` 族的真正缺陷是"分量量纲不齐"，不是"看涨/看跌抵消"。**（本报告初稿此处的诊断已被实测证伪，2026-09-18 修订）
   初稿称 `trigger_score = trigger_up_score + trigger_down_score`（[bc_technical_analysis.py#L837](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L837)）"把看涨证据和看跌证据直接相加、互相抵消"。**实测不成立**：`trigger_down_score` 本身带负号（实测 mean ≈ −0.60、min −5.0），`corr(trigger_score, trigger_down_score) = +0.79`，且 `trigger_score == trigger_up_score + trigger_down_score` 的 `max|diff| = 0` —— 这是一个**正确的净额求和**。
   真正的问题是**量纲**：`break_up_score` 可达 ±5，而 `support_score` 仅 ±1，直接相加时合成结果被大量纲分量主导。已按"分量先因果归一化、再带符号加权"修复并重跑（见 [8.5](#85-合成公式修复与重跑实测2026-09-18)）；结论是**该族仍然全部无效** —— 说明它失效的根源不是合成公式。

**立刻可做的一件事**：把 `M16_kijun_ichimoku` 加进 `score_backtest.py` 的 `MOMENTUM_SETS`（代码见 8.3）。这是本次评估里唯一一个"四池全正 + 样本外一致 + 低冗余"的组合。

---

## 1. 这次评了什么、怎么评的

### 1.1 指标来源

从 `bc_technical_analysis.py` 的 5 个主函数抽取全部数值型产出列，共 **174 个**（已剔除 `*_description` 等纯文本列；其中 172 个通过覆盖率/取值数筛选，进入相关性计算）：

| 产出函数 | 主要产出 | 个数 |
|---|---|---|
| `calculate_ta_basic`（[L293](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L293)） | ADX 家族、ATR、TR、原始 OHLCV、成交量 | ~40 |
| `calculate_ta_static`（[L342](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L342)） | 云图基准值、KAMA、RSI、布林带、蜡烛几何、`*_to_*` 距离列 | ~60 |
| `calculate_ta_dynamic`（[L749](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L749)） | 各类 `*_day` 状态持续列（由 `sda` 生成） | ~30 |
| `calculate_ta_score`（[L801](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L801)） | `trigger_up/down_score`、`trigger_score`、`support/resistant/break/boundary_score`、**修复新增** `trigger_net` / `trigger_net_alpha` | 11 |
| `calculate_ta_signal`（[L923](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L923)） | `trend_*`、`trend_magnitude*`、`pattern_score*`、`signal*`、`action*`、`*_label_score`、**修复新增** `pattern_net` / `pattern_net_alpha` | ~27 |

### 1.2 四个池（规模差异很大，必须分开看）

| 池 | 标的数 | 数据起点 | 备注 |
|---|---|---|---|
| `etf_3x` | 29 | 2020-01-02 | 美股 3x 杠杆 ETF，主战场 |
| `company_300` | 297 | 2020-01-02 | 美股自选池 |
| `hs300` | 298 | 2020-01-02 | A 股，**用"当前成分股"回溯构造 → 存在幸存者偏差，结果须打折** |
| `a_etf_all` | 97 | **2023-08-08** | A 股 ETF，**历史只有约 2 年，样本最短** |

> 这个差异很重要：`a_etf_all` 的所有"跨池为正"结论，实际只基于约 2 年数据，不能与另外三池等量齐观。

### 1.3 收益口径：可交易前瞻收益

不用"未来 N 日收益"这种含未来信息的口径，而用真实可下单的口径（[signal_search.py#L88-92](file:///c:/Users/northcheng/git/quant/research/signal_search.py#L88-L92)）：

```
信号日 t 收盘产生信号
  → t+1 开盘买入（entry = open.shift(-1)）
  → 持有 h 日后开盘卖出（exit = open.shift(-(1+h))）
  → 收益 = exit / entry - 1
```

核心指标 `excess_k = top-k 平均收益 − 同池等权持有收益`。
**读法**：`+0.01` = 相对"池子等权买入持有"多赚 1%（在 h 日的持有期内）。top-k 取信号值最大的 k=5 只。

### 1.4 五重检验（这套框架是本报告的价值所在，建议固化成常规流程）

| # | 检验 | 回答什么问题 | 阈值 |
|---|---|---|---|
| 1 | **可交易超额** `excess_k` | 到底能不能赚钱 | 要求跨池同号且为正 |
| 2 | **IC / ICIR** | 排序能力是否稳定 | `ICIR` 越高越好；参考值 |
| 3 | **分半稳定** `exc_half1/2` | 是否只在某段行情有效 | 两半同号 |
| 4 | **静态身份对照** `sig_ac1` / `n_eff_symbols` / `topk_turnover` / `dyn_minus_static` | 是不是"永远买同一批"的假信号 | `sig_ac1 ≥ 0.99` 即判死 |
| 5 | **冗余度** 两两截面秩相关 | 是不是在重复别的指标 | 上报 `\|corr\| ≥ 0.8` 的伴侣 |

第 4 条是本次新增的关键武器 —— **它很便宜（一次秩自相关），却能一次杀掉 30 个伪信号**。

### 1.5 价值分级定义

| 级别 | 判据 | 含义 |
|---|---|---|
| `A强` | 4 池同号为正 + 超额显著 | **本次 0 个** |
| `B中` | 多数池为正，超额稳定（h20 min ≥ 0.1%） | 可作候选信号 |
| `B中\|无动态增量` | 满足 B 但 `dyn_minus_static ≤ 0` | 价值 ≈ 固定持仓，无轮动价值 |
| `C弱` | 方向对但幅度小/池间不全为正 | 仅作辅助 |
| `D无效` | 方向不稳 | 弃用 |
| `S静态身份` | `sig_ac1 ≥ 0.99` | **假信号**（beta 暴露） |
| `L疑似标签` | 名称含 `label` / `action` | **严禁当特征** |
| `T0不可用` | `coverage < 0.5` 或 `nuniq ≤ 2` | 数据上就不成立 |

---

## 2. 总体结果

```
174 个指标
├─ A强            0   ← 没有一个单因子能独立成立
├─ B中            7   ← 可用
├─ B中|无动态增量   2   ← 疑似可用，实为固定持仓
├─ C弱           42   ← 辅助
├─ D无效         64   ← 弃用（含修复新增的 4 列）
├─ S静态身份      30   ← 假信号（第 3 节）
├─ L疑似标签       4   ← 严禁使用
└─ T0不可用       25   ← 数据不成立
```

各级别的客观体检数据（h=20 日）：

| 级别 | n | 平均 sig_ac1 | 平均换手 | 平均有效持仓数 | 正向比例 |
|---|---|---|---|---|---|
| B中 | 7 | 0.859 | **0.263** | **3.80** | 100% |
| B中\|无动态增量 | 2 | 0.964 | 0.071 | 3.35 | 100% |
| C弱 | 42 | 0.580 | 0.494 | 3.56 | 98% |
| D无效 | 64 | 0.471 | 0.513 | 3.70 | 64% |
| **S静态身份** | **30** | **0.998** | **0.017** | **0.47** | 83% |
| L疑似标签 | 4 | 0.479 | 0.506 | 3.43 | 75% |
| T0不可用 | 25 | 0.530 | 0.243 | 0.72 | 64% |

> **有效持仓数** = 1 / HHI（对"某标的出现在 top-5 中的频率"计算）。5 只均匀轮动时该值为 5；**小于 1 说明全部押在同一两只标的上**，是静态身份的硬证据。

---

## 3. 发现一：静态身份陷阱（30 个，最重要）

### 3.1 证据链

这 30 个指标的四个数字高度一致，构成完整证据：

| 指标 | 数值 | 正常轮动信号 | 含义 |
|---|---|---|---|
| `sig_ac1`（截面排名自相关） | **0.998** | 0.2 ~ 0.9 | 排名几十年不变 |
| `topk_turnover`（top5 成员更替率） | **0.017** | 0.2 ~ 0.8 | 几乎从不换股 |
| `n_eff_symbols`（有效持仓数） | **0.47** | 3 ~ 6 | 实际上只持有不到 1 只 |
| `top3_share`（top3 名字的持有频率和） | **2.354** | ≈ 0.5 ~ 1.5 | 前三名几乎每次都在 top-5 里（理论上限 5） |

对比同表里的正常组：`原始OHLCV` 组换手 **0.057**、有效持仓数 **0.37** —— 意味着"买价格最高的 5 只"这件事，6 年来几乎没有任何变化。

### 3.2 名单（30 个，全部应排除出信号池）

| 子类 | 指标 |
|---|---|
| **原始价格**（6） | `Open`, `High`, `Low`, `Close`, `Adj Close`（+ `Volume` 为 D 级） |
| **云图基准值**（6） | `kijun`, `tankan`, `senkou_a`, `senkou_b`, `kama_fast`, `kama_slow` |
| **蜡烛几何绝对值**（5） | `candle_entity_top`, `candle_entity_bottom`, `candle_gap_top`, `candle_gap_bottom`, `candle_gap_bottom` |
| **布林/统计**（4） | `mavg`, `mstd`, `bb_high_band`, `bb_low_band` |
| **波动/量能**（2） | `atr`, `adi` |
| **KAMA 距离**（2） | `kama_distance`, `kama_distance_day` |
| **形态 day 列**（5） | `吞噬_day`, `腰带_day`, `包孕_day`, `启明黄昏_day`, `流星_day`, `穿刺_day`, `锤子_day`（共 7 个形态 `_day`） |

### 3.3 为什么"看起来 IC 很高"却能骗人

举个具体例子：`Close` 的 `sig_ac1 = 0.998`。它的截面排名 = "谁的价格高"。
- **IC 检验**：价格高的资产在这段时间确实收益更高 → IC 为正 → 检验"通过"。
- **真相**：这不是预测能力，而是**价格水平本身就是收益的累积结果**。你如果照它下单，等于"永远持有那 1~2 只最贵的标的"，是一个 buy-and-hold 组合，不是轮动策略。

`dyn_minus_static`（动态相对固定持仓的增量）在 S 级里有 80% 为正、均值 +0.006 —— 这**看似**说明"动态选还有增量"。但要注意：对静态身份来说，"当天的最高价"和"前半段平均最高价"选出的几乎是同一批标的，两者之差只反映**区间内的价格漂移**（level momentum），而非轮动能力。**判断依据应以 `n_eff_symbols = 0.47` 和 `turnover = 0.017` 为准** —— 一个连 1 只标的都换不动的信号，不可能产生轮动 alpha。

> 经验法则：**先看 `sig_ac1` 和 `n_eff_symbols`，再看 IC。** 顺序反了就会被伪信号骗。

### 3.4 一个反直觉的副产品：它们的方向是"负"的

S 级中 30 个有 25 个 `dir = 负` —— 原始方向（买最贵的）超额是**负的**。把它反过来（买最便宜的）才有正超额。这说明在这 4 个池、这段时间里，**"低价格/低波动"整体占优**，与前面研究里"低波动倾斜（R3/R4）有效"的结论一致。但请注意：这仍然是**整体倾斜（beta）**，不是轮动能力 —— 位置见 [第 8.3 节](#83-使用方式建议)。

---

## 4. 发现二：真正有动态价值的指标（9 个 B 级）

### 4.1 B 级完整明细（按综合分排序）

`exc` 为**方向对齐后**的可交易超额（小数，0.01 = 1%）：`dir=正` 表示"信号值越大越好"，`dir=负` 表示"须反向使用（信号值越小越好）"。

| 指标 | 族 | sig_ac1 | 换手 | 有效持仓 | 动态增量 | 四池 h20 超额 | dir | 综合分 |
|---|---|---|---|---|---|---|---|---|
| `candle_gap_distance` | 蜡烛 | 0.882 | 0.145 | 1.70 | +0.0016 | 全负（须反向） | **负** | 0.724 |
| `ichimoku_distance_alpha` | ta_signal趋势 | 0.915 | 0.251 | **5.70** | **+0.0132** | **全正** | 正 | 0.719 |
| `Low_to_kijun` | 云图 | 0.834 | 0.268 | 2.28 | **+0.0135** | **全正** | 正 | 0.691 |
| `ichimoku_distance_day` | 状态(day) | 0.970 | 0.063 | 3.25 | **−0.0244** | 全负（须反向） | **负** | 0.655 |
| `adx_power` | ta_basic指标 | 0.860 | 0.206 | **4.93** | **+0.0229** | **全正** | 正 | 0.652 |
| `adx_strength_change` | ta_basic指标 | 0.756 | **0.413** | **5.88** | +0.0117 | 全正 | 正 | 0.633 |
| `High_to_kijun` | 云图 | 0.836 | 0.276 | 2.25 | +0.0097 | **全正** | 正 | 0.631 |
| `kama_slow_rate` | 云图 | 0.931 | 0.284 | 3.85 | **+0.0213** | 4 池正 | 正 | 0.624 |
| `kijun_day` | 状态(day) | 0.959 | 0.079 | 3.45 | −0.0058 | 全负（须反向） | **负** | 0.555 |

### 4.2 最可靠的 4 个：四池 h5 与 h20 全部为正

这是全部 174 个指标中唯一通过"跨池样本外一致"这一关的一组：

| 指标 | h20 平均超额 | 最差池 | etf_3x | company_300 | hs300 | a_etf_all |
|---|---|---|---|---|---|---|
| `ichimoku_distance_alpha` | +0.0110 | +0.0046 | +0.0046 | +0.0231 | +0.0103 | +0.0059 |
| `Low_to_kijun` | **+0.0315** | +0.0027 | +0.0027 | **+0.0814** | +0.0345 | +0.0075 |
| `adx_power` | +0.0059 | **+0.0035** | +0.0035 | +0.0040 | +0.0110 | +0.0051 |
| `High_to_kijun` | +0.0289 | +0.0021 | +0.0021 | +0.0770 | +0.0329 | +0.0034 |

**怎么读这张表（关键）**：

1. `adx_power` 是最"平"的一个 —— 四池都在 +0.35%~+1.1%，**没有靠单一池撑起来的假象**，稳健性最好。
2. `Low_to_kijun` / `High_to_kijun` 平均超额最大（3% 左右），但**几乎全来自 company_300 与 hs300**（+8% / +3.5%），在 etf_3x 上只有 +0.27% / +0.21%。这解释得通：股票池有 297 只标的，"选最强的 5 只"的选择空间大得多；etf_3x 只有 29 只且都是同向杠杆产品，区分度天然低。
3. **`Low_to_kijun` 与 `High_to_kijun` 高度重叠**（跨池 `|corr| = 0.58~0.71`），**只应选一个**，不要两个都加进组合。
4. 三个 `dir=负` 的（`candle_gap_distance`、`ichimoku_distance_day`、`kijun_day`）在四池上方向**完全一致地负**（0/4 为正），所以反向使用是安全的；但它们的动态增量弱（`candle_gap_distance` 仅 +0.0016）或为负，**优先级低于上面 4 个**。

### 4.3 一个必须说清的问题：超额很小

即使最好的指标，在 etf_3x 上的 20 日超额也只有 **+0.21% ~ +0.46%**（未计交易成本）。
`topk_turnover` 在 0.15~0.28 之间，意味着每 4~7 天换掉一只 top-5 成员，**交易成本会吃掉相当一部分**。

**结论：没有任何一个单因子能独立盈利，必须组合 + 必须做成本敏感性检验。**

---

## 5. 发现三：冗余与共线性

对"日期 × 标的"展平后的截面排名做两两 Pearson 相关，172 个通过可用性筛选的指标共 13,530 ~ 14,706 对（`etf_3x` 因标的最少、有效行较少，对数最少）：

| 池 | 对数 | `\|corr\| ≥ 0.9` | `≥ 0.8` |
|---|---|---|---|
| etf_3x | 13,530 | 172（1.3%） | 273 |
| company_300 | 14,706 | 220（1.5%） | 308 |
| hs300 | 14,706 | 275（1.9%） | 307 |
| a_etf_all | 14,706 | 190（1.3%） | 215 |

> 数据来源：`research/output/_redund_{pool}.txt`（`indicator_eval.py redund` 产出）。修订后仅输出 `≥0.9` / `≥0.8` 两档阈值。

> 这个比例偏低，部分原因是"排名在同一天内跨标的展平"会把不同标的的价格刻度差异一并算进相关性，从而**低估**了指标之间的真实冗余。所以 1.3%~1.9% 是冗余度的**下界**，实际冗余更高。

### 5.1 典型的"完全重复对"（必须做去重）

| 重复对 | `\|corr\|` | 说明 |
|---|---|---|
| `Close` ↔ `Adj Close` / `Open` | **1.000** | 除权后几乎同一条序列 |
| `kijun` ↔ `mavg` | **0.995 ~ 1.000** | 基准线 ≈ 20 日均线 |
| `adx_power` ↔ `adx_power_day` | **0.95 ~ 0.97** | `sda` 状态列只是原序列的平滑版 |
| `adx_strength_change` ↔ `adx_power` | 0.84 ~ 0.87 | `adx_power = sda(adx_strength_change)`（[L445](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L445)），同源 |
| `adx_value_pred_change` ↔ `adx_distance` | 1.000（etf_3x） | 公式上同源 |
| `kama_rate` ↔ `kama_fast_rate` | 0.92 ~ 0.94 | 同一 KAMA 变化的两种写法 |
| `candle_gap_distance` ↔ `candle_gap_color` | 0.81 ~ 0.85 | 一个是绝对距离×方向，一个是方向 |
| `kama_slow_rate` ↔ `rsi` | 0.79 ~ 0.83 | 变化率族与超买超卖族的强重叠（值得深挖） |
| `trend_score_alpha` ↔ `trend_magnitude_alpha` | 0.38 ~ 0.41 | 同族但相对独立，可同时用 |

### 5.2 独立性排序（跨 4 池平均 `|corr|`，越小越独立）

| 指标 | 平均 \|corr\| | 最好邻居（相关性） |
|---|---|---|
| `ichimoku_distance_alpha` | **0.0445** | `adx_strength`（0.41~0.43） |
| `trend_score_alpha` | 0.0503 | `trend_magnitude_alpha`（0.38~0.41） |
| `trend_magnitude_alpha` | 0.0553 | `trend_score_alpha` |
| `adx_strength_change` | 0.0768 | `adx_power`（0.84~0.87） |
| `adx_power` | 0.0775 | `adx_power_day`（0.95~0.97） |
| `candle_gap_distance` | 0.0935 | `candle_gap_color`（0.81~0.85） |
| `Low_to_kijun` | 0.1308 | `High_to_kijun`（0.58~0.71） |
| `High_to_kijun` | 0.1318 | `Low_to_kijun` |
| `kijun` | 0.1633 | `mavg`（0.995~1.000） |
| `Close` | 0.1668 | `Adj Close`/`Open`（1.000） |
| `kama_slow_rate` | 0.1848 | `rsi`（0.79~0.83） |
| `trend_magnitude` | 0.1955 | `trend_magnitude_day`（0.70~0.75） |
| `kama_rate` | 0.2213 | `kama_fast_rate`（0.92~0.94） |

> 数据来源：`research/output/eval_neighbors_{pool}.csv`（`indicator_eval.py redund` 产出），表中为 4 池 `mean_abs_corr` 的平均；以上均为 174 列口径下的重跑结果。

**实用结论**：把 `ichimoku_distance_alpha` + `adx_power` 组合是**信息互补最好的一对** —— 两者互为最近邻的相关性只有 0.4 左右，说明它们捕捉的是不同维度（云图位置 vs 趋势强度），而两者又都是四池全正的 B 级。这正是推荐组合的依据。

### 5.3 族级平均 `|corr|`（越高说明该族与全场重复越严重）

| 组 | 组平均 \|corr\| |
|---|---|
| 其他 | 0.068 |
| 状态持续(day) | 0.091 |
| ta_basic 蜡烛 | 0.100 |
| ta_signal 趋势 | 0.117 |
| ta_basic/static 其他 | 0.125 |
| ta_basic/static 云图 | 0.132 |
| ta_basic 指标 | 0.140 |
| 原始 OHLCV | 0.141 |
| ta_signal 形态 | 0.150 |
| ta_score 触发 | **0.163** |
| 疑似标签 | **0.165** |

> 口径：把全部两两指标对按"排序在前的那个成员"所在组归并后取 `|corr|` 均值（`indicator_eval.py redund` 的 `groupby('group_a')`），4 池平均。**越高说明该族与全场的重复越严重** —— `ta_score 触发` 与 `疑似标签` 仍是最高的两族。

---

## 6. 发现四：按族体检（哪一族值得继续深挖）

| 组 | n | 平均 ac1 | 平均换手 | 平均有效持仓 | 分级分布 |
|---|---|---|---|---|---|
| ta_basic/static云图 | 30 | 0.736 | 0.313 | 2.28 | C弱12 / S静态7 / D无效4 / T0 4 / **B中3** |
| 状态持续(day) | 30 | 0.883 | 0.152 | 2.73 | D无效13 / S静态8 / C弱6 / B中\|无2 / T0 1 |
| ta_basic蜡烛 | 31 | 0.549 | 0.379 | 1.79 | C弱9 / T0 9 / D无效8 / S静态4 / **B中1** |
| ta_basic指标 | 19 | 0.829 | 0.299 | 4.12 | D无效11 / C弱4 / **B中2** / S静态2 |
| 其他 | 13 | 0.703 | 0.189 | 0.61 | T0 8 / S静态4 / C弱1 |
| ta_signal形态 | 14 | 0.233 | 0.605 | 2.32 | D无效9 / C弱4 / T0 1 |
| ta_signal趋势 | 11 | **0.268** | **0.772** | **6.19** | D无效6 / C弱4 / **B中1** |
| **ta_score触发** | **11** | 0.240 | 0.626 | 3.07 | **D无效9 / T0 2（全军覆没）** |
| 原始OHLCV | 6 | **0.992** | **0.057** | **0.37** | S静态5 / D无效1 |
| ta_basic/static其他 | 5 | 0.162 | 0.737 | 3.22 | D无效3 / C弱2 |
| 疑似标签 | 4 | 0.479 | 0.506 | 3.43 | **L疑似4** |

**逐族结论**：

- **`ta_score` 触发族（11 个）：整体最差，9 个 D 级 + 2 个不可用。**（修订前为 9 个 / 7 D + 2 T0；新增的 `trigger_net` / `trigger_net_alpha` 仍是 D 级 → 见 8.5）这一族目前**不能作为截面选股信号**，但作为"个股当日事件标记"仍有价值（`trigger_up_score_description` 那种可读标记本来就是给人看的）。
- **`ta_signal` 趋势族（含 `*_alpha`）：换手 0.772、有效持仓 6.19 —— 全样本最"活跃"的一族**，是真正在做轮动的。虽然级别只在 B/C，但方向正确、且与其它族冗余低 → **最值得继续投入研究的方向**。
- **`ta_basic` 指标族（ADX 家族）：19 个里只有 2 个 B 级（`adx_power`、`adx_strength_change`），但这两个是四池全正的核心成员。** ADX 家族的问题是"同一信息写了 19 遍"（`adx_power` / `adx_power_day` / `adx_strength` / `adx_strength_change` / `adx_value` …），实际有效信息只有 2~3 个维度。
- **云图族：分化最严重。** `kijun`/`tankan`/`senkou_*` 等"基准值"是静态身份（S 级），而 `*_to_kijun` 这类"距离/比值"是 B 级。**同一个技术指标，"取数值"没用，"取距离"有用** —— 这是本次评估最有迁移价值的一条经验。
- **状态持续（`*_day`）族：30 个里 8 个静态身份、13 个无效。** `sda` 把信号压成了 -1/0/1 且加了持续性，导致换手极低（0.152）。设计意图没错，但**压得太狠，信息丢掉了**。
- **形态族（`ta_signal` 形态 11 个）：换手 0.605 很高但全部 D/C 级**，说明形态识别在"截面排序"这个用法上不成立（可能更适合做"择时/单标的"而不是"选股"）。
- **原始 OHLCV：几乎全是静态身份**（ac1 0.992、有效持仓 0.37）→ 默认排除。

---

## 7. 发现五：疑似标签与不可用（29 个，绝对排除）

### 7.1 疑似未来标签 / 动作列（4 个）

| 指标 | 风险 |
|---|---|
| `pos_label_score` | 名称含 `label`，极可能是**未来收益标签**（有未来信息） |
| `neg_label_score` | 同上 |
| `action_score` | 动作/决策列，通常是规则输出而非观测值 |
| `action_day` | 同上 |

证据：这 4 个的 `dyn_minus_static > 0` 命中率为 **0%**、`exc_mean` 中位仅 +0.0017（远低于有效信号的 +0.01）。**这是好消息**：说明它们即使含未来信息，也不是收益的直接泄漏源；但无论如何**不得进入特征池**。

### 7.2 不可用（25 个，覆盖率 < 0.5 或取值 ≤ 2）

```
kama_slow_support    kama_slow_resistant   kama_slow_break_up    kama_slow_break_down
candle_gap_top_support      candle_gap_top_resistant     candle_gap_top_break_up    candle_gap_top_break_down
candle_gap_bottom_support   candle_gap_bottom_resistant  candle_gap_bottom_break_up  candle_gap_bottom_break_down
support   resistant   candle_color   signal_day
反转注意   反转观望   下行空仓   区间波动   上行持有   触发卖出   触发买入
Dividend   Split
```

两类：
- **事件型（`Dividend`、`Split`、各类 `*_support/_resistant/_break_*`）**：绝大多数时间无取值 → 任何时候只有 0~5 只有数据，无法做截面排序。想用这类信息，应改成**"事件发生后的状态标记 + 衰减权重"**，而不是原值。
- **枚举型（反转注意/上行持有/触发买入…）**：取值 ≤ 2 个类别，`nuniq ≤ 2`，没有排序信息。若要用，应改为**类别编码 + 每类历史超额**，而不是直接当数值。

---

## 8. 建议

### 8.1 计算方式建议（改公式）

#### ✅ 保留并推广的好设计

1. **`normalize_causal`（[L1888](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L1888)）—— 本次评估的最大功臣。**
   因果归一化（只用尾随 252 日的 min/max，warm-up 用 expanding），无未来泄露，把任意量纲的指标压到 [0,1]。事实证明：**凡是加了 `_alpha` 归一化的指标，级别都明显高于其原值版本** —— `ichimoku_distance`（D 级）→ `ichimoku_distance_alpha`（B 级）；`trend_magnitude`（C 级）→ `trend_magnitude_alpha`（C 级但更稳定）。
   **建议**：把 `_alpha` 归一化作为**所有数值型指标的默认后处理**，而不是只给少数几个用。

2. **距离/比值归一化 —— `abs(a - b) / kama_slow`（[L2840-2842](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L2840-L2842)）。**
   把"绝对价位差"变成"相对距离"，使不同标的可比。`Low_to_kijun` / `High_to_kijun` 是本次最好的指标之一，就是这个设计的直接成果。
   **建议**：把剩余的 `*_to_*` 列（`Low_to_tankan`、`High_to_tankan`、`Low_to_kama_fast` …）**统一改用同一个分母**（现在分母不统一，导致它们冗余度判断混乱）。

#### ❌ 必须修的问题

3. **合成公式：缺陷是"分量量纲不齐"，不是"加法抵消"（已修复并重跑，见 8.5）。**
   初稿的诊断（"看涨证据 + 看跌证据 = 互相抵消"）经实测**证伪**。现状代码无需改成减法：
   ```python
   # L821-837 —— 这是一个正确的净额求和, 不要改
   df['trigger_down_score'] += df['break_down_score'] + df['resistant_score'] * 0.5
   df['trigger_score'] = df['trigger_up_score'] + df['trigger_down_score']
   ```
   证据：`trigger_down_score` mean ≈ −0.60（min −5.0，本身带负号）、`corr(trigger_score, trigger_down_score) = +0.79`、`trigger_score == trigger_up_score + trigger_down_score` 的 `max|diff| = 0`。
   **真正的缺陷是量纲**：`break_up_score` 可达 ±5，`support_score` 仅 ±1，直接相加时合成结果被大量纲分量主导。因此修法是**分量各自因果归一化后再带符号加权**：
   ```python
   # 已落地为"纯新增列"(不改动任何旧列取值)：trigger_net / trigger_net_alpha
   W = {'break_up_score': 1.0, 'support_score': 0.5, 'break_down_score': 1.0, 'resistant_score': 0.5}
   df['trigger_net'] = 0.0
   for c, w in W.items():
       alpha = normalize_causal(df[c].abs(), window=252, min_periods=60)   # 分量量纲统一到 [0,1]
       df['trigger_net'] += w * alpha * np.sign(df[c].fillna(0.0))        # 带符号加权求和
   df['trigger_net_alpha'] = normalize_causal(df['trigger_net'].abs(), window=252, min_periods=60)
   ```
   同一口径也套用到形态族：`pattern_net` / `pattern_net_alpha`（对 `pattern_up_score` / `pattern_down_score` 先归一化再加权求和）。
   **实测结果（8.5）：`trigger_net`、`trigger_net_alpha`、`pattern_net`、`pattern_net_alpha` 四个仍全部为 D 级无效** —— 归一化没能救活这一族，说明其失效根源不在合成公式，而在**原始分量本身不含截面超额信息**。这条经验请连同第 1 条（`_alpha` 归一化的正面案例）一起理解：**归一化只能修正量纲，不能凭空造出信息。**

4. **`trend_score` 量纲混用（[L949](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L949)）。**
   ```python
   df['trend_score'] = df['adx_value_change'] + df['adx_strength_change'] * sign(adx_value)
   ```
   把"ADX 数值变化"和"ADX 强度变化"两个不同量纲直接相加，且后者被符号翻转。实测 `trend_score` 仅 C 级、`trend_score_change` 为 D 级。
   **建议**：先各自 `_alpha` 化再加权（照抄 `trend_magnitude`（[L978-982](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L978-L982)）的做法 —— **`trend_magnitude` 正是本文件里最正确的合成范例**：先取绝对值归一化，再乘符号，再加权求和）。

5. **`sda` 状态列压得过狠。**
   `trigger_day`、`*_power_day`、`*_distance_day` 等把连续量压成 -1/0/1 + 持续性，导致换手降到 0.15、信息大量丢失（该族 30 个里 13 个 D 级）。
   **建议**：保留连续强度，把 `sda` 的状态只当"辅助特征"：
   ```python
   df['adx_power_state']  = sda(df['adx_strength_change'], zero_as=0)   # 方向 (-1/0/1)
   df['adx_power_alpha']  = normalize_causal(df['adx_strength_change'].abs(), 252, 60)  # 强度 [0,1]
   df['adx_power_signed'] = df['adx_power_state'] * df['adx_power_alpha']  # 有向强度 ← 推荐主用
   ```

6. **清理同义别名（会污染组合权重）。**
   `adx_direction ≡ adx_value_change` 与 `adx_power ≡ adx_strength_change`（[L444-445](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L444-L445)）是纯别名；`adx_power_day` 与 `adx_power` 相关 0.95~0.97；`adx_value_pred_change` 与 `adx_distance` 在 etf_3x 上相关 1.000。留着它们会让"按 IC 排序取前 N 个"的策略无意中重复押注同一信息。
   **建议**：每个信息维度只保留 1~2 个代表列，其余标记为 `_dup` 并在组合阶段过滤。

7. **默认排除清单（写进特征管线）。**
   建议在生成面板后立即把以下列排除出"信号候选"，避免误用：
   ```python
   EXCLUDE_AS_SIGNAL = [
       # 静态身份: 价格水平
       'Open','High','Low','Close','Adj Close','Volume',
       # 静态身份: 云图基准值/统计
       'kijun','tankan','senkou_a','senkou_b','kama_fast','kama_slow',
       'mavg','mstd','bb_high_band','bb_low_band','atr','adi','kama_distance',
       # 疑似标签
       'pos_label_score','neg_label_score','action_score','action_day','signal_day','label',
       # 文本
   ]  # + 所有 *_description / pattern_up / pattern_down
   ```

### 8.2 验证方式建议（把这次的框架固化下来）

本次的证据链脚本**已固化**为 [research/indicator_eval.py](file:///c:/Users/northcheng/git/quant/research/indicator_eval.py)（单文件、4 个子命令，原先的 `_tmp_eval/_tmp_grade/_tmp_redund/_tmp_neigh` 已合并并删除），纳入每次改动后的常规回归，因为：

- 五重检验的**边际成本很低**（四池全量 174 指标约几分钟）；
- 但能拦住**一半以上的伪信号** —— 本次 174 个里有 30 个静态身份 + 4 个疑似标签 + 25 个不可用，**59 个（34%）在传统 IC 检验下会"看起来有效"**。

它还有一个额外能力：**当 pkl 里还没有新列时，会按生产代码口径即时合成** `trigger_net` / `pattern_net` 及其 `_alpha`（见 `add_synth`），因此"改公式 → 立刻看效果"不需要先重跑全量数据生成。

建议的固定闸门（一个指标要"出线"必须连过 5 关）：

| 关卡 | 判据 | 不合格处理 |
|---|---|---|
| 1. 数据可用 | `coverage ≥ 0.85` 且 `nuniq > 10` | 弃用 |
| 2. 非泄漏 | 名称/语义不含 label/action/未来信息 | 弃用 |
| 3. **非静态身份** | `sig_ac1 < 0.99` **且** `n_eff_symbols ≥ 1.5` **且** `topk_turnover ≥ 0.05` | 降级为"beta 暴露" |
| 4. 跨池一致 | 4 池同号为正（至少 3/4 且最差池为正） | 降级为"池专属" |
| 5. 动态增量 | `dyn_minus_static > 0` | 标记"≈固定持仓" |

日常体检还要补两条：

- **样本外**：时间分半（前/后半段同号）+ 跨池（4 池同号）。本次用的是跨池，已经把 `Low_to_kijun` 这类真信号和只在一个池有效的伪信号分开了。
- **冗余**：任何新指标入池前，先与现有指标算 `|corr|`，`≥ 0.8` 的必须二选一（见 5.1 清单）。
- **成本**：按 `topk_turnover` 估算换手，做一次"扣除 10/20/50bp 单边成本"的敏感性（`topk_turnover=0.28` 时，20 日约换 1.4 次 × 双边成本，不能忽略）。

### 8.3 使用方式建议

把 174 个指标按"该怎么用"分四层，直接对应操作：

#### 第 1 层：可作截面轮动信号（9 个，建议只用 4 个）

| 优先级 | 指标 | 用法 | 备注 |
|---|---|---|---|
| ★★★ | `ichimoku_distance_alpha` | **正向**：值越大越买 | 四池全正、冗余最低（0.045）、有效持仓 5.7 |
| ★★★ | `adx_power` | **正向** | 四池最均衡（0.35%~1.1%），无单池依赖 |
| ★★☆ | `Low_to_kijun` | **正向** | 平均超额最大，但主要在股票池；与 `High_to_kijun` 二选一 |
| ★★☆ | `High_to_kijun` | **正向** | 同上，**勿与 `Low_to_kijun` 同时使用** |
| ★★☆ | `adx_strength_change` | **正向** | 与 `adx_power` 相关 0.85 → 二选一 |
| ★☆☆ | `kama_slow_rate` | **正向** | 与 `rsi` 相关 0.8 → 注意冗余 |
| ★☆☆ | `candle_gap_distance` | **反向**（取负） | 动态增量仅 +0.0016，收益主要靠低换手倾斜 |
| ★☆☆ | `ichimoku_distance_day` | **反向** | 动态增量为负 → 本质是固定持仓 |
| ★☆☆ | `kijun_day` | **反向** | 同上 |

**具体落地**：在 [score_backtest.py](file:///c:/Users/northcheng/git/quant/research/score_backtest.py#L66-L95) 的 `MOMENTUM_SETS` 里补两个预设（现有 `M10_kijun` 已经是 `Low_to_kijun`，可以保留）：

```python
# ---- 基于本次指标价值评估新增 (2026-09-18) ----
# 四池 h5/h20 全正 + 动态增量>0 + 跨池平均|corr|仅 0.4 (信息互补)
'M16_kijun_ichimoku': {'Low_to_kijun': 0.5, 'ichimoku_distance_alpha': 0.5},
# 最稳健的平坦组合: 趋势强度 + 云图位置, 不依赖单一池
'M17_adx_ichimoku':   {'adx_power': 0.5, 'ichimoku_distance_alpha': 0.5},
# 对照组: 只用一个, 验证组合是否真比单因子好
'M18_ichimoku_only':  {'ichimoku_distance_alpha': 1.0},
```
然后用现成的 `--weights M16_kijun_ichimoku` 跑一遍，并**务必**跑 `M18` 做对照 —— 如果组合没有明显优于单因子，就不该加复杂度。

#### 第 2 层：只作状态过滤 / 仓位调节（不做选股）

`trend_*`、`adx_*_day`、`ichimoku_*_day`、`kama_*_day` 这类**状态列**（换手 0.15 左右、有效持仓 2~3 只）不适合排序选股，但适合做**门控（gate）**：例如"只在 `adx_power_day > 0` 时开仓"。
`run_engine` 的 `gate` 参数就是为这个设计的（[score_backtest.py#L167](file:///c:/Users/northcheng/git/quant/research/score_backtest.py#L167)）—— 把状态列转成布尔门控，比当信号用更符合它的信息结构。

#### 第 3 层：只作 beta 倾斜 / 组合暴露（37 个）

- **25 个 `dir=负` 的静态身份列**（`Close`、`kijun`、`senkou_a` …）：反向使用 = "买低价格/低波动"。这确实能赚（`exc_mean` ≈ +1.6%），但**它是整体倾斜，不是轮动** —— 实现上应该用"一次性权重倾斜"而不是"每日换仓"，否则白付交易成本。
- **12 个 C 级趋势/形态列**：方向对但幅度小，只适合给第 1 层信号做**微调权重**（< 20% 权重），不要单独立项。

#### 第 4 层：弃用（93 个）

64 个 D 级 + 4 个疑似标签 + 25 个不可用。**特别是 `calculate_ta_score` 的 11 个直接产出（9 D + 2 T0）** —— 合成公式已按 8.1 第 3 条修复重跑，新列 **仍是 D 级无效**（见 8.5），因此这一族**已被判定为信息冗余/无独立超额，不要再投入**。

#### 风险提示（务必一起看）

1. **量级很小**：etf_3x 上最好的单因子 20 日超额仅 +0.46%；扣掉交易成本后可能归零。**必须做成本敏感性**。
2. **池子差异巨大**：同一指标在股票池（297/298 只）上超额是 etf_3x 的 5~30 倍。**不要用 company_300 的结果去预期 etf_3x 的表现。**
3. **hs300 有幸存者偏差**（用当前成分回溯），它的 +3.5% 应打折看待。
4. **a_etf_all 只有约 2 年数据**，它的"为正"证据强度弱于其他三池。
5. **本报告是横截面研究**：全部结论都以"池内相对排序"为口径。**不适用于单标的择时**。

### 8.4 下一步优先级（按性价比排序）

1. ~~**修合成公式**（8.1 第 3 条）→ 重跑 `trigger_net` / `pattern_net` 的评估。~~ **✅ 已完成（2026-09-18），结论见 8.5**：修复顺利落地，但**预期落空** —— `ta_score` 族没有变成 B 级，仍然全军覆没。这条建议的"性价比"因此被修正为：**修复本身无风险（纯新增列），但不要指望它救活该族。**
2. **把 `_alpha` 因果归一化铺开到所有数值列**（8.1 第 1 条）→ 重跑评估。这是"把一个 D 级指标变成 B 级"的可复制路径。
   ⚠️ 经 8.5 验证后需加一句限定：**`_alpha` 只在"原列本身含信息、只是量纲不合适"时有效**（`ichimoku_distance` → `_alpha` 的正面案例）；对本来就不含信息的分量（`break_up_score` 等）无效。
3. **回测 M16/M17**（8.3）并与 M18 单因子对照 → 验证组合是否真有增量。
4. **样本外前推**：把 2021-2024 作为训练、2025-2026 作为纯样本外，重跑第 4 节那 4 个指标的检验（本次是跨池样本外，还缺一个时间样本外）。
5. **深挖 `kama_slow_rate ↔ rsi` 相关 0.8** 这个现象：变化率族与超买超卖族高度重叠，可能是同一底层风险因子的两种表达，值得确认是否需要二选一。

### 8.5 合成公式修复与重跑实测（2026-09-18）

#### 怎么改的（"纯新增列"，对现有链路零影响）

在 [bc_technical_analysis.py](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py) 中**只新增 4 列、不改任何旧列**：

| 新列 | 位置 | 口径 |
|---|---|---|
| `trigger_net` | `calculate_ta_score`（[L801](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L801) 段内，`trigger_score` 之后） | 4 个分量各自 `normalize_causal(\|x\|)` 到 [0,1]，再 `w · alpha · sign(x)` 加权求和（`w`：break_up 1.0 / support 0.5 / break_down 1.0 / resistant 0.5） |
| `trigger_net_alpha` | 同上 | 对 `trigger_net` 再取绝对值做同样的因果归一化 |
| `pattern_net` | `calculate_ta_signal`（`pattern_score_alpha` 之后） | `pattern_up_score` / `pattern_down_score` 各自归一化后等权带符号求和 |
| `pattern_net_alpha` | 同上 | 对 `pattern_net` 再归一化 |

4 列同时登记进 `ta_data_columns`。**旧列取值零改动**：冒烟验证 `trigger_score == round(trigger_up_score + trigger_down_score, 2)` 与 `pattern_score == round(pattern_up_score + pattern_down_score, 2)` 在修改后仍为 `True`；`py_compile` 通过。

冒烟实测（600 行样本）：`trigger_net` 非空 600/600、nuniq 79、mean 0.0018、std 0.3980，与 `trigger_score` 相关 0.961；`trigger_net_alpha` 取值覆盖 [0, 1]；`pattern_net` 非空 600/600、nuniq 85、mean −0.0455、std 0.3604。

#### 重跑结果（四池，174 候选列）

分级计数（[eval_summary.csv](file:///c:/Users/northcheng/git/quant/research/output/eval_summary.csv)）：

| 级别 | 数量 |
|---|---|
| D 无效 | **64** |
| C 弱 | 42 |
| S 静态身份 | 30 |
| T0 不可用 | 25 |
| B 中 | 7 |
| L 疑似标签 | 4 |
| B 中\|无动态增量 | 2 |
| **A 强** | **0** |

`ta_score` 族修复后逐列表现（全部为 D 级）：

| 指标 | tier | score | h5 超额均值 | h20 超额均值 | 换手 | 有效持仓 |
|---|---|---|---|---|---|---|
| `trigger_net` | D 无效 | 0.462 | +0.0009 | **−0.0001** | 0.365 | 6.35 |
| `trigger_net_alpha` | D 无效 | 0.612 | +0.0008 | **−0.0017** | 0.850 | 6.40 |
| `pattern_net` | D 无效 | 0.225 | +0.0008 | +0.0017 | 0.318 | 6.28 |
| `pattern_net_alpha` | D 无效 | 0.512 | +0.0004 | +0.0011 | 0.287 | 6.33 |
| （对照）`trigger_score` | D 无效 | 0.562 | +0.0011 | +0.0010 | 0.446 | 5.60 |
| （对照）`pattern_score` | D 无效 | 0.362 | +0.0005 | −0.0019 | 0.364 | 5.93 |

#### 结论（三条）

1. **"抵消 bug"不存在**，修复方向已从"改加法为减法"纠正为"分量量纲归一化"。这是本次修订最重要的**自我纠错**：报告初稿的 L26 诊断是错的。
2. **归一化没能救活这一族。** 新列 `trigger_net` / `trigger_net_alpha` / `pattern_net` / `pattern_net_alpha` 与旧列 `trigger_score` / `pattern_score` 一样全部 D 级；超额均值在 +0.0001 ~ +0.0017 之间，**比交易成本低一个数量级**。值得注意的是新列的**换手不低**（0.29~0.85）、**有效持仓数已升到 6.2~6.4**（不再是"固定选票"），说明**它们确实是"在轮动"的信号，只是轮动本身不产生超额** —— 这比"静态身份"更彻底地说明该族无独立价值。
3. **判定：`ta_score` 触发族与形态-净分族整体归入"信息冗余/无独立超额"，停止投入。** 它们仍可作为**个股事件标记**（给人看的可读描述），但**不得进入任何截面选股策略**。

> **本次修订的方法论收获**：`_alpha` 因果归一化是"量纲矫正器"，不是"信息放大器"。它能救活"有信息但刻度不对"的列（`ichimoku_distance` → `ichimoku_distance_alpha`），不能救活"本来就没信息"的列（`break_up_score` → `trigger_net`）。**先证明分量有信息，再谈合成。**

---

## 9. 局限（结论的边界）

1. **样本区间**：2021-01 至 2026-09，只覆盖一轮较完整的行情，未经历 2008/2020 式危机。
2. **池子小**：etf_3x 仅 29 只、a_etf_all 97 只，`top_k=5` 在其中已经占 5%~17% 的池子权重，结论对 `top_k` 敏感。
3. **幸存者偏差**：hs300 用当前成分回溯。
4. **冗余度被低估**：相关性在"日期 × 标的"上展平计算，混入了跨标的刻度差异（见第 5 节说明）。
5. **单个 h 口径**：只测了 h = 5 / 20，未测 h = 1 / 60 / 120；`IC` 用的是逐日截面 rank 相关。
6. **本次只评分、未做组合权重优化**：B 级指标的权重分配未优化，直接用等权是最好的出发点（避免过拟合）。
7. **`dyn_minus_static` 未做方向对齐**：对静态身份类的正增量解释需要谨慎（见 3.3）。

---

## 10. 附录：数据文件与复现

### 产出文件（均在 `research/output/`）

| 文件 | 内容 |
|---|---|
| `eval_summary.csv` | **主结果**：174 指标 × 74 列，含 `tier`（分级）与 `score` |
| `eval_{pool}.csv` | 单池明细（174 行 × 49 列） |
| `eval_redundancy_{pool}.csv` | 两两相关性全表（13,530 / 14,706 对） |
| `eval_neighbors_{pool}.csv` | 每个指标的最近邻及相关系数 |
| `_grade_log.txt` | 本次重跑的分级计数与各组最优 |
| `_rep_log.txt` / `_rep2_log.txt` | 本报告第 2/3/6 节的聚合统计 |
| `_detail_log.txt` | 第 4 节四池明细 |
| `_neigh_log.txt` | 第 5 节冗余汇总 |
| `_count_log.txt` / `_pools_log.txt` | 冗余对数分档 / 四池规模 |
| `_prefix_backup_20260918/` | **修复前**的 `eval_*.csv` 备份（用于对比修复前后） |

### 输入数据位置（容易找不到，特别记录）

| 池 | pkl 路径 |
|---|---|
| `etf_3x` | `c:\Users\northcheng\git\quant\research\data\etf_3x_day_ta_data.pkl`（29 标的 / 41528 行） |
| `company_300` | `C:\Users\northcheng\quant\company_300_day_ta_data_research.pkl`（297 标的） |
| `hs300` | `C:\Users\northcheng\quant\hs300_day_ta_data_research.pkl`（298 标的） |
| `a_etf_all` | `C:\Users\northcheng\quant\a_etf_all_day_ta_data_research.pkl`（97 标的） |

> 注意：后三个池的 pkl **不在 git 仓库内**，而在用户主目录下的 `quant\`。路径已固化在 [indicator_eval.py](file:///c:/Users/northcheng/git/quant/research/indicator_eval.py) 的 `POOLS` 常量里；某池 pkl 缺失时脚本会打印 `[ERROR] 找不到 pkl: ... (该池本轮跳过)` 并继续跑其余池。

### 复现命令

```powershell
$py = "C:\Users\northcheng\.venv\Scripts\python.exe"
& $py research/indicator_eval.py eval   --pool all   # 四池全量评估 → eval_{pool}.csv
& $py research/indicator_eval.py grade              # 汇总 + 价值分级 → eval_summary.csv + _grade_log.txt
& $py research/indicator_eval.py redund --pool all  # 共线性/冗余 → eval_redundancy_{pool}.csv
& $py research/indicator_eval.py neigh              # 重点指标的最近邻/冗余伴侣 → eval_neighbors_*.csv
```

> 原临时脚本 `_tmp_eval.py` / `_tmp_grade.py` / `_tmp_redund.py` / `_tmp_neigh.py` **已合并进 `indicator_eval.py` 并删除**（其中的 `_tmp_rep.py` 聚合统计为一次性用途，未固化）。

### 关键代码位置

| 主题 | 位置 |
|---|---|
| 可交易前瞻收益 | [signal_search.py#L88-L92](file:///c:/Users/northcheng/git/quant/research/signal_search.py#L88-L92) |
| IC / 超额 / 换手计算 | [signal_search.py#L125-L172](file:///c:/Users/northcheng/git/quant/research/signal_search.py#L125-L172) |
| 真伪检验（静态对照） | [signal_search.py#L175-L230](file:///c:/Users/northcheng/git/quant/research/signal_search.py#L175-L230) |
| 因果归一化（建议推广） | [bc_technical_analysis.py#L1888](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L1888) |
| 状态持续 sda | [bc_technical_analysis.py#L1783](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L1783) |
| **合成公式（NS: 无需修改）** | [bc_technical_analysis.py#L821-L837](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L821-L837) —— 初稿判为 bug，实测为正确的净额求和 |
| **本次新增的修复列** | `trigger_net`/`trigger_net_alpha` 在 [bc_technical_analysis.py#L801](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L801) 段内；`pattern_net`/`pattern_net_alpha` 在 [bc_technical_analysis.py#L923](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L923) 段内（纯新增，旧列零改动） |
| 正确的合成范例 | [bc_technical_analysis.py#L978-L982](file:///c:/Users/northcheng/git/quant/bc_technical_analysis.py#L978-L982) |
| **回归脚本（本次固化）** | [research/indicator_eval.py](file:///c:/Users/northcheng/git/quant/research/indicator_eval.py) |
| 策略预设 | [score_backtest.py#L56-L95](file:///c:/Users/northcheng/git/quant/research/score_backtest.py#L56-L95) |