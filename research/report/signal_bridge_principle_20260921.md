# signal_bridge 信号计算原理(可解释性研究)

> 日期: 2026-09-21
> 问题: 从可解释性角度, signal_bridge 计算信号的原理是什么? 为什么要买这些股票?
> 证据来源: 源码逐行核实(行号以当日为准) + venv pandas 语义实测 + 此前回测实测数据

## 0. 一句话答案

company_300 桥每天买的是: 全池里「过去一年涨得最多」且「近 20 天走得最顺滑」综合排名前 12 的股票。
它**不预测任何股票的目标价**, 只做当日截面内的相对排名; 某只票排名弱了(跌出 top_k)就换掉。

## 1. 五步计算链(company_300 口径)

| 步骤 | 名称 | 公式 | 代码位置 |
|---|---|---|---|
| ① | F_mom121(12-1 动量) | `P(21天前) / P(252天前) − 1` | research/factor_mining.py L162 |
| ② | F_er20(趋势效率) | `|P − P(20天前)| / Σ|每日涨跌|` | research/factor_mining.py L88-90 |
| ③ | C_tmqmom(趋势质量动量) | `rank(F_mom121) + rank(F_er20)`(截面百分位) | research/alpha_mining.py L163 |
| ④ | composite 加权 | `0.5·rank(C_tmqmom) + 0.5·rank(F_er20)` | signal_bridge.py L227-238 |
| ⑤ | 截面 rank 三态 | `rank ≤ 12 → b`, `rank > 12 → s`, 中间 hold | signal_bridge.py L264-269 |

权重来源: `POOL_PRESETS['company_300'] = {'weights': {'C_tmqmom': 1.0, 'F_er20': 1.0}, 'top_k': 12}`(signal_bridge.py L82)。

## 2. 逐因子解释(为什么买)

### 2.1 F_mom121 — 涨得多, 但故意不看最近一个月

`close.shift(21) / close.shift(252) - 1.0`(factor_mining.py L162, 注释即「12-1 动量(剥离近月反转)」)。
分子分母同时前移 21 天, 度量的是「21 天前 ~ 252 天前」这 11 个月的涨幅, 完全绕开最近一个月。

**为什么剥离近月**: 学术界经典 12-1 动量(Jegadeesh & Titman)。近月刚大涨的票常伴随短期
反转回调, 直接用 12 个月涨幅容易买进「刚冲完顶」的票。往前挪一个月, 买的是
「涨过一轮、且已消化至少三周」的强势票。

### 2.2 F_er20 — 走得直: 同样涨 20%, 两种命运

`(close - close.shift(20)).abs() / close.diff().abs().rolling(20).sum()`(factor_mining.py L88-90),
即 Kaufman 效率比(ER): **净位移 ÷ 总路程**。

- 直线路径: 每天稳定涨 1%, 20 天净涨 20%, 总路程也约 20% → ER ≈ 1.0
- 锯齿路径: 涨 10% 跌 8% 再涨, 最终净涨 20%, 但总路程 60%+ → ER ≈ 0.3

同样 20% 涨幅, 前者是持续的资金共识, 后者是多空反复搏杀。桥要的是前者 —
趋势延续的摩擦最小。分母为 0 时(20 日零波动)置 NaN, 不参与排名。

### 2.3 C_tmqmom — 双 rank 相加 = 稳健合议

`rk(mined['F_mom121']) + rk(mined['F_er20'])`(alpha_mining.py L163, rk = 截面 pct rank)。
用截面百分位而非原始值相加: 只看位次不看幅度, 单因子第 1 名和第 30 名的 rank 差距极小。
这是有意为之的「中庸」— 不被极端值绑架, 两个维度都得靠前的票才冒头。

### 2.4 composite 与 F_er20 的二次加权 — 三份权重里两份是趋势质量

composite = `0.5·rank(C_tmqmom) + 0.5·rank(F_er20)`(signal_bridge.py L227-238)。
C_tmqmom 内部已含一份 rank(F_er20), 外层又给 F_er20 独立 1.0 权重, 展开后约为:

| 成分 | 有效权重 |
|---|---|
| 涨得多(动量) | ≈ 1/3 |
| 走得直(趋势质量) | ≈ 2/3 |

**策略真实偏好: 宁要「持续稳步上涨」, 不要「暴涨后横盘」** — 把仓位押在回撤控制上, 而非弹性上。

### 2.5 三态与「弱了就换」 — 相对排名的本质

signal_bridge.py L267: `state = 'b' if rk <= top_k else ('s' if rk > exit_rank else None)`;
hold 不进名单 = 维持现状。exit_rank 默认 12(DEFAULT_EXIT_RANK, L96), company_300 的
top_k 也是 12 → 买入/卖出阈值重合, 无缓冲带。

它从不预测绝对收益: 熊市里只要某 12 只票在全池「相对最强」就照买, 赌的是截面强者恒强,
不是大盘方向。代价是机械换手 — 任何一只票从第 12 滑到第 13 就触发换仓, 无预测性止盈止损:
- company_300 实测: 年换手 161.6x, 平均持仓约 3.4 天
- hs300 实测: 年换手 127.7x(弱点之一, 2026-07 单月 -14.8%)

## 3. composite 的 pandas NaN 精确语义(venv 实测)

signal_bridge.py L228-236 的口径:

```python
for s, w in weights.items():
    pct = cands[s].rank(axis=1, pct=True)      # 当日截面百分位, NaN 保留
    part = (w / total) * pct
    comp = part if comp is None else comp.add(part, fill_value=0.0)
    m = cands[s].notna()
    cover = m if cover is None else (cover | m)
agg = comp.fillna(0.5)
```

venv(3.x + pandas)实测结论:

```python
a = pd.DataFrame({'x': [np.nan, 1.0]}); b = pd.DataFrame({'x': [np.nan, 2.0]})
a.add(b, fill_value=0.0)      # -> [nan, 3.0]   双 NaN 相加仍为 NaN
0.5 * pd.DataFrame({'x': [np.nan, 0.8]})   # -> [nan, 0.4]  标量×NaN 仍为 NaN
```

即 `add(fill_value=0.0)` 的语义是: **部分缺失的成分按 0 参与相加, 全成分缺失才保留 NaN**,
再由 `fillna(0.5)` 统一填中性 0.5。这不是「把 NaN 全变 0」。

含义: 一只票只要有任何一个成分因子有值, 就会得到一个偏低(但非中性)的综合分;
只有当日完全无数据的票才是 0.5 中性分。rank 会把这些「部分缺失偏低分」和「中性 0.5」
一起参与排名 — 这是幻影名单的根源之一(见 §6)。

## 4. 决策截面选择(signal_bridge.py L240-261)

- 盘前模式(trade_date 晚于 pkl 最新数据日): 从后往前第一个**覆盖 ≥ 50%** 的截面, 用最新信息。
  原因: 个别标的当日数据更新延迟时, 直接取最后一行会产生小样本噪声名单。
- 历史模式(trade_date ≤ 最新数据日, 复盘用): 交易日之前最后一个覆盖 ≥ 50% 的截面, **防前视**,
  与 run_engine 的 s=dates[i-1] 一致(信号日收盘决策 → 交易日开盘执行)。
- gap 检查: 决策截面与交易日间隔超过 MAX_DATA_GAP=4 个自然日(含周末/长假)即报数据陈旧退出。

## 5. 全部池配置(signal_bridge.py L81-95)

| 池 | 权重 | top_k | 候选构建 |
|---|---|---|---|
| company_300 | C_tmqmom:1.0 + F_er20:1.0 | 12 | build_alpha_cands(美股) |
| company_1000 | G_vr20:1.0 + H_ichimoku_alpha:1.0 | 8 | build_alpha_cands; 无生产 pkl, 仅复盘 |
| etf_3x | F_mom121:0.5 + F_er20:0.3 + N_idiovol60:0.2 + F_obv20:0.1 | 5 | build_alpha_cands |
| hs300 | N_range20:-1.0 + D_ma20_dist:-1.0 + C_tmqmom:0.5 | 5 | build_a_combo_cands(A股) |
| a_etf_all | N_range20:+1.0 + G_oviv20:0.5 | 5 | build_a_combo_cands(A股) |

注意:
- A 股两池是 a_combo_search 20260920 终验 + is2023 切分复验的胜出配置, exit_rank=12。
- N_range20 两池方向**相反**(hs300 负权买高振幅波动反转 / a_etf_all 正权买低振幅防御),
  生产必须独立参数化。
- G_vr20 不在候选内, signal_bridge 本地补建(L221-223, 与 combo_search 同公式)。
- 剔除 DJT(无完整历史标的, L97), 保持与回测剔除口径一致。

## 6. 已知局限 / 风险

1. **fillna(0.5) 幻影病理**: 全成分缺失的票以中性 0.5 参与排名, 停牌/新股可能以中性分
   挤到名单边界 — hs300 top3 幻影已证实(此前研究)。部分缺失票则拿到偏低分, 同样扭曲位次。
2. **rank 对池子成分变化敏感**: 指数调样、退市、数据覆盖变化会让排名整体跳变。
3. **极高换手**: company_300 161.6x/年、hs300 127.7x/年, 实盘摩擦成本敏感。
4. **top_k 与 exit_rank 重合(均为 12)**: 无缓冲带, 边界票反复进出加剧换手
   (hs300 弱点, 待决策: 调 exit_rank/top_k)。
5. **company_1000 无生产 pkl**: 研究版 pkl 滞后 1~2 天, 仅复盘用。

## 7. 源码位置索引

| 内容 | 文件 | 行号 |
|---|---|---|
| POOL_PRESETS / A_POOL_PRESETS / exit_rank / gap | c:/Users/northcheng/signal_bridge.py(家目录, 非 git/quant 内) | L78-98 |
| composite 加权计算 + 覆盖 + fillna(0.5) + rank | 同上 | L218-238 |
| 决策截面选择(盘前/历史/防前视/gap) | 同上 | L240-261 |
| 三态 b/hold/s -> 名单 | 同上 | L263-271 |
| F_er20 / F_r2 / F_slopet 定义 | git/quant/research/factor_mining.py | L87-94 |
| F_mom121 / F_momaccel / F_momdd120 | 同上 | L161-166 |
| C_tmqmom / C_qmom60(截面秩组合) | git/quant/research/alpha_mining.py | L160-163 |

## 8. 相关实测数据(此前任务)

- hs300 桥口径 2026 年以来回测(桥口径 = rank_pct 加权 + fillna(0.5) + 覆盖门槛):
  策略 +10.4% vs 池等权基准 -5.3%(research/signal_timeseries.py, --trades-csv 可出逐笔标注)。
- 每笔交易的个股时序图核验已做, 结论: 换手集中在排名 12/13 边界的反复进出。
