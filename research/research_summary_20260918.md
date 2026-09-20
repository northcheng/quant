# 研究日总结：信号挖掘全链路（2026-09-18）

> 时间范围：2026-09-18 09:00 ~ 2026-09-19 00:15（跨夜，全文"今日"指 9/18）
> 项目仓库：`c:\Users\northcheng\git\quant`
> 数据目录：`C:\Users\northcheng\quant`（仓库外）
> 当日相关文档：[ic_research_review_20260918.md](ic_research_review_20260918.md) / [indicator_value_review_20260918.md](indicator_value_review_20260918.md)
> 最终产出：`research\output\alpha_mine_20260919_000017\`（48 候选 × 4 池 × h=5/20/60）

本文是工作台速查手册 + 当日研究复盘，供后续研究随时查阅。核心规则先看第二节（pkl 双轨制，最重要）。

---

## 一、Python 环境与运行惯例

| 项 | 值 |
|---|---|
| 解释器 | `C:\Users\northcheng\.venv\Scripts\python.exe`（Python 3.12） |
| 主要依赖 | pandas / numpy（research 脚本栈仅用这两个） |
| 运行目录约定 | **cwd 必须先切到 `c:\Users\northcheng\git\quant\research\`**，再 `python 脚本名.py`（脚本间 `import` 依赖此 cwd；写绝对 research\ 前缀会路径翻倍报错） |
| 输出位置 | 一律写入 `research\output\`（脚本自动建目录） |
两个券商平台的API分别是: 老虎: 两个券商平台的API分别是: 老虎: , 富途: , 富途: 两个券商平台的API分别是: 老虎: , 富途: 

典型运行方式：

```powershell
cd C:\Users\northcheng\git\quant\research
python alpha_mining.py --pools etf_3x,company_300,hs300,a_etf_all
```

坑（今日踩过）：
- **不要用 `python -c` 内嵌 f-string**——PowerShell 引号解析会报 `Unexpected token`。临时分析请写成 `research\_tmp_analysis.py` 之类的临时文件再执行，用完删除。
- `python research\xxx.py` 在 cwd 已是 research\ 时会路径翻倍，直接 `python xxx.py`。

---

## 二、pkl 数据文件规则（最重要：优先用 XXX_research.pkl）

### 2.1 位置与双轨制

所有池数据 pkl 在 **`C:\Users\northcheng\quant\`**（注意：不是 git 仓库内目录）：

| 池 | 生产版（勿作研究源） | 研究版（优先用） |
|---|---|---|
| etf_3x | `etf_3x_day_ta_data.pkl` 17.6MB | `etf_3x_day_ta_data_research.pkl` **69.9MB** |
| company_300 | `company_300_day_ta_data.pkl` 642.6MB | `company_300_day_ta_data_research.pkl` 657.4MB |
| hs300 | `hs300_day_ta_data.pkl` 681.5MB | `hs300_day_ta_data_research.pkl` 697.2MB |
| a_etf_all | `a_etf_all_day_ta_data.pkl` 151MB | `a_etf_all_day_ta_data_research.pkl` 154.5MB |
| company_star | `company_star_day_ta_data.pkl` 28.2MB（9/16 后未更新） | 无 research 版 |
| global | `global_day_ta_data.pkl` 5.2MB（9/16 后未更新） | 无 research 版 |

另有配套的 `{pool}_day_result.pkl` / `{pool}_day_result_research.pkl`（结果元数据，0.1~0.5MB，同样双轨）。

### 2.2 为什么必须用 _research 版

**生产 pkl 每次运行 `bc_technical_analysis.py`（数据管道）都会被覆盖**，且 etf_3x 生产版还会**被截短到 2025 年以后**（这就是 17.6MB vs 69.9MB 的差距；全历史为 2020-01-02 起 1686 个交易日）。今日实际发生的覆盖：

- 16:25~16:27：三池生产 pkl 被 16 点那次管道运行覆盖（a_etf_all 16:25:31 / company_300 16:26:29 / hs300 16:27:29）
- 21:55：etf_3x 生产 pkl 再次被覆盖（截短版）
- `etf_3x_day_ta_data_2026only_backup.pkl`（7.7MB，19:28）是截短版的备份，仅存档用

_research 版是全历史冻结快照，不受管道影响。今日的时间戳：三池 research 版为 9/18 凌晨 0:55~0:57 生成；etf_3x research 版为 9/18 19:42 从全历史恢复。

### 2.3 使用规则

1. **研究/回测一律优先 `XXX_day_ta_data_research.pkl`**，通过 `--pkl-path` 显式指定（signal_search / score_backtest / factor_mining 均支持）。
2. `alpha_mining.py` 的 POOLS 已内置路径，其中 etf_3x **已指向 research 版**，其余三池仍指生产版（若管道再跑，建议同样切换）。
3. 生产 pkl 只作"最新数据"刷新源；跑完管道后若要保留全历史，记得重新导出 research 快照。
4. `research\data\` 里的 `etf_3x_day_ta_data.pkl`、`etf_3x_day_ta_data_177d_backup.pkl` 是**遗留旧文件**，与主数据目录无关，勿混淆。

---

## 三、research\ 脚本位置与用法

全部脚本位于 `c:\Users\northcheng\git\quant\research\`，只读数据（不改 pkl），输出进 `research\output\`。

| 脚本 | 层级/用途 | 关键参数（默认值） |
|---|---|---|
| [factor_research.py](factor_research.py) | 早期 IC/分位数/事件研究（close→close 老口径，结论已由 signal_search 取代） | `--pool etf_3x` `--pkl-dir C:\Users\northcheng\quant` `--start` `--horizons 1,5,10,20,60` `--gate-col trend_magnitude_day` `--factors` `--skip-audit/ic/events` |
| [signal_search.py](signal_search.py) | 成分筛选：可交易口径收益 + NW-HAC t + BH-FDR（`nw_tstat`/`bh_qvals` 权威实现） | `--pool etf_3x` `--pkl-path` `--start 2021-01-01` `--horizons 5,20,60` `--top-k 5` `--min-cs 10` `--validate-top` `--signals` |
| [indicator_eval.py](indicator_eval.py) | 指标价值评估（分级/冗余/邻居），**子命令制** | `eval --pool etf_3x|all [--derived]`；`grade`；`redund --pool`；`neigh` |
| [factor_mining.py](factor_mining.py) | F_ 族批量因子挖掘 | `--pool etf_3x` `--pkl-dir C:\Users\northcheng\quant` `--pkl-path` `--start 2021-01-01` `--horizons 5,10,20,60` `--validate-top 8` `--signals` |
| [conditional_eval.py](conditional_eval.py) | 条件层：事件/状态网格下信号表现 | `--pool etf_3x|all` `--pkl` `--events` `--event-expr` `--grid s,m` `--state-expr` `--cooldown` `--min-days` |
| [score_backtest.py](score_backtest.py) | 集成层组合回测（gate/权重/仓位/成本） | `--pool etf_3x` `--pkl-path` `--start 2021-01-01` `--gate-col trend_magnitude_day` `--weights default` `--top-k 5` `--exit-rank 12` `--sizing tier` `--cost-bps 10` `--sensitivity` `--mined` `--derived` |
| [alpha_mining.py](alpha_mining.py) | **今日主工具**：48 候选信号四池挖掘 + 汇总分级 | `--pools etf_3x,company_300,hs300,a_etf_all` `--pkl "池=路径"` `--start 2021-01-01` `--horizons 5,20,60` `--signals`（只评指定候选） |
| [export_pool_context.py](export_pool_context.py) | 导出池上下文 csv 给外部（如 agent）分析 | `--pool etf_3x` `--pkl-path` `--out-path research/data/{pool}_context.csv` |
| [factor_signal.py](factor_signal.py) | gate 信号快照（当前持仓建议） | `--pool etf_3x` `--pkl-path` `--weights-preset gold4` `--top-k 5` `--exit-rank 12` `--as-of` `--recent 3` |

典型命令（研究源一律 research pkl）：

```powershell
cd C:\Users\northcheng\git\quant\research

# 今日主流程（48 候选四池）
python alpha_mining.py --pools etf_3x,company_300,hs300,a_etf_all

# 深挖指定信号
python alpha_mining.py --pools company_300 --signals H_trendmag_alpha,H_ichimoku_alpha

# 指定 research pkl 的单池筛查
python signal_search.py --pool company_300 --pkl-path C:\Users\northcheng\quant\company_300_day_ta_data_research.pkl

# 指标层全池评估
python indicator_eval.py eval --pool all

# 集成回测带敏感性
python score_backtest.py --pool etf_3x --pkl-path C:\Users\northcheng\quant\etf_3x_day_ta_data_research.pkl --sensitivity
```

alpha_mining.py 输出结构：`research\output\alpha_mine_{时间戳}\`，含 `summary.csv`（四池汇总+分级，144 行）、各池 `{pool}_h{5,20,60}_screen.csv`、`{pool}_validate.csv`、`report.txt`。

---

## 四、今日研究时间线（9/18 10:13 → 9/19 00:01）

| 时段 | 工作 | 产出（research\output\ 下） |
|---|---|---|
| 10:13~10:31 | indicator_eval 开发调试（checksign / smoke / eval_all / grade / redund / neigh） | `_*_log.txt`、`_prefix_backup_20260918\` |
| 16:23 | 174fix 备份 | `_backup_174fix_20260918\` |
| 16:25~16:27 | 数据管道运行，覆盖三池生产 pkl | （pkl 时间戳） |
| 16:33~16:36 | **indicator_eval 主产出**：四池指标价值分级 | `eval_{四池}.csv`、`eval_summary.csv`、`eval_redundancy_*.csv`、`eval_neighbors_*.csv` |
| 19:28~19:42 | etf_3x 数据抢救：截短版备份 + 全历史 research 版恢复 | （pkl） |
| 19:38~19:46 | score_backtest 四池集成回测（9 个目录） | `*_bt_20260918_19*` |
| 20:12~20:14 | factor_mining 两池 F_ 族因子挖掘 | `etf_3x_mine_20260918_201200\`、`company_300_mine_20260918_201232\` |
| 20:15~20:17 | 追加回测 7 个 | `*_bt_20260918_201*` |
| 21:55 | etf_3x 生产 pkl 又被管道覆盖（截短） | （pkl） |
| 22:38 | etf_3x 回测 + 敏感性分析 | `etf_3x_bt_20260918_223744\`（含 sensitivity.csv） |
| 23:42~23:49 | conditional_eval 四池条件层检验 | `cond_eval_{四池}.csv` |
| 23:59 | alpha_mining 单池冒烟 | `alpha_mine_20260918_235955\` |
| 9/19 00:01 | **alpha_mining 四池全量（最终结论来源）** | `alpha_mine_20260919_000017\` |

链路即三层验证漏斗：成分层（indicator_eval）→ 条件层（conditional_eval）→ 集成层（score_backtest / alpha_mining + signal_search 的统计验证）。

---

## 五、今日核心结论速查（依据 alpha_mine_20260919_000017）

### 5.1 推荐信号（4 个）

| 信号 | 最优周期 | 关键证据 | 备注 |
|---|---|---|---|
| **H_trendmag_alpha** | h=5 | company_300 nw_t=4.39，q≈0.000 | 构造 `normalize_causal(x.abs(), 252, 60)`；与 pkl 原列 P_trend_magnitude_alpha 复刻一致（口径自检通过） |
| **H_ichimoku_alpha** | h=5 | 四池 q 全<0.35 | m_trend_score_alpha 同源构造（本尊） |
| **C_tmqmom** | h=5 | company q=0.005，ICIR 0.14~0.18 全场最高 | 秩组合族 |
| **F_er20** | h=20 | 唯一长周期四池全同号，a_etf_all q=0.078 | 换手 0.34，成本友好 |

集成层够格的共 4 个（均为 company_300 h=5）。

### 5.2 结构性发现（比单信号更值钱）

1. **etf_3x 横截面塌缩**：top-5 恒选同 2 只 3x ETF（n_eff_symbols 均值 1.1，top3_share 0.72~0.78），超额 t 天然上不去——它是择时池不是选股池，应单列评级口径。
2. **长周期双保守效应**：h=60 时 NW-HAC(lag=59) 把最好 |nw_t| 压到 2.83，而 BH(48 候选) q≤0.10 需 |t|≈3.1+；IC 口径 t 高达 10~13 的低风险族（N_ulcer60 12.19、H_volchg_alpha 13.25）全被统计口径吃掉。
3. **无单池拖累**：三池显著而 a_etf_all 拖累的候选 = 0 个，瓶颈不在方向一致性。
4. **F_alpha60 池间分裂**：h=60 个股池 +1.6~7.7% vs a_etf_all -2.8%（q=0.06 反向显著）——高 alpha 个股延续、高 alpha 主题 ETF 反转。
5. **F_mom121 降级**：sig_ac1≥0.99，属"静态身份"而非真信号。

### 5.3 数据面

- q≤0.10 计数（48 候选）：h=5 → etf_3x 0、company_300 18、hs300 3、a_etf_all 1；h=20/60 仅 a_etf_all 各 4。
- 池 q 中位数：etf_3x 0.706 / company_300 0.492 / hs300 0.793 / a_etf_all 0.815。

---

## 六、遗留待办

1. **生产管道改造**（bc_technical_analysis.py）：把 m_trend_score_alpha / H_trendmag_alpha 配方进生产列；修复 etf_3x 生产 pkl 截短问题（全历史只存在于 research 版）。
2. alpha_mining.py POOLS 里 company_300 / hs300 / a_etf_all 三池切换到 _research pkl。
3. 4 个推荐信号进 conditional_eval（条件层）与 score_backtest（集成层）做组合验证。
4. 评级口径改进：q_max 改用"次弱池"、etf_3x 单列择时评级、长周期增加 IC-t 双口径。
5. factor_signal.py 的 gate 预设落地。

---
*生成：2026-09-19 00:20，由当日研究过程自动整理*
