# quant 项目全面审视与评价

> 审查对象: `C:\Users\northcheng\git\`（`quant\` 主包 + `research_2026-09\` 研究脚本）
> 审查方式: 全静态审查（沙箱禁用 shell，未执行任何回测/测试），逐文件读取 + 数据产出核对
> 审查范围: 36 个 Python 文件（`quant\`、`quant\factor\`、`quant\ml\`）+ **现役生产脚本 `~/signal_generator.py`（732 行）** + 研究产出 CSV/报告 + 生产链路（`~/quant`、`~/quant/quant/`、`~/automatic_trader.py`）
>
> **第 2 轮更正（依据用户澄清）**：`~/signal_generator.py` 是现役信号生成器；`~/git/research_2026-09\`（含 `signal_bridge.py`）是**上一轮研究**，已移出主文件夹。因此原 P0-0"生产信号桥断裂"**不成立**，已改写为 `signal_generator.py` 的回退路径缺陷；旧桥的断裂记录降级为附录性质（见 P0-0b）。

---

## 0. 执行摘要

**总体评价：研究方法的严谨程度显著高于工程实现水平。**

这个项目最反直觉的一点是：**它的"防自欺"能力（防前视、防过拟合、防静态选票）在我看来已经达到专业买方研究团队的水准**，而它的工程化（版本控制、测试、契约、依赖管理）还停留在个人脚本水平。绝大多数个人量化项目的短板是前者，你的短板是后者。

五个最需要立刻处理的问题（P0）—— 但实际整合四份深度审查后共 **6 条 P0**（含 P0-0，见第 3 节）：

| # | 问题 | 影响 |
|---|---|---|
| **P0-0** | **生产信号生成器的回退路径会用错池的因子/权重** —— `~/signal_generator.py`（现役，架构明显优于旧桥）在 `factor_config.json` 缺失/损坏/该池没跑过 `report` 时，**静默回退到写死的 `etf_3x` 冻结常量**；而回退只打一条 `logger.info`。`--pool hs300` 时无任何 warning | 信号会用 `etf_3x` 的 10 个因子和权重去跑 298 只 A 股，静默产出错误信号 |
| P0-1 | **整个工作副本不是 git 仓库**（无 `.git`），且产出与实际下单的全部生产代码不在版本控制内 —— `automatic_trader.py`（778 行）、`signal_generator.py`、`bt_core.py`、`score_backtest.py` 都在 `~/` 下 | 无历史、无 diff、无回滚、无法回答"这个数字是哪版代码算的"；研究到实盘之间没有可审计的冻结层 |
| P0-2 | **实盘下单的失败不改变资金记账** —— `bc_trader.py:458` 的 `trade_summary` 返回值从不被判断，`L462` 无条件扣减 `available_cash` | 下单失败仍扣钱 → 该账户当轮后续买入被系统性少买，多账户间配置漂移 |
| P0-3 | **研究/生产数据源静默降级** —— 生产 pkl 不存在时自动回退研究版全史 pkl（`a_etf_all`/`hs300`/`company_1000` 已发生），且 `signal_gen` 日志只给一行 warn | 实盘信号与研究报告的"冻结口径"可能不是同一份数据，而没有任何门禁拦住它 |
| P0-4 | **实盘三个保护条件全部失效** —— `ta_config.json` 的 `stop_loss=-0.8`/`stop_profit=10` 与小数量纲的 `rate` 比较（等价于"亏 80% 才止损、赚 1000% 才止盈"），而 `bc_trader.py:641` 的 `rate_inday` 被**硬编码为 0.0** → 日内止损永不触发。回测引擎里**根本没有止损** | 唯一的实盘下行保护未经验证且实际不生效；回测与实盘净值永久不可比 |
| P0-5 | **富途市价单的资金校验被绕过** —— `bc_trader.py:714-719` 把市价单的 `price` 占位成 `0.1`，`:729` 又用它调 `get_affordable_quantity` → 算出 `≈10×现金` 股，**任何数量都判为"买得起"**；这是默认路径 | 唯一"可无限量下单"的口子，且 `condition_trade` 的 quantity 来自配置文件、无上限 |
| P0-6 | **真实券商持仓与净值已入库** —— `quant\portfolio.json` 含 futu REAL 净值 28169.53、持仓明细与成本价；`position.json` 含三平台逐标的 cash/position | 泄露"钱的位置"；且运行时状态放在源码目录，一次 checkout 就可能覆盖实盘状态 |

另外有两个必须诚实面对的现实（不是 bug，是结论）：

- **这套因子的样本外优势很可能是有限的**。`factor\output\hs300_clean` 的实跑数据显示：41 因子等权在 OOS 的总收益 0.794，而**同期等权买入持有 0.802** —— 同等收益、波动 0.424 vs 0.200、最大回撤 -45.6% vs -23.5%。因子体系在这个池上没有跑赢基准，只是承担了两倍波动。你在 `research_summary_20260922.md` 里"生产要不要改 —— **不改**"的结论是对的，我同意。
- **美股三池的 73%~95% CAGR / Sharpe 1.34~1.39 需要打折看**。样本只有 2020+（一段历史级牛市），年换手 100~162x、平均持仓 3.4~6.8 天，回测假设了无限流动性、无最小手数、无市场冲击。

---

## 1. 项目是什么：一张地图

```
C:\Users\northcheng\
├── automatic_trader.py          ← 生产主循环（778 行）  ★不在 git
├── signal_bridge.py             ← 生产信号桥（池预设写死在源码）  ★不在 git
├── bt_core.py / score_backtest.py ← 生产回测引擎  ★不在 git
├── quant\                       ← 工作数据（pkl/csv/xlsx/png/密钥）
│   ├── *_day_ta_data_research.pkl  研究版全史（5 个池）
│   ├── *_day_ta_data.pkl           生产版盘前更新（只有 2 个池）
│   ├── api_key\api_key.json        ← 7 个数据源密钥（明文，好在不在 git 里）
│   ├── trader\user_info.json       ← 券商账号；rsa_*.pem 私钥
│   └── ta_model\                   每日信号 xlsx / png
└── git\                         ← 版本控制根（quant 是包名，故 git\ 是 sys.path 入口）
    ├── quant\
    │   ├── bc_data_io.py        2858 行  数据 IO/下载/邮件/config/pickle
    │   ├── bc_technical_analysis.py 7600+ 行  100+ 技术指标 + 绘图
    │   ├── bc_trader.py         1218 行  券商下单（Tiger / Futu）
    │   ├── bc_util.py            663 行  工具/日志/日期
    │   ├── factor\              研究流水线（自包含，8 层）
    │   │   data → factor → prepare → evaluate → combine → backtest → report
    │   │   + wf_step1..4 / wf_strat_search / wf_combo_search（walk-forward 系列）
    │   ├── ml\                  纯 numpy ML（线性/GBDT/MLP/GRU/LSTM）
    │   └── tests\               1 个测试文件，14 个断言
    └── research_2026-09\        研究脚本 + 产出 + 14 篇报告
```

**架构演化轨迹**（从 `__pycache__` 的残留可以还原）：

```
老体系: bc_factor_search / bc_combo_search / bc_backtest / signal_mine_* / regime_symbol_wf
        → 源码已删，只剩 .pyc；其中 bc_backtest 的 .pyc 还在，但 .py 已不存在
新体系: factor\ 包（自包含，不 import 任何 bc_*）
```

`factor\config.py:5` 把这件事写得很明确："那三个研究脚本后续可能被删除"。这说明你**已经在做正确的迁移**，但迁移没收尾：老的 `.pyc`、`.ipynb_checkpoints`、被删模块的产物还留在库里。

---

## 2. 做得好：这些是真正的专业级实践

先说好的，因为下面批评很多，但**这个项目的"防自欺"设计是我在个人量化项目里见过最好的之一**，而且是用工程手段强制执行的，不是靠自觉：

1. **前视偏差用不变量证明，而不是靠声明**（`factor\backtest.py:539` `verify_backtest`）
   - `oracle`（拿未来收益当分数）应该大赚、`shuffle`（打乱时序）应该≈随机 —— 引擎标定
   - **投毒不变量**：把 IS 窗外的 `fwd` 换成垃圾值，用同样表达式重新定权，权重必须**逐值不变**（`<=1e-12`）。这是真正证明"定权链路没读 IS 窗外数据"的方法，比写一百字注释有用。
   - 而且注释里解释了为什么**不**把 "honest <= lookahead 净值" 当判据（实测 59/70 因子集出现反向），这种对判据有效性的自我怀疑非常难得。

2. **标签口径与交易口径分离**（`evaluate.py:81` `forward_return`）
   - 标签：`t` 收盘决策 → `t+1` 开盘入场 → `t+1+h` 开盘出场，全部用 `open`。
   - 这避免了最常见的"用当日收盘价成交"的虚假收益，也与回测引擎 `run_engine` 的 open-to-open 结算严格一致。

3. **多重检验校正做对了**（`evaluate.py:325` `bh_qvals`）
   - 日级 IC 序列 → 因子重叠标签导致自相关 → **Newey-West HAC**（`_nw_tstat`，lag = max(h-1,1)，并解释了为什么 h=1 也要至少修 1 阶）→ **BH-FDR** 跨因子校正 → 分级 A/B/C。
   - 这个链路在个人项目里几乎见不到。多数人只报一个 IC 和 t 值。

4. **"这个 alpha 是不是只是押注了几个名字"被做成了探针**（`evaluate.py:395` `validate_one`）
   - 静态度检测 `sig_ac1 >= 0.99` → `is_static`
   - 分半 + 静态退化对照：用前半段"每标的平均分"当静态选票，看它能否复制后半段表现 → `dyn_minus_static <= 0` 即"动态选股没超过静态身份"
   - 持仓集中度 `top3_share / hhi / n_eff_symbols`

5. **对"没有 alpha"的结论照实写并据此不改生产**（`research_summary_20260920.md` §7.3、`astock_factor_strategy_20260922.md` §1）
   - "新挖 R/I/W、K、X 共 4 族（约 145 个候选组合）**全部无 A/B 级、无 FDR 显著**"
   - "以 `min(IS1,IS2)` 为目标从零贪心，两池都止步于单信号"
   - "算法按 obj 选出的唯一'改进'……**但 OOS 从 1.26 崩到 0.56**，被否决"
   - "**不改**。新增族进'观察名单'不上线"
   - 报告里还明确写了方法论教训："IS 双半窗能防'单窗过拟合'，但**防不了'只在熊市有效'**，OOS 复核不可省"。这是自己踩过坑后总结的真话。

6. **模块文档质量高**：36 个文件全部有 `# -*- coding: utf-8 -*-` + 长 module docstring，说明了"这一层的职责/与前几层的区别/最容易踩的坑/自检命令"。`backtest.py` 头部把"信号日 s 收盘决策 / 执行日 d 开盘成交"的时间线写成了 4 行规范，这是能防止后人改错的文档。

7. **组合层的缺失感知与去重逻辑是对的**
   - `dedupe` 用 `|rho|` 而不是 `rho` 判冗余（`combine.py:178`），并解释了原因："强负相关的两个因子同时等权入池会内部对冲……所以'冗余'看的是信息重复度 |rho|，不是符号"。这个细节很多人会做错。
   - `ic_weights` 带符号（负 RankIC 自动反向使用），而不是盲信先验 `direction`。
   - `IC_SHRINK=0.5` 向"方向对齐等权"收缩，理由写得很清楚：|RankIC| 是单期估计、噪声大。

---

## 3. P0：上线前必须修

### P0-0 生产信号生成器的回退路径会用错池的因子与权重

> **口径更正**：现役生产信号脚本是 `~/signal_generator.py`（732 行），它走 `quant.factor` 体系、**不 import 任何 `bc_factor_search/combo_search/backtest`**，所以下面这段"旧桥断裂"**不影响生产**。`research_2026-09\`（含 `signal_bridge.py`）是上一轮研究，已被移出主文件夹。此条据此重写。

**现役脚本的架构是对的（先说清楚，因为它明显优于旧桥）**：

| 设计 | 证据 | 为什么重要 |
|---|---|---|
| 只读 `quant.factor`，不碰老 `bc_*` 研究脚本 | `:67-71` | 与已废弃路径解耦，不会再出现"导入已移走的模块" |
| **不在生产数据上重定权** | `:29-33` 明文写着"本脚本绝不在生产数据上重新定权"；`:631-633` 用 `build_signals` 算因子值，权重直接搬 `factor_config.json` | **这条直接消掉了我原先担心的"线上 IC 含未来信息"**：权重来自研究版 IS，不是生产数据现算 |
| 决策截面严格 `< trade_date` | `:184-214`（`valid.index < trade_date`） | 盘前/复盘两种模式都防前视 |
| 显式校验数据陈旧 | `:159` `MAX_DATA_GAP = 4`，`:638-642` 超限即 `SystemExit` | 数据停更时**不产出信号**（fail-closed，正确取向） |
| 快照/Excel 带 `meta` sheet | `:664-683` | 记录了 pkl 路径、决策截面、权重，可追溯 |
| 用生产 pkl（不含研究版全史） | `:607-614` | 减少研究与生产的数据面差异，且回退时**有 warn 日志**（`:613`，`:2` 也确认走的是"生产 pkl 不存在 → 回退研究版"） |

**真正的缺陷在回退路径**：

```python
# ~/signal_generator.py:87-98
_FB_KEPT = ['mom_12_1', 'hi_252', 'mom_120', 'rev_10', 'rev_5',
            'slope_20', 'vol_20', 'mom_60', 'vratio_20', 'mom_20']
_FB_GREEDY = ['mom_12_1', 'rev_10', 'hi_252']
_FB_SCHEMES = { 'equal': ..., 'greedy': ..., 'ic': {...}, 'single': {'mom_12_1': 1.0} }
_FB_DEFAULT_SCHEME = 'equal'      # ← 注意: 不是 cd_ic
```

```python
# ~/signal_generator.py:140-151
conf = load_pool_config(pool, path)
schemes = (conf['schemes'] if conf else _FB_SCHEMES)      # ← 只看"有没有 conf", 不看"是不是这个池的"
default = p.get('default_scheme', _FB_DEFAULT_SCHEME)
```

**问题**：
1. `_FB_*` 全部是 **`etf_3x` 的早期 IS 导出**（`:81` 注释自己写明）。而 `resolve_pool(pool)` **不校验 `conf.meta.pool == pool`**。所以当 `factor_config.json` 里没有该池的条目时，**任意池都会拿到 `etf_3x` 的因子与权重**：
   - `--pool hs300`（298 只 A 股）会用 `mom_12_1/hi_252/rev_5/...` 这 10 个 ETF 因子的 IC 权重去打分；
   - 唯一提示是 `:604-605` 的一行 `logger.info('[conf]: factor_config.json 无 pool=hs300 的口径, 回退内置冻结常量')` —— **info 级、不中断、仍然产出 Excel 和实盘信号**。
   - 这属于"静默产出错误信号"，比崩溃危险。
2. **回退常量本身也已经过期**：当前 `factor_config.json` 的 `etf_3x.kept` 是 **16 个**因子、`default_scheme` 是 **`cd_ic`**；而 `_FB_*` 只有 10 个因子、默认方案是 `equal`。即使真的为 `etf_3x` 回退，用的也是**一轮之前**的因子集与权重。
3. `factor_config.json` 一旦**读不出来**（JSON 被截断、文件被占用、权限问题），`_load_config_doc`（`:116-121`）会 `except Exception: return {}` → 触发回退。也就是说"配置文件损坏"这个最需要报警的场景，会静默降级到陈旧口径。
   - 更耐人寻味的是 `factor\report.py:653-658` 的 `write_config` 也有 `except Exception: doc = {}`：**读失败**和**写失败**两条路径都可能把 `factor_config.json` 弄成不完整状态。
4. 回退常量里的因子名（`mom_12_1` 等）都在注册表里（已核实 `factor.py:187-361`），所以回退**能跑**；`get_factor` 对未知名字会 `raise KeyError`（`factor.py:829`），所以若 `factor_config.json` 里的方案引用了未注册因子（如 `C_tmqmom`——它在 `factor.py:636` 有注册，但 `a_etf_all` 的 `cd_ic` 里还引用了 `H_ichimoku_alpha`，此类"研究脚本自定义信号名"在注册表里是否齐全需要逐个核对）会**硬失败**。这两种表现（一个静默错、一个硬崩）都不理想。

**修复建议**：
1. `resolve_pool` 加池校验：`conf` 非空但 `conf['meta']['pool'] != pool` 时**抛异常**，不要回退。
2. 完全去掉 `_FB_*` 回退，改为**缺配置即 fail-fast**（`raise SystemExit`）。研究结论是生产的唯一参数来源，读不到就该停，而不是猜。
3. 若一定要保留回退，把它改成"**只对 `etf_3x` 生效**"，并且 `logger.warning` + 在 Excel 的 `meta` sheet 里写一行 `fallback=1`，让错误口径可见。
4. `_load_config_doc` 的 `except Exception: return {}` 改为区分"文件不存在"（可接受）与"文件存在但解析失败"（必须告警/中止）。

**关于旧桥断开的记录（不再是 P0，但仍建议清理）**：`research_2026-09\signal_bridge.py:68-69`、`strat_common.py:51-53`、`strat_walkforward.py:53`、`bc_backtest.py:32,484,1044-1047,1097,1612`、`bc_combo_search.py:27-28` 共 **12 处**导入 `quant.bc_factor_search / bc_combo_search / bc_backtest`，而这三个模块现在位于 `research_2026-09\`（`quant\` 下只剩 4 个 `.py`）。既然该目录已归档，建议：**要么删掉这批死导入，要么在目录里加一个 `README` 说明"已归档、不可直接运行"**，否则半年后（包括你自己）还会被它误导——这次就误导了我。

### P0-0b 旧桥断裂的完整记录（仅供参考，不影响现役生产）

**证据链**：

1. `research_2026-09\signal_bridge.py:68-69`
```python
from quant.bc_factor_search import load_panel, POOLS, build_alpha_cands
from quant.bc_combo_search import build_a_combo_cands
```
2. `quant\` 下的 `.py` 只有 **4 个**：`bc_util.py`、`bc_data_io.py`、`bc_technical_analysis.py`、`bc_trader.py`。**`bc_factor_search.py` / `bc_combo_search.py` / `bc_backtest.py` 都不在里面。**
3. 这三个文件的真实位置是**兄弟目录** `research_2026-09\`（已核实存在）。
4. `quant\__pycache__\bc_factor_search.cpython-312.pyc`、`bc_combo_search.cpython-312.pyc`、`bc_backtest.cpython-312.pyc` 还在 —— 说明它们**曾经**在 `quant\` 下，是后来被移走的。
5. `signal_bridge.py:57-58` 的 path 注入只加了 `git_path`（仓库根），**没有**注入脚本自身目录：
```python
root_paths['git_path'] = root_paths['home_path'] / 'git'
sys.path.append(str(root_paths['git_path']))       # quant 包(无 research 平铺路径)
```
注释里的"无 research 平铺路径"说明你**知道**这个目录不在 path 上 —— 但 import 语句没有跟着改。

**影响范围**：全树 **12 处**断裂导入，分布在 4 个文件：

| 文件 | 行号 |
|---|---|
| `research_2026-09\signal_bridge.py` | 68, 69 |
| `research_2026-09\strat_common.py` | 51, 52, 53 |
| `research_2026-09\strat_walkforward.py` | 53 |
| `research_2026-09\bc_backtest.py` | 32, 484, 1044, 1045, 1047, 1097, 1612 |
| `research_2026-09\bc_combo_search.py` | 27, 28 |

其中 `bc_combo_search.py:28` → `from quant.bc_backtest import ...`，而 `bc_backtest.py:32` → `from quant.bc_factor_search import ...` —— 三个模块**互相 import**，所以断裂是连带性的。

**时间线（重要）**：`factor\output\logs\signal_gen_2026-09-28.txt` 显示 2026-09-28 08:41 信号生成**成功**并落盘 xlsx。也就是说当时脚本还能跑 —— 那时它位于 `quant\research\` 下，`quant` 是合法包路径。之后目录被重命名为 `research_2026-09\` 并移到兄弟层级，**import 语句没有同步修改**。这同时解释了为什么全树 219 处文档引用 `quant/research` 全是死路径。

**注意**：如果哪天真的又去跑这个旧桥，`automatic_trader.py` 的调度逻辑（`automatic_trader_flow.md:16`）在桥失败时**不会中断主循环**（"当日 03:45 交易时回退读 legacy 信号文件"）→ 实际表现是"静默回退到 legacy 信号"。但因为现役脚本已是 `signal_generator.py`，这条只作为归档残留处理即可。

**处理建议**（既然该目录已归档）：
1. 在 `research_2026-09\` 里加一个 `README.md`，写明"**已归档，不可直接运行**：脚本 import 的 `quant.bc_factor_search / bc_combo_search / bc_backtest` 已不在 `quant\` 下"。
2. 或者把这 3 个模块移回 `quant\`（它们本来就是 `bc_*` 命名，属于 `quant` 包），让归档目录能自洽运行——但既然 `signal_generator.py` 已用 `quant.factor` 替代，**我更倾向第 1 个（标明归档）而不是恢复旧路径**。
3. 无论选哪个，**加一条 import 冒烟测试**（`python -c "import quant.factor; import quant.bc_data_io"` 级别）：这次断裂能潜伏数天、并误导审查者，就是因为全仓没有任何 import 级测试。

### P0-1 整个工作副本不是 git 仓库

**证据**：`glob .git/HEAD`（`git\` 与 `git\quant\` 都试）→ 无；`glob **/.git/HEAD` → 无；`read .git\HEAD` / `.git\config` → not found。全树 1439 个文件里唯一含 `.git` 字样的是 `git\quant\.gitignore`。

**但历史上确实用过 git**：`research_2026-09\report\research_20260918-21.md:6852` 有 `?? research/`、`git ls-files research/output` 为空的记录；`:6949` 记录了 `git status` clean、把带冲突标记的文件"已经推到了远端"、Dell 端 pull 到坏文件；`:6995` 是 Dell `git pull` 后重跑。→ **曾有 git，现在这个副本上没有 `.git`**（可能被剥离、可能从未拷回、可能在别处）。

**这改变了修复路径**：P0-1 不是"补提交几个文件"，而是"**先决定仓库在哪、要不要重建**"。建议：
1. 找到原远端（若存在），`git clone` 一份干净副本到新目录，再把当前工作副本的改动有选择地搬过去 —— **不要**直接把 1439 个文件（含 995 个研究产物、95 个 `.pyc`）一次性提交。
2. 若远端已丢失，`git init` 一个新的，但**先写好 `.gitignore` 再 `git add`**，否则第一次提交就会把 `research_2026-09\output\`（995 文件）和 `portfolio.json`（实盘持仓）一起收进去。
3. 提交前用 `git check-ignore -v <path>` 抽查关键路径，不要信任 `.gitignore` 的直觉。

### P0-1b 生产代码不在任何版本控制内

**证据**：`research_2026-09\report\automatic_trader_flow.md:3` 明示"对象: `c:\Users\northcheng\automatic_trader.py`（778 行）"。现役信号脚本 `~/signal_generator.py`（732 行）也在同一层级。这些文件都在 `C:\Users\northcheng\` 下，而工作树根是 `C:\Users\northcheng\git\`。

**为什么是 P0**：
- 实盘下单逻辑（778 行主循环 + 券商适配）与信号生成逻辑（732 行）没有历史、没有 diff、没有回滚点。改错一个数字无法定位引入时间。
- 生产参数目前有**两个来源并存**：已归档的 `research_2026-09\signal_bridge.py` 里写死的 `ALL_PRESETS`（源码常量），与现役 `signal_generator.py` 读的 `factor\factor_config.json`（`cd_ic` 等方案）。好消息是现役脚本**只读 `factor_config.json`**（`:73-79, :592-605`），所以实际生效的只有一套；但只要归档目录还在，就仍然会有人（包括审查者）去读那套已经失效的常量。
- 报告可以作为"事实上的文档"存在，但文档不能替代版本控制。

**建议**（按优先级）：
1. 立刻把 `automatic_trader.py`、**`signal_generator.py`**、`bt_core.py`、`score_backtest.py`、任何 `stop_loss_*.py` 纳入 git（哪怕是 `production/` 子目录，先入库再整理）。
2. 密钥/账号**不要**入库：在 `.gitignore` 里加 `api_key/`、`*user_info*`、`*.pem`、`trade_condition.json`，用 `api_key.json.example` 占位。
3. 删掉已归档的 `signal_bridge.ALL_PRESETS` 那套常量（或整个归档目录打上 `ARCHIVED` 标记），确保"权重→生产"的通道**唯一**且指向 `factor_config.json`。

### P0-2 下单失败仍扣减可用资金（空头记账错误）

**证据**：`quant\bc_trader.py:452-462`

```
quantity = math.floor((money_per_sec-trading_fee)/signal.loc[symbol, 'latest_price'])
if quantity > 0:
  ...
  trade_summary = self.trade(symbol=symbol, action='BUY', quantity=quantity, price=price, print_summary=False)
  self.logger.info(trade_summary)          # L459 —— 返回值只被记录，从不被判断
  available_cash -= quantity * signal.loc[symbol, 'latest_price']   # L462 —— 无条件扣减
```

而 `Tiger.trade`（`L947-1014`）和 `Futu.trade`（`L708-764`）在**失败时也不抛异常**，只把失败信息拼进返回字符串：

- `L986` `'FAILED: no response from server, order status unknown'`（下单返回 None）
- `L988` `'FAILED: Not affordable(...)'`
- `L1001` `'FAILED: Not enough stock to sell(...)'`
- `L1008` `'FAILED: {e}'`（异常被吞）
- `L757-758` Futu 同理

**后果**：一次下单失败（券商拒单、额度不足、盘前时段被拒、限价单未成交）会让 `available_cash` 凭空减少一笔，同一轮循环后续的买单被系统性少买；多账户（tiger_simu / tiger_real / futu_real 同时遍历）各自漂移。反向的危险同样存在：市价单实际成交价高于决策价时，`L462` 按 `latest_price`（延迟 15 分钟的实时价）估算，低估实际占用。

**这个缺陷在小账户上会被放大**。`quant\ta_config.json:119-139` 的实盘配置：
```json
"init_cash": { "tiger": { "real": 3000, "simu": 100000 }, "futu": { "real": 3000, "simu": 100000 } },
"pool":      { "tiger": { "real": "etf_3x", "simu": "company_300" },
               "futu":  { "real": "futu_pool", "simu": "futu_pool" } }
```
**tiger 实盘 3000、futu 实盘 3000**。若 `signal_trade` 拿到"买入 5 只、每只 `money_per_sec = 3000/5 = 600`"的配置，一次失败就吃掉 600（20% 的账户），后续 4 笔被压缩甚至因 `L434` 的 `available_cash <= minimum_position`（`minimum_position` 默认等于 `money_per_sec`，即 600）直接 `break`。**结果是：第一笔失败 → 当天剩余买入全部取消**。这不是"少买一点"，而是"信号被执行成了完全不同的组合"。

**顺带一个量级问题**：3000 的账户按 `top_k=5` 等权是每只 600。而 `etf_3x` 池里不少标的单价不低（从 `signal_gen_2026-09-28.txt` 的 A 股快照看，`hs300` 里已有 895.86、653.06 这类价格的标的），**600 的预算可能连 1 股都买不到**，而 `math.floor` 会算出 0 → `L453` `if quantity > 0` 为假 → `L464` 只打一行 `'[BUY]: not enough money'`。所以**这个账户规模下，"每只 1 股"的约束可能已经让策略退化** —— 这属于"回测假设无最小交易单位、实盘有"的典型落差，建议单独核对一遍实际下单历史，看成交笔数与信号笔数的比例。

**建议**：
1. 让 `trade()` 返回结构化结果，而不是给人看的字符串：
   ```python
   return {'ok': bool, 'order_id': str|None, 'detail': str}
   ```
   至少要有一个可判定的布尔。现有的 `trade_summary` 字符串可以保留在 `detail` 里给人看。
2. `signal_trade` 里 `if not res['ok']: continue`（失败**不**扣减、不占用该仓位额度）。
3. 更彻底：不要在客户端估算可用资金，改为每轮重新 `get_available_cash()` 查询券商真实余额（你现在已经有这个方法，只是没用在下单循环里）。

### P0-3 生产数据源静默降级到研究版全史 pkl

**证据**：`factor\output\logs\signal_gen_2026-09-28.txt`
```
L2  [warn]: 生产 pkl 不存在(a_etf_all_day_ta_data.pkl), 回退研究版 a_etf_all_day_ta_data_research.pkl
L34 [warn]: 生产 pkl 不存在(hs300_day_ta_data.pkl), 回退研究版 hs300_day_ta_data_research.pkl
```
`~/quant` 下实际存在的 pkl：
```
a_etf_all_day_ta_data_research.pkl   company_300_day_ta_data_research.pkl
company_1000_day_ta_data_research.pkl etf_3x_day_ta_data_research.pkl
hs300_day_ta_data_research.pkl        ← 研究版 5 个
company_300_day_ta_data.pkl           etf_3x_day_ta_data.pkl  ← 生产版只有 2 个
```

**为什么严重**：
- `config.py:57-70` 明确设计了研究版/生产版两套命名（"研究版(全史)、生产版(盘前更新, 通常更短)"），说明你**本来就打算区分**。但回退路径让这个设计在实践中失效 —— 3 个池在生产上一直吃的是研究版全史数据。
- 决定实盘买卖 5 只标的的信号，其因子值是用一份"包含未来才更新的研究数据"算出来的，而**唯一的表现是日志里一行 warn**。没有非零退出码、没有告警、没有落盘标记。
- 更隐蔽的一层：`factor_config.json` 里 `etf_3x` 的 `meta.is_window = [2021-01-01, 2024-12-31]`，但 `data_start = 2020-01-02`；而 `company_300` 的 `data_start = 2010-01-04` / `is_window = [2011-01-01, 2019-12-31]`。**同一个 `factor_config.json` 里 5 个池的 IS 窗口口径完全不同**（2011-2019 / 2021-2024 混用），而 `factor\config.py` 里的全局常量是 `TRADE_START='2011-01-01'`、`IS_END='2019-12-31'`。用 `evaluate_factors(..., window='is')` 去复核 `etf_3x` 时，截出来的是 2011-2019，而 config 声称 IS 是 2021-2024 —— **复核口径与出报告口径不一致**。

**建议**：
1. 把回退改成**失败退出**：生产模式下 pkl 不存在 → 抛异常 + 非零退出码，让 `automatic_trader` 的 skip/重试逻辑（它本来就有幂等设计）去处理，而不是默默用错数据。
2. 在生产 pkl 里写入元数据（生成时间、数据末日、行数、池成员数），信号生成时校验"数据末日 == 预期交易日前一交易日"，不符即拒绝。
3. 若要保留回退，至少在产出的 xlsx/邮件报告上打一个**显眼的降级标记**，让人肉眼可见。
4. 把每个池的 IS/OOS 窗口从 `factor_config.json` 读入 `evaluate`/`backtest`，不要在 `config.py` 里用一套全局常量覆盖所有池。

---

### P0-4 实盘风控参数本身就是一个风险点

**证据**：`quant\ta_config.json:140-179`（实盘参数，非模拟）
```json
"stop_profit":      { "tiger": { "real": 10 },    "futu": { "real": 10 } },
"stop_loss":        { "tiger": { "real": -0.8 },  "futu": { "real": -0.8 } },
"stop_profit_inday":{ "tiger": { "real": 10 },    "futu": { "real": 10 } },
"stop_loss_inday":  { "tiger": { "real": -0.3 },  "futu": { "real": -0.3 } }
```
而 `bc_trader.py:530-534` 的用法是：
```python
stop_loss_list   = position.query(f'rate < {stop_loss_rate}')
stop_profit_list = position.query(f'rate > {stop_profit_rate}')
```

**按字面读**：`stop_loss = -0.8` 表示**亏损超过 80% 才止损**；`stop_profit = 10` 表示**盈利超过 1000% 才止盈**（约等于不止盈）；只有 `stop_loss_inday = -0.3`（日内 -30%）是有实际作用的。而 `tiger.real` 的池正是 `etf_3x`（3 倍杠杆 ETF）—— 单日 -20% 并不罕见，留给 -80% 的缓冲意味着基本无法挽回。

**三个字段里两个用"比率"、一个用"10"，单位不统一，极可能是笔误**（本意可能是 `stop_profit = 1.0` 即 +100%、`stop_loss = -0.3` 即 -30%）。需要你确认。

**为什么是 P0**：这是直接作用于真钱的参数，而 `bc_trader.py:543` 的 `cash_out` 是**唯一的**下行保护 —— 回测引擎（`factor\backtest.py`）里**根本没有止损**（只有 rank 进出场 + band）。也就是说：**止损从来没有被回测验证过，它是在实盘上边跑边试的**。这同时意味着"回测净值"与"实盘净值"永远不可比。

**建议**：
1. 逐一确认 4 个参数的语义与单位，统一成"收益率小数"（`-0.3` / `+1.0`）。
2. 把止损/止盈加进 `run_engine`（可选规则），然后对比"有止损 vs 无止损"在 OOS 上的表现。这一步同时解决"回测与实盘口径不一致"的问题。
3. 注意止损与信号退出的相互干扰：`cash_out` 卖出后，下一个交易日信号可能又把它买回（若 rank 仍在前 5），产生来回摩擦与额外成本。回测也未建模。

### P0-5 真实券商持仓与净值已入库

**证据**（`quant\portfolio.json`，单行 JSON）：
- `futu.REAL.net_value = 28169.5337` + 持仓 `LTBR`
- `futu.SIMULATE.net_value = 2868148.37`
- `pingan.snowball` 的 A 股 ETF 持仓（`159100 / 159502 / 159561 / 159577 / 159937`）+ `net_value 26686.01`
- `tiger.simulation_account` 的 12 只美股持仓（ALM / AMD / ATRC / CHE / ERAS / HTHT / INTC / JLL / PNW / RIO / RTX / SBUX）与成本价

`quant\position.json`：三个平台的逐标的 cash / position 目标。`quant\selected_sec_list.json`：各池标的清单。

**判断依据**：报告里说"仓库内 `*.json` grep 密钥 → No matches"（这条是对的，`api_key.json` 确在仓库外的 `~/quant\api_key\`），但**账户状态本身泄露了**：实盘净值、持仓标的、成本价全在里面。这些是"钱的位置"信息，敏感度不低于一个数据源 API key。

**同时这是一个功能性风险**：`portfolio.json` / `position.json` 是**运行时状态**，却被放在源码目录里。任何一次 `git checkout`、分支切换、或误拷贝，都可能把实盘状态覆盖成历史快照 —— 而 `bc_trader.py` 的 `update_position_record` / `update_portfolio_record` 会读它。

**建议**：
1. 把这两个文件移到 `~/quant/`（数据目录，与 pkl 同级），代码里改成从 `config['config_path']` 读 —— 实际上 `bc_data_io` 的 config 机制已经在用 `config_path`/`result_path` 这套路径注入了，走同一条路即可。
2. 仓库里只留 `portfolio.json.example` 模板。
3. 若已提交过，`git rm --cached` 并从历史里清（`git filter-repo`）—— 但注意 P0-1：现在还没有 `.git`，所以这次实际上是"**别把它加进新仓库**"，成本反而更低。

### P0-6 复权处理的索引 bug，会让"昨日拆股"的复权被静默丢弃

这是本次审查中**唯一一处明确的、影响全链路数据正确性的代码 bug**。

**证据**：`quant\bc_technical_analysis.py:207-220`（`preprocess` 函数，生成 pkl 面板里的 OHLC）

```python
202  df['split_n1'] = df['Split'].shift(-1).fillna(1.0)      # 时间降序下, -1 = 后一日/更早
...
212  split_idx    = _safe_query(df, 'Split != 1.0 and (adj_rate >= 1.95 or adj_rate <= 0.45)')
213  df.loc[split_idx, 'adj_factor'] = 1 / df.loc[split_idx, 'Split']

215  # Split在昨日, 价格变化却在今日
216  split_n1_idx = _safe_query(df, 'split_n1 != 1.0 and (adj_rate >= 1.95 or adj_rate <= 0.45)')
217  df.loc[split_n1_idx, 'adj_factor'] = 1 / df.loc[split_idx, 'split_n1']   # ← 应为 split_n1_idx
218
219  df['adj_factor'] = df['adj_factor'].ffill()
220  df['Adj Close'] = df['Adj Close'] * df['adj_factor']
...
246  adj_rate = df['Adj Close'] / df['Close']
247  for col in ['High', 'Low', 'Open', 'Close']:
248    df[col] = df[col] * adj_rate
```

**为什么这是 bug**：`split_idx` 与 `split_n1_idx` 是按**不同条件**筛出来的两个集合，按定义基本不相交。`df.loc[split_n1_idx, 'split_n1']` 只在 `split_idx` 那几行上有值，赋到 `split_n1_idx` 行上时 **pandas 按标签对齐 → 全部落成 NaN**。随后 `:219` 的 `ffill()`（降序 = 向后/向更早已发生的日期填充）会把 NaN 格用**相邻的旧因子**补上 —— 本意要写的 `1/split_n1` 被彻底丢弃。

对照 `:225-240` 那段被注释掉的原版逐行循环实现，原版写的是 `adj_rate = 1/row['split_n1']`（`row` 就是当前行），**证明 `:217` 右边本该用 `split_n1_idx`**。

**波及范围（这是它成为 P0 的原因）**：
1. `Adj Close` 没能被正确复权 → 该砍掉的跳空留在序列里；
2. `:246-248` 用 `adj_rate = Adj Close / Close` 把 **OHLC 四列全部缩放** → 这个错误被**放大进所有价格列**；
3. pkl 面板的 OHLC 是你所有东西的唯一数据源：`factor\factor.py` 的 `ret_1 / vol_20 / mom_* / rev_*`、`ml\labels.py` 的标签、`ml\features.py` 的 155 个 TA 列、`research_2026-09` 的全部研究、以及**生产 `signal_bridge` 的实盘信号**；
4. 单次 2:1 拆股就会在该标的上制造一个约 **-50% 的假单日收益** —— 而 `factor\backtest.py:163` 的脏价格守卫阈值是 `r <= -0.9` 或 `r > 1.0`，**-50% 抓不到**。这与 P1-8（复权拼接）是同一个"守卫盲区"的两种来源。

**侧证**：`ml_2026-09-29.txt:236/458/504` 反复出现 `[factor.backtest] 价格卫生守卫: 剔除 2 个脏价格格` —— pkl 里确实存在脏价格，且**只有回测端在清洗**，特征/标签侧不清洗（见 P1-9）。

**修复**：
```python
217  df.loc[split_n1_idx, 'adj_factor'] = 1 / df.loc[split_n1_idx, 'split_n1']   # 改一个词
```
**并且必须**：
1. 补一个拆股单测（构造 t+1 日 2:1 拆股，断言复权后 `|pct_change()| < 0.5`）—— 这个 bug 能潜伏至今，就是因为没有这个测试。
2. **重跑 pkl 生成**，然后评估影响面：受影响的标的有哪些、那些标的的因子值/回测结论要重算多少。（这是"结论是否需要推翻"的问题，不只是"代码要改"。）
3. 把 `_sanitize_prices` 的判据从"|收益| > 100%"放宽到能覆盖**单次拆股量级**的跳空（如配合"成交量/振幅正常"做双重判据），并下沉到 `factor.data`，让特征/标签/回测共用同一个清洗入口（见 P1-9）。

---

## 4. P1：明确的缺陷与不一致

### P1-1 `combine_signals` 的分母与文档不一致（缺失感知实际失效）

**证据**：`factor\combine.py:218-250`

docstring 写：
> 缺失感知: 分子 = sum(w_i * sig_i)(逐格跳过 NaN), **分母 = sum(|w_i|) 只累加该格有效因子的权重**.

代码写：
```
244  num = num + s.fillna(0.0) * w                        # 分子: 逐格跳过 NaN ✓
245  den = den + s.notna().astype(float) * abs(w)         # 分母: 只累加有效权重 ✓
246  combo = num.div(den.where(den > 0))
...
248  total = sum(abs(float(weights[n])) for n in names)   # ← 无条件全部权重
249  combo = combo.where(den >= total - 1e-9)             # require_all 才用到
```

`den` 那行其实**是对的**（`s.notna()` 是 0/1 掩码）。问题在 `require_all` 的判据：`total` 是无条件全权重之和，所以 `den >= total` 几乎永不成立 —— `require_all=True` 实际上退化成"只要有一个因子缺失就整格作废"，而不是文档说的"所有因子都有效才给值"。语义上勉强算等价，但实现与注释描述的机制不同，且这种"注释说 A、代码做 B"的模式在有多个调用方时会变成真 bug。**结论：这是文档与实现不一致，需要统一（改注释或改代码），并补一个直接测试。**

### P1-2 `best_single` / `single` 方案忽略因子方向符号

**证据**：`factor\report.py:619-624`
```python
best = (max(sc, key=lambda n: abs(sc[n]) if pd.notna(sc[n]) else -1.0) if sc else None)
...
schemes['single'] = {best: 1.0}          # 权重恒 +1.0，忽略 sc[best] 的符号
```
`factor\backtest.py:692, 694` 同一模式：
```python
best = max(sc, key=lambda n: abs(sc[n]) if pd.notna(sc[n]) else -1.0)
combos[f'single:{best}'] = cmb.combine_signals(sigs, {best: 1.0}, min_cs=args.min_cs)
```

**后果**（实测证据）：`factor\output\hs300_clean\hs300\backtest_summary.csv`
```
single:rev_5,IS,1,2189,1.338,...,0.45,...          ← 勉强正
single:vol_60?  ← hs300 的 best_single 就是 vol_60, scores_is = -0.0352（强负 IC）
```
`factor_config.json` 显示 `hs300.best_single = "vol_60"`（`scores_is.vol_60 = -0.0352`）、`a_etf_all.best_single = "streak"`（`-0.0291`）、`company_300.best_single = "hi_252"`（`+0.0239`）。**5 个池里有 2~3 个的"最佳单因子"其实是强负相关因子，被以 +1 权重正向使用**，于是 `single:*` 那一行报告出来的收益是"反向使用"的结果。

这不是实盘 bug（生产用 `cd_ic`，其 `ic_weights` 是带符号的），但它是**报告层的误导**：`factor_config.json` 会被当接口读，`best_single` 字段的字面含义是"最好的单因子"，实际含义是"|IC| 最大的因子（方向未处理）"。

**建议**：`best_single` 改为带符号导出，或同时导出 `best_single` / `best_single_signed_weight`（= `sign(rank_ic)`），并在报告表里显示 `single` 方案的实际权重符号。至少把字段改名为 `max_abs_ic_single` 以免误读。

### P1-3 `wf_step3_sens.csv` 里 `sh_full` 与 `full` 不可比（疑似列污染）

**证据**：`factor\output\wf_step3_sens.csv`，同一行两列记录同一个量却差两个数量级：
```
pool,part,scheme,dim,value,n_w,agg,fold_obj,is,oos,full,ytd26,sh_full,full_ret,gap_is_oos
company_300,folds,cd_ic,folds=4|agg=mean,4.0,5.0,mean,2.1225,2.11,1.13,1.79,0.6,234.824,234.824,0.98
company_1000,folds,cd_ic,folds=6|agg=mean,6.0,5.0,mean,1.35,1.56,1.53,1.56,1.14,281.689,281.689,0.03
hs300,folds,cd_ic,folds=4|agg=mean,4.0,8.0,mean,1.34,1.42,1.79,1.51,0.81,34.49,34.49,-0.37
```
按 `factor\wf_step3_sens.py:99-100` 的赋值，`full`（低）与 `sh_full`（高）**应该来自同一个表达式** `m['full']['sharpe']`。同一行两个值不同 → 该 CSV 与当前代码不自洽，极可能是早期版本产出（或被外部脚本改写）。
另外 `gap_is_oos`（列尾）与 `_table()` 里 `df['is'] - df['oos']` 一致（`2.11-1.13=0.98` ✓），但 `fold_obj`（2.1225）≈ `sh_full`（234.824）毫无关系 —— 说明 `full` 列与 `is/oos` 列可能取自不同运行。

**风险**：如果 `sh_full` 被用作"哪个方案更好"的判据（`wf_step3_sens.py:207-215` 的 `d_sh_full` 对照就在这么用），那么用 234 和 1.79 比较会选出完全错误的方案。

**建议**：
1. 确认该 CSV 是哪个版本的产出；用当前代码重跑一次，比对是否复现。
2. 在所有产出 CSV 里加一列 `code_version`（git commit hash）和 `generated_at`，杜绝"不知道这份表是哪版代码跑的"。
3. `_measure` 里让 `full` 和 `sh_full` 字面上取同一个变量，避免将来再次分叉。

### P1-4 年化换手 135~257x，意味着策略被成本主导

**证据**：`factor\output\hs300_clean\hs300\backtest_summary.csv`
```
equal,IS,41,2189,...,sharpe 0.34,-0.633,ann_turnover 135.9,...,avg_days 3.8
ic,IS,41,2189,...,sharpe -0.26,...,ann_turnover 229.2,...,avg_days 2.2
shuffle(equal),IS,41,...,ann_turnover 456.4,...,avg_days 1.1
```
年化单边换手 135~257 意味着**平均每天换掉 54%~100% 的仓位**，`tier` 加权 + `top_k=5` + 日频调仓是高频摩擦策略。按 10bps 单边计，年成本拖累 = `ann_turnover × 10bps` ≈ 13.6%~25.7%。`company_300` 在 `signal_bridge` 报告里年换手 161.6x，报告自己算的年成本拖累约 16pp —— 也就是**扣费前 89% 的 CAGR 里有 16pp 是手续费**。

这不是 bug，但它是这类策略的核心脆弱点：
- 10bps 对美股大盘股或许够，对 A 股 ETF 可以（ETF 免印花税、佣金低），但对 `company_1000` 里的小盘股**不够**（还要加冲击成本、价差、借券费）。
- `run_engine` 的成本模型是 `equity *= (1 - turnover * cost_rate)`，一个乘性因子，**不建模冲击成本、不建模成交量约束、不检查流动性**（`backtest.py:322-327`）。`top_k=5` 却要在 276/298 只标的池里每天换仓，实际能不能成交、以什么价成交，回测没有回答。

**建议**：
1. 成本敏感性必须成为**标准输出**而不是可选：对每个方案跑 `cost_bps ∈ {5, 10, 20, 30, 50}` 的阶梯（`factor\output\a_etf_all\cost_ladder.csv` 这个文件说明你已经做了，把它提升为报告标配）。
2. 在引擎里加**成交量约束**：单日成交额不超过该标的当日成交额的 x%（如 5%），超出部分顺延或截断。哪怕是很粗的实现，也比无限流动性假设强。
3. 提高 `band`（当前默认 0.05）或加"最小持仓天数"来压换手，然后重新看 OOS。

### P1-5 样本外窗口偏短，Sharpe 点估计不可靠

**证据**：
- `etf_3x`/`hs300`/`a_etf_all`：`is_window = 2021-01-01 ~ 2024-12-31`（4 年），`oos_window = 2025-01-01 ~ None`（**约 21 个月，且随研究推进不断被"看"**）。
- 报告里的 OOS Sharpe 经常在 1.1~2.6 之间跳动（`astock_factor_strategy_20260922.md` §2：同一因子 `X_ovnshare60` 在 a_etf_all 上 `IS1 0.57 / IS2 2.11 / OOS 1.15`）。
- 你自己已经识别了这个问题的另一半："IS 双半窗能防'单窗过拟合'，但**防不了'只在熊市有效'**"。

**补充一个你报告里没写的点**：OOS 是 2025-01 至今，而每写一份新报告（0920 / 0922 …）都在同一个 OOS 上看结果。**这个 OOS 已经被偷看多次**，它的"样本外"属性在逐渐消耗。多重检验校正（BH-FDR）只校正了**一次评估内部**的因子族，不校正"跨报告反复看同一段 OOS"。

**建议**：
1. 把 OOS 冻结：声明"OOS 段每季度最多看一次"，并记录每次看的日期与当时的方案。或者留出一个**从未被看过的 holdout**（如最后 12 个月），仅在最终决策时打开一次。
2. 报告里除了点估计，给出**置信区间或 bootstrap 分布**。21 个月的 Sharpe 1.26，其 95% 区间大概会跨到 0 附近 —— 让人看到它有多不确定。
3. 用 `deflated Sharpe ratio` / `White's Reality Check` 之类考虑"搜索了多少个候选"的指标，替代裸 Sharpe 做方案选择判据。你的 `fold_obj`（折外 Sharpe）已经是朝这个方向的尝试，但它仍然是裸 Sharpe。

### P1-6 佣金/费用参数在系统内至少有三个不同的值

**证据**：
- `bc_trader.py:364` `signal_trade(..., trading_fee: float = 5, ...)`
- `bc_trader.py:153` `get_affordable_quantity(..., trading_fee: float = 3)`
- `bc_trader.py:452` 用 `money_per_sec - trading_fee`（=5）算买入数量
- 回测层用 `COST_BPS = 10.0`（单边 10bps），与上面两者都不同
- `signal_bridge_pool_performance_20260923.md:22` 写"单边 10 bps"

`get_affordable_quantity` 是按**金额**扣费（`trading_fee=3` 元/笔？），`signal_trade` 也按金额扣（5），而回测按 bps 比例扣。三处口径不同、默认值不同，且没有任何一处集中定义。

**建议**：建立单一的费用模型模块（`fees.py`），按市场/券商/账户类型定义（美股按股数或按笔、A股佣金+印花税、港股印花税+交易费），回测与实盘共用同一实现。`bc_util` 里已经有两个 `setup_logging` 级别的"统一入口"先例，费用应该照办。

### P1-7 数据下载层的具体缺陷

**证据**：

| 位置 | 问题 |
|---|---|
| `bc_data_io.py:49` | `default_eod_key = 'OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX'` —— API key 硬编码进源码并入库。虽然这是 EODHD 的公开测试 key（与 `~/quant/api_key/api_key.json` 的 `eod_test` 相同），但**模式是错的**：真实 key 也在同一份 json 里，一旦有人照抄就会把真 key 写进代码 |
| `bc_data_io.py:199-206` | `get_code_map_from_ak()` 用 `os.listdir()`（无参数）在当前工作目录里找 `updated_us_spot_em_*` 标记文件并**删除过期文件**。行为依赖 CWD；且它会删除工作目录里任何匹配该前缀的文件。这是"在别人的目录里删文件"的隐患 |
| `bc_data_io.py:159, 334, 383, 396` | 同一函数里混用 `logger` 和 `print`，且 `print(..., end=', ')` 用于进度显示。日志在长流程里会错行 |
| `bc_data_io.py:851` | `if len(tmp_data) > 0:` —— `get_data(...)` 在失败时返回 `None`（`L556`），`len(None)` 会抛 `TypeError`。同类问题在 `L928, 946, 1398` 等处重复（`benchmark_data.index.max()` 未判空） |
| `bc_data_io.py:1737` | `validate_standard_dataframe` 的返回值被用在 `if` 里（正确），但 `L509-510` 的调用点**丢弃返回值**，只留一条 warning 日志，坏数据继续流向调用方 |
| `bc_data_io.py:1488+` / `1367+` | `update_stock_data_from_eod`、`update_stock_data_from_ak`、`update_stock_data_new` 三个函数大量复制粘贴（realtime 合并那段几乎逐字重复三遍），修一处要改三处 |
| `factor\config.py:24` | `PKL_DIR = HOME / 'quant'` —— 数据路径硬编码为 `~/quant`，与代码仓库 `~/git/quant` 分离。虽然实际数据确实在那，但没有环境变量或配置文件覆盖入口，换机器/换目录就要改代码 |
| `bc_technical_analysis.py:115` | `config['result_path'] = config['quant_path'] / 'ta_model'` —— 路径由调用方传入的 dict 拼装，`ta_config.json` 是实际配置源，这个"配置的配置"链路没有 schema 校验 |

### P1-8 复权数据的一致性风险（需要你确认）

`bc_data_io.py:487-493`：ak 数据源的处理是
```python
df = df.rename(columns={... '收盘':'Close' ...})
df['Adj Close'] = df['Close'].copy()      # ← 前复权价直接当 Adjust Close
```

`get_data_from_ak(..., adjust='qfq')` 默认取**前复权**。前复权价的特性是：**每次分红送股，整段历史价格都会变**。而增量更新逻辑是 `start_date = 上次最后日期 - 3 天`（`L938`、`L1069`），只重下最近几天，然后

```
data[symbol] = pd.concat([data[symbol], new_data])
data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last')
```

**这意味着：本地历史是"几次不同时点的前复权价"拼接而成。** 举例：昨天收盘价 10 元，今天除权（10 送 10），前复权后昨天变成 5 元。今天增量更新的窗口是"昨天 -3 天"，只覆盖最近 4 天，所以本地第 5 天以前的旧价格仍然是**除权前的复权基准**，与最近 4 天不可比。结果就是历史序列里出现一个凭空跳空 —— 而这个虚假跳空会被因子（如 `ret_1`、`vol_20`、`mom_12_1`）当成真实收益。

`research_2026-09\exec_price_ab_test.py` 和 `factor\backtest.py:140` 的 `_sanitize_prices`（"研究 pkl 里偶发脏行……真实价 ~4.5 却被写成 ~9e5"）说明**你已经遇到过脏价格**，但那个守卫的判据是"单日 |收益| > 100% 或 <= -90%"。**前复权拼接造成的跳空通常只有 5%~50%，不会触发这个守卫。**

**需要你核实**：
1. `~/quant/stock_data/*.csv` 里的历史序列，是否存在"某日跳空但成交量正常"的可疑点？
2. 复权基准变化后，你的增量更新是否应该**全量重下**该标的？

**建议**：
1. 增量更新时，若检测到"新下载数据与本地重叠窗口（-3 天）的价格不一致"（同一日期同一列价格差异 > 0.1%），说明复权基准变了 → 触发该标的**全量重下**。这是低成本、高收益的守卫。
2. 或者干脆存不复权价 + 复权因子分两列，复权在读取时计算（专业做法，但改动大）。
3. 在 `_sanitize_prices` 之外，再加一个"复权跳空检测"：若某标的某日 `|跳空| > 8%` 且**同日成交量、振幅都正常**，标记为可疑并告警。

---

## 5. P2：可维护性与工程化

### P2-1 巨型模块与重复实现

| 文件 | 行数 | 问题 |
|---|---|---|
| `bc_technical_analysis.py` | ~7600 | 单文件承载 100+ 指标计算 + 30+ 绘图函数。`calculate_ta_static`（L342）单函数约 400 行，`calculate_ta_signal`（L930）约 400 行 |
| `bc_data_io.py` | 2858 | 数据下载 + 邮件 + config 原子写 + pickle 管理 + csv 读写全部混在一个文件 |
| `bc_trader.py` | 1218 | 抽象基类 + Futu + Tiger 混在一个文件；`Tiger` 类同时负责下单、市场状态、交易时间、idle 调度 |

指标实现有**三份**并存：`bc_technical_analysis.py`（老体系，`add_*_features`）、`factor\factor.py`（新体系，`@register` 注册表）、`ml\features.py`（"ML 特征"，`from quant.factor import factor as fct` —— 这一份是复用 factor 的，做法正确）。前两份的 `mom/vol/rev` 口径是否一致，没有测试保证。`repro_old_presets.py` 的存在本身就是证据：它需要"逐字照抄 git HEAD 公式"来复现老体系的因子，说明**新旧两套因子的等价性无法自动验证**。

### P2-2 测试覆盖严重不足

- `quant\tests\` 只有 **1 个文件、14 个断言**，覆盖 `bc_data_io` 的一部分（config 原子写 / 本地加载 / 邮件 / ak 容错）。
- **零测试**：`bc_trader.py`（下单！）、`bc_technical_analysis.py`（100+ 指标）、`factor\*`（整个研究流水线）、`ml\*`。
- 测试是**脚本式**的（`check(name, cond)` + 全局 `results` 列表 + 末尾 `ALL N TESTS PASSED`），不是 pytest。好处是零依赖、可直接双击；坏处是没有 fixture、没有参数化、不能单独跑一个用例、CI 里只能靠解析 stdout。
- 测试文件顶部注释写着绝对路径 `C:\Users\northcheng\.venv\Scripts\python.exe`，说明"必须用 venv 的 python"这个约束没有被工具化（没有 `pyproject.toml`、没有 `venv` 激活脚本、没有 CI）。
- 没有任何 CI 配置（无 `.github/workflows`）。

**最高优先级的补齐顺序**（按"出错代价 / 实现成本"排序）：
1. `factor\backtest.py` 的 `run_engine` —— 你已经有 `verify_backtest` 的 oracle/shuffle/投毒不变量，把它接进 pytest 就是现成的强测试。
2. `factor\combine.py` 的 `combine_signals`（P1-1 的不一致）、`dedupe`、`ic_weights` —— 纯函数，容易测。
3. `bc_trader.py` 的 `signal_trade` —— 用 mock trader（不发单）测"失败不扣现金"、"已持仓不重复买"、"现金不足停止买入"。
4. `bc_technical_analysis.py` 的指标 —— 用一只标的的固定输入 + 手工计算的期望值，先覆盖 `rsi/atr/macd/bb`。

### P2-3 依赖管理

`requirements.txt` 是 `pip freeze` 的全量快照：
- 142 行里包含 `jupyterlab`、`ipykernel`、`nbconvert`、`pyzmq`、`terminado`、`bleach`、`webcolors`、`json5`、`jupyter-lsp` 等**与项目无关**的 Jupyter 栈。
- 没有区分**直接依赖**与传递依赖 —— 无法判断哪些包删掉会坏。
- `vectorbt` 无版本号（第 137 行），且它的依赖闭包 `numba` / `llvmlite` **在 142 行里完全不存在**。而 `quant\__pycache__\backtest_etf_3x_vbt.cpython-312.pyc` 的存在说明 vectorbt 确实被用过 —— 也就是**这份 freeze 装不出当前环境**。
- **`ml` 的依赖确实缺失**（我最初按"纯 numpy 实现"的判断是错的，已修正）：`ml\models\gbdt.py:118,253` 有 `import lightgbm as lgb`，`ml\models\neural.py:134,227` 与 `ml\models\sequence.py:201,305` 有 `import torch`（都是**函数内延迟 import**，所以 `import quant.ml` 不会立刻失败，跑到 GBDT/神经网络路径才炸）。`requirements.txt` 里**没有 lightgbm、没有 torch**。
  - 修正一点：**没有** sklearn / xgboost 依赖。`ml\models\linear.py:23` 明确写"岭回归闭式解(纯 numpy, 不引入 sklearn)"。缺的就是 **lightgbm + torch**。
  - 延迟 import 这个设计本身是好的（不装 torch 也能用线性/因子部分），但它把"依赖缺失"从**导入期错误**变成了**运行期错误** —— 与 P0-0 是同一类问题的两种表现。
- `numpy==2.4.4` + `pandas==3.0.2` 与未固定的 `vectorbt` 共存 → 新环境很可能拉到不兼容组合（未实测）。
- 没有 `requirements-dev.txt`、没有 lock 文件、没有 `pyproject.toml`。

### P2-4 仓库卫生

- **`quant\.gitignore` 只有 9 行，且它管不到真正需要管的东西。** 文件位于 `quant\` 内，所以所有规则都锚定在 `quant/` 下：
  - `research/output/` 只能匹配 `quant/research/output/` —— 而这个目录**已经不存在了**（脚本搬走后留下的空规则）。
  - **真实产物在兄弟目录** `research_2026-09\output\`：**995 个文件，占全树 1439 个文件的 69%**。子目录里的 `.gitignore` 在原理上也管不到兄弟目录。
  - `quant\factor\output\`（163 个文件：logs / CSV / PNG）同样不在规则覆盖内。
  - 缺 `*.pkl`、`*.png`、`Thumbs.db`、`*.xmind`、`*.zip`、`*.json`（模板除外）。
  → 全树实测：**822 个 `.csv`、231 个 `.png`、168 个 `.txt`、11 个 `.pkl`、95 个 `.pyc`**。
  → 你也察觉到了：报告里引用的路径是 `c:/Users/northcheng/git/quant/research/output/pool_perf_signalbridge.csv`，而实际文件在 `git/research_2026-09/output/`。**目录改过名，ignore 规则和文档路径都没跟着改。**
- **95 个 `.pyc` 残留**（`quant\` 75 + `research_2026-09\` 20），跨 **5 套解释器**：`cpython-36`、`-37`、`-38`、`-39`、`-312`。其中包括**已被移走的模块**的字节码（`bc_factor_search`/`bc_combo_search`/`bc_backtest` 的 `.pyc` 还在 `quant\__pycache__\`）—— 这正是 P0-0 断裂的"考古证据"。
- 不该入库的东西（已核实存在）：`quant\Thumbs.db`、3 个 `.xmind`、`git\quant-2026-09-17.zip`（整个包的 zip 躺在工作树根）、`quant\.ipynb_checkpoints\bc_technical_analysis-checkpoint.py`（全树 **0 个 `.ipynb`**，这条 ignore 规则只在保护 1 个 py 副本）。
- **源码被当成产物存了一份**：`research_2026-09\output\_baseline_pre_split\` 下有 4 个 `.py`（`bt_core.py`、`score_backtest.py`、`signal_backtest.py`、`signal_timeseries.py`），与 `research_2026-09\` 顶层同名文件重复。这会造成"我改的是哪一份"的经典混乱 —— 而 P0-0 的断裂恰好就是这类分叉的后果。
- `quant\__pycache__\` 里有 **Python 3.6 / 3.7 / 3.8 / 3.9 / 3.12 五个版本**的 `.pyc`（`bc_technical_analysis.cpython-36/37/39`、`bc_machine_learning.cpython-38` 等），说明这个项目跨了整整 6 年的解释器版本。`.gitignore` 里虽然有 `__pycache__/`，但工作区里还躺着这些文件。
- 被删模块的 `.pyc` 残留：`bc_factor_search`、`bc_combo_search`、`bc_backtest`、`bc_finance`、`bc_futu`、`bc_tiger`、`bc_machine_learning`、`bc_trade_simulation`、`signal_mine_*`、`regime_symbol_wf`、`champion_daily`、`backtest_2026_log` —— **这些 .py 已经不存在，但 .pyc 还在**。它们的存在会误导审查（我一开始就以为 `bc_backtest.py` 存在）。
- 不该入库的文件：`quant\Thumbs.db`、`git\quant-2026-09-17.zip`（一个仓库快照压缩包放在仓库里）。
- 三个 `.xmind` 文件（`TA_structure.xmind`、`ta_signal.xmind`、`ta_data_columns.xmind`）是设计文档，有价值，但二进制格式无法 diff/评审。
- `quant\factor\output\` 下的产出（`*.csv`/`*.png`/`report_summary.txt`，含 5 个池 × 3 张图）**被跟踪了**。研究报告产出该不该入库是个取舍，但至少要一致（现在 `.gitignore` 忽略了 `research/output/` 却没忽略 `factor/output/`）。

### P2-5 敏感文件

**在工作树内（需要处理）**：
- `quant\bc_data_io.py:49` `default_eod_key = 'OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX'` —— EODHD key，硬编码且作为 **8 个公共函数的默认参数值**（`:293 / :515 / :559 / :708 / :734 / :756 / :1367 / :1623`）。该 key 被插值进 URL（`:319 / :341 / :373 / :594`），所以**它会随请求串进入日志**。
  - 已核实：这与 `~/quant/api_key/api_key.json` 里的 `eod_test` 是同一个值（EODHD 的公开测试 key）。**但真正危险的是旁边那个**：同一份 json 里的 `eod` 字段是**真实可用的 key**（本文档已脱敏，不在此处复现字面值）。任何人照抄 `default_eod_key` 这个模式，就会把真 key 写进源码。
- `quant\bc_data_io.py:2631` `from_addr='northcheng@qq.com', smtp_server='smtp.qq.com'` —— 邮箱与 SMTP 硬编码（SMTP 授权码走 `:2851 config['api_key']['qq']`，不在工作树内，这点是对的）。
- `quant\portfolio.json` / `quant\position.json` —— **真实账户净值与持仓**（详见 P0-5）。
- `quant\ta_config.json` —— 池定义 + 中英标的名称表 + 各平台 `init_cash` / 止损止盈阈值。我逐字读过：**不含**账号、邮箱、token。
- 已核实：**工作树内没有任何 `*.json` 含密钥字段**，也没有 `.env` / `*.pem` / `*.key` / `*.crt`；代码里读 `os.environ` 只有 6 处，且都是 `MPLBACKEND` / `STRAT_SILENT_INIT` 这类非敏感开关。**凭据管理在"不留在仓库里"这一点上是对的。**

**在工作树外（好）但缺乏规则保护**：
- `~/quant/api_key/api_key.json` —— 7 个数据源密钥，其中 `iex_sk`（secret key）、`fred`、`marketstack`、`snowball`（`xq_a_token=...`）看形态都是**真实可用**的 key，不是 test key。
- `~/quant/trader/rsa_private_key.pem` —— **券商 API 私钥明文**（富途/老虎用 RSA 签名）。**这是整个项目最高价值的凭据。**
- `~/quant/trader/user_info.json`、`user_info.csv`、`trade_condition.json` —— 券商账号信息。

**建议**：
1. 立刻为 `bc_data_io.py` 的 `default_eod_key` / 邮箱 / SMTP 做去硬编码：这些函数已经有 `api_key` 参数，改成 `None` 默认 + 从仓库外配置读即可，**不要留任何凭据作为默认值**。
2. `rsa_private_key.pem` 与 `api_key.json` 目前"安全"只是因为物理位置在 `git\` 之外 —— 这是**脆弱的保证**。写进 `.gitignore`（`api_key/`、`*_key.pem`、`*user_info*`、`trade_condition.json`、`portfolio.json`、`position.json`），并给 `api_key.json` 一个 `.example` 版本。
3. 检查 `~/quant/logs/*.txt` 与 `factor\output\logs\*.txt` 里有没有因为 URL 插值而落盘的 key。
4. 因为工作树现在**没有 `.git`**（P0-1），这次不需要 `filter-repo` 清历史 —— 只要在建新仓库前把 `.gitignore` 写对。这是"因祸得福"的一点。

### P2-6 两套并存的日志与两套并存的口径常量

- 日志：`bc_util.setup_logging()`（老体系，挂 root，`_ROOT_DONE` 用模块级 `len(root.handlers)==0` 判幂等）与 `factor\config.get_logger()`（新体系，用模块级 `_ROOT_DONE` 标志）。两者都往 root 挂 handler，**同进程混用会双写**（一个文件 handler + 一个文件 handler，日志文件里每条出现两次）。
- 口径：`factor\config.py` 的全局 `TRADE_START/IS_END/OOS_START` 与 `factor_config.json` 里**逐池**的 `is_window/oos_window` 并存且值不同（见 P0-3）。`repro_old_presets.py:90` 的 `WINDOWS` 又是第三份（`('oos', cfg.OOS_START, None)` —— **没有用 `OOS_END`**，所以 `OOS` 窗实际上一直延到数据末尾，与 `factor_config.json` 的 `oos_window=[2020-01-01, 2023-12-31]` 矛盾）。
  → 这直接影响结论：`etf_3x` 的 OOS 到底是 `2020-2023` 还是 `2020-至今`？两种口径下的 Sharpe 不可比。而 `repro_old_presets` 是用来"复现旧报告"的，口径漂移会让复现失败。

**建议**：口径常量单一来源化。把每池的窗口写进 `factor_config.json`，`config.py` 只保留 `DEFAULT_*`，所有 `WINDOWS` 从配置构造。

### P2-7 代码风格

- 缩进：主包（`bc_*`）与 `factor`/`ml` 全部用 **2 空格**。Python 社区标准是 4，2 空格会让 `flake8` 的 `E111` 报警，且嵌套深时可读性下降（`bc_technical_analysis.py` 里有些块嵌套 6 层 = 12 空格）。这是一致的（好），但偏离生态（坏）。
- 无 `flake8`/`ruff`/`black`/`isort` 配置。考虑到文件规模（7600 行），加一个 linter 会自动抓出一批真实 bug（未使用变量、`== None`、可变默认参数等）。**`bc_trader.py:452` 那类"返回值未判断"的问题，`ruff` 不一定抓得到，但 `_fast_query` 那种复杂代码里的未使用变量能抓到。**
- 语言：docstring 中英混杂（老模块是英文 Google/Epydoc 风格 `:param: :returns: :raises:`，新模块是中文）。新模块的中文 docstring 质量明显更高（讲清了"为什么"），建议统一到中文 + 保留 `:param:` 结构。
- `bc_util.py` 里 `print_when`、`plot_data` 等函数使用了过时的 API（`plt.gcf()`、`plt.xticks(rotation=90)` 作用于全局状态）。

---

## 6. 对研究结论的独立评价

你把我放进"评价"的位置，所以我直说几点对**研究结论本身**的看法（不是代码问题）：

### 6.1 我认同你"不改生产"的决定

`factor\output\hs300_clean\hs300\backtest_summary.csv` 是这个决定最硬的证据：

| kind | window | total_ret | vol | sharpe | max_dd | ann_turnover |
|---|---|---|---|---|---|---|
| equal（41 因子） | OOS | 0.794 | 0.424 | 0.57 | -0.456 | 138.6 |
| buyhold（276 只等权） | OOS | 0.802 | 0.200 | 0.87 | -0.235 | 1.1 |
| greedy / single:rev_5 | OOS | 0.288 | 0.354 | 0.36 | -0.530 | 190.3 |
| ic（IC 加权） | OOS | **-0.466** | 0.266 | -0.48 | -0.598 | 257.8 |
| shuffle(equal)（安慰剂） | OOS | -0.887 | 0.269 | -1.97 | -0.905 | 474.9 |

**因子体系在 OOS 没跑赢等权持有，却承担了 2 倍波动和 2 倍回撤。** `ic` 加权甚至是负的。安慰剂（shuffle）显著为负说明引擎没坏、信号确实"有点东西"（-0.887 vs -0.466），但这点"东西"不足以覆盖成本与波动。**在这种情况下不改生产是完全正确的判断**，比"因为我研究了三周所以要上线"专业得多。

### 6.2 `research_2026-09` 的结论（hs300 Sharpe 1.39）与 `factor\` 的结论（0.57）矛盾，需要解释

- `signal_bridge_pool_performance_20260923.md`：hs300 全期 Sharpe **1.39**、换手 76.1x、平均持仓 7.1 天。
- `factor\output\hs300_clean\hs300\backtest_summary.csv`：OOS Sharpe **0.57**、换手 138.6x、平均持仓 3.8 天。

两套系统都自称"与生产口径一致"，但**换手差 1.8 倍、Sharpe 差 2.4 倍**。差异来源可能是：
1. 时间窗不同（1.39 是 2021-2026 全期，0.57 是 OOS 2025+）—— 但 `signal_bridge` 报告的 OOS 也是 1.26，仍与 0.57 差很远。
2. 因子集不同（`signal_bridge` 用 3 个 `A_POOL_PRESETS` 因子 + `tier` 加权；`factor\` 用 41 因子等权 + `equal` 加权）。
3. 引擎不同（`bt_core.BacktestKit` vs `factor\backtest.run_engine`）—— **两套引擎并存**，且没有任何交叉验证证明它们对同一输入给出同一输出。

**这是我最担心的一点**：`bt_core`/`score_backtest`（生产回测引擎，不在 git）与 `factor\backtest`（研究引擎，在 git）是两套独立实现。**你所有"生产表现"的数字来自前者，所有"因子有效性"的结论来自后者。** 两者从未被证明等价。

**建议（这是我认为性价比最高的一项工作）**：
写一个**引擎对拍测试**：给定同一份 `open_w/close_w/composite/EngineParams`，要求 `bt_core` 与 `factor.backtest.run_engine` 输出的净值曲线逐点一致（或差异 < 1e-9）。如果做不到，找出并解释每一处差异。这一项工作可能只要一天，但它能消除"研究结论和实盘表现为什么对不上"的根本不确定性。

### 6.3 边际递减的结论我认同，但"没有 alpha"的前提要小心

你的报告反复出现"OHLCV 内单因子维度已彻底挖尽"、"四轮边际递减"。这个结论在你**已搜索的空间**内是对的，但要注意三个未搜索的维度：

1. **多标的/截面之外的信息**：现在全部因子都是单标的时序量 + 截面排序。没有用到"标的间关系"（协整、领先滞后、行业/主题联动）。`a_etf_all` 池里有巴西 ETF、黄金 ETF、纳指科技 ETF —— 这些标的间存在明显的宏观联动，可以构造配对/轮动信号。
2. **横截面之外的时序择时**：`ml\timing.py` 的存在说明你已经在探索，但它似乎没有进入主流水线。
3. **频率**：全是日频。你的池子（杠杆 ETF、ETF）流动性好、交易成本低，**日内/分钟级**的均值回复与量价背离可能还有空间 —— 而这恰好是这套高换手策略（日均持仓 3.8 天）勉强能触及的领域。

### 6.4 一个方法论上的观察

你的报告里"否决改进"的记录非常详细（"OOS 从 1.26 崩到 0.56，被否决"），这是好习惯。但我注意到**多个池的胜出方案都退化成单信号**（`astock_factor_strategy_20260922.md` §4："两池各自的 obj 榜首都是单信号…贪心在第 1 轮就停止（无新增能让 obj 提升 > margin 0.1）。→ A 股两池**没有多因子协同**"）。

这有两种解释：
- (a) 这些池子里确实没有多因子协同 —— 那 `A_POOL_PRESETS` 里的 3 因子（`N_range20:-1, D_ma20_dist:-1, C_tmqmom:0.5`）的"组合价值"就只是 regime 对冲，而你报告的 §5.1 已经给出了这个解释（"组合收益不是成分 alpha 的线性叠加，而是'低波/超卖打分 + 动量质量再排序'的交互"）。
- (b) 你的贪心停止判据（`min_gain = 0.1` 的 Sharpe 提升）太严，或目标函数（折外 Sharpe 均值）噪声太大，导致真协同被噪声淹没。

**建议验证 (b)**：把 `min_gain` 从 0.1 放到 0.02，看贪心是否会选出多因子组合、以及那个组合的 OOS 是否稳定。如果会，说明是判据问题；如果不会，你的 (a) 结论就站得更稳。

---

## 7. 建议的改进顺序（按投入产出比）

| 优先级 | 事项 | 预计工作量 | 理由 |
|---|---|---|---|
| 1 | 生产代码入库 + 密钥加 `.gitignore` | 2 小时 | 消除不可追溯风险；成本极低 |
| 2 | 修 `signal_trade` 的失败记账（P0-2） | 半天 | 直接影响真金白银；改动局部 |
| 3 | 生产 pkl 缺失改成 fail-fast（P0-3） | 2 小时 | 防止"用错数据产生信号" |
| 4 | `factor.backtest` vs `bt_core` 引擎对拍 | 1 天 | 消除两套引擎的口径不确定，是后续所有结论的基础 |
| 5 | 复权拼接守卫（P1-8） | 半天 | 可能解释一部分"脏价格"和异常因子值 |
| 6 | 统一费用模型 + 统一池窗口口径 | 1 天 | 消除多处参数漂移 |
| 7 | `verify_backtest` + `combine` 纯函数接进 pytest | 1 天 | 把已有的强自检变成回归测试 |
| 8 | 成交量约束 + 成本阶梯进报告标配 | 2 天 | 让高换手策略的可行性有真实答案 |
| 9 | 拆 `bc_technical_analysis.py` / `bc_data_io.py` | 3-5 天 | 长期可维护性；不紧急 |
| 10 | `.gitignore` 修正（`research_2026-09/output/`）+ 清 `.pyc` + 清 `Thumbs.db`/zip | 1 小时 | 仓库卫生 |

---

---

# 附录：四份并行深度审查的发现汇总

本附录汇总 4 个独立子代理对各模块的逐行审查结果（均与我本人的核实交叉验证；凡我亲自复核过的条目已在正文标注）。**未在正文展开的条目在此保留**。

## A1. 工程化与仓库（已整合进正文 P0-1/2/5、P2-3/4/5）

**已核实的新事实**：
- 全工作树 **1439 个文件**；`research_2026-09\output\` 占 **995 个（69%）**。
- **95 个 `.pyc`** 跨 5 套解释器（3.6/3.7/3.8/3.9/3.12）。
- **无** `setup.py` / `pyproject.toml` / `pytest.ini` / `conftest.py` / `.flake8` / `ruff.toml` / lock 文件 / `.github\`；全树无任何 `.yml/.yaml/.toml/.cfg/.ini`。
- `quant\__init__.py` **不存在** → `quant` 是 PEP 420 隐式命名空间包，所以"`pip install` 本包"这条路从来就不通，只能靠 `sys.path` 注入。
- **缩进系统性分裂**：`quant\` 一律 **2 空格**（126 处 class 方法），`research_2026-09\` 一律 **4 空格**。
- 测试是**纯脚本式**：模块级 `with tempfile.TemporaryDirectory()` + 自定义 `check()` + 文本哨兵 `ALL 14 TESTS PASSED` + `sys.exit(1)`。没有任何 runner 能单独跑一个用例。
- **关键空白**：因为全仓导入链是 `from quant import ...` 而**没有任何 import 冒烟测试**，P0-0 的断裂才能潜伏数天无人发现。

## A2. ML 包（15 个 .py）

**P0（数据泄漏风险）**：
1. `ml\config.py:62` `DEFAULT_FEATURE_SOURCE = 'both'` —— 默认特征 214 维里 **155 维（72%）是 pkl 预计算的 TA 列**，其因果性无法用截断一致性检验验证（`ml\features.py:15-20,64-67` 自己写明了这一点）。`ml\config.py:59` 也标注了"TA 列是 pkl 预计算值, 无独立生成器 → 因果性无法用截断检验验证"。
   - 实测差距（`ml_2026-09-28.txt:576-578` vs `:625-627`）：ridge h=5 用 `both` 的 IS RankIC 0.1494 / OOS 0.0625，用 `factor` 只有 0.1042 / 0.0447。**TA 把 IS 抬了 43%** —— 这个抬升正好落在"无法验证因果性"的那 72% 特征上。
   - 但 recent 回测**反向**：ridge(both) 0.935/Sharpe 0.72（`ml_2026-09-29.txt:247`）vs ridge(factor) 2.380/1.11（`ml_2026-09-28.txt:704`）。→ 结论方向不稳，不能靠 TA 提分。
   - 建议：默认改 `'factor'`；用 TA 必须显式 `--source` 并在报告抬头强制标注 TA 占比。
2. **序列模型的"排除 TA"守卫被移除（相对 09-28 是回归）**：日志 `ml_2026-09-28.txt:2175` 证明曾经有
   > `[predict] gru 为序列模型 -> 特征来源由 both 强制改为 factor(排除 TA: 其 shift(-1) 前视在长度 20 的窗口下被放大)`

   随后 F=59。而当前 `ml\predict.py:69-74` 已无此逻辑，`ml_2026-09-29.txt:493-495` 显示 gru/lstm 已在 `source=both`、**F=214**、L=20 下运行，日志自己打着"1 日前视被放大 20 倍"的 caveat。→ TA 让 gru 的验证 RankIC 从 0.0403（`09-28:2413`）升到 0.0659（`09-29:455`），**+63%**。
   - 建议：把守卫**下沉到 `seq.build_sequences`**（source 含 TA 且为序列样本时直接 raise），不要靠 CLI 纪律。

**值得记录的正面结论（与 factor 包形成矛盾，见 A4-P0-2）**：`ml\features.py:15-20` 明确区分"可验证/不可验证"，**拒绝给出虚假的 `causal=True`**。这是整个仓库对 TA 列最诚实的一段文字，而 `factor` 包对同一批列输出的是"全部因子因果"。

**P1（9 条，摘要）**：
- `requirements.txt` 全文无 `lightgbm` / `torch`（全仓只有 ml 包 import 这两个；ridge 是自写 numpy 闭式解，**不需要 sklearn**）→ 干净环境里 4 个模型全不可用。
- **IS/OOS 边界无 purge**：`ml\dataset.py:185-219` 的验证块末尾 = 2019-12-31，而标签用 `open(t+1)→open(t+1+h)`，所以最后 h+1=6 天的标签要用 **2020-01 的价格**；这些标签却参与早停轮数与 MLP/序列的超参选择 → 选参信息越界。
- **shuffle 对照臂无效**：`ml\models\neural.py:379-386`、`sequence.py:502-509` 把标签打乱后用**真实** `valid_fwd` 做早停判据 → 对照臂被人为抬高。日志形态吻合：只有 **MLP 的 shuffle RankIC 为正（+0.0112）**。
- **MLP 实际是"1 轮"模型**：`ml_2026-09-29.txt:409` `best_epoch=1`（`epochs=21/200`）；`09-28:1765` 搜参也选中 best_epoch=1 → 每次都判第 1 轮最优并回滚 ≈ 一次 SGD 步。**"DL 相比 GBDT 有没有增量"当前无法回答。**
- **无模型/特征清单持久化**：全包无 save/load，`ml\predict.py:178-179` 的文件名不含 `method/prep/F/L/seed`；同一天 factor 注册表已从 59→70、F 214→225（`09-29:530-532`），旧的分数 CSV 无字段记录 → **重跑即静默不可比**（这正是"训练/预测期特征错位"的土壤）。
- **对照臂不能判别能力**：`ml_2026-09-29.txt:462-471`（etf_3x, K=5/28 只）—— OOS gru 1.553/0.69 vs **shuffle +0.303/0.44**，recent gru 0.845/0.71 vs **shuffle 1.381/0.91（反超）**。而 `ml\predict.py:266-269` 自己的判据是"shuffle 应显著弱于模型"。IS 段 gbdt 159.2×/1.47 塌到 OOS −0.456 → **IS 净值不能作为性能证据**。

**ML 包的亮点**（与 factor 包一致，值得保留）：没有随机 KFold（时序切分 + `embargo = h+1`，且实际比严格需要多留 1 天，偏保守）；逐日截面 rank 不跨日、不拟合全局参数；`_col_fill`/`_col_fill` 的填补统计量只在 fit 段估（`linear.py:104-107`）；`predict` 强校验特征顺序（`linear.py:124`）；以及**"对检验本身做检验"**（`features.py:373-375,448-467` 注入"全样本跨日 rank"反例、要求必须被抓出；`timing.py:650-685` 用"截断到 2024-06-28 重跑，系数与 pred 逐值相同"同时证明"拟合只用 IS"与"全链因果"）。

## A3. 券商下单层（`bc_trader.py`，1218 行）

这一层是"真钱路径"，逐行审查后发现的 P0 数量最多：

**P0-A 富途市价单的资金检查被彻底绕过（走的是默认路径）** —— 已核实：
`bc_trader.py:714-719`（`price is None` → `OrderType.MARKET`，且 BUY 把 `price` 写成 `0.1`、SELL 写成 `1000000`）→ `:729` `get_affordable_quantity(symbol, price=price)` → `:167` `math.floor((available_cash - 3) / latest_price)`，价格变成 0.1 → 返回 `≈10×(cash−3)` 股 → **任何 `quantity` 都判为"买得起"**。
触发路径是默认的：`signal_trade(order_type='market')`（`:364, 457-458`）；`condition_trade`（`:499-500`）把 `trade_condition.json` 的 `quantity` 原样传入，**金额无上限**。

**P0-B 持仓查询失败被伪装成"空仓"** —— `:805-856`（Tiger）/ `:617-669`（Futu）在失败后 `self.position = result`，而 `result` 是空 DataFrame。连带两个后果：
- `get_in_position_quantity`（`:137-150`）返回 0 → `trade()` 卖出分支 `:744` `in_position_quantity >= quantity` 为假 → **止损/清仓单被静默跳过**（只留一行 "Not enough stock to sell(0/n)"）；
- `synchronize_position_record`（`:250-289`）在 `__init__` 每次执行，`self.position` 非 None（空表）→ `position_dict = {}` → 每个池内标的都命中 `:268` 的 `current_position != record_position` → `:271-273` 把 record 改成 `{'cash': init_cash, 'position': 0}`。**一次瞬时的 API 抖动就抹掉本地持仓台账并把现金还原成初始值。**

**P0-C 实盘止损/止盈永远不会触发**（这条我在正文 P0-4 只猜了一半，子代理给出了完整机制）：
- `bc_trader.py:640`（Tiger `:827`）`rate = round((latest_price - average_cost)/average_cost, 2)` → **rate 是小数比例**；
- `:530-533` `position.query(f'rate < {stop_loss_rate}')` → 配 `ta_config.json` 的 `-0.8`/`10` 就是 **-80% 才止损、+1000% 才止盈**；
- `:641`（Tiger `:828`）**`rate_inday = 0.0` 是硬编码**，只有 `get_briefs=True` 才重算，而 `cash_out(..., get_briefs=False)`（`:510`）是默认 → `rate_inday < -0.3` **永远查不到任何行**。
→ **结论：三个保护条件（止损、止盈、日内止损）在实盘上全部无效。**这是本次审查里对资金安全最直接的发现。

**P0-D 无幂等、不校验成交 → 重复下单**：`:384-391` 取一次持仓快照，`:431-465` 只要 `quantity == 0` 就买；刚提交的市价单不会立刻出现在 `position_list_query` 里 → **同一 bar 内重跑（cron 重试、手工补跑）会二次买入**。Futu `:732/:745` 传 `remark=None`、Tiger `:982/:995` 也没设外部订单号 → 没有去重键。下单后只把 `order_id` 拼进字符串（`:734/:747/:984/:997`），**不查询成交状态**。

**下单层 P1（选摘）**：
- `:130` `get_available_cash` 用 `asset.loc[0,'cash']`，而 Futu 侧已把 `avl_withdrawal_cash` 映射成 `available_cash`（`:685`）、Tiger 有 `available_funds`（`:870`）却都没用 → **用总现金而非可用资金做买入校验**，未交收资金被算进来。
- `:153-167` **不做最小交易单位/整手处理**（港股 lot size 100/500/1000）；`latest_price` 为 NaN 时 `math.floor(nan)` 抛 ValueError；`trading_fee` 默认 3 而 `signal_trade` 用 5（口径不一致，见正文 P1-6）。
- `:215-221` 只在 `new_cash >= 0 and new_position >= 0` 时更新台账，否则**只打日志** → 成交被永久丢弃，台账单向漂移且不自愈。
- `:572-593` 富途 `open_trade_client` 里**没有 `unlock_trade(password)`**，而 `:601` 只在 `close_trade_client` 里 `unlock_trade(is_unlock=False)` → 若无入口脚本补 unlock，**REAL 环境所有下单都会失败**；崩溃时也不会重新锁定。另 `:603` 在 except 之外调用 `logger.exception` → 日志里出现假的 "NoneType: None" 堆栈。
- `:947-961` Tiger `stock_contract(symbol=symbol, currency='USD')` **硬编码美元** → 港股/A 股不可用。
- `:488-490` `condition_trade` 把 `trade_condition.json` 的字符串**拼进 `df.query()`** → 表达式注入/任意求值（可用 `@` 访问局部变量），且每行重扫整个 DataFrame（O(n²)）。

**技术指标层的 P1（选摘，均有行号依据）**：
- `:217` 复权索引 bug → 已提升为正文 **P0-6**（我亲自核实）。
- `:246` `adj_rate = Adj Close / Close` 无除零保护 → Close=0 产生 ±inf；`:272` 的 `df.dropna()` **不会删 inf** → inf 一路进入指标。`:272/:280` 同时是"静默删数据"：任何一列 NaN 丢整行；只要最新一根 `Volume=0`（停牌）就 `df[:-1]` **丢掉最新 bar**。
- `:4550-4551`（TSI）`m.ewm(r).mean().ewm(s).mean()` —— **`ewm` 的第一个位置参数是 `com` 不是 `span`**，alpha=0.038 vs 本文件其余用 `span` 的 0.077 → TSI 与全文件口径不一致。
- `:4258-4281`（KAMA）`ER_den == 0`（一字板）→ `ER=NaN` → **NaN 永久传染到序列尾部**，`kama_fast/kama_slow` 全废。
- `:3923-3925`（ADI）/ `:3964-3966`（CMF）：`H==L` 且 `C!=H`（脏数据/停牌）时分子非零 → **±inf**，`.fillna(0.0)` 抓不到 inf → **整条 ADI 序列被污染成 inf**。
- `:3045` `if method in ('wilder')` —— `('wilder')` 是**字符串不是元组**，任何子串都走 Wilder 分支；且 `:3076-3082` 的 fillna 用 `locals().get(col)` 反射，而 wilder 分支从不写 `pdi/mdi/dx/adx` 到 df → **该 fillna 完全是空操作**。
- `:7430-7434`：`with ProcessPoolExecutor(...) as pool:` **覆盖了同名形参 `pool`（池名字符串）**，紧接着 `:7432` 把 executor 对象当 `pool` 参数传给 worker → 序列化失败 → 逐日绘图任务全部失败并被 `:7440-7441` 仅以日志吞掉 → 函数"成功"返回空结果。**该功能当前事实上不可用。**
- `:7157` `_safe_query_df(trade_info, f'code == "{title}" ...')`，而 `title` 是 `f'{symbol}({ed})'` 格式 → **成交标记永远匹配不上，图上不显示真实买卖点**（静默）。
- `:7246` 趋势行显示的是 `pattern_score` 的增量（`change_desc` 变量被复用）→ **报告正在给出错误的事实**。
- `:7626-7718` `ta_data_columns` 是**陈旧白名单**：含从未生成的列，却**不含 macd/cci/aroon/stoch/mfi/tsi/uo/wr/vortex/psar/kst/trix/stc/dpo/ao/eom/fi/nvi/obv/vpt/cmf** → 若下游按此选列，**整个动量族被静默丢弃**。
- `:7343/:7352/:7391` 等：`interval='year'` 在 `ta_config.json` 里没有对应键 → KeyError，而签名却声明 `Literal[..., 'year']`。

**下单层/指标层的亮点**：`rma()`（`:1722-1738`）是**真正的 Wilder 平滑**（前 n 个 SMA 作种子），RSI/ATR/KC/Wilder-ADX 确实用了它 → 核心指标公式标准（RSI 零涨跌取中性 50、TR/DM/DX 定义正确、除零有处置）。MACD（`em` 用 span）、Bollinger、Donchian/Stoch/UO/CCI/MFI/OBV/VPT/AO/Heikin-Ashi/EOM/Vortex 均对得上教科书。`normalize_causal`（`:1846-1875`）只用 trailing 窗口并说明为何不用 `normalize`。`_safe_query`（`:1466-1560`）用 AST 向量化快路径 + pandas 回退。`bc_trader` 的 `ACCOUNT_GROUPS`（`:31-43`）把"平台→账户→分组"契约集中并写清三方约定；`portfolio.json` "存在但不可读则跳过而不覆盖"（`:319-326`）是正确防御。

## A4. 因子研究包（`factor\`，16 个 .py，约 7300 行）

**P0（3 条，直接动摇研究结论）**：

1. **`least` 两套互斥的 IS/OOS 口径，且最终权重是用 OOS 数字选出来的。**
   - `factor\config.py:39-43`：IS 2011–2019 / OOS 2020–2023 / recent 2024+；
   - `factor\wf_strat_search.py:41-46`：**IS 2021-01-01~2024-12-31 / OOS 2025+**（注释却称"对齐 factor_config.json"）；
   - `factor\repro_old_presets.py:90-95` 的 `'oos'` 直接跑到数据末尾（**第三套**）。
   - 择优动作落在 OOS/FULL 上：`wf_step3_sens.py:148-165,200-216`（`gap_is_oos`/`d_gap`/`d_sh_full`）、`wf_step2_search.py:176-199`、`wf_combo_search.py:188-214`。连**折外目标的元参数**（4 折 vs 6 折、`mean` vs `min`）也是看 OOS 挑的（`wf_step3` → `wf_step4`）。
   - 冻结依据 `wf_step4_export_weights.py:1-9,28-31`（"Step3 结论：6 折+mean 最优"），而线上 `factor_config.json:13-20` 是 `is_window 2021–2024` / `oos_window 2025+`、`default_scheme = cd_ic`。
   - **后果**：`report_summary.txt` 把 2020–2023 标为 OOS，而线上 `cd_ic` 权重拟合于 2021–2024 → **报告的"样本外"对线上配置其实是样本内**。报告开头那句"权重只在 IS 期估计"不成立。
   - 建议：窗口单一真源；所有选择函数加 `assert window not in ('oos','recent')`；2020–2023 降级为开发验证集；2024+ 冻结为只看一次的 holdout；报告搜索配置总数与 deflated Sharpe。

2. **`verify_causal` 对 pkl 预计算列恒等失效，21 个 `H_*` 因子无因果证据却被报成"全部因子因果"。**
   - `factor\factor.py:892-934` 的截断一致性检验（截断前后逐值比对）对**包内自算的因子**有效；但 `factor.py:446-492` 的 `H_*` 族全部消费 pkl 的 `adx_*/atr/rsi/kama/ichimoku_*/candle_*/trigger_score/pattern_score` 等**预计算列**。`ds.slice()` 只截行不改值 → **截断前后逐值相等是恒等式，检验形同虚设**。
   - `factor\factor.py:726-728` **代码自己写着** candle_* "带前视嫌疑(bc 预计算)"；`:960-961` 却输出"全部因子因果"。
   - 影响面：`output/etf_3x/report_summary.txt:18` 显示保留的 45 个因子里 **17 个是 `H_*`**；`H_atr_alpha`(nw_t 3.88)、`H_gapdist_alpha`(3.20) 还**通过了 FDR**；`H_ichimoku_alpha` 是 company_1000 预设的两条腿之一（`repro_old_presets.py:39`）；company_300 保留 38 个里 H_ 占 15。
   - **子代理追查了上游，结论是"未发现已发生的泄漏"**：`candle_position_score`（`bc_technical_analysis.py:2194-2241`）与 ichimoku（`:3241-3307`，`senkou` 用 `.shift(+26)` 后移）都是因果的，且 `Split` 在本数据全为 1（所以正文 P0-6 的复权 bug 在这些数据上未触发）。**问题在于：这条通道结构性不可验证，而 `factor` 包报 `causal=True`、`ml` 包报"不可验证"，同仓库两个包对同一批列给出矛盾结论。**
   - 建议：照 ml 的做法加 `FactorSpec.verifiable` + 报告 caveat；用 OHLCV 在包内重算 atr/adx/rsi/kama/ichimoku/candle_* 与 pkl 列比对，不过者剔出 `kept`。

3. **生存偏差检验无效 + 引擎无退市处理。**
   - `factor\report.py:133-137` 用"上市起点是否同日"推断生存偏差，company_300 输出"142 个不同起点 → 生存偏差较小"（`output/company_300/report_summary.txt:10`）。
   - 但该池 2010–2026 全长只有 **297 个标的** —— 若为真实历史成分，16 年换手应产生 600+ 代码，**强烈指向"今天的成分股回填历史"**；而回填成分股天然有 142 个不同 IPO 日，所以这个判据**无法区分回填与全历史**，输出的安慰性结论不成立。
   - `factor\backtest.py:241-256`（当日收益 NaN 直接当 0%）、`:284-297`（卖单失败仍删权重）、`:340-352`（未平仓一律用最后收盘价补记）→ **无退市路径**。

**P1（选摘，14 条中的关键项）**：
- **因子筛选用了全样本相关矩阵**：`combine.py:828` 用全史算 `factor_corr`，`:845` 再传给每折的 `dedupe` → WF 折内选择被折外相关性污染；主链路同病（`report.py:427-429`、`combine.py:1038-1050`）→ **写进 `factor_config.json` 的 `kept` 含 OOS 信息**。
- **`--scheme inv_vol/rp` 静默空转**：`combine.py:65` 列了 7 个方案，权重分支只有 4 个（`report.py:431-445`）→ `--scheme inv_vol` 跑完全流程后 `weights={}`，报告实际输出 `single:best`。
- **`_sanitize_prices` 的 shift 方向与注释相反**：`backtest.py:163-164` `bad | bad.shift(-1)` 标的是"坏点的前一日"，导致 **k≥3 的常数垃圾段内部全部漏网**；且清洗在 `run_engine` 内对**已截窗**数据做（`:198`），同一日期在 IS/OOS/recent 的清洗结果不同。
- **"无信号"被当"清仓"**：`backtest.py:260-263`（composite 缺失→全部卖出）+ `prepare.py:52-66`（薄截面整行 NaN）→ 早期样本与数据缺口被解读为清仓，**凭空增加换手**。
- **静默回退改样本**：`data.py:210-212` 研究版 pkl 缺失时**无日志**回退生产版；线上配置 3/5 池 `data_start=2020-01-02` 而同池研究报告从 `2010-01-04` 开始 → **线上权重与研究报告不是同一份样本，产物也没记录用了哪个 pkl**。
- **`write_config` 有静默毁库路径**：`report.py:653-658` `except Exception: doc = {}` 吞掉 JSON 解析错误，随后 `:689-699` 以空 doc 重写文件 → **一次解析失败即把 5 个池的配置覆盖成只剩当前池**。
- **窗口边界标签越界**：`evaluate.py:613-621` 对 `fwd` 也按 `end` 截 → `t=end` 的标签用到 `end+1+h` 的价格，IS/OOS 分界不严格。
- **成本假设偏乐观**：成本阶梯显示 equal 的 OOS `total_ret` 在 50bps 单边降到 0.004（10bps 时 0.658）→ "OOS 有效"处在成本噪声带内，应写成"仅 ≤25bps 成立"。

**P2 中的三个明确 bug（值得单列）**：
1. **`factor\data.py` 的 pkl 缓存永久失效（我已核实）**：`:56` `key = str(pkl_path)`，但 `:74` 的循环 `for key, df in raw.items():` **把 `key` 覆盖成了最后一个标的的名字**，于是 `:94` `_PANEL_CACHE[key] = (sig, panel)` 用标的名为键，而 `:63` 查询用 pkl 路径（`:91-94` 附近）→ **永不命中**。而这段缓存的 docstring 明确写着"pkl 很大, 反序列化很贵"。修：循环变量改名 `sym_key`。
2. `factor\config.py:51` `VERIFY_CUT` 硬编码日历日 → 数据短于该日时 `verify_causal` **静默退化为恒真**（`factor.py:912-916`）。
3. `factor\backtest.py:59` 模块级可变全局 `_BAD_PX_SEEN` 跨调用累积；`combine.py:404-410` inv_vol 权重 ∝1/σ **无地板会爆炸**；`backtest.py:418-423` 分年收益丢掉跨年当日收益 → **连乘 ≠ total_ret**。

---

## A6. 数据 IO 层（`bc_data_io.py`，实际 **2858 行**，不是我以为的 1400+）

这一层是"所有东西的上游"——pkl 面板的原始价格就来自这里。四条 P0 全部关于数据正确性：

**P0-D1 增量拼接的复权基准漂移（与正文 P1-8 同源，但子代理给出了更精确的机制）**
- `:105/:231/:269-276` 默认 `adjust='qfq'`，`:488` 直接把 `Adj Close = Close`；
- ak 的 qfq 是**以请求时刻最新交易日为基准**的前复权价 → 增量只重下 `上次最后日期 - 3 天`（`:938/:1069`）再 `concat + keep='last'`，**每次除权除息都会在"老历史"与"新窗口"之间留一个人为台阶**（1~3%，大比例送转可 20%+）；3 天重叠只能刷新最近 3 行，**无法修复历史**。
- 额外影响：新窗口的复权因子含**未来的分红信息** → 构成前视偏差。
- 波及：收益率 / 波动率 / 均线 / 支撑压力 / 周月聚合 / 全部因子与回测，并会触发假突破交易。
- 修：把 -3 天重叠**当作一致性校验**（`np.allclose(old.loc[overlap,'Close'], new.loc[overlap,'Close'], rtol=1e-4)`，不一致就全量重下）；根治是本地存未复权 OHLCV + 本地复权。

**P0-D2 下载失败被吞 → 陈旧数据伪装成 success，5 次重试完全失效**
- `:553-556` / `:412-415` `get_data` 捕获**全部**异常后 `return None`；
- 于是 `download_single_symbol_eod` 的 5 次重试**永不因网络错误触发**；
- `pd.concat([旧, None])` 静默返回旧数据（子代理读了 venv 内 pandas 3.0.5 的 `concat.py:820-836` 确认 concat 会静默丢弃 `None`）；
- `:1106` `status='success'` **无条件执行** → `:1288/:1311` 把它计入"下载成功"与 progress_callback；
- 下游 `technical_analyst.py:319` / `technical_analyst_parallel.py:353` 用 `index.max() < required_date` **静默跳过**这些标的 → **universe 静默缩小、持仓标的当天不更新且无人知晓**。

**P0-D3 CSV 非原子写 + 同 symbol 并发写 + 读取无完整性校验**
- `:1654-1685` 的 `to_csv` 非原子；而**同一文件里已有正确范例** `_save_json_file:2257-2268`（tmp + `os.replace` + 失败清理）却未复用；
- `:1156-1159` symbols **未去重** → 同一 symbol 被多线程同时写同一个 csv；
- 唯一校验是 `> 2020-01-01`（`:921/:1049/:1429/:1554`）；
- 截断的 CSV 常能"正常"解析成更少行 → **静默丢历史**；也可能抛 ParserError 被 `:1723-1727` 当损坏文件直接删除 → 触发 eod 全量重下。

**P0-D4 Dividend/Split 被写成 0.0/1.0 并覆盖窗口内真实值**
- `:399-403` / `:489-490` / `:497-499` 把未下载的分红/拆股写成 `0.0`/`1.0`，而 `add_dividend=False` 是默认（生产 `technical_analyst_parallel.py:148/170` 的 `store_true` 默认关闭）；
- 配合 `:947/:1091` 的 `keep='last'` → **先跑一次带 `--update_dividend_split`、再跑一次不带，就会把最近 3 天的真实分红/拆股覆盖成 0/1**；
- "未下载"被表达成"没有分红" → 下游 `create_week_data:1824` / `create_month_data:1893` 的 `Dividend.sum()` 静默偏小。
- 修：未知写 NaN，合并时 `div/split` 用 `new.where(new.notna(), old)`。

**数据层 P1（选摘）**：
- `:2761` **`prefix` 未定义 → NameError**：只要某个 pool 的 signal xlsx 缺失就抛错，而**邮件是你唯一的主告警通道**，异常还在 `:2841` 的 try 之外。
- `:680/:687-688` 港股实时把 `time`（`HH:MM:SS`）当 `Date`（hkquote 同时返回 `date` 和 `time`）→ pandas 解析成"今天+该时刻" → realtime 模式插入**幻影 bar**（子代理核对了 `easyquotation/hkquote.py:41-42`）。sina 分支（`:676`）用的就是 `date`，两分支不一致。
- `:109` `int(symbol) < 600000 → '.SHE'` 把**上交所 ETF（510300/560000/588000）、北交所（430047/830799）、可转债（110/113）全标成 SHE**（应为 SHG）→ **eod 源对这批标的静默无数据**。
- `:1542-1547` vs `:1589`：`update_stock_data_from_ak` 的 refresh 对美股**完全失效**（`split('.')[0]` 把 `105.AAPL` 变成 `105`，exists/remove 作用在 `105.csv`，而保存名是 `105.AAPL.csv`）→ 删不掉旧文件、随后读到旧数据判定"已最新"。同一美股在两套更新器里还有**两种文件名**（`AAPL.csv` vs `105.AAPL.csv`）。
- `:1120/:1192` `sources=None` 无守卫 → TypeError（`api_keys` 在 `:1146-1148` 有守卫，`sources` 没有）；空 symbols / 路径不存在返回 `None`，而调用方 `technical_analyst.py:288`、`technical_analyst_parallel.py:306` 直接 `new_data.keys()` → AttributeError。
- `:1328` `'other'` 标的（`BRK.B` / `BABA.US` / 带后缀 CN）在 realtime 段 `sources[f'{mkt}_realtime']` → **KeyError 让整个函数抛出**（`:1320-1362` 无 try）；串行版 `:809-817` 只 log 不处理。
- `:1729-1748` `load_stock_data` 精心设计的"处理失败则保留原文件"**被所有调用方推翻**：`:926-929/:1061-1065/:1432-1434/:1557-1559` 在返回 None 时立刻 `os.remove` → 一次瞬时读取失败即删整段历史。
- `:199-226` `get_code_map_from_ak`：`os.listdir()` 依赖 CWD、重试无 sleep、**5 线程首跑同时写同一个 `us_spot_em.csv` 并互删过期文件**、mark 文件在 csv 写完后立刻落盘不校验（写坏则当天永不再下）、`:225` `pd.read_csv` 无 try。
- `:2367` `new_config.pop(key)` 无默认值 → KeyError 被 `:2372` 吞掉（"删除成功"其实没删）；add/remove/modify 是**无锁 read-modify-write**，trader 与 TA 并发写 `portfolio.json` 会**丢更新**。
- 异常反模式：`download_nytimes` 的 except 块用了未绑定的 `data`(`:2000`)/`docs`(`:2005`)/`return data`(`:2009`) → NameError；`filter_futu_exported:2205` 用了被注释掉的变量 → `market='a'` 必 NameError；`:413/:554` 的 `logger.exception(f'{symbol} {e}')` 会把**含 `api_token=` 的 URL 打进日志**；5 处 `requests.get` 无 session/retry/`raise_for_status`。

**数据层的亮点**：`_save_json_file`（`:2257-2268`）是全文最正确的一段（tmp + `os.replace` + 失败清理），三个 config 函数还用 `not new_config and exists and getsize>0`（`:2333/:2363/:2394`）挡住"损坏文件被清空"。数据契约写进了代码（`STANDARD_COLUMNS:63-69` + 单一 `validate_standard_dataframe:442-463` 守输入输出两端）。`load_stock_data`（`:1713-1748`）把"文件不可读(删) vs 列格式问题(留)"拆成两层 try 并写明理由。`get_data_from_easyquotation:433` 按腾讯**原始字段顺序**（Date,Open,Close,High,Low,Volume）而不是想当然的 OHLC。并行骨架（`:1275-1311`）worker 只写自己的文件、结果在主线程汇总，**并发模型选得对**（问题只在共享资源 `get_code_map_from_ak` 与文件写入）。

**最关键的一条测试缺口（这条直接影响你对生产的判断）**：现有 14 个用例**完全没有覆盖 eod 路径**，而生产的 us/cn/hk **三个市场全用 eod**；`download_eod_parallel` / `download_single_symbol_eod` / `update_stock_data_new` / `preprocess_symbol` **零覆盖**；**没有增量拼接的回归用例**（最危险的 P0-D1/D4 没有任何守护）。建议第一条测试就写：旧 CSV 3 行 + mock 新窗口含 2 行重叠且价格不同 → 断言应告警或全量刷新。

---

---

# 附录 B：修复记录（2026-09 第 1 轮，已落盘）

> **重要前提**：本机沙箱禁止执行 shell（`pwsh` 报 `Windows ACL temp root must be outside the workspace`），
> 因此**所有改动都是静态的、未经运行验证**。下面每一项都经过逐行重读确认，但没有跑过 pytest / 回测。
> 凡我判断正确但无法在本机完成的，列在本附录末尾的"待你执行"里。

## B1 已修改：`factor/data.py` — pkl 缓存永久失效

- 根因：`key = str(pkl_path)` 之后被 `for key, df in raw.items()` 覆盖，`:94` 用最后一个标的名作缓存键，`:63` 用路径查 → **永不命中**（而这段缓存的 docstring 明说 pkl 反序列化很贵）。
- 改动：缓存键改名 `cache_key`（含防回归注释），循环变量保留 `key`；新增 `_PANEL_CACHE_MAX = 4` 上限，避免长驻进程无限攒面板。

## B2 已修改：`bc_technical_analysis.py` — 复权索引 bug（本次最重要的一处数据正确性修复）

- `:217` `df.loc[split_n1_idx, 'adj_factor'] = 1 / df.loc[split_idx, 'split_n1']` → 源索引改为 `split_n1_idx`。
- 新增**守卫**：赋值后若目标行仍为 NaN（说明索引又写错/未命中），直接 `logger.error` 点名 symbol 与被漏掉的索引集 —— 这类"复权静默失效"必须能在日志里看出来。
- 同文件其他修复：
  - `plot_historical_evolution_parallel` 的 `with ProcessPoolExecutor(...) as pool:` **覆盖同名形参 `pool`**（池名字符串）→ executor 改名，worker 现在收到正确的池名。
  - 两处 gif 路径 `f'{plot_save_path}{symbol}...'`（缺分隔符，gif 落到目录旁边）→ 改用 `Path(...) / f'...'`。
  - 串行版 `images` 只在 `create_gif or plot_final` 时定义，而 `remove_origin=True` 是默认 → `images.append`/`os.remove` 可能 NameError；已确认定义路径并给删除加 `exists()` 判断。

## B3 已修改：`bc_trader.py` — 生产下单安全（护钱）

| 问题 | 改动 |
|---|---|
| **市价单资金校验被绕过** | `trade()` 里区分"下单用哨兵价"与"校验用真实价"（`ref_price`）；市价单现查最新价用于 `get_affordable_quantity`，**取不到参考价则拒单**（`FAILED: no valid reference price`）。`get_affordable_quantity` 对 NaN/≤0/占位价一律返回 0，不再算出"10 倍现金"的股数 |
| **下单失败仍扣预算** | `signal_trade` 买入循环改为**只有 `trade_summary` 含 `SUCCEED` 才扣减** `available_cash`，失败记 error 且不扣 |
| **每票预算被永久改小** | 删掉在循环里改 `money_per_sec` 的写法，改用不变量 `per_symbol_budget`；`budget = min(per_symbol_budget, available_cash)` |
| **可用资金取总现金** | `get_available_cash` 优先读 `available_cash` 列（Futu=`avl_withdrawal_cash`，Tiger=`available_funds`），仅在缺列时回退 `cash` 并告警；NaN/负值视为 0 |
| **止损/止盈量纲** | 新增 `_norm_rate`：`|值|>1` 视为百分比自动归一（`10 → 0.1`）并告警；每次 `cash_out` **打印生效阈值**；`rate_inday` 全为 0 而日内阈值已设时给出明确 warning |
| **持仓未知被当成空仓** | 新增 `position_known` 属性（Futu/Tiger 的 `update_position` 用 `answered` 标记），并在这五处 fail-safe：`synchronize_position_record`（不再抹台账）、`get_in_position_quantity`、`signal_trade`（直接中止本轮）、`trade()` 卖出分支（拒绝卖出而非静默跳过）、`cash_out`、`update_portfolio_record`（不用空持仓覆盖快照） |
| **成交被静默丢弃** | `update_position_record` 收集 `dropped` 列表，结尾统一 error 告警"有成交没入账，需人工对账" |
| **就地表污染调用方** | `update_portfolio_record` 去掉 `set_index(..., inplace=True)`（`position is self.position` 时会改到调用方对象） |

## B4 已修改：`bc_data_io.py` — 数据正确性

- **新增 `_check_overlap_consistency()`**：增量拼接前比对重叠窗口（`Open/High/Low/Close/Adj Close`）的新旧价格。这是对着"qfq 基准漂移"这条最隐性的数据风险加的可观测性 —— 不一致时 error 点名 symbol、坏格数、最大绝对差、受影响日期，并提示全量重下。
  > 注：它**只负责发现**，不做自动全量重下。自动重下会对数据源造成额外压力，建议你先看一段时间的日志再决定是否自动化。
- **新增 `_merge_dividend_split()`**：`add_dividend/add_split` 关闭时各源写 `Dividend=0.0 / Split=1.0`，配合 `keep='last'` 会把真值覆盖成占位。现在"新窗口是占位且本地已知非占位"时保留本地值，并 log 保留了几处。
- **下载失败不再伪装成功**：4 个合并点全部加了空返回判断（并行路径的 `status` 不再无条件 `'success'`）。
- **不再删历史**：4 处 `os.remove(symbol_file_name)` 中，3 处"max_idx 无效 / 空数据"的删除改为**保留原文件 + 全量重下**（`tmp_data_date = None`）；只保留 `update_mode == 'refresh'` 的显式删除。
- **CSV 原子写**：`save_stock_data` 改为临时文件 + `os.replace`，失败清理临时文件。
- **`prefix` NameError（邮件告警直接崩）**：`:2943` 的未定义变量 `prefix`，同时修掉与 `:2933` 文件名口径不一致的问题（现在打印真实文件名）。
- **交易所后缀**：`add_postfix_for_cn_symbol` 从 `int(symbol) < 600000` 一刀切改为按代码段判定（沪市 `60/68/51/56/58/50/11/90`、深市 `00/30/15/16/12/20`、北交所 `43/83/87/88/92`），无法归类的告警。
- **基准下载失败兜底**：`update_stock_data_from_eod` 里 `benchmark_data.index.max()` 原会在下载失败时 AttributeError，补上与 ak 分支一致的兜底。

## B5 已修改：`signal_generator.py`（新工作区）— 去掉静默回退

- 删除 `_FB_KEPT / _FB_GREEDY / _FB_SCHEMES`（全是 `etf_3x` 的陈旧口径）——**池信息绝不兜底**。
- `_load_config_doc`：区分"文件不存在"（返回 `{}`）与"存在但解析失败/非 dict"（**抛 ValueError**）。
- `load_pool_config`：池不存在、池节缺 `schemes`、`meta.pool` 与请求池不符 → **一律抛 ValueError**（fail-fast，宁可不产出信号也不产出错的信号）。
- `resolve_pool`：`default_scheme` 缺失或指向不存在的方案 → 抛 ValueError，不再静默退到 `greedy/first`。
- 生产 pkl 缺失回退研究版：`logger.info` 升级为 `logger.warning`，并说明"该池信号建立在研究版数据上"。
- 配置日志从"只打默认方案"改为**打全量**（pool / 方案列表 / 默认 / 本次使用）。

## B6 已修改：`factor/report.py` — 拒绝毁库 + 原子写

- `_load_config_doc`：解析失败不再 `doc = {}`（那会让 `write_config` 把 5 个池的配置覆盖成 1 个），改为**抛 ValueError**并提示先修复或删除文件。
- `write_config`：改为临时文件 + `fsync` + `os.replace`（原来直接覆盖写，中途失败会留下半截 JSON，而这是"研究→生产"的唯一接口）。
- 补 `import os`。

---

## B7 待你执行 / 待你确认（我无法在本机完成）

| # | 事项 | 为什么我做不了 / 需要你确认 |
|---|---|---|
| 1 | **跑一遍现有测试**：`quant\tests\test_bc_data_io_robustness.py`（我改了 `save_stock_data`/`load` 调用方/`add_postfix_for_cn_symbol`，必须验证不回归） | 沙箱禁止执行 python |
| 2 | **确认 `stop_profit` 的语义** | `ta_config.json` 现在是 `10`。我的 `_norm_rate` 会把它归一成 `0.1`（+10% 止盈）。**如果原意是 10%，那现在正好；如果原意是别的数，请直接改配置**。止损 `-0.8`（-80%）我没有动 —— 这个值是否需要收紧由你决定 |
| 3 | **`rate_inday` 目前恒为 0** | 它只在 `update_position(get_briefs=True)` 时才算，而 `cash_out` 默认 `get_briefs=False`。我加了 warning 但没改默认值（改默认值会增加每轮的网络请求，应该由你决定） |
| 4 | **`git init` + 写对 `.gitignore` 再 `git add`** | 需要你先决定仓库放哪；且必须先 ignore 掉 `research_2026-09/output/`（995 文件）、`portfolio.json`、`position.json`、`api_key/`、`*.pem`、`*.pyc` |
| 4b | **决定是否启用 `USE_IDEMPOTENT_REMARK`**（`bc_trader.py` 模块级开关，默认 `False`） | 打开后订单会带幂等键(`{platform}-{account}-{symbol}-{action}-{YYYYMMDD}`)。我无法确认两家券商对 `remark` 的语义（是否拒绝重复 remark/长度上限/字符集），所以**默认关闭**，要你确认后再开 |
| 5 | **轮换 EODHD key** | 需要联网操作后台 |
| 6 | **重跑 pkl 生成**（因为 B2 的复权索引修复会影响所有价格列） | 需要你的数据源与运行环境；且要评估"哪些标的的因子值/结论需要重算" |
| 7 | **装 `lightgbm` + `torch`**（`requirements.txt` 里没有） | 需要网络 |
| 8 | **`ml` 默认特征源改 `factor`**、**IS/OOS 窗口统一**、**`H_*` 因子因果性重算比对** | 这三项会改变研究结论，属于"需要你先拍板"的范畴，我没有擅自改动 |

## B8 我**没有**改的（及原因）

- **`ta_config.json` 的具体数值**：只加了代码侧的归一化与告警。改真钱参数应该由你确认（见 B7-2）。
- **`.gitignore`**：当前没有 `.git`，写了也暂时无人消费；等 B7-4 建仓库时一起做更合理。
- **两套回测引擎对拍**（`bt_core` vs `factor.backtest`）：这是"验证"而非"修复"，需要能跑代码。
- **测试的"pytest 化"**：我写的是**新增**回归用例（见 B10），沿用仓库既有脚本式风格；把整个 `tests/` 改造成 pytest（`conftest.py`/`pyproject.toml`/参数化）需要先能跑测试，否则改完也验证不了。
- **`ta_data_columns` 陈旧白名单** 与 **`plt.*` 全局状态**：见 B9 末尾说明（前者需要真实产出列集对齐，后者需要逐处对照 `use_ax`）。

---

## B9 已修改（第 2 轮）：`bc_technical_analysis.py` 指标层

> ⚠️ **这一批会改变指标数值**，因此**必须在重跑 pkl 之后重新复核受影响的因子/回测结论**。
> 每一项都注明了影响的是公式错误（应当修正）还是口径统一（需要你确认）。

| # | 位置 | 问题 | 改动与影响 |
|---|---|---|---|
| B9-1 | `add_tsi_features` | **`m.ewm(r)` 的首个位置参数是 `com` 不是 `span`** → alpha=1/(1+r)=0.038，而本文件其它指数平滑走 `em()`→`span=r` 即 alpha=2/(r+1)=0.077。同一份代码两套平滑口径，TSI 系统性偏慢 | 改用 `em()`，与文件内其余指标一致。**TSI 数值会变（变"更快"）**。另外把 `m2 → 0` 时的 ±inf 在产生阶段压成 NaN（原来只在 `fillna=True` 时才处理，默认 `fillna=False` 会漏到下游） |
| B9-2 | `cal_kama` | `ER_den == 0`（长期零波动/一字板）→ `ER = 0/0 = NaN` → `sc=NaN` → 循环里 `kama[i]=NaN`，**随后 `kama[i]=kama[i-1]+…` 把 NaN 永久传染到序列尾部**，`kama_fast/kama_slow` 全废 → 位置/趋势/支撑阻力连锁失效 | ① `ER = (ER_num/ER_den).where(ER_den > 0, 0.0)`，ER=0 是"无效率"的正确语义；② 循环里若 `prev` 为 NaN 则用当前收盘价重启状态，并在全程无法建立状态时 `logger.error`。**kama 序列会比原来"更长有效"（原来尾部可能是 NaN）** |
| B9-3 | `add_adi_features` | `H==L` 且 `C!=H`（脏数据/停牌）时分母 0、分子非 0 → **±inf**，`fillna(0.0)` 抓不到 inf，而 inf 一旦进入 `cumsum` 会把**整条 ADI 序列**污染成 inf | `clv` 与 `clv*volume` 都先 `.replace([inf,-inf], nan).fillna(0.0)` 再 `cumsum` |
| B9-4 | `add_cmf_features` | 同 B9-3（`fillna(0.0)` 抓不到 inf，inf 进 `rolling().sum()` 污染整段 CMF）；且 `vol_sum == 0`（全停牌）时会再产生 inf | 同 B9-3，并把分母 `vol_sum.where(vol_sum > 0)` |
| B9-5 | `add_wr_features` | `rolling(lbp, min_periods=0)` 暖机期拿"部分窗口"当真值，与本文件其它指标（`sm/em` 默认满窗）口径不一致；`hh==ll`（一字板）无保护 → ±inf 直接进信号 | `min_periods=lbp`（满窗）；`hh==ll` 的 bar 取中性 `-50`，**且只在窗口已满时替换，暖机期仍保持 NaN**（不制造假数据） |
| B9-6 | `add_adx_features` | `if method in ('wilder')` —— `('wilder')` 是**字符串不是元组**，任何子串（`'w'`/`'i'`/`''`）都算命中；且该分支的 `fillna` 用 `locals().get(col)` 反射 + 只写"已在 df 中的列"，而 wilder 分支从不写 `pdi/mdi/dx/adx` → **那几列的 fillna 完全是空操作** | 改为真元组 `('wilder', 'wilder_legacy')`；`fillna` 改为直接对 df 中实际存在的列处理（去掉 `locals()` 反射） |
| B9-7 | `calculate_ta_basic` | `df = eval(f'add_{indicator}_features(df=df)')`，`indicator` 直接来自 `ta_config.json` → **任意表达式执行**；且错误被外层宽 `except` 吞掉，只留一行日志、返回半成品 frame | 去掉 `eval`，改为 `globals().get(func_name)` + `callable()` 白名单校验；找不到实现时 `logger.error` 并跳过该指标（而不是静默不计算） |

### 第 2 轮**刻意未改**的两项（需要你决策）

1. **`ta_data_columns`（`:7686-7778`）陈旧白名单** —— 它含从未生成的列（`chikan`、`trend*`、`potential*`、`vis_*`、`l_trend` 等），却**不含 macd/cci/aroon/stoch/mfi/tsi/uo/wr/vortex/psar/kst/trix/stc/dpo/ao/eom/fi/nvi/obv/vpt/cmf**。
   我没有手工重写它，因为**没有实际产出列集可以对齐**（无法运行代码）——手工"补全"极可能引入新的错误。
   建议做法（需要能跑代码时执行一次）：
   ```python
   # 用真实产出对齐一次, 把结果固化; 并在 CI 里加一致性断言
   actual = set(df_full.columns)
   missing = actual - set(ta_data_columns)      # 会被下游静默丢掉的列
   stale   = set(ta_data_columns) - actual      # 列表里已不存在于产出的列
   ```
2. **一批 `plt.*` 全局调用**（`:7128-7129` 的 `plt.ylim/plt.yticks`、以及 `plt.savefig`/`plt.annotate` 等散布点）——多子图/并行流程下会把刻度或标注写到"当前 axes"而不是目标 axes，可能保存/标注错误的图。修它需要逐处对照 `use_ax` 语义，属于**图表正确性**而非资金安全，且不好静态验证，留作单独一批。

## B10 新增：回归测试 `tests/test_bc_data_io_incremental.py`（16 个用例）

写这份测试的动机：本次修复的 5 处缺陷**全都是"静默"型的**（不一致不报错、占位覆盖真值、写坏文件不报错、失败当成功、后缀错判后无数据），
没有测试就只能靠人肉记性。测试沿用仓库既有的脚本式风格（`check(name, cond)` + 末尾 `ALL N TESTS PASSED` + `sys.exit(1)`），零第三方依赖、不联网、不碰真实数据。

| 组 | 用例数 | 覆盖的行为 |
|---|---|---|
| 增量拼接一致性 | 5 | 重叠价一致→True 不报错；**复权基准漂移→False 且 error 点名 symbol**；无重叠→不误报；None/空表安全 |
| 分红/拆股真值保护 | 5 | 占位 `0.0/1.0` **不得**覆盖已知真值；新值非占位时以新值为准；老数据空时安全；**不就地修改传入的 new** |
| CSV 原子写 | 5 | 正常落盘；无 `.tmp` 残留；内容可解析；**写失败时向上抛异常**；写失败后原文件完整且不留 `.tmp` |
| 下载失败不伪装成功 | 2 | `get_data` 返回 `None` → `status='failed'`（原来是无条件 `'success'`）；messages 可诊断 |
| 交易所后缀判定 | 1 | 9 个代码：沪主板/科创板/**沪 ETF(510300/588000)**/深主板/创业板/深 ETF/**北交所(830799)**/港股 |

**运行方式与现有测试相同**：
```powershell
C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_incremental.py
```

> ⚠️ 这份测试**我没有运行过**（沙箱禁止执行 python）。如果其中某条断言与我的静态理解有偏差（尤其是 `download_single_symbol_eod` 的 `args` 元组顺序、以及 `_check_overlap_consistency` 的容差判定），请把失败输出贴给我，我按实际行为修正 —— 测试本身也可能有 bug，不能假定它一定对。

---

## B11 完整改动清单（可直接用作 code review 清单 / 提交信息）

**修改 6 个文件 + 新增 1 个测试文件**：

| 文件 | 改动摘要 |
|---|---|
| `quant/factor/data.py` | pkl 缓存键被循环变量覆盖 → 改名 `cache_key`（缓存**恢复生效**）；加 `_PANEL_CACHE_MAX=4` 上限 |
| `quant/factor/report.py` | `_load_config_doc` 解析失败改为抛异常（原来 `doc={}` 会让 `write_config` 把 5 个池覆盖成 1 个）；`write_config` 改临时文件 + `fsync` + `os.replace`；补 `import os` |
| `quant/bc_technical_analysis.py` | 复权索引 bug（`:217`）+ 未命中守卫；`ProcessPoolExecutor ... as pool` 遮蔽形参 `pool`；2 处 gif 路径缺分隔符；`images` NameError；TSI 的 `ewm` com/span + inf；KAMA 的 `ER_den==0` + NaN 传染；ADI/CMF 的 inf 污染；WR 的 `min_periods` + 一字板除零；ADX 的 `('wilder')` 字符串元组 + `locals()` 空操作 fillna；`eval` 派发改白名单查找 |
| `quant/bc_data_io.py` | 新增 `_check_overlap_consistency()` / `_merge_dividend_split()` / `_is_missing()` 并在 4 个拼接点接入；4 处空返回判失败（并行 `status` 不再无条件 `success`）；3 处"删历史再重下"改为保留原文件；`save_stock_data` 原子写；邮件 `prefix` NameError；交易所后缀按代码段判定；eod 基准下载兜底；`sources=None` 守卫；**hkquote 用 `date` 而非 `time`**（修幻影 bar） |
| `quant/bc_trader.py` | 市价单资金校验用真实参考价（取不到则拒单）；`get_affordable_quantity` 对占位/非法价返回 0；下单失败不扣预算；`per_symbol_budget` 不变量；`get_available_cash` 用可用资金列；`_norm_rate` 量纲归一 + 打印生效阈值 + `rate_inday` 告警；`position_known` 语义 + **7 处** fail-safe（含 Tiger 卖出分支）；成交丢弃告警；去掉 `set_index(inplace=True)`；**新增 `trade_ok()` 统一下单成功判定 + `idempotent_remark()` 幂等键（默认关闭）** |
| `~/signal_generator.py` | 删除 `_FB_KEPT/_FB_SCHEMES` 池兜底；配置缺失/损坏/池不符 → **fail-fast**；`default_scheme` 缺失/非法 → fail-fast；生产 pkl 回退升级为 `warning` |
| `quant/tests/test_bc_data_io_incremental.py` | **新增**，16 个回归用例（脚本式，零依赖、不联网） |

### 建议的提交顺序（若你用 git）

```bash
git add quant/factor/data.py quant/factor/report.py          # 缓存 + 配置安全（无行为风险）
git add quant/tests/test_bc_data_io_incremental.py           # 先加测试
git add quant/bc_data_io.py                                  # 数据正确性（会触发 overlap 告警）
git add quant/bc_trader.py                                   # 生产下单安全
git add quant/bc_technical_analysis.py                       # 指标层（会改变数值 → 需重跑 pkl）
git add ~/signal_generator.py                                # 注意: 它在工作树外, 需要先决定仓库位置
```

## B12 已修改（第 3 轮）：下单回执契约 + 数据层剩余两处

### 目标 P2 的剩余项：下单回执与幂等

上一轮我让 `signal_trade` 用 `'SUCCEED' in trade_summary` 判断成功 —— 那是**在脆弱契约上打补丁**：失败详情里若含 `'SUCCEED'` 字样就会误判为成功（把失败当成成功扣款或反之）。这一轮把根修掉：

| 改动 | 说明 |
|---|---|
| 新增 `trade_ok(trade_summary)` | 唯一的下单成功判定入口。规则：只取**第一个 `\|` 之前**的前缀段，该段含 `SUCCEED:` 且不含 `FAILED:` 才算成功。失败详情里的 `SUCCEED` 不再误判；无分隔符但两个标记都出现时**保守判失败**（宁可漏判成功，不可误判成功） |
| `signal_trade` 改用 `trade_ok()` | 不再自己 substring |
| 新增 `Trader.idempotent_remark(symbol, action)` | 生成 `{platform}-{account}-{symbol}-{action}-{YYYYMMDD}` 幂等键 |
| 新增模块开关 `USE_IDEMPOTENT_REMARK = False` | **默认关闭**。打开后 Futu 走 `place_order(remark=...)`、Tiger 走 `order.remark`。关闭的原因：我无法确认两家券商对 `remark` 的具体语义（是否拒绝重复 remark、长度上限、字符集），**盲目启用等于在生产上做未验证的行为变更** |
| `Tiger.trade` 卖出分支补 `position_known` 守卫 | 上一轮只修了 Futu，Tiger 的"持仓未知 → 静默跳过卖出"是同一个资金风险，本轮补齐 |

### 数据层剩余两处

| 位置 | 问题 | 改动 |
|---|---|---|
| `get_real_time_data_from_easyquotation` | **hkquote 把 `time`('HH:MM:SS') 映射到 `Date`**，再 `x[:10]`（对 8 字符时间串是空操作）→ pandas 解析成"今天+该时刻" → realtime 模式插入**幻影 bar**（同日两根），且时间戳不匹配会让 `.loc` 变成新增行。上行 sina 分支用的就是 `date` | 改用 `date`（并在字段缺失时 error 提示 easyquotation 版本契约可能已变） |
| `update_portfolio_support_resistant` | `np.isnan(support)`：值来自 portfolio.json / ta_result，**可能是字符串**（如 `'--'`）→ `TypeError`，整封告警邮件发不出 | 新增 `_is_missing(value)`：对 None / NaN / ±inf / 字符串占位（`''`/`'-'`/`'--'`/`'nan'`/`'none'`）/ 其它类型都安全判定，不再抛异常 |

### 测试更新

`tests/test_bc_data_io_incremental.py` 从 16 → **22 个用例**（含 `trade_ok` 组共 26）：
- 新增 `_is_missing` 的 2 个用例（含 `0.0` **不是**缺失、`'--'` 是缺失）；
- 新增 `trade_ok` 的 4 个用例，其中最关键的一条是 **"失败详情里含 `SUCCEED` 字样不得误判为成功"**。
- `trade_ok` 组用 `try: from quant.bc_trader import trade_ok` 包裹：**环境缺 `futu`/`tigeropen` 时打印一行 `[SKIP]` 而不是报假失败**（我确认了 `.venv\Lib\site-packages` 下没有这两个 SDK，所以这条保护是必要的）。

## B13 已修改（第 4 轮）：IS/OOS 口径单一真源（目标 P4 的剩余核心项）

这是"结论可信度"的根：审查发现 `factor_config.json` 里**逐池**的 `is_window`/`oos_window` 从来不被代码读取，纯属装饰；而 `factor/config.py` 的全局常量、`wf_strat_search.py` 的硬编码、`repro_old_presets.py` 的第三套、`ml/config.py` 的模块常量各不相同。后果是**"报告的样本外"对线上冻结权重可能其实是样本内**。

### 新增单一真源

`factor/config.py` 新增：
```python
resolve_windows(pool, path=None) -> {'is': (s,e), 'oos': (s,e), 'recent': (s,None), 'source': str}
```
优先级：**该池 `factor_config.json` 的 `meta.is_window`/`oos_window`** > 本模块全局默认。
- `recent` 从 **OOS 结束的次日**起算（用 `datetime` 做 +1 天），避免把 OOS 末日重复计入；
- `is_window` 形状不合法时**整体回退**全局默认（不"半采用"）；
- 配置缺失/损坏时回退全局默认且不抛异常；
- 带小缓存，避免逐次读盘。

### 接线（全部改为走真源）

| 文件 | 改动 |
|---|---|
| `factor/evaluate.py` | `window_bounds(window, start, end, pool=None)` 新增 `pool`；`evaluate_factors`/`__main__` 传 `ds.pool` |
| `factor/combine.py` | `verify_combine` 与 `main` 传 `ds.pool`（两处各 2 行） |
| `factor/backtest.py` | `verify_backtest`/`main` 传 `ds.pool`；`main` 的"全交易窗"起点改用该池 IS 起点 |
| `factor/report.py` | `run_report`/`build_summary` 传 `ds.pool`；`export_config` 传 `pool`（**导出的 meta 必须等于本次流水线实际用的窗口**，否则下一轮按 config 复核又会漂移）；`plot_equity(split=...)` 与全文"全交易窗"文案改用该池 IS 起点 |
| `factor/wf_strat_search.py` | 删掉硬编码的第三套窗口(2021-2024/2025/2026+)，改新增的 `pool_windows(pool)` → `cfg.resolve_windows` |
| `factor/repro_old_presets.py` | 新增 `windows_for(pool)`；**修掉 `'oos'` 用 `(OOS_START, None)` 的老 bug**（那让 OOS 一直延到数据末尾，把 recent 也吃进来了） |
| `factor/wf_step2/3/4`、`wf_combo_search.py` | 全部改用 `repro.windows_for(pool)` / `cfg.resolve_windows(pool)['is']`，不再引用全局常量 |
| `ml/config.py` | 新增 `windows_for(pool)` / `window_bounds(name, pool)` / `set_current_pool` / `current_pool`；`window_range` 改为委托并支持 `pool` |
| `ml/data.py` | `load()` 结束时调 `cfg.set_current_pool(ds.pool)` —— 深层拿不到 pool 的函数（SampleSet/FeaturePanel 不带池信息）也能切到正确窗口 |

### 为什么 ml 用"进程级当前池"而不是逐函数透传

`ml` 里切窗口的深层函数只拿得到 `SampleSet`/`FeaturePanel`，它们不携带池信息；逐层加 `pool` 参数要动 11 个文件的签名，风险远高于收益。`set_current_pool` 由**唯一的数据入口** `ml.data.load()` 设置，未设置时回退全局默认（行为与改动前一致，安全回退）。

### 新增回归测试（8 个用例）

`tests/test_bc_data_io_incremental.py` 从 22 → **30 个用例**（含 `trade_ok` 组 34）。新增的"窗口单一真源"组用一个**独立临时 config 文件**（不读仓库真实 `factor_config.json`，因此结果稳定、不受你本地数据影响）锁住四件事：
1. 该池 config 的 IS/OOS **优先于**全局默认；
2. `recent` 从 OOS 次日开始；
3. `is_window` 形状不合法时**整体回退**，不半采用；
4. 池不存在 / 配置文件缺失 / 文件损坏 → 都回退全局默认且不抛异常。

> ⚠️ **这一轮会改变多个脚本的窗口口径**（尤其 `etf_3x`/`hs300`/`a_etf_all` 从 2011-2019/2020-2023 变成 config 里的 2021-2024/2025+）。
> 这是**修正**而不是回归 —— 但意味着**这些池此前所有基于旧窗口的结论都需要重跑复核**。长历史池 `company_*` 的窗口本来就和全局一致，不受影响。

## B14 已修改（第 5 轮）：`bc_util` 两处 + `ta_data_columns` 守卫 + 作用域自检

### 1. 上一轮改动的**作用域自检**（这轮最重要的一件事）

上一轮我改了 12 个文件的窗口接线，而我无法运行代码 —— 一旦某个新引用的变量不在作用域内，整个流水线会以 `UnboundLocalError` 挂掉。因此我逐个函数核对了新引用的变量定义：

| 位置 | 新引用 | 定义处 | 结论 |
|---|---|---|---|
| `backtest.verify_backtest` | `is_s/is_e/oos_s/oos_e` | `:584-585` | ✅ |
| `backtest.main` | `is_s/oos_s/trade_start` | `:660-661`，`trade_start = is_s` | ✅ |
| `evaluate.window_bounds` | `win` | 函数内 `resolve_windows(pool) if pool` | ✅ |
| `evaluate.evaluate_factors` / `__main__` | `ds.pool` | 函数参数 | ✅ |
| `report.run_report` | `is_s/is_e/oos_s/oos_e/trade_start` | `:403-404`、`:498` | ✅ |
| `report.build_summary` | `_is/_oos` | `:808-809`（局部） | ✅ |
| `report.export_config` | `pool` | 函数参数 | ✅ |
| `combine.verify_combine` / `main` | `ds.pool` | 函数参数 | ✅ |
| `repro_old_presets.windows_for` | `cfg.resolve_windows` | `cfg` 已导入 | ✅ |
| `wf_step3_sens.run_pool` | `is_s/is_e` | `:89` | ✅ |
| `wf_step4_export_weights.run_pool` | `is_s/is_e` | `:51` | ✅ |
| `ml.data.load` | `cfg.set_current_pool` | 已定义 | ✅ |

同时全仓 grep 确认**没有遗留的全局常量引用**（`cfg.TRADE_START` / `cfg.IS_END` / `cfg.OOS_START` / `cfg.OOS_END`）——剩下的引用全部落在三类合法位置：`resolve_windows` 内部作为回退、`ml/config.py` 作为模块级默认、以及测试用例里断言"回退到全局默认"。

### 2. `bc_util.synchronize_file`：本地目录不存在时静默跳过整个目录

`:290`（文件夹分支）的 `newer_only` 判断直接 `os.path.getmtime(lfd_abs_path)`，而**本地目录在首次同步时并不存在** → `FileNotFoundError` → 被外层 `except` 吞掉 → 该目录被**静默跳过**，日志里只留一行 exception，看起来"同步跑过了"其实什么都没同步。
→ 已加 `and os.path.exists(lfd_abs_path)`（与文件分支 `:249` 的写法一致）。

### 3. `bc_util.run_script`：默认 `retry=1` 与注释"retry(10 times)"矛盾

`retry: int = 1` 表示**默认不重试**，而函数注释写的是 10 次；调用方不显式传参时拿到的是"一次就放弃"。
→ 默认值改为 10 以对齐注释，并在 docstring 里标为**行为变更**：若你依赖"默认只跑一次"（例如避免重复下单类脚本被自动重跑），**请显式传 `retry=1`**。

### 4. `ta_data_columns` 陈旧白名单：改为可验证，而不是手工猜写

这张手工清单的问题我在 B9 末尾说过。我没有手工"补全"它（没有真实产出列集可对齐，手工改极可能引入新错），而是**新增自检函数** `verify_ta_data_columns(df, strict=False)`：

- `missing` = 实际产出但不在白名单 → 下游按白名单选列时会被**静默丢弃**（Momentum 族的 macd/cci/aroon/stoch/mfi/tsi/uo/wr/vortex/psar/kst/trix/stc/dpo/ao/eom/fi/nvi/obv/vpt/cmf 全在此类；**目前 `ta_config.json` 未启用它们，所以是"启用即中招"的潜伏问题**）；
- `stale` = 白名单里有但本次没产出（含从未生成的 `chikan`/`trend*`/`potential*`/`vis_*`/`l_trend` 等）；
- `strict=True` 时发现 missing 直接抛 `ValueError`（适合放进 CI/自检）。

用法：算完一个池之后调一次 `verify_ta_data_columns(result_df)`，按其输出**一次性把白名单对齐到真实产出**并固化；之后 strict 模式就能防它再漂移。

### 5. 关于"测试 pytest 化"（P5 的剩余部分）：我判断**不做**

我核查了 `.venv\Lib\site-packages` —— **既没有 `pytest`，也没有 `akshare`**（更早还确认过没有 `futu`/`tigeropen`），而 `.venv\Scripts\python.exe` 是存在的。也就是说这个 venv 现有状态跑不了仓库自己的测试（`tests/README.md` 里"必须用 venv 的 python"这条约定目前不成立）。
在这个前提下把测试改成 pytest 是**净负收益**：既无法在你本机验证，又会让原本"零依赖、直接双击可跑"的脚本式测试变得跑不起来。所以 P5 的"补关键回归用例"我已完成（30+4 用例），"pytest 化"留待你能跑 pytest 之后再做。

## B15 新增（第 6 轮）：前置自检脚本 `quant/tools/preflight.py`

### 为什么做这个而不是继续改代码

我累计改了 8 个文件、新增 1 个测试文件，而**这些改动从未在任何环境里运行过**（改动时所在沙箱禁止执行 python）。继续在无法验证的前提下增加改动，边际收益是负的 —— 所以我这一轮做了**降低你验证成本**的东西。

### 它做什么（三步，每步失败都给明确的下一步）

```
python quant/tools/preflight.py                 # 全跑
python quant/tools/preflight.py --skip-tests    # 只做环境 + 导入冒烟
```

1. **环境体检**：python 版本 + 12 个关键第三方包是否存在（区分"必需"与"可选"）。
2. **模块导入冒烟**：逐个 import **本次改动涉及的 32 个模块**。这一步的价值在于：它能在**不跑完整回测**的前提下抓住"改坏了语法/缩进/引用"——而历史上那个潜伏数日的断裂导入（原 P0-0）正是因为没有 import 冒烟才没人发现。
3. **跑两个测试文件**：`subprocess` 调子进程、汇总退出码，测试输出原样打印。

### 一项关键的判定设计：区分"代码坏了"与"包没装"

你现在的 `.venv` **缺 akshare/pytest/futu/tigeropen**，所以导入冒烟一定会失败。如果这些失败被一律算成 FAIL，那么"我改坏了代码"和"环境没装包"就分不开——而那恰恰是你最需要分清的一件事。

所以脚本做了分类：
- `FAIL` = 需要排查（**非 ImportError 类型的异常**，如 `AttributeError` / `NameError` / `SyntaxError` / `IndentationError` —— 这些一定是改动引入的）；
- `WARN` = 第三方包缺失导致的 import 失败（装包即可）；
- `SKIP` = 可选重依赖（lightgbm/torch/futu/tigeropen）缺失，该模块本次无法冒烟。

汇总末尾会根据分类给出**不同的下一步指令**，并附上三条按需安装命令。

> ⚠️ 这份脚本**我同样没有运行过**（同一沙箱限制）。它本身也可能有 bug。如果它自己报 `SyntaxError` 或跑出与预期不符的结果，请把输出贴给我。

## B16 已修改（第 7 轮）：撤回一个有风险的默认值改动 + 补齐运维文档

### 1. 撤回 `run_script` 默认值的改动（**这是本轮最重要的一件事**）

上一轮我把 `bc_util.run_script` 的 `retry` 默认值从 **1 改成 10**，理由是"注释写 10 次、实现却是 1 次，不一致"。**这个改动有资金风险**：如果某个"不幂等"的调用方（下单类脚本）漏传 `retry`，它会被自动执行 10 次。

这一轮我去核实了所有生产调用点（`automatic_trader.py`，依据 `research_2026-09/report/automatic_trader_flow.md`）：

| 调用点 | retry | 说明 |
|---|---|---|
| `signal_bridge.py --pool company_300`（盘前桥） | **1** | 幂等设计：靠 `signal_bridge_{D}.xlsx` 是否存在来 skip，**刻意不希望自动重跑** |
| `technical_analyst_parallel --pool hs300`（A股数据） | 1 | |
| `--pool a`（A股数据+邮件） | 5 | |
| `company_300` 两次更新 / `us` 池 / `hs300` / `universe` / 周六 refresh | 5 | |

**结论：每个生产调用点都显式传了 `retry`，默认值在生产中从不生效** —— 所以我那个改动"没有实际影响"，但它会给未来漏传参数的调用方埋雷。

**因此撤回为 `retry: int = 1`**，并把注释改成与实现一致，docstring 里写清三件事：
- 默认 1 = 保守不重跑；
- 生产调用点分别用了 1 和 5（附证据来源）；
- **需要重试请显式传 `retry=N`**。

选择"默认保守"而不是"默认 10"的理由：漏传时"只跑一次"最坏只是没重试（可以人工再跑），而"自动跑 10 次"对不幂等脚本是不可逆的。**这类默认值应当向"安全"而不是向"注释"对齐。**

### 2. `tests/README.md` 重写

原文档只描述 1 个测试文件，而你正要照它操作。现在的版本包含：
- **一键自检入口**（`tools/preflight.py`）放在最前，因为那才是你该先跑的命令；
- 两个测试文件的用途与用例数（14 / 30+4）；
- **`FAIL` / `WARN` / `SKIP` 三态对照表**（把"我改坏了代码"与"环境没装包"分开）；
- 两条必须知道的警告：① 两个测试**从未运行过**，断言本身也可能是错的；② **"必须用 venv 的 python"这条约定当前不成立**（`.venv` 缺 `akshare`）；
- "什么时候跑"里新增一条：**重跑 pkl 之前先跑**；
- "优先补哪几类测试"按出错代价排序（增量拼接/复权、下单安全、引擎不变量、`_sanitize_prices` 的 k≥3 常数垃圾段）。

## B17 第 8 轮：交付物一致性审计 + 修掉一个我上轮引入的窗口归属 bug

### 1. 交付物一致性审计（我多轮报告了很多改动，但无法运行，所以必须核对"说的"与"文件里的"是否一致）

| 核对面 | 方法 | 结果 |
|---|---|---|
| 测试用例数 | grep 实际 `check(` 调用点 | **30 个无条件 + 4 个 `trade_ok` 条件用例** —— 与报告一致 ✅ |
| 测试分组 | 数分组头注释 | 6 组（拼接/D-S/原子写/下载+判空+回执/后缀/窗口），与 README 表格一致 ✅ |
| 窗口接线的作用域 | 逐个函数核对新引用变量的定义处 | 12 处全部 ✅（见 B14） |
| 残留全局窗口常量 | 全仓 grep `cfg.(TRADE_START\|IS_END\|OOS_START\|OOS_END\|RECENT_START)` | 仅剩三类合法引用：`resolve_windows` 内部回退、`ml/config` 模块级默认、测试断言"回退到全局默认" ✅ |
| `run_script` 改动的生产影响 | grep 全仓调用点 + 读 `automatic_trader_flow.md` | 所有生产调用点都显式传 `retry`，我的默认值改动是死代码（已在 B16 撤回） ✅ |

### 2. 修掉一个我上轮引入的真 bug：`ml/predict.py` 的窗口归属

上一轮我给 `ml.data.load()` 加了副作用 `cfg.set_current_pool(ds.pool)`（让深层切窗函数知道当前池）。但我漏了 `predict.py` 的**跨池训练**路径：

```python
ds_train = load(train_pool, a.interval)                     # 当前池 = 训练池
ds = ds_train if train_pool == a.pool else load(a.pool, ...) # 当前池 = 评估池 ← 覆盖了
```

于是 `predict_scores(ds_train, ...)` 内部按 `window_range('IS')` 切训练样本时，用的是**评估池**的窗口。当两个池的 `factor_config.json` 窗口不同时（例如 `--pool etf_3x --train-pool us_merged`：前者有池级窗口 2021-2024/2025+，后者回退全局 2011-2019/2020-2023），**训练样本会被评估池的窗口静默切错**。

**修法**：
- 显式 `cfg.set_current_pool(ds_train.pool)` —— 本脚本的 IS/OOS 切窗作用在**训练数据**上，所以窗口该归属训练池；
- 两个池窗口不同时打 `logger.warning`，把两套窗口都打出来，让你确认口径（而不是我替你假设）。

我也 grep 了全 `ml/`，确认 `--train-pool` 只出现在 `predict.py` 一处。

> 这个 bug 是"我上一轮为了减少改动面而引入的间接依赖"导致的 —— 恰好说明**为什么我不应该在无法运行的环境里继续做结构性改动**：我自己引入的问题自己看不见，需要靠这种事后审计才发现。

## B18 新增（第 9 轮）：仓库根 `.gitignore` + 提交前敏感信息守卫

这一轮不再改代码，而是把你**执行 `git init` 时必须具备的两样东西**准备好 —— 因为 `git add` 是**不可逆**的，顺序错了就要 `git filter-repo` 重写历史。

### 1. 仓库根 `.gitignore`（新建于 `C:\Users\northcheng\git\.gitignore`）

**位置是关键**：原 `.gitignore` 位于 `quant\` 内，所有规则都锚定在 `quant/` 下，**管不到兄弟目录** `research_2026-09/`；而那个目录的 `output/` 有 **995 个文件（占全树 69%）**。原规则写的 `research/output/` 对应的目录甚至已经不存在。

新文件按"数量最大 → 风险最高"排序覆盖：

| 段 | 覆盖内容 | 为什么 |
|---|---|---|
| Python 缓存 | `__pycache__/`、`*.py[cod]` | 工作区现有 **95 个 `.pyc`**，跨 5 套解释器 |
| 研究产物 | `research_2026-09/output/`、`quant/factor/output/`、`**/output/logs/`、`_backup_*` | **995 + 163 个文件**，是体积主体 |
| **运行时状态** | `portfolio.json`、`position.json` | **含真实券商净值与持仓明细**（futu REAL 28169.53 等） |
| **凭据** | `api_key/`、`*_key.pem`、`*user_info*`、`trade_condition.json`、`.env` | 目前实际在仓库外 `~/quant/`，但**规则必须写上**——"安全"不能只靠物理位置 |
| 行情数据 | `*.pkl`、`quant/stock_data/` | 体积大，应在 `~/quant/` |
| 杂项 | `Thumbs.db`、`*.xmind`、`*.zip`、Jupyter | `.xmind` 是二进制无法 diff；仓库里有 3 个 + 1 个整包 zip |

文件内附了**使用顺序**：① 放好 `.gitignore` → ② `git init` → ③ `git add` → ④ `git check-ignore -v` 抽查。顺序反了就要清历史。

### 2. `quant/tools/check_before_commit.py`（提交前守卫，只读）

人眼在 1439 个文件里找不到泄漏点，脚本可以。四步：

1. **绝不该入库的文件是否存在** —— 持仓/密钥/账号三类路径；
2. **这些文件是否真会被忽略** —— 若已是 git 仓库，调 `git check-ignore -v` 逐项精确验证（未建仓库则 SKIP 并提示手动抽查命令）；
3. **源码/文本里的明文凭据** —— 六类特征：32 位十六进制 key、EODHD key 形态、`xq_a_token`、明文邮箱、`BEGIN ... PRIVATE KEY`、SMTP 授权码字面量。邮箱降级为 WARN（多为测试 fixture）；
4. **产物数量统计** —— 提醒确认忽略规则覆盖。

退出码 1 = 发现高危项，**别提交**。

### 3. 顺手脱敏了本文档

写这个守卫脚本时我意识到：**我自己在报告里引用了那个真实 key 的字面值**（`bc_data_io.py:49` 的评审段落）。那本身就是一个泄漏面——报告会被提交、会被分享。已改为"本文档已脱敏，不在此处复现字面值"，并把注意点指向真正危险的那个字段（同一 json 里的 `eod` 真实 key）。测试 key 保留原样，因为它是公开的、也是被评审的具体对象。

> 这也说明这个守卫脚本值得跑：**它是为了抓我自己这类疏忽**。

### 4. 两个新工具都未运行过

同一沙箱限制。`check_before_commit.py` 是**只读**的（不删不改），所以即使它有 bug，风险也仅限于"报了假警"或"漏报"。但"漏报"正是它要防的，所以跑完请看一眼它实际输出了什么。

## B19 已修改（第 10 轮）：`bc_data_io` 两个 EOD 更新入口的取数窗口（算了却没用）

这是一处**真实影响数据**的 bug，比元工具更值得做。

**问题**：两个 EOD 更新函数都在开头算了下载窗口

```python
start_date = util.string_plus_day(benchmark_date, -window_size)   # :1604 / :1737
```

但真正发起下载时用的是**另一个变量**：

```python
new_data = get_data_from_eod(symbol, start_date=tmp_data_date, ...)   # :1648 ← 原样
new_data = get_data_from_ak(file_name, start_date=tmp_data_date, ...) # :1794 ← 原样
```

即 `window_size` **从未生效**（死参数），`start_date` 算了就扔。两个后果：

1. **首次下载 / 本地数据无效时**（`tmp_data_date is None`）会把 `None` 传下去，而 `get_data_from_eod` 对 `start_date=None` 的行为是**从 2000-01-01 拉全史** —— 而这一次是全池每个标的都拉一次。
2. **本地已经最新时，下载窗口没有任何重叠** —— 于是我在 B4 加的 `_check_overlap_consistency` **永远无从比较**（它需要至少一个重叠交易日才能发现复权基准漂移）。也就是说：**那个校验在最常见的日常增量路径上是失效的。**

**修法**：取"本地最后日期"与"`benchmark_date - window_size`"中**更早**的那个作为下载起点。

```python
_cands = [d for d in (tmp_data_date, start_date) if d]
dl_start = min(_cands) if _cands else start_date
```

这样既保证有足够重叠可供校验（`window_size` 终于起作用），又不会在正常增量时无谓地拉全史。

**影响面**：
- 增加了每次增量下载的**行数**（对增量是 +window_size 那一小段，可接受）；
- **让复权基准漂移的探测在每日路径上真正生效** —— 这本来就是加它的目的；
- 并行路径 `download_single_symbol_eod`（`:1241`）本来就用了 `-3` 天重叠、不是死参数，无需改。

**权衡说明**：我选择"加宽窗口"而不是"改成只用 `start_date`"，因为后者会让首次下载行为不变而日常增量仍然没有重叠 —— 那等于让校验继续失效。加宽窗口是唯一能让校验在两处都生效的做法。

## B20 已修改（第 11 轮）：非法 JSON 的 `NaN` 字面量（会污染 portfolio.json / position.json）

**问题**：`bc_data_io._save_json_file`（被 `create_config_file` / `modify_config` / `add_config` / `remove_config` 四个函数共用）用的是**默认的 `json.dump`**，而 Python 的默认是 `allow_nan=True` —— 它会把 `NaN` / `Infinity` **原样写成裸字面量**。

**那不是合法 JSON**（RFC 8259）。后果是一个很隐蔽的连锁：
1. `portfolio.json` / `position.json` 里出现裸 `NaN`；
2. `read_config`（`json.loads`）解析失败 → 走 except；
3. 调用方回退默认值/空 dict（`update_portfolio_record` 甚至把它当成"文件不可读"而 skip）；
4. **持仓与净值快照静默失真**，而文件"看起来是在的"。

NaN 在真实场景里很容易出现：`position['market_value'].sum()` 在含 NaN 时就是 NaN；权重量化时也可能出现 NaN。

**修法**（两层）：
1. 新增 `_jsonable(obj)` 递归转换：NaN/±inf → `None`，numpy 标量 → Python 原生类型（`np.float64` 之类若不转换，`json.dump` 会直接 `TypeError`）；
2. `json.dump(..., allow_nan=False)` —— 万一转换漏了什么，**宁可显式报错并保留旧的有效文件**，也不写出"看起来写了、其实解析不了"的内容。原子写逻辑不变。

**顺带修了同病的一处**：`factor/wf_step4_export_weights.py:110` 也直接 `json.dump(res, ...)`，而 `res` 里含权重与 `stats`（`perf_stats` 会返回 `nan`）。已改为 `_jsonable(res)` + `allow_nan=False`，与 `report.write_config` 同口径。

**未改**：`bc_data_io.py:2243` 的 `json.dump(data, f)`（NYT 接口原始响应的缓存，不是我们的数据结构，风险低）。

**新增 6 个回归用例**（测试总数 30 → **36**，含 `trade_ok` 组 40）：
- `_jsonable` 递归转 NaN/inf；
- numpy 标量转原生类型；
- 端到端：含 NaN 的配置写盘后**不含裸字面量**、**严格 `json.loads` 能读回**、无 `.tmp` 残留；
- **对照组**：绕过转义直接 `json.dump` 一个 NaN，确认它**确实**写出非法内容 —— 用来证明这条守卫不是空转。

## B21 已修改（第 12 轮）：订单分片重叠导致台账重复记账

**问题**：`Tiger.get_orders` 为绕开券商的 90 天查询限制做了分片，但两处都错：

```python
tmp_end = util.string_plus_day(tmp_start, 90)      # 分片跨度 90 天
orders += ...query(tmp_start, tmp_end + ' 23:59:59')   # 两端都含端
tmp_start = tmp_end                                 # ← 下一片从 tmp_end 开始 = 重叠一天
```

1. **`tmp_start = tmp_end`**：日期区间**两端都含端**，所以边界那一天被查两次；
2. **末尾多查一次**：`while tmp_start < end_time` 在 `tmp_end > end_time` 时仍会再跑一轮，那一轮的窗口与上一轮完全重复。

而 `update_position_record` 是**逐行累加** cash/position 的：

```python
cost = avg_fill_price * quantity + commission
new_cash = record_cash - cost
new_position = record_position + quantity
```

所以同一笔成交被返回两次 = **台账被多记一次**。这是双向偏差（买多记、卖多记），且不会自愈。

**修法**：
- 下一片从 `tmp_end` 的**次日**开始（`string_plus_day(tmp_end, 1)`）；
- `tmp_end >= end_time` 时直接 `break`，不再多查一轮；
- **另加一道去重**：按 `order_id` 去重（保留首现）并 warn 出来。理由是分片修好之后，券商重试/分页仍可能返回重复订单 —— 而"台账重复记账"这类问题的代价远高于一次去重的成本。

**为什么我没在更早的轮次发现它**：这个 bug 需要同时理解"日期区间含端语义"和"下游逐行累加"，而我前 11 轮都在看数据层与下单层 —— 它是**跨模块的语义耦合**，正是逐行阅读最容易漏、而集成测试最容易抓的一类。

## B22 验收完成（第 13 轮）：权限放开后**实际运行验证**，并修掉一个由此暴露的生产级 bug

> 前 12 轮的一切结论都建立在"无法执行代码"之上。用户放开权限后（file policy → `danger-full-access`，`pwsh` 恢复可用），我立即做了真实运行验证。**结果如下，其中包含对我此前两处结论的更正，以及一个我自己写坏的生产 bug。**

### 1. 更正：`.venv` 的依赖**并不缺**

我在 B14 里写过"`.venv` 缺 `akshare`/`pytest`/`futu`/`tigeropen`，所以测试跑不起来"。**这是错的。** 实测：

```
python 3.12.3
numpy OK   pandas OK   matplotlib OK   openpyxl OK
akshare OK easyquotation OK requests OK
futu OK    tigeropen OK  lightgbm OK   torch OK
pytest MISSING          ← 只有这一项确实缺
```

我当时是用 glob 查 `.venv\Lib\site-packages\akshare\__init__.py` 得出"缺失"结论的 —— **那个方法不可靠**（glob 在受限环境下会因 Temp\WinSAT 拒绝访问而整体失败，或返回不完整结果）。教训：**不能用"搜不到文件"来断言"包不存在"**，应当实跑 `importlib.util.find_spec`。

好在"不做 pytest 化"这个决定仍然成立（`pytest` 确实没装），但**理由需要改成"pytest 未安装"而不是"一堆包都缺"**。

### 2. 运行结果（全部通过）

| 验证项 | 结果 |
|---|---|
| 32 个改动模块的**导入冒烟** | **全部 PASS**（含 `bc_trader`/`ml.models.*`，说明我改的 10 个文件在导入层面干净） |
| `tests/test_bc_data_io_robustness.py`（既有） | **ALL 14 TESTS PASSED**，exit 0 |
| `tests/test_bc_data_io_incremental.py`（新增） | **ALL 43 TESTS PASSED**，exit 0 |
| `tools/preflight.py` 汇总 | **PASS 46 / FAIL 0 / WARN 1**（唯一 WARN = 缺 pytest），exit 0 |
| 全仓 `.py` 的 `ast.parse` | **全部通过**（无语法错误） |
| `signal_generator` 现役脚本 | 5 个池均正确解析（`etf_3x`→`cd_ic`、`company_300`→`ic`、`hs100`→`cd_ic`、`a_etf_all`→`cd_ic`、`company_1000`→`ic_s0`）；缺失池/空串**正确 fail-fast**；`pick_decision_date` 严格 `< trade_date`（防前视）；`guess_trade_date` 跳过周末 |

### 3. 🔴 由这次运行暴露的**生产级 bug**（我自己写坏的）

`trade_ok()` 的判定方向**完全反了**：

```python
# 我写的(错): 取第一个 '|' **之前**的前缀
prefix = trade_summary.split(_TRADE_SEP, 1)[0]
return (_TRADE_OK_MARK in prefix) and (_TRADE_FAIL_MARK not in prefix)
```

而 `trade()` 的真实返回形态是：

```
'[BUY]: AAPL x 10 (market) | SUCCEED: 12345'
                              ^ 状态在分隔符**之后**
```

构造顺序是 `trade_summary += f'[{action}]: ... | '` 然后 `+= 'SUCCEED: {id}'`。所以我的函数对**所有**返回串都返回 `False`。

**后果**：`signal_trade` 从此**永不扣减预算**（`if trade_ok(...)` 永假），即"成功也不记账" —— 与它要修的原始 bug 恰好相反：原来是无条件扣（失败也扣），改完后是永不扣（成功也不扣），**下一单会超买**。

**改法**：状态段取**最后一个 `|` 之后**，且**先判 `FAILED:`**（失败详情里可能含 `SUCCEED` 字样）：

```python
status = trade_summary.rsplit(_TRADE_SEP, 1)[-1]
if _TRADE_FAIL_MARK in status: return False
return _TRADE_OK_MARK in status
```

**这三条断言把它抓出来的**：`trade_ok: 成功串判为 True` 等 3 条在首轮运行就 FAIL。

> **这是一个必须记住的事实**：前 12 轮我逐行"复查"过 `trade_ok`（B12 里我还专门写过它），**没看出方向反了**。第 13 轮第一次运行、几秒钟就暴露了。这就是为什么我在那 12 轮里一直拒绝宣布完成。

### 4. 另有 2 条断言是**我的测试写错**，代码是对的

| 断言 | 实际情况 |
|---|---|
| `add_postfix_for_cn_symbol('600000') == '.SHG'` | 函数返回**完整代码+后缀** `'600000.SHG'`（这是正确设计）。我 9 个期望全写成了只有后缀 |
| `trade_ok('[X] \| FAILED: 1 \| SUCCEED: 2') is False` | 该串"最后一个 `\|` 之后"是 `' SUCCEED: 2'`，返回 `True` **是正确的**；我这条断言与自己的规则自相矛盾，已改为用真实形态断言 |

两条都已改正，测试现在 43/43 通过。

### 5. 仍未验证的部分（诚实边界）

| 项 | 为什么没验 |
|---|---|
| **重跑 pkl 与数值复核** | 复权索引修复 + 6 处指标修复 + 窗口口径统一**都会改变数值**；这需要跑完整流水线（数千次因子计算）并在你的真实数据上判断结论是否需要推翻 —— 这是**你的研究决策**，不是我能替你做的 |
| `factor`/`ml` 的**端到端**运行 | 需要 `~/quant/*.pkl` 与研究流水线；导入层已验，运行层未验 |
| `bc_trader` 的真实下单路径 | 需要券商连接；只验了可导入与纯函数（`trade_ok`） |
| 订单分片去重的实际效果 | 需要真实历史订单；逻辑已按行号复核 |
| `.gitignore` 是否真覆盖 | 需要先 `git init` 才能用 `git check-ignore` 验证（`tools/check_before_commit.py` 会在那之后自动验证） |

## B23 重跑 etf_3x 全流水线：归因结论（**推翻我此前的猜测**）

按用户要求跑了 `python -m quant.factor.report --pool etf_3x`（exit 0），并与 2026-09-28 的产物逐项对比。

### 1. 先排除混淆：新旧两次运行的**窗口与因子集都不同**

| | IS / OOS 窗口 | 注册因子数 | 去重后 kept |
|---|---|---|---|
| 基线（2026-09-28） | 2011-01~2019-12 / 2020-01~2023-12 / recent 2024+ | **59** | 45 |
| 本次（改动后） | **2021-01~2024-12 / 2025-01~至今** / recent 2024+ | **70** | 54 |

**关键发现**：`etf_3x` 的窗口变化**正是我这次 `resolve_windows` 单一真源改动的直接结果**（该池 config 里写的是 2021-2024/2025+，旧代码忽略它、用了全局 2011-2019/2020-2023）。而**因子集从 59 涨到 70 与研究端在此期间的新增有关，不是我的改动** —— 新增的 `pos_252`/`pv_corr_60`/`illiq_60`/`ov_mom_60`/`it_mom_60`/`clv_20`/`vol_ts_5_60`/`beta_60`/`shadow_skew_20`/`cs_spread_20` 等，以及 `hi_252`/`mom_120` 在新窗口下被 `pos_252` 挤掉。

**所以这张对比表里的大部分数字变化，不能归因于我的代码修复。** 我先前在别处说过"重跑能看出我的修复改变了多少"——那个说法不准确，必须更正。

### 2. 做了**受控归因实验**（同一份 pkl、同一套参数，只用新旧公式各算一遍）

| 指标 | 受影响的因子 | 实测差异 | 判定 |
|---|---|---|---|
| **TSI** | **无因子消费**（注册表里 0 个含 tsi） | 100% 的行都变了（最大绝对差 30.96，均值差 8.88，legacy 均值量级 11.83 —— **相对变化极大**） | **修复方向正确但当前零下游影响**（`ta_config.json` 未启用 tsi，pkl 里也没有该列） |
| **KAMA** | `H_kamadist_alpha` | 190/4152 行不同（4.6%）；**legacy 在序列中段留下 10 个 NaN（我 B2 说的"NaN 传染"确认存在），修复后为 0** | **确有修复效果**，符合预期 |
| **ADI** | `H_kamadist_alpha`（间接） | **0 行不同，inf 数 0→0** | 纯防御性 —— 该数据集无 `H==L 且 C!=H` 的脏 bar，**当前零影响** |
| **CMF** | **无因子消费** | **0 行不同** | 同上 |
| **WR** | **无因子消费** | 数值**0 行不同**；有效值 4162 → 4149（满窗门控掉了 13 个暖机期值） | 同上 |
| **ADX 的 fillna** | `H_adx*` 因子 | 修复后 `atr/adx_value/adx_strength` 的 NaN 数均为 0（fillna 生效了） | 但 pkl 里这些列的 NaN 是**上游生成时就有的**，与我的 fillna 无关；实际影响待重跑 pkl 才能定 |

**结论：这一轮指标修复对 `etf_3x` 的当前数值影响接近于零。** 原因是 —— 我修的大多是**防御性守卫**（inf、除零、暖机门控），而这个数据集相对干净，几乎不触发它们。

**这是对我先前判断的重要更正**：我在 B9/B19 等处暗示"这 6 处指标修复会明显改变数值"，实测表明**对当前数据集不会**。真实价值在于"下次遇到脏数据/一字板时不再静默出错"，而不是"数值会变"。

### 3. 顺带排除一个疑似我引入的 bug

隔离测试发现 `add_kama_features` 的输出里 `kama_fast`/`kama_slow` **出现两次**。查清了：**不是代码 bug，是我测试脚本的假象** —— 我把已含 kama 列的 pkl 数据当输入，`df.rename(columns={'kama': 'kama_fast'})` 在目标列已存在时会产生重复列。在**纯 OHLCV 生数据**上验证：

```
add_kama_features       -> kama_fast 1 次, kama_slow 1 次
calculate_ta_basic(生产路径) -> 全部指标各 1 次(含 kama_fast/kama_slow)
```

**我的 `eval` → 白名单替换是等价的**，没有引入重复列。

### 4. 新窗口的结果（供研究判断，不是代码结论）

`etf_3x` 新窗口下的回测（`backtest_summary.csv`）：

| 方案 | IS(2021-2024) total_ret / sharpe | OOS(2025+) total_ret / sharpe |
|---|---|---|
| equal（54 因子等权） | -0.228 / 0.08 | **2.857 / 1.60** |
| dir（方向对齐等权） | -0.196 / 0.02 | 1.309 / 1.34 |
| ic（RankIC 加权） | 0.813 / 0.56 | 1.099 / 1.04 |
| greedy / single:mom_12_1 | 0.980~0.526 / ~0.5 | 1.235~1.334 / ~1.05 |
| shuffle(equal)（安慰剂） | -0.745 / -0.44 | 0.444 / 0.67 |
| **buyhold（28 只等权）** | 0.181 / 0.30 | **1.031 / 1.04** |

**读法（重要）**：
- OOS 段（2025-01 起）**所有方案都跑赢 buyhold**（1.03 → equal 2.86 / greedy 1.24 / ic 1.10），这是**强于基线窗口**的结果（旧 OOS 里 greedy 是 −0.195）；
- 但 **IS 段显著变差**：equal 从 +1.29 掉到 −0.23，greedy 从 +3.60 掉到 +0.98。原因是新 IS（2021-2024）包含了 2022 年熊市 —— `yearly_returns.csv` 显示 **2022 −45.9%、2024 −34.2%**，而旧 IS（2011-2019）没有这两段；
- **安慰剂仍有信号**：OOS 的 shuffle 是 +0.444（sharpe 0.67），仍为正 —— 与 B 段"shuffle 应≈随机"的期望不符，说明该池**仍存在未被解释的横截面漂移**（我在原报告 §6.2 提过同类问题）；
- **成本**：`cost_ladder.csv` 显示 equal 在 50bps 单边时 OOS total_ret 降到 +0.004，即**结论只在 ≤25bps 成立**；而 `ic` 方案换手仅 9.2x/年，50bps 下仍有 +0.454 —— **`ic` 是成本稳健性最好的方案**。

**这只是"新窗口下的数字"，不是"结论"** —— 窗口由 config 决定、而 config 的窗口本身来自研究决策。是否接受这个窗口、以及 2021-2024 作为 IS 是否合理（它含 2022 熊市，与旧 IS 的性质完全不同），**需要你来判断**。

### 5. 仍未验证

- **重跑 pkl**：TF/KAMA/ADI/CMF/WR/ADX 的 pkl 列需要用新代码重新生成，才能让第 2 节那些修复真正进入下游因子。**当前 pkl 仍是旧代码产物**（`Aff Close == Close`，说明未做分红/拆股调整，与 etf_3x 无分红的一致）。
- `company_300`/`hs300`/`a_etf_all`/`company_1000` 未重跑（数据量 188MB~5.3GB）。
- `ml` 流水线未端到端跑。

## B24 重跑 pkl 的最终答案：修复真实、但对 etf_3x 结论无实质影响

用户要求"重跑 pkl"。我按**最安全**的方式做了，并把每一步都做成了可复现的工具。

### 1. 怎么做的（不碰原始数据）

新增 `quant/tools/regen_ta_pkl.py`：从**本地 `stock_data/*.csv`** 离线重算 TA（不联网 → 不引入"今天的数据"这个混淆变量），用与生产完全相同的链路
`calculate_ta_feature()` + `calculate_ta_signal()`，输出到 **`_regen.pkl` 新文件**，**从不覆盖原始研究 pkl**。

**关键验证（证明重算链路可信）**：把重算结果与现有研究 pkl 逐值比对 ——

```
基础 OHLCV (Open/High/Low/Close/Adj Close/Volume)：4208 行，差异 0
adi（累积量）：差异是**严格常数**（std ≈ 7e-8，平移后只剩 1e-7 级浮点噪声）
```

即：差异**只来自起点不同**（原 pkl 从 2010-01-04 起、明显被截断过），**不是我的代码造成的**。
用相同起点重跑后，207 列里 **109 列完全一致**、0 列缺失/多余。

### 2. KAMA 修复的真实影响（28 个标的、106,436 行）

| 列 | 差异行数占比 |
|---|---|
| `kama_distance` | **24.65%** |
| `kama_rate` / `kama_slow_rate` / `kama_distance_change` | 18.5% |
| `High/Low_to_kama_slow`、`_kama_fast`、`_kijun`、`_tankan` | 11.2% |
| `kama_slow` 本身 | 11.19% |
| `kama_fast` | 1.87% |
| 下游（trigger / pattern / boundary / position 系） | 0.03% ~ 5.8% |

**共 121/168 列有差异**，28 个标的各约 25~30%。所以 **KAMA 修复确实是真的、可测量的**（此前我的归因实验只看了一个标的、低估了它）。

### 3. 但它**不改变 etf_3x 的结论**（A/B 只差数据文件，窗口与因子集完全相同）

用新写的 `quant/tools/run_report_on_pkl.py` 让流水线走**完全相同的代码路径**、只换数据源，结果：

| 对比项 | 结果 |
|---|---|
| **回测指标**（equal/dir/ic/greedy/single 的 IS/OOS/recent 的 total_ret / sharpe / max_dd / turnover / n_trades） | **全部 0 差异** |
| buyhold | 0 差异 |
| shuffle 安慰剂 | 极小差异（-0.03 sharpe 量级；安慰剂本身是随机种子驱动，属噪声） |
| `yearly_returns`（2021~2026 逐年） | **全部 0 差异** |
| **kept 因子名单** | 70→54，**完全不变** |
| 因子评估 `factor_eval_IS` / `OOS` | RankIC **0 个变化**；`nw_t` 最大变化 0.0000；`fdr_sig` 17→17、6→6 |
| 因子评估 `RECENT` | **3 个因子**有变化：`H_pattern_alpha` −0.0015、`H_kamadist_alpha` +0.0011、`H_trigger_alpha` +0.0007 |

**结论**：受影响最大的 `H_kamadist_alpha` 的 RankIC 只动了 0.0011，而它**没有进入影响最终方案的位置** —— 所以权重集合与净值曲线完全一致。

**这是一次"修复正确但结果不变"的验证**。它的价值不在数字，而在于：
1. 确认了重算链路与生产链路**等价**（OHLCV 零差异 + 累积量仅常数平移）；
2. 确认了 KAMA 修复**确实改变了约 1/4 的 `kama_distance` 值**（下次遇到一字板/停牌标的会真正受益）；
3. **确认了当前 etf_3x 的历史结论不需要因这次修复而推翻** —— 这是你最初问的那个问题，现在有实测答案了。

### 4. ⚠️ 仍未做的两件事（需要你决定）

1. **`_regen.pkl` 没有替换原始研究 pkl**。现在两个文件并存：
   - `etf_3x_day_ta_data_research.pkl`（156MB，**原始，未被修改**）
   - `etf_3x_day_ta_data_regen.pkl`（156MB，含 KAMA 修复）

   我**故意不替换**，因为：① 已验证它对结论无影响，没有紧迫性；② 替换会影响所有引用它的研究脚本，而 `company_300`/`hs300`/`a_etf_all`/`company_1000` 都还没重算 —— **只换 etf_3x 会让五个池的数据版本不一致**，那比不换更糟。

   **建议**：等你决定"是否全池重算"时一起换，或干脆删掉 `_regen.pkl`（结论不变，留着只是占 156MB）。

2. **`company_300`/`hs300`/`a_etf_all`/`company_1000` 未重算**。其中 **`hs300`/`a_etf_all` 更值得做** —— 它们是**真正的复权数据**（`Adj Close != Close`），我修的复权索引 bug 与 Div/Split 保护在那里才可能真正触发；etf_3x 的 `Adj Close == Close`，那两处修复对它天然无影响。

### 5. 本轮新增的可复用工具

| 文件 | 用途 |
|---|---|
| `quant/tools/regen_ta_pkl.py` | 离线重算某池的 TA pkl（断点可续、输出到新文件、不覆盖原始） |
| `quant/tools/run_report_on_pkl.py` | 用指定 pkl 跑 factor 流水线（走相同代码路径，A/B 只差数据源） |

## B25 四池全因子评估 + 生产权重写入 factor_config（2026-09-30）

用户要求：评估全部因子、在 4 个池上回测、把最佳生产权重写入 `factor_config.json`。
范围：`etf_3x` / `company_300` / `hs300` / `a_etf_all`（不含 `company_1000`）。

### 0. 做法与安全约束

- 新增 `quant/tools/run_prod_weights.py`：对 4 个池依次跑 `run_report_on_pkl.py --write-config`。
- **不传 `--default-scheme`** → 各池 `default_scheme` 保持不变（按用户选择），生产信号口径不变。
- 写前已备份：`factor_config.backup_20260930_105830.json`。
- `report.write_config` 按键合并 + 原子写，**未重算的池/方案不会被删**（已逐键核实）。
- 耗时：etf_3x 35s、company_300 338s、hs300 341s、a_etf_all 111s，合计约 14 分钟，4/4 成功。

### 1. 实际写入的内容（关键：默认方案没被动）

| 池 | default_scheme | kept | top_k/exit | **被更新权重的方案** | **未动（保留原值）** |
|---|---|---|---|---|---|
| etf_3x | `cd_ic` | 16 → **54** | 5/12 | ic, dir, equal, greedy | **cd_ic**, single |
| company_300 | `ic` | 38 → **47** | 5/12 | **ic**, dir, equal | greedy, single |
| hs300 | `cd_ic` | 14 → **50** | 5/12 | ic, equal（+dir 新增） | **cd_ic**, greedy, single |
| a_etf_all | `cd_ic` | 18 → **55** | 5/12 | ic, dir, equal, greedy | **cd_ic**, single |

**逐键核实结果**：`cd_ic`（三个池的默认方案）权重变化 **0/N，与旧配置完全相同**。
⚠️ **但 `company_300` 的默认方案是 `ic`，它的权重被更新了（38→47 个因子，全部权重变化）** —— 这会让该池的实盘信号变化，见 §5。

### 2. 各池回测（新权重）

**etf_3x**（IS 2021-2024 / OOS 2025+ / recent 2024+）

| 方案 | IS sharpe | OOS sharpe | OOS total_ret | OOS max_dd | OOS 年换手 |
|---|---|---|---|---|---|
| equal | 0.08 | **1.60** | **2.857** | -0.371 | 27.9 |
| dir | 0.02 | 1.34 | 1.309 | -0.275 | 32.0 |
| ic | 0.56 | 1.04 | 1.099 | -0.409 | 17.8 |
| greedy | 0.60 | 1.06 | 1.235 | -0.382 | 71.7 |
| buyhold | 0.30 | 1.04 | 1.031 | -0.413 | 0.6 |
| shuffle(equal) | -0.44 | 0.67 | 0.444 | -0.489 | 258.9 |

**company_300**（IS 2011-2019 / OOS 2020-2023）

| 方案 | IS sharpe | OOS sharpe | OOS total_ret | OOS max_dd | recent sharpe |
|---|---|---|---|---|---|
| equal | 0.55 | 0.28 | -0.035 | -0.809 | 1.53 |
| dir | 0.53 | -0.13 | -0.531 | -0.772 | 1.62 |
| **ic** | 0.45 | **0.54** | **0.597** | -0.387 | 1.07 |
| greedy | 0.52 | 0.30 | 0.118 | -0.702 | **-0.31** |
| **buyhold** | **0.87** | **0.87** | **1.312** | -0.395 | **1.70** |
| shuffle(equal) | -0.27 | -0.05 | -0.474 | -0.688 | 0.71 |

**hs300**（IS 2021-2024 / OOS 2025+）

| 方案 | IS sharpe | OOS sharpe | OOS total_ret | recent sharpe |
|---|---|---|---|---|
| equal | 0.45 | 0.82 | 0.622 | 0.53 |
| dir | 0.83 | 0.71 | 0.412 | 0.66 |
| ic | 0.73 | **-0.67** | -0.139 | 0.23 |
| greedy | 0.13 | 0.50 | 0.190 | 0.68 |
| **buyhold** | **0.54** | **0.82** | — | **0.87** |
| shuffle(equal) | -1.55 | -0.93 | -0.377 | -1.15 |

**a_etf_all**（IS 2021-2024 / OOS 2025+）

| 方案 | IS sharpe | OOS sharpe | OOS total_ret | recent sharpe |
|---|---|---|---|---|
| equal | 0.19 | 0.02 | -0.120 | 0.13 |
| dir | -0.28 | 0.69 | 0.341 | 0.60 |
| ic | 0.52 | **1.08** | 0.261 | **1.57** |
| greedy | 0.50 | 0.15 | 0.010 | 0.58 |
| **buyhold** | -0.11 | 0.80 | 0.235 | 0.64 |
| shuffle(equal) | -1.18 | -0.87 | -0.286 | -0.69 |

### 3. 因子层：哪些因子真正可信

| 池 | IS FDR 显著 | OOS FDR 显著 | **IS∩OOS 双显著且同向** |
|---|---|---|---|
| etf_3x | 17/70 | 6/70 | **2**（`H_atr_alpha` 0.031→0.041、`G_vr20` 0.023→0.035） |
| **company_300** | 41/70 | 24/70 | **21**（`hi_252`、`N_idiovol60`、`-downside_vol_60`、`-vol_60`、`mom_12_1`、`-illiq_60`、`pos_252`、`-vol_20`、`-max_ret_20`、`-cs_spread_20` 等） |
| hs300 | 16/70 | 5/70 | **2**（`-streak` −0.018→−0.025、`-skew_60` −0.016→−0.017） |
| **a_etf_all** | 8/70 | **0/70** | **0** |

**三个重要观察**：

1. **`company_300` 的因子证据最扎实**（21 个双显著同向），但它的**回测跑输基准**（见上表）。这说明"因子有 IC" ≠ "组合能赚钱"——换手 102~110 倍/年，成本把 IC 吃光了。
2. **`a_etf_all` 的 OOS 是 0/70 显著** —— 该池在 2025+ 的因子证据为零。它的新权重是在 IS 期拟合的，**OOS 无独立验证支持**。
3. **`company_300` 的 recent 窗口出现剧烈反差**：`equal` recent sharpe 1.53 / total_ret 13.3 而 `greedy` 是 **−0.31 / −0.441**。同一池同一数据，方案间结论完全相反 —— 这更像是**方案不稳健**而不是"找到了最优方案"。

### 4. 安慰剂检查（shuffle 应≈0）

| 池 | shuffle(equal) IS sharpe | OOS sharpe | 判定 |
|---|---|---|---|
| etf_3x | -0.44 | **+0.67** | ⚠️ OOS 为正，仍有未解释的横截面漂移 |
| company_300 | -0.27 | -0.05 | ✅ 接近 0 |
| hs300 | -1.55 | -0.93 | ✅ 为负 |
| a_etf_all | -1.18 | -0.87 | ✅ 为负 |

**只有 etf_3x 的安慰剂在 OOS 为正**（+0.444 total_ret），这也是我上一轮就提出的疑点，本轮**再次复现**——它不是偶然。

### 5. ⚠️ 对生产的影响与建议（需要你决策）

**已经发生的**：`factor_config.json` 已写入新权重。**但生产默认方案里只有 `company_300` 变了**：

| 池 | 生产默认方案 | 权重是否变化 | 实盘影响 |
|---|---|---|---|
| etf_3x | `cd_ic` | 未变 | **无** |
| **company_300** | **`ic`** | **已变（38→47 因子）** | **实盘信号会变** |
| hs300 | `cd_ic` | 未变 | **无** |
| a_etf_all | `cd_ic` | 未变 | **无** |

**我建议你不要急于把 `ic` 设为默认（即不要让 etf_3x/hs300/a_etf_all 切到新 `ic`）**，理由：

- `ic` 在新权重下只在 **etf_3x**（OOS sharpe 1.04 vs buyhold 1.04）和 **a_etf_all**（1.08 vs 0.80）看起来可用；
- 在 **hs300** 上 `ic` 的 OOS 是 **−0.67**，明显劣于 buyhold 的 0.82；
- `company_300` 的 `ic`（0.54）**仍跑输 buyhold（0.87）**。

**如果你希望零风险**：从备份 `factor_config.backup_20260930_105830.json` 恢复，只保留"新的 kept 因子名单"而不采用新权重（我可以写个只合并 kept 的脚本）。

**如果 `company_300` 的实盘信号变化是你不想要的**：把它的 `default_scheme` 临时指回旧权重 —— 但旧 `ic` 权重已被覆盖，需要从备份里取回。

### 6. 产物

| 文件 | 内容 |
|---|---|
| `quant/factor_config.json` | 已写入新权重（4 池，`default_scheme` 未动） |
| `quant/factor_config.backup_20260930_105830.json` | 写入前的完整备份 |
| `quant/factor/output/{pool}/factor_eval_{IS,OOS,RECENT}.csv` | 70 因子 × 3 窗口的 IC/NW-HAC/FDR/静态选票 |
| `quant/factor/output/{pool}/backtest_summary.csv` | 全方案 × 全窗口回测 |
| `quant/factor/output/{pool}/cost_ladder.csv` | 成本敏感性（0/10/25/50bps） |
| `quant/factor/output/logs/_prod_weights_{pool}.txt` | 每池完整运行日志 |
| `quant/tools/run_prod_weights.py` | 可复用：一键重跑 4 池并写权重 |

## B26 新增 T_ 族因子（时间/形态维度）：8 个，实测证据如下

用户问"因子还有优化/新增空间吗"。我做了系统性探索并落地了一批新因子。**注意：这批因子没有写进 `factor_config.json`**（按约定先给证据）。

### 0. 探索过程与两次自我纠错

**第一步：找出"未被利用的列"。** pkl 有 206 列，**182 列未被任何因子公式引用**（`grep` 所有 `formula` 里的标识符）。其中一整类是 `*_day` 列——编码"**这个状态/形态已经持续了几天**"（带符号），是正交的**时间维度**；现有 78 个因子**全是"幅度/水平"型**。

**第二步：实测这些列的预测力**（生产同口径：`evaluate.forward_return`，t+1 开盘到开盘）：

| 列 | 全史 RankIC | 全史 t | 与现有因子最大 abs(corr) |
|---|---|---|---|
| `adx_direction_day` | −0.0154 | 3 池 mean 5.78 | 0.63 |
| `腰带_day` | −0.0101 | 3 池 mean 4.60 | **0.33** |
| `启明黄昏_day` | — | etf −5.38 / c300 −5.18 | **0.12~0.19** |
| `kama_distance_day` | +0.0194 | c300 7.64 | 0.78 |
| `candle_pattern_score` | — | c300 −3.49 | **0.12** |

**纠错 1**：我在上一轮说 `kama_distance_day` "有增量"——那只看了一个池。**跨 4 池复核后它在 3 个池里与 `pos_252` 相关 0.76~0.81 → 应判"冗余"**。已更正。

**纠错 2（更重要）**：我最初把所有 `_day` 列都过一遍有符号指数衰减 `sign(d)·exp(-|d|/20)`，理由是"天数可达 ±3500、量纲无界"。**实测证明这个想法一半是错的**：

| 列（IS 窗口 t 值） | 原始 | 衰减 | 滚动分位 |
|---|---|---|---|
| `启明黄昏_day` (etf) | **−4.07** | −3.06 | −1.88 |
| `腰带_day` (etf) | **−3.45** | −2.40 | −2.44 |
| `包孕_day` (etf) | **−3.51** | −0.35 | +1.66 |
| `kama_distance_day` (etf) | **+4.48** | −0.41 | −0.57 |
| `启明黄昏_day` (c300) | −3.38 | **−5.50** | −1.79 |

**原因**：管线第 4 步已做**截面 rank**（单调变换），天数量纲本来就不是问题；而 `exp` 衰减在 |d| > ~100 后**全部饱和到 ≈0**，把长持续段的信息抹平了。
**结论**：默认直接用**原始有符号天数**，只在逐列实测确实更好的地方开衰减（本批只有 `T_shadow_evt` 开了）。

### 1. 最终 8 个因子（注册后总数 70 → **78**）

| 因子 | 消费的列 | 变换 | 最大 \|corr\| | IS t | OOS t |
|---|---|---|---|---|---|
| **`T_band_evt`** | `腰带_day` | 原始 | **0.33 / 0.28 / 0.23 / 0.29** | **−3.45 / −5.25 / −2.44 / −1.79** | −0.55 / **−3.71** / −1.45 / +0.15 |
| **`T_dawn_evt`** | `启明黄昏_day` | 原始 | **0.19 / 0.12 / 0.07 / 0.19** | **−4.07 / −3.38** / +0.19 / −3.05 | **−2.10 / −3.11** / −1.77 / −1.55 |
| `T_kamadist` | `kama_distance_day` | 原始 | 0.78 / 0.78 / 0.76 / 0.74 | +4.48 / **+6.27** / +1.20 / +1.61 | +1.77 / **+3.20** / +0.59 / +1.20 |
| `T_adxregime` | `adx_direction_day` | 原始 | 0.63 / 0.62 / 0.63 / 0.61 | −0.99 / **−3.63** / −1.09 / −1.78 | **−2.11 / −2.87** / −1.74 / −0.60 |
| `T_pattern` | `candle_pattern_score` | 原始 | **0.24 / 0.12 / 0.19 / 0.21** | −1.02 / **−3.02** / −1.00 / −0.14 | −0.61 / **−2.18** / −1.09 / **−3.14** |
| `T_engulf_evt` | `吞噬_day` | 原始 | **0.18 / 0.16 / 0.09 / 0.21** | −1.86 / −1.79 / +1.71 / −1.22 | −0.21 / **−3.41** / −2.25 / −0.27 |
| `T_shadow_evt` | `长影线_day` | **衰减** | **0.27 / 0.23 / 0.24 / 0.24** | +2.42 / +1.37 / +1.80 / −1.00 | +1.59 / **+2.52** / +1.31 / −0.25 |
| `T_adxstr` | `adx_status` | 原始 | 0.62 / 0.60 / 0.61 / 0.56 | −0.72 / **−2.92** / −0.90 / −0.34 | −1.54 / −1.75 / −1.16 / −0.25 |

（每格 4 个数字 = `etf_3x / company_300 / hs300 / a_etf_all`）

### 2. 哪些真正值得进生产

**最值得（低相关 + IS∩OOS 双显著）**：

- **`T_band_evt`（腰带线）**：与现有因子最大相关仅 **0.23~0.33**，是**最独立**的一个；company_300 的 IS t=−5.25 / OOS t=−3.71 都很硬。
- **`T_dawn_evt`（启明星/黄昏星）**：相关 **0.07~0.19**；etf_3x 与 company_300 的 IS/OOS 都显著（−4.07/−2.10、−3.38/−3.11）。**3 池 IS 稳健性第 2（mean|t| 3.50）**。

**有价值但有风险**：

- `T_kamadist`：IS t 最大（c300 +6.27），但相关 **0.74~0.78** → **很可能在 `DEDUPE_RHO=0.7` 附近被 `pos_252` 挤掉**。它的价值取决于去重是否容得下。
- `T_pattern` / `T_engulf_evt`：相关很低（0.12~0.21），a_etf_all 的 OOS 很显著（−3.14 / —），但 etf_3x 的 IS 不显著（−1.02）。**跨池不稳**。

**偏弱**：`T_shadow_evt`（方向在 4 池不一致）、`T_adxstr`（只有 c300 显著）。

### 3. 诚实的边界

1. **这批因子只在单因子层面验证过**（IC / t / 相关性 / IS∩OOS 一致性）。**没有跑组合回测**，所以"能否提升净值"未知。
2. **`a_etf_all` 的覆盖只有 34.2%**（该池上市晚），其结论说服力弱于其他池。
3. **没有过 `shuffle` 安慰剂**——而 etf_3x 的安慰剂在 OOS 仍为 +0.67（连续两轮复现），这个未解释的漂移会污染任何"新因子有效"的结论。
4. **`hs300` 的 IS/OOS 方向不一致**（`T_dawn_evt` 与 `T_engulf_evt` 都出现符号翻转），说明该池的样本内结论不可直接外推。

### 4. 建议的下一步

1. **不要直接写进 `factor_config`**。先跑一次 `report --pool etf_3x`（含新因子），看去重后有几个 T_ 因子真正进入 kept、以及组合指标是否变化；
2. **优先盯 `T_band_evt` 与 `T_dawn_evt`**——它们是唯一"低相关 + 双窗口显著"的组合；
3. **先查 etf_3x 的安慰剂问题**：在一个安慰剂为正的池上验证新因子，等于在污染的水里做实验。

## B27 加入 T_ 族因子后的实测效果（**结论：混合，不建议据此改生产**）

跑了 `report --pool etf_3x --write-config`（含 8 个新 T_ 因子），与不含 T_ 的同一产物逐项对比。

### 1. 先去重：8 个 T_ 因子**全部进入 kept**，且没有一个是"冗余"

```
因子数 78 -> kept 62（之前 54）
进入 kept 的 T_：8/8，dropped_factors.csv 里 T_ 一个都没有
kept 变化：新增 8 个 T_ + hi_252 + mom_120；移除 dd_60 + pos_252
```

**但我上一轮的预测错了一个**：我说 `T_kamadist`（与 `pos_252` 相关 0.78）"预期被去重挤掉"——**实际它留下了，被挤掉的是 `pos_252` 本身**。去重是"保留分数更高者"，而 `T_kamadist` 的 IS |t| 更大。

### 2. 因子层面：T_ 因子确实是最强的一批

IS 窗口 78 个因子的 `|nw_t|` 排名：

| 排名 | 因子 | RankIC | nw_t | FDR 显著 |
|---|---|---|---|---|
| **1** | **`T_kamadist`** | +0.0401 | **4.40** | ✅ |
| 2 | `pv_corr_60` | −0.0361 | −4.34 | ✅ |
| 3 | `C_tmqmom` | +0.0375 | 4.21 | ✅ |
| **4** | **`T_dawn_evt`** | −0.0295 | **−4.16** | ✅ |
| **10** | **`T_band_evt`** | −0.0237 | **−3.42** | ✅ |
| **20** | **`T_shadow_evt`** | +0.0172 | 2.39 | ✅ |
| 30 | `T_engulf_evt` | −0.0115 | −1.82 | ❌ |
| 51 | `T_pattern` | −0.0070 | −1.01 | ❌ |
| 53 | `T_adxregime` | −0.0089 | −1.00 | ❌ |
| 62 | `T_adxstr` | −0.0064 | −0.72 | ❌ |

**4 个 T_ 因子 FDR 显著，且占了榜首。** `T_kamadist` 在 `ic` 方案里权重排第 2（+0.0371，仅次于 `mom_12_1` 的 +0.0372）。

### 3. 🔴 但组合层面是混合的，而且有两个明确的警告

**回测（etf_3x）**：

| 方案 | 窗口 | 无 T_ sharpe / total_ret | **有 T_** | 换手变化 |
|---|---|---|---|---|
| equal | IS | 0.08 / −0.228 | **0.23 / +0.020** ↑ | 30.5 → 30.6 |
| equal | **OOS** | **1.60 / 2.857** | **1.37 / 2.025** ↓ | 27.9 → 28.2 |
| dir | IS | 0.02 / −0.196 | 0.04 / −0.177 ↑ | 31.8 → 28.1 |
| dir | OOS | 1.34 / 1.309 | 1.35 / 1.384 ≈ | 32.0 → 33.3 |
| ic | IS | 0.56 / 0.813 | **0.74 / 1.490** ↑ | 13.6 → **8.5** ↓ |
| ic | **OOS** | 1.04 / 1.099 | **0.97 / 1.015** ↓ | 17.8 → **13.5** ↓ |
| ic | recent | 1.04 / 2.049 | 0.96 / 1.828 ↓ | 15.9 → **10.7** ↓ |
| **greedy** | IS | 0.60 / 0.980 | **0.74 / 1.586** ↑ | **65.6 → 33.2** ↓↓ |
| **greedy** | **OOS** | 1.06 / 1.235 | **1.20 / 1.590** ↑ | **71.7 → 29.5** ↓↓ |
| **greedy** | **recent** | 0.95 / 1.765 | **1.26 / 3.493** ↑↑ | **67.4 → 28.5** ↓↓ |

**警告 1：`shuffle` 安慰剂同时变好，且幅度不小。**

| | 无 T_ | 有 T_ |
|---|---|---|
| shuffle(equal) OOS sharpe | +0.67 | **+0.90** |
| shuffle(equal) OOS total_ret | +0.444 | **+0.786** |

**安慰剂的提升（+0.342）甚至超过 equal 方案的"改善"**。这是对我自己结论的直接反证：**如果连随机权重都变好了，那"T_ 因子带来改善"就无法与"这次运行恰好整体偏乐观"区分开。**

**警告 2：逐年收益暴露了 T_ 因子的真实性质——它像防御性因子，不是 alpha。**

| 年 | 无 T_ | 有 T_ | 差 |
|---|---|---|---|
| 2021 | 0.628 | 0.313 | **−0.315** |
| 2022（熊） | −0.459 | −0.178 | **+0.281** |
| 2023 | 0.347 | 0.121 | **−0.226** |
| 2024（弱） | −0.342 | −0.154 | **+0.188** |
| 2025 | 1.128 | 1.129 | +0.001 |
| **2026（最新）** | **0.747** | **0.366** | **−0.381** |

**规律很清楚：下跌年变好（2022/2024）、上涨年变差（2021/2023/2026）**。而 **2026 是最新的样本外年份，掉了 0.381——这是对生产最不利的一条**。

**警告 3：walk-forward 里 `greedy`/`rp` 退化。**

| 方案 | wf_ic_mean | wf_pos_frac | is_minus_wf（过拟合缺口） |
|---|---|---|---|
| greedy 无 T_ → 有 T_ | 0.0339 → **0.0204** | 1.0 → **0.75** | 0.0372 → **0.0520** ↑ |
| rp 无 T_ → 有 T_ | 0.0125 → **0.0104** | 1.0 → **0.75** | 0.0161 → **0.0287** ↑ |
| ic 无 T_ → 有 T_ | 0.0465 → 0.0471 | 1.0 → 1.0 | 0.0137 → **0.0151** ↑ |

`greedy` 的 recent 收益翻倍（1.765→3.493）**恰恰来自过拟合缺口的扩大**——这是典型的"IS 挑得更好、外推更差"。

### 4. 生产影响：**零**

`cd_ic`（三个池的默认方案、生产实际在用的）**权重逐位完全相同**：

```
mom_12_1 0.172597 / C_tmqmom 0.164821 / N_idiovol60 0.121714 /
G_vr20 0.061751 / H_ichimoku_alpha 0.067544 / obv_20 0.053613   <- 含 T_ 前后一字不差
```

所以 `signal_generator` 的信号**不变**。写入的只是 kept 名单（54→62）与 `ic`/`dir`/`equal`/`greedy` 方案。

### 5. 我的判断与建议

**不建议据此改生产，理由是三条硬证据**：OOS 变差（equal 1.60→1.37、ic 1.04→0.97）、最新年份 2026 掉 0.381、安慰剂同幅变好。

**但 T_ 因子在因子层面是真的强**（4 个 FDR 显著、占了 `|nw_t|` 榜首），所以"全盘否定"也不对。合理的下一步是**隔离变量**：

1. **只加 `T_dawn_evt` + `T_band_evt`**（唯一"低相关 0.07~0.33 + IS∩OOS 双显著"的两个），重跑看 OOS 是否不退化——8 个一起加会把好因子和弱因子混在一起，无法归因；
2. **查清 shuffle 为什么会随因子数变化**。如果安慰剂的强弱取决于因子集合大小/相关性结构，那这个安慰剂的**设计本身有问题**，它会污染所有"多因子 vs 少因子"的对比；
3. **用 `--exclude-static` 或成本阶梯复核**：`greedy` 的换手从 71.7 降到 29.5 是**唯一无条件的好消息**（成本稳健性），值得单独确认。

**当前状态**：`factor_config.json` 已含 T_（kept 62）。若要回到不含 T_ 的状态，用 `factor_config.backup_before_T_20260930_123258.json`。
我认为**可以先留着**（生产默认方案没变、`greedy` 的降换手是真实收益），但在警告 1/2 解释清楚之前不要据此提高 T_ 因子的权重。

## B28 只加 2 个 T_ 因子的隔离实验：**A/B 结论不可靠，但发现两个更重要的事实**

### 1. 实验本身的结果

用新写的 `quant/tools/abo_factor_subset.py`（可把因子集合收窄到"baseline + 指定因子"，走同一条 `run_report` 流水线）：

只加 `T_dawn_evt` + `T_band_evt`（其余 70 个不变），etf_3x：

| 方案 | 窗口 | 无 T_（54 kept） | **2 个 T_（56 kept）** | 8 个 T_（62 kept） |
|---|---|---|---|---|
| equal | OOS | 1.60 / 2.857 | 1.38 / 2.062 ↓ | 1.37 / 2.025 ↓ |
| ic | OOS | 1.04 / 1.099 | **1.19 / 1.464 ↑** | 0.97 / 1.015 ↓ |
| ic | recent | 1.04 / 2.049 | **1.16 / 2.656 ↑** | 0.96 / 1.828 ↓ |
| dir | OOS | 1.34 / 1.309 | 1.09 / 0.864 ↓ | 1.35 / 1.384 ≈ |
| greedy | OOS | 1.06 / 1.235 | 1.06 / 1.235（未变） | 1.20 / 1.590 ↑ |

**2 个 T_ 对 `ic` 方案是改善（OOS 1.04→1.19、recent 1.04→1.16），对 `equal` 是退化。** 但这与 8 个 T_ 的结果**高度相似**（都是 equal 退化），说明**8 个一起加并没有比 2 个更糟**——我原来"8 个混在一起所以无法归因"的假设没有证据。

### 2. ⚠️ 但我发现自己的 A/B 测试有缺陷，上面的数字**不能作为决策依据**

决定性的一点：**2 个 T_ 的 kept=56 与"n_fac 脚本 k=70"的 kept=56 产出了完全相同的回测数字**（equal OOS 2.062、ic OOS 1.464，逐位一致），而**官方生产运行（kept=62）是 2.857 / 1.099**。

也就是说：**OOS 结果由"最终 kept 集合"决定，而我构造 ablation 的方式（先按 IS |RankIC| 排序再截断）会改变 kept 集合** → 我的对照里**同时变了两个东西**（因子集 + 去重后的集合），**无法归因到 T_ 因子本身**。

**第二条缺陷**：我在 `_nfac_test.py` 的排序步骤里写错了接口（用了不存在的 `rep.evaluate_all`），使排序返回 0 个因子、每次只保留 2 个——**那一轮的 5 个 k 值实际跑的是同一个配置**。看到的"k 不影响结果"是我的 bug，不是结论。该脚本已删。

### 3. ✅ 但这个有缺陷的实验意外地确认了一件重要的事：**安慰剂是稳定的**

在（坏掉的）n_fac 运行里，5 个 k 值虽然等价，但产出的是**同一配置的 5 次独立运行**：

```
shuffle(equal) OOS sharpe: 0.86 / 0.94 / 0.91 / 1.04 / 0.84   （同一配置重复 5 次）
equal          OOS sharpe: 1.63 / 1.45 / 1.45 / 1.61 / 1.38
```

**波动仅 0.84~1.04，不是 0.67→1.34 那种幅度。** 结合 `seed=42` 是写死的（`bt.shuffle_composite(..., seed=42)`、`report.py:451`），**shuffle 是确定性的**——

**所以 B27 里"安慰剂从 0.67 涨到 0.90/1.34"不是随机噪声，而是因子集合变化的真实反应。** 这一条**排除**了我上轮提出的"安慰剂设计有问题"的怀疑（至少排除了"它随机不稳定"这一半）。剩下的问题是：**为什么随机权重会随因子集合系统性变好**——那更可能是"OOS 窗口内存在可通过各种组合捕捉的横截面漂移"，即 B27 警告 2 的另一种表现。

### 4. 关键背景：OOS 窗口只有 434 个交易日

三次运行的 OOS 都是 **434 天（≈1.7 年，2025-01 起）**。**1.7 年样本上的 Sharpe 差异（1.37 vs 1.60）本就不该作为切换生产方案的依据**——这比任何因子层面的问题都更根本。

### 5. 我的结论与建议（修正 B27）

1. **B27 里"不建议据此改生产"的结论仍然成立**，但**理由要换**：不是"T_ 因子让 OOS 变差"，而是**"我无法把 OOS 的变化归因到 T_ 因子"**（见 §2）+ **"OOS 只有 1.7 年，差异不显著"**（见 §4）。
2. **T_ 因子的因子层面证据不变**（4 个 FDR 显著、占 `|nw_t|` 榜首）——那部分是单因子 IC，与 ablation harness 无关，仍然可信。
3. **要真正归因，正确做法是**：固定**去重后的 kept 集合**（例如强制 `kept = baseline 的 54 个 + 指定的 T_`，跳过再排序），只改这一个变量重跑。`abo_factor_subset.py` 需要改成"保留 baseline 完整 kept + 追加 T_"，而不是"截断 IS 排序"。**这一步我还没做。**
4. **`ic` 方案 + 2 个 T_ 是唯一同时改善 OOS 与 recent 的组合**（1.04→1.19、1.04→1.16），值得用第 3 点的正确做法复核一次。

### 6. 当前生产状态（未变）

```
etf_3x: kept=62（含 8 个 T_）
default_scheme = cd_ic，权重 = mom_12_1 / C_tmqmom / G_vr20 / N_idiovol60 /
                              H_ichimoku_alpha / obv_20   <- 与加 T_ 之前逐位相同
```

`signal_generator` 的信号**不受影响**。

## B29 数据切分（IS/OOS/recent）的现状 + 修掉一个"recent 窗变成全历史"的静默 bug

用户问"现在的数据是怎么分割的"。我把真源、实际生效值、以及一个由此暴露的 bug 一起说清。

### 1. 真源：`factor_config.json` 的 `pools[pool].meta`，不是代码常量

`cfg.resolve_windows(pool)` 的优先级：**该池 config 的 `meta.is_window` / `meta.oos_window`** > 全局默认（`config.py` 的 `TRADE_START/IS_END/OOS_START/OOS_END/RECENT_START`）。

`recent` **不是独立配置项**，而是由 `oos_window[1] + 1 天` 推导。

### 2. 五个池的实际切分（实测交易日数）

| 池 | IS | OOS | recent | IS 天 | OOS 天 | recent 天 |
|---|---|---|---|---|---|---|
| etf_3x | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ **None(开放)** | 无 | 1005 | 434 | **0** |
| hs300 | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ None | 无 | 969 | 421 | **0** |
| a_etf_all | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ None | 无 | 969 | 421 | **0** |
| company_300 | 2011-01-01 ~ 2019-12-31 | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 至今 | 2264 | 1006 | 686 |
| company_1000 | 2011-01-01 ~ 2019-12-31 | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 至今 | — | — | — |

IS 末到 OOS 初相隔 1 天（无空洞）；company_300/1000 的 recent 从 OOS 末**次日**起算（无重叠）。
数据总跨度：etf_3x 4208 天（2010-01-04~）、company_300 4208、hs300 4065、a_etf_all 3594。

**关键结构差异**：`company_*` 是**三段式**（IS 9 年 / OOS 4 年 / recent 2.7 年）；
`etf_3x`/`hs300`/`a_etf_all` 是**两段式**（IS 4 年 / OOS 1.7 年，OOS 开放到数据末尾）。

### 3. 🔴 修掉的 bug：开放 OOS 时 `recent` 被静默当成"全历史"

**症状**（修复前实测）：

| 池 | `window_bounds('recent', pool=...)` 实际覆盖 | 天数 | 应为 |
|---|---|---|---|
| etf_3x | 2010-01-04 ~ 2026-09-25 | **4208** | 无该窗 |
| hs300 | 2010-01-04 ~ 2026-09-24 | **4065** | 无该窗 |
| a_etf_all | 2011-12-09 ~ 2026-09-24 | **3594** | 无该窗 |
| company_300 | 2024-01-02 ~ 2026-09-25 | 686 ✅ | 686 |

**成因**（`config.py:resolve_windows`）：`oos_window = ['2025-01-01', None]` 时，代码执行
`datetime.strptime(str(None), '%Y-%m-%d')` → `ValueError` → 被 `except (ValueError, TypeError)` 吞掉 → `win['recent'] = (None, None)`。
而 `(None, None)` 在 `window_bounds` / `Dataset.slice` 里的语义是 **"不限"= 全窗**。

**为什么危险**：它**不报错**，而且"recent 表现"会算成整段历史 —— 数字看起来完全正常。
更隐蔽的是**它只在第二次运行时出现**：第一次跑 `report`（此时 config 里还没有该池的窗口）会走全局默认 `RECENT_START='2024-01-01'` → recent 正确；一旦 `--write-config` 把 `oos_window=[..., None]` 写进 config，**此后每次运行都退化**。

**修法**（三处，已在 2026-09-30 完成）：
1. `config.resolve_windows`：显式判断 `oos_w[1] is None`，置 `recent=(None, None)` 并写 `recent_note='OOS 开放到数据末尾, 无独立 recent 窗'`（把隐式异常路径变成显式分支，语义可被下游读取）；
2. `evaluate.window_bounds`：新增 `EMPTY_START='9999-12-31'` 哨兵 —— **仅当该池配置确实声明了开放 OOS** 时，`recent` 返回空区间哨兵，避免被当成全窗；无 `pool`/无配置时**保持原行为**（`('2024-01-01', None)`），零向后兼容破坏；
3. `report.py`：`eval_recent` 与回测的 `windows` 列表都改为走 `window_bounds('recent', pool=ds.pool)`，不再硬用全局 `cfg.RECENT_START`（后者正是 etf_3x 报出"recent 686 天"、且那 686 天**落在 OOS 区间内部**的原因）。

**修后行为**（已验证）：

```
etf_3x       recent=('9999-12-31','9999-12-31')   回测只出 IS / OOS 两窗
company_300  recent=('2024-01-01', None)          回测出 IS / OOS / recent 三窗
无 pool      recent=('2024-01-01', None)          向后兼容不变
```

`preflight` 复跑：**PASS 46 / FAIL 0 / WARN 1**。

### 4. 一个必须连带指出的隐患（不是 bug，是方法论）

**`etf_3x` 的 IS 窗（2021-2024）本身几乎没有信号**：

```
equal 方案 IS 窗: total_ret 0.020 / sharpe 0.23 / max_dd -0.469
```

也就是说，**生产权重是拟合在一段"策略不赚钱"的样本内的**。再叠加 OOS 只有 434 天（1.7 年），
**任何"IS 选出来的权重"都先天缺乏统计支撑**。这比因子层面的任何优化都更值得先解决。

**对照**：`company_300` 的 IS 是 2011-2019（9 年、2264 天），样本量是 etf_3x 的 2.25 倍、时间跨度 2.25 倍 —— 这也是为什么它的因子证据（21 个 IS∩OOS 双显著同向）远强于 etf_3x（2 个）。

## B30 用户更新了 4 个 research pkl（2026-09-30 13:21~13:26）：核实 + 重跑

### 1. 新 pkl 完整性核实（全部通过）

| 池 | 列数 | 标的 | 行数 | 末日（旧 → 新） | 重复列 | 78 因子可算 |
|---|---|---|---|---|---|---|
| etf_3x | 206 | 28 | 106,492 | 2026-09-25 → **09-29** | 0 | ✅ 0 失败 |
| company_300 | 206 | 297 | 872,279 | 2026-09-25 → **09-29** | 0 | ✅ 0 失败 |
| hs300 | 206 | 298 | 981,557 | 2026-09-24 → **09-29** | 0 | ✅ 0 失败 |
| a_etf_all | 206 | 98 | 120,544 | 2026-09-24 → **09-29** | 0 | ✅ 0 失败 |

基础列（OHLCV/Adj Close/Volume）与 T_ 族所用列**均齐备**；所有窗口仍落在数据范围内。
`company_1000` **未更新**（pkl 仍是 2026-09-27，且其 config 里 `kept=42`、**0 个 T_**）。

### 2. 重跑结果：3 个池的生产权重未变，1 个池变化（但要看清原因）

| 池 | default_scheme | kept | **默认方案权重** |
|---|---|---|---|
| etf_3x | `cd_ic` | 62 → 62 | **0/6 变化 → 相同** |
| **company_300** | **`ic`** | 47 → **55** | 46/46「变化」← **是归一化伪象，见下** |
| hs300 | `cd_ic` | 50 → 57 | **0/8 变化 → 相同** |
| a_etf_all | `cd_ic` | 55 → 63 | **0/6 变化 → 相同** |

**`company_300` 的"权重变化"不是信号变化**：共同 46 个因子的权重**相关系数 0.9999**，
只是整体幅度缩了 **0.814 倍**（`ic_weights` 按 Σ|w|=1 归一化，因子数 47→55 后每个权重自动变小）。
即：**同一个信号，只是被摊薄到更多因子上**。

### 3. ⚠️ 这次对比混了两个变量，必须说明

查备份发现：**上一个备份里只有 `etf_3x` 含 T_ 因子，company_300 / hs300 / a_etf_all 的 kept 里一个 T_ 都没有**（我此前只对 etf_3x 跑过 `--write-config`）。

所以这次 4 池一起重跑 = **同时引入了「新 pkl」+「T_ 因子」两个变量**。
严格说不能把变化归因给数据更新。要单独看数据更新的影响，需要"新 pkl + 不含 T_"再跑一次。

### 4. ✅ 但意外得到一个有价值的结果：T_ 因子在 4 个池里 7~8/8 全部存活

| 池 | kept | T_ 存活 | 存活明细 |
|---|---|---|---|
| etf_3x | 62 | **8/8** | 全部 |
| company_300 | 55 | **7/8** | 除 `T_adxstr` |
| hs300 | 57 | **7/8** | 除 `T_adxstr` |
| a_etf_all | 63 | **8/8** | 全部 |
| company_1000 | 42 | 0/8 | （未用新 pkl 重跑，未评估） |

且在 `ic` 方案里 `T_kamadist` 在**每个池都是权重最大的 T_**（etf_3x +0.0371 / company_300 +0.0472 / hs300 +0.0163 / a_etf_all +0.0201）。

**这与 B26/B27 的单因子结论一致**，而且是在**四个独立池**上复核过的——比之前只看 etf_3x 强得多。

### 5. 新数据下的 OOS 表现（各池，`backtest_summary.csv`）

| 池 | OOS 天数 | buyhold | equal | ic | greedy | shuffle(equal) |
|---|---|---|---|---|---|---|
| etf_3x | 436 | 1.01 | **1.37** | 0.95 | 1.18 | **0.77** ⚠️ |
| company_300 | 1006 | **0.87** | 0.28 | 0.69 | 0.30 | 0.46 |
| hs300 | 423 | **0.75** | 0.58 | **−0.18** | 0.46 | −1.21 |
| a_etf_all | 423 | 0.75 | 0.39 | **0.85** | 0.13 | −0.32 |

**读法**：
- **只有 `a_etf_all` 的 `ic`（0.85 > 0.75）和 `etf_3x` 的 `equal`（1.37 > 1.01）跑赢买入持有**；
- **`hs300` 的 `ic` 是 −0.18**，是该池最差方案；
- **`company_300`（生产默认用 `ic`）OOS 0.69 < buyhold 0.87** —— 该池的默认方案仍跑输基准；
- **`etf_3x` 的安慰剂仍是正的（0.77）**，这是它连续第三轮出现，仍然是该池最该查的问题。

`company_300` 的 recent 窗（686 天）反而是它最好的：`equal` 1.74 / `ic` 1.18 / buyhold 1.68 —— 但 recent 不参与任何定参，只做事后汇报。

### 6. 顺带修掉的第二处同类 bug：`wf_strat_search.pool_windows`

`pool_windows()` 也把 `(None, None)` 直接透传给回测，导致 `wf_strat_search.py` 的 "recent" 段
对一个开放 OOS 的池会算**全历史**。已改为：**只在配置给出真实 recent 起点时才加入该窗**
（`rs not in (None, '9999-12-31')`），与 `report.py` 的处理口径一致。

### 7. 待办

1. **`company_1000` 需要同步**：新数据没给它，且它的 config 还是 2026-09-28 的（无 T_ 因子）。要么补一份新 pkl 重跑，要么从 config 里撤掉这个池以免口径不一致；
2. **数据更新的净影响未单独测量**（见 §3）——若要严格归因，需"新 pkl + 不含 T_"跑一次；
3. `factor_config.json` 已写入新权重，备份在 `factor_config.backup_before_pkl_update_20260930_132903.json`。

## B31 用生产冻结权重回测 vs 流水线现算权重（**生产权重全面更强，但有重要前提**）

### 0. 为什么要单独跑这一遍

`factor_config.json` 里三个池的默认方案是 `cd_ic`，而 **`cd_ic` 不在 `run_report` 重算的方案集合里**
（日志里那句"保留既有未重算方案 ['cd_ic']"就是这个意思）。所以此前所有 `backtest_summary.csv`
**都没有"真实生产权重"这一行** —— 我之前引用的 `ic`/`equal` 数字都不是生产实际在用的。

新增 `quant/tools/bt_prod_weights.py`：直接读 config 的 `schemes`，用 `combine_signals` 合成后
喂给同一个 `bt.run_engine`（同一份数据、同一引擎参数、同一窗口），复刻真实生产回测。

### 1. 结果：生产冻结权重在 4 个池的 OOS 上都明显更强

| 池 | 生产方案(因子数) | OOS Sharpe | OOS total_ret | OOS 换手 |
|---|---|---|---|---|
| etf_3x | **`cd_ic`(6)** | **1.69** | **3.705** | 19.5 |
| company_300 | `ic`(55) | 0.69 | 0.973 | 47.2 |
| hs300 | **`cd_ic`(8)** | **1.74** | **2.727** | 46.9 |
| a_etf_all | **`cd_ic`(6)** | **1.55** | **0.565** | **7.4** |

对照（同一份数据/引擎/窗口，流水线现算）：

| 池 | 现算 `equal` | 现算 `ic` | buyhold | 安慰剂 |
|---|---|---|---|---|
| etf_3x | 1.37 | 0.95 | 1.01 | 0.77 |
| company_300 | 0.28 | 0.69 | 0.87 | 0.46 |
| hs300 | 0.58 | **−0.18** | 0.75 | −1.21 |
| a_etf_all | 0.39 | 0.85 | 0.75 | −0.32 |

**读法**：
- **`hs300` 反差最大**：生产 `cd_ic` 1.74 vs 现算 `ic` −0.18 —— 差 1.92。现算 `ic` 的 103.6x 年换手是主因；
- **`a_etf_all` 的生产 `cd_ic` 换手只有 7.4x**（全池最低），却有 1.55 的 OOS Sharpe —— 成本稳健性最好；
- **`etf_3x` 的生产 `cd_ic` 在 IS 上也最好**（0.93，而现算 `equal` 是 0.23）；
- **唯一的例外是 `company_300`**：它的生产方案是 `ic`（不是 `cd_ic`），OOS 0.69 **仍跑输 buyhold 0.87**。

### 2. ✅ 一个有意义的一致性交叉验证

`company_300` 的生产权重来自 config 的 `ic` 方案，而它在 `backtest_summary.csv` 里也有一行
现算 `ic`。两者**逐位一致**（OOS total_ret 0.973 / sharpe 0.69 / max_dd −0.343 / 换手 47.2，
recent 1.677 / 1.18）：

- 说明 `ic_weights` 在 `run_report` 里算出的权重与 `export_config` 存进 config 的**是同一套**；
- 也说明我这个"读 config 权重 + 自己合成"的脚本与流水线**口径一致**（否则不可能逐位吻合）。

### 3. ⚠️ 但这不构成"生产权重更好"的独立证据 —— 必须说清

`cd_ic` 的定义是：**以 `ic` 为初值做连续坐标下降，目标函数 = 折外 Sharpe**（`wf_step2_search.py:16`、
`wf_step3_sens.py` 冻结为 6 折等分 + agg=mean）。也就是说：

**这套权重是"以折外表现为目标"搜索出来的。** 它在 OOS 上强，**部分是选择过程本身的结果**，
而不是纯粹的样本外泛化能力。我在 B27 里对 T_ 因子提出的同样标准（"安慰剂 / 选择偏差"），
在这里必须同样适用。

**能减轻这个担忧的三点**：
1. 目标用的是 **6 折等分 + agg=mean**（不是单看某一段），跨度包含 IS 起点到数据末，比"只调 OOS"稳健；
2. `cd_ic` 的因子数很少（6/8/6），**自由度低**，过拟合空间比 55 因子小得多；
3. 它在**IS 段也不差**（etf_3x 0.93 / hs300 1.22 / a_etf_all 0.94），不是"只在 OOS 亮"。

**但它仍然不是干净的样本外**：目标函数里包含 OOS 段。要真正验证，需要**留出一段从未参与任何选择的窗口**
（例如把数据切成 pretrain / select / holdout 三段，前者只用于 select）。

### 4. 由此修正我之前几轮的说法

- B27/B28 里我说"T_ 因子让 equal 的 OOS 从 1.60 变 1.37、所以有风险"——**那个对比用的是现算 `equal`，
  不是生产权重**。生产 `cd_ic` 在同一数据上是 1.69，**比 1.60 和 1.37 都好**。
- 也就是说：**前几轮我一直在用"现算权重"的表现推断"生产表现"，这个推断方向是错的。**
  生产在用的是少数几个因子 + 坐标下降权重，与 55~63 因子的 `equal`/`ic` 是两回事。
- **结论**：关于 etf_3x/hs300/a_etf_all，"生产权重是否退化"这个问题，**答案是"没有退化，反而是最强的"**。

### 5. 仍未解决 / 下一步

1. **`company_300` 是唯一跑输 buyhold 的生产池**（0.69 vs 0.87，换手 47.2x 是主因）。它用的是 `ic` 而非 `cd_ic` —— 值得试一次把它的默认方案换成 `cd_ic`（该池 `schemes` 里**没有** `cd_ic`，需要先跑 `wf_step4`）；
2. **`cd_ic` 需要重建以跟上新数据**：目前三池的 `cd_ic` 是历史上导出的，**没有用 2026-09-29 的新数据重算过**。新数据只多加 2 个交易日，影响应当极小，但严格说配置与数据不同源；
3. **真正的样本外验证**：需要专门的 holdout 窗口（见 §3）。

---

## 9. 建议的改进顺序（整合四份深度审查后的最终版）

| # | 事项 | 预计 | 为什么排这里 |
|---|---|---|---|
| 1 | **`signal_generator.py` 去掉静默回退**（P0-0）：`resolve_pool` 加 `conf.meta.pool == pool` 校验；缺配置即 `raise SystemExit`；`_load_config_doc` 区分"文件不存在"与"解析失败" | 1–2 h | 现在 `--pool hs300` 在配置缺失时会用 `etf_3x` 的因子和权重静默出信号；这是"错误信号"而非"崩溃" |
| 2 | **实盘止损/止盈量纲 + `rate_inday`**（P0-3/A3-P0-C）：统一到小数、禁止 round 后比阈值、给 `cash_out` 加"至少一条保护生效"断言 | 2–4 h | **当前三个保护条件全部失效** —— 真钱裸奔 |
| 3 | **富途市价单资金校验**（A3-P0-A）：用真实参考价校验，加单笔金额/数量上限 | 2–4 h | 唯一"可无限量下单"的口子，且在默认路径上 |
| 4 | **持仓查询失败不得伪装空仓**（A3-P0-B）：失败上报 `None`/异常，台账同步在数据不可信时 return | 半天 | 修完 #2 才可能真正执行止损；同时阻止台账被抹 |
| 5 | **建仓库 + `.gitignore` 先写对**（P0-1/2）：`git init`，先 ignore 掉 `research_2026-09/output/`、`portfolio.json`、`position.json`、`api_key/`、`*.pem`、`*.pyc`、`*.xmind`、`*.zip`，再 `git add` | 2–4 h | 现在**没有 `.git`**，所以这一步是"从零建对"，成本最低；顺序反了就要 `filter-repo` |
| 6 | **复权索引 bug**（P0-6）：改一个词 + 补拆股单测 + **重跑 pkl 并评估影响面** | 半天 + 重跑 | 影响所有价格列；虽然 `Split` 在现有数据全为 1 未触发，但这是"下次一定触发"的雷 |
| 7 | **统一因子窗口口径 + 重置 holdout**（A4-P0-1）：窗口单一真源、选择函数禁 OOS、2024+ 冻结 | 1–2 天 | **"样本外"对线上配置其实是样本内** —— 这条不修，后面所有结论都要打折 |
| 8 | **`requirements` 声明 lightgbm/torch + 拆 base/ml/dev**（A2-P1-1） | 2–4 h | 干净环境装不出可用环境；换机器复现的第一道坎 |
| 9 | **引擎对拍**（正文 §6.2）：`bt_core` vs `factor.backtest` 同输入逐点比对 | 1 天 | 消除"生产表现"与"研究结论"对不上账的根本不确定 |
| 10 | **`ml` 默认特征源改 `factor` + 序列模型 TA 守卫下沉**（A2-P0-1/2） | 半天 | 214 维里 155 维因果性不可验证；守卫靠 CLI 纪律已经失效过一次 |
| 11 | **`.gitignore` 覆盖 + 清 95 个 `.pyc`/杂项**（P2-4） | 1–2 h | 与 #5 同批做 |
| 12 | **`write_config` 原子写 + pkl 溯源 + 回退告警**（A4-P1） | 半天 | 现在一次 JSON 解析失败就把 5 个池配置覆盖成 1 个 |
| 13 | **补测试网**：`verify_backtest`/`combine` 纯函数接 pytest、`bc_trader` 的 mock 下单测试（"资金不足必须拒绝""重复调用不得重复下单"） | 2–3 天 | 7718 行指标 + 1218 行下单层**零测试** |
| 14 | **`_sanitize_prices` 掩码修正 + 下沉公共层**（A4-P1、P0-6） | 半天 | 现在特征/标签侧不清洗、回测侧清洗，两个口径 |
| 15 | **成交量约束 + 成本阶梯进报告标配**（P1-4） | 2 天 | 让"高换手策略可行吗"有真实答案 |
| 16 | **文档整合 + 死代码清理**（A1）：根 README、修 219 处死路径、pkl 数据字典、`archive/` 会话转录 | 1–2 天 | 不紧急，但决定"半年后你还能不能接手" |

**建议节奏**：**1→2→3→4（约 1 天，纯护钱）**，然后 **5→11（半天，把地基铺对）**，然后 **6→7→8→9→10（约 3 天，修正确性与口径）**，最后 12–16 按研究节奏铺开。

## 10. 一句话总结

**这个项目的研究纪律（防前视不变量、NW-HAC + BH-FDR、静态选票探针、诚实记录"否决改进"、以及"对检验本身做检验"的反例注入）已经超过绝大多数个人量化项目，甚至超过不少机构的实际水平。它的风险不在"想错了"，而在"做出来的东西没被锁住"。**

四份深度审查合起来，最刺眼的是这一组事实：

**关于"钱"（前 5 条，建议一周内集中处理）**：
- **现役信号生成器 `signal_generator.py` 的架构是对的**（只读 `quant.factor`、**不在生产数据上重定权**、决策截面严格 `< trade_date`、数据陈旧时 fail-closed）—— 但它有一条**静默回退**：配置缺失时会用 `etf_3x` 的因子和权重去跑任意池，只打一条 info 日志；
- 实盘的止损、止盈、日内止损**三个保护条件全部无效**（量纲错误 + `rate_inday` 硬编码 0）；
- 富途市价单的**资金校验被一个占位价 0.1 绕过**，任何数量都判为"买得起"；
- 下单失败**仍扣可用资金**，在 3000 本金 + 5 只标的的实盘上，第一笔失败会让当天剩余买入全部取消；而每只 600 的预算可能连 1 股都买不到；
- 持仓查询失败被**伪装成"空仓"**，于是止损单静默跳过、本地持仓台账被抹平并还原初始现金；
- 工作副本**根本没有 `.git`**，而产出与实际下单的代码（含 `signal_generator.py`、`automatic_trader.py`）从不在版本控制内。

**关于"结论"（后 3 条，建议在挖下一个因子之前解决）**：
- pkl 的复权处理有**一处索引写错**（`bc_technical_analysis.py:217`），影响所有价格列，而现有脏价格守卫的阈值（|收益|>100%）抓不到它；
- 数据 IO 层有**四条 P0**：qfq 增量拼接的复权基准漂移、下载失败被吞后陈旧数据伪装成"success"、CSV 非原子写与并发写、分红/拆股被 `0.0/1.0` 覆盖真值 —— 而**测试完全没有覆盖 eod 路径**（生产三个市场全用 eod）；
- 因子包内部**有两套互斥的 IS/OOS 口径**，最终线上权重是用 OOS 数字从数百组配置里选出来的 —— 报告里的"样本外"对线上配置其实是样本内；
- 21 个 `H_*` 因子的因果性**结构性不可验证**，却被报告为"全部因子因果"，而同一个仓库的 ml 包对同一批列写的是"无法验证"；
- ml 默认特征里 **155/214（72%）是那批因果性不可验证的 TA 列**，而序列模型"排除 TA"的守卫已经丢失（相对 09-28 是回归）。

还有一条**架构层面的隐患值得单独提**：`signal_generator.py:24` 声称"只要生产 pkl 的起点距最新日 > 最大窗口(252)，最新一日的因子值就与研究版**逐值相等**"。这个前提在"两者都是 qfq 前复权、但下载时点不同"的情况下**不自动成立** —— 详见 P1-8 与附录 A6 的 P0-D1。而这条假设是"生产 pkl 短、研究权重直接可用"整套设计的地基。建议做一次实证：取生产 pkl 与研究 pkl 的最后 252 个交易日，对同一批因子逐值比对，确认差异量级。

如果"样本外"不是样本外、如果一半特征的因果性没有证据、如果上游价格本身可能带有人为台阶，那么**继续挖新因子的边际价值是零** —— 你会把噪声当成 alpha，并用真钱去验证它。

**你在"没有找到 alpha 就不改生产"这件事上做对了 —— 那是这个项目最专业的一次判断。现在把同样的纪律用到工程上。**
