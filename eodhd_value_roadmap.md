# EODHD 数据源价值最大化规划报告

> 生成日期：2026-10-12 ｜ 面向：次日晨读
> 口径：账户实测（MCP `get_user_details`）+ 100 个 MCP 工具描述 + 官方文档（api-limits / commercial-license / data-sources）+ 本地量化系统现状
> 配套图：[eodhd_roadmap.png](file:///c:/Users/northcheng/git/quant/factor/output/eodhd_roadmap.png) ｜ 已有实测清单：[eodhd_new_data_sources.md](file:///c:/Users/northcheng/git/quant/eodhd_new_data_sources.md)

---

## 摘要（Executive Summary）

**一句话**：EODHD 的"存量数据"（日行情 / 基本面 / 情绪）已被现有系统基本榨干；真正未兑现的价值集中在两类——**结构（universe / point-in-time）** 与 **事件（smart money / 事件窗口）**，而非再堆更多价量或比率因子。

**账号定位**（实测）
- 套餐：**ALL-IN-ONE**（`subscriptionType=yearly`、`subscriptionMode=paid`、Stripe）
- 额度：**100,000 calls/日**（GMT 00:00 重置）、**1,000 requests/min**、额外 reserve = 0
- 今日用量：**20,474 / 100,000（20.5%）**，剩 79,526 → 额度非常充裕，真正的约束是 **req/min**
- 主订阅含 **45 个 data feed**；`availableMarketplaceDataFeeds=[]` → **所有 `/mp/` 路径（期权 / ESG / PRAAMS / TradingHours / S&P·DJ 指数）REST 与 MCP 均 403，不可用**

**核心判断**
1. 现有因子库已达 **106 个**（价量·技术 78 + 基本面 26 + 情绪 2），边际信息增益已很低；继续在同一池同一数据上加比率，属于低 ROI。
2. 已落地的最大单项收益来自 **修 PIT 宇宙口径**：均值 |IC| 0.0035 → 0.0075（**≈2.1×**），nw_t 显著因子 27 → 55。这是"同一批因子、更干净的宇宙"带来的免费收益。
3. 尚未兑现且 ROI 最高的三件事：**① PIT 宇宙补全到基本面；② 内部人 Form 4 净买入 + 国会跟单；③ VIX / 利率 / 利差 regime 门控**。

**三条立即可做（低成本高收益，见文末路线图 P0）**
- 补退市股基本面 → 让 `F_*` 族也在 PIT 宇宙内评估（当前 delta≈0，因退市股无基本面）
- 内部人 Form 4 净买入因子（MCP 独立授权通道已实测可用）
- 股东收益率（分红 + 回购）/ 市值 —— 补 value 族缺失的回购维度，成本仅 2 个端点

![剩余价值象限](file:///c:/Users/northcheng/git/quant/factor/output/eodhd_roadmap.png)

---

## 一、账号与额度经济学

### 1.1 权限矩阵（实测，2026-10-12）

| 状态 | 内容 |
|---|---|
| ✅ 已用 | EOD 日行情、拆分、分红、公司基本面全量、历史市值、情绪、SEC filings 计数 |
| 🟡 可用未接 | 内部人 Form 4（MCP 通道）、国会交易、财报日历/趋势、新闻词权重、VIX、UST 收益率、利率/信用利差、指数成分、退市股清单 |
| 🔴 无权限 | 全部 `/mp/*`：美股期权链/EOD、S&P Global 指数、Investverte ESG、PRAAMS 风险/财报、TradingHours |

> 注：`get_insider_transactions_form4`、`get_us_tick_data` 走 **MCP 独立授权通道**可用（REST 直连 404）。批量任务仍建议优先 REST + `api_key`。

### 1.2 计价规则（"call" 是账户货币）

| 类别 | 单次消耗 |
|---|---|
| 单标的端点（EOD / 市值 / 分红 …） | **1 call** |
| 技术指标 / 日内 / 新闻 / 情绪 | **5 calls**（新闻类 `5 + 5×N`，`s=` 可批量 15–20 只摊薄） |
| 基本面 / 期权 / 债券 / 所有 marketplace | **10 calls** |
| Bulk 整所 | **100 + N calls** |

### 1.3 额度核算

- 全池基本面一次：us_500（501 只）× 10 = **5,010 calls**（一次性回填）
- 全部新增源每日刷新合计仅 **万级 calls** → 相对 100k/日 **绰绰有余**
- **唯一硬约束 = 1,000 req/min**：并行下载时必须限速（退市股 2,000 只回填即属此类，需分批）

### 1.4 许可红线

- 套餐默认 **个人使用**；**商业用途**需申请 professional user 状态（约 3 个工作日，且需向交易所报备）
- 数据来源：美股 EOD 来自 **Nasdaq Cloud API**；欧洲 **Cboe Europe**；澳洲 **ASX**；NEO/TSX 来自 Quotemedia（延迟 15 分钟）
- 基本面来自 sec.gov / sedar.com，经 NLP/LLM 解析（大中盘有人工复核）

---

## 二、EODHD 能力地图（100 个 MCP 工具 × 三态）

> 状态：✅ 已用 ｜ 🟡 可用未接 ｜ 🔴 无权限

| 数据域 | 代表端点/工具 | 状态 | 备注 |
|---|---|---|---|
| 日行情 EOD | `get_historical_stock_prices` | ✅ | 系统主数据 |
| 日内/实时 | `get_intraday_historical_data`、`get_live_price_data`、`get_us_live_extended_quotes`、WS | 🟡 | 执行层，非 alpha 研究 |
| 逐笔 Tick | `get_us_tick_data`（MCP） | 🟡 | 贵（10 calls/req），微观结构专用 |
| 基本面 | `get_fundamentals_data`、`get_bulk_fundamentals` | ✅ | 三表/比率/Highlights/SharesStats/Earnings/Dividends |
| SEC 结构化 | `get_sec_companies / financials / ratios / growth / ttm / segments / snapshot / filings` | 🟡 | 8 工具，与 fundamentals 双视图 |
| 历史市值 | `get_historical_market_cap` | ✅ | size 因子 |
| 分红/拆股/事件 | `get_historical_dividends / splits`、`get_upcoming_*` | 🟡 | 分红历史已可接 |
| 内部人 | `get_insider_transactions`、`get_insider_transactions_form4` | 🟡 | Form4 更全（含衍生品/脚注） |
| 国会交易 | `get_congressional_trades` | 🟡 | 46,133 条，1990+，**稀有** |
| 财报 | `get_upcoming_earnings`、`get_earnings_trends` | 🟡 | PEAD / 预期修正 |
| 新闻/情绪 | `get_company_news`、`get_sentiment_data`、`get_news_word_weights` | ✅/🟡 | 情绪已用；词权重未用 |
| 宏观 | `get_macro_indicator`、`get_economic_events` | 🟡 | 年频偏粗；事件日历有用 |
| 利率 | `get_ust_yield_rates / bill / long_term / real_yield`、`get_rates_*`（3） | 🟡 | 1990+，regime 门控 |
| 信用 | `get_credit_*` 共 7 工具 | 🟡 | 主权 CDS 年频（Damodaran），企业 HQM/CMDI |
| 指数 | `get_cboe_indices_list / index_data`、`mp_indices_list / components` | 🟡/🔴 | CBOE 可用（VIX）；`mp_*` 403 |
| 成分股 | `/fundamentals/{INDEX}.INDX`（经 `get_fundamentals_data`） | 🟡 | **仅当前快照**，无历史增删 |
| 商品/房地产/制裁 | `get_historical_commodity_prices`、`get_real_estate_*`、`get_sanctions_*` | 🟡 | 低优先 |
| 交易所/搜索 | `get_exchanges_list / details / tickers`、`resolve_ticker`、`get_stocks_from_search`、`get_symbol_change_history` | 🟡 | 退市股清单入口 |
| Marketplace | 期权 / ESG / PRAAMS / TradingHours / S&P | 🔴 | **全部 403** |
| 工具类 | `stock_screener`、`get_support_resistance_levels`、`get_technical_indicators`、`retrieve_description_by_id` | 🟡 | technical 与自研 TA 重复，**不建议** |

**结构性观察**
- 存在大量功能重叠（insider 旧版 vs Form4；`get_fundamentals_data` vs `get_sec_*`；search vs `resolve_ticker`；`exchange_details` vs `tradinghours`）→ 选一条主通道即可，避免重复付费。
- `/mp/` 全不可用，规划时**不要**把期权 / ESG / PRAAMS 纳入路线。

---

## 三、现有量化系统现状（数据层 + 因子层）

### 3.1 数据层已落地（`~/quant/`）

| 资产 | 路径 | 现状 |
|---|---|---|
| 日行情 | `stock_data/` | us_500，501 只 EOD |
| 基本面 | `fundamental_data/fundamentals/` | 501 只 JSON（PIT 事件长表 `pit_events.pkl`） |
| 历史市值 | `market_cap/` | 已接 |
| PIT 退市股 | `newdata/pit_stock_data/` | 抽样 2,000 → 成功 1,998 只 |
| 宏观 | `macro_us.pkl` / `macro_long.pkl` | VIX、UST |
| 情绪 | `sentiment_data/us_500_sentiment.pkl` | 长表，**仅 2022+**（2022 前 NaN） |

### 3.2 因子库：106 个（`factor.py` / `fundamentals.py` / `sentiment.py`）

- **价量·技术 78**：动量 / 反转 / 波动 / 量能 / 趋势 / regime（`T_*`）/ 残差动量（`R_*`）/ 微观结构（`G_*`）/ 门控（`K_*`）/ 流动性（`illiq_60`、`cs_spread_20`）/ 技术派生（`H_*_alpha`）
- **基本面 26**（前缀 `F_`）：size / value / quality / growth / surprise / issuance
- **情绪 2**（前缀 `S_`）：`S_sent_lvl`、`S_news_attn` —— 定位为**控制变量**（前序评估：无稳健独立 alpha，`S_news_attn` 近静态大盘暴露）

### 3.3 PIT 宇宙实验结论（已落地，方向 1）

| 维度 | biased（今日 us_500） | PIT（含退市股） | 变化 |
|---|---|---|---|
| 均值 \|IC\| | 0.0035 | 0.0075 | **≈2.1×** |
| nw_t 显著 | 27 / 79 | 55 / 79 | +28 |

- 代表：`mom_12_1` 0.0147→0.0201；`pos_252` 0.0071→0.0183；`hi_252` 0.0006→0.0186
- **符号被修正**：`vol_60` 0.0003→−0.0221、`illiq_60` 0.0037→−0.0279(t=−15.6) → 含退市股后**低波/流动性异象才显现**
- **未覆盖**：`F_*` 基本面因子 IC 几乎不变（退市股无基本面缓存）→ 见路线图 P0

---

## 四、剩余价值空间（核心章节，按 ROI 排序）

### P0-1 ｜ PIT 宇宙补全到基本面（**最高优先**）
- **目标**：让 `F_*` 族也在含退市股的 PIT 宇宙内评估，消除幸存者偏差对基本面的系统性低估
- **端点**：`/fundamentals/{SYM}.US`（10 calls）对 1,998 只退市股回填 → ≈ 19,980 calls（一次性，额度内）
- **前置**：退市股 EOD 已有（`pit_stock_data/`）；须按 `filing_date` 做 avail_date，缺失回退"报告期 + 90/45 天"再 ffill 到交易日
- **预期**：价量族已验证 2.1× 增益；基本面族大概率同向
- **风险**：退市股年报缺漏多、字段质量差 → 需 min_cs 与覆盖率护栏

### P0-2 ｜ 内部人 Form 4 净买入（smart money）
- **端点**：`get_insider_transactions_form4`（MCP 通道已验证可用，AAPL 返回 617 条 filings）
- **因子设想**：滚动窗口内 **净买入金额 / 市值**、高管 vs 董事、买入簇（cluster buying）
- **PIT 要点**：以 **filed date**（非 transaction date）为可用日
- **预期**：低相关、事件型 alpha 来源

### P0-3 ｜ 股东收益率（Shareholder Yield）补回购维度
- **端点**：`get_historical_dividends`（1 call）+ 回购（`/fundamentals` 现金流 or shares 变化）+ `get_historical_market_cap`（1 call）
- **因子设想**：`(分红 + 净回购) / 市值`，补 value 族缺失的"回购"腿
- **成本**：极低（分红 + 市值两个端点）

### P1-1 ｜ VIX / 利率 / 利差 regime 门控
- **端点**：`get_historical_stock_prices` `VIX.INDX`（**1990-01-02 起**）、`get_ust_yield_rates`（1990+，2,250 条）、`get_ust_real_yield_rates`、`get_rates_funding_stress`、`get_credit_sovereign_cds_spreads`
- **因子设想**：外部 regime 门控（高波动/倒挂/信用走阔 → 降暴露或切风格），非选股因子
- **成本**：每源 1 call/日

### P1-2 ｜ PEAD 事件窗口
- **端点**：`get_upcoming_earnings`（日历）+ `get_earnings_trends`（预期修正）+ 已有 `F_sue_*` / `F_eps_surp`
- **因子设想**：盈余公告后 N 日窗口内的漂移强度、预期修正方向
- **PIT 要点**：以公告日（含盘前/盘后）对齐

### P1-3 ｜ 国会交易跟单
- **端点**：`get_congressional_trades`（46,133 条，1990+）
- **⚠ 陷阱**：`transaction_date` 年份被污染 → **必须用 `disclosure_date`** 作可用日（披露滞后本身也是信号）
- **因子设想**：议员净买入强度、委员会相关性（细分）

### P2-1 ｜ 新闻主题（word weights）
- **端点**：`get_news_word_weights`（已验证：AAPL 2024-09，news_processed=300）
- **因子设想**：主题集中度 / 情绪分歧，作为情绪层的补充

### P2-2 ｜ 机构持仓 13F 拥挤度
- **端点**：`/fundamentals/{T}.US?filter=Holders`（10 calls；现 JSON 仅 2 条太薄，需走季度 13F）
- **成本高、更新慢** → 暂缓

### 暂缓 / 不做
- 实时/盘中（执行层，非 alpha 研究）
- `/technical/{T}`（与自研 TA 重复）
- 全部 `/mp/*`（无权限）
- 宏观年频、主权 CDS 年频（粒度太粗，仅作长期背景）

---

## 五、分阶段落地路线图

| 阶段 | 内容 | 关键端点 | 额度 | 里程碑 |
|---|---|---|---|---|
| **P0（立刻）** | 退市股基本面回填 + 内部人 Form4 + 股东收益率 | fundamentals / form4 / div+marketcap | ≈2万 calls 一次性 | `F_*` 在 PIT 内评估；新增 3 族因子 |
| **P1（下一轮）** | VIX/利率 regime 门控 + PEAD + 国会跟单 | VIX / ust / earngs / congress | <1k calls/日 | 择时门控 + 2 事件族 |
| **P2（储备）** | 新闻主题 / 机构持仓 13F | word-weights / Holders 13F | 视情 | 情绪层增强、拥挤度 |

**执行顺序原则**：先修**结构**（universe），再补**事件**，最后做**择时与舆情**。每次只引入一族新数据，用现有评估管线（`evaluate.py` + `combine.py`）做增量正交检验，避免"数据一多就过拟合"。

---

## 六、反模式清单（不要做）

1. ❌ 在**今日成分股**池上继续加因子 → 幸存者偏差会**系统性低估**真实 alpha（已验证 2.1×）
2. ❌ 用 `/technical/*` 重复自研 TA → 5 calls 换等价信息，纯浪费
3. ❌ 依赖任何 `/mp/*` 端点 → 全部 403
4. ❌ 把新生儿数据当长历史用 → **情绪仅 2022+**、宏观/主权 CDS **年频**
5. ❌ 用 `transaction_date` 对齐国会交易 → 年份被污染，必须 `disclosure_date`
6. ❌ 把 `get_fundamentals_data` 与 `get_sec_*` 都拉一遍 → 双视图重复付费，选一条
7. ❌ 忽略 `shuffle_composite` 是**按日整行置换** → 对截面 IC 不是有效零假设（引擎口径才是红线）

---

## 七、数据质量红线与陷阱（实测）

| 数据源 | 实测范围 / 陷阱 |
|---|---|
| VIX | 1990-01-02 起（开盘 17.24），可接 |
| UST 收益率 | 1990-01-02 起，2,250 条 |
| 宏观指标 USA | **1960–2025，年频** → 粒度粗 |
| 主权 CDS | 年频，Damodaran 源 → 粒度粗 |
| 国会交易 | 46,133 条 1990+；**`transaction_date` 年份污染**，用 `disclosure_date` |
| SEC filings（AAPL） | 10-K 11 / 10-Q 35 / 8-K 108 / Form4 617 |
| 分红（AAPL） | 1990 起全量（declaration/record/payment 齐全） |
| 指数成分（GSPC.INDX） | 含 Weight/Sector/Industry，但**仅当前快照**，无历史增删 |
| 新闻词权重（AAPL） | 2024-09 正常，news_processed=300 |
| CBOE 指数列表 | 123 个，**多为欧洲区域指数**，非 VIX |
| 退市股清单 | `/exchange-symbol-list/US?delisted=1` → 60,420（Common Stock 33,175） |

---

## 八、官方文档导航（关键 3 页）

| 文档 | 价值 |
|---|---|
| `api-limits` | call 计价、100k/日、1,000 req/min、额外 calls 购买 |
| `commercial-license` | 个人 vs 商业用途、professional user 申请 |
| `data-sources` | 各 feed 的上游来源（Nasdaq / Cboe Europe / ASX / Quotemedia） |

---

## 附录：本地脚本与产物索引

| 用途 | 路径 |
|---|---|
| EODHD 可接数据源实测清单 | [eodhd_new_data_sources.md](file:///c:/Users/northcheng/git/quant/eodhd_new_data_sources.md) |
| 一次性回填脚本（symbols/div/macro/news/earnings/pit） | [_tmp_fetch_newdata.py](file:///c:/Users/northcheng/git/quant/_tmp_fetch_newdata.py) |
| PIT 宇宙实验 | [_tmp_pit_lab.py](file:///c:/Users/northcheng/git/quant/_tmp_pit_lab.py) → [pit_lab_us.csv](file:///c:/Users/northcheng/git/quant/factor/output/fundamentals/pit_lab_us.csv) |
| 新数据源调研 | [_tmp_survey_newdata.py](file:///c:/Users/northcheng/git/quant/_tmp_survey_newdata.py) → [newdata_survey.csv](file:///c:/Users/northcheng/git/quant/factor/output/fundamentals/newdata_survey.csv) |
| 本报告配图脚本 | [_tmp_eodhd_roadmap.py](file:///c:/Users/northcheng/git/quant/_tmp_eodhd_roadmap.py) |

---

*本报告基于 2026-10-12 账户实测与仓库现状生成；额度与权限如有变动，以 `get_user_details` 与实测为准。*