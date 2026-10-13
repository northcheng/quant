# EODHD 可接新数据源清单（实测版）

> 账户实测（`/api/internal-user`）：`subscriptionMode=paid`、`subscriptionType=yearly`、`dailyRateLimit=100,000`；主订阅含 **45 个 data feed**；`availableMarketplaceDataFeeds=[]`（**无 Marketplace 权益**）。
> 口径：ALL-IN-ONE = 100,000 calls/day、1,000 requests/min、30+ 年历史、个人使用。
> "call"（API call，接口调用额度）≠ "request"（HTTP 请求）：一次 request 可消耗多个 call。
> 标的格式 `{TICKER}.US`；class share 用连字符 `BRK-B.US`。
> 下表 **状态** 列：✓=本次实测 HTTP 200；✗=实测 403/404（无权限或路径不符）；未测=未验证。

## 一、基线（现状，无需新增）

| 数据 | 端点 | 成本 | 状态 |
|---|---|---|---|
| EOD（End of Day，日终行情 OHLCV） | `/eod/{T}.US` | 1 | ✓ 已用 |
| 拆分 / 分红 | `/splits/{T}` `/div/{T}` | 1 | 已用 |

## 二、可新增数据源（按对美股 alpha 价值排序）

### 优先级 1 — 基本面 / 持仓（value·quality·size 因子）
| 数据 | 端点 | 成本 | 状态 | 频率/历史 | alpha 用途 |
|---|---|---|---|---|---|
| 公司基本面全量 | `/fundamentals/{T}.US` | **10** | ✓ | 30+ 年，季/年报 | 财务三表（Income/BS/CF）、估值比率、Highlights、SharesStats、Earnings、Dividends |
| 机构持仓 | `/fundamentals/{T}.US?filter=Holders` | **10** | ✓ | 季度 13F | **机构持股变化、拥挤度因子** |
| 历史市值序列 | `/historical-market-cap/{T}.US` | 1 | ✓ | 周频多年 | **size 因子（log 市值）、市值动量** |
| 批量基本面 | `/bulk-fundamentals/US`（或带 `symbols=`） | 100(+N) | 未测 | 同上 | 整所/子集一次拉，摊薄成本 |

> 全史测算：us_500(501只) 基本面 = 501×10 = **5,010 calls**（一次），远低于 100k/日。

### 优先级 2 — 内部人 / 国会 / 事件（smart money、PEAD）
| 数据 | 端点 | 成本 | 状态 | 频率/历史 | 用途 |
|---|---|---|---|---|---|
| 内部人交易 | `/insider-transactions?code=US` | 1 | ✓ | 逐笔多年 | 高管增减持 → **内部人净买入因子** |
| **国会交易** | `/congressional-trades` | 1 | ✓ | 逐笔，含议员/方向/金额/日期 | **议员交易跟单因子**（数据源稀有，价值高） |
| SEC 文件目录 | `/sec-filings/{T}.US` | 1 | ✓ | 10-K/10-Q/8-K 计数与最新日 | 财报发布节奏、事件窗口 |
| 财报日历 | `/calendar/earnings?from=&to=` | 1 | ✓ | 按区间 | EPS 预期/实际/超预期 → **PEAD（Post-Earnings Announcement Drift，盈余公告后漂移）** |
| 财报趋势 | `/calendar/trends?symbols=` | 1 | ✓ | 季度 | 分析师预期修正方向 |
| 分红/拆股/IPO 日历 | `/calendar/dividends` `splits` `ipos` | 1 | ✓ | 按区间 | 事件驱动、除权效应 |

### 优先级 3 — 新闻与情绪（attention / sentiment）
| 数据 | 端点 | 成本 | 状态 | 用途 |
|---|---|---|---|---|
| 公司新闻 | `/news?s={T}.US` | **5+5×N** | ✓ | 新闻时间线 + 情绪标签 |
| 情绪聚合 | `/sentiments?s=` | **5+5×N** | ✓ | 日度情绪分 → 情绪动量 |
| 新闻词权重 | `/news-word-weights?s=` | **5+5×N** | 未测 | 词级权重 → 主题/文本因子 |

> `s=` 可批量 15–20 只/次，批量摊薄单位成本。

### 优先级 4 — 宏观 / 利率 / 信用（择时层 regime）
| 数据 | 端点 | 成本 | 状态 | 用途 |
|---|---|---|---|---|
| 宏观指标 | `/macro-indicator/{COUNTRY}` | 1 | ✓ | 各国 50+ 指标（GDP、CPI、失业…）→ regime |
| 美债收益率 | `/ust/yield-rates` | 1 | ✓ | 10Y、期限利差 |
| 美债短/长/实际利率 | `/ust/bill-rates` `long-term-rates` `real-yield-rates` | 1 | ✓ | 实际利率、倒挂 |
| 政策利率 | `/rates/policy-rates` | 1 | ✓ | 全球央行利率 |
| 参考利率 | `/rates/reference-rates` | 1 | ✓ | SOFR/ESTR 等资金利率 |
| 融资压力利差 | `/spreads/funding-stress` | 1 | ✓ | EFFR-SOFR 等 → 流动性 regime |
| 信用风险 | `/credit-risk/sovereign/cds-spreads` 等 | 1 | ✓ | 主权 CDS、评级 → 风险偏好 |
| 经济事件 | `/economic-events` | 1 | ✓ | FOMC/CPI 发布日窗口 |

### 优先级 5 — 指数成分 / universe
| 数据 | 端点 | 成本 | 状态 | 用途 |
|---|---|---|---|---|
| 指数成分 | `/fundamentals/{INDEX}.INDX`（如 `GSPC.INDX`） | 10 | ✓ | ⚠ **实测只有当前成分快照**（Code/Exchange/Name/Sector/Industry/Weight），**无历史增删日期** → 不能单独构造 point-in-time universe；须结合退市股名单 + EOD 可得性 |
| CBOE 指数列表 | `/cboe/indices` | 1 | ✓ | 123 个 CBOE 指数目录 |
| CBOE 指数数据 | `/cboe/index?filter[index_code]=VIX` | 1 | ✓ | VIX 等波动率序列 |
| 交易所标的全清单 | `/exchange-symbol-list/US` | 1 | 未测 | 含 delisted → 扩池 |

### 优先级 6 — 辅助（低）
| 数据 | 端点 | 成本 | 状态 |
|---|---|---|---|
| 标的搜索 | `/search/{QUERY}` | 1 | 未测 |
| 代码变更历史 | `/symbol-change-history` | 1 | 未测 |
| 制裁名单 | `/sanctions/entities` 等 | 1 | 未测 |
| 房地产指标 | `/real-estate/{code}` | 1 | 未测 |

### 优先级 7 — 技术指标端点（**不建议**）
| 数据 | 端点 | 成本 | 说明 |
|---|---|---|---|
| EODHD 技术指标 | `/technical/{T}` | **5** | 与自有 TA（Technical Analysis，技术指标）重复 → 跳过 |

### 优先级 8 — 实时/盘中（执行层，非 alpha 研究）
| 数据 | 端点 | 成本 | 状态 |
|---|---|---|---|
| 盘中 K 线 | `/intraday/{T}.US` | 5 | 未测 |
| 实时报价（15min 延迟） | `/real-time/{T}.US` | 1 | 未测 |
| 盘前盘后报价 | `/us-quote-delayed` | 1 | 未测 |
| WebSocket 实时流 | `wss://ws.eodhistoricaldata.com/ws/us` | 无 per-request 限制 | 未测 |

## 三、权限实测（2026-10-12 经 MCP 复核）

### 3.1 经 MCP 通道**已可用**（原 REST 直连报 404，MCP 下正常）

| 数据 | MCP 工具 | 成本 | 实测结果 |
|---|---|---|---|
| SEC Form 4 内部人交易（含衍生品/脚注） | `get_insider_transactions_form4` | 10 calls/req | ✓ AAPL 返回 **617** 条 filed filings，字段完整 |
| 美股逐笔 Tick | `get_us_tick_data` | 10 calls/req | ✓ AAPL 30 分钟窗口返回逐笔（ts/price/shares/mkt/seq） |

> 说明：这两类走 MCP 的**独立授权通道**，与 REST 直连 API key 的权限不同；批量任务仍建议优先 REST，MCP 适合交互式单点取数。

### 3.2 实测**无权限**（403/404）——不要依赖

| 数据 | 端点 / MCP 工具 | 实测 |
|---|---|---|
| 美股期权链 / EOD | `/mp/unicornbay/options/*`、`get_us_options_contracts` | **403**（feed 里虽列 "Stock Options Data"，实际 marketplace 为空） |
| S&P Global 指数列表 | `/mp/unicornbay/spglobal/list`、`mp_indices_list` | **403** |
| Investverte ESG / PRAAMS / TradingHours | `/mp/investverte/*` `/mp/praams/*` `/mp/tradinghours/*` | 未测（marketplace 空，预期无权限） |
| SEC Financial Data（XBRL 结构化） | `/sec-financial-data` | 未测（路径待查；财务数据已可经 `/fundamentals` 获得） |

> 结论：**所有 `/mp/` 路径不可用**（REST 与 MCP 均 403）；index components 走 `/fundamentals/{INDEX}` 而非 `/mp/`。

## 四、额度粗算（us_500=501 只，100k/day）

| 任务 | 单只 | 全池 | 建议频率 |
|---|---|---|---|
| EOD 增量 | 1 | ~501（bulk 仅 100） | 每日 |
| 基本面全量 | 10 | 5,010 | 每季 |
| 历史市值 | 1 | 501 | 每日/周 |
| 内部人 / 国会 / 财报日历 | 全市场 1 次 | 1 | 每日 |
| 新闻/情绪 | 5+5N（批量20≈105/次） | ~2,600 | 每日 |

**全部新增源每日刷新也仅万级 calls，100k 额度充裕**；真正约束是 1,000 req/min（批量时需限速）。

## 五、建议落地顺序（美股优先）

1. 基本面 + 机构持仓 + 历史市值 → value/quality/size 因子（**信息增量最大**）
2. 内部人交易 + **国会交易** + 财报日历/趋势 → 事件/smart money 因子
3. 新闻 + 情绪 → attention/sentiment 因子
4. 宏观 + 利率 + 信用利差 → 择时层 regime
5. `/fundamentals/{INDEX}` 取权威成分 → 升级 universe（point-in-time）

---
*生成于 2026-10-12，基于账户实测 + eodhd-claude-skills 仓库文档。路径以 `eodhd.com/api/` 为前缀。*

## 六、数据范围重检（2026-10-12 v2 实测复核）

> 目的：在既有因子库（106 个：价量/技术 78 + 基本面 26 + 情绪 2）基础上，盘点**尚未挖掘**的数据方向并实测其历史回溯范围。

### 6.1 本轮新确认的历史范围（实测）

| 数据源 | 通道 | 实测范围 | 结论 |
|---|---|---|---|
| VIX（CBOE Volatility Index，芝加哥期权交易所波动率指数） | `get_historical_stock_prices` `VIX.INDX` | **1990-01-02 起**（17.24） | ✓ 可接 |
| UST Yield Rates（US Treasury，美债收益率） | `get_ust_yield_rates` | 1990-01-02 起，2250 条 | ✓ |
| Macro Indicator（宏观指标） | `get_macro_indicator` USA | **1960–2025，年频** | ✓ 粒度粗 |
| Sovereign CDS（主权信用违约互换利差） | `get_credit_sovereign_cds_spreads` | 年频，Damodaran 源 | ✓ 粒度粗 |
| Congressional Trades（国会交易） | `get_congressional_trades` | **46,133 条，1990 至今** | ⚠ `transaction_date` 年份被污染，须用 `disclosure_date` |
| SEC Filings（美国证监会文件） | `get_sec_filings` AAPL | 10-K 11 / 10-Q 35 / 8-K 108 / **Form 4 617** | ✓ |
| Dividends（分红） | `get_historical_dividends` AAPL | **1990 起全量**（declaration/record/payment 日齐全） | ✓ |
| Index Constituents（指数成分） | `get_fundamentals_data` `GSPC.INDX` | ✓ 含 Weight/Sector/Industry，但**仅当前快照** | ⚠ 无历史增删 → 改走「退市股名单 + EOD 可得性」 |
| News Word Weights（新闻词权重） | `get_news_word_weights` | ✓（AAPL 2024-09，news_processed=300） | ✓ |
| CBOE Index List | `get_cboe_indices_list` | 123 个，**多为欧洲区域指数，非 VIX** | 低价值 |
| Form 4 / Tick（逐笔） | `get_insider_transactions_form4` / `get_us_tick_data` | ✓ 经 MCP 独立授权可用 | ✓ |
| 所有 `/mp/`（Marketplace） | 期权/ESG/PRAAMS 等 | **403** | ✗ 无权限 |

### 6.2 尚未挖掘方向（按价值排序）

1. **Point-in-Time Universe（时点成分）** — 现 us_500 为**今日**成分 → 严重幸存者偏差；`GSPC.INDX` + `exchange_tickers?delisted=true` 可修口径（结构性问题，收益最大）。
2. **Insider Net Buy（内部人净买入）** — Form 4 逐笔。
3. **Congressional Net Buy（议员交易跟单）** — 46k 条稀有数据（须按 disclosure_date 对齐）。
4. **Shareholder Yield（股东收益率 = 分红+回购/市值）** — 补 value 族缺的回购维度。
5. **Volatility Regime（波动率状态）** — VIX + 期限利差/实际利率 → 外部 regime 门控。
6. **News Theme（新闻主题）** — word weights → 主题集中度/情绪分歧。
7. **PEAD 事件窗口（Post-Earnings Announcement Drift，盈余公告后漂移）** — 财报日历 + surprise 族。
8. **Institutional Holders（机构持仓）** — 现 JSON Holders 仅 2 条太薄，需改走季度 13F（成本高）。
9. **宏观/信用择时层** — 年频粒度太粗，仅作长期背景。

### 6.3 建议落地顺序

1. PIT 成分/退市股（先修口径） → 2. 内部人 Form 4 + 国会跟单 → 3. 股东收益率 → 4. VIX/利率 regime 门控 → 5. 新闻主题 / PEAD。

> 结论：存量数据已接近榨干；真正剩余空间在**结构（universe）**与**事件（smart money / 事件窗口）**两类，而非更多价量/基本面比率。

---

### 6.4 落地结果：方向 1 PIT 宇宙（2026-10-12 实测）

**取数**：`/exchange-symbol-list/US?delisted=1` → 退市股 60,420（Common Stock 33,175）；限美股主板 6 交易所 + 排除已在 `stock_data/` 者 → 随机抽样 2,000 只，并行下载 EOD 成功 **1,998 只** → `~/quant/newdata/pit_stock_data/{SYM}.csv`（脚本 `_tmp_fetch_newdata.py --what pit --pit-n 2000`）。

**宇宙**：PIT = us_500(501) ∪ 退市股(1,935) = **2,436 只**；研究版 pkl `~/quant/newdata/us_pit_day_ta_data_research.pkl`（1962-01-02~2026-10-09，16,400 交易日）。

**对比**：脚本 `_tmp_pit_lab.py`；同一因子信号分别在 biased（今日 us_500）与 PIT 两个宇宙内做截面 rank 归一化后评估（h=1 / top-k=20 / Q=5 / min_cs=8），产出 `factor/output/fundamentals/pit_lab_us.csv`（79 因子；跳过 27 个需 TA 列的 `T_*`/`_alpha_reg`）。

| 维度 | biased（今日 us_500） | PIT（含退市股） | 变化 |
|---|---|---|---|
| 均值 \|IC\| | 0.0035 | 0.0075 | **≈2.1×（+115%）** |
| nw_t 显著 (\|t\|>2) | 27 / 79 | 55 / 79 | +28 |

代表因子（IC_biased → IC_pit）：
- 动量：`mom_12_1` 0.0147→0.0201、`C_tmqmom` 0.0095→0.0143、`mom_120` 0.0066→0.0123
- 高位/趋势：`pos_252` 0.0071→0.0183、`hi_252` 0.0006→0.0186、`dd_60` −0.0016→0.0128
- **低波/流动性（符号被修正）**：`vol_60` 0.0003→−0.0221、`illiq_60` 0.0037→−0.0279(t=−15.6)、`cs_spread_20` 0.0020→−0.0244 —— 含退市股后低波异象才显现。

**结论**：现 us_500 回测**系统性低估价量因子有效性**；因退市股无基本面缓存，`F_*` 基本面因子 IC 几乎不变（delta≈0），即方向 1 的修口径**尚未覆盖 F_ 族**（需后续补退市股基本面）。注意 PIT 宇宙 = biased ∪ 2k 退市股抽样（非完整历史成分），故上述增幅为**下界估计**。

---

### 6.5 落地结果：方向 2 内部人 Form 4 + 国会跟单（2026-10-12 实测）

**取数**（脚本 `_tmp_fetch_newdata.py --what insider` / `--what congress`）：
- **内部人交易（Insider Transactions，美国 SEC Form 4 披露）**：REST `/insider-transactions?code=US`（批量 `offset` 无效，改用 `from`/`to` 日期切片 + 触达 `limit=1000` 即递归二分）→ 2011-01-01~2026-10-12 全市场 **104,484 行**、6,633 只 → `~/quant/newdata/insider_us.pkl`。
- **国会交易（Congressional Trades，议员 STOCK Act 披露）**：REST `/congressional-trades`（真实分页参数为 JSON:API 风格 `page[offset]`/`page[limit]≤100`，`offset`/`limit` 被忽略）→ 全量 **42,612 行**、3,401 只 → `~/quant/newdata/congress_trades.pkl`；`transaction_date` 年份被污染，统一改用 `disclosure_date`。

**PIT（Point-in-Time，时点）对齐**：内部人按 `reportDate`（申报日）、国会按 `disclosure_date`（披露日）作为**可用日**；事件日映射到「≥ 该日的最近交易日」，避免前视。逐 symbol-date 聚合净买入流 → 滚动 w 个交易日求和得因子：

| 因子 | 含义 |
|---|---|
| `I_shares_{20,60,120}` | 内部人公开市场净买入**股数**（P 记 +，S/F 记 −），滚动 20/60/120 日 |
| `I_val_{20,60,120}` | 内部人净买入**金额**（股数 × 成交价） |
| `P_val_{20,60,120}` | 国会议员净买入**金额**（purchase +，sale −） |
| `P_cnt_60` | 国会议员净买入**笔数** |

**回测**（脚本 `_tmp_dir2_lab.py`）：us_500 池、full 窗口（1990~2026-10-09）、method=`rank`、top-k=20、Q=5、min_cs=8；跨 horizon h∈{1,5,20,60} 评估 → `factor/output/fundamentals/dir2_lab_us.csv`（64 行）、`dir2_lab_us.png`。

**结论 1 — 内部人净买入是「慢变量」，h=60 才显著**：IC 随 horizon **单调上升**（h=1≈0 → h=60 显著），符合内部人信息逐步反映。

| 因子 (h=60) | 覆盖率 | Rank IC | nw_t | ls_spread (Q5−Q1) |
|---|---|---|---|---|
| `I_val_120` | 2.9% | **0.0316** | **2.50** | +0.0036 |
| `I_val_60` | 2.1% | 0.0234 | 2.11 | +0.0099 |
| `I_shares_120` | 2.9% | 0.0254 | 2.00 | −0.0012 |
| `I_shares_60` | 2.1% | 0.0189 | 1.67 | +0.0074 |

对照控制组（h=60）：`vol_60` IC 0.049 / `illiq_60` 0.054 / `beta_60` 0.044（`illiq_60` nw_t=6.1）。

**结论 2 — 国会交易在 us_500 池上无预测力**：全 horizon IC≈0 或为负，`P_cnt_60` h=60 nw_t=−1.62；覆盖虽略高（9~13%）但披露滞后 + 噪声淹没信号。

**结论 3 — 极端桶（top-20）不稳定，须用分层**：内部人因子 IC 为正但 `excess`（top-20 相对池等权）为负，而 `ls_spread`（Q5−Q1）为正 → 覆盖仅 1~3%，top-20 极端桶高度集中且噪声大；信号方向在**分层（quintile）/行业中性**层面成立，不宜直接用于超集中 top-k 组合。

**落地结论**：内部人净买入可保留为**慢频（h≈60）/ 分层**候选因子，并需与流动性、波动因子做正交化；国会跟单在 us_500 上**不纳入**。

---

### 6.6 落地结果：方向 3 股东收益率（分红 + 回购）（2026-10-12 实测）

**取数**（脚本 `_tmp_fetch_newdata.py --what px_mcap` / `--what dividends`）：
- **原始收盘价 + 市值**：REST `/eod/{T}.US`（取 `close` 原始价，非 `adjusted_close`）+ `/historical-market-cap/{T}.US` → `~/quant/newdata/px_mcap/{SYM}.csv`（列 `date,close_raw,mcap,symbol`），**501/501 全部成功**。
- **分红**：REST `/div/{T}.US` → `~/quant/newdata/dividends_us.pkl`（列 `symbol/date/declarationDate/recordDate/paymentDate/value/unadjustedValue/period`），**49,464 行、424 只**。

**⚠️ 关键数据陷阱（本方向实测发现）**：研究版面板 `ds.wide('Dividend')` 的 **非零值占比 = 0.0**（虽 notna 占比 0.79），即面板内 `Dividend` 全为 0 → **不能用于股息率**，必须回源 `/div`。另有：`/historical-market-cap` 为**周频、仅约 5 年**（2021+），故回购类因子覆盖受限（±预期）。

**PIT 对齐**：分红用 `paymentDate`（缺失回退 `declarationDate`/`date`）作为可用日，映射到「≥ 该日的最近交易日」，避免前视；金额优先 `unadjustedValue`（回退 `value`）。回购用 `shares = mcap / close_raw`。

**因子定义**（`_tmp_dir3_lab.py`）：

| 因子 | 含义 |
|---|---|
| `DY_ttm` | 股息率（Dividend Yield，TTM）= 滚动 252 日分红和 / 原始收盘价 |
| `DY_gr` | 分红增长 = 股息率 / 去年同期 − 1 |
| `BY` | 回购收益率（Buyback Yield）= −(股本同比增速) |
| `ShY` | 股东收益率（Shareholder Yield）= `DY_ttm + BY` |

**回测**（脚本 `_tmp_dir3_lab.py`）：us_500 池、full 窗口（2011-01-01~2026-10-09）、method=`rank`、top-k=20、Q=5、min_cs=8；跨 horizon h∈{1,5,20,60} → `factor/output/fundamentals/dir3_lab_us.csv`（40 行）、`dir3_lab_us.png`。

| 因子 | 覆盖率 | h=1 IC / t | h=5 IC / t | h=20 IC / t | h=60 IC / t |
|---|---|---|---|---|---|
| `DY_ttm` | 40.3% | −0.0054 / −1.83 | **−0.0121 / −2.30** | −0.0189 / −1.87 | −0.0290 / −1.59 |
| `DY_gr` | 29.2% | −0.0048 / −1.61 | −0.0008 / −0.15 | 0.0076 / 0.76 | 0.0136 / 0.97 |
| `BY` | 11.0% | 0.0038 / 1.19 | 0.0070 / 1.21 | **0.0151 / 1.48** | 0.0119 / 0.71 |
| `ShY` | 11.0% | 0.0000 / 0.00 | 0.0020 / 0.26 | 0.0067 / 0.54 | 0.0042 / 0.22 |

对照控制组（h=60）：`vol_60` IC 0.049（t=2.1）/ `illiq_60` 0.054（t=6.1）/ `beta_60` 0.044（t=1.7）。

**结论 1 — 股息率 `DY_ttm` 对前瞻收益「弱负相关」**：IC 全 horizon 为负、随 h 单调走低，h=5 最显著（IC=−0.0121、nw_t=−2.30）；即在 us_500 大池中，**高股息率并非超额来源**（多为低增长/防御/价值陷阱），与「高股息=高质量」的直觉相反，需与质量/成长因子正交后再用。
**结论 2 — 分红增长 `DY_gr` / 回购收益率 `BY` 弱正但不显著**：`DY_gr` h=60 IC=0.0136（t=0.97）、`BY` h=20 IC=0.0151（t=1.48），方向为正、未过 |t|>2；`BY` 覆盖仅 11%（受 market-cap 仅 5 年所限）。
**结论 3 — 股东收益率 `ShY` 近中性**：IC≈0（各 h |t|<0.6），因正的回购与负的股息相互抵消；说明简单加总 `DY+BY` 无效，须分项使用。
**落地结论**：方向 3 在 us_500 上**不纳入简单合成口径**；如使用，建议仅保留 `BY`（h≈20、分层）作为独立回购信号，并**禁用面板 `Dividend` 列**（恒 0）改走 `/div` 回源。

---

### 6.7 落地结果：方向 4 宏观 regime 门控（VIX / 美债期限利差）（2026-10-12 实测）

**取数**（脚本 `_tmp_fetch_newdata.py --what macro`）：REST `/eod/{T}` → `~/quant/newdata/macro_long.pkl`，键 `vix / us10y / tnx / us3m / us2y`（`close` 列）。覆盖：VIX 1990+（9,300 行）、US10Y 1990+（9,330）、US3M 1990+（9,353）、TNX 1993+（8,268）、US2Y 仅 2012+（3,531）。

**思路**：宏观状态在 t 日**可观测**，用它给因子「前瞻有效性」分状态评估——即衡量因子在何种市场状态下才有效（门控）。两种状态：
- **VIX regime**：按全样本分位 [1/3, 2/3]（q1=15.2、q2=20.8）→ `low / mid / high`。
- **期限利差 regime**：`Term Spread = US10Y − US3M`；`inverted(<0)`（倒挂，衰退前瞻信号）/ `normal(≥0)`。

**回测**（脚本 `_tmp_dir4_lab.py`）：us_500 池、full 窗口（2011-01-01~2026-10-09）、method=`rank`、min_cs=8；对 6 个基础因子取逐日 Rank IC 序列（`ev.ic_series`），再按 regime 分组汇总（`ev.ic_summary`，nw_lag=h−1）→ `factor/output/fundamentals/dir4_lab_us.csv`（108 行）、`dir4_lab_us.png`。

**核心发现 — 风险/低波/反转因子是「压力状态因子」**（h=60，括号内 NW-t）：

| 因子 | VIX low | VIX high | 利差 normal | 利差 inverted | ALL |
|---|---|---|---|---|---|
| `beta_60` | −0.0074 (−0.24) | **0.1599 (3.07)** | 0.0243 (0.88) | **0.1320 (3.22)** | 0.0436 |
| `vol_60` | 0.0141 (0.50) | **0.1393 (2.79)** | 0.0354 (1.40) | **0.1138 (2.97)** | 0.0494 |
| `rev_5`（短反转） | 0.0038 (0.64) | **0.0376 (3.49)** | 0.0083 (1.57) | 0.0193 (1.43) | 0.0103 |
| `illiq_60` | 0.0318 (2.84) | **0.0944 (6.10)** | 0.0599 (6.29) | 0.0244 (1.43) | 0.0536 |
| `mom_60`（动量） | −0.0005 (−0.03) | **−0.0464 (−1.63)** | −0.0137 (−0.89) | −0.0371 (−1.83) | −0.0179 |
| `vratio_20` | −0.0080 (−2.93) | 0.0001 (0.03) | −0.0038 (−1.61) | −0.0057 (−1.22) | −0.0041 |

**结论 1 — 高 VIX / 利差倒挂时，低 beta、低波、反转、非流动性因子 IC 放大约 3~7 倍且显著**（`beta_60`：low −0.007→high 0.160；`vol_60` 0.014→0.139）。即**风险规避因子在压力状态下才真正提供超额**，平静期几乎失效。
**结论 2 — 动量在压力状态「受伤」**：`mom_60` IC 在高 VIX / 倒挂时明显更负（h=60 −0.046 / −0.037，vs 平静期 ≈0），符合「动量崩溃（momentum crash）」在危机后反转的市场事实。
**结论 3 — `illiq_60`（非流动性）最稳健**：各状态均为正且显著（t 2.8~6.3），是**跨状态通用因子**；`vratio_20` 全状态下无效。
**落地结论**：方向 4 具备**门控价值**——建议在组合层引入 **VIX / 期限利差状态开关**：高 VIX 或倒挂期**放大**低 beta/低波/反转因子权重、**降低**动量权重；平静期反之。recall：宏观数据（VIX、UST 各期限）全历史可得，门控**无覆盖限制**（对比方向 3 受 market-cap 5 年所限）。

---

### 6.8 落地结果：方向 5 PEAD（财报公告后漂移）（2026-10-12 实测）

**取数**（脚本 `_tmp_fetch_newdata.py --what earnings`）：REST `/calendar/earnings`（按年切片 `from`/`to`）→ `~/quant/newdata/earnings_calendar.pkl`，**2,223,314 行**（2011-01-01~2026-10-12）。列：`code / report_date(公告日) / date(财季) / actual / estimate / difference / percent`。其中 `estimate` 仅 51%、`percent` 仅 33% 非空。

**新闻主题**：REST `/news-word-weights`（脚本 `--what wordweights`）仅做**抽样**（20 只）→ `~/quant/newdata/wordweights_sample.pkl`；覆盖标的太少，**不足以做全池因子评估**，本方向以 PEAD 为主。

**因子定义**（`_tmp_dir5_lab.py`）：
- 盈余意外（Standardized Earnings Surprise，盈余意外）`surprise = (actual − estimate)/|estimate|`，裁剪到 [−1, 1]；us_500 命中 **29,404 条**、498 只。
- **PIT 对齐**：`report_date`（公告日）为可用日，信号置于公告日，前瞻收益从**次日开盘**起算（`forward_return` 的 `open_.shift(-1)`），无前视。
- **PEAD 漂移**：信号在公告后 L 个交易日内保持（`ffill(limit=L)`）→ `PEAD_20`（L=20，覆盖 13.3%）/ `PEAD_60`（L=60，覆盖 37.7%）。

**回测**（`_tmp_dir5_lab.py`）：us_500 池、full 窗口、method=`rank`、top-k=20、Q=5、min_cs=8 → `factor/output/fundamentals/dir5_lab_us.csv`（32 行）、`dir5_lab_us.png`。

| 因子 | 覆盖率 | h=1 IC / t | h=5 IC / t | h=20 IC / t | h=60 IC / t | 单调性(h=20) |
|---|---|---|---|---|---|---|
| `PEAD_60` | 37.7% | **0.0056 / 4.67** | **0.0088 / 4.07** | **0.0108 / 2.52** | 0.0073 / 1.02 | 0.70 |
| `PEAD_20` | 13.3% | 0.0058 / 2.43 | 0.0090 / 2.11 | 0.0112 / 1.28 | 0.0070 / 0.67 | 0.50 |

对照（h=20）：`illiq_60` 0.0283（t=4.99）/ `rev_5` 0.0089（t=1.77）/ `mom_60` −0.0098。

**结论 1 — PEAD 在 us_500 上显著且稳健**：`PEAD_60` 在 h=1/5/20 均 nw_t>2.5（h=1 最强 t=4.67），IC 随 h 先升后回、quintile 单调性 0.60~0.70，`ls_spread`（Q5−Q1）在 h=20 达 +0.22% / 20 日 —— 说明**盈余意外后的漂移在公告后约 1~20 交易日持续**，与经典 PEAD 文献一致。
**结论 2 — 长漂移窗（L=60）优于短窗**：`PEAD_60` 覆盖为 `PEAD_20` 的 2.8 倍、t 值更高更稳；短窗虽短端 IC 略高但降速快、显著性弱。
**落地结论**：方向 5 的 **PEAD 具备落地价值**（大池、显著、低换手，turnover 仅 5.5%）；建议纳入因子库作为**事件驱动 / 盈余意外**独立因子（h≈5~20、分层或中性化后使用）。**新闻词权重**因端点仅支持逐只拉取、全池成本高且历史受限，暂**仅保留抽样验证**，不纳入因子。

**总结**：方向 5 为建议顺序（§6.3）最后一环，至此五个方向（PIT 宇宙 → 内部人/国会 → 股东收益率 → 宏观 regime → PEAD）**全部落地闭环**。可落地优先级：**PEAD（§6.8）、内部人净买入（§6.5）、宏观 regime 门控（§6.7）> PIT 口径修正（§6.4，需补退市股基本面）> 股东收益率（§6.6，仅 `BY`）**。