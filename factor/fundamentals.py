# -*- coding: utf-8 -*-
"""
factor.fundamentals — 基本面因子层(自包含)
==========================================
职责: 把 EODHD fundamentals JSON(~ /quant/fundamental_data/fundamentals/*.json) 转成
      **PIT(Point-in-Time, 时点)可比**的 (date × symbol) 宽表, 并注册为因子.

为什么必须 PIT:
  财报是季度数据, 但披露有滞后 —— 某季(如 2026-06-30)的报表要到 2026-07-31 才披露.
  若直接把 "报告期" 当 "可用日", 就等于在 7 月 31 日之前就用了 6 月 30 日的数字 -> 前视偏差.
  本模块一律以 EODHD 的 `filing_date`(披露日) 作为**可用日(avail_date)**;
  `filing_date` 缺失时回退 `报告期 + 90 天`. 再用 ffill 把事件对齐到交易日网格,
  这样 t 日的因子值只依赖 <= t 已披露的报表, 通过 verify_causal(截断一致性检验).

数据口径(实测 501/501 可用):
  * 金额为原始美元, 股数为原始股数(EODHD 不缩放);
  * TTM(Trailing Twelve Months, 滚动十二个月) = 最近已披露 4 季之和, 可用日取第 4 季的 filing_date;
  * 市值(Market Cap) = 当日收盘价 × PIT 股数(Balance_Sheet.quarterly.commonStockSharesOutstanding);
    —— 不用 market_cap/*.csv(仅 ~5 年周频), 这样可得全历史.
  * Holders(机构持仓) 仅最新快照、无历史序列 -> 本期不做持仓因子.

产物:
  * 事件长表缓存: output/fundamentals/pit_events.pkl(带文件签名, 命中即复用);
  * 分析师预期缓存: output/fundamentals/earnings_events.pkl(来源 JSON 的 Earnings.History);
  * 因子: size(F_logmcap) / value(F_ep,F_bp,F_sp,F_fcf_yield,F_ebitda_yield) /
          quality(F_roe,F_roa,F_gross_margin,F_op_margin,F_debt2equity,F_current_ratio) /
          growth(F_rev_yoy,F_eps_yoy;  F_ann_eps_yoy,F_eps_grow_streak,F_eps_cagr_3y 年度EPS趋势) /
          surprise(F_sue_eps,F_sue_rev 季节随机游走;  F_eps_surp,F_sue_analyst,F_eps_surp_4q
                   分析师预期差;  F_beat_streak,F_surp_accel 超预期持续性/加速度) /
          issuance(F_net_issuance,F_net_issuance_3y 净股票发行异象).

  ⚠ 只有 Financials(三表) / Earnings.History / Earnings.Annual / outstandingShares 是
    **多年时序**(可 PIT 回测);
    Valuation/SharesStats/AnalystRatings/Holders/InsiderTransactions 等板块在 EODHD
    的 JSON 里**仅最新快照**, 无历史序列, 不可用于历史回测(会引入前视).
  ⚠ 后两类(Earnings.Annual 无分析师预期,F 只含 epsActual;  outstandingShares 无披露日)
    缺披露时间戳 -> 一律用"报告期 + 回退滞后"作为可用日(年报 90 天 / 季报 45 天, 偏保守).

自检:
  cd ~/git && python -m quant.factor.fundamentals
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factor import config as cfg
from quant.factor import factor as _fct
from quant.factor.data import Dataset

# 以 `python -m quant.factor.fundamentals` 直接运行时, factor.py 底部导入会先生成
# canonical 副本(quant.factor.fundamentals)并完成注册, 再轮到本 __main__ 副本执行装饰器
# -> 会重复注册. 此处令 __main__ 副本复用 no-op 装饰器, 只跑自检, 不再注册.
_RERUN = __name__ == '__main__' and 'quant.factor.fundamentals' in sys.modules
_register = _fct.register if not _RERUN else (lambda *a, **k: (lambda fn: fn))

FUND_DIR = cfg.HOME / 'quant' / 'fundamental_data'
FUND_JSON_DIR = FUND_DIR / 'fundamentals'
CACHE_DIR = cfg.OUT_DIR / 'fundamentals'
EVENTS_PKL = CACHE_DIR / 'pit_events.pkl'
EARN_PKL = CACHE_DIR / 'earnings_events.pkl'
EXTRA_PKL = CACHE_DIR / 'extra_events.pkl'
FALLBACK_LAG_DAYS = 90                          # filing_date 缺失时的回退披露滞后
EARN_FALLBACK_LAG_DAYS = 45                     # Earnings reportDate 缺失时的回退披露滞后
ANN_FALLBACK_LAG_DAYS = 90                      # Earnings.Annual(10-K)缺失披露日时的回退滞后
SHARE_FALLBACK_LAG_DAYS = 45                    # outstandingShares(10-Q)缺失披露日时的回退滞后
_SCHEMA = 3                                     # 缓存结构版本: 解析列/口径变更时 +1, 自动失效旧缓存

INC_FIELDS = ['totalRevenue', 'grossProfit', 'netIncome', 'ebit',
              'ebitda', 'operatingIncome']
BAL_FIELDS = ['commonStockSharesOutstanding', 'totalStockholderEquity', 'totalAssets',
              'totalCurrentAssets', 'totalCurrentLiabilities', 'totalLiab',
              'longTermDebt']
CF_FIELDS = ['freeCashFlow', 'totalCashFromOperatingActivities', 'capitalExpenditures']
FLOW_FIELDS = ['totalRevenue', 'grossProfit', 'netIncome', 'ebitda',
               'operatingIncome', 'freeCashFlow', 'totalCashFromOperatingActivities']

# 事件长表最终列(供因子直接消费)
EVENT_COLS = ['shares', 'equity', 'total_assets', 'total_liab', 'current_assets',
              'current_liab', 'long_term_debt', 'ttm_net_income', 'ttm_revenue',
              'ttm_gross_profit', 'ttm_ebitda', 'ttm_op_income', 'ttm_fcf',
              'ttm_ocf', 'ttm_eps', 'rev_yoy', 'eps_yoy',
              'eps_q', 'sue_eps', 'sue_rev']

# 各列在 JSON 里的源字段名
_SRC = {
  'shares': 'commonStockSharesOutstanding',
  'equity': 'totalStockholderEquity',
  'total_assets': 'totalAssets',
  'total_liab': 'totalLiab',
  'current_assets': 'totalCurrentAssets',
  'current_liab': 'totalCurrentLiabilities',
  'long_term_debt': 'longTermDebt',
  'ttm_net_income': 'netIncome',
  'ttm_revenue': 'totalRevenue',
  'ttm_gross_profit': 'grossProfit',
  'ttm_ebitda': 'ebitda',
  'ttm_op_income': 'operatingIncome',
  'ttm_fcf': 'freeCashFlow',
  'ttm_ocf': 'totalCashFromOperatingActivities',
}

# ---- 内存缓存 ----
_EVENTS_CACHE: pd.DataFrame = None              # 事件长表(symbol, avail_date, ...)
_WIDE_CACHE: dict = {}                          # 列名 -> 以 avail_date 为索引的宽表
_EARN_CACHE: pd.DataFrame = None                # 分析师预期长表(symbol, avail_date, ...)
_EARN_WIDE_CACHE: dict = {}                     # 同上(earnings 专用)


def _num(v):
  """JSON 数值 -> float; 缺失/非法 -> NaN."""
  if v is None:
    return np.nan
  try:
    return float(v)
  except (TypeError, ValueError):
    return np.nan


def _ts(s):
  """日期字符串 -> Timestamp; 非法 -> NaT."""
  return pd.to_datetime(s, errors='coerce')


def _avail(period_ts, filings):
  """可用日: 优先最大 filing_date, 否则 报告期 + FALLBACK_LAG_DAYS."""
  fds = [t for t in (_ts(f) for f in filings) if pd.notna(t)]
  if fds:
    return max(fds)
  return period_ts + pd.Timedelta(days=FALLBACK_LAG_DAYS)


def _symbol_events(sym: str, raw: dict) -> list:
  """单个标的: 三大报表按报告期合并 -> 每期一条记录(含 TTM/YoY 前的原始字段)."""
  fin = raw.get('Financials') or {}
  periods: dict = {}
  for recs, flds in [((fin.get('Income_Statement') or {}).get('quarterly'), INC_FIELDS),
                     ((fin.get('Balance_Sheet') or {}).get('quarterly'), BAL_FIELDS),
                     ((fin.get('Cash_Flow') or {}).get('quarterly'), CF_FIELDS)]:
    if not isinstance(recs, dict):
      continue
    for k, r in recs.items():
      if not isinstance(r, dict):
        continue
      p = r.get('date') or k
      d = periods.setdefault(p, {'period': _ts(p), '_filings': []})
      fd = r.get('filing_date')
      if fd:
        d['_filings'].append(fd)
      for f in flds:
        if r.get(f) is not None:
          d[f] = _num(r[f])

  rows = []
  for p, d in periods.items():
    if pd.isna(d['period']):
      continue
    rows.append(d)
  if not rows:
    return []

  df = pd.DataFrame(rows).sort_values('period').reset_index(drop=True)
  # TTM: 最近 4 季滚动求和(不足 4 季 -> NaN)
  for f in FLOW_FIELDS:
    if f in df.columns:
      df['ttm_' + f] = df[f].rolling(4, min_periods=4).sum()
  # 每股收益(近似): TTM 净利 / 当期股数
  df['ttm_eps'] = df.get('ttm_netIncome', np.nan) / df['commonStockSharesOutstanding'] \
      if 'commonStockSharesOutstanding' in df.columns else np.nan
  # YoY: TTM 同比(报告期对齐, shift(4) = 去年同期)
  df['rev_yoy'] = df['ttm_totalRevenue'] / df['ttm_totalRevenue'].shift(4) - 1 \
      if 'ttm_totalRevenue' in df.columns else np.nan
  df['eps_yoy'] = df['ttm_eps'] / df['ttm_eps'].shift(4) - 1
  # 季度 EPS(近似): 季度净利 / 当期股数
  if 'netIncome' in df.columns and 'commonStockSharesOutstanding' in df.columns:
    df['eps_q'] = df['netIncome'] / df['commonStockSharesOutstanding'].replace(0, np.nan)
  else:
    df['eps_q'] = np.nan
  # SUE(Standardized Unexpected Earnings, 标准化盈余意外):
  #   季节性随机游走意外 = X_t - X_{t-4}(同报告期同比差分), 再除以该意外的滚动标准差.
  #   捕捉"预期差"(变化量)而非"水平量", 与 YoY 的差别在于以历史波动做标准化.
  for _src, _name in [('eps_q', 'sue_eps'), ('totalRevenue', 'sue_rev')]:
    if _src in df.columns:
      _d = df[_src] - df[_src].shift(4)
      df[_name] = _d / _d.rolling(8, min_periods=4).std()
    else:
      df[_name] = np.nan
  # 可用日: 用 df 内已随排序对齐的 _filings 列计算(勿用未排序的 rows)
  df['avail_date'] = [_avail(per, fil)
                      for per, fil in zip(df['period'], df['_filings'])]

  out = pd.DataFrame({'symbol': sym, 'avail_date': df['avail_date']})
  for col, src in _SRC.items():
    key = 'ttm_' + src if col.startswith('ttm_') else src   # ttm_* 取滚动列, 其余取原始列
    out[col] = df[key].values if key in df.columns else np.nan
  out['ttm_eps'] = df['ttm_eps'].values
  out['rev_yoy'] = df['rev_yoy'].values
  out['eps_yoy'] = df['eps_yoy'].values
  out['eps_q'] = df['eps_q'].values
  out['sue_eps'] = df['sue_eps'].values
  out['sue_rev'] = df['sue_rev'].values
  return out.to_dict('records')


# ---------- Earnings(分析师预期) 板块 ----------
# JSON 的 Earnings.History: 每季一条 {date(报告期), reportDate(披露日), epsActual,
# epsEstimate, epsDifference, surprisePercent}. 这是**独立于三表**的第二条多年时序,
# 直接给出"实际 vs 分析师一致预期"的预期差, 是 PEAD(盈余公告后漂移)因子的标准输入.
# 可用日一律取 reportDate(公告日) —— 比三表的 filing_date 更贴近信息到达市场的时点.
def _symbol_earnings(sym: str, raw: dict) -> list:
  """单个标的: Earnings.History -> 每季一条(未披露的实际值跳过, 避免前视)."""
  hist = (raw.get('Earnings') or {}).get('History') or {}
  rows = []
  for k, r in hist.items():
    if not isinstance(r, dict):
      continue
    actual = _num(r.get('epsActual'))
    if np.isnan(actual):                        # 未来季度(尚未披露) -> 跳过
      continue
    est = _num(r.get('epsEstimate'))
    diff = _num(r.get('epsDifference'))
    if np.isnan(diff) and not np.isnan(est):
      diff = actual - est
    surp = _num(r.get('surprisePercent'))
    if np.isnan(surp) and not np.isnan(diff) and not np.isnan(est) and est != 0:
      surp = diff / abs(est) * 100.0
    period = _ts(r.get('date') or k)
    rd = _ts(r.get('reportDate'))
    if pd.notna(rd):
      avail = rd
    elif pd.notna(period):
      avail = period + pd.Timedelta(days=EARN_FALLBACK_LAG_DAYS)
    else:
      continue
    rows.append({'symbol': sym, 'avail_date': avail, 'eps_diff': diff,
                 'eps_surp': surp / 100.0 if not np.isnan(surp) else np.nan})
  if not rows:
    return []
  df = pd.DataFrame(rows).sort_values('avail_date').reset_index(drop=True)
  # 标准化预期差(SUE-analyst): 意外 / 该公司过去 8 季意外的波动(同 F_sue_eps 口径)
  d = df['eps_diff']
  df['sue_analyst'] = d / d.rolling(8, min_periods=4).std()
  # 持续性预期差: 最近 4 季 eps_surp 的均值(更平滑, 换手更低)
  df['eps_surp_4q'] = df['eps_surp'].rolling(4, min_periods=2).mean()
  # 超预期持续性(beat streak): 连续"实际>预期"(eps_diff>0)的季度数; 断一次归零.
  pos = (df['eps_diff'] > 0).astype(int)
  df['beat_streak'] = pos * (pos.groupby((pos != pos.shift()).cumsum()).cumcount() + 1)
  # 预期差加速度: 本季 eps_surp 相对上季的变化(二阶动量)
  df['surp_accel'] = df['eps_surp'] - df['eps_surp'].shift(1)
  return df.to_dict('records')


def build_earnings(force: bool = False) -> pd.DataFrame:
  """
  扫描 fundamentals JSON 的 Earnings.History -> PIT 分析师预期长表
  (symbol, avail_date, eps_diff, eps_surp, sue_analyst, eps_surp_4q). 带磁盘缓存.

  :param force: True 强制重建
  :returns: 长表(按 symbol, avail_date 排序, 同 (symbol,avail_date) 去重取最后)
  """
  global _EARN_CACHE
  if _EARN_CACHE is not None and not force:
    return _EARN_CACHE

  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  sig = _signature()
  if not force and EARN_PKL.exists():
    try:
      cached = pd.read_pickle(EARN_PKL)
      if isinstance(cached, pd.DataFrame) and cached.attrs.get('sig') == sig:
        _EARN_CACHE = cached
        return cached
    except Exception:
      pass

  recs = []
  for p in sorted(FUND_JSON_DIR.glob('*.json')):
    try:
      raw = json.load(open(p, encoding='utf-8'))
    except Exception:
      continue
    recs.extend(_symbol_earnings(p.stem, raw))

  ev = pd.DataFrame(recs)
  if ev.empty:
    raise RuntimeError(f'未解析出任何 Earnings 事件: {FUND_JSON_DIR}')
  ev = (ev.sort_values(['symbol', 'avail_date'])
          .drop_duplicates(['symbol', 'avail_date'], keep='last')
          .reset_index(drop=True))
  ev.attrs['sig'] = sig
  ev.to_pickle(EARN_PKL)
  _EARN_CACHE = ev
  return ev


def _earn_wide(col: str) -> pd.DataFrame:
  """Earnings 长表某列 -> (avail_date × symbol) 宽表(带内存缓存)."""
  if col in _EARN_WIDE_CACHE:
    return _EARN_WIDE_CACHE[col]
  ev = build_earnings()
  w = (ev.pivot_table(index='avail_date', columns='symbol', values=col,
                      aggfunc='last', dropna=False)
         .sort_index())
  _EARN_WIDE_CACHE[col] = w
  return w


def _ewide(ds: Dataset, col: str) -> pd.DataFrame:
  """Earnings 某列对齐到 ds 交易日网格(每交易日取 <= t 最近一次公告的值, ffill)."""
  w = _earn_wide(col)
  idx = w.index.union(ds.dates)
  aligned = w.reindex(idx).ffill().reindex(ds.dates)
  return aligned.reindex(columns=ds.symbols)


def _signature() -> tuple:
  """JSON 目录签名(文件数 + 最大 mtime + 结构版本), 用于判断缓存是否失效."""
  files = sorted(FUND_JSON_DIR.glob('*.json'))
  mtime = max((p.stat().st_mtime for p in files), default=0.0)
  return (len(files), mtime, _SCHEMA)


def build_events(force: bool = False) -> pd.DataFrame:
  """
  扫描 fundamentals JSON -> PIT 事件长表 DataFrame(symbol, avail_date, EVENT_COLS).
  带磁盘缓存(output/fundamentals/pit_events.pkl): 文件签名未变则直接复用.

  :param force: True 强制重建
  :returns: 事件长表(按 symbol, avail_date 排序, 同 (symbol,avail_date) 去重取最后)
  """
  global _EVENTS_CACHE
  if _EVENTS_CACHE is not None and not force:
    return _EVENTS_CACHE

  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  sig = _signature()
  if not force and EVENTS_PKL.exists():
    try:
      cached = pd.read_pickle(EVENTS_PKL)
      if isinstance(cached, pd.DataFrame) and cached.attrs.get('sig') == sig:
        _EVENTS_CACHE = cached
        return cached
    except Exception:
      pass

  files = sorted(FUND_JSON_DIR.glob('*.json'))
  recs = []
  for p in files:
    try:
      raw = json.load(open(p, encoding='utf-8'))
    except Exception:
      continue
    recs.extend(_symbol_events(p.stem, raw))

  ev = pd.DataFrame(recs)
  if ev.empty:
    raise RuntimeError(f'未解析出任何基本面事件: {FUND_JSON_DIR}')
  ev = ev.dropna(subset=['avail_date'])
  ev = (ev.sort_values(['symbol', 'avail_date'])
          .drop_duplicates(['symbol', 'avail_date'], keep='last')
          .reset_index(drop=True))
  ev.attrs['sig'] = sig
  ev.to_pickle(EVENTS_PKL)
  _EVENTS_CACHE = ev
  return ev


def _wide_by_avail(col: str) -> pd.DataFrame:
  """事件长表某列 -> (avail_date × symbol) 宽表(带内存缓存, 索引升序)."""
  if col in _WIDE_CACHE:
    return _WIDE_CACHE[col]
  ev = build_events()
  w = (ev.pivot_table(index='avail_date', columns='symbol', values=col,
                      aggfunc='last', dropna=False)
         .sort_index())
  _WIDE_CACHE[col] = w
  return w


def _fwide(ds: Dataset, col: str) -> pd.DataFrame:
  """
  某列对齐到 ds 交易日网格: 每个交易日 t 取"<= t 最近一次披露"的值(ffill).
  严格向后看, 因此因果 —— 与喂给 ds 的数据长度无关(截断一致性成立).
  """
  w = _wide_by_avail(col)
  idx = w.index.union(ds.dates)
  aligned = w.reindex(idx).ffill().reindex(ds.dates)
  return aligned.reindex(columns=ds.symbols)


# ---------- 附加多年时序: Earnings.Annual(年度 EPS) 与 outstandingShares(季度股数) ----------
# 这两块都**没有披露时间戳**(Annual 只有 reportDate/date; outstandingShares 只有报告期末),
# 因此可用日一律用 "报告期末 + 回退滞后"(年报 90 天 / 季报 45 天), 取保守值避免前视.
# Earnings.Annual 仅含 epsActual(无分析师预期) -> 只能做"年度水平/趋势"类因子;
# outstandingShares.quarterly 是长历史股数序列 -> 做经典"净股票发行(NSI)"异象.
_EXTRA_CACHE: dict = None                       # {'annual': df, 'shares': df}
_EXTRA_WIDE: dict = {}                          # (kind, col) -> 宽表


def _symbol_annual(sym: str, raw: dict) -> list:
  """单个标的: Earnings.Annual -> 每财年一条(ann_eps + 年度同比/连续增长年数/3年CAGR)."""
  ea = (raw.get('Earnings') or {}).get('Annual') or {}
  rows = []
  for k, r in ea.items():
    if not isinstance(r, dict):
      continue
    eps = _num(r.get('epsActual'))
    per = _ts(r.get('date') or k)
    if np.isnan(eps) or pd.isna(per):
      continue
    rows.append({'symbol': sym, 'avail_date': per + pd.Timedelta(days=ANN_FALLBACK_LAG_DAYS),
                 '_period': per, 'ann_eps': eps})
  if not rows:
    return []
  df = pd.DataFrame(rows).sort_values('_period').reset_index(drop=True)
  # 年度 EPS 同比
  df['ann_eps_yoy'] = df['ann_eps'] / df['ann_eps'].shift(1).replace(0, np.nan) - 1
  # 连续增长年数: 同比>0 连续累加, 断一次归零
  grow = (df['ann_eps_yoy'] > 0).astype(int)
  df['eps_grow_streak'] = grow * (grow.groupby((grow != grow.shift()).cumsum()).cumcount() + 1)
  # 3 年 CAGR(仅当两端 EPS 均为正才有意义)
  base = df['ann_eps'].shift(3)
  ratio = df['ann_eps'] / base.replace(0, np.nan)
  df['eps_cagr_3y'] = np.where((df['ann_eps'] > 0) & (base > 0), ratio ** (1 / 3) - 1, np.nan)
  cols = ['symbol', 'avail_date', 'ann_eps', 'ann_eps_yoy', 'eps_grow_streak', 'eps_cagr_3y']
  return df[cols].to_dict('records')


def _symbol_shares(sym: str, raw: dict) -> list:
  """单个标的: outstandingShares.quarterly -> 每季一条(净股票发行 NSI 的基础序列)."""
  q = (raw.get('outstandingShares') or {}).get('quarterly') or {}
  rows = []
  for k, r in q.items():
    if not isinstance(r, dict):
      continue
    sh = _num(r.get('shares'))
    per = _ts(r.get('dateFormatted') or r.get('date') or k)
    if np.isnan(sh) or sh <= 0 or pd.isna(per):
      continue
    rows.append({'symbol': sym, 'avail_date': per + pd.Timedelta(days=SHARE_FALLBACK_LAG_DAYS),
                 '_period': per, 'os_shares': sh})
  if not rows:
    return []
  df = pd.DataFrame(rows).sort_values('_period').reset_index(drop=True)
  # 净股票发行(Net Share Issuance): 股数同比/三年变化; 发行(>0)负向预测收益(经典异象)
  df['nsi_1y'] = df['os_shares'] / df['os_shares'].shift(4).replace(0, np.nan) - 1
  df['nsi_3y'] = df['os_shares'] / df['os_shares'].shift(12).replace(0, np.nan) - 1
  return df[['symbol', 'avail_date', 'nsi_1y', 'nsi_3y']].to_dict('records')


_EXTRA_PARSERS = {'annual': _symbol_annual, 'shares': _symbol_shares}


def build_extra(force: bool = False) -> dict:
  """
  扫描 fundamentals JSON -> {'annual': 长表, 'shares': 长表}(带磁盘缓存 + 签名校验).

  :param force: True 强制重建
  :returns: {'annual': DataFrame, 'shares': DataFrame, '__sig': tuple}
  """
  global _EXTRA_CACHE
  if _EXTRA_CACHE is not None and not force:
    return _EXTRA_CACHE
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  sig = _signature()
  if not force and EXTRA_PKL.exists():
    try:
      cached = pd.read_pickle(EXTRA_PKL)
      if isinstance(cached, dict) and cached.get('__sig') == sig:
        _EXTRA_CACHE = cached
        return cached
    except Exception:
      pass

  recs = {k: [] for k in _EXTRA_PARSERS}
  for p in sorted(FUND_JSON_DIR.glob('*.json')):
    try:
      raw = json.load(open(p, encoding='utf-8'))
    except Exception:
      continue
    for kind, parser in _EXTRA_PARSERS.items():
      recs[kind].extend(parser(p.stem, raw))

  out = {}
  for kind, rs in recs.items():
    df = pd.DataFrame(rs)
    if not df.empty:
      df = (df.sort_values(['symbol', 'avail_date'])
              .drop_duplicates(['symbol', 'avail_date'], keep='last')
              .reset_index(drop=True))
    out[kind] = df
  out['__sig'] = sig
  pd.to_pickle(out, EXTRA_PKL)
  _EXTRA_CACHE = out
  return out


def _extra_wide(kind: str, col: str) -> pd.DataFrame:
  """extra 长表某列 -> (avail_date × symbol) 宽表(带内存缓存)."""
  key = (kind, col)
  if key in _EXTRA_WIDE:
    return _EXTRA_WIDE[key]
  d = build_extra()
  w = (d[kind].pivot_table(index='avail_date', columns='symbol', values=col,
                           aggfunc='last', dropna=False)
         .sort_index())
  _EXTRA_WIDE[key] = w
  return w


def _xwide(ds: Dataset, kind: str, col: str) -> pd.DataFrame:
  """extra 某列对齐到 ds 交易日网格(每交易日取 <= t 最近一次的值, ffill)."""
  w = _extra_wide(kind, col)
  idx = w.index.union(ds.dates)
  return w.reindex(idx).ffill().reindex(ds.dates).reindex(columns=ds.symbols)


def _mcap(ds: Dataset) -> pd.DataFrame:
  """PIT 市值 = 当日收盘价 × PIT 股数."""
  return ds.wide('Close') * _fwide(ds, 'shares')


def _ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
  """比值; 分母为 0 / NaN 处 -> NaN(保留, 不置零)."""
  return num / den.replace(0, np.nan)


# ============================ 注册因子 ============================
# 命名前缀 F_(fundamental). group 取 size/value/quality/growth.

# ---- size ----
@_register('F_logmcap', 'size', 'log(close[t] × PIT shares[t])',
               window=1, direction=-1)
def _f_logmcap(ds: Dataset) -> pd.DataFrame:
  m = _mcap(ds)
  return np.log(m.where(m > 0))


# ---- value ----
@_register('F_ep', 'value', 'TTM net income[t] / mcap[t]', window=1)
def _f_ep(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_net_income'), _mcap(ds))


@_register('F_bp', 'value', 'book equity[t] / mcap[t]', window=1)
def _f_bp(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'equity'), _mcap(ds))


@_register('F_sp', 'value', 'TTM revenue[t] / mcap[t]', window=1)
def _f_sp(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_revenue'), _mcap(ds))


@_register('F_fcf_yield', 'value', 'TTM free cash flow[t] / mcap[t]', window=1)
def _f_fcf(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_fcf'), _mcap(ds))


@_register('F_ebitda_yield', 'value', 'TTM EBITDA[t] / mcap[t]', window=1)
def _f_ebitda(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_ebitda'), _mcap(ds))


# ---- quality ----
@_register('F_roe', 'quality', 'TTM net income[t] / book equity[t]', window=1)
def _f_roe(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_net_income'), _fwide(ds, 'equity'))


@_register('F_roa', 'quality', 'TTM net income[t] / total assets[t]', window=1)
def _f_roa(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_net_income'), _fwide(ds, 'total_assets'))


@_register('F_gross_margin', 'quality', 'TTM gross profit[t] / TTM revenue[t]', window=1)
def _f_gm(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_gross_profit'), _fwide(ds, 'ttm_revenue'))


@_register('F_op_margin', 'quality', 'TTM operating income[t] / TTM revenue[t]', window=1)
def _f_om(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'ttm_op_income'), _fwide(ds, 'ttm_revenue'))


@_register('F_debt2equity', 'quality', 'total liabilities[t] / book equity[t]',
               window=1, direction=-1)
def _f_d2e(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'total_liab'), _fwide(ds, 'equity'))


@_register('F_current_ratio', 'quality', 'current assets[t] / current liabilities[t]', window=1)
def _f_cr(ds: Dataset) -> pd.DataFrame:
  return _ratio(_fwide(ds, 'current_assets'), _fwide(ds, 'current_liab'))


# ---- growth ----
@_register('F_rev_yoy', 'growth', 'TTM revenue[t] / TTM revenue[t-4q] - 1', window=1)
def _f_rev_yoy(ds: Dataset) -> pd.DataFrame:
  return _fwide(ds, 'rev_yoy')


@_register('F_eps_yoy', 'growth', 'TTM EPS[t] / TTM EPS[t-4q] - 1', window=1)
def _f_eps_yoy(ds: Dataset) -> pd.DataFrame:
  return _fwide(ds, 'eps_yoy')


# ---- surprise(预期差) ----
@_register('F_sue_eps', 'surprise',
               'SUE: (EPS_q[t] - EPS_q[t-4q]) / std(季节意外, 8q)', window=1)
def _f_sue_eps(ds: Dataset) -> pd.DataFrame:
  return _fwide(ds, 'sue_eps')


@_register('F_sue_rev', 'surprise',
               'SUE: (revenue_q[t] - revenue_q[t-4q]) / std(季节意外, 8q)', window=1)
def _f_sue_rev(ds: Dataset) -> pd.DataFrame:
  return _fwide(ds, 'sue_rev')


# ---- surprise(分析师预期差, 来自 Earnings.History) ----
@_register('F_eps_surp', 'surprise',
               'analyst EPS surprise: (actual - consensus) / |consensus|', window=1)
def _f_eps_surp(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'eps_surp')


@_register('F_sue_analyst', 'surprise',
               'SUE(analyst): (actual-consensus) / std(意外, 8q)', window=1)
def _f_sue_analyst(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'sue_analyst')


@_register('F_eps_surp_4q', 'surprise',
               'mean analyst EPS surprise over last 4 quarters', window=1)
def _f_eps_surp_4q(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'eps_surp_4q')


# ---- surprise(Earnings.History 扩展: 超预期持续性/加速度) ----
@_register('F_beat_streak', 'surprise',
               'consecutive quarters with positive analyst EPS surprise', window=1)
def _f_beat_streak(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'beat_streak')


@_register('F_surp_accel', 'surprise',
               'analyst EPS surprise acceleration: surp[t] - surp[t-1q]', window=1)
def _f_surp_accel(ds: Dataset) -> pd.DataFrame:
  return _ewide(ds, 'surp_accel')


# ---- growth(Earnings.Annual: 年度 EPS 趋势) ----
@_register('F_ann_eps_yoy', 'growth', 'FY EPS[t] / FY EPS[t-1y] - 1', window=1)
def _f_ann_eps_yoy(ds: Dataset) -> pd.DataFrame:
  return _xwide(ds, 'annual', 'ann_eps_yoy')


@_register('F_eps_grow_streak', 'growth',
               'consecutive years of positive annual EPS growth', window=1)
def _f_eps_grow_streak(ds: Dataset) -> pd.DataFrame:
  return _xwide(ds, 'annual', 'eps_grow_streak')


@_register('F_eps_cagr_3y', 'growth', '3-year CAGR of annual EPS', window=1)
def _f_eps_cagr_3y(ds: Dataset) -> pd.DataFrame:
  return _xwide(ds, 'annual', 'eps_cagr_3y')


# ---- issuance(outstandingShares: 净股票发行异象) ----
@_register('F_net_issuance', 'issuance',
               'net share issuance: shares[t] / shares[t-4q] - 1', window=1, direction=-1)
def _f_net_issuance(ds: Dataset) -> pd.DataFrame:
  return _xwide(ds, 'shares', 'nsi_1y')


@_register('F_net_issuance_3y', 'issuance',
               'net share issuance (3y): shares[t] / shares[t-12q] - 1', window=1, direction=-1)
def _f_net_issuance_3y(ds: Dataset) -> pd.DataFrame:
  return _xwide(ds, 'shares', 'nsi_3y')


# ============================ 自检 ============================
def main():
  log = cfg.get_logger('factor.fundamentals')
  ev = build_events()
  log.info(f'[PIT 事件表] {len(ev):,} 行, {ev["symbol"].nunique()} 标的, '
           f'avail_date {ev["avail_date"].min():%Y-%m-%d} ~ {ev["avail_date"].max():%Y-%m-%d}')
  log.info('[列覆盖(非空占比)]')
  cov = (ev[EVENT_COLS].notna().mean()).round(3).sort_values(ascending=False)
  log.info('\n' + cov.to_string())
  for s in ['AAPL', 'MSFT', 'NVDA']:
    sub = ev[ev['symbol'] == s].tail(2)
    if not sub.empty:
      log.info(f'\n[{s}] 最近 2 期:\n' + sub[['symbol', 'avail_date'] + EVENT_COLS[:8]].to_string(index=False))


if __name__ == '__main__':
  main()