# bc_data_io 新增 EOD 数据方法单元测试（回归测试）
#
# 覆盖 bc_data_io.py 中 "#-- EOD New Data --#" 区段：
#   9 个私有助手 + 44 个公开 get_xxx_from_eod 方法
#
# 运行：
#   C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_eod_new_data.py
# 预期：每行 PASS，末尾 ALL 87 TESTS PASSED，退出码 0
#
# 注意：
# - 必须用 venv 的 python（系统 python 缺 pandas 依赖）
# - 全部 mock 底层 _get_json_from_eod / requests.get，不联网、不碰真实数据
# - 重点锁定 history 修复过的：filter 键/路径、intraday unix 秒、envelope 展平
import logging
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from quant import bc_data_io as io

# 静默 logger.warning / logger.exception，保持测试输出干净（断言只看返回值）
logging.disable(logging.CRITICAL)

results = []


def check(name, cond):
  results.append((name, bool(cond)))
  print(f'{name}: {"PASS" if cond else "FAIL"}')


# 通用 mock envelope：data 为 list（覆盖绝大多数端点）
ROWS = {'data': [{'date': '2024-01-01', 'v': 1}, {'date': '2024-01-02', 'v': 2}]}


def probe(call, envelope, expect_path, expect_params=None):
  """
  mock 底层 _get_json_from_eod，执行被测方法，返回 (输出, path 命中, params 命中).
  传入的 call 为 lambda（方法调用）.
  """
  with mock.patch.object(io, '_get_json_from_eod', return_value=envelope) as m:
    out = call()
  args, kwargs = m.call_args
  path = args[0] if args else kwargs.get('path')
  params = kwargs.get('params') or {}
  ok = (path == expect_path)
  if expect_params is not None:
    ok = ok and all(params.get(k) == v for k, v in expect_params.items())
  return out, ok, path, params


# ============================== 私有助手 ==============================
ev = io._envelope_to_rows
check('envelope: None -> []', ev(None) == [])
check('envelope: list 过滤非 dict', ev([{'a': 1}, 2, {'b': 2}]) == [{'a': 1}, {'b': 2}])
check('envelope: {data: list}', ev({'data': [{'a': 1}]}) == [{'a': 1}])
check('envelope: {data: dict-of-dicts} 取 value 为行', ev({'meta': {}, 'data': {'AAPL': {'p': 1}, 'MSFT': {'p': 2}}}) == [{'p': 1}, {'p': 2}])
check('envelope: 单键 section 取 list', ev({'type': 'x', 'earnings': [{'a': 1}]}) == [{'a': 1}])
check('envelope: 普通 dict 包成单行', ev({'a': 1}) == [{'a': 1}])
check('envelope: 标量 -> []', ev(123) == [])

check('unix: None -> None', io._to_unix_seconds(None) is None)
check('unix: int 原样', io._to_unix_seconds(123) == 123)
check('unix: 数字串转 int', io._to_unix_seconds('123') == 123)
check('unix: 日期串转秒', io._to_unix_seconds('2024-01-01') == int(pd.Timestamp('2024-01-01', tz='UTC').timestamp()))
check('unix: 非法串 -> None', io._to_unix_seconds('not-a-date') is None)

_flat = io._flatten_nested_rows([{'member': {'name': 'A'}, 'transaction': {'amount': 5}, 'id': 1}, 'bad', {'x': 2}])
check('flatten: 嵌套 dict 合并到顶层并跳过非 dict', _flat == [{'name': 'A', 'amount': 5, 'id': 1}, {'x': 2}])

_pdf = io._parse_date_columns(pd.DataFrame({'date': ['2024-01-01'], 'other': ['x']}))
check('datecols: 转换 date 列', str(_pdf['date'].dtype).startswith('datetime64'))
check('datecols: 不碰非日期列', _pdf['other'].iloc[0] == 'x')
check('datecols: 空 df 原样返回', len(io._parse_date_columns(pd.DataFrame())) == 0)
check('datecols: None 原样返回', io._parse_date_columns(None) is None)


# ============================== 底层 HTTP / 分页 ==============================
class _Resp:
  def __init__(self, status_code=200, payload=None, text=''):
    self.status_code = status_code
    self._payload = payload
    self.text = text

  def json(self):
    if self._payload is None:
      raise ValueError('no json')
    return self._payload


with mock.patch.object(io.requests, 'get', return_value=_Resp(200, {'ok': 1})) as rg:
  _r = io._get_json_from_eod('/x', api_key='KEY', params={'a': 1, 'b': None})
_args, _kw = rg.call_args
check('http: 200 返回 json', _r == {'ok': 1})
check('http: url 拼 eodhd 域名', _args[0] == 'https://eodhd.com/api/x')
check('http: query 含 api_token/fmt 且丢弃 None', _kw['params'] == {'api_token': 'KEY', 'fmt': 'json', 'a': 1})

with mock.patch.object(io.requests, 'get', return_value=_Resp(403, text='forbidden')):
  check('http: 403 -> None', io._get_json_from_eod('/x') is None)
with mock.patch.object(io.requests, 'get', return_value=_Resp(404, text='not found')):
  check('http: 404 -> None', io._get_json_from_eod('/x') is None)
with mock.patch.object(io.requests, 'get', return_value=_Resp(500, text='err')):
  check('http: 其它状态码 -> None', io._get_json_from_eod('/x') is None)
with mock.patch.object(io.requests, 'get', side_effect=RuntimeError('net down')):
  check('http: 请求异常 -> None 不抛', io._get_json_from_eod('/x') is None)

with mock.patch.object(io, '_get_json_from_eod', return_value={'data': [{'a': 1}, 'str']}):
  check('rows: envelope 展平并过滤非 dict', io._get_rows_from_eod('/p') == [{'a': 1}])

# 多页累积 + page[offset]/page[limit]
_calls = []
_pages = [[{'n': i} for i in range(100)], [{'n': 100}]]


def _fake_rows(path, api_key=None, params=None, timeout=60, is_print=False):
  i = len(_calls)
  _calls.append(dict(params or {}))
  return _pages[i] if i < len(_pages) else []


with mock.patch.object(io, '_get_rows_from_eod', side_effect=_fake_rows):
  _rows = io._get_paged_rows_from_eod('/p', page_size=100, max_pages=10)
check('paged: 多页累积', len(_rows) == 101)
check('paged: 逐页递增 page[offset]/[limit]',
      _calls[0].get('page[offset]') == 0 and _calls[0].get('page[limit]') == 100 and _calls[1].get('page[offset]') == 100)

# 服务端忽略分页（相邻页相同）提前停止
_calls2 = []


def _fake_same(path, api_key=None, params=None, timeout=60, is_print=False):
  _calls2.append(1)
  return [{'x': 1}]


with mock.patch.object(io, '_get_rows_from_eod', side_effect=_fake_same):
  _rows2 = io._get_paged_rows_from_eod('/p', page_size=1, max_pages=10)
check('paged: 忽略分页时提前停止', _rows2 == [{'x': 1}] and len(_calls2) == 2)

with mock.patch.object(io, '_get_rows_from_eod', return_value=[]):
  check('paged: 首页为空直接返回', io._get_paged_rows_from_eod('/p') == [])

# ust 私有助手：filter 键 + 日期解析
with mock.patch.object(io, '_get_paged_rows_from_eod', return_value=[{'date': '2024-01-01', 'tenor': '1M'}]) as m:
  _udf = io._get_ust_rates_from_eod('yield-rates', start_date='2024-01-01', end_date='2024-01-05', year=2024)
_ukw = m.call_args.kwargs
check('ust: filter 键(year/date_from/date_to)',
      _ukw['params'] == {'filter[year]': 2024, 'filter[date_from]': '2024-01-01', 'filter[date_to]': '2024-01-05'})
check('ust: 日期列转 datetime', str(_udf['date'].dtype).startswith('datetime64'))
check('ust: path=/ust/{dataset}', m.call_args.args[0] == '/ust/yield-rates')

# calendar 私有助手：from/to/symbols 拼接
with mock.patch.object(io, '_get_rows_from_eod', return_value=[{'date': '2024-01-01', 'code': 'AAPL.US'}]) as m:
  io._get_calendar_from_eod('earnings', start_date='2024-01-01', end_date='2024-01-05', symbols=['AAPL', 'MSFT'])
check('calendar: path=/calendar/earnings', m.call_args.args[0] == '/calendar/earnings')
check('calendar: from/to/symbols 逗号拼接',
      m.call_args.kwargs['params'] == {'from': '2024-01-01', 'to': '2024-01-05', 'symbols': 'AAPL,MSFT'})


# ============================== Universe / Bulk ==============================
out, ok, _, _ = probe(lambda: io.get_exchanges_list_from_eod(), ROWS, '/exchanges-list')
check('universe: exchanges_list path+返回', ok and len(out) == 2)

out, ok, _, p = probe(lambda: io.get_exchange_symbol_list_from_eod('US', delisted=True, type='etf', symbols=['AAPL', 'MSFT']),
                      ROWS, '/exchange-symbol-list/US',
                      {'delisted': 1, 'type': 'etf', 'symbols': 'AAPL,MSFT'})
check('universe: exchange_symbol_list path+参数', ok)

out, ok, _, p = probe(lambda: io.get_bulk_fundamentals_from_eod('US', offset=0, limit=100),
                      {'data': {'AAPL': {'mktcap': 1}}}, '/bulk-fundamentals/US', {'offset': 0, 'limit': 100})
check('universe: bulk_fundamentals 取 data dict', ok and out == {'AAPL': {'mktcap': 1}})


# ============================== News / Sentiment ==============================
out, ok, _, _ = probe(lambda: io.get_news_from_eod(symbols=['AAPL'], tag='x', start_date='2024-01-01',
                                                   end_date='2024-01-05', limit=10, offset=5),
                      ROWS, '/news',
                      {'s': 'AAPL', 't': 'x', 'from': '2024-01-01', 'to': '2024-01-05', 'limit': 10, 'offset': 5})
check('news: path /news + s/t/from/to/limit/offset', ok and len(out) == 2)

out, ok, _, _ = probe(lambda: io.get_sentiments_from_eod(symbols='AAPL', start_date='2024-01-01', end_date='2024-01-02'),
                      {'AAPL': [{'date': '2024-01-01', 'count': 3, 'normalized': 0.5}]},
                      '/sentiments', {'s': 'AAPL', 'from': '2024-01-01', 'to': '2024-01-02'})
check('sentiments: symbol 列 + date 解析', ok and 'symbol' in out.columns and out['symbol'].iloc[0] == 'AAPL'
      and str(out['date'].dtype).startswith('datetime64'))

out, ok, _, _ = probe(lambda: io.get_news_word_weights_from_eod('AAPL', start_date='2024-01-01', end_date='2024-01-02'),
                      {'AAPL': {'2024-01-01': {'good': 1.0}}},
                      '/news-word-weights', {'s': 'AAPL', 'filter[date_from]': '2024-01-01', 'filter[date_to]': '2024-01-02'})
check('news-word-weights: filter[date_from/to] + 返回 dict', ok and out == {'AAPL': {'2024-01-01': {'good': 1.0}}})


# ============================== Smart money / Filings ==============================
out, ok, _, _ = probe(lambda: io.get_insider_transactions_from_eod(code='US', start_date='2024-01-01',
                                                                   end_date='2024-01-05', limit=1000),
                      ROWS, '/insider-transactions',
                      {'code': 'US', 'limit': 1000, 'from': '2024-01-01', 'to': '2024-01-05'})
check('insider: path + code/limit/from/to', ok)

out, ok, _, _ = probe(lambda: io.get_congressional_trades_from_eod(page_size=100),
                      {'data': [{'member': {'name': 'A'}, 'asset': {'ticker': 'AAPL'}, 'disclosure_date': '2024-01-01'}]},
                      '/congressional-trades')
check('congressional: 嵌套展平 + 日期解析', ok and 'name' in out.columns and 'ticker' in out.columns
      and str(out['disclosure_date'].dtype).startswith('datetime64'))

out, ok, _, _ = probe(lambda: io.get_sec_filings_from_eod('AAPL', form='10k'),
                      ROWS, '/sec-filings/AAPL.US/10k', {'limit': 20, 'offset': 0})
check('sec_filings: 指定 form 走 path 明细', ok)

out, ok, _, _ = probe(lambda: io.get_sec_filings_from_eod('AAPL'),
                      {'data': {'ticker': 'AAPL', 'name': 'Apple', 'cik': '123',
                                'filings': {'10-K': {'count': 10, 'latest': '2024-01-01', 'url': 'u'}}}},
                      '/sec-filings/AAPL.US')
check('sec_filings: overview 解析为汇总表', ok and list(out.columns) == ['ticker', 'name', 'cik', 'form', 'count', 'latest', 'url']
      and out['form'].iloc[0] == '10-K')


# ============================== Calendar / Macro ==============================
out, ok, _, _ = probe(lambda: io.get_earnings_calendar_from_eod(start_date='2024-01-01', end_date='2024-01-05', symbols=['AAPL']),
                      ROWS, '/calendar/earnings', {'from': '2024-01-01', 'to': '2024-01-05', 'symbols': 'AAPL'})
check('earnings_calendar: path + from/to/symbols', ok)

out, ok, _, _ = probe(lambda: io.get_earnings_trends_from_eod(['AAPL']),
                      {'type': 'x', 'trends': [{'a': 1}]}, '/calendar/trends', {'symbols': 'AAPL'})
check('earnings_trends: 返回 dict', ok and out.get('type') == 'x')

out, ok, _, _ = probe(lambda: io.get_upcoming_dividends_from_eod(symbol='AAPL', date_eq='2024-01-01'),
                      ROWS, '/calendar/dividends', {'filter[symbol]': 'AAPL.US', 'filter[date_eq]': '2024-01-01'})
check('upcoming_dividends: filter[symbol]/[date_eq]', ok)

out, ok, _, _ = probe(lambda: io.get_upcoming_splits_from_eod(start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/calendar/splits', {'from': '2024-01-01', 'to': '2024-01-05'})
check('upcoming_splits: path /calendar/splits', ok)

out, ok, _, _ = probe(lambda: io.get_upcoming_ipos_from_eod(start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/calendar/ipos', {'from': '2024-01-01', 'to': '2024-01-05'})
check('upcoming_ipos: path /calendar/ipos', ok)

out, ok, _, _ = probe(lambda: io.get_macro_indicator_from_eod('USA', indicator='gdp_current_usd'),
                      ROWS, '/macro-indicator/USA', {'indicator': 'gdp_current_usd'})
check('macro: path /macro-indicator/USA + indicator', ok)


# ============================== Rates ==============================
for _name, _fn, _ds in [('yield', io.get_ust_yield_rates_from_eod, 'yield-rates'),
                        ('bill', io.get_ust_bill_rates_from_eod, 'bill-rates'),
                        ('long_term', io.get_ust_long_term_rates_from_eod, 'long-term-rates'),
                        ('real_yield', io.get_ust_real_yield_rates_from_eod, 'real-yield-rates')]:
  out, ok, _, _ = probe(lambda f=_fn: f(start_date='2024-01-01', end_date='2024-01-05', year=2024),
                        ROWS, f'/ust/{_ds}',
                        {'filter[year]': 2024, 'filter[date_from]': '2024-01-01', 'filter[date_to]': '2024-01-05'})
  check(f'rates: ust {_name} path+filter', ok)

out, ok, _, _ = probe(lambda: io.get_reference_rates_from_eod(code='USD', currency='USD',
                                                             start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/rates/reference-rates',
                      {'filter[code]': 'USD', 'filter[currency]': 'USD', 'filter[from]': '2024-01-01', 'filter[to]': '2024-01-05'})
check('rates: reference_rates filter[code/currency/from/to]', ok)

out, ok, _, _ = probe(lambda: io.get_policy_rates_from_eod(code='X', country='US', central_bank='FED',
                                                           start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/rates/policy-rates',
                      {'filter[code]': 'X', 'filter[country]': 'US', 'filter[central_bank]': 'FED',
                       'filter[from]': '2024-01-01', 'filter[to]': '2024-01-05'})
check('rates: policy_rates filter[code/country/central_bank/from/to]', ok)

out, ok, _, _ = probe(lambda: io.get_funding_stress_from_eod(), ROWS, '/spreads/funding-stress')
check('rates: funding_stress path', ok and len(out) == 2)


# ============================== Credit risk ==============================
out, ok, _, _ = probe(lambda: io.get_credit_sovereign_risk_premium_from_eod(country='USA', region='x', as_of='2024-01-01'),
                      ROWS, '/credit-risk/sovereign/risk-premium',
                      {'filter[country]': 'USA', 'filter[region]': 'x', 'filter[as_of]': '2024-01-01'})
check('credit: sovereign risk_premium filter[country/region/as_of]', ok)

for _name, _fn, _p in [('credit_ratings', io.get_credit_sovereign_credit_ratings_from_eod, '/credit-risk/sovereign/credit-ratings'),
                       ('cds_spreads', io.get_credit_sovereign_cds_spreads_from_eod, '/credit-risk/sovereign/cds-spreads'),
                       ('default_spreads', io.get_credit_sovereign_default_spreads_from_eod, '/credit-risk/sovereign/default-spreads')]:
  out, ok, _, _ = probe(_fn, ROWS, _p)
  check(f'credit: sovereign {_name} path', ok)

out, ok, _, _ = probe(lambda: io.get_credit_corporate_cmdi_from_eod(start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/credit-risk/corporate/cmdi', {'filter[from]': '2024-01-01', 'filter[to]': '2024-01-05'})
check('credit: cmdi filter[from/to]', ok)

out, ok, _, _ = probe(lambda: io.get_credit_corporate_hqm_yields_from_eod(start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/credit-risk/corporate/hqm-yields', {'filter[from]': '2024-01-01', 'filter[to]': '2024-01-05'})
check('credit: hqm_yields filter[from/to]', ok)

out, ok, _, _ = probe(lambda: io.get_credit_cds_market_aggregates_from_eod(metric='gross_notional', dimension='grade',
                                                                           start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/credit-risk/cds-market/aggregates',
                      {'filter[metric]': 'gross_notional', 'filter[dimension]': 'grade',
                       'filter[from]': '2024-01-01', 'filter[to]': '2024-01-05'})
check('credit: cds_market_aggregates 修正后的 path+filter', ok)


# ============================== Events / Index / Aux ==============================
out, ok, _, _ = probe(lambda: io.get_economic_events_from_eod(country='US', comparison='yoy', event_type='x',
                                                              start_date='2024-01-01', end_date='2024-01-05',
                                                              limit=5, offset=0),
                      [{'date': '2024-01-01', 'event': 'x'}], '/economic-events',
                      {'country': 'US', 'comparison': 'yoy', 'type': 'x', 'from': '2024-01-01',
                       'to': '2024-01-05', 'limit': 5, 'offset': 0})
check('events: 用 plain limit/offset + country/comparison/type', ok and len(out) == 1)

out, ok, _, _ = probe(lambda: io.get_cboe_indices_list_from_eod(), ROWS, '/cboe/indices')
check('index: cboe indices_list path', ok)

out, ok, _, _ = probe(lambda: io.get_cboe_index_data_from_eod('BDE30P', 'snapshot_official_closing', '2024-01-01'),
                      ROWS, '/cboe/index',
                      {'filter[index_code]': 'BDE30P', 'filter[feed_type]': 'snapshot_official_closing', 'filter[date]': '2024-01-01'})
check('index: cboe index_data 三 filter', ok)

out, ok, _, _ = probe(lambda: io.get_search_from_eod('apple', limit=5), ROWS, '/search/apple', {'limit': 5})
check('aux: search path + limit', ok)

out, ok, _, _ = probe(lambda: io.get_symbol_change_history_from_eod(start_date='2024-01-01', end_date='2024-01-05'),
                      ROWS, '/symbol-change-history', {'from': '2024-01-01', 'to': '2024-01-05'})
check('aux: symbol_change_history 全局 feed path+from/to', ok)


# ============================== Sanctions ==============================
out, ok, _, _ = probe(lambda: io.get_sanctions_entities_from_eod(source='OFAC', type='entity', program='X',
                                                                country='RU', q='abc', active='true'),
                      ROWS, '/sanctions/entities',
                      {'source': 'OFAC', 'type': 'entity', 'program': 'X', 'country': 'RU', 'q': 'abc', 'active': 'true'})
check('sanctions: entities path + 各过滤键', ok)

out, ok, _, _ = probe(lambda: io.get_sanctions_vessels_from_eod(), ROWS, '/sanctions/vessels')
check('sanctions: vessels path', ok)

out, ok, _, _ = probe(lambda: io.get_sanctions_programs_from_eod(), ROWS, '/sanctions/programs')
check('sanctions: programs path', ok)

out, ok, _, _ = probe(lambda: io.get_sanctions_sources_from_eod(), ROWS, '/sanctions/sources')
check('sanctions: sources path', ok)


# ============================== Real estate ==============================
out, ok, _, _ = probe(lambda: io.get_real_estate_countries_from_eod(), ROWS, '/real-estate/countries')
check('realestate: countries path', ok)

out, ok, _, _ = probe(lambda: io.get_real_estate_from_eod(code='US', type='x', metric='y', start_period='2020', end_period='2021'),
                      ROWS, '/real-estate/US', {'type': 'x', 'metric': 'y', 'from_period': '2020', 'to_period': '2021'})
check('realestate: prices path + 过滤键', ok)

out, ok, _, _ = probe(lambda: io.get_real_estate_detailed_from_eod(code='US'), ROWS, '/real-estate/US/detailed')
check('realestate: detailed path', ok)

out, ok, _, _ = probe(lambda: io.get_real_estate_detailed_series_from_eod(code='US'),
                      ROWS, '/real-estate/US/detailed/series')
check('realestate: detailed/series path', ok)


# ============================== Intraday / Delayed ==============================
_intraday_env = {'data': [{'datetime': '2024-01-01 09:30:00', 'open': 1.0, 'high': 2.0, 'low': 0.5,
                           'close': 1.5, 'volume': 100}]}
out, ok, _, p = probe(lambda: io.get_intraday_data_from_eod('AAPL', interval='5m', start_date='2024-01-01',
                                                            end_date='2024-01-02', split_dt=True),
                      _intraday_env, '/intraday/AAPL.US',
                      {'interval': '5m', 'from': int(pd.Timestamp('2024-01-01', tz='UTC').timestamp()),
                       'to': int(pd.Timestamp('2024-01-02', tz='UTC').timestamp()), 'split-dt': 1})
check('intraday: 日期串转 Unix 秒 + split-dt', ok)
check('intraday: 列重命名 + datetime 索引',
      'Open' in out.columns and 'Volume' in out.columns and isinstance(out.index, pd.DatetimeIndex))

out, ok, _, _ = probe(lambda: io.get_us_quote_delayed_from_eod(['AAPL', 'MSFT']),
                      {'data': {'AAPL': {'close': 1}, 'MSFT': {'close': 2}}}, '/us-quote-delayed', {'s': 'AAPL,MSFT'})
check('usquote: data dict-of-dicts 展平为两行', ok and len(out) == 2)
check('usquote: symbols 为空直接返回空 df', len(io.get_us_quote_delayed_from_eod([])) == 0)


# ============================== 错误路径（不抛、返回空） ==============================
with mock.patch.object(io, '_get_json_from_eod', return_value=None):
  check('robust: None 时 news 返回空 df', len(io.get_news_from_eod(symbols='AAPL')) == 0)
  check('robust: None 时 bulk_fundamentals 返回 {}', io.get_bulk_fundamentals_from_eod() == {})
  check('robust: None 时 sentiments 返回空 df', len(io.get_sentiments_from_eod(symbols='AAPL')) == 0)
  check('robust: None 时 news_word_weights 返回 {}', io.get_news_word_weights_from_eod('AAPL') == {})
  check('robust: None 时 earnings_trends 返回 {}', io.get_earnings_trends_from_eod(['AAPL']) == {})
  check('robust: None 时 reference_rates 返回空 df', len(io.get_reference_rates_from_eod()) == 0)


print()
failed = [n for n, ok in results if not ok]
if failed:
  print(f'FAILED: {failed}')
  sys.exit(1)
print(f'ALL {len(results)} TESTS PASSED')