# quant 回归测试: 增量拼接 / 原子写 / 复权保护 / 窗口单一真源
#
# 覆盖 2026-09 修复的数据正确性缺陷（对应 CODE_REVIEW_2026-09.md 附录 B4 / B9 / B13）：
#   1. 增量拼接一致性校验  _check_overlap_consistency  —— 复权基准漂移的探针
#   2. 分红/拆股真值保护    _merge_dividend_split      —— 占位 0.0/1.0 不得覆盖已知真值
#   3. CSV 原子写          save_stock_data           —— 不留 .tmp、失败不破坏原文件
#   4. 下载失败不再伪装成功 download_single_symbol_eod —— get_data 返回 None 必须是 failed
#   5. 安全判空            _is_missing               —— np.isnan(字符串) 不再让告警链路崩
#   6. 下单成功判定        bc_trader.trade_ok        —— 失败详情含 'SUCCEED' 不得误判成功
#   7. 交易所后缀判定      add_postfix_for_cn_symbol  —— 沪 ETF/科创板/北交所不再误判 .SHE
#   8. 窗口单一真源        factor.config.resolve_windows —— 该池 config 的 IS/OOS 优先于全局默认
#   9. JSON 合法性         _jsonable/_save_json_file   —— NaN/inf 不得写成裸字面量(非法 JSON)
#
# 文件名保留 ..._incremental.py 是历史原因(最初只覆盖 bc_data_io 增量路径), 现已扩到
# 跨模块回归; 若你更在意命名, 可自行重命名为 test_quant_regression.py(不影响运行).
#
# 运行：
#   C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_incremental.py
# 预期：每行 PASS（trade_ok 组在缺 futu/tigeropen 时打印一行 [SKIP]），
#       末尾 ALL 36 TESTS PASSED（含 trade_ok 组时为 ALL 40 TESTS PASSED），退出码 0
#
# 注意：
# - 必须用 venv 的 python（系统 python 缺 pandas/akshare 依赖）
# - 所有用例 mock 外部依赖（下载），文件读写全部在临时目录，不联网、不碰真实数据文件
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from quant import bc_data_io as io

results = []


def check(name, cond):
  results.append((name, bool(cond)))
  print(f'{name}: {"PASS" if cond else "FAIL"}')


def _daily(dates, close, dividend=0.0, split=1.0):
  """构造一个符合标准 schema 的宽表(DatetimeIndex + STANDARD_DATA_COLUMNS)。"""
  idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
  n = len(idx)
  df = pd.DataFrame({
    'Open': close, 'High': close, 'Low': close, 'Close': close,
    'Adj Close': close, 'Volume': [1000] * n,
    'Dividend': [dividend] * n, 'Split': [split] * n,
  }, index=idx)
  return df


# ============================== 1. 增量拼接一致性校验 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)

  # 1) 重叠日价格一致 -> 无告警, 返回 True
  old = _daily(['2026-01-02', '2026-01-05'], [10.0, 11.0])
  same = _daily(['2026-01-05', '2026-01-06'], [11.0, 12.0])
  with mock.patch.object(io.logger, 'error') as m_err:
    ok = io._check_overlap_consistency(old, same, 'AAA')
  check('拼接: 重叠价格一致时返回 True 且不报错', ok is True and not m_err.called)

  # 2) 重叠日价格不一致(模拟 qfq 基准漂移) -> 返回 False 且 error 点名 symbol
  drifted = _daily(['2026-01-05', '2026-01-06'], [9.35, 12.0])   # 11.0 -> 9.35 (除权后整体平移)
  with mock.patch.object(io.logger, 'error') as m_err:
    ok = io._check_overlap_consistency(old, drifted, 'AAA')
  check('拼接: 复权基准漂移被检出(返回 False)', ok is False)
  check('拼接: 漂移时 error 日志点名 symbol',
        m_err.called and 'AAA' in str(m_err.call_args))

  # 3) 无重叠 -> 无从比较, 返回 True(不应误报)
  disjoint = _daily(['2026-02-02'], [12.0])
  check('拼接: 无重叠日时返回 True(不误报)', io._check_overlap_consistency(old, disjoint, 'AAA') is True)

  # 4) 空/None 输入不得抛异常
  check('拼接: None / 空表输入安全返回 True',
        io._check_overlap_consistency(None, same, 'AAA') is True
        and io._check_overlap_consistency(old, None, 'AAA') is True
        and io._check_overlap_consistency(pd.DataFrame(), same, 'AAA') is True)


# ============================== 2. 分红/拆股真值保护 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)

  # 1) 新窗口是"未知占位"(Dividend 0.0 / Split 1.0), 老数据已知 0.5 / 2.0 -> 必须保留老值
  old = _daily(['2026-01-05'], [11.0], dividend=0.5, split=2.0)
  new_placeholder = _daily(['2026-01-05', '2026-01-06'], [11.0, 12.0], dividend=0.0, split=1.0)
  merged = io._merge_dividend_split(old, new_placeholder)
  check('D/S: 占位不覆盖已知 Dividend', float(merged.loc['2026-01-05', 'Dividend']) == 0.5)
  check('D/S: 占位不覆盖已知 Split', float(merged.loc['2026-01-05', 'Split']) == 2.0)
  check('D/S: 未重叠的新行保持原样', float(merged.loc['2026-01-06', 'Dividend']) == 0.0)

  # 2) 新窗口带真实值 -> 以新值为准(不能被"保留老值"逻辑挡住)
  old2 = _daily(['2026-01-05'], [11.0], dividend=0.5, split=2.0)
  new_real = _daily(['2026-01-05'], [11.0], dividend=0.7, split=3.0)
  merged2 = io._merge_dividend_split(old2, new_real)
  check('D/S: 新值非占位时以新值为准',
        float(merged2.loc['2026-01-05', 'Dividend']) == 0.7
        and float(merged2.loc['2026-01-05', 'Split']) == 3.0)

  # 3) 老数据为空 -> 原样返回, 不得抛异常
  check('D/S: 老数据为空时安全返回', len(io._merge_dividend_split(pd.DataFrame(), new_placeholder)) == 2)

  # 4) 不修改传入的 new(避免调用方被就地改动)
  before = new_placeholder.loc['2026-01-05', 'Dividend']
  io._merge_dividend_split(old, new_placeholder)
  check('D/S: 不就地修改传入的 new', new_placeholder.loc['2026-01-05', 'Dividend'] == before)


# ============================== 3. CSV 原子写 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)
  df = _daily(['2026-01-05', '2026-01-06'], [11.0, 12.0])

  io.save_stock_data(df=df, file_path=tmp, file_name='AAA')
  target = tmp / 'AAA.csv'
  check('原子写: 正常落盘', target.exists())

  io.save_stock_data(df=df, file_path=tmp, file_name='AAA')
  check('原子写: 无 .tmp 残留', not (tmp / 'AAA.csv.tmp').exists())
  reloaded = pd.read_csv(target, encoding='utf8')
  check('原子写: 内容可正常解析(行数正确)', len(reloaded) == 2)

  # 覆盖已有文件失败时, 原文件必须保持完整(to_csv 抛错 -> 清理 tmp 并 re-raise)
  good = target.read_text(encoding='utf8')
  with mock.patch.object(pd.DataFrame, 'to_csv', side_effect=OSError('disk full')):
    try:
      io.save_stock_data(df=df, file_path=tmp, file_name='AAA')
      raised = False
    except OSError:
      raised = True
  check('原子写: 写失败时向上抛异常', raised)
  check('原子写: 写失败后原文件未被破坏', target.read_text(encoding='utf8') == good)
  check('原子写: 写失败后不留 .tmp', not (tmp / 'AAA.csv.tmp').exists())


# ============================== 4. 下载失败不得伪装成功 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)
  # 先放一份本地数据, 让函数走"增量拼接"分支
  io.save_stock_data(df=_daily(['2026-01-05'], [11.0]), file_path=tmp, file_name='AAA')

  # args 顺序必须与 download_single_symbol_eod 的解包一致:
  # (symbol, market, start_date, required_date, benchmark_date, update_mode, source,
  #  api_key, stock_data_path, file_format, is_save, add_dividend, add_split, adjust)
  args = ('AAA', 'us', None, None, '2026-01-06', 'eod', 'eod',
          None, tmp, '.csv', True, False, False, 'qfq')
  with mock.patch.object(io, 'preprocess_symbol', return_value={'AAA': 'AAA'}), \
       mock.patch.object(io, 'get_data', return_value=None), \
       mock.patch.object(io.time, 'sleep', return_value=None):
    symbol, query_symbol, data, status, messages = io.download_single_symbol_eod(args)
  check('下载: get_data 返回 None 时 status=failed(不再伪装 success)', status == 'failed')
  check('下载: 失败时给出可诊断的 messages', bool(messages))


# ============================== 4b. 安全判空 _is_missing ==============================
# 这条是为"邮件/组合更新里 np.isnan(字符串) 会崩"补的守卫
missing_cases = [None, float('nan'), float('inf'), '', '-', '--', 'nan', 'NONE', [], {}]
present_cases = [0.0, 1.5, -3, '1.23', ' 2.5 ']
check('判空: None/NaN/inf/占位字符串 均判为缺失',
      all(io._is_missing(v) for v in missing_cases))
check('判空: 0.0/正数/数字字符串 均判为有效(0.0 不是缺失)',
      not any(io._is_missing(v) for v in present_cases))


# ============================== 4c. trade_ok 成功判定 ==============================
# bc_trader 在模块顶层 import futu / tigeropen; 环境缺这两个 SDK 时跳过(而不是报假失败)
try:
  from quant.bc_trader import trade_ok
except Exception as e:                                          # pragma: no cover
  print(f'[SKIP] trade_ok 用例跳过(bc_trader 不可导入: {type(e).__name__}: {e})')
else:
  ok_cases = [
    '[BUY]: AAPL x 10 (market) | SUCCEED: 12345',
    '[SELL]: AAPL x 10 (NORMAL:1.5) | SUCCEED: abc',
  ]
  fail_cases = [
    None, '',
    '[BUY]: AAPL x 10 (market) | FAILED: Not affordable(0/10 @ 1.5)',
    '[BUY]: AAPL x 10 (market) | FAILED: Not enough stock to sell(0/10)',
    '[BUY]: AAPL x 10 (market) | FAILED: no response from server, order status unknown',
    '[BUY]: AAPL x 10 (market) | FAILED: position unknown (query failed), refuse to sell',
    # 关键: 失败详情里**含** 'SUCCEED' 字样时不得误判为成功
    '[BUY]: AAPL x 10 (market) | FAILED: broker said previous SUCCEED order was rejected',
  ]
  check('trade_ok: 成功串判为 True', all(trade_ok(s) for s in ok_cases))
  check('trade_ok: 失败串/非字符串/含 SUCCEED 的失败详情 均判为 False',
        not any(trade_ok(s) for s in fail_cases))
  # 状态段 = 最后一个 '|' 之后. 真实返回串里状态永远在末尾, 这里用真实形态断言.
  check('trade_ok: 真实成功串(状态在末尾)判 True',
        trade_ok('[BUY]: AAPL x 10 (market) | SUCCEED: 12345') is True)
  check('trade_ok: 真实失败串(状态在末尾)判 False',
        trade_ok('[BUY]: AAPL x 10 (market) | FAILED: Not affordable(0/10)') is False)
  # 无分隔符时整串就是状态段; 先判 FAILED 优先命中 -> 保守判 False
  check('trade_ok: 同时含两个标记且无分隔符时先判 FAILED -> False',
        trade_ok('SUCCEED: 1 FAILED: 2') is False)
  # 反向: 状态段里 SUCCEED 在前、FAILED 在后(真实失败详情形态)也必须判 False
  check('trade_ok: 失败详情含 SUCCEED 字样仍判 False',
        trade_ok('[X] | FAILED: broker returned SUCCEED-like text') is False)
  # 无任何标记(旧格式/异常) -> 保守判 False, 不能当成功
  check('trade_ok: 无任何标记时保守判 False', trade_ok('[X] | something else') is False)


# ============================== 5. 交易所后缀判定 ==============================
# 注意: add_postfix_for_cn_symbol 返回的是「完整代码 + 后缀」(如 '510300.SHG'), 不是只有后缀.
cases = {
  '600000': '600000.SHG',   # 沪市主板
  '688111': '688111.SHG',   # 科创板
  '510300': '510300.SHG',   # 沪市 ETF(原实现误判 .SHE)
  '588000': '588000.SHG',   # 科创 50 ETF(原实现误判 .SHE)
  '000001': '000001.SHE',   # 深市主板
  '300750': '300750.SHE',   # 创业板
  '159919': '159919.SHE',   # 深市 ETF
  '830799': '830799.BSE',   # 北交所(原实现误判 .SHE)
  '00700':  '00700.HK',     # 港股
}
bad = {s: io.add_postfix_for_cn_symbol(s) for s, want in cases.items()
       if io.add_postfix_for_cn_symbol(s) != want}
check(f'后缀: 9 个代码交易所判定正确(错误: {bad or "无"})', not bad)


# ============================== 6. 窗口单一真源 resolve_windows ==============================
# 目的: 锁住"该池 config 的 meta 窗口优先于全局默认"这一优先级.
# 用独立临时 config 文件, 不依赖仓库里的真实 factor_config.json, 因此结果稳定.
import json as _json
from quant.factor import config as fcfg

with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)
  cfg_path = tmp / 'factor_config.json'
  cfg_path.write_text(_json.dumps({
    'default_pool': 'poolA',
    'pools': {
      'poolA': {'meta': {'pool': 'poolA',
                         'is_window': ['2021-01-01', '2024-12-31'],
                         'oos_window': ['2025-01-01', '2025-12-31']},
                'schemes': {'x': {'f': 1.0}}},
      'broken': {'meta': {'pool': 'broken', 'is_window': ['2022-01-01']},
                 'schemes': {'x': {'f': 1.0}}},          # is_window 长度不合法 -> 应回退全局
    }
  }, ensure_ascii=False), encoding='utf-8')

  wa = fcfg.resolve_windows('poolA', path=cfg_path)
  check('窗口: 该池 config 的 IS 生效(优先于全局默认)',
        wa['is'] == ('2021-01-01', '2024-12-31'))
  check('窗口: 该池 config 的 OOS 生效', wa['oos'] == ('2025-01-01', '2025-12-31'))
  check('窗口: recent 从 OOS 次日开始(不重复计入 OOS 末日)',
        wa['recent'] == ('2026-01-01', None))
  check('窗口: 来源标记为 factor_config.json', wa['source'] == 'factor_config.json')

  # 形状不合法的窗口必须整体回退到全局默认, 不能"半采用"
  wb = fcfg.resolve_windows('broken', path=cfg_path)
  check('窗口: is_window 形状非法时回退全局默认',
        wb['is'] == (fcfg.TRADE_START, fcfg.IS_END) and wb['source'] == 'global')

  # 配置里没有的池 -> 全局默认
  wc = fcfg.resolve_windows('not_there', path=cfg_path)
  check('窗口: 池不在 config 中时回退全局默认',
        wc['is'] == (fcfg.TRADE_START, fcfg.IS_END))

  # 配置文件不存在 / 损坏 -> 全局默认(且不抛异常)
  wd = fcfg.resolve_windows('poolA', path=tmp / 'no_such.json')
  check('窗口: 配置文件缺失时回退全局默认',
        wd['is'] == (fcfg.TRADE_START, fcfg.IS_END))
  bad_path = tmp / 'bad.json'
  bad_path.write_text('{"pools": ', encoding='utf-8')
  we = fcfg.resolve_windows('poolA', path=bad_path)
  check('窗口: 配置文件损坏时回退全局默认(不抛异常)',
        we['is'] == (fcfg.TRADE_START, fcfg.IS_END))


# ============================== 7. JSON 合法性: NaN/inf 不得写成裸字面量 ==============================
# 背景: json.dump 默认 allow_nan=True, 会把 NaN/Infinity 写成裸字面量 —— 那不是合法 JSON
# (RFC 8259), 严格解析器一律拒绝. portfolio.json / position.json 一旦中招,
# read_config 解析失败 -> 调用方回退默认值 -> 持仓与净值快照静默失真.
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)

  # 1) _jsonable 递归转换
  conv = io._jsonable({'a': float('nan'), 'b': float('inf'), 'c': [1.5, float('-inf')],
                       'd': {'e': 'ok'}, 'f': None})
  check('jsonable: NaN/inf 递归转 None, 其余保持',
        conv['a'] is None and conv['b'] is None and conv['c'] == [1.5, None]
        and conv['d'] == {'e': 'ok'} and conv['f'] is None)

  # 2) numpy 标量也要转成原生类型(否则 json.dump 直接 TypeError)
  conv2 = io._jsonable({'i': np.int64(3), 'f': np.float64(1.25), 'b': np.bool_(True),
                        'n': np.float64('nan')})
  check('jsonable: numpy 标量转为原生类型',
        conv2['i'] == 3 and isinstance(conv2['i'], int)
        and conv2['f'] == 1.25 and conv2['b'] is True and conv2['n'] is None)

  # 3) 端到端: 含 NaN 的配置写盘后必须是"合法 JSON"且能读回同样语义
  io._save_json_file({'net_value': float('nan'), 'cash': 100.0,
                      'portfolio': {'quantity': {'A': 1}, 'support': {'A': float('inf')}}},
                     file_path=tmp, file_name='portfolio.json')
  raw = (tmp / 'portfolio.json').read_text(encoding='utf-8')
  check('json: 落盘内容不含裸 NaN 字面量',
        'NaN' not in raw and 'Infinity' not in raw)
  # 用**严格**解析器验证(标准 json 即可, 因为已无裸字面量)
  parsed = json.loads(raw)
  check('json: 严格解析可读回', parsed['net_value'] is None and parsed['cash'] == 100.0
        and parsed['portfolio']['support']['A'] is None)
  check('json: 无 .tmp 残留', not (tmp / 'portfolio.json.tmp').exists())

  # 4) 反向验证: 若绕过转义直接写 NaN, 标准解析应当失败(证明这条守卫不是空转)
  with open(tmp / 'bad.json', 'w', encoding='utf-8') as f:
    json.dump({'x': float('nan')}, f)
  bad_raw = (tmp / 'bad.json').read_text(encoding='utf-8')
  check('json: 对照组——未转义的 NaN 确实写出非法内容(证明守卫必要)',
        'NaN' in bad_raw)


print()
failed = [n for n, ok in results if not ok]
if failed:
  print(f'FAILED: {failed}')
  sys.exit(1)
print(f'ALL {len(results)} TESTS PASSED')
