# bc_data_io 健壮性行为验证（回归测试）
#
# 覆盖四块：邮件报告 / ak 下载容错 / config 原子写 / 本地数据文件加载
#
# 运行：
#   C:\Users\northcheng\.venv\Scripts\python.exe c:\Users\northcheng\git\quant\tests\test_bc_data_io_robustness.py
# 预期：每行 PASS，末尾 ALL 14 TESTS PASSED，退出码 0
#
# 注意：
# - 必须用 venv 的 python（系统 python 缺 pandas/akshare 依赖）
# - 所有用例 mock 外部依赖（ak 下载），文件读写全部在临时目录，不联网、不碰真实数据文件
# - 修改 bc_data_io.py / bc_trader.py 后跑一遍，防止回归
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from quant import bc_data_io as io

results = []


def check(name, cond):
  results.append((name, bool(cond)))
  print(f'{name}: {"PASS" if cond else "FAIL"}')


# ============================== config 原子写与防覆写 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)
  cfg_file = tmp / 'cfg.json'

  # 1) modify_config on a valid file
  cfg_file.write_text('{"a": 1}', encoding='utf8')
  io.modify_config('b', 2, file_path=tmp, file_name='cfg.json')
  d = json.loads(cfg_file.read_text(encoding='utf8'))
  check('config: modify_config 更新键后文件可正常解析', d == {'a': 1, 'b': 2})
  check('config: 原子写无 .tmp 残留', not (tmp / 'cfg.json.tmp').exists())

  # 2) modify_config on a corrupt(unreadable but non-empty) file -> skip, keep original
  cfg_file.write_text('{"a": 1, "truncat', encoding='utf8')
  original = cfg_file.read_text(encoding='utf8')
  io.modify_config('b', 2, file_path=tmp, file_name='cfg.json')
  check('config: modify_config 拒绝覆写损坏文件', cfg_file.read_text(encoding='utf8') == original)

  # 3) add/remove on valid file
  cfg_file.write_text('{"a": 1}', encoding='utf8')
  io.add_config('c', 3, file_path=tmp, file_name='cfg.json')
  io.remove_config('a', file_path=tmp, file_name='cfg.json')
  d = json.loads(cfg_file.read_text(encoding='utf8'))
  check('config: add_config / remove_config 正常工作', d == {'c': 3})

  # 4) create_config_file
  cfg_file.unlink()
  io.create_config_file({'x': [1, 2]}, file_path=tmp, file_name='cfg.json')
  d = json.loads(cfg_file.read_text(encoding='utf8'))
  check('config: create_config_file 正常且无 .tmp 残留', d == {'x': [1, 2]} and not (tmp / 'cfg.json.tmp').exists())

  # 5) corrupt file guard for add_config
  cfg_file.write_text('garbage-not-json', encoding='utf8')
  original = cfg_file.read_text(encoding='utf8')
  io.add_config('y', 9, file_path=tmp, file_name='cfg.json')
  check('config: add_config 拒绝覆写损坏文件', cfg_file.read_text(encoding='utf8') == original)


# ============================== 本地数据文件加载 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)

  # 1) valid csv but missing standard columns -> processing error, file KEPT
  f = tmp / 'AAPL.csv'
  f.write_text('Date,Close\n2024-01-01,100\n', encoding='utf8')
  df = io.load_stock_data(file_path=tmp, file_name='AAPL', standard_columns=True)
  check('加载: 处理失败(列缺失)时保留原文件并返回 None', df is None and f.exists())

  # 2) corrupt csv -> read failure, file removed for re-download
  f2 = tmp / 'BAD.csv'
  f2.write_bytes(b'\xff\xfe\x00\x01garbage\xff\xfe\x00')
  df = io.load_stock_data(file_path=tmp, file_name='BAD')
  check('加载: 文件损坏不可读时删除待重下', df is None and not f2.exists())

  # 3) normal csv still loads
  f3 = tmp / 'OK.csv'
  f3.write_text('Date,Open,High,Low,Close,Volume,Adj Close,Dividend,Split\n2024-01-01,1,2,0.5,1.5,100,1.5,0,1\n', encoding='utf8')
  df = io.load_stock_data(file_path=tmp, file_name='OK', standard_columns=True)
  check('加载: 正常文件按标准列加载', df is not None and len(df) == 1 and 'Adj Close' in df.columns)


# ============================== 邮件报告健壮性 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)
  # portfolio.json missing 'futu'/'pingan' keys entirely, updated=None
  (tmp / 'portfolio.json').write_text(json.dumps({'tiger': {'global_account': {'net_value': 100, 'updated': None, 'portfolio': None}}}), encoding='utf8')
  config = {'config_path': tmp, 'result_path': tmp, 'log_path': tmp}
  try:
    ret = io.send_result_by_email(config=config, to_addr='t@t.com', platform=['tiger', 'futu', 'pingan'], test=True)
    check('邮件: portfolio 缺平台键 + updated=None 时不崩溃', ret == 'test')
  except Exception as e:
    print(f'  exception: {type(e).__name__}: {e}')
    check('邮件: portfolio 缺平台键 + updated=None 时不崩溃', False)


# ============================== ak 下载容错 ==============================
with tempfile.TemporaryDirectory() as tmp:
  tmp = Path(tmp)

  # 1) code map is None -> abort cleanly instead of TypeError
  with mock.patch.object(io, 'get_data_from_ak', side_effect=RuntimeError('network down')), \
       mock.patch.object(io, 'get_code_map_from_ak', return_value=None):
    ret = io.update_stock_data_from_ak(symbols=['AAPL'], stock_data_path=tmp, is_return=True)
    check('ak: code map 为 None 时干净中止', ret is None)

  # 2) benchmark download raises -> fallback to today, symbol download failure guarded, loop completes
  code_map = pd.DataFrame({'代码': ['105.AAPL']})
  with mock.patch.object(io, 'get_data_from_ak', side_effect=RuntimeError('network down')), \
       mock.patch.object(io, 'get_code_map_from_ak', return_value=code_map):
    try:
      ret = io.update_stock_data_from_ak(symbols=['AAPL'], stock_data_path=tmp, is_return=True, is_print=True)
      check('ak: 网络全部失败时回退基准日并返回 dict', isinstance(ret, dict))
    except Exception as e:
      print(f'  exception: {type(e).__name__}: {e}')
      check('ak: 网络全部失败时回退基准日并返回 dict', False)

  # 3) happy path: benchmark + symbol download succeed
  # mock returns what the real get_data_from_ak returns: already post-processed(DatetimeIndex, standard columns)
  ak_raw = pd.DataFrame({
    '日期': ['2024-01-01', '2024-01-02'], '开盘': [10.0, 11.0], '收盘': [10.5, 11.5],
    '最高': [11.0, 12.0], '最低': [9.5, 10.5], '成交量': [1000, 1100]})
  ak_df = io.post_process_download_data(ak_raw, 'ak')
  with mock.patch.object(io, 'get_data_from_ak', return_value=ak_df.copy()), \
       mock.patch.object(io, 'get_code_map_from_ak', return_value=code_map):
    try:
      ret = io.update_stock_data_from_ak(symbols=['AAPL'], stock_data_path=tmp, is_return=True, is_save=False, is_print=True)
      ok = isinstance(ret, dict) and len(ret.get('105.AAPL', pd.DataFrame())) == 2
      check('ak: 正常下载路径返回数据', ok)
    except Exception as e:
      print(f'  exception: {type(e).__name__}: {e}')
      check('ak: 正常下载路径返回数据', False)

  # 4) happy path with is_save=True for us stock(cn_stock=False), data_to_save must be defined for us too
  with mock.patch.object(io, 'get_data_from_ak', return_value=ak_df.copy()), \
       mock.patch.object(io, 'get_code_map_from_ak', return_value=code_map):
    try:
      ret = io.update_stock_data_from_ak(symbols=['AAPL'], stock_data_path=tmp, is_return=True, is_save=True)
      saved = pd.read_csv(tmp / '105.AAPL.csv', encoding='utf8')
      ok = isinstance(ret, dict) and len(saved) == 2 and 'Date' in saved.columns
      check('ak: 美股保存路径正常落盘(无 NameError)', ok)
    except Exception as e:
      print(f'  exception: {type(e).__name__}: {e}')
      check('ak: 美股保存路径正常落盘(无 NameError)', False)


print()
failed = [n for n, ok in results if not ok]
if failed:
  print(f'FAILED: {failed}')
  sys.exit(1)
print(f'ALL {len(results)} TESTS PASSED')
