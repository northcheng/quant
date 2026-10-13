# -*- coding: utf-8 -*-
"""
preflight — quant 项目改动后的前置自检(环境 + 可导入性 + 测试)
==============================================================
为什么需要它
------------
2026-09 的一批修复(见 CODE_REVIEW_2026-09.md 附录 B)修改了 8 个文件、新增了回归测试,
但**这些改动从未在任何环境里运行过**(改动时所在沙箱禁止执行 python). 因此第一步不是
"相信它", 而是**先确认环境能不能跑、模块能不能导入**, 再跑测试.

本脚本做三件事, 每步失败都给出明确下一步:
  1. 环境体检: python 版本 / 关键第三方包是否装了(pandas/akshare/futu/tigeropen/lightgbm/torch/pytest);
  2. 可导入性冒烟: 逐个 import 本次改动涉及的模块 —— 这能抓住"改坏了语法/缩进/引用"的问题,
     而不需要跑完整回测(历史上 P0-0 的断裂导入正是因为没有 import 冒烟才潜伏数日);
  3. 跑两个测试文件(脚本式, 用 subprocess 调子进程并汇总退出码).

用法
----
    C:\\Users\\northcheng\\.venv\\Scripts\\python.exe c:\\Users\\northcheng\\git\\quant\\tools\\preflight.py
    # 只做环境 + 导入冒烟, 不跑测试:
    C:\\Users\\northcheng\\.venv\\Scripts\\python.exe ...\\preflight.py --skip-tests
    # 只跑测试:
    C:\\Users\\northcheng\\.venv\\Scripts\\python.exe ...\\preflight.py --skip-import

退出码: 0 = 全部通过; 1 = 有失败项(详情见输出)。
"""
import argparse
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

# 仓库根: 本文件在 <repo>/quant/tools/ 下 -> parents[2]
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

# ---- 本次改动涉及的模块(导入冒烟清单) ----
# 说明: bc_trader 依赖 futu/tigeropen, 缺失时会 SKIP 而不是 FAIL(环境问题, 不是代码问题)
IMPORT_TARGETS = [
    ('quant.bc_util', False),
    ('quant.bc_data_io', False),
    ('quant.bc_technical_analysis', False),
    ('quant.bc_trader', True),                 # optional: 需要 futu/tigeropen
    ('quant.factor.config', False),
    ('quant.factor.data', False),
    ('quant.factor.prepare', False),
    ('quant.factor.factor', False),
    ('quant.factor.evaluate', False),
    ('quant.factor.combine', False),
    ('quant.factor.backtest', False),
    ('quant.factor.report', False),
    ('quant.factor.repro_old_presets', False),
    ('quant.factor.wf_strat_search', False),
    ('quant.factor.wf_step1_post', False),
    ('quant.factor.wf_step2_search', False),
    ('quant.factor.wf_step3_sens', False),
    ('quant.factor.wf_step4_export_weights', False),
    ('quant.factor.wf_combo_search', False),
    ('quant.ml.config', False),
    ('quant.ml.data', False),
    ('quant.ml.features', False),
    ('quant.ml.labels', False),
    ('quant.ml.dataset', False),
    ('quant.ml.seq', False),
    ('quant.ml.timing', False),
    ('quant.ml.train', False),
    ('quant.ml.predict', False),
    ('quant.ml.models.linear', False),
    ('quant.ml.models.gbdt', True),            # optional: 函数内延迟 import lightgbm
    ('quant.ml.models.neural', True),          # optional: 延迟 import torch
    ('quant.ml.models.sequence', True),
]

# ---- 关键第三方包(缺失会影响能力, 不一定阻塞导入) ----
PACKAGES = [
    ('numpy', '必需'),
    ('pandas', '必需'),
    ('matplotlib', '必需(绘图)'),
    ('openpyxl', '必需(xlsx 输出)'),
    ('akshare', '数据源(A股/港股/美股 eod)'),
    ('easyquotation', '数据源(CN/HK 实时)'),
    ('requests', 'HTTP'),
    ('futu', '券商 SDK(港股/美股; bc_trader 需要)'),
    ('tigeropen', '券商 SDK(美股; bc_trader 需要)'),
    ('lightgbm', 'ml.models.gbdt 需要'),
    ('torch', 'ml.models.neural / sequence 需要'),
    ('pytest', '仅"pytest 化测试"需要(当前测试是脚本式, 不依赖它)'),
]

TEST_FILES = [
    'quant/tests/test_bc_data_io_robustness.py',      # 既有 14 例(回归)
    'quant/tests/test_bc_data_io_incremental.py',     # 新增 30 例(+4 可跳过)
    'quant/tests/test_bc_data_io_eod_new_data.py',    # 新增 87 例(EOD New Data 区段: 9 助手 + 44 方法)
]

_results = []


def _record(name, status, detail=''):
  _results.append((name, status, detail))
  icon = {'PASS': 'PASS', 'FAIL': 'FAIL', 'SKIP': 'SKIP', 'WARN': 'WARN'}[status]
  line = f'[{icon}] {name}'
  if detail:
    line += f' -- {detail}'
  print(line)


def check_env():
  print('\n===== 1/3 环境体检 =====')
  print(f'python: {sys.version.split()[0]} ({sys.executable})')
  if sys.version_info < (3, 8):
    _record('python 版本 >= 3.8', 'FAIL', f'当前 {sys.version.split()[0]}')
  else:
    _record('python 版本 >= 3.8', 'PASS', sys.version.split()[0])

  for mod, why in PACKAGES:
    try:
      spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
      spec = None
    if spec is None:
      # 券商 SDK / ml 重依赖 / pytest 缺失不算"代码被改坏", 但对能力有影响 -> WARN
      status = 'WARN' if mod in ('futu', 'tigeropen', 'lightgbm', 'torch', 'pytest') else 'FAIL'
      _record(f'包 {mod} 可用', status, f'未安装({why})')
    else:
      _record(f'包 {mod} 可用', 'PASS', why)


def _is_third_party_missing(err: Exception) -> bool:
  """判断一个 import 失败是否只是"第三方包没装", 而不是我们改坏了代码."""
  if not isinstance(err, ImportError):
    return False
  msg = str(err).lower()
  # ModuleNotFoundError: No module named 'xxx' / 或 Couldn't import ...
  return msg.startswith('no module named') or 'no module named' in msg


def check_imports():
  print('\n===== 2/3 模块导入冒烟 =====')
  for mod, optional in IMPORT_TARGETS:
    try:
      importlib.import_module(mod)
      _record(f'import {mod}', 'PASS')
    except Exception as e:
      kind = f'{type(e).__name__}: {e}'
      if optional:
        # 延迟 import 的重依赖缺失不应算代码错
        _record(f'import {mod}', 'SKIP', kind)
      elif _is_third_party_missing(e):
        # 例如 pandas/akshare 没装 -> 环境问题, 不是代码问题. 标 WARN:
        # 既不掩盖"跑不起来", 也不会在汇总里被误当成"代码被改坏"。
        _record(f'import {mod}', 'WARN', f'第三方依赖缺失 -> {kind}')
      else:
        _record(f'import {mod}', 'FAIL', kind)


def check_tests():
  print('\n===== 3/3 回归测试 =====')
  for rel in TEST_FILES:
    path = REPO / rel
    if not path.exists():
      _record(f'test {Path(rel).name}', 'FAIL', f'文件不存在: {path}')
      continue
    print(f'\n--- 运行 {rel} ---')
    try:
      proc = subprocess.run([sys.executable, str(path)],
                            cwd=str(REPO), capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
      _record(f'test {Path(rel).name}', 'FAIL', '超时(1800s)')
      continue
    out = (proc.stdout or '') + (proc.stderr or '')
    print(out.rstrip())
    tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
    ok = (proc.returncode == 0)
    _record(f'test {Path(rel).name}', 'PASS' if ok else 'FAIL',
            f'exit={proc.returncode}; 末尾: {tail}')


def main():
  ap = argparse.ArgumentParser(description='quant 改动后的前置自检(环境 + 导入 + 测试)')
  ap.add_argument('--skip-env', action='store_true', help='跳过环境体检')
  ap.add_argument('--skip-import', action='store_true', help='跳过导入冒烟')
  ap.add_argument('--skip-tests', action='store_true', help='跳过测试')
  a = ap.parse_args()

  print(f'repo: {REPO}')
  if not a.skip_env:
    check_env()
  if not a.skip_import:
    check_imports()
  if not a.skip_tests:
    check_tests()

  print('\n===== 汇总 =====')
  n_fail = sum(1 for _, s, _ in _results if s == 'FAIL')
  n_warn = sum(1 for _, s, _ in _results if s == 'WARN')
  n_skip = sum(1 for _, s, _ in _results if s == 'SKIP')
  n_pass = sum(1 for _, s, _ in _results if s == 'PASS')
  print(f'PASS {n_pass} / FAIL {n_fail} / WARN {n_warn} / SKIP {n_skip}')
  print('\n读法:')
  print('  FAIL = 需要排查(代码问题, 或必需包缺失)')
  print('  WARN = 第三方依赖缺失导致跑不了 -> 装包即可(见下), 不是代码被改坏')
  print('  SKIP = 可选重依赖(lightgbm/torch/futu/tigeropen)缺失, 该模块本次无法导入冒烟')
  if n_warn:
    print('\n安装建议(按需):')
    print('  pip install -r quant/requirements.txt        # 基础(pandas/akshare/matplotlib/openpyxl...)')
    print('  pip install lightgbm torch                   # ml.models 的 gbdt/neural/sequence')
    print('  pip install futu-api tigeropen               # bc_trader(券商下单) 与 ml 无关')
  if n_fail:
    print('\n先修 FAIL 项再谈其他。若 FAIL 的 import 报的是 "No module named \'xxx\'"(xxx 为'
          '\n第三方包), 那属于装包问题; 若是其它类型异常(AttributeError/NameError/SyntaxError'
          '\n/IndentationError 等), 请把那一行原样发出来 —— 那是改动引入的问题。')
    sys.exit(1)
  print('\n全部通过。下一步建议: 重跑 pkl 并复核受影响的因子/回测结论'
        '(复权索引修复 + 6 处指标修复 + 窗口口径统一都会改变数值)。')
  sys.exit(0)


if __name__ == '__main__':
  main()
