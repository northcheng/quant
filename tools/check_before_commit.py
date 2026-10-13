# -*- coding: utf-8 -*-
"""
check_before_commit — 提交前敏感信息与产物守卫
================================================
为什么需要它
------------
`git add` 是**不可逆**的: 一旦把实盘持仓、券商私钥或 API key 提交进去, 即便删除也要
`git filter-repo` 重写历史. 人眼在 1439 个文件里找不到这些东西, 但一个脚本可以.

本脚本**只读**, 不改任何文件、不删任何东西. 它做四件事:
  1. 扫描"绝不该入库"的文件是否存在于工作区(持仓/密钥/账号);
  2. 扫描这些文件是否**会被 git 跟踪**(在已建仓库的前提下, 用 .gitignore 规则模拟);
  3. 统计"体积大/数量多"的产物类别, 提醒你确认忽略规则是否覆盖;
  4. 扫描源码与文本文件里是否残留明文凭据(api key / 邮箱 / 私钥片段).

用法
----
    C:\\Users\\northcheng\\.venv\\Scripts\\python.exe c:\\Users\\northcheng\\git\\quant\\tools\\check_before_commit.py
    # 只看高危项:
    C:\\Users\\northcheng\\.venv\\Scripts\\python.exe ...\\check_before_commit.py --quiet

退出码: 0 = 未发现高危项; 1 = 发现高危项(不要提交)。
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ---- 1. 绝不该入库的路径(存在即高危) ----
FORBIDDEN_PATTERNS = [
    ('券商持仓与净值', ['quant/portfolio.json', 'quant/position.json']),
    ('券商账号/私钥', ['quant/trader/user_info.json', 'quant/trader/user_info.csv',
                       'quant/trader/rsa_private_key.pem', 'quant/trader/trade_condition.json']),
    ('API 密钥目录', ['quant/api_key/api_key.json']),
]

# ---- 2. 源码里不该出现的明文凭据特征 ----
# 说明: 命中不等于一定是真凭据(可能是占位/测试值), 需要人看一眼.
CREDENTIAL_PATTERNS = [
    ('疑似硬编码 api key(32位十六进制)', re.compile(r"['\"][0-9a-fA-F]{32}['\"]")),
    ('疑似 eodhd key(形如 xxxxx.yyyyy)', re.compile(r"['\"][0-9a-f]{14}\.\d{8}['\"]")),
    ('snowball/xq token', re.compile(r'xq_a_token')),
    ('明文邮箱', re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ('私钥头', re.compile(r'BEGIN [A-Z ]*PRIVATE KEY')),
    ('smtp 授权码字面量', re.compile(r"['\"][a-z]{16}['\"]\s*#?\s*.*(smtp|授权码|qq)", re.I)),
]

# 这些文件里的邮箱/密钥是"已知且可接受"的(测试 fixture 或文档占位), 从扫描中排除
CREDENTIAL_SCAN_EXCLUDE = {
    'quant/tests/test_bc_data_io_robustness.py',      # 用 t@t.com 做测试地址
    'quant/tests/test_bc_data_io_incremental.py',
    'quant/tools/preflight.py',
    'quant/tools/check_before_commit.py',
    'quant/CODE_REVIEW_2026-09.md',                   # 报告里必然引用 key 的形式
    '.gitignore',
}

SCAN_SUFFIXES = {'.py', '.json', '.md', '.txt', '.cfg', '.ini', '.toml', '.yml', '.yaml'}

# ---- 3. 产物类别(数量统计, 提醒确认忽略覆盖) ----
ARTIFACT_SUFFIXES = ['.csv', '.png', '.pkl', '.xlsx', '.pyc', '.zip', '.xmind', '.db']

SKIP_DIRS = {'.git', 'node_modules', '.venv', 'venv', '__pycache__'}

_fail = []
_warn = []


def _rel(p: Path) -> str:
  try:
    return p.relative_to(REPO).as_posix()
  except ValueError:
    return p.as_posix()


def _iter_files():
  for p in REPO.rglob('*'):
    if not p.is_file():
      continue
    if any(part in SKIP_DIRS for part in p.parts):
      continue
    yield p


def check_forbidden():
  print('\n===== 1) 绝不该入库的文件是否存在 =====')
  found_any = False
  for label, rels in FORBIDDEN_PATTERNS:
    for rel in rels:
      p = REPO / rel
      if p.exists():
        found_any = True
        _warn.append(f'{label}: {rel} 存在于工作区')
        print(f'[WARN] {label}: {rel} 存在 —— 确认 .gitignore 已覆盖, 不要 git add 它')
  if not found_any:
    print('[ OK ] 未发现持仓/密钥/账号文件(它们应在仓库外的 ~/quant/ 下)')


def check_ignored():
  """若有 .git 与 git 命令, 用 git check-ignore 精确验证; 否则退化为打印规则清单."""
  print('\n===== 2) 这些文件是否真的会被忽略 =====')
  git_dir = REPO / '.git'
  if not git_dir.exists():
    print('[SKIP] 当前还不是 git 仓库 —— 建仓库后请重跑本脚本,')
    print('       或手动抽查: git check-ignore -v quant/portfolio.json')
    return
  import subprocess
  targets = [rel for _, rels in FORBIDDEN_PATTERNS for rel in rels if (REPO / rel).exists()]
  targets += ['research_2026-09/output', 'quant/factor/output']
  for rel in targets:
    try:
      r = subprocess.run(['git', 'check-ignore', '-v', rel], cwd=str(REPO),
                         capture_output=True, text=True, timeout=60)
    except Exception as e:
      print(f'[SKIP] 无法调用 git({e}) —— 请手动 git check-ignore -v {rel}')
      return
    if r.returncode == 0:
      print(f'[ OK ] {rel} 被忽略: {r.stdout.strip()}')
    else:
      _fail.append(f'{rel} **未**被 .gitignore 覆盖(会被 git add 收进去)')
      print(f'[FAIL] {rel} 未被忽略 —— 补 .gitignore 规则!')


def check_credentials():
  print('\n===== 3) 源码/文本里的明文凭据 =====')
  hits = 0
  for p in _iter_files():
    if p.suffix.lower() not in SCAN_SUFFIXES:
      continue
    rel = _rel(p)
    if rel in CREDENTIAL_SCAN_EXCLUDE:
      continue
    try:
      text = p.read_text(encoding='utf-8', errors='ignore')
    except OSError:
      continue
    for label, rx in CREDENTIAL_PATTERNS:
      for m in rx.finditer(text):
        # 取所在行做展示(截断, 避免把整行密钥打到日志里太长)
        line_no = text.count('\n', 0, m.start()) + 1
        snippet = text.splitlines()[line_no - 1].strip()[:110]
        hits += 1
        if label.startswith('明文邮箱'):
          _warn.append(f'{rel}:{line_no} {label}')
        else:
          _fail.append(f'{rel}:{line_no} {label}')
        print(f'[{"WARN" if label.startswith("明文邮箱") else "FAIL"}] {rel}:{line_no} {label}'
              f'\n        {snippet}')
  if hits == 0:
    print('[ OK ] 未在源码/配置中发现明文凭据特征')


def check_artifacts():
  print('\n===== 4) 产物数量统计(确认忽略规则覆盖) =====')
  counts = {}
  total = 0
  for p in _iter_files():
    total += 1
    counts[p.suffix.lower()] = counts.get(p.suffix.lower(), 0) + 1
  print(f'工作区文件总数(已排除 .git/.venv/__pycache__): {total}')
  for suf in ARTIFACT_SUFFIXES:
    n = counts.get(suf, 0)
    if n:
      print(f'  {suf:8s} {n:5d}')
  print('提示: 这些应全在 .gitignore 覆盖范围内(见仓库根 .gitignore);'
        '\n      其中 research_2026-09/output/ 与 quant/factor/output/ 是数量最大的两块。')


def main():
  ap = argparse.ArgumentParser(description='提交前敏感信息与产物守卫(只读)')
  ap.add_argument('--quiet', action='store_true', help='只打印 FAIL/WARN 与汇总')
  args = ap.parse_args()

  print(f'repo: {REPO}')
  check_forbidden()
  check_ignored()
  check_credentials()
  if not args.quiet:
    check_artifacts()

  print('\n===== 汇总 =====')
  print(f'FAIL {len(_fail)} / WARN {len(_warn)}')
  for x in _fail:
    print(f'  [FAIL] {x}')
  for x in _warn:
    print(f'  [WARN] {x}')
  if _fail:
    print('\n**不要提交**: 先修 FAIL 项(补 .gitignore 或删除/外移文件), 再 git add。')
    sys.exit(1)
  print('\n未发现高危项。WARN 项请自行确认是否需要处理。')
  sys.exit(0)


if __name__ == '__main__':
  main()
