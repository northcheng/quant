# -*- coding: utf-8 -*-
"""
对 4 个池跑完整 factor 流水线并把权重写入 factor_config.json。

顺序(每池):
  1) run_report_on_pkl.py --pkl <研究pkl> --pool <pool> --write-config
     -> 评估全部因子(IS/OOS/recent) + 各方案定权 + 回测 + 成本阶梯 + 图
     -> 通过 report.write_config 按键合并写回 factor_config.json (原子写, 不丢其它池)

注意:
  * **不传 --default-scheme** -> 保留各池既有的 default_scheme, 生产信号口径不变;
  * 写入前已由外部备份 factor_config.json;
  * 每池独立计时并落日志, 单池失败不影响其它池。
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
QUANT = Path.home() / 'quant'
WRAP = REPO / 'quant' / 'tools' / 'run_report_on_pkl.py'
LOGDIR = REPO / 'quant' / 'factor' / 'output' / 'logs'
LOGDIR.mkdir(parents=True, exist_ok=True)

POOLS = ['etf_3x', 'company_300', 'hs300', 'a_etf_all']

results = []
for i, pool in enumerate(POOLS, 1):
    pkl = QUANT / f'{pool}_day_ta_data_research.pkl'
    logf = LOGDIR / f'_prod_weights_{pool}.txt'
    print(f'\n{"="*100}\n[{i}/{len(POOLS)}] {pool}   pkl={pkl.name} '
          f'({pkl.stat().st_size/1024/1024:.0f} MB)\n{"="*100}', flush=True)
    if not pkl.exists():
        print(f'  !! pkl 不存在, 跳过', flush=True)
        results.append((pool, 'MISSING', 0.0))
        continue

    cmd = [PY, '-u', str(WRAP), '--pkl', str(pkl), '--pool', pool, '--write-config']
    t0 = time.time()
    with open(logf, 'w', encoding='utf-8') as fh:
        p = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = 'OK' if p.returncode == 0 else f'FAIL(exit={p.returncode})'
    print(f'  -> {status}  用时 {dt:.1f}s  日志 {logf.name}', flush=True)
    results.append((pool, status, dt))

print(f'\n{"="*100}\n汇总\n{"="*100}')
for pool, status, dt in results:
    print(f'  {pool:14s} {status:16s} {dt:8.1f}s')
