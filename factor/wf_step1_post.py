# -*- coding: utf-8 -*-
"""
factor.wf_step1_post — Step1 结果**后处理(临时脚本)**
====================================================
只读 wf_combo_search 产出的 CSV, 不重跑搜索. 产出两部分:
  1) **缺口对比表**: 对每个方案(preset/cur_ic/equal/new)算出 IS->OOS 缺口
     (`is - oos`) 与 `fold_obj - oos`. 这是红线的实质判据 —— "把接受判据换成
     折外指标"到底有没有让样本外更稳.
  2) **数据质量诊断**: 各池 close-to-close 日收益 |r|>50% 的 symbol-day 计数与
     最极端样本, 用于解释 company 池 FULL total_ret 高达 200x+ 的异常.

用法: cd ~/git && python -u -m quant.factor.wf_step1_post
"""
import argparse

import numpy as np
import pandas as pd

from quant.factor.data import load_pool

CSV = r'c:\Users\northcheng\git\quant\factor\output\wf_combo_search_step1.csv'
WIN_ORDER = ['is', 'oos', 'full', 'ytd26']


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
  """逐池x方案: 各窗 sharpe + fold_obj + IS->OOS 缺口."""
  piv = df.pivot_table(index=['pool', 'scheme'], columns='window',
                       values='sharpe', aggfunc='first')
  ret = df.pivot_table(index=['pool', 'scheme'], columns='window',
                       values='total_ret', aggfunc='first')
  fo = df.groupby(['pool', 'scheme'])['fold_obj'].first()
  t = pd.DataFrame(index=piv.index)
  t['fold_obj'] = fo.round(3)
  for w in WIN_ORDER:
    if w in piv.columns:
      t[f'sh_{w}'] = piv[w].round(2)
  t['gap_is_oos'] = (piv['is'] - piv['oos']).round(2)      # >0 => IS 好于 OOS(过拟合迹象)
  t['gap_fold_oos'] = (fo - piv['oos']).round(2)           # 折外目标与真 OOS 的落差
  t['full_ret'] = ret['full'].round(2)
  return t


def dq_report(pool: str, topn: int = 5) -> dict:
  """单池数据质量: |日收益|>50% 的 symbol-day 计数 + 最极端样本."""
  ds = load_pool(pool)
  cl = ds.wide('close')
  ret = cl / cl.shift(1) - 1.0
  absr = ret.abs()
  mask = absr > 0.5
  n_ext = int(mask.sum().sum())
  n_cells = int(absr.notna().sum().sum())
  # 取 |r| 最大的 top-n symbol-day
  flat = absr.stack().reset_index()
  flat.columns = ['date', 'symbol', 'absr']
  flat['ret'] = [ret.at[d, s] for d, s in zip(flat['date'], flat['symbol'])]
  ext = flat[flat['absr'] > 0.5].sort_values('absr', ascending=False).head(topn)
  return {'pool': pool, 'n_ext': n_ext, 'n_cells': n_cells,
          'pct': round(100.0 * n_ext / max(n_cells, 1), 4),
          'top': [(f"{r.symbol}@{r.date:%Y-%m-%d}", round(float(r.ret), 3))
                  for r in ext.itertuples()]}


def main():
  ap = argparse.ArgumentParser(description='Step1 后处理(临时)')
  ap.add_argument('--csv', default=CSV)
  ap.add_argument('--dq', action='store_true', default=True)
  ap.add_argument('--pools', default='company_300,company_1000,etf_3x,hs300,a_etf_all')
  a = ap.parse_args()

  df = pd.read_csv(a.csv)
  print('\n========== Step1 缺口对比 (sharpe / 缺口) ==========')
  t = gap_table(df)
  print(t.to_string(float_format=lambda x: f'{x:g}'))

  print('\n========== 各方案 IS->OOS 缺口汇总 (gap_is_oos, 越小越稳) ==========')
  g = t['gap_is_oos'].unstack('scheme')
  print(g.to_string(float_format=lambda x: f'{x:g}'))
  print('\n  [解读] gap_is_oos>0 表示样本内明显好于样本外(过拟合迹象);'
        ' new 是"用折外目标选出来"的方案, 若它 gap 反而更大, 说明"只把判据换成折外"不够.')

  if a.dq:
    print('\n========== 数据质量诊断 (close-to-close |日收益|>50%) ==========')
    for pool in [s.strip() for s in a.pools.split(',') if s.strip()]:
      r = dq_report(pool)
      print(f"  {r['pool']:14s} 极端 symbol-day: {r['n_ext']:5d} / {r['n_cells']} "
            f"({r['pct']}%)  top: {r['top']}")


if __name__ == '__main__':
  main()
