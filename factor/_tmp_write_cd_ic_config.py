# -*- coding: utf-8 -*-
"""
factor._tmp_write_cd_ic_config — Step 收尾 **临时脚本**: 把 cd_ic+6f 冻结权重写进 factor_config
================================================================================
来源: Step3 结论(折外目标 = 6 折等分 + agg=mean) + Step4 导出(见 output/_tmp_step4_weights.json),
  在 4 个池上重跑坐标下降得到的**逐因子冻结权重**.

关键: Step4 的权重用**旧预设因子名**(F_mom121 / F_er20 / ...), 而 factor_config.json /
  signal_generator 走**注册表名**. 本脚本先把名字映射到注册表名, 并**逐池验证**
  「旧名信号 @ 冻结权重」与「注册表名信号 @ 映射后权重」逐点相等(max_diff==0),
  再写入 factor_config.json(只加 cd_ic 方案 + 改 default_scheme, 其余 meta 原样保留).

这是**临时脚本**: 只做一次性的一次性映射/校验/落盘.

自检:
  cd ~/git && python -u -m quant.factor._tmp_write_cd_ic_config            # 只校验不写
  cd ~/git && python -u -m quant.factor._tmp_write_cd_ic_config --write    # 校验并写库
"""
import argparse

from quant.factor import combine as cmb
from quant.factor import config as cfg
from quant.factor import prepare as prep
from quant.factor import report as rep
from quant.factor import repro_old_presets as repro
from quant.factor.data import load_pool

SCHEME = 'cd_ic'          # 新方案名(与 Step2/Step3 口径命名一致)

# 旧预设名 -> 注册表名(等价关系由 repro.native_signals 定义: 见其 reg(...) 映射);
# 未列出的名字(N_/C_/G_/H_ 前缀)本身已是注册表名, 原样保留.
NAME_MAP = {
    'F_mom121': 'mom_12_1',      # repro.native_signals: reg('mom_12_1')
    'F_er20': 'er_20',           # reg('er_20')
    'F_obv20': 'obv_20',         # reg('obv_20')
    'D_ma20_dist': 'bias_20',    # reg('bias_20')
}

# Step4 导出的冻结权重(旧预设名; 逐字取自 output/_tmp_step4_weights.json 的 weights 字段)
CD_IC = {
    'company_300': {'F_mom121': -0.303016, 'G_oviv20': 0.114337, 'F_er20': 0.039333,
                    'F_obv20': 0.077606, 'D_ma20_dist': 0.063589},
    'etf_3x': {'F_mom121': 0.172597, 'C_tmqmom': 0.164821, 'G_vr20': 0.061751,
               'N_idiovol60': 0.121714, 'H_ichimoku_alpha': 0.067544, 'F_obv20': 0.053613},
    'hs300': {'N_idiovol60': -0.252531, 'F_mom121': 0.254458, 'D_ma20_dist': -0.010259,
              'F_obv20': 0.008461, 'H_ichimoku_alpha': 0.091454, 'G_oviv20': 0.033635,
              'G_vr20': 0.042592, 'F_er20': 0.059245},
    'a_etf_all': {'N_range20': 0.336694, 'G_oviv20': 0.575652, 'H_ichimoku_alpha': 0.086126,
                  'G_vr20': 0.154336, 'F_er20': 0.06114, 'F_obv20': 0.050263},
}


def to_registry(weights: dict) -> dict:
  """旧预设名权重 -> 注册表名权重(权重值不变, 只换键)."""
  return {NAME_MAP.get(k, k): float(v) for k, v in weights.items()}


def verify(pool: str, log) -> float:
  """验证映射等价: 旧名信号 @ 旧权重 与 注册表名信号 @ 新权重 逐点比较, 返回 max|diff|."""
  ds = load_pool(pool)
  old_sigs = repro.native_signals(ds)                       # 旧预设名
  old_w = CD_IC[pool]
  reg_w = to_registry(old_w)
  new_sigs = cmb.build_signals(ds, prep.DEFAULT_PREP, sorted(reg_w), cfg.MIN_CS)

  comp_old = cmb.combine_signals(old_sigs, old_w, min_cs=cfg.MIN_CS)
  comp_new = cmb.combine_signals(new_sigs, reg_w, min_cs=cfg.MIN_CS)
  a, b = comp_old.align(comp_new, join='inner')
  diff = float((a - b).abs().max().max())
  log.info(f'[verify]: {pool} 旧名/注册表名 权重 {len(old_w)} 个, '
           f'映射 {[k for k in old_w if k in NAME_MAP]}, max|diff| = {diff:.3e}')
  return diff


def main():
  ap = argparse.ArgumentParser(description='临时脚本: 写 cd_ic+6f 冻结权重进 factor_config')
  ap.add_argument('--write', action='store_true', help='校验通过后写入 factor_config.json')
  a = ap.parse_args()

  log = cfg.get_logger('write_cd_ic')
  diffs = {}
  for pool in CD_IC:
    diffs[pool] = verify(pool, log)

  bad = {p: d for p, d in diffs.items() if d > 1e-10}
  if bad:
    log.error(f'[abort]: 映射不等价, 不写库 -> {bad}')
    return

  doc = rep._load_config_doc(cfg.FACTOR_CONFIG)
  for pool, w in CD_IC.items():
    conf = doc['pools'].get(pool)
    if not conf:
      log.error(f'[skip]: factor_config 无池 {pool}')
      continue
    conf['schemes'][SCHEME] = to_registry(w)
    old_default = conf['params'].get('default_scheme')
    conf['params']['default_scheme'] = SCHEME
    log.info(f'[plan]: {pool} 方案 {SCHEME} = {to_registry(w)}; '
             f'default_scheme {old_default} -> {SCHEME}')

  if not a.write:
    log.info('[dry-run]: 未写库(--write 才落盘)')
    return

  for pool, conf in doc['pools'].items():
    if pool in CD_IC:
      rep.write_config(conf, cfg.FACTOR_CONFIG)
  log.info(f'[done]: 已写入 {cfg.FACTOR_CONFIG} (池: {list(CD_IC)})')


if __name__ == '__main__':
  main()
