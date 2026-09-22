# -*- coding: utf-8 -*-
"""strat_common.py — 组合层三项研究(参数扫描 / 稳健性治理 / 跨池迁移)的公共原语.

只做组合层的事, 不引入任何新数据; 口径全部对齐生产(signal_bridge + bc_backtest):
  PRESETS   : 池 -> (weights, top_k). 直取 signal_bridge.ALL_PRESETS(单一真源).
  make_kit  : BacktestKit(pool, 全历史) + 注册三套候选构建器
              (a_combo > alpha > combo 优先级, 名称冲突时先注册者胜).
  composite : 全历史组合分数(成分当日截面 rank_pct -> |w| 归一加权 -> fillna(0.5)),
              与 compute_composite / 桥的加权段同式.
  build_gate: 池级体制门('none' 常开 / 'breadth20' 截面宽度 / 'vr_trend' 趋势体制) -> 布尔宽表.
  run_cfg   : 窗口切片 -> run_config(payload). 时间线(信号日收盘决策 -> 次日开盘执行)由
              run_engine 保证, 本模块只负责切片与门构造, 无前视.
  bundle    : build/save/load/run_bundle —— 只保留回测必需的宽表(不含 kit), 供子进程复用.
  stat_row  : payload -> 一行统计 dict(带 pool/window/... 标签).

运行环境: 项目 venv(.venv, Python 3.12), 其中 matplotlib/pytz/requests/scipy 齐备.
"""
import gc
import os
import pickle
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))                   # .../git/quant/research
_GIT = os.path.dirname(os.path.dirname(_HERE))                       # .../git(quant 包所在)
_ROOT = os.path.dirname(_GIT)                                        # 项目根(signal_bridge 所在)
for _p in (_GIT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_SILENT_INIT = os.environ.get('STRAT_SILENT_INIT') == '1'


def log(msg: str):
    print(msg, flush=True)


def log_init(msg: str):
    """初始化期的环境/预设提示. 子进程里由 STRAT_SILENT_INIT=1 静默(父进程已打印过)."""
    if not _SILENT_INIT:
        print(msg, flush=True)


from quant.bc_backtest import BacktestKit, EngineParams, run_config       # noqa: E402
from quant.bc_combo_search import build_a_combo_cands, build_combo_cands  # noqa: E402
from quant.bc_factor_search import build_alpha_cands                      # noqa: E402

A_POOLS = ('hs300', 'a_etf_all')

STAT_KEYS = ['total_ret', 'cagr', 'sharpe', 'max_dd', 'calmar', 'vol',
             'ann_turnover', 'n_trades', 'win_rate', 'avg_ret', 'avg_days', 'n_days']


def load_presets() -> dict:
    """池 -> (weights, top_k), 直取 signal_bridge.ALL_PRESETS(单一真源, 不做本地副本)."""
    from signal_bridge import ALL_PRESETS
    log_init(f'[presets]: 取自 signal_bridge.ALL_PRESETS({len(ALL_PRESETS)} 池)')
    return {k: (dict(v['weights']), v.get('top_k')) for k, v in ALL_PRESETS.items()}


PRESETS = load_presets()


def wdesc(weights: dict) -> str:
    """权重描述: 按 |w| 降序, 如 'K_tmqr:1,H_trendmag_alpha:0.5'."""
    return ','.join(f'{k}:{v:g}' for k, v in
                    sorted(weights.items(), key=lambda kv: -abs(kv[1])))


def slice_w(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None and end is None:
        return df
    if start is None:
        return df.loc[:end]
    if end is None:
        return df.loc[start:]
    return df.loc[start:end]


def make_kit(pool: str) -> BacktestKit:
    """一个池的 BacktestKit(全历史, 不截窗 —— 窗口由 run_cfg 切片, 便于同一 kit 跑多窗)."""
    kit = BacktestKit(pool=pool)
    kit.register(build_a_combo_cands, label='a_combo候选')
    kit.register(build_alpha_cands, label='alpha候选')
    kit.register(build_combo_cands, label='combo候选')
    return kit


def composite(kit: BacktestKit, weights: dict) -> pd.DataFrame:
    """全历史组合分数宽表(rank 口径). 与窗口无关(截面 rank 逐日), 可跨窗复用."""
    comp = kit._composite(dict(weights), 'rank')
    return comp.reindex(index=kit.open_wide.index, columns=kit.open_wide.columns)


def always_on(like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=like.index, columns=like.columns)


def build_bundle(pool: str) -> dict:
    """一个池跑回测所需的全部轻型数据(全历史): open/close/atr 宽表 + 组合分数.

    刻意不保留 BacktestKit —— kit 持有 panel_full 与全部候选列(大池可达数 GB),
    而 run_config 只需要这几张宽表(大池约数十 MB), 于是子进程内存从 GB 级降到 MB 级,
    并行度不再受单池体积限制."""
    kit = make_kit(pool)
    bundle = {'pool': pool,
              'open_wide': kit.open_wide, 'close_wide': kit.close_wide,
              'atr_wide': kit.atr_wide,
              'composite': composite(kit, PRESETS[pool][0])}
    del kit
    gc.collect()
    return bundle


def save_bundle(bundle: dict, path: str):
    with open(path, 'wb') as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str) -> dict:
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def run_bundle(bundle: dict, name: str, p: EngineParams,
               start=None, end=None, gate: str = 'none',
               comp: pd.DataFrame = None) -> dict:
    """按窗口跑一个配置(不依赖 kit). 与 run_cfg 同为切片 + 门构造 + run_config.

    comp: 可选的组合分数覆盖(sign-flip 置换对照用); 缺省用 bundle 内的生产分数.
    """
    ow = slice_w(bundle['open_wide'], start, end)
    if len(ow) < 2:
        raise ValueError(f'窗口不足 2 个交易日: {start}~{end}')
    cw = slice_w(bundle['close_wide'], start, end)
    c = (bundle['composite'] if comp is None else comp).reindex(index=ow.index,
                                                                columns=ow.columns)
    g = build_gate(gate, bundle['close_wide'], ow)
    aw = slice_w(bundle['atr_wide'], start, end) if bundle['atr_wide'] is not None else None
    return run_config(name, ow, cw, c, g, p, atr_wide=aw)


def _vr20(close: pd.DataFrame) -> pd.DataFrame:
    """方差比(20): 与 combo_search.build_combo_cands / 桥 build_g_vr20 同公式."""
    ret1 = close.pct_change()
    var20 = close.pct_change(20).rolling(120, min_periods=60).var()
    var1 = ret1.rolling(120, min_periods=60).var()
    return var20 / (20.0 * var1.replace(0, np.nan))


def _g_breadth20(close: pd.DataFrame) -> pd.Series:
    """截面宽度: 池内收盘价站上 MA20 的标的占比 >= 50% 视为开门."""
    ma = close.rolling(20, min_periods=10).mean()
    above = (close > ma).where(close.notna())
    return above.mean(axis=1) >= 0.5


def _g_vr_trend(close: pd.DataFrame) -> pd.Series:
    """趋势体制: 池内 G_vr20 截面中位数 > 其自身 252 日中位数(方差比 > 1 偏趋势化)."""
    vr = _vr20(close).median(axis=1)
    return vr > vr.rolling(252, min_periods=60).median()


GATES = {'none': None, 'breadth20': _g_breadth20, 'vr_trend': _g_vr_trend}


def build_gate(kind: str, close_full: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """体制门 -> 布尔宽表(与 like 同形).

    kind: 'none' 常开 / GATES 内的门名. 门序列在全历史上计算后按 like.index 对齐,
      门外/未算出的日子一律 False(不开新仓并触发退出), 与 _resolve_gate 的 fillna(False) 同义.
    """
    fn = GATES.get(kind)
    if fn is None:
        return always_on(like)
    on = fn(close_full).reindex(like.index).fillna(False).to_numpy(dtype=bool)
    return pd.DataFrame(np.repeat(on[:, None], like.shape[1], axis=1),
                        index=like.index, columns=like.columns)


def run_cfg(kit: BacktestKit, comp: pd.DataFrame, name: str, p: EngineParams,
            start=None, end=None, gate: str = 'none') -> dict:
    """按窗口跑一个配置. comp 为全历史组合分数(内部按窗重切); gate 为门名(默认常开)."""
    ow = slice_w(kit.open_wide, start, end)
    if len(ow) < 2:
        raise ValueError(f'窗口不足 2 个交易日: {start}~{end}')
    cw = slice_w(kit.close_wide, start, end)
    c = comp.reindex(index=ow.index, columns=ow.columns)
    g = build_gate(gate, kit.close_wide, ow)
    aw = slice_w(kit.atr_wide, start, end) if kit.atr_wide is not None else None
    return run_config(name, ow, cw, c, g, p, atr_wide=aw)


def stat_row(payload: dict, **extra) -> dict:
    """payload -> 一行统计(pool/window/... 标签由 extra 传入)."""
    st = payload['stats']
    return {**extra, **{k: st.get(k) for k in STAT_KEYS}}


def daily_ret(payload: dict) -> pd.Series:
    return payload['equity'].pct_change().dropna()


def sharpe_per(payload: dict) -> float:
    """非年化(逐期)sharpe —— DSR / 多重检验校正需要未取整的逐期值."""
    r = daily_ret(payload)
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std())


def split_windows(start: str, is_end: str, oos_start: str) -> dict:
    """IS / OOS / FULL 三窗(交易窗口, 信号一律全历史构建)."""
    return {'IS': (start, is_end), 'OOS': (oos_start, None), 'FULL': (start, None)}


def base_params(cost_bps: float = 10.0) -> EngineParams:
    """生产基线引擎参数(与 EngineParams 默认值一致, cost 可覆盖)."""
    return EngineParams(cost_bps=cost_bps)