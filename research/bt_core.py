# -*- coding: utf-8 -*-
"""
bt_core.py — 通用回测核心: 任意信号 → 组合回测 → 统一结果对象
==============================================================

把 score_backtest 的引擎层(run_engine, 与信号完全解耦)包装成程序化 API:
  - BacktestKit:    数据加载 + 信号注册 + 三种喂法 + 对照 + 批量
  - BacktestResult: stats/equity/trades + monthly/yearly/top_trades/save

引擎层零改动(直接 import score_backtest), CLI 与 API 跑同一引擎, 口径逐位一致。

信号时间线(防前视, 与 score_backtest/signal_backtest 一致):
  全历史构建信号(rolling 需 warmup) → 截取交易窗口 [start, end] →
  信号日 s 收盘决策 → 执行日 d=s+1 开盘成交 → 单边成本。

两种分数口径(不互相统一, 各自回归各自基线):
  - mode='rank'  (默认): Σ wᵢ × 当日截面 rank_pct, NaN 以 0.5 参与
                        —— score_backtest 的 compute_composite 口径
  - mode='unit01'      : 分量 to_unit01 归一化到 [0,1] 后加权均值, NaN 保留
                        —— signal_backtest 的「汇总信号」口径

用法:
  from bt_core import BacktestKit, EngineParams, monthly_returns
  from alpha_mining import build_alpha_cands
  from signal_search import build_derived
  from factor_mining import build_mined

  kit = BacktestKit(pool='etf_3x', start='2026-01-01')
  kit.register(build_alpha_cands)          # builder: panel -> {名: 宽表}, 自动注入 panel 列
  kit.register(build_derived)              #   (rank 模式即可直接 --weights D_*)
  kit.register(build_mined)
  kit.register('my_signal', my_fn)         # 单信号: panel -> 宽表

  r1 = kit.run('H_trendmag_alpha')         # 单信号(rank 口径)
  r2 = kit.run({'H_trendmag_alpha': .4, 'D_mom250': .3, 'F_er20': .3})   # 权重组合
  r3 = kit.run(['H_ichimoku_alpha', 'H_trendmag_alpha', 'C_tmqmom', 'F_er20'],
               mode='unit01', name='agg_signal')                          # 汇总信号口径
  r4 = kit.run(ready_wide_df, name='现成宽表')                             # 直接喂宽表

  r1.stats; r1.equity; r1.trades
  r1.yearly(); r1.monthly(); r1.top_trades(5); r1.bottom_trades(5); r1.save(out_dir)

  kit.compare([r2, kit.shuffle(r2), kit.buyhold()])      # 对照汇总(打印+返回表)
  results = kit.run_many(['H_trendmag_alpha', 'D_mom250', {...}, ...])   # 批量

门(gate): BacktestKit(gate_col='trend_magnitude_day') 设默认门(>0 为开);
  run(gate=...) 可逐次覆盖: None=kit 默认 / 'none'=常开 / 列名 / DataFrame。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_research import load_panel, normalize_causal   # noqa: E402
from alpha_mining import POOLS                              # noqa: E402
# 引擎层(零改动复用): run_engine/run_config/EngineParams/perf_stats 等全部来自 score_backtest
from score_backtest import (                                # noqa: E402
    EngineParams, run_config, perf_stats, yearly_returns,
    shuffled_composite, compute_composite, summary_table, build_alpha_synths)


# ================================================================ 通用工具 ================================================================ #
def to_unit01(name: str, sig: pd.DataFrame) -> pd.DataFrame:
    """把信号统一到 [0,1] 尺度: 已知尺度的直接映射, 未知的因果归一化兜底.

    (自 signal_timeseries 迁入, 两处共用同一实现, 保证可视化与回测口径一致)
    """
    if name.startswith(('H_', 'P_')) and name.endswith('_alpha'):
        return sig.clip(0, 1)                       # 构造上已在 [0,1]
    if name.startswith('F_er'):
        return sig.clip(0, 1)                       # 效率比天然 [0,1]
    if name.startswith('C_'):
        return (sig / 2.0).clip(0, 1)               # 两个 rank_pct 之和, 上限 2
    # 兜底: 逐标的因果归一化(与 alpha 列同口径)
    out = {}
    for c in sig.columns:
        out[c] = normalize_causal(sig[c].astype(float), window=252, min_periods=60)
    return pd.DataFrame(out)[sig.columns]


def monthly_returns(equity: pd.Series) -> pd.Series:
    """月度收益: 月末净值/月初净值 - 1. (自 signal_backtest 迁入)"""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    g = equity.groupby([equity.index.year, equity.index.month])
    out = (g.last() / g.first() - 1.0).round(4)
    out.index = [f'{y}-{m:02d}' for y, m in out.index]
    return out


# ================================================================ 结果对象 ================================================================ #
class BacktestResult:
    """统一回测结果对象.

    属性访问 + dict 兼容(__getitem__ 委托内部 payload), 可直接传 summary_table
    等既有接收 run_config dict 的函数。
    """

    def __init__(self, payload: dict, comp: pd.DataFrame = None,
                 gate: pd.DataFrame = None, p: EngineParams = None):
        self._d = payload                     # run_config 输出: name/stats/daily/pos/trades/equity/ann_turnover
        self._comp, self._gate, self._p = comp, gate, p   # 生成材料(shuffle 对照复用)

    # ---- 字段(委托 payload) ----
    @property
    def name(self):
        return self._d['name']

    @property
    def stats(self) -> dict:
        return self._d['stats']

    @property
    def daily(self) -> pd.DataFrame:
        return self._d['daily']

    @property
    def pos(self) -> pd.DataFrame:
        return self._d['pos']

    @property
    def trades(self) -> pd.DataFrame:
        return self._d['trades']

    @property
    def equity(self) -> pd.Series:
        return self._d['equity']

    @property
    def ann_turnover(self) -> float:
        return self._d['ann_turnover']

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def __repr__(self):
        s = self.stats
        return (f"<BacktestResult '{self.name}' total={s.get('total_ret')} "
                f"sharpe={s.get('sharpe')} maxdd={s.get('max_dd')} "
                f"trades={s.get('n_trades', 0)}>")

    # ---- 派生报告 ----
    def yearly(self) -> pd.Series:
        return yearly_returns(self.equity)

    def monthly(self) -> pd.Series:
        return monthly_returns(self.equity)

    def _done_trades(self) -> pd.DataFrame:
        td = self.trades
        if len(td) and 'completed' in td.columns:
            return td[td['completed']]
        return td

    def top_trades(self, n: int = 5) -> pd.DataFrame:
        """盈利最大 n 笔(已完成)."""
        done = self._done_trades()
        if not len(done):
            return pd.DataFrame()
        cols = ['symbol', 'entry_exec', 'exit_exec', 'days', 'ret']
        return done.sort_values('ret', ascending=False)[cols].round({'ret': 4}).head(n)

    def bottom_trades(self, n: int = 5) -> pd.DataFrame:
        """亏损最大 n 笔(已完成)."""
        done = self._done_trades()
        if not len(done):
            return pd.DataFrame()
        cols = ['symbol', 'entry_exec', 'exit_exec', 'days', 'ret']
        return done.sort_values('ret', ascending=False)[cols].round({'ret': 4}).tail(n)

    # ---- 落盘 ----
    def save(self, out_dir: str) -> str:
        """落盘 3 件套: equity_curve.csv / positions.csv / trades.csv(文件名与 CLI 一致)."""
        os.makedirs(out_dir, exist_ok=True)
        self.daily.to_csv(os.path.join(out_dir, 'equity_curve.csv'), encoding='utf-8-sig')
        self.pos.to_csv(os.path.join(out_dir, 'positions.csv'), index=False, encoding='utf-8-sig')
        if len(self.trades):
            self.trades.to_csv(os.path.join(out_dir, 'trades.csv'), index=False, encoding='utf-8-sig')
        return out_dir


# ================================================================ 回测套件 ================================================================ #
class BacktestKit:
    """通用回测套件: 一个池一个窗口一套引擎参数, 注册任意信号源后 run/compare/run_many.

    窗口语义(与 signal_backtest 一致): 价格宽表在全历史上 unstack 后截行,
    信号在全历史构建后截行; 窗口内无数据的标的以 NaN 列参与, 不占截面名次。
    """

    def __init__(self, pool: str = None, pkl_path: str = None, interval: str = 'day',
                 start: str = None, end: str = None, gate_col: str = None,
                 engine_params: EngineParams = None):
        """pool 与 pkl_path 至少给一个(直接指定 pkl 优先); start/end 为交易窗口
        (None=全历史; 信号一律全历史构建, 只截交易窗口, 无 warmup 污染);
        gate_col=None 常开(通用默认), 或 panel 列名(>0 视为开)。"""
        self.pkl_path = pkl_path or POOLS.get(pool)
        if not self.pkl_path or not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f'pkl 不存在: {self.pkl_path}(pool={pool})')
        self.pool = pool or os.path.basename(self.pkl_path)
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.gate_col = gate_col
        self.engine_params = engine_params or EngineParams()

        _, panel = load_panel(self.pkl_path, interval)
        self.panel_full = panel          # 全历史(注入后含注册信号列)
        self._inject_log = []            # [(label, 列名)] 记录已注入列(CLI 打印用)
        self._inject([('alpha复现', build_alpha_synths)])   # 与 score_backtest 一致: 无条件注入
        self._builders = []              # [(label, builder)]
        self._cands = None               # 惰性缓存: {信号名: 全历史宽表}
        self._study = None               # 惰性: 交易窗口内的 panel

        ow = self.panel_full['Open'].unstack('symbol').sort_index()
        cw = self.panel_full['Close'].unstack('symbol').sort_index()
        self.open_wide = ow.loc[self.start:self.end]
        self.close_wide = cw.loc[self.start:self.end]
        if len(self.open_wide) < 2:
            raise ValueError(f'交易窗口不足 2 个交易日: '
                             f'{self.open_wide.index.min()}~{self.open_wide.index.max()}')

    # ---------------- 信号注册 ---------------- #
    def register(self, obj, fn=None, label: str = None):
        """注册信号源, 两种形式:
          kit.register(builder)       builder(panel)->{信号名: 宽表}(如 build_alpha_cands)
          kit.register('名字', fn)     fn(panel)->宽表(单信号, 任意自定义函数)
        builder 输出会注入 panel 列(已存在则跳过), 使 rank 模式可直接引用这些信号名。"""
        if fn is None:
            self._builders.append((label or getattr(obj, '__name__', str(obj)), obj))
        else:
            wrap = lambda panel, _fn=fn, _n=obj: {_n: _fn(panel)}  # noqa: E731
            self._builders.append((label or str(obj), wrap))
        self._cands = None   # 失效缓存
        self._study = None   # 新列注入后窗口 study 需重建
        return self

    def _inject(self, injectors: list):
        """在完整历史上计算信号再按 panel 行序注入, 避免错位(score_backtest 同口径)。
        已存在列跳过(不覆盖), 全 NaN 列不注入; 注入结果记入 _inject_log。"""
        panel = self.panel_full
        mi = panel.index
        key = pd.MultiIndex.from_arrays([mi.get_level_values('date'), mi.get_level_values('symbol')],
                                        names=['date', 'symbol'])
        for label, builder in injectors:
            for name, wide in builder(panel).items():
                if name in panel.columns:
                    continue
                vals = wide.stack().reindex(key).to_numpy(dtype=float)
                if not np.isnan(vals).all():
                    panel = panel.assign(**{name: vals})
                    self._inject_log.append((label, name))
        self.panel_full = panel

    @property
    def study(self) -> pd.DataFrame:
        """交易窗口内的 panel(rank 模式成分列的来源)。"""
        if self._study is None:
            st = self.panel_full
            if self.start is not None:
                st = st[st.index.get_level_values('date') >= self.start]
            if self.end is not None:
                st = st[st.index.get_level_values('date') <= self.end]
            self._study = st
        return self._study

    def _get_cands(self) -> dict:
        """构建并缓存全部注册信号(全历史, 惰性; 同时注入 panel 列)。"""
        if self._cands is None:
            inject, cands = [], {}
            for label, builder in self._builders:
                out = builder(self.panel_full)
                for name, wide in out.items():
                    if name not in cands:
                        cands[name] = wide
                inject.append((label, lambda p, _o=out: _o))
            if inject:
                self._inject(inject)
                self._study = None
            self._cands = cands
        return self._cands

    # ---------------- 分数构造 ---------------- #
    def _composite(self, spec, mode: str) -> pd.DataFrame:
        """spec → 全历史/窗口组合分数宽表。"""
        if isinstance(spec, pd.DataFrame):
            return spec                                        # 现成宽表: 原样(截行在 run 统一做)
        if isinstance(spec, str):
            if mode == 'unit01':
                cands = self._get_cands()
                if spec not in cands:
                    raise KeyError(f'未注册信号: {spec}(已注册: {sorted(cands)})')
                return to_unit01(spec, cands[spec])
            return self._rank_composite({spec: 1.0})           # rank 口径单信号 = 截面 rank_pct
        if isinstance(spec, (list, tuple)):
            spec = {s: 1.0 for s in spec}
        if not isinstance(spec, dict) or not spec:
            raise TypeError(f'run 只接受 str/list/dict/DataFrame, 收到: {type(spec)}')
        if mode == 'unit01':
            # 汇总信号口径: 分量归一化到 [0,1] 后加权 skipna 均值 —— 逐格 Σ wᵢxᵢ / Σ|wᵢ|
            # (有效分量)。等权时与 build_symbol_view 的 groupby.mean(skipna) 逐位一致:
            # 某分量缺失当日不参与, 全分量缺失才 NaN; 负权 = 反向分量(不做 clip)。
            cands = self._get_cands()
            missing = [k for k in spec if k not in cands]
            if missing:
                raise KeyError(f'未注册信号: {missing}(已注册: {sorted(cands)})')
            num = den = None
            for k, w in spec.items():
                u = to_unit01(k, cands[k])
                v = u.fillna(0.0) * w
                d = u.notna().astype(float) * abs(w)
                num = v if num is None else num.add(v, fill_value=0.0)
                den = d if den is None else den.add(d, fill_value=0.0)
            return num / den.replace(0.0, np.nan)
        return self._rank_composite(spec)

    def _rank_composite(self, weights: dict) -> pd.DataFrame:
        """rank 口径: 成分全部在窗口 panel 列时逐位走 compute_composite(与 score_backtest
        完全一致); 否则成分可来自注册信号(公式相同, 成分源更宽)。"""
        if all(k in self.study.columns for k in weights):
            return compute_composite(self.study, weights)
        cands = self._get_cands()
        total = float(sum(abs(w) for w in weights.values()))
        if total <= 0:
            raise ValueError('权重绝对值和为 0')
        comp = None
        for col, w in weights.items():
            if col in self.panel_full.columns:
                wide = pd.to_numeric(self.panel_full[col], errors='coerce') \
                              .unstack('symbol').sort_index()
            elif col in cands:
                wide = cands[col].astype(float)
            else:
                raise KeyError(f'组合分数成分不存在(既非 panel 列也非注册信号): {col}')
            pct = wide.rank(axis=1, pct=True)                  # 当日截面, NaN 保留
            part = (w / total) * pct
            comp = part if comp is None else comp.add(part, fill_value=0.0)
        return comp.fillna(0.5)

    def _resolve_gate(self, gate, like: pd.DataFrame) -> pd.DataFrame:
        """gate 参数 → 布尔宽表: None=kit 默认 / 'none'=常开 / 列名(>0 开) / DataFrame。"""
        g = self.gate_col if gate is None else gate
        if g is None or (isinstance(g, str) and g.lower() == 'none'):
            return pd.DataFrame(True, index=like.index, columns=like.columns)
        if isinstance(g, pd.DataFrame):
            return g.reindex(index=like.index, columns=like.columns).fillna(False)
        if g not in self.panel_full.columns:
            raise KeyError(f'门列不存在: {g}')
        gw = pd.to_numeric(self.panel_full[g], errors='coerce').unstack('symbol').sort_index() > 0
        return gw.reindex(index=like.index, columns=like.columns).fillna(False)

    @staticmethod
    def _auto_name(spec, mode: str) -> str:
        if isinstance(spec, pd.DataFrame):
            return 'wide(现成宽表)'
        if isinstance(spec, str):
            return spec
        keys = list(spec.keys()) if isinstance(spec, dict) else list(spec)
        tag = 'agg' if mode == 'unit01' else 'rank'
        return f'{tag}({",".join(keys)})'[:80]

    # ---------------- 核心执行 ---------------- #
    def run(self, spec, mode: str = 'rank', name: str = None, gate=None,
            engine_params: EngineParams = None) -> BacktestResult:
        """跑一个配置, 返回 BacktestResult.

        spec: 信号名 str / 权重 dict / 信号名 list(等权) / 现成分数宽表 DataFrame
        mode: 'rank'(默认, score_backtest 口径) 或 'unit01'(汇总信号口径)
        gate: None=kit 默认 / 'none'=常开 / panel 列名 / 布尔 DataFrame
        """
        comp = self._composite(spec, mode)
        comp = comp.reindex(index=self.open_wide.index, columns=self.open_wide.columns)
        g = self._resolve_gate(gate, comp)
        p = engine_params or self.engine_params
        payload = run_config(name or self._auto_name(spec, mode),
                             self.open_wide, self.close_wide, comp, g, p)
        return BacktestResult(payload, comp=comp, gate=g, p=p)

    def run_many(self, specs: list, mode: str = 'rank', gate=None,
                 engine_params: EngineParams = None) -> list:
        """批量: specs 为 spec 列表(或 (name, spec) 元组), 返回 BacktestResult 列表。"""
        out = []
        for item in specs:
            if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], (int, float)):
                nm, sp = item
            else:
                nm, sp = None, item
            out.append(self.run(sp, mode=mode, name=nm, gate=gate,
                                engine_params=engine_params))
        return out

    # ---------------- 内置对照(同一引擎, 公平核算) ---------------- #
    def shuffle(self, base, seed: int = 42, name: str = 'shuffle对照',
                engine_params: EngineParams = None) -> BacktestResult:
        """时序置换对照: 接收 run 的结果或分数宽表, 破坏分数-时间对齐(保留截面分布)。"""
        comp = base._comp if isinstance(base, BacktestResult) else base
        comp_shuf = shuffled_composite(comp, seed)
        g = base._gate if isinstance(base, BacktestResult) else self._resolve_gate(None, comp)
        p = engine_params or (base._p if isinstance(base, BacktestResult) else self.engine_params)
        payload = run_config(name, self.open_wide, self.close_wide,
                             comp_shuf.reindex(index=self.open_wide.index,
                                               columns=self.open_wide.columns), g, p)
        return BacktestResult(payload, comp=comp, gate=g, p=p)

    def _const_comp(self) -> pd.DataFrame:
        return pd.DataFrame(0.5, index=self.open_wide.index, columns=self.open_wide.columns)

    def buyhold(self, name: str = 'buyhold_pool(等权持有)',
                engine_params: EngineParams = None) -> BacktestResult:
        """全池等权买入持有: 常数分数 + 常开门 + 大K 等权。"""
        comp = self._const_comp()
        g = pd.DataFrame(True, index=comp.index, columns=comp.columns)
        p_ge = (engine_params or self.engine_params).copy(top_k=9999, exit_rank=99999, sizing='equal')
        payload = run_config(name, self.open_wide, self.close_wide, comp, g, p_ge)
        return BacktestResult(payload, comp=comp, gate=g, p=p_ge)

    def gate_equal(self, name: str = 'gate_equal(门内等权)',
                   engine_params: EngineParams = None) -> BacktestResult:
        """门内全等权(近似无分数行为): 常数分数 + kit 默认门 + 大K 等权。"""
        comp = self._const_comp()
        g = self._resolve_gate(None, comp)
        p_ge = (engine_params or self.engine_params).copy(top_k=9999, exit_rank=99999, sizing='equal')
        payload = run_config(name, self.open_wide, self.close_wide, comp, g, p_ge)
        return BacktestResult(payload, comp=comp, gate=g, p=p_ge)

    def buyhold_symbol(self, sym: str, name: str = None,
                       engine_params: EngineParams = None) -> BacktestResult:
        """单标的买入持有(如 TQQQ/SPY 基准)。"""
        if sym not in self.open_wide.columns:
            raise KeyError(f'{sym} 不在池 {self.pool} 中')
        ob, cb = self.open_wide[[sym]], self.close_wide[[sym]]
        comp = self._const_comp()[[sym]]
        g = pd.DataFrame(True, index=comp.index, columns=[sym])
        p1 = (engine_params or self.engine_params).copy(top_k=1, exit_rank=2,
                                                        sizing='equal', per_symbol_cap=1.0)
        payload = run_config(name or f'buyhold_{sym}', ob, cb, comp, g, p1)
        return BacktestResult(payload, comp=comp, gate=g, p=p1)

    # ---------------- 汇总 ---------------- #
    def compare(self, results: list, print_table: bool = True) -> pd.DataFrame:
        """对照汇总表(BacktestResult 与 run_config dict 混合均可)。"""
        t = summary_table(results)
        if print_table:
            print(t.to_string(index=False))
        return t

    def register_defaults(self):
        """注册常用信号源(等价于显式 register 三个 builder)。"""
        from alpha_mining import build_alpha_cands
        from signal_search import build_derived
        try:
            from factor_mining import build_mined
        except Exception:
            build_mined = None
        self.register(build_alpha_cands, label='alpha候选')
        self.register(build_derived, label='衍生信号')
        if build_mined is not None:
            self.register(build_mined, label='挖矿信号')
        return self
