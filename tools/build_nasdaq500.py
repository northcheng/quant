# -*- coding: utf-8 -*-
"""
构建 nasdaq_500 池(方案 B: 纳斯达克挂牌按市值前 ~500).

分阶段执行, 中间产物落盘可复用/断点续跑:
  list  : 拉取 NASDAQ 交易所 common_stock 全集 -> univ list csv
  cap   : 拉取 bulk-fundamentals 取当前市值 -> 合并排名, 取前 N -> univ ranked csv
  check : 核查本地 stock_data/{symbol}.csv 覆盖
  fetch : 用 bc_data_io 补下载缺失标的的 OHLCV 到 stock_data/
  regen : 复用 tools/regen_ta_pkl.py 生成 research pkl
  reg   : 注册池到 selected_sec_list.json + factor_config.json

用法:
  python quant/tools/build_nasdaq500.py --stage list
  python quant/tools/build_nasdaq500.py --stage cap --top 500
  ...
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant import bc_data_io as io  # noqa: E402

HOME = Path.home()
QUANT = HOME / 'quant'
STOCK = QUANT / 'stock_data'

LIST_CSV = QUANT / 'nasdaq_all_list.csv'      # 交易所挂牌全集(含类型)
SCREEN_CSV = QUANT / 'nasdaq_screen.csv'      # screener 原始拉取(NASDAQ 按市值降序, 最多 1000)
RANK_CSV = QUANT / 'nasdaq_500_univ.csv'      # 排名后选定名单(含市值/排名)
KEY = io.default_eod_key

POOL = 'nasdaq_500'
SEL_JSON = REPO / 'quant' / 'selected_sec_list.json'
FACTOR_CONF = REPO / 'quant' / 'factor_config.json'
RESEARCH_PKL = QUANT / f'{POOL}_day_ta_data_research.pkl'


def stage_list():
    """拉取 NASDAQ 交易所挂牌的 common_stock 全集(1 call)."""
    df = io.get_exchange_symbol_list_from_eod(exchange='NASDAQ', type='common_stock', is_print=True)
    if df is None or len(df) == 0:
        print('[list] 空结果 —— 检查 API key / 权限')
        return
    df.to_csv(LIST_CSV, index=False, encoding='utf-8')
    print(f'[list] {len(df)} 个 NASDAQ common_stock -> {LIST_CSV}')
    print(f'[list] 列: {list(df.columns)}')
    print(df.head(8).to_string())


def _screener_pull(max_rows: int = 1000, page: int = 100):
    """分页拉取 NASDAQ screener(按市值降序). offset 上限 999 -> 最多 1000 行. 落盘复用."""
    if SCREEN_CSV.exists():
        df = pd.read_csv(SCREEN_CSV, encoding='utf-8')
        print(f'[cap] 复用已拉取 screener: {len(df)} 行')
        return df
    filters = json.dumps([['exchange', '=', 'NASDAQ']])
    rows, off = [], 0
    while len(rows) < max_rows and off <= 900:
        j = io._get_json_from_eod('/screener',
                                  params={'filters': filters, 'sort': 'market_capitalization.desc',
                                          'limit': page, 'offset': off}, is_print=True)
        data = (j or {}).get('data') or []
        if not data:
            print(f'[cap] offset={off} 空返回, 停止')
            break
        rows.extend(data)
        off += page
        print(f'[cap] 已拉取 {len(rows)} 行 (offset={off})')
        if len(data) < page:
            break
    df = pd.DataFrame(rows)
    df.to_csv(SCREEN_CSV, index=False, encoding='utf-8')
    return df


def stage_cap(top: int):
    """NASDAQ screener(市值降序) ∩ NASDAQ 交易所名单(强制为普通股) -> 取前 top."""
    if not LIST_CSV.exists():
        print('[cap] 先跑 --stage list'); return
    nas = pd.read_csv(LIST_CSV, encoding='utf-8')
    nas_syms = set(nas['Code'].astype(str))
    print(f'[cap] NASDAQ common_stock 全集 {len(nas_syms)}')

    sc = _screener_pull()
    if sc is None or len(sc) == 0:
        print('[cap] screener 空'); return
    sc['code'] = sc['code'].astype(str)
    sc['market_capitalization'] = pd.to_numeric(sc['market_capitalization'], errors='coerce')
    print(f'[cap] screener 共 {len(sc)} 行')

    keep = sc[sc['code'].isin(nas_syms)].copy()
    drop = sc[~sc['code'].isin(nas_syms)]['code'].tolist()
    print(f'[cap] 过滤非普通股(权证/单元/基金等)剔除 {len(drop)}: {drop[:20]}')
    print(f'[cap] 保留 {len(keep)} 只')

    keep = keep.sort_values('market_capitalization', ascending=False).reset_index(drop=True)
    keep['rank'] = keep.index + 1
    sel = keep.head(top).copy()
    out = sel[['code', 'name', 'market_capitalization', 'rank', 'sector']]
    out.to_csv(RANK_CSV, index=False, encoding='utf-8')
    print(f'[cap] 选定前 {top} -> {RANK_CSV}')
    print(out.head(12).to_string())
    print('...')
    print(out.tail(6).to_string())
    print(f'[cap] 第 {top} 名市值={out["market_capitalization"].iloc[-1]/1e9:.1f}B USD')


def stage_check():
    """核查选定名单的本地 CSV 覆盖."""
    sel = pd.read_csv(RANK_CSV, encoding='utf-8')
    syms = sel['code'].astype(str).tolist()
    have = [s for s in syms if (STOCK / f'{s}.csv').exists()]
    miss = [s for s in syms if not (STOCK / f'{s}.csv').exists()]
    print(f'[check] 选定 {len(syms)}  本地已有 {len(have)}  缺 {len(miss)}')
    if miss:
        print('[check] 缺失:', ' '.join(miss))
        (QUANT / 'nasdaq_500_missing.txt').write_text('\n'.join(miss), encoding='utf-8')
        print(f'[check] 缺失名单 -> {QUANT}/nasdaq_500_missing.txt')


def stage_fetch():
    """补下载缺失标的 OHLCV 到 stock_data/."""
    p = QUANT / 'nasdaq_500_missing.txt'
    if not p.exists():
        print('[fetch] 无缺失名单, 跳过'); return
    syms = [s.strip() for s in p.read_text(encoding='utf-8').split('\n') if s.strip()]
    if not syms:
        print('[fetch] 名单为空, 跳过'); return
    print(f'[fetch] 下载 {len(syms)} 个标的 -> {STOCK}')
    res = io.download_eod_parallel(syms, stock_data_path=str(STOCK), sources=io.default_data_sources,
                                   api_keys={'eod': KEY}, update_mode='eod', is_print=True, max_workers=5)
    print(f'[fetch] 完成, 返回 {len(res) if res else 0} 项')

    # 复核仍缺失的, 写回 missing.txt(便于下一轮续跑)
    still = [s for s in syms if not (STOCK / f'{s}.csv').exists()]
    (QUANT / 'nasdaq_500_missing.txt').write_text('\n'.join(still), encoding='utf-8')
    print(f'[fetch] 仍缺 {len(still)} 个 -> {QUANT}/nasdaq_500_missing.txt')


def stage_reg():
    """注册池: selected_sec_list.json(名单) + factor_config.json(仿 us_500 结构)."""
    sel = pd.read_csv(RANK_CSV, encoding='utf-8')
    syms = sorted(sel['code'].astype(str).str.upper().tolist())

    # 1) selected_sec_list.json
    with open(SEL_JSON, encoding='utf-8') as f:
        lst = json.load(f)
    lst[POOL] = syms
    lst[f'{POOL}_update_time'] = datetime.date.today().isoformat()
    with open(SEL_JSON, 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False)
    print(f'[reg] selected_sec_list.json: {POOL} = {len(syms)} 只')

    # 2) factor_config.json(结构/reuse 权重参照 us_500)
    with open(FACTOR_CONF, encoding='utf-8') as f:
        conf = json.load(f)
    base = json.loads(json.dumps(conf['pools']['us_500']))   # 深拷贝
    base['meta']['pool'] = POOL
    base['meta']['generated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    base['meta']['source'] = f'manual:from us_500 frozen weights (pool={POOL})'
    base['meta']['data_start'] = None
    base['meta']['data_end'] = None
    conf['pools'][POOL] = base
    with open(FACTOR_CONF, 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)
    print(f'[reg] factor_config.json: 新增 pools["{POOL}"] '
          f'(kept={len(base["kept"])}, scheme=cd_ic, top_k={base["params"]["top_k"]})')


def stage_regen():
    """生成研究版 pkl: 复用 tools/regen_ta_pkl.py 流程."""
    if POOL not in json.load(open(SEL_JSON, encoding='utf-8')):
        print(f'[regen] 池 {POOL} 未注册, 先跑 --stage reg'); return
    cmd = [sys.executable, str(REPO / 'quant' / 'tools' / 'regen_ta_pkl.py'),
           '--pool', POOL, '--interval', 'day', '--out', str(RESEARCH_PKL)]
    print('[regen] ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=False)


def stage_verify():
    """验证: load_pool 加载研究版 pkl + 复用冻结生产权重(cd_ic)回测 IS/OOS/全窗."""
    from quant.factor import combine as cmb
    from quant.factor import backtest as bt
    from quant.factor import evaluate as ev
    from quant.factor import data as fdata
    from quant.factor.data import window as _win

    conf = json.load(open(FACTOR_CONF, encoding='utf-8'))['pools'][POOL]
    kept = conf['kept']
    params = conf['params']
    scheme = params.get('default_scheme', 'cd_ic')
    weights = conf['schemes'][scheme]
    method = params.get('prep', 'rank')
    min_cs, min_factors = params.get('min_cs', 8), params.get('min_factors', 5)

    ds = fdata.load_pool(POOL, 'day')
    print('[verify] ' + (ds.summary() if hasattr(ds, 'summary') else f'loaded {POOL}'))
    print(f'[verify] frozen scheme={scheme} kept={len(kept)} top_k={params["top_k"]} exit_rank={params["exit_rank"]}')

    sigs = cmb.build_signals(ds, method, kept, min_cs)
    comb = cmb.combine_signals(sigs, weights, min_cs, min_factors)
    ow, cw = ds.wide('open'), ds.wide('close')
    p = bt.EngineParams(top_k=params['top_k'], exit_rank=params['exit_rank'],
                        capital=params.get('capital'))

    windows = [('IS', ev.window_bounds('is', pool=POOL)),
               ('OOS', ev.window_bounds('oos', pool=POOL)),
               ('FULL', (None, None))]
    for tag, (s, e) in windows:
        o, c, wc = _win(ow, s, e), _win(cw, s, e), _win(comb, s, e)
        if len(o) < 2:
            print(f'[verify] {tag}: 数据不足, 跳过'); continue
        res = bt.run_engine(o, c, wc, p)
        st = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
        bh_res = bt.buyhold(o, c)
        bh = bt.perf_stats(bh_res['equity'], bh_res['ann_turnover'], bh_res['trades'])
        print(f'[verify] {tag:4s} {str(s):10s}~{str(e):10s} '
              f'strat: ret={st["total_ret"]:+.2f} cagr={st["cagr"]:+.2f} sharpe={st["sharpe"]} '
              f'dd={st["max_dd"]} n_trades={st["n_trades"]} | bh ret={bh["total_ret"]:+.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['list', 'cap', 'check', 'fetch', 'reg', 'regen', 'verify'])
    ap.add_argument('--top', type=int, default=500)
    a = ap.parse_args()
    if a.stage == 'list':
        stage_list()
    elif a.stage == 'cap':
        stage_cap(a.top)
    elif a.stage == 'check':
        stage_check()
    elif a.stage == 'fetch':
        stage_fetch()
    elif a.stage == 'reg':
        stage_reg()
    elif a.stage == 'regen':
        stage_regen()
    elif a.stage == 'verify':
        stage_verify()


if __name__ == '__main__':
    main()