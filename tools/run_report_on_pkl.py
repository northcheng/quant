# -*- coding: utf-8 -*-
"""
用"重算后的 pkl"跑同一套 factor 流水线(走完全相同的代码路径, 只换数据源).

做法: 在导入 report 模块后, 把它的 load_pool 引用替换成"固定 pkl_path"的包装版,
其余一切(窗口/因子/回测/报告)完全不变。这样 A/B 只差一个数据文件。

用法:
  python quant/tools/run_report_on_pkl.py --pkl <path> --pool etf_3x
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant.factor import data as fdata
from quant.factor import report as rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--pool', default='etf_3x')
    ap.add_argument('--interval', default='day')
    # 其余参数(如 --write-config / --cost-ladder / --no-fig)一律透传给 report。
    # 注意: 不能用 nargs='*' 的位置参数 —— argparse 不允许在它之后再出现 --flag。
    a, extra = ap.parse_known_args()

    pkl = str(Path(a.pkl))
    if not Path(pkl).exists():
        raise SystemExit(f'pkl 不存在: {pkl}')

    _orig = rep.load_pool

    def patched(pool, interval='day', **kw):
        # 强制用指定 pkl, 忽略调用方传的 pool 解析
        kw.pop('pkl_path', None)
        kw.pop('research', None)
        return _orig(pool, interval, pkl_path=pkl, **kw)

    rep.load_pool = patched
    sys.argv = ['report', '--pool', a.pool, '--interval', a.interval] + list(extra)
    print(f'[wrapper] 数据源固定为 {pkl}')
    print(f'[wrapper] 透传参数: {extra}')
    rep.main()


if __name__ == '__main__':
    main()
