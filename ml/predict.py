# -*- coding: utf-8 -*-
"""
ml.predict — 预测层: 出分数宽表 -> 复用 factor.backtest 出净值(第4步之三)
=========================================================================
职责: 用 train 定好的参数在 IS 上拟合(--model 选 ridge / gbdt / mlp / gru / lstm), 对全历史打分,
      然后**直接复用 factor.backtest 的引擎**把分数变成净值和绩效 —— 不另写一套回测.

--- 训练宇宙 ≠ 评估宇宙(--train-pool) ---
  默认训练宇宙 = 评估池(--pool). 若指定 --train-pool(可为合并池别名, 见 config.POOL_ALIASES),
  则: 特征/标签/拟合都在训练宇宙上做(截面 rank 在训练宇宙内计算), 打分得到训练宇宙的分数宽表,
  再**按评估池的网格切片**, 回测只推进评估池 —— 即"用大体量宇宙训练, 只在目标池交易".
  这样特征分布(rank 的分母)训练/推理一致, 而交易口径仍是目标池(TOP_K/EXIT_RANK 等保持可比).

--- 为什么复用 factor.backtest ---
  回测里最容易出错的地方(成交时点、成本、滞回、涨跌口径)在 factor 端已经被
  verify_backtest 标定过(oracle 大赚 / shuffle≈随机). ml 的分数宽表与 factor 的
  composite 同构, 直接喂进同一个引擎, 两边净值和绩效才逐字可比.

--- 一条必须知道的口径 ---
  分数宽表只在"有标签的样本日"上有值(末尾 h+1 日、薄截面日无值); 引擎在分数为
  NaN 的当日会把该票视为"无信号"而清出。这与 factor 体系里信号在薄日缺值的行为
  一致, 不是 ml 特有 bug。

--- 红线 ---
  拟合只在 IS 上做; OOS/recent 只用已定参数打分, 不参与任何拟合.
  ridge: 填补统计量 + 系数在 IS 上估; gbdt: 早停验证块也在 IS 内部用**时序**切分
  (dataset.holdout_split, 带 embargo=h+1), 绝不拿 OOS 选轮数.

自检:
  cd ~/git && python -m quant.ml.predict --pool etf_3x --alpha 100
  cd ~/git && python -m quant.ml.predict --pool etf_3x --model gbdt --source both
  cd ~/git && python -m quant.ml.predict --pool etf_3x --model lstm --source both
     (gru/lstm 共用同一套序列管道, 同口径对照)
  cd ~/git && python -m quant.ml.predict --pool etf_3x --timing --timing-regime
     (满仓 / +voltarget(连续波动率目标) / +regime(离散状态固定档位) 三臂同引擎对比)
  cd ~/git && python -m quant.ml.predict --pool etf_3x --timing --timing-regime --timing-score
     (再加 +scorebreadth(分数超阈值占比) / +scorestrength(TopK 平均分) 两臂, 用选股分数本身定敞口;
      详见 ml/timing.py 第 9 节)
  cd ~/git && python -m quant.ml.predict --pool etf_3x --train-pool us_merged --model gbdt --source factor
     (合并池训练/etf_3x 回测: 特征/标签/拟合在 1077 标的宇宙上做, 分数切回 etf_3x 后回测)
"""
import argparse

import pandas as pd

from quant.factor import backtest as bt
from quant.factor import config as fcfg
from quant.factor import evaluate as ev
from quant.factor.data import window as _win

from quant.ml import config as cfg
from quant.ml import timing
from quant.ml.data import load
from quant.ml.dataset import assemble, holdout_split
from quant.ml.models import make_model
from quant.ml.seq import build_sequences


def predict_scores(ds, model_name: str, alpha: float, fill: str, h: int, method: str,
                   prep: str, source: str, min_cs: int, valid_frac: float,
                   window_len: int = cfg.SEQ_WINDOW, stride: int = cfg.SEQ_STRIDE,
                   feature_names: list = None, log=None) -> pd.DataFrame:
  """
  在 IS 上拟合指定模型, 对全历史打分.

  :param model_name: 'ridge' / 'gbdt' / 'mlp' / 'gru' / 'lstm'(见 config.MODELS)
  :param alpha/fill: ridge 的超参
  :param valid_frac: gbdt/mlp/gru/lstm 早停验证块占 IS 交易日比例
  :param window_len/stride: 序列模型(gru/lstm)的窗口长度与信号日起步
  :param feature_names: 显式特征名(None = 按 source 自动取全量); 供特征剪枝消融用
  :returns: (date × symbol) 分数宽表, 索引/列对齐到 Dataset 的完整网格
  """
  if model_name in cfg.SEQ_MODELS:
    # 序列模型: 样本换成 D_{t-L+1}..D_t 的窗口; 特征来源沿用 source(默认 both, 与截面
    # 模型同口径). 含 TA 时 build_sequences 会打印 TA_CAVEAT —— TA 的 1 日前视不可验证,
    # 且在长度 L 的窗口下被放大 L 倍, 其 OOS 结论仅作参考.
    s = build_sequences(ds, feature_names, h, method, prep, min_cs,
                        window_len=window_len, stride=stride, source=source, log=log)
  else:
    s = assemble(ds, feature_names, h, method, prep, min_cs, feature_source=source, log=log)
  is_s = s.split('IS')
  # 红线: 权重/早停只在 IS 内部做. 所有模型(含 ridge)都在**同一时序切分**下的训练块 tr 上拟合,
  # 使模型间对比公平 —— 若 ridge 用全 IS 而其余用 tr, 两边的训练样本量/时段不同, 净值不可比.
  tr, va, info = holdout_split(is_s, valid_frac, embargo=h + 1)
  if log is not None:
    log.info(f'[predict] {model_name} 拟合块(train<= {info["train_end"]:%Y-%m-%d}, '
             f'embargo={info["embargo"]}, valid {info["valid_start"]:%Y-%m-%d}~'
             f'{info["valid_end"]:%Y-%m-%d})')
  if model_name == 'ridge':
    # 闭式解, 无早停: 直接在 tr 上拟合(与其余模型同训练块, 口径可比)
    model = make_model('ridge', alpha=alpha, fill=fill).fit(tr, log=log)
  else:
    # gbdt / mlp / gru / lstm: IS 内时序早停块, 判据统一为 RankIC(越大越好) ——
    # 低信噪比下 MSE 最小点不对应排序能力最强点(会停在第 1 轮), 故不用 MSE.
    fwd = ev.forward_return(ds, h)                 # 可交易前向收益(标签必须用未来)
    model = make_model(model_name).fit(tr, valid=va, log=log,
                                       valid_fwd=fwd, min_cs=min_cs,
                                       criterion='rank_ic')

  open_w = ds.wide('open')
  pred = model.predict(s).reindex(index=open_w.index, columns=open_w.columns)
  if log is not None:
    log.info(f'[predict] 分数宽表 {pred.shape[0]} 行 × {pred.shape[1]} 列, '
             f'有值占比 {float(pred.notna().to_numpy().mean()):.1%}')
  return pred


def main():
  ap = argparse.ArgumentParser(description='ml 预测层: 分数宽表 -> factor.backtest 出净值')
  ap.add_argument('--pool', default='etf_3x')
  ap.add_argument('--train-pool', default=None,
                  help='训练宇宙池名(默认 = --pool); 支持合并池别名(见 config.POOL_ALIASES), '
                       '如 --train-pool us_merged = 合并池训练 / --pool 为评估池回测')
  ap.add_argument('--interval', default='day')
  ap.add_argument('--h', type=int, default=cfg.DEFAULT_HORIZON)
  ap.add_argument('--method', default=cfg.DEFAULT_LABEL, choices=list(cfg.LABEL_METHODS))
  ap.add_argument('--prep', default=cfg.FEATURE_PREP, choices=('raw', 'z', 'rank', 'mad_z', 'q_z'))
  ap.add_argument('--model', default=cfg.DEFAULT_MODEL, choices=list(cfg.MODELS),
                  help='模型: ridge / gbdt / mlp / gru / lstm')
  ap.add_argument('--source', default=cfg.DEFAULT_FEATURE_SOURCE,
                  choices=list(cfg.FEATURE_SOURCES), help='特征来源: factor / ta / both'
                  '(序列模型 gru/lstm 与截面模型同口径, 默认 both)')
  ap.add_argument('--window-len', type=int, default=cfg.SEQ_WINDOW,
                  help='序列模型(gru/lstm)的输入窗口长度 L')
  ap.add_argument('--stride', type=int, default=cfg.SEQ_STRIDE,
                  help='序列模型(gru/lstm)的信号日起步(>1 降采样)')
  ap.add_argument('--alpha', type=float, default=cfg.RIDGE_ALPHA,
                  help='Ridge L2 强度(用 train.py 选出的值)')
  ap.add_argument('--fill', default=cfg.FILL_STRATEGY, choices=('median', 'mean', 'zero'))
  ap.add_argument('--valid-frac', type=float, default=cfg.GBDT_VALID_FRAC,
                  help='gbdt 早停验证块占 IS 交易日比例')
  ap.add_argument('--min-cs', type=int, default=fcfg.MIN_CS)
  # 交易规则(与 factor 口径一致; 同为参数, 严格说只在 IS 调 —— 这里沿用 config 默认)
  ap.add_argument('--top-k', type=int, default=fcfg.TOP_K)
  ap.add_argument('--exit-rank', type=int, default=fcfg.EXIT_RANK)
  ap.add_argument('--sizing', default='equal', choices=bt.SIZINGS)
  ap.add_argument('--cost-bps', type=float, default=fcfg.COST_BPS)
  ap.add_argument('--band', type=float, default=0.05)
  ap.add_argument('--strength', action='store_true',
                  help='路线A(权重与标的个数解耦): 用**截面 z 入场门槛 θ** 筛入场券 '
                       '(rank<=top_k 且 z>=θ; 达标标的少则持仓只数 < top_k, 空缺不回填), '
                       '持仓内按强度 0.5+max(z-θ,0) 加权, 配合更大的 per_symbol_cap 让单票可 >1 份')
  ap.add_argument('--strength-threshold', type=float, default=None,
                  help='路线A 截面 z 入场门槛 θ(仅 --strength 生效). 默认 None = 路线A2: '
                       '改用**池无关口径** θ_eff = baseline - margin, 其中 baseline 用 IS 估计'
                       '的"Top-K 平均截面 z"; 给了具体值则用该绝对 θ(路线A 原始口径)')
  ap.add_argument('--strength-margin', type=float, default=0.25,
                  help='路线A2 池无关门槛的余量 margin: θ_eff = IS(Top-K 平均 z) - margin; '
                       'margin 越大门槛越低、持仓越多(仅 --strength 且未显式给 --strength-threshold 生效)')
  ap.add_argument('--strength-cap', type=float, default=0.5,
                  help='路线A 单票权重上限(默认 0.5 = 2.5 份; 仅 --strength 生效)')
  ap.add_argument('--save', action='store_true', help='把分数宽表存到 ml/output/')
  # 择时层(第5步, 可选): 只改变"总仓位上多少", 不改选股; 详见 ml/timing.py
  ap.add_argument('--timing', action='store_true',
                  help='启用择时层: IS 内训练波动率预测器 -> 逐日降仓, 与满仓基线同引擎对照')
  ap.add_argument('--timing-alpha', type=float, default=cfg.RIDGE_ALPHA,
                  help='择时波动率预测器的 Ridge L2 强度')
  ap.add_argument('--timing-z-window', type=int, default=None,
                  help='择时市场特征滚动标准化窗口(None = TIMING_DD_WINDOW)')
  ap.add_argument('--timing-target-vol', type=float, default=None,
                  help='择时目标年化波动率(None = IS 中位数估计, 红线)')
  ap.add_argument('--timing-regime', action='store_true',
                  help='启用状态分类臂: IS 分位二值化未来波动率 -> 线性概率模型判 高/低波动 -> 固定档位')
  ap.add_argument('--timing-regime-q', type=float, default=cfg.TIMING_REGIME_THRESHOLD_Q,
                  help='波动率二值化分位(⚠ 只在 IS 内取, 红线)')
  ap.add_argument('--timing-regime-on', type=float, default=cfg.TIMING_REGIME_ON_EXPOSURE,
                  help='低波动状态敞口(默认满仓 1.0)')
  ap.add_argument('--timing-regime-off', type=float, default=cfg.TIMING_REGIME_OFF_EXPOSURE,
                  help='高波动状态敞口(默认降仓 0.3; 仍 >= 0, 不做空/不加杠杆)')
  ap.add_argument('--timing-regime-ret', action='store_true',
                  help='启用趋势状态臂: IS 分位二值化**未来收益** -> 线性概率模型判 涨/跌 -> '
                       '下跌降仓 / 上涨满仓(与 regime 臂同特征, 只换标签口径; 默认档位方向相反)')
  ap.add_argument('--timing-regime-riskon', action='store_true',
                  help='启用风险偏好(risk-on/risk-off)状态臂: IS 分位二值化**风险偏好价差**'
                       '(未来高波组 - 低波组超额收益) -> 线性概率模型判 risk-off/risk-on -> '
                       'risk-off 降仓 / risk-on 满仓(与 regime 臂同特征, 只换标签口径)')
  ap.add_argument('--timing-regime-riskonfeat', action='store_true',
                  help='把风险偏好**降级为特征**喂给现有 regime 臂: 在 regime(波动率状态)臂的'
                       '市场特征里追加 riskon(过去窗口"高波组-低波组"实现超额收益, 因果)一列; '
                       '对比"加特征"是否胜过基线 regime 臂')
  ap.add_argument('--timing-score', action='store_true',
                  help='启用信号分数阈值臂: 用**选股分数本身**定逐日总敞口, 一次产出两臂 —— '
                       'scorebreadth(超阈值占比) 与 scorestrength(TopK 平均分); '
                       '与 voltarget/regime 信息源不同(不看市场波动)')
  ap.add_argument('--timing-score-tau-q', type=float, default=cfg.TIMING_SCORE_TAU_Q,
                  help='scorebreadth 的分数阈值分位(⚠ 只在 IS 内取, 红线)')
  ap.add_argument('--timing-score-lo-q', type=float, default=cfg.TIMING_SCORE_LO_Q,
                  help='敞口线性映射下限分位(⚠ IS 内取)')
  ap.add_argument('--timing-score-hi-q', type=float, default=cfg.TIMING_SCORE_HI_Q,
                  help='敞口线性映射上限分位(⚠ IS 内取)')
  a = ap.parse_args()

  log = cfg.get_logger('ml.predict')
  # 训练宇宙(--train-pool, 默认 = 评估池 --pool). 两者不同时: 特征/标签/拟合都在训练宇宙上做,
  # 打分后把分数宽表切回评估池 -> 回测只在评估池推进(用大体量宇宙训练, 只在目标池交易).
  train_pool = a.train_pool or a.pool
  ds_train = load(train_pool, a.interval)
  ds = ds_train if train_pool == a.pool else load(a.pool, a.interval)
  # ⚠ 窗口归属: load() 会把"进程级当前池"(ml.config.set_current_pool)设成它刚加载的那个池,
  # 因此两次 load 之后当前池 = **评估池**. 但本脚本的 IS/OOS 切窗(assemble/holdout_split)
  # 是作用在**训练数据**上的, 所以这里显式把当前池改回训练池 —— 否则当两个池的
  # factor_config.json 窗口不同时, 训练样本会被评估池的窗口切, 静默错位.
  cfg.set_current_pool(ds_train.pool)
  if ds is not ds_train and cfg.windows_for(ds.pool) != cfg.windows_for(ds_train.pool):
    log.warning(f'[窗口] 训练池 {ds_train.pool} 与评估池 {ds.pool} 的 IS/OOS 窗口不同: '
                f'训练切窗用前者 {cfg.window_range("IS", pool=ds_train.pool)}'
                f'~{cfg.window_range("OOS", pool=ds_train.pool)}; '
                f'评估回测用后者 {cfg.window_range("IS", pool=ds.pool)}'
                f'~{cfg.window_range("OOS", pool=ds.pool)} —— 请确认这是你要的口径')
  log.info(f'训练宇宙 {ds_train.pool}: {len(ds_train.symbols)} 标的 × '
           f'{len(ds_train.dates)} 交易日')
  if ds is not ds_train:
    log.info(f'评估宇宙 {ds.pool}: {len(ds.symbols)} 标的 × {len(ds.dates)} 交易日 '
             f'(分数宽表打完后切回该宇宙再回测)')
  log.info(f'[口径] model={a.model}, h={a.h}, label={a.method}, prep={a.prep}, '
           f'source={a.source}, alpha={a.alpha:g}, fill={a.fill}')

  pred = predict_scores(ds_train, a.model, a.alpha, a.fill, a.h, a.method, a.prep,
                        a.source, a.min_cs, a.valid_frac, a.window_len, a.stride, log=log)
  open_w, close_w = ds.wide('open'), ds.wide('close')
  if ds is not ds_train:                       # 切回评估池网格(date × 评估池标的)
    pred = pred.reindex(index=open_w.index, columns=open_w.columns)
    log.info(f'[predict] 分数宽表切回评估宇宙 {ds.pool}: {pred.shape[0]} 行 × {pred.shape[1]} 列')
  if a.save:
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = ds.pool if ds.pool == ds_train.pool else f'{ds.pool}_train-{ds_train.pool}'
    fn = cfg.OUT_DIR / (f'scores_{tag}_{a.model}_{a.source}_h{a.h}'
                        f'_a{a.alpha:g}.csv')
    pred.to_csv(fn)
    log.info(f'[save] 分数宽表 -> {fn}')

  p = bt.EngineParams(top_k=a.top_k, exit_rank=a.exit_rank, sizing=a.sizing,
                      max_exposure=1.0, per_symbol_cap=0.30,
                      cost_bps=a.cost_bps, band=a.band)
  log.info('引擎: ' + p.brief())

  # 路线A(权重与标的个数解耦)参数集: sizing=strength + 截面 z 门槛 θ + 放宽的单票上限
  # θ 口径: 显式给 --strength-threshold -> 绝对口径(路线A 原始); 否则走路线A2 池无关口径:
  #   θ_eff = baseline - margin, baseline = IS 窗内"Top-K 平均截面 z"(池越大自动越高),
  #   故同一 margin 在不同规模池上对应相同分位强度. baseline 只在 IS 估计 -> 无前视.
  p_str = None
  if a.strength:
    if a.strength_threshold is not None:
      theta_eff = float(a.strength_threshold)
      log.info(f'[路线A] 绝对门槛 θ={theta_eff:g}')
    else:
      ws, we = cfg.window_range('IS', pool=ds.pool)
      base_z = bt.topk_z_baseline(pred, a.top_k, window=(ws, we))
      theta_eff = base_z - a.strength_margin
      log.info(f'[路线A2 池无关门槛] IS({ws}~{we}) Top-K 平均 z baseline={base_z:.3f}, '
               f'margin={a.strength_margin:g} -> θ_eff={theta_eff:.3f}')
    p_str = bt.EngineParams(top_k=a.top_k, exit_rank=a.exit_rank, sizing='strength',
                            max_exposure=1.0, per_symbol_cap=a.strength_cap,
                            cost_bps=a.cost_bps, band=a.band,
                            strength_threshold=theta_eff)
    log.info('引擎(路线A): ' + p_str.brief())

  # ---- 择时层(可选): 只用 IS 拟合, 得到全历史逐日敞口; 两臂共用同一引擎 ----
  expo_vol, expo_reg, expo_reg_ret, expo_reg_riskon = None, None, None, None
  expo_reg_riskonfeat = None
  if a.timing:
    log.info('\n[择时-波动率目标] 在 IS 内训练波动率预测器 -> 逐日敞口 e_t ∈ [0,1] '
             '(方案B: 波动率高则降仓; 不做空/不加杠杆)')
    expo_vol = timing.build_exposure(ds, h=a.h, alpha=a.timing_alpha,
                                     z_window=a.timing_z_window,
                                     target_vol=a.timing_target_vol, log=log)
  if a.timing_regime:
    log.info('\n[择时-状态分类] IS 分位二值化未来波动率 -> 线性概率模型判状态 -> 固定档位 '
             '(低波动满仓 / 高波动降仓; 不做空/不加杠杆)')
    expo_reg = timing.build_regime_exposure(ds, h=a.h, alpha=a.timing_alpha,
                                            z_window=a.timing_z_window, q=a.timing_regime_q,
                                            on_exposure=a.timing_regime_on,
                                            off_exposure=a.timing_regime_off, log=log)
  if a.timing_regime_ret:
    log.info('\n[择时-趋势状态] IS 分位二值化**未来收益** -> 线性概率模型判状态 -> 固定档位 '
             '(下跌降仓 / 上涨满仓; 不做空/不加杠杆)')
    expo_reg_ret = timing.build_regime_exposure(ds, h=a.h, alpha=a.timing_alpha,
                                                z_window=a.timing_z_window, q=a.timing_regime_q,
                                                on_exposure=a.timing_regime_on,
                                                off_exposure=a.timing_regime_off,
                                                kind='ret', log=log)
  if a.timing_regime_riskon:
    log.info('\n[择时-风险偏好状态] IS 分位二值化**风险偏好价差**(未来高波组-低波组超额收益) '
             '-> 线性概率模型判 risk-off/risk-on -> 固定档位 '
             '(risk-off 降仓 / risk-on 满仓; 不做空/不加杠杆)')
    expo_reg_riskon = timing.build_regime_exposure(ds, h=a.h, alpha=a.timing_alpha,
                                                   z_window=a.timing_z_window, q=a.timing_regime_q,
                                                   on_exposure=a.timing_regime_on,
                                                   off_exposure=a.timing_regime_off,
                                                   kind='riskon', log=log)
  if a.timing_regime_riskonfeat:
    log.info('\n[择时-风险偏好特征] 把风险偏好**降级为特征**: 在 regime(波动率状态)臂的市场特征里'
             '追加 riskon(过去窗口"高波组-低波组"实现超额收益, 因果)一列, 标签仍为波动率状态; '
             '对照基线 regime 臂看"加特征"是否有效')
    expo_reg_riskonfeat = timing.build_regime_exposure(ds, h=a.h, alpha=a.timing_alpha,
                                                       z_window=a.timing_z_window, q=a.timing_regime_q,
                                                       on_exposure=a.timing_regime_on,
                                                       off_exposure=a.timing_regime_off,
                                                       kind='vol', riskon_feat=True, log=log)
  # 信号分数阈值臂: 与上面两臂信息源不同 —— 只用选股分数本身(不看市场波动率/回撤)
  expo_sb, expo_ss, expo_sb_sh, expo_ss_sh = None, None, None, None
  if a.timing_score:
    log.info('\n[择时-信号分数阈值] 用选股分数本身构造逐日总敞口 '
             '(弱信号->降仓 / 强信号->满仓; 阈值与映射分位只在 IS 取, 不做空/不加杠杆)')
    kw = dict(tau_q=a.timing_score_tau_q, lo_q=a.timing_score_lo_q,
              hi_q=a.timing_score_hi_q, top_k=a.top_k)
    log.info('[分数-广度] stat=breadth: 每日"有效分数 > 阈值"的占比 -> 敞口')
    expo_sb = timing.build_score_exposure(pred, stat='breadth', **kw, log=log)
    log.info('[分数-强度] stat=topk_mean: 每日 TopK 平均分 -> 敞口')
    expo_ss = timing.build_score_exposure(pred, stat='topk_mean', **kw, log=log)
    # 诊断对照: 用"打乱后的分数"构造同款敞口(应与模型臂拉开, 否则敞口机制本身存疑)
    sh_full = bt.shuffle_composite(pred, seed=42)
    expo_sb_sh = timing.build_score_exposure(sh_full, stat='breadth', **kw, log=None)
    expo_ss_sh = timing.build_score_exposure(sh_full, stat='topk_mean', **kw, log=None)

  def arms_for(cp, sh):
    """一个分数宽表 -> 同引擎下的全部对照臂(选股不变, 只换敞口/信号/参数集).

    返回 4 元组 (kind, comp, expo, params): params 为该臂使用的 EngineParams
    (大多数臂用基线 p; 路线A 的 strength 臂用 p_str).
    """
    arms = [(a.model, cp, None, p)]
    if expo_vol is not None:
      arms.append((f'{a.model}+voltarget', cp, expo_vol, p))
    if expo_reg is not None:
      arms.append((f'{a.model}+regime', cp, expo_reg, p))
    if expo_reg_ret is not None:
      arms.append((f'{a.model}+regime_ret', cp, expo_reg_ret, p))
    if expo_reg_riskon is not None:
      arms.append((f'{a.model}+regime_riskon', cp, expo_reg_riskon, p))
    if expo_reg_riskonfeat is not None:
      arms.append((f'{a.model}+regime_riskonfeat', cp, expo_reg_riskonfeat, p))
    if expo_sb is not None:
      arms.append((f'{a.model}+scorebreadth', cp, expo_sb, p))
    if expo_ss is not None:
      arms.append((f'{a.model}+scorestrength', cp, expo_ss, p))
    if p_str is not None:
      arms.append((f'{a.model}+strength', cp, None, p_str))
    arms.append((f'shuffle({a.model})', sh, None, p))
    if expo_vol is not None:
      arms.append((f'shuffle({a.model})+voltarget', sh, expo_vol, p))
    if expo_reg is not None:
      arms.append((f'shuffle({a.model})+regime', sh, expo_reg, p))
    if expo_reg_ret is not None:
      arms.append((f'shuffle({a.model})+regime_ret', sh, expo_reg_ret, p))
    if expo_reg_riskon is not None:
      arms.append((f'shuffle({a.model})+regime_riskon', sh, expo_reg_riskon, p))
    if expo_reg_riskonfeat is not None:
      arms.append((f'shuffle({a.model})+regime_riskonfeat', sh, expo_reg_riskonfeat, p))
    if expo_sb_sh is not None:
      arms.append((f'shuffle({a.model})+scorebreadth', sh, expo_sb_sh, p))
    if expo_ss_sh is not None:
      arms.append((f'shuffle({a.model})+scorestrength', sh, expo_ss_sh, p))
    if p_str is not None:
      arms.append((f'shuffle({a.model})+strength', sh, None, p_str))
    arms.append(('buyhold', None, None, p))
    return arms

  # ---- 各窗独立推进引擎: 模型 / buyhold / shuffle / [各择时臂] ----
  rows = []
  for win in ('IS', 'OOS', 'recent'):
    ws, we = cfg.window_range(win)
    ow, cw, cp = _win(open_w, ws, we), _win(close_w, ws, we), _win(pred, ws, we)
    if len(ow) < 2:
      continue
    sh = bt.shuffle_composite(cp, seed=42)             # 每窗一次, 各 shuffle 臂复用同种子
    for kind, comp, exp, pp in arms_for(cp, sh):
      res = (bt.buyhold(ow, cw) if comp is None
             else bt.run_engine(ow, cw, comp, pp, exposure=exp))
      st = bt.perf_stats(res['equity'], res['ann_turnover'], res['trades'])
      rows.append({'window': win, 'kind': kind,
                   **{k: st.get(k) for k in bt.STAT_KEYS}})
  tab = pd.DataFrame(rows)
  log.info('\n[回测汇总] 分数只在 IS 上拟合; 各窗独立推进引擎(含单边成本 %d bps)'
           % a.cost_bps)
  log.info('\n' + tab.to_string(index=False))

  # ---- 择时敞口的量级(信号口径) ----
  expo_arms = [(tag, e) for tag, e in (('voltarget', expo_vol), ('regime', expo_reg),
                                       ('regime_ret', expo_reg_ret),
                                       ('regime_riskon', expo_reg_riskon),
                                       ('regime_riskonfeat', expo_reg_riskonfeat),
                                       ('scorebreadth', expo_sb),
                                       ('scorestrength', expo_ss)) if e is not None]
  if expo_arms:
    erows = []
    for tag, e in expo_arms:
      for win in ('IS', 'OOS', 'recent'):
        ws, we = cfg.window_range(win)
        v = _win(e.to_frame('e'), ws, we)['e']
        erows.append({'arm': tag, 'window': win, 'days': int(v.notna().sum()),
                      'e_mean': round(float(v.mean()), 4),
                      'e_min': round(float(v.min()), 4),
                      'e_max': round(float(v.max()), 4)})
    log.info('\n[择时敞口] 各臂各窗 e_t 日均(信号口径, 未含单票上限):')
    log.info('\n' + pd.DataFrame(erows).to_string(index=False))

  # ---- OOS 分年收益(诚实窗口): 全部对照臂并排 ----
  ws, we = cfg.window_range('OOS')
  ow, cw, cp = _win(open_w, ws, we), _win(close_w, ws, we), _win(pred, ws, we)
  sh = bt.shuffle_composite(cp, seed=42)
  cols = {}
  for kind, comp, exp, pp in arms_for(cp, sh):
    eq = (bt.buyhold(ow, cw)['equity'] if comp is None
          else bt.run_engine(ow, cw, comp, pp, exposure=exp)['equity'])
    cols[kind] = bt.yearly_returns(eq)
  log.info(f'\n[OOS 分年收益] {a.model}(alpha={a.alpha:g}) 及全部对照臂:')
  log.info('\n' + pd.DataFrame(cols).to_string())

  log.info('\n[说明] IS 是拟合段(净值偏乐观); OOS/recent 未参与拟合, 是诚实数字. '
           'shuffle 应显著弱于模型本身, 否则说明引擎或信号口径有疑. '
           '各择时臂只改总敞口(voltarget=连续波动率目标 / regime=波动率状态固定档位 / '
           'regime_ret=趋势状态固定档位 / regime_riskon=风险偏好状态固定档位 / '
           'regime_riskonfeat=波动率状态固定档位+风险偏好特征 / '
           'scorebreadth,scorestrength=用选股分数本身定敞口 / '
           'strength(路线A/A2)=截面 z 门槛 θ 筛入场券(持仓只数可 < top_k)+ 持仓内强度加权 '
           '(θ 口径: 默认路线A2 池无关 θ_eff=IS(TopK 平均 z)-margin; 显式给 --strength-threshold 则用绝对 θ)), '
           '与满仓版同分数、同引擎, 可直接对比; 各臂拟合/定标均只喂 IS. '
           '注: strength 臂用的是解耦参数集(sizing=strength, per_symbol_cap 放宽), '
           '与其余臂的等权/0.3 上限不同 —— 对比 baseline 与 shuffle+strength 判断该机制真伪.')


if __name__ == '__main__':
  main()