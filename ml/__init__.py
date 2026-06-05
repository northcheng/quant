# -*- coding: utf-8 -*-
"""
Machine Learning ranking layer for signal enhancement.

Reuses the existing ta_data.pkl (8 technical scores + OHLCV) as raw features,
adds market-relative and volatility-based features, then trains a lightgbm
model per (market, pool, horizon) to predict the probability of the price
moving up / neutral / down over the next N trading days.

Public entry points:
  - quant.ml.features.build_features
  - quant.ml.dataset.build_label
  - quant.ml.dataset.walk_forward_split
  - quant.ml.train.train_pool
  - quant.ml.train.train_single
  - quant.ml.predict.predict
  - quant.ml.predict.load_latest_model
  - quant.ml.evaluate.ic_score
  - quant.ml.evaluate.layered_backtest
  - quant.ml.integration.attach_ml_scores  <-- bridge to bc_technical_analysis

CLI quickstart (after requirements.txt install):
  python -m quant.ml.train --pool company_300 --market us --horizon 5 --train-date 2026-06-04
  python -m quant.ml.predict --pool company_300 --market us --horizon 5
  python -m quant.ml.evaluate --pool company_300 --market us --horizon 5

Integration with bc_technical_analysis.calculate_ta_signal:
    df = calculate_ta_signal(
        df, market='us', pool='company_300', horizon=5,
        config=config, signal_source='auto',
    )
  Appends 5 standardized columns:
    ml_proba_up / ml_proba_neutral / ml_proba_down / ml_signal / ml_score

:author: Beichen Chen
"""

from quant.ml.integration import attach_ml_scores, ML_COLUMNS, VALID_SOURCES

__all__ = [
    'attach_ml_scores',
    'ML_COLUMNS',
    'VALID_SOURCES',
]
