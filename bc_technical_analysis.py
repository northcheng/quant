# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

# rank
# cumsum

# 除去NA值
def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)]  # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df

# 获取最大/最小值
def get_min_max(x1, x2, f='min'):
    if not np.isnan(x1) and not np.isnan(x2):
        if f == 'max':
            max(x1, x2)
        elif f == 'min':
            min(x1, x2)
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    else:
        return np.nan    


# 简单移动窗口
def sm(series, periods, fillna=False):
    if fillna:
        return series.rolling(window=periods, min_periods=0)
    return series.rolling(window=periods, min_periods=periods)

# 指数移动窗口
def em(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0)
    return series.ewm(span=periods, min_periods=periods)  

# 计算SMA
def sma(series, periods, fillna=False):
    if fillna:
        return series.rolling(window=periods, min_periods=0).mean()
    return series.rolling(window=periods, min_periods=periods).mean()

# 计算EMA
def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()







