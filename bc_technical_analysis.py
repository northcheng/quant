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

# # 计算SMA
# def sma(series, periods, fillna=False):
#     if fillna:
#         return series.rolling(window=periods, min_periods=0).mean()
#     return series.rolling(window=periods, min_periods=periods).mean()

# # 计算EMA
# def ema(series, periods, fillna=False):
#     if fillna:
#         return series.ewm(span=periods, min_periods=0).mean()
#     return series.ewm(span=periods, min_periods=periods).mean()

def cal_joint_signal(data, positive_col, negative_col):

    data = data.copy()

    # 计算两条线之间的差
    data['diff'] = data[positive_col] - data[negative_col]
  
    # 计算信号
    data['signal'] = 'n'
    last_value = None
    for index, row in data.iterrows():
    
        # 判断前值是否存在
        current_value = row['diff']
        if last_value is None:
            last_value = current_value
            continue
    
        # 正线从下往上穿越负线, 买入
        if last_value < 0 and current_value > 0:
            data.loc[index, 'signal'] = 'b'

        # 正线从上往下穿越负线, 卖出
        elif last_value > 0 and current_value < 0:
            data.loc[index, 'signal'] = 's'
      
        last_value = current_value
    
    return data




