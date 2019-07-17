# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
from quant import finance_util

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


# 计算交叉信号
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


# MACD(Moving Average Convergence Divergence)信号
def cal_macd_signal(df, n_fast=50, n_slow=105):
    data = df.copy()
    data['macd_diff']  = ta.macd_diff(close=data.Close, n_fast=n_fast, n_slow=n_slow)
    data['zero'] = 0
    data = ta_util.cal_joint_signal(data=data, positive_col='macd_diff', negative_col='zero')
    data.rename(columns={'signal': 'macd_signal'}, inplace=True)

    return data[['macd_signal']]   


# RSI(Relative Strength Index)信号
def cal_rsi_signal(df, n=14, up=70, low=30):
    # divergence / failed swing not implemented
    # only implemented up/low bound
    data = df.copy()
    data['rsi'] = ta.rsi(close=data.Close, n=n)
    data['rsi_signal'] = 'n'
    over_buy_idx = data.query('rsi > %(up)s' % dict(up=up)).index
    over_sell_idx = data.query('rsi < %(low)s' % dict(low=low)).index
  
    data.loc[over_buy_idx, 'rsi_signal'] = 's'
    data.loc[over_sell_idx, 'rsi_signal'] = 'b'
  
    return data[['rsi_signal']]


# Aroon Indicator 信号
def cal_aroon_signal(df, up=90, low=10):
    data = df.copy()
    data['aroon_up'] = ta.aroon_up(close=data.Close)
    data['aroon_down'] = ta.aroon_down(close=data.Close)
    data['aroon_signal'] = 'n'
  
    bull_idx = data.query('aroon_up > %(up)s and aroon_down < %(low)s' % dict(up=up, low=low)).index
    bear_idx = data.query('aroon_down > %(up)s and aroon_up < %(low)s' % dict(up=up, low=low)).index
  
    data.loc[bull_idx, 'aroon_signal'] = 'b'
    data.loc[bear_idx, 'aroon_signal'] = 's'
  
    return data[['aroon_signal']]    


# CCI(Commidity Channel Indicator)信号
def cal_cci_signal(df, up=200, low=-200):
    data = df.copy()
    data['cci'] = ta.cci(high=data.High, low=data.Low, close=data.Close)
    data['cci_signal'] = 'n'
    over_buy_idx = data.query('cci > %(up)s' % dict(up=up)).index
    over_sell_idx = data.query('cci < %(low)s' % dict(low=low)).index
  
    data.loc[over_buy_idx, 'cci_signal'] = 's'
    data.loc[over_sell_idx, 'cci_signal'] = 'b'
  
    return data[['cci_signal']]


# Ichimoku
def cal_ichimoku(df, method='original'):
  
    data = df.copy()
  
    if method == 'original':
        data = finance_util.cal_moving_average(df=data, dim='High', ma_windows=[9, 26, 52], window_type='sm')
        data = finance_util.cal_moving_average(df=data, dim='Low', ma_windows=[9, 26, 52], window_type='sm')

        data['tankan'] = (data['High_ma_9'] + data['Low_ma_9']) / 2
        data['kijun'] = (data['High_ma_26'] + data['Low_ma_26']) / 2
        data['senkou_a'] = (data['tankan'] + data['kijun']) / 2
        data['senkou_b'] = (data['High_ma_52'] + data['Low_ma_52']) / 2
        data['chikan'] = data.Close.shift(-26)
    
    elif method == 'ta':
        data['tankan'] = (data.High.rolling(9, min_periods=0).max() + data.Low.rolling(9, min_periods=0).min()) / 2
        data['kijun'] = (data.High.rolling(26, min_periods=0).max() + data.Low.rolling(26, min_periods=0).min()) / 2
        data['senkou_a'] = (data['tankan'] + data['kijun']) / 2
        data['senkou_b'] = (data.High.rolling(52, min_periods=0).max() + data.Low.rolling(52, min_periods=0).min()) / 2
        data['chikan'] = data.Close.shift(-26)
  
    data = ta_util.cal_joint_signal(data=data, positive_col='senkou_a', negative_col='senkou_b')
    return data


# Ichimoku signal
def cal_ichimoku_signal(df):
    data = df.copy()
  
    data['signal_cloud'] = 0
    buy_idx = data.query('Close > senkou_a and senkou_a >= senkou_b').index
    sell_idx = data.query('Close < senkou_a').index
    data.loc[buy_idx, 'signal_cloud'] = 1
    data.loc[sell_idx, 'signal_cloud'] = -1
  
    data['signal_tankan_kijun'] = 0
    buy_idx = data.query('tankan > kijun').index
    sell_idx = data.query('tankan < kijun').index
    data.loc[buy_idx, 'signal_tankan_kijun'] = 1
    data.loc[sell_idx, 'signal_tankan_kijun'] = -1
  
    data['signal_sum'] = data['signal_cloud'] + data['signal_tankan_kijun']
    buy_idx = data.query('signal_sum == 2').index
    sell_idx = data.query('signal_sum == -2').index

    data['signal'] = 'n'
    data.loc[buy_idx, 'signal'] = 'b'
    data.loc[sell_idx, 'signal'] = 's'
    return data
  

# Plot Ichimoku
def plot_ichimoku(data, start=None, end=None, plot_signal=True, title=None):
  
    plot_data = data[start:end]
  
    fig = plt.figure(figsize=(20, 5))
    ax = plt.gca()
    ax.plot(plot_data.index, plot_data.Close, color='black')
  
    ax.plot(plot_data.index, plot_data.tankan, color='magenta', linestyle='-.')
    ax.plot(plot_data.index, plot_data.kijun, color='blue', linestyle='-.')
  
    ax.plot(plot_data.index, plot_data.senkou_a, color='green')
    ax.plot(plot_data.index, plot_data.senkou_b, color='red')
  
    ax.fill_between(plot_data.index, plot_data.senkou_a, plot_data.senkou_b, where=plot_data.senkou_a > plot_data.senkou_b, facecolor='green', interpolate=True, alpha=0.6)
    ax.fill_between(plot_data.index, plot_data.senkou_a, plot_data.senkou_b, where=plot_data.senkou_a <= plot_data.senkou_b, facecolor='red', interpolate=True, alpha=0.6)
  
    if plot_signal:
        if 'signal' in plot_data.columns.tolist():
        buy_signal = plot_data.query('signal == "b"')
        sell_signal = plot_data.query('signal == "s"')
        ax.scatter(buy_signal.index, buy_signal.Close, marker='^', color='green', )
        ax.scatter(sell_signal.index, sell_signal.Close, marker='v', color='red', )
  
    plt.legend()  
    plt.title(title)
