# -*- coding: utf-8 -*-
import math
import sympy
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
# from quant import finance_util

# rank
# cumsum
#----------------------------- 基础运算 -----------------------------------#
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


#----------------------------- 移动窗口 -----------------------------------#
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


#----------------------------- 信号计算 -----------------------------------#
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


#----------------------------- 均值回归模型 -----------------------------------#
# 计算涨跌幅/累计涨跌幅
def cal_change_rate(df, dim, period=1, add_accumulation=True, add_prefix=False):
  
    # 复制 dataframe
    df = df.copy()
  
    # 设置列名前缀
    prefix = ''
    if add_prefix:
        prefix = dim + '_'

    # 设置列名
    rate_dim = prefix + 'rate'
    acc_rate_dim = prefix + 'acc_rate'
    acc_day_dim = prefix + 'acc_day'

    # 计算涨跌率
    df[rate_dim] = df[dim].pct_change(periods=period)
  
    # 计算累计维度列
    if add_accumulation:
        df[acc_rate_dim] = 0
        df.loc[df[rate_dim]>0, acc_day_dim] = 1
        df.loc[df[rate_dim]<0, acc_day_dim] = -1
  
        # 计算累计值
        idx = df.index.tolist()
        for i in range(1, len(df)):
            current_idx = idx[i]
            previous_idx = idx[i-1]
            current_rate = df.loc[current_idx, rate_dim]
            previous_acc_rate = df.loc[previous_idx, acc_rate_dim]
            previous_acc_days = df.loc[previous_idx, acc_day_dim]

            # 如果符号相同则累加, 否则重置
            if previous_acc_rate * current_rate > 0:
                df.loc[current_idx, acc_rate_dim] = current_rate + previous_acc_rate
                df.loc[current_idx, acc_day_dim] += previous_acc_days
            else:
                df.loc[current_idx, acc_rate_dim] = current_rate

    df.dropna(inplace=True) 

    return df


# 计算当前值与移动均值的差距离移动标准差的倍数
def cal_mean_reversion(df, dim, window_size=100, window_type='sm', start_date=None, end_date=None):
  
    # 日收益率计算
    original_columns = df.columns
    data = cal_change_rate(df=df, dim=dim, period=1, add_accumulation=True)[start_date:end_date]
  
    # 选择移动窗口类型
    if window_type == 'em':
        mw_func = em
    else:
        mw_func = sm

    # 计算变化率/累计变化率/累计天数: (偏离均值的距离)/方差
    new_columns = [x for x in data.columns if x not in original_columns]
    for d in new_columns:
    
        # 计算累计变化率的移动平均及移动标准差
        mw = mw_func(series=data[d], periods=window_size)
        tmp_mean = mw.mean()
        tmp_std = mw.std()
    
        # 计算偏差
        data[d+'_bias'] = (data[d] - tmp_mean) / (tmp_std)
  
    return data


# 计算均值回归信号
def cal_mean_reversion_signal(df, time_std=2, triger_dim=['rate_bias', 'acc_rate_bias', 'acc_day_bias'], triger_threshold=2, start_date=None, end_date=None):
  
    # 复制 dataframe
    mr_df = df.copy()

    # 选择包含 'bias' 的列
    target_dim = [x for x in mr_df.columns if 'bias' in x]
    for t in triger_dim:
        if t not in target_dim:
            print(t, 'not found in columns!')
            triger_dim = [x for x in triger_dim if x != t]

    # 初始化信号
    mr_df['signal'] = 0

    # 计算每种 bias 的信号
    for dim in triger_dim:
        signal_dim = dim.replace('bias', 'signal')
        mr_df[signal_dim] = 0
    
        # 超买信号
        mr_df.loc[mr_df[dim] > time_std, signal_dim] = 1
    
        # 超卖信号
        mr_df.loc[mr_df[dim] < -time_std, signal_dim] = -1

        # 综合信号
        mr_df['signal'] = mr_df['signal'] + mr_df[signal_dim]
  
        # 将信号从数字转化为字符  
        sell_signals = mr_df.loc[mr_df['signal'] >= triger_threshold, ].index
        buy_signals = mr_df.loc[mr_df['signal'] <= -triger_threshold, ].index
        mr_df['signal'] = 'n'
        mr_df.loc[sell_signals, 'signal'] = 's'
        mr_df.loc[buy_signals, 'signal'] = 'b'
  
    return mr_df


# 计算触发信号所需的累积涨跌
def cal_mean_reversion_expected_rate(df, rate_dim, window_size, time_std):
  
    x = sympy.Symbol('x')
  
    rate_data = np.hstack((df.tail(window_size-1)[rate_dim].values, x))
    ma = rate_data.mean()
    std = sympy.sqrt(sum((rate_data - ma)**2)/(window_size-1))
    result = sympy.solve(((x - ma)**2) - ((time_std*std)**2), x)
  
    return result


# 画出均值回归偏差图
def plot_mean_reversion(df, times_std, window_size, start_date=None, end_date=None):
  
    # 需要绘出的维度
    plot_dims = [x for x in df.columns if '_bias' in x]
    
    # 创建图片
    plt.figure()
    plot_data = df[plot_dims][:end_date].tail(window_size)
    plot_data['upper'] = times_std
    plot_data['lower'] = -times_std
  
    # 画出信号
    plot_data.plot(figsize=(20, 3))
    plt.legend(loc='best')


#----------------------------- 均线模型 -----------------------------------#
# 计算移动平均信号
def cal_moving_average(df, dim, ma_windows=[50, 105], start_date=None, end_date=None, window_type='em'):

    # 截取数据  
    df = df[start_date:end_date].copy()

    # 选择移动窗口类型
    if window_type == 'em':
        mw_func = em
    else:
        mw_func = sm

    # 计算移动平均
    for mw in ma_windows:
        ma_dim = '%(dim)s_ma_%(window_size)s' % dict(dim=dim, window_size=mw)
        df[ma_dim] = mw_func(series=df[dim], periods=mw).mean()
    
    return df

# 计算移动平均信号
def cal_moving_average_signal(ma_df, dim, short_ma, long_ma, short_ma_col=None, long_ma_col=None, start_date=None, end_date=None):
  
    ma_df = ma_df.copy()
  
    if short_ma_col is None:
        short_ma = '%(dim)s_ma_%(window_size)s' %dict(dim=dim, window_size=short_ma)
    else:
        short_ma = short_ma_col

    if long_ma_col is None:
        long_ma = '%(dim)s_ma_%(window_size)s' %dict(dim=dim, window_size=long_ma)
    else:
        long_ma = long_ma_col

    if long_ma not in ma_df.columns or short_ma not in ma_df.columns:
        print("%(short)s or %(long)s on %(dim)s not found" % dict(short=short_ma, long=long_ma, dim=dim))
        return pd.DataFrame()
  
    # 计算长短均线之差
    ma_df['ma_diff'] = ma_df[short_ma] - ma_df[long_ma]
  
    # 计算信号
    ma_df['signal'] = 'n'
    last_value = None
    for index, row in ma_df[start_date : end_date].iterrows():
    
        # 当前与之前的长短期均线差值
        current_value = row['ma_diff']
    
        if last_value is None:
            last_value = current_value
            continue
    
        # 短线从下方穿过长线, 买入
        if last_value < 0 and current_value > 0:
            ma_df.loc[index, 'signal'] = 'b'

        # 短线从上方穿过长线, 卖出
        elif last_value > 0 and current_value < 0:
            ma_df.loc[index, 'signal'] = 's'
      
            last_value = current_value
    
    return ma_df

# 画出移动平均图
def plot_moving_average(df, dim, short_ma, long_ma, window_size, start_date=None, end_date=None):
  
    plot_dims = ['Close']
  
    short_ma = '%(dim)s_ma_%(window_size)s' %dict(dim=dim, window_size=short_ma)
    long_ma = '%(dim)s_ma_%(window_size)s' %dict(dim=dim, window_size=long_ma)
    if long_ma not in df.columns or short_ma not in df.columns:
        print("%(short)s or %(long)s on %(dim)s not found" % dict(short=short_ma, long=long_ma, dim=dim))
    
    else:
        plot_dims += [long_ma, short_ma]
  
    # 创建图片
    plt.figure()
    plot_data = df[plot_dims][start_date:end_date].tail(window_size)
  
    # 画出信号
    plot_data.plot(figsize=(20, 3))
    plt.legend(loc='best')



#----------------------------- 技术指标 -----------------------------------#
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


# # Ichimoku
# def cal_ichimoku(df, method='original'):
  
#     data = df.copy()
  
#     if method == 'original':
#         data = finance_util.cal_moving_average(df=data, dim='High', ma_windows=[9, 26, 52], window_type='sm')
#         data = finance_util.cal_moving_average(df=data, dim='Low', ma_windows=[9, 26, 52], window_type='sm')

#         data['tankan'] = (data['High_ma_9'] + data['Low_ma_9']) / 2
#         data['kijun'] = (data['High_ma_26'] + data['Low_ma_26']) / 2
#         data['senkou_a'] = (data['tankan'] + data['kijun']) / 2
#         data['senkou_b'] = (data['High_ma_52'] + data['Low_ma_52']) / 2
#         data['chikan'] = data.Close.shift(-26)
    
#     elif method == 'ta':
#         data['tankan'] = (data.High.rolling(9, min_periods=0).max() + data.Low.rolling(9, min_periods=0).min()) / 2
#         data['kijun'] = (data.High.rolling(26, min_periods=0).max() + data.Low.rolling(26, min_periods=0).min()) / 2
#         data['senkou_a'] = (data['tankan'] + data['kijun']) / 2
#         data['senkou_b'] = (data.High.rolling(52, min_periods=0).max() + data.Low.rolling(52, min_periods=0).min()) / 2
#         data['chikan'] = data.Close.shift(-26)
  
#     data = ta_util.cal_joint_signal(data=data, positive_col='senkou_a', negative_col='senkou_b')
#     return data


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
