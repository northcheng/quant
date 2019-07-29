# -*- coding: utf-8 -*-
import math
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta



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

def cal_peak_trough(df, target_col, result_col='signal', peak_signal='p', trough_signal='t'):
  
    df = df.copy()
    previous_target_col = 'previous_' + target_col
    df[previous_target_col] = df[target_col].shift(1)
    peaks = df.query('%(t)s < %(pt)s' % dict(t=target_col, pt=previous_target_col)).index
    troughs = df.query('%(t)s > %(pt)s' % dict(t=target_col, pt=previous_target_col)).index

    df[result_col] = 'n'
    df.loc[peaks, result_col] = peak_signal
    df.loc[troughs, result_col] = trough_signal

    df[result_col] = df[result_col].shift(-1)
    df = remove_redundant_signal(signal=df, keep='first')
    print(df)
    return df
  
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


#----------------------------- 计算变化 -----------------------------------#
# 计算变化值/累计变化值
def cal_change(df, dim, period=1, add_accumulation=True, add_prefix=False):
  
    # 复制 dataframe
    df = df.copy()
  
    # 设置列名前缀
    prefix = ''
    if add_prefix:
        prefix = dim + '_'

    # 设置列名
    change_dim = prefix + 'change'
    acc_change_dim = prefix + 'acc_change'
    acc_change_day_dim = prefix + 'acc_change_count'

    # 计算涨跌率
    df[change_dim] = df[dim] - df[dim].shift(1)
  
    # 计算累计维度列
    if add_accumulation:
        df[acc_change_dim] = 0
        df.loc[df[change_dim]>0, acc_change_day_dim] = 1
        df.loc[df[change_dim]<0, acc_change_day_dim] = -1
  
        # 计算累计值
        idx = df.index.tolist()
        for i in range(1, len(df)):
            current_idx = idx[i]
            previous_idx = idx[i-1]
            current_change = df.loc[current_idx, change_dim]
            previous_acc_change = df.loc[previous_idx, acc_change_dim]
            previous_acc_change_days = df.loc[previous_idx, acc_change_day_dim]

            # 如果符号相同则累加, 否则重置
            if previous_acc_change * current_change > 0:
                df.loc[current_idx, acc_change_dim] = current_change + previous_acc_change
                df.loc[current_idx, acc_change_day_dim] += previous_acc_change_days
            else:
                df.loc[current_idx, acc_change_dim] = current_change

    df.dropna(inplace=True) 

    return df    


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


#----------------------------- 信号处理 -----------------------------------#
# 计算交叉信号
def cal_joint_signal(df, positive_col, negative_col):
    data = df.copy()

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


# 计算趋势信号
def cal_trend_signal(df, trend_dim, buy_window=3, sell_window=2, threshold=0.7):

    # 计算连续变化天数
    data = cal_change(df=df, dim=trend_dim)

    # 上涨趋势和下降趋势
    up_trend_idx = data.query('acc_change_count >= %s' % buy_window).index
    down_trend_idx = data.query('acc_change_count <= %s' % -sell_window).index

    # 设置信号
    data['signal'] = 'n'
    data.loc[up_trend_idx, 'signal'] = 'b'
    data.loc[down_trend_idx, 'signal'] = 's'

    # 移除冗余信号
    data = remove_redundant_signal(signal=data)

    return data


# 去除冗余信号
def remove_redundant_signal(signal, keep='first'):

    # copy signal
    clear_signal = signal.copy()

    # 初始化
    buy_sell_signals = clear_signal.query('signal != "n"')
    valid_signals = []
    last_signal = 'n'
    last_index = None
    
    # 遍历信号数据 
    for index, row in buy_sell_signals.iterrows():

        # 获取当前信号
        current_index = index
        current_signal = row['signal']  

        # 对比前后信号
        if keep == 'first':
            # 如果当前信号与上一信号不一致, 则记录信号
            if current_signal != last_signal:
                valid_signals.append(current_index)
        
        elif keep == 'last':
            # 如果当前信号与上一信号不一致, 则跳过
            if current_signal != last_signal:
                valid_signals.append(last_index)
        else:
            print('invalid method to keep signal: %s' % keep)
            break
              
        # 更新
        last_index = current_index
        last_signal = current_signal
      
    # 后处理
    if keep == 'last' and last_signal != 'n':
        valid_signals.append(last_index)
    valid_siganls = [x for x in valid_signals if x is not None]
    
    # 移除冗余的信号
    redundant_signals = [x for x in clear_signal.index.tolist() if x not in valid_signals]
    clear_signal.loc[redundant_signals, 'signal'] = 'n'



# 画出信号位置
def plot_signal(signal, signal_dim, price_dim='Close', pos_signal='b', neg_signal='s', none_signal='n', figsize=(20, 5), start=None, end=None, title=None):

    plot_data = signal[start:end]

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.plot(plot_data.index, plot_data[price_dim], color='black')

    positive_signal = plot_data.query('%(signal)s == "%(pos_signal)s"' % dict(signal=signal_dim, pos_signal=pos_signal))
    negative_signal = plot_data.query('%(signal)s == "%(neg_signal)s"' % dict(signal=signal_dim, neg_signal=neg_signal))
    ax.scatter(positive_signal.index, positive_signal[price_dim], marker='^', color='green', alpha=0.6)
    ax.scatter(negative_signal.index, negative_signal[price_dim], marker='v', color='red', alpha=0.6)

    plt.legend()  
    plt.title(title)


#----------------------------- 均值回归 -----------------------------------#
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
        mr_df['signal'] = mr_df['signal'].astype(int) + mr_df[signal_dim].astype(int)

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



#----------------------------- 移动平均 -----------------------------------#
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
def cal_moving_average_signal(ma_df, short_ma_col=None, long_ma_col=None, start_date=None, end_date=None):

    ma_df = ma_df[start_date:end_date].copy()

    ma_df = cal_joint_signal(df=ma_df, positive_col=short_ma_col, negative_col=long_ma_col)

    return ma_df


# 画出移动平均图
def plot_moving_average(df, short_ma, long_ma, window_size, start_date=None, end_date=None):
  
    plot_dims = ['Close']
  
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
    data = cal_joint_signal(df=data, positive_col='macd_diff', negative_col='zero')
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
        data = cal_moving_average(df=data, dim='High', ma_windows=[9, 26, 52], window_type='sm')
        data = cal_moving_average(df=data, dim='Low', ma_windows=[9, 26, 52], window_type='sm')

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
  
    data = cal_joint_signal(df=data, positive_col='senkou_a', negative_col='senkou_b')
    return data


# Ichimoku signal
def cal_ichimoku_signal(df):
    data = df.copy()
  
    # 云信号
    data['signal_cloud'] = 0
    buy_idx = data.query('Close > senkou_a and senkou_a >= senkou_b').index
    sell_idx = data.query('Close < senkou_a').index
    data.loc[buy_idx, 'signal_cloud'] = 1
    data.loc[sell_idx, 'signal_cloud'] = -1
  
    # 基准/转换信号
    data['signal_tankan_kijun'] = 0
    buy_idx = data.query('tankan > kijun').index
    sell_idx = data.query('tankan < kijun').index
    data.loc[buy_idx, 'signal_tankan_kijun'] = 1
    data.loc[sell_idx, 'signal_tankan_kijun'] = -1
  
    # 合并信号
    data['signal_sum'] = data['signal_cloud'] + data['signal_tankan_kijun']
    buy_idx = data.query('signal_sum == 2').index
    sell_idx = data.query('signal_sum == -2').index

    data['ichimoku_signal'] = 'n'
    data.loc[buy_idx, 'ichimoku_signal'] = 'b'
    data.loc[sell_idx, 'ichimoku_signal'] = 's'

    return data[['ichimoku_signal']]
  

# Plot Ichimoku
def plot_ichimoku(data, start=None, end=None, signal_dim=None, title=None, save_path=None):
  
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
  
    if signal_dim is not None:
        if signal_dim in plot_data.columns.tolist():
            buy_signal = plot_data.query('%s == "b"' % signal_dim)
            sell_signal = plot_data.query('%s == "s"' % signal_dim)
            ax.scatter(buy_signal.index, buy_signal.Close, marker='^', color='green', )
            ax.scatter(sell_signal.index, sell_signal.Close, marker='v', color='red', )
  
    plt.legend()  
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path + title + '.png')


# KST Oscillator
def cal_kst_signal(df):
    data = df.copy()
    data['kst'] = ta.kst(close=data.Close)
    data['kst_sig'] = ta.kst_sig(close=data.Close)
    data['0'] = 0

    data['kst_signal'] = cal_joint_signal(df=data, positive_col='kst', negative_col='kst_sig')
    
    return data[['kst_signal']]

#----------------------------- 技术指标可视化 -----------------------------------#
# 画出以benchmark为界的柱状图, 上升为绿, 下降为红
def plot_indicator_around_benchmark(data, target_col, benchmark=0, title=None, start_date=None, end_date=None, color_mode='up_down', plot_close=True, figsize=(20, 5)):

    # 拷贝数据创建图片
    plot_data = data[start_date:end_date].copy()
    fig = plt.figure(figsize=figsize)
    
    # 指标
    ax1 = fig.add_subplot(111)

    # 如果是上涨/下跌模式, 下跌为红, 上涨为绿
    if color_mode == 'up_down':  
        plot_data['color'] = 'red'
        previous_target_col = 'previous_'+target_col
        plot_data[previous_target_col] = plot_data[target_col].shift(1)
        plot_data.loc[plot_data[target_col] > plot_data[previous_target_col], 'color'] = 'green'
  
    # 如果是阈值模式, 阈值之下为红, 之上为绿
    elif color_mode == 'benchmark':
        plot_data['color'] = 'red'
        plot_data.loc[plot_data[target_col] > benchmark, 'color'] = 'green'

        # 画出阈值
        plot_data['benchmark'] = benchmark
        ax1.plot(plot_data.index, plot_data['benchmark'], color='black')
        
    else:
        print('Unknown Color Mode')
        return None

    # 画出指标数据
    ax1.bar(plot_data.index, height=plot_data[target_col], color=plot_data.color, alpha=0.5)
    ax1.set_title(title)

    # 添加收盘价
    if plot_close:
        ax2=ax1.twinx()
        ax2.plot(plot_data.Close, color='blue' )


# 画出指标与价格
def plot_indicator(data, target_col, title=None, start_date=None, end_date=None, plot_close=True, figsize=(20, 5)):
  
    plot_data = data[start_date:end_date].copy()

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(plot_data[target_col], color='red', alpha=0.5)
    ax1.set_title(title)
  
    if plot_close:
        ax2=ax1.twinx()
        ax2.plot(plot_data.Close, color='blue' ) 


#----------------------------- 蜡烛图维度 -----------------------------------#
# 为数据添加蜡烛图维度
def add_candle_dims_for_data(original_df):
  
  data = original_df.copy()
  
  # 影线范围
  data['shadow'] = (data['High'] - data['Low'])    
  
  # 实体范围
  data['entity'] = abs(data['Close'] - data['Open'])
  
  # 筛选涨跌
  up_idx = data.Open < data.Close
  down_idx = data.Open >= data.Close

  # 上影线/下影线
  data['upper_shadow'] = 0
  data['lower_shadow'] = 0
  data['candle_color'] = 0
  
  # 涨
  data.loc[up_idx, 'candle_color'] = 1
  data.loc[up_idx, 'upper_shadow'] = (data.loc[up_idx, 'High'] - data.loc[up_idx, 'Close'])
  data.loc[up_idx, 'lower_shadow'] = (data.loc[up_idx, 'Open'] - data.loc[up_idx, 'Low'])
  
  # 跌
  data.loc[down_idx, 'candle_color'] = -1
  data.loc[down_idx, 'upper_shadow'] = (data.loc[down_idx, 'High'] - data.loc[down_idx, 'Open'])
  data.loc[down_idx, 'lower_shadow'] = (data.loc[down_idx, 'Close'] - data.loc[down_idx, 'Low'])
  
  return data