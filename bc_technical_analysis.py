# -*- coding: utf-8 -*-
import math
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta



#----------------------------- Basic calculation -----------------------------------#
def dropna(df):
  """
  Drop rows with "Nans" values

  :param df: original dfframe
  :returns: dfframe with Nans dropped
  :raises: none
  """
  df = df[df < math.exp(709)]  # big number
  df = df[df != 0.0]
  df = df.dropna()
  return df


def get_min_max(x1, x2, f='min'):
  """
  Get Max or Min value from 2 values

  :param x1: value 1
  :param x2: value 2
  :param f: which one do you want: max/min
  :returns: max or min value
  :raises:
  """    
  if not np.isnan(x1) and not np.isnan(x2):
    if f == 'max':
      max(x1, x2)
    elif f == 'min':
      min(x1, x2)
    else:
      raise ValueError('"f" variable value should be "min" or "max"')
  else:
    return np.nan    
    

#----------------------------- Rolling windows -------------------------------------#
def sm(series, periods, fillna=False):
  """
  Simple Moving Window

  :param series: series to roll
  :param periods: size of the moving window
  :param fillna: make the min_periods = 0
  :returns: a rolling window with window size 'periods'
  :raises: none
  """  
  if fillna:  
    return series.rolling(window=periods, min_periods=0)
  return series.rolling(window=periods, min_periods=periods)


def em(series, periods, fillna=False):
  """
  Exponential Moving Window

  :param series: series to roll
  :param periods: size of the moving window
  :param fillna: make the min_periods = 0
  :returns: an exponential weighted rolling window with window size 'periods'
  :raises: none
  """  
  if fillna:
    return series.ewm(span=periods, min_periods=0)
  return series.ewm(span=periods, min_periods=periods)  


#----------------------------- Change calculation ----------------------------------#
def cal_change(df, target_col, periods=1, add_accumulation=True, add_prefix=False):
  """
  Calculate change of a column with a sliding window
  
  :param df: original dfframe
  :param target_col: change of which column to calculate
  :param periods: calculate the change within the period
  :param add_accumulation: wether to add accumulative change in a same direction
  :param add_prefix: whether to add prefix for the result columns (when there are multiple target columns to calculate)
  """
  # copy dfframe
  df = df.copy()

  # set prefix for result columns
  prefix = ''
  if add_prefix:
    prefix = target_col + '_'

  # set result column names
  change_dim = prefix + 'change'
  acc_change_dim = prefix + 'acc_change'
  acc_change_day_dim = prefix + 'acc_change_count'

  # calculate change within the period
  df[change_dim] = df[target_col] - df[target_col].shift(periods)
  
  # calculate accumulative change in a same direction
  if add_accumulation:
    df[acc_change_dim] = 0
    df.loc[df[change_dim]>0, acc_change_day_dim] = 1
    df.loc[df[change_dim]<0, acc_change_day_dim] = -1
  
    # go through each row, add values with same symbols (+/-)
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current_change = df.loc[current_idx, change_dim]
      previous_acc_change = df.loc[previous_idx, acc_change_dim]
      previous_acc_change_days = df.loc[previous_idx, acc_change_day_dim]

      if previous_acc_change * current_change > 0:
        df.loc[current_idx, acc_change_dim] = current_change + previous_acc_change
        df.loc[current_idx, acc_change_day_dim] += previous_acc_change_days
      else:
        df.loc[current_idx, acc_change_dim] = current_change
  df.dropna(inplace=True) 

  return df    


def cal_change_rate(df, target_col, periods=1, add_accumulation=True, add_prefix=False):
  """
  Calculate change rate of a column with a sliding window
  
  :param df: original dfframe
  :param target_col: change rate of which column to calculate
  :param periods: calculate the change rate within the period
  :param add_accumulation: wether to add accumulative change rate in a same direction
  :param add_prefix: whether to add prefix for the result columns (when there are multiple target columns to calculate)
  """
  # copy dfframe
  df = df.copy()
  
  # set prefix for result columns
  prefix = ''
  if add_prefix:
    prefix = target_col + '_'

  # set result column names
  rate_dim = prefix + 'rate'
  acc_rate_dim = prefix + 'acc_rate'
  acc_day_dim = prefix + 'acc_day'

  # calculate change rate within the period
  df[rate_dim] = df[target_col].pct_change(periods=periods)
  
  # calculate accumulative change rate in a same direction
  if add_accumulation:
    df[acc_rate_dim] = 0
    df.loc[df[rate_dim]>0, acc_day_dim] = 1
    df.loc[df[rate_dim]<0, acc_day_dim] = -1
  
    # go through each row, add values with same symbols (+/-)
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current_rate = df.loc[current_idx, rate_dim]
      previous_acc_rate = df.loc[previous_idx, acc_rate_dim]
      previous_acc_days = df.loc[previous_idx, acc_day_dim]

      if previous_acc_rate * current_rate > 0:
        df.loc[current_idx, acc_rate_dim] = current_rate + previous_acc_rate
        df.loc[current_idx, acc_day_dim] += previous_acc_days
      else:
        df.loc[current_idx, acc_rate_dim] = current_rate
  df.dropna(inplace=True) 

  return df


#----------------------------- Signal processing -----------------------------------#
def cal_crossover_signal(df, fast_line, slow_line, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate signal generated from the crossover of 2 lines
  When fast line breakthrough slow line from the bottom, positive signal will be generated
  When fast line breakthrough slow line from the top, negative signal will be generated 

  :param df: original dffame which contains a fast line and a slow line
  :param fast_line: columnname of the fast line
  :param slow_line: columnname of the slow line
  :param result_col: columnname of the result
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: series of the result column
  :raises: none
  """
  df = df.copy()

  # calculate the distance between fast and slow line
  df['diff'] = df[fast_line] - df[slow_line]

  # calculate signal by going through the whole dfframe
  df[result_col] = none_signal
  last_value = None
  for index, row in df.iterrows():
  
    # whehter the last_value exists
    current_value = row['diff']
    if last_value is None:
      last_value = current_value
      continue

    # fast line breakthrough slow line from the bottom
    if last_value < 0 and current_value > 0:
      df.loc[index, result_col] = pos_signal

    # fast line breakthrough slow line from the top
    elif last_value > 0 and current_value < 0:
      df.loc[index, result_col] = neg_signal
  
    last_value = current_value
  
  return df[[result_col]]


def cal_trend_signal(df, trend_col, up_window=3, down_window=2, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate signal generated from trend change

  :param df: original dataframe
  :param trend_col: columnname of the trend value
  :param up_window: the window size to judge the up trend
  :param down_window: the window size to judge the down trned
  :param result_col: columnname of the result
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: series of the result column
  :raises: none   
  """
  # calculate the change of the trend column
  df = cal_change(df=df, target_col=trend_col)

  # up trend and down trend
  up_trend_idx = df.query('acc_change_count >= %s' % up_window).index
  down_trend_idx = df.query('acc_change_count <= %s' % -down_window).index

  # set signal
  df[result_col] = none_signal
  df.loc[up_trend_idx, result_col] = pos_signal
  df.loc[down_trend_idx, result_col] = neg_signal

  # remove redundant signals (duplicated continuous signals)
  df = remove_redundant_signal(df=df, signal_col=result_col, keep='first', pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal)

  return df[[result_col]]


def remove_redundant_signal(df, signal_col='signal', keep='first', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Remove redundant (duplicated continuous) signals, keep only the first or the last one

  :param df: signal dataframe
  :param signal_col: columnname of the signal value
  :param keep: which one to keep: first/last
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal  
  :returns: signal dataframe with redundant signal removed
  :raises: none
  """
  # copy dataframe
  df = df.copy()

  # initialization
  none_empty_signals = df.query('%(signal)s != "%(none_signal)s"' % dict(signal=signal_col, none_signal=none_signal))
  valid_signals = []
  last_signal = none_signal
  last_index = None
  
  # go through the none empty signals
  for index, row in none_empty_signals.iterrows():

    # get current signals
    current_index = index
    current_signal = row[signal_col]  

    # compare current signal with previous one
    if keep == 'first':
      # if current signal is different from the last one, keep the current one(first)
      if current_signal != last_signal:
        valid_signals.append(current_index)
    
    elif keep == 'last':
      # if current signal is differnet from the last one, keep the last one(last)
      if current_signal != last_signal:
        valid_signals.append(last_index)
    else:
      print('invalid method to keep signal: %s' % keep)
      break
          
    # update last_index, last_signal
    last_index = current_index
    last_signal = current_signal
    
  # post-processing
  if keep == 'last' and last_signal != none_signal:
    valid_signals.append(last_index)
  valid_siganls = [x for x in valid_signals if x is not None]
  
  # set redundant signals to none_signal
  redundant_signals = [x for x in df.index.tolist() if x not in valid_signals]
  df.loc[redundant_signals, signal_col] = none_signal

  return df


def plot_signal(df, signal_col, price_col='Close', pos_signal='b', neg_signal='s', none_signal='n', start=None, end=None, title=None, figsize=(20, 5)):
  """
  Plot signals along with the price

  :param df: dataframe with price and signal columns
  :param signal_col: columnname of the signal values
  :param price_col: columnname of the price values
  :param keep: which one to keep: first/last
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :param start: start row to plot
  :param end: end row to stop
  :param title: plot title
  :param figsize: figsize
  :returns: a signal plotted price chart
  :raises: none
  """
  # copy dataframe within the specific period
  df = df[start:end]

  # create figure
  fig = plt.figure(figsize=figsize)
  ax = plt.gca()

  # plot price
  ax.plot(df.index, df[price_col], color='black')

  # plot signals
  positive_signal = df.query('%(signal)s == "%(pos_signal)s"' % dict(signal=signal_col, pos_signal=pos_signal))
  negative_signal = df.query('%(signal)s == "%(neg_signal)s"' % dict(signal=signal_col, neg_signal=neg_signal))
  ax.scatter(positive_signal.index, positive_signal[price_col], marker='^', color='green', alpha=0.6)
  ax.scatter(negative_signal.index, negative_signal[price_col], marker='v', color='red', alpha=0.6)

  # legend and title
  plt.legend()  
  plt.title(title)


#----------------------------- Support/Resistant -----------------------------------#
def cal_peak_trough(df, target_col, result_col='signal', peak_signal='p', trough_signal='t', none_signal='n', further_filter=True):
  """
  Calculate the position (signal) of the peak/trough of the target column

  :param df: original dataframe which contains target column
  :param result_col: columnname of the result
  :param peak_signal: the value of the peak signal
  :param trough_signal: the value of the trough signal
  :param none_signal: the value of the none signal
  :further_filter: if the peak/trough value is higher/lower than the average of its former and later peak/trough values, this peak/trough is valid
  :returns: series of peak/trough signal column
  :raises: none
  """
  # copy dataframe
  df = df.copy()

  # previous value of the target column
  previous_target_col = 'previous_' + target_col
  df[previous_target_col] = df[target_col].shift(1)

  # when value goes down, it means it is currently at peak
  peaks = df.query('%(t)s < %(pt)s' % dict(t=target_col, pt=previous_target_col)).index
  # when value goes up, it means it is currently at trough
  troughs = df.query('%(t)s > %(pt)s' % dict(t=target_col, pt=previous_target_col)).index

  # set signal values
  df[result_col] = none_signal
  df.loc[peaks, result_col] = peak_signal
  df.loc[troughs, result_col] = trough_signal

  # shift the signal back 1 unit
  df[result_col] = df[result_col].shift(-1)

  # remove redundant signals
  df = remove_redundant_signal(df=df, signal_col=result_col, keep='first', pos_signal=peak_signal, neg_signal=trough_signal, none_signal=none_signal)

  # further filter the signals
  if further_filter:
      
    # get all peak/trough signals
    peak = df.query('%(r)s == "%(p)s"' % dict(r=result_col, p=peak_signal)).index.tolist()
    trough = df.query('%(r)s == "%(t)s"' % dict(r=result_col, t=trough_signal)).index.tolist()

    # filter peak signals
    for i in range(1, len(peak)-1):
      previous_idx = peak[i-1]
      current_idx = peak[i]
      next_idx = peak[i+1]

      if df.loc[current_idx, target_col] < ((df.loc[previous_idx, target_col]+df.loc[next_idx, target_col])/2):
        df.loc[current_idx, result_col] = none_signal

    # filter trough signals
    for i in range(1, len(trough)-1):        
      previous_idx = trough[i-1]
      current_idx = trough[i]
      next_idx = trough[i+1]

      if df.loc[current_idx, target_col] > ((df.loc[previous_idx, target_col]+df.loc[next_idx, target_col])/2):
        df.loc[current_idx, result_col] = none_signal

  return df[[result_col]]
  

#----------------------------- mean reversion --------------------------------------#
def cal_mean_reversion(df, target_col, window_size=100, start_date=None, end_date=None, window_type='sm'):
  """
  Calculate (current value - moving avg) / moving std

  :param df: original dataframe which contains target column
  :param window_size: window size of the moving window
  :param start_date: start row
  :param end_date: end row
  :window_type: which type of moving window is going to be used: sm/em
  :returns: dataframe with mean-reversion result columns
  :raises: none
  """
  # calculate change rate by day
  original_columns = df.columns
  df = cal_change_rate(df=df, target_col=target_col, periods=1, add_accumulation=True)[start_date:end_date]

  # select the type of moving window
  if window_type == 'em':
    mw_func = em
  elif window_type == 'sm':
    mw_func = sm
  else:
    print('Unknown type of moving window')
    return df

  # calculate the (current value - moving avg) / moving std
  new_columns = [x for x in df.columns if x not in original_columns]
  for d in new_columns:
    mw = mw_func(series=df[d], periods=window_size)
    tmp_mean = mw.mean()
    tmp_std = mw.std()
    df[d+'_bias'] = (df[d] - tmp_mean) / (tmp_std)

  return df


def cal_mean_reversion_signal(df, std_multiple=2, final_signal_threshold=2, start_date=None, end_date=None, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate signal from mean reversion data

  :param df: dataframe which contains mean reversion columns
  :param std_multiple: the multiple of moving std to triger signals
  :param final_signal_threshold: how many columns triger signals at the same time could triger the final signal
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param result_col: columnname of the result signal
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: dataframe with signal columns
  :raises: none
  """
  # copy dataframe
  df = df[start_date : end_date].copy()

  # check whether triger columns are in the dataframe
  triger_cols = [x for x in df.columns if 'bias' in x]
  for t in triger_cols:
    if t not in triger_cols:
      print(t, 'not found in columns!')
      triger_cols = [x for x in triger_cols if x != t]

  # initialization
  df[result_col] = 0

  # calculate signal for each triger column
  for col in triger_cols:
    signal_col = col.replace('bias', result_col)
    df[signal_col] = 0

    # over buy
    df.loc[df[col] > std_multiple, signal_col] = 1

    # over sell
    df.loc[df[col] < -std_multiple, signal_col] = -1

    # final signal
    df[result_col] = df[result_col].astype(int) + df[signal_col].astype(int)

  # conver final singal from int to specific values  
  sell_signals = df.loc[df[result_col] >= final_signal_threshold, ].index
  buy_signals = df.loc[df[result_col] <= -final_signal_threshold, ].index
  
  df[result_col] = none_signal
  df.loc[sell_signals, result_col] = neg_signal
  df.loc[buy_signals, result_col] = pos_signal

  return df[result_col]


def cal_mean_reversion_expected_rate(df, rate_col, window_size, std_multiple):
  """
  Calculate the expected rate change to triger mean-reversion signals

  :param df: original dataframe which contains rate column
  :param rate_col: columnname of the change rate values
  :param window_size: windowsize of the moving window
  :param std_multiple: the multiple of moving std to triger signals
  :returns: the expected up/down rate to triger signals
  :raises: none
  """
  x = sympy.Symbol('x')

  df = np.hstack((df.tail(window_size-1)[rate_col].values, x))
  ma = df.mean()
  std = sympy.sqrt(sum((df - ma)**2)/(window_size-1))
  result = sympy.solve(((x - ma)**2) - ((std_multiple*std)**2), x)

  return result


def plot_mean_reversion(df, std_multiple, window_size, start_date=None, end_date=None, title=''):
  """
  Plot mean reversion charts

  :param df: original dataframe which contains plot data
  :param std_multiple: the multiple of moving std to triger signals
  :param window_size: window_size of the data to plot
  :param start_date: start date of plot data
  :param end_date: end date of plot data
  :param title: title of the plot
  :returns: plot of mean-reversion
  :raises: none
  """
  
  # columns to be plotted
  plot_dims = [x for x in df.columns if '_bias' in x]
  
  # create figure
  plt.figure()

  # rows to be plotted
  df = df[plot_dims][:end_date].tail(window_size)

  # plot boundaries
  df['upper'] = std_multiple
  df['lower'] = -std_multiple

  # plot signals
  df.plot(figsize=(20, 3))

  # plot legend and title
  plt.legend(loc='best')
  plt.title(title)


#----------------------------- moving average --------------------------------------#
def cal_moving_average(df, target_col, ma_windows=[50, 105], start_date=None, end_date=None, window_type='em'):
  """
  Calculate moving average of the tarhet column with specific window size

  :param df: original dataframe which contains target column
  :param ma_windows: a list of moving average window size to be calculated
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param window_type: which moving window to be used: sm/em
  :returns: dataframe with moving averages
  :raises: none
  """
  # copy dataframe
  df = df[start_date:end_date].copy()

  # select moving window type
  if window_type == 'em':
    mw_func = em
  elif window_type == 'sm':
    mw_func = sm
  else:
    print('Unknown moving window type')
    return df

  # calculate moving averages
  for mw in ma_windows:
    ma_col = '%(target_col)s_ma_%(window_size)s' % dict(target_col=target_col, window_size=mw)
    df[ma_col] = mw_func(series=df[target_col], periods=mw).mean()
  
  return df


def cal_moving_average_signal(df, short_ma_col=None, long_ma_col=None, start_date=None, end_date=None):
  """
  Calculate moving avergae signals gernerated from fast/slow moving average crossover

  :param df: original dataframe which contains short/ling ma columns
  :param short_ma_col: columnname of the short ma
  :param long_ma_col: columnname of the long ma
  :param start_date: start date of the data
  :param end_date: end date of the data
  :returns: dataframe with ma crossover signal
  :raises: none
  """
  # copy dataframe
  df = df[start_date:end_date].copy()

  # calculate ma crossover signal
  df['signal'] = cal_crossover_signal(df=df, fast_line=short_ma_col, slow_line=long_ma_col)

  return df


def plot_moving_average(df, short_ma_col, long_ma_col, price_col, window_size, start_date=None, end_date=None):
  """
  Plot moving average chart

  :param df: original dataframe which contains ma columns
  :param short_ma_col: columnname of the short ma
  :param long_ma_col: columnname of the long ma
  :param price_col: columnname of the price
  :param window_size: window size of the data to be plotted
  :param start_date: start date of the data
  :param end_date: end date of the data
  :returns: plot of moving average
  :raises: none
  """
  # columns to be plotted
  plot_dims = ['Close']
  if long_ma_col not in df.columns or short_ma_col not in df.columns:
    print("%(short)s or %(long)s on %(dim)s not found" % dict(short=short_ma_col, long=long_ma_col, dim=dim))
  else:
    plot_dims += [long_ma_col, short_ma_col]

  # create figure
  plt.figure()

  # select data
  df = df[plot_dims][start_date:end_date].tail(window_size)

  # plot data
  df.plot(figsize=(20, 3))
  plt.legend(loc='best')


#----------------------------- technical indicators --------------------------------#
def cal_macd_signal(df, n_fast=50, n_slow=105):
  """
  Calculate MACD(Moving Average Convergence Divergence) signals

  :param df: original OHLCV dataframe
  :param n_fast: ma window of fast ma
  :param n_slow: ma window of slow ma
  :returns: macd signals
  :raises: none
  """
  df = df.copy()
  df['macd_diff']  = ta.macd_diff(close=df.Close, n_fast=n_fast, n_slow=n_slow)
  df['zero'] = 0
  df['signal'] = cal_crossover_signal(df=df, fast_line='macd_diff', slow_line='zero')
  df.rename(columns={'signal': 'macd_signal'}, inplace=True)

  return df[['macd_signal']]   


def cal_rsi_signal(df, n=14, up=70, low=30):
  """
  Calculate RSI(Relative Strength Index) signals

  :param df: original OHLCV dataframe
  :param n: windowsize
  :param up: up boundary
  :param low: low boundary
  :returns: rsi signal
  :raises: none
  """
  # divergence / failed swing not implemented
  # only implemented up/low bound
  df = df.copy()
  df['rsi'] = ta.rsi(close=df.Close, n=n)
  df['rsi_signal'] = 'n'
  over_buy_idx = df.query('rsi > %(up)s' % dict(up=up)).index
  over_sell_idx = df.query('rsi < %(low)s' % dict(low=low)).index

  df.loc[over_buy_idx, 'rsi_signal'] = 's'
  df.loc[over_sell_idx, 'rsi_signal'] = 'b'

  return df[['rsi_signal']]


def cal_aroon_signal(df, up=90, low=10):
  """
  Calculate Aroon Indicator signals

  :param df: original OHLCV dataframe
  :param up: up boundary
  :param low: low bounday
  :returns: aroon signals
  :raises: none
  """
  df = df.copy()
  df['aroon_up'] = ta.aroon_up(close=df.Close)
  df['aroon_down'] = ta.aroon_down(close=df.Close)
  df['aroon_signal'] = 'n'

  bull_idx = df.query('aroon_up > %(up)s and aroon_down < %(low)s' % dict(up=up, low=low)).index
  bear_idx = df.query('aroon_down > %(up)s and aroon_up < %(low)s' % dict(up=up, low=low)).index

  df.loc[bull_idx, 'aroon_signal'] = 'b'
  df.loc[bear_idx, 'aroon_signal'] = 's'

  return df[['aroon_signal']]    


def cal_cci_signal(df, up=200, low=-200):
  """
  Calculate CCI(Commidity Channel Indicator) signal

  :param df: original OHLCV dataframe
  :param up: up boundary
  :param low: low bounday
  :returns: CCI signals
  :raises: none
  """
  df = df.copy()
  df['cci'] = ta.cci(high=df.High, low=df.Low, close=df.Close)
  df['cci_signal'] = 'n'
  over_buy_idx = df.query('cci > %(up)s' % dict(up=up)).index
  over_sell_idx = df.query('cci < %(low)s' % dict(low=low)).index

  df.loc[over_buy_idx, 'cci_signal'] = 's'
  df.loc[over_sell_idx, 'cci_signal'] = 'b'

  return df[['cci_signal']]


def cal_ichimoku(df, method='original'):
  """
  Calculate Ichimoku indicators

  :param df: origianl OHLCV dataframe
  :param method: how to calculate Ichimoku indicators: original/ta
  :returns: dataframe with ichimoku indicators
  :raises: none
  """
  # copy dataframe
  df = df.copy()

  # use original method to calculate ichimoku indicators
  if method == 'original':
    df = cal_moving_average(df=df, target_col='High', ma_windows=[9, 26, 52], window_type='sm')
    df = cal_moving_average(df=df, target_col='Low', ma_windows=[9, 26, 52], window_type='sm')

    df['tankan'] = (df['High_ma_9'] + df['Low_ma_9']) / 2
    df['kijun'] = (df['High_ma_26'] + df['Low_ma_26']) / 2
    df['senkou_a'] = (df['tankan'] + df['kijun']) / 2
    df['senkou_b'] = (df['High_ma_52'] + df['Low_ma_52']) / 2
    df['chikan'] = df.Close.shift(-26)
  
  # use ta method to calculate ichimoku indicators
  elif method == 'ta':
    df['tankan'] = (df.High.rolling(9, min_periods=0).max() + df.Low.rolling(9, min_periods=0).min()) / 2
    df['kijun'] = (df.High.rolling(26, min_periods=0).max() + df.Low.rolling(26, min_periods=0).min()) / 2
    df['senkou_a'] = (df['tankan'] + df['kijun']) / 2
    df['senkou_b'] = (df.High.rolling(52, min_periods=0).max() + df.Low.rolling(52, min_periods=0).min()) / 2
    df['chikan'] = df.Close.shift(-26)

  return df


def cal_ichimoku_signal(df, final_signal_threshold=2):
  """
  Calculate ichimoku signals 

  :param df: dataframe with ichimoku indicator columns
  :returns: ichimoku signals
  """
  # copy dataframe
  df = df.copy()

  # calculate crossover signals
  df['signal_senkou'] = cal_crossover_signal(df=df, fast_line='senkou_a', slow_line='senkou_b', pos_signal=1, neg_signal=-1, none_signal=0)

  # calculate cloud signals
  df['signal_cloud'] = 0
  buy_idx = df.query('Close > senkou_a and senkou_a >= senkou_b').index
  sell_idx = df.query('Close < senkou_a').index
  df.loc[buy_idx, 'signal_cloud'] = 1
  df.loc[sell_idx, 'signal_cloud'] = -1

  # calculate kijun/tankan signal
  df['signal_tankan_kijun'] = 0
  buy_idx = df.query('tankan > kijun').index
  sell_idx = df.query('tankan < kijun').index
  df.loc[buy_idx, 'signal_tankan_kijun'] = 1
  df.loc[sell_idx, 'signal_tankan_kijun'] = -1

  # final signal
  df['signal_sum'] = df['signal_senkou'] + df['signal_cloud'] + df['signal_tankan_kijun'] 
  buy_idx = df.query('signal_sum == %s' % final_signal_threshold).index
  sell_idx = df.query('signal_sum == %s' % -final_signal_threshold).index

  df['ichimoku_signal'] = 'n'
  df.loc[buy_idx, 'ichimoku_signal'] = 'b'
  df.loc[sell_idx, 'ichimoku_signal'] = 's'

  return df[['ichimoku_signal']]
  

def plot_ichimoku(df, start=None, end=None, signal_col=None, title=None, save_path=None):
  """
  Plot ichimoku chart

  :param df: dataframe with ichimoku indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param signal_col: columnname of signal values
  :param title: title of the plot
  :param save_path: path to save the plot
  :returns: ichimoku plot
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end]

  # create figure
  fig = plt.figure(figsize=(20, 5))
  ax = plt.gca()

  # plot price
  ax.plot(df.index, df.Close, color='black')

  # plot kijun/tankan lines
  ax.plot(df.index, df.tankan, color='magenta', linestyle='-.')
  ax.plot(df.index, df.kijun, color='blue', linestyle='-.')

  # plot senkou lines
  ax.plot(df.index, df.senkou_a, color='green')
  ax.plot(df.index, df.senkou_b, color='red')

  # plot clouds
  ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a > df.senkou_b, facecolor='green', interpolate=True, alpha=0.6)
  ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a <= df.senkou_b, facecolor='red', interpolate=True, alpha=0.6)

  # plot signals
  if signal_col is not None:
    if signal_col in df.columns.tolist():
      buy_signal = df.query('%s == "b"' % signal_col)
      sell_signal = df.query('%s == "s"' % signal_col)
      ax.scatter(buy_signal.index, buy_signal.Close, marker='^', color='green', )
      ax.scatter(sell_signal.index, sell_signal.Close, marker='v', color='red', )

  plt.legend()  
  plt.title(title)

  # save image
  if save_path is not None:
    plt.savefig(save_path + title + '.png')


def cal_kst_signal(df):
  """
  Calculate kst signal

  :param df: original OHLCV dataframe
  :returns: kst signals
  :raises: none
  """
  # copy dataframe
  df = df.copy()
  df['kst'] = ta.kst(close=df.Close)
  df['kst_sig'] = ta.kst_sig(close=df.Close)
  df['kst_signal'] = cal_crossover_signal(df=df, fast_line='kst', slow_line='kst_sig')
  
  return df[['kst_signal']]


#----------------------------- Indicator visualization -----------------------------#
def plot_indicator_around_benchmark(df, target_col, benchmark=0, title=None, start_date=None, end_date=None, color_mode='up_down', plot_close=True, figsize=(20, 5)):
  """
  Plot indicators around a benchmark

  :param df: dataframe which contains target columns
  :param target_col: columnname of the target indicator
  :param benchmark: benchmark, a fixed value
  :param title: title of the plot
  :param start_date: start date of the data
  :param end_date: end_date of the data
  :param color_mode: which color mode to use: benckmark/up_down
  :param plot_close: whether to plot close price
  :param figsize: figure size
  :returns: figure with indicators and close price plotted
  :raises: none
  """
  # select data
  df = df[start_date:end_date].copy()
  
  # create figure
  fig = plt.figure(figsize=figsize)
  
  # set indicator colors
  ax1 = fig.add_subplot(111)

  # plot in up_down mode
  if color_mode == 'up_down':  
    df['color'] = 'red'
    previous_target_col = 'previous_'+target_col
    df[previous_target_col] = df[target_col].shift(1)
    df.loc[df[target_col] > df[previous_target_col], 'color'] = 'green'

  # plot in benchmark mode
  elif color_mode == 'benchmark':
    df['color'] = 'red'
    df.loc[df[target_col] > benchmark, 'color'] = 'green'
    df['benchmark'] = benchmark
    ax1.plot(df.index, df['benchmark'], color='black')
    
  else:
    print('Unknown Color Mode')
    return None

  # plot indicator
  ax1.bar(df.index, height=df[target_col], color=df.color, alpha=0.5)
  ax1.set_title(title)

  # plot close price
  if plot_close:
    ax2=ax1.twinx()
    ax2.plot(df.Close, color='blue' )


def plot_indicator(df, target_col, title=None, start_date=None, end_date=None, plot_close=True, figsize=(20, 5)):
  """
  Plot indicators

  :param df: dataframe which contains target columns
  :param target_col: columnname of the target indicator
  :param title: title of the plot
  :param start_date: start date of the data
  :param end_date: end_date of the data
  :param plot_close: whether to plot close price
  :param figsize: figure size
  :returns: figure with indicators and close price plotted
  :raises: none
  """
  # copy dataframe  
  df = df[start_date:end_date].copy()

  # create figure
  fig = plt.figure(figsize=figsize)
  ax1 = fig.add_subplot(111)

  # plot indicator
  ax1.plot(df[target_col], color='red', alpha=0.5)
  ax1.set_title(title)

  # plot close price
  if plot_close:
    ax2=ax1.twinx()
    ax2.plot(df.Close, color='blue' ) 


#----------------------------- Candlesticks ----------------------------------------#
def add_candle_dims_for_df(df):
  """
  Add candlestick dimentions for dataframe

  :param df: original OHLCV dataframe
  :returns: dataframe with candlestick columns
  :raises: none
  """
  # copy dataframe
  df = df.copy()
  
  # shadow
  df['shadow'] = (df['High'] - df['Low'])    
  
  # entity
  df['entity'] = abs(df['Close'] - df['Open'])
  
  # up and down rows
  up_idx = df.Open < df.Close
  down_idx = df.Open >= df.Close

  # upper/lower shadow
  df['upper_shadow'] = 0
  df['lower_shadow'] = 0
  df['candle_color'] = 0
  
  # up
  df.loc[up_idx, 'candle_color'] = 1
  df.loc[up_idx, 'upper_shadow'] = (df.loc[up_idx, 'High'] - df.loc[up_idx, 'Close'])
  df.loc[up_idx, 'lower_shadow'] = (df.loc[up_idx, 'Open'] - df.loc[up_idx, 'Low'])
  
  # down
  df.loc[down_idx, 'candle_color'] = -1
  df.loc[down_idx, 'upper_shadow'] = (df.loc[down_idx, 'High'] - df.loc[down_idx, 'Open'])
  df.loc[down_idx, 'lower_shadow'] = (df.loc[down_idx, 'Close'] - df.loc[down_idx, 'Low'])
  
  return df
