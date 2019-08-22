# -*- coding: utf-8 -*-
import math
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
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


def cal_change_rate(df, target_col, periods=1, add_accumulation=True, add_prefix=False, drop_na=True):
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

  if drop_na:        
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


def cal_boundary_signal(df, upper_col, lower_col, upper_boundary, lower_boundary, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate signal generated from triger of boundaries
  When upper_col breakthrough upper_boundary, positive signal will be generated
  When lower_col breakthrough lower_boundary, negative signal will be generated 

  :param df: original dffame which contains a fast line and a slow line
  :param upper_col: columnname of the positive column
  :param lower_col: columnname of the negative column
  :param upper_boundary: upper boundary
  :param lower_boundary: lower boundary
  :param result_col: columnname of the result
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: series of the result column
  :raises: none
  """
  # copy dataframe
  df = df.copy()

  # calculate signals
  df[result_col] = none_signal
  pos_idx = df.query('%(column)s > %(value)s' % dict(column=upper_col, value=upper_boundary)).index
  neg_idx = df.query('%(column)s < %(value)s' % dict(column=lower_col, value=lower_boundary)).index
  df.loc[pos_idx, result_col] = pos_signal
  df.loc[neg_idx, result_col] = neg_signal

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


def remove_redundant_signal(df, signal_col='signal', pos_signal='b', neg_signal='s', none_signal='n', keep='first'):
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


#----------------------------- Support/resistant -----------------------------------#
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

  # shift the signal back by 1 unit
  df[result_col] = df[result_col].shift(-1)

  # remove redundant signals
  df = remove_redundant_signal(df=df, signal_col=result_col, keep='first', pos_signal=peak_signal, neg_signal=trough_signal, none_signal=none_signal)

  # further filter the signals
  if further_filter:
      
    # get all peak/trough signals
    peak = df.query('%(r)s == "%(p)s"' % dict(r=result_col, p=peak_signal)).index.tolist()
    trough = df.query('%(r)s == "%(t)s"' % dict(r=result_col, t=trough_signal)).index.tolist()
        
    # peak/trough that not qualified
    false_peak = []
    false_trough = []
    
    # filter peak signals
    for i in range(len(peak)):
      current_idx = peak[i]
      benchmark = 0
      
      # the peak is not qualified if it is lower that the average of previous 2 troughs
      previous_troughs = df[:current_idx].query('%(r)s == "%(t)s"' % dict(r=result_col, t=trough_signal)).tail(2)
      if len(previous_troughs) > 0:
        benchmark = previous_troughs[target_col].mean()
      
      if df.loc[current_idx, target_col] < benchmark:
        false_peak.append(current_idx)
        
    # filter trough signals
    for i in range(len(trough)):        
      current_idx = trough[i]
      benchmark = 0

      # the trough is not qualified if it is lower that the average of previous 2 peaks
      previous_peaks = df[:current_idx].query('%(r)s == "%(p)s"' % dict(r=result_col, p=peak_signal)).tail(2)
      if len(previous_peaks) > 0:
        benchmark = previous_peaks[target_col].mean()
      
      if df.loc[current_idx, target_col] > benchmark:
        false_trough.append(current_idx)
          
    df.loc[false_peak, result_col] = none_signal
    df.loc[false_trough, result_col] = none_signal

  return df[[result_col]]
  

#----------------------------- Mean reversion --------------------------------------#
def cal_mean_reversion(df, target_col, window_size=100, start=None, end=None, window_type='sm'):
  """
  Calculate (current value - moving avg) / moving std

  :param df: original dataframe which contains target column
  :param window_size: window size of the moving window
  :param start: start row
  :param end: end row
  :window_type: which type of moving window is going to be used: sm/em
  :returns: dataframe with mean-reversion result columns
  :raises: none
  """
  # calculate change rate by day
  original_columns = df.columns
  df = cal_change_rate(df=df, target_col=target_col, periods=1, add_accumulation=True)[start:end]

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


def cal_mean_reversion_signal(df, std_multiple=2, final_signal_threshold=2, start=None, end=None, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate signal from mean reversion data

  :param df: dataframe which contains mean reversion columns
  :param std_multiple: the multiple of moving std to triger signals
  :param final_signal_threshold: how many columns triger signals at the same time could triger the final signal
  :param start: start date of the data
  :param end: end date of the data
  :param result_col: columnname of the result signal
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: dataframe with signal columns
  :raises: none
  """
  # copy dataframe
  df = df[start : end].copy()
  col_to_drop = []

  # check whether triger columns are in the dataframe
  triger_cols = ['rate_bias', 'acc_rate_bias', 'acc_day_bias']

  # initialization
  df['final_signal'] = 0
  col_to_drop.append('final_signal')

  # calculate signal for each triger column
  for col in triger_cols:
    col_signal = col.replace('bias', 'signal')
    col_to_drop.append(col_signal)

    df[col_signal] = cal_boundary_signal(df=df, upper_col=col, lower_col=col, upper_boundary=std_multiple, lower_boundary=-std_multiple, pos_signal=1, neg_signal=-1, none_signal=0)
    df['final_signal'] = df['final_signal'].astype(int) + df[col_signal].astype(int)

  # conver final singal from int to specific values 
  df[result_col]  = cal_boundary_signal(df=df, upper_col='final_signal', lower_col='final_signal', upper_boundary=final_signal_threshold, lower_boundary=-final_signal_threshold, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal)

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


#----------------------------- Moving average --------------------------------------#
def cal_moving_average(df, target_col, ma_windows=[50, 105], start=None, end=None, window_type='em'):
  """
  Calculate moving average of the tarhet column with specific window size

  :param df: original dataframe which contains target column
  :param ma_windows: a list of moving average window size to be calculated
  :param start: start date of the data
  :param end: end date of the data
  :param window_type: which moving window to be used: sm/em
  :returns: dataframe with moving averages
  :raises: none
  """
  # copy dataframe
  df = df[start:end].copy()

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


def cal_moving_average_signal(df, target_col='Close', ma_windows=[50, 105], start=None, end=None, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate moving avergae signals gernerated from fast/slow moving average crossover

  :param df: original dataframe which contains short/ling ma columns
  :param short_ma_col: columnname of the short ma
  :param long_ma_col: columnname of the long ma
  :param start: start date of the data
  :param end: end date of the data
  :param result_col: columnname of the result signal
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: dataframe with ma crossover signal
  :raises: none
  """
  # calculate moving average
  df = df[start : end].copy()
  df[result_col] = none_signal

  if len(ma_windows) > 2:
    print('There should be only 2 moving average lines')

  else:
    short_ma_col = '%(col)s_ma_%(window_size)s' % dict(col=target_col, window_size=min(ma_windows))
    long_ma_col = '%(col)s_ma_%(window_size)s' % dict(col=target_col, window_size=max(ma_windows))

    # calculate ma crossover signal
    df[result_col] = cal_crossover_signal(df=df, fast_line=short_ma_col, slow_line=long_ma_col, result_col=result_col, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal)

  return df[result_col]


#----------------------------- TA trend indicators ---------------------------------#
# def add_adx_features()


def add_aroon_features(df, n=25, close='Close', open='Open', high='High', low='Low', volume='Volume', fillna=False, cal_signal=True, boundary=[50, 50]):
  """
  Calculate Aroon

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param close: column name of the close
  :param open: column name of the open
  :param high: column name of the high
  :param low: column name of the low
  :param volume: column name of the volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: upper and lower boundary for calculating signal
  :returns: dataframe with new features generated
  """
  # calculate aroon up and down indicators
  aroon_up = df[close].rolling(n, min_periods=0).apply(lambda x: float(np.argmax(x) + 1) / n * 100, raw=True)
  aroon_down = df[close].rolling(n, min_periods=0).apply(lambda x: float(np.argmin(x) + 1) / n * 100, raw=True)
  
  # fill na value with 0
  if fillna:
    aroon_up = aroon_up.replace([np.inf, -np.inf], np.nan).fillna(0)
    aroon_down = aroon_down.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign values to df
  df['aroon_up'] = aroon_up
  df['aroon_down'] = aroon_down

  # calculate aroon signal
  if cal_signal:
    upper_boundary = max(boundary)
    lower_boundary = min(boundary)
    pos_idx = df.query('aroon_up > %(upper_boundary)s and aroon_down < %(lower_boundary)s' % dict(upper_boundary=upper_boundary, lower_boundary=lower_boundary)).index
    neg_idx = df.query('aroon_up < %(upper_boundary)s and aroon_down > %(lower_boundary)s' % dict(upper_boundary=upper_boundary, lower_boundary=lower_boundary)).index

    df['aroon_signal'] = 'n'
    df.loc[pos_idx, 'aroon_signal'] = 'b'
    df.loc[neg_idx, 'aroon_signal'] = 's'

  return df


def add_cci_features(df, n=20, c=0.015, close='Close', open='Open', high='High', low='Low', volume='Volume', fillna=False, cal_signal=True, boundary=[200, -200]):
  """
  Calculate CCI(Commidity Channel Indicator) 

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param c: constant value used in cci calculation
  :param close: column name of the close
  :param open: column name of the open
  :param high: column name of the high
  :param low: column name of the low
  :param volume: column name of the volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: upper and lower boundary for calculating signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # calculate cci
  pp = (df[high] + df[low] + df[close]) / 3.0
  mad = lambda x : np.mean(np.abs(x-np.mean(x)))
  cci = (pp - pp.rolling(n, min_periods=0).mean()) / (c * pp.rolling(n).apply(mad,True))

  # assign values to dataframe
  df['cci'] = cci

  # calculate siganl
  if cal_signal:
    upper_boundary = max(boundary)
    lower_boundary = min(boundary)
    df['cci_signal'] = cal_boundary_signal(df=df, upper_col='cci', lower_col='cci', upper_boundary=upper_boundary, lower_boundary=lower_boundary)

  return df


def add_dpo_features(df, n=20, close='Close', open='Open', high='High', low='Low', volume='Volume', fillna=False, cal_signal=True):
  """
  Calculate DPO(Detrended Price Oscillator ) 

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param close: column name of the close
  :param open: column name of the open
  :param high: column name of the high
  :param low: column name of the low
  :param volume: column name of the volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # calculate dpo
  dpo = df[close].shift(int((0.5 * n) + 1)) - df[close].rolling(n, min_periods=0).mean()
  if fillna:
    dpo = dpo.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign values to df
  df['dpo'] = dpo

  # calculate_signal
  if cal_signal:
    df['zero'] = 0
    df['dpo_signal'] = cal_crossover_signal(df=df, fast_line='dpo', slow_line='zero')
    df.drop(labels='zero', axis=1, inplace=True)

  return df


def add_ichimoku_features(df, n_short=9, n_medium=26, n_long=52, method='ta', is_shift=True, close='Close', open='Open', high='High', low='Low', volume='Volume', fillna=False, cal_status=True, cal_signal=True, signal_threhold=2):
  """
  Calculate Ichimoku indicators

  :param df: original OHLCV dataframe
  :param n_short: short window size
  :param n_medium: medium window size
  :param n_long: long window size
  :param method: original/ta way to calculate ichimoku indicators
  :param is_shift: whether to shift senkou_a and senkou_b n_medium units
  :param close: column name of the close
  :param open: column name of the open
  :param high: column name of the high
  :param low: column name of the low
  :param volume: column name of the volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  col_to_drop = []

  # use original method to calculate ichimoku indicators
  if method == 'original':
    df = cal_moving_average(df=df, target_col=high, ma_windows=[n_short, n_medium, n_long], window_type='sm')
    df = cal_moving_average(df=df, target_col=low, ma_windows=[n_short, n_medium, n_long], window_type='sm')

    # generate column names
    short_high = '%(col)s_ma_%(p)s' % dict(col=high, p=n_short)
    short_low = '%(col)s_ma_%(p)s' % dict(col=low, p=n_short)
    medium_high = '%(col)s_ma_%(p)s' % dict(col=high, p=n_medium)
    medium_low = '%(col)s_ma_%(p)s' % dict(col=low, p=n_medium)
    long_high = '%(col)s_ma_%(p)s' % dict(col=high, p=n_long)
    long_low = '%(col)s_ma_%(p)s' % dict(col=low, p=n_long)
    col_to_drop += [short_high, medium_high, long_high, short_low, medium_low, long_low]

    # calculate ichimoku indicators
    df['tankan'] = (df[short_high] + df[short_low]) / 2
    df['kijun'] = (df[medium_high] + df[medium_low]) / 2
    df['senkou_a'] = (df['tankan'] + df['kijun']) / 2
    df['senkou_b'] = (df[long_high] + df[long_low]) / 2
    df['chikan'] = df[close].shift(-n_medium)
  
  # use ta method to calculate ichimoku indicators
  elif method == 'ta':
    df['tankan'] = (df[high].rolling(n_short, min_periods=0).max() + df[low].rolling(n_short, min_periods=0).min()) / 2
    df['kijun'] = (df[high].rolling(n_medium, min_periods=0).max() + df[low].rolling(n_medium, min_periods=0).min()) / 2
    df['senkou_a'] = (df['tankan'] + df['kijun']) / 2
    df['senkou_b'] = (df[high].rolling(n_long, min_periods=0).max() + df[low].rolling(n_long, min_periods=0).min()) / 2
    df['chikan'] = df[close].shift(-n_medium)

  # shift senkou_a and senkou_b n_medium units
  if is_shift:
    df['senkou_a'] = df['senkou_a'].shift(n_medium)
    df['senkou_b'] = df['senkou_b'].shift(n_medium)

  # calculate ichmoku status
  if cal_status:
    # cloud color change
    df['cloud_shift'] = cal_crossover_signal(df=df, fast_line='senkou_a', slow_line='senkou_b', pos_signal=1, neg_signal=-1, none_signal=0)
    df['cloud_height'] = round((df['senkou_a'] - df['senkou_b'])/df[close], ndigits=3)
    green_idx = df.query('cloud_height > 0').index
    red_idx = df.query('cloud_height <= 0').index

    # cloud width (how long does it last)
    df['cloud_width'] = 0
    df.loc[green_idx, 'cloud_width'] = 1
    df.loc[red_idx, 'cloud_width'] = -1

    # cloud top and bottom
    df['cloud_top'] = 0
    df.loc[green_idx, 'cloud_top'] = df['senkou_a']
    df.loc[red_idx, 'cloud_top'] = df['senkou_b']

    df['cloud_bottom'] = 0
    df.loc[green_idx, 'cloud_bottom'] = df['senkou_b']
    df.loc[red_idx, 'cloud_bottom'] = df['senkou_a']

    # calculate how long current cloud has lasted
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current_cloud_period = df.loc[current_idx, 'cloud_width']
      previous_cloud_period = df.loc[previous_idx, 'cloud_width']

      # calculate how long the cloud has last
      if current_cloud_period * previous_cloud_period > 0:
        df.loc[current_idx, 'cloud_width'] += previous_cloud_period

    # calculate distance between Close and each ichimoku lines    
    line_weight = {'kijun':1, 'tankan':0.5, 'cloud_top':1, 'cloud_bottom':1}
    df['break_up'] = ''
    df['break_down'] = ''
    df['breakthrough'] = 0
    col_to_drop.append('breakthrough')

    for line in line_weight.keys():
      # set weight for this line
      weight = line_weight[line]

      # calculate breakthrough
      line_signal = cal_crossover_signal(df=df, fast_line=close, slow_line=line, pos_signal=weight, neg_signal=-weight, none_signal=0)
      line_signal_name = 'signal_%s' % line
      col_to_drop.append(line_signal_name)

      # record breakthrough
      up_idx = line_signal.query('signal == %s' % weight).index
      down_idx = line_signal.query('signal == %s' % -weight).index      
      df.loc[up_idx, 'break_up'] = df.loc[up_idx, 'break_up'] + line + ','
      df.loc[down_idx, 'break_down'] = df.loc[down_idx, 'break_down'] + line + ','
      
      # accumulate breakthrough signals
      df[line_signal_name] = line_signal
      df['breakthrough'] = df['breakthrough'].astype(int) +df[line_signal_name].astype(int)

      # calculate distance between close price and indicator
      df['close_to_' + line] = round((df['Close'] - df[line]) / df['Close'], ndigits=3)

    # calculate ichimoku signal
    if cal_signal:

      # identify trend
      up_idx = df.query('%s > cloud_top' % close).index
      down_idx = df.query('%s < cloud_bottom' % close).index
      df['trend'] = 0
      df.loc[up_idx, 'trend'] = 2
      df.loc[down_idx, 'trend'] = -2

      # cloud color
      up_idx = df.query('cloud_height > 0').index
      down_idx = df.query('cloud_height <= 0').index
      df['cloud_color'] = 0
      df.loc[up_idx, 'cloud_color'] = 1
      df.loc[down_idx, 'cloud_color'] = -1

      # identify takan-kijun crossover
      df['tankan_kijun_crossover'] = cal_crossover_signal(df=df, fast_line='tankan', slow_line='kijun', pos_signal=1, neg_signal=-1, none_signal=0)   

      # sum up signals
      df['ichimoku_idx'] = df['trend'].astype(float) + df['cloud_color'].astype(float) + df['cloud_shift'].astype(float) + df['breakthrough'].astype(float) + df['tankan_kijun_crossover'].astype(float)

      # final signal
      buy_idx = df.query('ichimoku_idx > %s' % signal_threhold).index
      sell_idx = df.query('ichimoku_idx < %s' % -signal_threhold).index
      df['ichimoku_signal'] = 'n'
      df.loc[buy_idx, 'ichimoku_signal'] = 'b'
      df.loc[sell_idx, 'ichimoku_signal'] = 's'
      col_to_drop += ['trend', 'cloud_color', 'tankan_kijun_crossover']

    # drop redundant columns  
    # df.drop(col_to_drop, axis=1, inplace=True)

  return df


def add_kst_features(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsign=9, close='Close', open='Open', high='High', low='Low', volume='Volume', fillna=False, cal_signal=True, signal_mode='mix'):
  """
  Calculate KST(Know Sure Thing)

  :param df: original OHLCV dataframe
  :param r_1: r1 window size
  :param r_2: r2 window size
  :param r_3: r3 window size
  :param r_4: r4 window size
  :param n_1: n1 window size
  :param n_2: n2 window size
  :param n_3: n3 window size
  :param n_4: n4 window size
  :param n_sign: kst signal window size
  :param close: column name of the close
  :param open: column name of the open
  :param high: column name of the high
  :param low: column name of the low
  :param volume: column name of the volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  col_to_drop = []

  # calculate kst
  rocma1 = ((df[close] - df[close].shift(r1)) / df[close].shift(r1)).rolling(n1, min_periods=0).mean()
  rocma2 = ((df[close] - df[close].shift(r2)) / df[close].shift(r2)).rolling(n2, min_periods=0).mean()
  rocma3 = ((df[close] - df[close].shift(r3)) / df[close].shift(r3)).rolling(n3, min_periods=0).mean()
  rocma4 = ((df[close] - df[close].shift(r4)) / df[close].shift(r4)).rolling(n4, min_periods=0).mean()
  
  kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
  kst_sign = kst.rolling(nsign, min_periods=0).mean()

  # fill na value
  if fillna:
    kst = kst.replace([np.inf, -np.inf], np.nan).fillna(0)
    kst_sign = kst_sig.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign values to df
  df['kst'] = kst
  df['kst_sign'] = kst_sign

  # calculate signal
  if cal_signal:
    df['zero'] = 0
    col_to_drop.append('zero')

    # kst-0 crossover
    if signal_mode == 'zero':  
      df['kst_signal'] = cal_crossover_signal(df=df, fast_line='kst', slow_line='zero')
      
    # kst-kst_sign crossovers
    elif signal_mode == 'kst_sign':
      df['kst_signal'] = cal_crossover_signal(df=df, fast_line='kst', slow_line='kst_sign')

    # buy with kst-0 crossover, sell with kst-kst_sign crossover
    elif signal_mode == 'mix':
      df['signal_kst'] = cal_crossover_signal(df=df, fast_line='kst', slow_line='zero', pos_signal=1, neg_signal=0, none_signal=0)
      df['signal_kst_sign'] = cal_crossover_signal(df=df, fast_line='kst', slow_line='kst_sign', pos_signal=0, neg_signal=-1, none_signal=0)
      col_to_drop += ['signal_kst', 'signal_kst_sign']

      df['kst_signal'] = df['signal_kst'].astype(int) + df['signal_kst_sign'].astype(int)
      buy_idx = df.query('kst_signal == 1').index
      sell_idx = df.query('kst_signal == -1').index
      
      df['kst_signal'] = 'n'
      df.loc[buy_idx, 'kst_signal'] = 'b'
      df.loc[sell_idx, 'kst_signal'] = 's'
    
    else:
      print('unknown signal mode')
      df['kst_signal'] = 'n'

  df.drop('zero', axis=1, inplace=True)
  return df


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


def cal_ichimoku_status(df, add_change_rate=True, is_save=False, file_name='ichimoku_status.xlsx', save_path='drive/My Drive/ichimoku/'):
  """
  Calculate relationship between close price and ichimoku indicators

  :param df: dataframe with close price and ichimoku indicator columns
  :param is_add_change_rate: whether to add change rate of close price
  :param is_save: whether to save the result into excel file
  :param file_name: destination filename
  :patam save_path: where to save the file
  :returns: dataframe with ichimoku status
  :raises: none
  """
  # copy dataframe
  df = df.copy()

  # calculate cloud size/color and color shift
  df['cloud_shift'] = cal_crossover_signal(df=df, fast_line='senkou_a', slow_line='senkou_b', pos_signal=1, neg_signal=-1, none_signal=0)
  df['cloud_height'] = round((df['senkou_a'] - df['senkou_b'])/df['Close'], ndigits=3)
  df['cloud_width'] = 0
  df['cloud_color'] = 0
  df['cloud_top'] = 0
  df['cloud_bottom'] = 0
  df['break_up'] = ''
  df['break_down'] = ''

  # initialize values according to ichimoku indicators
  green_idx = df.query('cloud_height > 0').index
  red_idx = df.query('cloud_height <= 0').index
  df.loc[green_idx, 'cloud_width'] = 1
  df.loc[green_idx, 'cloud_color'] = 1
  df.loc[green_idx, 'cloud_top'] = df['senkou_a']
  df.loc[green_idx, 'cloud_bottom'] = df['senkou_b']
  df.loc[red_idx, 'cloud_width'] = -1
  df.loc[red_idx, 'cloud_color'] = -1
  df.loc[red_idx, 'cloud_top'] = df['senkou_b']
  df.loc[red_idx, 'cloud_bottom'] = df['senkou_a']

  # calculate how long current cloud has lasted
  idx = df.index.tolist()
  for i in range(1, len(df)):
    current_idx = idx[i]
    previous_idx = idx[i-1]
    current_cloud_period = df.loc[current_idx, 'cloud_width']
    previous_cloud_period = df.loc[previous_idx, 'cloud_width']

    # calculate how long the cloud has last
    if current_cloud_period * previous_cloud_period > 0:
      df.loc[current_idx, 'cloud_width'] += previous_cloud_period

  # calculate distance between Close and each ichimoku lines    
  lines = ['kijun', 'tankan', 'cloud_top', 'cloud_bottom']
  for line in lines:

    # breakthrough
    line_signal = cal_crossover_signal(df=df, fast_line='Close', slow_line=line, result_col='signal', pos_signal='up', neg_signal='down', none_signal='')
    up_idx = line_signal.query('signal == "up"').index
    down_idx = line_signal.query('signal == "down"').index
    df.loc[up_idx, 'break_up'] = df.loc[up_idx, 'break_up'] + line + ','
    df.loc[down_idx, 'break_down'] = df.loc[down_idx, 'break_down'] + line + ','

    # calculate distance between close price and indicator
    df['close_to_' + line] = round((df['Close'] - df[line]) / df['Close'], ndigits=3)

  # save result to files
  if is_save:
    df.to_excel(save_path+file_name)

  return df


def cal_ichimoku_signal(df, result_col='signal', pos_signal='b', neg_signal='s', none_signal='n', filter_signal='first', final_signal_threshold=2):
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

  df[result_col] = none_signal
  df.loc[buy_idx, result_col] = pos_signal
  df.loc[sell_idx, result_col] = neg_signal

  if filter_signal in ['first', 'last']:
    df = remove_redundant_signal(df=df, signal_col=result_col, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal, keep=filter_signal)

  return df[[result_col]]
  
# def add_kst_features
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


def add_macd_features(df, n_fast=12, n_slow=26, n_sign=9, close='Close', open='Open', high='High', low='Low', volume='Volume', fillna=False, cal_signal=True):
  """
  Calculate MACD(Moving Average Convergence Divergence)

  :param df: original OHLCV dataframe
  :param n_fast: ma window of fast ma
  :param n_slow: ma window of slow ma
  :paran n_sign: ma window of macd signal line
  :param close: column name of the close
  :param open: column name of the open
  :param high: column name of the high
  :param low: column name of the low
  :param volume: column name of the volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # calculate fast and slow ema of close price
  emafast = em(series=df[close], periods=n_fast, fillna=fillna).mean()
  emaslow = em(series=df[close], periods=n_slow, fillna=fillna).mean()
  
  # calculate macd, ema(macd), macd-ema(macd)
  macd = emafast - emaslow
  macd_sign = em(series=macd, periods=n_sign, fillna=fillna).mean()
  macd_diff = macd - macd_sign

  # fill na value with 0
  if fillna:
      macd = macd.replace([np.inf, -np.inf], np.nan).fillna(0)
      macd_sign = macd_sign.replace([np.inf, -np.inf], np.nan).fillna(0)
      macd_diff = macd_diff.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign valuse to df
  df['macd'] = macd
  df['macd_sign'] = macd_sign
  df['macd_diff'] = macd_diff

  # calculate crossover signal
  if cal_signal:
    df['zero'] = 0
    df['macd_signal'] = cal_crossover_signal(df=df, fast_line='macd_diff', slow_line='zero')
    df.drop(labels='zero', axis=1, inplace=True)

  return df


# def add_mi_features


# def add_trix_features

# def add_vi_features
def cal_vi_signal(df,  result_col='signal', pos_signal='b', neg_signal='s', none_signal='n'):
  """
  Calculate Vortex Indicators

  :param df: original OHLCV dataframe
  :param result_col: columnname of the result signal
  :param pos_signal: the value of positive signal
  :param neg_siganl: the value of negative signal
  :param none_signal: the value of none signal
  :returns: vortex signal
  :raises: none
  """
  # copy dataframe
  df = df.copy()
  
  # calculate vortex indicators
  df['vi_pos'] = ta.vortex_indicator_pos(high=df.High, low=df.Low, close=df.Close)
  df['vi_neg'] = ta.vortex_indicator_neg(high=df.High, low=df.Low, close=df.Close)
  
  # calculate vortex signal
  df[result_col]= cal_crossover_signal(df=df, fast_line='vi_pos', slow_line='vi_neg', result_col=result_col, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal)

  return df[[result_col]]


#----------------------------- TA volume indicators --------------------------------#
# def add_adi_features()


# def add_cmf_features()

# def add_eom_features()
def cal_eom_signal(df):
  """
  Calculate Ease of movement signal

  :param df: original OHLCV dataframe
  :returns: eom signal
  :raises: none
  """

  # copy dataframe
  df = df.copy()

  # calculate eom indicator and signal
  df['eom'] = ta.ease_of_movement(high=df.High, low=df.Low, close=df.Close, volume=df.Volume)
  df['zero'] = 0
  df['signal'] = cal_crossover_signal(df=df, fast_line='eom', slow_line='zero')
  
  return df[['signal']]


# def add_fi_features()


# def add_nvi_features()


# def add_obv_features()


# def add_vpt_features()


#----------------------------- TA momentum indicators ------------------------------#


#----------------------------- TA volatility indicators ----------------------------#


#----------------------------- Indicator visualization -----------------------------#
def plot_signal(df, start=None, end=None, price_col='Close', signal_col='signal', pos_signal='b', neg_signal='s', none_signal='n', filter_signal=None, title=None, figsize=(20, 5), use_ax=None):
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
  :param use_ax: the already-created ax to draw on
  :returns: a signal plotted price chart
  :raises: none
  """
  # copy dataframe within the specific period
  df = df[start:end]

  # create figure
  ax = use_ax
  if ax is None:
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

  # plot price
  ax.plot(df.index, df[price_col], color='black', label=price_col, alpha=0.5)

  # plot signal
  if signal_col in df.columns:

    # remove redundant signals
    if filter_signal in ['first', 'last']:
      signal = remove_redundant_signal(df, signal_col=signal_col, keep=filter_signal, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal)
    else:
      signal = df
      
    # plot positive and negative signlas
    positive_signal = signal.query('%(signal)s == "%(pos_signal)s"' % dict(signal=signal_col, pos_signal=pos_signal))
    negative_signal = signal.query('%(signal)s == "%(neg_signal)s"' % dict(signal=signal_col, neg_signal=neg_signal))
    ax.scatter(positive_signal.index, positive_signal[price_col], label='%s' % pos_signal, marker='^', color='green', alpha=0.8)
    ax.scatter(negative_signal.index, negative_signal[price_col], label='%s' % neg_signal, marker='v', color='red', alpha=0.8)

  # legend and title
  ax.legend(loc='upper left')  
  ax.set_title(title)

  # return ax
  if use_ax is not None:
    return ax


def plot_ichimoku(df, price_col='Close', start=None, end=None, signal_col='signal',  pos_signal='b', neg_signal='s', none_signal='n', filter_signal='first', save_path=None, title=None, figsize=(20, 5), use_ax=None):
  """
  Plot ichimoku chart

  :param df: dataframe with ichimoku indicator columns
  :param signal_col: columnname of signal values
  :param price_col: columnname of the price
  :param start: start row to plot
  :param end: end row to plot
  :param save_path: path to save the plot
  :param title: title of the plot
  :param figsize: figsize
  :param use_ax: the already-created ax to draw on
  :returns: ichimoku plot
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end]

  # create figure
  ax = use_ax
  if ax is None:
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

  # plot price and signal
  ax = plot_signal(df, price_col=price_col, signal_col=signal_col, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal, filter_signal=filter_signal, use_ax=ax)

  # plot kijun/tankan lines
  ax.plot(df.index, df.tankan, color='magenta', linestyle='-.')
  ax.plot(df.index, df.kijun, color='blue', linestyle='-.')

  # plot senkou lines
  ax.plot(df.index, df.senkou_a, color='green')
  ax.plot(df.index, df.senkou_b, color='red')

  # plot clouds
  ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a > df.senkou_b, facecolor='green', interpolate=True, alpha=0.3)
  ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a <= df.senkou_b, facecolor='red', interpolate=True, alpha=0.3)

  # title and legend
  ax.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.) 
  ax.set_title(title)

  # save image
  if save_path is not None:
    plt.savefig(save_path + title + '.png')

  if use_ax is not None:
    return ax


def plot_indicator(df, target_col, start=None, end=None, benchmark=0, boundary=None, color_mode='up_down', price_col='Close', signal_col='signal', pos_signal='b', neg_signal='s', none_signal='n', filter_signal=None, title=None, figsize=(20, 5), use_ax=None):
  """
  Plot indicators around a benchmark

  :param df: dataframe which contains target columns
  :param target_col: columnname of the target indicator
  :param price_col: columnname of the price values
  :param start: start date of the data
  :param end: end of the data
  :param benchmark: benchmark, a fixed value
  :param color_mode: which color mode to use: benckmark/up_down
  :param title: title of the plot
  :param figsize: figure size
  :param use_ax: the already-created ax to draw on
  :returns: figure with indicators and close price plotted
  :raises: none
  """
  # select data
  df = df[start:end].copy()
  
  # create figure
  ax = use_ax
  if ax is None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

  # plot benchmark
  if benchmark is not None:
    df['benchmark'] = benchmark
    ax.plot(df.index, df['benchmark'], color='black', linestyle='--', label='%s'%benchmark)

  if boundary is not None:
    if len(boundary) > 0:
      df['upper_boundary'] = max(boundary)
      df['lower_boundary'] = min(boundary)
      ax.plot(df.index, df['upper_boundary'], color='green', linestyle='--', label='%s'% max(boundary))
      ax.plot(df.index, df['lower_boundary'], color='red', linestyle='--', label='%s'% min(boundary))

  # plot indicator(s)
  for col in target_col:
    ax.plot(df.index, df[col], label=col, alpha=0.8)

  # plot color bars if there is only one indicator to plot
  if len(target_col) == 1:
    tar = target_col[0]

    # plot in up_down mode
    if color_mode == 'up_down':  
      df['color'] = 'red'
      previous_target_col = 'previous_' + tar
      df[previous_target_col] = df[tar].shift(1)
      df.loc[df[tar] > df[previous_target_col], 'color'] = 'green'

    # plot in benchmark mode
    elif color_mode == 'benchmark' and benchmark is not None:
      df['color'] = 'red'
      df.loc[df[tar] > benchmark, 'color'] = 'green'
      
    # plot indicator
    if 'color' in df.columns:
      ax.bar(df.index, height=df[tar], color=df.color, alpha=0.5)

  # plot close price
  if price_col in df.columns:
    ax2=ax.twinx()
    plot_signal(df, price_col=price_col, signal_col=signal_col, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal, filter_signal=filter_signal, use_ax=ax2)
    ax2.legend(loc='lower left')

  # plot title and legend
  ax.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.) 
  ax.set_title(title)

  # return ax
  if use_ax is not None:
    return ax


def plot_multiple_indicators(df, args={'plot_ratio': {'ichimoku':1.5, 'mean_reversion':1}, 'mean_reversion': {'std_multiple': 2}}, start=None, end=None, title=None, save_path=None, show_image=False, ws=0, hs=0, xp=0, yp=0, unit_size=3):
  """
  Plot Ichimoku and mean reversion in a same plot

  :param df: dataframe with ichimoku and mean reversion columns
  :param std_multiple: std_multiple for mean reversion
  :param start: start of the data
  :param end: end of the data
  :param title: title of the figure
  :param save_path: path where the figure will be saved to
  :param use_ax: the already-created ax to draw on
  :returns: plot
  :raises: none
  """
  # select plot data
  plot_data = df[start:end].copy()

  # get indicator names and plot ratio
  plot_ratio = args.get('plot_ratio')
  if plot_ratio is None :
    print('No indicator to plot')
    return None

  indicators = list(plot_ratio.keys())
  ratios = list(plot_ratio.values())
  num_indicators = len(indicators)
  
  # create figures
  fig = plt.figure(figsize=(20, num_indicators*unit_size))  
  gs = gridspec.GridSpec(num_indicators, 1, height_ratios=ratios)
  gs.update(wspace=ws, hspace=hs)

  axes = {}
  for i in range(num_indicators):
    tmp_indicator = indicators[i]
    tmp_args = args.get(tmp_indicator)
    axes[tmp_indicator] = plt.subplot(gs[i]) 
    axes[tmp_indicator].patch.set_alpha(0.5)

    if tmp_indicator == 'ichimoku':
      plot_ichimoku(df=plot_data, signal_col='ichimoku', title=tmp_indicator, use_ax=axes[tmp_indicator])

    else:
      target_col = tmp_args.get('target_col')
      benchmark = tmp_args.get('benchmark')
      boundary = tmp_args.get('boundary')
      color_mode = tmp_args.get('color_mode')
      price_col = tmp_args.get('price_col')
      signal_col = tmp_args.get('signal_col')
      pos_signal = tmp_args.get('pos_signal')
      neg_signal = tmp_args.get('neg_signal')
      none_signal = tmp_args.get('none_signal')
      filter_signal = tmp_args.get('filter_signal')
      plot_indicator(
        df=plot_data, target_col=target_col, benchmark=benchmark, boundary=boundary, color_mode=color_mode, 
        price_col=price_col, signal_col=signal_col, pos_signal=pos_signal, neg_signal=neg_signal, none_signal=none_signal, filter_signal=filter_signal,
        title=tmp_indicator, use_ax=axes[tmp_indicator])

  # adjust plot layout
  fig.suptitle(title, x=xp, y=yp)

  # save image
  if save_path is not None:
    plt.savefig(save_path + title + '.png')
    
  # close image
  if not show_image:
    plt.close(fig)

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
