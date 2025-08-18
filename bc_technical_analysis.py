# -*- coding: utf-8 -*-
"""
Technical Analysis Calculation and Visualization functions

:author: Beichen Chen
"""
import os
import math
import sympy
import datetime
import ta
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Literal, Optional, Any
from scipy.stats import linregress
from numpy.lib.stride_tricks import as_strided
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from quant import bc_util as util
from quant import bc_data_io as io_util

# set font for chinese characters 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# default values
default_ohlcv_col = {'close':'Close', 'open':'Open', 'high':'High', 'low':'Low', 'volume':'Volume'}
default_trend_val = {'pos_trend':'u', 'neg_trend':'d', 'none_trend':'', 'wave_trend':'n'}
default_signal_val = {'pos_signal':'b', 'neg_signal':'s', 'none_signal':'', 'wave_signal': 'n'}

# default indicators and dynamic trend for calculation
default_indicators = {'trend': ['ichimoku', 'kama', 'adx'], 'volume': [], 'momentum':['rsi'], 'volatility': [], 'other': []}
default_perspectives = ['candle', 'support_resistant']
default_support_resistant_col = ['kama_fast', 'kama_slow', 'tankan', 'kijun', 'renko_h', 'renko_l', 'candle_gap_top', 'candle_gap_bottom']

# default arguments for visualization
default_candlestick_color = {'color_up':'green', 'color_down':'red', 'shadow_color':'black', 'entity_edge_color':'black', 'alpha':1}
default_main_indicator = {'candlestick'}
default_plot_args = {'figsize':(30, 3), 'title_rotation':'vertical', 'xaxis_position': 'bottom', 'yaxis_position': 'right', 'title_x':-0.01, 'title_y':0, 'bbox_to_anchor':(1.02, 0.), 'loc':3, 'ncol':1, 'borderaxespad':0.0}

# zorders
default_zorders = {}
counter = 1
for item in ['default', 'price', 'gap', 'ichimoku', 'kama', 'renko', 'candle_pattern', 'candle_shadow', 'candle_entity', 'extended', ]:
  default_zorders[item] = counter
  counter += 1

# ================================================ Load configuration =============================================== # 
# load configuration
def load_config(root_paths: dict):
  """ 
  Load configuration from json file

  :param root_paths: a dictionary that contains home_path and git_path (differ by platforms)
  :returns: dictionary of config arguments
  :raises: None
  """
  # copy root paths
  config = root_paths.copy()

  # validate root paths
  for p in config.keys():
    if not os.path.exists(config[p]):
      print(f'{p} not exists!')

  # add derived paths
  config['config_path'] = config['git_path']    +   'quant/'      # github files
  config['quant_path'] = config['home_path']    +   'quant/'      # local files

  config['log_path'] = config['quant_path']     +   'logs/'       # logs of script execution
  config['api_path'] = config['quant_path']     +   'api_key/'    # configuration for data api, etc.
  config['futu_path'] = config['quant_path']    +   'futuopen/'   # configuration for futu platform
  config['tiger_path'] = config['quant_path']   +   'tigeropen/'  # configuration for tiger platform
  config['trader_path'] = config['quant_path']  +   'trader/'     # configuration for trader platforms
  config['data_path'] = config['quant_path']    +   'stock_data/' # downloaded stock data (OHLCV)
  config['result_path'] = config['quant_path']  +   'ta_model/'   # results of script execution
  
  # load self-defined pools (lists of stock symbols)
  config['selected_sec_list'] = io_util.read_config(file_path=config['config_path'], file_name='selected_sec_list.json')

  # load data api
  config['api_key'] = io_util.read_config(file_path=config['api_path'], file_name='api_key.json')

  # load calculation and visulization parameters
  ta_config = io_util.read_config(file_path=config['config_path'], file_name='ta_config.json')
  config.update(ta_config)

  return config

# load locally saved data(sec_data, ta_data, results)
def load_data(target_list: dict, config: dict, interval: str = 'day', load_empty_data: bool = False, load_derived_data: bool = False):
  """ 
  Load locally saved data(sec_data, ta_data, results)
  
  :param target_list: list of target pools(each pool is a list of stock symbols)
  :param config: dictionary of config arguments
  :param interval: data interval
  :param load_empty_data: whether to load empyt data dict
  :param load_derived_data: whether to load ta_data, result, final_result besides sec_data
  :returns: dictionary of data load from local files
  :raises: None
  """
  # init data dictionary
  data = {'sec_data': {}, 'ta_data': {}, 'result': {}, 'final_result': {}}
  for target in target_list:
    ti = f'{target}_{interval}'
    data['sec_data'][ti] = {}
    data['ta_data'][ti] = {}
    data['result'][ti] = {}

  # load sec data (data_path/[symbol].csv)
  if not load_empty_data:
    data_path = config['data_path']
    for target in target_list:
      ti = f'{target}_{interval}'
      for symbol in target_list[target]:
        if os.path.exists(data_path+f'{symbol.split(".")[0]}.csv'):
          data['sec_data'][ti][f'{symbol}_day'] = io_util.load_stock_data(file_path=data_path, file_name=symbol, standard_columns=True)
        else:
          data['sec_data'][ti][f'{symbol}_day'] = None

    # load derived data (quant_path/[ta_data/result].pkl)
    if load_derived_data:
      file_path = config["quant_path"]
      for target in target_list:
        ti = f'{target}_{interval}'
        for f in ['ta_data', 'result']:
          file_name = f'{ti}_{f}.pkl'
          if os.path.exists(f'{file_path}{file_name}'):
            data[f][ti] = io_util.pickle_load_data(file_path=file_path, file_name=f'{file_name}')
          else:
            print(f'{file_name} not exists')

  return data


# ================================================ Core progress ==================================================== # 
# preprocess stock data (OHLCV)
def preprocess(df: pd.DataFrame, symbol: str, print_error: bool = True):
  '''
  Preprocess stock data (OHLCV)

  :param df: downloaded stock data
  :param symbol: symbol of stock
  :param print_error: whether print error information or not
  :returns: preprocessed dataframe
  :raises: None
  '''
  # check whether data is empty or None
  if df is None or len(df) == 0:
    print(f'No data for preprocess')
    return None

  # drop duplicated rows, keep the first
  df = util.remove_duplicated_index(df=df, keep='first')

  # adjust close price manually (if split not updated)
  # adj_close = close
  # adj_rate = adj_close_p1 / adj_close   
  df['split_n1'] = df['Split'].shift(-1).fillna(1.0)
  df['adj_close_p1'] = df['Adj Close'].shift(1)
  df['adj_rate'] = df['adj_close_p1'] / df['Adj Close']
  df = df.sort_index(ascending=False)

  # 初始化 adj_factor
  start_idx = df.index[0]
  df.loc[start_idx, 'adj_factor'] = 1

  # Split在今日, 价格变化也在今日
  split_idx = df.query('Split != 1.0 and (adj_rate >= 1.95 or adj_rate <= 0.45)').index
  df.loc[split_idx, 'adj_factor'] = 1 / df.loc[split_idx, 'Split']

  # Split在昨日, 价格变化却在今日
  split_n1_idx = df.query('split_n1 != 1.0 and (adj_rate >= 1.95 or adj_rate <= 0.45)').index
  df.loc[split_n1_idx, 'adj_factor'] = 1 / df.loc[split_idx, 'split_n1']

  df['adj_factor'] = df['adj_factor'].fillna(method='ffill')
  df['Adj Close'] = df['Adj Close'] * df['adj_factor']

  if len(split_idx) != 0 or len(split_n1_idx) != 0:
    print(f'Price need to be adjusted due to Split on {split_idx}, {split_n1_idx}')

  # # 逐行处理, 以最新价格为基准, 向前复权(Forward Adjustment)   
  # adj_rate = 1
  # for idx, row in df.iterrows():

  #   df.loc[idx, 'Adj Close'] *= adj_rate

  #   # 当split != 1 (存在split), 且价格变化剧烈(adj_rate > 1.95 或 < 0.45)时
  #   # 将adj_rate设置为1/split, 往下走
  #   if row['Split'] != 1.0:
  #     if row['adj_rate'] >= 1.95 or row['adj_rate'] <= 0.45:
  #       adj_rate = 1/row['Split']
  #   # 当split_n1 != 1 (昨日存在split), 且价格变化剧烈(adj_rate > 1.95 或 < 0.45)时
  #   # 将adj_rate设置为1/split_n1, 往下走        
  #   elif row['split_n1'] != 1.0:
  #     if row['adj_rate'] >= 1.95 or row['adj_rate'] <= 0.45:
  #       adj_rate = 1/row['split_n1']

  df = df.sort_index()
  df.drop(['adj_rate', 'adj_close_p1', 'split_n1', 'adj_factor'], axis=1, inplace=True)

  # adjust open/high/low/close/volume values
  adj_rate = df['Adj Close'] / df['Close']
  for col in ['High', 'Low', 'Open', 'Close']:
    df[col] = df[col] * adj_rate

  # process NA and 0 values
  max_idx = df.index.max()
  na_cols = []
  zero_cols = []
  error_info = ''

  # check whether 0 or NaN values exists in the latest record
  extra_cols = ['Split', 'Dividend']
  cols = [x for x in df.columns if x not in extra_cols]
  for col in cols:
    if df.loc[max_idx, col] == 0:
      zero_cols.append(col)
    if np.isnan(df.loc[max_idx, col]):
      na_cols.append(col)
    
  # process NaN values
  if len(na_cols) > 0:
    error_info += 'NaN values found in '
    for col in na_cols:
      error_info += f'{col}'
  df['Split'] = df['Split'].fillna(1.0)
  df['Dividend'] = df['Dividend'].fillna(0.0)
  df = df.dropna()
    
  # process 0 values
  if len(zero_cols) > 0:
    error_info += '0 values found in '
    for col in zero_cols:
      error_info += f'{col}'
    df = df[:-1].copy()
  
  # print error information
  if print_error and len(error_info) > 0:
    error_info = f'[{symbol}]: on {max_idx.date()}, {error_info}'
    print(error_info)

  # add symbol and change rate of close price
  df['symbol'] = symbol
  
  return df

# calculate indicators according to definition
def calculate_ta_basic(df: pd.DataFrame, indicators: dict = default_indicators):
  '''
  Calculate indicators according to definition

  :param df: preprocessed stock data
  :param indicators: dictionary of different type of indicators to calculate
  :returns: dataframe with technical indicator columns
  :raises: None
  '''
  # check whether data is empty or None
  if df is None or len(df) == 0:
    print(f'No data for calculate_ta_basic')
    return None  

  # copy dataframe
  df = df.copy()

  # indicator calculation
  indicator = None
  try:
    # calculate close change rate
    phase = 'calculate close rate' 
    df = cal_change_rate(df=df, target_col=default_ohlcv_col['close'], add_accumulation=False)
    
    # calculate candlestick features
    phase = 'calculate candlestick' 
    df = add_candlestick_features(df=df)

    # add indicator features
    phase = 'calculate indicators' 
    indicator_calculated = []
    for i in indicators.keys():
      tmp_indicators = indicators[i]
      for indicator in tmp_indicators:
        if indicator not in indicator_calculated:
          df = eval(f'add_{indicator}_features(df=df)')
          indicator_calculated.append(indicator)
        else:
          print(f'{indicator} already calculated!')

  except Exception as e:
    print(f'[Exception]: @ {phase} - {indicator}, {e}')

  return df

# calculate static trend according to indicators
def calculate_ta_static(df: pd.DataFrame, indicators: dict = default_indicators):
  """
  Calculate static trend according to indicators (which is static to different start/end time).

  :param df: dataframe with several ta features
  :param indicators: dictionary of different type of indicators to calculate
  :returns: dataframe with static trend columns
  :raises: Exception 
  """
  # check whether data is empty or None
  if df is None or len(df) == 0:
    print(f'No data for calculate_ta_static')
    return None

  # columns to drop
  col_to_drop = []

  # trend calculation
  try:
    phase = 'calculate trend for trend_indicators'
    
    # ================================ ichimoku/kama trend ====================
    lines = {'ichimoku': {'fast': 'tankan', 'slow': 'kijun'}, 'kama': {'fast': 'kama_fast', 'slow': 'kama_slow'}}
    for target_indicator in ['ichimoku', 'kama']:
      if target_indicator in indicators['trend']:

        # fast/slow line column names
        fl = lines[target_indicator]['fast']
        sl = lines[target_indicator]['slow']
        fl_rate = f'{fl}_rate'
        sl_rate = f'{sl}_rate'
        fs_rate = f'{target_indicator}_rate'
 
        # distance related column names
        distance = f'{target_indicator}_distance'
        distance_middle = f'{target_indicator}_distance_middle'
        
        # distance
        df[distance] = df[fl] - df[sl]
        df[distance] = df[distance] / df[sl]

        # distance change(normalized by slow_line)
        # distance_change = f'{target_indicator}_distance_change'
        # df[distance_change] = df[distance] - df[distance].shift(1)

        # distance_day (fl & sl crossover)
        distance_day = f'{target_indicator}_distance_day'
        df[distance_day] = df[distance] > 0
        df[distance_day] = df[distance_day].replace({True: 1, False: -1})
        df[distance_day] = sda(series=df[distance_day], zero_as=None).astype(int)

        pos_none_idx = df.query(f'0 < {distance_day} and {distance} == 0').index
        neg_none_idx = df.query(f'0 > {distance_day} and {distance} == 0').index
        df.loc[pos_none_idx, distance_day] = 1
        df.loc[neg_none_idx, distance_day] = -1       
        
        # fl/sl change rate and fl & Close, sl & Close crossover
        threshold = 0.00
        for col in [fl, sl]:
          df = cal_change_rate(df=df, target_col=col, periods=1, add_accumulation=False, add_prefix=col, drop_na=False)
          df[f'{col}_day'] = cal_crossover_signal(df=df, fast_line='Close', slow_line=col, pos_signal=1, neg_signal=-1, none_signal=0)
          df[f'{col}_day'] = sda(series=df[f'{col}_day'], zero_as=1)

          # # fl/sl with none zero rate          
          # rate_col = f'{col}_rate'
          # none_zero_col = f'{rate_col}_none_zero'
          # df[none_zero_col] = np.NaN
          # none_zero_idx = df.query(f'{rate_col} > {threshold} or {rate_col} < {threshold}').index
          # df.loc[none_zero_idx, none_zero_col] = df.loc[none_zero_idx, rate_col]
          # df[none_zero_col] = df[none_zero_col].fillna(method='ffill')

        # # fl & sl crossover  
        # fs_day = f'{target_indicator}_cross_day'
        # df[fs_day] = cal_crossover_signal(df, fast_line=fl, slow_line=sl, result_col=fs_day, pos_signal='u', neg_signal='d', none_signal='n')
        # df[fs_day] = df[fs_day].replace({'u':1, 'd':-1, 'n':0})
        # df[fs_day] = sda(series=df[fs_day], zero_as=1)
        
        # # change
        # result_col = f'{target_indicator}_change'
        # status_col = f'{target_indicator}_status'
        # df[result_col] = 0
        # for col in [fl_rate, sl_rate]:
        #   df[result_col] += df[col]
        # col_symbol = (df[result_col] > 0).replace({True: 1, False: -1})
        # df[result_col] = normalize(df[result_col].abs()) * col_symbol

        # # status
        # threshold = 0.00
        # conditions = {
        #   'up':       f'({fl_rate} > {threshold} and {sl_rate} > {threshold}) or ({fl_rate} > {threshold} and {distance_change} >= {threshold})', 
        #   'down':     f'({fl_rate} < {threshold} and {sl_rate} < {threshold}) or ({fl_rate} < {threshold} and {distance_change} <= {threshold})',
        # } 
        # values = {
        #   'up':       1,
        #   'down':     -1
        # }
        # df = assign_condition_value(df=df, column=status_col, condition_dict=conditions, value_dict=values, default_value=0)

        # cloud top and bototm
        cloud_top_col = f'{target_indicator}_cloud_top'
        cloud_bottom_col = f'{target_indicator}_cloud_bottom'
        col_to_drop.append(cloud_top_col)
        col_to_drop.append(cloud_bottom_col)

        green_idx = df.query(f'{distance} > 0').index
        red_idx = df.query(f'{distance} <= 0').index
        df.loc[green_idx, cloud_top_col] = df.loc[green_idx, fl]
        df.loc[green_idx, cloud_bottom_col] = df.loc[green_idx, sl]
        df.loc[red_idx, cloud_top_col] = df.loc[red_idx, sl]
        df.loc[red_idx, cloud_bottom_col] = df.loc[red_idx, fl]

        df[distance_middle] = (df[cloud_top_col] + df[cloud_bottom_col])/2
        df[distance_middle] = df[distance_middle] - df[distance_middle].shift(1)
        df[fs_rate] = df[fl_rate] + df[sl_rate]

        conditions = {
          '上方': f'(candle_entity_bottom >= {target_indicator}_cloud_top)',
          '中上': f'((candle_entity_top > {target_indicator}_cloud_top) and ({target_indicator}_cloud_top > candle_entity_bottom >= {target_indicator}_cloud_bottom))',
          '中间': f'((candle_entity_top <= {target_indicator}_cloud_top) and (candle_entity_bottom >= {target_indicator}_cloud_bottom))',
          '穿刺': f'((candle_entity_top > {target_indicator}_cloud_top) and (candle_entity_bottom < {target_indicator}_cloud_bottom))',
          '中下': f'((candle_entity_bottom < {target_indicator}_cloud_bottom) and ({target_indicator}_cloud_top >= candle_entity_top > {target_indicator}_cloud_bottom))',
          '下方': f'(candle_entity_top <= {target_indicator}_cloud_bottom)'}
        values = {
          '上方': 'up', '中上': 'mid_up',
          '中间': 'mid', '穿刺': 'out',
          '中下': 'mid_down', '下方': 'down'}
        df = assign_condition_value(df=df, column=f'相对{target_indicator}位置', condition_dict=conditions, value_dict=values, default_value='')

    # ================================ adx trend ==============================
    target_indicator = 'adx'
    if target_indicator in indicators['trend']:
      
      # adx value and strength
      # df['adx_value'] = df['adx_diff_ma']
      # df['adx_strength'] = df['adx']
      df = cal_change(df=df, target_col='adx_value', add_accumulation=False, add_prefix=True)
      df = cal_change(df=df, target_col='adx_strength', add_accumulation=False, add_prefix=True)

      # # mute micro changes
      # wave_idx = df.query('-0.01 < adx_value_change < 0.01').index
      # df.loc[wave_idx, 'adx_value_change'] = 0
      # wave_idx = df.query('-0.01 < adx_strength_change < 0.01').index
      # df.loc[wave_idx, 'adx_strength_change'] = 0
      
      df['adx_direction'] = df['adx_value_change'] # sda of adx_value_change
      df['adx_power'] = df['adx_strength_change']  # sda of adx_strength_change
      # wave_idx = df.query('-1 < adx_direction < 1 and -0.5 < adx_power < 0.5').index

      # adx_value_prediction
      df['adx_value_prediction'] = df['adx_value'] + em(series=df['adx_value_change'], periods=3).mean()
      df['adx_value_prediction'] = em(series=df['adx_value_prediction'], periods=5).mean()
      df['adx_value_pred_change'] = df['adx_value_prediction'] - df['adx_value_prediction'].shift(1) 

      # direction(of value) and power(of strength)
      for col in ['adx_direction', 'adx_power']:
        # df.loc[wave_idx, col] = 0
        df[col] = sda(series=df[col], zero_as=0)

        threshold = 0
        conditions = {
          'up': f'{col} > {threshold}', 
          'down': f'{col} < {-threshold}', 
          'wave': f'{-threshold} <= {col} <= {threshold}'} 
        values = {
          'up': 1, 
          'down': -1,
          'wave': 0}
        df = assign_condition_value(df=df, column=f'{col}_day', condition_dict=conditions, value_dict=values, default_value=0) 
        df[f'{col}_day'] = sda(series=df[f'{col}_day'], zero_as=1) 

      # highest(lowest) value of adx_value of previous uptrend(downtrend)
      base_columns = {'adx_direction': 'adx_value', 'adx_power': 'adx_strength'}
      for col in ['adx_direction', 'adx_power']:
        start_col = f'{col}_start'
        day_col = f'{col}_day'
        base_col = base_columns[col]
        prev_col = f'prev_{base_col}'
        col_to_drop.append(prev_col)
        
        df[prev_col] = df[base_col].shift(1)
        extreme_idx = df.query(f'{day_col} == 1 or {day_col} == -1').index.tolist()

        for i in range(len(extreme_idx)):
          tmp_idx = extreme_idx[i]
          if i == 0:
            start = df.index.min()
            end = extreme_idx[i]
          else:
            start = extreme_idx[i-1]
            end = extreme_idx[i]

          tmp_day = df.loc[tmp_idx, day_col]
          # tmp_extreme = df[start:end][base_col].max() if tmp_day < 0 else df[start:end][base_col].min()
          tmp_start = df.loc[end, prev_col]
          df.loc[tmp_idx, start_col] = tmp_start

          if col == 'adx_power':
            df.loc[tmp_idx, 'adx_power_start_adx_value'] = df.loc[tmp_idx, 'adx_value']
        
        if start_col not in df.columns:
          df[start_col] = np.nan
        else:
          df[start_col] = df[start_col].fillna(method='ffill')

        if col == 'adx_power':
          if 'adx_power_start_adx_value' not in df.columns:
            df['adx_power_start_adx_value'] = np.nan
          else:
            df['adx_power_start_adx_value'] = df['adx_power_start_adx_value'].fillna(method='ffill')

      # previous adx_trend duration
      df['prev_adx_direction_day'] = df['adx_direction_day'].shift(1)
      col_to_drop.append('prev_adx_direction_day')
      start_idx = df.query('adx_direction_day == 1 or adx_direction_day == -1').index
      df.loc[start_idx, 'prev_adx_duration'] = df.loc[start_idx, 'prev_adx_direction_day']
      df['prev_adx_duration'] = df['prev_adx_duration'].fillna(method='ffill')

      # whether strength is strong or weak
      adx_strong_weak_threshold = 25
      conditions = {
        'up': f'adx_strength >= {adx_strong_weak_threshold}', 
        'down': f'adx_strength < {adx_strong_weak_threshold}'} 
      values = {
        'up': 1, 
        'down': -1}
      df = assign_condition_value(df=df, column='adx_strong_day', condition_dict=conditions, value_dict=values, default_value=0)
      df['adx_strong_day'] = sda(series=df['adx_strong_day'], zero_as=1)

      # whether value is waving around 0
      adx_wave_threshold = 10
      conditions = {
        'wave': f'{-adx_wave_threshold} <= adx_direction_start <= {adx_wave_threshold} and {-adx_wave_threshold} <= adx_value <= {adx_wave_threshold}', 
        'none': f'{-adx_wave_threshold} > adx_direction_start or adx_direction_start > {adx_wave_threshold} or {-adx_wave_threshold} > adx_value or adx_value > {adx_wave_threshold}'} 
      values = {
        'wave': 1, 
        'none': 0}
      df = assign_condition_value(df=df, column='adx_wave_day', condition_dict=conditions, value_dict=values, default_value=0)
      df['adx_wave_day'] = sda(series=df['adx_wave_day'], zero_as=None)

      # adx_trend
      # adx_distance: (实际值 - 预测值与), adx_status: ± of adx_distance
      df['adx_distance'] = df['adx_value'] - df['adx_value_prediction']
      df['adx_status'] = (df['adx_distance'] > 0).replace({True: 1, False: -1})
      df['adx_distance'] = normalize(df['adx_distance'].abs()) * df['adx_status']
      df['adx_distance_day'] = sda(series=df['adx_status'], zero_as=None)
      df['adx_distance_change'] = df['adx_distance'] - df['adx_distance'].shift(1)  
      
      df['adx_distance_middle'] = (df['adx_value'] + df['adx_value_prediction']) / 2
      df['adx_distance_middle'] = df['adx_distance_middle'] - df['adx_distance_middle'].shift(1)

      df['adx_rate'] = df['adx_value_change'] + df['adx_value_pred_change']

      # the true adx trend
      conditions = {
        'neg_u':      f'(adx_value < 0 and adx_value_change > 0 and adx_strength_change < 0)', 
        'neg_d':      f'(adx_value < 0 and adx_value_change < 0)',
        'pos_u':      f'(adx_value >=0 and adx_value_change > 0 and ((adx_value <= 10 and adx_strength_change < 0 and adx_direction_start < -5) or (adx_value > 10 and adx_strength_change > 0)))', 
        'pos_d':      f'(adx_value >=0 and adx_value_change < 0 and ((adx_value <= 10 and adx_value_change < -0.5) or (adx_value > 10 and adx_strength_change < 0)))',
      } 
      values = {
        'neg_u':      1, 
        'neg_d':      -1,
        'pos_u':      1, 
        'pos_d':      -1,
      }
      df = assign_condition_value(df=df, column='adx_trend', condition_dict=conditions, value_dict=values, default_value=0)
      df['adx_day'] = sda(df['adx_trend'], zero_as=None)

      # the true adx syn
      conditions = {
        'neg_u':      f'(adx_value < 0 and adx_value_change > 0 and adx_strength_change < 0)', 
        'neg_d':      f'(adx_value < 0 and adx_value_change < 0 and adx_strength_change > 0)',
        'pos_u':      f'(adx_value > 5 and adx_value_change > 0 and adx_strength_change > 0)', 
        'pos_d':      f'(adx_value > 5 and adx_value_change < 0 and adx_strength_change < 0)',
      } 
      values = {
        'neg_u':      1, 
        'neg_d':      -1,
        'pos_u':      1, 
        'pos_d':      -1,
      }
      df = assign_condition_value(df=df, column='adx_syn', condition_dict=conditions, value_dict=values, default_value=0)
      df['adx_syn'] = sda(df['adx_syn'], zero_as=None)
      
    # ================================ aroon trend ============================
    target_indicator = 'aroon'
    if target_indicator in indicators['trend']:
      aroon_col = ['aroon_up', 'aroon_down', 'aroon_gap']
      df[aroon_col] = df[aroon_col].round(1)
      for col in aroon_col:
        df = cal_change(df=df, target_col=col, add_prefix=True, add_accumulation=True)
        col_to_drop.append(f'{col}_change')
        col_to_drop.append(f'{col}_acc_change')
        col_to_drop.append(f'{col}_acc_change_count')

      # calculate aroon trend
      df['aroon_trend'] = ''

      # it is going up when:
      # 1+. aroon_up is extramely positive(96+)
      # 2+. aroon_up is strongly positive [88,100], aroon_down is strongly negative[0,12]
      up_idx = df.query('(aroon_up>=96) or (aroon_up>=88 and aroon_down<=12)').index
      df.loc[up_idx, 'aroon_trend'] = 'u'

      # it is going down when
      # 1-. aroon_down is extremely positive(96+)
      # 2-. aroon_up is strongly negative[0,12], aroon_down is strongly positive[88,100]
      down_idx = df.query('(aroon_down>=96) or (aroon_down>=88 and aroon_up<=12)').index
      df.loc[down_idx, 'aroon_trend'] = 'd'

      # otherwise up trend
      # 3+. aroon_down is decreasing, and (aroon_up is increasing or aroon_down>aroon_up)
      up_idx = df.query('(aroon_trend!="u" and aroon_trend!="d") and ((aroon_down_change<0) and (aroon_up_change>=0 or aroon_down>aroon_up))').index # and (aroon_up_acc_change_count <= -2 or aroon_down_acc_change_count <= -2)
      df.loc[up_idx, 'aroon_trend'] = 'u'

      # otherwise down trend
      # 3-. aroon_up is decreasing, and (aroon_down is increasing or aroon_up>aroon_down)
      down_idx = df.query('(aroon_trend!="u" and aroon_trend!="d") and ((aroon_up_change<0) and (aroon_down_change>=0 or aroon_up>aroon_down))').index #and (aroon_up_acc_change_count <= -2 or aroon_down_acc_change_count <= -2)
      df.loc[down_idx, 'aroon_trend'] = 'd'

      # it is waving when
      # 1=. aroon_gap keep steady and aroon_up/aroon_down keep changing toward a same direction
      wave_idx = df.query('-32<=aroon_gap<=32 and ((aroon_gap_change==0 and aroon_up_change==aroon_down_change<0) or ((aroon_up_change<0 and aroon_down<=4) or (aroon_down_change<0 and aroon_up<=4)))').index
      df.loc[wave_idx, 'aroon_trend'] = 'n'

    # ================================ kst trend ==============================
    target_indicator = 'kst'
    if target_indicator in indicators['trend']:
      conditions = {
        'up': 'kst_diff > 0', 
        'down': 'kst_diff <= 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='kst_trend', condition_dict=conditions, value_dict=values) 
    
    # ================================ cci trend ==============================
    target_indicator = 'cci'
    if target_indicator in indicators['trend']:
      conditions = {
        'up': 'cci_ma > 0', 
        'down': 'cci_ma <= 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='cci_trend', condition_dict=conditions, value_dict=values) 

    # ================================ trix trend =============================
    target_indicator = 'trix'
    if target_indicator in indicators['trend']:
      df['trix_trend'] = 'n'
      up_mask = df['trix'] > df['trix'].shift(1)
      down_mask = df['trix'] < df['trix'].shift(1)
      df.loc[up_mask, 'trix_trend'] = 'u'
      df.loc[down_mask, 'trix_trend'] = 'd'

    # ================================ psar trend =============================
    target_indicator = 'psar'
    if target_indicator in indicators['trend']:
      conditions = {
        'up': 'psar_up > 0', 
        'down': 'psar_down > 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='psar_trend', condition_dict=conditions, value_dict=values, default_value='')

    # ================================ stoch trend =============================
    target_indicator = 'stoch'
    if target_indicator in indicators['trend']:
      conditions = {
        'up': 'stoch_diff > 1 or stoch_k > 80', 
        'down': 'stoch_diff < -1 or stoch_k < 20'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='stoch_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # =========================================================================

    phase = 'calculate trend for volume_indicators'

    # ================================ eom trend ==============================
    target_indicator = 'eom'
    if target_indicator in indicators['volume']:
      conditions = {
        'up': 'eom_diff > 0', 
        'down': 'eom_diff <= 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='eom_trend', condition_dict=conditions, value_dict=values) 

    # ================================ fi trend ===============================
    target_indicator = 'fi'
    if target_indicator in indicators['volume']:
      conditions = {
        'up': 'fi_ema > 0', 
        'down': 'fi_ema <= 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='fi_trend', condition_dict=conditions, value_dict=values) 

    # =========================================================================
    
    phase = 'calculate trend for momentum_indicators'

    # ================================ eom trend ==============================
    target_indicator = 'rsi'
    # if target_indicator in indicators['momentum']:
    #   conditions = {
    #     'up': 'eom_diff > 0', 
    #     'down': 'eom_diff <= 0'} 
    #   values = {
    #     'up': 'u', 
    #     'down': 'd'}
    #   df = assign_condition_value(df=df, column='eom_trend', condition_dict=conditions, value_dict=values) 


    phase = 'calculate trend for volatility_indicators'

    # ================================ bb trend ===============================
    target_indicator = 'bb'
    if target_indicator in indicators['volatility']:
      conditions = {
        'up': 'Close < bb_low_band', 
        'down': 'Close > bb_high_band'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='bb_trend', condition_dict=conditions, value_dict=values, default_value='')

    # =========================================================================
    
    phase = 'calculate trend for other_indicators'

    # ================================ ao trend ===============================
    target_indicator = 'ao'
    if target_indicator in indicators['other']:
      df['ao_trend'] = 'n'
      up_mask = df['ao'] > df['ao'].shift(1)
      down_mask = df['ao'] < df['ao'].shift(1)
      df.loc[up_mask, 'ao_trend'] = 'u'
      df.loc[down_mask, 'ao_trend'] = 'd'
    
    # =========================================================================

    phase = 'calculate trend overall'

    # # ================================ overall trend ==========================
    # target_indicator = 'trend_day'
    # if target_indicator > '':
    
    #   # specify all indicators and specify the exclusives
    #   all_indicators = []
    #   for i in indicators.keys():
    #     all_indicators += [x for x in indicators[i] if x not in all_indicators]

    #   for indicator in all_indicators:
    #     trend_col = f'{indicator}_trend'
    #     day_col = f'{indicator}_day'

    #     if trend_col not in df.columns:
    #       df[trend_col] = 'n'
    #     if day_col not in df.columns:
    #       df[day_col] = 0

    #     # calculate number of days since trend shifted
    #     if indicator in ['bb']:
    #       df[day_col] = sda(series=df[trend_col].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1, one_restart=True) 
    #     else:
    #       df[day_col] = sda(series=df[trend_col].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1) 

    # remove redandunt columns
    for col in col_to_drop:
      if col in df.columns:
        df.drop(col, axis=1, inplace=True)

  except Exception as e:
    print(f'[Exception]: @ {phase} - {target_indicator}, {e}')
    
  return df

# calculate dynamic trend according to indicators and static trend
def calculate_ta_dynamic(df: pd.DataFrame, perspective: list = default_perspectives):
  """
  Calculate dynamic trend according to indicators and static trend (which is static to different start/end time).

  :param df: dataframe with technical indicators and their derivatives
  :param perspective: for which indicators[renko, linear, candle, support_resistant], derivative columns that need to calculated 
  :returns: dataframe with dynamic trend columns
  :raises: None
  """
  # check whether data is empty or None
  if df is None or len(df) == 0:
    print(f'No data for calculate_ta_dynamic')
    return None  

  # copy dataframe
  df = df.copy()
  
  # derivatives calculation
  try:

    # ================================ renko analysis ============================
    phase = 'renko analysis'
    if 'renko' in perspective:
      
      ## 在解决[不同数据开始日期导致renko动态变化]问题之前, 不建议使用renko, 容易造成混乱
      # add renko features
      df = add_renko_features(df=df)
      
    # ================================ candle analysis ===========================
    phase = 'candle analysis'
    if 'candle' in perspective:
      
      df = add_candlestick_patterns(df=df)

    # ================================ linear analysis ===========================
    phase = 'linear analysis'
    if 'linear' in perspective:
      
      # add linear features
      df = add_linear_features(df=df)

    #   # crossover signals between Close and linear_fit_high/linear_fit_low, # , 'support', 'resistant'
    #   for col in ['linear_fit_high', 'linear_fit_low']:
    #     signal_col = None

    #     if col in df.columns:
    #       signal_col = f'{col}_signal'
    #       df[signal_col] = cal_crossover_signal(df=df, fast_line='Close', slow_line=col, pos_signal=1, neg_signal=-1, none_signal=0)

    #       if signal_col in df.columns:
    #         df[signal_col] = sda(series=df[signal_col], zero_as=1)
    #     else:
    #       print(f'{col} not in df.columns')
      
    #   # trends from linear fit 
    #   # hitpeak or rebound
    #   conditions = {
    #     'up': '((linear_fit_low_stop >= 1 and linear_fit_support < candle_entity_bottom) and (Close > linear_fit_low_stop_price))', 
    #     'down': '((linear_fit_high_stop >= 1 and linear_fit_resistant > candle_entity_top) and (Close < linear_fit_high_stop_price))',
    #     'wave': '(linear_fit_high_slope == 0 and linear_fit_low_slope == 0)'} 
    #   values = {
    #     'up': 'u', 
    #     'down': 'd',
    #     'wave': 'n'}
    #   df = assign_condition_value(df=df, column='linear_bounce_trend', condition_dict=conditions, value_dict=values)
    #   df['linear_bounce_trend'] = df['linear_bounce_trend'].fillna(method='ffill').fillna('')
    #   df['linear_bounce_day'] = sda(series=df['linear_bounce_trend'].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1)
      
    #   # break through up or down
    #   conditions = {
    #     'up': '((linear_fit_high_stop >= 5 or linear_fit_high_slope == 0) and linear_fit_high_signal >= 1 and ((candle_color == 1 and candle_entity_top > linear_fit_resistant) or (candle_entity_bottom > linear_fit_resistant)))', 
    #     'down': '((linear_fit_low_stop >= 5 or linear_fit_low_slope == 0) and linear_fit_low_signal <= -1 and ((candle_color == -1 and candle_entity_bottom < linear_fit_support) or (candle_entity_top < linear_fit_support)))'} 
    #   values = {
    #     'up': 'u', 
    #     'down': 'd'}
    #   df = assign_condition_value(df=df, column='linear_break_trend', condition_dict=conditions, value_dict=values)
    #   df['linear_break_trend'] = df['linear_break_trend'].fillna(method='ffill').fillna('')
    #   df['linear_break_day'] = sda(series=df['linear_break_trend'].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1)

    #   # focus on the last row only
    #   max_idx = df.index.max()
    #   valid_idxs = df.query('linear_slope == linear_slope').index

    # ================================ support & resistant =======================
    phase = 'support and resistant'
    if 'support_resistant' in perspective:
      df = add_support_resistance(df)

  except Exception as e:
    print(phase, e)

  return df

# generate description for ta features
def calculate_ta_score(df: pd.DataFrame):
  """
  Generate description for latest ta_data of symbols(aka. result).

  :param df: dataframe which contains latest ta_data of symbols, each row for a symbol
  :returns: dataframe with description
  :raises: None
  """

  # initialization
  df['trigger_score'] = 0
  col_to_drop = []

  # ================================ calculate trigger/position score =======
  df['up_score'] = 0
  df['down_score'] = 0
  df['up_score_description'] = ''
  df['down_score_description'] = ''

  # support/resistant, break_up/bread_down, candle_pattern description
  df['up_score'] += df['break_up_score'] + df['support_score'] * 0.5 # + df['up_pattern_score'] + df['cross_up_score']  
  df['down_score'] += df['break_down_score'] + df['resistant_score'] * 0.5 # + df['down_pattern_score'] + df['cross_down_score'] 

  # descriptions
  names = {'break_up': '+突破', 'break_down': '-跌落', 'support':'+支撑', 'resistant': '-阻挡', }  # 'up_pattern': '+蜡烛', 'down_pattern': '-蜡烛', 
  for col in names.keys():

    desc = df[f'{col}_score'].apply(lambda x: '' if x == 0 else f'{names[col]}({x}), ')
    # df[f'{col}_description'].apply(lambda x: '' if x == '' else f'{names[col]}:[{x}], ')

    if col in ['break_up', 'support', 'up_pattern']:
      df['up_score_description'] = (desc + df['up_score_description'])
    else:
      df['down_score_description'] = (desc + df['down_score_description'])

  # trigger_score sum up
  df['trigger_score'] = (df['up_score'] + df['down_score']).round(2) # 
  df['up_score_description'] = df['up_score_description'].apply(lambda x: x[:-2] if (len(x) >=2 and x[-2] == ',') else x)
  df['down_score_description'] = df['down_score_description'].apply(lambda x: x[:-2] if (len(x) >=2 and x[-2] == ',') else x)

  # trigger days
  trigger_conditions = {
    'pos':    f'trigger_score > 0.0', 
    'neg':    f'trigger_score < 0.0',
  } 
  trigger_values = {
    'pos':    1, 
    'neg':    -1,
  }
  df = assign_condition_value(df=df, column='trigger_day', condition_dict=trigger_conditions, value_dict=trigger_values, default_value=0)
  df['trigger_day'] = sda(series=df['trigger_day'], zero_as=1)

  # ================================ calculate overall change & status ======
  df['overall_change'] = 0
  df['overall_status'] = 0

  # adx/ichimoku/kama distance
  threhold = 0.00
  for col in ['adx', 'ichimoku', 'kama']:

    distance_col = f'{col}_distance'
    distance_middle_col = f'{col}_distance_middle'
    result_col = f'{col}_distance_status'
    status_col = f'{col}_status'
    rate_col = f'{col}_rate'
    df[result_col] = ''

    col_to_drop += [distance_middle_col, status_col, rate_col]

    flr = {'kama_distance': 'kama_fast_rate', 'ichimoku_distance': 'tankan_rate', 'adx_distance': 'adx_value_change'}[distance_col]
    slr = {'kama_distance': 'kama_slow_rate', 'ichimoku_distance': 'kijun_rate', 'adx_distance': 'adx_value_pred_change'}[distance_col]

    # distance 决定 color 与 alpha
    distance_conditions = {
      'pos': f'{distance_col} > {threhold}', 
      'neg': f'{distance_col} < {-threhold}', 
      'none':f'{-threhold} <= {distance_col} <= {threhold}'
    }
    
    # distance_change 决定 marker
    distance_change_conditions = {
      'up': f'''
            (
              ({flr} > {threhold} and {slr} > {threhold}) or 
              ({rate_col} > 0 and {distance_middle_col} > {threhold})
            )
            '''.replace('\n', ''), 

      'down': f'''
            (
              ({flr} < {-threhold} and {slr} < {-threhold}) or 
              ({rate_col} < {-threhold} and {distance_middle_col} < {-threhold})
            )
            '''.replace('\n', ''), 

      'none':f'''
            (
              ({-threhold} <= {flr} <= {threhold}) and ({-threhold} <= {slr} <= {threhold}) or
              ({rate_col} < {-threhold} and {distance_middle_col} > {threhold}) or
              ({rate_col} > {threhold} and {distance_middle_col} < {-threhold})
            )
            '''.replace('\n', ''),  
    }
    
    # 综合 distance 和 distance_change
    for d in distance_conditions.keys():
      for dc in distance_change_conditions.keys():
        tmp_condition = distance_conditions[d] + ' and ' + distance_change_conditions[dc]
        tmp_match = df.query(tmp_condition).index
        df.loc[tmp_match, result_col] = f'{d}{dc}'

    df[status_col] = (df[rate_col] > 0).replace({True: 1, False: -1})
    df['overall_change'] += normalize(df[rate_col].abs()) * df[status_col]
    df['overall_status'] += df[status_col]

  # overall change and status 
  df['overall_change_day'] = (df['overall_change'] > 0).replace({True: 1, False: -1})
  df['overall_change'] = normalize(df['overall_change'].abs()) * df['overall_change_day']
  df['overall_change_diff'] = df['overall_change'] - df['overall_change'].shift(1)
  df['overall_change_day'] = sda(series=df['overall_change_day'], zero_as=1) 

  # remove redandunt columns
  for col in col_to_drop:
    if col in df.columns:
      df.drop(col, axis=1, inplace=True)

  return df

# calculate all features (ta_basic + ta_static + ta_dynamic + ta_score) all at once
def calculate_ta_feature(df: pd.DataFrame, symbol: str, start_date: str = None, end_date: str = None, indicators: dict = default_indicators):
  """
  Calculate all features (ta_data + ta_static + ta_dynamic) all at once.

  :param df: original dataframe with hlocv features
  :param symbol: symbol of the data
  :param start_date: start date of calculation
  :param end_date: end date of calculation
  :param indicators: dictionary of different type of indicators to calculate
  :returns: dataframe with ta indicators, static/dynamic trend
  :raises: None
  """
  # check whether data is empty or None
  if df is None or len(df) == 0:
    print(f'{symbol}: No data for calculate_ta_feature')
    return None   
  
  time_counter = {}
  time_counter['start'] = datetime.datetime.now()
  try:    
    # # preprocess sec_data
    phase = 'preprocess'
    time_counter[phase] = datetime.datetime.now()
    df = preprocess(df=df[start_date:end_date], symbol=symbol).copy()
    
    # calculate TA indicators
    phase = 'cal_ta_basic_features' 
    time_counter[phase] = datetime.datetime.now()
    df = calculate_ta_basic(df=df, indicators=indicators)

    # calculate TA static trend
    phase = 'cal_ta_static_features'
    time_counter[phase] = datetime.datetime.now()
    df = calculate_ta_static(df=df, indicators=indicators)

    # calculate TA dynamic trend
    phase = 'cal_ta_dynamic_features'
    time_counter[phase] = datetime.datetime.now()
    df = calculate_ta_dynamic(df)

    # calculate TA scores
    phase = 'cal_ta_score'
    time_counter[phase] = datetime.datetime.now()
    df = calculate_ta_score(df)

    time_counter['end'] = datetime.datetime.now()

  except Exception as e:
    print(symbol, phase, e)

  # for i in range(1, len(time_counter)):
  #   ls = list(time_counter.keys())[i-1]
  #   le = list(time_counter.keys())[i]
  #   ts = time_counter[ls]
  #   te = time_counter[le]
  #   print(f'{ls}: {te-ts}')
  # print(f'total: {time_counter["end"] - time_counter["start"]}')
    
  return df

# calculate signal according to features
def calculate_ta_signal(df: pd.DataFrame):
  """
  Calculate signal according to features.

  :param df: dataframe with ta features and derived features for calculating signals
  :raturns: dataframe with signal
  :raises: None
  """

  if df is None or len(df) == 0:
    print(f'No data for calculate_ta_signal')
    return None

  # columns to drop          
  df['prev_adx_day'] = sda(df['adx_trend'].shift(1), zero_as=0)
  df['adx_wave'] = df['adx_value_change'].abs() + df['adx_strength_change'].abs()
  df['ki_distance_sum'] = df['ichimoku_distance'] + df['kama_distance']
  df['final_score'] = df['adx_distance_change'] + df['overall_change_diff']
  df['final_score_change'] = df['final_score'] - df['final_score'].shift(1)
  col_to_drop = ['prev_adx_day', 'adx_wave', 'ki_distance_sum']
  
  # ================================ calculate position =====================
  if 'position' > '':

    # 低/中低/中高/高
    position_conditions = {
      '低':        f'ki_distance in ["rr"]',
      '中低':      f'ki_distance in ["rg", "rn"]',
      '中高':      f'ki_distance in ["gr", "gn"]',
      '高':        f'ki_distance in ["gg"]',

    } 
    position_values = {
      '低':        f'l',
      '中低':      f'ml',
      '中高':      f'mh',
      '高':        f'h',
    }
    df = assign_condition_value(df=df, column='位置', condition_dict=position_conditions, value_dict=position_values, default_value='') 
    
  # ================================ calculate trend ========================
  if 'trend'  > '':
  
    # trend
    df['trend'] = ''
    df['trend_score'] = 0 
    df['trend_description'] = ''

    # up: 基础方向向上
    up_data = df.query('adx_day > 0 or (adx_day == 0 and adx_value_change > 1)')
    df.loc[up_data.index, 'trend'] = 'up'
    up_condition = {
      '低位':   [
        # 起始位置低
        'adx_direction_start < -20', 
        # ichimoku红云
        'ichimoku_distance < 0', 
        # kama红云
        'kama_distance < 0', 
        # 处于低(中低)位
        'position_score < 0'
      ],

      '反转':   [
        # adx_trend由负转正
        'adx_day == 0 and prev_adx_day < 0 and adx_value_change > 1', 
        # adx_distance红云收窄
        'adx_distance_status == "negup"', 
        # ichimoku_distance红云收窄
        'ichimoku_distance_status == "negup"', 
        # kama_distance红云收窄
        'kama_distance_status == "negup"'
      ],

      '上行':   [
        # adx_trend正向
        'adx_day > 0', 
        # adx_distance绿云扩大
        'adx_distance_status == "posup"', 
        # 整体趋势向上
        'overall_change > 0 and overall_change_diff > 0'
      ]
    }
    for uc in up_condition.keys():
      
      # 当前方向分数列名
      tmp_score_col = f'trend_{uc}'    
      col_to_drop.append(tmp_score_col)

      # 初始化
      df[tmp_score_col] = 0
      tmp_idx_merge = None

      # 遍历细分方向条件
      for q in up_condition[uc]:
        tmp_idx = up_data.query(q).index

        condition_value = (df.loc[tmp_idx, 'position_score'] * -0.5) if q == 'position_score < 0' else 1
        df.loc[tmp_idx, tmp_score_col] += condition_value

        if tmp_idx_merge is None:
          tmp_idx_merge = pd.Index(tmp_idx)
        else:
          index2 = pd.Index(tmp_idx)
          tmp_idx_merge = tmp_idx_merge.union(index2)
      
      df['trend_score'] += df[tmp_score_col]
      df.loc[tmp_idx_merge, 'trend_description'] += f' {uc}(' + df.loc[tmp_idx_merge, tmp_score_col].astype(str) + ')'

    # down: 基础方向向下
    down_data = df.query('adx_day < 0 or (adx_day == 0 and adx_value_change < -1)')
    df.loc[down_data.index, 'trend'] = 'down'
    down_condition = {
      '高位':   [
        # 起始位置高
        'adx_direction_start > 20', 
        # ichimoku绿云
        'ichimoku_distance > 0', 
        # kama绿云
        'kama_distance > 0', 
        # 处于高(中高)位
        'position_score > 0'
      ],

      '反转':   [
        # adx_trend由正转负
        'adx_day == 0 and prev_adx_day > 0 and adx_value_change < -1', 
        # adx_distance绿云收窄
        'adx_distance_status == "posdown"', 
        # ichimoku_distance绿云收窄
        'ichimoku_distance_status == "posdown"', 
        # kama_distance绿云收窄
        'kama_distance_status == "posdown"'
      ],

      '下行':   [
        # adx趋势向下
        'adx_day < 0', 
        # adx_distance红云扩大
        'adx_distance_status == "negdown"', 
        # 整体趋势向下
        'overall_change < 0 and overall_change_diff < 0'
      ],
    }
    for dc in down_condition.keys():  
      
      # 当前方向分数列名
      tmp_score_col = f'trend_{dc}'    
      col_to_drop.append(tmp_score_col)

      # 初始化
      df[tmp_score_col] = 0
      tmp_idx_merge = None

      # 遍历细分方向条件
      for q in down_condition[dc]:
        tmp_idx = down_data.query(q).index

        condition_value = (df.loc[tmp_idx, 'position_score'] * -0.5) if q == 'position_score > 0' else -1
        df.loc[tmp_idx, tmp_score_col] += condition_value

        if tmp_idx_merge is None:
          tmp_idx_merge = pd.Index(tmp_idx)
        else:
          index2 = pd.Index(tmp_idx)
          tmp_idx_merge = tmp_idx_merge.union(index2)

      df['trend_score'] += df[tmp_score_col]
      df.loc[tmp_idx_merge, 'trend_description'] += f' {dc}(' + df.loc[tmp_idx_merge, tmp_score_col].astype(str) + ')'

    # 通用: 波动趋势
    wave_condition = {
      '波动':   [
        # adx强度弱(adx_strong_day < 0), 当前值在波动区间(-10 < adx_value < 10)
        '(adx_strong_day < 0) and (-10 < adx_value < 10)',
        # adx强度弱(adx_strong_day < 0), 起始于波动区间(-10 < adx_direction_start < 10)
        '(adx_strong_day < 0) and (-10 < adx_direction_start < 10)',
        # adx强度弱(adx_strong_day < 0), 长时间弱势(adx_strong_day <= -20)
        '(adx_strong_day < 0) and (adx_strong_day <= -20)',
      ]
    }
    for wc in wave_condition.keys():
      
      # 当前方向分数列名
      tmp_score_col = f'trend_{wc}'    
      # col_to_drop.append(tmp_score_col)

      # 初始化
      df[tmp_score_col] = 0
      tmp_idx_merge = None

      # 遍历细分方向条件
      for q in wave_condition[wc]:
        tmp_idx = df.query(q).index

        condition_value = -1
        df.loc[tmp_idx, tmp_score_col] += condition_value

        if tmp_idx_merge is None:
          tmp_idx_merge = pd.Index(tmp_idx)
        else:
          index2 = pd.Index(tmp_idx)
          tmp_idx_merge = tmp_idx_merge.union(index2)
      
      df['trend_score'] += df[tmp_score_col]
      df.loc[tmp_idx_merge, 'trend_description'] += f' {wc}(' + df.loc[tmp_idx_merge, tmp_score_col].astype(str) + ')'
    
    # 蜡烛: 相对蜡烛位置
    df['trend_score'] += df['candle_position_score']
    df['trend_score'] = df['trend_score'].round(2)
    df['trend_description'] += f' 蜡烛(' + df['candle_position_score'].astype(str) + ')'

    # exceptions: 例外情况
    none_trend_conditions = {
      'false_up':   'adx_value_change > 0 and adx_day == 0 and (candle_position_score < 0)',
      'false_down': 'adx_value_change < 0 and adx_day == 0 and (candle_position_score > 0)'
    } 
    for c in none_trend_conditions.keys():
      tmp_condition = none_trend_conditions[c]
      tmp_idx = df.query(tmp_condition).index
      df.loc[tmp_idx, 'trend'] = ''
    
    df['trend_day'] = sda(df['trend'].replace({'':0, 'up':1, 'down': -1}), zero_as=1)
    df['prev_trend_day'] = df['trend_day'].shift(1)
  
  # ================================ calculate pattern ======================
  if 'pattern'  > '':
    df['pattern_up_score'] = 0
    df['pattern_down_score'] = 0
    df['pattern_score'] = 0
    df['pattern_description'] = ''
    df['pattern_up'] = ''
    df['pattern_down'] = ''    
    col_to_drop += ['pattern_up', 'pattern_down']

    # mark pattern
    # pattern_up = []
    # pattern_down = []
    pattern_conditions = {

      # 超卖
      '超买超卖_up':            '''
                            (rsi < 30)
                            '''.replace('\n', ''),
      # 超买
      '超买超卖_down':          '''
                            (rsi > 70)
                            '''.replace('\n', ''),

      # 关键交叉(ichimoku, kama)
      '关键交叉i_up':            '''
                            (
                              (ichimoku_distance_day == 1)
                            )
                            '''.replace('\n', ''),

      # 关键交叉(ichimoku, kama)
      '关键交叉i_down':          '''
                            (
                              (ichimoku_distance_day == -1)
                            )
                            '''.replace('\n', ''),

      # 关键交叉(ichimoku, kama)
      '关键交叉k_up':            '''
                            (
                              (kama_distance_day == 1)
                            )
                            '''.replace('\n', ''),

      # 关键交叉(ichimoku, kama)
      '关键交叉k_down':          '''
                            (
                              (kama_distance_day == -1)
                            )
                            '''.replace('\n', ''),

      # 关键边界(ichimoku, kama, gap)
      '长线边界_up':            '''
                            (
                              位置 in ['h', 'mh'] and
                              (
                                (kama_slow_support > 0) or 
                                (kama_slow_break_down < 0 and candle_color == 1 and 长影线_trend == "u") or
                                (kama_slow_break_up > 0 and (candle_position_score > 0 or candle_color == 1))
                              )
                            )
                            '''.replace('\n', ''),

      # 关键边界(ichimoku, kama, gap)
      '长线边界_down':          '''
                            (
                              位置 in ['l', 'ml'] and
                              (
                                (kama_slow_resistant < 0) or 
                                (kama_slow_break_up > 0 and (candle_color == -1 or 长影线_trend == "d" or resistant_score < 0)) or
                                (kama_slow_break_down < 0 and (candle_position_score < 0 or candle_color == -1))
                              ) 
                            )
                            '''.replace('\n', ''),
    } 

    # calculate pattern score and description
    p_up_desc = {
      '超买超卖': '超卖',
      '关键突破': '突破',
      '关键交叉i': '金叉(ichi)',
      '关键交叉k': '金叉(kama)',
      '长线边界': '长线支撑'
    }
    p_down_desc = {
      '超买超卖': '超买',
      '关键突破': '跌落',
      '关键交叉i': '死叉(ichi)',
      '关键交叉k': '死叉(kama)',
      '长线边界': '长线阻挡'
    }
    for c in pattern_conditions.keys():
      
      name, direction = c.split('_')
      if name not in df.columns:
        df[name] = 0

      # # get index which matches the condition
      tmp_condition = pattern_conditions[c]
      tmp_idx = df.query(tmp_condition).index

      # mark up/down pattern
      if direction == 'up':
        df.loc[tmp_idx, name] += 1
        df.loc[tmp_idx, 'pattern_score'] += 1
        df.loc[tmp_idx, 'pattern_up_score'] += 1
        df.loc[tmp_idx, 'pattern_up'] += p_up_desc[name] + ', '

      elif direction == 'down':
        df.loc[tmp_idx, name] = -1
        df.loc[tmp_idx, 'pattern_score'] -= 1
        df.loc[tmp_idx, 'pattern_down_score'] -= 1
        df.loc[tmp_idx, 'pattern_down'] += p_down_desc[name] + ', '

      else:
        pass

    # # exceptions
    # none_pattern_conditions = {
    # } 
    # for c in none_pattern_conditions.keys():
    #   tmp_condition = none_pattern_conditions[c]
    #   tmp_idx = df.query(tmp_condition).index
    #   df.loc[tmp_idx, c] = 0

    # for c in pattern_conditions.keys():
    #   p_name = c.split('_')[0]

    #   if 'up' in c:
    #     tmp_idx = df.query(f'{c} == 1').index
    #     df.loc[tmp_idx, 'pattern_score'] += 1
    #     df.loc[tmp_idx, 'pattern_up_score'] += 1
    #     df.loc[tmp_idx, 'pattern_up'] += p_up_desc[p_name] + ', '
    #   elif 'down' in c:
    #     tmp_idx = df.query(f'{c} == -1').index
    #     df.loc[tmp_idx, 'pattern_score'] -= 1
    #     df.loc[tmp_idx, 'pattern_down_score'] -= 1
    #     df.loc[tmp_idx, 'pattern_down'] += p_down_desc[p_name] + ', '
    #   else:
    #     pass
  
    # final post-processing
    df['pattern_score'] = df['pattern_score'].round(2)
    df['pattern_up'] = df['pattern_up'].apply(lambda x: '+[' + x[:-2] + ']' if len(x) > 0 else '')
    df['pattern_down'] = df['pattern_down'].apply(lambda x: '-[' + x[:-2] + ']' if len(x) > 0 else '')
    df['pattern_description'] = df['pattern_up'] + ' | ' + df['pattern_down']
    df['pattern_description'] = df['pattern_description'].apply(lambda x: '' if x == ' | ' else x)
    df['pattern_description'] = df['pattern_description'].apply(lambda x: x[:-2] if (len(x) > 2 and x[-2] == '|') else x)
    df['pattern_description'] = df['pattern_description'].apply(lambda x: x[2:] if (len(x) > 2 and x[1] == '|') else x)
    df['pattern_description'] = df['pattern_description'].apply(lambda x: x[:-1] if (len(x) > 2 and x[-2] == ']') else x)

  # ================================ calculate signal =======================
  if 'signal'  > '':
  
    # signal
    df['signal'] = ''
    df['signal_day'] = 0
    df['signal_description'] = ''

    # # signal conditions
    # conditions = {

    #   '趋势_buy':   '''
    #                 Close< 0
    #                 '''.replace('\n', ''),

    #   '趋势_sell':  '''
    #                 Close< 0
    #                 '''.replace('\n', ''),   
      
    #   '转换_buy':   '''
    #                 (trend_day == 1 and prev_trend_day < 0) and
    #                 (adx_day > 0 or (final_score > 0.25))
    #                 '''.replace('\n', ''),

    #   '转换_sell':  '''
    #                 (trend_day == -1 and prev_trend_day > 0) and
    #                 (adx_day < 0 or (final_score < -0.25))
    #                 '''.replace('\n', ''),   

    #   '触发_buy':   '''
    #                 (Close < 0)
    #                 '''.replace('\n', ''),

    #   '触发_sell':  '''
    #                 (Close < 0)
    #                 '''.replace('\n', ''),    

    #   '前瞻_buy':   '''
    #                 (Close < 0)
    #                 '''.replace('\n', ''),

    #   '前瞻_sell':  '''
    #                 (Close < 0)
    #                 '''.replace('\n', ''),    
      
    #   '蜡烛_buy':   '''
    #                 (candle_pattern_score > 0)
    #                 '''.replace('\n', ''),

    #   '蜡烛_sell':  '''
    #                 (candle_pattern_score < 0)
    #                 '''.replace('\n', ''),  
    # }
    # for c in conditions.keys():
    #   tmp_name, tmp_action = c.split('_')
    #   if tmp_name not in df.columns:
    #     df[tmp_name] = 0

    #   tmp_query = conditions[c]
    #   tmp_idx = df.query(tmp_query).index
    #   df.loc[tmp_idx, tmp_name] = 1 if tmp_action == 'buy' else -1
    
    # buy_idx = df.query('转换 > 0').index
    # df.loc[buy_idx, 'signal'] = 'b'

    # sell_idx = df.query('转换 < 0').index
    # df.loc[sell_idx, 'signal'] = 's'

    # # disable some false alarms
    # none_signal_idx = []
    # none_signal_conditions = {
      
    #   # B|S:  无adx强度数据  
    #   '信号不全':           '''
    #                         (signal == "b" or signal == "s") and (adx_power_day == 0)
    #                         '''.replace('\n', ''),

    #   # B: 去下降趋势中的买入信号  
    #   '距离过大':           '''
    #                         (signal == "b") and
    #                         (ki_distance in ['rr']) and
    #                         (
    #                           (ichimoku_distance < -0.15 and kama_distance < -0.15) or
    #                           (ki_distance_sum < -0.125)
    #                         )
    #                         '''.replace('\n', ''),

    #   # B: 受到阻挡  
    #   '受到阻挡':           '''
    #                         (signal == "b") and
    #                         (resistant_score < 0) and
    #                         (candle_color == -1) and 
    #                         (
    #                           (candle_position_score < 0) or 
    #                           (candle_upper_shadow_pct > 0.4 > candle_lower_shadow_pct) or
    #                           (entity_trend != "d" and candle_entity_pct > 0.6)
    #                         )
    #                         '''.replace('\n', ''),

    #   # B: 高位买入  
    #   '高位买入':           '''
    #                         (signal == "b") and
    #                         (
    #                           (位置 in ['mh', 'h'] and (十字星_trend != "n" or position_score > 1))
    #                         )
    #                         '''.replace('\n', ''),

    #   # B: 中高位波动买入  
    #   '波动买入':           '''
    #                         (signal == "b") and
    #                         (
    #                           (位置 in ['mh', 'h'] and adx_strong_day < 0 and adx_wave_day > 0 and adx_direction < 5)
    #                         )
    #                         '''.replace('\n', ''),

    #   # S: 高位卖出  
    #   '高位卖出':           '''
    #                         (signal == "s") and
    #                         (
    #                           (位置 in ['h'] and ki_distance == 'gg' and position_score >= 2 and break_down_score == 0)
    #                         )
    #                         '''.replace('\n', ''),

    #   # B: 低位下行  
    #   '低位下行':           '''
    #                         (signal == "b") and
    #                         (
    #                           (位置 in ['l'] and ki_distance == 'rr' and kama_fast > kijun)
    #                         )
    #                         '''.replace('\n', '')

    # } 
    # for c in none_signal_conditions.keys():
    #   df[c] = 0
    #   tmp_condition = none_signal_conditions[c]
    #   tmp_idx = df.query(tmp_condition).index
    #   df.loc[tmp_idx, c] = -1
    #   df.loc[tmp_idx, 'signal_description'] += f'{c}, '
    #   none_signal_idx += tmp_idx.tolist()    
    # none_signal_idx = list(set(none_signal_idx))
    # df.loc[none_signal_idx, 'signal'] = 'n' + df.loc[none_signal_idx, 'signal']
    # df['signal_description'] = df['signal_description'].apply(lambda x: x[:-2])

    # # signal day
    # df['signal_day'] = sda(df['signal'].replace({'b': 1, 's': -1, '': 0, 'nb': 1, 'ns': -1}), zero_as=1)  

    # signal score
    df['signal_score'] = 0

    signal_conditions = {

      # '首日触发':            '''
      #                       ki_distance == "rr" and adx_direction_day == 1
      #                       '''.replace('\n', ''),

      '趋势向上':            '''
                            trend == "up"
                            '''.replace('\n', ''),

      '低位':               '''
                            位置 in ['l']
                            '''.replace('\n', ''),

      '上行':               '''
                            candle_position_score > 0
                            '''.replace('\n', ''),
      
      '突破':               '''
                            break_up_score > 0
                            '''.replace('\n', ''),

      '无波动':             '''
                            trend_波动 == 0
                            '''.replace('\n', ''),

      '无阻挡':             '''
                            resistant_score == 0
                            '''.replace('\n', ''),

      '正向模式':            '''
                            up_pattern_score > 0
                            '''.replace('\n', ''),

      '高位下降':            '''
                            位置 in ['h'] and 相对ichimoku位置 in ['up'] and 相对kama位置 in ['up'] and (candle_color == -1 or 长影线_trend == "d")
                            '''.replace('\n', ''),
    } 
    signal_condition_weights = {'趋势向上': 1, '低位': 1, '上行': 1, '突破': 1, '无波动': 1, '无阻挡': 0.5, '正向模式':0.5, '高位下降': -1}

    for c in signal_conditions.keys():
      
      tmp_condition = signal_conditions[c]
      tmp_idx = df.query(tmp_condition).index

      df.loc[tmp_idx, 'signal_score'] += signal_condition_weights[c]
      df.loc[tmp_idx, 'signal_description'] += c + ', '

    df['total_score'] = (df['trend_score'] + df['trigger_score'] + df['pattern_score']).round(2)
    df['signal_score'] = (df['signal_score'] + df['total_score']).round(2)
    df['signal_score_change'] = (df['signal_score'] - df['signal_score'].shift(1)).round(2)

    threshold = 4
    signal_change_idx = df.query(f'signal_score_change >= {threshold} and adx_day == 1 or adx_direction_day == 1').index
    df.loc[signal_change_idx, 'signal_score'] = (df.loc[signal_change_idx, 'signal_score'] + df.loc[signal_change_idx, 'signal_score_change']/threshold).round(2)
    df.loc[signal_change_idx, 'signal_description'] += df.loc[signal_change_idx, 'signal_score_change'].apply(lambda x: f'分数剧增({x}), ')

    df['desc_score'] = df['signal_score'].copy()
    df['signal_description'] = df['signal_description'].apply(lambda x: x[:-2] if len(x) > 0 else '')


  # drop redundant columns
  for col in col_to_drop:
    if col in df.columns:
      df.drop(col, axis=1, inplace=True)

  return df

# visualize features and signals
def visualization(df: pd.DataFrame, start: str = None, end: str = None, interval: str = 'day', title: str = None, save_path: str = None, visualization_args: dict = {}, trade_info: dict = None):
  """
  Visualize features and signals.

  :param df: dataframe with ta indicators
  :param start: start date to draw
  :param end: end date to draw
  :param title: title of the plot
  :param save_path: to where the plot will be saved
  :param visualization_args: arguments for plotting
  :returns: None
  :raises: Exception
  """
  if df is None or len(df) == 0:
    print(f'No data for visualization')
    return None

  try:
    # visualize 
    phase = 'visualization'
    is_show = visualization_args.get('show_image')
    is_save = visualization_args.get('save_image')
    plot_args = visualization_args.get('plot_args')
    
    plot_multiple_indicators(
      df=df, title=title, args=plot_args, start=start, end=end, interval=interval,
      show_image=is_show, save_image=is_save, save_path=save_path, trade_info=trade_info)
  except Exception as e:
    print(phase, e)

# postprocess
def postprocess(df: pd.DataFrame, keep_columns: dict, drop_columns: list, sec_names: dict, target_interval: str = ''):
  """
  Postprocess

  :param df: dataframe with ta features and ta derived features
  :param keep_columns: columns to keep for the final result
  :param drop_columns: columns to drop for the final result
  :param sec_names: security names mapping
  :returns: postprocessed dataframe
  :raises: None
  """

  if df is None or len(df) == 0:
    print(f'No data for postprocess')
    return pd.DataFrame()

  # reset index(as the index(date) of rows are all the same)
  df = df.reset_index().copy()

  # sort symbols
  df = df.sort_values(['adx_direction_day', 'adx_direction'], ascending=[True, False])

  # add names for symbols
  df['name'] = df['symbol']
  df['name'] = df['name'].apply(lambda x: sec_names[x] if x in sec_names.keys() else x)

  # rename columns, keep 3 digits
  df = df[list(keep_columns.keys())].rename(columns=keep_columns).round(3)
  
  # add target-interval info
  df['ti'] = target_interval

  # drop columns
  df = df.drop(drop_columns, axis=1)
  
  return df


# ================================================ Condition calculation ============================================ #
# drop na values for dataframe
def dropna(df: pd.DataFrame) -> pd.DataFrame:
  """
  Drop rows with "Nans" values

  :param df: original dataframe
  :returns: dataframe with Nans dropped
  :raises: none
  """
  df = df[df < math.exp(709)]  # big number
  df = df[df != 0.0]
  df = df.dropna()
  return df

# fill na values for dataframe
def fillna(series: pd.Series, fill_value: float = 0) -> pd.Series:
  """
  Fill na value for a series with specific value

  :param series: series to fillna
  :param fill_value: the value to replace na values
  :returns: series with na values filled
  :raises: none
  """
  series.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
  return series

# get max/min in 2 values
def get_min_max(x1: float, x2: float, f: Literal['min','max'] = 'min') -> float:
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
      return max(x1, x2)
    elif f == 'min':
      return min(x1, x2)
    else:
      raise ValueError('"f" variable value should be "min" or "max"')
  else:
    return np.nan    
    
# filter index that meet conditions
def filter_idx(df: pd.DataFrame, condition_dict: dict) -> dict:
  """
  Filter index that meet conditions. Usually used in func[assign_condition_value]

  :param df: dataframe to search
  :param condition_dict: dictionary of conditions
  :returns: dictionary of index that meet corresponding conditions
  :raises: None
  """
  # target index
  result = {}
  for condition in condition_dict.keys():
    result[condition] = df.query(condition_dict[condition]).index

  return result

# set value to column of indices that meet specific conditions
def assign_condition_value(df: pd.DataFrame, column: str, condition_dict: dict, value_dict: dict, default_value: Any = None) -> pd.DataFrame:
  """
  Set value to column of index that meet conditions

  :param df: dataframe to search
  :param column: target column
  :param condition_dict: dictionary of conditions
  :param value_dict: corresponding value to assign to the column
  :param default_value: default value of the column
  :returns: dataframe with the column assigned with corresponding values
  :raises: None
  """
  # copy dataframe
  df = df.copy()
  
  # set default value of column
  if default_value is not None:
    df[column] = default_value
  
  # set condition value to filterd index
  filtered_idx = filter_idx(df=df, condition_dict=condition_dict)
  for k in value_dict.keys():
    condition_idx = filtered_idx.get(k)
    condition_value = value_dict[k]

    if condition_idx is None:
      print(f'{k} not found in filter')
    else:
      df.loc[condition_idx, column] = condition_value
  
  return df

# calculate score according to conditions
def assign_condition_score(df: pd.DataFrame, condition_dict: dict, up_score_col: str, down_score_col: str) -> pd.DataFrame:
  """
  Calculate up(+) and down(-) score according to corresponding conditions
  
  :param df: original dataframe
  :param condition_dict: dictionary of conditions and corresponding scores
  :param up_score_col: column name of up score (positive score)
  :param down_score_col: column name of down score (negative score)
  :returns: dataframe with up_score_col and down_score_col columns
  :raises: none
  """
  # initialization
  if up_score_col not in df.columns:
    df[up_score_col] = 0
  if down_score_col not in df.columns:
    df[down_score_col] = 0
  
  df[f'{up_score_col}_description'] = ''
  df[f'{down_score_col}_description'] = ''

  # extract score, label and condition from dict
  labels = {}
  scores = {}
  conditions = {}
  for k in condition_dict.keys():
    scores[k] = condition_dict[k][0]
    labels[k] = condition_dict[k][1]
    conditions[k] = condition_dict[k][2]
  
  # calculate score and score_description
  for c in conditions.keys():
    tmp_idx = df.query(conditions[c]).index
    
    # scores
    if c[0] == '+':
      df.loc[tmp_idx, up_score_col] += scores[c]
      df.loc[tmp_idx, f'{up_score_col}_description'] += f'{c}, '
    elif c[0] == '-':
      df.loc[tmp_idx, down_score_col] +=scores[c]
      df.loc[tmp_idx, f'{down_score_col}_description'] += f'{c}, '
    else:
      print(f'{c} not recognized')

  # postprocess
  df[up_score_col] = df[up_score_col].round(2)
  df[f'{up_score_col}_description'] = df[f'{up_score_col}_description'].apply(lambda x: x[:-2])  
  if down_score_col != up_score_col:
    df[down_score_col] = df[down_score_col].round(2)
    df[f'{down_score_col}_description'] = df[f'{down_score_col}_description'].apply(lambda x: x[:-2])

  return df


# ================================================ Rolling windows ================================================== #
# simple moving window
def sm(series: pd.Series, periods: int, fillna: bool = False) -> pd.core.window.rolling.Rolling:
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

# exponential moving window
def em(series: pd.Series, periods: int, fillna: bool = False) -> pd.core.window.ewm.ExponentialMovingWindow:
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

# weighted moving average
def wma(series: pd.Series, periods: int, fillna: bool = False) -> pd.Series:
  """
  Weighted monving average from simple moving window

  :param series: series to calculate
  :param periods: size of the moving window
  :param fillna: make the min_periods = 0
  :returns: a rolling weighted average of 'series' with window size 'periods'
  :raises: none
  """
  weight = pd.Series([i * 2 / (periods * (periods + 1)) for i in range(1, periods + 1)])
  weighted_average = sm(series=series, periods=periods, fillna=fillna).apply(lambda x: (weight * x).sum(), raw=True)
  return weighted_average

# same direction accumulation
def sda(series: pd.Series, zero_as: Optional[float] = None, one_restart: bool = False) -> pd.Series:
  """
  Accumulate values with same symbol (+/-), once the symbol changed, start over again

  :param series: series to calculate
  :param accumulate_by: if None, accumulate by its own value, other wise, add by specified value
  :param zero_val: action when encounter 0: if None pass, else add(minus) spedicied value according to previous symbol 
  :returns: series with accumulated value
  :raises: None
  """
  # copy series
  lst = series.values.tolist()
  idx = series.index

  # initialize
  result = [lst[0]]  
  for i in range(1, len(lst)):

    # get current and previous number and their sign
    current_num = lst[i]
    current_num_sign = np.sign(current_num)
    previous_num = lst[i - 1]
    previous_num_sign = np.sign(previous_num)
    cumsum_sign = np.sign(result[i-1])
    
    # 当前值为0的情况
    if current_num_sign == 0:
      if zero_as is None or (result[i-1]) == 0:
        result.append(current_num)
      else:
        result.append(result[i-1] + zero_as*(1 if cumsum_sign == 1 else -1))

    # 当前值非0的情况
    else:

      # 特殊情况: 如果当前值为1且one_restart
      if (one_restart and current_num in [1, -1]):
        result.append(current_num)

      # 一般情况
      else:
        
        # 如果当前值与累计值的符号不同
        if (current_num_sign != cumsum_sign):
          result.append(current_num) 

        # 如果当前值与累计值的符号相同       
        else:
          result.append(result[i-1] + current_num)
        

  result_series = pd.Series(result, index=idx)
  return result_series

# moving slope
def moving_slope(series: pd.Series, periods: int) -> tuple:
  """
  Moving Slope

  :param series: series to calculate
  :param periods: size of the moving window
  :returns: a tuple of series: (slope, intercepts)
  :raises: none
  """  
  # convert series to numpy array
  np_series = series.to_numpy()
  stride = np_series.strides
  slopes, intercepts = np.polyfit(np.arange(periods), as_strided(np_series, (len(series)-periods+1, periods), stride+stride).T, deg=1)

  # padding NaNs on the top
  padding_nan = np.full((periods-1, ), np.nan)
  padded_slopes = np.concatenate([padding_nan, slopes])
  padded_intercepts = np.concatenate([padding_nan, intercepts])
  
  return (padded_slopes, padded_intercepts)

# normilization
def normalize(series: pd.Series, fillna: Optional[str] = None, method: str = 'default') -> pd.Series:
  """
  Normalize a series.

  :param series: The input series to be normalized.
  :param fillna: Optional value to fill NaN values in the series. If not provided, NaN values will not be filled.
  :returns: The normalized series.
  :raises: None
  """  
  normalaized = series
  
  # fill NA values if specified
  if fillna is not None:  
    normalaized = series.fillna(method=fillna)

  # normalization
  normalaized = (normalaized - normalaized.min()) / (normalaized.max() - normalaized.min())

  # Scale the normalized seriesto the range [-1, 1]
  if method == 'minmax':
    normalized = normalized * 2 - 1
  # Scale the normalized seriesto the range [0, 1]
  else:
    pass

  return normalaized


# ================================================ Change calculation =============================================== #
# calculate change of a column in certain period
def cal_change(df: pd.DataFrame, target_col: str, periods: int = 1, add_accumulation: bool = True, add_prefix: bool = False, drop_na: bool = False) -> pd.DataFrame:
  """
  Calculate change of a column with a sliding window
  
  :param df: original dataframe
  :param target_col: change of which column to calculate
  :param periods: calculate the change within the period
  :param add_accumulation: wether to add accumulative change in a same direction
  :param add_prefix: whether to add prefix for the result columns (when there are multiple target columns to calculate)
  :param drop_na: whether to drop na values from dataframe:
  :returns: dataframe with change columns
  :raises: none
  """
  # copy dateframe
  df = df.copy()

  # set prefix for result columns
  prefix = ''
  if add_prefix:
    prefix = f'{target_col}_'

  # set result column names
  change_col = f'{prefix}change'
  acc_change_col = f'{prefix}acc_change'
  acc_change_count_col = f'{prefix}acc_change_count'

  # calculate change within the period
  df[change_col] = df[target_col].diff(periods=periods)
  
  # calculate accumulative change in a same direction
  if add_accumulation:

    df[acc_change_col] = sda(series=df[change_col], zero_as=0)

    df[acc_change_count_col] = 0
    df.loc[df[change_col]>0, acc_change_count_col] = 1
    df.loc[df[change_col]<0, acc_change_count_col] = -1
    df[acc_change_count_col] = sda(series=df[acc_change_count_col], zero_as=1)
    
  # drop NA values
  if drop_na:        
    df.dropna(inplace=True)

  return df 

# calculate change rate of a column in certain period
def cal_change_rate(df: pd.DataFrame, target_col: str, periods: int = 1, add_accumulation: bool = True, add_prefix: bool = False, drop_na: bool = False) -> pd.DataFrame:
  """
  Calculate change rate of a column with a sliding window
  
  :param df: original dataframe
  :param target_col: change rate of which column to calculate
  :param periods: calculate the change rate within the period
  :param add_accumulation: wether to add accumulative change rate in a same direction
  :param add_prefix: whether to add prefix for the result columns (when there are multiple target columns to calculate)
  :param drop_na: whether to drop na values from dataframe:
  :returns: dataframe with change rate columns
  :raises: none
  """
  # copy dataframe
  df = df.copy()
  
  # set prefix for result columns
  prefix = ''
  if add_prefix:
    prefix = f'{target_col}_'

  # set result column names
  rate_col = f'{prefix}rate'
  acc_rate_col = f'{prefix}acc_rate'
  acc_day_col = f'{prefix}acc_day'

  # calculate change rate within the period
  df[rate_col] = df[target_col].pct_change(periods=periods)
  
  # calculate accumulative change rate in a same direction
  if add_accumulation:
    df[acc_rate_col] = 0
    df.loc[df[rate_col]>=0, acc_day_col] = 1
    df.loc[df[rate_col]<0, acc_day_col] = -1

    # add continuous values which has the same symbol (+/-)
    df[acc_rate_col] = sda(series=df[rate_col], zero_as=0)
    df[acc_day_col] = sda(series=df[acc_day_col], zero_as=1)

    # fill NA in acc_day_col with 0
    df[acc_rate_col] = df[acc_rate_col].fillna(0.0)
    df[acc_day_col] = df[acc_day_col].fillna(0).astype(int) 
  if drop_na:        
    df.dropna(inplace=True) 

  return df


# ================================================ Signal processing ================================================ #
# calculate signal that generated from 2 lines crossover
def cal_crossover_signal(df: pd.DataFrame, fast_line: str, slow_line: str, result_col: str = 'signal', pos_signal: str = 'b', neg_signal: str = 's', none_signal: str = 'n') -> pd.DataFrame:
  """
  Calculate signal generated from the crossover of 2 lines
  When fast line breakthrough slow line from the bottom, positive signal will be generated
  When fast line breakthrough slow line from the top, negative signal will be generated 

  :param df: original dffame which contains a fast line and a slow line
  :param fast_line: columnname of the fast line
  :param slow_line: columnname of the slow line
  :param result_col: columnname of the result
  :param pos_signal: the value of positive signal
  :param neg_signal: the value of negative signal
  :param none_signal: the value of none signal
  :returns: series of the result column
  :raises: none
  """
  df = df.copy()

  # calculate the distance between fast and slow line
  df['diff'] = df[fast_line] - df[slow_line]
  df['diff_prev'] = df['diff'].shift(1)

  # get signals from fast/slow lines cross over
  conditions = {
    'up': '(diff >= 0 and diff_prev < 0) or (diff > 0 and diff_prev <= 0)', 
    'down': '(diff <= 0 and diff_prev > 0) or (diff < 0 and diff_prev >= 0)'} 
  values = {
    'up': pos_signal, 
    'down': neg_signal}
  df = assign_condition_value(df=df, column=result_col, condition_dict=conditions, value_dict=values, default_value=none_signal)
  
  return df[[result_col]]

# calculate signal that generated from trigering boundaries
def cal_boundary_signal(df: pd.DataFrame, upper_col: str, lower_col: str, upper_boundary: float, lower_boundary: float, result_col: str = 'signal', pos_signal: str = 'b', neg_signal: str = 's', none_signal: str = 'n') -> pd.DataFrame:
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
  conditions = {
    'up': f'{upper_col} > {upper_boundary}', 
    'down': f'{lower_col} < {lower_boundary}'} 
  values = {
    'up': pos_signal, 
    'down': neg_signal}
  df = assign_condition_value(df=df, column=result_col, condition_dict=conditions, value_dict=values, default_value=none_signal)

  return df[[result_col]]


# ================================================ Self-defined TA ================================================== #
# linear regression
def linear_fit(df: pd.DataFrame, target_col: str, periods: int) -> dict:
  """
  Calculate slope for selected piece of data

  :param df: dataframe
  :param target_col: target column name
  :param periods: input data length 
  :returns: slope of selected data from linear regression
  :raises: none
  """

  # not enough data
  if len(df) <= periods:
    return {'slope': 0, 'intecept': 0}
  
  # calculate linear regression prediction
  else:
    x = range(1, periods+1)
    y = df[target_col].fillna(0).tail(periods).values.tolist()
    lr = linregress(x, y)

    return {'slope': lr[0], 'intecept': lr[1]}

# calculate moving average 
def cal_moving_average(df: pd.DataFrame, target_col: str, ma_windows: list = [50, 105], start: Optional[str] = None, end: Optional[str] = None, window_type: str = 'em') -> pd.DataFrame:
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
    ma_col = f'{target_col}_ma_{mw}'
    df[ma_col] = mw_func(series=df[target_col], periods=mw).mean()
  
  return df

# add position features
def cal_position_score(df: pd.DataFrame) -> pd.DataFrame:
  """
  Calculate inter-candlestick position score

  :param df: original OHLCV dataframe
  :returns: dataframe with new columns [position, position_score, ki_distance]
  :raises: none
  """
  df = df.copy()
  col_to_drop = [] 

  df['position_score'] = 0

  # position score
  rp_cols = {"kama":"相对kama位置", "ichimoku":"相对ichimoku位置"}
  for col in ['ichimoku', 'kama']:
    
    col_p = rp_cols[col]
    col_d = f'{col}_distance'
    col_v = f'{col}_position_score'
    col_to_drop.append(col_v)

    if col_p not in df.columns or col_d not in df.columns:
      df[col_v] = 0
      continue
    else:
      position_conditions = {

        'down_from_low':        f'{col_p} in ["down"] and {col_d} < 0',
        'down_from_high':       f'{col_p} in ["down"] and {col_d} > 0',
        'mid_down_from_low':    f'{col_p} in ["mid_down"] and {col_d} < 0',
        'mid_down_from_high':   f'{col_p} in ["mid_down"] and {col_d} > 0',
        'mid_from_low':         f'{col_p} in ["mid"] and {col_d} < 0',
        'mid_from_high':        f'{col_p} in ["mid"] and {col_d} > 0',
        'mid_up_from_low':      f'{col_p} in ["mid_up"] and {col_d} < 0',
        'mid_up_from_high':     f'{col_p} in ["mid_up"] and {col_d} > 0',
        'up_from_low':          f'{col_p} in ["up"] and {col_d} < 0',
        'up_from_high':         f'{col_p} in ["up"] and {col_d} > 0',
        'out_from_low_red':     f'{col_p} in ["out"] and {col_d} < 0 and candle_color == -1',
        'out_from_high_red':    f'{col_p} in ["out"] and {col_d} > 0 and candle_color == -1',
        'out_from_low_green':   f'{col_p} in ["out"] and {col_d} < 0 and candle_color == 1',
        'out_from_high_green':  f'{col_p} in ["out"] and {col_d} > 0 and candle_color == 1',

      } 
      position_values = {
        
        'up_from_high':         1,
        'up_from_low':          1,

        'down_from_low':        -1,
        'down_from_high':       -1,

        'mid_down_from_low':    -0.5,
        'mid_down_from_high':   -0.5,

        'mid_from_low':         0,
        'mid_from_high':        0,

        'mid_up_from_low':      0.5,
        'mid_up_from_high':     0.5,
        
        'out_from_low_red':     -0.5,
        'out_from_high_red':    -0.5,

        'out_from_low_green':   0.5,
        'out_from_high_green':  0.5,
      }
      df = assign_condition_value(df=df, column=col_v, condition_dict=position_conditions, value_dict=position_values, default_value=0) 
    
    df['position_score'] += df[col_v]

  # ichimoku-kama position
  threshold = 1
  position_conditions = {   
    'up':           f'position_score > {threshold}',
    'mid_up':       f'0 < position_score <= {threshold}', 
    'mid':          f'position_score == 0',
    'mid_down':     f'0 > position_score >= {-threshold}', 
    'down':         f'position_score < {-threshold}',
  } 
  position_values = {
    'up':           'up',
    'mid_up':       'mid_up',
    'mid':          'mid',
    'mid_down':     'mid_down',
    'down':         'down', 
  }
  df = assign_condition_value(df=df, column='position', condition_dict=position_conditions, value_dict=position_values, default_value='')

  # ================================ calculate overall distance =============
  term_trend_conditions = {
    'rr':    f'kama_distance <  0 and ichimoku_distance <  0', 
    'rn':    f'kama_distance <  0 and ichimoku_distance == 0', 
    'rg':    f'kama_distance <  0 and ichimoku_distance >  0', 
    'nr':    f'kama_distance == 0 and ichimoku_distance <  0', 
    'nn':    f'kama_distance == 0 and ichimoku_distance == 0',
    'ng':    f'kama_distance == 0 and ichimoku_distance >  0', 
    'gr':    f'kama_distance >  0 and ichimoku_distance <  0', 
    'gn':    f'kama_distance >  0 and ichimoku_distance == 0',
    'gg':    f'kama_distance >  0 and ichimoku_distance >  0', 
  } 
  term_trend_values = {
    'rr':    f'rr', 
    'rn':    f'rn', 
    'rg':    f'rg', 
    'nr':    f'nr', 
    'nn':    f'nn',
    'ng':    f'ng',
    'gr':    f'gr', 
    'gn':    f'gn',
    'gg':    f'gg',
  }
  df = assign_condition_value(df=df, column='ki_distance', condition_dict=term_trend_conditions, value_dict=term_trend_values, default_value='n')

  # drop unnecessary columns
  for col in col_to_drop:
    if col in df.columns:
      df.drop(col, axis=1, inplace=True)

  return df

# add candle stick features 
def add_candlestick_features(df: pd.DataFrame, ohlcv_col: dict = default_ohlcv_col) -> pd.DataFrame:
  """
  Add candlestick features for dataframe

  :param df: original OHLCV dataframe
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :returns: dataframe with candlestick columns
  :raises: none
  """
  # copy dataframe
  df = df.copy()
  col_to_drop = [] 

  # set column names
  open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']
  
  # candle color
  df['candle_color'] = 0
  up_idx = df[open] < df[close]
  down_idx = df[open] >= df[close]
  df.loc[up_idx, 'candle_color'] = 1
  df.loc[down_idx, 'candle_color'] = -1
  
  # shadow
  df['candle_shadow'] = (df[high] - df[low])
  
  # entity
  df['candle_entity'] = abs(df[close] - df[open])
  
  # ======================================= shadow/entity ============================================ #
  df['candle_entity_top'] = 0
  df['candle_entity_bottom'] = 0
  df.loc[up_idx, 'candle_entity_top'] = df.loc[up_idx, close]
  df.loc[up_idx, 'candle_entity_bottom'] = df.loc[up_idx, open]
  df.loc[down_idx, 'candle_entity_top'] = df.loc[down_idx, open]  
  df.loc[down_idx, 'candle_entity_bottom'] = df.loc[down_idx, close]

  df['candle_upper_shadow'] = df[high] - df['candle_entity_top']
  df['candle_lower_shadow'] = df['candle_entity_bottom'] - df[low]

  df['candle_upper_shadow_pct'] = df['candle_upper_shadow'] / df['candle_shadow']
  df['candle_lower_shadow_pct'] = df['candle_lower_shadow'] / df['candle_shadow']
  df['candle_entity_pct'] = df['candle_entity'] / df['candle_shadow']

  # calculate relative candle
  df['pre_candle_top'] = df['candle_entity_top'].shift(1)
  df['pre_candle_bottom'] = df['candle_entity_bottom'].shift(1) 
  df['pre_candle_color'] = df['candle_color'].shift(1)
  
  # window position status (beyond/below/among window)
  conditions = {
    '上方': '(candle_entity_bottom >= pre_candle_top)',
    '中上': '((candle_entity_top > pre_candle_top) and (pre_candle_top > candle_entity_bottom >= pre_candle_bottom))',
    '中间': '((candle_entity_top <= pre_candle_top) and (candle_entity_bottom >= pre_candle_bottom))',
    '穿刺': '((candle_entity_top > pre_candle_top) and (candle_entity_bottom < pre_candle_bottom))',
    '中下': '((candle_entity_bottom < pre_candle_bottom) and (pre_candle_top >= candle_entity_top > pre_candle_bottom))',
    '下方': '(candle_entity_top <= pre_candle_bottom)'}
  values = {
    '上方': 'up', '中上': 'mid_up',
    '中间': 'mid', '穿刺': 'out',
    '中下': 'mid_down', '下方': 'down'}
  df = assign_condition_value(df=df, column='相对candle位置', condition_dict=conditions, value_dict=values, default_value='')

  # candle_position_score
  candle_conditions = {
    'green_down':           f'candle_color == 1 and 相对candle位置 == "down"', 
    'green_up':             f'candle_color == 1 and 相对candle位置 == "up"', 
    'green_out':            f'candle_color == 1 and 相对candle位置 == "out"', 

    'green_green_mid_down': f'pre_candle_color == 1 and candle_color == 1 and 相对candle位置 in ["mid_down"]',
    'green_green_mid':      f'pre_candle_color == 1 and candle_color == 1 and 相对candle位置 in ["mid"]',
    'green_green_mid_up':   f'pre_candle_color == 1 and candle_color == 1 and 相对candle位置 in ["mid_up"]', 
    'red_green_mid_down':   f'pre_candle_color ==-1 and candle_color == 1 and 相对candle位置 in ["mid_down"]',
    'red_green_mid':        f'pre_candle_color ==-1 and candle_color == 1 and 相对candle位置 in ["mid"]',
    'red_green_mid_up':     f'pre_candle_color ==-1 and candle_color == 1 and 相对candle位置 in ["mid_up"]', 
    
    'red_down':             f'candle_color == -1 and 相对candle位置 == "down"', 
    'red_up':               f'candle_color == -1 and 相对candle位置 == "up"', 
    'red_out':              f'candle_color == -1 and 相对candle位置 == "out"', 

    'green_red_mid_down':   f'pre_candle_color == 1 and candle_color ==-1 and 相对candle位置 in ["mid_down"]',
    'green_red_mid':        f'pre_candle_color == 1 and candle_color ==-1 and 相对candle位置 in ["mid"]',
    'green_red_mid_up':     f'pre_candle_color == 1 and candle_color ==-1 and 相对candle位置 in ["mid_up"]', 
    'red_red_mid_down':     f'pre_candle_color ==-1 and candle_color ==-1 and 相对candle位置 in ["mid_down"]',
    'red_red_mid':          f'pre_candle_color ==-1 and candle_color ==-1 and 相对candle位置 in ["mid"]',
    'red_red_mid_up':       f'pre_candle_color ==-1 and candle_color ==-1 and 相对candle位置 in ["mid_up"]', 
  } 
  candle_values = {
    'green_up':             0.99, 
    'green_down':           -0.33, 
    'green_out':            0.66, 

    'green_green_mid_down': -0.33,
    'green_green_mid':      -0.33,
    'green_green_mid_up':   0.33, 
    'red_green_mid_down':   0.33,
    'red_green_mid':        0.33,
    'red_green_mid_up':     0.66, 

    'red_down':             -0.99, 
    'red_up':               0.33, 
    'red_out':              -0.66, 

    'red_red_mid_up':       0.33, 
    'red_red_mid':          0.33,
    'red_red_mid_down':     -0.33,
    'green_red_mid_up':     -0.33, 
    'green_red_mid':        -0.33,
    'green_red_mid_down':   -0.66,    
  }
  df = assign_condition_value(df=df, column='candle_position_score', condition_dict=candle_conditions, value_dict=candle_values, default_value=0)

  col_to_drop += ['candle_upper_shadow', 'candle_lower_shadow', 'pre_candle_top', 'pre_candle_bottom', 'pre_candle_color']

  # ======================================= gap ====================================================== #
  # gap_up / gap_down  
  for col in [open, close, high, low, 'candle_color', 'candle_entity']:
    prev_col = f'prev_{col}' 
    df[prev_col] = df[col].shift(1)
    col_to_drop.append(prev_col)
  
  # initialization
  df['candle_gap'] = 0
  df['candle_gap_color'] = np.NaN
  df['candle_gap_top'] = np.NaN
  df['candle_gap_bottom'] = np.NaN
  
  # gap up
  df['low_prev_high'] = df[low] - df[f'prev_{high}']
  gap_up_idx = df.query(f'low_prev_high > 0').index
  df.loc[gap_up_idx, 'candle_gap'] = 1
  gap_up_mean = df.loc[gap_up_idx, 'low_prev_high'].mean() # df['low_prev_high'].nlargest(10).values[-1] # 
  gap_up_mean = 0 if np.isnan(gap_up_mean) else gap_up_mean
  strict_gap_up_idx = df.query(f'low_prev_high >= {gap_up_mean} and low_prev_high > 0').index
  if len(strict_gap_up_idx) / len(df) < 0.05:
    df.loc[strict_gap_up_idx, 'candle_gap'] = 2
    df.loc[strict_gap_up_idx, 'candle_gap_color'] = 1
    df.loc[strict_gap_up_idx, 'candle_gap_top'] = df.loc[strict_gap_up_idx, f'{low}']
    df.loc[strict_gap_up_idx, 'candle_gap_bottom'] = df.loc[strict_gap_up_idx, f'prev_{high}']
  
  # gap down
  df['prev_low_high'] = df[f'prev_{low}'] - df[high]
  gap_down_idx = df.query(f'prev_low_high > 0').index  
  df.loc[gap_down_idx, 'candle_gap'] = -1
  gap_down_mean = df.loc[gap_down_idx, 'prev_low_high'].mean() # df['prev_low_high'].nlargest(10).values[-1] # 
  gap_down_mean = 0 if np.isnan(gap_down_mean) else gap_down_mean
  strict_gap_down_idx = df.query(f'prev_low_high >= {gap_down_mean} and prev_low_high > 0').index  
  if len(strict_gap_down_idx) / len(df) < 0.05:
    df.loc[strict_gap_down_idx, 'candle_gap'] = -2
    df.loc[strict_gap_down_idx, 'candle_gap_color'] = -1
    df.loc[strict_gap_down_idx, 'candle_gap_top'] = df.loc[strict_gap_down_idx, f'prev_{low}']
    df.loc[strict_gap_down_idx, 'candle_gap_bottom'] = df.loc[strict_gap_down_idx, f'{high}']
  
  # gap height, color, top and bottom
  df['candle_gap_top'] = df['candle_gap_top'].fillna(method='ffill')
  df['candle_gap_bottom'] = df['candle_gap_bottom'].fillna(method='ffill') 
  df['candle_gap_color'] = df['candle_gap_color'].fillna(method='ffill').fillna(0)
  df['candle_gap_distance'] = ((df['candle_gap_top'] - df['candle_gap_bottom']).fillna(method='ffill')) * df['candle_gap_color']

  # # window position status (beyond/below/among window)
  # conditions = {
  #   '上方': '(candle_entity_bottom >= candle_gap_top)',
  #   '中上': '((candle_entity_top > candle_gap_top) and (candle_gap_top > candle_entity_bottom >= candle_gap_bottom))',
  #   '中间': '((candle_entity_top <= candle_gap_top) and (candle_entity_bottom >= candle_gap_bottom))',
  #   '穿刺': '((candle_entity_top > candle_gap_top) and (candle_entity_bottom < candle_gap_bottom))',
  #   '中下': '((candle_entity_bottom < candle_gap_bottom) and (candle_gap_top >= candle_entity_top > candle_gap_bottom))',
  #   '下方': '(candle_entity_top <= candle_gap_bottom)'}
  # values = {
  #   '上方': 'up', '中上': 'mid_up',
  #   '中间': 'mid', '穿刺': 'out',
  #   '中下': 'mid_down', '下方': 'down'}
  # df = assign_condition_value(df=df, column='相对gap位置', condition_dict=conditions, value_dict=values, default_value='')
  
  # drop intermidiate columns
  for c in ['low_prev_high', 'prev_low_high']:
    col_to_drop.append(c)
  df = df.drop(col_to_drop, axis=1)

  return df

# add candle stick patterns
def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
  """
  Add candlestick patterns for dataframe

  :param df: dataframe with candlestick features
  :returns: dataframe with candlestick patterns
  :raises: none
  """
  # columns to drop
  col_to_drop = []

  # global position
  if 'position' > '':
    
    # add position features
    df = cal_position_score(df)
    df['moving_max'] = sm(series=df['High'], periods=10).max()
    df['moving_min'] = sm(series=df['Low'], periods=10).min()
    
    conditions = {
      # 位置_trend == "u", 近10日最高价
      '顶部': '(position == "up" and moving_max == High)', 
      # 位置_trend == "d", 近10日最低价
      '底部': '(position == "down" and moving_min == Low)'}
    values = {'顶部': 'u', '底部': 'd'}
    df = assign_condition_value(df=df, column='极限_trend', condition_dict=conditions, value_dict=values, default_value='n')
    col_to_drop += ['moving_max', 'moving_min', '极限_trend']

    # add gap features
    conditions = {
      # 位置_trend == "u", 近10日最高价
      'gap_up':   '(candle_gap > 1)', 
      # 位置_trend == "d", 近10日最低价
      'gap_down': '(candle_gap < -1)'}
    values = {'gap_up': 'u', 'gap_down': 'd'}
    df = assign_condition_value(df=df, column='窗口_trend', condition_dict=conditions, value_dict=values, default_value='n')

  # shadow and entity
  if 'shadow_entity' > '':

    # X_diff: (X-mean(X, 30))/std(X, 30)
    ma_period = 30
    upper_factor = 1
    lower_factor = -0.5
    for col in ['entity', 'shadow']:
      df[f'{col}_ma'] = sm(series=df[f'candle_{col}'], periods=ma_period).mean()
      # df[f'{col}_std'] = sm(series=df[f'candle_{col}'], periods=ma_period).std()
      df[f'{col}_diff'] = (df[f'candle_{col}'] - df[f'{col}_ma'])/df[f'{col}_ma'] # df[f'{col}_std']
      col_to_drop += [f'{col}_ma' , f'{col}_diff'] # , f'{col}_std'

    # long/short shadow
    conditions = {
      '价格波动范围大': f'shadow_diff >= {upper_factor}', 
      '价格波动范围小': f'shadow_diff <= {lower_factor}'}
    values = {'价格波动范围大': 'u', '价格波动范围小': 'd'}
    df = assign_condition_value(df=df, column='shadow_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # long/short entity
    conditions = {
      '长实体': f'(entity_diff >= {1.5*upper_factor}) or (entity_diff >= {upper_factor} and (shadow_trend == "u" and candle_entity_pct >= 0.8))', 
      '短实体': f'(entity_diff <= {lower_factor})'} 
    values = {'长实体': 'u', '短实体': 'd'}
    df = assign_condition_value(df=df, column='entity_trend', condition_dict=conditions, value_dict=values, default_value='n')

  # patterns that consist only 1 candlestick
  if '1_candle' > '':
    
    # cross/highwave
    conditions = {
      # 十字星1: 实体占比<15%, 影线σ<-1.5
      '十字星_1': '(candle_entity_pct < 0.15) and (shadow_diff < -0.5 or (shadow_diff < 0 and entity_diff < -0.5))',
      # 高浪线: 实体占比<15%, 影线σ>-1.5
      '高浪线': '(candle_entity_pct < 0.15) and (shadow_diff > 1.5)',
      # 十字星2: 实体占比<5%
      '十字星_2': '(candle_entity_pct < 0.05 or (shadow_diff < -1 and entity_diff < -1))',
      }
    values = {'十字星_1': 'd', '高浪线': 'u', '十字星_2': 'd', } 
    df = assign_condition_value(df=df, column='十字星_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # hammer/meteor
    conditions = {
      # 影线σ > 1, 长下影线, 非长实体, 上影线占比 < 5%, 实体占比 <= 30%, 下影线占比 >= 60%
      '锤子': '(shadow_diff > 1) and (candle_upper_shadow_pct < 0.1 and entity_trend != "u") and (candle_upper_shadow_pct < 0.05 and 0.05 <= candle_entity_pct <= 0.3 and candle_lower_shadow_pct >= 0.6)',
      # 影线σ > 1, 长上影线, 非长实体, 上影线占比 >= 60%, 实体占比 <= 30%, 下影线占比 < 5%
      '流星': '(shadow_diff > 1) and (candle_lower_shadow_pct < 0.1 and entity_trend != "u") and (candle_lower_shadow_pct < 0.05 and 0.05 <= candle_entity_pct <= 0.3 and candle_upper_shadow_pct >= 0.6)'}
    values = {'锤子': 'u', '流星': 'd'}
    df = assign_condition_value(df=df, column='锤子', condition_dict=conditions, value_dict=values, default_value='n')
    col_to_drop.append('锤子')

    # meteor
    conditions = {
      # 高位, 近10日最高, 形态 == '流星'
      '流星线': '(position == "up" and 极限_trend == "u") and (锤子 == "d")',
      # 低位, 近10日最低, 形态 == '流星'
      '倒锤线': '(position == "down" and 极限_trend == "d") and (锤子 == "d")'}
    values = {'流星线': 'd', '倒锤线': 'u'}
    df = assign_condition_value(df=df, column='流星_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # hammer
    conditions = {
      # 高位, 近10日最高, 形态 == '锤子'
      '吊颈线': '(position == "up" and 极限_trend == "u") and (锤子 == "u")',
      # 低位, 近10日最低, 形态 == '锤子'
      '锤子线': '(position == "down" and 极限_trend == "d") and (锤子 == "u")'}
    values = {
      '吊颈线': 'd', '锤子线': 'u'}
    df = assign_condition_value(df=df, column='锤子_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # belt
    conditions = {
      # 非短实体, 低位, 价格上涨, 实体占比 > 50%, 下影线占比 <= 5%, 上影线占比 >= 15%
      '看多腰带': '(entity_trend != "d" and candle_entity_pct > 0.5) and (position == "down" and candle_lower_shadow_pct <= 0.05 and candle_upper_shadow_pct >= 0.15 and candle_color == 1)',
      # 非短实体, 高位, 价格下跌, 实体占比 > 50%, 上影线占比 <= 5%, 下影线占比 >= 15%
      '看空腰带': '(entity_trend != "d" and candle_entity_pct > 0.5) and (position == "up" and candle_upper_shadow_pct <= 0.05 and candle_lower_shadow_pct >= 0.15 and candle_color == -1)'}
    values = {'看多腰带': 'u', '看空腰带': 'd'}
    df = assign_condition_value(df=df, column='腰带_trend', condition_dict=conditions, value_dict=values, default_value='n')

  # patterns that consist multiple candlesticks
  if 'multi_candle' > '':

    # intermediate columns
    df['high_diff'] = ((df['High'] - df['High'].shift(1)) / df['Close']).abs()
    df['low_diff'] = ((df['Low'] - df['Low'].shift(1)) / df['Close']).abs()
    df['candle_entity_middle'] = (df['candle_entity_top'] + df['candle_entity_bottom']) * 0.5
    df['candle_upper_shadow_pct_diff'] = df['candle_upper_shadow_pct'] - df['candle_upper_shadow_pct'].shift(1)
    df['candle_lower_shadow_pct_diff'] = df['candle_lower_shadow_pct'] - df['candle_lower_shadow_pct'].shift(1)
    col_to_drop += ['high_diff', 'low_diff', 'candle_entity_middle', 'candle_upper_shadow_pct_diff', 'candle_lower_shadow_pct_diff']

    # previous_columns(row-1)
    for col in ['candle_color', 'candle_entity_top', 'candle_entity_bottom', 'candle_entity_middle', 'candle_entity_pct', 'High', 'Low', 'position', 'entity_trend', '相对candle位置', '极限_trend']:
      prev_col = f'prev_{col}'
      df[prev_col] = df[col].shift(1)
      col_to_drop.append(prev_col)

      # previous_previous_column(row-2)
      if col in ['candle_color', 'candle_entity_top', 'candle_entity_bottom', 'candle_entity_middle']:
        prev_prev_col = f'prev_prev_{col}'
        df[prev_prev_col] = df[col].shift(2)
        col_to_drop.append(prev_prev_col)

    # 平头顶/平头底
    conditions = {
      # 非十字星/高浪线, adx趋势向上(或高位), 高位, [1.近10日最高, 顶部差距<0.2%, 2.顶部差距<0.1%, 3.顶部差距<0.4%, 价格下跌, 上影线差距在5%内]
      '平头顶': '(十字星_trend == "n") and (position == "up") and ((high_diff <= 0.002) or (position == "up" and ((high_diff <= 0.001) or (high_diff <= 0.004 and rate <= 0 and -0.05 <= candle_upper_shadow_pct_diff <= 0.05))))',
      # 非十字星/高浪线, adx趋势向下(或低位), 低位, 近10日最低, 底部差距<0.2%
      '平头底': '(十字星_trend == "n") and (position == "down")  and (low_diff <= 0.002) and (candle_position_score > 0.33 or candle_color == 1)'}
    values = {'平头顶': 'd', '平头底': 'u'}
    df = assign_condition_value(df=df, column='平头_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # 吞噬形态
    conditions = {
      # 相对candle位置 == "out", 蜡烛非短实体, 实体占比 > 75%, 位于底部, 1-红, 2-绿
      '多头吞噬': '(相对candle位置 == "out") and (position == "down") and (prev_entity_trend != "d" and entity_trend != "d" and candle_entity_pct > 0.75) and (prev_candle_color == -1 and candle_color == 1)',
      # 相对candle位置 == "out", 蜡烛非短实体, 实体占比 > 75%, 位于顶部, 1-绿, 2-红
      '空头吞噬': '(相对candle位置 == "out") and (position == "up") and (prev_entity_trend != "d" and entity_trend != "d" and candle_entity_pct > 0.75) and (prev_candle_color == 1 and candle_color == -1)'}
    values = {'多头吞噬': 'u', '空头吞噬': 'd'}
    df = assign_condition_value(df=df, column='吞噬_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # 包孕形态
    conditions = {
      # 相对candle位置 == "mid", 前一蜡烛非短实体, 实体占比 > 80%, 当前蜡烛实体占比> 50%, 位于底部, 1-红, 2-绿
      '多头包孕': '(相对candle位置 == "mid") and (position == "down") and (prev_entity_trend != "d" and entity_trend != "d" and prev_candle_entity_pct > 0.5 and candle_entity_pct > 0.75) and (prev_candle_color == -1 and candle_color == 1)', # and (prev_High > High and prev_Low < Low) 
      # 相对candle位置 == "mid", 前一蜡烛非短实体, 实体占比 > 80%, 当前蜡烛实体占比> 50%, 位于顶部, 1-绿, 2-红
      '空头包孕': '(相对candle位置 == "mid") and (position == "up") and (prev_entity_trend != "d" and entity_trend != "d" and prev_candle_entity_pct > 0.5 and candle_entity_pct > 0.75) and (prev_candle_color == 1 and candle_color == -1)'} # (prev_High > High and prev_Low < Low) and 
    values = {'多头包孕': 'u', '空头包孕': 'd'}
    df = assign_condition_value(df=df, column='包孕_trend', condition_dict=conditions, value_dict=values, default_value='n')

    # 穿刺形态
    conditions = {
      # 相对candle位置 == "mid_down", 前一蜡烛位于底部, 1-必须为红色, 2-必须为绿色长实体, 顶部<=前顶部, 底部<前底部, 顶部穿过前中点
      '多头穿刺': '(相对candle位置 == "mid_down") and (prev_position == "down" and prev_candle_color == -1 and candle_color == 1 and entity_trend == "u" and prev_entity_trend != "d") and (prev_candle_entity_middle < candle_entity_top)',
      # 相对candle位置 == "mid_up", 前一蜡烛位于顶部, 1-必须为绿色, 2-必须为红色长实体, 顶部>前顶部, 底部>前底部, 底部穿过前中点
      '空头穿刺': '(相对candle位置 == "mid_up") and (prev_position == "up" and prev_candle_color == 1 and candle_color == -1 and entity_trend == "u" and prev_entity_trend != "d") and (prev_candle_entity_middle > candle_entity_bottom)'}
    values = {'多头穿刺': 'u', '空头穿刺': 'd'}
    df = assign_condition_value(df=df, column='穿刺_trend', condition_dict=conditions, value_dict=values, default_value='n')
    
    # 启明星/黄昏星
    conditions = {
      # 2-1 down, 3-2up, 2处于低位, 1-红色非小实体, 3-长实体 或 3-middle > 1-top, 2-非长实体, 2-顶部 < 1/3-底部
      '启明星': '(prev_相对candle位置 == "down" and 相对candle位置 == "up" and position == "down") and (ki_distance == "rr" or prev_极限_trend == "d") and (prev_position == "down") and (candle_color == 1 and entity_trend != "d") and (entity_trend == "u" or candle_entity_middle > prev_prev_candle_entity_top) and (prev_entity_trend != "u") and (prev_candle_entity_top < prev_prev_candle_entity_bottom) and (prev_candle_entity_top < candle_entity_bottom)',
      # 2-1 up, 3-2 down, 2处于高位, 1-绿色, 3-红色, 3-长实体 或 3-middle > 1-middle, 2-非长实体, 2-底部 > 1/3-顶部
      '黄昏星': '(prev_相对candle位置 == "up" and 相对candle位置 == "down" and position == "up") and (ki_distance == "gg" or prev_极限_trend == "u") and (prev_position == "up") and (prev_prev_candle_color == 1 and candle_color == -1) and (entity_trend == "u" or candle_entity_middle < prev_prev_candle_entity_middle) and (prev_entity_trend != "u")'}
    values = {'启明星': 'u', '黄昏星': 'd'}
    df = assign_condition_value(df=df, column='启明黄昏_trend', condition_dict=conditions, value_dict=values, default_value='n')

  # other customized paterns
  if 'customized' > '':

    # # 蜡烛位置
    # conditions = {
    #   # 
    #   '上行': '(position == "down") and (entity_trend != "d") and (candle_position_score > 0.66)',
    #   # 
    #   '下行': '(position == "up") and (entity_trend != "d") and (candle_position_score < -0.66)'}
    # values = {'上行': 'u', '下行': 'd'}
    # df = assign_condition_value(df=df, column='蜡烛位置_trend', condition_dict=conditions, value_dict=values, default_value='n')
    
    # 上下影线
    # long up/down shadow
    conditions = {
      # 长上影线: 
      '长上影': '((shadow_diff > 0.5) and (candle_upper_shadow_pct > 0.5)) or ((shadow_diff > 0.25) and (candle_upper_shadow_pct > 0.75))',
      # 长下影线: 
      '长下影': '((shadow_diff > 0.5) and (candle_lower_shadow_pct > 0.5)) or ((shadow_diff > 0.25) and (candle_lower_shadow_pct > 0.75))',
      }
    values = {'长上影': 'd', '长下影': 'u'} 
    df = assign_condition_value(df=df, column='长影线_trend', condition_dict=conditions, value_dict=values, default_value='n')

  # days since signal triggered
  df['up_pattern_score'] = 0
  df['down_pattern_score'] = 0
  df['up_pattern_description'] = ''
  df['down_pattern_description'] = ''
  
  pattern_weights = {
    '十字星': 0, '长影线': 1, '流星': 1, '锤子': 1, '腰带': 1, '平头': 1.1, 
    '穿刺': 1, '包孕': 1, '吞噬': 1, 
    '启明黄昏': 2, '窗口': 2, 
  }
  up_pattern_names = {
    '十字星': '十字星', '长影线': '长下影', '流星': '倒锤', '锤子': '锤子', '腰带': '多腰带', '平头': '平头底', 
    '穿刺': '多穿刺', '包孕': '多包孕', '吞噬': '多吞噬', '启明黄昏': '启明星', '窗口': '向上跳空', 
  }
  down_pattern_names = {
    '十字星': '十字星', '长影线': '长上影', '流星': '流星', '锤子': '墓碑', '腰带': '空腰带', '平头': '平头顶', 
    '穿刺': '空穿刺', '包孕': '空包孕', '吞噬': '空吞噬', '启明黄昏': '黄昏星', '窗口': '向下跳空',
  }
  all_candle_patterns = list(pattern_weights.keys())
  for col in all_candle_patterns:
    day_col = f'{col}_day'
    trend_col = f'{col}_trend'
    df[day_col] = df[trend_col].replace({'u':1, 'd':-1, 'n':0, '': 0}).fillna(0).astype(int)
    df[day_col] = sda(series=df[day_col], zero_as=1, one_restart=True)

    up_idx = df.query(f'{day_col} == 1').index
    down_idx = df.query(f'{day_col} == -1').index

    if col in ['十字星']:
      up_idx = [] # df.query(f'{day_col} == 1').index
      down_idx = df.query(f'{day_col} == -1 or {day_col} == 1').index

    df.loc[up_idx, 'up_pattern_score'] += pattern_weights[col]
    df.loc[up_idx, 'up_pattern_description'] += f'{up_pattern_names[col]}, '

    df.loc[down_idx, 'down_pattern_score'] -= pattern_weights[col]
    df.loc[down_idx, 'down_pattern_description'] += f'{down_pattern_names[col]}, '

  df['candle_pattern_score'] = df['up_pattern_score'] + df['down_pattern_score']
  df['up_pattern_description'] = df['up_pattern_description'].apply(lambda x: x[:-2] if (len(x) >=2 and x[-2] == ',') else x)
  df['down_pattern_description'] = df['down_pattern_description'].apply(lambda x: x[:-2] if (len(x) >=2 and x[-2] == ',') else x)

  # redundant intermediate columns
  for col in col_to_drop:
    if col in df.columns:
      df.drop(col, axis=1, inplace=True)

  return df

# add heikin-ashi candlestick features
def add_heikin_ashi_features(df: pd.DataFrame, ohlcv_col: dict = default_ohlcv_col, replace_ohlc: bool = False, dropna: bool = True) -> pd.DataFrame:
  """
  Add heikin-ashi candlestick dimentions for dataframe

  :param df: original OHLCV dataframe
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :returns: dataframe with candlestick columns
  :raises: none
  """
  # copy dataframe
  df = df.copy()

  # set column names
  open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # add previous stick
  columns_to_drop = []
  for col in [open, high, low, close]:
    tmp_col = f'prev_{col}'
    df[tmp_col] = df[col].shift(1)
    columns_to_drop.append(tmp_col)

  # calculate heikin-ashi ohlc
  df['H_Close'] = (df[open] + df[high] + df[low] + df[close])/4
  df['H_Open'] = (df[f'prev_{open}'] + df[f'prev_{close}'])/2
  df['H_High'] = df[[f'prev_{high}', 'H_Open', 'H_Close']].max(axis=1)
  df['H_Low'] = df[[f'prev_{low}', 'H_Open', 'H_Close']].min(axis=1)
    
  # replace original ohlc with heikin-ashi ohlc
  if replace_ohlc:
    for col in [open, high, low, close]:
      df.drop(f'{col}', axis=1, inplace=True)
    df.rename(columns={'H_Close': close, 'H_Open': open, 'H_High': high, 'H_Low': low}, inplace=True)

  # dropna values
  if dropna:
    df.dropna(inplace=True)

  # drop redundant columns
  for col in columns_to_drop:
    df.drop(col, axis=1, inplace=True)
  
  return df

# linear regression for recent high and low values
def add_linear_features(df: pd.DataFrame, max_period: int = 60, min_period: int = 5, is_print: bool = False) -> pd.DataFrame:
  """
  Add linear regression for High/Low by the most recent peaks and troughs

  :param df: original OHLCV dataframe
  :param max_period: maximum length of the input series
  :param min_period: minimum length of the input series  
  :returns: dataframe with candlestick columns
  :raises: none
  """
  # get all indexes
  idxs = df.index.tolist()
  skip_latest = 3

  # get current date, renko_color, earliest-start date, latest-end date
  earliest_start = idxs[-60] if len(idxs) >= 60 else idxs[0] # df.tail(max_period).index.min()
  latest_end = idxs[-skip_latest]
  if is_print:
    print(earliest_start, latest_end)

  # sampling: recent high/low 
  recent_period = int(max_period / 2)
  possible_idxs = df[:latest_end].index.tolist()
  middle_high = df[possible_idxs[-recent_period]:latest_end]['High'].idxmax()
  long_high = df[possible_idxs[-max_period]:latest_end]['High'].idxmax()
  middle_low =  df[possible_idxs[-recent_period]:latest_end]['Low'].idxmin()
  long_low = df[possible_idxs[-max_period]:latest_end]['High'].idxmin()

  # get slice of data
  start = earliest_start
  end = latest_end
  candidates = [middle_high, middle_low, long_high, long_low]
  start_candidates = [x for x in candidates if x < possible_idxs[-min_period]]
  start_candidates.sort(reverse=True)
  end_candidates = [x for x in candidates if x > possible_idxs[-min_period]]
  if len(end_candidates) > 0:
    end = max(end_candidates)
  if len(start_candidates) > 0:
    for s in start_candidates:
      if (end-s).days > min_period:
        start = s
        break
  if is_print:
    print(start, end)

  # get peaks and troughs
  tmp_data = df[start:end].copy()
  tmp_idxs = tmp_data.index.tolist()
  
  # gathering high and low points for linear regression
  high = {'x':[], 'y':[]}
  low = {'x':[], 'y':[]}
  for ti in tmp_idxs:
    x = idxs.index(ti)
    y_high = df.loc[ti, 'High']  
    y_low = df.loc[ti, 'Low']  
    high['x'].append(x)
    high['y'].append(y_high)
    low['x'].append(x)
    low['y'].append(y_low)

  # linear regression for high/low values
  highest_high = df[start:latest_end]['High'].max()
  lowest_low = df[start:latest_end]['Low'].min()

  # high linear regression
  if len(high['x']) < 2: 
    high_linear = (0, highest_high, 0, 0)
  else:
    high_linear = linregress(high['x'], high['y'])
    high_range = round((max(high['y']) + min(high['y']))/2, 3)
    slope_score = round(abs(high_linear[0])/high_range, 5)
    if slope_score < 0.001:
      high_linear = (0, highest_high, 0, 0)

  # low linear regression
  if len(low['x']) < 2:
    low_linear = (0, lowest_low, 0, 0)
  else:
    low_linear = linregress(low['x'], low['y'])
    low_range = round((max(low['y']) + min(low['y']))/2, 3)
    slope_score = round(abs(low_linear[0])/low_range, 5)
    if slope_score < 0.001:
      low_linear = (0, lowest_low, 0, 0)

  # # add high/low fit values
  idx_min = min(min(high['x']), min(low['x']))
  std_high = df[start:end]['High'].std()
  std_low = df[start:end]['Low'].std()
  std_factor = 0.3
  idx_num = range(1, len(idxs)+1)

  df['linear_fit_high'] = idx_num
  df['linear_fit_high'] = df['linear_fit_high'].apply(lambda x: np.nan if (x < idx_min) else (high_linear[0] * x + high_linear[1] + std_factor * std_high))
  df['linear_fit_low'] = idx_num
  df['linear_fit_low'] = df['linear_fit_low'].apply(lambda x: np.nan if (x < idx_min) else (low_linear[0] * x + low_linear[1] - std_factor * std_low))
  df[['linear_fit_high', 'linear_fit_low']] = df[['linear_fit_high', 'linear_fit_low']].fillna(method='ffill')
  
  return df
 
# linear regression for recent kama and ichimoku fast slow lines
def add_ma_linear_features(df: pd.DataFrame, period: int = 5, target_col: list = ['kama_fast', 'kama_slow', 'tankan', 'kijun']) -> dict:
  """
  Add linear regression for ichimoku/kama fast and slow lines

  :param df: dataframe with ichimoku/kama features
  :param period: length of input data  
  :param target_col: columns that need to predict  
  :returns: dataframe with prediction values
  :raises: none
  """
  result = {}
  
  for col in target_col:
    if col in df.columns:
      y = df[col].tail(period).values.tolist()
      x = list(range(1, period+1))
      reg_result = linregress(x, y)
      result[col] = reg_result
    else:
      continue

  return result

# kama and ichimoku fast/slow line support and resistance
def add_support_resistance(df: pd.DataFrame, target_col: list = default_support_resistant_col) -> pd.DataFrame:
  """
  Add support and resistance 

  :param df: dataframe with ta features
  :param target_col: column that support or resistant price
  :returns: dataframe with support or resistance columns
  :raises: none
  """
  # copy dataframe
  df = df.copy()
  col_to_drop = []
  
  # remove columns that not exists
  target_col = [x for x in target_col if x in df.columns]
  key_cols = ['tankan', 'kijun', 'kama_fast', 'kama_slow']
  other_cols = [x for x in target_col if x not in key_cols]

  for col in target_col:
    df[col] = df[col].round(3)
  
  # calculate cross-over day (with Close)
  for col in other_cols:
    if f'{col}_day' not in df.columns:
      df[f'{col}_day'] = cal_crossover_signal(df=df, fast_line='Close', slow_line=col, pos_signal=1, neg_signal=-1, none_signal=0)
      df[f'{col}_day'] = sda(series=df[f'{col}_day'], zero_as=1)
      col_to_drop.append(f'{col}_day')

  # calculate middle price
  df['mid_price'] = (df['High'] + df['Low']) / 2
  col_to_drop.append('mid_price')

  generated_cols = {'High': [], 'Low': []}
  # calculate support and resistance
  for col_1 in ['High', 'Low']:
    for col_2 in target_col:
      
      # calculate mutual distnace
      distance_col = f'{col_1}_to_{col_2}'
      col_to_drop.append(distance_col)

      tmp_distance = abs(df[col_1] - df[col_2]) / df[col_1]
      df[distance_col] = tmp_distance
      generated_cols[col_1].append(distance_col)

  df['support_score'] = 0
  df['support_description'] = ''
  df['resistant_score'] = 0
  df['resistant_description'] = ''
  df['break_up_score'] = 0
  df['break_up_description'] = ''
  df['break_down_score'] = 0
  df['break_down_description'] = ''

  # ================================ breakthorough =====================================
  break_weight = {}
  for col in target_col:
    if col in key_cols:
      break_weight[col] = 1
    elif col in ['candle_gap_top', 'candle_gap_bottom']:
      break_weight[col] = 0
    else:
      break_weight[col] = 0.5

  for col in target_col:

    df[f'{col}_break_up'] = 0
    df[f'{col}_break_down'] = 0

    up_query = f'(({col}_day == 1 or (candle_color == 1 and candle_entity_top > {col} and candle_entity_bottom < {col})) and (十字星_trend == "n" or (十字星_trend != "n" and candle_entity_bottom > {col}))'
    if 'renko' in col:
      up_query += ' and renko_real != "red") or (renko_real == "green")'
    elif 'candle_gap' in col:
      up_query += ') and (candle_gap != 2)'
    else:
      up_query += ')'
    break_up_idx = df.query(up_query).index # entity_diff > -0.5 and 
    df.loc[break_up_idx, 'break_up_description'] += f'{col}, '
    df.loc[break_up_idx, f'{col}_break_up'] += break_weight[col]

    down_query = f'(({col}_day == -1 or (candle_color == -1 and candle_entity_top > {col} and candle_entity_bottom < {col})) and (十字星_trend == "n" or (十字星_trend != "n" and candle_entity_top < {col}))'
    if 'renko' in col:
      down_query += ' and renko_real != "green") or (renko_real == "red")'
    elif 'candle_gap' in col:
      down_query += ') and (candle_gap != -2)'
    else:
      down_query += ')'
    break_down_idx = df.query(down_query).index # entity_diff > -0.5 and 
    df.loc[break_down_idx, 'break_down_description'] += f'{col}, '
    df.loc[break_down_idx, f'{col}_break_down'] -= break_weight[col]

  df['break_up_description'] = df['break_up_description'].apply(lambda x: ', '.join(list(set(x[:-2].split(', ')))))
  df['break_down_description'] = df['break_down_description'].apply(lambda x: ', '.join(list(set(x[:-2].split(', ')))))

  # ================================ intra-day support and resistant ===================
  # calculate support
  distance_threshold = 0.01
  distance_threshold_strict = 0.0075
  shadow_pct_threhold = 0.2
  for col in generated_cols['Low']:

    tmp_col = col.split('_to_')[-1]
    df[f'{tmp_col}_support'] = 0
    support_query = f'''
    (
      (candle_entity_bottom > {tmp_col}) and
      ({tmp_col}_break_up == 0)
    ) and
    (
      (
        (
          (candle_color == 1 and mid_price > {tmp_col}) or 
          (candle_color == -1 and Close > {tmp_col})
        ) and 
        (
          (candle_lower_shadow_pct > {shadow_pct_threhold}) or 
          (candle_lower_shadow_pct > candle_upper_shadow_pct)
        ) and 
        (
          (十字星_trend == "n") or 
          (十字星_trend != "n" and Low > {tmp_col}) 
        ) and
        (
          ({col} < {distance_threshold})
        )
      ) or 
      (
        (candle_color == 1 and {tmp_col}_day != 1) and
        ({col} < {distance_threshold_strict})
      )
    )
    '''.replace('\n', ' ')
    support_idx = df.query(support_query).index.tolist()
    df.loc[support_idx, 'support_description'] += f'{tmp_col}, '
    df.loc[support_idx, f'{tmp_col}_support'] += 1

  # calculate resistance
  for col in generated_cols['High']:

    tmp_col = col.split('_to_')[-1]
    df[f'{tmp_col}_resistant'] = 0
    resistant_query = f'''
    (
      (candle_entity_top < {tmp_col}) and
      ({tmp_col}_break_down == 0)
    ) and
    (
      (
        (
          (candle_color == -1 and mid_price < {tmp_col}) or 
          (candle_color == 1 and Close < {tmp_col})
        ) and 
        (
          (candle_upper_shadow_pct > {shadow_pct_threhold}) or 
          (candle_upper_shadow_pct > candle_lower_shadow_pct)
        ) and 
        (
          ({col} < {distance_threshold})
        )
      ) or 
      (
        (十字星_trend != "n") and 
        (High > {tmp_col} > Low)
      ) or
      (
        (candle_color == -1 and {tmp_col}_day != -1) and
        ({col} < {distance_threshold_strict})
      )
    )
    '''.replace('\n', ' ')
    resistant_idx = df.query(resistant_query).index.tolist()
    df.loc[resistant_idx, 'resistant_description'] += f'{tmp_col}, '
    df.loc[resistant_idx, f'{tmp_col}_resistant'] -= 1

  # ================================ in-day support and resistant ======================
  for col in target_col:
    
    up_query = f'((Open > {col} and Low <= {col} and Close >= {col}) or ({col}_day != 1 and Open < {col} and Close > {col})) and ({col}_support == 0) and (candle_entity_bottom >= {col}) and ({col}_break_up == 0)'
    if 'renko' in col:
      up_query += ' and (renko_real != "red")'
    support_idx = df.query(up_query).index
    df.loc[support_idx, 'support_description'] += f'{col}, '
    df.loc[support_idx, f'{col}_support'] += 1
    
    down_query = f'((Open < {col} and High >= {col} and Close <= {col}) or ({col}_day != -1 and Open > {col} and Close < {col})) and ({col}_resistant == 0) and (candle_entity_top <= {col}) and ({col}_break_down == 0)'
    if 'renko' in col:
      down_query += ' and (renko_real != "green")'
    resistant_idx = df.query(down_query).index
    df.loc[resistant_idx, 'resistant_description'] += f'{col}, '
    df.loc[resistant_idx, f'{col}_resistant'] -= 1

  df['support_description'] = df['support_description'].apply(lambda x: ', '.join(list(set(x[:-2].split(', ')))))
  df['resistant_description'] = df['resistant_description'].apply(lambda x: ', '.join(list(set(x[:-2].split(', ')))))
  
  # ================================ supporter and resistanter =========================
  # add support/supporter, resistant/resistanter for the last row
  max_idx = df.index.max() 
  df['resistant'] = np.nan
  df['resistanter'] = ''
  df['support'] = np.nan
  df['supporter'] = ''
  if df.loc[max_idx, 'resistant_description'] > '':
    resistanter_candidates = df.loc[max_idx, 'resistant_description'].split(', ')
    resistanters = {}
    for r in resistanter_candidates:
      resistanters[r] = df.loc[max_idx, r]
    resistanter = min(resistanters, key=resistanters.get)
    df.loc[max_idx, 'resistanter'] = resistanter
    df.loc[max_idx, 'resistant'] = df.loc[max_idx, resistanter]

  if df.loc[max_idx, 'support_description'] > '':
    supporter_candidates = df.loc[max_idx, 'support_description'].split(', ')
    supporters = {}
    for s in supporter_candidates:
      supporters[s] = df.loc[max_idx, s]
    supporter = max(supporters, key=supporters.get)
    df.loc[max_idx, 'supporter'] = supporter
    df.loc[max_idx, 'support'] = df.loc[max_idx, supporter]

  # ================================ bondary and break score ===========================
  df['boundary_score'] = 0
  df['break_score'] = 0

  # calculate scores
  for col in target_col:
    for idx in ['support', 'resistant', 'break_up', 'break_down']:
      tmp_col = f'{col}_{idx}'
      tmp_value = df[tmp_col] * 1 #(1 if col in key_cols else 0.5)

      score_col = f'{idx}_score'
      df[score_col] += tmp_value

      if idx == 'support':
        df['boundary_score'] += tmp_value
      elif idx == 'resistant':
        df['boundary_score'] += tmp_value
      elif idx == 'break_up':
        df['break_score'] += tmp_value
      elif idx == 'break_down':
        df['break_score'] += tmp_value
      else:
        print(f'error: {idx} not defined')

      if 'kama_slow' not in tmp_col:
        col_to_drop.append(tmp_col)

  # drop unnecessary columns
  for col in col_to_drop:
    if col in df.columns and col not in ['candle_gap_top_resistant', 'candle_gap_bottom_support']:
      df.drop(col, axis=1, inplace=True)

  return df


# ================================================ Trend indicators ================================================= #
# ADX(Average Directional Index) 
def add_adx_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, adx_threshold: int = 25) -> pd.DataFrame:
  """
  Calculate ADX(Average Directional Index)

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param adx_threshold: the threshold to filter none-trending signals
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  # col_to_drop = []

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  # close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate true range
  df = add_atr_features(df=df, n=n, cal_signal=False)

  # difference of high/low between 2 continuouss days
  df['high_diff'] = df[high] - df[high].shift(1)
  df['low_diff'] = df[low].shift(1) - df[low]
  
  # plus/minus directional movements
  df['zero'] = 0
  df['pdm'] = df['high_diff'].combine(df['zero'], lambda x1, x2: get_min_max(x1, x2, 'max'))
  df['mdm'] = df['low_diff'].combine(df['zero'], lambda x1, x2: get_min_max(x1, x2, 'max'))
  
  # plus/minus directional indicators
  df['pdi'] = 100 * em(series=df['pdm'], periods=n).mean() / df['atr']
  df['mdi'] = 100 * em(series=df['mdm'], periods=n).mean() / df['atr']

  # directional movement index
  df['dx'] = 100 * abs(df['pdi'] - df['mdi']) / (df['pdi'] + df['mdi'])

  # Average directional index
  df['adx'] = em(series=df['dx'], periods=n).mean()

  idx = df.index.tolist()
  for i in range(n*2, len(df)-1):
    current_idx = idx[i]
    previous_idx = idx[i-1]
    df.loc[current_idx, 'adx'] = (df.loc[previous_idx, 'adx'] * (n-1) + df.loc[current_idx, 'dx']) / n

  # (pdi-mdi) / (adx/25)
  df['adx_diff'] = (df['pdi'] - df['mdi'])# * (df['adx']/adx_threshold)
  df['adx_diff_ma'] = em(series=df['adx_diff'], periods=5).mean()

  # rename columns: adx_diff_ma -> adx_value; adx -> adx_strength
  df['adx_value'] = df['adx_diff_ma']
  df['adx_strength'] = df['adx']

  # fill na values
  if fillna:
    for col in ['pdm', 'mdm', 'atr', 'pdi', 'mdi', 'dx', 'adx']:
      df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
  
  # drop redundant columns
  df.drop(['high_diff', 'low_diff', 'zero', 'pdm', 'mdm', 'pdi', 'mdi', 'dx', 'adx', 'adx_diff', 'adx_diff_ma'], axis=1, inplace=True)

  return df

# Aroon # 25
def add_aroon_features(df: pd.DataFrame, n: int = 25, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [50, 50]) -> pd.DataFrame:
  """
  Calculate Aroon

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: upper and lower boundary for calculating signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

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

  # calculate gap between aroon_up and aroon_down
  df['aroon_gap'] = (df['aroon_up'] - df['aroon_down'])

  return df

# CCI(Commidity Channel Indicator)
def add_cci_features(df: pd.DataFrame, n: int = 20, c: float = 0.015, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [200, -200]) -> pd.DataFrame:
  """
  Calculate CCI(Commidity Channel Indicator) 

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param c: constant value used in cci calculation
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: upper and lower boundary for calculating signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  
  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate cci
  pp = (df[high] + df[low] + df[close]) / 3.0
  mad = lambda x : np.mean(np.abs(x-np.mean(x)))
  cci = (pp - pp.rolling(n, min_periods=0).mean()) / (c * pp.rolling(n).apply(mad,True))

  # assign values to dataframe
  df['cci'] = cci

  # calculate siganl
  df['cci_ma'] = sm(series=df['cci'], periods=5).mean() 

  return df

# DPO(Detrended Price Oscillator)
def add_dpo_features(df: pd.DataFrame, n: int = 20, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate DPO(Detrended Price Oscillator) 

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  
  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

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

# Ichimoku 
def add_ichimoku_features(df: pd.DataFrame, n_short: int = 9, n_medium: int = 26, n_long: int = 52, method: str = 'ta', is_shift: bool = True, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_status: bool = True) -> pd.DataFrame:
  """
  Calculate Ichimoku indicators

  :param df: original OHLCV dataframe
  :param n_short: short window size (9)
  :param n_medium: medium window size (26)
  :param n_long: long window size (52)
  :param method: original/ta way to calculate ichimoku indicators
  :param is_shift: whether to shift senkou_a and senkou_b n_medium units
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated


  # signal: 在tankan平移, kijun向下的时候应该卖出; 在tankan向上, kijun向上或平移的时候应该买入
  """
  # copy dataframe
  df = df.copy()
  col_to_drop = []

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # use original method to calculate ichimoku indicators
  if method == 'original':
    df = cal_moving_average(df=df, target_col=high, ma_windows=[n_short, n_medium, n_long], window_type='sm')
    df = cal_moving_average(df=df, target_col=low, ma_windows=[n_short, n_medium, n_long], window_type='sm')

    # generate column names
    short_high = f'{high}_ma_{n_short}'
    short_low = f'{low}_ma_{n_short}'
    medium_high = f'{high}_ma_{n_medium}'
    medium_low = f'{low}_ma_{n_medium}'
    long_high = f'{high}_ma_{n_long}'
    long_low = f'{low}_ma_{n_long}'
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

  # drop redundant columns  
  df.drop(col_to_drop, axis=1, inplace=True)

  return df

# KST(Know Sure Thing)
def add_kst_features(df: pd.DataFrame, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30, n1: int = 10, n2: int = 10, n3: int = 10, n4: int = 15, nsign: int = 9, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False) -> pd.DataFrame:
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
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  # col_to_drop = []

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

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
    kst_sign = kst_sign.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign values to df
  df['kst'] = kst
  df['kst_sign'] = kst_sign
  df['kst_diff'] = df['kst'] - df['kst_sign']
  df['kst_diff'] = (df['kst_diff'] - df['kst_diff'].mean()) / df['kst_diff'].std()

  return df

# MACD(Moving Average Convergence Divergence)
def add_macd_features(df: pd.DataFrame, n_fast: int = 12, n_slow: int = 26, n_sign: int = 9, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate MACD(Moving Average Convergence Divergence)

  :param df: original OHLCV dataframe
  :param n_fast: ma window of fast ma
  :param n_slow: ma window of slow ma
  :paran n_sign: ma window of macd signal line
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  
  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

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

# Mass Index
def add_mi_features(df: pd.DataFrame, n: int = 9, n2: int = 25, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Mass Index

  :param df: original OHLCV dataframe
  :param n: ema window of high-low difference
  :param n_2: window of cumsum of ema ratio
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  
  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  # close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  amplitude = df[high] - df[low]
  ema1 = em(series=amplitude, periods=n, fillna=fillna).mean()
  ema2 = em(series=ema1, periods=n, fillna=fillna).mean()
  mass = ema1 / ema2
  mass = mass.rolling(n2, min_periods=0).sum()
  
  # fillna value  
  if fillna:
    mass = mass.replace([np.inf, -np.inf], np.nan).fillna(n2)

  # assign value to df
  df['mi'] = mass

  # calculate signal
  if cal_signal:
    df['benchmark'] = 27
    df['triger_signal'] = cal_crossover_signal(df=df, fast_line='mi', slow_line='benchmark', pos_signal='b', neg_signal='n', none_signal='n')
    df['benchmark'] = 26.5
    df['complete_signal'] = cal_crossover_signal(df=df, fast_line='mi', slow_line='benchmark', pos_signal='n', neg_signal='s', none_signal='n')

    conditions = {
      'up': 'triger_signal == "b"', 
      'down': 'complete_signal == "s"'} 
    values = {
      'up': 'b', 
      'down': 's'}
    df = assign_condition_value(df=df, column='mi_signal', condition_dict=conditions, value_dict=values, default_value='n')

    df.drop(['benchmark', 'triger_signal', 'complete_signal'], axis=1, inplace=True)

  return df

# TRIX
def add_trix_features(df: pd.DataFrame, n: int = 15, n_sign: int = 9, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, signal_mode: str = 'mix') -> pd.DataFrame:
  """
  Calculate TRIX

  :param df: original OHLCV dataframe
  :param n: ema window of close price
  :param n_sign: ema window of signal line (ema of trix)
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  
  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate trix
  ema1 = em(series=df[close], periods=n, fillna=fillna).mean()
  ema2 = em(series=ema1, periods=n, fillna=fillna).mean()
  ema3 = em(series=ema2, periods=n, fillna=fillna).mean()
  trix = (ema3 - ema3.shift(1)) / ema3.shift(1)
  trix *= 100

  # fillna value
  if fillna:
    trix = trix.replace([np.inf, -np.inf], np.nan).fillna(0)
  
  # assign value to df
  df['trix'] = trix
  df['trix_sign'] = em(series=trix, periods=n_sign, fillna=fillna).mean()
  df['trix_diff'] = df['trix'] - df['trix_sign']

  return df

# Vortex
def add_vortex_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Vortex indicator

  :param df: original OHLCV dataframe
  :param n: ema window of close price
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
  
  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate vortex
  tr = (df[high].combine(df[close].shift(1), max) - df[low].combine(df[close].shift(1), min))
  trn = tr.rolling(n).sum()

  vmp = np.abs(df[high] - df[low].shift(1))
  vmm = np.abs(df[low] - df[high].shift(1))

  vip = vmp.rolling(n, min_periods=0).sum() / trn
  vin = vmm.rolling(n, min_periods=0).sum() / trn

  if fillna:
    vip = vip.replace([np.inf, -np.inf], np.nan).fillna(1)
    vin = vin.replace([np.inf, -np.inf], np.nan).fillna(1)
  
  # assign values to df
  df['vortex_pos'] = vip
  df['vortex_neg'] = vin
  df['vortex_diff'] = df['vortex_pos'] - df['vortex_neg']
  # df['vortex_diff'] = df['vortex_diff'] - df['vortex_diff'].shift(1)

  # calculate signal
  if cal_signal:
    df['vortex_signal'] = cal_crossover_signal(df=df, fast_line='vortex_pos', slow_line='vortex_neg')

  return df

# PSAR
def add_psar_features(df: pd.DataFrame, ohlcv_col: dict = default_ohlcv_col, step: float = 0.02, max_step: float = 0.10, fillna: bool = False) -> pd.DataFrame:
  """
  Calculate Parabolic Stop and Reverse (Parabolic SAR) indicator

  :param df: original OHLCV dataframe
  :param step: unit of a step
  :param max_step: up-limit of step
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
   
  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  df['psar'] = df[close].copy()
  df['psar_up'] = np.NaN
  df['psar_down'] = np.NaN

  up_trend = True
  af = step
  idx = df.index.tolist()
  up_trend_high = df.loc[idx[0], high]
  down_trend_low = df.loc[idx[0], low]
  
  for i in range(2, len(df)):
    current_idx = idx[i]
    previous_idx = idx[i-1]
    previous_previous_idx = idx[i-2]

    reversal = False
    max_high = df.loc[current_idx, high]
    min_low = df.loc[current_idx, low]

    if up_trend:
      df.loc[current_idx, 'psar'] = df.loc[previous_idx, 'psar'] + (af * (up_trend_high - df.loc[previous_idx, 'psar']))

      if min_low < df.loc[current_idx, 'psar']:
        reversal = True
        df.loc[current_idx, 'psar'] = up_trend_high
        down_trend_low = min_low
        af = step
      else:
        if max_high > up_trend_high:
          up_trend_high = max_high
          af = min(af+step, max_step)

        l1 = df.loc[previous_idx, low]
        l2 = df.loc[previous_previous_idx, low]
        if l2 < df.loc[current_idx, 'psar']:
          df.loc[current_idx, 'psar'] = l2
        elif l1 < df.loc[current_idx, 'psar']:
          df.loc[current_idx, 'psar'] = l1

    else:
      df.loc[current_idx, 'psar'] = df.loc[previous_idx, 'psar'] - (af * (df.loc[previous_idx, 'psar'] - down_trend_low))

      if max_high > df.loc[current_idx, 'psar']:
        reversal = True
        df.loc[current_idx, 'psar'] = down_trend_low
        up_trend_high = max_high
        af = step
      else:
        if min_low < down_trend_low:
          down_trend_low = min_low
          af = min(af+step, max_step)

        h1 = df.loc[previous_idx, high]
        h2 = df.loc[previous_previous_idx, high]
        if h2 > df.loc[current_idx, 'psar']:
          df.loc[current_idx, 'psar'] = h2
        elif h1 > df.loc[current_idx, 'psar']:
          df.loc[current_idx, 'psar'] = h1

    up_trend = (up_trend != reversal)

    if up_trend:
      df.loc[current_idx, 'psar_up'] = df.loc[current_idx, 'psar']
    else:
      df.loc[current_idx, 'psar_down'] = df.loc[current_idx, 'psar']

  # fill na values
  if fillna:
    for col in ['psar', 'psar_up', 'psar_down']:
      df[col] = df[col].fillna(method='ffill').fillna(-1)

  return df

# Schaff Trend Cycle
def add_stc_features(df: pd.DataFrame, n_fast: int = 23, n_slow: int = 50, n_cycle: int = 10, n_smooth: int = 3, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False) -> pd.DataFrame:
  """
  Calculate Schaff Trend Cycle indicator

  :param df: original OHLCV dataframe
  :param n_fast: short window
  :param n_slow: long window
  :param n_cycle: cycle size
  :param n_smooth: ema period over stoch_d and stock_kd
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()
   
  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # ema and macd
  ema_fast = em(series=df[close], periods=n_fast, fillna=fillna).mean()
  ema_slow = em(series=df[close], periods=n_slow, fillna=fillna).mean()
  macd = ema_fast - ema_slow
  macd_min = sm(series=macd, periods=n_cycle, fillna=fillna).min()
  macd_max = sm(series=macd, periods=n_cycle, fillna=fillna).max()

  stoch_k = 100 * (macd - macd_min) / (macd_max - macd_min)
  stoch_d = em(series=stoch_k, periods=n_smooth, fillna=fillna).mean()
  stoch_d_min = sm(series=stoch_d, periods=n_cycle).min()
  stoch_d_max = sm(series=stoch_d, periods=n_cycle).max()
  stoch_kd = 100 * (stoch_d - stoch_d_min) / (stoch_d_max - stoch_d_min)

  stc = em(series=stoch_kd, periods=n_smooth, fillna=fillna).mean()

  df['stc'] = stc
  df['25'] = 25
  df['75'] = 75
  df['stc_signal'] = cal_boundary_signal(df=df, upper_col='stc', upper_boundary='25', lower_col='stc', lower_boundary='75')
  
  df.drop(['25', '75'], axis=1, inplace=True)

  return df

# Renko
def add_renko_features(df: pd.DataFrame, brick_size_factor: float = 0.05, dynamic_brick: bool = True, merge_duplicated: bool = True) -> pd.DataFrame:
  """
  Calculate Renko indicator
  :param df: original OHLCV dataframe
  :param brick_size_factor: if not using atr, brick size will be set to Close*brick_size_factor
  :param merge_duplicated: whether to merge duplicated indexes in the final result
  :param dynamic_brick: whether to use dynamic brick size
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """

  # reset index and copy df
  original_df = util.remove_duplicated_index(df=df, keep='last')
  df = original_df.copy()

  if dynamic_brick:
    # use dynamic brick size: brick_size_factor * Close price
    df['bsz'] = (df['Close'] * brick_size_factor).round(3)
  
  else:
    # use static brick size: brick_size_factor * Close price
    df['bsz'] = (df['Close'].values[0] * brick_size_factor).round(3)

  na_bsz = df.query('bsz != bsz').index 
  df = df.drop(index=na_bsz).reset_index()
  brick_size = df['bsz'].values[0]

  # construct renko_df, initialize values for first row
  columns = ['Date', 'Open', 'High', 'Low', 'Close']
  renko_df = pd.DataFrame(columns=columns, data=[], )
  renko_df.loc[0, columns] = df.loc[0, columns]
  close = df.loc[0, 'Close'] // brick_size * brick_size
  renko_df.loc[0, ['Open', 'High', 'Low', 'Close']] = [close-brick_size, close, close-brick_size, close]
  renko_df['uptrend'] = True
  renko_df['renko_distance'] = brick_size
  columns = ['Date', 'Open', 'High', 'Low', 'Close', 'uptrend', 'renko_distance']

  # go through the dataframe
  for index, row in df.iterrows():

    # get current date and close price
    date = row['Date']
    close = row['Close']
    
    # get previous trend and close price
    row_p1 = renko_df.iloc[-1]
    uptrend = row_p1['uptrend']
    close_p1 = row_p1['Close']
    brick_size_p1 = row_p1['renko_distance']

    # calculate bricks    
    bricks = int((close - close_p1) / brick_size)
    data = []
    
    # if in a uptrend and close_diff is larger than 1 brick
    if uptrend and bricks >=1 :
      for i in range(bricks):
        r = [date, close_p1, close_p1+brick_size, close_p1, close_p1+brick_size, uptrend, brick_size]
        data.append(r)
        close_p1 += brick_size
      brick_size = row['bsz']

    # if in a uptrend and closs_diff is less than -2 bricks
    elif uptrend and bricks <= -2:
      # flip trend
      uptrend = not uptrend
      bricks += 1
      close_p1 -= brick_size_p1

      for i in range(abs(bricks)):
        r = [date, close_p1, close_p1, close_p1-brick_size, close_p1-brick_size, uptrend, brick_size]
        data.append(r)
        close_p1 -= brick_size
      brick_size = row['bsz']

    # if in a downtrend and close_diff is less than -1 brick
    elif not uptrend and bricks <= -1:
      for i in range(abs(bricks)):
        r = [date, close_p1, close_p1, close_p1-brick_size, close_p1-brick_size, uptrend, brick_size]
        data.append(r)
        close_p1 -= brick_size
      brick_size = row['bsz']

    # if in a downtrend and close_diff is larger than 2 bricks
    elif not uptrend and bricks >= 2:
      # flip trend
      uptrend = not uptrend
      bricks -= 1
      close_p1 += brick_size_p1

      for i in range(abs(bricks)):
        r = [date, close_p1, close_p1+brick_size, close_p1, close_p1+brick_size, uptrend, brick_size]
        data.append(r)
        close_p1 += brick_size
      brick_size = row['bsz']

    else:
      continue
      
    # construct the [1:] rows and attach it to the first row
    tmp_row = pd.DataFrame(data=data, columns=columns)
    renko_df = pd.concat([renko_df, tmp_row], sort=False)

  # get back to original dataframe
  df = original_df

  # convert renko_df to time-series, add extra features(columns) to renko_df
  renko_df = util.df_2_timeseries(df=renko_df, time_col='Date')
  renko_df.rename(columns={'Open':'renko_o', 'High': 'renko_h', 'High': 'renko_h', 'Low': 'renko_l', 'Close':'renko_c', 'uptrend': 'renko_color'}, inplace=True)
  
  # renko brick start/end points
  renko_df['renko_start'] = renko_df.index.copy()
  renko_df['renko_end'] = renko_df['renko_start'].shift(-1).fillna(df.index.max())
  # renko_df['renko_duration'] = renko_df['renko_end'] - renko_df['renko_start']
  # renko_df['renko_duration'] = renko_df['renko_duration'].apply(lambda x: x.days+1).astype(float)

  # renko color(green/red), trend(u/d), flip_point(renko_real), same-direction-accumulation(renko_brick_sda), sda-moving sum(renko_brick_ms), number of bricks(for later calculation)
  renko_df['renko_color'] = renko_df['renko_color'].replace({True: 'green', False:'red'})
  renko_df['renko_real'] = renko_df['renko_color'].copy()
  renko_df['renko_brick_number'] = 1
  
  # merge rows with duplicated date (e.g. more than 1 brick in a single day)
  if merge_duplicated:
    
    # remove duplicated date index
    duplicated_idx = list(set(renko_df.index[renko_df.index.duplicated()]))
    for idx in duplicated_idx:
      tmp_rows = renko_df.loc[idx, ].copy()

      # make sure they are in same color
      colors = tmp_rows['renko_color'].unique()
      if len(colors) == 1:
        color = colors[0]
        if color == 'green':
          renko_df.loc[idx, 'renko_o'] = tmp_rows['renko_o'].min()
          renko_df.loc[idx, 'renko_l'] = tmp_rows['renko_l'].min()
          renko_df.loc[idx, 'renko_h'] = tmp_rows['renko_h'].max()
          renko_df.loc[idx, 'renko_c'] = tmp_rows['renko_c'].max()
          renko_df.loc[idx, 'renko_distance'] = tmp_rows['renko_distance'].sum()
          renko_df.loc[idx, 'renko_brick_number'] = tmp_rows['renko_brick_number'].sum()
        elif color == 'red':
          renko_df.loc[idx, 'renko_o'] = tmp_rows['renko_o'].max()
          renko_df.loc[idx, 'renko_l'] = tmp_rows['renko_l'].min()
          renko_df.loc[idx, 'renko_h'] = tmp_rows['renko_h'].max()
          renko_df.loc[idx, 'renko_c'] = tmp_rows['renko_c'].min()
          renko_df.loc[idx, 'renko_distance'] = tmp_rows['renko_distance'].sum()
          renko_df.loc[idx, 'renko_brick_number'] = tmp_rows['renko_brick_number'].sum() 
        else:
          print(f'unknown renko color {color}')
          continue 
      else:
        print('duplicated index with different renko colors!')
        continue
    renko_df = util.remove_duplicated_index(df=renko_df, keep='last')

  # drop currently-existed renko_df columns from df, merge renko_df into df 
  for col in df.columns:
    if 'renko' in col:
      df.drop(col, axis=1, inplace=True)
  df = pd.merge(df, renko_df, how='left', left_index=True, right_index=True)

  # for rows in downtrend, renko_distance = -renko_distance
  red_idx = df.query('renko_color == "red"').index
  df.loc[red_idx, 'renko_distance'] = -df.loc[red_idx, 'renko_distance']

  # fill na values
  renko_columns = ['renko_o', 'renko_h','renko_l', 'renko_c', 'renko_color', 'renko_distance', 'renko_brick_number','renko_start', 'renko_end'] # 'renko_series_short', 'renko_series_long', 'renko_series_short_idx', 'renko_series_long_idx', 'renko_duration_p1', 'renko_direction', 'renko_duration'
  for col in renko_columns:
    df[col] = df[col].fillna(method='ffill')

  # calculate length(number of days to the end of current brick) 
  # calculate of each brick(or merged brick): renko_brick_length, renko_countdown_days(for ploting)
  # max_idx = df.index.max()
  # if merge_duplicated:
  #   df['s']  = df.index
  #   if df['s'].max() == max_idx:
  #     max_idx = max_idx + datetime.timedelta(days=1)
  #   # df['renko_countdown_days'] = df['renko_end'] - df['s'] 
  #   # df['renko_brick_length'] = df['s'] - df['renko_start']
  #   # df['renko_brick_length'] = df['renko_brick_length'].apply(lambda x: x.days+1).astype(float)
  #   df.drop('s', axis=1, inplace=True)
  # else:
  #   # df['renko_countdown_days'] = 1
  #   # df['renko_brick_length'] = 1

  # # below/among/above renko bricks  
  # above_idx = df.query('Close > renko_h').index
  # among_idx = df.query('renko_l <= Close <= renko_h').index
  # below_idx = df.query('Close < renko_l').index
  # df.loc[above_idx, 'renko_position'] = 1
  # df.loc[among_idx, 'renko_position'] = 0
  # df.loc[below_idx, 'renko_position'] = -1 

  # # renko support and resistant
  # df.loc[above_idx, 'renko_support'] = df.loc[above_idx, 'renko_h']
  # df.loc[below_idx, 'renko_resistant'] = df.loc[below_idx, 'renko_l']
  # df.loc[among_idx, 'renko_support'] = df.loc[among_idx, 'renko_l']
  # df.loc[among_idx, 'renko_resistant'] = df.loc[among_idx, 'renko_h']

  # number of days since renko triggered
  df['renko_day'] = df['renko_color'].replace({'green':1, 'red': -1})
  real_idx = df.query('renko_real in ["red", "green"]').index
  df.loc[real_idx, 'renko_day'] = 0
  df['renko_day'] = sda(df['renko_day'], zero_as=None)

  # renko position status (beyond/below/among brick)
  conditions = {
    '上方': '((candle_entity_bottom >= renko_h) or (renko_real == "green"))',
    '中上': '((renko_real not in ["green", "red"]) and (candle_entity_top > renko_h) and (renko_h > candle_entity_bottom >= renko_l))',
    '中间': '((renko_real not in ["green", "red"]) and (candle_entity_top <= renko_h) and (candle_entity_bottom >= renko_l))',
    '穿刺': '((renko_real not in ["green", "red"]) and (candle_entity_top > renko_h) and (candle_entity_bottom < renko_l))',
    '中下': '((renko_real not in ["green", "red"]) and (candle_entity_bottom < renko_l) and (renko_h >= candle_entity_top > renko_l))',
    '下方': '((candle_entity_top <= renko_l) or (renko_real == "red"))'}
  values = {
    '上方': 'up', '中上': 'mid_up',
    '中间': 'mid', '穿刺': 'out',
    '中下': 'mid_down', '下方': 'down'}
  df = assign_condition_value(df=df, column='相对renko位置', condition_dict=conditions, value_dict=values, default_value='')

  return df


# ================================================ Volume indicators ================================================ #
# Accumulation Distribution Index
def add_adi_features(df: pd.DataFrame, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Accumulation Distribution Index

  :param df: original OHLCV dataframe
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """

  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate ADI
  clv = ((df[close] - df[low]) - (df[high] - df[close])) / (df[high] - df[low])
  clv = clv.fillna(0.0)  # float division by zero
  ad = clv * df[volume]
  ad = ad + ad.shift(1)

  # fill na values
  if fillna:
    ad = ad.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign adi to df
  df['adi'] = ad

  # calculate signals
  if cal_signal:
    df['adi_signal'] = 'n'

  return df

# *Chaikin Money Flow (CMF)
def add_cmf_features(df: pd.DataFrame, n: int = 20, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Chaikin Money FLow

  :param df: original OHLCV dataframe
  :param n: ema window of close price
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """

  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate cmf
  mfv = ((df[close] - df[low]) - (df[high] - df[close])) / (df[high] - df[low])
  mfv = mfv.fillna(0.0)  # float division by zero
  mfv *= df[volume]
  cmf = (mfv.rolling(n, min_periods=0).sum() / df[volume].rolling(n, min_periods=0).sum())

  # fill na values
  if fillna:
    cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign cmf to df
  df['cmf'] = cmf

  # calculate signals
  if cal_signal:
    df['cmf_signal'] = cal_boundary_signal(df=df, upper_col='cmf', lower_col='cmf', upper_boundary=0.05, lower_boundary=-0.05)

  return df

# Ease of movement (EoM, EMV)
def add_eom_features(df: pd.DataFrame, n: int = 20, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False) -> pd.DataFrame:
  """
  Calculate Vortex indicator

  :param df: original OHLCV dataframe
  :param n: ema window of close price
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """

  # copy dataframe
  df = df.copy()
  # col_to_drop = []

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  # close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate eom
  eom = (df[high].diff(periods=1) + df[low].diff(periods=1)) * (df[high] - df[low]) / (df[volume] * 2)
  eom = eom.rolling(window=n, min_periods=0).mean()

  # fill na values
  if fillna:
    eom = eom.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign eom to df
  df['eom'] = eom
  
  # calculate eom_ma_14 and eom - eom_ma_14
  df = cal_moving_average(df=df, target_col='eom', ma_windows=[14], window_type='sm')
  df['eom_diff'] = df['eom'] - df['eom_ma_14']
  df['eom_diff'] = (df['eom_diff'] - df['eom_diff'].mean()) / df['eom_diff'].std()

  return df

# Force Index (FI)
def add_fi_features(df: pd.DataFrame, n1: int = 2, n2: int = 22, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Force Index

  :param df: original OHLCV dataframe
  :param n: ema window of close price
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """

  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate fi
  fi = df[close].diff(n1) * df[volume]#.diff(n)
  fi_ema = em(series=fi, periods=n2).mean()

  # fill na values
  if fillna:
    fi = fi.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign fi to df
  df['fi'] = fi
  df['fi_ema'] = fi_ema

  # calculate signals
  if cal_signal:
    df['fi_signal'] = 'n'

  return df

# *Negative Volume Index (NVI)
def add_nvi_features(df: pd.DataFrame, n: int = 255, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Negative Volume Index (NVI)

  :param df: original OHLCV dataframe
  :param n: ema window of close price
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate nvi
  price_change = df[close].pct_change()*100
  vol_decress = (df[volume].shift(1) > df[volume])

  nvi = pd.Series(data=np.nan, index=df[close].index, dtype='float64', name='nvi')

  nvi.iloc[0] = 1000
  for i in range(1, len(nvi)):
    if vol_decress.iloc[i]:
      nvi.iloc[i] = nvi.iloc[i-1] + (price_change.iloc[i])
    else:
      nvi.iloc[i] = nvi.iloc[i-1]

  # fill na values
  if fillna:
    nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign nvi to df
  df['nvi'] = nvi
  df['nvi_ema'] = em(series=nvi, periods=n).mean()

  # calculate signal
  if cal_signal:
    df['nvi_signal'] = 'n'

  return df

# *On-balance volume (OBV)
def add_obv_features(df: pd.DataFrame, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Force Index

  :param df: original OHLCV dataframe
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate obv
  df['OBV'] = np.nan
  c1 = df[close] < df[close].shift(1)
  c2 = df[close] > df[close].shift(1)
  if c1.any():
    df.loc[c1, 'OBV'] = - df[volume]
  if c2.any():
    df.loc[c2, 'OBV'] = df[volume]
  obv = df['OBV'].cumsum()
  obv = obv.fillna(method='ffill')

  # fill na values
  if fillna:
    obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign obv to df
  df['obv'] = obv

  df.drop('OBV', axis=1, inplace=True)
  return df

# *Volume-price trend (VPT)
def add_vpt_features(df: pd.DataFrame, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Vortex indicator

  :param df: original OHLCV dataframe
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate vpt
  df['close_change_rate'] = df[close].pct_change(periods=1)
  vpt = df[volume] * df['close_change_rate']
  vpt = vpt.shift(1) + vpt

  # fillna values
  if fillna:
    vpt = vpt.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign vpt value to df
  df['vpt'] = vpt

  # calculate signals
  if cal_signal:
    df['vpt_signal'] = 'n'

  # drop redundant columns
  df.drop(['close_change_rate'], axis=1, inplace=True)

  return df


# ================================================ Momentum indicators ============================================== #
# Awesome Oscillator
def add_ao_features(df: pd.DataFrame, n_short: int = 5, n_long: int = 34, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Awesome Oscillator

  :param df: original OHLCV dataframe
  :param n_short: short window size for calculating sma
  :param n_long: long window size for calculating sma
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  # close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate ao
  mp = 0.5 * (df[high] + df[low])
  ao = mp.rolling(n_short, min_periods=0).mean() - mp.rolling(n_long, min_periods=0).mean()

  # fill na values
  if fillna:
    ao = ao.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign ao to df
  df['ao'] = ao
  df['ao_ma'] = sm(series=df['ao'], periods=2).mean()

  return df

# Kaufman's Adaptive Moving Average (KAMA)
def cal_kama(df: pd.DataFrame, n1: int = 10, n2: int = 2, n3: int = 30, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False) -> pd.DataFrame:
  """
  Calculate Kaufman's Adaptive Moving Average

  :param df: original OHLCV dataframe
  :param n1: number of periods for Efficiency Ratio(ER)
  :param n2: number of periods for the fastest EMA constant
  :param n3: number of periods for the slowest EMA constant
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate kama
  close_values = df[close].values
  vol = pd.Series(abs(df[close] - np.roll(df[close], 1)))

  ER_num = abs(close_values - np.roll(close_values, n1))
  ER_den = vol.rolling(n1).sum()
  ER = ER_num / ER_den
  ER = ER.fillna(method='ffill')

  sc = ((ER * (2.0/(n2+1.0) - 2.0/(n3+1.0)) + 2.0/(n3+1.0)) ** 2.0).values

  kama = np.zeros(sc.size)
  N = len(kama)
  first_value = True

  for i in range(N):
    if np.isnan(sc[i]):
      kama[i] = np.nan
    else:
      if first_value:
        kama[i] = close_values[i]
        first_value = False
      else:
        kama[i] = kama[i-1] + sc[i] * (close_values[i] - kama[i-1])
  kama = pd.Series(kama, name='kama', index=df[close].index)

  # fill na values
  if fillna:
    kama = kama.replace([np.inf, -np.inf], np.nan).fillna(df[close])

  # assign kama to df
  df['kama'] = kama

  return df

# Kaufman's Adaptive Moving Average (KAMA: short [10, 2, 30], long: [10, 5, 30])
def add_kama_features(df: pd.DataFrame, n_param: dict = {'kama_fast': [10, 2, 30], 'kama_slow': [15, 5, 42]}, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False) -> pd.DataFrame:
  """
  Calculate Kaufman's Adaptive Moving Average Signal

  :param df: original OHLCV dataframe
  :param n_param: series of n parameters fro calculating kama in different periods
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate fast and slow kama
  for k in n_param.keys():
    tmp_n = n_param[k]
    if len(tmp_n) != 3:
      print(k, ' please provide all 3 parameters')
      continue
    else:
      n1 = tmp_n[0]
      n2 = tmp_n[1]
      n3 = tmp_n[2]
      df = cal_kama(df=df, n1=n1, n2=n2, n3=n3, ohlcv_col=ohlcv_col)
      df.rename(columns={'kama': k}, inplace=True)
  
  return df

# Money Flow Index(MFI)
def add_mfi_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [20, 80]) -> pd.DataFrame:
  """
  Calculate Money Flow Index Signal

  :param df: original OHLCV dataframe
  :param n: ma window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: boundaries for overbuy/oversell
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  volume = ohlcv_col['volume']

  # calculate adi
  typical_price = (df[high] + df[low] + df[close])  / 3.0

  df['up_or_down'] = 0
  df.loc[(typical_price > typical_price.shift(1)), 'up_or_down'] = 1
  df.loc[(typical_price < typical_price.shift(1)), 'up_or_down'] = -1

  money_flow = typical_price * df[volume] * df['up_or_down']

  n_positive_mf = money_flow.rolling(n).apply(
    lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), 
    raw=True)

  n_negative_mf = abs(money_flow.rolling(n).apply(
    lambda x: np.sum(np.where(x < 0.0, x, 0.0)),
    raw=True))

  mfi = n_positive_mf / n_negative_mf
  mfi = (100 - (100 / (1 + mfi)))

  # fill na values, as 50 is the central line (mfi wave between 0-100)
  if fillna:
    mfi = mfi.replace([np.inf, -np.inf], np.nan).fillna(50)

  # assign mfi to df
  df['mfi'] = mfi

  # calculate signals
  if cal_signal:
    df['mfi_signal'] = cal_boundary_signal(df=df, upper_col='mfi', lower_col='mfi', upper_boundary=max(boundary), lower_boundary=min(boundary))

  df.drop('up_or_down', axis=1, inplace=True)
  return df

# Relative Strength Index (RSI)
def add_rsi_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [30, 70]) -> pd.DataFrame:
  """
  Calculate Relative Strength Index

  :param df: original OHLCV dataframe
  :param n: ma window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: boundaries for overbuy/oversell
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate RSI
  diff = df[close].diff(1)#pct_change(1)
  
  up = diff.copy()
  up[diff < 0] = 0
  
  down = -diff.copy()
  down[diff > 0] = 0
  
  emaup = up.ewm(com=n-1, min_periods=0).mean()
  emadown = down.ewm(com=n-1, min_periods=0).mean()

  rsi = 100 * emaup / (emaup + emadown)

  # fill na values, as 50 is the central line (rsi wave between 0-100)
  if fillna:
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)

  # assign rsi to df
  df['rsi'] = rsi

  # calculate signals
  if cal_signal:
    df['rsi_signal'] = cal_boundary_signal(df=df, upper_col='rsi', lower_col='rsi', upper_boundary=max(boundary), lower_boundary=min(boundary), pos_signal='s', neg_signal='b', none_signal='n')

  return df

# Stochastic RSI
def add_srsi_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [20, 80]) -> pd.DataFrame:
  """
  Calculate Stochastic RSI

  :param df: original OHLCV dataframe
  :param n: ma window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: boundaries for overbuy/oversell
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate rsi
  df = add_rsi_features(df, n=n, ohlcv_col=ohlcv_col, cal_signal=False)
  
  # calculate stochastic
  rsi_min = df['rsi'].rolling(n, min_periods=0).min()
  rsi_max = df['rsi'].rolling(n, min_periods=0).max()
  stoch_rsi = (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
  
  # fill na values, as 50 is the central line (rsi wave between 0-100)
  if fillna:
    stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan).fillna(50)

  # assign stochastic values to df
  df['srsi'] = stoch_rsi

  # calculate signals
  if cal_signal:
    df['srsi_signal'] = cal_boundary_signal(df=df, upper_col='srsi', lower_col='srsi', upper_boundary=max(boundary), lower_boundary=min(boundary), pos_signal='s', neg_signal='b', none_signal='n')

  return df

# Stochastic Oscillator
def add_stoch_features(df: pd.DataFrame, n: int = 14, d_n: int = 3, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [20, 80]) -> pd.DataFrame:
  """
  Calculate Stochastic Oscillator

  :param df: original OHLCV dataframe
  :param n: ma window size
  :param d_n: ma window size for stoch
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param boundary: boundaries for overbuy/oversell
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate stochastic
  stoch_min = df[low].rolling(n, min_periods=0).min()
  stoch_max = df[high].rolling(n, min_periods=0).max()
  stoch_k = 100 * (df[close] - stoch_min) / (stoch_max - stoch_min)
  stoch_d = stoch_k.rolling(d_n, min_periods=0).mean()

  # fill na values, as 50 is the central line (rsi wave between 0-100)
  if fillna:
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50)
    stoch_d = stoch_d.replace([np.inf, -np.inf], np.nan).fillna(50)

  # assign stochastic values to df
  df['stoch_k'] = stoch_k
  df['stoch_d'] = stoch_d
  df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
  # df['stoch_diff'] = df['stoch_diff'] - df['stoch_diff'].shift(1)

  return df

# True strength index (TSI)
def add_tsi_features(df: pd.DataFrame, r: int = 25, s: int = 13, ema_period: int = 7, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate True strength index

  :param df: original OHLCV dataframe
  :param r: ma window size for high
  :param s: ma window size for low
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate tsi
  m = df[close] - df[close].shift(1, fill_value=df[close].mean())
  m1 = m.ewm(r).mean().ewm(s).mean()
  m2 = abs(m).ewm(r).mean().ewm(s).mean()
  tsi = 100 * (m1 / m2)
  tsi_sig = em(series=tsi, periods=ema_period).mean()

  # fill na values
  if fillna:
    tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    tsi_sig = tsi_sig.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign tsi to df
  df['tsi'] = tsi
  df['tsi_sig'] = tsi_sig

  # calculate signal
  if cal_signal:
    df['zero'] = 0
    df['tsi_fast_slow_signal'] = cal_crossover_signal(df=df, fast_line='tsi', slow_line='tsi_sig', result_col='signal', pos_signal='b', neg_signal='s', none_signal='n')
    df['tsi_centerline_signal'] = cal_crossover_signal(df=df, fast_line='tsi', slow_line='zero', result_col='signal', pos_signal='b', neg_signal='s', none_signal='n')

  return df

# Ultimate Oscillator
def add_uo_features(df: pd.DataFrame, s: int = 7, m: int = 14, l: int = 28, ws: float = 4.0, wm: float = 2.0, wl: float = 1.0, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = False) -> pd.DataFrame:
  """
  Calculate Ultimate Oscillator

  :param df: original OHLCV dataframe
  :param s: short ma window size 
  :param m: mediem window size
  :param l: long window size
  :param ws: weight for short period
  :param wm: weight for medium period
  :param wl: weight for long period
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate uo
  min_l_or_pc = df[close].shift(1, fill_value=df[close].mean()).combine(df[low], min)
  max_h_or_pc = df[close].shift(1, fill_value=df[close].mean()).combine(df[high], max)

  bp = df[close] - min_l_or_pc
  tr = max_h_or_pc - min_l_or_pc

  avg_s = bp.rolling(s, min_periods=0).sum() / tr.rolling(s, min_periods=0).sum()
  avg_m = bp.rolling(m, min_periods=0).sum() / tr.rolling(m, min_periods=0).sum()
  avg_l = bp.rolling(l, min_periods=0).sum() / tr.rolling(l, min_periods=0).sum()

  uo = 100.0 * ((ws * avg_s) + (wm * avg_m) + (wl * avg_l)) / (ws + wm + wl)

  # fill na values
  if fillna:
    uo = uo.replace([np.inf, -np.inf], np.nan).fillna(0)

  # assign uo to df
  df['uo'] = uo
  df['uo_diff'] = df['uo'] - df['uo'].shift(1)

  return df

# Williams %R
def add_wr_features(df: pd.DataFrame, lbp: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True, boundary: list = [-20, -80]) -> pd.DataFrame:
  """
  Calculate Williams %R

  :param df: original OHLCV dataframe
  :param lbp: look back period
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate wr
  hh = df[high].rolling(lbp, min_periods=0).max()
  ll = df[low].rolling(lbp, min_periods=0).min()

  wr = -100 * (hh - df[close]) / (hh - ll)

  # fill na values
  if fillna:
    wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)

  # assign wr to df
  df['wr'] = wr

  # calulate signal
  if cal_signal:
    df['wr_signal'] = cal_boundary_signal(df=df, upper_col='wr', lower_col='wr', upper_boundary=max(boundary), lower_boundary=min(boundary))

  return df


# ================================================ Volatility indicators ============================================ #
# Average True Range
def add_atr_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Average True Range

  :param df: original OHLCV dataframe
  :param n: ema window
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate true range
  df['h_l'] = df[low] - df[low]
  df['h_pc'] = abs(df[high] - df[close].shift(1))
  df['l_pc'] = abs(df[low] - df[close].shift(1))
  df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)

  # calculate average true range
  df['atr'] = sm(series=df['tr'], periods=n, fillna=True).mean()
  
  idx = df.index.tolist()
  for i in range(n, len(df)):
    current_idx = idx[i]
    previous_idx = idx[i-1]
    df.loc[current_idx, 'atr'] = (df.loc[previous_idx, 'atr'] * 13 + df.loc[current_idx, 'tr']) / 14

  # fill na value
  if fillna:
    df['atr'] = df['atr'].replace([np.inf, -np.inf], np.nan).fillna(0)

  # calculate signal
  if cal_signal:
    df['atr_diff'] = df['tr'] - df['atr']

  df.drop(['h_l', 'h_pc', 'l_pc'], axis=1, inplace=True)

  return df

# Price that will triger mean reversion signal
def cal_mean_reversion_expected_rate(df: pd.DataFrame, rate_col: str, n: int = 100, mr_threshold: int = 2) -> tuple:
  """
  Calculate the expected rate change to triger mean-reversion signals

  :param df: original dataframe which contains rate column
  :param rate_col: columnname of the change rate values
  :param n: windowsize of the moving window
  :param mr_threshold: the multiple of moving std to triger signals
  :returns: the expected up/down rate to triger signals
  :raises: none
  """
  x = sympy.Symbol('x')

  df = np.hstack((df.tail(n-1)[rate_col].values, x))
  ma = df.mean()
  std = sympy.sqrt(sum((df - ma)**2)/(n-1))
  result = sympy.solve(((x - ma)**2) - ((mr_threshold*std)**2), x)

  return result

# Bollinger Band
def add_bb_features(df: pd.DataFrame, n: int = 20, ndev: int = 2, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False) -> pd.DataFrame:
  """
  Calculate Bollinger Band

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ndev: standard deviation factor
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate bollinger band 
  mavg = sm(series=df[close], periods=n).mean()
  mstd = sm(series=df[close], periods=n).std(ddof=0)
  high_band = mavg + ndev*mstd
  low_band = mavg - ndev*mstd

  # fill na values
  if fillna:
      mavg = mavg.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
      mstd = mstd.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
      high_band = high_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
      low_band = low_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
      
  # assign values to df
  df['mavg'] = mavg
  df['mstd'] = mstd
  df['bb_high_band'] = high_band
  df['bb_low_band'] = low_band

  return df

# Donchian Channel
def add_dc_features(df: pd.DataFrame, n: int = 20, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Donchian Channel

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  # high = ohlcv_col['high']
  # low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate dochian channel
  high_band = df[close].rolling(n, min_periods=0).max()
  low_band = df[close].rolling(n, min_periods=0).min()
  middle_band = (high_band + low_band)/2

  # fill na values
  if fillna:
    high_band = high_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    low_band = low_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    middle_band = middle_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

  # assign values to df
  df['dc_high_band'] = high_band
  df['dc_low_band'] = low_band
  df['dc_middle_band'] = middle_band

  # calculate signals
  if cal_signal:
    conditions = {
      'up': f'{close} <= dc_low_band', 
      'down': f'{close} >= dc_high_band'} 
    values = {
      'up': 'b', 
      'down': 's'}
    df = assign_condition_value(df=df, column='dc_signal', condition_dict=conditions, value_dict=values, default_value='n')

  return df

# Keltner channel (KC)
def add_kc_features(df: pd.DataFrame, n: int = 10, ohlcv_col: dict = default_ohlcv_col, method: str = 'atr', fillna: bool = False, cal_signal: bool = True) -> pd.DataFrame:
  """
  Calculate Keltner channel (KC)

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param method: 'atr' or 'ta'
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate keltner channel
  typical_price = (df[high] +  df[low] + df[close]) / 3.0
  middle_band = typical_price.rolling(n, min_periods=0).mean()

  if method == 'atr':
    df = add_atr_features(df=df)
    high_band = middle_band + 2 * df['atr']
    low_band = middle_band - 2 * df['atr']

  else:
    typical_price = ((4*df[high]) - (2*df[low]) + df[close]) / 3.0
    high_band = typical_price.rolling(n, min_periods=0).mean()

    typical_price = ((-2*df[high]) + (4*df[low]) + df[close]) / 3.0
    low_band = typical_price.rolling(n, min_periods=0).mean()

  # fill na values
  if fillna:
    middle_band = middle_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    high_band = high_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    low_band = low_band.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

  # assign values to df
  df['kc_high_band'] = high_band
  df['kc_middle_band'] = middle_band
  df['kc_low_band'] = low_band

  # calculate signals
  if cal_signal:
    conditions = {
      'up': f'{close} < kc_low_band', 
      'down': f'{close} > kc_high_band'} 
    values = {
      'up': 'b', 
      'down': 's'}
    df = assign_condition_value(df=df, column='kc_signal', condition_dict=conditions, value_dict=values, default_value='n')

  return df

# Ulcer Index
def add_ui_features(df: pd.DataFrame, n: int = 14, ohlcv_col: dict = default_ohlcv_col, fillna: bool = False, cal_signal: bool = False) -> pd.DataFrame:
  """
  Calculate Ulcer Index

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """
  # copy dataframe
  df = df.copy()

  # set column names
  # open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']

  # calculate UI
  recent_max_close = df[close].rolling(n, min_periods=0).max()
  pct_drawdown = ((df[close] - recent_max_close) / recent_max_close) * 100
  sqr_avg = (pct_drawdown ** 2).rolling(n).mean()
  ui = np.sqrt(sqr_avg)

  # fill na values
  if fillna:
    ui = ui.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

  # assign values to df
  df['ui'] = ui

  return df


# ================================================ Other indicators ================================================= #


# ================================================ Indicator visualization  ========================================= #
# plot bar
def plot_bar(df: pd.DataFrame, target_col: str, start: Optional[str] = None, end: Optional[str] = None, width: float = 0.8, alpha: float = 1, color_mode: Literal['up_down', 'benchmark'] = 'up_down', edge_color: tuple = (0,0,0,0.1), benchmark: Optional[float] = None, add_line: bool = False, title: Optional[str] = None, use_ax: Optional[plt.Axes] = None, ytick_roration: int = 0, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot a series in bar
  :param df: time-series dataframe which contains target columns
  :param target_col: column name of the target column
  :param start: start date to plot
  :param end: end date to plot
  :param width: width of the bar
  :param alpha: alpha value
  :param color_mode: 'up_down' or 'benchmark'
  :param edge_color: color of the bar edge
  :param benchmark: benchmark value
  :param add_line: whether to add a line along bars
  :param title: plot title
  :param use_ax: the already-created ax to draw on
  :param ytick_roration: rotation degree of ytick
  :param plot_args: plot arguments
  :returns: figure with bar plotted
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.family': 'SimHei', 'axes.unicode_minus': 'False'})
    ax = fig.add_subplot(1,1,1, style=s)

  # plot bar
  current = target_col
  previous = 'previous_' + target_col
  if color_mode == 'up_down':  
    df['color'] = 'red'  
    df[previous] = df[current].shift(1)
    df.loc[df[current] >= df[previous], 'color'] = 'green'

  # plot in benchmark mode
  elif color_mode == 'benchmark' and benchmark is not None:
    df['color'] = 'red'
    df.loc[df[current] > benchmark, 'color'] = 'green'

  # plot indicator
  if 'color' in df.columns:
    ax.bar(df.index, height=df[target_col], color=df.color, width=width , alpha=alpha, label=target_col, edgecolor=edge_color)
    
  if add_line:
    color = 'gray'
    ax.plot(df[target_col], color=color, alpha=alpha, label=target_col)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='both', linestyle='-', linewidth=0.5, alpha=0.3)
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot volume
def plot_scatter(df: pd.DataFrame, target_col: str, start: Optional[str] = None, end: Optional[str] = None, marker: str = '.', alpha: float = 1, color_mode: Literal['up_down', 'benchmark'] = 'up_down', benchmark: Optional[float] = None, add_line: bool = False, title: Optional[str] = None, use_ax: Optional[plt.Axes] = None, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot a series in scatters
  :param df: time-series dataframe which contains target columns
  :param target_col: column name of the target column
  :param start: start date to plot
  :param end: end date to plot
  :param marker: marker type
  :param alpha: alpha value
  :param color_mode: 'up_down' or 'benchmark'
  :param benchmark: benchmark value
  :param add_line: whether to add a line along scatters
  :param title: plot title
  :param use_ax: the already-created ax to draw on
  :param plot_args: plot arguments
  :returns: figure with scatter plotted
  :raises: none
  """

  # copy dataframe within a specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  # plot bar
  current = target_col
  previous = 'previous_' + target_col
  if color_mode == 'up_down':  
    df['color'] = 'red'  
    df[previous] = df[current].shift(1)
    df.loc[df[current] >= df[previous], 'color'] = 'green'

  # plot in benchmark mode
  elif color_mode == 'benchmark' and benchmark is not None:
    df['color'] = 'red'
    df.loc[df[current] > benchmark, 'color'] = 'green'

  # plot indicator
  if 'color' in df.columns:
    # df = cal_change_rate(df=df, target_col='Volume', add_accumulation=False, add_prefix=True)
    # threshold = 0.2
    # strong_trend_idx = df.query(f'Volume_rate >= {threshold} or Volume_rate <= {-threshold}').index
    # weak_trend_idx = df.query(f'{-threshold} < Volume_rate < {threshold}').index
    # ax.scatter(strong_trend_idx, df.loc[strong_trend_idx, 'Volume'], color=df.loc[strong_trend_idx, 'color'], label=target_col, alpha=0.4, marker='s')
    # ax.scatter(weak_trend_idx, df.loc[weak_trend_idx, 'adx_strength'], color=df.loc[weak_trend_idx, 'color'], alpha=0.4, marker='_')
    ax.scatter(df.index, df[target_col], marker=marker,color=df.color, alpha=alpha, label=target_col)

  if add_line:
    ax.plot(df.index, df[target_col], color='black', linestyle='--', marker='.', alpha=alpha)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  # ax.grid(True, axis='both', linestyle='-', linewidth=0.5, alpha=0.3)

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot trend index
def plot_up_down(df: pd.DataFrame, col: str = 'trend_idx', start: Optional[str] = None, end: Optional[str] = None, use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot indicators around a benchmark

  :param df: dataframe which contains target columns
  :param start: start row to plot
  :param end: end row to stop
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :returns: figure with indicators and close price plotted
  :raises: none
  """
  # select data
  df = df[start:end].copy() 
  
  # calculate score moving average
  df[f'up_{col}_ma'] = em(series=df[f'up_{col}'], periods=3).mean()
  df[f'down_{col}_ma'] = em(series=df[f'down_{col}'], periods=3).mean()
  df[f'{col}_ma'] = em(series=df[f'{col}'], periods=3).mean()

  # change of score ma
  df[f'up_{col}_ma_change'] = df[f'up_{col}_ma'] - df[f'up_{col}_ma'].shift(1)
  df[f'down_{col}_ma_change'] = df[f'down_{col}_ma'] - df[f'down_{col}_ma'].shift(1)
  df[f'{col}_ma_change'] = df[f'{col}_ma'] - df[f'{col}_ma'].shift(1)
  # conditions = {
  #   'up': f'((up_{col}_ma_change > 0) and (down_{col}_ma_change > 0) and ({col} > 0))', 
  #   'down': f'((up_{col}_ma_change < 0) and (down_{col}_ma_change < 0)) or ({col}_ma_change < -0.5)'} 
  # values = {
  #   'up': 'u', 
  #   'down': 'd'}
  # df = assign_condition_value(df=df, column='{col}_trend', condition_dict=conditions, value_dict=values, default_value=np.nan)
  # df['{col}_trend'] = df['{col}_trend'].fillna(method='ffill')

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
  alpha = 0.4

  # plot benchmark(0)
  df['zero'] = 0
  ax.plot(df.index, df['zero'], color='grey', alpha=0.5*alpha)
  # # plot up_trend and down_trend
  # ax.fill_between(df.index, df.up_trend_idx_ma, df.zero, facecolor='green', interpolate=False, alpha=0.3, label='trend up') 
  # ax.fill_between(df.index, df.down_trend_idx_ma, df.zero, facecolor='red', interpolate=False, alpha=0.3, label='trend down')

  # # plot trend
  # df['prev_ta_trend'] = df['ta_trend'].shift(1)
  # green_mask = ((df.ta_trend == 'u') | (df.prev_ta_trend == 'u'))
  # red_mask = ((df.ta_trend == 'd') | (df.prev_ta_trend == 'd'))
  # ax.fill_between(df.index, df.up_trend_idx_ma, df.down_trend_idx_ma, where=green_mask,  facecolor='green', interpolate=False, alpha=0.3, label='trend up') 
  # ax.fill_between(df.index, df.up_trend_idx_ma, df.down_trend_idx_ma, where=red_mask, facecolor='red', interpolate=False, alpha=0.3, label='trend down')

  threshold = 0.5
  for direction in ['up', 'down']:
    
    direction_change_col = f'{direction}_{col}_ma_change'
    target_col = f'{direction}_{col}_direction'
    plot_col = f'{direction}_{col}_ma'
    
    conditions = {
      'up': f'{direction_change_col} > {threshold}' + (f' and {col} > 0' if direction == 'down' else ''), 
      'down': f'{direction_change_col} < {-threshold}' + (f' and down_{col} < 0' if direction == 'up' else '')} 
    values = {
      'up': 'u', 
      'down': 'd'}
    df = assign_condition_value(df=df, column=target_col, condition_dict=conditions, value_dict=values, default_value=np.nan)
    df[target_col] = df[target_col].fillna(method='ffill')
    green_mask = ((df[target_col] == "u") | (df[target_col].shift(1) == "u"))
    red_mask = ((df[target_col] == "d") | (df[target_col].shift(1) == "d"))
    ax.fill_between(df.index, df[plot_col], df.zero, where=green_mask,  facecolor='green', interpolate=False, alpha=0.3) 
    ax.fill_between(df.index, df[plot_col], df.zero, where=red_mask, facecolor='red', interpolate=False, alpha=0.15)
  
  # # ichimoku_distance_day
  # max_idx = df.index.max()
  # x_signal = max_idx + datetime.timedelta(days=2)
  # y_signal = 0
  # text_signal = int(df.loc[max_idx, 'ichimoku_distance_day'])
  # text_color = 'red' if text_signal < 0 else 'green'
  # plt.annotate(f'ich: {text_signal}', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, alpha=0.05))

  # title and legend
  # ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  # ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot score and trigger score
def plot_score(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, width: float = 0.8, use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot overall score and trigger score

  :param df: dataframe which contains target columns
  :param start: start row to plot
  :param end: end row to stop
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :returns: figure with indicators and close price plotted
  :raises: none
  """

  # select data
  df = df[start:end].copy() 

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
  alpha = 0.4

  # plot benchmark(0)
  df['zero'] = 0
  ax.plot(df.index, df['zero'], color='grey', alpha=0.5*alpha)

  # plot score
  df['prev_score'] = df['score'].shift(1)
  green_mask = ((df.score > 0) | (df.prev_score > 0))
  red_mask = ((df.score < 0) | (df.prev_score < 0))
  ax.fill_between(df.index, df.score, df.zero, where=green_mask,  facecolor='green', interpolate=False, alpha=0.2, label='trend up') 
  ax.fill_between(df.index, df.score, df.zero, where=red_mask, facecolor='red', interpolate=False, alpha=0.2, label='trend down')

  # plot trigger_score
  up_idx = df.query(f'trigger_score > 0').index
  down_idx = df.query(f'trigger_score < 0').index
  df['trigger_score_color'] = 'white'
  df.loc[up_idx, 'trigger_score_color'] = 'green'
  df.loc[down_idx, 'trigger_score_color'] = 'red'
  # ax.scatter(up_idx, df.loc[up_idx, 'trigger_score'], color='green', alpha=1, marker='_')
  # ax.scatter(down_idx, df.loc[down_idx, 'trigger_score'], color='red', alpha=1, marker='_')
  ax.bar(df.index, height=df['trigger_score'], color=df['trigger_score_color'], width=width, alpha=0.8)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  # ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot signals on price line
def plot_signal(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, signal_x: str = 'signal', signal_y: str = 'Close', use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, interval: str = 'day', trend_val: dict = default_trend_val, signal_val: dict = default_signal_val, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot signals along with the price

  :param df: dataframe with price and signal columns
  :param start: start row to plot
  :param end: end row to stop
  :param signal_x: columnname of the signal x values (default 'signal')
  :param signal_y: columnname of the signal y values (default 'Close')
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param trend_val: value of different kind of trends (e.g. 'u'/'d'/'n')
  :param signal_val: value of different kind of signals (e.g. 'b'/'s'/'n')
  :param plot_args: other plot arguments
  :returns: a signal plotted price chart
  :raises: none
  """
  # copy dataframe within the specific period
  df = df[start:end].copy()
  
  # use existed figure or create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  # plot settings
  settings = {
    'main': {'pos_signal_marker': 's', 'neg_signal_marker': 's', 'pos_trend_marker': 's', 'neg_trend_marker': 's', 'wave_trend_marker': '_', 'signal_alpha': 0.8, 'trend_alpha': 0.5, 'pos_color':'green', 'neg_color':'red', 'wave_color':'orange'},
    'other': {'pos_signal_marker': '^', 'neg_signal_marker': 'v', 'pos_trend_marker': '2', 'neg_trend_marker': '1', 'wave_trend_marker': '_', 'signal_alpha': 0.5, 'trend_alpha': 0.3, 'pos_color':'green', 'neg_color':'red', 'wave_color':'orange'}}
  style = settings['main'] if signal_x in ['signal'] else settings['other']

  # plot signal base line
  signal_label = signal_y
  ax.plot(df.index, df[signal_y], label=signal_label, alpha=0)

  # plot trend
  trend_col = signal_x.replace('signal', 'trend')
  if trend_col in df.columns:
    if signal_x not in ['signal']:
      for i in ['pos', 'neg', 'wave']:
        tmp_trend_value = trend_val[f'{i}_trend']
        tmp_data = df.query(f'{trend_col} == "{tmp_trend_value}"')
        ax.scatter(tmp_data.index, tmp_data[signal_y], marker=style[f'{i}_trend_marker'], color=style[f'{i}_color'], alpha=style['trend_alpha'])
    else:
      pass

  # buy and sell
  if signal_x == ' ':

    pass
    # types = ['转换', '触发', '前瞻']
    # settings = {
    #   '转换': {'pos_marker': '.', 'neg_marker': '.', 'pos_color': 'green', 'neg_color': 'red', 'pos_color_edge': 'none', 'neg_color_edge': 'none', 'alpha': 1},
    #   '触发': {'pos_marker': '|', 'neg_marker': '|', 'pos_color': 'green', 'neg_color': 'red', 'pos_color_edge': 'none', 'neg_color_edge': 'none', 'alpha': 1},
    #   '前瞻': {'pos_marker': '.', 'neg_marker': '.', 'pos_color': 'green', 'neg_color': 'red', 'pos_color_edge': 'none', 'neg_color_edge': 'none', 'alpha': 1},
    #   '默认': {'pos_marker': '^', 'neg_marker': 'v', 'pos_color': 'green', 'neg_color': 'red', 'pos_color_edge': 'green', 'neg_color_edge': 'red', 'alpha': 1},
    # }
    # for t in types:

    #   tmp_setting = settings[t]

    #   pos_data = df.query(f'{t} == 1')
    #   if len(pos_data) > 0:
    #     ax.scatter(pos_data.index, pos_data[signal_y], marker=tmp_setting['pos_marker'], color=tmp_setting['pos_color'], edgecolor=tmp_setting['pos_color_edge'], alpha=tmp_setting['alpha'])

    #   neg_data = df.query(f'{t} == -1')
    #   if len(neg_data) > 0:
    #     ax.scatter(neg_data.index, neg_data[signal_y], marker=tmp_setting['neg_marker'], color=tmp_setting['neg_color'], edgecolor=tmp_setting['neg_color_edge'], alpha=tmp_setting['alpha'])
    
  # trigger_score
  if signal_x in ['trigger']:

    # triggers
    threhold = 0
    alpha = 0.5

    # support/resistant
    tmp_data = df.query(f'(support_score > {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='_', color='green', alpha=alpha) # 'none', edgecolor=
  
    tmp_data = df.query(f'(resistant_score < {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='_', color='red', alpha=alpha) # 'none', edgecolor=

    # break_up/down
    tmp_data = df.query(f'(break_up_score > {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='^', color='none', edgecolor='green', alpha=alpha) # 'none', edgecolor=
  
    tmp_data = df.query(f'(break_down_score < {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='v', color='none', edgecolor='red', alpha=alpha) # 'none', edgecolor=

  # candle position and patterns
  if signal_x in ['candle']:

    threhold = 0
    alpha = 1
    
    # candle position
    tmp_data = df.query(f'(up_pattern_score > {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='o', color='none', edgecolor='green', alpha=alpha) # 'none', edgecolor=
  
    tmp_data = df.query(f'(down_pattern_score < {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='o', color='none', edgecolor='red', alpha=alpha) # 'none', edgecolor=

    df['candle_position_alpha'] = df['candle_position_score'].abs()
    # candle pattern
    tmp_data = df.query(f'(candle_position_score > {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='2', color='green', edgecolor='none', alpha=tmp_data['candle_position_alpha'].fillna(0)) # 'none', edgecolor=
  
    tmp_data = df.query(f'(candle_position_score < {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='1', color='red', edgecolor='none', alpha=tmp_data['candle_position_alpha'].fillna(0)) # 'none', edgecolor=

  # trend
  if signal_x in ['trend']:

    # trigger_score
    tmp_col_v = f'{signal_x}_score'
    tmp_col_a = f'{signal_x}_score_alpha'
    outer_alpha = 0.66

    tmp_data = df.query(f'(trend == "up")')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='s', color='none', edgecolor='green', alpha=outer_alpha) # outer_alpha

    tmp_data = df.query(f'(trend == "down")')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='s', color='none', edgecolor='red', alpha=outer_alpha)

    # final_score = adx_distance_change + overall_change_diff
    df['final_score_alpha'] = normalize(df['final_score'].abs())
    
    # pos_up_idx = df.query('final_score > 0 and final_score_change > 0').index
    # pos_down_idx = df.query('final_score > 0 and final_score_change <= 0').index
    # ax.scatter(pos_up_idx, df.loc[pos_up_idx, signal_y], marker='3', color='green', alpha=df.loc[pos_up_idx, 'final_score_alpha'].fillna(0))
    # ax.scatter(pos_down_idx, df.loc[pos_down_idx, signal_y], marker='4', color='red', alpha=df.loc[pos_down_idx, 'final_score_alpha'].fillna(0))
    
    # neg_up_idx = df.query('final_score < 0 and final_score_change > 0').index
    # neg_down_idx = df.query('final_score < 0 and final_score_change <=0 ').index
    # ax.scatter(neg_up_idx, df.loc[neg_up_idx, signal_y], marker='4', color='green', alpha=df.loc[neg_up_idx, 'final_score_alpha'].fillna(0))
    # ax.scatter(neg_down_idx, df.loc[neg_down_idx, signal_y], marker='3', color='red', alpha=df.loc[neg_down_idx, 'final_score_alpha'].fillna(0))

    # none_up_idx = df.query('final_score == 0 and final_score_change > 0').index
    # none_down_idx = df.query('final_score == 0 and final_score_change <=0 ').index
    # ax.scatter(none_up_idx, df.loc[none_up_idx, signal_y], marker='.', color='green', alpha=df.loc[neg_up_idx, 'final_score_alpha'].fillna(0))
    # ax.scatter(none_down_idx, df.loc[none_down_idx, signal_y], marker='.', color='red', alpha=df.loc[neg_down_idx, 'final_score_alpha'].fillna(0))

    up_idx = df.query('((adx_day > 0 or overall_change > 0) and (adx_distance_change > 0 and overall_change_diff > 0) and final_score > 0) or final_score > 0.2').index
    down_idx = df.query('((adx_day < 0 or overall_change < 0) and (adx_distance_change < 0 and overall_change_diff < 0) and final_score < 0) or final_score < -0.2').index
    ax.scatter(up_idx, df.loc[up_idx, signal_y], marker='s', color='green', edgecolor='none', alpha=df.loc[up_idx, 'final_score_alpha'].fillna(0))
    ax.scatter(down_idx, df.loc[down_idx, signal_y], marker='s', color='red', edgecolor='none', alpha=df.loc[down_idx, 'final_score_alpha'].fillna(0))


    # annotate info
    ylim = ax.get_ylim()
    y_max = ylim[1]
    max_idx = df.index.max()
    interval_factor = {'day':2, 'week': 10, 'month': 45}
    x_signal = max_idx + datetime.timedelta(days=1 * interval_factor[interval])

    # # annotate overall_change (overall_change_diff)
    # v = round(df.loc[max_idx, 'signal_score'],1)
    # v_change = str(df['signal_score'].round(1).values[-3:].tolist()).replace(' ', '')
    # y_signal = y_max - 1.5 # round(y_middle)
    # text_color = 'green' if v > 0 else 'red'
    # plt.annotate(f'{v_change}', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

    # annotate final_score (adx_distance_change+overall_change_diff)
    df['prev_final_score'] = df['final_score'].shift(1)
    v = round(df.loc[max_idx, 'final_score'], 2)
    text_color = 'green' if v > 0 else 'red'
    v_prev = round(df.loc[max_idx, 'prev_final_score'], 2)
    v_change = round(df.loc[max_idx, 'final_score_change'],2)
    v_change = f'+{v_change}' if v_change > 0 else f'{v_change}'
    y_signal = y_max - 0.5
    plt.annotate(f'{v_prev:0<4}{v_change:0<4}→{v:0<4}', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

    # annotate adx_distance (adx_distance_change)
    v = round(df.loc[max_idx, 'adx_distance'], 1)
    v_change = round(df.loc[max_idx, 'adx_distance_change'],2)
    text_color = 'green' if v_change > 0 else 'red'
    v_change = f'+{v_change}' if v_change > 0 else f'{v_change}'
    y_signal = y_max - 1.75
    plt.annotate(f'[短]{v:0<4}({v_change})', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

    # annotate overall_change (overall_change_diff)
    v = round(df.loc[max_idx, 'overall_change'],1)
    v_change = round(df.loc[max_idx, 'overall_change_diff'],2)
    text_color = 'green' if v_change > 0 else 'red'
    v_change = f'+{v_change}' if v_change > 0 else f'{v_change}'
    y_signal = y_max - 3 # round(y_middle)
    plt.annotate(f'[总]{v:0<4}({v_change})', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

    # # annotate adx/ichimoku/kama distance_status
    # adx_distance_status = df.loc[max_idx, "adx_distance_status"].replace('pos', '+').replace('neg', '-').replace('none', '=').replace('up', '↑').replace('down', '↓')
    # ichimoku_distance_status = df.loc[max_idx, "ichimoku_distance_status"].replace('pos', '+').replace('neg', '-').replace('none', '=').replace('up', '↑').replace('down', '↓')
    # kama_distance_status = df.loc[max_idx, "kama_distance_status"].replace('pos', '+').replace('neg', '-').replace('none', '=').replace('up', '↑').replace('down', '↓')
    # v = f'[A]{adx_distance_status}\n[I]{ichimoku_distance_status}\n[K]{kama_distance_status}'
    # v = f'  短  中  长  \n {adx_distance_status}  {ichimoku_distance_status}  {kama_distance_status} '
    # y_signal = y_max - 8 # round(y_middle + y_range/4)
    # text_color = 'black'
    # if (df.loc[max_idx, "adx_distance_status"] in ['posup', 'negup'] and df.loc[max_idx, "ichimoku_distance_status"] in ['posup', 'negup', 'posnone', 'negnone']):
    #   text_color = 'green'
    # elif (df.loc[max_idx, "adx_distance_status"] in ['posdown', 'negdown'] and df.loc[max_idx, "ichimoku_distance_status"] in ['posdown', 'negdown', 'posnone', 'negnone']):
    #   text_color = 'red'
    # else:
    #   pass
    # plt.annotate(f'{v}', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

    # title and legend
    ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
    ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
    ax.grid(True, axis='x', linestyle='-', linewidth=0.5, alpha=0.1)
    ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # 位置(及波动标识)
  if signal_x in ['position']:

    # position
    tmp_col_v = f'位置'
    tmp_color = 'orange'
    tmp_color_mid = 'grey'
    
    values = {'低': 'l', '中低': 'ml', '中高': 'mh', '高': 'h'}
    markers = {'低': '.', '中低': '.', '中高': 'o', '高': 'o'}
    colors = {'低': tmp_color, '中低': 'none', '中高': 'none', '高': tmp_color}
    edgecolors = {'低': tmp_color, '中低': tmp_color_mid, '中高': tmp_color_mid, '高': tmp_color}
    alphas = {'低': 0.7, '中低': 1, '中高': 1, '高': 0.7}

    for p in markers.keys():
      tmp_data = df.query(f'({tmp_col_v} == "{values[p]}")')
      if len(tmp_data) > 0:
        ax.scatter(tmp_data.index, tmp_data[signal_y], marker=markers[p], color=colors[p], edgecolor=edgecolors[p], alpha=alphas[p])

    # 模式
    alpha = 1

    # 超卖
    tmp_data = df.query(f'(超买超卖 == 1)')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='o', color='none', edgecolor='green', alpha=alpha) # 
    
    # 超买
    tmp_data = df.query(f'(超买超卖 == -1)')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker='o', color='none', edgecolor='red', alpha=alpha) # 'none', edgecolor=

    # # 其他pattern
    # tmp_data = df.query(f'(pattern_score > 0 and 超买超卖 == 0)')
    # if len(tmp_data) > 0:
    #   ax.scatter(tmp_data.index, tmp_data[signal_y], marker='s', color='none', edgecolor='green', alpha=alpha) # 
    
    # # 其他pattern
    # tmp_data = df.query(f'(pattern_score < 0 and 超买超卖 == 0)')
    # if len(tmp_data) > 0:
    #   ax.scatter(tmp_data.index, tmp_data[signal_y], marker='s', color='none', edgecolor='red', alpha=alpha) # 'none', edgecolor=

  # adx_syn(whether adx_value and adx_strength goes the same direction)
  if signal_x in ['adx', 'overall']:
    
    # columns
    outer_cols = {'adx': 'adx_day', 'overall': 'overall_change'}
    inner_cols = {'adx': 'adx_distance_change', 'overall': 'overall_change_diff'}
    outer_tmp_col_v = outer_cols[signal_x]
    inner_tmp_col_v = inner_cols[signal_x]

    # markers and alpha
    outer_pos_marker = 'o'
    outer_neg_marker = 'o'
    outer_alpha = 0.25
    inner_pos_marker = 'o'
    inner_neg_marker = 'o'
    df['inner_alpha'] = normalize(df[inner_tmp_col_v].abs())
    none_marker = '_'
    
    threhold = 0
    # outer circle
    tmp_data = df.query(f'({outer_tmp_col_v} > {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker=outer_pos_marker, color='none', edgecolor='green', alpha=outer_alpha)
  
    tmp_data = df.query(f'({outer_tmp_col_v} < {-threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker=outer_neg_marker, color='none', edgecolor='red', alpha=outer_alpha)

    # inner point
    tmp_data = df.query(f'({inner_tmp_col_v} > {threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker=inner_pos_marker, color='green', edgecolor='none', alpha=tmp_data['inner_alpha'].fillna(0))
  
    tmp_data = df.query(f'({inner_tmp_col_v} < {-threhold})')
    if len(tmp_data) > 0:
      ax.scatter(tmp_data.index, tmp_data[signal_y], marker=inner_neg_marker, color='red', edgecolor='none', alpha=tmp_data['inner_alpha'].fillna(0))

  # ichimoku/kama distance
  if signal_x in ["短期", "中期", "长期"]:

    term_indicator = {"短期":'adx', "中期":'ichimoku', "长期":'kama'}

    tmp_col_v = f'{term_indicator[signal_x]}_distance'
    tmp_col_s = f'{term_indicator[signal_x]}_distance_status'
    tmp_col_a = f'{term_indicator[signal_x]}_distance_alpha'
    defalut_alpha = 0.2
    df[tmp_col_a] = normalize(df[tmp_col_v].abs()).apply(lambda x: x if x > defalut_alpha else defalut_alpha)

    none_idx = df.query(f'{tmp_col_s} in ["noneup", "nonedown", "nonenone"]').index
    df.loc[none_idx, tmp_col_a] = 1
    
    # marker
    markers = {
      'posup': f'3', 'posdown': f'4', 'posnone': '_', 
      'negup': f'4', 'negdown': f'3', 'negnone': '_', 
      'noneup': '.', 'nonedown': '.', 'nonenone': '.'
    }
    
    # color
    colors = {
      'posup': f'green', 'posdown': f'green', 'posnone': f'green', 
      'negup': f'red', 'negdown': f'red', 'negnone': f'red', 
      'noneup': 'green', 'nonedown': 'red', 'nonenone': 'orange', 
    }
    
    for ds in markers.keys():
      tmp_data = df.query(f'{tmp_col_s} == "{ds}"')
      tmp_color = colors[ds]
      tmp_marker = markers[ds]
      if len(tmp_data) > 0:
        ax.scatter(tmp_data.index, tmp_data[signal_y], marker=tmp_marker, color=tmp_color, alpha=tmp_data[tmp_col_a].fillna(0))
  
  # return ax
  if use_ax is not None:
    return ax

# plot adx chart
def plot_adx(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, plot_args: dict = default_plot_args, interval: Literal['day', 'week', 'month', 'year'] = 'day') -> Optional[plt.Axes]:
  """
  Plot adx chart

  :param df: dataframe with ichimoku indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param date_col: column name of Date
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param plot_args: other plot arguments
  :returns: ichimoku plot
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end].copy()
  
  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  if interval == 'day':
    bar_width = datetime.timedelta(days=1)
  elif interval == 'week':
    bar_width = datetime.timedelta(days=7)
  elif interval == 'month':
    bar_width = datetime.timedelta(days=30)
  elif interval == 'year':
    bar_width = datetime.timedelta(days=365)
  else:
    pass
  ax.fill_between(df.index, 10, -10, hatch=None, linewidth=1, facecolor='yellow', edgecolor='black', alpha=0.2, zorder=0)

  # plot adx_value and adx_direction
  df['zero'] = 0
  ax.plot(df.index, df.zero, color='black', alpha=0.25, zorder=0)

  adx_threshold = 25
  # overlap_mask = green_mask & red_mask
  df['adx_color'] = 'orange'
  df.loc[df.adx_power_day > 0, 'adx_color'] = 'green'
  df.loc[df.adx_power_day < 0, 'adx_color'] = 'red'

  strong_trend_idx = df.query(f'adx_strength >= {adx_threshold}').index
  weak_trend_idx = df.query(f'adx_strength < {adx_threshold}').index
  ax.scatter(strong_trend_idx, df.loc[strong_trend_idx, 'adx_strength'], color=df.loc[strong_trend_idx, 'adx_color'], label='adx strong', alpha=0.5, marker='s', zorder=3)
  ax.scatter(weak_trend_idx, df.loc[weak_trend_idx, 'adx_strength'], color=df.loc[weak_trend_idx, 'adx_color'], label='adx weak', alpha=0.5, marker='_', zorder=3)

  # plot moving average value of adx_value
  ax.plot(df.index, df.adx_value_prediction, color='black', label='adx prediction',linestyle='-', alpha=0.5, zorder=3) # 

  target_col = 'adx_value'
  plot_bar(df=df, target_col=target_col, alpha=0.4, width=bar_width, color_mode='up_down', edge_color=(0.5,0.5,0.5,0), benchmark=0, title='', use_ax=ax, plot_args=default_plot_args)

  # annotate adx (adx_strength_change)
  interval_factor = {'day':2, 'week': 10, 'month': 45}
  ylim = ax.get_ylim()
  y_min = ylim[0]
  y_max = ylim[1]
  y_range = (y_max - y_min)
  y_middle = (y_max + y_min)/2

  max_idx = df.index.max()
  before_max_idx = df.index[-2]
  x_signal = max_idx + datetime.timedelta(days=1 * interval_factor[interval])
  v = round(df.loc[max_idx, 'adx_strength'], 1)
  v_change = round(df.loc[max_idx, 'adx_strength_change'],1)
  y_signal = round(y_middle + y_range/4)
  text_color = 'green' if v_change > 0 else 'red'
  text_color = 'green' if df.loc[max_idx, 'adx_strength_change'] > 0 else 'red'
  v_change = f'+{v_change}' if v_change > 0 else f'{v_change}'
  plt.annotate(f'[强]{v:0<5}({v_change})', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

  

  # annotate adx_value(adx_value_change)
  x_signal = max_idx + datetime.timedelta(days=1 * interval_factor[interval])
  v = round(df.loc[max_idx, 'adx_value'],1)
  v_change = round(df.loc[max_idx, 'adx_value_change'],1)
  y_signal = round(y_middle)
  text_color = 'green' if v_change > 0 else 'red'
  v_change = f'+{v_change}' if v_change > 0 else f'{v_change}'
  plt.annotate(f'[值]{v:0<5}({v_change})', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

  # annotate adx_value_prediction(adx_value_prediction - adx_value)
  x_signal = max_idx + datetime.timedelta(days=1 * interval_factor[interval])
  v = round(df.loc[max_idx, 'adx_value']-df.loc[max_idx, 'adx_value_prediction'],1)
  v_change = round(v - (df.loc[before_max_idx, 'adx_value']-df.loc[before_max_idx, 'adx_value_prediction']),1)
  y_signal = round(y_middle - y_range/4)
  text_color = 'green' if v_change > 0 else 'red'
  v_change = f'+{v_change}' if v_change > 0 else f'{v_change}'
  plt.annotate(f'[差]{v:0<5}({v_change})', xy=(x_signal, y_signal), xytext=(x_signal, y_signal), fontsize=12, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=text_color, edgecolor='none', alpha=0.1))

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot candlestick chart
def plot_candlestick(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, date_col: str = 'Date', add_on: list = ['split', 'gap', 'support_resistant', 'pattern'], use_ax: Optional[plt.Axes] = None, ohlcv_col: dict = default_ohlcv_col, color: dict = default_candlestick_color, plot_args: dict = default_plot_args, interval: Literal['day', 'week', 'month', 'year'] = 'day') -> Optional[plt.Axes]:
  """
  Plot candlestick chart

  :param df: dataframe with price and signal columns
  :param start: start row to plot
  :param end: end row to stop
  :param date_col: columnname of the date values
  :param add_on: add ons beyond basic candlesticks
  :param width: width of candlestick
  :param use_ax: the already-created ax to draw on
  :param ohlcv_col: columns names of Open/High/Low/Close/Volume
  :param color: up/down color of candlestick
  :param plot_args: other plot arguments
  :param interval: interval of the sec_data(day/week/month/year)
  :returns: a candlestick chart
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
  
  # set column names
  open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  
  # get indexes and max index
  idxs = df.index.tolist()
  max_idx = idxs[-1]
  annotation_idx = max_idx + datetime.timedelta(days=1)
  min_idx = df.index.min()
  padding = (df.High.max() - df.Low.min()) / 100

  # annotate split
  if 'split' in add_on and 'Split' in df.columns:
    
    splited = df.query('Split != 1.0').index
    all_idx = df.index.tolist()
    for s in splited:
      x = s
      y = df.loc[s, 'High']
      x_text = all_idx[max(0, all_idx.index(s)-2)]
      y_text = df.High.max()
      sp = round(df.loc[s, 'Split'], 4)
      plt.annotate(f'splited {sp}', xy=(x, y), xytext=(x_text,y_text), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='->', alpha=0.5), bbox=dict(boxstyle="round", fc="1.0", alpha=0.5))
  
  # annotate gaps
  if 'gap' in add_on:

    # for gap which start before 'start_date'
    if df.loc[min_idx, 'candle_gap_top'] > df.loc[min_idx, 'candle_gap_bottom']:
      df.loc[min_idx, 'candle_gap'] = df.loc[min_idx, 'candle_gap_color'] * 2

    # invalidate all gaps if there are too many gaps in the data
    up_gap_idxs = df.query('candle_gap == 2').index.tolist()
    down_gap_idxs = df.query('candle_gap == -2').index.tolist()
    if len(up_gap_idxs) > 10:
      up_gap_idxs = []
    if len(down_gap_idxs) > 10:
      down_gap_idxs = []

    # plot valid gaps
    gap_idxs = up_gap_idxs + down_gap_idxs
    for idx in gap_idxs:

      # gap start
      start = idx
      top_value = df.loc[start, 'candle_gap_top']
      bottom_value = df.loc[start, 'candle_gap_bottom']
      gap_color = 'green' if df.loc[start, 'candle_gap'] > 0 else 'red' # 'lightyellow' if df.loc[start, 'candle_gap'] > 0 else 'grey' # 
      gap_hatch = '||||' # '////' if df.loc[start, 'candle_gap'] > 0 else '\\\\\\\\' # 'xxxx'# 
      gap_hatch_color = 'black' # 'darkgreen' if df.loc[start, 'candle_gap'] > 0 else 'darkred' 
      
      # gap end
      end = None
      tmp_data = df[start:]
      for i, r in tmp_data.iterrows(): 
        if (r['candle_gap_top'] != top_value) or (r['candle_gap_bottom'] != bottom_value):  
          break      
        end = i

      # shift gap-start 1 day earlier
      pre_i = idxs.index(start)-1
      pre_start = idxs[pre_i] if pre_i > 0 else start
      tmp_data = df[start:end]
      ax.fill_between(df[pre_start:end].index, top_value, bottom_value, hatch=gap_hatch, facecolor=gap_color, interpolate=True, alpha=0.2, edgecolor=gap_hatch_color, linewidth=0.1, zorder=default_zorders['gap']) #,  

    # # gap support & resistant
    # ax.scatter(support_idx, df.loc[support_idx, 'Low'] * 0.98, marker='^', color='black', edgecolor='black', zorder=21)
    # ax.scatter(resistant_idx, df.loc[resistant_idx, 'High'] * 1.02, marker='v', color='black', edgecolor='black', zorder=21)

  # annotate close price, support/resistant(if exists)
  if 'support_resistant' in add_on:

    # annotate close price
    y_close = None
    y_text_close = None
    
    y_close_padding = padding*5
    y_close = df.loc[max_idx, 'Close'].round(3)
    y_text_close = y_close
    plt.annotate(f'{y_close}', xy=(annotation_idx, y_text_close), xytext=(annotation_idx, y_text_close), fontsize=13, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", alpha=0))

    y_resistant = None
    y_text_resistant = None
    y_support = None
    y_text_support = None

    # annotate resistant
    if df.loc[max_idx, 'resistanter'] is not None and df.loc[max_idx, 'resistanter'] > '':
      resistants = df.loc[max_idx, 'resistant_description'].split(', ') if df.loc[max_idx, 'resistant_description'] is not None else []
      resistants = [x for x in resistants if x != '']
      resistant_score = len(resistants)
      y_resistant = df.loc[max_idx, 'resistant'].round(3)
      y_text_resistant = y_resistant

      diff = y_text_resistant - y_text_close
      if diff < y_close_padding:
        y_text_resistant = y_text_close + y_close_padding
      plt.annotate(f'{y_resistant}[{resistant_score}]', xy=(annotation_idx, y_text_resistant), xytext=(annotation_idx, y_text_resistant), fontsize=13, xycoords='data', textcoords='data', color='black', va='bottom',  ha='left', bbox=dict(boxstyle="round", facecolor='red', alpha=0.1*resistant_score)) # 
    
    # annotate support 
    if df.loc[max_idx, 'supporter'] is not None and df.loc[max_idx, 'supporter'] > '':
      supports = df.loc[max_idx, 'support_description'].split(', ') if df.loc[max_idx, 'support_description'] is not None else []
      supports = [x for x in supports if x != '']
      support_score = len(supports)
      y_support = df.loc[max_idx, 'support'].round(3)
      y_text_support = y_support
      
      diff = y_text_close - y_text_support
      if diff < y_close_padding:
        y_text_support = y_text_close - y_close_padding
      plt.annotate(f'{y_support}[{support_score}]', xy=(annotation_idx, y_text_support), xytext=(annotation_idx, y_text_support), fontsize=13, xycoords='data', textcoords='data', color='black', va='top',  ha='left', bbox=dict(boxstyle="round", facecolor='green', alpha=0.1*support_score)) #

  # annotate candle patterns
  if 'pattern' in add_on:

    # plot cross star
    df['high_top'] = df['High'] * 1.02
    t_idx = df.query(f'十字星_trend != "n"').index
    t_color = 'purple' 
    ax.scatter(t_idx, df.loc[t_idx, 'high_top'], color=t_color, alpha=1, s=20, marker='x', zorder=default_zorders['price'])
    
    # plot flat-top/bottom
    len_unit = datetime.timedelta(days=1)
    rect_high = padding*1.5
    for t in ['u', 'd']:
      t_idx = df.query(f'平头_trend == "{t}"').index
      t_color = 'green' if t == 'u' else 'red'
      for i in t_idx:
        x = idxs.index(i)
        x = x - 1 if x > 1 else x
        x = idxs[x]
        rect_len = (i - x) 
        x = x - (len_unit * 0.5)
        rect_len = rect_len + len_unit
        y = df.loc[i, 'Low'] - 2*padding if t == 'u' else df.loc[i, 'High'] + 0.5*padding
        flat = Rectangle((x, y), rect_len, rect_high, facecolor='yellow', edgecolor=t_color, linestyle='-', linewidth=1, fill=True, alpha=0.8, zorder=default_zorders['candle_pattern'])
        ax.add_patch(flat)
        
    # settings for annotate candle patterns
    pattern_info = {
      
      # '十字星_trend': {'u': '高浪线', 'd': '十字星'},
      '腰带_trend': {'u': '腰带', 'd': '腰带'},
      '锤子_trend': {'u': '锤子', 'd': '吊颈'},
      '流星_trend': {'u': '倒锤', 'd': '流星'},

      # '平头_trend': {'u': '平底', 'd': '平顶'},
      '穿刺_trend': {'u': '穿刺', 'd': '乌云'},
      '吞噬_trend': {'u': '吞噬', 'd': '吞噬'},
      '包孕_trend': {'u': '包孕', 'd': '包孕'},

      '启明黄昏_trend': {'u': '启明星', 'd': '黄昏星'},
      # 'linear_bounce_trend': {'u': '反弹', 'd': '回落'},
      # 'linear_break_trend': {'u': '突破', 'd': '跌落'}
    }
    settings = {
      'normal': {'fontsize':12, 'fontcolor':'black', 'va':'center', 'ha':'center', 'up':'green', 'down':'red', 'alpha': 0.15, 'arrowstyle': '-'},
      'emphasis': {'fontsize':12, 'fontcolor':'black', 'va':'center', 'ha':'center', 'up':'yellow', 'down':'purple', 'alpha': 0.15, 'arrowstyle': '-'},
    }

    # plot other patterns
    up_pattern_annotations = {}
    down_pattern_annotations = {}
    for p in pattern_info.keys():
      
      style = 'normal'

      if p in df.columns:

        tmp_up_idx = df.query(f'{p} == "u"').index
        tmp_down_idx = df.query(f'{p} == "d"').index

        # positive patterns
        tmp_up_info = pattern_info[p]['u']
        if len(tmp_up_info) > 0: # and len(tmp_up_idx) < 10
          for i in tmp_up_idx:
            k = util.time_2_string(i.date())
            if k not in up_pattern_annotations: 
              up_pattern_annotations[k] = {'x': k, 'y': df.loc[i, 'Low'] - padding, 'text': tmp_up_info, 'style': style}
            else:
              up_pattern_annotations[k]['text'] = up_pattern_annotations[k]['text']  + f'/{tmp_up_info}'
              if up_pattern_annotations[k]['style'] == 'normal':
                up_pattern_annotations[k]['style'] = style 

        # negative patterns
        tmp_down_info = pattern_info[p]['d']
        if len(tmp_down_info) > 0: # and len(tmp_down_idx) < 10
          for i in tmp_down_idx:
            k = util.time_2_string(i.date())
            if k not in down_pattern_annotations:
              down_pattern_annotations[k] = {'x': k, 'y': df.loc[i, 'High'] + padding, 'text': tmp_down_info, 'style': style}
            else:
              down_pattern_annotations[k]['text'] = down_pattern_annotations[k]['text']  + f'/{tmp_down_info}'
              if down_pattern_annotations[k]['style'] == 'normal':
                down_pattern_annotations[k]['style'] = style 

    # candle pattern annotation
    annotations = {'up': up_pattern_annotations, 'down': down_pattern_annotations}
    y_text_padding = {0 : padding*0, 1: padding*5}
    for a in annotations.keys():
      
      # sort patterns by date
      tmp_a = annotations[a]
      tmp_annotation = {}
      sorted_keys = sorted(tmp_a.keys())
      for sk in sorted_keys:
        tmp_annotation[sk] = tmp_a[sk]

      # annotate patterns
      counter = 0
      for k in tmp_annotation.keys():

        x = tmp_annotation[k]['x']
        y = tmp_annotation[k]['y']
        text = tmp_annotation[k]['text']
        style = settings[tmp_annotation[k]['style']]
        if a == 'up':
          y_text = df.Low.min() - y_text_padding[counter % 2]
        else:
          y_text = df.High.max() + y_text_padding[counter % 2]
          
        plt.annotate(f'{text}', xy=(x, y), xytext=(x,y_text), fontsize=style['fontsize'], rotation=0, color=style['fontcolor'], va=style['va'],  ha=style['ha'], xycoords='data', textcoords='data', arrowprops=dict(arrowstyle=style['arrowstyle'], alpha=0.5, color='black'), bbox=dict(boxstyle="round", facecolor=style[a], edgecolor='none', alpha=style['alpha']))
        counter += 1

  # transform date to numbers, plot candlesticks
  # df.reset_index(inplace=True)
  # df[date_col] = df[date_col].apply(mdates.date2num)
  # plot_data = df[[date_col, open, high, low, close]]
  # ls, rs = candlestick_ohlc(ax=ax, quotes=plot_data.values, width=width, colorup=color['color_up'], colordown=color['color_down'], alpha=color['alpha'])
  
  # set offset, bar_width according to data interval
  if interval == 'day':
    OFFSET = datetime.timedelta(days=0.5)
    entity_width = datetime.timedelta(days=1)
  elif interval == 'week':
    OFFSET = datetime.timedelta(days=3.5)
    entity_width = datetime.timedelta(days=7)
  elif interval == 'month':
    OFFSET = datetime.timedelta(days=15)
    entity_width = datetime.timedelta(days=30)
  elif interval == 'year':
    OFFSET = datetime.timedelta(days=182.5)
    entity_width = datetime.timedelta(days=365)
  else:
    pass

  # set colors
  alpha = color['alpha'] 
  shadow_color = color['shadow_color']
  entity_edge_color = (0,0,0,0.1)
  ax.fill_between(df.index, df.loc[min_idx, 'High'], df.loc[min_idx, 'Low'], facecolor=None, interpolate=True, alpha=0, linewidth=1, zorder=default_zorders['candle_pattern'])

  # plot each candlestick
  for idx, row in df.iterrows():
    
    # set entity_color
    if row[close] >= row[open]:
      entity_color = color['color_up']
      lower = row[open]
      height = row[close] - row[open]
    else:
      entity_color = color['color_down']
      lower = row[close]
      height = row[open] - row[close]
    
    # set shadow_color
    if shadow_color is not None:
      line_color = shadow_color
    else:
      line_color = entity_color
    
    # plot shadow
    vline = Line2D(xdata=(idx, idx), ydata=(row[low], row[high]), color=line_color, linewidth=1, antialiased=True, zorder=default_zorders['candle_shadow'])
    
    # plot entity
    x = idx - OFFSET
    rect = Rectangle(xy=(x, lower), width=entity_width, height=height, facecolor=entity_color, linewidth=1, edgecolor=entity_edge_color, alpha=alpha, zorder=default_zorders['candle_entity'])

    # add shadow and entity to plot
    ax.add_line(vline)
    ax.add_patch(rect)
    
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot ichimoku chart
def plot_main_indicators(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, date_col: str = 'Date', add_on: list = ['split', 'gap', 'support_resistant', 'pattern'], target_indicator: list = ['price', 'ichimoku', 'kama', 'candlestick', 'bb', 'psar', 'renko', 'linear'], interval: Literal['day', 'week', 'month', 'year'] = 'day', use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, candlestick_color: dict = default_candlestick_color, ohlcv_col: dict = default_ohlcv_col, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot ichimoku chart

  :param df: dataframe with ichimoku indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param date_col: column name of Date
  :param add_on: additional parts on candlestick
  :param target_indicator: indicators need to be ploted on the chart
  :param interval: interval of the sec_data, which dicides the width of candlestick
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param ohlcv_col: columns names of Open/High/Low/Close/Volume
  :param candlestick_color: up/down color of candlestick
  :param plot_args: other plot arguments
  :returns: ichimoku plot
  :raises: none
  """
  # copy dataframe within a specific period
  max_idx = df.index.max()
  
  # add extention data
  extended = 1
  ext_columns = ['tankan', 'kijun', 'kama_fast', 'kama_slow']
  linear_cols = ['linear_fit_high', 'linear_fit_low']
  renko_cols = ['renko_color', 'renko_o', 'renko_h', 'renko_l', 'renko_c',  'renko_start', 'renko_distance', 'renko_brick_number']
  candle_gap_cols = ['candle_gap', 'candle_gap_top', 'candle_gap_bottom']
  support_resistant_cols = ['Close', 'support', 'supporter', 'support_score', 'support_description', 'resistant', 'resistanter', 'resistant_score', 'resistant_description']
  current_idx = max_idx
  next_idx = None
  interval_factor = {'day':1, 'week': 56, 'month': 259}
  
  if interval in ["day", "week", "month"]:
    
    # pred = add_ma_linear_features(df, period=period, target_col=ext_columns)

    for i in range(extended):

      next_idx = current_idx + datetime.timedelta(days = interval_factor[interval])
      df.loc[next_idx, candle_gap_cols] = df.loc[max_idx, candle_gap_cols]
      df.loc[next_idx, support_resistant_cols] = df.loc[max_idx, support_resistant_cols]
      df.loc[next_idx, ext_columns] = df.loc[max_idx, ext_columns]
      # for ec in ext_columns:
      #   slope = pred[ec][0]
      #   intercept = pred[ec][1]
      #   df.loc[next_idx, ec] = (period + i + 1) * ( slope) + intercept 

      if 'linear_fit_high' in df.columns and 'linear_fit_low' in df.columns:
        df.loc[next_idx, linear_cols] = df.loc[max_idx, linear_cols]

      if 'renko_real' in df.columns:
        df.loc[next_idx, 'renko_real'] = np.nan
        df.loc[next_idx, renko_cols] = df.loc[max_idx, renko_cols]

      current_idx = next_idx

  else:
    extended = None

  
  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
  
  # as we usually ploting data within a year, therefore log_y is not necessary
  # ax.set_yscale("log")

  # # get local max and min
  # local_maxima_indices = argrelextrema(df['Open'].values, np.greater, order=1)
  # local_minima_indices = argrelextrema(df['Close'].values, np.less, order=1)
  # lmax_idx = df.index[local_maxima_indices]
  # lmin_idx = df.index[local_minima_indices]
  # ax.plot(lmax_idx, df.loc[lmax_idx, 'Open'], color='red')
  # ax.plot(lmin_idx, df.loc[lmin_idx, 'Low'], color='red')
  
  # plot close price
  if 'price' in target_indicator:
    alpha = 0.2
    ax.plot(df.index, df[default_ohlcv_col['close']], label='close', color='black', linestyle='--', alpha=alpha, zorder=default_zorders['price'])
  
  # plot senkou lines, clouds, tankan and kijun
  if 'ichimoku' in target_indicator:
    
    # tankan/kijun
    alpha = 0.4
    ax.plot(df.index, df.tankan, label='tankan', color='green', linestyle='-', alpha=alpha, zorder=default_zorders['ichimoku']) # magenta
    ax.plot(df.index, df.kijun, label='kijun', color='red', linestyle='-', alpha=alpha, zorder=default_zorders['ichimoku']) # blue
    alpha = 0.1
    ax.fill_between(df.index, df.tankan, df.kijun, where=df.tankan > df.kijun, facecolor='green', interpolate=True, alpha=alpha, zorder=default_zorders['ichimoku'])
    ax.fill_between(df.index, df.tankan, df.kijun, where=df.tankan <= df.kijun, facecolor='red', interpolate=True, alpha=alpha, zorder=default_zorders['ichimoku'])

    # # senkou_a/b, chikan
    # alpha = 0.8
    # ax.plot(df.index, df.senkou_a, label='senkou_a', color='darkgreen', linestyle='-', alpha=alpha, zorder=default_zorders['ichimoku']) # magenta
    # ax.plot(df.index, df.senkou_b, label='senkou_b', color='darkred', linestyle='-', alpha=alpha, zorder=default_zorders['ichimoku']) # blue
    # ax.plot(df.index, df.chikan, label='chikan', color='purple', linestyle='-', alpha=alpha, zorder=default_zorders['ichimoku']) # blue
    # alpha = 0.25
    # ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a > df.senkou_b, facecolor='green', interpolate=True, alpha=alpha, zorder=default_zorders['ichimoku'])
    # ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a <= df.senkou_b, facecolor='red', interpolate=True, alpha=alpha, zorder=default_zorders['ichimoku'])


    if extended is not None:
      alpha = 0.6
      extended_idx = df.index[-extended:]
      tmp_df = df.loc[extended_idx].copy()
      tmp_hatch = None
      ax.fill_between(tmp_df.index, tmp_df.tankan, tmp_df.kijun, where=tmp_df.tankan > tmp_df.kijun, hatch=tmp_hatch, facecolor='white', edgecolor='grey', interpolate=True, alpha=alpha, zorder=default_zorders['ichimoku'])
      ax.fill_between(tmp_df.index, tmp_df.tankan, tmp_df.kijun, where=tmp_df.tankan <= tmp_df.kijun, hatch=tmp_hatch, facecolor='white', edgecolor='grey', interpolate=True, alpha=alpha, zorder=default_zorders['ichimoku'])
  
  # plot kama_fast/slow lines 
  if 'kama' in target_indicator:
    alpha = 0.8
    ax.plot(df.index, df.kama_fast, label='kama_fast', color='magenta', linestyle='-', alpha=alpha, zorder=default_zorders['kama']) # magenta
    ax.plot(df.index, df.kama_slow, label='kama_slow', color='blue', linestyle='-', alpha=alpha, zorder=default_zorders['kama'])

    # alpha = 0.1
    # ax.fill_between(df.index, df.kama_fast, df.kama_slow, where=df.kama_fast > df.kama_slow, facecolor='green', interpolate=True, alpha=alpha, zorder=-1)
    # ax.fill_between(df.index, df.kama_fast, df.kama_slow, where=df.kama_fast <= df.kama_slow, facecolor='red', interpolate=True, alpha=alpha, zorder=-1)
  
  # plot bollinger bands
  if 'bb' in target_indicator:
    alpha = 0.2
    alpha_fill = 0.02
    ax.plot(df.index, df.bb_high_band, label='bb_high_band', color='black', linestyle='-', alpha=alpha, zorder=default_zorders['default'])
    ax.plot(df.index, df.bb_low_band, label='bb_low_band', color='black', linestyle='-', alpha=alpha, zorder=default_zorders['default'])
    ax.plot(df.index, df.mavg, label='mavg', color='black', linestyle=':', alpha=alpha*3, zorder=default_zorders['default'])
    ax.fill_between(df.index, df.mavg, df.bb_high_band, facecolor='green', interpolate=True, alpha=alpha_fill, zorder=default_zorders['default'])
    ax.fill_between(df.index, df.mavg, df.bb_low_band, facecolor='red', interpolate=True, alpha=alpha_fill, zorder=default_zorders['default'])
  
  # plot average true range
  if 'atr' in target_indicator:
    alpha = 0.6
    ax.plot(df.index, df.atr, label='atr', color='green', alpha=alpha, zorder=default_zorders['default'])
    # ax.plot(df.index, df.bb_low_band, label='bb_low_band', color='red', alpha=alpha)
    # ax.plot(df.index, df.mavg, label='mavg', color='grey', alpha=alpha)
    # ax.fill_between(df.index, df.mavg, df.bb_high_band, facecolor='green', interpolate=True, alpha=0.1)
    # ax.fill_between(df.index, df.mavg, df.bb_low_band, facecolor='red', interpolate=True, alpha=0.2)
  
  # plot psar dots
  if 'psar' in target_indicator:
    alpha = 0.6
    s = 10
    ax.scatter(df.index, df.psar_up, label='psar', color='green', alpha=alpha, s=s, marker='o', zorder=default_zorders['default'])
    ax.scatter(df.index, df.psar_down, label='psar', color='red', alpha=alpha, s=s, marker='o', zorder=default_zorders['default'])
  
  # plot high/low trend
  if 'linear' in target_indicator:
    
    # --------------------------------- stright line -------------------------------#
    # # calculate linear feature
    # linear_result = add_linear_features(df)
    # high_linear = linear_result.get('high')
    # low_linear = linear_result.get('low')
    # start_x = linear_result.get('idx_start')
    # end_x = len(df.index)
    # start_point = df.index[start_x]
    # end_point = df.index.max()
    # std_ma = df[start_point:end_point]['Close'].std()

    # # calculate start and end points
    # std_factor = 0.75
    # start_high = high_linear[0] * start_x + high_linear[1] + std_factor * std_ma
    # start_low = low_linear[0] * start_x + low_linear[1] - std_factor * std_ma
    # end_high = high_linear[0] * end_x + high_linear[1] + std_factor * std_ma
    # end_low = low_linear[0] * end_x + low_linear[1] - std_factor * std_ma
    # highs = [start_high, end_high]
    # lows = [start_low, end_low]

    # # determine color for ploting
    # hs =  high_linear[0]
    # ls = low_linear[0]
    # hls = hs + ls
    
    # if hs >= 0 and ls >= 0 and hls > 0:
    #   linear_color = 'green'
    # elif hs <= 0 and ls <= 0 and hls < 0:
    #   linear_color = 'red'
    # elif hs >= 0 and ls <= 0 or hs <= 0 or ls >= 0 or hls == 0:
    #   linear_color = 'orange'
    # else:
    #   linear_color = 'grey'

    # # plot line
    # line_alpha = 0.5
    # ax.plot([start_point, end_point], highs, label='linear_fit_high', color=linear_color, linestyle='-.', alpha=line_alpha, zorder=default_zorders['default'])
    # ax.plot([start_point, end_point], lows, label='linear_fit_low', color=linear_color, linestyle='-.', alpha=line_alpha, zorder=default_zorders['default'])

    # # fill between linear_fit_high and linear_fit_low
    # fill_alpha = 0.25
    # linear_hatch = '--' # hatches[linear_direction]
    # ax.fill_between([start_point, end_point], highs, lows, facecolor='white', edgecolor=linear_color, hatch=linear_hatch, interpolate=True, alpha=fill_alpha, zorder=default_zorders['default'])    

    # --------------------------------- weekend line -------------------------------#
    linear_color = 'black'
    line_alpha = 0.5
    ax.plot(df.index, df.linear_fit_high, label='linear_fit_high', color=linear_color, linestyle='-.', alpha=line_alpha, zorder=default_zorders['default'])
    ax.plot(df.index, df.linear_fit_low, label='linear_fit_low', color=linear_color, linestyle='-.', alpha=line_alpha, zorder=default_zorders['default'])

    # fill between linear_fit_high and linear_fit_low
    fill_alpha = 0.25
    linear_range = df.linear_fit_high == df.linear_fit_high
    linear_hatch = '--'
    ax.fill_between(df.index, df.linear_fit_high, df.linear_fit_low, where=linear_range, facecolor='white', edgecolor=linear_color, hatch=linear_hatch, interpolate=True, alpha=fill_alpha, zorder=default_zorders['default'])    
  
  # plot candlestick
  if 'candlestick' in target_indicator:
    ax = plot_candlestick(df=df, start=start, end=end, date_col=date_col, add_on=add_on, ohlcv_col=ohlcv_col, color=candlestick_color, use_ax=ax, plot_args=plot_args, interval=interval)
  
  # plot renko bricks
  if 'renko' in target_indicator:
    ax = plot_renko(df, use_ax=ax, plot_args=default_plot_args, close_alpha=0)

  # plot mask for extended
  if extended is not None:
    extended_data = df[ext_columns].tail(extended).copy()
    ax.plot(extended_data, linestyle=':', color='white', zorder=default_zorders['extended'])

  # plot key line prices
  if 'add_line_value' > '':
    interval_factor = {'day':1, 'week': 6, 'month': 25}
    annotation_idx = max_idx  + datetime.timedelta(days=24*interval_factor[interval])
    ylim = ax.get_ylim()
    y_min = ylim[0]
    y_max = ylim[1]
    y_mid = (y_max + y_min) / 2

    # lines beyond/below close
    up_key_col = {}
    down_key_col = {}
    close_price = df.loc[max_idx, 'Close']
    col_names = {'tankan':'tankan', 'kijun':'kijun ', 'kama_fast':'km_fst', 'kama_slow':'km_slw', 'renko_h':'renk_h', 'renko_l':'renk_l', 'candle_gap_top':'gp_top', 'candle_gap_bottom':'gp_btm', 'linear_fit_high': 'lnr_hi', 'linear_fit_low': 'lnr_lo'}
    for col in col_names.keys():
      if col in df.columns:
        tmp_col_value = df.loc[max_idx, col]
        if np.isnan(tmp_col_value):
          continue
        else:
          tmp_col_value = round(tmp_col_value, 3)

        if tmp_col_value > close_price:
          up_key_col[col] = tmp_col_value
        else:
          down_key_col[col] = tmp_col_value
    
    # sort lines by their values from high to low
    sorted_up_key_col = dict(sorted(up_key_col.items(), key=lambda item: item[1], reverse=True))
    sorted_down_key_col = dict(sorted(down_key_col.items(), key=lambda item: item[1], reverse=True))
    
    # construct information string
    up_price = ''
    counter = 0
    for k in sorted_up_key_col:
      counter += 1
      up_price += f'{col_names[k]}: {sorted_up_key_col[k]:05.3f}'
      if counter < len(sorted_up_key_col):
        up_price += '\n'

    down_price = ''
    counter = 0
    for k in sorted_down_key_col:
      counter += 1
      down_price += f'{col_names[k]}: {sorted_down_key_col[k]:05.3f}'
      if counter < len(sorted_down_key_col):
        down_price += '\n'

    # add candle_upper_shadow_pct and candle_lower_shadow_pct and OHLC
    # open_price = round(df.loc[max_idx, "Open"], 3)
    high_price = round(df.loc[max_idx, "High"], 3)
    low_price = round(df.loc[max_idx, "Low"], 3)
    # close_price = round(df.loc[max_idx, "Close"], 3)
    upper_shadow = round(df.loc[max_idx, "candle_upper_shadow_pct"], 3) * 100
    entity = round(df.loc[max_idx, "candle_entity_pct"], 3) * 100
    lower_shadow = round(df.loc[max_idx, "candle_lower_shadow_pct"], 3) * 100
    price_info = f'{up_price}\n\n' + f'    [{high_price:05.3f}]    \n----{upper_shadow:5.1f}%----' + f'\n----{entity:5.1f}%----\n' + f'----{lower_shadow:5.1f}%----\n    [{low_price:05.3f}]    ' + f'\n\n{down_price}'
    
    # add the string to the chart
    plt.text(
      x=annotation_idx, y=y_mid, 
      s=price_info,
      fontsize=12, color='black', va='center', ha='left', bbox=dict(boxstyle="round", facecolor='white', edgecolor='black', alpha=0.25)
    )

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='x', linestyle=':', linewidth=0.5)
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot aroon chart
def plot_aroon(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None, use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot aroon chart

  :param df: dataframe with ichimoku indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param date_col: column name of Date
  :param ohlcv_col: columns names of Open/High/Low/Close/Volume
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param plot_args: other plot arguments
  :returns: ichimoku plot
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  # plot aroon_up/aroon_down lines 
  ax.plot(df.index, df.aroon_up, label='aroon_up', color='green', marker='.', alpha=0.2)
  ax.plot(df.index, df.aroon_down, label='aroon_down', color='red', marker='.', alpha=0.2)

  # fill between aroon_up/aroon_down
  ax.fill_between(df.index, df.aroon_up, df.aroon_down, where=df.aroon_up > df.aroon_down, facecolor='green', interpolate=True, alpha=0.2)
  ax.fill_between(df.index, df.aroon_up, df.aroon_down, where=df.aroon_up <= df.aroon_down, facecolor='red', interpolate=True, alpha=0.2)

  # # plot waving areas
  # wave_idx = (df.aroon_gap_change==0)&(df.aroon_up_change==df.aroon_down_change)#&(df.aroon_up<96)&(df.aroon_down<96)
  # for i in range(len(wave_idx)):
  #   try:
  #     if wave_idx[i]:
  #         wave_idx[i-1] = True
  #   except Exception as e:
  #     print(e)
  # ax.fill_between(df.index, df.aroon_up, df.aroon_down, facecolor='grey', where=wave_idx, interpolate=False, alpha=0.3)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='both', linestyle='--', linewidth=0.5)
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot renko chart
def plot_renko(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None, use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, close_alpha: float = 0.5, save_path: Optional[str] = None, save_image: bool = False, show_image: bool = False, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot renko chart

  :param df: dataframe with renko indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param close_alpha: alpha of close price
  :param save_path: path to save image
  :param save_image: whether to save image
  :param show_image: whether to show image
  :param plot_args: other plot arguments
  :returns: renko ploted ax
  """

  # copy data frame
  df = df[start:end].copy()
  min_idx = df.index.min()
  max_idx = df.index.max()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
    
  # plot close for displaying the figure
  ax.plot(df.Close, alpha=close_alpha)

  # whether to plot in date axes  
  if df.loc[min_idx, 'renko_real'] not in ['green', 'red']:
    df.loc[min_idx, 'renko_real'] = df.loc[min_idx, 'renko_color'] 
    df.loc[min_idx, 'renko_start'] = min_idx 
  
  # get rows where renko starts
  df = df.query('renko_real == "green" or renko_real =="red"').copy()
  df.loc[df.index.max(), 'renko_end'] = max_idx
  
  # plot renko
  legends = {'green': 'u', 'red': 'd', np.nan:' '}
  for index, row in df.iterrows():
    
    brick_length = (row['renko_end'] - row['renko_start'])
    hatch = '----'
    facecolor = 'white'
    edgecolor = 'black' if row['renko_color'] == 'green' else 'red'
    alpha = 0.4 if row['renko_color'] == 'green' else 0.5
    renko = Rectangle((index, row['renko_o']), brick_length, row['renko_distance'], facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, linewidth=0.1, fill=False, alpha=0.5, label=legends[row['renko_real']], zorder=default_zorders['renko']) #  edgecolor=row['renko_color'], linestyle='-', linewidth=5, 
    legends[row['renko_real']] = "_nolegend_"
    ax.add_patch(renko)
  
  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  # ax.grid(True, axis='x', linestyle='--', linewidth=0.5)

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])
  
  # return ax
  if use_ax is not None:
    return ax
  else:
    # save image
    if save_image and (save_path is not None):
      plt.savefig(save_path + title + '.png')
      
    # show image
    if show_image:
      plt.show()

# plot rsi chart
def plot_rsi(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None, use_ax: Optional[plt.Axes] = None, title: Optional[str] = None, plot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot aroon chart

  :param df: dataframe with ichimoku indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param date_col: column name of Date
  :param ohlcv_col: columns names of Open/High/Low/Close/Volume
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param plot_args: other plot arguments
  :returns: ichimoku plot
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  df['0'] = 0
  df['30'] = 30
  df['50'] = 50
  df['70'] = 70
  df['100'] = 100
  df['prev_rsi'] = df['rsi'].shift(1)
  df['rsi_marker'] = 'v'
  up_idx = df.query('rsi > prev_rsi').index
  df.loc[up_idx, 'rsi_marker'] = '^'

  ax.fill_between(df.index, df['0'], df['30'], facecolor='green', interpolate=True, alpha=0.2)
  ax.fill_between(df.index, df['70'], df['100'], facecolor='red', interpolate=True, alpha=0.2)

  # plot rsi and standard lines 
  for col in [70, 50, 30]:
    tmp_col_name = f'{col}'
    ax.plot(df.index, df[tmp_col_name], label=None, color='grey', alpha=0.5)

  for m in ['^', 'v']:
    tmp_df = df.query(f'rsi_marker == "{m}"')

    # overbuy
    ob_idx = tmp_df.query('rsi > 70').index
    ax.plot(ob_idx, tmp_df.loc[ob_idx, 'rsi'], label='RSI', color='black', marker=m, alpha=0.2)

    # oversell
    os_idx = tmp_df.query('rsi < 30').index
    ax.plot(os_idx, tmp_df.loc[os_idx, 'rsi'], label='RSI', color='black', marker=m, alpha=0.2)

    # up
    up_idx = tmp_df.query('50 < rsi < 70').index
    ax.plot(up_idx, tmp_df.loc[up_idx, 'rsi'], label='RSI', color='green', marker=m, alpha=0.5)

    # down
    down_idx = tmp_df.query('50 > rsi > 30').index
    ax.plot(down_idx, tmp_df.loc[down_idx, 'rsi'], label='RSI', color='red', marker=m, alpha=0.5)

  # # fill between aroon_up/aroon_down
  # ax.fill_between(df.index, df.aroon_up, df.aroon_down, where=df.aroon_up > df.aroon_down, facecolor='green', interpolate=True, alpha=0.2)
  # ax.fill_between(df.index, df.aroon_up, df.aroon_down, where=df.aroon_up <= df.aroon_down, facecolor='red', interpolate=True, alpha=0.2)

  # # plot waving areas
  # wave_idx = (df.aroon_gap_change==0)&(df.aroon_up_change==df.aroon_down_change)#&(df.aroon_up<96)&(df.aroon_down<96)
  # for i in range(len(wave_idx)):
  #   try:
  #     if wave_idx[i]:
  #         wave_idx[i-1] = True
  #   except Exception as e:
  #     print(e)
  # ax.fill_between(df.index, df.aroon_up, df.aroon_down, facecolor='grey', where=wave_idx, interpolate=False, alpha=0.3)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(False, axis='both', linestyle='--', linewidth=0.5)

  # return ax
  if use_ax is not None:
    return ax

# plot rate and trigger_score/trend_score for each target list
def plot_summary(data: dict, width: int = 20, unit_size: float = 0.3, wspace: float = 0.2, hspace: float = 0.1, plot_args: dict = default_plot_args, config: Optional[dict] = None, save_path: Optional[str] = None) -> plt.Axes:
  """
  Plot rate and trigger_score/overall_change for each target list
  :param data: dict of dataframes including 'result'
  :param figsize:  figure size
  :param save_path:  path to save the figure
  :returns: none
  :raises: none
  """

  # get pools and number of symbols in pools
  pools = list(data['result'].keys())
  num_symbols_in_pool = {}
  for p in pools:

    # get result and ta_data
    tmp_result_data = data['result'][p]
    tmp_len = len(tmp_result_data)
    num_symbols_in_pool[p] = tmp_len
    if tmp_len <= 0:
      continue

  # remove empty pool
  pools = list(num_symbols_in_pool.keys())
  
  # create axes for each pool
  n_row = len(num_symbols_in_pool.keys())  
  n_symbols = list(num_symbols_in_pool.values())
  fig = plt.figure(figsize=(width, sum(n_symbols)*unit_size))  
  gs = gridspec.GridSpec(n_row, 2, height_ratios=n_symbols, width_ratios=[1,1])
  plt.subplots_adjust(wspace=wspace, hspace=hspace)
  axes = {}

  key_crateria = ['trend_score', 'trigger_score', 'pattern_score', 'desc_score', 'signal_score', 'candle_pattern_score']
  prev_key_crateria = ['prev_' + x for x in key_crateria]
  sort_crateria = ['rate', 'signal_score'] # 'rate', 'pattern_score', 'trigger_score'
  sort_order = [True, True]

  # plot rate and score
  for i in range(n_row):

    num_total = n_symbols[i]
    num_down = 0
    t = pools[i]

    # use previous data
    if 'use_previous_data' > '':
      tmp_data = pd.DataFrame()
      for s in data['ta_data'][t].keys():
        tmp_data_s = data['ta_data'][t][s].copy()
        tmp_data_s[prev_key_crateria] = tmp_data_s[key_crateria].shift(1)
        tmp_data = pd.concat([tmp_data, tmp_data_s.tail(1)])
    # use current_data
    else:
      tmp_data = data['result'][t].copy()

    tmp_data = tmp_data.sort_values(by=sort_crateria, ascending=sort_order)
    tmp_data = tmp_data[['symbol', 'rate'] + key_crateria + prev_key_crateria].set_index('symbol')
    tmp_data['name'] = tmp_data.index.values

    # get data
    if ('a_company' in t) or ('hs300' in t) or('a_etf' in t):
      if config is not None:        
        names = config['visualization']['plot_args']['sec_name']
        for idx, row in tmp_data.iterrows():
          tmp_name = names.get(idx)
          if tmp_name is not None:
            tmp_data.loc[idx, 'name'] += f'({tmp_name})'

    tmp_data = tmp_data.set_index('name')
    tmp_data['rate'] = tmp_data['rate'] * 100

    # get ax
    if i == 0:
      rate_ax = plt.subplot(gs[i*2], zorder=1) 
      score_ax = plt.subplot(gs[i*2+1], zorder=0)
      axes['rate'] = rate_ax
      axes['score'] = score_ax
    else:
      rate_ax = plt.subplot(gs[i*2], sharex=axes['rate'], zorder=1) 
      score_ax = plt.subplot(gs[i*2+1], sharex=axes['score'], zorder=0)
    
    # plot signal score
    if 'plot_signal_score' > '':
      # plot signal score
      tmp_data['score_color'] = 'yellow'
      rate_ax.barh(tmp_data.index, tmp_data['signal_score'], color=tmp_data['score_color'], label='signal_score', alpha=0.5, edgecolor='k') #, edgecolor='k'  
      
      # plot rate
      tmp_data['rate_color'] = 'green'
      down_idx = tmp_data.query('rate <= 0').index    
      tmp_data.loc[down_idx, 'rate_color'] = 'red'
      num_down = len(down_idx)
      title_color = 'green' if num_total/2 > num_down else 'red'  
      rate_ax.barh(tmp_data.index, tmp_data['rate'], color=tmp_data['rate_color'], label='rate', alpha=0.5) #, edgecolor='k'
      rate_ax.set_xlabel(f'[{t.replace("_day", "")}] Rate ({num_total-num_down}/{num_total})', labelpad = 10, fontsize = 20) 
      rate_ax.legend(loc='upper left', ncol=plot_args['ncol']) 

    # plot previous score
    if 'plot_previous_score' > '':
      # plot trigger/position/pattern/trend score
      tmp_data['max'] = 0
      tmp_data['min'] = 0
      colors = {'trigger_score': 'red', 'trend_score': 'green', 'pattern_score': 'blue', 'signal_score': 'white', 'desc_score': 'purple'}
      for col in ['signal_score', 'desc_score', 'trend_score', 'pattern_score', 'trigger_score']:
        
        value_col = f'prev_{col}'
        edgecolor = 'k' if col in ['signal_score'] else 'none'
        hatch ='///' if col in ['signal_score'] else None

        tmp_data['tmp_value_pos'] = tmp_data[value_col]
        tmp_data['tmp_value_neg'] = tmp_data[value_col]
        
        tmp_data['tmp_value_pos'] = tmp_data['tmp_value_pos'].apply(lambda x: max(0, x))
        tmp_data['tmp_value_neg'] = tmp_data['tmp_value_neg'].apply(lambda x: min(0, x))

        score_ax.barh(tmp_data.index, tmp_data['tmp_value_pos'], color=colors[col], left=tmp_data['max'], alpha=0.5, edgecolor=edgecolor, hatch=hatch)
        score_ax.barh(tmp_data.index, tmp_data['tmp_value_neg'], color=colors[col], left=tmp_data['min'], label=col, alpha=0.5, edgecolor=edgecolor, hatch=hatch)

        if col not in ['signal_score']:
          tmp_data['max'] += tmp_data['tmp_value_pos']
          tmp_data['min'] += tmp_data['tmp_value_neg']

      score_ax.legend(loc='upper left', ncol=plot_args['ncol']) 
     
      # reverse X axis
      # score_ax.invert_xaxis()
      score_ax.set_xlabel(f'Previous Scores', labelpad = 10, fontsize = 20) 

    # borders
    rate_ax.spines['right'].set_alpha(0)
    score_ax.spines['left'].set_alpha(0)

    # y label
    rate_ax.yaxis.set_ticks_position("right")
    score_ax.yaxis.set_ticks_position("left")
    # plt.setp(rate_ax.get_yticklabels(), visible=False)
    plt.setp(score_ax.get_yticklabels(), visible=False)
    
    # grid
    rate_ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    rate_ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=1)
    score_ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    # score_ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=1)

    # rate_ax.xaxis.set_ticks_position('top') 
    rate_ax.xaxis.set_label_position('top')

    # score_ax.xaxis.set_ticks_position('top') 
    score_ax.xaxis.set_label_position('top')

    rate_ax.xaxis.label.set_color(title_color)
    score_ax.xaxis.label.set_color(title_color)

  # save image
  if save_path is not None:
    plt.savefig(save_path, bbox_inches = 'tight')

  return score_ax

# plot review of signal's price
def plot_review(prefix: str, df: pd.DataFrame, sort_factors: list = ['信号分数', "模式分数", 'ADX天数', 'ADX起始'], sort_orders: list = [False, True, False, False], width: int = 20, unit_size: float = 0.3, wspace: float = 0.2, hspace: float = 0.1, plot_args: dict = default_plot_args, config: Optional[dict] = None, save_path: Optional[str] = None) -> plt.Axes:
  """
  Plot rate and signal indicators for signal
  :param df: signal dataframe
  :param figsize:  figure size
  :param save_path:  path to save the figure
  :returns: none
  :raises: none
  """

  if df is None or len(df) == 0:
    print(f'data not exists:')
    return None

  # get pools and number of symbols in pools
  n_row = 1
  num_symbols = len(df)

  # create axes for each pool
  fig = plt.figure(figsize=(width, num_symbols*unit_size))  
  gs = gridspec.GridSpec(n_row, 2, height_ratios=[num_symbols], width_ratios=[1,1])
  # gs.update(wspace=wspace, hspace=hspace)
  plt.subplots_adjust(wspace=wspace, hspace=hspace)
  axes = {}

  primary_factor = sort_factors[0]
  secondary_factor = sort_factors[1]
  validation_statistic = f'{len(df.query("验证 > 0"))}/{len(df)}'

  # plot rate and score
  for i in range(n_row):

    # get target data
    tmp_data = df.sort_values(by=sort_factors, ascending=sort_orders).copy()
    tmp_data = tmp_data.set_index('名称') 
    tmp_data['name'] = tmp_data.index.values
    tmp_data['验证'] = tmp_data['验证'] * 100

    # get ax
    if i == 0:
      rate_ax = plt.subplot(gs[i*2], zorder=1) 
      score_ax = plt.subplot(gs[i*2+1], zorder=0)
      axes['signal'] = rate_ax
      axes['review'] = score_ax
    else:
      rate_ax = plt.subplot(gs[i*2], sharex=axes['signal'], zorder=1) 
      score_ax = plt.subplot(gs[i*2+1], sharex=axes['review'], zorder=0)
    
    # plot signal rank
    tmp_data['score_color'] = 'yellow'
    rate_ax.barh(tmp_data.index, tmp_data[primary_factor], color=tmp_data['score_color'], label=primary_factor, alpha=0.5, edgecolor='k')
    
    # plot rate
    tmp_data['pattern_color'] = 'green'
    down_idx = tmp_data.query(f'{secondary_factor} <= 0').index    
    tmp_data.loc[down_idx, 'pattern_color'] = 'red'
    title_color = 'black' 
    rate_ax.barh(tmp_data.index, tmp_data[secondary_factor], color=tmp_data['pattern_color'], label=secondary_factor, alpha=0.5) #, edgecolor='k'
    rate_ax.set_xlabel(f'{primary_factor} - {secondary_factor}', labelpad = 10, fontsize = 20) 
    rate_ax.legend(loc='upper right', ncol=plot_args['ncol']) 

    # plot trigger score
    # score_ax.barh(tmp_data.index, tmp_data['验证'], color='yellow', label='trigger_score', alpha=0.5, edgecolor='k')
    # tmp_data['score_bottom'] = tmp_data['trigger_score']
    # for index, row in tmp_data.iterrows():
    #   if False: #(#row['trigger_score'] > 0 and row['trend_score_change'] > 0) or (row['trigger_score'] < 0 and row['trend_score_change'] < 0):
    #     continue
    #   else:
    #     tmp_data.loc[index, 'score_bottom'] = 0

    # plot score
    tmp_data['score_color'] = 'green'
    down_idx = tmp_data.query('验证 <= 0').index    
    tmp_data.loc[down_idx, 'score_color'] = 'red'
    score_ax.barh(tmp_data.index, tmp_data['验证'], color=tmp_data['score_color'], left=0,label='验证', alpha=0.5) #, edgecolor='k'  
    score_ax.set_title(f'验证结果 - {validation_statistic}', fontsize=20)
    score_ax.legend(loc='upper left', ncol=plot_args['ncol']) 

    # borders
    rate_ax.spines['right'].set_alpha(0)
    score_ax.spines['left'].set_alpha(0)

    # y label
    rate_ax.yaxis.set_ticks_position("right")
    score_ax.yaxis.set_ticks_position("left")
    # plt.setp(rate_ax.get_yticklabels(), visible=False)
    plt.setp(score_ax.get_yticklabels(), visible=False)
    
    # grid
    rate_ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    rate_ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=1)
    score_ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    score_ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=1)

    # rate_ax.xaxis.set_ticks_position('top') 
    rate_ax.xaxis.set_label_position('top')

    # score_ax.xaxis.set_ticks_position('top') 
    score_ax.xaxis.set_label_position('top')

    rate_ax.xaxis.label.set_color(title_color)
    score_ax.xaxis.label.set_color(title_color)

  # save image
  if save_path is not None:
    plt.savefig(save_path, bbox_inches = 'tight')

  return score_ax

# plot selected 
def plot_selected(data: dict, config: dict, make_pdf: bool = False, dst_path: Optional[str] = None, file_name: Optional[str] = None) -> pd.DataFrame:
  """
  Plot  signals
  :param data: dict of dataframes including'result'
  :param config: global parameter config
  :param make_pdf: whether to make pdf from images
  :param dst_path: destination path of pdf
  :param file_name: file name of pdf
  :returns: selected_data
  :raises: none
  """

  # initialization
  selected_data = pd.DataFrame()
  
  # get image path
  if 'result' in data.keys():
    for ti in data['result'].keys():
      if len(data['result'][ti]) > 0:
        
        # get target_list and interval, then construct image path
        ti_split = ti.split('_')
        target_list = target_list = '_'.join(ti_split[:-1])
        interval = ti_split[-1]
        img_path = config['result_path'] + target_list + f'/{interval}/'
        data['result'][ti]['img_path'] = data['result'][ti]['symbol'].apply(lambda x: f'{img_path}{x}.png') 
        selected_data = pd.concat([selected_data, data['result'][ti]])

  # calculate rank and sort by rank    
  # selected_data['rank'] = selected_data['signal_rank'] + selected_data['inday_trend_score']
  selected_data = selected_data.query('adx_direction > 0').sort_values(['trend_score', 'signal_day', 'adx_direction_start'], ascending=[True, True, True])

  # make pdf from images
  if make_pdf:
    img_to_pdf = selected_data['img_path'].tolist()
    dst_path = config['home_path'] + 'Desktop/view' if dst_path is None else dst_path
    if not os.path.exists(dst_path):
      os.mkdir(dst_path)
    file_name = dst_path + '/' + 'selected.pdf' if file_name is None else dst_path + '/' + file_name
    util.image_2_pdf(img_to_pdf, save_name=file_name, is_print=True)
    print(f'{len(img_to_pdf)}/{len(selected_data)} images saved into {file_name}')

  return selected_data

# plot multiple indicators on a same chart
def plot_multiple_indicators(df: pd.DataFrame, args: dict = {}, start: Optional[str] = None, end: Optional[str] = None, interval: Literal['day', 'week', 'month', 'year'] = 'day', trade_info: Optional[pd.DataFrame] = None, save_path: Optional[str] = None, save_image: bool = False, show_image: bool = False, title: Optional[str] = None, subplot_args: dict = default_plot_args) -> Optional[plt.Axes]:
  """
  Plot Ichimoku and mean reversion in a same plot
  :param df: dataframe with ichimoku and mean reversion columns
  :param args: dict of args for multiple plots
  :param start: start of the data
  :param end: end of the data
  :param save_path: path where the figure will be saved to, if set to None, then image will not be saved
  :param show_image: whether to display image
  :param title: title of the figure
  :param unit_size: height of each subplot
  :param ws: wide space 
  :param hs: heighe space between subplots
  :param subplot_args: plot args for subplots
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

  # get figure parameters from ta_config
  size_args = args.get('size')
  unit_size = size_args.get('unit_size')
  wspace = size_args.get('wspace')
  hspace = size_args.get('hspace')
  width_factor = size_args.get('width_factor')
  interval_factor = size_args.get('interval_factor')
  
  # dynamically calculate figure size
  start = util.time_2_string(plot_data.index.min())
  end = util.time_2_string(plot_data.index.max())
  width = (util.num_days_between(start, end) / interval_factor[interval]) * width_factor
  width = 10 if width < 10 else width

  # create axes for each indicator
  # fig = mpf.figure(figsize=(width, num_indicators*unit_size))
  fig = plt.figure(figsize=(width, num_indicators*unit_size))  
  gs = gridspec.GridSpec(num_indicators, 1, height_ratios=ratios)
  gs.update(wspace=wspace, hspace=hspace)
  axes = {}
  for i in range(num_indicators):

    tmp_indicator = indicators[i]
    tmp_args = args.get(tmp_indicator)
    
    # put the main_indicators at the bottom, share_x
    if i == 0:
      axes[tmp_indicator] = plt.subplot(gs[i]) 
    else:
      axes[tmp_indicator] = plt.subplot(gs[i], sharex=axes[indicators[0]])
      
    # shows the x_ticklabels on the top at the first ax
    if i != 0: #i%2 == 0: # 
      axes[tmp_indicator].xaxis.set_ticks_position("none")
      plt.setp(axes[tmp_indicator].get_xticklabels(), visible=False)
      axes[tmp_indicator].patch.set_alpha(0.5)
    else:
      axes[tmp_indicator].xaxis.set_ticks_position("top")
      axes[tmp_indicator].patch.set_alpha(0.5)

      # set base
      tmp_data = plot_data[['Close']].copy()
      tmp_data['test'] = 0
      max_idx = tmp_data.index.max()
      extra_days = 7
      for i in range(extra_days):
        tmp_idx = max_idx + datetime.timedelta(days=i+1)
        tmp_data.loc[tmp_idx, 'test'] = 0
      axes[tmp_indicator].plot(tmp_data.index, tmp_data['test'], alpha=0)

    # plot ichimoku with candlesticks
    if tmp_indicator == 'main_indicators':
      # get candlestick color and target indicator from ta_config
      target_indicator = tmp_args.get('target_indicator') if tmp_args.get('target_indicator') is not None else default_main_indicator
      candlestick_color = tmp_args.get('candlestick_color') if tmp_args.get('candlestick_color') is not None else default_candlestick_color
      plot_main_indicators(
        df=plot_data, target_indicator=target_indicator, interval=interval, candlestick_color=candlestick_color,
        use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args)

    # plot aroon
    elif tmp_indicator == 'aroon':
      plot_aroon(df=plot_data, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args)

    # plot adx
    elif tmp_indicator == 'adx':
      plot_adx(df=plot_data, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args, interval=interval)

    # plot volume  
    elif tmp_indicator == 'volume':

      # set bar_width according to data interval
      bar_width = datetime.timedelta(days=1)
      if interval == 'day':
        bar_width = datetime.timedelta(days=1)
      elif interval == 'week':
        bar_width = datetime.timedelta(days=7)
      elif interval == 'month':
        bar_width = datetime.timedelta(days=30)
      elif interval == 'year':
        bar_width = datetime.timedelta(days=365)
      else:
        pass

      # plot_bar(df=plot_data, target_col='Volume', width=bar_width, alpha=0.5, color_mode="up_down", benchmark=None, title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=default_plot_args)
      plot_bar(df=plot_data, target_col='Volume', width=bar_width, alpha=0.4, color_mode="up_down", benchmark=None, edge_color='grey', title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=default_plot_args)
      
    # plot score
    elif tmp_indicator == 'score':
      # set bar_width according to data interval
      bar_width = datetime.timedelta(days=1)
      if interval == 'day':
        bar_width = datetime.timedelta(days=1)
      elif interval == 'week':
        bar_width = datetime.timedelta(days=7)
      elif interval == 'month':
        bar_width = datetime.timedelta(days=30)
      elif interval == 'year':
        bar_width = datetime.timedelta(days=365)
      else:
        pass
      
      # plot_data = cal_change(df=plot_data, target_col='signal_score', add_prefix=True, add_accumulation=False)
      plot_bar(df=plot_data, target_col='signal_score', width=bar_width, alpha=0.4, color_mode="benchmark", edge_color='grey', benchmark=0, title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=default_plot_args)
      # up_idx = plot_data.query('break_score > 0').index
      # down_idx = plot_data.query('break_score < 0').index
      # axes[tmp_indicator].scatter(up_idx, plot_data.loc[up_idx, 'signal_score'], color='green', edgecolor='black', label='signal_score', alpha=0.5, marker='^', zorder=3)
      # axes[tmp_indicator].scatter(down_idx, plot_data.loc[down_idx, 'signal_score'], color='red', edgecolor='black', label='signal_score', alpha=0.5, marker='v', zorder=3)

    # plot renko
    elif tmp_indicator == 'renko':
      plot_renko(plot_data, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=default_plot_args)

    # plot ta signals or candle patterns
    elif tmp_indicator == 'signals':
      
      # get signal_list from ta_config
      signals = tmp_args.get('signal_list')

      # plot one by one
      labels = {' ': '信号', 'trigger': '触发(突破/边界)', 'trend': '趋势(方向/变化)', 'candle': '蜡烛(模式/位置)', 'position': '位置(超买/超卖)'}
      signal_bases = []
      signal_names = []
      if signals is not None:
        for i in range(len(signals)):
          signal_name = signals[i]
          signal_name_split = signal_name.split('_')
          
          if len(signal_name_split) > 1:
            signal_name_split = [x for x in signal_name_split if x not in ['trend', 'signal', 'score']]
            signal_name_label = '_'.join(signal_name_split)
          else:
            signal_name_label = signal_name

          tmp_label = labels.get(signal_name_label)
          signal_name_label = tmp_label if tmp_label is not None else signal_name_label
          signal_names.append(signal_name_label)
          plot_data[f'signal_base_{signal_name}'] = i
          signal_bases.append(i)
          plot_signal(
            df=plot_data, signal_x=signal_name, signal_y=f'signal_base_{signal_name}', interval=interval,
            title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=subplot_args)

      # signal_names = [{'adx':'adx(dst_chg)','overall':'overall(chg_dif)'}[x] if x in ['adx','overall'] else x for x in signal_names ]
      # legend and title
      
      plt.ylim(ymin=min(signal_bases)-1 , ymax=max(signal_bases)+1)
      plt.yticks(signal_bases, signal_names)
      axes[tmp_indicator].legend().set_visible(False)

    # plot other indicators
    else:
      print(f'method for indicator ({tmp_indicator}) not defined')

    # set border color
    spine_alpha = 0.3
    for position in ['top', 'bottom']: # , 'left', 'right'
      if (i in [0, 1, 2, 3] and position in ['top']) or (position in ['bottom'] and i != 4):
        axes[tmp_indicator].spines[position].set_alpha(0)
      else:
        axes[tmp_indicator].spines[position].set_alpha(spine_alpha)

  # adjust plot layout
  max_idx = df.index.max()
  up_down_symbol = {True: '↑', False: '↓'}
  close_rate = (df.loc[max_idx, "rate"]*100).round(2)
  signal_score = df.loc[max_idx, "signal_score"]
  title_color = 'green' if close_rate > 0 else 'red'
  desc_color = 'green' if signal_score > 0 else 'red'
  title_symbol = up_down_symbol[close_rate > 0]
  plt.rcParams['font.sans-serif'] = ['SimHei'] 
  plt.rcParams['axes.unicode_minus'] = False

  # plot trade info
  if trade_info is not None:

    trade_info['date'] = trade_info['updated_time'].apply(lambda x: util.string_plus_day(x, diff_days=-0.54, date_format="%Y-%m-%d %H:%M:%S"))
    trade_info['date'] = trade_info['date'].apply(lambda x: x[:10])
    trade_info = trade_info.query(f'code == "{title}" and date >= "{start}" and date <= "{end}"')

    trade_info = util.df_2_timeseries(trade_info, time_col='date')
    trade_info = pd.merge(trade_info, df[['High', 'Low']], how='left', left_index=True, right_index=True)

    if len(trade_info) > 0:
      buy_data = trade_info.query('trd_side == "BUY"')
      sell_data = trade_info.query('trd_side == "SELL"')

      if len(buy_data) > 0:
        axes['main_indicators'].scatter(buy_data.index, buy_data['High']*1.05, marker='^', color='black', alpha=1, zorder=10)

      if len(sell_data) > 0:
        axes['main_indicators'].scatter(sell_data.index, sell_data['Low']*0.95, marker='v', color='black', alpha=1, zorder=10)

  # get name of the symbol, and trend, trigger, pattern descriptions
  new_title = args['sec_name'].get(title.split('(')[0]) 
  
  # descroption
  before_max_idx = df.index[-2]
  for idx in [max_idx]:

    # signal desc
    signal_score = df.loc[idx, "signal_score"]
    desc_score = df.loc[idx, 'desc_score']
    desc = df.loc[idx, 'signal_description']
    signal_desc_title = (f'[{signal_score}]' + (f' {desc}' if len(desc) > 0 else ''))

    # total_score desc
    total_score = df.loc[idx, "total_score"]
    change = round(df.loc[idx, "total_score"] - df.loc[before_max_idx, "total_score"], 2)
    change_desc = f'+{change}' if change >= 0 else f'{change}'
    total_desc = f'----------------------\n{total_score:<6} ({change_desc:<6})'

    # trend desc
    desc = df.loc[idx, "trend_description"]
    change = round(df.loc[idx, "trend_score"] - df.loc[before_max_idx, "trend_score"], 2)
    change_desc = f'+{change}' if change >= 0 else f'{change}'
    trend_desc = (f' {desc}' if len(desc) > 0 else '') + f' | 趋势 {df.loc[idx, "trend_score"]:<6} ({change_desc:<6})'

    # trigger desc (break_through)
    up_desc = df.loc[idx, "break_up_score"] # break_up_description
    down_desc = df.loc[idx, "break_down_score"]
    up_desc = f'突破({up_desc})' if up_desc > 0 else ''
    down_desc = f'跌落({down_desc})' if down_desc < 0 else ''
    if len(up_desc) > 0 and len(down_desc) > 0:
      desc = f'{up_desc} | {down_desc}'
    else:
      desc = up_desc + down_desc
    change = round(df.loc[idx, "break_score"] - df.loc[before_max_idx, "break_score"], 2)
    change_desc = f'+{change}' if change >= 0 else f'{change}'
    break_desc = (f' {desc}' if len(desc) > 0 else '') + f' | 突破 {df.loc[idx, "break_score"]:<6} ({change_desc:<6})'

    # trigger desc (boundary)
    up_desc = df.loc[idx, "support_score"]
    down_desc = df.loc[idx, "resistant_score"]
    up_desc = f'支撑({up_desc*0.5})' if up_desc > 0 else ''
    down_desc = f'阻挡({down_desc*0.5})' if down_desc < 0 else ''
    if len(up_desc) > 0 and len(down_desc) > 0:
      desc = f'{up_desc} | {down_desc}'
    else:
      desc = up_desc + down_desc
    change = round(df.loc[idx, "boundary_score"] - df.loc[before_max_idx, "boundary_score"], 2)
    change_desc = f'+{change}' if change >= 0 else f'{change}'
    boundary_desc = (f' {desc}' if len(desc) > 0 else '') + f' | 边界 {df.loc[idx, "boundary_score"]*0.5:<6} ({change_desc:<6})'

    # pattern desc
    desc = df.loc[idx, "pattern_description"]
    change = round(df.loc[idx, "pattern_score"] - df.loc[before_max_idx, "pattern_score"], 2)
    change_desc = f'+{change}' if change >= 0 else f'{change}'
    pattern_desc = (f' {desc}' if len(desc) > 0 else '') + f' | 模式 {df.loc[idx, "pattern_score"]:<6} ({change_desc:<6})'
    pattern_desc_title = (f'{desc}' if len(desc) > 0 else '')

    # candle pattern desc
    up_desc = df.loc[idx, "up_pattern_description"]
    up_desc = f'+[{up_desc}]' if len(up_desc) > 0 else up_desc
    down_desc = df.loc[idx, "down_pattern_description"]
    down_desc = f'-[{down_desc}]' if len(down_desc) > 0 else down_desc
    if len(up_desc) > 0 and len(down_desc) > 0:
      desc = f'{up_desc} | {down_desc}'
    else:
      desc = up_desc + down_desc
    change = round(df.loc[idx, "candle_pattern_score"] - df.loc[before_max_idx, "candle_pattern_score"], 2)
    change_desc = f'+{change}' if change >= 0 else f'{change}'
    # candle_pattern_desc = (f' {desc}' if len(desc) > 0 else '') + f' * 蜡烛 {df.loc[idx, "candle_pattern_score"]:<6} ({change_desc:<6})'
    candle_desc_title = (f' {desc}' if len(desc) > 0 else '')

    plt.figtext(0.973, 1.05, f'{pattern_desc}\n{break_desc}\n{boundary_desc}\n{trend_desc}\n{total_desc}', fontsize=16, color='black', ha='right', va='top', bbox=dict(boxstyle="round", fc=desc_color, ec="1.0", alpha=abs(signal_score*0.025)))


  # construct super title
  if new_title is None:
    new_title == ''
  super_title = f' {title}({new_title})  {close_rate}% {title_symbol}'

  # super title description
  connection_symbol = '+' if total_score >=0 else ' '
  score_title = (f'{desc_score}{connection_symbol}{total_score}={signal_desc_title}' if signal_desc_title != '' else '')
  pattern_title = (f'{pattern_desc_title}' if pattern_desc_title != '' else '')
  candle_title = (f'{candle_desc_title}' if candle_desc_title != '' else '')
  fig.suptitle(f'{super_title}\n{score_title}\n{pattern_title}{candle_title}', ha='center', va='top', x=0.5, y=1.05, fontsize=24, bbox=dict(boxstyle="round", fc=title_color, ec="1.0", alpha=0.05), linespacing = 1.8)
  
  # save image
  if save_image and (save_path is not None):
    plt_save_name = save_path + title + '.png'
    plt.savefig(plt_save_name, bbox_inches = 'tight')

  # show image
  if show_image:
    plt.show()

  # close figures
  plt.cla()
  plt.clf()
  plt.close()

# calculate ta indicators, trend and derivatives for historical data
def plot_historical_evolution(df: pd.DataFrame, symbol: str, interval: Literal['day', 'week', 'month', 'year'], config: dict, his_start_date: Optional[str] = None, his_end_date: Optional[str] = None, indicators: list = default_indicators, is_print: bool = False, create_gif: bool = False, plot_final: bool = False, remove_origin: bool = True, plot_save_path: Optional[str] = None) -> pd.DataFrame:
  """
  Calculate selected ta features for dataframe

  :param df: original dataframe with hlocv features
  :param symbol: symbol of the data
  :param interval: interval of the data
  :param his_start_date: start date
  :param his_end_date: end_date
  :param indicators: indicators to calculate
  :param is_print: whether to print current progress
  :param create_gif: whether to create gif from images
  :param plot_final: whether to combine all images to a final one
  :param plot_save_path: path to save plots
  :param remove_origin: whether to remove original images for each day
  :returns: dataframe with ta features, derivatives, signals
  :raises: None
  """
  # copy dataframe
  df = df.copy()

  # check the data date
  df_max_idx = util.time_2_string(df.index.max())
  if df_max_idx < his_end_date:
    print(f'can only evolve to {df_max_idx}')
    his_end_date = df_max_idx
  
  if df is None or len(df) == 0:
    print(f'{symbol}: No data for calculation')
    return None   
  else:
    data_start_date = util.string_plus_day(string=his_start_date, diff_days=-config['calculation']['look_back_window'][interval])
    df = df[data_start_date:]
    plot_start_date = data_start_date

  # summary setting
  if create_gif or plot_final:
    if plot_save_path is None:
      print('Please specify plot save path in parameters, create_gif disable for this time')
      create_gif = False
    else:
      config['visualization']['show_image'] = False
      config['visualization']['save_image'] = True
      images = []
  
  # calculate static data at once
  # calculate dynamic data and signal day by day
  try:
    
    # preprocess sec_data
    phase = 'preprocess'
    df = preprocess(df=df, symbol=symbol)
    
    # calculate TA indicators
    phase = 'cal_ta_basic_features' 
    df = calculate_ta_basic(df=df, indicators=indicators)
    
    # calculate TA derivatives for historical data for period [his_start_date ~ his_end_date]
    phase = 'cal_ta_dynamic_features_and_signals'
    historical_ta_data = pd.DataFrame()
    ed = his_start_date

    # calculate dynamic data
    current_max_idx = None
    while ed <= his_end_date:   

      # calculate sd = ed - interval, get max_idx in df[sd:ed]
      sd = util.string_plus_day(string=ed, diff_days=-config['visualization']['plot_window'][interval])
      tmp_max_idx = df[sd:ed].index.max()
      
      # decide whether to skip current loop (mainly for weekends)
      skip = False
      if current_max_idx is not None and tmp_max_idx <= current_max_idx:
        skip = True

      # update current_max_idx
      if (current_max_idx is None) or (tmp_max_idx > current_max_idx):
        current_max_idx = tmp_max_idx

      # calculation and visualization
      if skip:
        if is_print:
          print(f'{ed} - ({sd} ~ {util.time_2_string(tmp_max_idx)}) - skip')

      else:
        if is_print:
          print(f'{ed} - ({sd} ~ {util.time_2_string(tmp_max_idx)})')
        
        # calculate the dynamic part: linear features
        # calculate TA trend
        phase = 'cal_ta_dynamic_features_day_by_day'
        ta_data = calculate_ta_static(df=df, indicators=indicators)
        ta_data = calculate_ta_dynamic(df=ta_data[sd:ed])
        ta_data = calculate_ta_score(df=ta_data)
        ta_data = calculate_ta_signal(df=ta_data)
        historical_ta_data = pd.concat([historical_ta_data, ta_data.tail(1)])

        # create image for gif
        phase = 'visualization_day_by_day'
        if create_gif:
          visualization(df=ta_data, start=plot_start_date, title=f'{symbol}({ed})', save_path=plot_save_path, visualization_args=config['visualization'])
          images.append(f'{plot_save_path}{symbol}({ed}).png')

      # update ed
      ed = util.string_plus_day(string=ed, diff_days=1)

    # append data
    phase = 'final_data_construction'
    historical_ta_data = pd.concat([ta_data, historical_ta_data])
    df = util.remove_duplicated_index(df=historical_ta_data, keep='last')

    # create gif
    phase = 'plot_gif'
    if create_gif:
      util.image_2_gif(image_list=images, save_name=f'{plot_save_path}{symbol}({his_start_date}-{his_end_date}).gif')

    # remove original images
    phase = 'plot_gif'
    if remove_origin:
      for img in images:
        os.remove(img)

    # if plot final data
    phase = 'final_plot_visualization'
    if plot_final: 
      visualization(df=df, start=plot_start_date, title=f'{symbol}(final)', save_path=plot_save_path, visualization_args=config['visualization'])

  except Exception as e:
    print(symbol, phase, e)

  return df

