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
from numpy.lib.stride_tricks import as_strided

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from mplfinance.original_flavor import candlestick_ohlc
from quant import bc_util as util
from quant import bc_data_io as io_util
try:
  from scipy.signal import find_peaks
except Exception as e:
  print(e)

# default values
default_signal_val = {'pos_signal':'b', 'neg_signal':'s', 'none_signal':'', 'wave_signal': 'n'}
default_candlestick_color = {'colorup':'green', 'colordown':'red', 'alpha':0.8}
default_ohlcv_col = {'close':'Close', 'open':'Open', 'high':'High', 'low':'Low', 'volume':'Volume'}  
default_plot_args = {'figsize':(30, 5), 'title_rotation':'vertical', 'xaxis_position': 'bottom', 'yaxis_position': 'right', 'title_x':-0.01, 'title_y':0.3, 'bbox_to_anchor':(1.02, 0.), 'loc':3, 'ncol':1, 'borderaxespad':0.0}

# ================================================ Load configuration =============================================== # 
# load configuration
def load_config(root_paths):
  """ 
  Load configuration from file

  :param root_paths: a dictionary contains home_path and git_path, differ by platforms
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
  config['config_path'] = config['git_path'] + 'quant/'
  config['quant_path'] = config['home_path'] + 'quant/'

  config['log_path'] = config['quant_path'] + 'logs/'
  config['api_path'] = config['quant_path'] + 'api_key/'
  config['futu_path'] = config['quant_path'] + 'futuopen/'
  config['tiger_path'] = config['quant_path'] + 'tigeropen/'
  config['data_path'] = config['quant_path'] + 'stock_data/'
  config['result_path'] = config['quant_path'] + 'ta_model/'
  
  # load sec lists
  config['selected_sec_list'] = io_util.read_config(file_path=config['config_path'], file_name='selected_sec_list.json')

  # load data-spurce api
  config['api_key'] = io_util.read_config(file_path=config['api_path'], file_name='api_key.json')

  # load calculation and visulization parameters
  ta_config = io_util.read_config(file_path=config['config_path'], file_name='ta_config.json')
  config.update(ta_config)

  return config

# load locally saved data(sec_data, ta_data, results)
def load_data(symbols, config, load_empty_data=False, load_derived_data=False):
  """ 
  Load data from local files
  
  :param config: dictionary of config arguments
  :param load_empty_data: whether to load empyt data dict
  :param update: whether to update local data
  :param required_date: the required date to update
  :param is_print: whether to print update information
  :returns: dictionary of data load from local files
  :raises: None
  """
  # init data
  data = {'sec_data': {}, 'ta_data': {}, 'result': {}, 'final_result': {}}

  # load local data
  if not load_empty_data:

    # load stock data
    for symbol in symbols:
      if os.path.exists(config['data_path']+f'{symbol}.csv'):
        data['sec_data'][f'{symbol}_day'] = io_util.load_stock_data(file_path=config['data_path'], file_name=symbol, standard_columns=True)
      else:
        data['sec_data'][f'{symbol}_day'] = None

    # load derived data
    if load_derived_data:
      file_path = config["quant_path"]
      file_names = config["calculation"]["file_name"]
      for f in file_names.keys():
        file_name = file_names[f]
        if os.path.exists(f'{file_path}{file_name}'):
          data[f] = io_util.pickle_load_data(file_path=file_path, file_name=file_name)
        else:
          print(f'{file_name} not exists')
      
  return data


# ================================================ Core calculation ================================================= # 
# pre-process downloaded stock data
def preprocess_sec_data(df, symbol, print_error=True):
  '''
  Preprocess downloaded data

  :param df: downloaded stock data
  :param symbol: symbol
  :param print_error: whether print error information or not
  :returns: preprocessed dataframe
  :raises: None
  '''
  # if raw data is empty
  if len(df) == 0:
    print(f'No data for preprocessing')
    return None

  # drop duplicated rows, keep the first
  df = util.remove_duplicated_index(df=df, keep='first')

  # adjust close price manually
  adj_rate = 1
  df['split_n1'] = df['Split'].shift(-1).fillna(1.0)
  df['adj_close_p1'] = df['Adj Close'].shift(1)
  df['adj_rate'] = df['adj_close_p1'] / df['Adj Close']
  df = df.sort_index(ascending=False)
  
  for idx, row in df.iterrows():
    df.loc[idx, 'Adj Close'] *= adj_rate
    if row['Split'] != 1:
      if row['adj_rate'] > 2 or row['adj_rate'] < 0.5:
        adj_rate = 1/row['Split']
    elif row['split_n1'] != 1:
      if row['adj_rate'] > 2 or row['adj_rate'] < 0.5:
        adj_rate = 1/row['split_n1']
  df = df.sort_index()
  df.drop(['adj_rate', 'adj_close_p1', 'split_n1'], axis=1, inplace=True)

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

# calculate technical-analysis indicators
def calculate_ta_indicators(df, trend_indicators, volume_indicators, volatility_indicators, other_indicators):

  # copy dataframe
  df = df.copy()
  if df is None or len(df) == 0:
    print(f'No data for calculate_ta_indicator')
    return None  

  # indicator calculation
  try:
    phase = 'calculate close rate' 
    df = cal_change_rate(df=df, target_col=default_ohlcv_col['close'], add_accumulation=False)
    
    phase = 'calculate candlestick' 
    df = add_candlestick_features(df=df)

    phase = 'calculate indicators' 
    all_indicators = []
    all_indicators += [x for x in trend_indicators if x not in all_indicators]
    all_indicators += [x for x in volume_indicators if x not in all_indicators]
    all_indicators += [x for x in volatility_indicators if x not in all_indicators]
    all_indicators += [x for x in other_indicators if x not in all_indicators]

    for indicator in all_indicators:
      to_eval = f'add_{indicator}_features(df=df)'
      df = eval(to_eval)

  except Exception as e:
    print(f'[Exception]: @ {phase} - {indicator}, {e}')

  return df

# calculate trends from ta indicators
def calculate_ta_trend(df, trend_indicators, volume_indicators, volatility_indicators, other_indicators, signal_threshold=0.001):
  """
  Adding derived features such as trend, momentum, etc.

  :param df: dataframe with several ta features
  :param trend_indicators: trend indicators such as ichimoku, kama, aroon, adx, psar, renko, etc
  :param volatility_indicators: volatility indicators bollinger bands
  :param other_indicators: other indicators
  :param signal_threshold: threshold for main indicators trigerments
  :returns: dataframe with extra fetures
  :raises: Exception 
  """
  # if df is empty
  if len(df) == 0:
    print(f'No data for calculate_ta_trend')
    return None
  
  # trend calculation
  try:
    phase = 'calculate trend for trend_indicators'

    # ================================ ichimoku trend =========================
    if 'ichimoku' in trend_indicators:
      signal_col = f'ichimoku_signal'
      trend_col = f'ichimoku_trend'

      for col in ['tankan', 'kijun']:
        df[f'{col}_day'] = sda(series=df[f'{col}_signal'], zero_as=1)
        df = cal_change_rate(df=df, target_col=f'{col}', periods=1, add_accumulation=False, add_prefix=f'{col}', drop_na=False)
        df[f'{col}_rate_ma'] = em(series=df[f'{col}_rate'], periods=5, fillna=False).mean()

      fl = 'tankan'
      sl = 'kijun'
      fld = 'tankan_day'
      sld = 'kijun_day'
      df[trend_col] = ''

      # it is going up when
      ichimoku_up_conditions = {
        'at least 1 triggered': [
          f'(close_to_{fl} >= close_to_{sl} > {signal_threshold})',
          f'(close_to_{sl} >= close_to_{fl} > {signal_threshold})',
          f'((close_to_{fl}>={signal_threshold}) and (close_to_{sl}<={-signal_threshold}) and (abs({fld})<abs({sld})))',
          f'((close_to_{fl}<={-signal_threshold}) and (close_to_{sl}>={signal_threshold}) and (abs({fld})>abs({sld})))',
        ],
        'must all triggered': [
          # f'((tankan_rate_ma > 0) and (kijun_rate_ma > 0))',
          '(Close > 0)'
        ]
      }
      ichimoku_up_query_or = ' or '.join(ichimoku_up_conditions['at least 1 triggered'])
      ichimoku_up_query_and = ' and '.join(ichimoku_up_conditions['must all triggered']) 
      ichimoku_up_query = f'({ichimoku_up_query_and}) and ({ichimoku_up_query_or})'
      up_idx = df.query(f'{ichimoku_up_query}').index
      df.loc[up_idx, trend_col] = 'u'
     
      # it is going down when
      ichimoku_down_conditions = {
        'at least 1 triggered': [
          f'(close_to_{fl} <= close_to_{sl} < {-signal_threshold})',
          f'(close_to_{sl} <= close_to_{fl} < {-signal_threshold})',
          f'((close_to_{sl}<{-signal_threshold}) and (close_to_{fl}>{signal_threshold}) and (abs({fld})>abs({sld})))',
          f'((close_to_{sl}>{signal_threshold}) and (close_to_{fl}<{-signal_threshold}) and (abs({fld})<abs({sld})))',
        ],
        'must all triggered': [
          # f'((tankan_rate_ma < 0) or (kijun_rate_ma < 0))',
          f'(Close<kijun)',
          # 'Close > 0' # when there is no condition
        ],
      }
      ichimoku_down_query_or = ' or '.join(ichimoku_down_conditions['at least 1 triggered'])
      ichimoku_down_query_and = ' and '.join(ichimoku_down_conditions['must all triggered'])
      ichimoku_down_query = f'({ichimoku_down_query_and}) and ({ichimoku_down_query_or})'
      down_idx = df.query(ichimoku_down_query).index
      df.loc[down_idx, trend_col] = 'd'
      
      # it is waving when
      # 1. (-0.01 < kijun_rate_ma < 0.01)
      wave_idx = df.query(f'(({trend_col} != "u") and ({trend_col} != "d")) and ((kijun_rate == 0) and (tankan < kijun))').index
      df.loc[wave_idx, trend_col] = 'n'

      # drop intermediate columns
      df.drop(['tankan_day', 'tankan_rate', 'tankan_rate_ma', 'kijun_day', 'kijun_rate', 'kijun_rate_ma'], axis=1, inplace=True)

    # ================================ aroon trend ============================
    if 'aroon' in trend_indicators:
      aroon_col = ['aroon_up', 'aroon_down', 'aroon_gap']
      df[aroon_col] = df[aroon_col].round(1)
      for col in aroon_col:
        df = cal_change(df=df, target_col=col, add_prefix=True, add_accumulation=True)

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

      # drop intermediate columns
      df.drop(['aroon_up_change', 'aroon_up_acc_change', 'aroon_up_acc_change_count', 'aroon_down_change', 'aroon_down_acc_change', 'aroon_down_acc_change_count', 'aroon_gap_change', 'aroon_gap_acc_change', 'aroon_gap_acc_change_count'], axis=1, inplace=True)

    # ================================ adx trend ==============================
    if 'adx' in trend_indicators:
      conditions = {
        'up': 'adx_diff > 0', 
        'down': 'adx_diff <= 0',
        'none': 'adx < 20'} 
      values = {
        'up': 'u', 
        'down': 'd',
        'none': 'n'}
      df = assign_condition_value(df=df, column='adx_trend', condition_dict=conditions, value_dict=values, default_value='')           

    # ================================ kst trend ==============================
    if 'kst' in trend_indicators:
      conditions = {
        'up': 'kst_diff > 0', 
        'down': 'kst_diff <= 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='kst_trend', condition_dict=conditions, value_dict=values) 

    # ================================ psar trend =============================
    if 'psar' in trend_indicators:
      conditions = {
        'up': 'psar_up > 0', 
        'down': 'psar_down > 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='psar_trend', condition_dict=conditions, value_dict=values, default_value='')

    # =========================================================================

    phase = 'calculate trend for volume_indicators'

    # ================================ eom trend ==============================
    if 'eom' in volume_indicators:
      conditions = {
        'up': 'eom_diff > 0', 
        'down': 'eom_diff <= 0'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='eom_trend', condition_dict=conditions, value_dict=values) 

    # =========================================================================
    
    phase = 'calculate trend for volatility_indicators'

    # ================================ bb trend ===============================
    if 'bb' in volatility_indicators:
      conditions = {
        'up': 'Close < bb_low_band', 
        'down': 'Close > bb_high_band'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='bb_trend', condition_dict=conditions, value_dict=values, default_value='')

    # =========================================================================

    phase = 'calculate trend overall'

    # ================================ overall trend ==========================
    df['trend_idx'] = 0
    df['up_trend_idx'] = 0
    df['down_trend_idx'] = 0

    # specify all indicators and specify the exclusives
    all_indicators = list(set(trend_indicators + volume_indicators + volatility_indicators + other_indicators))
    exclude_indicators = ['bb']
    for indicator in all_indicators:
      trend_col = f'{indicator}_trend'
      signal_col = f'{indicator}_signal'
      day_col = f'{indicator}_day'

      # calculate number of days since trend shifted
      df[day_col] = sda(series=df[trend_col].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1) 
      
      # signal of individual indicators are set to 'n'
      if signal_col not in df.columns:
        df[signal_col] = 'n'

      # skype the exclusive indicators
      if indicator in exclude_indicators:
        continue

      # calculate the overall trend value
      up_idx = df.query(f'{trend_col} == "u"').index
      down_idx = df.query(f'{trend_col} == "d"').index
      df.loc[up_idx, 'up_trend_idx'] += 1
      df.loc[down_idx, 'down_trend_idx'] -= 1
    
    # calculate overall trend index and its moving average
    df['trend_idx'] = df['up_trend_idx'] + df['down_trend_idx']
    
    # calculate direction of overall trend
    df['trend_idx_ma'] = sm(series=df['trend_idx'], periods=5).mean()
    df['pre_trend_idx_ma'] = df['trend_idx_ma'].shift(1)
    conditions = {
      'up': 'trend_idx >=3 and (trend_idx_ma > pre_trend_idx_ma or (trend_idx_ma == pre_trend_idx_ma and trend_idx_ma > 0))', 
      'down': 'trend_idx <=-2 and (trend_idx_ma < pre_trend_idx_ma or (trend_idx_ma == pre_trend_idx_ma and trend_idx_ma < 0))'} 
    values = {
      'up': 1, 
      'down': -1}
    df = assign_condition_value(df=df, column='trend_direction', condition_dict=conditions, value_dict=values, default_value=0) 
    df['trend_direction'] = sda(series=df['trend_direction'], zero_as=1)

  except Exception as e:
    print(f'[Exception]: @ {phase}, {e}')
    
  return df

# technical analyze for ta_data
def calculate_ta_derivatives(df, perspective=['renko', 'linear', 'candle', 'support_resistant', 'overall']):
  """
  analysze support and resistant
  :param df: dataframe with ta indicators
  :param perspective: for which indicators, derivative columns that need to calculated 
  :returns: dataframe with derivative columns
  :raises: None
  """

  # copy dataframe
  df = df.copy()
  if df is None or len(df) == 0:
    print(f'No data for calculate_ta_derivatives')
    return None  
  
  # derivatives calculation
  try:
    phase = 'renko analysis'
    # ================================ renko analysis ============================
    if 'renko' in perspective:
      
      # add renko features
      df = add_renko_features(df=df)

      # calculate renko trend
      conditions = {
        'up': '(candle_color == 1) and ((renko_color == "red" and Low > renko_h) or (renko_color == "green"))', 
        'down': '(renko_color == "red") or (renko_color == "green" and Close < renko_l)'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='renko_trend', condition_dict=conditions, value_dict=values, default_value='n')
      wave_idx = df.query('(renko_trend != "u" and renko_trend != "d") and ((renko_brick_length >= 20 ) and (renko_brick_length>3*renko_duration_p1))').index
      df.loc[wave_idx, 'renko_trend'] = 'n'

      # calculate renko signal
      df['renko_signal'] = 'n'
      df['renko_day'] = sda(series=df['renko_trend'].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1)
      
    phase = 'candle analysis'
    # ================================ candle analysis ===========================
    if 'candle' in perspective:
      
      ma_period = 30
      std_factor = 0.75

      # ============================== fundamental components ====================
      if 'basics' > '':
        
        # tops/bottoms
        df['位置_signal'] = 'n'
        conditions = {
          'top': 'tankan > kijun', 
          'bottom': 'tankan < kijun',
          'middle': '(Close <= tankan and Close >= kijun) or (Close <= kijun and Close >= tankan)' 
          }
        values = {
          'top': 'u', 
          'bottom': 'd',
          'middle': 'n'}
        df = assign_condition_value(df=df, column='位置_trend', condition_dict=conditions, value_dict=values, default_value='')
        df['位置_day'] = df['位置_trend'].replace({'u':1, 'd':-1, 'n':0, '': 0}).fillna(0).astype(int)
        df['位置_day'] = sda(series=df['位置_day'], zero_as=None)

        # large/small volume 
        df['成交量_signal'] = 'n'
        df['volume_ma'] = sm(series=df['Volume'], periods=ma_period).mean()
        df['volume_to_ma'] = (df['Volume'] - df['volume_ma'])/df['volume_ma']

        conditions = {
          'large': 'volume_to_ma > 1',  #  or Volume_rate > 0.3
          'small': 'volume_to_ma < -0.5'} #  or Volume_rate < -0.3
        values = {
          'large': 'u', 
          'small': 'd'}
        df = assign_condition_value(df=df, column='成交量_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # entity/shadow to close rate
        df['candle_entity_to_close'] = df['candle_entity'] / df['Close']
        df['candle_shadow_to_close'] = df['candle_shadow'] / df['Close']
        df['candle_shadow_pct_diff'] = df['candle_upper_shadow_pct'] - df['candle_lower_shadow_pct']

      if 'shadow_entity' > '':
        
        # long/short entity
        df['entity_signal'] = 'n'
        df['entity_ma'] = sm(series=df['candle_entity'], periods=ma_period).mean()
        df['entity_std'] = sm(series=df['candle_entity'], periods=ma_period).std()
        df['entity_diff'] = (df['candle_entity'] - df['entity_ma'])/df['entity_std']
        conditions = {
          'long': f'entity_diff >= {std_factor}', 
          'short': f'entity_diff <= {-std_factor}'} 
        values = {
          'long': 'u', 
          'short': 'd'}
        df = assign_condition_value(df=df, column='entity_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # long/short shadow
        df['shadow_signal'] = 'n'
        df['shadow_ma'] = sm(series=df['candle_shadow'], periods=ma_period).mean()
        df['shadow_std'] = sm(series=df['candle_shadow'], periods=ma_period).std()
        df['shadow_diff'] = (df['candle_shadow'] - df['shadow_ma'])/df['shadow_std']
        conditions = {
          'long': f'shadow_diff >= {std_factor}', 
          'short': f'shadow_diff <= {-std_factor}'}
        values = {
          'long': 'u', 
          'short': 'd'}
        df = assign_condition_value(df=df, column='shadow_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # upper shadow
        df['upper_shadow_signal'] = 'n'
        conditions = {
          'long': '(candle_upper_shadow_pct > 0.3)', 
          'short': '(candle_upper_shadow_pct < 0.1)'}
        values = {
          'long': 'u', 
          'short': 'd'}
        df = assign_condition_value(df=df, column='upper_shadow_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # lower shadow
        df['lower_shadow_signal'] = 'n'
        conditions = {
          'long': '(candle_lower_shadow_pct > 0.3)', 
          'short': '(candle_lower_shadow_pct < 0.1)'}
        values = {
          'long': 'u', 
          'short': 'd'}
        df = assign_condition_value(df=df, column='lower_shadow_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # cross
        conditions = {
          'cross': '(entity_trend == "d" and candle_entity_pct <= 0.1 and shadow_trend == "d")',
          'spindle': '(entity_trend == "d" and candle_entity_pct <= 0.25 and shadow_trend == "n")',
          'highwave': '(entity_trend == "d" and candle_entity_pct <= 0.25 and shadow_trend == "u")'}
        values = {
          'cross': 'd', 
          'spindle': 'n',
          'highwave': 'u'}
        df = assign_condition_value(df=df, column='十字星', condition_dict=conditions, value_dict=values, default_value='')

        # hammer/meteor
        conditions = {
          'hammer': '(entity_trend != "u" and candle_entity_pct <= 0.25) and (shadow_trend == "u" and upper_shadow_trend == "d" and candle_lower_shadow_pct >= 0.5)',
          'meteor': '(entity_trend != "u" and candle_entity_pct <= 0.25) and (shadow_trend == "u" and lower_shadow_trend == "d" and candle_upper_shadow_pct >= 0.5)'}
        values = {
          'hammer': 'u', 
          'meteor': 'd'}
        df = assign_condition_value(df=df, column='锤子', condition_dict=conditions, value_dict=values, default_value='n')

      if 'windows' > '':
        
        # candle entity middle
        df['candle_entity_middle'] = (df['candle_entity_top'] + df['candle_entity_bottom']) * 0.5
        
        # window(gap)
        df['窗口_signal'] = 'n'
        conditions = {
          'up': 'candle_gap > 1', 
          'down': 'candle_gap < -1'}
        values = {
          'up': 'u', 
          'down': 'd'}
        df = assign_condition_value(df=df, column='窗口_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # window position days (days beyond/below window)
        conditions = {
          'up': '((candle_entity_bottom >= candle_gap_top) or (candle_entity_top > candle_gap_top and candle_entity_bottom < candle_gap_bottom and candle_color == 1))',
          'down': '((candle_entity_top <= candle_gap_bottom) or (candle_entity_top > candle_gap_top and candle_entity_bottom < candle_gap_bottom and candle_color ==-1))'}
        values = {
          'up': 1, 
          'down': -1}
        df = assign_condition_value(df=df, column='window_position_days', condition_dict=conditions, value_dict=values, default_value=0)
        df['window_position_days'] = sda(series=df['window_position_days'], zero_as=1)
        df['previous_window_position_days'] = df['window_position_days'].shift(1)

        # window position status (beyond/below/among window)
        conditions = {
          'up': '(candle_entity_bottom >= candle_gap_top)',
          'mid_up': '((candle_entity_top > candle_gap_top) and (candle_gap_top > candle_entity_bottom >= candle_gap_bottom))',
          'mid': '((candle_entity_top <= candle_gap_top) and (candle_entity_bottom >= candle_gap_bottom))',
          'out': '((candle_entity_top > candle_gap_top) and (candle_entity_bottom < candle_gap_bottom))',
          'mid_down': '((candle_entity_bottom < candle_gap_bottom) and (candle_gap_top >= candle_entity_top > candle_gap_bottom))',
          'down': '(candle_entity_top <= candle_gap_bottom)'}
        values = {
          'up': 'up', 
          'mid_up': 'mid_up',
          'mid': 'mid',
          'out': 'out',
          'mid_down': 'mid_down',
          'down': 'down'}
        df = assign_condition_value(df=df, column='window_position_status', condition_dict=conditions, value_dict=values, default_value='')#, default_value=0)
        df['previous_window_position_status'] = df['window_position_status'].shift(1)
        df['previous_candle_color'] = df['candle_color'].shift(1)

        # rebound or hitpeak
        df['反弹_signal'] = ''
        conditions = {
          'rebound': '(candle_gap != 2 and window_position_days > 2) and ((window_position_status == "up") and (previous_window_position_status == "mid_up" or previous_window_position_status == "mid" or ((previous_window_position_status == "mid_down" or previous_window_position_status == "out")and previous_candle_color == 1)))', 
          'hitpeak': '(candle_gap !=-2 and window_position_days <-2) and ((window_position_status == "down") and (previous_window_position_status == "mid_down" or previous_window_position_status == "mid" or ((previous_window_position_status == "mid_up" or previous_window_position_status == "out") and previous_candle_color == -1)))'} 
        values = {
          'rebound': 'u',
          'hitpeak': 'd'}
        df = assign_condition_value(df=df, column='反弹_trend', condition_dict=conditions, value_dict=values, default_value='n')
        
        # break through up or down
        df['突破_signal'] = ''
        conditions = {
          'break_up': '((candle_gap != 2 and previous_window_position_days < 0) and ((candle_color == 1 and (window_position_status == "out" or (entity_trend != "d" and candle_entity_middle > candle_gap_top))) or (window_position_status == "up")))',
          'break_down': '((candle_gap != -2 and previous_window_position_days > 0) and ((candle_color ==-1 and window_position_status == "out" or (entity_trend != "d" and candle_entity_middle < candle_gap_bottom)) or (window_position_status == "down")))'}
        values = {
          'break_up': 'u', 
          'break_down': 'd'}
        df = assign_condition_value(df=df, column='突破_trend', condition_dict=conditions, value_dict=values, default_value='n')
        df['next_突破_trend'] = df['突破_trend'].shift(-1)
        redundant_idx = df.query('(突破_trend == "u" and next_突破_trend == "u") or (突破_trend == "d" and next_突破_trend == "d")').index
        df.loc[redundant_idx, '突破_trend'] = 'n'
        
      # ============================== 1 candle pattern  =========================
      if '1_candle' > '':
        
        # belt
        df['腰带_signal'] = 'n'
        conditions = {
          'up': '(位置_trend == "d") and (candle_lower_shadow_pct <= 0.05) and (entity_trend == "u") and (candle_color == 1)',
          'down': '(位置_trend == "u") and (candle_upper_shadow_pct <= 0.05) and (entity_trend == "u") and (candle_color == -1)'}
        values = {
          'up': 'u',
          'down': 'd'}
        df = assign_condition_value(df=df, column='腰带_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # cross/spindle/highwave
        df['十字星_signal'] = 'n'
        conditions = {
          'cross': '(位置_trend != "n") and (十字星 == "d")',
          'highwave': '(位置_trend != "n") and (十字星 == "u")'}
        values = {
          'cross': 'd', 
          'highwave': 'u'}
        df = assign_condition_value(df=df, column='十字星_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # hammer
        df['锤子_signal'] = 'n'
        conditions = {
          'hammer': '((位置_trend == "d" and 位置_day < -5) and (锤子 == "u"))', 
          'hanging': '((位置_trend == "u" and 位置_day > 5) and (锤子 == "u"))'}
        values = {
          'hammer': 'u', 
          'hanging': 'd'}
        df = assign_condition_value(df=df, column='锤子_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # meteor
        df['流星_signal'] = 'n'
        conditions = {
          'meteor': '((位置_trend == "u" and 位置_day > 5) and (锤子 == "d"))',
          'reverse_hammer': '((位置_trend == "d" and 位置_day < -5) and (锤子 == "d"))'}
        values = {
          'meteor': 'd',
          'reverse_hammer': 'u'}
        df = assign_condition_value(df=df, column='流星_trend', condition_dict=conditions, value_dict=values, default_value='n')

      # ============================== 2+ candle pattern =========================  
      if 'multi_candle' > '':

        # initialize multi-candle pattern trend
        idxs = df.index.tolist()
        for t in ['平头', '穿刺', '吞噬', '包孕', '启明黄昏']:
          df[f'{t}_trend'] = 'n'
          # df[f'{t}_signal'] = np.nan

        df['previous_entity_top'] = df['candle_entity_top'].shift(1)
        df['previous_entity_bottom'] = df['candle_entity_bottom'].shift(1)
        df['previous_candle_color'] = df['candle_color'].shift(1)
        df['top_diff'] = abs(df['candle_entity_top'] - df['previous_entity_top'])/df['Close']
        df['bottom_diff'] = abs(df['candle_entity_bottom'] - df['previous_entity_bottom'])/df['Close']

        # flat top/bottom 
        df['平头_signal'] = 'n'
        conditions = {
          'top': '(位置_trend == "u" and top_diff <= 0.001 and entity_trend != "d" and (candle_color == 1 and previous_candle_color == 1))', 
          'bottom': '(位置_trend == "d" and bottom_diff <= 0.001 and entity_trend != "d" and (candle_color == -1 and previous_candle_color == -1))'}
        values = {
          'top': 'd', 
          'bottom': 'u'}
        df = assign_condition_value(df=df, column='平头_trend', condition_dict=conditions, value_dict=values, default_value='n')

        # iterate through dataframe by window_size of 2 and 3
        previous_row = None
        previous_previous_row = None
        for idx in idxs:
          
          # current row(3), previous row(2) and previous previous row(1)
          row = df.loc[idx]

          previous_i = idxs.index(idx) - 1
          if previous_i < 0:
            continue
          else:
            previous_idx = idxs[previous_i]
            previous_row = df.loc[previous_idx]

          previous_previous_i = previous_i - 1
          if previous_previous_i < 0:
            previous_previous_row = None
          else:
            previous_previous_idx = idxs[previous_previous_i]
            previous_previous_row = df.loc[previous_previous_idx]

          # 当前蜡烛为长实体
          if row['entity_trend'] != 'u':
            pass
          else:
            # 吞噬形态: 2-底部 < 1-底部 & 2-顶部 > 1-顶部
            if (previous_row['candle_entity_top'] < row['candle_entity_top']) and (previous_row['candle_entity_bottom'] > row['candle_entity_bottom']):
            
              # 位于顶部
              if (row['位置_trend'] == "u"):
                # 空头吞噬: 1-绿, 2-红
                if (previous_row['candle_color'] == 1 and row['candle_color'] == -1):
                  df.loc[idx, '吞噬_trend'] = 'd'
                # 最后顶吞噬: 1-绿, 2-绿, 都在顶部
                elif (previous_row['candle_color'] == 1 and row['candle_color'] == 1) and (previous_row['位置_trend'] == "u"):
                  df.loc[idx, '吞噬_trend'] = 'd'

              # 位于底部
              elif (row['位置_trend'] == "d"):
                # 多头吞噬: 位于底部, 1-红, 2-绿
                if (previous_row['candle_color'] == -1 and row['candle_color'] == 1):
                  df.loc[idx, '吞噬_trend'] = 'u'
                # 最后底吞噬: 1-红, 2-红, 都在底部
                elif (previous_row['candle_color'] == -1 and row['candle_color'] == -1) and (previous_row['位置_trend'] == "d"):
                  df.loc[idx, '吞噬_trend'] = 'u'  

          # 前一蜡烛为长实体 
          if previous_row['entity_trend'] != 'u':
            pass
          else:
            # 包孕形态: 2-底部 > 1-底部 & 2-顶部 < 1-顶部
            if (previous_row['candle_entity_top'] > row['candle_entity_top']) and (previous_row['candle_entity_bottom'] < row['candle_entity_bottom']):
              
              # 位于顶部
              if row['位置_trend'] == 'u':
                # 空头包孕: 1-绿, 2-红
                if (previous_row['candle_color'] == 1 and row['candle_color'] == -1):
                  df.loc[idx, '包孕_trend'] = 'd'

              # 位于底部
              elif row['位置_trend'] == 'd':
                # 多头包孕: 1-红, 2-绿
                if (previous_row['candle_color'] == -1 and row['candle_color'] == 1):
                  df.loc[idx, '包孕_trend'] = 'u'
          
          # 顶部:乌云盖顶, 黄昏星
          if (previous_row['位置_trend'] == "u"):

            # =================================== 乌云盖顶  =================================== #
            # 1-必须为绿色, 2-必须为红色长实体
            if (row['entity_trend'] != 'u' or row['candle_color'] != -1 or previous_row['candle_color'] != 1):
              pass
            else:
              # 顶部>前顶部, 底部>前底部, 底部穿过前中点
              if (previous_row['candle_entity_top'] < row['candle_entity_top']) and (previous_row['candle_entity_bottom'] < row['candle_entity_bottom']) and (previous_row['candle_entity_middle'] > row['candle_entity_bottom']):
                df.loc[idx, '穿刺_trend'] = 'd'

            # =================================== 黄昏星  ===================================== #
            if previous_previous_row is None:
              pass
            else:
              # 1-绿色非小实体, 2-高位, 3-红色非小实体
              if (previous_row['位置_trend'] == "u") and (previous_previous_row['candle_color'] == 1 and previous_previous_row['entity_trend'] != 'd') and (row['candle_color'] == -1 and row['entity_trend'] != 'd'):
                # 3-长实体 或 3-Low < 1-Low 或 3-top < 1-bottom
                if row['entity_trend'] == 'u' or (row['Low'] < previous_previous_row['Low']) or (row['candle_entity_top'] < previous_previous_row['candle_entity_bottom']):
                  # 2-小实体, 2-底部 > 1/3-顶部
                  if (previous_row['entity_trend'] == 'd') and (previous_row['candle_entity_bottom'] > previous_previous_row['candle_entity_top']) and (previous_row['candle_entity_bottom'] > row['candle_entity_top']): #(previous_row['High'] > previous_previous_row['High']) and (previous_row['High'] > row['High']):
                    df.loc[idx, '启明黄昏_trend'] = 'd'
                  # # 2-非小实体, 2-底部 > 1/3-顶部
                  # elif (previous_row['entity_trend'] == 'n') and (previous_row['candle_entity_bottom'] > previous_previous_row['candle_entity_top']) and (previous_row['candle_entity_bottom'] > row['candle_entity_top']):
                  #     df.loc[idx, '启明黄昏_trend'] = 'd'

          # 底部:穿刺形态, 启明星
          elif (previous_row['位置_trend'] == "d"):

            # =================================== 穿刺形态  =================================== #
            # 1-必须为红色, 2-必须为绿色长实体
            if (row['entity_trend'] != 'u' or row['candle_color'] != 1 or previous_row['candle_color'] != -1):
              pass
            else:
              # 顶部<=前顶部, 底部<前底部, 顶部穿过前中点
              if (previous_row['candle_entity_top'] >= row['candle_entity_top']) and (previous_row['candle_entity_bottom'] > row['candle_entity_bottom']) and (previous_row['candle_entity_middle'] < row['candle_entity_top']):
                df.loc[idx, '穿刺_trend'] = 'u'
            
            # =================================== 启明星  ===================================== #
            if previous_previous_row is None:
              pass
            else:
              # 1-红色非小实体, 2-高位, 3-绿色非小实体
              if (previous_row['位置_trend'] == "d") and (previous_previous_row['candle_color'] == -1 and previous_previous_row['entity_trend'] != 'd') and (row['candle_color'] == 1 and row['entity_trend'] != 'd'):
                # 3-长实体 或 3-High > 1-High 或 3-bottom > 1-top
                if row['entity_trend'] == 'u' or (row['High'] > previous_previous_row['High']) or (row['candle_entity_bottom'] > previous_previous_row['candle_entity_top']):
                  # 2-小实体, 2-顶部 < 1/3-底部
                  if (previous_row['entity_trend'] == 'd') and (previous_row['candle_entity_top'] < previous_previous_row['candle_entity_bottom']) and (previous_row['candle_entity_top'] < row['candle_entity_bottom']): #(previous_row['Low'] < previous_previous_row['Low']) and (previous_row['Low'] < row['Low']):
                    df.loc[idx, '启明黄昏_trend'] = 'u'
                  # # 2-非小实体, 2-顶部 < 1/3-底部
                  # elif (previous_row['entity_trend'] == 'n') and (previous_row['candle_entity_top'] < previous_previous_row['candle_entity_bottom']) and (previous_row['candle_entity_top'] < row['candle_entity_bottom']):
                  #     df.loc[idx, '启明黄昏_trend'] = 'u'

        # multi-candle signals
        for t in ['穿刺', '吞噬', '包孕', '启明黄昏']:
          df[f'{t}_signal'] = df[f'{t}_trend']

      # ============================== overall results  ==========================
      if 'candle pattern description and index' > '':  

        # candle pattern description
        pattern_info = {
          '窗口_trend': {'u': '上升窗口', 'd': '下降窗口'},
          '突破_trend': {'u': '突破窗口阻挡', 'd': '跌落窗口支撑'},
          '反弹_trend': {'u': '触底反弹', 'd': '触顶回落'},
          '锤子_trend': {'u': '锤子线', 'd': '流星线'},
          '十字星_trend': {'u': '高浪线', 'd': '十字星'},
          '平头_trend': {'u': '平头顶', 'd': '平头底'},
          '穿刺_trend': {'u': '穿刺', 'd': '乌云盖顶'},
          '吞噬_trend': {'u': '多头吞噬', 'd': '空头吞噬'},
          '包孕_trend': {'u': '多头包孕', 'd': '空头包孕'},
          '启明黄昏_trend': {'u': '启明星', 'd': '黄昏星'},
        }

        # candle pattern weight
        pattern_weight = {
          '窗口_trend': {'u': 2, 'd': -2},
          '突破_trend': {'u': 2, 'd': -2},
          '反弹_trend': {'u': 2, 'd': -2},
          '锤子_trend': {'u': 0.5, 'd': -0.5},
          '十字星_trend': {'u': 0, 'd': 0},
          '平头_trend': {'u': 0.5, 'd': -0.5},
          '穿刺_trend': {'u': 1, 'd': -1},
          '吞噬_trend': {'u': 0.5, 'd': -0.5},
          '包孕_trend': {'u': 0.5, 'd': -0.5},
          '启明黄昏_trend': {'u': 2, 'd': -2},
        }
        
        # concate descriptions
        df['candle_pattern_idx'] = 0
        df['candle_pattern_description'] = ''

        target_df = df.query('窗口_trend != "n" or 突破_trend != "n" or 反弹_trend != "n" or 锤子_trend != "n" or 十字星_trend != "n" or 穿刺_trend != "n" or 吞噬_trend != "n" or 包孕_trend != "n" or 启明黄昏_trend != "n"')
        for index, row in target_df.iterrows():
          candle_patterns = pattern_info.keys()
          for t in candle_patterns:
            if t in df.columns:
              tmp_trend = df.loc[index, t]
              if tmp_trend is not None:
                tmp_info = pattern_info[t].get(tmp_trend)
                tmp_weight = pattern_weight[t].get(tmp_trend)
                if tmp_info is not None:
                  df.loc[index, 'candle_pattern_description'] += f'/{tmp_info}'
                if tmp_weight is not None:
                  df.loc[index, 'candle_pattern_idx'] += tmp_weight
        df.loc[index, 'candle_pattern_description'] = df.loc[index, 'candle_pattern_description'][1:]
        empty_description_idx = df.query('candle_pattern_description == ""').index
        df.loc[empty_description_idx, 'candle_pattern_description'] = ' '

      # drop unnecessary columns
      for col in [
        # '位置_signal', '成交量_signal', 
        # '腰带_signal', '十字星_signal', '平头_signal', '包孕_signal', '吞噬_signal', '锤子_signal', '流星_signal', 
        # '启明黄昏_signal', '突破_signal', '窗口_signal', '穿刺_signal', 'window_position_status', 
        'window_position_days', 'previous_window_position_days', 'previous_window_position_status', 'previous_candle_color', 'next_突破_trend',
        # 'entity_signal', 'shadow_signal', 'upper_shadow_signal', 'lower_shadow_signal', 
        'candle_entity_to_close', 'candle_shadow_to_close', 'candle_shadow_pct_diff', 'candle_entity_middle'
        'previous_high', 'previous_low', 'high_diff', 'low_diff'
        ]:
        if col in df.columns:
          df.drop(col, axis=1, inplace=True)

    phase = 'linear analysis'
    # ================================ linear analysis ===========================
    if 'linear' in perspective:

      # add linear features
      df = add_linear_features(df=df)

      # crossover signals between Close and linear_fit_high/linear_fit_low, # , 'support', 'resistant'
      for col in ['linear_fit_high', 'linear_fit_low']:
        signal_col = None

        if col in df.columns:
          signal_col = f'{col}_signal'
          df[signal_col] = cal_crossover_signal(df=df, fast_line='Close', slow_line=col, pos_signal=1, neg_signal=-1, none_signal=0)

          if signal_col in df.columns:
            df[signal_col] = sda(series=df[signal_col], zero_as=1)
        else:
          print(f'{col} not in df.columns')
      
      # trends from linear fit 
      # hitpeak or rebound
      conditions = {
        'up': '((candle_color == 1 and linear_fit_low_stop > 1 and linear_fit_support < Close) and ((candle_color_day >= 3) or (candle_color == 1 and shadow_trend == "u" and lower_shadow_trend == "u") or (吞噬_trend == "u" or 穿刺_trend == "u") or (tankan_kijun_signal > 0 or Close > tankan) or (candle_entity_bottom > linear_fit_high)))', 
        'down': '((candle_color == -1 and linear_fit_high_stop > 1 and linear_fit_resistant > Close) and ((candle_color_day <= -3) or (candle_color == -1 and shadow_trend == "u" and upper_shadow_trend == "u") or (吞噬_trend == "d" or 穿刺_trend == "d") or (tankan_kijun_signal < 0 or Close < tankan) or (candle_entity_top < linear_fit_low)))'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='linear_bounce_trend', condition_dict=conditions, value_dict=values, default_value='')
      df['linear_bounce_day'] = sda(series=df['linear_bounce_trend'].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1)
      df = remove_redundant_signal(df=df, signal_col='linear_bounce_trend', pos_signal='u', neg_signal='d', none_signal='', keep='first')

      # break through up or down
      conditions = {
        'up': '((linear_fit_high_stop > 0 and linear_fit_resistant > 0 and linear_fit_high_signal > 1))', 
        'down': '((linear_fit_low_stop > 0 and linear_fit_resistant > 0 and linear_fit_low_signal < -1))'} 
      values = {
        'up': 'u', 
        'down': 'd'}
      df = assign_condition_value(df=df, column='linear_break_trend', condition_dict=conditions, value_dict=values, default_value='')
      df['linear_break_day'] = sda(series=df['linear_break_trend'].replace({'': 0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1)

      # linear fit description and score
      df['linear_fit_description'] = ''
      df['linear_fit_idx'] = 0

      conditions = {
        'rebound': 'linear_bounce_trend == "u"', 
        'hitpeak': 'linear_bounce_trend == "d"',
        'break_up': 'linear_break_trend == "u"',
        'break_down': 'linear_break_trend == "d"'} 
      trend_values = {
        'rebound': 'u', 
        'hitpeak': 'd',
        'break_up': 'u',
        'break_down': 'd'} 
      dscription_values = {
        'rebound': '触底反弹', 
        'hitpeak': '触顶回落',
        'break_up': '突破阻挡',
        'break_down': '跌落支撑'}
      idx_values = {
        'rebound': 1, 
        'hitpeak': -1,
        'break_up': 2,
        'break_down': -2}
      df = assign_condition_value(df=df, column='拟合_trend', condition_dict=conditions, value_dict=trend_values, default_value=' ')
      df = assign_condition_value(df=df, column='linear_fit_description', condition_dict=conditions, value_dict=dscription_values, default_value=' ')
      df = assign_condition_value(df=df, column='linear_fit_idx', condition_dict=conditions, value_dict=idx_values, default_value=0)
      
      valid_idxs = df.query('linear_slope == linear_slope and 拟合_trend == ""').index
      df.loc[valid_idxs, '拟合_trend'] = 'n'
      df['拟合_signal'] = 'n' #df['拟合_trend'].replace({'d': 's', 'u': 'b', 'n': ''})

    phase = 'support and resistant'
    # ================================ support and resistant =====================
    if 'support_resistant' in perspective:

      # focus on the last row only
      max_idx = df.index.max()
      valid_idxs = df.query('linear_slope == linear_slope').index

      # calculate support and resistant from renko, linear_fit and candle_gap
      support_candidates = {'linear': df.loc[max_idx, 'linear_fit_support'], 'candle': df.loc[max_idx, 'candle_gap_support']}
      resistant_candidates = {'linear': df.loc[max_idx, 'linear_fit_resistant'], 'candle': df.loc[max_idx, 'candle_gap_resistant']}

      # support
      to_pop = []
      for k in support_candidates.keys():
        if np.isnan(support_candidates[k]):
          to_pop += [k]
      for k in to_pop:
        support_candidates.pop(k)

      supporter = ''
      support = np.nan      
      if len(support_candidates) > 0:
        supporter = max(support_candidates, key=support_candidates.get)
        support = support_candidates[supporter]
      df.loc[valid_idxs, 'support'] = support
      df.loc[valid_idxs, 'supporter'] = supporter

      # resistant
      to_pop = []
      for k in resistant_candidates.keys():
        if np.isnan(resistant_candidates[k]):
          to_pop += [k]
      for k in to_pop:
        resistant_candidates.pop(k)

      resistanter = ''
      resistant = np.nan
      if len(resistant_candidates) > 0:
        resistanter = min(resistant_candidates, key=resistant_candidates.get)
        resistant = resistant_candidates[resistanter]
      df.loc[valid_idxs, 'resistant'] = resistant
      df.loc[valid_idxs, 'resistanter'] = resistanter

      # crossover signals between Close and linear_fit_high/linear_fit_low, # , 'support', 'resistant'
      for col in ['support', 'resistant']:
        signal_col = None

        if col in df.columns:
          signal_col = f'{col}_signal'
          df[signal_col] = cal_crossover_signal(df=df, fast_line='Close', slow_line=col, pos_signal=1, neg_signal=-1, none_signal=0)

          if signal_col in df.columns:
            df[signal_col] = sda(series=df[signal_col], zero_as=1)
        else:
          print(f'{col} not in df.columns')

    phase = 'overall'
    # ================================ overall description =======================
    if 'overall' in perspective:

      # days since signal triggered
      for col in ['窗口', '突破', '反弹', '启明黄昏', '穿刺', '包孕', '吞噬', '平头', '十字星', '锤子', '流星', '腰带']:
        df[f'{col}_day'] = df[f'{col}_trend'].replace({'u':1, 'd':-1, 'n':0, '': 0}).fillna(0).astype(int)

      # iterate through df
      previous_idx = None
      for index, row in df.iterrows():
        if previous_idx is not None:
          for col in ['窗口', '突破', '反弹', '启明黄昏', '穿刺', '包孕', '吞噬', '平头', '十字星', '锤子', '流星', '腰带']:
            
            # get current day and previous day
            day_col = f'{col}_day'
            current_day = row[day_col]
            previous_day = df.loc[previous_idx, day_col]
 
            # only operate when current day is 0
            if current_day == 0 and previous_day >= 1:
              df.loc[index, day_col] = previous_day + 1
            elif current_day == 0 and previous_day <= -1:
              df.loc[index, day_col] = previous_day - 1
            elif current_day == 1 and previous_day == -1:
              df.loc[index, day_col] = 0
            else:
              pass
        else:
          pass

        # update previous index
        previous_idx = index

      # for col in ['窗口', '突破', '反弹', '启明黄昏']: # '十字星', '锤子', '流星', '穿刺', '包孕', '吞噬', '平头'
      #   valid_up_idx = df.query(f'(0 < {col}_day <= 3)').index
      #   valid_down_idx = df.query(f'(0 > {col}_day >= -3)').index
      #   valid_wave_idx = df.query(f'({col}_day == 0)').index
      #   df.loc[valid_up_idx, f'{col}_trend'] = 'u'
      #   df.loc[valid_down_idx, f'{col}_trend'] = 'd'    
      #   df.loc[valid_wave_idx, f'{col}_trend'] = 'n'      

      # ================================ linear analysis ========================
      row = df.loc[max_idx,].copy()
      period_threhold = 5
      df['category'] = ''
      df['description'] = ''

      for col in ['tankan', 'kijun']:
        signal_col = f'{col}_signal'
        if signal_col in df.columns:
          df[signal_col] = sda(series=df[signal_col], zero_as=1)
        else:
          print(f'{signal_col} not in df.columns')

      # define conditions
      conditions = {

        # tankan-kijun(ichimoku信号)
        'T/K': (row['tankan_kijun_signal'] > 0 and row['tankan_kijun_signal'] <= 10) or (row['tankan_kijun_signal'] < 0 and row['tankan_kijun_signal'] >= -10),

        # linear trend
        '强势': (row['linear_slope'] >= 0.1 or row['linear_slope'] <= -0.1) and (row['linear_fit_high_slope'] * row['linear_fit_low_slope'] > 0),
        '弱势': (row['linear_slope'] >-0.1 and row['linear_slope'] < 0.1 ) or ((row['linear_fit_high_slope'] * row['linear_fit_low_slope']) < 0),
        '上行': (row['linear_fit_high_slope'] > 0 and row['linear_fit_low_slope'] >= 0) or (row['linear_fit_high_slope'] >= 0 and row['linear_fit_low_slope'] > 0),
        '下行': (row['linear_fit_high_slope'] < 0 and row['linear_fit_low_slope'] <= 0) or (row['linear_fit_high_slope'] <= 0 and row['linear_fit_low_slope'] < 0),
        '波动': (row['linear_fit_high_slope'] * row['linear_fit_low_slope'] < 0) or (row['linear_fit_high_slope']==0 and row['linear_fit_low_slope']==0),
        '轨道中': (row['linear_fit_high'] >= row['Close']) and (row['linear_fit_low'] <= row['Close']),
        '轨道上方': (row['linear_fit_high'] < row['Close']) and (row['linear_fit_low'] < row['Close']),
        '轨道下方': (row['linear_fit_high'] > row['Close']) and (row['linear_fit_low'] > row['Close']),

        # linear support and resistant
        '跌破支撑': (row['linear_break_trend'] == "d"),
        '突破阻挡': (row['linear_break_trend'] == "u"), 
        '触顶回落': (row['linear_bounce_trend'] == "d"),
        '触底反弹': (row['linear_bounce_trend'] == "u"), 

        # technical indicators
        '上穿快线': (row['tankan_signal'] > 0 and row['tankan_signal'] < period_threhold),
        '上穿慢线': (row['kijun_signal'] > 0 and row['kijun_signal'] < period_threhold),
        '上穿底部': (row['linear_fit_low_signal'] > 0 and row['linear_fit_low_signal'] < period_threhold),
        '上穿顶部': (row['linear_fit_high_signal'] > 0 and row['linear_fit_high_signal'] < period_threhold),
        '下穿快线': (row['tankan_signal'] < 0 and row['tankan_signal'] > -period_threhold),
        '下穿慢线': (row['kijun_signal'] < 0 and row['kijun_signal'] > -period_threhold),
        '下穿底部': (row['linear_fit_low_signal'] < 0 and row['linear_fit_low_signal'] > -period_threhold),
        '下穿顶部': (row['linear_fit_high_signal'] < 0 and row['linear_fit_high_signal'] > -period_threhold)
      }

      # initialize empty dict
      classification = []

      # breakthrough resistant
      if conditions['强势'] and conditions['上行'] and conditions['突破阻挡']:
        classification.append('up_x_resistant')

      # fall below support
      if conditions['强势'] and conditions['下行'] and conditions['跌破支撑']:
        classification.append('down_x_support')

      # rebound
      if conditions['下行'] and conditions['触底反弹'] and conditions['上穿快线'] and (conditions['上穿顶部'] or conditions['上穿慢线']):
        classification.append('rebound')

      # peak back
      if conditions['上行'] and conditions['触顶回落'] and (conditions['下穿底部'] or conditions['下穿快线'] or conditions['下穿慢线']):
        classification.append('hitpeak')

      # uptrending
      if conditions['上行'] and (conditions['轨道中'] or conditions['轨道上方']) and ('hitpeak' not in classification and 'up_x_resistant' not in classification):
        classification.append('uptrending')

      # downtrending
      if conditions['下行'] and (conditions['轨道中'] or conditions['轨道下方']) and ('rebound' not in classification and 'down_x_support' not in classification):
        classification.append('downtrending')

      # waving
      if (conditions['波动'] or conditions['弱势']) and ('rebound' not in classification and 'hitpeak' not in classification and 'uptrending' not in classification and 'downtrending' not in classification):
        classification.append('waving')

      # others
      if len(classification) == 0:
        classification.append('others')

      # generate description: trend analysis
      description = (f'[{row["tankan_kijun_signal"]}]')
      prev_k = None
      for k in conditions.keys():
        segment_s = ''
        segment_e = ''
        s = None
        e = None
        addition = ''

        # if condition not triggered, continue
        if not conditions[k] or ('上穿' in k) or ('下穿' in k) or k =='T/K':
          continue
        
        # add addition info for conditions
        elif k in ['触底反弹']:
          addition = f'({row["linear_fit_low_stop"]})'
        elif k in ['触顶回落']:
          addition = f'({row["linear_fit_high_stop"]})'
        elif k in ['跌破支撑']:
          addition = f'({row["support"]})'
        elif k in ['突破阻挡']:
          addition = f'({row["resistant"]})'

        # add segment for conditions
        if k in ['强势', '弱势', '上行', '下行', '波动']:
          segment_s = '['
          
        if k in ['轨道中', '轨道上方', '轨道下方']:
          segment_e = f']' # ({row["rate_direction"].round(2)})
          
        if k in ['跌破支撑', '突破阻挡', '触顶回落', '触底反弹']:
          segment_s = '['
          segment_e = ']'
          
        if k in ['上行', '下行', '波动'] and prev_k in ['强势', '弱势']:
          segment_s = ''
          
        description += f'{segment_s}{k[s:e]}{addition}{segment_e}'
        prev_k = k

      # assign category
      category = '/'.join(classification)
      df.loc[max_idx, 'category'] = category

      # assign description
      description += '' if (description[-1] == ']') else ']'
      df.loc[max_idx, 'description'] = description

  except Exception as e:
    print(phase, e)

  return df

# calculate ta signal
def calculate_ta_signal(df):
  """
  Calculate signals from ta features

  :param df: dataframe with ta features and derived features for calculating signals
  :raturns: dataframe with signal
  :raises: None
  """
  if len(df) == 0:
    print(f'No data for calculate_ta_signal')
    return None

  # copy data, initialize
  df = df.copy()

  df = cal_change_rate(df=df, target_col='tankan', periods=1, add_accumulation=True, add_prefix='tankan', drop_na=False)

  # ================================ buy and sell signals ==========================
  df['trend'] = ''

  # buy conditions
  buy_conditions = {
    # # stable version
    # 'ichimoku/aroon/adx/psar are all up trending': '(trend_idx == 4)',
    # 'renko is up trending': '(renko_trend == "u")',
    # 'bb is not over-buying': '(bb_trend != "d")',

    # developing version
    'at least 4 in ichimoku/aroon/adx/kst/psar are up trending': '(trend_idx >= 3)',
    'renko is up trending': '(renko_trend == "u")',
    'bb is not over-buying': '(bb_trend != "d")',
    'positive candle patterns': '((candle_color == 1) or (0< 窗口_day <=3) or (0< 突破_day <=3) or (0< 反弹_day <=3) or (0< 启明黄昏_day <=3))',
    'not a cross or highwave': '(十字星_trend != "u" and 十字星_trend != "d")',
    'not hanging or meteor on the top': '((位置_trend != "u") or (位置_trend == "u" and 锤子_trend != "d" and 锤子_trend != "u"))',
  }
  up_idx = df.query(' and '.join(buy_conditions.values())).index 
  df.loc[up_idx, 'trend'] = 'u'

  # sell conditions
  sell_conditions = {
    # # stable version
    # 'High is below kijun line': '(High < kijun)',
    # 'no individual trend is up and overall trend is down': '(trend_idx < -1 and up_trend_idx == 0)',
    # 'price went down through brick': '(renko_trend == "d")', 
    # 'bb is not over-selling': '(bb_trend != "u")',
    
    # developing version
    'High is below kijun line': '(High < kijun)',
    'no individual trend is up and overall trend is down': '(trend_idx < -1 and up_trend_idx == 0)',
    'price went down through brick': '(renko_trend == "d")', 
    'bb is not over-selling': '(bb_trend != "u" or (Close < renko_l and renko_duration >= 150))',
  } 
  down_idx = df.query(' and '.join(sell_conditions.values())).index 
  df.loc[down_idx, 'trend'] = 'd'

  # ================================ Calculate overall siganl ======================
  df['signal_day'] = sda(series=df['trend'].replace({'':0, 'n':0, 'u':1, 'd':-1}).fillna(0), zero_as=1)

  df['signal'] = '' 
  df.loc[df['signal_day'] == 1, 'signal'] = 'b'
  df.loc[df['signal_day'] ==-1, 'signal'] = 's'


  return df

# calculate ta indicators, trend and derivatives fpr latest data
def calculation(df, symbol, start_date=None, end_date=None, trend_indicators=['ichimoku', 'aroon', 'adx', 'psar'], volume_indicators=[], volatility_indicators=['bb'], other_indicators=[], signal_threshold=0.001):
  """
  Calculate selected ta features for dataframe

  :param df: original dataframe with hlocv features
  :param symbol: symbol of the data
  :param trend_indicators: trend indicators
  :param volumn_indicators: volume indicators
  :param volatility_indicators: volatility indicators
  :param other_indicators: other indicators
  :param signal_threshold: threshold for kama/ichimoku trigerment
  :returns: dataframe with ta features, derivatives, signals
  :raises: None
  """
  # copy dataframe
  df = df.copy()
  if df is None or len(df) == 0:
    print(f'{symbol}: No data for calculate_ta_data')
    return None   
  
  try:
    # # preprocess sec_data
    phase = 'preprocess_sec_data'
    df = preprocess_sec_data(df=df, symbol=symbol)[start_date:end_date].copy()
    
    # calculate TA indicators
    phase = 'cal_ta_indicators' 
    df = calculate_ta_indicators(df=df, trend_indicators=trend_indicators, volume_indicators=volume_indicators, volatility_indicators=volatility_indicators, other_indicators=other_indicators)

    # calculate TA trend
    phase = 'cal_ta_trend'
    df = calculate_ta_trend(df=df, trend_indicators=trend_indicators, volume_indicators=volume_indicators, volatility_indicators=volatility_indicators, other_indicators=other_indicators, signal_threshold=signal_threshold)

    # calculate TA derivatives
    phase = 'cal_ta_derivatives'
    df = calculate_ta_derivatives(df)

    # calculate TA final signal
    phase = 'cal_ta_signals'
    df = calculate_ta_signal(df=df)

  except Exception as e:
    print(symbol, phase, e)

  return df

# visualize ta indicators
def visualization(df, start=None, end=None, title=None, save_path=None, visualization_args={}):
  """
  visualize ta data
  :param df: dataframe with ta indicators
  :param start: start date to draw
  :param end: end date to draw
  :param title: title of the plot
  :param save_path: to where the plot will be saved
  :param visualization_args: arguments for plotting
  :returns: None
  :raises: Exception
  """
  if len(df) == 0:
    print(f'No data for visualize_ta_data')
    return None

  try:
    # visualize 
    phase = 'visualization'
    is_show = visualization_args.get('show_image')
    is_save = visualization_args.get('save_image')
    plot_args = visualization_args.get('plot_args')
    plot_multiple_indicators(
      df=df, title=title, args=plot_args,  start=start, end=end,
      show_image=is_show, save_image=is_save, save_path=save_path)
  except Exception as e:
    print(phase, e)

# post-process calculation results
def postprocess(df, keep_columns, drop_columns, target_interval=''):
  """
  Postprocess reulst data (last rows of symbols in a list)

  :param df: dataframe with ta features and ta derived features
  :param keep_columns: columns to keep for the final result
  :param drop_columns: columns to drop for the final result
  :param watch_columns: list of indicators to keep watching
  :returns: postprocessed dataframe
  :raises: None
  """
  if len(df) == 0:
    print(f'No data for postprocessing')
    return pd.DataFrame()

  # reset index(as the index(date) of rows are all the same)
  df = df.reset_index().copy()

  # overbuy/oversell
  df['obos'] = df['bb_trend'].replace({'d': '超买', 'u': '超卖', 'n': ''})

  # candle pattern index and description
  conditions = {
    # 'breakthrough or rebound': '(category == "up_x_resistant" or category == "rebound")',
    'candlestick window': '(反弹_trend == "u" or 突破_trend == "u" or 窗口_trend == "u" or 启明黄昏_trend == "u")',
    # 'ichimoku signal': '(0 < tankan_kijun_signal < 5)',
    'linear uptrend': 'linear_bounce_trend == "u" or linear_break_trend == "u"',
    # 'negative candle patterns': 'candle_pattern_idx < 0',
    # 'long upper shadow': '(upper_shadow_trend == "u")',
    # 'long red entity': '(candle_color == -1 and entity_trend != "d")',
    # 'under candlestick window': '(candle_gap_resistant == candle_gap_resistant and Low < candle_gap_resistant)',
    # 'waving falling or hitpeak': '(category == "waving" or category == "down_x_support" or category == "hitpeak")',
    'signal': '(signal == "b" or signal == "s")'}
  values = {
    # 'breakthrough or rebound': 'potential',
    'candlestick window': 'potential',
    # 'ichimoku signal': 'potential',
    'linear uptrend': 'potential',
    # 'negative candle patterns': '',
    # 'long red entity': '',
    # 'long upper shadow': '',
    # 'under candlestick window': '',
    # 'waving falling or hitpeak': '',
    'signal': 'signal'}
  df = assign_condition_value(df=df, column='label', condition_dict=conditions, value_dict=values, default_value='')

  # rename columns, keep 3 digits
  df = df[list(keep_columns.keys())].rename(columns=keep_columns).round(3)
  
  # add target-interval info
  df['ti'] = target_interval

  # drop columns
  df = df.drop(drop_columns, axis=1)
  
  # sort by operation and symbol
  df = df.sort_values(['交易信号', '代码'], ascending=[True, True])
  
  return df


# ================================================ Basic calculation ================================================ #
# drop na values for dataframe
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

# fill na values for dataframe
def fillna(series, fill_value=0):
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
      return max(x1, x2)
    elif f == 'min':
      return min(x1, x2)
    else:
      raise ValueError('"f" variable value should be "min" or "max"')
  else:
    return np.nan    
    
# filter index that meet conditions
def filter_idx(df, condition_dict):
  """
  # Filter index that meet conditions

  :param df: dataframe to search
  :param condition_dict: dictionary of conditions
  :returns: dictionary of index that meet corresponding conditions
  :raises: None
  """
  # target index
  result = {}
  for condition in condition_dict.keys():
    result[condition] = df.query(condition_dict[condition]).index

  # other index
  other_idx = df.index
  for i in result.keys():
    other_idx = [x for x in other_idx if x not in result[i]]
  result['other'] = other_idx

  return result

# set value to column of indeies that meet specific conditions
def assign_condition_value(df, column, condition_dict, value_dict, default_value=None):
  """
  # set value to column of index that meet conditions

  :param df: dataframe to search
  :param column: target column
  :param condition_dict: dictionary of conditions
  :param value_dict: corresponding value to assign to the column
  :param default_value: default value of the column
  :returns: dataframe with the column set
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

# set index-column with specific value
def set_idx_col_value(df, idx, col, values, set_on_copy=True):
  """
  Set specific index-column with specific values

  :param df: dataframe to search
  :param idx: dictionary of index
  :param col: target column
  :param values: dictionary of values, with same keys as idx
  :param set_on_copy: whether to set values on a copy of df
  :returns: dataframe with value set
  :raises: None
  """
  
  # copy dataframe
  if set_on_copy:
    df = df.copy()

  # set values to specific index, column
  for i in idx.keys():
    df.loc[idx[i], col] = values[i]

  return df


# ================================================ Rolling windows ================================================== #
# simple moving window
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

# exponential moving window
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

# weighted moving average
def wma(series, periods, fillna=False):
  """
  Weighted monving average from simple moving window

  :param series: series to calculate
  :param periods: size of the moving window
  :param fillna: make the min_periods = 0
  :returns: a rolling weighted average of 'series' with window size 'periods'
  :raises: none
  """
  weight = pd.Series([i * 2 / (periods * (periods + 1)) for i in range(1, periods + 1)])
  weighted_average = sm(series=series, periods=periods).apply(lambda x: (weight * x).sum(), raw=True)
  return weighted_average

# same direction accumulation
def sda(series, zero_as=None):
  """
  Accumulate value with same symbol (+/-), once the symbol changed, start over again

  :param series: series to calculate
  :param accumulate_by: if None, accumulate by its own value, other wise, add by specified value
  :param zero_val: action when encounter 0: if None pass, else add(minus) spedicied value according to previous symbol 
  :returns: series with same direction accumulation
  :raises: None
  """
  # copy series
  target_col = series.name
  index_col = series.index.name
  new_series = series.reset_index()

  previous_idx = None
  current_idx = None
  for index, row in new_series.iterrows():
    
    # record current index
    current_idx = index

    # for the first loop
    if previous_idx is None:
      pass

    # for the rest of loops
    else:
      current_val = new_series.loc[current_idx, target_col]
      previous_val = new_series.loc[previous_idx, target_col]

      # with same direction
      if current_val * previous_val > 0:
        new_series.loc[current_idx, target_col] = current_val + previous_val
      
      # current value is 0 and previous value is not 0
      elif current_val == 0 and previous_val != 0:
        if zero_as is not None:
          if previous_val > 0:
            new_series.loc[current_idx, target_col] = previous_val + zero_as
          else: 
            new_series.loc[current_idx, target_col] = previous_val - zero_as

      # otherwise(different direction, previous(and current) value is 0)
      else:
        pass

    # record previous index
    previous_idx = index

  # reset index back
  new_series = new_series.set_index(index_col)[target_col].copy()

  return new_series

# moving slope
def moving_slope(series, periods):
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

# ================================================ Change calculation =============================================== #
# calculate change of a column in certain period
def cal_change(df, target_col, periods=1, add_accumulation=True, add_prefix=False, drop_na=False):
  """
  Calculate change of a column with a sliding window
  
  :param df: original dfframe
  :param target_col: change of which column to calculate
  :param periods: calculate the change within the period
  :param add_accumulation: wether to add accumulative change in a same direction
  :param add_prefix: whether to add prefix for the result columns (when there are multiple target columns to calculate)
  :param drop_na: whether to drop na values from dataframe:
  :returns: dataframe with change rate columns
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
def cal_change_rate(df, target_col, periods=1, add_accumulation=True, add_prefix=False, drop_na=False):
  """
  Calculate change rate of a column with a sliding window
  
  :param df: original dfframe
  :param target_col: change rate of which column to calculate
  :param periods: calculate the change rate within the period
  :param add_accumulation: wether to add accumulative change rate in a same direction
  :param add_prefix: whether to add prefix for the result columns (when there are multiple target columns to calculate)
  :param drop_na: whether to drop na values from dataframe:
  :returns: dataframe with change rate columns
  :raises: none
  """
  # copy dfframe
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
  
    # go through each row, add values with same symbols (+/-)
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current_rate = df.loc[current_idx, rate_col]
      previous_acc_rate = df.loc[previous_idx, acc_rate_col]
      previous_acc_days = df.loc[previous_idx, acc_day_col]

      if previous_acc_rate * current_rate > 0:
        df.loc[current_idx, acc_rate_col] = current_rate + previous_acc_rate
        df.loc[current_idx, acc_day_col] += previous_acc_days
      else:
        df.loc[current_idx, acc_rate_col] = current_rate

  if drop_na:        
    df.dropna(inplace=True) 

  return df


# ================================================ Signal processing ================================================ #
# calculate signal that generated from 2 lines crossover
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
  conditions = {
    'up': f'{upper_col} > {upper_boundary}', 
    'down': f'{lower_col} < {lower_boundary}'} 
  values = {
    'up': pos_signal, 
    'down': neg_signal}
  df = assign_condition_value(df=df, column=result_col, condition_dict=conditions, value_dict=values, default_value=none_signal)

  return df[[result_col]]

# replace signal values 
def replace_signal(df, signal_col='signal', replacement={'b':1, 's':-1, 'n': 0}):
  """
  Replace signals with different values
  :param df: df that contains signal column
  :param signal_col: column name of the signal
  :param replacement: replacement, key is original value, value is the new value
  :returns: df with signal values replaced
  :raises: none
  """
  # copy dataframe
  new_df = df.copy()

  # find and replace
  for i in replacement.keys():
    new_df[signal_col].replace(to_replace=i, value=replacement[i], inplace=True)

  return new_df

# remove duplicated signals
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
  
  # initialize
  signals = df.query(f'{signal_col} != "{none_signal}"').copy()
  movement = {'first': 1, 'last': -1}.get(keep)

  # find duplicated signals and set to none_signal
  if len(signals) > 0 and movement is not None:
    signals['is_dup'] = signals[signal_col] + signals[signal_col].shift(movement)
    dup_idx = signals.query(f'is_dup == "{pos_signal}{pos_signal}" or is_dup == "{neg_signal}{neg_signal}"').index

    if len(dup_idx) > 0:
      df.loc[dup_idx, signal_col] = none_signal

  return df


# ================================================ Self-defined TA ================================================== #
# linear regression
def linear_fit(df, target_col, periods):
  """
  Calculate slope for selected piece of data

  :param df: dataframe
  :param target_col: target column name
  :param periods: input data length 
  :returns: slope of selected data from linear regression
  :raises: none
  """

  if len(df) <= periods:
    return {'slope': 0, 'intecept': 0}
  
  else:
    x = range(1, periods+1)
    y = df[target_col].fillna(0).tail(periods).values.tolist()
    lr = linregress(x, y)

    return {'slope': lr[0], 'intecept': lr[1]}
   
# calculate peak / trough in price
def cal_peak_trough(df, target_col, height=None, threshold=None, distance=None, width=None):
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

  # set result values
  result_col='peak_trend'
  peak_signal='d'
  trough_signal='u'
  none_signal=''
  
  try:
    # find peaks 
    peaks, _ = find_peaks(df[target_col], height=height, threshold=threshold, distance=distance, width=width)
    peaks = df.iloc[peaks,].index

    # find troughs
    troughs, _ = find_peaks(-df[target_col], height=height, threshold=threshold, distance=distance, width=width)
    troughs = df.iloc[troughs,].index

    # set signal values
    df[result_col] = none_signal
    df.loc[peaks, result_col] = peak_signal
    df.loc[troughs, result_col] = trough_signal
    
  except Exception as e:
    print(e)

  return df[[result_col]]

# calculate moving average 
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
    ma_col = f'{target_col}_ma_{mw}'
    df[ma_col] = mw_func(series=df[target_col], periods=mw).mean()
  
  return df

# add candle stick features 
def add_candlestick_features(df, ohlcv_col=default_ohlcv_col, pattern_recognition=False):
  """
  Add candlestick dimentions for dataframe

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
  
  # candle color
  df['candle_color'] = 0
  up_idx = df[open] < df[close]
  down_idx = df[open] >= df[close]
  df.loc[up_idx, 'candle_color'] = 1
  df.loc[down_idx, 'candle_color'] = -1
  df['candle_color_day'] = sda(series=df['candle_color'], zero_as=1)
  
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

  # ======================================= gap ====================================================== #
  # gap_up / gap_down
  col_to_drop = [] 
  for col in [open, close, high, low, 'candle_color', 'candle_entity']:
    prev_col = f'prev_{col}' 
    df[prev_col] = df[col].shift(1)
    col_to_drop.append(prev_col)
  
  # gap up
  df['candle_gap'] = 0
  df['pct_close'] = df[close] * 0.01

  df['low_prev_high'] = df[low] - df[f'prev_{high}']
  gap_up_idx = df.query(f'low_prev_high > 0').index
  df.loc[gap_up_idx, 'candle_gap'] = 1
  strict_gap_up_idx = df.query(f'low_prev_high >= pct_close').index
  df.loc[strict_gap_up_idx, 'candle_gap'] = 2
  df.loc[strict_gap_up_idx, 'candle_gap_color'] = 1
  df.loc[strict_gap_up_idx, 'candle_gap_top'] = df.loc[strict_gap_up_idx, f'{low}']
  df.loc[strict_gap_up_idx, 'candle_gap_bottom'] = df.loc[strict_gap_up_idx, f'prev_{high}']
  
  # gap down
  df['prev_low_high'] = df[f'prev_{low}'] - df[high]
  gap_down_idx = df.query(f'prev_low_high > 0').index  
  df.loc[gap_down_idx, 'candle_gap'] = -1
  strict_gap_down_idx = df.query(f'prev_low_high >= pct_close').index  
  df.loc[strict_gap_down_idx, 'candle_gap'] = -2
  df.loc[strict_gap_down_idx, 'candle_gap_color'] = -1
  df.loc[strict_gap_down_idx, 'candle_gap_top'] = df.loc[strict_gap_down_idx, f'prev_{low}']
  df.loc[strict_gap_down_idx, 'candle_gap_bottom'] = df.loc[strict_gap_down_idx, f'{high}']

  # # if there's too many gaps, disnote all gaps
  # up_gaps = df.query('candle_gap == 2').index
  # down_gaps = df.query('candle_gap == -2').index
  # disnote_gaps = []
  # if len(up_gaps) >= 20:
  #   disnote_gaps += up_gaps
  # if len(down_gaps) >= 20:
  #   disnote_gaps += down_gaps
  # for g in disnote_gaps:
  #   df.loc[g, 'candle_gap'] = 0
  #   for col in ['candle_gap_color', 'candle_gap_top', 'candle_gap_bottom']:
  #     df.loc[g, col] = np.NaN

  # gap color, top and bottom
  df['candle_gap_top'] = df['candle_gap_top'].fillna(method='ffill') 
  df['candle_gap_bottom'] = df['candle_gap_bottom'].fillna(method='ffill') 
  df['candle_gap_color'] = df['candle_gap_color'].fillna(method='ffill')

  # gap support and resistant
  support_idx = df.query(f'{close} > candle_gap_bottom').index
  resistant_idx = df.query(f'{close} < candle_gap_top').index
  df.loc[support_idx, 'candle_gap_support'] = df.loc[support_idx, 'candle_gap_bottom']
  df.loc[resistant_idx, 'candle_gap_resistant'] = df.loc[resistant_idx, 'candle_gap_top']
  
  # drop intermidiate columns
  for c in ['low_prev_high', 'prev_low_high', 'pct_close']:
    col_to_drop.append(c)
  # df = df.drop(col_to_drop, axis=1)

  return df

# add heikin-ashi candlestick features
def add_heikin_ashi_features(df, ohlcv_col=default_ohlcv_col, replace_ohlc=False, dropna=True):
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
  for col in [open, high, low, close]:
    df[f'prev_{col}'] = df[col].shift(1)

  # calculate heikin-ashi ohlc
  df['H_Close'] = (df[open] + df[high] + df[low] + df[close])/4
  df['H_Open'] = (df[f'prev_{open}'] + df[f'prev_{close}'])/2
  df['H_High'] = df[[f'prev_{high}', 'H_Open', 'H_Close']].max(axis=1)
  df['H_Low'] = df[[f'prev_{low}', 'H_Open', 'H_Close']].min(axis=1)

  # drop previous stick
  for col in [open, high, low, close]:
    df.drop(f'prev_{col}', axis=1, inplace=True)
    
  # replace original ohlc with heikin-ashi ohlc
  if replace_ohlc:
    for col in [open, high, low, close]:
      df.drop(f'{col}', axis=1, inplace=True)
    df.rename(columns={'H_Close': close, 'H_Open': open, 'H_High': high, 'H_Low': low}, inplace=True)

  # dropna values
  if dropna:
    df.dropna(inplace=True)
  
  return df

# linear regression for recent high and low values
def add_linear_features(df, max_period=60, min_period=15, is_print=False):

  # get all indexes
  idxs = df.index.tolist()

  # get current date, renko_color, earliest-start date, latest-end date
  current_date = df.index.max()
  current_color = df.loc[current_date, 'renko_color']
  earliest_start = df.tail(max_period).index.min() #current_date - datetime.timedelta(days=max_period)
  if (idxs[-1] - idxs[-2]).days >= 7:
    latest_end = idxs[-2]
  else:
    latest_end = current_date - datetime.timedelta(days=(current_date.weekday()+1))

  # recent extreme as the latest_end
  if df[idxs[-min_period]:]['High'].max() == df[idxs[-max_period]:]['High'].max():
    extreme_high = df[idxs[-min_period]:]['High'].idxmax()
  else:
    extreme_high = None

  if df[idxs[-min_period]:]['Low'].min() == df[idxs[-max_period]:]['Low'].min():
    extreme_low = df[idxs[-min_period]:]['Low'].idxmin()
  else:
    extreme_low = None
  latest_end = min(latest_end, extreme_high) if extreme_high is not None else latest_end
  latest_end = min(latest_end, extreme_low) if extreme_low is not None else latest_end

  # get slice according to renko bricks, allows only 1 different color brick
  start=None  
  renko_list = df.query('renko_real == renko_real').index.tolist()
  renko_list.reverse()
  for idx in renko_list:
    tmp_color = df.loc[idx, 'renko_color']
    tmp_start = df.loc[idx, 'renko_start']
    if tmp_color != current_color:
      break
    else:
      if tmp_start < earliest_start:
        start = earliest_start
        break
  start = max(tmp_start, earliest_start)
  end = latest_end
  
  # make the slice length at least min_period
  if len(df[start:end]) < min_period: #(end - start).days < min_period:
    start = df[:end].tail(min_period).index.min() # end - datetime.timedelta(days=min_period)
  if len(df[start:end]) > min_period: #(end - start).days > max_period:
    start = df[:end].tail(max_period).index.min() # end - datetime.timedelta(days=max_period)
  if is_print:
    print(start, end)

  # calculate peaks and troughs
  tmp_data = df[start:end].copy()
  tmp_idxs = tmp_data.index.tolist()
  
  # find the highest high and lowest low
  hh = tmp_data['High'].idxmax()
  ll = tmp_data['Low'].idxmin()
  if is_print:
    print(hh, ll)

  # get slice from highest high and lowest low
  if hh > ll:
    if len(df[hh:latest_end]) >= min_period: # (latest_end - hh).days >= min_period:
      start = hh
    elif len(df[ll:latest_end]) >= min_period: # (latest_end - ll).days >= min_period:
      start = ll
    else:
      end = max(hh, ll)
  else:
    if len(df[ll:latest_end]) >= min_period: #(latest_end - ll).days >= min_period:
      start = ll
    elif len(df[hh:latest_end]) >= min_period: # (latest_end - hh).days >= min_period:
      start = hh
    else:
      end = max(hh, ll)

  # if start != earliest_start:
  #   # start = start - datetime.timedelta(days=3)
  #   si = df.index.tolist().index(start)
  #   si = si - 1
  #   start = df.index.tolist()[si]

  # get peaks and troughs
  tmp_data = df[start:end].copy()
  tmp_idxs = tmp_data.index.tolist()
  num_points = 4 #int(len(tmp_data) / 3)
  distance = math.floor(len(tmp_data) / num_points)
  distance = 1 if distance < 1 else distance
  day_gap = math.floor(len(tmp_data) / 2)
  highest_high = df[start:]['High'].max() # tmp_data['High'].max()
  highest_high_idx = tmp_data['High'].idxmax()
  lowest_low = df[start:]['Low'].min()# tmp_data['Low'].min()
  lowest_low_idx = tmp_data['Low'].idxmin()

  # peaks
  peaks,_ = find_peaks(x=tmp_data['High'], distance=distance)
  peaks = [tmp_idxs[x] for x in peaks]
  if is_print:
    print(df.loc[peaks, 'High'])

  # divide peaks by highest peak, take the longer half
  if len(peaks) >= 2:
    peak_value = [df.loc[x, 'High'] for x in peaks]
    hp = peak_value.index(max(peak_value))

    if hp+1 > len(peak_value)/2:
      peaks = peaks[:hp+1]
    elif hp+1 <= math.ceil(len(peak_value)/2):
      peaks = peaks[hp:]

  s = start
  e = start
  while e < end:
    e = s + datetime.timedelta(days=day_gap)
    t_data = df[s:e].copy()
    if len(t_data) == 0:
      s = e
      continue
    else:
      # highest high 
      hh_idx = t_data['High'].idxmax()
      if hh_idx not in peaks:
        peaks = np.append(peaks, hh_idx)
      s = e
  if is_print:
    print(df.loc[peaks, 'High'])

  # troughs
  troughs, _ = find_peaks(x=-tmp_data['Low'], distance=distance)
  troughs = [tmp_idxs[x] for x in troughs]
  if is_print:
    print(df.loc[troughs, 'Low'])

  # divide troughs by lowest trough, take the longer half
  if len(troughs) >= 2:
    trough_value = [df.loc[x, 'Low'] for x in troughs]
    lt = trough_value.index(min(trough_value))
    
    if lt+1 > len(trough_value)/2:
      troughs = troughs[:lt+1]
    elif lt+1 <= math.ceil(len(trough_value)/2):
      troughs = troughs[lt:]
    
  # else:
  s = start
  e = start
  while e < end:
    e = s + datetime.timedelta(days=day_gap)
    t_data = df[s:e].copy()
    if len(t_data) == 0:
      s = e
      continue
    else:

      # lowest_low
      ll_idx = t_data['Low'].idxmin()
      troughs = np.append(troughs, ll_idx)
      
      # update end date
      s = e
  if is_print:
    print(df.loc[troughs, 'Low'])

  # gathering high and low points
  high = {'x':[], 'y':[]}
  low = {'x':[], 'y':[]}

  for p in peaks:
    x = idxs.index(p)
    y = df.loc[p, 'High'] #+ df[start:end]['High'].std()*0.5
    high['x'].append(x)
    high['y'].append(y)

  for t in troughs:
    x = idxs.index(t)
    y = df.loc[t, 'Low'] #- df[start:end]['Low'].std()*0.5
    low['x'].append(x)
    low['y'].append(y)

  # linear regression for high/low values
  if len(high['x']) < 2: 
    high_linear = (0, highest_high, 0, 0)
  else:
    high_linear = linregress(high['x'], high['y'])
    high_range = round((max(high['y']) + min(high['y']))/2, 3)
    slope_score = round(abs(high_linear[0])/high_range, 5)
    if slope_score < 0.001:
      high_linear = (0, highest_high, 0, 0)

  if len(low['x']) < 2:
    low_linear = (0, lowest_low, 0, 0)
  else:
    low_linear = linregress(low['x'], low['y'])
    low_range = round((max(low['y']) + min(low['y']))/2, 3)
    slope_score = round(abs(low_linear[0])/low_range, 5)
    if slope_score < 0.001:
      low_linear = (0, lowest_low, 0, 0)

  # add high/low fit values
  counter = 0
  idx_max = len(idxs)
  idx_min = min(min(high['x']), min(low['x']))
  for x in range(idx_min, idx_max):
    
    idx = idxs[x]
    counter += 1
    df.loc[idx, 'linear_day_count'] = counter

    # calculate linear fit values    
    linear_fit_high = high_linear[0] * x + high_linear[1]
    linear_fit_low = low_linear[0] * x + low_linear[1]

    # linear fit high
    df.loc[idx, 'linear_fit_high_slope'] = high_linear[0]

    if (linear_fit_high <= highest_high and linear_fit_high >= lowest_low): 
      df.loc[idx, 'linear_fit_high'] = linear_fit_high
    elif linear_fit_high > highest_high:
      df.loc[idx, 'linear_fit_high'] = highest_high
    elif linear_fit_high < lowest_low:
      df.loc[idx, 'linear_fit_high'] = lowest_low
    else:
      df.loc[idx, 'linear_fit_high'] = np.NaN
    
    if  high_linear[0] > 0 and idx >= highest_high_idx and df.loc[idx, 'linear_fit_high'] <= highest_high:
      df.loc[idx, 'linear_fit_high'] = highest_high

    # linear fit low
    df.loc[idx, 'linear_fit_low_slope'] = low_linear[0]

    if (linear_fit_low <= highest_high and linear_fit_low >= lowest_low): 
      df.loc[idx, 'linear_fit_low'] = linear_fit_low
    elif linear_fit_low > highest_high:
      df.loc[idx, 'linear_fit_low'] = highest_high
    elif linear_fit_low < lowest_low:
      df.loc[idx, 'linear_fit_low'] = lowest_low
    else:
      df.loc[idx, 'linear_fit_low'] = np.NaN

    if  low_linear[0] < 0 and idx >= lowest_low_idx and df.loc[idx, 'linear_fit_low'] >= lowest_low:
      df.loc[idx, 'linear_fit_low'] = lowest_low

  # high/low fit stop
  df['linear_fit_high_stop'] = 0
  df['linear_fit_low_stop'] = 0
  reach_top_idx = df.query(f'High=={highest_high} and linear_fit_high == {highest_high} and linear_fit_high_slope >= 0').index
  reach_bottom_idx = df.query(f'Low=={lowest_low} and linear_fit_low == {lowest_low} and linear_fit_low_slope <= 0').index
  df.loc[reach_top_idx, 'linear_fit_high_stop'] = 1
  df.loc[reach_bottom_idx, 'linear_fit_low_stop'] = 1
  df.loc[reach_top_idx, 'linear_top_entity_top'] = df.loc[reach_top_idx, 'candle_entity_top']
  df.loc[reach_top_idx, 'linear_top_entity_bottom'] = df.loc[reach_top_idx, 'candle_entity_bottom']
  df.loc[reach_bottom_idx, 'linear_bottom_entity_top'] = df.loc[reach_bottom_idx, 'candle_entity_top']
  df.loc[reach_bottom_idx, 'linear_bottom_entity_bottom'] = df.loc[reach_bottom_idx, 'candle_entity_bottom']
  
  for col in ['linear_fit_high_stop', 'linear_fit_low_stop', 'linear_top_entity_top', 'linear_top_entity_bottom', 'linear_bottom_entity_top', 'linear_bottom_entity_bottom']:
    df[col] = df[col].fillna(method='ffill')
  df['linear_fit_low_stop'] = sda(df['linear_fit_low_stop'], zero_as=1)
  df['linear_fit_high_stop'] = sda(df['linear_fit_high_stop'], zero_as=1)

  # support and resistant
  resistant_idx = df.query(f'linear_fit_high == {highest_high} and linear_fit_high_stop > 0').index
  if len(resistant_idx) > 0:
    df.loc[min(resistant_idx), 'linear_fit_resistant'] = highest_high
  else:
    df['linear_fit_resistant'] = np.nan

  support_idx = df.query(f'linear_fit_low == {lowest_low} and linear_fit_low_stop > 0').index
  if len(support_idx) > 0:
    df.loc[min(support_idx), 'linear_fit_support'] = lowest_low
  else:
    df['linear_fit_support'] = np.nan

  for col in ['linear_fit_support', 'linear_fit_resistant']:
    df[col] = df[col].fillna(method='ffill')

  # overall slope of High and Low
  df['linear_slope']  = df['linear_fit_high_slope'] + df['linear_fit_low_slope']

  # direction means the slopes of linear fit High/Low
  conditions = {
    'up': '(linear_fit_high_slope > 0 and linear_fit_low_slope > 0) or (linear_fit_high_slope > 0 and linear_fit_low_slope == 0) or (linear_fit_high_slope == 0 and linear_fit_low_slope > 0)', 
    'down': '(linear_fit_high_slope < 0 and linear_fit_low_slope < 0) or (linear_fit_high_slope < 0 and linear_fit_low_slope == 0) or (linear_fit_high_slope == 0 and linear_fit_low_slope < 0)',
    'none': '(linear_fit_high_slope > 0 and linear_fit_low_slope < 0) or (linear_fit_high_slope < 0 and linear_fit_low_slope > 0) or (linear_fit_high_slope == 0 and linear_fit_low_slope == 0)'} 
  values = {
    'up': 'u', 
    'down': 'd',
    'none': 'n'}
  df = assign_condition_value(df=df, column='linear_direction', condition_dict=conditions, value_dict=values, default_value='')

  # price direction
  df['price_direction'] = 0.5
  df['rate_direction'] = 0.5

  min_idx = tmp_idxs[0]
  reach_top = None
  reach_bottom = None

  if len(reach_top_idx) > 0:
    reach_top = reach_top_idx[0]
  if len(reach_bottom_idx) > 0:
    reach_bottom = reach_bottom_idx[0]

  if reach_top is None and reach_bottom is None:
    start = min_idx
  elif reach_top is None and reach_bottom is not None:
    start = reach_bottom
  elif reach_top is not None and reach_bottom is None:
    start = reach_top
  else:
    start = max(reach_top, reach_bottom)

  stop_data = df[start:].copy()
  if len(stop_data) > 0:
    counter = 0
    for index, row in stop_data.iterrows():
      counter += 1
      if index == start:
        continue
      else:
        
        x1 = (df[start:index]['candle_color'] > 0).sum()
        y1 = x1 / counter

        x2 = (df[start:index]['rate'] > 0).sum()
        y2 = x2 / counter

        df.loc[index, 'price_direction'] = y1
        df.loc[index, 'rate_direction'] = y2
        
  return df
 

# ================================================ Trend indicators ================================================= #
# ADX(Average Directional Index) 
def add_adx_features(df, n=14, ohlcv_col=default_ohlcv_col, fillna=False, adx_threshold=25):
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
  # df['adx_diff'] = (df['adx_diff'] - df['adx_diff'].mean()) / df['adx_diff'].std()

  # fill na values
  if fillna:
    for col in ['pdm', 'mdm', 'atr', 'pdi', 'mdi', 'dx', 'adx']:
      df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
  
  # drop redundant columns
  df.drop(['high_diff', 'low_diff', 'zero', 'pdm', 'mdm'], axis=1, inplace=True)

  return df

# Aroon
def add_aroon_features(df, n=25, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, boundary=[50, 50]):
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
def add_cci_features(df, n=20, c=0.015, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, boundary=[200, -200]):
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
  df = cal_moving_average(df=df, target_col='cci', ma_windows=[3, 5])

  return df

# DPO(Detrended Price Oscillator)
def add_dpo_features(df, n=20, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_ichimoku_features(df, n_short=9, n_medium=26, n_long=52, method='ta', is_shift=True, ohlcv_col=default_ohlcv_col, fillna=False, cal_status=True):
  """
  Calculate Ichimoku indicators

  :param df: original OHLCV dataframe
  :param n_short: short window size
  :param n_medium: medium window size
  :param n_long: long window size
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

  # tankan-kijun signal
  df['tankan_kijun_signal'] = cal_crossover_signal(df=df, fast_line='tankan', slow_line='kijun', pos_signal=1, neg_signal=-1, none_signal=0)
  df['tankan_kijun_signal'] = sda(series=df['tankan_kijun_signal'], zero_as=1)

  if cal_status:
    # ================================ Cloud status ===================================
    # cloud color change, cloud height (how thick is the cloud)
    # df['cloud_shift'] = cal_crossover_signal(df=df, fast_line='senkou_a', slow_line='senkou_b', pos_signal=1, neg_signal=-1, none_signal=0)
    df['cloud_height'] = round((df['senkou_a'] - df['senkou_b'])/df[close], ndigits=3)
    green_idx = df.query('cloud_height > 0').index
    red_idx = df.query('cloud_height <= 0').index

    # cloud width (how has it last)
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

    # ================================ Close breakthrough =============================
    # calculate distance between Close and each ichimoku lines    
    line_weight = {'kijun':1, 'tankan':1, 'cloud_top':1, 'cloud_bottom':1}
    line_name = {"kijun":"基准", "tankan":"转换", "cloud_top":"云顶", "cloud_bottom":"云底"} 
    df['break_up'] = ''
    df['break_down'] = ''
    df['breakthrough'] = 0
    col_to_drop.append('breakthrough')

    for line in line_weight.keys():
      # set weight for this line
      weight = line_weight[line]

      # calculate breakthrough
      line_signal_name = f'{line}_signal'
      df[line_signal_name] = cal_crossover_signal(df=df, fast_line=close, slow_line=line, pos_signal=weight, neg_signal=-weight, none_signal=0)
      
      # record breakthrough
      up_idx = df.query(f'{line_signal_name} == {weight}').index
      down_idx = df.query(f'{line_signal_name} == {-weight}').index      
      df.loc[up_idx, 'break_up'] = df.loc[up_idx, 'break_up'] + line_name[line] + ','
      df.loc[down_idx, 'break_down'] = df.loc[down_idx, 'break_down'] + line_name[line] + ','
      
      # accumulate breakthrough signals
      df['breakthrough'] = df['breakthrough'].astype(int) +df[line_signal_name].astype(int)

      # calculate distance between close price and indicator
      df['close_to_' + line] = round((df[close] - df[line]) / df[close], ndigits=3)
      
    # drop redundant columns  
    df.drop(col_to_drop, axis=1, inplace=True)

  return df

# KST(Know Sure Thing)
def add_kst_features(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsign=9, ohlcv_col=default_ohlcv_col, fillna=False):
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
def add_macd_features(df, n_fast=12, n_slow=26, n_sign=9, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_mi_features(df, n=9, n2=25, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_trix_features(df, n=15, n_sign=9, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, signal_mode='mix'):
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
def add_vortex_features(df, n=14, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
  df['vortex_diff'] = df['vortex_diff'] - df['vortex_diff'].shift(1)

  # calculate signal
  if cal_signal:
    df['vortex_signal'] = cal_crossover_signal(df=df, fast_line='vortex_pos', slow_line='vortex_neg')

  return df

# PSAR
def add_psar_features(df, ohlcv_col=default_ohlcv_col, step=0.02, max_step=0.10, fillna=False):
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
def add_stc_features(df, n_fast=23, n_slow=50, n_cycle=10, n_smooth=3, ohlcv_col=default_ohlcv_col, fillna=False):
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
def add_renko_features(df, brick_size_factor=0.1, merge_duplicated=True):
  """
  Calculate Renko indicator
  :param df: original OHLCV dataframe
  :param brick_size_factor: if not using atr, brick size will be set to Close*brick_size_factor
  :param merge_duplicated: whether to merge duplicated indexes in the final result
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :returns: dataframe with new features generated
  """

  # reset index and copy df
  original_df = util.remove_duplicated_index(df=df, keep='last')
  df = original_df.copy()

  # use dynamic brick size: brick_size_factor * Close price
  df['bsz'] = (df['Close'] * brick_size_factor).round(3)

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
  renko_df['renko_brick_height'] = brick_size
  columns = ['Date', 'Open', 'High', 'Low', 'Close', 'uptrend', 'renko_brick_height']

  # go through the dataframe
  for index, row in df.iterrows():

    # get current date and close price
    date = row['Date']
    close = row['Close']
    
    # get previous trend and close price
    row_p1 = renko_df.iloc[-1]
    uptrend = row_p1['uptrend']
    close_p1 = row_p1['Close']
    brick_size_p1 = row_p1['renko_brick_height']

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
  renko_df['renko_duration'] = renko_df['renko_end'] - renko_df['renko_start']
  renko_df['renko_duration'] = renko_df['renko_duration'].apply(lambda x: x.days+1).astype(float)
  renko_df['renko_duration_p1'] = renko_df['renko_duration'].shift(1)

  # renko color(green/red), trend(u/d), flip_point(renko_real), same-direction-accumulation(renko_brick_sda), sda-moving sum(renko_brick_ms), number of bricks(for later calculation)
  renko_df['renko_color'] = renko_df['renko_color'].replace({True: 'green', False:'red'})
  renko_df['renko_direction'] = renko_df['renko_color'].replace({'green':'u', 'red':'d'})
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
          renko_df.loc[idx, 'renko_brick_height'] = tmp_rows['renko_brick_height'].sum()
          renko_df.loc[idx, 'renko_brick_number'] = tmp_rows['renko_brick_number'].sum()
        elif color == 'red':
          renko_df.loc[idx, 'renko_o'] = tmp_rows['renko_o'].max()
          renko_df.loc[idx, 'renko_l'] = tmp_rows['renko_l'].min()
          renko_df.loc[idx, 'renko_h'] = tmp_rows['renko_h'].max()
          renko_df.loc[idx, 'renko_c'] = tmp_rows['renko_c'].min()
          renko_df.loc[idx, 'renko_brick_height'] = tmp_rows['renko_brick_height'].sum()
          renko_df.loc[idx, 'renko_brick_number'] = tmp_rows['renko_brick_number'].sum() 
        else:
          print(f'unknown renko color {color}')
          continue 
      else:
        print('duplicated index with different renko colors!')
        continue
    renko_df = util.remove_duplicated_index(df=renko_df, keep='last')

  # calculate accumulated renko trend (so called "renko_series")
  series_len_short = 4
  series_len_long = 8
  renko_df['renko_series_short'] = 'n' * series_len_short
  renko_df['renko_series_long'] = 'n' * series_len_long
  prev_idx = None
  for idx, row in renko_df.iterrows():
    if prev_idx is not None:
      renko_df.loc[idx, 'renko_series_short'] = (renko_df.loc[prev_idx, 'renko_series_short'] + renko_df.loc[idx, 'renko_direction'])[-series_len_short:]
      renko_df.loc[idx, 'renko_series_long'] = (renko_df.loc[prev_idx, 'renko_series_long'] + renko_df.loc[idx, 'renko_direction'])[-series_len_long:]
    prev_idx = idx
  renko_df['renko_series_short_idx'] = renko_df['renko_series_short'].apply(lambda x: x.count('u') - x.count('d')).astype(int)
  renko_df['renko_series_long_idx'] = renko_df['renko_series_long'].apply(lambda x: x.count('u') - x.count('d')).astype(int)
    
  # drop currently-existed renko_df columns from df, merge renko_df into df 
  for col in df.columns:
    if 'renko' in col:
      df.drop(col, axis=1, inplace=True)
  df = pd.merge(df, renko_df, how='left', left_index=True, right_index=True)

  # for rows in downtrend, renko_brick_height = -renko_brick_height
  red_idx = df.query('renko_color == "red"').index
  df.loc[red_idx, 'renko_brick_height'] = -df.loc[red_idx, 'renko_brick_height']

  # fill na values
  renko_columns = ['renko_o', 'renko_h','renko_l', 'renko_c', 'renko_color', 'renko_brick_height', 'renko_brick_number','renko_start', 'renko_end', 'renko_duration', 'renko_duration_p1', 'renko_direction', 'renko_series_short_idx', 'renko_series_long_idx'] # , 'renko_direction', 'renko_series_short', 'renko_series_long'
  for col in renko_columns:
    df[col] = df[col].fillna(method='ffill')

  for col in ['renko_series_short_idx', 'renko_series_long_idx']:
    df[col] = df[col].fillna(0)

  # calculate length(number of days to the end of current brick) 
  # calculate of each brick(or merged brick): renko_brick_length, renko_countdown_days(for ploting)
  max_idx = df.index.max()
  if merge_duplicated:
    df['s']  = df.index
    if df['s'].max() == max_idx:
      max_idx = max_idx + datetime.timedelta(days=1)
    df['renko_countdown_days'] = df['renko_end'] - df['s'] 
    df['renko_brick_length'] = df['s'] - df['renko_start']
    df['renko_brick_length'] = df['renko_brick_length'].apply(lambda x: x.days+1).astype(float)
    df.drop('s', axis=1, inplace=True)
  else:
    df['renko_countdown_days'] = 1
    df['renko_brick_length'] = 1

  # number of days below/among/above renko bricks
  for col in ['above_renko_h', 'among_renko', 'below_renko_l']:
    df[col] = 0
  
  above_idx = df.query('Close > renko_h').index
  among_idx = df.query('renko_l <= Close <= renko_h').index
  below_idx = df.query('Close < renko_l').index
  df.loc[above_idx, 'above_renko_h'] = 1
  df.loc[below_idx, 'below_renko_l'] = 1
  df.loc[among_idx, 'among_renko'] = 1

  renko_swift_idx = df.query('renko_real == renko_real').index

  for col in ['above_renko_h', 'among_renko', 'below_renko_l']:
    df.loc[renko_swift_idx, col] = 0
    df[col] = sda(df[col], zero_as=None, )   

  # renko support and resistant
  df.loc[above_idx, 'renko_support'] = df.loc[above_idx, 'renko_h']
  df.loc[below_idx, 'renko_resistant'] = df.loc[below_idx, 'renko_l']
  df.loc[among_idx, 'renko_support'] = df.loc[among_idx, 'renko_l']
  df.loc[among_idx, 'renko_resistant'] = df.loc[among_idx, 'renko_h']

  return df


# ================================================ Volume indicators ================================================ #
# Accumulation Distribution Index
def add_adi_features(df, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_cmf_features(df, n=20, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_eom_features(df, n=20, ohlcv_col=default_ohlcv_col, fillna=False):
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
def add_fi_features(df, n1=2, n2=22, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_nvi_features(df, n=255, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_obv_features(df, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_vpt_features(df, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_ao_features(df, n_short=5, n_long=34, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
  df['ao_diff'] = df['ao'] - df['ao'].shift(1)

  return df

# Kaufman's Adaptive Moving Average (KAMA)
def cal_kama(df, n1=10, n2=2, n3=30, ohlcv_col=default_ohlcv_col, fillna=False):
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

# Kaufman's Adaptive Moving Average (KAMA)
def add_kama_features(df, n_param={'kama_fast': [10, 2, 30], 'kama_slow': [10, 5, 30]}, ohlcv_col=default_ohlcv_col, fillna=False):
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

  # calculate distance between close price and indicator
  kama_lines = ['kama_fast', 'kama_slow'] 
  for line in kama_lines:
    df[f'close_to_{line}'] = round((df[close] - df[line]) / df[close], ndigits=3)
    df[f'{line}_signal'] = cal_crossover_signal(df=df, fast_line=close, slow_line=line, pos_signal=1, neg_signal=-1, none_signal=0)
  
  return df

# Money Flow Index(MFI)
def add_mfi_features(df, n=14, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, boundary=[20, 80]):
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
    df = remove_redundant_signal(df=df, signal_col='mfi_signal', pos_signal='s', neg_signal='b', none_signal='n', keep='first')

  df.drop('up_or_down', axis=1, inplace=True)
  return df

# Relative Strength Index (RSI)
def add_rsi_features(df, n=14, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, boundary=[30, 70]):
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
  diff = df[close].pct_change(1)
  
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
    df = remove_redundant_signal(df=df, signal_col='rsi_signal', pos_signal='s', neg_signal='b', none_signal='n', keep='first')

  return df

# Stochastic Oscillator
def add_stoch_features(df, n=14, d_n=3, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, boundary=[20, 80]):
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
  df['stoch_diff'] = df['stoch_diff'] - df['stoch_diff'].shift(1)

  return df

# True strength index (TSI)
def add_tsi_features(df, r=25, s=13, ema_period=7, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_uo_features(df, s=7, m=14, l=28, ws=4.0, wm=2.0, wl=1.0, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=False):
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
def add_wr_features(df, lbp=14, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, boundary=[-20, -80]):
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
def add_atr_features(df, n=14, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
    df['atr_signal'] = df['tr'] - df['atr']

  df.drop(['h_l', 'h_pc', 'l_pc'], axis=1, inplace=True)

  return df

# Mean Reversion
def add_mean_reversion_features(df, n=100, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True, mr_threshold=2):
  """
  Calculate Mean Reversion

  :param df: original OHLCV dataframe
  :param n: look back window size
  :param ohlcv_col: column name of Open/High/Low/Close/Volume
  :param fillna: whether to fill na with 0
  :param cal_signal: whether to calculate signal
  :param mr_threshold: the threshold to triger signal
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

  # calculate change rate of close price
  df = cal_change_rate(df=df, target_col=close, periods=1, add_accumulation=True)
  target_col = ['rate', 'acc_rate', 'acc_day']

  # calculate the (current value - moving avg) / moving std
  for col in target_col:
    mw = sm(series=df[col], periods=n)
    tmp_mean = mw.mean()
    tmp_std = mw.std()
    df[col+'_bias'] = (df[col] - tmp_mean) / (tmp_std)

  # calculate the expected change rate that will triger signal
  result = cal_mean_reversion_expected_rate(df=df, rate_col='acc_rate', n=n, mr_threshold=mr_threshold)
  last_acc_rate = df['acc_rate'].tail(1).values[0]
  last_close = df[close].tail(1).values[0]

  up = down = 0
  if last_acc_rate > 0:
    up = max(result) - last_acc_rate
    down = min(result)
  else:
    up = max(result)
    down = min(result) - last_acc_rate

  up_price = round((1+up) * last_close, ndigits=2)
  down_price = round((1+down) * last_close, ndigits=2)
  up = round(up * 100, ndigits=0) 
  down = round(down * 100, ndigits=0) 
  df['mr_price'] = f'{up_price}({up}%%),{down_price}({down}%%)'

  # calculate mr signal
  if cal_signal:
    df['rate_signal'] = cal_boundary_signal(df=df, upper_col='rate_bias', lower_col='rate_bias', upper_boundary=mr_threshold, lower_boundary=-mr_threshold, pos_signal=1, neg_signal=-1, none_signal=0)
    df['acc_rate_signal'] = cal_boundary_signal(df=df, upper_col='acc_rate_bias', lower_col='acc_rate_bias', upper_boundary=mr_threshold, lower_boundary=-mr_threshold, pos_signal=1, neg_signal=-1, none_signal=0)
    df['mr_signal'] = df['rate_signal'].astype(int) + df['acc_rate_signal'].astype(int)
    df = replace_signal(df=df, signal_col='mr_signal', replacement={0: 'n', 1: 'n', -1:'n', 2:'b', -2:'s'})
    df.drop(['rate_signal', 'acc_rate_signal'], axis=1, inplace=True)   

  return df

# Price that will triger mean reversion signal
def cal_mean_reversion_expected_rate(df, rate_col, n=100, mr_threshold=2):
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
def add_bb_features(df, n=20, ndev=2, ohlcv_col=default_ohlcv_col, fillna=False):
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
def add_dc_features(df, n=20, ohlcv_col=default_ohlcv_col, fillna=False, cal_signal=True):
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
def add_kc_features(df, n=10, ohlcv_col=default_ohlcv_col, method='atr', fillna=False, cal_signal=True):
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


# ================================================ Indicator visualization  ========================================= #
# plot signals on price line
def plot_signal(
  df, start=None, end=None, price_col='Close', price_alpha=1,
  signal_col='signal', trend_col='trend', signal_val=default_signal_val, plot_on_price=True, 
  use_ax=None, title=None, plot_args=default_plot_args):
  """
  Plot signals along with the price

  :param df: dataframe with price and signal columns
  :param start: start row to plot
  :param end: end row to stop
  :param price_col: columnname of the price values
  :param signal_col: columnname of the signal values
  :param signal_val: value of different kind of signals
  :param plot_on_price: whether plot signal on price line
  :param use_ax: the already-created ax to draw on
  :param title: plot title
  :param plot_args: other plot arguments
  :returns: a signal plotted price chart
  :raises: none
  """
  # copy dataframe within the specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  # plot price
  if not plot_on_price:
    df['signal_base'] = df[price_col].min() - 1
    label = None

  else:
    df['signal_base'] = df[price_col]
    label = price_col
  ax.plot(df.index, df['signal_base'], color='black', marker='.', label=label, alpha=price_alpha)

  # get signal values
  pos_signal = signal_val['pos_signal']
  neg_signal = signal_val['neg_signal']
  wave_signal = signal_val['wave_signal']
  none_signal = signal_val['none_signal']

  # plot signals
  if signal_col in df.columns:
    
    signal_alpha = 1 if signal_col in ['signal'] else 0.3
    trend_alpha = 0.5 if signal_col in ['signal'] else 0.25
    positive_signal = df.query(f'{signal_col} == "{pos_signal}"')
    negative_signal = df.query(f'{signal_col} == "{neg_signal}"')
    wave_signal = df.query(f'{signal_col} == "{wave_signal}"')
    none_signal = df.query(f'{signal_col} == "{none_signal}"')
    ax.scatter(positive_signal.index, positive_signal['signal_base'], label=None, marker='^', color='green', alpha=signal_alpha)
    ax.scatter(negative_signal.index, negative_signal['signal_base'], label=None, marker='v', color='red', alpha=signal_alpha)
    
    if trend_col in df.columns:
      pos_trend = df.query(f'{trend_col} == "u"')
      neg_trend = df.query(f'{trend_col} == "d"') 
      wave_trend = df.query(f'{trend_col} == "n"')
      ax.scatter(pos_trend.index, pos_trend['signal_base'], label=None, marker='s', color='green', alpha=trend_alpha)
      ax.scatter(neg_trend.index, neg_trend['signal_base'], label=None, marker='s', color='red', alpha=trend_alpha)
      ax.scatter(wave_trend.index, wave_trend['signal_base'], label=None, marker='s', color='orange', alpha=trend_alpha/2)

  # legend and title
  ax.legend(loc='upper left')  
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='both', linestyle='-', linewidth=0.5)

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot candlestick chart
def plot_candlestick(
  df, start=None, end=None, date_col='Date', ohlcv_col=default_ohlcv_col, 
  width=0.8, color=default_candlestick_color, 
  use_ax=None, plot_args=default_plot_args):
  """
  Plot candlestick chart

  :param df: dataframe with price and signal columns
  :param start: start row to plot
  :param end: end row to stop
  :param date_col: columnname of the date values
  :param ohlcv_col: columns names of Open/High/Low/Close/Volume
  :param width: width of candlestick
  :param color: up/down color of candlestick
  :param use_ax: the already-created ax to draw on
  :param plot_args: other plot arguments
  :returns: a candlestick chart
  :raises: none
  """
  # copy dataframe within a specific period
  df = df[start:end].copy()
  
  # for gap which start before 'start_date'
  min_idx = df.index.min()
  if df.loc[min_idx, 'candle_gap_top'] > df.loc[min_idx, 'candle_gap_bottom']:
    df.loc[min_idx, 'candle_gap'] = df.loc[min_idx, 'candle_gap_color'] * 2
  
  # set column names
  open = ohlcv_col['open']
  high = ohlcv_col['high']
  low = ohlcv_col['low']
  close = ohlcv_col['close']
  # volume = ohlcv_col['volume']
  padding = (df.High.max() - df.Low.min()) / 100

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
  
  # annotate split
  if 'Split' in df.columns:
    splited = df.query('Split != 1.0').index
    all_idx = df.index.tolist()

    for s in splited:
      x = s
      x_text = all_idx[max(0, all_idx.index(s)-2)]
      y = df.loc[s, 'High']
      y_text = y + df.High.max()*0.1
      sp = round(df.loc[s, 'Split'], 4)
      plt.annotate(f'splited {sp}', xy=(x, y), xytext=(x_text,y_text), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='->', alpha=0.5), bbox=dict(boxstyle="round",fc="1.0", alpha=0.5))
  
  # get indexes and max index
  idxs = df.index.tolist()
  max_idx = idxs[-1]
  max_x = max_idx + datetime.timedelta(days=1)

  # annotate gaps
  gap_idxs = df.query('candle_gap == 2 or candle_gap == -2').index
  for idx in gap_idxs:

    # gap start and it top/bottom
    start = idx
    top_value = df.loc[start, 'candle_gap_top']
    bottom_value = df.loc[start, 'candle_gap_bottom']

    gap_color = 'yellow' if df.loc[start, 'candle_gap'] > 0 else 'purple' # 
    # gap_hatch = '/' if df.loc[start, 'candle_gap'] > 0 else '\\'
    gap_hatch_color = 'green' if df.loc[start, 'candle_gap'] > 0 else 'red'
    
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
    ax.fill_between(df[pre_start:end].index, top_value, bottom_value, facecolor=gap_color, interpolate=True, alpha=0.25, edgecolor=gap_hatch_color, linewidth=3) #,  

  y_resistant = None
  y_text_resistant = None
  y_close = None
  y_text_close = None
  y_support = None
  y_text_support = None

  # annotate resistant
  resistant = df.query('resistant == resistant')
  if len(resistant) > 0:
    resistant_to_plot = resistant.tail(30)
    ax.plot(resistant_to_plot.index, resistant_to_plot['resistant'], color='red', linestyle='--', label='resistant')
    y_resistant = df.loc[max_idx, 'resistant'].round(2)
    y_text_resistant = y_resistant
    plt.annotate(f'{y_resistant}', xy=(max_x, y_text_resistant), xytext=(max_x, y_text_resistant), fontsize=13, xycoords='data', textcoords='data', color='black', va='bottom',  ha='left', bbox=dict(boxstyle="round", facecolor='red', alpha=0.1))

  # annotate close price
  y_close_padding = padding*5
  y_close = df.loc[max_idx, 'Close'].round(2)
  y_text_close = y_close
  if y_text_resistant is not None:
    diff = y_text_resistant - y_text_close
    if diff > 0 and diff < y_close_padding:
      y_text_close -= y_close_padding
    elif diff < 0 and diff > -y_close_padding:
      y_text_close += y_close_padding
  # rate = (df.loc[max_idx, 'rate'] * 100).round(2)
  close_color = 'blue'
  plt.annotate(f'{y_close}', xy=(max_x, y_text_close), xytext=(max_x, y_text_close), fontsize=13, xycoords='data', textcoords='data', color='black', va='center',  ha='left', bbox=dict(boxstyle="round", facecolor=close_color, alpha=0.1))
  
  # annotate support 
  support = df.query('support == support')
  if len(support) > 0:
    support_to_plot = support.tail(30)
    ax.plot(support_to_plot.index, support_to_plot['support'], color='green', linestyle='--', label='support')
    y_support = df.loc[max_idx, 'support'].round(2)
    y_text_support = y_support
    diff = y_text_close - y_text_support
    if diff > 0 and diff < y_close_padding:
      y_text_support -= y_close_padding
    elif diff < 0 and diff > -y_close_padding:
      y_text_support += y_close_padding
    plt.annotate(f'{y_support}', xy=(max_x, y_text_support), xytext=(max_x, y_text_support), fontsize=13, xycoords='data', textcoords='data', color='black', va='top',  ha='left', bbox=dict(boxstyle="round", facecolor='green', alpha=0.1))

  # settings for annotate candle patterns
  pattern_info = {
    '窗口_day': {1: '窗口', -1: '窗口'},
    '反弹_day': {1: '反弹', -1: '回落'},
    '突破_day': {1: '突破', -1: '跌落'},
    '启明黄昏_day': {1: '启明星', -1: '黄昏星'},

    # '腰带_day': {1: '腰带', -1: '腰带'},
    # '十字星_day': {1: '高浪线', -1: '十字星'},
    '锤子_day': {1: '锤子', -1: '吊颈'},
    '流星_day': {1: '倒锤', -1: '流星'},

    # '平头_day': {1: '平底', -1: '平顶'},
    '穿刺_day': {1: '穿刺', -1: '乌云'},
    # '吞噬_day': {1: '吞噬', -1: '吞噬'},
    # '包孕_day': {1: '包孕', -1: '包孕'},

    'linear_bounce_day': {1: '反弹', -1: '回落'},
    'linear_break_day': {1: '突破', -1: '跌落'}
  }
  settings = {
    'normal': {'fontsize':12, 'fontcolor':'black', 'va':'center', 'ha':'center', 'up':'green', 'down':'red', 'alpha': 0.15},
    'emphasis': {'fontsize':12, 'fontcolor':'black', 'va':'center', 'ha':'center', 'up':'yellow', 'down':'purple', 'alpha': 0.15},
  }
  up_pattern_annotations = {}
  down_pattern_annotations = {}
  for p in pattern_info.keys():
    stn = 'normal' if p not in ['窗口_day', '突破_day', '反弹_day'] else 'emphasis'
    if p in df.columns:
      tmp_up_idx = df.query(f'{p} == 1').index
      tmp_down_idx = df.query(f'{p} == -1').index

      # positive patterns
      tmp_up_info = pattern_info[p][1]
      if len(tmp_up_info) > 0:
        for i in tmp_up_idx:
          k = util.time_2_string(i.date())
          if k not in up_pattern_annotations: 
            up_pattern_annotations[k] = {'x': k, 'y': df.loc[i, 'Low'] - padding, 'text': tmp_up_info, 'stn': stn}
          else:
            up_pattern_annotations[k]['text'] = up_pattern_annotations[k]['text']  + f'/{tmp_up_info}'
            if up_pattern_annotations[k]['stn'] == 'normal':
              up_pattern_annotations[k]['stn'] = stn 

      # negative patterns
      tmp_down_info = pattern_info[p][-1]
      if len(tmp_down_info) > 0:
        for i in tmp_down_idx:
          k = util.time_2_string(i.date())
          if k not in down_pattern_annotations:
            down_pattern_annotations[k] = {'x': k, 'y': df.loc[i, 'High'] + padding, 'text': tmp_down_info, 'stn': stn}
          else:
            down_pattern_annotations[k]['text'] = down_pattern_annotations[k]['text']  + f'/{tmp_down_info}'
            if down_pattern_annotations[k]['stn'] == 'normal':
              down_pattern_annotations[k]['stn'] = stn 

  # candle pattern annotation
  annotations = {'up': up_pattern_annotations, 'down': down_pattern_annotations}
  y_text_padding = {0 : padding*0, 1: padding*5}
  for a in annotations.keys():
    
    # sort dictionary by date
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
      if a == 'up':
        y_text = df.Low.min() - y_text_padding[counter % 2]
      else:
        y_text = df.High.max() + y_text_padding[counter % 2]
        
      text = tmp_annotation[k]['text']
      stn = settings[tmp_annotation[k]['stn']]
      plt.annotate(f'{text}', xy=(x, y), xytext=(x,y_text), fontsize=stn['fontsize'], rotation=0, color=stn['fontcolor'], va=stn['va'],  ha=stn['ha'], xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='-|>', alpha=0.3, color='black'), bbox=dict(boxstyle="round", facecolor=stn[a], alpha=stn['alpha']))
      counter += 1

  # transform date to numbers
  df.reset_index(inplace=True)
  df[date_col] = df[date_col].apply(mdates.date2num)
  plot_data = df[[date_col, open, high, low, close]]
  
  # plot candlesticks
  candlestick_ohlc(
    ax=ax, quotes=plot_data.values, width=width, 
    colorup=color['colorup'], colordown=color['colordown'], alpha=color['alpha']
  )
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot ichimoku chart
def plot_main_indicators(
  df, start=None, end=None, date_col='Date', ohlcv_col=default_ohlcv_col, 
  target_indicator = ['price', 'ichimoku', 'kama', 'candlestick', 'bb', 'psar', 'renko', 'linear'],
  candlestick_width=0.8, candlestick_color=default_candlestick_color, 
  use_ax=None, title=None, plot_args=default_plot_args):
  """
  Plot ichimoku chart

  :param df: dataframe with ichimoku indicator columns
  :param start: start row to plot
  :param end: end row to plot
  :param date_col: column name of Date
  :param ohlcv_col: columns names of Open/High/Low/Close/Volume
  :param candlestick_width: width of candlestick
  :param candlestick_color: up/down color of candlestick
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
 
  # plot close price
  if 'price' in target_indicator:
    alpha = 0.2
    ax.plot(df.index, df[default_ohlcv_col['close']], label='close', color='black', linestyle='--', alpha=alpha)
  
  # plot senkou lines, clouds, tankan and kijun
  if 'ichimoku' in target_indicator:
    # alpha = 0.2
    # ax.plot(df.index, df.senkou_a, label='senkou_a', color='green', alpha=alpha)
    # ax.plot(df.index, df.senkou_b, label='senkou_b', color='red', alpha=alpha)
    # ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a > df.senkou_b, facecolor='green', interpolate=True, alpha=alpha)
    # ax.fill_between(df.index, df.senkou_a, df.senkou_b, where=df.senkou_a <= df.senkou_b, facecolor='red', interpolate=True, alpha=alpha)

    alpha = 0.6
    ax.plot(df.index, df.tankan, label='tankan', color='magenta', linestyle='--', alpha=alpha)
    ax.plot(df.index, df.kijun, label='kijun', color='blue', linestyle='--', alpha=alpha)
  
  # plot kama_fast/slow lines 
  if 'kama' in target_indicator:
    alpha = 0.6
    ax.plot(df.index, df.kama_fast, label='kama_fast', color='magenta', alpha=alpha)
    ax.plot(df.index, df.kama_slow, label='kama_slow', color='blue', alpha=alpha)
  
  # plot bollinger bands
  if 'bb' in target_indicator:
    alpha = 0.6
    ax.plot(df.index, df.bb_high_band, label='bb_high_band', color='green', alpha=alpha)
    ax.plot(df.index, df.bb_low_band, label='bb_low_band', color='red', alpha=alpha)
    ax.plot(df.index, df.mavg, label='mavg', color='grey', alpha=alpha)
    ax.fill_between(df.index, df.mavg, df.bb_high_band, facecolor='green', interpolate=True, alpha=0.1)
    ax.fill_between(df.index, df.mavg, df.bb_low_band, facecolor='red', interpolate=True, alpha=0.2)
  
  # plot psar dots
  if 'psar' in target_indicator:
    alpha = 0.6
    s = 10
    ax.scatter(df.index, df.psar_up, label='psar', color='green', alpha=alpha, s=s, marker='o')
    ax.scatter(df.index, df.psar_down, label='psar', color='red', alpha=alpha, s=s, marker='o')

  # plot renko bricks
  if 'renko' in target_indicator:
    ax = plot_renko(df, use_ax=ax, plot_args=default_plot_args, plot_in_date=True, close_alpha=0)

  # plot high/low trend
  if 'linear' in target_indicator:
    # plot aroon_up/aroon_down lines 
    ax.plot(df.index, df.linear_fit_high, label='linear_fit_high', color='black', alpha=0.5)
    ax.plot(df.index, df.linear_fit_low, label='linear_fit_low', color='black', alpha=0.5)

    # fill between linear_fit_high and linear_fit_low
    up_direction = df.linear_direction == 'u'
    down_direction = df.linear_direction == 'd'
    none_direction = df.linear_direction == 'n'
    ax.fill_between(df.index, df.linear_fit_high, df.linear_fit_low, where=up_direction, facecolor='green', interpolate=True, alpha=0.1)
    ax.fill_between(df.index, df.linear_fit_high, df.linear_fit_low, where=down_direction, facecolor='red', interpolate=True, alpha=0.1)
    ax.fill_between(df.index, df.linear_fit_high, df.linear_fit_low, where=none_direction, facecolor='yellow', interpolate=True, alpha=0.1)

  # plot candlestick
  if 'candlestick' in target_indicator:
    ax = plot_candlestick(df=df, start=start, end=end, date_col=date_col, ohlcv_col=ohlcv_col, width=candlestick_width, color=candlestick_color, use_ax=ax, plot_args=plot_args)
  
  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='x', linestyle=':', linewidth=0.5)

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot aroon chart
def plot_aroon(
  df, start=None, end=None, ohlcv_col=default_ohlcv_col, 
  use_ax=None, title=None, plot_args=default_plot_args):
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

  # return ax
  if use_ax is not None:
    return ax

# plot adx chart
def plot_adx(
  df, start=None, end=None, ohlcv_col=default_ohlcv_col, 
  use_ax=None, title=None, plot_args=default_plot_args):
  """
  Plot adx chart

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

  # # plot pdi/mdi/adx 
  # ax.plot(df.index, df.pdi, label='pdi', color='green', marker='.', alpha=0.3)
  # ax.plot(df.index, df.mdi, label='mdi', color='red', marker='.', alpha=0.3)
  # ax.plot(df.index, df.adx, label='adx', color='black', marker='.', alpha=0.3)
  
  # # fill between pdi/mdi
  # ax.fill_between(df.index, df.pdi, df.mdi, where=df.pdi > df.mdi, facecolor='green', interpolate=True, alpha=0.1)
  # ax.fill_between(df.index, df.pdi, df.mdi, where=df.pdi <= df.mdi, facecolor='red', interpolate=True, alpha=0.1)

  # plot adx_diff
  ax.plot(df.index, df['adx_diff'], label='adx_diff', color='black', linestyle='--', marker='.', alpha=0.5)
  df['zero'] = 0
  ax.fill_between(df.index, df.adx_diff, df.zero, where=df.adx_diff > df.zero, facecolor='green', interpolate=True, alpha=0.1)
  ax.fill_between(df.index, df.adx_diff, df.zero, where=df.adx_diff <= df.zero, facecolor='red', interpolate=True, alpha=0.1)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='both', linestyle='--', linewidth=0.5)

  # return ax
  if use_ax is not None:
    return ax

# plot renko chart
def plot_renko(
  df, start=None, end=None, ohlcv_col=default_ohlcv_col, 
  use_ax=None, title=None, plot_args=default_plot_args, plot_in_date=True, close_alpha=0.5, 
  save_path=None, save_image=False, show_image=False):

  # copy data frame
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')
    
  # plot close for displaying the figure
  ax.plot(df.Close, alpha=close_alpha)

  # whether to plot in date axes
  min_idx = df.index.min()
  max_idx = df.index.max()
  if df.loc[min_idx, 'renko_real'] != 'green' or df.loc[min_idx, 'renko_real'] != 'red':
    df.loc[min_idx, 'renko_real'] = df.loc[min_idx, 'renko_color'] 
    df.loc[min_idx, 'renko_countdown_days'] = df.loc[min_idx, 'renko_countdown_days'] 
  
  if plot_in_date:
    df = df.query('renko_real == "green" or renko_real =="red"').copy()
  else:
    df = df.query('renko_real == "green" or renko_real =="red"').reset_index()
  
  # plot renko
  legends = {'u': 'u', 'd': 'd', 'n':'n', '':''}
  for index, row in df.iterrows():
    renko = Rectangle((index, row['renko_o']), row['renko_countdown_days'], row['renko_brick_height'], facecolor=row['renko_color'], edgecolor=None, linestyle='-', linewidth=2, fill=True, alpha=0.15, label=legends[row['renko_trend']]) #  edgecolor=row['renko_color'], linestyle='-', linewidth=5, 
    legends[row['renko_trend']] = "_nolegend_"
    ax.add_patch(renko)
  
  # modify axes   
  if not plot_in_date:
    ax.get_figure().canvas.draw()
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(xlabels)):  
      try:
        idx = int(xlabels[i])
        if idx in df.index:
          xlabels[i] = f'{df.loc[idx, "Date"].date()}'
        else:
          xlabels[i] = f'{df.index.max().date()}'
          continue
      except Exception as e:
        continue
    ax.set_xticklabels(xlabels)
  
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

# plot volume
def plot_bar(
  df, target_col, start=None, end=None, width=0.8, alpha=1, color_mode='up_down', benchmark=None, 
  add_line=False, title=None, use_ax=None, plot_args=default_plot_args):

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
    ax.bar(df.index, height=df[target_col], width=width,color=df.color, alpha=alpha, label=target_col)

  if add_line:
    ax.plot(df.index, df[target_col], color='black', linestyle='--', marker='.', alpha=alpha)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='y', linestyle='-', linewidth=1)

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot overall trend of ta indicators
def plot_trend_idx(
  df, start=None, end=None, title=None, use_ax=None, plot_args=default_plot_args):

  # copy dataframe within a specific period
  df = df[start:end].copy()

  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  # plot trend_idx and trend_idx_ma
  if 'trend_idx' in df.columns:
    df['0'] = 0
    ax.plot(df.index, df['trend_idx'], color='black', linestyle='-', label='trend_idx', alpha=0.1)

    above_0 = df.query('trend_idx > 0')
    below_0 = df.query('trend_idx <= 0')

    alpha=0.25
    ax.fill_between(df.index, df['trend_idx'], df['0'], where=df['trend_idx'] > 0, facecolor='green', interpolate=True, alpha=alpha)
    ax.fill_between(df.index, df['trend_idx'], df['0'], where=df['trend_idx'] < 0, facecolor='red', interpolate=True, alpha=alpha)
  if 'trend_idx_ma' in df.columns:
    ax.plot(df.index, df['trend_idx_ma'], color='blue', linestyle='--', label='trend_idx_ma', alpha=0.5)

  # title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])
  ax.grid(True, axis='y', linestyle='-', linewidth=1)

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot general ta indicators
def plot_indicator(
  df, target_col, start=None, end=None, price_col='Close', price_alpha=1,
  signal_col='signal', signal_val=default_signal_val, 
  plot_price_in_twin_ax=False, plot_signal_on_price=None,
  benchmark=None, boundary=None, color_mode=None, 
  use_ax=None, title=None, plot_args=default_plot_args):
  """
  Plot indicators around a benchmark

  :param df: dataframe which contains target columns
  :param target_col: columnname of the target indicator
  :param start: start date of the data
  :param end: end of the data
  :param price_col: columnname of the price
  :param plot_price_in_twin_ax: whether plot price and signal in a same ax or in a twin ax
  :param signal_col: columnname of signal values
  :param signal_val: values of different kind of signals
  :param plot_signal_on_price: if not None, plot signal on (true) or under (false) price line 
  :param benchmark: benchmark, a fixed value
  :param boundary: upper/lower boundaries, a list of fixed values
  :param color_mode: which color mode to use: benckmark/up_down
  :param use_ax: the already-created ax to draw on
  :param title: title of the plot
  :param plot_args: other plot arguments
  :returns: figure with indicators and close price plotted
  :raises: none
  """
  # select data
  df = df[start:end].copy()
  # df = df.fillna(0)
  
  # create figure
  ax = use_ax
  if ax is None:
    fig = mpf.figure(figsize=plot_args['figsize'])
    ax = fig.add_subplot(1,1,1, style='yahoo')

  # plot benchmark
  if benchmark is not None:
    df['benchmark'] = benchmark
    ax.plot(df.index, df['benchmark'], color='black', linestyle='--', label='%s'%benchmark, alpha=0.5)

  if boundary is not None:
    if len(boundary) > 0:
      df['upper_boundary'] = max(boundary)
      df['lower_boundary'] = min(boundary)
      ax.plot(df.index, df['upper_boundary'], color='green', linestyle='--', label='%s'% max(boundary), alpha=0.5)
      ax.plot(df.index, df['lower_boundary'], color='red', linestyle='--', label='%s'% min(boundary), alpha=0.5)

  # plot indicator(s)
  unexpected_col = [x for x in target_col if x not in df.columns]
  if len(unexpected_col) > 0:
    print('column not found: ', unexpected_col)
  target_col = [x for x in target_col if x in df.columns]
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
      df.loc[df[tar] >= df[previous_target_col], 'color'] = 'green'

    # plot in benchmark mode
    elif color_mode == 'benchmark' and benchmark is not None:
      df['color'] = 'red'
      df.loc[df[tar] > benchmark, 'color'] = 'green'

    # plot indicator
    if 'color' in df.columns:
      ax.bar(df.index, height=df[tar], color=df.color, alpha=0.3)

  # plot close price
  plot_on_price = plot_signal_on_price if plot_signal_on_price is not None else True
  if price_col in df.columns:
    if plot_price_in_twin_ax:
      ax2=ax.twinx()
      plot_signal(df, price_col=price_col, price_alpha=price_alpha, signal_col=signal_col, signal_val=signal_val, plot_on_price=plot_on_price, use_ax=ax2)
      ax2.legend(loc='lower left')
    else:
      plot_signal(df, price_col=price_col, price_alpha=price_alpha, signal_col=signal_col, signal_val=signal_val, plot_on_price=plot_on_price, use_ax=ax)

  # plot title and legend
  ax.legend(bbox_to_anchor=plot_args['bbox_to_anchor'], loc=plot_args['loc'], ncol=plot_args['ncol'], borderaxespad=plot_args['borderaxespad']) 
  ax.set_title(title, rotation=plot_args['title_rotation'], x=plot_args['title_x'], y=plot_args['title_y'])

  ax.yaxis.set_ticks_position(default_plot_args['yaxis_position'])

  # return ax
  if use_ax is not None:
    return ax

# plot multiple indicators on a same chart
def plot_multiple_indicators(
  df, args={}, start=None, end=None, save_path=None, save_image=False, show_image=False, 
  title=None, width=25, unit_size=4, wspace=0, hspace=0.15, subplot_args=default_plot_args):
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
  
  # create figures
  fig = plt.figure(figsize=(width, num_indicators*unit_size))  
  gs = gridspec.GridSpec(num_indicators, 1, height_ratios=ratios)
  gs.update(wspace=wspace, hspace=hspace)

  axes = {}
  for i in range(num_indicators):
    tmp_indicator = indicators[i]
    tmp_args = args.get(tmp_indicator)

    zorder = 10 if tmp_indicator == 'main_indicators' else 1

    if i == 0:
      axes[tmp_indicator] = plt.subplot(gs[i], zorder=zorder)     
    else:
      axes[tmp_indicator] = plt.subplot(gs[i], sharex=axes[indicators[0]], zorder=zorder)
      
    if i%2 != 0:
      axes[tmp_indicator].xaxis.set_ticks_position("top")
      plt.setp(axes[tmp_indicator].get_xticklabels(), visible=False)
      axes[tmp_indicator].patch.set_alpha(0.5)

    axes[tmp_indicator].spines['top'].set_alpha(0.2)
    axes[tmp_indicator].spines['bottom'].set_alpha(0.2)
    axes[tmp_indicator].spines['left'].set_alpha(0.2)
    axes[tmp_indicator].spines['right'].set_alpha(0.2)

    # get extra arguments
    target_col = tmp_args.get('target_col')
    price_col = tmp_args.get('price_col')
    price_alpha = tmp_args.get('price_alpha')
    signal_col = tmp_args.get('signal_col')
    signal_val = tmp_args.get('signal_val')
    benchmark = tmp_args.get('benchmark')
    boundary = tmp_args.get('boundary')
    color_mode = tmp_args.get('color_mode')
    plot_signal_on_price = tmp_args.get('plot_signal_on_price')
    plot_price_in_twin_ax = tmp_args.get('plot_price_in_twin_ax')
    price_alpha = price_alpha if price_alpha is not None else 1
    signal_val = signal_val if signal_val is not None else default_signal_val
    plot_price_in_twin_ax = plot_price_in_twin_ax if plot_price_in_twin_ax is not None else False
    
    # plot ichimoku with candlesticks
    if tmp_indicator == 'main_indicators':
      # get candlestick width and color
      candlestick_color = tmp_args.get('candlestick_color') if tmp_args.get('candlestick_color') is not None else default_candlestick_color
      width = tmp_args.get('candlestick_width') if tmp_args.get('candlestick_width') is not None else 1
      target_indicator = tmp_args.get('target_indicator') if tmp_args.get('target_indicator') is not None else ['price']

      plot_main_indicators(
        df=plot_data, target_indicator=target_indicator, candlestick_width=width, candlestick_color=candlestick_color,
        use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args)

    # plot aroon
    elif tmp_indicator == 'aroon':
      plot_aroon(df=plot_data, ohlcv_col=default_ohlcv_col, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args)

    # plot adx
    elif tmp_indicator == 'adx':
      plot_adx(df=plot_data, ohlcv_col=default_ohlcv_col, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args)

    # plot renko
    elif tmp_indicator == 'renko':
      plot_renko(df=plot_data, ohlcv_col=default_ohlcv_col, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=subplot_args)

    # plot ta signals
    elif tmp_indicator == 'signals':
      signals = tmp_args.get('signal_list')
      signal_bases = []
      signal_names = []
      if signals is not None:
        for i in range(len(signals)):
          signal_name = signals[i]
          trend_name = signal_name.replace('signal', 'trend')
          signal_names.append(signal_name.split('_')[0])

          plot_data[f'signal_base_{signal_name}'] = i
          signal_bases.append(i)

          plot_signal(
            df=plot_data, price_col=f'signal_base_{signal_name}', price_alpha=price_alpha,
            signal_col=signal_name, trend_col=trend_name, signal_val=signal_val, 
            title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=subplot_args)

      # legend and title
      plt.ylim(ymin=min(signal_bases)-1 , ymax=max(signal_bases)+1)
      plt.yticks(signal_bases, signal_names)
      axes[tmp_indicator].legend().set_visible(False)

    # plot candlestick pattern signals
    elif tmp_indicator == 'candle':
      signals = tmp_args.get('signal_list')
      signal_bases = []
      signal_names = []
      if signals is not None:
        for i in range(len(signals)):
          signal_name = signals[i]
          trend_name = signal_name.replace('signal', 'trend')
          signal_names.append(signal_name.split('_')[0])

          plot_data[f'signal_base_{signal_name}'] = i
          signal_bases.append(i)

          plot_signal(
            df=plot_data, price_col=f'signal_base_{signal_name}', price_alpha=price_alpha,
            signal_col=signal_name, trend_col=trend_name, signal_val=signal_val, 
            title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=subplot_args)

      # legend and title
      plt.ylim(ymin=min(signal_bases)-1 , ymax=max(signal_bases)+1)
      plt.yticks(signal_bases, signal_names)
      axes[tmp_indicator].legend().set_visible(False)

    # plot Volume  
    elif tmp_indicator == 'volume':
      width = tmp_args.get('bar_width') if tmp_args.get('bar_width') is not None else 1
      alpha = tmp_args.get('alpha') if tmp_args.get('alpha') is not None else 1
      plot_bar(df=plot_data, target_col=target_col, width=width, alpha=alpha, color_mode=color_mode, benchmark=None, title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=default_plot_args)
    
    elif tmp_indicator == 'TA_overall':
      plot_trend_idx(plot_data, title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=default_plot_args)

    # plot renko
    elif tmp_indicator == 'renko':
      plot_in_date = tmp_args.get('plot_in_date') if tmp_args.get('plot_in_date') is not None else True
      plot_renko(plot_data, use_ax=axes[tmp_indicator], title=tmp_indicator, plot_args=default_plot_args, plot_in_date=plot_in_date)

    # plot other indicators
    else:
      plot_indicator(
        df=plot_data, target_col=target_col, 
        price_col=price_col, price_alpha=price_alpha, 
        signal_col=signal_col, signal_val=signal_val, 
        plot_price_in_twin_ax=plot_price_in_twin_ax, 
        plot_signal_on_price=plot_signal_on_price, 
        benchmark=benchmark, boundary=boundary, color_mode=color_mode,
        title=tmp_indicator, use_ax=axes[tmp_indicator], plot_args=subplot_args)

  # adjust plot layout
  max_idx = df.index.max()
  close_rate = (df.loc[max_idx, "rate"]*100).round(2)
  title_color = 'green' if close_rate > 0 else 'red'
  
  plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
  plt.rcParams['axes.unicode_minus'] = False

  # get name of the symbol
  new_title = args['sec_name'].get(title)
  rate = (df.loc[df.index.max(), "rate"] * 100).round(2)
  linear_desc = f'\n[拟合]: {df.loc[df.index.max(), "description"]}'
  candle_desc = f'\n[蜡烛]: {df.loc[df.index.max(), "candle_pattern_description"]}'
  candle_desc = '' if candle_desc[-2] == ':' else candle_desc
  if new_title is None:
    new_title == ''
  fig.suptitle(f'{title} - {new_title}({rate}%){linear_desc}{candle_desc}', color=title_color, x=0.5, y=0.97, fontsize=20)
  
  # save image
  if save_image and (save_path is not None):
    plt.savefig(save_path + title + '.png')
    
  # show image
  if show_image:
    plt.show()

  # close figures
  plt.cla()
  plt.clf()
  plt.close()

# calculate ta indicators, trend and derivatives for historical data
def plot_historical_evolution(df, symbol, interval, config, his_start_date=None, his_end_date=None, trend_indicators=['ichimoku', 'aroon', 'adx', 'psar'], volume_indicators=[], volatility_indicators=['bb'], other_indicators=[], signal_threshold=0.001, is_print=False, create_gif=False, plot_final=False, plot_save_path=None):
  """
  Calculate selected ta features for dataframe

  :param df: original dataframe with hlocv features
  :param symbol: symbol of the data
  :param interval: interval of the data
  :param trend_indicators: trend indicators
  :param volumn_indicators: volume indicators
  :param volatility_indicators: volatility indicators
  :param other_indicators: other indicators
  :param signal_threshold: threshold for kama/ichimoku trigerment
  :returns: dataframe with ta features, derivatives, signals
  :raises: None
  """
  # copy dataframe
  df = df.copy()
  today = util.time_2_string(time_object=df.index.max())
  
  if df is None or len(df) == 0:
    print(f'{symbol}: No data for calculate_ta_data')
    return None   
  else:
    data_start_date = util.string_plus_day(string=his_start_date, diff_days=-config['calculation']['look_back_window'][interval])
    df = df[data_start_date:]
    plot_start_date = data_start_date

  if create_gif or plot_final:
    if plot_save_path is None:
      print('Please specify plot save path in parameters, create_gif disable for this time')
      create_gif = False
    else:
      config['visualization']['show_image'] = False
      config['visualization']['save_image'] = True
      images = []
  
  try:
    # preprocess sec_data
    phase = 'preprocess_sec_data'
    df = preprocess_sec_data(df=df, symbol=symbol)
    
    # calculate TA indicators
    phase = 'cal_ta_indicators' 
    df = calculate_ta_indicators(df=df, trend_indicators=trend_indicators, volume_indicators=volume_indicators, volatility_indicators=volatility_indicators, other_indicators=other_indicators)

    # calculate TA trend
    phase = 'cal_ta_trend'
    df = calculate_ta_trend(df=df, trend_indicators=trend_indicators, volume_indicators=volume_indicators, volatility_indicators=volatility_indicators, other_indicators=other_indicators, signal_threshold=signal_threshold)

    # calculate TA derivatives for historical data for period [his_start_date ~ his_end_date]
    phase = 'cal_ta_derivatives(historical)'
    historical_ta_data = pd.DataFrame()
    ed = his_start_date
    while ed <= his_end_date:   

      # current max index     
      sd = util.string_plus_day(string=ed, diff_days=-config['visualization']['plot_window'][interval])
      current_max_idx = df[sd:ed].index.max()

      # next_ed = ed + 1day
      next_ed = util.string_plus_day(string=ed, diff_days=1)
      next_sd = util.string_plus_day(string=next_ed, diff_days=-config['visualization']['plot_window'][interval])
      next_max_idx = df[next_sd:next_ed].index.max()

      # if next_ed is weekend or holiday(on which no trading happened), skip; else do the calculation
      if next_max_idx != current_max_idx:
        
        # print current end_date
        if is_print:
          print(sd, ed)
        
        # calculate the dynamic part: linear features
        ta_data = calculate_ta_derivatives(df=df[sd:ed])
        ta_data = calculate_ta_signal(ta_data)
        historical_ta_data = historical_ta_data.append(ta_data.tail(1))

        # create image for gif
        if create_gif:
          visualization(df=ta_data, start=plot_start_date, title=f'{symbol}({ed})', save_path=plot_save_path, visualization_args=config['visualization'])
          images.append(f'{plot_save_path}{symbol}({ed}).png')

      # update ed
      ed = next_ed

    # append data
    historical_ta_data = ta_data.append(historical_ta_data)  
    df = util.remove_duplicated_index(df=historical_ta_data, keep='last')

    # create gif
    if create_gif:
      util.image_2_gif(image_list=images, save_name=f'{plot_save_path}{symbol}({his_start_date}-{his_end_date}).gif')

    # if plot final data
    if plot_final: 
      visualization(df=df, start=plot_start_date, title=f'{symbol}(final)', save_path=plot_save_path, visualization_args=config['visualization'])

  except Exception as e:
    print(symbol, phase, e)

  return df
