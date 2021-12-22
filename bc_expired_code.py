# # expired
# from alpha_vantage.timeseries import TimeSeries
# def get_data_from_tiger(sec_code, interval, start_date=None, end_date=None, time_col='time', minute_level=False, quote_client=None, download_limit=1200, is_print=False):
#   """
#   Download stock data from Tiger Open API
#   :param sec_code: symbol of the stock to download
#   :param start_date: start date of the data
#   :param end_date: end date of the data 
#   :param time_col: time column in that data  
#   :param interval: period of data: day/week/month/year/1min/5min/15min/30min/60min
#   :param quote_client: quote_client used for querying data from API
#   :param download limit: the limit of number of records in each download
#   :param is_print: whether to print the download information
#   :returns: dataframe 
#   :raises: none
#   """  
#   try:     
#     # initialization
#     data = pd.DataFrame()
#     begin_time = 0
#     end_time = round(time.time() * 1000)

#     # transfer start/end date to timestamp instance
#     if start_date is not None:
#       begin_time = round(time.mktime(util.string_2_time(start_date).timetuple()) * 1000)
#     if end_date is not None:
#       end_time = round(time.mktime(util.string_2_time(end_date).timetuple()) * 1000)
      
#     # start downloading data
#     tmp_len = download_limit
#     while tmp_len >= download_limit:  
#       tmp_data = quote_client.get_bars([sec_code], begin_time=begin_time, end_time=end_time, period=interval, limit=download_limit)
#       tmp_len = len(tmp_data)
#       data = tmp_data.append(data)
#       end_time = int(tmp_data.time.min())
    
#     # process downloaded data
#     data_length = len(data)
#     if data_length > 0:
#       data.drop('symbol', axis=1, inplace=True)
      
#       # drop duplicated data
#       if minute_level:
#         data[time_col] = data[time_col].apply(lambda x: util.timestamp_2_time(x))
#       else:
#         data[time_col] = data[time_col].apply(lambda x: util.timestamp_2_time(x).date())
#       data = data.drop_duplicates(subset=time_col, keep='last')
#       data.sort_values(by=time_col,  inplace=True)
      
#       # change column names
#       data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'time': 'Date'}, inplace=True)
#       data['Adj Close'] = data['Close']
#       data = util.df_2_timeseries(data, time_col='Date')
      
#     # print download result
#     if is_print:
#       print('[From tiger]{sec_code}: {start} - {end}, 下载记录 {data_length}'.format(sec_code=sec_code, start=data.index.min().date(), end=data.index.max().date(), data_length=len(data)))
      
#   except Exception as e:
#     print(sec_code, e)   
    
#   # return dataframe
#   return data


# def update_stock_data_from_alphavantage(symbols, stock_data_path, api_key, file_format='.csv', required_date=None, is_print=False):
#   """
#   update local stock data from alphavantage

#   :param symbols: symbol list
#   :param stock_data_path: in where the local stock data files(.csv) are stored
#   :param api_key: api key for accessing alphavantage
#   :param file_format: default is .csv
#   :param required_date: if the local data have already meet the required date, it won't be updated
#   :param is_print: whether to print info when downloading
#   :returns: dataframe of latest stock data, per row each symbol
#   :raises: none
#   """
#   # get current date if required date is not specified
#   if required_date is None:
#     required_date = util.time_2_string(datetime.datetime.today())
  
#   # assume it will cost 1 api call (which is limitted to 5/min for free users)
#   api_call = 1
  
#   # go through symbols
#   for symbol in symbols:
#     download_info = f'{symbol}: '
  
#     # if stock data file already exists, load existing data
#     if os.path.exists(f'{stock_data_path}{symbol}{file_format}'):
      
#       # load existed data
#       old_data = load_stock_data(file_path=stock_data_path, file_name=symbol)
      
#       # check period between existing data and current date, if small than 100 days, download in compact mode
#       old_data_date = util.time_2_string(old_data.index.max())
#       download_info += f'exists({old_data_date}), '
      
#       # if existed data is uptodate, cancel the api call
#       diff_days = util.num_days_between(old_data_date, required_date)
#       if diff_days == 0:
#         download_info += f'up-to-date...'
#         api_call = 0
#       # else if it is in 100 days from required date, download in compact mode
#       elif diff_days > 0 and diff_days <= 100:
#         download_info += f'updating...'
#         outputsize='compact'
#       # otherwise redownload the whole data
#       else:
#         download_info += f'redownloading...'
#         outputsize='full'
      
#     # else if the local data is not exist, download in full mode
#     else:
#       download_info += 'not found, downloading...'
#       old_data = pd.DataFrame()
#       outputsize='full'
      
#     # download data
#     if api_call == 1:
#       new_data = get_data_from_alphavantage(symbol=symbol, api_key=api_key, outputsize=outputsize)
#     else:
#       new_data = pd.DataFrame()

#     # append new data to the old data
#     data = old_data.append(new_data, sort=True)
    
#     # remove duplicated index, keep the latest
#     data = util.remove_duplicated_index(df=data, keep='last')
  
#     # save data to the specified path with <symbol>.<file_format>
#     save_stock_data(df=data, file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True)
#     download_info += f'done, latest date({data.index.max().date()})'
    
#     # print download info
#     if is_print:
#       print(download_info)

#     return api_call


# def get_data_from_alphavantage(symbol, api_key, interval='d', start_date=None, end_date=None, time_col='Date', is_print=False, outputsize='compact'):
#   """
#   Download stock data from alpha vantage

#   :param symbol: symbol of the stock to download
#   :param api_key: alpha vantage api_key
#   :param interval: period of data: d/w/m
#   :param start_date: start date of the data
#   :param end_date: end date of the data
#   :param time_col: time column in that data
#   :param is_print: whether to print the download information
#   :param outputsize: either 'compact' for latest 100 records or 'full' for all records
#   :returns: dataframe 
#   :raises: none
#   """
#   try:
#     # get timeseries instance
#     ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='integer')

#     # set output size
#     if start_date is not None and end_date is not None:
#       diff_days = util.num_days_between(start_date, end_date)
#       if diff_days > 100:
#         outputsize = 'full'
    
#     # download data
#     if interval == 'd':
#       data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)  
#     elif interval == 'w':
#       data, meta_data = ts.get_weekly_adjusted(symbol=symbol)  
#     elif interval == 'm':
#       data, meta_data = ts.get_monthly_adjusted(symbol=symbol)  

#     # post process data: rename columns, transfer it to timeseries data
#     data.rename(columns={'index':'Date', '1. open':'Open', '2. high':'High', '3. low':'Low', '4. close':'Close', '5. adjusted close': 'Adj Close', '6. volume':'Volume', '7. dividend amount': 'Dividend', '8. split coefficient': 'Split'}, inplace=True)   
#     data = util.df_2_timeseries(df=data, time_col=time_col)

#     # print download result
#     if is_print:
#       print(f'[From AlphaVantage]{symbol}: {data.index.min().date()} - {data.index.max().date()}, 下载记录 {len(data)}')
  
#   except Exception as e:
#     print(symbol, e)
#     data = None

#   # return dataframe for selected period
#   return data[start_date:end_date]


## ichimoku and kama fastline/slowlines
# # fast / slow lines in each indicator
    # fast_slow_lines = {
    #   'kama': {'fast': 'kama_fast', 'slow': 'kama_slow'}, 
    #   'ichimoku': {'fast': 'tankan', 'slow': 'kijun'}
    # }

    # for indicator in main_indicators:

    #   # construct column names according to indicator name
    #   signal_col = f'{indicator}_signal'
    #   trend_col = f'{indicator}_trend'
    #   fl = fast_slow_lines[indicator]['fast']
    #   sl = fast_slow_lines[indicator]['slow']
    #   fld = f'{fl}_day'
    #   sld = f'{sl}_day'

    #   # calculate number of days since fast/slow line triggered
    #   for col in [fl, sl]:
    #     df[f'{col}_day'] = sda(series=df[f'{col}_signal'], zero_as=1)

## aroon peak falling signal
# # notice when aroon_down falls from the peak
      # df['apf_signal'] = 'n'

      # df['previous_aroon_down'] = df['aroon_down'].shift(1)
      # peak_idx = df.query('aroon_down_acc_change_count == -1').index
      # df.loc[peak_idx, 'aroon_down_peak'] = df.loc[peak_idx, 'previous_aroon_down']
      # df['aroon_down_peak'] = df['aroon_down_peak'].fillna(method='ffill')
      # up_idx = df.query('(aroon_down_peak==100 and aroon_down_acc_change_count<=-4 and aroon_up_acc_change_count<=-4 and aroon_gap<-32)  or (aroon_up==100)').index
      # df.loc[up_idx, 'apf_trend'] = 'u'

      # df['previous_aroon_up'] = df['aroon_up'].shift(1)
      # peak_idx = df.query('aroon_up_acc_change_count == -1').index
      # df.loc[peak_idx, 'aroon_up_peak'] = df.loc[peak_idx, 'previous_aroon_up']
      # df['aroon_up_peak'] = df['aroon_up_peak'].fillna(method='ffill')
      # down_idx = df.query('(aroon_up_peak==100 and aroon_up_acc_change_count<=-4 and aroon_down_acc_change_count<=-4 and aroon_gap>32) or (aroon_down==1001)').index
      # df.loc[down_idx, 'apf_trend'] = 'd'

      # # calculate aroon_gap same-direction-accumulation(sda)
      # df['aroon_sda'] = sda(series=df['aroon_gap'])
      # sell_idx = df.query('aroon_sda < -1000').index
      # buy_idx = df.query('aroon_sda > 1000').index
      # df.loc[buy_idx, 'aroon_sda_trend'] = 'u'
      # df.loc[sell_idx, 'aroon_sda_trend'] = 'd'
      # df['aroon_sda_signal'] = 'n'


# 2021-12-22
# original columns
'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividend', 'Split', 

# L1 derived columns
'symbol', 'rate', 
       
# L2 candlestick derived columns
'candle_color', 'candle_entity', 'candle_entity_top', 'candle_entity_bottom',
'candle_shadow', 'candle_upper_shadow', 'candle_lower_shadow', 
'candle_upper_shadow_pct', 'candle_lower_shadow_pct', 'candle_entity_pct', 
'candle_gap', 'candle_gap_color', 'candle_gap_top', 'candle_gap_bottom', 'candle_gap_height', 'candle_gap_support', 'candle_gap_resistant',

# L2 ichimoku derived columns
'tankan', 'kijun', 'senkou_a', 'senkou_b', 'chikan', 
'cloud_height', 'cloud_width', 'cloud_top', 'cloud_bottom', 
'tankan_signal', 'kijun_signal', 'tankan_to_kijun', 'tankan_kijun_signal', 

# L2 other technical indicators derived columns
'aroon_up', 'aroon_down', 'aroon_gap', 
'tr', 'atr',
'pdi', 'mdi', 
'dx', 'adx', 'adx_diff', 
'psar', 'psar_up', 'psar_down',
'mavg', 'mstd', 
'bb_high_band', 'bb_low_band', 

# L3 technical indicators trend derived columns
'tankan_day', 'tankan_rate', 'kijun_day', 'kijun_rate', 'tankan_day_plus_kijun_day',  
'adx_value', 'adx_value_change', 'adx_direction', 'adx_direction_day', 'prev_adx_extreme', 
'adx_strength', 'adx_strength_change', 'adx_power', 'adx_power_day', 'adx_strong_day',
'up_trend_idx', 'down_trend_idx', 'trend_idx', 
'ichimoku_trend', 'aroon_trend', 'adx_trend', 'psar_trend', 'stoch_trend','bb_trend',
'ichimoku_day', 'aroon_day', 'adx_day', 'psar_day', 'stoch_day', 'bb_day', 
'ichimoku_signal', 'aroon_signal', 'adx_signal', 'psar_signal', 'stoch_signal', 'bb_signal', 

# L3 renko derived columns
'renko_o', 'renko_h', 'renko_l', 'renko_c', 'renko_color', 'renko_real', 
'renko_brick_height', 'renko_brick_length', 'renko_brick_number', 
'renko_start', 'renko_end', 'renko_duration', 'renko_duration_p1', 'renko_countdown_days', 
'renko_series_short', 'renko_series_long', 'renko_series_short_idx', 'renko_series_long_idx',
'renko_position', 'renko_support', 'renko_resistant',
'renko_trend', 'renko_day', 'renko_signal',

# L3 linear fit derived columns
'linear_fit_high', 'linear_fit_low', 'linear_fit_high_slope', 'linear_fit_low_slope', 'linear_slope', 'linear_fit_high_signal', 'linear_fit_low_signal', 'linear_fit_high_stop', 'linear_fit_low_stop', 
'linear_day_count', 'linear_fit_resistant', 'linear_fit_support', 
'linear_direction', 
'linear_bounce_trend', 'linear_break_trend', 'linear_trend', 
'linear_bounce_day', 'linear_break_day', 
# '拟合_trend', 'linear_fit_idx', '拟合_signal', 
'linear_fit_category', 'linear_fit_description',
        
# L3 candlestick patterns derived columns
'位置_trend', '位置_day', 
'entity_trend', 'shadow_trend', 'upper_shadow_trend', 'lower_shadow_trend', 

'相对窗口位置', '十字星', '锤子', 
'窗口_trend', '反弹_trend', '突破_trend', '窗口_day', '突破_day', '反弹_day', 
'十字星_trend', '锤子_trend', '腰带_trend', '流星_trend', '平头_trend', '穿刺_trend', '吞噬_trend', '包孕_trend', '启明黄昏_trend', 
'十字星_day', '锤子_day', '腰带_day', '流星_day', '平头_day', '穿刺_day', '吞噬_day', '包孕_day', '启明黄昏_day', 
'candle_pattern_idx', 'candle_pattern_description',

# L4 ultimate results
'support', 'supporter', 
'resistant', 'resistanter', 
'ichimoku_mean', 
'trend', 'uncertain_trend', 'trend_day', 'signal',


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
 