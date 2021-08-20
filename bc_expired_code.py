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


# # 2021-08-20
# # original columns
# 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividend', 'Split', 

# # L1 derived columns
# 'symbol', 'rate', 
       
# # candlestick derived columns
# 'candle_color', 'candle_shadow', 'candle_entity', 
# 'candle_entity_top', 'candle_entity_bottom', 'candle_upper_shadow', 'candle_lower_shadow', 
# 'candle_upper_shadow_pct', 'candle_lower_shadow_pct', 'candle_entity_pct', 
# 'candle_gap', 'candle_gap_top', 'candle_gap_bottom', 'candle_gap_support', 'candle_gap_resistant', 

# # L2 ichimoku derived columns
# 'tankan', 'kijun', 'senkou_a', 'senkou_b', 'chikan', 
# 'cloud_height', 'cloud_width', 'cloud_top', 'cloud_bottom', 
# 'break_up', 'break_down', 
# 'close_to_kijun', 'close_to_tankan', 'close_to_cloud_top', 'close_to_cloud_bottom', 
# 'tankan_signal', 'kijun_signal', 'tankan_kijun_signal', 'cloud_top_signal', 'cloud_bottom_signal',

# # L2 other technical indicators derived columns
# 'aroon_up', 'aroon_down', 'aroon_gap', 
# 'tr', 'atr'
# 'pdi', 'mdi', 
# 'dx', 'adx', 'adx_diff', 
# 'psar', 'psar_up', 'psar_down',
# 'mavg', 'mstd', 
# 'bb_high_band', 'bb_low_band', 

# # L3 technical indicators trend derived columns
# 'ichimoku_trend', 'aroon_trend', 'adx_trend', 'psar_trend', 'bb_trend', 'trend_idx', 'up_trend_idx', 'down_trend_idx', 
# 'ichimoku_day', 'aroon_day', 'adx_day', 'psar_day', 'bb_day', 
# 'ichimoku_signal', 'aroon_signal', 'adx_signal', 'psar_signal', 'bb_signal', 

# # L3 renko derived columns
# 'renko_o', 'renko_h', 'renko_l', 'renko_c', 'renko_color', 'renko_brick_height', 'renko_brick_length', 'renko_brick_number', 
# 'renko_start', 'renko_end', 'renko_duration', 'renko_duration_p1','renko_real', 'renko_countdown_days', 'above_renko_h', 'among_renko', 'below_renko_l',
# 'renko_trend', 'renko_day', 'renko_signal',

# # L3 linear fit derived columns
# 'linear_fit_high', 'linear_fit_low', 'linear_fit_high_slope', 'linear_fit_low_slope', 'linear_slope', 'linear_fit_high_signal', 'linear_fit_low_signal', 'linear_fit_high_stop', 'linear_fit_low_stop', 
# 'linear_day_count', 'linear_fit_resistant', 'linear_fit_support', 
# 'linear_direction', 'price_direction', 'rate_direction',
# 'linear_trend', 'linear_day', 'linear_signal', 

# # L3 candlestick patterns derived columns
# 'position_signal', 'belt_signal', 'flat_signal', 'embrace_signal',
# 'position_trend', 'volume_trend',
# 'window_trend', 'window_position_trend', 
# 'color_trend', 'entity_trend',
# 'shadow_trend', 'upper_shadow_trend', 'lower_shadow_trend',
# 'hammer_trend', 'meteor_trend', 'belt_trend', 'cross_signal', 'cross_trend',
# 'flat_trend', 'wrap_trend', 'embrace_trend', 
# 'star_trend', 

# # L4 ultimate results
# 'support', 'resistant', 'support_signal', 'resistant_signal',
# 'candle_patterns', 'category', 'description', 'trend', 'signal_day', 'signal'