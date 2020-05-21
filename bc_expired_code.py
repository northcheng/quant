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