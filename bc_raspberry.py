# -*- coding: utf-8 -*-
"""
Utilities used in Google Colaboratory
:author: Beichen Chen
"""
import pandas as pd
import numpy as np
import time
import os
import pandas_datareader.data as web 
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from quant import bc_util as util


def get_symbols(remove_invalid=True, remove_not_fetched=True, not_fetched_list='drive/My Drive/probabilistic_model/yahoo_not_fetched_sec_code.csv'):
  """
  Get Nasdaq stock list
  :param remove_invalid: whether to remove invalid stock symbols
  :param remove_not_fetched: whether to remove the not-fetched stock symbols
  :param not_fetched_list: the not-fetched stock symbols list file
  :returns: dataframe of stock symbols
  :raises: exception when error reading not-fetched symbols list
  """
  # use pandas_datareader to get the symbols
  try:
    symbols = get_nasdaq_symbols()
    symbols = symbols.loc[symbols['Test Issue'] == False,]
  
  # when the pandas datareader is not accessible
  # download symbols from Nasdaq website directly
  except Exception as e:
    symbols = pd.read_table('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt', sep='|', index_col='Symbol').drop(np.NaN)
    symbols = symbols.loc[symbols['Test Issue'] == 'N',]
  sec_list = symbols.index.tolist()

  # remove invalid symbols
  if remove_invalid:
    original_len = len(sec_list)
    sec_list = [x for x in sec_list if '$' not in x]
    sec_list = [x for x in sec_list if '.' not in x]

  # remove not-fetched symbols
  if remove_not_fetched:
    original_len = len(sec_list)
    yahoo_not_fetched_list = []
    try: 
      yahoo_not_fetched_list = pd.read_csv(not_fetched_list).sec_code.tolist()
    except Exception as e:
      print(e)
    sec_list = [x for x in sec_list if x not in yahoo_not_fetched_list]
  
  return symbols.loc[sec_list, ]


def read_stock_data(sec_code, time_col, file_path='drive/My Drive/stock_data_us/', file_format='.csv', source='google_drive', start_date=None, end_date=None, drop_cols=[], drop_na=False, sort_index=True):
  """
  Read stock data from Google Drive or pandas_datareader
  :param sec_code: the target stock symbol
  :param time_col: time column in the stock data
  :param file-path: the path where stock file (.csv) stored
  :param file_format: the file format of stock data file
  :param source: where to read the data from: 'google_drive' or 'web'
  :param start_date: the start date to read
  :param end_date: the end date to read
  :param dro_cols: columns that will be dropped
  :param drop_na: whether to drop records that contains na values
  :param sort_index: whether to sort the data by index
  :returns: timeseries-dataframe of stock data
  :raises: exception when error reading data
  """
  try:
    # read data from google drive files
    if source == 'google_drive':
    
      # construct filename by sec_code, file_path and file_format
      filename = file_path + sec_code + file_format
      
      # if the file not exists, print information, return an empty dataframe
      if not os.path.exists(filename):
        print(filename, ' not exists')
        data = pd.DataFrame()
        
      # if the file exists
      else:
        # load file
        stage = 'reading_from_google_drive'
        if file_format == '.csv':
          data = pd.read_csv(filename, encoding='utf8', engine='python')
        elif file_format == '.xlsx':
          data = pd.read_excel(filename)

        # convert dataframe to timeseries dataframe
        stage = 'transforming_to_timeseries'
        data = util.df_2_timeseries(df=data, time_col=time_col)
        
        # handle invalid data
        stage = 'handling_invalid_data'
        # drop specific columns
        data.drop(drop_cols, axis=1, inplace=True)
        # drop na rows
        if drop_na:
          data.dropna(axis=1, inplace=True)
        # resort dataframe by its index (timeseries information)
        if sort_index:
          data.sort_index(inplace=True)

    # read data from pandas_datareader
    elif source == 'web':
      
      # download data from yahoo
      stage = 'reading_from_pandas_datareader'
      data = web.DataReader(sec_code, 'yahoo', start=start_date, end=end_date)
      
    # read data from other undefined sources
    else:
      print('source %s not found' % source)
      data = pd.DataFrame()
  except Exception as e:
    print(sec_code, stage, e)
    data = pd.DataFrame()

  return data[start_date:end_date]


def download_stock_data(sec_code, source, time_col, quote_client=None, download_limit=1200, start_date=None, end_date=None, file_path='drive/My Drive/stock_data_us/', file_format='.csv', is_return=False, is_print=True):
  """
  Download stock data from web sources
  :param sec_code: symbol of the stock to download
  :param source: the datasrouce to download data from
  :param time_col: time column in that data
  :param quote_client: quote client when using tiger_open_api
  :param download_limit: download_limit when using tiger_open_api
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param file_path: path to store the download data
  :param file_format: the format of file that data will be stored in
  :param is_return: whether to return the download data in dataframe format
  :param is_print: whether to print the download information
  :returns: dataframe is is_return=True
  :raises: none
  """
  # download stock data from yahoo finance api via pandas_datareader
  if source == 'yahoo':
    return download_stock_data_from_yahoo(sec_code=sec_code, time_col=time_col, start_date=start_date, end_date=end_date, file_path=file_path, file_format=file_format, is_return=is_return, is_print=is_print)
  # download stock data by using tiger open api
  elif source == 'tiger':
    return download_stock_data_from_tiger(sec_code=sec_code, time_col=time_col, quote_client=quote_client, download_limit=download_limit,  start_date=start_date, end_date=end_date, file_path=file_path, file_format=file_format, is_return=is_return, is_print=is_print)


def download_stock_data_from_yahoo(sec_code, time_col='Date', start_date=None, end_date=None, file_path='drive/My Drive/stock_data_us/', file_format='.csv', is_return=False, is_print=True):
  """
  Download stock data from Yahoo finance api via pandas_datareader
  :param sec_code: symbol of the stock to download
  :param time_col: time column in that data
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param file_path: path to store the download data
  :param file_format: the format of file that data will be stored in
  :param is_return: whether to return the download data in dataframe format
  :param is_print: whether to print the download information
  :returns: dataframe is is_return=True
  :raises: none
  """
  # construct filename by sec_code, file_path and file_format
  filename = file_path + sec_code + file_format
  
  # start downloading
  stage = 'downloading_started'
  try:
    # check whether historical data exists
    stage = 'loading_existed_data'
    data = pd.DataFrame()
    if os.path.exists(filename):
      data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col=time_col)
    
    # record the number of existed records, update the start_date
    init_len = len(data)
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y-%m-%d')

    # append new data onto the existed data
    stage = 'appending_new_data'
    tmp_data = web.DataReader(sec_code, 'yahoo', start=start_date, end=end_date)
    if len(tmp_data) > 0:
      data = data.append(tmp_data, sort=False)

      # save data
      stage = 'saving_data'
      data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
      data.to_csv(filename, index=False) 
      
    # calculate the number of new records
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('%(sec_code)s: %(first_date)s - %(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
        diff_len=diff_len, final_len=final_len, first_date=data[time_col].min().date(), latest_date=data[time_col].max().date(), sec_code=sec_code))
  except Exception as e:
      print(sec_code, stage, e)

  # return dataframe
  if is_return:
    data = util.df_2_timeseries(data, time_col=time_col)
    return data 


def download_stock_data_from_tiger(sec_code, time_col='time', quote_client=None, download_limit=1200, start_date=None, end_date=None, file_path='drive/My Drive/stock_data_us/', file_format='.csv', is_return=False, is_print=True):
  """
  Download stock data from Tiger Open API
  :param sec_code: symbol of the stock to download
  :param time_col: time column in that data
  :param quote_client: quote_client used for querying data from API
  :param download limit: the limit of number of records in each download
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param file_path: path to store the download data
  :param file_format: the format of file that data will be stored in
  :param is_return: whether to return the download data in dataframe format
  :param is_print: whether to print the download information
  :returns: dataframe is is_return=True
  :raises: none
  """  
  # construct filename by sec_code, file_path and file_format
  filename = file_path + sec_code + file_format
  
  # start downloading
  stage = 'downloading_started'
  try:
    # check whether historical data exists
    stage = 'loading_existed_data'
    data = pd.DataFrame()
    if os.path.exists(filename):  
      data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col='Date')
      
    # record the number of existed records, update the start_date
    init_len = len(data)  
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y-%m-%d')
      
    # download data from tiger open api
    stage = 'downloading_new_data'

    # transfer start/end date to timestamp instance
    if start_date is not None:
      begin_time = round(time.mktime(util.string_2_time(start_date).timetuple()) * 1000)
    else:
      begin_time = 0
    if end_date is not None:
      end_time = round(time.mktime(util.string_2_time(end_date).timetuple()) * 1000)
    else:
      end_time = round(time.time() * 1000)

    # start downloading data
    tmp_len = download_limit
    new_data = pd.DataFrame()
    while tmp_len >= download_limit:  
      tmp_data = quote_client.get_bars([sec_code], begin_time=begin_time, end_time=end_time, limit=download_limit)
      tmp_len = len(tmp_data)
      new_data = tmp_data.append(new_data)
      end_time = int(tmp_data.time.min())
    
    # process newly downloaded data
    stage = 'processing_new_data'
    if len(new_data) > 0:
      new_data.drop('symbol', axis=1, inplace=True)
      new_data[time_col] = new_data[time_col].apply(lambda x: util.timestamp_2_time(x).date())
      new_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'time': 'Date'}, inplace=True)
      new_data['Adj Close'] = new_data['Close']
      time_col = 'Date'
      new_data = util.df_2_timeseries(df=new_data, time_col=time_col)
    
      # append new data onto existed data
      data = data.append(new_data, sort=False)

      # drop duplicated data
      stage = 'saving_data'
      data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
      data.sort_values(by=time_col,  )
      data.to_csv(filename, index=False) 
      
    # calculate the number of new records
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('[From Tiger]%(sec_code)s: %(first_date)s - %(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
        diff_len=diff_len, final_len=final_len, first_date=data[time_col].min().date(), latest_date=data[time_col].max().date(), sec_code=sec_code))
      
  except Exception as e:
    print(sec_code, stage, e)   
    
  # return dataframe
  if is_return:
    data = util.df_2_timeseries(data, time_col=time_col)
    return data
