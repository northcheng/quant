# -*- coding: utf-8 -*-
"""
Utilities used for data IO

:author: Beichen Chen
"""

import pandas as pd
import numpy as np
import requests
import datetime
import zipfile
import time
import json
import os
import pandas_datareader.data as web 
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
try:
  import yfinance as yf
except Exception as e:
  print(e)
from quant import bc_util as util


#----------------------------- Stock Data -------------------------------------#
def get_symbols(remove_invalid=True, remove_not_fetched=False, not_fetched_list=None):
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
  
  # get list of all symbols
  sec_list = symbols.index.tolist()

  # remove invalid symbols
  if remove_invalid:
    original_len = len(sec_list)
    sec_list = [x for x in sec_list if '$' not in x]
    sec_list = [x for x in sec_list if '.' not in x]

  # remove not-fetched symbols
  if remove_not_fetched and not_fetched_list is not None:
    original_len = len(sec_list)
    yahoo_not_fetched_list = []
    try: 
      yahoo_not_fetched_list = pd.read_csv(not_fetched_list).sec_code.tolist()
    except Exception as e:
      print(e)
    sec_list = [x for x in sec_list if x not in yahoo_not_fetched_list]
  
  return symbols.loc[sec_list, ]


def get_data_from_yahoo(sec_code, interval='d', start_date=None, end_date=None, time_col='Date', is_print=False):
  """
  Download stock data from Yahoo finance api via pandas_datareader

  :param sec_code: symbol of the stock to download
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param time_col: time column in that data
  :param interval: period of data: d/w/m/v
  :param is_print: whether to print the download information
  :returns: dataframe 
  :raises: none
  """

  try:
    # download data
    data = web.get_data_yahoo(sec_code, start_date, end_date, interval=interval)
      
    # print download result
    if is_print:
      print('[From Yahoo]{sec_code}: {start} - {end}, 下载记录 {data_length}'.format(sec_code=sec_code, start=data.index.min().date(), end=data.index.max().date(), data_length=len(data)))

  except Exception as e:
      print(sec_code, e)

  # return dataframe
  return data


def get_data_from_yfinance(sec_code, interval='1d', start_date=None, end_date=None, time_col='Date', is_print=False):
  """
  Download stock data from Yahoo finance api via yfinance

  :param sec_code: symbol of the stock to download
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param time_col: time column in that data
  :param interval: period of data: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
  :returns: dataframe
  :raises: none
  """
  try:
    # download data
    ticker = yf.Ticker(sec_code)
    data = ticker.history(start=start_date, end=end_date, interval=interval).drop(columns=['Dividends', 'Stock Splits'])
    data['Adj Close'] = data['Close']
          
    # print download result
    if is_print:
      print('[From YFinance]{sec_code}: {start} - {end}, 下载记录 {data_length}'.format(sec_code=sec_code, start=data.index.min().date(), end=data.index.max().date(), data_length=len(data)))

  except Exception as e:
      print(sec_code, e)

  # return dataframe
  return data 


def get_data_from_tiger(sec_code, interval, start_date=None, end_date=None, time_col='time', quote_client=None, download_limit=1200, is_print=False):
  """
  Download stock data from Tiger Open API
  :param sec_code: symbol of the stock to download
  :param start_date: start date of the data
  :param end_date: end date of the data 
  :param time_col: time column in that data  
  :param interval: period of data: day/week/month/year/1min/5min/15min/30min/60min
  :param quote_client: quote_client used for querying data from API
  :param download limit: the limit of number of records in each download
  :param is_print: whether to print the download information
  :returns: dataframe 
  :raises: none
  """  
  try:     
    # initialization
    data = pd.DataFrame()
    begin_time = 0
    end_time = round(time.time() * 1000)

    # transfer start/end date to timestamp instance
    if start_date is not None:
      begin_time = round(time.mktime(util.string_2_time(start_date).timetuple()) * 1000)
    if end_date is not None:
      end_time = round(time.mktime(util.string_2_time(end_date).timetuple()) * 1000)
      
    # start downloading data
    tmp_len = download_limit
    while tmp_len >= download_limit:  
      tmp_data = quote_client.get_bars([sec_code], begin_time=begin_time, end_time=end_time, period=interval, limit=download_limit)
      tmp_len = len(tmp_data)
      data = tmp_data.append(data)
      end_time = int(tmp_data.time.min())
    
    # process downloaded data
    data_length = len(data)
    if data_length > 0:
      data.drop('symbol', axis=1, inplace=True)
      
      # drop duplicated data
      data[time_col] = data[time_col].apply(lambda x: util.timestamp_2_time(x).date())
      data = data.drop_duplicates(subset=time_col, keep='last')
      data.sort_values(by=time_col,  inplace=True)
      
      # change column names
      data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'time': 'Date'}, inplace=True)
      data['Adj Close'] = data['Close']
      data = util.df_2_timeseries(data, time_col='Date')
      
    # print download result
    if is_print:
      print('[From tiger]{sec_code}: {start} - {end}, 下载记录 {data_length}'.format(sec_code=sec_code, start=data.index.min().date(), end=data.index.max().date(), data_length=len(data)))
      
  except Exception as e:
    print(sec_code, e)   
    
  # return dataframe
  return data


def download_stock_data(sec_code, start_date=None, end_date=None, source='yahoo', time_col='Date', interval='d', quote_client=None, download_limit=1200, is_print=False, is_return=False, is_save=False, file_path=None, file_name=None):
  """
  Download stock data from web sources

  :param sec_code: symbol of the stock to download
  :param file_path: path to store the download data
  :param file_name: name of the stock data file
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param source: the datasrouce to download data from
  :param time_col: time column in that data
  :param interval: period of data, for yahoo: d/w/m/v; for tiger day/week/month/year/1min/5min/15min/30min/60min
  :param quote_client: quote client when using tiger_open_api
  :param download_limit: download_limit when using tiger_open_api
  :param is_return: whether to return the download data in dataframe format
  :param is_print: whether to print the download information
  :returns: dataframe is is_return=True
  :raises: none
  """
  try:
    # get data
    data = pd.DataFrame()

    # yahoo
    if source == 'yahoo':
      data = get_data_from_yahoo(sec_code=sec_code, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
    # yfinance
    elif source == 'yfinance':
      data = data = get_data_from_yfinance(sec_code=sec_code, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
    # dtigeropen
    elif source == 'tiger':
      data = get_data_from_tiger(sec_code=sec_code, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print, quote_client=quote_client ,download_limit=download_limit)

    # save data
    if is_save:
      # construct file_name by sec_code, file_path and file_format
      if file_name is None:
        file_name = '{path}{name}.csv'.format(path=file_path, name=sec_code)
      else:
        file_name = '{path}{name}.csv'.format(path=file_path, name=file_name)
      
      # if dataframe is not empty, save it into a csv file
      data_length = len(data)
      if data_length > 0:
        data.reset_index().drop_duplicates(subset=time_col, keep='last').to_csv(file_name, index=False)  

  except Exception as e:
    print(sec_code, e)

  if is_return:
    return data


def read_stock_data(sec_code, time_col, file_path, file_name=None, start_date=None, end_date=None, drop_na=False, sort_index=True):
  """
  Read stock data from Google Drive

  :param sec_code: the target stock symbol
  :param time_col: time column in the stock data
  :param file_path: the path where stock file (.csv) stored
  :param file_name: name of the stock data file
  :param start_date: the start date to read
  :param end_date: the end date to read
  :param drop_na: whether to drop records that contains na values
  :param sort_index: whether to sort the data by index
  :returns: timeseries-dataframe of stock data
  :raises: exception when error reading data
  """
  # construct file_name by sec_code, file_path and file_format
  stage = 'initialization'
  if file_name is None:
    file_name = file_path + sec_code + '.csv'
  else:
    file_name = file_path + file_name + '.csv'

  # read data from google drive
  try:
    # if the file not exists, print information, return an empty dataframe
    if not os.path.exists(file_name):
      print(file_name, ' not exists')

    else:
      # load file
      stage = 'reading_from_google_drive'
      data = pd.read_csv(file_name, encoding='utf8', engine='python')

      # convert dataframe to timeseries dataframe
      stage = 'transforming_to_timeseries'
      data = util.df_2_timeseries(df=data, time_col=time_col)
      
      # handle invalid data
      stage = 'handling_invalid_data'
      if drop_na:
        data.dropna(axis=1, inplace=True)
      
      if sort_index:
        data.sort_index(inplace=True)

  except Exception as e:
    print(sec_code, stage, e)

  return data[start_date:end_date]


def remove_stock_data(sec_code, file_path, file_name=None):
  '''
  Remove stock data file from drive

  :param sec_code: symbol of the stock to download
  :param file_path: path to store the download data
  :param file_format: the format of file that data will be stored in
  :returns: None
  :raises: None
  '''
  if file_name is None:
    file_name = file_path + sec_code + '.csv'
  else:
    file_name = file_path + file_name + '.csv'
  
  try:
    os.remove(file_name)
  
  except Exception as e:
    print(sec_code, e) 

    

#----------------------------- Preprocess / Postprocess -------------------------#    
def preprocess_stock_data(df, interval, print_error=True):
  '''
  Preprocess downloaded data

  :param df: downloaded stock data
  :param interval: interval of the downloaded data
  :param print_error: whether print error information or not
  :returns: preprocessed dataframe
  :raises: None
  '''    
  # initialization
  na_cols = []
  zero_cols = []
  error_info = ''
  max_idx = df.index.max()
        
  # check invalid records
  for col in df.columns:
    if df.loc[max_idx, col] == 0:
      zero_cols.append(col)
    if df[col].isna().sum() > 0:
      na_cols.append(col)
    
  # delete NaN value records
  if len(na_cols) > 0:
    error_info = 'NaN values found in '
    for col in na_cols:
      error_info += '{col}, '.format(col=col)
    df = df.dropna()
    
  # delete 0 value records
  if len(zero_cols) > 0:
    error_info += '0 values found in '
    for col in zero_cols:
      error_info += '{col}, '.format(col=col)
    error_info += '数据出错.'
    df = df[:-1].copy()

  # if interval is week, keep data until the most recent monday
  if interval == 'week':
    df = df[~df.index.duplicated(keep='first')]
    if df.index.max().weekday() != 0:
      df = df[:-1].copy()
      
  # print error information
  if print_error and len(error_info)>0:
    print(error_info)
  
  return df
  
def postprocess_stock_data(df, keep_columns, drop_columns):
  '''
  Postprocess downloaded data

  :param df: downloaded stock data
  :param keep_columns: columns to keep for the final result
  :param drop_columns: columns to drop for the final result
  :returns: postprocessed dataframe
  :raises: None
  '''     
  # reset index, keep 3 digits for numbers
  df = df.round(3).reset_index()

  # analysis indicators
  df['趋势'] = ''
  df['超买/超卖'] = ''
  df['信号'] = ''
  df['操作'] = ''
  df['分数'] = 0

  # ================================ 趋势 ==========================================
  df['趋势'] += '['

  # KAMA 趋势
  up_idx = df.query('Close > kama_fast and rate > 0').index
  down_idx = df.query('Close < kama_fast and rate < 0').index
  other_idx = [x for x in df.index if x not in up_idx and x not in down_idx]
  df.loc[up_idx, '趋势'] += '+, '
  df.loc[down_idx, '趋势'] += '-, '
  df.loc[other_idx, '趋势'] += ' , '

  # ICHIMOKU 趋势
  up_idx = df.query('Close > cloud_top and cloud_height > 0').index
  down_idx = df.query('Close < cloud_bottom and cloud_height < 0').index
  other_idx = [x for x in df.index if x not in up_idx and x not in down_idx]
  df.loc[up_idx, '趋势'] += '+, '
  df.loc[down_idx, '趋势'] += '-, '
  df.loc[other_idx, '趋势'] += ' , '

  # KST 趋势
  up_idx = df.query('kst > kst_sign').index
  down_idx = df.query('kst < kst_sign').index
  other_idx = [x for x in df.index if x not in up_idx and x not in down_idx]
  df.loc[up_idx, '趋势'] += '+'
  df.loc[down_idx, '趋势'] += '-'
  df.loc[other_idx, '趋势'] += ' '

  df['趋势'] += ']'
  # =============================== 超买超卖 =======================================
  df['超买/超卖'] += '['

  # 布林线 超买/超卖
  up_idx = df.query('bb_signal == "b"').index
  down_idx = df.query('bb_signal == "s"').index
  other_idx = [x for x in df.index if x not in up_idx and x not in down_idx]
  df.loc[up_idx, '超买/超卖'] += '+, '
  df.loc[down_idx, '超买/超卖'] += '-, '
  df.loc[other_idx, '超买/超卖'] += ' , '

  # RSI 超买/超卖
  up_idx = df.query('rsi_signal == "b"').index
  down_idx = df.query('rsi_signal == "s"').index
  other_idx = [x for x in df.index if x not in up_idx and x not in down_idx]
  df.loc[up_idx, '超买/超卖'] += '+'
  df.loc[down_idx, '超买/超卖'] += '-'
  df.loc[other_idx, '超买/超卖'] += ' '

  df['超买/超卖'] += ']'
  # ================================ 信号/分数 =====================================
  df['信号'] += '['

  # KAMA 信号
  buy_idx = df.query('kama_signal == "b"').index
  sell_idx = df.query('kama_signal == "s"').index
  other_idx = [x for x in df.index if x not in buy_idx and x not in sell_idx]
  df.loc[buy_idx, '信号'] += 'KAMA+, '
  df.loc[sell_idx, '信号'] += 'KAMA-, '
  df.loc[other_idx, '信号'] += (df.loc[other_idx, 'kama_days'].astype(str) + ', ')
  df.loc[buy_idx, '分数'] += 1
  df.loc[sell_idx, '分数'] += -1

  # Ichimoku 信号
  buy_idx = df.query('break_up > ""').index
  sell_idx = df.query('break_down > ""').index
  other_idx = [x for x in df.index if x not in buy_idx and x not in sell_idx]
  df.loc[buy_idx, '信号'] += 'ICHI+, '
  df.loc[sell_idx, '信号'] += 'ICHI-, '
  df.loc[other_idx, '信号'] += (df.loc[other_idx, 'ichimoku_days'].astype(str) + ', ')
  df.loc[buy_idx, '分数'] += 1
  df.loc[sell_idx, '分数'] += -1

  # KST 信号
  buy_idx = df.query('kst_signal == "b"').index
  sell_idx = df.query('kst_signal == "s"').index
  other_idx = [x for x in df.index if x not in buy_idx and x not in sell_idx]
  df.loc[buy_idx, '信号'] += 'KST+'
  df.loc[sell_idx, '信号'] += 'KST-'
  df.loc[other_idx, '信号'] += df.loc[other_idx, 'kst_days'].astype(str)
  df.loc[buy_idx, '分数'] += 1
  df.loc[sell_idx, '分数'] += -1

  df['信号'] += ']'       
  # =============================== 列名处理 =======================================

  df = df[list(keep_columns.keys())].rename(columns=keep_columns)
  if set(['上穿', '下穿']) < set(df.columns):
    for index, row in df.iterrows():
      for i in en_2_cn.keys():
        df.loc[index, '上穿'] = df.loc[index, '上穿'].replace(i, en_2_cn[i])
        df.loc[index, '下穿'] = df.loc[index, '下穿'].replace(i, en_2_cn[i])
    
  # 删除冗余的列
  df = df.drop(drop_columns, axis=1)

  return df
  
#----------------------------- NYTimes Data -------------------------------------#

def download_nytimes(year, month, api_key, file_path, file_format='.json', is_print=False, is_return=False):
  '''
  download news from newyork times api

  :param year: year to download
  :param month: month to download
  :param api_key: nytimes api key
  :param file_path: where the data will be save to
  :param file_format: which format the data will be saved in
  :param is_print: whether to print download information
  :param is_return: whether to return data
  :returns: data if is_return=True
  :raises: None
  '''
  # construct URL
  url = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}" 
  url = url.format(year=year, month=month, api_key=api_key)

  # construct file_name
  file_name = '{file_path}{year}-{month:02}{file_format}'.format(file_path=file_path, year=year, month=month, file_format=file_format)

  # get data
  items = requests.get(url)

  # resolve data
  try:
    if file_format == '.json':
      data = items.json()
      
      # get all news
      docs = data["response"]["docs"]
      
      # drop duplicated news
      doc_id = []
      duplicated_doc_index = []
      for i in range(len(docs)):
        if docs[i]['_id'] not in doc_id:
          doc_id.append(docs[i]['_id'])
        else:
          duplicated_doc_index.append(i)
      duplicated_doc_index.sort(reverse=True)
      
      for i in duplicated_doc_index:
        docs.pop(i)
        
      # update
      data["response"]["docs"] = docs
      data['response']['meta']['hits'] = len(docs)
      
      # save
      with open(file_name, 'w') as f:
        json.dump(data, f)
        
  except Exception as e:
    print(e)
    print(data)
    pass

  # print info
  if is_print:
    print("Finished downloading {}/{} ({}hints)".format(year, month, len(docs)))

  # return data
  if is_return:
    return data


def read_nytimes(year, month, file_path, file_format='.json'):
  """
  read nytimes files into dataframe

  :param year: year to read
  :param month: month to read
  :param file_path: where to read the file
  :file_format: what is the file format of the file
  :returns: dataframe
  :raises: None
  """
  # construct file_name
  file_name = '{file_path}{year}-{month:02}{file_format}'.format(file_path=file_path, year=year, month=month, file_format=file_format)
  
  # load json data
  with open(file_name) as data_file:    
    NYTimes_data = json.load(data_file)
  
  # convert json to dataframe
  date_list = []
  df = pd.DataFrame()  
  df['News'] = None
  num_hits = NYTimes_data['response']['meta']['hits']
  print('读取 {year}/{month} 新闻, {num_hits}条'.format(year=year, month=month, num_hits=num_hits))

  # original columns
  columns = [
      'title',
      'pub_date',
      'news_desk',
      'section_name',
      'snippet',
      'lead_paragraph',
      'web_url',
      'word_count'
  ]
  
  # initialization
  result = {}
  for col in columns:
    result[col] = []

  # go through json
  for article_number in range(num_hits):
    
    tmp_news = NYTimes_data["response"]["docs"][article_number]

    result['title'].append(tmp_news.get('headline').get('main'))  
    result['pub_date'].append(util.time_2_string(datetime.datetime.strptime(tmp_news.get('pub_date'), "%Y-%m-%dT%H:%M:%S+0000"), date_format='%Y-%m-%d %H:%M:%S'))
    result['news_desk'].append(tmp_news.get('news_desk'))
    result['section_name'].append(tmp_news.get('section_name'))
    result['word_count'].append(tmp_news.get('word_count'))       
    result['snippet'].append(tmp_news.get('snippet'))     
    result['lead_paragraph'].append(tmp_news.get('lead_paragraph')) 
    result['web_url'].append(tmp_news.get('web_url')) 

  df = pd.DataFrame(result)

  # return dataframe
  return df  


#----------------------------- Global Parameter Setting -------------------------#
def create_config_file(config_dict, file_path, file_name):
  """
  Create a config file and save global parameters into the file

  :param config_dict: config parameter of keys and values
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :returns: None
  :raises: save error
  """
  try:
    with open(file_path + file_name, 'w') as f:
      json.dump(config_dict, f)
    print('Config saved successfully')

  except Exception as e:
    print(e)


def read_config(file_path, file_name):
  """
  Read config from a specific file

  :param file_path: the path to save the file
  :param file_name: the name of the file
  :returns: config dict
  :raises: read error
  """
  try:
    # read existing config
    with open(file_path + file_name, 'r') as f:
      config_dict = json.loads(f.read())

  except Exception as e:
    print(e)

  return config_dict


def add_config(config_key, config_value, file_path, file_name):
  """
  Add a new config in to the config file

  :param config_key: name of the new config
  :param config_value: value of the config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :returns: None
  :raises: save error
  """

  try:
    # read existing config
    new_config = read_config(file_path, file_name)
    new_config[config_key] = config_value

    with open(file_path + file_name, 'w') as f:
      json.dump(new_config, f)
      print('Config added successfully')

  except Exception as e:
    print(e)


def remove_config(config_key, file_path, file_name):
  """
  remove a config from the config file

  :param config_key: name of the new config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :returns: None
  :raises: save error
  """
  try:
    # read existing config
    new_config = read_config(file_path, file_name)
    new_config.pop(config_key)

    with open(file_path + file_name, 'w') as f:
      json.dump(new_config, f)
      print('Config removed successfully')

  except Exception as e:
    print(e)


def modify_config(config_key, config_value, file_path, file_name):
  """
  modify the value of a config with certain config_key

  :param config_key: name of the new config
  :param config_value: value of the config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :returns: None
  :raises: save error
  """
  try:
    # read existing config
    new_config = read_config(file_path, file_name)
    new_config[config_key] = config_value

    with open(file_path + file_name, 'w') as f:
      json.dump(new_config, f)
      print('Config modified successfully')

  except Exception as e:
    print(e)



#----------------------------- Zip file ----------------------------------------#
def zip_folder(folder_path, destination_path, zip_file_name):
  """
  Zip folder
  :param folder_path: full path of the folder
  :param destination_path: where you want the zip file to be
  :param zip_file_name: name of the zip file
  :returns: zip file name
  :raises: none
  """
  # create zip file
  start_dir = folder_path
  zip_file_name = '{path}{name}.zip'.format(path=destination_path, name=zip_file_name)
  zip_writer = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)

  # zip files in the folder into zip file
  for dir_path, dir_names, file_names in os.walk(start_dir):
    short_path = dir_path.replace(start_dir, '')
    short_path = short_path + os.sep if short_path is not None else ''
    for f in file_names:
      zip_writer.write(os.path.join(dir_path, f),short_path+f)
  zip_writer.close()
  
  return zip_file_name
