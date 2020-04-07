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
import pickle
import time
import json
import os
import yfinance as yf
import pandas_datareader.data as web 
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
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
    sec_list = [x for x in sec_list if '$' not in x]
    sec_list = [x for x in sec_list if '.' not in x]

  # remove not-fetched symbols
  if remove_not_fetched and not_fetched_list is not None:
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


def get_data_from_tiger(sec_code, interval, start_date=None, end_date=None, time_col='time', minute_level=False, quote_client=None, download_limit=1200, is_print=False):
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
      if minute_level:
        data[time_col] = data[time_col].apply(lambda x: util.timestamp_2_time(x))
      else:
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


def download_stock_data(sec_code, start_date=None, end_date=None, source='yahoo', time_col='Date', interval='d', quote_client=None, minute_level=False, is_return=True, is_print=False, is_save=False, file_path=None, file_name=None):
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
      data = get_data_from_yfinance(sec_code=sec_code, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
    # tiger
    elif source == 'tiger':
      data = get_data_from_tiger(sec_code=sec_code, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, quote_client=quote_client, minute_level=minute_level, is_print=is_print)
    else:
      print('data source {source} not found'.format(source=source))
      return None

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


def get_stock_briefs(symbols, period='1d', interval='1m'):
  """
  Get latest stock data for symbols

  :param symbols: symbol list
  :param period: how long the period to download
  :param interval: in which interval to download
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """
  # get minute data for recent 1 day
  ticker_data = yf.download(tickers=symbols, period=period, interval=interval, group_by='ticker')
  min_idx = ticker_data.index.min()
  max_idx = ticker_data.index.max()

  # gather latest data for each symbol
  latest_data = pd.DataFrame()
  for symbol in symbols:

    tmp_ticker_data = ticker_data[symbol].dropna()

    # get the latest row of the data
    tmp_data = tmp_ticker_data.tail(1).reset_index().copy()
    
    # assign symbol to the row
    tmp_data['symbol'] = symbol

    # update the Open/High/Low
    tmp_data.loc[0, 'Open'] = tmp_ticker_data.loc[min_idx, 'Open'].round(2)
    tmp_data.loc[0, 'High'] = tmp_ticker_data['High'].max().round(2)
    tmp_data.loc[0, 'Low'] = tmp_ticker_data['Low'].min().round(2)
    tmp_data.loc[0, 'Volume'] = tmp_ticker_data['Volume'].sum()
    tmp_data['Close'] = tmp_data['Close'].round(2)
    tmp_data['Adj Close'] = tmp_data['Close'] 
    

    # append the data into result
    latest_data = latest_data.append(tmp_data)

  # process date of the data
  latest_data = latest_data.rename(columns={'Datetime': 'latest_time'})
  latest_data['latest_price'] = latest_data['Close'].copy()
  latest_data['Date'] = latest_data['latest_time'].copy()
  latest_data['Date'] = latest_data['Date'].apply(util.time_2_string, args=(0, '%Y-%m-%d',))
  latest_data['Date'] = latest_data['Date'].apply(util.string_2_time,args=('%Y-%m-%d',))
  

  return latest_data


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
    file_name = '{path}{name}.csv'.format(path=file_path, name=sec_code) 
  else:
    file_name = '{path}{name}.csv'.format(path=file_path, name=file_name)

  # read data from google drive
  try:
    # if the file not exists, print information, return an empty dataframe
    if not os.path.exists(file_name):
      print('{file} not exists'.format(file=file_name))

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
    file_name = '{path}{name}.csv'.format(path=file_path, name=sec_code) 
  else:
    file_name = '{path}{name}.csv'.format(path=file_path, name=file_name)
  
  try:
    os.remove(file_name)
  
  except Exception as e:
    print(sec_code, e) 

  
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
  url = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}".format(year=year, month=month, api_key=api_key)

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
    print("Finished downloading {year}/{month} ({docs}hints)".format(year=year, month=month, docs=len(docs)))

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
def create_config_file(config_dict, file_path, file_name, print=False):
  """
  Create a config file and save global parameters into the file

  :param config_dict: config parameter of keys and values
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param print: whether to print result
  :returns: None
  :raises: save error
  """
  try:
    with open(file_path + file_name, 'w') as f:
      json.dump(config_dict, f)
    
    if print:
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
    with open(file_path + file_name, 'r', encoding='UTF-8') as f:
      config_dict = json.loads(f.read())

  except Exception as e:
    print(e)

  return config_dict


def add_config(config_key, config_value, file_path, file_name, print=False):
  """
  Add a new config in to the config file

  :param config_key: name of the new config
  :param config_value: value of the config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param print: whether to print result
  :returns: None
  :raises: save error
  """

  try:
    # read existing config
    new_config = read_config(file_path, file_name)
    new_config[config_key] = config_value

    with open(file_path + file_name, 'w', encoding='UTF-8') as f:
      json.dump(new_config, f)
      if print:
        print('Config added successfully')

  except Exception as e:
    print(e)


def remove_config(config_key, file_path, file_name, print=False):
  """
  remove a config from the config file

  :param config_key: name of the new config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param print: whether to print result
  :returns: None
  :raises: save error
  """
  try:
    # read existing config
    new_config = read_config(file_path, file_name)
    new_config.pop(config_key)

    with open(file_path + file_name, 'w', encoding='UTF-8') as f:
      json.dump(new_config, f)
      if print:
        print('Config removed successfully')

  except Exception as e:
    print(e)


def modify_config(config_key, config_value, file_path, file_name, print=False):
  """
  modify the value of a config with certain config_key

  :param config_key: name of the new config
  :param config_value: value of the config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param print: whether to print result
  :returns: None
  :raises: save error
  """
  try:
    # read existing config
    new_config = read_config(file_path, file_name)
    new_config[config_key] = config_value

    with open(file_path + file_name, 'w', encoding='UTF-8') as f:
      json.dump(new_config, f)
      if print:
        print('Config modified successfully')

  except Exception as e:
    print(e)


#----------------------- Solidify data ---------------------# 
def dict_2_excel(dictionary, file_path, file_name, keep_index=False):

  # 打开文件
  writer = pd.ExcelWriter('{path}{name}'.format(path=file_path, name=file_name))

  # 写入
  for k in dictionary.keys():
    dictionary[k].to_excel(writer, sheet_name=k, index=keep_index)

  # 关闭文件并保存
  writer.save()
  writer.close()


def folder_2_zip(folder_path, destination_path, zip_file_name):
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
  zip_file_name = '{path}{name}'.format(path=destination_path, name=zip_file_name)
  zip_writer = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)

  # zip files in the folder into zip file
  for dir_path, dir_names, file_names in os.walk(start_dir):
    short_path = dir_path.replace(start_dir, '')
    short_path = short_path + os.sep if short_path is not None else ''
    for f in file_names:
      zip_writer.write(os.path.join(dir_path, f),short_path+f)
  zip_writer.close()
  
  return zip_file_name


def pickle_dump_data(data, file_path, file_name):
  """
  pickle data into a file
  :param data: data to dump
  :param file_path: destination file path
  :param file_name: destination file name
  :raises: None
  :returns: None
  """
  file_name = file_path + file_name
  with open(file_name, 'wb') as f:
    pickle.dump(data, f)


def pickle_load_data(file_path, file_name):
  """
  load data from pickled file
  :param file_path: source file path
  :param file_name: source file name
  :raises: None
  :returns: pickled data
  """
  file_name = file_path + file_name
  data = None
  with open(file_name, 'rb') as f:
    data = pickle.load(f)

  return data
