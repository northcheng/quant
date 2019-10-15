# -*- coding: utf-8 -*-
"""
Utilities used for data IO

:author: Beichen Chen
"""

import pandas as pd
import numpy as np
import datetime
import time
import os
import requests
import json
import pandas_datareader.data as web 
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from quant import bc_util as util


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
  # initialization
  data = pd.DataFrame()

  # read data from google drive files
  try:
    # construct filename by sec_code, file_path and file_format
    if file_name is None:
      file_name = file_path + sec_code + '.csv'
    else:
      file_name = file_path + file_name + '.csv'
    
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


def download_stock_data(sec_code, file_path, file_name=None, start_date=None, end_date=None, source='yahoo', time_col='Date', interval='d', quote_client=None, download_limit=1200, is_return=False, is_save=True, is_print=True):
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
  # download stock data from yahoo finance api via pandas_datareader
  if source == 'yahoo':
    return download_stock_data_from_yahoo(sec_code=sec_code, file_path=file_path, file_name=file_name, start_date=start_date, end_date=end_date, time_col=time_col, interval=interval, is_return=is_return, is_save=is_save, is_print=is_print)
  # download stock data by using tiger open api
  elif source == 'tiger':
    return download_stock_data_from_tiger(sec_code=sec_code, file_path=file_path, file_name=file_name, start_date=start_date, end_date=end_date, time_col=time_col, interval=interval, is_return=is_return, is_save=is_save, is_print=is_print, quote_client=quote_client, download_limit=download_limit)


def download_stock_data_from_yahoo(sec_code, file_path, file_name=None, start_date=None, end_date=None, time_col='Date', interval='d', is_return=False, is_save=True, is_print=True):
  """
  Download stock data from Yahoo finance api via pandas_datareader

  :param sec_code: symbol of the stock to download
  :param file_path: path to store the download data
  :param file_name: name of the stock data file
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param time_col: time column in that data
  :param interval: period of data: d/w/m/v
  :param is_return: whether to return the download data in dataframe format
  :param is_print: whether to print the download information
  :returns: dataframe is is_return=True
  :raises: none
  """
  # construct filename by sec_code, file_path and file_format
  if file_name is None:
    file_name = file_path + sec_code + '.csv'
  else:
    file_name = file_path + file_name + '.csv'
  
  try:
    # download data
    stage = 'downloading_data'
    data = web.get_data_yahoo(sec_code, start_date, end_date, interval=interval)
    
    # save data
    stage = 'saving_data'
    data_length = len(data)
    if data_length > 0:
      data = data.reset_index().drop_duplicates(subset=time_col, keep='last')

      if  is_save:
        data.to_csv(file_name, index=False)
      
    # print download result
    if is_print:
      print('[From Yahoo]%(sec_code)s: %(first_date)s - %(latest_date)s, 下载记录 %(data_length)s, ' % dict(
        sec_code=sec_code, first_date=data[time_col].min().date(), latest_date=data[time_col].max().date(), data_length=data_length))

  except Exception as e:
      print(sec_code, stage, e)

  # return dataframe
  if is_return: 
    return util.df_2_timeseries(data, time_col=time_col) 


def download_stock_data_from_tiger(sec_code, file_path, file_name=None, start_date=None, end_date=None, time_col='time', interval='day', is_return=False, is_save=True, is_print=True, quote_client=None, download_limit=1200):
  """
  Download stock data from Tiger Open API
  :param sec_code: symbol of the stock to download
  :param file_path: path to store the download data
  :param file_name: name of the stock data file
  :param start_date: start date of the data
  :param end_date: end date of the data 
  :param time_col: time column in that data  
  :param interval: period of data: day/week/month/year/1min/5min/15min/30min/60min
  :param is_return: whether to return the download data in dataframe format
  :param is_print: whether to print the download information
  :param quote_client: quote_client used for querying data from API
  :param download limit: the limit of number of records in each download
  :returns: dataframe is is_return=True
  :raises: none
  """  
  # construct filename by sec_code, file_path and file_format
  if file_name is None:
    file_name = file_path + sec_code + '.csv'
  else:
    file_name = file_path + file_name + '.csv'
  
  try:     
    # download data from tiger open api
    stage = 'downloading_data'

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
    data = pd.DataFrame()
    tmp_len = download_limit
    while tmp_len >= download_limit:  
      tmp_data = quote_client.get_bars([sec_code], begin_time=begin_time, end_time=end_time, period=interval, limit=download_limit)
      tmp_len = len(tmp_data)
      data = tmp_data.append(data)
      end_time = int(tmp_data.time.min())
    
    # process newly downloaded data
    stage = 'processing_data'
    data_length = len(data)
    if data_length > 0:
      data.drop('symbol', axis=1, inplace=True)
      data[time_col] = data[time_col].apply(lambda x: util.timestamp_2_time(x).date())
      data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'time': 'Date'}, inplace=True)
      data['Adj Close'] = data['Close']

      # drop duplicated data
      stage = 'saving_data'
      time_col = 'Date'
      data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
      data.sort_values(by=time_col,  inplace=True)
      if is_save:
        data.to_csv(file_name, index=False) 
      
    # print download result
    if is_print:
      print('[From Tiger]%(sec_code)s: %(first_date)s - %(latest_date)s, 下载记录 %(data_length)s, ' % dict(
        sec_code=sec_code, first_date=data[time_col].min().date(), latest_date=data[time_col].max().date(), data_length=data_length))
      
  except Exception as e:
    print(sec_code, stage, e)   
    
  # return dataframe
  if is_return:
    return util.df_2_timeseries(data, time_col=time_col)


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

  # construct filename
  filename = '{file_path}{year}-{month:02}{file_format}'.format(file_path=file_path, year=year, month=month, file_format=file_format)

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
      with open(filename, 'w') as f:
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
  # construct filename
  filename = '{file_path}{year}-{month:02}{file_format}'.format(file_path=file_path, year=year, month=month, file_format=file_format)
  
  # load json data
  with open(filename) as data_file:    
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

  df = pd.DataFrame(result)

  # return dataframe
  return df  