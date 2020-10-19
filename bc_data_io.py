# -*- coding: utf-8 -*-
"""
Utilities used for data IO

:author: Beichen Chen
"""

# regular
import pandas as pd
import numpy as np
import requests
import datetime
import zipfile
import pickle
import json
import os

# data source
import yfinance as yf
import pandas_datareader.data as web 
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

# mail modules
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# self defined
from quant import bc_util as util


#----------------------------- Stock Data -------------------------------------#
def get_symbols(remove_invalid=True):
  """
  Get Nasdaq stock list

  :param remove_invalid: whether to remove invalid stock symbols from external stock list (.csv)
  :returns: dataframe of stock symbols
  :raises: exception when error reading not-fetched symbols list
  """
  # get the symbols from pandas_datareader
  try:
    symbols = get_nasdaq_symbols()
    symbols = symbols.loc[symbols['Test Issue'] == False,]
  
  # get symbols from Nasdaq website directly when the pandas datareader is not available
  except Exception as e:
    symbols = pd.read_table('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt', sep='|', index_col='Symbol').drop(np.NaN)
    symbols = symbols.loc[symbols['Test Issue'] == 'N',]
    print(e)
  
  # get list of all symbols and remove invalid symbols
  sec_list = symbols.index.tolist()
  if remove_invalid:
    sec_list = [x for x in sec_list if '$' not in x]
    sec_list = [x for x in sec_list if '.' not in x]
  
  return symbols.loc[sec_list, ]


def get_data_from_yahoo(symbol, start_date=None, end_date=None, time_col='Date', interval='d', is_print=False):
  """
  Download stock data from Yahoo finance api via pandas_datareader

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param time_col: column name of datetime column
  :param interval: available values - d/w/m/v
  :param is_print: whether to print download information
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  data = None

  try:
    # download data
    data = web.get_data_yahoo(symbol, start_date, end_date, interval=interval)
    data = post_process_download_data(df=data, source='yahoo')
      
    # print download information
    if is_print:
      print(f'[From Yahoo]{symbol}: {data.index.min().date()} - {data.index.max().date()}, 下载记录 {len(data)}')

  except Exception as e:
      print(symbol, e)
      
  return data


def get_data_from_yfinance(symbol, start_date=None, end_date=None, time_col='Date', interval='1d', is_print=False):
  """
  Download stock data from Yahoo finance api via yfinance

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param time_col: column name of datetime column
  :param interval: available values - 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
  :param is_print: whether to print download information
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  data = None

  try:
    # +1 day for end_date to ensure the data for the end_date is included
    if end_date is not None:
      end_date = util.string_plus_day(end_date, 1)

    # download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval, actions=True, auto_adjust=True, back_adjust=False)
    data = post_process_download_data(df=data, source='yfinance')

    # print download result
    if is_print:
      print(f'[From YFinance]{symbol}: {data.index.min().date()} - {data.index.max().date()}, 下载记录 {len(data)}')

  except Exception as e:
      print(symbol, e)
      
  return data 


def get_data(symbol, start_date=None, end_date=None, source='yfinance', time_col='Date', interval='1d', is_print=False):
  """
  Download stock data from web sources

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param source: datasrouce: 'yahoo' or 'yfinance'
  :param time_col: column name of datetime column
  :param interval: period, for yahoo: d/w/m/v; for yfinance: 1d/1wk/1mo;
  :param is_print: whether to print download information
  :returns: dataframe 
  :raises: exception when downloading failed
  """
  data = None

  try:
    # yahoo
    if source == 'yahoo':
      data = get_data_from_yahoo(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
    
    # yfinance
    elif source == 'yfinance':
      data = get_data_from_yfinance(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
    
    # otherwise
    else:
      print(f'data source {source} not found')

  except Exception as e:
    print(symbol, e)

  return data


def get_stock_briefs_from_yfinance(symbols, period='1d', interval='1m'):
  """
  Get latest stock data for symbols

  :param symbols: list of target symbols
  :param period: how long the period to download
  :param interval: available values - 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
  :returns: dataframe of latest data, per row for each symbol
  :raises: none
  """
  latest_data = None

  # if the target list is empty
  if len(symbols) == 0:
    print('symbols is empty')

  else:
    # get minute data for the most recent 1 day
    ticker_data = yf.download(tickers=symbols, period=period, interval=interval, group_by='ticker')
    if len(ticker_data) > 0:

      # process if there is only one symbol in the list
      if len(symbols) == 1:
        ticker_data = {symbols[0]: ticker_data}

      # gather latest data for each symbol
      latest_data = pd.DataFrame()
      for symbol in symbols:

        # get the latest data for current symbol 
        tmp_ticker_data = ticker_data[symbol].dropna()
        min_idx = tmp_ticker_data.index.min() 
        tmp_data = tmp_ticker_data.tail(1).reset_index().copy()
        
        # update the Open/High/Low/Close/Volume/Adj Close/symbol
        tmp_data.loc[0, 'Open'] = round(tmp_ticker_data.loc[min_idx, 'Open'], 2)
        tmp_data.loc[0, 'High'] = round(tmp_ticker_data['High'].max(), 2)
        tmp_data.loc[0, 'Low'] = round(tmp_ticker_data['Low'].min(), 2)
        tmp_data.loc[0, 'Volume'] = tmp_ticker_data['Volume'].sum()
        tmp_data['Close'] = tmp_data['Close'].round(2)
        tmp_data['Adj Close'] = tmp_data['Close'] 
        tmp_data['symbol'] = symbol

        # append the data into result
        latest_data = latest_data.append(tmp_data)

      # process date of the data
      latest_data = latest_data.rename(columns={'Datetime': 'latest_time'})
      latest_data['latest_price'] = latest_data['Close'].copy()
      latest_data['Date'] = latest_data['latest_time'].copy()
      latest_data['Date'] = latest_data['Date'].apply(util.time_2_string, args=(0, '%Y-%m-%d',))
      latest_data['Date'] = latest_data['Date'].apply(util.string_2_time,args=(0, '%Y-%m-%d',))

    else:
      print('ticker_data is empty')

    

  return latest_data


def get_stock_briefs(symbols, source='yfinance', period='1d', interval='1m'):
  """
  Get latest stock data for symbols

  :param symbols: symbol list
  :param source: data source
  :param period: how long the period to download
  :param interval: in which interval to download
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """
  # initialize
  briefs = None

  # currently only yfinance is available
  if source == 'yfinance':
    briefs =  get_stock_briefs_from_yfinance(symbols=symbols, period=period, interval=interval)
  else:
    print(f'Unknown source {source}')

  return briefs


def post_process_download_data(df, source):
  """
  Post process data downloaded from certain data source

  :param df: stock data dataframe downloaded from source
  :param source: data source
  :returns: post processed data
  :raises: none
  """
  if df is not None:

    # copy dataframe
    df = df.copy()

    # post process data downloaded from yfinance
    if source == 'yfinance':
      df = df.rename(columns={'Dividends':'Dividend', 'Stock Splits': 'Split'})
      df.Split = df.Split.replace(0, 1)
      df = df.dropna()

      if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    # post process data downloaded from yahoo
    elif source == 'yahoo':
      df['Split'] = 1
      df['Dividend'] = 0

  return df


def update_stock_data_from_yfinance(symbol, data=None, is_print=True):
  """
  Get latest stock data for a symbol in current trading day

  :param symbols: target symbol
  :param data: existed data
  :returns: dataframe of updated stock data
  :raises: none
  """
  # calculate start_date, end_date
  if data is not None:
    start = data.index.max()
  else:
    start = '1991-01-01'
    data = pd.DataFrame()
  end = util.time_2_string(datetime.datetime.today().date())

  # get the most recent data, append it to the original data
  realtime_data = get_data_from_yfinance(symbol=symbol, interval='1d', start_date=start, end_date=end, is_print=is_print)
  
  # append new data to the existed data
  data = data.append(realtime_data, sort=True)
  data = util.remove_duplicated_index(df=data, keep='last').dropna()

  return data


def update_stock_data_from_yfinance_by_stock(symbols, stock_data_path, file_format='.csv', required_date=None, is_print=False, is_return=False, is_save=True):
  """
  update local stock data from yfinance

  :param symbols: list of target symbols
  :param stock_data_path: where the local stock data files(.csv) stored
  :param file_format: default is .csv
  :param required_date: if the local data have already meet the required date, it won't be updated
  :param is_print: whether to print info when downloading
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """
  # init
  data = {}

  # get current date if required date is not specified
  if required_date is None:
    required_date = util.time_2_string(datetime.datetime.today())

  # go through symbols
  for symbol in symbols:

    # init
    old_data_date = '1991-01-01'
    download_info = f'{symbol}: '
  
    # if local data existed, load local data
    if os.path.exists(f'{stock_data_path}{symbol}{file_format}'):
      old_data = load_stock_data(file_path=stock_data_path, file_name=symbol)
      old_data_date = util.time_2_string(old_data.index.max())
      download_info += f'exists({old_data_date}), updating...'
      
    # if not exist, download from 1991-01-01
    else:
      download_info += 'not found, downloading...'
      old_data = pd.DataFrame()

    # print download info
    if is_print:
      print(download_info, end=' ')
      
    # download new data
    new_data = get_data_from_yfinance(symbol, interval='1d', start_date=old_data_date, end_date=required_date, time_col='Date', is_print=is_print)
    if new_data['Split'].product() != 1 and old_data_date != '1991-01-01':
      if is_print:
        print('Split action detected, redownloading data...')
      new_data = get_data_from_yfinance(symbol, interval='1d', start_date='1991-01-01', end_date=required_date, time_col='Date', is_print=is_print)
    
    # append new data to the old data, remove duplicated index, keep the latest
    tmp_data = old_data.append(new_data, sort=True)
    tmp_data = util.remove_duplicated_index(df=tmp_data, keep='last').dropna()
    data[symbol] = tmp_data

  # save data to the specified path with <symbol>.<file_format>
  if is_save:
    for symbol in data.keys():
      save_stock_data(df=tmp_data, file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True)

  # return
  if is_return:
    return data
    

def update_stock_data_from_yfinance_by_date(symbols, stock_data_path, file_format='.csv', required_date=None, is_print=False, is_return=False, is_save=True):
  """
  update local stock data from alphavantage

  :param symbols: list of target symbols
  :param stock_data_path: where the local stock data files(.csv) stored
  :param file_format: default is .csv
  :param required_date: if the local data have already meet the required date, it won't be updated
  :param is_print: whether to print info when downloading
  :param is_return: whether to return the updated data
  :param is_save: whether to save the updated data to local files
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """
  # get current date if required date is not specified
  if required_date is None:
    required_date = util.time_2_string(datetime.datetime.today())

  # get the existed data and its latest date for each symbols
  data = {}
  data_date = {'1991-01-01': []}
  for symbol in symbols:

    # if local data exists, load local data
    if os.path.exists(f'{stock_data_path}{symbol}{file_format}'):
      tmp_data = load_stock_data(file_path=stock_data_path, file_name=symbol)
      tmp_data_date = util.time_2_string(tmp_data.index.max())
      data[symbol] = tmp_data.copy()

      # group symbols by their latest date
      if data_date.get(tmp_data_date) is None:
        data_date[tmp_data_date] = [symbol]
      else:
        data_date[tmp_data_date].append(symbol)

    # if local data not exists, download from 1991-01-01
    else:
      data[symbol] = pd.DataFrame()
      data_date['1991-01-01'].append(symbol)

  # download and update by group(latest date)
  download_info = ''
  for d in data_date.keys():

    # get symbols with same latest date, it symbols is empty, skip
    tmp_symbols = data_date[d]
    if len(tmp_symbols) == 0:
      continue

    # download, construct download information for printing
    tmp_batch_data = yf.download(tickers=tmp_symbols, start=d, interval='1d', group_by='ticker', actions=True)
    download_info += f'{tmp_symbols} updated from {d}\n'

    if len(tmp_symbols) == 1:
      tmp_batch_data = {tmp_symbols[0]: tmp_batch_data}

    # update data for current batch, rename, replace, append, dropna
    for symbol in tmp_symbols:
      new_data = tmp_batch_data[symbol].copy()
      new_data = post_process_download_data(df=new_data, source='yfinance')
      data[symbol] = data[symbol].append(new_data, sort=True)
      data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

  # print
  if is_print:
    print(download_info)

  # save 
  if is_save:
    for symbol in data.keys():
      save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)

  # return
  if is_return:
    return data


def update_stock_data(symbols, stock_data_path, file_format='.csv', source='yfinance', by='date', required_date=None, is_print=False, is_return=False, is_save=True):
  """
  update local stock data

  :param symbols: symbol list
  :param stock_data_path: in where the local stock data files(.csv) are stored
  :param source: data source to download latest stock data, yfinance only for now
  :param file_format: default is .csv
  :param required_date: if the local data have already meet the required date, it won't be updated
  :param by: 'stock'-update one by one; 'date'-update for batch of stocks which have same latest dates
  :param is_print: whether to print info when downloading
  :param is_return: whether to return the updated data
  :param is_save: whether to save the updated data to local files
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """

  result = None

  if source == 'yfinance':
    if by == 'date':
      result = update_stock_data_from_yfinance_by_date(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save)
    elif by == 'stock':
      result = update_stock_data_from_yfinance_by_stock(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save)
  else:
    print(f'unknown source: {source}')

  if is_return:
    return result


def save_stock_data(df, file_path, file_name, file_format='.csv', reset_index=True, index=False):
  """
  save stock data (dataframe) to .csv file

  :param df: stock data to save
  :param file_path: to where the file will be save to
  :param file_name: name of the file to save
  :param file_format: default is .csv
  :param reset_index: whether to reset index
  :param index: whether to save index in the .csv file
  :returns: none
  :raises: none
  """
  # construct filename
  file_name = f'{file_path}{file_name}{file_format}'

  # reset index
  if reset_index:
    df = df.sort_index().reset_index()
  
  # save file
  if file_format == '.csv':
    df.to_csv(file_name, index=index)
  else:
    print(f'Unknown format {file_format}')


def load_stock_data(file_path, file_name, file_format='.csv', time_col='Date', standard_columns=False, sort_index=True):
  """
  load stock data (dataframe) from .csv file

  :param file_path: to where the file will be save to
  :param file_name: name of the file to save
  :param file_format: default is .csv
  :param time_col: column name of the time col, default is Date
  :param standard_columns: whether to return dataframe with standard columns (OHLCV, Adj Close, Dividend, Split)
  :param sort_index: whether to sort index
  :returns: none
  :raises: none
  """
  # contruct filename
  file_name = f'{file_path}{file_name}{file_format}'
  
  # initialize data
  df = None
  
  # read data from local file
  try:
    # if the file not exists, print information
    if not os.path.exists(file_name):
      print(f'{file_name} not exists')

    else:
      # load file
      df = pd.read_csv(file_name, encoding='utf8', engine='python')
      
      # transform dataframe to timeseries
      df = util.df_2_timeseries(df=df, time_col=time_col)
      
      # select standard columns
      if standard_columns:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividend', 'Split']].copy()
      
      # sort index
      if sort_index:
        df.sort_index(inplace=True)

  except Exception as e:
    print(e)

  return df


def remove_stock_data(symbol, file_path, file_format='.csv'):
  '''
  Remove stock data file from drive

  :param symbol: symbol of the stock to download
  :param file_path: path to store the download data
  :param file_format: the format of file that data will be stored in
  :returns: None
  :raises: None
  '''
  # construct filename
  file_name = f'{file_path}{symbol}{file_format}'
  
  # remove file
  try:
    if os.path.exists(file_name):
      os.remove(file_name)
    else:
      print(f'{file_name} not exists')
      
  except Exception as e:
    print(symbol, e) 


def create_week_data(df):
  '''
  convert day-interval data into week-interval 

  :param df: day-interval OHLCV data
  :returns: None
  :raises: None
  '''
  # count weeks
  index = df.index
  df['week_day'] = [x.weekday() for x in index]
  df['previous_week_day'] = df['week_day'].shift(1)
  df.fillna(-1, inplace=True)

  # if current week_day < previous week_day, means new week started
  week_count = 0
  df['week_count'] = 0
  for index, row in df.iterrows():
    if row['week_day'] < row['previous_week_day']:
      week_count += 1
    df.loc[index, 'week_count'] = week_count

  # create an empty dict for storing result
  week_data = {'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [], 'Dividend':[], 'Split':[]}

  # go through weeks
  for week in range(week_count+1):

    tmp_data = df.query(f'week_count == {week}')
    week_data['Date'].append(tmp_data.index.min())
    week_data['Open'].append(tmp_data.loc[tmp_data.index.min(), 'Open'])
    week_data['Close'].append(tmp_data.loc[tmp_data.index.max(), 'Close'])
    week_data['High'].append(tmp_data['High'].max())
    week_data['Low'].append(tmp_data['Low'].min())
    week_data['Volume'].append(tmp_data['Volume'].sum())

    if 'Dividend' in tmp_data.columns:
      week_data['Dividend'].append(tmp_data['Dividend'].sum())
    else:
      week_data['Dividend'].append(0)

    if 'Split' in tmp_data.columns:
      week_data['Split'].append(tmp_data['Split'].product())
    else:
      week_data['Split'].append(1.0)
  
  # convert result dict to timeseries dataframe
  week_data = pd.DataFrame(week_data)
  week_data['Adj Close'] = week_data['Close']
  week_data = util.df_2_timeseries(df=week_data, time_col='Date')
  
  return week_data


def create_month_data(df):
  '''
  convert day-interval data into month-interval 

  :param df: day-interval OHLCV data
  :returns: None
  :raises: None
  '''
  # get minimum and maximum index
  min_index = df.index.min()
  max_index = df.index.max()
  
  # get start year and end year
  start_year = min_index.year
  end_year = max_index.year
  
  # get start month and end month
  start_month = util.time_2_string(min_index)[:7]
  end_month = util.time_2_string(max_index)[:7]
  
  # create an empty dict for storing result
  month_data = {'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [], 'Dividend':[], 'Split':[]}
  
  # go through each month
  for year in range(start_year, end_year+1):
    for month in range(1, 12+1):
      
      # get current month
      tmp_period = f'{year}-{month:02}'
      if (tmp_period >= start_month ) and (tmp_period <= end_month):
        tmp_data = df[tmp_period]
        
        # skip current period if no data
        if len(tmp_data) == 0:
          continue
          
        month_data['Date'].append(tmp_period)
        month_data['Open'].append(tmp_data.loc[tmp_data.index.min(), 'Open'])
        month_data['Close'].append(tmp_data.loc[tmp_data.index.max(), 'Close'])
        month_data['High'].append(tmp_data['High'].max())
        month_data['Low'].append(tmp_data['Low'].min())
        month_data['Volume'].append(tmp_data['Volume'].sum())

        if 'Dividend' in tmp_data.columns:
          month_data['Dividend'].append(tmp_data['Dividend'].sum())
        else:
          month_data['Dividend'].append(0)

        if 'Split' in tmp_data.columns:
          month_data['Split'].append(tmp_data['Split'].product())
        else:
          month_data['Split'].append(1.0)

      else:
        continue
  
  # convert result dictionary to dataframe and post process
  month_data = pd.DataFrame(month_data)
  month_data['Adj Close'] = month_data['Close']
  month_data = util.df_2_timeseries(df=month_data, time_col='Date')

  return month_data
    

def switch_data_interval(df, interval):
  '''
  convert day-interval data into week-interval or month-interval data

  :param df: day-interval OHLCV data
  :param interval: interval of target data week/month
  :returns: None
  :raises: None
  '''
  # initialize result
  result = None

  # convert data
  if df is not None:
    
    result = df.copy()
    if interval == 'day':
      pass

    elif interval == 'week':
      result = create_week_data(result)

    elif interval == 'month':
      result = create_month_data(result)

    else:
      print(f'unknown interval {interval}')
      
  return result


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
  url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"

  # construct file_name
  file_name = f'{file_path}{year}-{month:02}{file_format}'

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
    print(f"Finished downloading {year}/{month} ({len(docs)}hints)")

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
  file_name = f'{file_path}{year}-{month:02}{file_format}'
  
  # load json data
  with open(file_name) as data_file:    
    NYTimes_data = json.load(data_file)
  
  # convert json to dataframe
  df = pd.DataFrame()  
  df['News'] = None
  num_hits = NYTimes_data['response']['meta']['hits']
  print(f'读取 {year}/{month} 新闻, {num_hits}条')

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
    config_dict = {}
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


#----------------------------- Solidify data ------------------------------------# 
def dict_2_excel(dictionary, file_path, file_name, keep_index=False):

  # 打开文件
  writer = pd.ExcelWriter(f'{file_path}{file_name}')

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
  zip_file_name = f'{destination_path}{zip_file_name}'
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


#----------------------- Email sending ---------------------------#
def send_result_by_email(config, to_addr, from_addr, smtp_server, password, subject=None, platform=['tiger'], signal_file_date=None, log_file_date=None):

  # construct email 
  m = MIMEMultipart()
  if subject is not None:
    m['Subject'] = subject
  else:
    m['Subject'] = f'[auto_trade] {datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}'
  
  # get asset summary
  assets = {}
  if 'tiger' in platform:
    if os.path.exists(config['config_path']+'tiger_position_record.json'):
      pr = read_config(file_path=config['config_path'], file_name='tiger_position_record.json')
      assets['glob'] = pr.get('global_account')
      assets['simu'] = pr.get('simulation_account') 

  if 'futu' in platform:
    if os.path.exists(config['config_path']+'futu_position_record.json'):
      pr = read_config(file_path=config['config_path'], file_name='futu_position_record.json')
      assets['REAL'] = pr.get('REAL')
      assets['SIMU'] = pr.get('SIMULATE')

  asset_info = '<h3>Assets</h3><ul>'
  for portfilio in assets.keys():
    if assets[portfilio] is not None:
      net_value = assets[portfilio].get('net_value')
      updated = assets[portfilio].get('updated')
    else:
      net_value = '(Not Found)'
      updated = '/'  
    asset_info += f'<li><b><p>{portfilio}: ${net_value}</b></p>({updated})</li>'
  asset_info += '</ul>'

  # get signal summary
  signal_info = '<h3>Signals</h3><ul>'
  signal_color = {'b':'green', 's':'red', 'n':'grey'}
  if signal_file_date is None:
    signal_file_date =datetime.datetime.now().date().strftime(format='%Y-%m-%d')
  signal_file = f'{config["result_path"]}{signal_file_date}.xlsx'
  if os.path.exists(signal_file):
    signals = pd.read_excel(signal_file, sheet_name='signal')
    for s in ['b', 's', 'n']:
      font_color = signal_color[s]
      tmp_signals = signals.query(f'交易信号 == "{s}"')['代码'].tolist()
      signal_info += f'<li>[ <b>{s}</b> ]: <font color="{font_color}">{", ".join(tmp_signals)}</font></li>'
  else:
    signal_info += f'<li><p>[Not Found]: {signal_file}</p></li>'
  signal_info += '</ul>'

  # attachment 1: log file
  log_info = '<h3>Log</h3><ul>'
  if log_file_date is None:
    log_file_date = util.string_plus_day(string=signal_file_date, diff_days=-1)
  log_file = f'{config["quant_path"]}tiger_trade_log_{log_file_date}.txt'
  if os.path.exists(log_file):
    log_part = MIMEApplication(open(log_file, 'rb').read())
    log_part.add_header('Content-Disposition', 'attachment', filename=log_file)
    log_info += f'<li><p>[Attached]</p></li>'
  else:
    log_info += f'<li><p>[Not Found]: {log_file}</p></li>'
    log_part = None
  log_info += '</ul>'
  
  # attachment 2: signal images
  image_info = f'<h3> Images</h3><ul><li>[Requested]: {signal_file_date}'
  images = []
  wrong_date = {}
  for symbol in config['selected_sec_list'][config['trade']['pool']['global_account']]:
    signal_image = f'{config["result_path"]}signal/{symbol}_day.png'
    if os.path.exists(signal_image):

      # check whether the signal image is up-to-date
      image_create_date = util.timestamp_2_time(timestamp=os.path.getctime(signal_image), unit='s').date().strftime(format='%Y-%m-%d')
      if image_create_date != signal_file_date and image_create_date not in wrong_date.keys():
        wrong_date[image_create_date] = [symbol]
      
      if image_create_date in wrong_date:
        wrong_date[image_create_date].append(symbol)

      with open(signal_image, 'rb') as fp:
        symbol_image = MIMEImage(fp.read())
      images.append(symbol_image)

    else:
      image_info += f'<li><p>[Not Found]: image for {symbol}</p></li>'
  
  if len(wrong_date) > 0:
    for wd in wrong_date.keys():
      image_info += f'<li><p>[Wrong Date]: {wd}</p>{", ".join(wrong_date[wd])}</li>'
  image_info += '</ul>'

  msg_part = MIMEText(f'<html><body>{asset_info}{signal_info}{log_info}{image_info}</body></html>','html','utf-8')
  m.attach(msg_part)

  if log_part is not None:
    m.attach(log_part)

  for i in images:
    m.attach(i)

  # start SMTP service, send email, stop SMTP service
  server=smtplib.SMTP_SSL(smtp_server)
  server.connect(smtp_server,465)
  server.login(from_addr, password)
  server.sendmail(from_addr, to_addr, m.as_string())
  ret = server.quit()

  return ret