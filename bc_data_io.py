# -*- coding: utf-8 -*-
"""
Data IO Utilities

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

# mail process
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# self defined
from quant import bc_util as util

# global variables
default_eod_key = 'OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX'


#----------------------------- Stock Data -------------------------------------#
def get_symbols(remove_invalid=True, save_path=None, save_name='symbol_list.csv', local_file=None):
  """
  Get Nasdaq stock list

  :param remove_invalid: whether to remove invalid stock symbols from external stock list (.csv)
  :param save_path: where to save the symbol list, generally it will be saved at ~/quant/stock/data/
  :param save_name: the name of the saved symbol list file, defaultly it will be symbol_list.csv
  :returns: dataframe of stock symbols
  :raises: exception when error reading not-fetched symbols list
  """
  # get the symbols from pandas_datareader
  if local_file is not None and os.path.exists(local_file):
    symbols = pd.read_csv(local_file).set_index('Symbol')

  else:
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

    symbols = symbols.loc[sec_list, ].copy()

    if save_path is not None:
      symbols.reset_index().to_csv(f'{save_path}{save_name}', index=False)
  
  return symbols


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
      print(f'[From Yahoo] {symbol:4}: {data.index.min().date()} - {data.index.max().date()}, 下载记录 {len(data)}')

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
      print(f'[From YFinance] {symbol:4}: {data.index.min().date()} - {data.index.max().date()}, 下载记录 {len(data)}')

  except Exception as e:
      print(symbol, e)
      
  return data 


def get_data_from_eod(symbol, start_date=None, end_date=None, interval='d', is_print=False, api_key=default_eod_key, add_dividend=True, add_split=True):
  """
  Download stock data from EOD

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param interval: available values - d, w, m
  :param is_print: whether to print download information
  :param api_key: api token to access eod data
  :param add_dividend: whether to add dividend data
  :param add_split: whether to add split data
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  

  data = None

  try:
    # initialize from - to parameters
    from_to = ''
    if start_date is not None:
      from_to += f'&from={start_date}'
    if end_date is not None:
      from_to += f'&to={end_date}'

    # get eod data (ohlcv)
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}?api_token={api_key}&period={interval}&fmt=json{from_to}'
    response = requests.get(url)
    if response.status_code==200: 
      eod = response.json()
      response_status = 'o' 
    else:
      eod = []
      response_status = 'x' 
    
    # post process eod data, add dividend and split(optional)
    if len(eod) > 0:

      # convert downloaded data from json to dataframe
      data = pd.DataFrame(eod)
      if is_print:
        print(f'{symbol:4}: {data.date.min()} - {data.date.max()}, 下载记录 {len(data)}, eod({response_status})', end=', ')
      
      # add dividend data
      if add_dividend:

        # get dividend data
        url = f'https://eodhistoricaldata.com/api/div/{symbol}?api_token={api_key}&fmt=json{from_to}'
        response = requests.get(url)
        if response.status_code==200: 
          dividend = response.json()
          response_status = 'o' 
        else:
          dividend = []
          response_status = 'x' 

        if is_print:
          print(f'dividend({response_status})', end=', ')

        # post process dividend data
        if len(dividend) > 0:
          dividend_data = pd.DataFrame(dividend)
          dividend_data = dividend_data[['date', 'value']].copy()
          dividend_data = dividend_data.rename(columns={'value':'dividend'})
          data = pd.merge(data, dividend_data, how='left', left_on='date', right_on='date')
          data['dividend'] = data['dividend'].fillna(0.0)
      
      # add split data
      if add_split:

        # get split data
        url = f'https://eodhistoricaldata.com/api/splits/{symbol}?api_token={api_key}&fmt=json{from_to}'  
        response = requests.get(url)
        if response.status_code==200: 
          split = response.json()
          response_status = 'o' 
        else:
          split = []
          response_status = 'x' 

        if is_print:
          print(f'split({response_status})')
        
        # post process dividend data
        if len(split) > 0:
          split_data = pd.DataFrame(split)
          split_data.split = split_data.split.apply(lambda x: float(x.split('/')[0])/float(x.split('/')[1]))
          data = pd.merge(data, split_data, how='left', left_on='date', right_on='date')
          data['split'] = data['split'].fillna(1.0)
          
      # fill na values for dividend and split    
      if 'dividend' not in data.columns:
        data['dividend'] = 0.0
      if 'split' not in data.columns:
        data['split'] = 1.0
        
    # if eod data not available, return empty dataframe
    else:
      data = pd.DataFrame()

    # convert to timeseries dataframe, rename columns
    data = post_process_download_data(df=data, source='eod')

  except Exception as e:
    print(symbol, e)

  return data


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

    # post process data downloaded from eod
    elif source == 'eod':
      df = df.rename(columns={'date':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume', 'dividend':'Dividend', 'split': 'Split'})
      df = util.df_2_timeseries(df=df, time_col='Date')   

  return df


def get_data(symbol, start_date=None, end_date=None, interval='d', is_print=False, source='eod', time_col='Date', api_key=default_eod_key, add_dividend=True, add_split=True):
  """
  Download stock data from web sources

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param source: datasrouce: 'yahoo' or 'yfinance'
  :param time_col: column name of datetime column
  :param interval: period, for yahoo: d/w/m/v; for yfinance: 1d/1wk/1mo; for eod: d/w/m
  :param is_print: whether to print download information
  :param api_key: api token to access eod data
  :param add_dividend: whether to add dividend data
  :param add_split: whether to add split data
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
    
    # eod
    elif source == 'eod':
      data = get_data_from_eod(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, is_print=is_print, api_key=api_key, add_dividend=add_dividend, add_split=add_split)
    
    # otherwise
    else:
      print(f'data source {source} not found')

  except Exception as e:
    print(symbol, e)

  return data


def get_real_time_data_from_eod(symbols, api_key=default_eod_key, is_print=False, batch_size=15):
  """
  Download real-time stock data from EOD

  :param symbols: list of symbols
  :param api_key: api token to access eod data
  :param is_print: whether to print download information
  :param batch_size: batch size of symbols of getting real-time data
  :returns: dataframe or None
  :raises: exception when downloading failed
  """

  # initialize result
  result = pd.DataFrame()

  # divid symbol list into batches
  symbol_list_len = len(symbols)
  batch_start = 0
  while batch_start < symbol_list_len:
    
    # calculate batch end according to batch start and batch size
    batch_end = batch_start + batch_size
    if batch_end > symbol_list_len:
      batch_end = symbol_list_len
    
    # get temporal symbol list
    tmp_list = symbols[batch_start:batch_end]
    first_symbol = tmp_list[0]
    other_symbols = ','.join(tmp_list[1:])
  
    # get real-time data for current batch
    url = f'https://eodhistoricaldata.com/api/real-time/{first_symbol}?api_token={api_key}&fmt=json&s={other_symbols}'
    response = requests.get(url)
    real_time = [] if response.status_code!=200 else response.json()
    real_time = [real_time] if isinstance(real_time, dict) else real_time
    
    if is_print:
      print(f'updating real-time for symbol: {batch_start+1:3} -{batch_end:3} - {response}')      
  
    # post process downloaded real-time data
    real_time_data = pd.DataFrame(real_time)
    tz = datetime.timezone(datetime.timedelta(hours=-5))
    for idx, row in real_time_data.iterrows():
      real_time_data.loc[idx, 'latest_time'] = util.timestamp_2_time(real_time_data.loc[idx, 'timestamp'], unit='s').astimezone(tz).replace(tzinfo=None)
      real_time_data.loc[idx, 'code'] = real_time_data.loc[idx, 'code'].split('.')[0]

    real_time_data['Date'] = real_time_data['latest_time'].apply(lambda x: x.date())
    real_time_data['Adj Close'] = real_time_data['close'].copy()
    real_time_data['dividend'] = 0.0
    real_time_data['split'] = 1.0
    
    # concate batches of real-time data 
    result = result.append(real_time_data)
    batch_start = batch_end
    
  result.rename(columns={'code':'symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume', 'dividend':'Dividend', 'split': 'Split'}, inplace=True)
  result.drop(['timestamp', 'gmtoffset'], axis=1, inplace=True)
    
  return result


def get_real_time_data_from_yfinance(symbols, period='1d', interval='1m'):

  # initialize result
  latest_data = pd.DataFrame()

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
      for symbol in symbols:

        # get the latest data for current symbol        
        tmp_data = ticker_data[symbol].dropna().reset_index()
        tmp_latest_data = tmp_data.tail(1).copy()
        
        # process latest data for current symbol
        tmp_latest_data['symbol'] = symbol
        tmp_latest_data['Open'] = tmp_data.Open.values[0]
        tmp_latest_data['High'] = tmp_data.High.max()
        tmp_latest_data['Low'] = tmp_data.Low.min()
        tmp_latest_data['Volume'] = tmp_data.Volume.sum()
        tmp_latest_data['Date'] = tmp_latest_data['Datetime'].max().date()
        
        # append to final result
        latest_data = latest_data.append(tmp_latest_data)
      
      # process final result
      latest_data['change'] = latest_data['Close'] - latest_data['Open']
      latest_data['change_p'] = latest_data['change'] / latest_data['Open'] * 100
      latest_data['previousClose'] =  latest_data['Open']
      latest_data['Dividend'] = 0.0
      latest_data['Split'] = 1.0
      
      latest_data = latest_data.reset_index().drop('index', axis=1).round(3)
      latest_data = latest_data.rename(columns={'Datetime': 'latest_time'})
      latest_data = latest_data[['change', 'change_p', 'Close', 'symbol', 'High', 'Low', 'Open', 'previousClose', 'Volume', 'latest_time', 'Date', 'Adj Close', 'Dividend', 'Split']]

  return latest_data


def get_stock_briefs_from_eod(symbols, api_key=default_eod_key, batch_size=15):
  """
  Get latest stock data for symbols from eod

  :param symbols: list of target symbols
  :param api_key: api token to access eod data
  :param batch_size: batch size of symbols of getting real-time data
  :returns: dataframe of latest data, per row for each symbol
  :raises: none
  """
  latest_data = get_real_time_data_from_eod(symbols=symbols, api_key=api_key, batch_size=batch_size, is_print=True)

  if len(latest_data) > 0:
    latest_data['latest_price'] = latest_data['Close'].copy()
    latest_data = latest_data[['latest_time', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'symbol', 'latest_price', 'Date']]
  else:
    print('real time data from eod is empty')

  return latest_data


def get_stock_briefs_from_yfinance(symbols, period='1d', interval='1m'):
  """
  Get latest stock data for symbols from yahoo

  :param symbols: list of target symbols
  :param period: how long the period you want to download data
  :param interval: available values - 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
  :returns: dataframe of latest data, per row for each symbol
  :raises: none
  """
  latest_data = get_real_time_data_From_yfinance(symbols=symbols, period=period, interval=interval)

  if len(latest_data) > 0:
    latest_data['latest_price'] = latest_data['Close'].copy()
    latest_data = latest_data[['latest_time', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'symbol', 'latest_price', 'Date']]
  else:
    print('real time data from yfinance is empty')

  return latest_data


def get_stock_briefs(symbols, source='eod', period='1d', interval='1m', api_key=default_eod_key, batch_size=15):
  """
  Get latest stock data for symbols

  :param symbols: symbol list
  :param source: data source
  :param period: how long the period to download
  :param interval: in which interval to download
  :param api_key: api token to access eod data
  :param batch_size: batch size of symbols of getting real-time data
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """
  # initialize
  briefs = None

  # currently only yfinance is available
  if source == 'yfinance':
    briefs =  get_stock_briefs_from_yfinance(symbols=symbols, period=period, interval=interval)

  elif source == 'eod':
    briefs = get_stock_briefs_from_eod(symbols=symbols, api_key=api_key, batch_size=batch_size)

  else:
    print(f'Unknown source {source}')

  return briefs


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
  update local stock data from yfinance by stock

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
  update local stock data from yahoo by date

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


def update_stock_data_from_eod(symbols, stock_data_path, file_format='.csv', required_date=None, window_size=3, is_print=False, is_return=False, is_save=True, api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15, update_mode='eod'):
  """
  update local stock data from eod

  :param symbols: list of target symbols
  :param stock_data_path: where the local stock data files(.csv) stored
  :param file_format: default is .csv
  :param required_date: if the local data have already meet the required date, it won't be updated
  :param is_print: whether to print info when downloading
  :param is_return: whether to return the updated data
  :param is_save: whether to save the updated data to local files
  :param api_key: api token to access eod data
  :param add_dividend: whether to add dividend data
  :param add_split: whether to add split data
  :param batch_size: batch size of symbols of getting real-time data
  :param update_mode: how to update data - realtime/eod/both(eod+realtime)/refresh(delete and redownload)
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """

  # verify update_mode
  if update_mode not in ['realtime', 'eod', 'both', 'refresh']:
    print(f'unknown update mode: {update_mode}')
    return None

  # get the benchmark of eod data
  today = util.time_2_string(datetime.datetime.today().date())
  start_date = util.string_plus_day(today, -7)
  benchmark_data = get_data_from_eod(symbol='AAPL', start_date=start_date, end_date=today, interval='d', is_print=False, api_key=api_key, add_dividend=False, add_split=False)
  benchmark_date = util.time_2_string(benchmark_data.index.max())
  start_date = util.string_plus_day(benchmark_date, -window_size)

  # get the existed data and its latest date for each symbols
  print(f'******************** updating eod-data ********************')
  data = {}
  for symbol in symbols:
    
    # init symbol data and its most recent date
    data[symbol] = pd.DataFrame()
    tmp_data_date = None

    # if local data exists, load existed data, update its most current date
    symbol_file_name = f'{stock_data_path}{symbol}{file_format}'
    if os.path.exists(symbol_file_name):
      
      # delete local data if update_mode == refresh
      if update_mode == 'refresh':
        os.remove(symbol_file_name)
        
      # load local data and update its most recent date
      else:
        data[symbol] = load_stock_data(file_path=stock_data_path, file_name=symbol)
        if len(data[symbol]) > 0:
          tmp_data_date = util.time_2_string(data[symbol].index.max())

    # update eod data, print updating info
    if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_date):
      if is_print:
        print(f'updating from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
      
      # download latest data for current symbol
      new_data = get_data_from_eod(symbol, start_date=tmp_data_date, end_date=required_date, interval='d', is_print=is_print, api_key=api_key, add_dividend=add_dividend, add_split=add_split)
    
      # append new data to the origin
      data[symbol] = data[symbol].append(new_data, sort=True)
      data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

      # save data to local csv files
      if is_save:
        save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
    
    else:
      if is_print:
        print(f'updating from {tmp_data_date} {symbol:4}: already up-to-date, skip')

  # add real-time data when requiring data return and data will NOT be saved
  if update_mode in ['realtime', 'both']:
    print('***************** querying real-time data *****************')

    # get real-time data from EOD, convert it into time-series data
    real_time_data = get_real_time_data_from_eod(symbols=symbols, api_key=api_key, is_print=is_print, batch_size=batch_size)
    real_time_data = util.df_2_timeseries(df=real_time_data, time_col='Date')

    # append it corresponding eod data according to symbols
    for symbol in symbols:
        tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
        data[symbol] = data[symbol].append(tmp_data)
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

  # return
  if is_return:
    return data


def update_stock_data(symbols, stock_data_path, file_format='.csv', source='eod', by='date', required_date=None, is_print=False, is_return=False, is_save=True, api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15, update_mode='eod'):
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
  :param update_mode: how to update data - realtime/eod/both(eod+realtime)/refresh(delete and redownload)
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """

  result = None

  if source == 'yfinance':
    if by == 'date':
      result = update_stock_data_from_yfinance_by_date(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save)
    elif by == 'stock':
      result = update_stock_data_from_yfinance_by_stock(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save)
  elif source == 'eod':
    result = update_stock_data_from_eod(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, api_key=api_key, add_dividend=add_dividend, add_split=add_split, batch_size=batch_size, update_mode=update_mode)
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


#----------------------------- Json File Processing---- -------------------------#
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
  """
  Save dictionary into an excel file
  :param dictionary: target dict
  :param file_path: where to save the excel file
  :param file_name: name of the exccel file
  :param keep_index: whether to keep index
  :return: None
  :raise: None
  """
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


#----------------------------- Email sending ------------------------------------#
def send_result_by_email(config, to_addr, from_addr, smtp_server, password, subject=None, platform=['tiger'], signal_file_date=None, log_file_date=None, position_summary={}, test=False):
  """
  send automatic_trader's trading result by email

  :param config: dictionary of config, include pathes, etc.
  :param to_addr: destination email address
  :param from_addr: email address used for sending email
  :param smtp_server: smtp server address
  :param password: pwd for smtp server
  :param subject: subject of the email
  :param platform: trading platforms include ['tiger', 'futu']
  :param signal_file_date: date of signal file which will be attached on email
  :param log_file_date: date of log file which will be attached on email
  :param position_summary: dictionary of dataframes which contain position info
  :param test: whether on test mode(print message rather than send the email)
  :return: smtp ret code
  :raise: none
  """

  # get current time
  current_time = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")

  # construct email 
  m = MIMEMultipart()
  if subject is not None:
    m['Subject'] = subject
  else:
    m['Subject'] = f'[auto_trade] {current_time}'
  
  # get portfolio record
  assets = {}
  positions = {}
  if os.path.exists(config['config_path']+'portfolio.json'):
    portfolio_record = read_config(file_path=config['config_path'], file_name='portfolio.json')
    if 'tiger' in platform:
      pr = portfolio_record.get('tiger')
      assets['glob'] = pr.get('global_account')
      assets['simu'] = pr.get('simulation_account') 

    if 'futu' in platform:
      pr = portfolio_record.get('futu')
      assets['REAL'] = pr.get('REAL')
      assets['SIMU'] = pr.get('SIMULATE')

  # construct asset summary
  asset_info = '<h3>Assets</h3><ul>'
  for portfilio in assets.keys():
    if assets[portfilio] is not None:
      net_value = assets[portfilio].get('net_value')
      updated = assets[portfilio].get('updated')

      # add position summary if provided
      position = position_summary.get(portfilio)
      position = position.drop('latest_time', axis=1).to_html() if position is not None else ''
    else:
      net_value = '--'
      updated = '--'  
      position = ''

    # mark update-time if not not match with current time
    if updated[:16] == current_time[:16]:
      updated = ''
    else:
      updated = f'({updated})'

    # asset summary for current portfilio
    asset_info += f'<li><b><p>{portfilio}: ${net_value}</b></p>{position}{updated}</li>'
  asset_info += '</ul>'

  # get signal summary
  signal_info = '<h3>Signals</h3><ul>'
  signal_color = {'b':'green', 's':'red', 'n':'grey'}
  if signal_file_date is not None:
    signal_file = f'{config["result_path"]}{signal_file_date}.xlsx'
    if os.path.exists(signal_file):
      signals = pd.read_excel(signal_file, sheet_name='signal')
      for s in ['b', 's', 'n']:
        font_color = signal_color[s]
        tmp_signals = signals.query(f'交易信号 == "{s}"')['代码'].tolist()
        signal_info += f'<li>[ <b>{s}</b> ]: <font color="{font_color}">{", ".join(tmp_signals)}</font></li>'
    else:
      signal_info += f'<li><p>[Not Found]: {signal_file}</p></li>'
  else:
    signal_info += '<li><p>[Not Required]</p></li>'
  signal_info += '</ul>'

  # attachment 1: log file
  log_info = '<h3>Log</h3><ul>'
  if log_file_date is not None:
    # log_file_date = util.string_plus_day(string=signal_file_date, diff_days=-1)

    log_file = f'{config["log_path"]}automatic_trade_log_{log_file_date}.txt'
    if os.path.exists(log_file):
      log_part = MIMEApplication(open(log_file, 'rb').read())
      log_part.add_header('Content-Disposition', 'attachment', filename=log_file)
      log_info += f'<li><p>[Attached]</p></li>'
    else:
      log_info += f'<li><p>[Not Found]: {log_file}</p></li>'
      log_part = None
  else:
    log_info += '<li><p>[Not Required]</p></li>'
    log_part = None
  log_info += '</ul>'
  
  # attachment 2: signal images
  image_info = f'<h3> Images</h3><ul>'
  images = []
  if signal_file_date is not None:
    image_info += f'<li>[Requested]: {signal_file_date}</li>'
    wrong_date = {}
    for symbol in config['selected_sec_list']['all']: #[config['trade']['pool']['global_account']]:
      signal_image = f'{config["result_path"]}signal/{symbol}_day.png'
      if os.path.exists(signal_image):

        # check whether the signal image is up-to-date
        image_create_date = util.timestamp_2_time(timestamp=os.path.getmtime(signal_image), unit='s').date().strftime(format='%Y-%m-%d')
        if image_create_date != signal_file_date and image_create_date not in wrong_date.keys():
          wrong_date[image_create_date] = [symbol]
        
        if image_create_date in wrong_date:
          wrong_date[image_create_date].append(symbol)

        with open(signal_image, 'rb') as fp:
          symbol_image = MIMEImage(fp.read())
        images.append(symbol_image)

      else:
        if symbol in config['selected_sec_list']['auto']: 
          image_info += f'<li><p>[Not Found]: image for {symbol}</p></li>'
        else:
          continue

    if len(wrong_date) > 0:
      for wd in wrong_date.keys():
        image_info += f'<li><p>[Actual Date]: {wd}</p>{", ".join(wrong_date[wd])}</li>'
  else:
    image_info += f'<li><p>[Not Required]</p></li>'
  image_info += '</ul>'

  # construct message part by concating info parts
  full_info = f'<html><body>{asset_info}{signal_info}{log_info}{image_info}</body></html>'
  msg_part = MIMEText(full_info,'html','utf-8')
  m.attach(msg_part)

  if log_part is not None:
    m.attach(log_part)

  for i in images:
    m.attach(i)

  # if test, print info parts, else send the email with attachments
  if test:
    print(full_info)
    ret = 'test'
  else:
    # start SMTP service, send email, stop SMTP service
    server=smtplib.SMTP_SSL(smtp_server)
    server.connect(smtp_server,465)
    server.login(from_addr, password)
    server.sendmail(from_addr, to_addr, m.as_string())
    ret = server.quit()

  return ret