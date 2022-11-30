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
import pytz
import zipfile
import pickle
import json
import os

# data source
import yfinance as yf
import pandas_datareader.data as web 
import easyquotation as eq
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

# mail process
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# self defined
from quant import bc_util as util

# global variables
est_tz = pytz.timezone('US/Eastern')
utc_tz = pytz.timezone('UTC')
default_eod_key = 'OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX'
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36',
  }


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


def get_data_from_yahoo(symbol, start_date=None, end_date=None, time_col='Date', is_print=False):
  """
  Download daily stock data from Yahoo finance api via pandas_datareader

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param time_col: column name of datetime column
  :param is_print: whether to print download information
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  data = None

  try:
    # download data
    data = web.DataReader(symbol, 'yahoo', start_date, end_date)
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
    response = requests.get(url, headers=headers)
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
        print(f'{symbol:5}: {data.date.min()} - {data.date.max()}, 下载记录 {len(data)}, eod({response_status})', end=', ')
      
      # add dividend data
      if add_dividend:

        # get dividend data
        url = f'https://eodhistoricaldata.com/api/div/{symbol}?api_token={api_key}&fmt=json{from_to}'
        response = requests.get(url, headers=headers)
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
        response = requests.get(url, headers=headers)
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


def get_data_from_iex(symbol, interval, api_key, chart_last=None, include_today=True, chart_by_day=True, sandbox=False, is_print=False):
  """
  Download stock data from IEX cloud
  https://iexcloud.io/docs/api/#historical-prices

  :param symbol: target symbol
  :param interval: dynamic/date/5dm/1mm/5d/1m/3m/6m/ytd/1y/2y/5y/max
  :param is_print: whether to print download information
  :param api_key: api token to access eod data
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  data = None
  
  # select a base url for normal/sandbox mode
  url = f'https://cloud.iexapis.com/stable/'
  if sandbox:
    url = f'https://sandbox.iexapis.com/stable/'  
    
  # add symbol, interval and api_key
  url += f'stock/{symbol}/chart/{interval}?token={api_key}'
  
  # add extra parameters
  if chart_last is not None:
    url += f'&chartLast={chart_last}'
  if include_today:
    url += '&includeToday=true'
  if chart_by_day:
    url += '&chartByDay=true'
  
  # request data from iex
  response_code = None
  try:
    response = requests.get(url, headers=headers)
    response_code = response.status_code

    # get data in json format
    if response_code==200:
      data_json = response.json()

      # convert json to dataframe
      if len(data_json) > 0:
        data = pd.DataFrame(data_json)
        data = post_process_download_data(df=data, source='iex')

        if is_print:
          print(f'{symbol:4}: {data.index.min()} - {data.index.max()}, 下载记录 {len(data)} from iex cloud')
      
    else:
      if is_print:
        print(f'{symbol}({response_code})')

  except Exception as e:
    print(f'{symbol}({response_code}): {e}')    

  return data


def request_marketstack(field, parameters, is_print=False):
  """
  request data from marketstack

  :param field: target field, such as eod/intraday/exchanges/currencies/tickers, etc.
  :param parameters: dictionary of parameters for http request
  :returns: http response
  :raises: None
  """

  base_url = 'http://api.marketstack.com/v1/'
  url = f'{base_url}{field}'
  if is_print:
    print(url, parameters)
  response = requests.get(url, parameters, headers=headers)
  
  return response


def get_data_from_marketstack(symbol, api_key, start_date=None, end_date=None, limit=1000, is_print=False):
  """
  Download stock data from marketstack

  :param symbol: target symbol
  :param api_key: api token to access eod data
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param limit: number of rows to download, max is 1000
  :param is_print: whether to print download information
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  # initialize result
  data = pd.DataFrame()
  
  # set default dates
  if end_date is None:
    end_date = util.time_2_string(datetime.datetime.today())
  if start_date is None:
    start_date = util.string_plus_day(end_date, -limit)

  # print(start_date, end_date)
  # return None
  
  # request data
  finish_downloading = False
  while not finish_downloading:

    # url parameters
    field = 'eod'
    parameters = {
      'access_key': api_key,
      'symbols': symbol,
      'date_from': start_date,
      'date_to': end_date,
      'sort': 'ASC',
      'limit': limit
    }
    response = request_marketstack(field=field, parameters=parameters)

    # extract data from response
    response_json = util.response_2_json(response=response, print_status=False)
    if response_json is not None:

      # initialize
      tmp_end = None
      tmp_data = None
      tmp_len = 0
      
      # get dataframe from response.json()
      json_data = response_json.get('data')
      if json_data is not None:
        tmp_data = pd.DataFrame(json_data)
        tmp_len = len(tmp_data)
        tmp_end = tmp_data.date.max()[:10]
        data = pd.concat(data, tmp_data) 
      else:
        print('no data in json')

      # print downloading info
      if is_print:
        print(f'status({response.status_code}), {symbol:4}: ', end='')
        if tmp_data is not None and tmp_len > 0:
          print(f'{tmp_data.date.min()[:10]} - {tmp_data.date.max()[:10]}, 下载记录 {tmp_len}')
        else:
          print(f'get empty data')

      # whether downloading finished
      if (tmp_len == 0) or (0 < tmp_len and tmp_len < limit) or (tmp_end >= end_date):
        finish_downloading = True
      else:
        start_date = tmp_end
    else:
        print('no json in response')

  # postprocess downloaded data
  data = post_process_download_data(df=data, source='marketstack')    

  return data


def post_process_download_data(df, source):
  """
  Post process data downloaded from certain data source

  :param df: stock data dataframe downloaded from source
  :param source: data source
  :returns: post processed data
  :raises: none
  """
  if df is not None and len(df)>0:

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
    
    # post process data downloaded from iex
    elif source == 'iex':
      new_cols = {'fOpen': 'Open', 'fHigh': 'High', 'fLow': 'Low', 'fClose': 'Close', 'fVolume': 'Volume', 'date': 'Date'}
      df = df.rename(columns=new_cols)
      df['Split'] = 1.0
      df['Dividend'] = 0.0
      df['Adj Close'] = df['Close']
      df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split', 'Date']].copy()
      df = util.df_2_timeseries(df=df, time_col='Date')

    # post process data download from marketstack
    elif source == 'marketstack':
      df['date'] = df['date'].apply(lambda x: x[:10])
      df = df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'adj_close': 'Adj Close', 'split_factor':'Split', 'date': 'Date'})
      df['Dividend'] = 0.0
      df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividend', 'Split']].copy()
      df = util.df_2_timeseries(df=df, time_col='Date')
    
    # remove duplicated index and sort data by index
    df = util.remove_duplicated_index(df, keep='last')
    df.sort_index(ascending=True, inplace=True)

  return df


def get_data(symbol, start_date=None, end_date=None, interval='d', is_print=False, source='eod', time_col='Date', api_key=default_eod_key, add_dividend=True, add_split=True, chart_last=None, include_today=True, chart_by_day=True, sandbox=False, limit=1000):
  """
  Download stock data from web sources

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param source: datasrouce: 'yahoo' or 'yfinance'
  :param time_col: column name of datetime column
  :param interval: period, for yahoo: d/w/m/v; for yfinance: 1d/1wk/1mo; for eod: d/w/m
  :param is_print: whether to print download information
  :param api_key: api token to access eod or iex data
  :param add_dividend: [eod] whether to add dividend data
  :param add_split: [eod] whether to add split data
  :param chart_last: [iex] how many rows to download from current date
  :param include_today: [iex] include the data of current date
  :param chart_by_day: [iex] download the data in day interval
  :param sandbox: [iex] whether to use sandbox environment for testing
  :param limit: [marketstack] number of rows to download
  :returns: dataframe 
  :raises: exception when downloading failed
  """
  data = None

  try:
    # yahoo
    if source in ['yahoo', 'yfinance']:
      # data = get_data_from_yahoo(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
      # data = get_data_from_yfinance(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, time_col=time_col, is_print=is_print)
      print('Yahoo finance as the data source is very un-stable, please use other sources')
      
    # eod
    elif source == 'eod':
      data = get_data_from_eod(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, is_print=is_print, api_key=api_key, add_dividend=add_dividend, add_split=add_split)
    
    # iex
    elif source == 'iex':
      data = get_data_from_iex(symbol=symbol, interval=interval, api_key=api_key, chart_last=chart_last, include_today=include_today, chart_by_day=chart_by_day, sandbox=sandbox, is_print=is_print)

    # marketstack
    elif source == 'marketstack':
      data = get_data_from_marketstack(symbol=symbol, api_key=api_key, start_date=start_date, end_date=end_date, limit=limit, is_print=is_print)

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
    response = requests.get(url, headers=headers)
    try:
      real_time = [] if response.status_code!=200 else response.json()
    except Exception as e:
      print(e)
      print(response.status_code, response) 
      real_time = []
    real_time = [real_time] if isinstance(real_time, dict) else real_time
    
    if is_print:
      print(f'updating real-time for symbol: {batch_start+1:3} -{batch_end:3} - {response}')      
  
    # post process downloaded real-time data
    real_time_data = pd.DataFrame(real_time)
    to_drop = []
    for idx, row in real_time_data.iterrows():
      
      try:
        real_time_data.loc[idx, 'latest_time'] = util.timestamp_2_time(real_time_data.loc[idx, 'timestamp'], unit='s').astimezone(est_tz).replace(tzinfo=None)
        real_time_data.loc[idx, 'code'] = real_time_data.loc[idx, 'code'].split('.')[0]
      except Exception as e:
        print(e)
        print(row)
        to_drop.append(idx)
        continue
    
    real_time_data.drop(to_drop, inplace=True)
    if 'latest_time' in real_time_data.columns:
      real_time_data['Date'] = real_time_data['latest_time'].apply(lambda x: x.date())
    else:
      real_time_data['Date'] = None
      
    real_time_data['Adj Close'] = real_time_data['close'].copy()
    real_time_data['dividend'] = 0.0
    real_time_data['split'] = 1.0
    
    # concate batches of real-time data 
    result = pd.concat([result, real_time_data]) 
    batch_start = batch_end
    
  result.rename(columns={'code':'symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume', 'dividend':'Dividend', 'split': 'Split'}, inplace=True)
  result.drop(['timestamp', 'gmtoffset'], axis=1, inplace=True)

  result

  return result


def get_real_time_data_from_yfinance(symbols, period='1d', interval='1m'):
  """
  Download real-time stock data from yfinance

  :param symbols: list of symbols
  :param period: period of the downloaded data
  :param interval: interval of the downloaded data
  :returns: dataframe
  :raises: None
  """

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
        latest_data = pd.concat([latest_data, tmp_latest_data]) 
      
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


def get_real_time_data_from_easyquotation(symbols, source='sina'):
  """
  Download real-time stock data (chinese stock only) from easyquotation

  :param symbols: list of symbols
  :param source: data source, sina/tecent/qq
  :returns: dataframe or None
  :raises: exception when downloading failed
  """

  df = pd.DataFrame()

  # create quoataion entity
  quotation = eq.use(source)
  print(f'updating real-time for symbols from easyquotation({source})')

  # get stock data
  codes = [x.split('.')[0] for x in symbols]
  qt = quotation.stocks(codes) 
  df = pd.DataFrame(qt).T

  idxs = df.index.tolist()
  idx_symbols = [x for x in symbols if x.split('.')[0] in idxs]

  # post process
  df = df.reset_index()
  df['index'] = idx_symbols
  
  columns_to_keep = {'index':'symbol', 'open':'Open', 'high':'High', 'low':'Low', 'now':'Close', 'turnover':'Volume', 'date':'Date'}
  
  df = df.rename(columns=columns_to_keep)[columns_to_keep.values()].copy()
  df = util.df_2_timeseries(df=df, time_col='Date')
  df['Adj Close'] = df['Close']
  df['Dividend'] = 0.0
  df['Split'] = 1.0
  
  df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

  return df


def get_real_time_data_from_marketstack(symbols, api_key, is_print=False, batch_size=100, limit=1000):
  """
  Download real-time stock data from marketstack
  date in UTC(EST=UTC-5, EDT=UTC-4)

  :param symbols: list of symbols
  :param api_key: api token to access eod data
  :param is_print: whether to print download information
  :param limit: limit of rows for getting real-time data
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  # initialize result
  result = pd.DataFrame()

  # get current time
  utc_now = datetime.datetime.utcnow()#  util.time_2_string(, date_format='%Y-%m-%dT%H:%M:%S+0000')

  # if it is not valid period
  valid_period = True
  if utc_now.weekday() in [5, 6]:
    valid_period = False
    print('no data on weekends')
  if utc_now.time().hour < 13 or utc_now.time().hour > 21:
    valid_period = False
    print('no data when market closed')
  
  # get realtime symbol data by batches
  symbol_list_len = len(symbols)
  batch_start = 0
  while batch_start < symbol_list_len:
    
    # calculate batch end according to batch start and batch size
    batch_end = batch_start + batch_size
    if batch_end > symbol_list_len:
      batch_end = symbol_list_len
      
    # get temporal symbol list
    tmp_list = symbols[batch_start:batch_end]
    tmp_symbols = ','.join(tmp_list)

    # if it is valid period
    if valid_period:
      date_from = utc_now - datetime.timedelta(hours=1)
      date_from = util.time_2_string(date_from, date_format='%Y-%m-%dT%H:%M:%S+0000')
      date_to = util.time_2_string(utc_now, date_format='%Y-%m-%dT%H:%M:%S+0000')

      print(f'from {date_from} to {date_to}')
    
      # set request parameters
      field = 'intraday'
      parameters = {
        'access_key': api_key,
        'symbols': tmp_symbols,
        'date_from': date_from,
        'date_to': date_to,
        'interval': '15min',
        'limit': limit
      }

      # request data, extract data from response
      response = request_marketstack(field=field, parameters=parameters)
      response_json = util.response_2_json(response=response, print_status=False)
      if is_print:
        print(f'status({response.status_code}), updating real-time for symbols: {batch_start+1:3} -{batch_end:3}')

      # convert data from json to dataframe
      if response_json is not None:
        json_data = response_json.get('data')
        if json_data is not None:
          data = pd.DataFrame(json_data)
          
          # append latest data of a symbol onto the result
          if len(data) > 0:
            for symbol in tmp_list:
              symbol_data = data.dropna().query(f'symbol == "{symbol}"')
              if len(symbol_data) > 0:
                result = pd.concat([result, symbol_data.head(1)]) 

    # if not in valid period, get latest eod data instead
    else:
      field = 'eod/latest'
      parameters = {
            'access_key': api_key,
            'symbols': tmp_symbols,
            'limit': limit
      }

      # request data
      response = request_marketstack(field=field, parameters=parameters)
      response_json = util.response_2_json(response=response, print_status=False)
      if is_print:
        print(f'status({response.status_code}), getting latest eod for symbols: {batch_start+1:3} -{batch_end:3}')
     
      # convert data from json to dataframe
      if response_json is not None:
        json_data = response_json.get('data')
        if json_data is not None:
          data = pd.DataFrame(json_data)
          
          # append latest eod data of a symbols onto the result
          if len(data) > 0:
            result = pd.concat([result, data]) 
    
    # start next batch
    batch_start = batch_end
  
  # postprocess downloaded data
  if len(result) > 0:
    
    if valid_period:
      # reset index
      result.reset_index(inplace=True, drop=True)
      
      # convert timezone from UTC to EST
      result['date'] = result['date'].apply(lambda x: util.convert_timezone(util.string_2_time(x, date_format='%Y-%m-%dT%H:%M:%S+0000'), from_tz=utc_tz, to_tz=est_tz))
          
      # rename columns
      result.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'last':'Close', 'close':'previousClose', 'volume': 'Volume', 'date':'latest_time'}, inplace=True)
      
      # create extra columns
      result['Date'] = result['latest_time'].apply(lambda x: x.date())
      result['Adj Close'] = result['Close']
      result['Dividend'] = 0.0
      result['Split'] = 1.0

      # drop unnecessary columns
      result.drop(['exchange'], axis=1, inplace=True)

    else:
      result['date'] = result['date'].apply(lambda x: x[:10])
      result['latest_time'] = result['date'] + ' 16:00:00'
      result['previousClose'] = None
      result['Dividend'] = 0.0

      # rename columns
      result.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume': 'Volume', 'adj_close':'Adj Close', 'date':'Date', 'split_factor': 'Split'}, inplace=True)
    
      # drop unnecessary columns
      result.drop(['exchange', 'adj_high', 'adj_low', 'adj_open', 'adj_volume'], axis=1, inplace=True)
       
  return result


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


def get_stock_briefs_from_marketstack(symbols, api_key=default_eod_key, batch_size=100):
  """
  Get latest stock data for symbols from marketstack

  :param symbols: list of target symbols
  :param api_key: api token to access eod data
  :param batch_size: batch size of symbols of getting real-time data
  :returns: dataframe of latest data, per row for each symbol
  :raises: none
  """
  latest_data = get_real_time_data_from_marketstack(symbols=symbols, api_key=api_key, batch_size=batch_size, is_print=True)

  if len(latest_data) > 0:
    latest_data['latest_price'] = latest_data['Close'].copy()
    latest_data = latest_data[['latest_time', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'symbol', 'latest_price', 'Date']]
  else:
    print('real time data from marketstack is empty')

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
    tmp_data = pd.concat([old_data, new_data]) 
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
      data[symbol] = pd.concat([data[symbol], new_data])
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


def update_stock_data_from_eod(symbols, stock_data_path, file_format='.csv', required_date=None, window_size=3, is_print=False, is_return=False, is_save=True, api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15, update_mode='eod', cn_stock=False):
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
  :param cn_stock: whether updating chinese stocks
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
  data = {}
  up_to_date_symbols = []
  for symbol in symbols:
    
    # init symbol data and its most recent date
    data[symbol] = pd.DataFrame()
    tmp_data_date = None

    # for chinese stocks
    file_name = symbol
    splited = file_name.split('.')
    if len(splited) ==2 and splited[1] in ['SHG', 'SS', 'SHE', 'SZ']:
      file_name = splited[0]      

    # if local data exists, load existed data, update its most current date
    symbol_file_name = f'{stock_data_path}{file_name}{file_format}'
    if os.path.exists(symbol_file_name):
      
      # delete local data if update_mode == refresh
      if update_mode == 'refresh':
        os.remove(symbol_file_name)
        
    # load local data and update its most recent date
    existed_data = load_stock_data(file_path=stock_data_path, file_name=symbol)
    if existed_data is not None and len(existed_data) > 0:
      max_idx = existed_data.index.max()
      if max_idx > util.string_2_time('2020-01-01'):
        data[symbol] = existed_data
        tmp_data_date = util.time_2_string(max_idx)
      else:
        print(f'max index of {symbol} is invalid({max_idx}), refreshing data')
        os.remove(symbol_file_name)

    # update eod data, print updating info
    if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_date):
      if is_print:
        print(f'from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
      
      # download latest data for current symbol
      new_data = get_data_from_eod(symbol, start_date=tmp_data_date, end_date=required_date, interval='d', is_print=is_print, api_key=api_key, add_dividend=add_dividend, add_split=add_split)
    
      # append new data to the origin
      data[symbol] = pd.concat([data[symbol], new_data])
      data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

      # save data to local csv files
      if is_save:
        save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
    
    else:
      up_to_date_symbols.append(symbol)
      
  num_symbol_up_to_date = len(up_to_date_symbols)
  if num_symbol_up_to_date > 0:
    if is_print:
      print(f'from {tmp_data_date} ***** - [skip]: <already up-to-date {num_symbol_up_to_date}/{len(symbols)} >')

  # add real-time data when requiring data return and data will NOT be saved
  if update_mode in ['realtime', 'both']:
    print('***************** querying real-time data *****************')

    if not cn_stock:
      # get real-time data from EOD, convert it into time-series data
      real_time_data = get_real_time_data_from_eod(symbols=symbols, api_key=api_key, is_print=is_print, batch_size=batch_size)
      real_time_data = util.df_2_timeseries(df=real_time_data, time_col='Date')
    else:
      # get real-time data from easyquotation, convert it into time-series data
      real_time_data = get_real_time_data_from_easyquotation(symbols=symbols)

    # append it corresponding eod data according to symbols
    for symbol in symbols:
      tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
      if len(tmp_data) == 0:
        print(f'real-time data not found for {symbol}')
        continue
      else:
        tmp_idx = tmp_data.index.max()
        for col in data[symbol].columns:
          data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

  # return
  if is_return:
    return data


def update_stock_data_from_marketstack(symbols, stock_data_path, api_key, file_format='.csv', required_date=None, window_size=3, is_print=False, is_return=False, is_save=True, batch_size=100, update_mode='eod', cn_stock=False):
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
  :param cn_stock: whether updating chinese stocks
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """

  # verify update_mode
  if update_mode not in ['realtime', 'eod', 'both', 'refresh']:
    print(f'unknown update mode: {update_mode}')
    return None

  # get the benchmark of eod data
  today = util.time_2_string(datetime.datetime.today().date())
  start_date = util.string_plus_day(today, -window_size)
  benchmark_data = get_data_from_marketstack(symbol='AAPL', api_key=api_key, start_date=start_date, end_date=today)
  benchmark_date = util.time_2_string(benchmark_data.index.max())
  start_date = util.string_plus_day(benchmark_date, -window_size)

  # get the existed data and its latest date for each symbols
  data = {}
  for symbol in symbols:
    
    # init symbol data and its most recent date
    data[symbol] = pd.DataFrame()
    tmp_data_date = None

    # for chinese stocks
    file_name = symbol
    splited = file_name.split('.')
    if len(splited) ==2 and splited[1] in ['XSHG', 'SHG', 'SS', 'XSHE', 'SHE', 'SZ']:
      file_name = splited[0]      

    # if local data exists, load existed data, update its most current date
    symbol_file_name = f'{stock_data_path}{file_name}{file_format}'
    if os.path.exists(symbol_file_name):
      
      # delete local data if update_mode == refresh
      if update_mode == 'refresh':
        os.remove(symbol_file_name)
        
    # load local data and update its most recent date
    existed_data = load_stock_data(file_path=stock_data_path, file_name=symbol)
    if existed_data is not None and len(existed_data) > 0:
      max_idx = existed_data.index.max()
      if max_idx > util.string_2_time('2020-01-01'):
        data[symbol] = existed_data
        tmp_data_date = util.time_2_string(max_idx)
      else:
        print(f'max index of {symbol} is invalid({max_idx}), refreshing data')
        os.remove(symbol_file_name)

    # update eod data, print updating info
    if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_date):
      if is_print:
        print(f'from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
      
      # download latest data for current symbol
      new_data = get_data_from_marketstack(symbol, start_date=tmp_data_date, end_date=required_date, is_print=is_print, api_key=api_key)
    
      # append new data to the origin
      data[symbol] = pd.concat([data[symbol], new_data])
      data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

      # save data to local csv files
      if is_save:
        save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
    
    else:
      if is_print:
        print(f'from {tmp_data_date} {symbol:4}: already up-to-date, skip')

  # add real-time data when requiring data return and data will NOT be saved
  if update_mode in ['realtime', 'both']:
    print('***************** querying real-time data *****************')

    if not cn_stock:
      # get real-time data from EOD, convert it into time-series data
      real_time_data = get_real_time_data_from_marketstack(symbols=symbols, api_key=api_key, is_print=is_print, batch_size=batch_size)
      real_time_data = util.df_2_timeseries(df=real_time_data, time_col='Date')
    else:
      # get real-time data from easyquotation, convert it into time-series data
      real_time_data = get_real_time_data_from_easyquotation(symbols=symbols)

    # append it corresponding eod data according to symbols
    for symbol in symbols:
      tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
      if len(tmp_data) == 0:
        print(f'real-time data not found for {symbol}')
        continue
      else:
        tmp_idx = tmp_data.index.max()
        for col in data[symbol].columns:
          data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

  # return
  if is_return:
    return data


def update_stock_data(symbols, stock_data_path, file_format='.csv', source='eod', by='date', required_date=None, is_print=False, is_return=False, is_save=True, api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15, update_mode='eod', cn_stock=False):
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
  :param cn_stock: whether updating chinese stocks
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
    result = update_stock_data_from_eod(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, api_key=api_key, add_dividend=add_dividend, add_split=add_split, batch_size=batch_size, update_mode=update_mode, cn_stock=cn_stock)
  elif source == 'marketstack':
    result = update_stock_data_from_marketstack(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, api_key=api_key, batch_size=batch_size, update_mode=update_mode, cn_stock=cn_stock)
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
  # for chinese stocks
  splited = file_name.split('.')
  if len(splited) ==2 and splited[1] in ['SHG', 'SS', 'SHE', 'SZ']:
    file_name = splited[0]

  # construct filename
  file_name = f'{file_path}{file_name}{file_format}'

  if len(df) > 0:
    # reset index
    if reset_index:
      df = df.sort_index().reset_index()
    # save file
    if file_format == '.csv':
      df.to_csv(file_name, index=index)
    else:
      print(f'Unknown format {file_format}')
  else:
    print('Empty dataframe to save, skipped')
  

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
  # for chinese stocks
  splited = file_name.split('.')
  if len(splited) ==2 and splited[1] in ['SHG', 'SS', 'SHE', 'SZ']:
    file_name = splited[0]

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
  items = requests.get(url, headers=headers)

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


#----------------------------- Stock List Management ----------------------------#
def process_futu_exported(file_path, file_name):
  """
  process stock info that exported from futu-niuniu

  :param file_path: file path
  :param file_name: file name (.csv file)
  :returns: dataframe
  :raises: None
  """
  # load futu exported excel file 
  universe = pd.read_csv(file_path + file_name, dtype={'代码': str})
  
  # categorize columns
  id_columns = ['代码', '名称', '所属行业']
  basic_columns = ['涨跌速率%', '换手率%', '振幅%', '今开', '昨收', '最高', '最低', '最新价' ]
  rate_change_columns = ['涨跌幅', '5日涨跌幅', '10日涨跌幅', '20日涨跌幅', '60日涨跌幅', '120日涨跌幅', '250日涨跌幅', '年初至今涨跌幅']
  volume_columns = ['成交量', '成交额', '总市值', '流通市值', '总股本', '流通股本']
  fundamental_columns = ['股息率TTM', '市盈率TTM', '市盈率(静)', '市净率']

  # turn characters to numbers
  unit = {'万': 10000, '亿': 100000000}
  for col in volume_columns:
    if universe[col].dtype == str:
      universe[col] = universe[col].replace('-', '0')
      universe[col] = universe[col].apply(lambda x: (float(x[:-1]) *unit[x[-1]]) if not x.isdigit() else float(x))
    
  # turn '%' to numbers
  for col in rate_change_columns:
    universe[col] = universe[col].apply(lambda x: float(x.replace('%', ''))/100)  
  universe['股息率TTM'] = universe['股息率TTM'].apply(lambda x: float(x.replace('%', ''))/100)

  # process fundamental columns
  for col in ['市盈率TTM', '市盈率(静)', '市净率']:
    universe[col] = universe[col].replace('亏损', '-1')
    universe[col] = universe[col].replace('-', '0')
    universe[col] = universe[col].astype(float)
  universe['流通率'] = (universe['流通股本'] / universe['总股本']).round(2)
  
  # for chinese stock symbols
  for index, row in universe.iterrows():
    current_symbol = str(row['代码'])

    if len(current_symbol) == 6:
      new_symbol = current_symbol
      if current_symbol[0] == '6':
        new_symbol += '.SHG'
      elif current_symbol[0] in ['0', '3']:
        new_symbol += '.SHE'
      universe.loc[index, '代码'] = new_symbol

  # convert column names from chinese to english
  final_columns = {
    '代码': 'symbol', '名称': 'name', '所属行业': 'category', 
    '换手率%': 'turnover', '振幅%': 'close_range', '今开': 'open', '昨收': 'close_p1', '最高': 'high', '最低': 'low', '最新价': 'close', 
    '涨跌幅': 'close_rate', '5日涨跌幅': 'rate_5d', '10日涨跌幅': 'rate_10d', '20日涨跌幅': 'rate_20d', '60日涨跌幅': 'rate_60d', '120日涨跌幅':'rate_120d', '250日涨跌幅':'rate_250d', '年初至今涨跌幅': 'rate_this_year',
    '流通率': 'circulation', '成交量': 'volume', '成交额': 'volume_value', '总市值': 'market_value', '流通市值': 'market_value_circulation', '总股本': 'stock_total', '流通股本': 'stock_circulation',
    '股息率TTM': 'dividend_rate', '市盈率TTM': 'PE_TTM', '市盈率(静)': 'PE', '市净率': 'PB', 
    
  }
  id_columns_en = [final_columns[x] for x in id_columns if x in final_columns.keys()]
  basic_columns_en = [final_columns[x] for x in basic_columns if x in final_columns.keys()]
  rate_change_columns_en = [final_columns[x] for x in rate_change_columns if x in final_columns.keys()]
  volume_columns_en = [final_columns[x] for x in volume_columns if x in final_columns.keys()]
  fundamental_columns_en = [final_columns[x] for x in fundamental_columns if x in final_columns.keys()]

  universe = universe[final_columns.keys()].copy()
  universe = universe.rename(columns=final_columns)

  # classify stock type (e.g company, etf/etn, adr, etc.)
  universe['type'] = '-'

  company_filter = universe.query('category != "-"').index
  universe.loc[company_filter, 'type'] = 'company'

  adr_filter =universe.name.apply(lambda x: True if (('ADR' in x)) else False)
  universe.loc[adr_filter, 'type'] = 'adr'

  etf_filter = universe.name.apply(lambda x: True if (('指数' in x) or ('基金' in x) or ('做多' in x) or ('做空' in x)or ('ETN' in x) or ('ETF' in x)) else False)
  universe.loc[etf_filter, 'type'] = 'etf'

  return universe


def filter_futu_exported(df, condition=None, q=0.7, price_limit=[5, 1000], market='us'):
  """
  filter symbols from data that exported from futu

  :param df: dataframe of stock information
  :param condition: dict of filter conditions
  :returns: dataframe
  :raises: None
  """

  filtered_df = df
  volume_threshold = df.volume.quantile(q)
  volume_value_threshold = df.volume_value.quantile(q)
  market_value_threshold = df.market_value.quantile(q)

  # drop abnormal symbols
  to_drop = []

  # for us stocks, remove those who's symbol contains '.'
  if market == 'us':
    
    PB_threshold = -1.0
    PE_threshold = -1.0
    filtered_df = filtered_df.query(f'({price_limit[0]} <= close <= {price_limit[1]}) and (type == "company")').copy()
    filtered_df = filtered_df.query(f'(volume >= {volume_threshold} or volume_value >= {volume_value_threshold} or market_value >= {market_value_threshold}) and PE >= {PE_threshold} and PB >= {PB_threshold}').copy()
    filtered_df = filtered_df.sort_values('market_value', ascending=False)
    
    for index, row in filtered_df.iterrows():
      if ('.' in row['symbol']):
        to_drop.append(index)

  # for chinese stocks, remove those who's name contains '*' or 'ST'
  elif market == 'a':
    
    PB_threshold = 0
    PE_threshold = 0
    filtered_df = filtered_df.query(f'({price_limit[0]} <= close <= {price_limit[1]}) and (PE_TTM > 0)').copy()
    filtered_df = filtered_df.query(f'(volume >= {volume_threshold} or volume_value >= {volume_value_threshold} or market_value >= {market_value_threshold}) and PE >= {PE_threshold} and PB >= {PB_threshold}').copy()
    filtered_df = filtered_df.sort_values('PE')

    for index, row in filtered_df.iterrows():
      if ('*' in row['name'] or 'ST' in row['name']):
        to_drop.append(index)
    
  else:
    print(f'market "{market}" not defined')
  
  filtered_df = filtered_df.drop(to_drop)

  # remove duplicated stocks
  filtered_df = filtered_df.loc[~filtered_df['name'].duplicated(),].copy()
    
  return filtered_df


def import_futu_exported(df, num=100):
  """
  get list and dict from stock info dataframe, for importing into selected_sec_list.json and ta_config.json

  :param df: dataframe of stock information
  :param config: dict of config
  :returns: dict
  :raises: None
  """
  df = df.head(num).copy()
  df['name'] = df['name'] + '(' + df['category'] + ')'
  codes = df[['symbol', 'name']].set_index('symbol')

  # selected_sec_list
  code_list = codes.index.tolist()
  # code_list = [x for x in code_list if x not in config['visualization']['plot_args']['sec_name'].keys()]

  # ta_config
  code_names = codes.sort_index().to_dict()['name']
  # to_pop = [x for x in code_names.keys() if x not in code_list]
  # for symbol in to_pop:
  #   code_names.pop(symbol)
    
  return {'selected_sec_list': code_list, 'ta_config': code_names}


#----------------------------- Json File Processing---- -------------------------#
def create_config_file(config_dict, file_path, file_name, print=False, ensure_ascii=True):
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
      json.dump(config_dict, f, ensure_ascii=ensure_ascii)
    
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
  # writer.save()
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


#----------------------------- portfolio updating -------------------------------#
def update_portfolio_support_resistant(config, data, portfolio_file_name='portfolio.json', is_return=False):
  """
  record support and resistant for symbols in portfolio
  :param config: config dict
  :param data: data dict
  :param portfolio_file_name: portfolio file name, default "portfolio.json"
  :param is_return: whether to return portfolio
  :raises: None
  :returns: portfolio dict if is_return
  """

  # read portfolio information from file
  portfolio = read_config(file_path=config['config_path'], file_name=portfolio_file_name)

  # get ta result data
  ta_result = pd.DataFrame()
  for ti in data['result'].keys():
    ta_result = pd.concat([ta_result, data['result'][ti]])
  ta_result = ta_result.set_index('symbol')
  ta_result = util.remove_duplicated_index(df=ta_result)
  
  # iterate through portfolios of different platform and account
  for platform in portfolio.keys():
    
    # get portfolio for current platform-account
    tmp_platform = portfolio[platform]
    for account in tmp_platform.keys():
      tmp_account = tmp_platform[account]
      tmp_portfolio = tmp_account.get('portfolio')
            
      # if portfolio not exists, continue for next account
      if tmp_portfolio is None:
        print(f'portfolio for {account} not exists')
        continue

      else:  

        # update price related fields for chinese stocks
        if platform in ['pingan'] and account in ['snowball']:
          cn_stock = {}
          cn_stock['quantity'] = tmp_portfolio.get('quantity')
          cn_stock['latest_price'] = tmp_portfolio.get('latest_price')
          cn_stock['average_cost'] = tmp_portfolio.get('average_cost')
          cn_stock['market_value'] = tmp_portfolio.get('market_value')
          cn_stock['rate'] = tmp_portfolio.get('rate')

          # manually update portfolio(except quantity) for chinese stocks
          if cn_stock['quantity'] is not None:

            # remove symbols that no longer in position
            for d in ['latest_price', 'average_cost', 'market_value', 'rate']:
              if cn_stock[d] is None:
                cn_stock[d] = {}
              else:
                to_pop = [x for x in cn_stock[d].keys() if x not in cn_stock['quantity'].keys()]
                for symbol in to_pop:
                  cn_stock[d].pop(symbol)                

            # update latest_price, market_value and earning_rate
            for symbol in cn_stock['quantity'].keys():
              if symbol in ta_result.index:
                cn_stock['latest_price'][symbol] = ta_result.loc[symbol, 'Close']
                cn_stock['market_value'][symbol] = cn_stock['latest_price'][symbol] * cn_stock['quantity'][symbol]
                if symbol in cn_stock['average_cost'].keys():
                  cn_stock['rate'][symbol] = round((cn_stock['latest_price'][symbol] - cn_stock['average_cost'][symbol]) / cn_stock['average_cost'][symbol], 2)
                else:
                  print(f'please update average_cost for {symbol}')
                  cn_stock['average_cost'][symbol] = None
                  cn_stock['rate'][symbol] = None
            
            # assign the updated values to original portfolio, and update tmp_portfolio
            for d in cn_stock.keys():
              portfolio[platform][account]['portfolio'][d] = cn_stock[d]

            portfolio[platform][account]['market_value'] = sum(cn_stock['market_value'].values())
            portfolio[platform][account]['net_value'] = portfolio[platform][account]['market_value'] + portfolio[platform][account]['cash']
            portfolio[platform][account]['updated'] = datetime.datetime.now().strftime(format='%Y-%m-%d %H:%M:%S')
            tmp_portfolio = tmp_account.get('portfolio')

        # for all portfolios (us and cn)
        tmp_price = tmp_portfolio.get('latest_price')

        # update support and resistant for symbols in current portfolio
        if tmp_price is not None:

          # get current support 
          tmp_support = tmp_portfolio.get('support')      
          if tmp_support is not None:
            expired_symbols = [x for x in tmp_support.keys() if x not in tmp_price.keys()]
            for symbol in expired_symbols:
              tmp_support.pop(symbol)
          else:
            tmp_support = {}

          # get current resistant
          tmp_resistant = tmp_portfolio.get('resistant')
          if tmp_resistant is not None:
            expired_symbols = [x for x in tmp_resistant.keys() if x not in tmp_price.keys()]
            for symbol in expired_symbols:
              tmp_resistant.pop(symbol)
          else:
            tmp_resistant = {}

          # looking for symbol info from the ta result  
          for symbol in tmp_price.keys():
            close = tmp_price.get(symbol)
            support = tmp_support.get(symbol)
            resistant = tmp_resistant.get(symbol)
            
            # remove prefix for symbol in futu portfolios (e.g. US.AAPL)
            if platform == 'futu':
              converted_symbol = ''.join(symbol.split('.')[1:])
            else:
              converted_symbol = symbol
            
            # get support, resistant and latest price
            if converted_symbol in ta_result.index:
              close = ta_result.loc[converted_symbol, 'Close'].round(2)
              support = ta_result.loc[converted_symbol, 'support'].round(2)
              resistant = ta_result.loc[converted_symbol, 'resistant'].round(2)
              
            # record support, resistant and latest price
            tmp_support[symbol] = None if (support is None or np.isnan(support)) else support
            tmp_resistant[symbol] = None if (resistant is None or np.isnan(resistant)) else resistant
            tmp_price[symbol] = tmp_price[symbol] if (close is None or np.isnan(close)) else close

          # update portfolio
          portfolio[platform][account]['portfolio']['latest_price'] = tmp_price
          portfolio[platform][account]['portfolio']['support'] = tmp_support
          portfolio[platform][account]['portfolio']['resistant'] = tmp_resistant

  # update portfolio file
  create_config_file(config_dict=portfolio, file_path=config['config_path'], file_name=portfolio_file_name)
  
  # return portfolios
  if is_return:
    return portfolio


#----------------------------- Email sending ------------------------------------#
def send_result_by_email(config, to_addr, from_addr, smtp_server, password, subject=None, platform=['tiger'], signal_file_date=None, log_file_date=None, test=False, cn_stock=False):
  """
  send automatic_trader's trading result and technical_analyst's calculation result by email

  :param config: dictionary of config, include pathes, etc.
  :param to_addr: destination email address
  :param from_addr: email address used for sending email
  :param smtp_server: smtp server address
  :param password: pwd for smtp server
  :param subject: subject of the email
  :param platform: trading platforms include ['tiger', 'futu']
  :param signal_file_date: date of signal file which will be attached on email
  :param log_file_date: date of log file which will be attached on email
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
  if os.path.exists(config['config_path']+'portfolio.json'):
    portfolio_record = read_config(file_path=config['config_path'], file_name='portfolio.json')
    
    # for us_stock
    if not cn_stock:
      if 'tiger' in platform:
        pr = portfolio_record.get('tiger')
        assets['glob'] = pr.get('global_account')
        # assets['simu'] = pr.get('simulation_account') 

      if 'futu' in platform:
        pr = portfolio_record.get('futu')
        assets['REAL'] = pr.get('REAL')
        # assets['SIMU'] = pr.get('SIMULATE')
    
    # for cn_stock
    else:
      pr = portfolio_record.get('pingan')
      assets['snowball'] = pr.get('snowball')


  # construct asset summary
  asset_info = '<h3>Assets</h3><ul>'
  for portfolio in assets.keys():
    if assets[portfolio] is not None:
      net_value = assets[portfolio].get('net_value')
      updated = assets[portfolio].get('updated')
      position = assets[portfolio].get('portfolio')
      if position is not None:
        position = pd.DataFrame(position).fillna(np.nan)
        if len(position) > 0:
          
          # support and resistant
          lower_than_support = position.query('latest_price <= support').index.tolist()
          higher_than_resistant = position.query('latest_price >= resistant').index.tolist()
          
          # symbol name (for cn stocks only)
          idx_list = position.index.tolist()
          idx_list = [x.replace('US.', '') for x in idx_list]
          position['name'] = idx_list
          position['name'] = position['name'].apply(lambda x: config['visualization']['plot_args']['sec_name'].get(x))

          # convert to html format
          position = position.drop('latest_time', axis=1)[['name', 'quantity', 'rate', 'market_value', 'average_cost', 'latest_price', 'support', 'resistant']].to_html()
          
          for l in lower_than_support:
            position = position.replace(l, f'<font color="red">{l}</front>')
          for h in higher_than_resistant:
            position = position.replace(h, f'<font color="green">{h}</front>')

        else:
          position = None

      # add position summary if provided
      position = position if position is not None else ''

    else:
      net_value = '--'
      updated = '--'  
      position = ''

    # mark update-time if not not match with current time
    if updated[:16] == current_time[:16]:
      updated = ''
    else:
      updated = f'({updated})'

    # asset summary for current portfolio
    asset_info += f'<li><b><p>{portfolio}: ${net_value}</b></p>{position}{updated}</li>'
  asset_info += '</ul>'

  # get signal and summary
  signal_info = '<h3>Signals</h3><ul>'
  summary_info = '<h3>Summary</h3><ul>'
  signal_color = {'b':'green', 's':'red', 'n':'grey'}
  
  if signal_file_date is not None:
    
    prefix = 'a_' if cn_stock else ''
    signal_file = f'{config["result_path"]}{prefix}{signal_file_date}.xlsx'
    
    if os.path.exists(signal_file):

      signal_file_content =pd.read_excel(signal_file, sheet_name=None)
      for s in signal_file_content.keys():
        if s == 'signal':
          signals = signal_file_content.get('signal') # pd.read_excel(signal_file, sheet_name='signal')
          if signals is not None:
            # if len(signals) > 0:
            for s in ['b', 's', 'n']:
              font_color = signal_color[s]
              tmp_signals = signals.query(f'交易信号 == "{s}"')['代码'].tolist()
              signal_info += f'<li>[ <b>{s}</b> ]: <font color="{font_color}">{", ".join(tmp_signals)}</font></li>'
        
        elif s == 'potential':
          potentials = signal_file_content.get('potential') # pd.read_excel(signal_file, sheet_name='potential')
          if potentials is not None:
            if len(potentials) > 0:
              # symbol name (for cn stocks only)
              potentials['名称'] = potentials['代码']
              potentials['名称'] = potentials['名称'].apply(lambda x: config['visualization']['plot_args']['sec_name'][x])
              potentials = potentials.set_index('代码')[config['postprocess']['send_columns']].sort_values('总分', ascending=False).to_html()
              signal_info += f'</b></p>{potentials}'
        else:
          tmp_sheet = signal_file_content.get(s)
          if tmp_sheet is not None:
            num_up = len(tmp_sheet.query('涨跌 > 0'))
            num_total = len(tmp_sheet)
            font_color = 'green' if num_up > num_total/2 else 'red'
            summary_info += f'<li>[ <b>{s}</b> ]: <font color="{font_color}">{num_up} / {num_total}</font></li>'

    else:
      signal_info += f'<li><p>[Not Found]: {prefix}{signal_file_date}.xlsx</p></li>'
  
  else:
    signal_info += '<li><p>[Not Required]</p></li>'
    summary_info += '<li><p>[Not Required]</p></li>'
  signal_info += '</ul>'
  summary_info += '</ul>'

  # attachment 1: log file
  log_info = '<h3>Log</h3><ul>'
  if log_file_date is not None:

    log_file = f'{config["log_path"]}automatic_trade_log_{log_file_date}.txt'
    
    if os.path.exists(log_file):
      log_part = MIMEApplication(open(log_file, 'rb').read())
      log_part.add_header('Content-Disposition', 'attachment', filename=log_file)
      log_info += f'<li><p>[Attached]</p></li>'
    
    else:
      log_info += f'<li><p>[Not Found]: automatic_trade_log_{log_file_date}.txt</p></li>'
      log_part = None
  
  else:
    log_info += '<li><p>[Not Required]</p></li>'
    log_part = None
  log_info += '</ul>'
  
  # attachment 2: images in pdf files(signal/portfolio/index/potential)
  image_info = f'<h3> Images</h3><ul>'
  pdfs = []
  if signal_file_date is not None:

    # initialize header, attach pdfs
    image_info += f'<li>[Requested]: {signal_file_date}</li>'
    pdf_names = ['portfolio', 'signal', 'potential', 'index'] if not cn_stock else ['a_portfolio', 'a_signal', 'a_potential']
    for p in pdf_names:

      # consstruct pdf file path
      tmp_pdf = f'{config["result_path"]}{p}.pdf'

      # if pdf file exists, check its create date and attach its content
      if os.path.exists(tmp_pdf):
        tmp_pdf_create_date = util.timestamp_2_time(timestamp=os.path.getmtime(tmp_pdf), unit='s').date().strftime(format='%Y-%m-%d')
        
        # if it is not up-to-date, notify in image_info
        if tmp_pdf_create_date < signal_file_date:
          image_info += f'<li>[{p}.pdf]: {tmp_pdf_create_date}</li>'
        
        # other wise attach it
        else:
          with open(tmp_pdf, 'rb') as fp:
            tmp_pdf_content = MIMEBase('application', "octet-stream")
            tmp_pdf_content.set_payload(fp.read())
            tmp_pdf_content.add_header('Content-Disposition', 'attachment', filename=f'{p}_{tmp_pdf_create_date}.pdf')
            encoders.encode_base64(tmp_pdf_content)
          pdfs.append(tmp_pdf_content)

      else:        
        image_info += f'<li><p>[{p}.pdf]: Not Found</p></li>'
  else:
    image_info += '<li><p>[Not Required]</p></li>'
  image_info += '</ul>'

  # construct message part by concating info parts
  full_info = f'<html><body>{asset_info}{summary_info}{signal_info}{image_info}{log_info}</body></html>'
  msg_part = MIMEText(full_info,'html','utf-8')
  m.attach(msg_part)

  # text files (log)
  if log_part is not None:
    m.attach(log_part)

  # pdf files (images)
  for i in pdfs:
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