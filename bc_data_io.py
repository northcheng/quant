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

# mute warnings
import warnings
warnings.filterwarnings('ignore')

# data source
import yfinance as yf
import pandas_datareader.data as web 
import easyquotation as eq
import akshare as ak
# from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

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
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'}

# standards
# STANDARD_US_SYMBOL = 'AAPL'
# STANDARD_CN_SYMBOL = '000001' # 00700
# STANDARD_INTERVAL = 'd' # /w/m

# EOD is mainly used for US stock eod and realtime(15min-delayed) price:  AAPL
# EOD is able to access CN stock eod price, but the price is un-adjusted: 000001.SHE
# ak is recommanded for CN/HK stock eod price, adjusted(qfq/hfq) or not:  000001/00700
# easyquotation is used for getting CN stock realtime price:              000001/00700
# Yahoo, Yfinance, iex, marketstack datasource have been deprecated

# US_EOD_CANDIDATES: eod, ak
# US_REALTIME_CANDIDATES: eod
# CN_EOD_CANDIDATES: eod(unjusted), ak(recommanded)
# CN_REALTIME_CANDIDATES: easyquotation
# HK_EOD_CANDIDATES: eod, ak, easyquotation
# HK_REALTIME_CANDIDATES: easyquotation

# 跑数据用eod/easyquotation, 更新A股数据用ak
default_data_sources = {
  'us_eod': 'ak', 
  'us_realtime': 'eod', 
  'cn_eod': 'ak', 
  'cn_realtime': 'easyquotation', 
  'hk_eod': 'ak', 
  'hk_realtime': 'easyquotation'
}


#----------------------------- Stock Data -------------------------------------#
def add_postfix_for_cn_symbol(symbol):
  """
  Add postfix for chinese stock symbols

  :param symbol: cn symbol in STANDARD_CN_SYMBOL format
  :returns: symbol with postfix added
  :raises: none
  """
  # cn symbol should be all digit
  if symbol.isdigit():
    
    # mainland market
    if len(symbol) == 6:
      postfix = '.SHE' if int(symbol) < 600000 else '.SHG'

    # hongkong market
    elif len(symbol) < 6:
      postfix = '.HK'

    else:
      print(f'{symbol}: cn symbol should be 5(hk) or 6(mainland) digits')
      postfix = ''

    symbol = symbol + postfix

  else:
    print(f'{symbol} is not all digit')

  return symbol

# preprocess symbol for downloading
def preprocess_symbol(symbols, style):
  """
  Add preprocess list of symbols to meet the style of different datasource

  :param symbols: list of symbols in STANDARD_CN_SYMBOL/STANDARD_CN_SYMBOL format
  :param style: name of the datasource
  :returns: list of processed symbols
  :raises: none
  """

  result = {}

  # classify symbols
  us_symbols = [x.upper() for x in symbols if x.isalpha()]
  cn_symbols = [x for x in symbols if x.isdigit()]
  other_symbols = [x for x in symbols if x not in us_symbols and x not in cn_symbols]
  if len(other_symbols) > 0:
    print(f'other symbols found: {other_symbols}')

  # e.g. 'AAPL', '000001.SHE'/'600001.SHG/'0700.HK'
  if style == 'eod':
    # us symbols remains tha same
    for s in us_symbols:
      result[s] = s
    
    # cn mainland symbols add postfix 
    for s in cn_symbols:
      result[s] = add_postfix_for_cn_symbol(s[1:] if len(s) == 5 else s)
    
  # e.g. '105.AAPL', '000001'/'600001'
  elif style == 'ak':
    # us symbols add prefix
    us_symbol_list = ak.stock_us_spot_em()
    us_symbol_list['symbol'] = us_symbol_list['代码'].apply(lambda x: x.split('.')[1])
    us_symbol_list = us_symbol_list.set_index('symbol')    
    for s in us_symbols:
      if s in us_symbol_list.index:
        result[s] = us_symbol_list.loc[s, '代码']

    # cn_symbols remains the same
    for s in cn_symbols:
      result[s] = s

  # e.g. '000001'/'600001'/'00700'
  elif style == 'easyquotation':
    # remove us symbols
    
    # cn symbols remains the same
    for s in cn_symbols:
      result[s] = s

  else:
    print(f'Unknown symbol style {style}')
  
  return result

# # get symbols from Nasdaq
# def get_symbols(remove_invalid=True, save_path=None, save_name='symbol_list.csv', local_file=None):
#   """
#   Get Nasdaq stock list

#   :param remove_invalid: whether to remove invalid stock symbols from external stock list (.csv)
#   :param save_path: where to save the symbol list, generally it will be saved at ~/quant/stock/data/
#   :param save_name: the name of the saved symbol list file, defaultly it will be symbol_list.csv
#   :returns: dataframe of stock symbols
#   :raises: exception when error reading not-fetched symbols list
#   """
#   # get the symbols from pandas_datareader
#   if local_file is not None and os.path.exists(local_file):
#     symbols = pd.read_csv(local_file).set_index('Symbol')

#   else:
#     try:
#       symbols = get_nasdaq_symbols()
#       symbols = symbols.loc[symbols['Test Issue'] == False,]
    
#     # get symbols from Nasdaq website directly when the pandas datareader is not available
#     except Exception as e:
#       symbols = pd.read_table('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt', sep='|', index_col='Symbol').drop(np.NaN)
#       symbols = symbols.loc[symbols['Test Issue'] == 'N',]
#       print(e)
    
#     # get list of all symbols and remove invalid symbols
#     sec_list = symbols.index.tolist()
#     if remove_invalid:
#       sec_list = [x for x in sec_list if '$' not in x]
#       sec_list = [x for x in sec_list if '.' not in x]

#     symbols = symbols.loc[sec_list, ].copy()

#     if save_path is not None:
#       symbols.reset_index().to_csv(f'{save_path}{save_name}', index=False)
  
#   return symbols

# get ohlcv data from eod(US/CN/HK)
def get_data_from_eod(symbol, start_date=None, end_date=None, interval='d', is_print=False, api_key=default_eod_key, add_dividend=True, add_split=True):
  """
  Download stock data from EOD

  :param symbol: target symbol, e.g. 'AAPL', '000001.SHE'
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

    # postprocess: convert to timeseries dataframe, rename columns
    data = post_process_download_data(df=data, source='eod')

  except Exception as e:
    print(symbol, e)

  return data

# get ohlcv data from ak(US/CN/HK)
def get_data_from_ak(symbol, start_date=None, end_date=None, interval='daily', is_print=False, adjust='qfq'):
  """
  Download stock data from akshare
  https://github.com/akfamily/akshare

  :param symbol: target symbol, e.g. '105.AAPL', '000001', '00700'
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param interval: daily/weekly/monthly
  :param adjust: the way to adjust price, ''/''qfq'/'hfq'
  :param is_print: whether to print download information
  :returns: dataframe or None
  :raises: exception when downloading failed
  """
  # initialize
  market = None
  result = None

  # start and end date
  start_date = '2000-01-01' if start_date is None else start_date
  end_date = util.time_2_string(datetime.datetime.today().date()) if end_date is None else end_date
  start_date = start_date.replace('-', '')
  end_date = end_date.replace('-', '')
  
  # decide market from the first character of the symbol
  symbol_end = symbol[-1]
  if symbol_end.isalpha():
    market = 'us'
  elif symbol_end.isdigit():
    if len(symbol) == 6:
      market = 'cn'
    elif len(symbol) == 5:
      market = 'hk'
    else:
      print(f'symbol for unknown market: {symbol}')

  # get data
  if market == 'us':
    result = ak.stock_us_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)

  elif market == 'cn':
    result = ak.stock_zh_a_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)
  
  elif market == 'hk':
    result = ak.stock_hk_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)

  else:
    pass

  # postprocess: rename columns, add extra columns
  result = post_process_download_data(result, 'ak')

  if is_print:
    print(f'{symbol:5}: {result.index.min()} - {result.index.max()}, 下载记录 {len(result)} from ax')
  
  return result

# get ohlcv data from easyquotation(HK: 1500rows)
def get_data_from_easyquotation(symbol, is_print=False):
  """
  Download HK stock data from easyquotation
  https://github.com/shidenggui/easyquotation

  :param symbol: target symbol, e.g. '00700'
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param is_print: whether to print download information
  :returns: dataframe or None
  :raises: exception when downloading failed
  """

  # hk daykline
  quotation = eq.use("daykline")
  data = quotation.real([symbol])  

  df = pd.DataFrame(data[symbol], columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Note', 'Turnover', 'Money'])
  df = post_process_download_data(df=df, source='easyquotation_daykline')

  if is_print:
    print(f'{symbol:5}: {df.index.min()} - {df.index.max()}, 下载记录 {len(df)} from easyquotation_daykline')

  return df

# postprocess downloaded data
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

    # data from eod
    if source == 'eod':
      df = df.rename(columns={'date':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume', 'dividend':'Dividend', 'split': 'Split'})
      df = util.df_2_timeseries(df=df, time_col='Date')   
  
    # data from ak   
    elif source == 'ak':
      df = df.rename(columns={'日期':'Date', '股票代码':'symbol', '开盘':'Open', '收盘':'Close', '最高':'High', '最低':'Low', '成交量':'Volume', '成交额':'Money', '振幅':'Inday Rate', '涨跌幅':'Change Rate', '涨跌额': 'Change Price', '换手率': 'Turnover'})
      df['Adj Close'] = df['Close'].copy()
      df['Dividend'] = 0.0
      df['Split'] = 1.0
          
      df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']].copy()
      df = util.df_2_timeseries(df, time_col='Date')
    
    # data from easyquotation
    elif source == 'easyquotation_daykline':
      df['Adj Close'] = df['Close'].copy()
      df['Dividend'] = 0.0
      df['Split'] = 1.0
          
      df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']].copy()
      df = util.df_2_timeseries(df, time_col='Date')
    
    # remove duplicated index and sort data by index
    df = util.remove_duplicated_index(df, keep='last')
    df.sort_index(ascending=True, inplace=True)

  return df

# get data from specific datasource
def get_data(symbol, start_date=None, end_date=None, interval='d', is_print=False, source='eod', api_key=default_eod_key, add_dividend=True, add_split=True, adjust='qfq'):
  """
  Download stock data from web sources

  :param symbol: target symbol
  :param start_date: start date of the data
  :param end_date: end date of the data
  :param source: datasrouce: 'eod'/'ak'/'easyquotation'
  :param interval: period, e.g. d/w/m for eod, daily/weekly/monthly for ak
  :param is_print: whether to print download information
  :param api_key: [eod] api token to access eod or iex data
  :param add_dividend: [eod] whether to add dividend data
  :param add_split: [eod] whether to add split data
  :param adjust: [ak] adjustment method for price
  :returns: dataframe 
  :raises: exception when downloading failed
  """
  data = None

  try:
    
    # eod(US/CN/HK)
    if source == 'eod':
      data = get_data_from_eod(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date, is_print=is_print, api_key=api_key, add_dividend=add_dividend, add_split=add_split)
    
    # ak(US/CN/HK)
    elif source == 'ak':
      interval = {'d':'daily', 'w':'weekly', 'm':'monthly'}[interval]
      data = get_data_from_ak(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval, is_print=is_print, adjust=adjust)

    # easyquotation(HK)
    elif source == 'easyquotation':
      data = get_data_from_easyquotation(symbol=symbol, is_print=is_print)

    # otherwise
    else:
      print(f'data source {source} not found')

  except Exception as e:
    print(symbol, e)

  return data

# get 15min-delayed US market realtime data from eod
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
    tmp_list_len = len(tmp_list)
    first_symbol = tmp_list[0]
    if tmp_list_len == 1:
      other_symbols = first_symbol
    else:
      other_symbols = ','.join(tmp_list[1:])
  
    # get real-time data for current batch
    url = f'https://eodhistoricaldata.com/api/real-time/{first_symbol}?api_token={api_key}&fmt=json&s={other_symbols}'
    response = requests.get(url, headers=headers)
    try:
      real_time = [] if response.status_code!=200 else response.json()
    except Exception as e:
      print(e, response.status_code, response)
      real_time = []

    real_time = [real_time] if isinstance(real_time, dict) else real_time
    
    if is_print:
      print(f'updating real-time for symbol: {batch_start+1:3} -{batch_end:3} - {response}')      
  
    # post process downloaded real-time data
    real_time_data = pd.DataFrame(real_time)
    if not real_time_data.empty:

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

    else:

      print('get empty response for realtime data')
      
    # concate batches of real-time data 
    result = pd.concat([result, real_time_data]) 
    batch_start = batch_end

  # postprocess
  if not result.empty:
    result = result.reset_index().drop('index', axis=1)
    result.rename(columns={'code':'symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume', 'dividend':'Dividend', 'split': 'Split'}, inplace=True)
    for col in ['timestamp', 'gmtoffset']:
      if col in result.columns:
        result.drop([col], axis=1, inplace=True)
  else:
    print('real time data from eod is empty')

  return result

# get realtime CN market realtime data from easyquotation
def get_real_time_data_from_easyquotation(symbols, source='sina'):
  """
  Download real-time stock data (chinese stock only) from easyquotation
  https://github.com/shidenggui/easyquotation

  :param symbols: list of symbols, e.g.[000001, 600001] / [00700, 00001]
  :param source: data source, sina/tecent/qq
  :returns: dataframe or None
  :raises: exception when downloading failed
  """

  df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Date'])
  columns_to_keep = {}

  # create quoataion entity
  quotation = eq.use(source)
  print(f'updating real-time for symbols from easyquotation({source})')

  # get stock data from sina
  codes = [x.split('.')[0] for x in symbols]

  if source == 'sina':
    qt = quotation.stocks(codes) 
    columns_to_keep = {'index':'symbol', 'open':'Open', 'high':'High', 'low':'Low', 'now':'Close', 'turnover':'Volume', 'date':'Date'}

  elif source == 'hkquote':
    qt = quotation.real(codes) 
    columns_to_keep = {'index':'symbol', 'openPrice':'Open', 'high':'High', 'low':'Low', 'price':'Close', 'amount':'Volume', 'time':'Date'}

  else:
    print(f'unknown source {source}')

  # turn into dataframe
  df = pd.DataFrame(qt).T if len(qt) > 0 else df
  if source == 'hkquote':
    df['time'] = df['time'].apply(lambda x: x[:10])

  idxs = df.index.tolist()
  idx_symbols = [x for x in symbols if x.split('.')[0] in idxs]

  # post process
  df = df.reset_index()
  df['index'] = idx_symbols
  
  df = df.rename(columns=columns_to_keep)[columns_to_keep.values()].copy()
  df = util.df_2_timeseries(df=df, time_col='Date')
  df['Adj Close'] = df['Close']
  df['Dividend'] = 0.0
  df['Split'] = 1.0
  
  df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

  return df

# get realtime data from specific datasource
def get_real_time_data(symbols, source='eod', sub_source='sina', api_key=default_eod_key, is_print=False, batch_size=15):

  data = None

  try:
    
    symbols_for_source = list(preprocess_symbol(symbols=symbols, style=source).values())

    # eod(US)
    if source == 'eod':
      data = get_real_time_data_from_eod(symbols=symbols_for_source, is_print=is_print, api_key=api_key, batch_size=batch_size)
    
    # easyquotation (CN/HK)
    elif source == 'easyquotation':
      data = get_real_time_data_from_easyquotation(symbols=symbols_for_source, source=sub_source)

    # otherwise
    else:
      print(f'data source {source} not found')

  except Exception as e:
    print(symbols, e)

  return data

# get realtime data for a list of stocks from eod
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
    latest_data = pd.DataFrame(columns=['latest_time', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'symbol', 'latest_price', 'Date'])
    print('real time data from eod is empty')

  return latest_data

# get realtime data for a list of stocks from specified datasource
def get_stock_briefs(symbols, source='eod', api_key=default_eod_key, batch_size=15):
  """
  Get latest stock data for symbols

  :param symbols: symbol list
  :param source: data source
  :param api_key: api token to access eod data
  :param batch_size: batch size of symbols of getting real-time data
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """
  # initialize
  briefs = None

  # currently only eod is available for us stocks
  if source == 'eod':
    briefs = get_stock_briefs_from_eod(symbols=symbols, api_key=api_key, batch_size=batch_size)

  else:
    print(f'Unknown source {source}')

  return briefs

# update stock data (eod and/or realtime)
def update_stock_data_new(symbols, stock_data_path, file_format='.csv', update_mode='eod', required_date=None, window_size=3, is_print=False, is_return=False, is_save=True, sources=default_data_sources, api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15, adjust='qfq'):
  """
  update local stock data from eod

  :param symbols: list of target symbols
  :param stock_data_path: where the local stock data files(.csv) stored
  :param file_format: default is .csv
  :param update_mode: how to update data - realtime/eod/both(eod+realtime)/refresh(delete and redownload eod)
  :param required_date: if the local data have already meet the required date, it won't be updated
  :param is_print: whether to print info when downloading
  :param is_return: whether to return the updated data
  :param is_save: whether to save the updated data to local files
  :param sources: data sources for different markets
  :param api_key: api token to access eod data
  :param add_dividend: whether to add dividend data
  :param add_split: whether to add split data
  :param batch_size: batch size of symbols of getting real-time data  
  :returns: dataframe of latest stock data, per row each symbol
  :raises: none
  """

  # verify update_mode
  if update_mode not in ['realtime', 'eod', 'both', 'refresh']:
    print(f'unknown update mode: {update_mode}')
    return None

  # classify symbols
  us_symbols = [x for x in symbols if x.isalpha()]
  cn_symbols = [x for x in symbols if x.isdigit() and len(x) == 6]
  hk_symbols = [x for x in symbols if x.isdigit() and len(x) == 5]
  other_symbols = [x for x in symbols if (x not in us_symbols and x not in cn_symbols and x not in hk_symbols)]
  symbol_class = {'us': us_symbols, 'cn': cn_symbols, 'hk': hk_symbols, 'other': other_symbols}
  symbol_count = {'us': len(us_symbols), 'cn': len(cn_symbols), 'hk': len(hk_symbols), 'other': len(other_symbols)}
  print(f'US({symbol_count["us"]}), CN({symbol_count["cn"]}), HK({symbol_count["hk"]}), Other({symbol_count["other"]})')
  if symbol_count['other'] > 0:
    print(f'Unexpected symbols found: {other_symbols}')

  # default dates
  today = datetime.datetime.today().date()
  today = util.time_2_string(today)
  start_date = util.string_plus_day(today, -7)

  # set benchmarks for different markets
  benchmark_symbols = {'us': 'AAPL', 'cn': '000001', 'hk': '00700'}
  benchmark_dates = {}
  for mkt in benchmark_symbols.keys():
    benchmark_source = sources[f'{mkt}_eod']
    mkt_symbol_count = symbol_count[mkt]
    mkt_benchmark_symbol = benchmark_symbols[mkt]
    mkt_benchmark_symbol = '105.AAPL' if (mkt == 'us' and benchmark_source == 'ak') else mkt_benchmark_symbol
    
    if symbol_count[mkt] > 0:
      
      retry_count = 0
      while retry_count < 5:
        try:

          retry_count += 1
          print(f'[data]: querying benchmark date for [{mkt.upper()}] from {benchmark_source} (try #{retry_count})')        

          tmp_data = get_data(mkt_benchmark_symbol, start_date=start_date, end_date=today, interval='d', is_print=False, source=benchmark_source, api_key=api_key, add_dividend=False, add_split=False, adjust='qfq')
          benchmark_dates[mkt] = util.time_2_string(tmp_data.index.max())        
          
          break

        except Exception as e:
          
          print(f'[erro]: querying failed for [{mkt} market], try({retry_count}/5), {type(e)} - {e}')        
          time.sleep(5)
          continue

      tmp_benchmark_date = benchmark_dates.get(mkt)
      if tmp_benchmark_date is None:
        benchmark_dates[mkt] = today
        print(f'[-{mkt.upper()}-]: symbols({mkt_symbol_count}), benchmark({mkt_benchmark_symbol}), date(failed to get benchmark data, use today - {today})') # 
      else:
        print(f'[-{mkt.upper()}-]: symbols({mkt_symbol_count}), benchmark({mkt_benchmark_symbol}), date({benchmark_dates[mkt]})')
    else:
      pass
  
  print()

  # for different markets
  data = {}
  for mkt in benchmark_symbols.keys():

    if len(symbol_class[mkt]) == 0:
      continue
    
    tmp_source = sources[f'{mkt}_eod']
    tmp_source_symbols = preprocess_symbol(symbols=symbol_class[mkt], style=tmp_source)
    up_to_date_symbols = []

    # retry 5 times
    retry_count = 0
    while retry_count < 5:
      try:

        retry_count += 1
        print(f'[data]: updating data for [{mkt.upper()}] from {tmp_source} (try #{retry_count})')

        # get the existed data and its latest date for each symbols
        for symbol in symbol_class[mkt]:
          
          # init symbol data and its most recent date
          data[symbol] = pd.DataFrame()
          tmp_data_date = None

          # if local data exists, load existed data, update its most current date
          symbol_file_name = f'{stock_data_path}{symbol}{file_format}'
          if os.path.exists(symbol_file_name):
            
            # delete local data if update_mode == refresh
            if update_mode == 'refresh':
              os.remove(symbol_file_name)
            
            # otherwise load local data and update its most recent date
            else:
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
          if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_dates[mkt] or (tmp_data_date <= benchmark_dates[mkt] and tmp_source == 'ak')):
            
            if is_print:
              print(f'from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
            
            # download latest data for current symbol
            start_date = util.string_plus_day(tmp_data_date, -3) if tmp_data_date is not None else tmp_data_date
            tmp_symbol = tmp_source_symbols.get(symbol)
            
            if tmp_symbol is not None:

              # get new data
              new_data = get_data(symbol=tmp_symbol, start_date=start_date, end_date=required_date, interval='d', is_print=is_print, source=tmp_source, api_key=api_key, add_dividend=add_dividend, add_split=add_split, adjust=adjust)
          
              # append new data to the origin
              data[symbol] = pd.concat([data[symbol], new_data])
              data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

              # save data to local csv files
              if is_save:
                save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
            
            else:
              new_data = None
              print(f'{symbol:5}: symbol not found from source {tmp_source}')

            
          else:
            up_to_date_symbols.append(symbol)
            
        num_symbol_up_to_date = len(up_to_date_symbols)
        if num_symbol_up_to_date > 0:
          if is_print:
            print(f'from {tmp_data_date} ***** - [skip]: <already up-to-date {num_symbol_up_to_date}/{len(symbol_class[mkt])} >')

        # add real-time data when requiring data return and data will NOT be saved
        if update_mode in ['realtime', 'both']:
          print('***************** querying real-time data *****************')
          tmp_source = sources[f'{mkt}_realtime']
          tmp_sub_source = 'hkquote' if mkt == 'hk' else 'sina'

          if sources[f'{mkt}_eod'] in ['ak'] and update_mode in 'both':
            print(f'--------real-time data comes with eod data from ak--------')

          else:
            real_time_data = get_real_time_data(symbols=symbol_class[mkt], source=tmp_source, sub_source=tmp_sub_source, api_key=api_key, is_print=is_print, batch_size=batch_size)
            if tmp_source == 'eod':
              real_time_data = util.df_2_timeseries(df=real_time_data, time_col='Date')

            # append it corresponding eod data according to symbols
            for symbol in symbol_class[mkt]:
              tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
              if len(tmp_data) == 0:
                print(f'real-time data not found for {symbol}')
                continue
              else:
                tmp_idx = tmp_data.index.max()
                for col in data[symbol].columns:
                  data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
                data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

        break
      except Exception as e:
        print(f'[erro]: updating data failed for [{mkt} market], try({retry_count}/5), {type(e)} - {e}')
        continue
    
    print()
  
  # return
  if is_return:
    return data


def update_stock_data_from_eod(symbols, stock_data_path, file_format='.csv', update_mode='eod', required_date=None, window_size=3, is_print=False, is_return=False, is_save=True, cn_stock=False, api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15):
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
  benchmark_symbol = 'AAPL' if not cn_stock else '000001.SHE'
  benchmark_data = get_data_from_eod(symbol=benchmark_symbol, start_date=start_date, end_date=today, interval='d', is_print=False, api_key=api_key, add_dividend=False, add_split=False)
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


def update_stock_data_from_ak(symbols, stock_data_path, file_format='.csv', update_mode='eod', required_date=None, window_size=3, is_print=False, is_return=False, is_save=True, cn_stock=False):

  # verify update_mode
  if update_mode not in ['eod', 'refresh', 'both', 'realtime']:
    print(f'unknown update mode: {update_mode}')
    return None

  # get the benchmark of eod data
  today = util.time_2_string(datetime.datetime.today().date())
  start_date = util.string_plus_day(today, -7)
  benchmark_symbol = '105.AAPL' if not cn_stock else '000001'
  benchmark_data = get_data_from_ak(symbol=benchmark_symbol, start_date=start_date, end_date=today, interval='daily')
  benchmark_date = util.time_2_string(benchmark_data.index.max())
  start_date = util.string_plus_day(benchmark_date, -window_size)

  # get the existed data and its latest date for each symbols
  data = {}
  up_to_date_symbols = []
  
  # for us stocks
  if not cn_stock:
    symbol_list = ak.stock_us_spot_em()
    symbol_list['symbol'] = symbol_list['代码'].apply(lambda x: x.split('.')[1])
    symbol_list = symbol_list.set_index('symbol')
    symbols = symbol_list.loc[symbols, '代码'].to_list()
    
  # # for cn stocks
  # else:
  #   symbols = [x.split('.')[0] for x in symbols]

  for symbol in symbols:

    # init symbol data and its most recent date
    data[symbol] = pd.DataFrame()
    tmp_data_date = None

    # filename, if local data exists, load existed data, update its most current date
    file_name = symbol.split('.')[0]   
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
      print(file_name)
      new_data = get_data_from_ak(file_name, start_date=tmp_data_date, end_date=required_date, interval='daily', is_print=is_print)

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


def update_stock_data(symbols, stock_data_path, file_format='.csv', update_mode='eod', cn_stock=False, required_date=None, is_print=False, is_return=False, is_save=True, source='eod', api_key=default_eod_key, add_dividend=True, add_split=True, batch_size=15):
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

  if source == 'eod':
    result = update_stock_data_from_eod(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, api_key=api_key, add_dividend=add_dividend, add_split=add_split, batch_size=batch_size, update_mode=update_mode, cn_stock=cn_stock)
  elif source == 'ak':
    result = update_stock_data_from_ak(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, update_mode=update_mode, cn_stock=cn_stock)
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
    print(file_name, e)

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
    
  # use adjusted_close as close, and calculate related open/high/low
  for col in ['Open', 'High', 'Low']:
    df[f'{col}_to_Close'] = (df[col] / df['Close'])
  df['Close'] = df['Adj Close']
  for col in ['Open', 'High', 'Low']:
    df[col] = df[f'{col}_to_Close'] * df['Close']

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
  
  # use adjusted_close as close, and calculate related open/high/low
  for col in ['Open', 'High', 'Low']:
    df[f'{col}_to_Close'] = (df[col] / df['Close'])
  df['Close'] = df['Adj Close']
  for col in ['Open', 'High', 'Low']:
    df[col] = df[f'{col}_to_Close'] * df['Close']
  

  # create an empty dict for storing result
  month_data = {'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [], 'Dividend':[], 'Split':[]}
  
  # go through each month
  for year in range(start_year, end_year+1):
    for month in range(1, 12+1):
      
      # get current month
      tmp_period = f'{year}-{month:02}'
      if (tmp_period >= start_month ) and (tmp_period <= end_month):
        tmp_data = df.loc[tmp_period].copy()
        
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
    filtered_df = filtered_df.query(f'(volume >= {volume_threshold} and volume_value >= {volume_value_threshold} and market_value >= {market_value_threshold}) and PE >= {PE_threshold} and PB >= {PB_threshold} and rate_5d > 0 and rate_10d > 0').copy()
    filtered_df = filtered_df.sort_values('market_value_circulation', ascending=False)
    
    for index, row in filtered_df.iterrows():
      if ('.' in row['symbol']):
        to_drop.append(index)

  # for chinese stocks
  elif market == 'a':
    
    PB_threshold = 0
    PE_threshold = 0
    filtered_df = filtered_df.query(f'({price_limit[0]} <= close <= {price_limit[1]}) and (PE_TTM > 0)').copy()
    filtered_df = filtered_df.query(f'(volume >= {volume_threshold} or volume_value >= {volume_value_threshold} or market_value >= {market_value_threshold}) and PE >= {PE_threshold} and PB >= {PB_threshold} and rate_5d > 0 and rate_10d > 0').copy()
    filtered_df = filtered_df.sort_values('market_value_circulation', ascending=False)

    for index, row in filtered_df.iterrows():
      tmp_symbol_start = row['symbol'].split('.')[0][:2]
      tmp_name = row['name']
      
      # remove those who's symbol not starts with ['60', '00'], or who's name contains ['*', 'ST']
      if tmp_symbol_start not in ['60', '00'] or ('*' in tmp_name or 'ST' in tmp_name):
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


def add_config(config_key, config_value, file_path, file_name, is_print=False):
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
      if is_print:
        print('Config added successfully')

  except Exception as e:
    print(e)


def remove_config(config_key, file_path, file_name, is_print=False):
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
      if is_print:
        print('Config removed successfully')

  except Exception as e:
    print(e)


def modify_config(config_key, config_value, file_path, file_name, is_print=False):
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
      if is_print:
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
                
                if symbol in cn_stock['average_cost'].keys() and cn_stock['average_cost'][symbol] is not None:
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
def send_result_by_email(config, to_addr, from_addr, smtp_server, password, subject=None, platform=['tiger'], signal_file_date=None, log_file_date=None, test=False, pool=None):
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

  # default pool
  pool = '' if pool is None else pool

  # get current time
  current_time = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")

  # construct email 
  m = MIMEMultipart()
  if subject is not None:
    m['Subject'] = subject
  else:
    m['Subject'] = f'[auto_trade] {current_time}'
  
  m["From"] = f'Autotrade <{from_addr}>' 
  
  # get portfolio record
  assets = {}
  if os.path.exists(config['config_path']+'portfolio.json'):
    portfolio_record = read_config(file_path=config['config_path'], file_name='portfolio.json')

    # for us_stock
    if 'tiger' in platform:
      pr = portfolio_record.get('tiger')
      assets['glob'] = pr.get('global_account')
      # assets['simu'] = pr.get('simulation_account') 

    if 'futu' in platform:
      pr = portfolio_record.get('futu')
      assets['REAL'] = pr.get('REAL')
      # assets['SIMU'] = pr.get('SIMULATE')
    
    # for cn_stock
    pr = portfolio_record.get('pingan')
    assets['snowball'] = pr.get('snowball')

  # construct asset summary
  asset_info = '<h3>Portfolio</h3><ul>'
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
          position = position.drop('latest_time', axis=1)[['name', 'quantity', 'rate', 'market_value']].to_html()

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
    
    prefix = '' if pool in ['us', ''] else f'{pool}_' # 'a_' if cn_stock else ''
    signal_file = f'{config["result_path"]}{prefix}{signal_file_date}.xlsx'
    
    if os.path.exists(signal_file):

      signal_file_content =pd.read_excel(signal_file, sheet_name=None, dtype={'代码': str})
      for s in signal_file_content.keys():
        if s == 'signal':
          signals = signal_file_content.get('signal') # pd.read_excel(signal_file, sheet_name='signal')
          if signals is not None:
            # if len(signals) > 0:
            for s in ['b', 's', 'n']:
              font_color = signal_color[s]
              tmp_signals = signals.query(f'交易信号 == "{s}"')['代码'].astype(str).tolist()
              signal_info += f'<li>[ <b>{s}</b> ]: <font color="{font_color}">{", ".join(tmp_signals)}</font></li>'
        
        elif s == 'potential':
          potentials = signal_file_content.get('potential') # pd.read_excel(signal_file, sheet_name='potential')
          if potentials is not None:
            if len(potentials) > 0:
              # symbol name (for cn stocks only)
              potentials['名称'] = potentials['代码']
              potentials['名称'] = potentials['名称'].apply(lambda x: config['visualization']['plot_args']['sec_name'][x])
              potentials = potentials.set_index('代码')[config['postprocess']['send_columns']].sort_values('信号排名', ascending=False).to_html()
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
    pdf_names = ['portfolio', 'signal', 'potential', 'index'] if pool in ['us', ''] else [f'{pool}_portfolio', f'{pool}_signal', f'{pool}_potential']
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