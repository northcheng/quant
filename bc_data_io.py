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
import logging
import time
import pytz
import zipfile
import pickle
import json
import os

# mute warnings
import warnings
warnings.filterwarnings('ignore')

# data source
import easyquotation as eq
import akshare as ak

# mail process
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Literal, Optional
from pathlib import Path

# self defined
from quant import bc_util as util
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

# global variables
est_tz = pytz.timezone('US/Eastern')
utc_tz = pytz.timezone('UTC')
default_eod_key = 'OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'}
tqdm_ncols = 91

# module logger: 库层只发日志不配 handler, handler 配置权交给入口脚本(见 bc_util.setup_logging)
logger = logging.getLogger(__name__)

# eod api token: 优先读取本地 ~/quant/api_key/api_key.json 的 'eod'(正式 token),
# 读取失败则回退 demo/test token(仅白名单标的 AAPL/MSFT/AMZN/TSLA/MCD 可用).
# 入口脚本仍可通过 api_key 参数显式覆盖. 此处只记日志不打印 token 本身.
_eod_key_file = Path.home() / 'quant' / 'api_key' / 'api_key.json'
try:
  with open(_eod_key_file, 'r', encoding='utf-8') as f:
    _eod_cfg = json.loads(f.read())
  if _eod_cfg.get('eod'):
    default_eod_key = _eod_cfg['eod']
    logger.info(f'default_eod_key: loaded from {_eod_key_file}')
  else:
    logger.warning(f"default_eod_key: no 'eod' entry in {_eod_key_file}, fallback to demo/test token")
except Exception as e:
  logger.warning(f'default_eod_key: fallback to demo/test token ({_eod_key_file} unavailable): {e}')

# standards
# STANDARD_US_SYMBOL = 'AAPL'
# STANDARD_CN_SYMBOL = '000001' # 00700
# STANDARD_INTERVAL = 'd' # /w/m
BENCHMARK_SYMBOL = {'us': 'SPY', 'cn': '000001', 'hk': '00700'}
REFRESH_START_DATE = '2010-01-01'

# internal standard dataframe schema (single source of truth of the data contract):
# all stock dataframes converge to this shape no matter where they come from or go to.
#   - output boundary: post_process_download_data() produces it, validate_standard_dataframe() guards it
#   - input boundary:  load_stock_data(standard_columns=True) selects it, guarded by the same validator
STANDARD_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
# same as above without the time column ('Date' becomes the DatetimeIndex after df_2_timeseries)
STANDARD_DATA_COLUMNS = [col for col in STANDARD_COLUMNS if col != 'Date']

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
  'us_eod': 'eod', 
  'us_realtime': 'eod', 
  'cn_eod': 'eod', 
  'cn_realtime': 'easyquotation', 
  'hk_eod': 'eod', 
  'hk_realtime': 'easyquotation'
}


#----------------------------- Stock Data -------------------------------------#
def add_postfix_for_cn_symbol(symbol: str) -> str:
  """
  Add postfix for chinese stock symbols

  交易所后缀判定(修正): 原实现用 `int(symbol) < 600000` 一刀切, 把上交所 ETF
  (510300/560000/588000...)、科创板(688xxx)、北交所(430047/830799/870xxx) 全判成
  .SHE —— 它们在上交所/北交所, eod 源会**静默返回空数据**, 且从日志里看不出原因.
  现在按代码段显式判定, 并把无法归类的代码告警出来.

  :param symbol: cn symbol in STANDARD_CN_SYMBOL format
  :returns: symbol with postfix added
  :raises: none
  """
  # cn symbol should be all digit
  if symbol.isdigit():

    # mainland market (6 digits)
    if len(symbol) == 6:
      if symbol.startswith(('60', '68', '51', '56', '58', '50', '11', '90')):
        postfix = '.SHG'                                  # 沪市: 主板/科创板/沪 ETF/沪转债/B股
      elif symbol.startswith(('00', '30', '15', '16', '12', '20')):
        postfix = '.SHE'                                  # 深市: 主板/创业板/深 ETF/深转债/B股
      elif symbol.startswith(('43', '83', '87', '88', '92')):
        postfix = '.BSE'                                  # 北交所(原新三板精选层)
      else:
        logger.warning(f'{symbol}: 无法判定交易所前缀, 默认按深市 .SHE 处理')
        postfix = '.SHE'

    # hongkong market
    elif len(symbol) < 6:
      postfix = '.HK'

    else:
      logger.warning(f'{symbol}: cn symbol should be 5(hk) or 6(mainland) digits')
      postfix = ''

    symbol = symbol + postfix

  else:
    logger.warning(f'{symbol} is not all digit')

  return symbol

# preprocess symbol for downloading
def preprocess_symbol(symbols: list, style: Literal['eod', 'ak', 'easyquotation']) -> dict:
  """
  Add preprocess list of symbols to meet the style of different datasource

  :param symbols: list of symbols in STANDARD_CN_SYMBOL/STANDARD_CN_SYMBOL format
  :param style: name of the datasource: 'eod'/'ak'/'easyquotation'
  :returns: dict of {original symbol: processed symbol}
  :raises: none
  """

  result = {}

  # classify symbols
  us_symbols = [x.upper() for x in symbols if x.isalpha()]
  cn_symbols = [x for x in symbols if x.isdigit()]
  other_symbols = [x for x in symbols if x not in us_symbols and x not in cn_symbols]
  if len(other_symbols) > 0:
    logger.info(f'other symbols found: {other_symbols}')

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
    us_symbol_list = get_code_map_from_ak() # ak.stock_us_spot_em()
    # guard None code map(ak download failed and no local cache) to avoid TypeError
    if us_symbol_list is None:
      logger.error(f'[erro]: failed getting us code map from ak, us symbols will not be preprocessed: {us_symbols}')
      us_symbol_list = pd.DataFrame(columns=['代码', 'symbol'])
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
    logger.error(f'Unknown symbol style {style}')
  
  return result

# get us_spot_em from ak, for mapping US symbols to ak format
def get_code_map_from_ak() -> pd.DataFrame:

  # initialization
  today = f'{datetime.datetime.today().date()}'
  us_spot_em = None

  # mark-file that used for checking whether the file is up-to-date
  mark_file = f'updated_us_spot_em_{today}'
  local_file_name = f'us_spot_em.csv'
  
  # update us_spot_em if it is not updated today  
  if not os.path.exists(mark_file):
    
    # remove old files
    files = os.listdir()
    expired_local_file_name = [x for x in files if 'updated_us_spot_em_' in x]
    for elf in expired_local_file_name:
      os.remove(elf)

    # get new file
    retry_time = 5
    try_count = 0
    while try_count < retry_time:
      try_count += 1
      try:
        us_spot_em = ak.stock_us_spot_em()
        us_spot_em.to_csv(local_file_name, index=False)
        with open(mark_file, 'w', encoding='utf-8') as f:
          pass
        break
      except Exception as e:
        logger.exception(f'When calling ak.stock_us_spot(): {e}')
        continue
  
  # read old file if update failed
  if us_spot_em is None:
    if os.path.exists(local_file_name):
      us_spot_em = pd.read_csv(local_file_name)
      logger.info(f'Using existed us_spot_em.csv')

  return us_spot_em

# get ohlcv data from ak(US/CN/HK)
def get_data_from_ak(symbol: str, start_date: str = None, end_date: str = None, interval: str = 'daily', is_print: bool = False, adjust: str = 'qfq') -> pd.DataFrame:
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
      logger.error(f'symbol for unknown market: {symbol}')

  # get data
  if market == 'us':
    result = ak.stock_us_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)

  elif market == 'cn':
    if symbol.startswith('0') or symbol.startswith('3') or symbol.startswith('6'):
      result = ak.stock_zh_a_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)
    if symbol.startswith('15') or symbol.startswith('51') or symbol.startswith('56') or symbol.startswith('58'):
      # result = ak.stock_zh_a_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)      
      result = ak.fund_etf_hist_em(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)
  
  elif market == 'hk':
    result = ak.stock_hk_hist(symbol=symbol, period=interval, start_date=start_date, end_date=end_date, adjust=adjust)

  else:
    pass

  # postprocess: rename columns, add extra columns
  result = post_process_download_data(result, 'ak')

  if is_print and result is not None and len(result) > 0:
    logger.info(f'{symbol:8}: {result.index.min()} - {result.index.max()}, 下载记录 {len(result)} from ak')
  
  return result

# get ohlcv data from eod(US/CN/HK)
def get_data_from_eod(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: Literal['d', 'w', 'm'] = 'd', is_print: bool = False, api_key: str = default_eod_key, add_dividend: bool = False, add_split: bool = False) -> Optional[pd.DataFrame]:
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
    url = f'https://eodhd.com/api/eod/{symbol}?api_token={api_key}&period={interval}&fmt=json{from_to}'
    response = requests.get(url, headers=headers, timeout=30)
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

        try:
          # get dividend data
          url = f'https://eodhd.com/api/div/{symbol}?api_token={api_key}&fmt=json{from_to}'
          response = requests.get(url, headers=headers, timeout=30)
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
        
        except Exception as e:
          logger.exception(f'failed to get dividend({e}), filled with 0')
      else:
        if is_print:
          print(f'dividend(-)', end=', ' )
        pass

      # add split data
      if add_split:

        try:
          # get split data
          url = f'https://eodhd.com/api/splits/{symbol}?api_token={api_key}&fmt=json{from_to}'  
          response = requests.get(url, headers=headers, timeout=30)
          if response.status_code==200: 
            split = response.json()
            response_status = 'o' 
          else:
            split = []
            response_status = 'x' 

          if is_print:
            logger.info(f'split({response_status})')
          
          # post process dividend data
          if len(split) > 0:
            split_data = pd.DataFrame(split)
            split_data.split = split_data.split.apply(lambda x: float(x.split('/')[0])/float(x.split('/')[1]))
            data = pd.merge(data, split_data, how='left', left_on='date', right_on='date')
            data['split'] = data['split'].fillna(1.0)
        
        except Exception as e:
          logger.exception(f'failed to get split({e}), filled with 1')
      else:
        if is_print:
          logger.info(f'split(-)')
        pass
        
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
    logger.exception(f'{symbol} {e}')

  return data

# get ohlcv data from easyquotation(HK: 1500rows)
def get_data_from_easyquotation(symbol: str, is_print: bool = False) -> pd.DataFrame:
  """
  Download HK stock data from easyquotation
  https://github.com/shidenggui/easyquotation

  :param symbol: target symbol, e.g. '00700'
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
    logger.info(f'{symbol:5}: {df.index.min()} - {df.index.max()}, 下载记录 {len(df)} from easyquotation_daykline')

  return df

# validate a dataframe against the standard schema (data contract guard, ~20 lines on purpose)
def validate_standard_dataframe(df: pd.DataFrame, source: str = '') -> bool:
  """
  Check whether a dataframe conforms to the internal standard schema:
  DatetimeIndex + all STANDARD_DATA_COLUMNS present.

  :param df: dataframe to check
  :param source: hint of where the dataframe comes from, used in diagnostics
  :returns: True if valid, False otherwise
  :raises: none
  """
  prefix = f'[data contract] {source}: ' if source else '[data contract]: '
  if not isinstance(df, pd.DataFrame) or len(df) == 0:
    logger.warning(f'{prefix}dataframe is empty or not a dataframe')
    return False
  if not isinstance(df.index, pd.DatetimeIndex):
    logger.warning(f'{prefix}index is not DatetimeIndex but {type(df.index).__name__}')
    return False
  missing = [col for col in STANDARD_DATA_COLUMNS if col not in df.columns]
  if len(missing) > 0:
    logger.warning(f'{prefix}missing standard columns: {missing}')
    return False
  return True

# postprocess downloaded data
def post_process_download_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
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
          
      df = df[STANDARD_COLUMNS].copy()
      df = util.df_2_timeseries(df, time_col='Date')
    
    # data from easyquotation
    elif source == 'easyquotation_daykline':
      df['Adj Close'] = df['Close'].copy()
      df['Dividend'] = 0.0
      df['Split'] = 1.0
          
      df = df[STANDARD_COLUMNS].copy()
      df = util.df_2_timeseries(df, time_col='Date')
    
    # remove duplicated index and sort data by index
    df = util.remove_duplicated_index(df, keep='last')
    df.sort_index(ascending=True, inplace=True)

    # guard the standard schema at the output boundary
    if source in ('eod', 'ak', 'easyquotation_daykline'):
      validate_standard_dataframe(df=df, source=f'post_process_download_data({source})')

  return df

# get data from specific datasource
def get_data(symbol: str, start_date: str = None, end_date: str = None, interval: str = 'd', is_print: bool = False, source: str = 'eod', api_key: str = default_eod_key, add_dividend: bool = False, add_split: bool = False, adjust: str = 'qfq') -> pd.DataFrame:
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
      logger.error(f'data source {source} not found')

  except Exception as e:
    logger.exception(f'{symbol} {e}')

  return data

# get 15min-delayed US market realtime data from eod
def get_real_time_data_from_eod(symbols: list, api_key: str = default_eod_key, is_print: bool = False, batch_size: int = 15) -> pd.DataFrame:
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
    url = f'https://eodhd.com/api/real-time/{first_symbol}?api_token={api_key}&fmt=json&s={other_symbols}'
    response = requests.get(url, headers=headers, timeout=30)
    try:
      real_time = [] if response.status_code!=200 else response.json()
    except Exception as e:
      logger.exception(f'{e} {response.status_code} {response}')
      real_time = []

    real_time = [real_time] if isinstance(real_time, dict) else real_time
    
    if is_print:
      logger.info(f'[{"realtime":^19}] - updating real-time for symbol: {batch_start+1:3} -{batch_end:3} - {response}')      
  
    # post process downloaded real-time data
    real_time_data = pd.DataFrame(real_time)
    if not real_time_data.empty:

      to_drop = []
      for idx, row in real_time_data.iterrows():
        try:
          real_time_data.loc[idx, 'latest_time'] = util.timestamp_2_time(real_time_data.loc[idx, 'timestamp'], unit='s').astimezone(est_tz).replace(tzinfo=None)
          real_time_data.loc[idx, 'code'] = real_time_data.loc[idx, 'code'].split('.')[0]
        except Exception as e:
          logger.exception(e)
          logger.exception(row)
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

      logger.warning('get empty response for realtime data')
      
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
    logger.warning('real time data from eod is empty')

  return result

# get realtime CN market realtime data from easyquotation
def get_real_time_data_from_easyquotation(symbols: list, source: str = 'sina') -> pd.DataFrame:
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
  logger.info(f'[{"realtime":^19}] - updating real-time for symbols from easyquotation({source})')

  # get stock data from sina
  codes = [x.split('.')[0] for x in symbols]

  if source == 'sina':
    qt = quotation.stocks(codes) 
    columns_to_keep = {'index':'symbol', 'open':'Open', 'high':'High', 'low':'Low', 'now':'Close', 'turnover':'Volume', 'date':'Date'}

  elif source == 'hkquote':
    qt = quotation.real(codes) 
    # hkquote 同时返回 `date`('YYYY-MM-DD')与 `time`('HH:MM:SS'). 原来把 `time` 映射到
    # `Date`, 再 `x[:10]`(对 8 字符的时间串是空操作) —— pandas 会把它解析成"今天+该时刻",
    # 于是 realtime 模式插入**幻影 bar**(同日两根), 且时间戳与历史行不匹配会让 .loc 变成新增行.
    # 用 `date` 才是正确语义(上行 sina 分支用的就是 date).
    columns_to_keep = {'index':'symbol', 'openPrice':'Open', 'high':'High', 'low':'Low', 'price':'Close', 'amount':'Volume', 'date':'Date'}

  else:
    logger.error(f'unknown source {source}')

  # turn into dataframe
  df = pd.DataFrame(qt).T if len(qt) > 0 else df
  if source == 'hkquote':
    if 'Date' in df.columns:
      # 只取日期部分(个别源可能带时间后缀)
      df['Date'] = df['Date'].astype(str).str.slice(0, 10)
    else:
      logger.error('[realtime]: hkquote 返回缺少 date 字段, Date 列不可用; '
                   '请检查 easyquotation 版本(字段契约可能已变)')

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
def get_real_time_data(symbols: list, source: str = 'eod', sub_source: str = 'sina', api_key: str = default_eod_key, is_print: bool = False, batch_size: int = 15) -> pd.DataFrame:

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
      logger.error(f'data source {source} not found')

  except Exception as e:
    logger.exception(f'{symbols} {e}')

  return data

# get realtime data for a list of stocks from eod
def get_stock_briefs_from_eod(symbols: list, api_key: str = default_eod_key, batch_size: int = 15) -> pd.DataFrame:
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
    logger.warning('real time data from eod is empty')

  return latest_data

# get realtime data for a list of stocks from specified datasource
def get_stock_briefs(symbols: list, source: str = 'eod', api_key: str = default_eod_key, batch_size: int = 15) -> pd.DataFrame:
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
    logger.error(f'Unknown source {source}')

  return briefs

# update stock data (eod and/or realtime)
def update_stock_data_new(symbols: list, stock_data_path: str, file_format: str = '.csv', update_mode: str = 'eod', required_date: str = None, window_size: int = 3, is_print: bool = False, is_return: bool = False, is_save: bool = True, sources: dict = default_data_sources, api_keys: dict = {}, add_dividend: bool = False, add_split: bool = False, batch_size: int = 15, adjust: str = 'qfq', query_benchmark: bool = True) -> dict:
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
  :param api_keys: api tokens for different data sources
  :param add_dividend: whether to add dividend data
  :param add_split: whether to add split data
  :param batch_size: batch size of symbols of getting real-time data  
  :returns: dict of dataframes with symbol as key
  :raises: none
  """

  global BENCHMARK_SYMBOL

  # verify update_mode
  if update_mode not in ['realtime', 'eod', 'both', 'refresh']:
    logger.error(f'unknown update mode: {update_mode}')
    return None

  # classify symbols
  us_symbols = [x for x in symbols if x.isalpha()]
  cn_symbols = [x for x in symbols if x.isdigit() and len(x) == 6]
  hk_symbols = [x for x in symbols if x.isdigit() and len(x) == 5]
  other_symbols = [x for x in symbols if (x not in us_symbols and x not in cn_symbols and x not in hk_symbols)]
  symbol_class = {'us': us_symbols, 'cn': cn_symbols, 'hk': hk_symbols, 'other': other_symbols}
  symbol_count = {'us': len(us_symbols), 'cn': len(cn_symbols), 'hk': len(hk_symbols), 'other': len(other_symbols)}
  # print(f'US({symbol_count["us"]}), CN({symbol_count["cn"]}), HK({symbol_count["hk"]}), Other({symbol_count["other"]})')
  if symbol_count['other'] > 0:
    logger.error(f'[{"pool":^19}] - [erro]: Unexpected symbols found: {other_symbols}')

  # default dates
  today = datetime.datetime.today().date()
  today = util.time_2_string(today)
  start_date = util.string_plus_day(today, -7)

  # set benchmarks for different markets
  benchmark_symbols = BENCHMARK_SYMBOL
  benchmark_dates = {}
  benchmark_api_keys = {}

  stock_data_path = Path(stock_data_path)
  for mkt in benchmark_symbols.keys():
    benchmark_source = sources[f'{mkt}_eod']
    benchmark_api_keys[mkt] = api_keys.get(benchmark_source)
    mkt_symbol_count = symbol_count[mkt]
    mkt_benchmark_symbol = preprocess_symbol([benchmark_symbols[mkt]], benchmark_source).get(benchmark_symbols[mkt])
    # mkt_benchmark_symbol = '105.AAPL' if (mkt == 'us' and benchmark_source == 'ak') else mkt_benchmark_symbol
    
    # when query_benchmark required
    if query_benchmark:
      # when there's symbol for this market
      if symbol_count[mkt] > 0 and mkt_benchmark_symbol is not None:
        
        # try 5 times if there's something wrong querying the data
        retry_count = 0
        while retry_count < 5:
          try:
            retry_count += 1
            # print(f'[data]: querying benchmark({mkt_benchmark_symbol}) date for [{mkt.upper()}] from {benchmark_source} (try #{retry_count})')        

            # get data for benchmark symbol of current market
            tmp_data = get_data(mkt_benchmark_symbol, start_date=start_date, end_date=today, interval='d', is_print=False, source=benchmark_source, api_key=benchmark_api_keys[mkt], add_dividend=False, add_split=False, adjust='qfq')
            if len(tmp_data) > 0:
              benchmark_dates[mkt] = util.time_2_string(tmp_data.index.max()) 
            else:   
              logger.warning(f'[-{mkt.upper()} - [{mkt.upper():^4}]: benchmark({mkt_benchmark_symbol}) data is empty, use today - {today}')
              benchmark_dates[mkt] = today
            
            # break when finish
            break

          except Exception as e:
            
            # if failed, sleep 5 seconds and try again
            logger.exception(f'[{"benchmark":^19}] - [erro]: querying benchmark failed for [{mkt} market], try({retry_count}/5), {type(e)} - {e}')        
            time.sleep(5)
            continue
          
    # check whether got behchmart data successfully
    if mkt_symbol_count > 0:
      tmp_benchmark_date = benchmark_dates.get(mkt)
      if tmp_benchmark_date is None:
        benchmark_dates[mkt] = today
        logger.info(f'[{mkt.upper()+" market":^19}] - [pool]: symbols({mkt_symbol_count}), benchmark({mkt_benchmark_symbol}), date(skip benchmark query, use today - {today})') # 
      else:
        logger.info(f'[{mkt.upper()+" market":^19}] - [pool]: symbols({mkt_symbol_count}), benchmark({mkt_benchmark_symbol}), date({benchmark_dates[mkt]})')

  time.sleep(2)

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
    max_retries = 5
    retry_delay = 2
    while retry_count < max_retries:
      try:
        time.sleep(1)
        retry_count += 1
        # print(f'[data]: updating data for [{mkt.upper()}] from {tmp_source} (try #{retry_count})')

        # get the existed data and its latest date for each symbols
        process_counter = 0
        for symbol in symbol_class[mkt]:
          process_counter += 1

          # init symbol data and its most recent date
          data[symbol] = pd.DataFrame()
          tmp_data_date = None

          # if local data exists, load existed data, update its most current date
          symbol_file_name = stock_data_path / f'{symbol}{file_format}'
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
                  # 不删文件: load_stock_data 已负责"确实不可读才删". 这里删掉再下载,
                  # 一旦下载失败就是**永久丢历史**. 改为原地重下(全量窗口).
                  logger.warning(f'max index of {symbol} is invalid({max_idx}), 将全量重下(保留原文件)')
                  tmp_data_date = None
              else:
                # 同上: 返回 None 可能是"列格式问题"(文件仍有效), 不能一律删掉
                logger.warning(f'data {symbol} 未能加载(空/格式问题), 将全量重下(保留原文件)')
                tmp_data_date = None

          # update eod data, print updating info
          if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_dates[mkt] or (tmp_data_date <= benchmark_dates[mkt] and tmp_source == 'ak')):
            
            if is_print:
              print(f'[{process_counter:04}]from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
            
            # download latest data for current symbol
            start_date = util.string_plus_day(tmp_data_date, -3) if tmp_data_date is not None else tmp_data_date
            tmp_symbol = tmp_source_symbols.get(symbol)
            
            if tmp_symbol is not None:

              # get new data
              new_data = get_data(symbol=tmp_symbol, start_date=start_date, end_date=required_date, interval='d', is_print=is_print, source=tmp_source, api_key=benchmark_api_keys[mkt], add_dividend=add_dividend, add_split=add_split, adjust=adjust)
              # 空返回(= get_data 吞掉异常)不得当成"已更新": 否则旧数据被保留却被视为
              # 下载成功, 下游按"已是最新"跳过该标的, universe 静默缩小.
              if new_data is None or len(new_data) == 0:
                logger.error(f'{symbol:5}: 下载返回空({tmp_source}), 视为更新失败, 保留本地旧数据')
                new_data = None
              else:
                # 增量拼接一致性校验(复权基准变化会让历史留下人为台阶)
                _check_overlap_consistency(data[symbol], new_data, symbol)
                # 保留本地已知的分红/拆股真值(新窗口若是未知占位 0/1 则不得覆盖)
                new_data = _merge_dividend_split(data[symbol], new_data)
                # append new data to the origin
                data[symbol] = pd.concat([data[symbol], new_data])
                data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

                # save data to local csv files
                if is_save:
                  save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
            
            else:
              new_data = None
              logger.warning(f'{symbol:5}: symbol not found from source {tmp_source}')
            
          else:
            if symbol not in up_to_date_symbols:
              up_to_date_symbols.append(symbol)
            
        num_symbol_up_to_date = len(up_to_date_symbols)
        if num_symbol_up_to_date > 0:
          if is_print:
            logger.warning(f'[{"eod":^19}] - [skip] : {num_symbol_up_to_date}/{len(symbol_class[mkt])} already up-to-date')

        # add real-time data when requiring data return and data will NOT be saved
        if update_mode in ['realtime', 'both']:
          # print('***************** querying real-time data *****************')
          tmp_source = sources[f'{mkt}_realtime']
          tmp_sub_source = 'hkquote' if mkt == 'hk' else 'sina'

          if sources[f'{mkt}_eod'] in ['ak'] and update_mode in ['both']:
            logger.info(f'[{"realtime":^19}] - real-time data comes with eod data from ak')

          else:
            real_time_data = get_real_time_data(symbols=symbol_class[mkt], source=tmp_source, sub_source=tmp_sub_source, api_key=benchmark_api_keys[mkt], is_print=is_print, batch_size=batch_size)
            if tmp_source == 'eod':
              real_time_data = util.df_2_timeseries(df=real_time_data, time_col='Date')

            # append it corresponding eod data according to symbols
            for symbol in symbol_class[mkt]:
              tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
              if len(tmp_data) == 0:
                logger.warning(f'real-time data not found for {symbol}')
                continue
              else:
                tmp_idx = tmp_data.index.max()
                for col in data[symbol].columns:
                  data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
                data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

        break
      except Exception as e:
        logger.exception(f'[erro]: updating data failed for [{mkt} market], try({retry_count}/5), {type(e)} - {e}')
        time.sleep(retry_delay)
        continue
  
  # return
  if is_return:
    return data

# 增量拼接一致性校验的重叠列(只比价格, 不比 Volume/派生列)
_OVERLAP_PRICE_COLS = ['Open', 'High', 'Low', 'Close', 'Adj Close']


def _is_missing(value) -> bool:
  """
  安全判空(对 None / NaN / 字符串占位都成立, 且不抛异常).

  用途: 从 portfolio.json 或 ta_result 取出的值**可能是字符串**(例如占位 '--' 或 ''),
  这时 `np.isnan(value)` 会抛 TypeError, 让整条告警链路崩掉. 本函数统一处理:
    * None -> True
    * 字符串 -> 空串 / '-' / '--' / 'nan' / 'none' / 'null' 视为缺失
    * 数值(含 numpy 标量) -> NaN/±inf 视为缺失
    * 其它类型(如 list/dict) -> True(无法当价格用)

  :param value: 待判定的值
  :returns: True = 缺失/不可用
  :raises: none
  """
  if value is None:
    return True
  if isinstance(value, str):
    return value.strip().lower() in ('', '-', '--', 'nan', 'none', 'null')
  try:
    return bool(pd.isna(value)) or not np.isfinite(float(value))
  except (TypeError, ValueError):
    return True


# 除权除息列的"未知"占位值: add_dividend/add_split 关闭时, 各数据源会把这些列写成 0.0/1.0.
# 0.0/1.0 本身是合法取值, 所以"未知"与"确实为 0/1"无法区分 —— 一旦覆盖到已有真值, 就是
# 把真实分红/拆股记录抹掉(Dividend 会影响 /1 复权与周月聚合, Split 会影响复权).
_UNKNOWN_DIVIDEND = 0.0
_UNKNOWN_SPLIT = 1.0


def _merge_dividend_split(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
  """
  Merge a freshly downloaded window over the stored data **without clobbering known
  dividend/split values with "unknown" placeholders**.

  Default is `add_dividend=False` / `add_split=False`, in which case every source writes
  `Dividend = 0.0` and `Split = 1.0`. Combined with `keep='last'` in the caller, running
  once with `add_dividend=True` and then once without would silently overwrite the real
  values in the newly downloaded window back to 0.0/1.0.

  Rule: keep the incoming value unless it is exactly the placeholder **and** the stored
  value is different (i.e. we actually knew something better before).

  :param old: stored data (already reindexed/sorted, may be empty)
  :param new: freshly downloaded window
  :returns: `new` with Dividend/Split restored where the new window only has placeholders
  :raises: none
  """
  if old is None or len(old) == 0:
    return new
  new = new.copy()
  for col, placeholder in (('Dividend', _UNKNOWN_DIVIDEND), ('Split', _UNKNOWN_SPLIT)):
    if col not in new.columns or col not in old.columns:
      continue
    common = new.index.intersection(old.index)
    if len(common) == 0:
      continue
    old_vals = pd.to_numeric(old.loc[common, col], errors='coerce')
    new_vals = pd.to_numeric(new.loc[common, col], errors='coerce')
    # 新值是占位 且 老值是"已知且不是占位" -> 用老值
    restore = (new_vals == placeholder) & old_vals.notna() & (old_vals != placeholder)
    if bool(restore.any()):
      idx = restore.index[restore.to_numpy()]
      logger.info(f'[merge]: {col} 保留本地已知值 {len(idx)} 处'
                  f'(新数据为未知占位 {placeholder})')
      new.loc[idx, col] = old_vals.loc[idx].to_numpy()
  return new


def _check_overlap_consistency(existed: pd.DataFrame, new: pd.DataFrame, symbol: str,
                                rtol: float = 1e-3) -> bool:
  """
  Compare the overlapping dates of the locally stored data and a freshly downloaded
  window, and report a mismatch.

  Why this matters: akshare returns **front-adjusted (qfq)** prices normalised to the
  latest trading day *at request time*. Every dividend/split shifts the whole history.
  Incremental updates only re-download the last few days (`last_date - 3`), so after an
  ex-dividend event the stored older rows keep the OLD adjustment basis while the newly
  appended rows use the NEW one -- an artificial step (1~3%, up to 20%+ for large
  splits) that the factor pipeline reads as a real return. A large enough step is
  invisible to `_sanitize_prices` in factor.backtest (its threshold is |ret| > 100%).

  This check cannot repair the history by itself (it only sees the overlap); it makes
  the problem **visible** so the caller can force a full re-download / adjustment.

  :param existed: locally stored data (may be None)
  :param new: freshly downloaded window
  :param symbol: symbol, for the log message
  :param rtol: relative tolerance for price equality
  :returns: True if consistent (or nothing to compare), False if a mismatch was found
  :raises: none
  """
  if existed is None or len(existed) == 0 or new is None or len(new) == 0:
    return True
  try:
    common = existed.index.intersection(new.index)
    if len(common) == 0:
      return True
    cols = [c for c in _OVERLAP_PRICE_COLS if (c in existed.columns and c in new.columns)]
    if not cols:
      return True
    old = existed.loc[common, cols].astype(float)
    fresh = new.loc[common, cols].astype(float)
    # 两侧都有效的格子才比; 相对误差按 |old| 归一(老价为 0 时退化为绝对比较)
    both = old.notna() & fresh.notna()
    if not bool(both.to_numpy().any()):
      return True
    diff = (fresh - old).abs()
    tol = old.abs() * rtol
    tol = tol.where(tol > 0, rtol)                    # old == 0 -> 绝对容差
    bad = both & (diff > tol)
    if not bool(bad.to_numpy().any()):
      return True

    n_bad = int(bad.to_numpy().sum())
    bad_dates = sorted(set(bad.index[bad.any(axis=1)]))
    worst = float((diff.where(both)).max().max())
    logger.error(
        f'[overlap]: {symbol} 重叠窗口 {len(common)} 行中有 {n_bad} 个价格格不一致'
        f'(最大绝对差 {worst:.6g}, 容差 {rtol:g}); 疑似复权基准变化 —— '
        f'受影响的日期 {[util.time_2_string(d) for d in bad_dates[:5]]}'
        f'{"..." if len(bad_dates) > 5 else ""}; '
        f'增量拼接会在历史里留下人为台阶, 建议对该标的全量重下(refresh)')
    return False
  except Exception as e:
    logger.exception(f'[overlap]: {symbol} 一致性校验本身失败({e}), 跳过校验')
    return True


# 下载单个标的的EOD数据（用于多线程并行）
def download_single_symbol_eod(args) -> tuple:
  """
  下载单个标的的EOD数据（用于多线程并行）
  
  :param args: tuple of (ssymbol, market, start_date, required_date, benchmark_date, update_mode, source, api_key, stock_data_path, file_format, is_save, add_dividend, add_split, adjust)
  :returns: tuple of (symbol, data, status, error_msg)
  """

  # unpack args
  symbol, market, start_date, required_date, benchmark_date, update_mode, source, api_key, stock_data_path, file_format, is_save, add_dividend, add_split, adjust = args
  # initialize parameters
  retry_count = 0
  max_retries = 5
  retry_delay = 2

  # try to download data, up to 5 times
  data = None
  # initialize query_symbol to avoid NameError in the return statement when the first try fails before assignment
  query_symbol = None
  while retry_count < max_retries:

    retry_count += 1
    try:

      # preprocess symbol
      query_symbol = preprocess_symbol(symbols=[symbol], style=source).get(symbol)
      if query_symbol is None:
        status = 'symbol_not_found'
        messages = f'[erro]: Symbol {symbol} not found from source {source}'
        break

      # get local data
      symbol_file_name = stock_data_path / f'{symbol}{file_format}'
      tmp_data_date = None
      if os.path.exists(symbol_file_name):

        # delete local data if refresh mode
        if update_mode == 'refresh':
          os.remove(symbol_file_name)

        # otherwise load local data and get its most recent date
        else:
          existed_data = load_stock_data(file_path=stock_data_path, file_name=symbol, file_format=file_format)
          if existed_data is not None and len(existed_data) > 0:
            
            max_idx = existed_data.index.max()
            if max_idx > util.string_2_time('2020-01-01'):
              tmp_data_date = util.time_2_string(max_idx)
              data = existed_data

              if tmp_data_date >= benchmark_date:   
                           
                status = 'already_up_to_date'
                messages = None
                break

            else:
              # 不删文件(删除+下载失败 = 永久丢历史); 置 tmp_data_date=None 即走全量重下
              logger.warning(f'max index of {symbol} is invalid({max_idx}), 将全量重下(保留原文件)')
              tmp_data_date = None

          else:
            logger.warning(f'data {symbol} 未能加载(空/格式问题), 将全量重下(保留原文件)')
            tmp_data_date = None

      # update eod data for those which local data date < benchmark date
      if tmp_data_date is not None:
        start_date = util.string_plus_day(tmp_data_date, -3)
      else:
        start_date = None

      # get eod data
      new_data = get_data(
        symbol=query_symbol,
        start_date=start_date,
        end_date=required_date,
        interval='d',
        is_print=False,
        source=source,
        api_key=api_key,
        add_dividend=add_dividend,
        add_split=add_split,
        adjust=adjust
      )

      # 红线: get_data 失败时返回 None(它自己吞掉异常). 若把 None 算作"已更新",
      # pd.concat([旧, None]) 会静默返回旧数据, 而下面 status='success' 无条件执行 ——
      # 结果是**陈旧数据被计入"下载成功"**, 5 次重试永不因网络错误触发, 下游还会
      # 按"已是最新"跳过这些标的(universe 静默缩小). 必须显式判空并按失败处理.
      if new_data is None or len(new_data) == 0:
        status = 'failed'
        messages = f'get_data 返回空({source})'
        if retry_count < max_retries:
          time.sleep(retry_delay)
        continue

      # append new data to local data if not already updated
      if data is not None:
        # 增量拼接一致性校验: 重叠窗口(-3 天)里新旧价格必须吻合. 不吻合说明数据源
        # 复权基准变了(除权除息会整体平移前复权价), 此时只刷新最近几行会在历史里
        # 留一个人为台阶, 且新因子含未来分红信息(前视). 发现不一致即报出来.
        _check_overlap_consistency(existed_data, new_data, symbol)
        # 保留本地已知的分红/拆股真值(新窗口若是未知占位 0/1 则不得覆盖)
        new_data = _merge_dividend_split(existed_data, new_data)
        data = pd.concat([existed_data, new_data])
        # remove duplicated dates from the overlapping download window(-3 days), keep the newly downloaded rows
        data = util.remove_duplicated_index(df=data, keep='last').sort_index()
      else:
        data = new_data
      
      # save data if specified
      if is_save and data is not None and len(data) > 0:
        save_stock_data(
          df=data,
          file_path=stock_data_path,
          file_name=symbol,
          file_format=file_format,
          reset_index=True,
          index=False
        )
      
      status = 'success'
      messages = None
      break
      
    except Exception as e:
      logger.exception(f'[erro]: failed updating eod data for [{symbol}](parallely), try({retry_count}/5), {type(e)} - {e}')
      status = 'failed'
      messages = f'{type(e).__name__}: {str(e)}'
      if retry_count < max_retries:
        time.sleep(retry_delay)
  
  return (symbol, query_symbol, data, status, messages)

# 并行下载EOD数据
def download_eod_parallel(symbols: List[str], stock_data_path: str, file_format: str = '.csv', sources: Dict[str, str] = None, api_keys: Dict[str, str] = None, update_mode: str = 'eod', required_date: str = None, is_print: bool = False, is_save: bool = True, add_dividend: bool = False, add_split: bool = False, adjust: str = 'qfq', progress_callback=None, query_benchmark: bool = True, max_workers: int = 5, batch_size: int = 15) -> Dict[str, pd.DataFrame]:
  """
  并行下载多个标的的EOD数据和REALTIME数据

  :param symbols: 标的列表
  :param stock_data_path: 本地存储路径
  :param file_format: 文件格式，默认为 '.csv'
  :param sources: 数据源配置，默认为 default_data_sources
  :param api_keys: API密钥配置
  :param update_mode: 更新模式，'eod' 时仅更新比本地新的数据，'refresh' 时强制重新下载
  :param required_date: 要求的数据截止日期
  :param is_print: 是否打印详细信息
  :param is_save: 是否保存到本地文件
  :param add_dividend: 是否添加分红数据
  :param add_split: 是否添加拆股数据
  :param adjust: 复权类型，'qfq' 前复权 / 'hfq' 后复权 / 'none' 不复权
  :param progress_callback: 进度回调函数，签名为 callback(success_count, failed_count, total_count)
  :param query_benchmark: 是否查询各市场基准标的获取最新数据日期
  :param max_workers: 最大并行线程数，默认为 5
  :param batch_size: 批处理大小，默认为 15
  :returns: dict，{symbol: dataframe}，包含所有标的的EOD数据（已与本地数据合并）
  """

  # get benchmark symbols
  global BENCHMARK_SYMBOL

  # guard None api_keys so .get() calls below won't crash
  if api_keys is None:
    api_keys = {}

  # default dates
  today = datetime.datetime.today().date()
  today = util.time_2_string(today)
  start_date = util.string_plus_day(today, -7)

  # verify symbols
  num_symbols = len(symbols)
  if num_symbols == 0:
    logger.warning('empty symbols list')
    return None
  
  # verify stock_data_path
  stock_data_path = Path(stock_data_path)
  if not stock_data_path.exists():
    logger.warning(f'stock_data_path {stock_data_path} not exist')
    return None

  # verify update_mode
  if update_mode not in ['realtime', 'eod', 'both', 'refresh']:
    logger.error(f'unknown update mode: {update_mode}')
    return None

  # verify sources: api_keys 已有 None 守卫(见下), sources 没有 —— None 会在
  # sources[f'{mkt}_eod'] 处抛 TypeError, 而函数签名允许 None 默认值. 补齐同一层守卫.
  if sources is None:
    logger.error('sources is None: 需要 {market}_eod / {market}_realtime 映射, 无法继续')
    return None
  missing_sources = [k for k in ('us_eod', 'cn_eod', 'hk_eod') if k not in sources]
  if missing_sources:
    logger.error(f'sources 缺少必需的键 {missing_sources}')
    return None

  # classify symbols by market
  us_symbols = [x for x in symbols if x.isalpha()]
  cn_symbols = [x for x in symbols if x.isdigit() and len(x) == 6]
  hk_symbols = [x for x in symbols if x.isdigit() and len(x) == 5]
  other_symbols = [x for x in symbols if (x not in us_symbols and x not in cn_symbols and x not in hk_symbols)]
  
  symbol_count = {'us': len(us_symbols), 'cn': len(cn_symbols), 'hk': len(hk_symbols), 'other': len(other_symbols)}
  symbol_class = {'us': us_symbols, 'cn': cn_symbols, 'hk': hk_symbols, 'other': other_symbols}
  # print(f'[{"pool":^19}] - US({symbol_count["us"]}), CN({symbol_count["cn"]}), HK({symbol_count["hk"]}), Other({symbol_count["other"]})')
  if symbol_count['other'] > 0:
    logger.error(f'[{"pool":^19}] - [erro]: Unexpected symbols found: {other_symbols}')

  # -------------------------------------------------- benchmark --------------------------------------------------#
  # set benchmarks for different markets
  benchmark_symbols = BENCHMARK_SYMBOL
  benchmark_dates = {}
  benchmark_api_keys = {}

  # query benchmark dates for each market
  for mkt in benchmark_symbols.keys():
    benchmark_source = sources[f'{mkt}_eod']
    benchmark_api_keys[mkt] = api_keys.get(benchmark_source)
    mkt_symbol_count = symbol_count[mkt]
    mkt_benchmark_symbol = preprocess_symbol([benchmark_symbols[mkt]], benchmark_source).get(benchmark_symbols[mkt])
    # mkt_benchmark_symbol = '105.AAPL' if (mkt == 'us' and benchmark_source == 'ak') else mkt_benchmark_symbol
    
    # when query_benchmark required, there's symbol for this market and benchmark symbol is not None
    if query_benchmark:
      if mkt_symbol_count > 0 and mkt_benchmark_symbol is not None:
        
        # try to get benchmark data for current market, up to 5 times if failed, use today as default date
        retry_count = 0
        while retry_count < 5:
          try:
            retry_count += 1
            
            # get data for benchmark symbol of current market
            tmp_data = get_data(mkt_benchmark_symbol, start_date=start_date, end_date=today, interval='d', is_print=False, source=benchmark_source, api_key=benchmark_api_keys[mkt], add_dividend=False, add_split=False, adjust='qfq')
            if tmp_data is not None and len(tmp_data) > 0:
              benchmark_dates[mkt] = util.time_2_string(tmp_data.index.max()) 
            else:                 
              benchmark_dates[mkt] = today
              logger.warning(f'[{"benchmark":^19}] - [{mkt.upper():^4}]: benchmark({mkt_benchmark_symbol}) data is empty, use today - {today}')
            
            # break when finish
            break

          except Exception as e:
            
            # if failed, sleep 5 seconds and try again
            logger.exception(f'[{"benchmark":^19}] - [erro]: failed querying benchmark for [{mkt} market], try({retry_count}/5), {type(e)} - {e}')        
            time.sleep(5)
            continue

    # check whether got behchmart data successfully
    if mkt_symbol_count > 0:
      tmp_benchmark_date = benchmark_dates.get(mkt)
      if tmp_benchmark_date is None:
        benchmark_dates[mkt] = today
        logger.info(f'[{mkt.upper()+" market":^19}] - [pool]: symbols({mkt_symbol_count}), benchmark({mkt_benchmark_symbol}), date(skip benchmark query, use today - {today})') # 
      else:
        logger.info(f'[{mkt.upper()+" market":^19}] - [pool]: symbols({mkt_symbol_count}), benchmark({mkt_benchmark_symbol}), date({benchmark_dates[mkt]})')
    else:
      continue
  time.sleep(2)

  # -------------------------------------------------- eod data ---------------------------------------------------#
  # construct download tasks
  download_tasks = []
  for market, market_symbols in symbol_class.items():
    
    # skip market with an empty list
    if len(market_symbols) == 0:
      continue

    # get source and api key for current market
    tmp_source = sources.get(f'{market}_eod')
    tmp_api_key = api_keys.get(tmp_source, '')
    
    # get benchmark date for current market
    mkt_benchmark_date = benchmark_dates.get(market)
    if mkt_benchmark_date is None:
      mkt_benchmark_date = today
    
    start_date = None

    # construct download tasks for each symbol
    for symbol in market_symbols:

      download_tasks.append(
        (symbol, market, start_date, required_date, mkt_benchmark_date, update_mode, tmp_source, tmp_api_key, stock_data_path, file_format, is_save, add_dividend, add_split, adjust)
      )
  
  # parallel download eod data
  symbol_downloaded = []
  symbol_up_to_date = []
  symbol_not_found = []
  symbol_download_failed = []
  
  # parallelly run download tasks
  data = {}
  if len(download_tasks) > 0:

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {executor.submit(download_single_symbol_eod, task): task[0] for task in download_tasks}
      
      for future in tqdm(as_completed(futures), total=len(futures), ncols=tqdm_ncols, desc=f'[{"downloading":^19}] '):
        # symbol = futures[future]
        
        try:
          # get thread result from future
          result_symbol, query_symbol, result_data, status, error_msg = future.result()
          if query_symbol is not None and result_symbol is not None and result_data is not None and len(result_data) > 0:
            data[result_symbol] = result_data

          # if thread download successfully
          if status == 'success' and result_data is not None:
            symbol_downloaded.append(result_symbol)
            
          elif status == 'symbol_not_found':
            symbol_not_found.append(result_symbol)

          elif status == 'already_up_to_date':
            symbol_up_to_date.append(result_symbol)

          elif status == 'failed':
            symbol_download_failed.append(result_symbol)

          else:
            symbol_download_failed.append(result_symbol)

          if error_msg:
              logger.error(f'[{"downloading failed":^19}] : {result_symbol}: {error_msg}')
            
        except Exception as e:
          
          logger.exception(f'[erro] : {futures[future]}: {str(e)}')
        
        if progress_callback:
          progress_callback(len(symbol_downloaded)+len(symbol_up_to_date), len(symbol_download_failed) + len(symbol_not_found), len(download_tasks))
  else:
    pass

  if is_print and len(symbol_up_to_date) > 0:
    logger.warning(f'[{"eod":^19}] - [skip] : {len(symbol_up_to_date)}/{num_symbols} already up-to-date')
  
  # -------------------------------------------------- realtime data ----------------------------------------------#
  # download real-time data
  if update_mode in ['realtime', 'both']:
    for mkt in symbol_class.keys():

      # skip market with an empty list
      if len(symbol_class[mkt]) == 0:
        continue
      
      # print(f'[{"realtime data":^19}] - ')
      tmp_source = sources[f'{mkt}_realtime']
      tmp_sub_source = 'hkquote' if mkt == 'hk' else 'sina'

      # with akshare, real-time data comes with eod data
      if sources[f'{mkt}_eod'] in ['ak'] and update_mode in ['both']:
        logger.info(f'[{"realtime":^19}] - real-time data comes with eod data from ak')

      else:
        query_symbol_list = preprocess_symbol(symbols=symbol_class[mkt], style=tmp_source)
        real_time_data = get_real_time_data(symbols=query_symbol_list, source=tmp_source, sub_source=tmp_sub_source, api_key=benchmark_api_keys[mkt], batch_size=batch_size, is_print=is_print)

        # skip current market when real-time data is not available
        if real_time_data is None or len(real_time_data) == 0:
          logger.error(f'[{"realtime":^19}] - [erro]: real-time data is empty for {mkt} market, skip')
          continue
        if tmp_source == 'eod':
          real_time_data = util.df_2_timeseries(df=real_time_data, time_col='Date')
        
        # append it corresponding eod data according to symbols
        for symbol in symbol_class[mkt]:
          # skip symbols whose eod data failed to download
          if symbol not in data:
            logger.error(f'[{"realtime":^19}] - [erro]: eod data not found for {symbol}, skip real-time update')
            continue
          tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
          
          if len(tmp_data) == 0:
            logger.warning(f'real-time data not found for {symbol}')
            continue

          else:
            tmp_idx = tmp_data.index.max()
            for col in data[symbol].columns:
              data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
            data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

  return data

# update stock data from eod 
def update_stock_data_from_eod(symbols: list, stock_data_path: str, file_format: str = '.csv', update_mode: str = 'eod', required_date: str = None, window_size: int = 3, is_print: bool = False, is_return: bool = False, is_save: bool = True, cn_stock: bool = False, api_key: str = default_eod_key, add_dividend: bool = False, add_split: bool = False, batch_size: int = 15) -> dict:
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
  :returns: dict of dataframes with symbol as key
  :raises: none
  """

  # verify update_mode
  if update_mode not in ['realtime', 'eod', 'both', 'refresh']:
    logger.error(f'unknown update mode: {update_mode}')
    return None

  # get the benchmark of eod data
  today = util.time_2_string(datetime.datetime.today().date())
  start_date = util.string_plus_day(today, -7)
  benchmark_symbol = 'AAPL' if not cn_stock else '000001.SHE'
  benchmark_data = get_data_from_eod(symbol=benchmark_symbol, start_date=start_date, end_date=today, interval='d', is_print=False, api_key=api_key, add_dividend=False, add_split=False)
  # 基准下载失败时 get_data_from_eod 返回 None/空表 -> 原来直接 .index.max() 会 AttributeError,
  # 让整个更新函数崩掉(ak 分支已有兜底, 这里补齐). 回退到今天 = 尽量去下最新数据.
  if benchmark_data is None or len(benchmark_data) == 0:
    logger.warning(f'[-{benchmark_symbol}]: benchmark data is empty, use today - {today}')
    benchmark_date = today
  else:
    benchmark_date = util.time_2_string(benchmark_data.index.max())
  start_date = util.string_plus_day(benchmark_date, -window_size)

  # get the existed data and its latest date for each symbols
  stock_data_path = Path(stock_data_path)
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
    symbol_file_name = stock_data_path / f'{file_name}{file_format}'
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
        # 不删文件(删除+下载失败 = 永久丢历史); tmp_data_date=None -> 全量重下
        logger.warning(f'max index of {symbol} is invalid({max_idx}), 将全量重下(保留原文件)')
        tmp_data_date = None

    # update eod data, print updating info
    if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_date):
      if is_print:
        print(f'from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
      
      # download latest data for current symbol
      # 取数窗口: 原实现传的是 `tmp_data_date`(本地最后日期), 而上面用 window_size
      # 算出的 `start_date` **从未被使用** —— 两个后果:
      #   1) 首次下载/本地数据无效(tmp_data_date is None)时会从 2000-01-01 拉全史;
      #   2) 本地已最新时下载窗口**没有重叠**, 于是 _check_overlap_consistency 永远
      #      无从比较(它需要至少一个重叠交易日才能发现复权基准漂移).
      # 现在改成取"本地最后日期"与"benchmark_date - window_size"中**更早**的那个,
      # 保证既有足够重叠可供校验, 又不会无谓地拉全史.
      _cands = [d for d in (tmp_data_date, start_date) if d]
      dl_start = min(_cands) if _cands else start_date
      new_data = get_data_from_eod(symbol, start_date=dl_start, end_date=required_date, interval='d', is_print=is_print, api_key=api_key, add_dividend=add_dividend, add_split=add_split)
    
      # 空返回不得当成"已更新"(否则旧数据被保留却被视为下载成功, universe 静默缩小)
      if new_data is None or len(new_data) == 0:
        logger.error(f'{symbol:5}: eod 下载返回空, 视为更新失败, 保留本地旧数据')
      else:
        # 增量拼接一致性校验(复权基准变化会让历史留下人为台阶)
        _check_overlap_consistency(data[symbol], new_data, symbol)
        # 保留本地已知的分红/拆股真值
        new_data = _merge_dividend_split(data[symbol], new_data)
        # append new data to the origin
        data[symbol] = pd.concat([data[symbol], new_data])
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

        # save data to local csv files
        if is_save:
          save_stock_data(df=data[symbol], file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
    
    else:
      if symbol not in up_to_date_symbols:
        up_to_date_symbols.append(symbol)
      
  num_symbol_up_to_date = len(up_to_date_symbols)
  if num_symbol_up_to_date > 0:
    if is_print:
      logger.warning(f'from {tmp_data_date} ***** - [skip]: <already up-to-date {num_symbol_up_to_date}/{len(symbols)} >')

  # add real-time data when requiring data return and data will NOT be saved
  if update_mode in ['realtime', 'both']:
    logger.info('***************** querying real-time data *****************')

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
        logger.warning(f'real-time data not found for {symbol}')
        continue
      else:
        tmp_idx = tmp_data.index.max()
        for col in data[symbol].columns:
          data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

  # return
  if is_return:
    return data

# update stock data from ak
def update_stock_data_from_ak(symbols: list, stock_data_path: str, file_format: str = '.csv', update_mode: str = 'eod', required_date: str = None, window_size: int = 3, is_print: bool = False, is_return: bool = False, is_save: bool = True, cn_stock: bool = False) -> dict:

  # verify update_mode
  if update_mode not in ['eod', 'refresh', 'both', 'realtime']:
    logger.error(f'unknown update mode: {update_mode}')
    return None

  # get the benchmark of eod data
  current_weekday = datetime.datetime.now().weekday()
  current_hour = datetime.datetime.now().hour
  today = util.time_2_string(datetime.datetime.today().date())
  start_date = util.string_plus_day(today, -7)
  benchmark_symbol = '105.AAPL' if not cn_stock else '000001'

  # benchmark download may fail(ak raises on network errors), fall back to today
  benchmark_data = None
  try:
    benchmark_data = get_data_from_ak(symbol=benchmark_symbol, start_date=start_date, end_date=today, interval='daily')
  except Exception as e:
    logger.exception(f'[erro]: failed downloading benchmark data from ak: {e}')
  if benchmark_data is None or len(benchmark_data) == 0:
    logger.error(f'[erro]: benchmark data from ak is empty, fallback to today({today}) as benchmark date')
    benchmark_date = today
  else:
    benchmark_date = util.time_2_string(benchmark_data.index.max())
  start_date = util.string_plus_day(benchmark_date, -window_size)

  # get the existed data and its latest date for each symbols
  data = {}
  up_to_date_symbols = []
  
  # for us stocks
  if not cn_stock:
    symbol_list = get_code_map_from_ak() # ak.stock_us_spot_em()
    # guard None code map(ak download failed and no local cache), us symbols can not be mapped without it
    if symbol_list is None:
      logger.error(f'[erro]: failed getting us code map from ak, aborting update for {symbols}')
      return None
    symbol_list['symbol'] = symbol_list['代码'].apply(lambda x: x.split('.')[1])
    symbol_list = symbol_list.set_index('symbol')
    symbols = symbol_list.loc[symbols, '代码'].to_list()
    
  # # for cn stocks
  # else:
  #   symbols = [x.split('.')[0] for x in symbols]
  stock_data_path = Path(stock_data_path)
  for symbol in symbols:

    # init symbol data and its most recent date
    data[symbol] = pd.DataFrame()
    tmp_data_date = None

    # filename, if local data exists, load existed data, update its most current date
    file_name = symbol.split('.')[0]   
    symbol_file_name = stock_data_path / f'{file_name}{file_format}'
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
        # 不删文件(删除+下载失败 = 永久丢历史); tmp_data_date=None -> 全量重下
        logger.warning(f'max index of {symbol} is invalid({max_idx}), 将全量重下(保留原文件)')
        tmp_data_date = None
       
    # update eod data, print updating info
    if (update_mode in ['eod', 'both', 'refresh']) and (tmp_data_date is None or tmp_data_date < benchmark_date):
      if is_print:
        print(f'from ', end='0000-00-00 ' if tmp_data_date is None else f'{tmp_data_date} ')
      
      # download latest data for current symbol, guard download failure so one symbol won't abort the whole update
      logger.info(file_name)
      new_data = None
      # 同 update_stock_data_from_eod: 取"本地最后日期"与"benchmark_date - window_size"
      # 中更早的那个, 保证有重叠可用于复权基准校验(原实现只用 tmp_data_date, 而上面算出的
      # start_date 从未被使用, 于是本地已最新时没有任何重叠窗口).
      _cands = [d for d in (tmp_data_date, start_date) if d]
      ak_start = min(_cands) if _cands else start_date
      try:
        new_data = get_data_from_ak(file_name, start_date=ak_start, end_date=required_date, interval='daily', is_print=is_print)
      except Exception as e:
        logger.exception(f'[erro]: failed downloading {file_name} from ak: {e}')

      # append new data to the origin
      if new_data is not None and len(new_data) > 0:
        # 增量拼接一致性校验(复权基准变化会让历史留下人为台阶)
        _check_overlap_consistency(data[symbol], new_data, symbol)
        # 保留本地已知的分红/拆股真值
        new_data = _merge_dividend_split(data[symbol], new_data)
        data[symbol] = pd.concat([data[symbol], new_data])
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()

        # save data to local csv files
        # as ak data contains the real-time data
        if is_save:
          if cn_stock:
            if current_weekday in [5, 6] or current_hour >= 15:
              data_to_save = data[symbol]
            else:
              data_to_save = data[symbol][:-1]
          else:
            data_to_save = data[symbol]
          save_stock_data(df=data_to_save, file_path=stock_data_path, file_name=symbol, file_format=file_format, reset_index=True, index=False)
    
    else:
      if symbol not in up_to_date_symbols:
        up_to_date_symbols.append(symbol)
      
  num_symbol_up_to_date = len(up_to_date_symbols)
  if num_symbol_up_to_date > 0:
    if is_print:
      logger.warning(f'from {tmp_data_date} ***** - [skip]: <already up-to-date {num_symbol_up_to_date}/{len(symbols)} >')
      
  # add real-time data when requiring data return and data will NOT be saved
  if update_mode in ['realtime', 'both']:
    logger.info('***************** querying real-time data *****************')

    # get real-time data from easyquotation, convert it into time-series data
    real_time_data = get_real_time_data_from_easyquotation(symbols=symbols)
    # append it corresponding eod data according to symbols
    for symbol in symbols:
      tmp_data = real_time_data.query(f'symbol == "{symbol}"')[data[symbol].columns].copy()
      if len(tmp_data) == 0:
        logger.warning(f'real-time data not found for {symbol}')
        continue
      else:
        tmp_idx = tmp_data.index.max()
        for col in data[symbol].columns:
          data[symbol].loc[tmp_idx, col] = tmp_data.loc[tmp_idx, col]
        data[symbol] = util.remove_duplicated_index(df=data[symbol], keep='last').dropna()    

  # return
  if is_return:
    return data

# update stock data (old, aborted)
def update_stock_data(symbols: list, stock_data_path: str, file_format: str = '.csv', update_mode: str = 'eod', cn_stock: bool = False, required_date: str = None, is_print: bool = False, is_return: bool = False, is_save: bool = True, source: str = 'eod', api_key: str = default_eod_key, add_dividend: bool = False, add_split: bool = False, batch_size: int = 15) -> dict:
  """
  update local stock data

  :param symbols: symbol list
  :param stock_data_path: in where the local stock data files(.csv) are stored
  :param source: data source to download latest stock data
  :param file_format: default is .csv
  :param required_date: if the local data have already meet the required date, it won't be updated
  :param is_print: whether to print info when downloading
  :param is_return: whether to return the updated data
  :param is_save: whether to save the updated data to local files
  :param update_mode: how to update data - realtime/eod/both(eod+realtime)/refresh(delete and redownload)
  :param cn_stock: whether updating chinese stocks
  :returns: dict of dataframes with symbol as key or None
  :raises: none
  """

  result = None

  if source == 'eod':
    result = update_stock_data_from_eod(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, api_key=api_key, add_dividend=add_dividend, add_split=add_split, batch_size=batch_size, update_mode=update_mode, cn_stock=cn_stock)
  elif source == 'ak':
    result = update_stock_data_from_ak(symbols=symbols, stock_data_path=stock_data_path, file_format=file_format, required_date=required_date, is_print=is_print, is_return=is_return, is_save=is_save, update_mode=update_mode, cn_stock=cn_stock)
  else:
    logger.error(f'unknown source: {source}')

  if is_return:
    return result

# save downloaded stock data
def save_stock_data(df: pd.DataFrame, file_path: str, file_name: str, file_format: str = '.csv', reset_index: bool = True, index: bool = False) -> None:
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
  file_name = Path(file_path) / f'{file_name}{file_format}'

  if len(df) > 0:
    # reset index
    if reset_index:
      df = df.sort_index().reset_index()
    # save file
    if file_format == '.csv':
      # 原子写: 先写同目录临时文件再 os.replace.
      # 直接 to_csv 覆盖时, 进程被杀/磁盘满/并发同写会留下**截断的 CSV**, 而截断文件
      # 往往仍能被 pandas 正常解析成"更少行" -> 静默丢历史; 或者抛 ParserError 被
      # 读取端当"文件损坏"删掉 -> 触发全量重下.
      tmp_name = file_name.with_name(f'{file_name.name}.tmp')
      try:
        df.to_csv(tmp_name, index=index)
        os.replace(tmp_name, file_name)
      except Exception:
        if tmp_name.exists():
          try:
            tmp_name.unlink()
          except OSError:
            pass
        raise
    else:
      logger.error(f'Unknown format {file_format}')
  else:
    logger.warning('Empty dataframe to save, skipped')
  
# load data from stock data file (csv file)
def load_stock_data(file_path: str, file_name: str, file_format: str = '.csv', time_col: str = 'Date', standard_columns: bool = False, sort_index: bool = True) -> pd.DataFrame:
  """
  load stock data (dataframe) from .csv file

  :param file_path: to where the file will be save to
  :param file_name: name of the file to save
  :param file_format: default is .csv
  :param time_col: column name of the time col, default is Date
  :param standard_columns: whether to return dataframe with standard columns (OHLCV, Adj Close, Dividend, Split)
  :param sort_index: whether to sort index
  :returns: dataframe of stock data
  :raises: none
  """
  # for chinese stocks
  splited = file_name.split('.')
  if len(splited) ==2 and splited[1] in ['SHG', 'SS', 'SHE', 'SZ']:
    file_name = splited[0]

  # contruct filename
  file_name = Path(file_path) / f'{file_name}{file_format}'
  
  # initialize data
  df = None
  
  # read data from local file
  try:
    # if the file not exists, print information
    if not os.path.exists(file_name):
      logger.warning(f'{file_name} not exists')

    else:
      # load file, a failure here means the file itself is unreadable(corrupted),
      # remove it so it can be re-downloaded later
      try:
        df = pd.read_csv(file_name, encoding='utf8')
      except Exception as e:
        logger.exception(f'{file_name} {e}')
        logger.exception(f'remove unreadable file: {file_name}')
        os.remove(file_name)
        return df

      # process the loaded data, a failure here is a column/format issue,
      # keep the original file and only report the error
      try:
        # transform dataframe to timeseries
        df = util.df_2_timeseries(df=df, time_col=time_col)
        
        # select standard columns, guarded by the standard schema validator at the input boundary
        if standard_columns:
          if validate_standard_dataframe(df=df, source=f'load_stock_data: {file_name}'):
            df = df[STANDARD_DATA_COLUMNS].copy()
          else:
            df = None
        
        # sort index
        if sort_index:
          df.sort_index(inplace=True)

      except Exception as e:
        logger.exception(f'{file_name} {e}')
        df = None

  except Exception as e:
    logger.exception(f'{file_name} {e}')

  return df

# remove stock data file (csv file)
def remove_stock_data(symbol: str, file_path: str, file_format: str = '.csv') -> None:
  '''
  Remove stock data file from drive

  :param symbol: symbol of the stock to download
  :param file_path: path to store the download data
  :param file_format: the format of file that data will be stored in
  :returns: None
  :raises: None
  '''
  # construct filename
  file_name = Path(file_path) / f'{symbol}{file_format}'
  
  # remove file
  try:
    if os.path.exists(file_name):
      os.remove(file_name)
    else:
      logger.warning(f'{file_name} not exists')
      
  except Exception as e:
    logger.exception(f'{symbol} {e}') 

# create weekly data from daily data
def create_week_data(df: pd.DataFrame) -> pd.DataFrame:
  '''
  convert day-interval data into week-interval 

  :param df: day-interval OHLCV data
  :returns: week-interval OHLCV data
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

# create monthly data from daily data
def create_month_data(df: pd.DataFrame) -> pd.DataFrame:
  '''
  convert day-interval data into month-interval 

  :param df: day-interval OHLCV data
  :returns: month-interval OHLCV data
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
    
# switch data interval (daily, weekly, monthly)
def switch_data_interval(df: pd.DataFrame, interval: str) -> pd.DataFrame:
  '''
  convert day-interval data into week-interval or month-interval data

  :param df: day-interval OHLCV data
  :param interval: interval of target data week/month
  :returns: converted data with specified interval
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
      logger.error(f'unknown interval {interval}')
      
  return result


#----------------------------- NYTimes Data -------------------------------------#

def download_nytimes(year: int, month: int, api_key: str, file_path: str, file_format: str = '.json', is_print: bool = False, is_return: bool = False) -> dict:
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
  file_name = Path(file_path) / f'{year}-{month:02}{file_format}'

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
    logger.exception(e)
    logger.exception(data)
    pass

  # print info
  if is_print:
    logger.info(f"Finished downloading {year}/{month} ({len(docs)}hints)")

  # return data
  if is_return:
    return data


def read_nytimes(year: int, month: int, file_path: str, file_format: str = '.json') -> pd.DataFrame:
  """
  read nytimes files into dataframe

  :param year: year to read
  :param month: month to read
  :param file_path: where to read the file
  :param file_format: what is the file format of the file
  :returns: dataframe
  :raises: None
  """
  # construct file_name
  file_name = Path(file_path) / f'{year}-{month:02}{file_format}'
  
  # load json data
  with open(file_name) as data_file:    
    NYTimes_data = json.load(data_file)
  
  # convert json to dataframe
  df = pd.DataFrame()  
  df['News'] = None
  num_hits = NYTimes_data['response']['meta']['hits']
  logger.info(f'读取 {year}/{month} 新闻, {num_hits}条')

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
def process_futu_exported(file_path: str, file_name: str) -> pd.DataFrame:
  """
  process stock info that exported from futu-niuniu

  :param file_path: file path
  :param file_name: file name (.csv file)
  :returns: dataframe
  :raises: None
  """
  # load futu exported excel file 
  universe = pd.read_csv(Path(file_path) / file_name, dtype={'代码': str})
  
  # categorize columns
  id_columns = ['代码', '名称', '所属行业']
  basic_columns = ['今开', '昨收', '最高', '最低', '最新价', '成交量', '综合热度']
  rate_change_columns = ['换手率', '涨跌速率', '振幅', '涨跌幅', '5日涨跌幅', '10日涨跌幅', '20日涨跌幅', '60日涨跌幅', '120日涨跌幅', '250日涨跌幅', '年初至今']
  volume_columns = ['成交额', '总市值', '流通值', '总股本']
  fundamental_columns = ['股息率TTM', '市盈率TTM', '市盈率(静)', '市净率']

  # turn prices to numbers
  for col in basic_columns:
    universe[col] = universe[col].replace('-', '0')
    universe[col] = universe[col].astype(float)

  # # turn characters to numbers
  # unit = {'万': 10000, '亿': 100000000}
  # for col in volume_columns:
  #   if universe[col].dtype == str:
  #     universe[col] = universe[col].replace('-', '0')
  #     universe[col] = universe[col].apply(lambda x: (float(x[:-1]) *unit[x[-1]]) if not x.isdigit() else float(x))
    
  # turn '%' to numbers
  for col in rate_change_columns:
    universe[col] = universe[col].replace('-', '0')
    universe[col] = universe[col].apply(lambda x: float(x.replace('%', ''))/100)  
  universe['股息率TTM'] = universe['股息率TTM'].apply(lambda x: float(x.replace('%', ''))/100)

  # process fundamental columns
  for col in ['市盈率TTM', '市盈率(静)', '市净率']:
    universe[col] = universe[col].replace('亏损', '-1')
    universe[col] = universe[col].replace('-', '0')
    universe[col] = universe[col].astype(float)
    universe['流通率'] = (universe['流通股'] / universe['总股本']).round(2)
  
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
    '代码': 'symbol', '名称': 'name', '所属行业': 'category', '综合热度': 'hotness',
    '换手率': 'turnover', '振幅': 'close_range', '今开': 'open', '昨收': 'close_p1', '最高': 'high', '最低': 'low', '最新价': 'close', 
    '涨跌幅': 'close_rate', '5日涨跌幅': 'rate_5d', '10日涨跌幅': 'rate_10d', '20日涨跌幅': 'rate_20d', '60日涨跌幅': 'rate_60d', '120日涨跌幅':'rate_120d', '250日涨跌幅':'rate_250d', '年初至今': 'rate_this_year',
    '成交量': 'volume', '成交额': 'volume_value', '总市值': 'market_value', '流通值': 'market_value_circulation', '总股本': 'stock_total', '流通股': 'stock_circulation',
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


def filter_futu_exported(df: pd.DataFrame, condition: dict = None, q: float = 0.7, price_limit: list = [5, 1000], market: str = 'us') -> pd.DataFrame:
  """
  filter symbols from data that exported from futu

  :param df: dataframe of stock information
  :param condition: dict of filter conditions
  :param q: quantile for volume thresholds
  :param price_limit: price range limit
  :param market: market type (us/a)
  :returns: dataframe
  :raises: None
  """

  filtered_df = df
  # volume_threshold = df.volume.quantile(q)
  # volume_value_threshold = df.volume_value.quantile(q)
  # market_value_threshold = df.market_value.quantile(q)
  market_value_threshold = 500000000

  # drop abnormal symbols
  to_drop = []

  # for us stocks, remove those who's symbol contains '.'
  if market == 'us':
    
    PB_threshold = -1.0
    PE_threshold = -1.0
    filtered_df = filtered_df.query(f'({price_limit[0]} <= close <= {price_limit[1]}) and (type == "company")').copy()
    filtered_df = filtered_df.query(f'(market_value >= {market_value_threshold})').copy() # volume >= {volume_threshold} and volume_value >= {volume_value_threshold} and  and PE >= {PE_threshold} and PB >= {PB_threshold} and rate_5d > 0 and rate_10d > 0
    filtered_df = filtered_df.sort_values('hotness', ascending=False)
    
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
    logger.error(f'market "{market}" not defined')
  
  filtered_df = filtered_df.drop(to_drop)

  # remove duplicated stocks
  filtered_df = filtered_df.loc[~filtered_df['name'].duplicated(),].copy()
    
  return filtered_df


def import_futu_exported(df: pd.DataFrame, num: int = 100) -> dict:
  """
  get list and dict from stock info dataframe, for importing into selected_sec_list.json and ta_config.json

  :param df: dataframe of stock information
  :param num: number of stocks to import
  :returns: dict
  :raises: None
  """
  df = df.head(num).copy()
  df['name'] = df['name'] + '(' + df['category'] + ')'
  codes = df[['symbol', 'name']].set_index('symbol')

  # selected_sec_list
  code_list = codes.index.tolist()
  # code_list.sort()
  # code_list = [x for x in code_list if x not in config['visualization']['plot_args']['sec_name'].keys()]

  # ta_config
  code_names = codes.sort_index().to_dict()['name']
  # to_pop = [x for x in code_names.keys() if x not in code_list]
  # for symbol in to_pop:
  #   code_names.pop(symbol)
    
  return {'selected_sec_list': code_list, 'ta_config': code_names}


#----------------------------- Json File Processing---- -------------------------#
# write json file atomically: dump into a temp file first, then replace the target file,
# so a concurrent reader never sees a half-written(potentially unparsable) file
def _jsonable(obj):
  """
  把 numpy 标量 / NaN / ±inf 递归转成**严格合法 JSON** 的值(NaN/inf -> None).

  为什么必须做: `json.dump` 默认 `allow_nan=True`, 会把 NaN / Infinity **原样写成裸字面量**
  `NaN` / `Infinity` —— 那**不是合法 JSON**(RFC 8259), 严格解析器一律拒绝, 连 Python 自己的
  `json.loads` 之外的工具链(以及任何 `parse_constant` 未放宽的调用)都会失败.
  而真实场景里 NaN 很容易出现: 例如 `position['market_value'].sum()` 在含 NaN 时是 NaN,
  被写进 portfolio.json 后就留下一个"文件看起来在、但读出来是空"的坑
  (read_config 解析失败 -> 调用方回退默认值 -> 持仓/净值快照静默失真).

  :param obj: 任意可 JSON 化结构(dict/list/tuple/标量)
  :returns: 同构结构, 其中 NaN/±inf -> None, numpy 标量 -> Python 标量
  :raises: none
  """
  if isinstance(obj, dict):
    return {k: _jsonable(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_jsonable(v) for v in obj]
  if isinstance(obj, (np.floating, float)):
    f = float(obj)
    return f if np.isfinite(f) else None
  if isinstance(obj, (np.integer,)):
    return int(obj)
  if isinstance(obj, (np.bool_,)):
    return bool(obj)
  # 其它 numpy 标量(如 np.str_)统一转成 Python 原生类型
  if isinstance(obj, np.generic):
    return _jsonable(obj.item())
  return obj


def _save_json_file(config_dict: dict, file_path: str, file_name: str, ensure_ascii: bool = True) -> None:
  target_file = Path(file_path) / file_name
  tmp_file = Path(file_path) / f'{file_name}.tmp'
  try:
    with open(tmp_file, 'w', encoding='UTF-8') as f:
      # allow_nan=False: 万一 _jsonable 漏了什么, 宁可在这里显式报错(文件保持旧的有效版本),
      # 也不要写出一个"看起来写了、其实解析不了"的非法 JSON.
      json.dump(_jsonable(config_dict), f, ensure_ascii=ensure_ascii, allow_nan=False)
    os.replace(tmp_file, target_file)
  except Exception:
    # remove the leftover temp file if anything failed before the replace
    if os.path.exists(tmp_file):
      os.remove(tmp_file)
    raise


def create_config_file(config_dict: dict, file_path: str, file_name: str, print: bool = False, ensure_ascii: bool = True) -> None:
  """
  Create a config file and save global parameters into the file

  :param config_dict: config parameter of keys and values
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param print: whether to print result
  :param ensure_ascii: whether to ensure ascii encoding
  :returns: None
  :raises: save error
  """
  try:
    _save_json_file(config_dict=config_dict, file_path=file_path, file_name=file_name, ensure_ascii=ensure_ascii)
    
    if print:
      logger.info('Config saved successfully')

  except Exception as e:
    logger.exception(e)


def read_config(file_path: str, file_name: str) -> dict:
  """
  Read config from a specific file

  :param file_path: the path to save the file
  :param file_name: the name of the file
  :returns: config dict
  :raises: read error
  """
  try:
    # read existing config
    with open(Path(file_path) / file_name, 'r', encoding='UTF-8') as f:
      config_dict = json.loads(f.read())

  except Exception as e:
    config_dict = {}
    logger.exception(e)

  return config_dict


def add_config(config_key: str, config_value: any, file_path: str, file_name: str, is_print: bool = False) -> None:
  """
  Add a new config in to the config file

  :param config_key: name of the new config
  :param config_value: value of the config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param is_print: whether to print result
  :returns: None
  :raises: save error
  """

  try:
    # read existing config
    new_config = read_config(file_path, file_name)

    # if the file exists but is unreadable(read returns {}), skip to avoid clobbering it with a partial record
    target_file = Path(file_path) / file_name
    if not new_config and os.path.exists(target_file) and os.path.getsize(target_file) > 0:
      logger.warning(f'skip adding config: {target_file} exists but is unreadable')
      return

    new_config[config_key] = config_value
    _save_json_file(config_dict=new_config, file_path=file_path, file_name=file_name)
    if is_print:
      logger.info('Config added successfully')

  except Exception as e:
    logger.exception(e)


def remove_config(config_key: str, file_path: str, file_name: str, is_print: bool = False) -> None:
  """
  remove a config from the config file

  :param config_key: name of the config to remove
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param is_print: whether to print result
  :returns: None
  :raises: save error
  """
  try:
    # read existing config
    new_config = read_config(file_path, file_name)

    # if the file exists but is unreadable(read returns {}), skip to avoid clobbering it with a partial record
    target_file = Path(file_path) / file_name
    if not new_config and os.path.exists(target_file) and os.path.getsize(target_file) > 0:
      logger.warning(f'skip removing config: {target_file} exists but is unreadable')
      return

    new_config.pop(config_key)
    _save_json_file(config_dict=new_config, file_path=file_path, file_name=file_name)
    if is_print:
      logger.info('Config removed successfully')

  except Exception as e:
    logger.exception(e)


def modify_config(config_key: str, config_value: any, file_path: str, file_name: str, is_print: bool = False) -> None:
  """
  modify the value of a config with certain config_key

  :param config_key: name of the config to modify
  :param config_value: new value of the config
  :param file_path: the path to save the file
  :param file_name: the name of the file
  :param is_print: whether to print result
  :returns: None
  :raises: save error
  """
  try:
    # read existing config
    new_config = read_config(file_path, file_name)

    # if the file exists but is unreadable(read returns {}), skip to avoid clobbering it with a partial record
    target_file = Path(file_path) / file_name
    if not new_config and os.path.exists(target_file) and os.path.getsize(target_file) > 0:
      logger.warning(f'skip modifying config: {target_file} exists but is unreadable')
      return

    new_config[config_key] = config_value
    _save_json_file(config_dict=new_config, file_path=file_path, file_name=file_name)
    if is_print:
      logger.info('Config modified successfully')

  except Exception as e:
    logger.exception(e)


#----------------------------- Solidify data ------------------------------------# 
def dict_2_excel(dictionary: dict, file_path: str, file_name: str, keep_index: bool = False) -> None:
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
  writer = pd.ExcelWriter(Path(file_path) / file_name)

  # 写入
  for k in dictionary.keys():
    dictionary[k].to_excel(writer, sheet_name=k, index=keep_index)

  # 关闭文件并保存
  # writer.save()
  writer.close()


def folder_2_zip(folder_path: str, destination_path: str, zip_file_name: str) -> str:
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
  zip_file_name = Path(destination_path) / zip_file_name
  zip_writer = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)

  # zip files in the folder into zip file
  for dir_path, dir_names, file_names in os.walk(start_dir):
    short_path = dir_path.replace(start_dir, '')
    short_path = short_path + os.sep if short_path is not None else ''
    dir_path = Path(dir_path)
    short_path  = Path(short_path)
    for f in file_names:
      zip_writer.write(dir_path / f, short_path / f)
  zip_writer.close()
  
  return zip_file_name


def pickle_dump_data(data: any, file_path: str, file_name: str) -> None:
  """
  pickle data into a file
  :param data: data to dump
  :param file_path: destination file path
  :param file_name: destination file name
  :raises: None
  :returns: None
  """
  file_name = Path(file_path) / file_name
  with open(file_name, 'wb') as f:
    pickle.dump(data, f)


def pickle_load_data(file_path: str, file_name: str) -> any:
  """
  load data from pickled file
  :param file_path: source file path
  :param file_name: source file name
  :raises: None
  :returns: pickled data
  """
  file_name = Path(file_path) / file_name
  data = None
  with open(file_name, 'rb') as f:
    data = pickle.load(f)

  return data


#----------------------------- EOD Fundamentals --------------------------------#
# 优先级1新数据源: 基本面(fundamentals: 财务三表/估值/盈利) + 机构持仓(Holders) +
# 历史市值(historical-market-cap), REST 直连 eodhd, 无需 MCP.
#   - 原始基本面是多层嵌套 JSON, 以 {symbol}.json **原样落盘**(权威副本, 不做有损裁剪);
#     需要用的整洁表(财务三表/持仓/估值快照)由 extract_* 函数**按需**展开, 避免重复存储.
#   - 历史市值是周频时间序列, 复用 save_stock_data/load_stock_data 以 .csv 落盘.
FUNDAMENTAL_DIRNAME = 'fundamental_data'   # 默认落盘根目录(挂在 quant_path 下, 与 stock_data 同级)


def _normalize_us_symbol(symbol: str) -> str:
  """
  Normalize a US ticker to the exchange-suffixed form eod expects.

  eod 的 fundamentals / historical-market-cap 端点: 裸码('AAPL')与带后缀('AAPL.US')都返回 200,
  但带后缀是官方推荐写法. 已带任意交易所后缀(如 '.US'/'.SHE')的原样返回.

  :param symbol: us ticker, e.g. 'AAPL' or 'AAPL.US'
  :returns: exchange-suffixed ticker, e.g. 'AAPL.US'
  :raises: none
  """
  return symbol if '.' in symbol else f'{symbol}.US'


def get_fundamentals_from_eod(symbol: str, api_key: str = default_eod_key, section: str = None, is_print: bool = False) -> dict:
  """
  Download fundamentals(financial statements + valuation + earnings + holders ...) from EOD.

  endpoint: /fundamentals/{symbol}  (full)  or  ?filter={section}  (single section)

  :param symbol: target symbol, e.g. 'AAPL' or 'AAPL.US'
  :param api_key: api token to access eod data
  :param section: None = full payload; otherwise a section name, e.g. 'Holders'/'General'/
                  'Highlights'/'Valuation'/'SharesStats'/'Earnings'/'Financials'
  :param is_print: whether to log download information
  :returns: dict of fundamentals (empty dict on failure)
  :raises: none
  """
  data = {}

  try:
    symbol = _normalize_us_symbol(symbol)
    url = f'https://eodhd.com/api/fundamentals/{symbol}?api_token={api_key}&fmt=json'
    if section:
      url += f'&filter={section}'

    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code == 200:
      data = response.json()
      if is_print:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        logger.info(f'{symbol}: fundamentals({section or "full"}) 下载成功, sections={keys}')
    elif response.status_code == 403:
      # eodhd 对 demo/test token 只开放极少数标的白名单(如 AAPL/MSFT/AMZN/TSLA/MCD),
      # 其余一律 403 'Forbidden'; 同时超额/越权也返回 403. 这里点明, 免得被误当限速重试.
      logger.warning(f'{symbol}: fundamentals({section or "full"}) HTTP 403 Forbidden '
                     f'(token 无该标的权限: demo/test token 仅开放白名单标的, 或额度用尽)')
    else:
      logger.warning(f'{symbol}: fundamentals({section or "full"}) HTTP {response.status_code}')

  except Exception as e:
    logger.exception(f'{symbol} fundamentals: {e}')

  return data


def get_historical_market_cap_from_eod(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  Download historical market capitalization(历史市值) as a weekly timeseries.

  endpoint: /historical-market-cap/{symbol}
  响应是 **以数字字符串为键的 dict**({'0': {'date':..., 'value':...}}), 这里归一化为
  以 Date 为索引、含 'Market Cap' 列的 DataFrame. 该端点仅覆盖 NYSE/NASDAQ 且自 2020 年起.

  :param symbol: target symbol, e.g. 'AAPL' or 'AAPL.US'
  :param start_date: start date of the data, e.g. '2020-01-01'
  :param end_date: end date of the data
  :param api_key: api token to access eod data
  :param is_print: whether to log download information
  :returns: dataframe(DatetimeIndex + 'Market Cap') or empty dataframe
  :raises: none
  """
  df = pd.DataFrame()

  try:
    symbol = _normalize_us_symbol(symbol)
    from_to = ''
    if start_date is not None:
      from_to += f'&from={start_date}'
    if end_date is not None:
      from_to += f'&to={end_date}'

    url = f'https://eodhd.com/api/historical-market-cap/{symbol}?api_token={api_key}&fmt=json{from_to}'
    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code == 200:
      raw = response.json()
      # normalize the digit-string-keyed dict into a list of rows
      if isinstance(raw, dict):
        if len(raw) > 0 and all(str(k).isdigit() for k in raw.keys()):
          rows = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
        else:
          rows = list(raw.values())
      elif isinstance(raw, list):
        rows = raw
      else:
        rows = []

      if len(rows) > 0:
        df = pd.DataFrame(rows)
        df = df.rename(columns={'date': 'Date', 'value': 'Market Cap'})
        df = util.df_2_timeseries(df=df, time_col='Date')
        df = util.remove_duplicated_index(df, keep='last')
        df.sort_index(inplace=True)
        if is_print:
          logger.info(f'{symbol}: historical-market-cap {df.index.min()} - {df.index.max()}, 下载记录 {len(df)}')
      else:
        logger.warning(f'{symbol}: historical-market-cap 空数据')
    elif response.status_code == 403:
      # 与 fundamentals 一致: demo/test token 仅开放白名单标的, 其余 403 'Forbidden'.
      logger.warning(f'{symbol}: historical-market-cap HTTP 403 Forbidden '
                     f'(token 无该标的权限: demo/test token 仅开放白名单标的, 或额度用尽)')
    else:
      logger.warning(f'{symbol}: historical-market-cap HTTP {response.status_code}')

  except Exception as e:
    logger.exception(f'{symbol} historical-market-cap: {e}')

  return df


def save_fundamental_data(data: dict, file_path: str, file_name: str) -> None:
  """
  Save fundamentals / holders JSON(dict) to disk (atomic write).

  :param data: fundamentals dict to save
  :param file_path: destination directory
  :param file_name: destination file name, e.g. 'AAPL.json'
  :returns: none
  :raises: none
  """
  if data is None or len(data) == 0:
    logger.warning('Empty fundamental data to save, skipped')
    return

  try:
    Path(file_path).mkdir(parents=True, exist_ok=True)
    _save_json_file(config_dict=data, file_path=file_path, file_name=file_name, ensure_ascii=False)
  except Exception as e:
    logger.exception(f'{file_path}/{file_name} {e}')


def load_fundamental_data(file_path: str, file_name: str) -> dict:
  """
  Load fundamentals / holders JSON(dict) from disk.

  :param file_path: source directory
  :param file_name: source file name, e.g. 'AAPL.json'
  :returns: fundamentals dict (empty dict when missing / unreadable)
  :raises: none
  """
  data = {}

  try:
    file_name = Path(file_path) / file_name
    if not os.path.exists(file_name):
      logger.warning(f'{file_name} not exists')
    else:
      with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

  except Exception as e:
    logger.exception(f'{file_name} {e}')

  return data


def extract_financial_statements(data: dict) -> dict:
  """
  Flatten the nested Financials(in 三表: Income_Statement/Balance_Sheet/Cash_Flow) into tidy frames.

  每个报表的 yearly/quarterly 是 {报告期日期: {字段...}} 的 dict, 这里转成
  以报告期日期为索引的 DataFrame, 供因子/研究直接使用.

  :param data: full fundamentals dict (as returned by get_fundamentals_from_eod)
  :returns: dict of {f'{statement}_{freq}': DataFrame}, 缺失的报表/频率不出现在结果里
  :raises: none
  """
  fin = (data or {}).get('Financials', {}) or {}
  result = {}

  for statement in ('Income_Statement', 'Balance_Sheet', 'Cash_Flow'):
    block = fin.get(statement, {}) or {}
    for freq in ('yearly', 'quarterly'):
      period = block.get(freq, {}) or {}
      if isinstance(period, dict) and len(period) > 0:
        df = pd.DataFrame.from_dict(period, orient='index')
        if 'date' in df.columns:
          df = df.set_index('date')
        df = df[~df.index.duplicated(keep='last')].sort_index()
        result[f'{statement}_{freq}'] = df

  return result


def extract_fundamental_snapshot(data: dict) -> pd.Series:
  """
  Flatten scalar fields of General/Highlights/Valuation/SharesStats into a single Series.

  用于截面估值/质量因子(如 PERatio / PriceBookMRQ / ReturnOnEquityTTM / MarketCapitalization).
  只保留标量字段, 嵌套结构(如 Address/Listings)一律跳过.

  :param data: full fundamentals dict
  :returns: Series, index 形如 'Highlights.PERatio'
  :raises: none
  """
  snapshot = {}

  for section in ('General', 'Highlights', 'Valuation', 'SharesStats'):
    block = (data or {}).get(section, {}) or {}
    if isinstance(block, dict):
      for key, value in block.items():
        if value is None or isinstance(value, (str, int, float, bool)):
          snapshot[f'{section}.{key}'] = value

  return pd.Series(snapshot)


def extract_holders(data: dict) -> pd.DataFrame:
  """
  Flatten institutional holders(Holders.Institutions) into a tidy frame.

  Holders.Institutions 是 {数字字符串: {持仓字段...}} 的 dict(不是 list), 这里取 value 组成行.

  :param data: full fundamentals dict (or a filtered {'Institutions':...} dict)
  :returns: DataFrame of holders (empty when unavailable)
  :raises: none
  """
  institutions = ((data or {}).get('Holders', {}) or {}).get('Institutions', {}) or {}
  rows = list(institutions.values()) if isinstance(institutions, dict) else list(institutions)
  return pd.DataFrame(rows)


def download_fundamental_data(symbols: list, data_path: str, api_key: str = default_eod_key, with_market_cap: bool = True, overwrite: bool = False, max_workers: int = 5, is_print: bool = False) -> dict:
  """
  Batch download fundamentals(full) + historical market cap and persist them.

  落盘结构(默认 data_path 建议取 quant_path/'fundamental_data'):
    {data_path}/fundamentals/{symbol}.json   # 原始 full fundamentals(含 Financials/Holders/Earnings)
    {data_path}/market_cap/{symbol}.csv      # Date + Market Cap 周频序列(with_market_cap=True)

  已有文件默认跳过(增量); 单只失败只记状态不中断整体.

  :param symbols: list of us tickers, e.g. ['AAPL', 'MSFT']
  :param data_path: destination root directory
  :param api_key: api token to access eod data
  :param with_market_cap: whether to download historical market cap as well
  :param overwrite: whether to re-download symbols whose files already exist
  :param max_workers: thread pool size (mind the 1000 req/min rate limit)
  :param is_print: whether to show progress bar and summary
  :returns: dict of {symbol: {'fundamentals': 'ok'/'skip'/'fail', 'market_cap': ...}}
  :raises: none
  """
  fund_dir = Path(data_path) / 'fundamentals'
  mc_dir = Path(data_path) / 'market_cap'
  fund_dir.mkdir(parents=True, exist_ok=True)
  if with_market_cap:
    mc_dir.mkdir(parents=True, exist_ok=True)

  def _download_one(symbol: str) -> dict:
    result = {'symbol': symbol, 'fundamentals': 'skip', 'market_cap': 'skip'}
    code = _normalize_us_symbol(symbol)

    # fundamentals(full -> includes Financials/Earnings/Holders)
    if overwrite or not (fund_dir / f'{symbol}.json').exists():
      data = get_fundamentals_from_eod(code, api_key=api_key)
      if len(data) > 0:
        save_fundamental_data(data, str(fund_dir), f'{symbol}.json')
        result['fundamentals'] = 'ok'
      else:
        result['fundamentals'] = 'fail'

    # historical market cap(reuse the stock-data csv save/load contract)
    if with_market_cap and (overwrite or not (mc_dir / f'{symbol}.csv').exists()):
      market_cap = get_historical_market_cap_from_eod(code, api_key=api_key)
      if len(market_cap) > 0:
        save_stock_data(market_cap, str(mc_dir), symbol, file_format='.csv')
        result['market_cap'] = 'ok'
      else:
        result['market_cap'] = 'fail'

    return result

  summary = {}
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_download_one, s): s for s in symbols}
    for future in tqdm(as_completed(futures), total=len(symbols), ncols=tqdm_ncols, disable=not is_print):
      try:
        result = future.result()
        summary[result['symbol']] = result
      except Exception as e:
        symbol = futures[future]
        summary[symbol] = {'symbol': symbol, 'error': str(e)}
        logger.exception(f'{symbol} {e}')

  if is_print:
    ok_f = sum(1 for r in summary.values() if r.get('fundamentals') == 'ok')
    ok_m = sum(1 for r in summary.values() if r.get('market_cap') == 'ok')
    logger.info(f'download_fundamental_data: {len(symbols)} 只, fundamentals ok={ok_f}, market_cap ok={ok_m}')

  return summary


#-- EOD New Data --#
# 说明: 本区段对应 eodhd_new_data_sources.md 中列出的"新数据"端点(EODHD 45 feeds 中的增量部分),
# 风格与 get_real_time_data_from_eod() / get_fundamentals_from_eod() 保持一致.
# 统一约定:
#   * 参数 api_key=default_eod_key, is_print=False, 批量类额外 batch_size/page_size;
#   * URL 统一 f'https://eodhd.com/api{path}' + params(api_token & fmt=json + from_to/filter);
#   * requests.get(url, headers=headers, params=..., timeout=...);
#   * status_code==200 正常解析, 403 记 warning(无权限/需 All-in-One), 其它记 warning, 异常 logger.exception;
#   * 返回 pd.DataFrame / dict, 空结果返回 pd.DataFrame() / {}, 绝不返回 None;
#   * 多序列数据集(rates/credit/sanctions/macro/calendar/news/events/lists)返回扁平 DataFrame(仅解析日期列, 不设索引,
#     避免多序列同日塌缩); 真正单标的时序(intraday)才用 util.df_2_timeseries 设索引.
_DATE_COLUMNS = ['date', 'Date', 'as_of', 'as_of_date', 'release_date', 'filing_date',
                 'reportDate', 'report_date', 'disclosure_date', 'transaction_date',
                 'transactionDate', 'declarationDate', 'recordDate', 'paymentDate',
                 'datetime', 'timestamp', 'effective_date', 'last_updated']


def _envelope_to_rows(raw) -> list:
  """
  把 eodhd 各种响应 envelope 统一展平为 list of dict
  :param raw: 原始 json(可能为 list / {'data': [...]} / 单键 section / 普通 dict)
  :returns: list of dict
  """
  if raw is None:
    return []
  if isinstance(raw, list):
    return [x for x in raw if isinstance(x, dict)]
  if isinstance(raw, dict):
    data = raw.get('data')
    if isinstance(data, list):
      return [x for x in data if isinstance(x, dict)]
    # {meta, data: {SYMBOL/KEY: {...}}} 形态(如 /us-quote-delayed), 取每个 value 为一行
    if isinstance(data, dict):
      values = list(data.values())
      if values and all(isinstance(v, dict) for v in values):
        return values
    # 单键 section: 如 {type, description, from, to, earnings:[...]} / {..., splits:[...]} / {..., trends:[...]}
    for value in raw.values():
      if isinstance(value, list):
        return value
    # 单标的时间序列 dict(如 {date: value} 或 {symbol: {...}}), 包成单行
    return [raw]
  return []


def _to_unix_seconds(value):
  """
  把日期串/时间戳统一转成 Unix 秒(供 /intraday 等需要数字时间戳的端点使用)
  :param value: None / int / float / 数字串 / 日期串
  :returns: int 或 None
  """
  if value is None:
    return None
  if isinstance(value, (int, float)):
    return int(value)
  text = str(value)
  if text.isdigit():
    return int(text)
  try:
    return int(pd.Timestamp(text, tz='UTC').timestamp())
  except Exception as e:
    logger.exception(f'_to_unix_seconds {value} {e}')
    return None


def _flatten_nested_rows(rows: list) -> list:
  """
  把形如 {'member': {...}, 'asset': {...}, 'transaction': {...}} 的嵌套行展平为单层 dict
  :param rows: list of dict
  :returns: list of dict
  """
  flat = []
  for row in rows:
    if not isinstance(row, dict):
      continue
    item = {}
    for key, value in row.items():
      if isinstance(value, dict):
        item.update(value)
      else:
        item[key] = value
    flat.append(item)
  return flat


def _parse_date_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
  """
  把常见日期字符串列转成 datetime(不设索引, 避免多序列同日塌缩)
  :param df: 原始 DataFrame
  :param columns: 需要转换的列名, 默认 _DATE_COLUMNS
  :returns: 转换后的 DataFrame
  """
  if df is None or len(df) == 0:
    return df
  cols = columns if columns is not None else _DATE_COLUMNS
  for col in cols:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors='coerce')
  return df


def _get_json_from_eod(path: str, api_key: str = default_eod_key, params: dict = None,
                       timeout: int = 60, is_print: bool = False):
  """
  底层单次 GET: 统一 api_token/fmt=json 拼接与 403/404/异常日志
  :param path: 如 '/ust/yield-rates'(不含域名)
  :param api_key: eod api token
  :param params: 额外 query 参数(dict, 值为 None 的键会被丢弃)
  :param timeout: 超时秒数
  :param is_print: 是否打印
  :returns: 原始 json(list/dict) 或 None
  """
  query = {'api_token': api_key, 'fmt': 'json'}
  if params:
    query.update({k: v for k, v in params.items() if v is not None})
  url = f'https://eodhd.com/api{path}'
  try:
    response = requests.get(url, headers=headers, params=query, timeout=timeout)
  except Exception as e:
    logger.exception(f'{url} {e}')
    return None
  if response.status_code == 200:
    try:
      raw = response.json()
    except Exception as e:
      logger.exception(f'{url} {e}')
      return None
    if is_print:
      logger.info(f'get {path} success')
    return raw
  elif response.status_code == 403:
    logger.warning(f'get {path} failed: 403 forbidden(no permission or require All-in-One subscription)')
    return None
  elif response.status_code == 404:
    logger.warning(f'get {path} failed: 404 not found')
    return None
  else:
    logger.warning(f'get {path} failed: status_code={response.status_code}, text={response.text[:200]}')
    return None


def _get_rows_from_eod(path: str, api_key: str = default_eod_key, params: dict = None,
                       timeout: int = 60, is_print: bool = False) -> list:
  """
  GET 后把响应 envelope 展平为 list of dict
  :returns: list of dict
  """
  return _envelope_to_rows(_get_json_from_eod(path, api_key=api_key, params=params, timeout=timeout, is_print=is_print))


def _get_paged_rows_from_eod(path: str, api_key: str = default_eod_key, params: dict = None,
                             page_size: int = 100, max_pages: int = 500, timeout: int = 60,
                             is_print: bool = False) -> list:
  """
  按 page[offset]/page[limit] 分页拉取
  (实测 eodhd 的 rates/sanctions/credit/real-estate/congressional 统一使用该分页参数;
   offset/limit 会被忽略. 若服务端忽略分页(相邻两页完全相同)则提前停止, 防止死循环)
  :param page_size: 每页条数(一般 max 100, real-estate max 500)
  :param max_pages: 最大页数保护
  :returns: list of dict
  """
  rows = []
  base_params = dict(params) if params else {}
  offset = 0
  last_page = None
  for _ in range(max_pages):
    page_params = dict(base_params)
    page_params['page[offset]'] = offset
    page_params['page[limit]'] = page_size
    page = _get_rows_from_eod(path, api_key=api_key, params=page_params, timeout=timeout, is_print=is_print)
    if not page:
      break
    if page == last_page:
      break
    rows.extend(page)
    if len(page) < page_size:
      break
    last_page = page
    offset += page_size
  if is_print:
    logger.info(f'get {path} total rows={len(rows)}')
  return rows


def _get_ust_rates_from_eod(dataset: str, api_key: str = default_eod_key, start_date: str = None,
                            end_date: str = None, year=None, is_print: bool = False) -> pd.DataFrame:
  """
  美国国债利率家族私有助手
  :param dataset: 'yield-rates' | 'bill-rates' | 'long-term-rates' | 'real-yield-rates'
  :param start_date: 起始日期 YYYY-MM-DD
  :param end_date: 结束日期 YYYY-MM-DD
  :param year: 按年过滤(1900+)
  :returns: pd.DataFrame
  """
  path = f'/ust/{dataset}'
  params = {}
  if year is not None:
    params['filter[year]'] = year
  if start_date:
    params['filter[date_from]'] = start_date
  if end_date:
    params['filter[date_to]'] = end_date
  rows = _get_paged_rows_from_eod(path, api_key=api_key, params=params, page_size=100, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def _get_calendar_from_eod(dataset: str, start_date: str = None, end_date: str = None,
                           symbols=None, api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  通用公司日历私有助手
  :param dataset: 'earnings' | 'splits' | 'ipos'
  :param start_date: 起始日期(from)
  :param end_date: 结束日期(to)
  :param symbols: 标的列表或逗号串(部分日历支持)
  :returns: pd.DataFrame
  """
  path = f'/calendar/{dataset}'
  params = {}
  if start_date:
    params['from'] = start_date
  if end_date:
    params['to'] = end_date
  if symbols:
    params['symbols'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  rows = _get_rows_from_eod(path, api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


# --- Universe / Bulk ---
def get_exchanges_list_from_eod(api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get the list of supported exchanges(direct REST /exchanges-list, 1 call)
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame with columns [Name, Code, OperatingMIC, Country, Currency, CountryISO2, CountryISO3]
  """
  rows = _get_rows_from_eod('/exchanges-list', api_key=api_key, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows)


def get_exchange_symbol_list_from_eod(exchange: str = 'US', delisted: bool = False, type: str = None,
                                      symbols=None, api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get the tickers of one exchange(direct REST /exchange-symbol-list/{exchange}, 1 call)
  :param exchange: exchange code, default 'US'
  :param delisted: whether to include delisted symbols
  :param type: 'common_stock' or 'etf'
  :param symbols: list or comma string, filter by symbols
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame with columns [Code, Name, Country, Exchange, Currency, Type, Isin]
  """
  path = f'/exchange-symbol-list/{exchange}'
  params = {}
  if delisted:
    params['delisted'] = 1
  if type:
    params['type'] = type
  if symbols:
    params['symbols'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  rows = _get_rows_from_eod(path, api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows)


def get_bulk_fundamentals_from_eod(exchange: str = 'US', symbols=None, offset: int = 0, limit: int = 100,
                                   api_key: str = default_eod_key, is_print: bool = False) -> dict:
  """
  get the whole exchange fundamentals snapshot(direct REST /bulk-fundamentals/{exchange}, 100+N calls)
  note: 需要 bulk 权限, 无权限时服务端返回 403, 会记 warning 并返回 {}
  :param exchange: exchange code, default 'US'
  :param symbols: list or comma string, filter by symbols
  :param offset: pagination offset
  :param limit: pagination limit
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: dict {SYMBOL: fundamentals dict}
  """
  path = f'/bulk-fundamentals/{exchange}'
  params = {'offset': offset, 'limit': limit}
  if symbols:
    params['symbols'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  raw = _get_json_from_eod(path, api_key=api_key, params=params, timeout=90, is_print=is_print)
  if not isinstance(raw, dict):
    return {}
  if 'data' in raw and isinstance(raw['data'], dict):
    return raw['data']
  return raw


# --- News / Sentiment ---
def get_news_from_eod(symbols=None, tag: str = None, start_date: str = None, end_date: str = None,
                      limit: int = 50, offset: int = 0, api_key: str = default_eod_key,
                      is_print: bool = False) -> pd.DataFrame:
  """
  get financial news(direct REST /news, 5+5*N calls)
  :param symbols: list or comma string of tickers(s= param)
  :param tag: news tag(t= param)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param limit: page size
  :param offset: pagination offset
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame with columns [date, title, content, link, symbols]
  """
  params = {'limit': limit, 'offset': offset}
  if symbols:
    params['s'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  if tag:
    params['t'] = tag
  if start_date:
    params['from'] = start_date
  if end_date:
    params['to'] = end_date
  rows = _get_rows_from_eod('/news', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_sentiments_from_eod(symbols=None, start_date: str = None, end_date: str = None,
                            api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get aggregated sentiment time series(direct REST /sentiments, 5+5*N calls)
  :param symbols: list or comma string of tickers(s= param, required by api)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame with columns [symbol, date, count, normalized]
  """
  params = {}
  if symbols:
    params['s'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  if start_date:
    params['from'] = start_date
  if end_date:
    params['to'] = end_date
  raw = _get_json_from_eod('/sentiments', api_key=api_key, params=params, is_print=is_print)
  # 响应为 {symbol: [{date, count, normalized}, ...]}
  if not isinstance(raw, dict):
    return pd.DataFrame()
  rows = []
  for symbol, items in raw.items():
    if not isinstance(items, list):
      continue
    for item in items:
      if isinstance(item, dict):
        row = dict(item)
        row['symbol'] = symbol
        rows.append(row)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_news_word_weights_from_eod(symbols, start_date: str = None, end_date: str = None,
                                   api_key: str = default_eod_key, is_print: bool = False) -> dict:
  """
  get news word weights(direct REST /news-word-weights, 5+5*N calls)
  :param symbols: list or comma string of tickers(s= param, required by api)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: dict {symbol: {date: {word: weight}}}
  """
  params = {}
  if symbols:
    params['s'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  if start_date:
    params['filter[date_from]'] = start_date
  if end_date:
    params['filter[date_to]'] = end_date
  raw = _get_json_from_eod('/news-word-weights', api_key=api_key, params=params, timeout=120, is_print=is_print)
  return raw if isinstance(raw, dict) else {}


# --- Smart money / Corporate events ---
def get_insider_transactions_from_eod(code: str = 'US', start_date: str = None, end_date: str = None,
                                      limit: int = 1000, api_key: str = default_eod_key,
                                      is_print: bool = False) -> pd.DataFrame:
  """
  get insider transactions(direct REST /insider-transactions, 1 call)
  :param code: exchange code, default 'US'
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param limit: max records(api max 1000)
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame (PIT 请用 reportDate 作为 avail_date)
  """
  params = {'code': code, 'limit': limit}
  if start_date:
    params['from'] = start_date
  if end_date:
    params['to'] = end_date
  rows = _get_rows_from_eod('/insider-transactions', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_congressional_trades_from_eod(page_size: int = 100, max_pages: int = 500,
                                      api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get congressional trades(direct REST /congressional-trades, 1 call/page)
  note: 分页 page[offset]/page[limit](limit<=100); 展平 member/asset/transaction 嵌套
  :param page_size: records per page(<=100)
  :param max_pages: max pages guard
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame (PIT 请用 disclosure_date; transaction_date 年份有污染)
  """
  rows = _get_paged_rows_from_eod('/congressional-trades', api_key=api_key, params=None,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(_flatten_nested_rows(rows)))


def get_sec_filings_from_eod(symbol: str, form: str = None, limit: int = 20, offset: int = 0,
                             api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get SEC filings(direct REST /sec-filings/{symbol}, 10 calls/req, 需 All-in-One 订阅)
  note: form 走 path(/sec-filings/{symbol}/{form}); 省略 form 返回 overview(data.filings dict)
  :param symbol: ticker(自动补 .US)
  :param form: '10k' | '10q' | '8k' | 'form4' 等, 省略则返回 overview 汇总
  :param limit: records per page(1..100)
  :param offset: pagination offset
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame; overview 返回 [ticker, name, cik, form, count, latest, url]; 指定 form 返回明细
  """
  symbol = _normalize_us_symbol(symbol)
  if form:
    path = f'/sec-filings/{symbol}/{form}'
    rows = _get_rows_from_eod(path, api_key=api_key, params={'limit': limit, 'offset': offset}, is_print=is_print)
    if not rows:
      return pd.DataFrame()
    return _parse_date_columns(pd.DataFrame(rows))
  raw = _get_json_from_eod(f'/sec-filings/{symbol}', api_key=api_key, is_print=is_print)
  data = raw.get('data') if isinstance(raw, dict) else None
  if not isinstance(data, dict) or not isinstance(data.get('filings'), dict):
    return pd.DataFrame()
  rows = []
  for form_key, info in data['filings'].items():
    info = info if isinstance(info, dict) else {}
    rows.append({'ticker': data.get('ticker'), 'name': data.get('name'), 'cik': data.get('cik'),
                 'form': form_key, 'count': info.get('count'), 'latest': info.get('latest'),
                 'url': info.get('url')})
  return pd.DataFrame(rows)


def get_earnings_calendar_from_eod(start_date: str = None, end_date: str = None, symbols=None,
                                   api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get earnings calendar(direct REST /calendar/earnings, 1 call)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param symbols: list or comma string(symbols= param)
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  return _get_calendar_from_eod('earnings', start_date=start_date, end_date=end_date,
                                symbols=symbols, api_key=api_key, is_print=is_print)


def get_earnings_trends_from_eod(symbols, api_key: str = default_eod_key, is_print: bool = False) -> dict:
  """
  get earnings trends(direct REST /calendar/trends, 1 call)
  :param symbols: list or comma string of tickers(symbols= param)
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: dict {type, description, symbols, trends: [...]}
  """
  params = {}
  if symbols:
    params['symbols'] = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  raw = _get_json_from_eod('/calendar/trends', api_key=api_key, params=params, is_print=is_print)
  return raw if isinstance(raw, dict) else {}


def get_upcoming_dividends_from_eod(symbol: str = None, date_eq: str = None, start_date: str = None,
                                    end_date: str = None, api_key: str = default_eod_key,
                                    is_print: bool = False) -> pd.DataFrame:
  """
  get dividends calendar(direct REST /calendar/dividends, 1 call)
  note: api 要求至少提供 symbol 或 date_eq, 否则返回 422
  :param symbol: ticker(自动补 .US)
  :param date_eq: exact date YYYY-MM-DD
  :param start_date: range start(filter[date_from])
  :param end_date: range end(filter[date_to])
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  params = {}
  if symbol:
    params['filter[symbol]'] = _normalize_us_symbol(symbol)
  if date_eq:
    params['filter[date_eq]'] = date_eq
  if start_date:
    params['filter[date_from]'] = start_date
  if end_date:
    params['filter[date_to]'] = end_date
  rows = _get_rows_from_eod('/calendar/dividends', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_upcoming_splits_from_eod(start_date: str = None, end_date: str = None,
                                 api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get splits calendar(direct REST /calendar/splits, 1 call)
  :param start_date: from date YYYY-MM-DD(default api: 未来 7 天)
  :param end_date: to date YYYY-MM-DD
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  return _get_calendar_from_eod('splits', start_date=start_date, end_date=end_date,
                                api_key=api_key, is_print=is_print)


def get_upcoming_ipos_from_eod(start_date: str = None, end_date: str = None,
                               api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get ipos calendar(direct REST /calendar/ipos, 1 call)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  return _get_calendar_from_eod('ipos', start_date=start_date, end_date=end_date,
                                api_key=api_key, is_print=is_print)


# --- Macro / Rates / Credit ---
def get_macro_indicator_from_eod(country: str, indicator: str = 'gdp_current_usd',
                                 api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get country macro indicator(direct REST /macro-indicator/{country}, 1 call)
  :param country: ISO-3 alpha code, e.g. 'USA', 'FRA', 'DEU'
  :param indicator: indicator name, default 'gdp_current_usd'
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame with columns [CountryCode, CountryName, Indicator, Date, Period, Value]
  """
  rows = _get_rows_from_eod(f'/macro-indicator/{country}', api_key=api_key,
                            params={'indicator': indicator}, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_ust_yield_rates_from_eod(start_date: str = None, end_date: str = None, year=None,
                                 api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get US Treasury par yield curve rates(direct REST /ust/yield-rates, 1 call)
  :returns: pd.DataFrame (tenors: 1M..30Y)
  """
  return _get_ust_rates_from_eod('yield-rates', api_key=api_key, start_date=start_date,
                                 end_date=end_date, year=year, is_print=is_print)


def get_ust_bill_rates_from_eod(start_date: str = None, end_date: str = None, year=None,
                                api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get US Treasury bill discount/coupon rates(direct REST /ust/bill-rates, 1 call)
  :returns: pd.DataFrame (tenors: 4WK..52WK)
  """
  return _get_ust_rates_from_eod('bill-rates', api_key=api_key, start_date=start_date,
                                 end_date=end_date, year=year, is_print=is_print)


def get_ust_long_term_rates_from_eod(start_date: str = None, end_date: str = None, year=None,
                                     api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get US Treasury long-term rates(direct REST /ust/long-term-rates, 1 call)
  :returns: pd.DataFrame
  """
  return _get_ust_rates_from_eod('long-term-rates', api_key=api_key, start_date=start_date,
                                 end_date=end_date, year=year, is_print=is_print)


def get_ust_real_yield_rates_from_eod(start_date: str = None, end_date: str = None, year=None,
                                      api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get US Treasury real yield rates(direct REST /ust/real-yield-rates, 1 call)
  :returns: pd.DataFrame
  """
  return _get_ust_rates_from_eod('real-yield-rates', api_key=api_key, start_date=start_date,
                                 end_date=end_date, year=year, is_print=is_print)


def get_reference_rates_from_eod(code: str = None, currency: str = None, start_date: str = None,
                                 end_date: str = None, page_size: int = 100, max_pages: int = 500,
                                 api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get reference rates(direct REST /rates/reference-rates, 1 call/page, paginated)
  :param code: rate code
  :param currency: currency code
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param page_size: records per page(max 100)
  :param max_pages: max pages guard
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  params = {}
  if code:
    params['filter[code]'] = code
  if currency:
    params['filter[currency]'] = currency
  if start_date:
    params['filter[from]'] = start_date
  if end_date:
    params['filter[to]'] = end_date
  rows = _get_paged_rows_from_eod('/rates/reference-rates', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_policy_rates_from_eod(code: str = None, country: str = None, central_bank: str = None,
                              start_date: str = None, end_date: str = None, page_size: int = 100,
                              max_pages: int = 500, api_key: str = default_eod_key,
                              is_print: bool = False) -> pd.DataFrame:
  """
  get central bank policy rates(direct REST /rates/policy-rates, 1 call/page, paginated)
  :param code: rate code
  :param country: country/jurisdiction
  :param central_bank: central bank name
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param page_size: records per page(max 100)
  :param max_pages: max pages guard
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  params = {}
  if code:
    params['filter[code]'] = code
  if country:
    params['filter[country]'] = country
  if central_bank:
    params['filter[central_bank]'] = central_bank
  if start_date:
    params['filter[from]'] = start_date
  if end_date:
    params['filter[to]'] = end_date
  rows = _get_paged_rows_from_eod('/rates/policy-rates', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_funding_stress_from_eod(api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get funding stress indicators(direct REST /spreads/funding-stress, 1 call, not paginated)
  :returns: pd.DataFrame
  """
  rows = _get_rows_from_eod('/spreads/funding-stress', api_key=api_key, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_sovereign_risk_premium_from_eod(country: str = None, region: str = None, as_of: str = None,
                                               page_size: int = 100, max_pages: int = 500,
                                               api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get sovereign country risk premium(Damodaran-style)(direct REST /credit-risk/sovereign/risk-premium, paginated)
  :param country: ISO-3 code or name
  :param region: region filter
  :param as_of: as-of date YYYY-MM-DD
  :param page_size: records per page(max 100)
  :param max_pages: max pages guard
  :returns: pd.DataFrame
  """
  params = {}
  if country:
    params['filter[country]'] = country
  if region:
    params['filter[region]'] = region
  if as_of:
    params['filter[as_of]'] = as_of
  rows = _get_paged_rows_from_eod('/credit-risk/sovereign/risk-premium', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_sovereign_credit_ratings_from_eod(page_size: int = 100, max_pages: int = 500,
                                                 api_key: str = default_eod_key,
                                                 is_print: bool = False) -> pd.DataFrame:
  """
  get sovereign credit ratings(direct REST /credit-risk/sovereign/credit-ratings, paginated)
  :returns: pd.DataFrame
  """
  rows = _get_paged_rows_from_eod('/credit-risk/sovereign/credit-ratings', api_key=api_key, params=None,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_sovereign_cds_spreads_from_eod(page_size: int = 100, max_pages: int = 500,
                                              api_key: str = default_eod_key,
                                              is_print: bool = False) -> pd.DataFrame:
  """
  get sovereign CDS spreads(direct REST /credit-risk/sovereign/cds-spreads, paginated)
  :returns: pd.DataFrame
  """
  rows = _get_paged_rows_from_eod('/credit-risk/sovereign/cds-spreads', api_key=api_key, params=None,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_sovereign_default_spreads_from_eod(page_size: int = 100, max_pages: int = 500,
                                                  api_key: str = default_eod_key,
                                                  is_print: bool = False) -> pd.DataFrame:
  """
  get sovereign default spreads(direct REST /credit-risk/sovereign/default-spreads, paginated)
  :returns: pd.DataFrame
  """
  rows = _get_paged_rows_from_eod('/credit-risk/sovereign/default-spreads', api_key=api_key, params=None,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_corporate_cmdi_from_eod(start_date: str = None, end_date: str = None, page_size: int = 100,
                                       max_pages: int = 500, api_key: str = default_eod_key,
                                       is_print: bool = False) -> pd.DataFrame:
  """
  get corporate market-based default indicator(CMDI)(direct REST /credit-risk/corporate/cmdi, paginated)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :returns: pd.DataFrame (market / IG / HY sub-indices)
  """
  params = {}
  if start_date:
    params['filter[from]'] = start_date
  if end_date:
    params['filter[to]'] = end_date
  rows = _get_paged_rows_from_eod('/credit-risk/corporate/cmdi', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_corporate_hqm_yields_from_eod(start_date: str = None, end_date: str = None, page_size: int = 100,
                                             max_pages: int = 500, api_key: str = default_eod_key,
                                             is_print: bool = False) -> pd.DataFrame:
  """
  get corporate high-quality market(HQM) yields(direct REST /credit-risk/corporate/hqm-yields, paginated)
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :returns: pd.DataFrame
  """
  params = {}
  if start_date:
    params['filter[from]'] = start_date
  if end_date:
    params['filter[to]'] = end_date
  rows = _get_paged_rows_from_eod('/credit-risk/corporate/hqm-yields', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_credit_cds_market_aggregates_from_eod(metric: str = None, dimension: str = None, start_date: str = None,
                                              end_date: str = None, page_size: int = 100, max_pages: int = 500,
                                              api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get aggregated CDS market statistics(direct REST /credit-risk/cds-market-aggregates, paginated)
  :param metric: currently 'gross_notional'
  :param dimension: 'grade' or 'cleared_status'
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :returns: pd.DataFrame
  """
  params = {}
  if metric:
    params['filter[metric]'] = metric
  if dimension:
    params['filter[dimension]'] = dimension
  if start_date:
    params['filter[from]'] = start_date
  if end_date:
    params['filter[to]'] = end_date
  rows = _get_paged_rows_from_eod('/credit-risk/cds-market/aggregates', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_economic_events_from_eod(country: str = None, comparison: str = None, event_type: str = None,
                                 start_date: str = None, end_date: str = None, limit: int = 50, offset: int = 0,
                                 api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get macro economic calendar events(direct REST /economic-events, 1 call)
  note: 该端点用 offset/limit(非 page[]), 返回纯 list
  :param country: ISO-2 country code
  :param comparison: 'mom' | 'qoq' | 'yoy'
  :param event_type: event type filter
  :param start_date: from date YYYY-MM-DD
  :param end_date: to date YYYY-MM-DD
  :param limit: page size
  :param offset: pagination offset
  :returns: pd.DataFrame
  """
  params = {'limit': limit, 'offset': offset}
  if country:
    params['country'] = country
  if comparison:
    params['comparison'] = comparison
  if event_type:
    params['type'] = event_type
  if start_date:
    params['from'] = start_date
  if end_date:
    params['to'] = end_date
  rows = _get_rows_from_eod('/economic-events', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


# --- Index / Auxiliary / Sanctions / Real estate ---
def get_cboe_indices_list_from_eod(api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get the list of supported CBOE indices(direct REST /cboe/indices, 1 call)
  :returns: pd.DataFrame
  """
  rows = _get_rows_from_eod('/cboe/indices', api_key=api_key, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows)


def get_cboe_index_data_from_eod(index_code: str, feed_type: str, date: str,
                                 api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get CBOE index data(direct REST /cboe/index, 10 calls/req)
  note: 三个参数均为必需, 否则返回 422
  :param index_code: e.g. 'VIX', 'BDE30P'
  :param feed_type: e.g. 'snapshot_official_closing'
  :param date: YYYY-MM-DD
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  params = {'filter[index_code]': index_code, 'filter[feed_type]': feed_type, 'filter[date]': date}
  rows = _get_rows_from_eod('/cboe/index', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_search_from_eod(query: str, limit: int = 50, api_key: str = default_eod_key,
                        is_print: bool = False) -> pd.DataFrame:
  """
  search tickers/instruments by keyword(direct REST /search/{query}, 1 call)
  :param query: search keyword
  :param limit: max results
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame with columns [Code, Exchange, Name, Type, Country, Currency, ISIN]
  """
  rows = _get_rows_from_eod(f'/search/{query}', api_key=api_key, params={'limit': limit}, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows)


def get_symbol_change_history_from_eod(start_date: str = None, end_date: str = None,
                                       api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get symbol change history(direct REST /symbol-change-history, 5 calls per ticker, 1 call)
  :param start_date: from date YYYY-MM-DD(filter 可能被忽略)
  :param end_date: to date YYYY-MM-DD
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  params = {}
  if start_date:
    params['from'] = start_date
  if end_date:
    params['to'] = end_date
  rows = _get_rows_from_eod('/symbol-change-history', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_sanctions_entities_from_eod(source: str = None, type: str = None, program: str = None, country: str = None,
                                    q: str = None, active: str = None, page_size: int = 100, max_pages: int = 500,
                                    api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get sanctioned entities(direct REST /sanctions/entities, paginated)
  :param source: sanctions source
  :param type: 'individual' | 'entity' | 'vessel' | 'aircraft'
  :param program: sanctions program
  :param country: country filter
  :param q: keyword(min 2 chars)
  :param active: active filter
  :param page_size: records per page(max 100)
  :param max_pages: max pages guard
  :returns: pd.DataFrame
  """
  params = {}
  if source:
    params['source'] = source
  if type:
    params['type'] = type
  if program:
    params['program'] = program
  if country:
    params['country'] = country
  if q:
    params['q'] = q
  if active:
    params['active'] = active
  rows = _get_paged_rows_from_eod('/sanctions/entities', api_key=api_key, params=params,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_sanctions_vessels_from_eod(page_size: int = 100, max_pages: int = 500,
                                   api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get sanctioned vessels(direct REST /sanctions/vessels, paginated)
  :returns: pd.DataFrame
  """
  rows = _get_paged_rows_from_eod('/sanctions/vessels', api_key=api_key, params=None,
                                  page_size=min(page_size, 100), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_sanctions_programs_from_eod(api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get sanctions programs(direct REST /sanctions/programs, not paginated)
  :returns: pd.DataFrame
  """
  rows = _get_rows_from_eod('/sanctions/programs', api_key=api_key, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows)


def get_sanctions_sources_from_eod(api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get sanctions sources(direct REST /sanctions/sources, not paginated)
  :returns: pd.DataFrame
  """
  rows = _get_rows_from_eod('/sanctions/sources', api_key=api_key, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows)


def get_real_estate_countries_from_eod(page_size: int = 100, max_pages: int = 500,
                                       api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get real estate covered countries(direct REST /real-estate/countries, 5 calls, paginated)
  :returns: pd.DataFrame
  """
  rows = _get_paged_rows_from_eod('/real-estate/countries', api_key=api_key, params=None,
                                  page_size=min(page_size, 500), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_real_estate_from_eod(code: str = 'US', type: str = None, metric: str = None, start_period: str = None,
                             end_period: str = None, page_size: int = 100, max_pages: int = 500,
                             api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get real estate prices(direct REST /real-estate/{code}, 5 calls, paginated)
  :param code: country code, default 'US'
  :param type: data type filter
  :param metric: metric filter
  :param start_period: from period
  :param end_period: to period
  :returns: pd.DataFrame
  """
  params = {}
  if type:
    params['type'] = type
  if metric:
    params['metric'] = metric
  if start_period:
    params['from_period'] = start_period
  if end_period:
    params['to_period'] = end_period
  rows = _get_paged_rows_from_eod(f'/real-estate/{code}', api_key=api_key, params=params,
                                  page_size=min(page_size, 500), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_real_estate_detailed_from_eod(code: str = 'US', type: str = None, metric: str = None, start_period: str = None,
                                      end_period: str = None, page_size: int = 100, max_pages: int = 500,
                                      api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get real estate detailed prices(direct REST /real-estate/{code}/detailed, 5 calls, paginated)
  :returns: pd.DataFrame
  """
  params = {}
  if type:
    params['type'] = type
  if metric:
    params['metric'] = metric
  if start_period:
    params['from_period'] = start_period
  if end_period:
    params['to_period'] = end_period
  rows = _get_paged_rows_from_eod(f'/real-estate/{code}/detailed', api_key=api_key, params=params,
                                  page_size=min(page_size, 500), max_pages=max_pages, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


def get_real_estate_detailed_series_from_eod(code: str = 'US', type: str = None, metric: str = None,
                                             start_period: str = None, end_period: str = None,
                                             api_key: str = default_eod_key, is_print: bool = False) -> pd.DataFrame:
  """
  get real estate detailed series(direct REST /real-estate/{code}/detailed/series, 5 calls, not paginated)
  :returns: pd.DataFrame
  """
  params = {}
  if type:
    params['type'] = type
  if metric:
    params['metric'] = metric
  if start_period:
    params['from_period'] = start_period
  if end_period:
    params['to_period'] = end_period
  rows = _get_rows_from_eod(f'/real-estate/{code}/detailed/series', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


# --- Intraday / Delayed ---
def get_intraday_data_from_eod(symbol: str, interval: str = '5m', start_date: str = None, end_date: str = None,
                               split_dt: bool = False, api_key: str = default_eod_key,
                               is_print: bool = False) -> pd.DataFrame:
  """
  get intraday OHLCV candles(direct REST /intraday/{symbol}, 5 calls)
  note: 单标的时序, 返回后按 datetime 设为时间索引
  :param symbol: ticker(自动补 .US)
  :param interval: '1m' | '5m' | '1h'
  :param start_date: from(日期串或 Unix 秒)
  :param end_date: to(日期串或 Unix 秒)
  :param split_dt: whether to request split date/time fields
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame (index=datetime, columns=Open/High/Low/Close/Volume)
  """
  symbol = _normalize_us_symbol(symbol)
  params = {'interval': interval}
  if start_date:
    params['from'] = _to_unix_seconds(start_date)
  if end_date:
    params['to'] = _to_unix_seconds(end_date)
  if split_dt:
    params['split-dt'] = 1
  rows = _get_rows_from_eod(f'/intraday/{symbol}', api_key=api_key, params=params, is_print=is_print)
  if not rows:
    return pd.DataFrame()
  df = pd.DataFrame(rows)
  df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
  if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = util.df_2_timeseries(df, time_col='datetime')
    df = util.remove_duplicated_index(df, keep='last')
  return df


def get_us_quote_delayed_from_eod(symbols, page_size: int = 100, api_key: str = default_eod_key,
                                  is_print: bool = False) -> pd.DataFrame:
  """
  get extended delayed quotes for US stocks(direct REST /us-quote-delayed, 1 call/ticker)
  :param symbols: list or comma string of US tickers
  :param page_size: records per page(max 100)
  :param api_key: eod api token
  :param is_print: whether to print
  :returns: pd.DataFrame
  """
  if not symbols:
    return pd.DataFrame()
  query = ','.join(symbols) if isinstance(symbols, (list, tuple)) else symbols
  rows = _get_paged_rows_from_eod('/us-quote-delayed', api_key=api_key, params={'s': query},
                                  page_size=min(page_size, 100), is_print=is_print)
  if not rows:
    return pd.DataFrame()
  return _parse_date_columns(pd.DataFrame(rows))


#----------------------------- portfolio updating -------------------------------#
def update_portfolio_support_resistant(config: dict, data: dict, portfolio_file_name: str = 'portfolio.json', is_return: bool = False) -> dict:
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
    if len(data['result'][ti]) > 0:
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
        logger.warning(f'portfolio for {account} not exists')
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
                  logger.warning(f'please update average_cost for {symbol}')
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
            # 原来直接 np.isnan(value): value 来自 portfolio.json 或 ta_result, **可能是字符串**
            # (例如 latest_price 的占位 '--'), np.isnan('--') 会抛 TypeError 让整封邮件崩掉.
            # 改用一个对 str/None/NaN 都安全的判空.
            tmp_support[symbol] = None if _is_missing(support) else support
            tmp_resistant[symbol] = None if _is_missing(resistant) else resistant
            tmp_price[symbol] = tmp_price[symbol] if _is_missing(close) else close

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
def send_result_by_email(config: dict, to_addr: str, from_addr: str = 'northcheng@qq.com', smtp_server: str = 'smtp.qq.com', subject: str = None, platform: list = ['tiger'], signal_file_date: str = None, log_file_date: str = None, test: bool = False, market: str = 'us') -> any:
  """
  send automatic_trader's trading result and technical_analyst's calculation result by email

  :param config: dictionary of config, include pathes, etc.
  :param to_addr: destination email address
  :param from_addr: email address used for sending email
  :param smtp_server: smtp server address
  :param subject: subject of the email
  :param platform: trading platforms include ['tiger', 'futu']
  :param signal_file_date: date of signal file which will be attached on email
  :param log_file_date: date of log file which will be attached on email
  :param test: whether on test mode(print message rather than send the email)
  :param market: market type, 'us' or 'cn'
  :return: smtp ret code
  :raise: none
  """

  # pools for each market
  if market == 'us':
    pools = ['etf_3x', 'company_300'] 
  elif market == 'cn':
    pools = ['a_etf_all', 'hs300']
  else:
    pools = []

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
  if os.path.exists(Path(config['config_path']) / 'portfolio.json'):
    portfolio_record = read_config(file_path=config['config_path'], file_name='portfolio.json')

    # for us_stock, guard None platform records so chained .get() won't crash
    if 'tiger' in platform:
      pr = portfolio_record.get('tiger') or {}
      assets['glob'] = pr.get('global_account')
      # assets['simu'] = pr.get('simulation_account') 

    if 'futu' in platform:
      pr = portfolio_record.get('futu') or {}
      assets['REAL'] = pr.get('REAL')
      # assets['SIMU'] = pr.get('SIMULATE')
    
    # for cn_stock
    pr = portfolio_record.get('pingan') or {}
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
            position = position.replace(l, f'<font color="red">{l}</font>')
          for h in higher_than_resistant:
            position = position.replace(h, f'<font color="green">{h}</font>')

        else:
          position = None

      # add position summary if provided
      position = position if position is not None else ''

    else:
      net_value = '--'
      updated = '--'  
      position = ''

    # mark update-time if not not match with current time
    # guard None updated so slicing won't crash
    if updated is None:
      updated = ''
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
  signal_color = {'b':'green', 'h':'grey'} # , 's':'red'

  if signal_file_date is not None:    
    for p in pools:
      signal_info += f'<li><p>{p}</p></li>'
      tmp_signal_file = Path(config["result_path"]) / f'signal_{p}_{signal_file_date}.xlsx'
      if os.path.exists(tmp_signal_file):
        tmp_signal_list = pd.read_excel(tmp_signal_file, sheet_name='signal', dtype={'symbol': str})
        if len(tmp_signal_list) > 0:
          for s in signal_color.keys():
            font_color = signal_color[s]
            tmp_signals = tmp_signal_list.query(f'signal == "{s}"')['symbol'].astype(str).tolist()            
            signal_info += f'<li>[ <b>{s}</b> ]: <font color="{font_color}">{", ".join(tmp_signals)}</font></li>'

      else:
        # 原来写的是 {prefix}{signal_file_date}.xlsx —— prefix 在本函数里未定义,
        # 只要某个池的信号文件缺失(正是最需要发告警邮件的时候)就抛 NameError,
        # 邮件(唯一主告警通道)整封发不出去. 这里同时修掉未定义变量与文件名口径不一致.
        signal_info += f'<li><p>[Not Found]: {tmp_signal_file.name}</p></li>'
  
  else:
    signal_info += '<li><p>[Not Required]</p></li>'
    summary_info += '<li><p>[Not Required]</p></li>'
  signal_info += '</ul>'
  summary_info += '</ul>'

  # attachment 1: log file
  log_info = '<h3>Log</h3><ul>'
  if log_file_date is not None:

    log_file = config["log_path"] / f'automatic_trade_log_{log_file_date}.txt'
    
    if os.path.exists(log_file):
      log_part = MIMEApplication(open(log_file, 'rb').read())
      log_part.add_header('Content-Disposition', 'attachment', filename=f'automatic_trade_log_{log_file_date}.txt')
      log_info += f'<li><p>[Attached]</p></li>'
    
    else:
      log_info += f'<li><p>[Not Found]: automatic_trade_log_{log_file_date}.txt</p></li>'
      log_part = None
  
  else:
    log_info += '<li><p>[Not Required]</p></li>'
    log_part = None
  log_info += '</ul>'
  
  # attachment 2: images in pdf files(signal/portfolio/index)
  image_info = f'<h3> Images</h3><ul>'
  pdfs = []
  if signal_file_date is not None:

    # initialize header, attach pdfs
    image_info += f'<li>[Requested]: {signal_file_date}</li>'
    pdf_names = [f'{p}_signal' for p in pools]
    pdf_names.append('portfolio')
    
    for pdf in pdf_names:

      # consstruct pdf file path
      tmp_pdf = config["result_path"] / f'{pdf}.pdf'

      # if pdf file exists, check its create date and attach its content
      if os.path.exists(tmp_pdf):
        tmp_pdf_modify_date = util.timestamp_2_time(timestamp=os.path.getmtime(tmp_pdf), unit='s').date().strftime(format='%Y-%m-%d')
        
        # if it is not up-to-date, notify in image_info
        if tmp_pdf_modify_date < signal_file_date:
          image_info += f'<li>[{pdf}.pdf]: {tmp_pdf_modify_date}</li>'
        
        # other wise attach it
        else:
          with open(tmp_pdf, 'rb') as fp:
            tmp_pdf_content = MIMEBase('application', "octet-stream")
            tmp_pdf_content.set_payload(fp.read())
            tmp_pdf_content.add_header('Content-Disposition', 'attachment', filename=f'{pdf}_{tmp_pdf_modify_date}.pdf')
            encoders.encode_base64(tmp_pdf_content)
          pdfs.append(tmp_pdf_content)

      else:        
        image_info += f'<li><p>[{pdf}.pdf]: Not Found</p></li>'
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
  try:
    if test:
      logger.info(full_info)
      ret = 'test'
    else:
      # start SMTP service, send email, stop SMTP service
      server = smtplib.SMTP_SSL(smtp_server)
      # Use port from config if available, otherwise default to 465
      port = config.get('email_port', 465)
      server.connect(smtp_server, port)
      server.login(from_addr, config['api_key']['qq'])
      server.sendmail(from_addr, to_addr, m.as_string())
      ret = server.quit()
  except Exception as e:
    logger.exception(f"Error sending email: {str(e)}")
    ret = f"Error: {str(e)}"

  return ret