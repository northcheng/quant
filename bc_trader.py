import math
import logging
import pytz
import time
import datetime
import pandas as pd

from futu import *
import datetime
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.common.consts import (Language,  Market, BarPeriod, QuoteRight) # 语言, 市场, k线周期, 复权类型
from tigeropen.common.util.contract_utils import (stock_contract, option_contract, future_contract) # 股票合约, 期权合约, 期货合约
from tigeropen.common.util.order_utils import (market_order, limit_order, stop_order, stop_limit_order, trail_order, order_leg) # 市价单, 限价单, 止损单, 限价止损单, 移动止损单, 附加订单


DEFAULT_USER_INFO_FILENAME = 'user_info.json'

class Trader(object):
  
  # init
  def __init__(self, platform, account_type, config, logger_name=None):
    
    self.quote_client = None
    self.trade_client = None
    
    # get logger
    logger_name = f'{platform}_{account_type}' if logger_name is None else logger_name
    logger_level = logging.INFO
    self.logger = logging.getLogger(logger_name)
    self.logger.setLevel(level = logger_level)
    
    logger_console_format = logging.Formatter('[%(asctime)s] - %(message)s ', '%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logger_console_format)
    handlers = self.logger.handlers
    for hdlr in handlers:
      self.logger.removeHandler(hdlr)
    self.logger.addHandler(console)
     
    # read user i nfo, position record from local files
    self.user_info = io_util.read_config(file_path=config['trader_path'], file_name=DEFAULT_USER_INFO_FILENAME).get(platform)
    self.position_record = io_util.read_config(file_path=config['config_path'], file_name=f'{platform}_position_record.json')
    self.record = self.position_record[account_type].copy()
    self.eod_api_key = config['api_key']['eod']
    
    # set account, account type
    self.account = self.user_info[account_type]
    self.account_type = account_type
    
    self.logger.info(f'[{platform}]: instance created: {logger_name}')
  
  # get quote client
  def get_quote_client():
    pass
  
  # close quote client
  def close_trade_client():
    pass
  
  # get trade client
  def get_trade_client():
    pass
  
  # close trade client
  def close_quote_client():
    pass
  

  
class Futu(Trader):
  
  def __init__(self, platform, account_type, config, logger_name=None, market='US', is_encrypt=False):
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)
    
    # get quote and trade context, assets, positions
    self.get_quote_client()
    self.get_trade_client(market=market)
    
    if self.trade_client is not None:
      self.trade_client.unlock_trade(self.user_info['unlock_pwd'])
      ret_positions, self.positions = self.trade_client.position_list_query(trd_env=account_type)
      ret_assets,self.assets = self.trade_client.accinfo_query(trd_env=account_type)
    else:
      self.positions = None
      self.assets = None
      self.logger.error('trade_context not available')
  
  # get quote client
  def get_quote_client(self):
    try:
      host = self.user_info['host']
      port = self.user_info['port']
      is_encrypt = self.user_info['is_encrypt']
      self.quote_client = OpenQuoteContext(host=host, port=port, is_encrypt=is_encrypt)
    except Exception as e:
      self.quote_client = None
      self.logger.exception(f'[erro]: can not create quote context:{e}')
  
  # exit current quote client
  def close_quote_client(self):
    if self.quote_client is not None:
      self.quote_client.close()
      self.quote_client = None
      
  # get trade context
  def get_trade_client(self, market='US'):
    try:
      self.market = market
      host = self.user_info['host']
      port = self.user_info['port']
      is_encrypt = self.user_info['is_encrypt']

      if market == 'US':
        self.trade_client = OpenUSTradeContext(host=host, port=port, is_encrypt=is_encrypt)
      elif market == 'HK':    
        self.trade_client = OpenHKTradeContext(host=host, port=port, is_encrypt=is_encrypt)
      else:
        print(f'Unknown market {market}')
        self.trade_client = None
    except Exception as e:
      self.trade_client = None
      self.logger.exception(f'[erro]: can not create trade context:{e}')
  
  # exit current trade client
  def close_trade_client(self):
    if self.trade_client is not None:
      self.trade_client.close()
      self.trade_client = None
    
  # synchronize position record with real position status
  def synchronize_position_record(self, config):

    account_type = self.account_type
    
    # initialize position record for symbols that not in position record
    init_cash = config['trade']['init_cash'][account_type]
    pool = config['selected_sec_list'][config['trade']['pool'][account_type]]  
    for symbol in pool:
      if symbol not in self.record.keys():
        self.record[symbol] = {'cash': init_cash, 'position': 0}

    # get real position (dict)
    position_dict = dict([(x.contract.symbol, x.quantity) for x in self.positions])

    # compare position record with real position
    record_conflicted = False
    for symbol in self.record.keys():

      if symbol not in pool:
        continue
      
      # update position in record
      record_position = self.record[symbol]['position']
      current_position = 0 if (symbol not in position_dict.keys()) else position_dict[symbol]
      if current_position != record_position:
        record_conflicted = True
        if current_position > 0:
          self.record[symbol] = {'cash': 0, 'position': current_position}
        else:
          self.record[symbol] = {'cash': init_cash, 'position': 0}
        self.logger.error(f'[{account_type[:4]}]: {symbol} position({current_position}) rather than ({record_position}), reset record')

    # add record for position that not recorded
    for symbol in [x for x in position_dict.keys() if (x in pool and x not in self.record.keys())]:
      record_conflicted = True
      self.record[symbol] = {'cash': 0, 'position': position_dict[symbol]}
      self.logger.error(f'[{account_type[:4]}]: {symbol} position({position_dict[symbol]}) not in record, add record')

    # update __position_record
    if record_conflicted:
      self.__position_record[self.account_type] = self.record.copy()
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='tiger_position_record.json')

    
    
    
    
    
class Tiger(Trader):
  
  def __init__(self, platform, account_type, config, logger_name=None, sandbox_debug=False):
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)  
    
    # initialize client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=sandbox_debug)
    self.client_config.private_key = read_private_key(config['trader_path'] + self.user_info['private_key_name'])
    self.client_config.tiger_id = str(self.user_info['tiger_id'])
    self.client_config.language = Language.en_US
    self.client_config.account = self.account 
    
    # get quote/trade clients, assets, positions
    self.get_quote_client()
    self.get_trade_client()
    self.positions = self.trade_client.get_positions(account=self.account)
    self.assets = self.trade_client.get_assets(account=self.account)

    # get market status and trade time
    self.update_trade_time()

    # update position record
    self.synchronize_position_record(config=config)

  
  # get quote client
  def get_quote_client(self):
    self.quote_client = QuoteClient(self.client_config)
    
  # exit current quote client
  def close_quote_client(self):
    if self.quote_client is not None:
      self.quote_client = None
  
  # get trade client
  def get_trade_client(self):
    self.trade_client = TradeClient(self.client_config)
    
  # exit current trade client
  def close_trade_client(self):
    if self.trade_client is not None:
      self.trade_client = None
  
  # update trade time
  def update_trade_time(self, market=Market, tz='Asia/Shanghai', open_time_adj=0, close_time_adj=0):

    # get local timezone
    tz = pytz.timezone(tz)
    
    # get US market trade time
    try:
      # get open_time
      status = self.quote_client.get_market_status(market=Market.US)[0]
      current_status = status.status
      open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
      # open_time = open_time + datetime.timedelta(hours=open_time_adj)

      # if program runs after market open, api will return trade time for next trade day, 
      # trade time for current trade day need to be calculated manually
      if status.status in ['Trading', 'Post-Market Trading']:
        if open_time.weekday() == 0:
          open_time = open_time - datetime.timedelta(days=3)
        else:
          open_time = open_time - datetime.timedelta(days=1)

      # calculate close time, pre_open_time, post_close_time
      close_time = open_time + datetime.timedelta(hours=6.5) #  + close_time_adj
      pre_open_time = open_time - datetime.timedelta(hours=5.5)
      post_close_time = close_time + datetime.timedelta(hours=4)

    except Exception as e:
      self.logger.error(e)
      current_status = None
      open_time = None
      close_time = None
      pre_open_time = None
      post_close_time = None
      
    # get CN market trade time
    try:
      # get open_time
      cn_status = self.quote_client.get_market_status(market=Market.CN)[0]
      cn_current_status = cn_status.status
      cn_open_time = cn_status.open_time.astimezone(tz).replace(tzinfo=None)
      # cn_open_time = cn_open_time + datetime.timedelta(hours=open_time_adj)

      # if program runs after market open, api will return trade time for next trade day, 
      # trade time for current trade day need to be calculated manually
      if status.status in ['Trading', 'Post-Market Trading']:
        if open_time.weekday() == 0:
          cn_open_time = cn_open_time - datetime.timedelta(days=3)
        else:
          cn_open_time = cn_open_time - datetime.timedelta(days=1)

      # calculate close time, pre_open_time, post_close_time
      cn_close_time = cn_open_time + datetime.timedelta(hours=6.5) #  + close_time_adj

    except Exception as e:
      self.logger.error(e)
      cn_current_status = None
      cn_open_time = None
      cn_close_time = None

    self.trade_time = {
      'status': current_status, 'tz': tz,
      'pre_open_time': pre_open_time, 'open_time': open_time,
      'close_time': close_time, 'post_close_time': post_close_time,
      
      'a_status': cn_current_status,
      'a_open_time': cn_open_time, 'a_close_time': cn_close_time
    }
    
