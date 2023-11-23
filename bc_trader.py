import math
import logging
import pytz
import time
import datetime
import pandas as pd

from futu import (OpenQuoteContext, OpenUSTradeContext, OpenHKTradeContext)
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
    
    # get logger
    logger_level = logging.INFO
    logger_name = f'{platform}_{account_type}' if logger_name is None else logger_name
    logger_format = logging.Formatter('[%(asctime)s] - %(message)s ', '%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logger_level)
    console.setFormatter(logger_format)
    self.logger = logging.getLogger(logger_name)
    self.logger.setLevel(level = logger_level)
    for hdlr in self.logger.handlers:
      self.logger.removeHandler(hdlr)
    self.logger.addHandler(console)
     
    # read user info, position record from local files
    self.user_info = io_util.read_config(file_path=config['trader_path'], file_name=DEFAULT_USER_INFO_FILENAME).get(platform)
    self.position_record = io_util.read_config(file_path=config['config_path'], file_name=f'{platform}_position_record.json')
    self.record = self.position_record[account_type].copy()
    self.eod_api_key = config['api_key']['eod']
    
    # set account, account type
    self.account = self.user_info[account_type]
    self.account_type = account_type
    
    # get quote and trade context, positions, assets
    self.set_client_config()
    self.open_quote_client()
    self.open_trade_client()
    self.update_position()
    self.update_asset()
    
    # update position record
    self.synchronize_position_record(config=config)
    
    self.logger.info(f'[{platform}]: instance created: {logger_name}')
  
  # get client config
  def set_client_config(self):
    pass
  
  # get quote client
  def open_quote_client(self):
    self.quote_client = None
  
  # close quote client
  def close_trade_client(self):
    pass
  
  # get trade client
  def open_trade_client(self):
    self.trade_client = None
  
  # close trade client
  def close_quote_client(self):
    pass
  
  # finalize trader
  def finalize(self):
    pass
  
  # get positions
  def update_position(self):
    self.positions = None
  
  # get assets
  def update_asset(self):
    self.assets = None
    
  def get_available_cash(self):
    
    available_cash = 0
    
    if self.trade_client is None:
      pass
    else:
      self.get_asset_summary()
      if len(self.assets) > 0:
        available_cash = self.assets.loc[0, 'cash']
      else:
        print('Not able to get available cash')
        
    return available_cash
  
  # synchronize position record with real position status
  def synchronize_position_record(self, config):
    
    if self.positions is None:
      pass
    
    else:
      account_type = self.account_type

      # initialize position record for symbols that not in position record
      init_cash = config['trade']['init_cash'][account_type]
      pool = config['selected_sec_list'][config['trade']['pool'][account_type]]  
      for symbol in pool:
        if symbol not in self.record.keys():
          self.record[symbol] = {'cash': init_cash, 'position': 0}

      # get record position and real position then compare to each other
      record_conflicted = False
      
      position_dict = {}
      if len(self.positions) > 0:
        position_dict = dict(self.positions[['symbol', 'quantity']].values)

      # compare position record with real position
      to_reset = []
      for symbol in self.record.keys():

        if symbol not in pool:
          # to_reset.append(symbol)
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
      
      # # reset symbols not in position
      # for symbol in to_reset:
      #   self.record[symbol] = {'cash': init_cash, 'position': 0}

      # add record for position that not recorded
      for symbol in [x for x in position_dict.keys() if (x in pool and x not in self.record.keys())]:
        record_conflicted = True
        self.record[symbol] = {'cash': 0, 'position': position_dict[symbol]}
        self.logger.error(f'[{account_type[:4]}]: {symbol} position({position_dict[symbol]}) not in record, add record')

      # update position_record
      if record_conflicted:
        self.position_record[self.account_type] = self.record.copy()
        io_util.create_config_file(config_dict=self.position_record, file_path=config['config_path'], file_name='tiger_position_record.json')

    
    
  

  
class Futu(Trader):
  
  def __init__(self, platform, account_type, config, logger_name=None):
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)
    
  
  # get quote client
  def open_quote_client(self):
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
  def open_trade_client(self, market='US'):
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

  # finalize trader
  def finalize(self):
    self.close_quote_client()
    self.close_trade_client()
    self.logger.info(f'[fin]: Futu instance finalized: {self.logger.name}')
    
  # get summary of positions
  def update_position(self, get_briefs=False):
    result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'rate_inday':[], 'market_value':[], 'latest_time':[]})
    
    try:
      ret_positions, positions = self.trade_client.position_list_query(trd_env=self.account_type)

      if len(positions) > 0:
        positions.rename(columns={'code':'symbol', 'qty':'quantity', 'cost_price': 'average_cost'}, inplace=True)
        positions['symbol'] = positions['symbol'].apply(lambda x: x.split('.')[1])
        if get_briefs:
          status = io_util.get_stock_briefs(symbols=[positions.symbol.tolist()], source='eod', period='1d', interval='1m', api_key=self.eod_api_key)
          status['symbol'] = f'{self.market}.' + status['symbol']
          positions = pd.merge(positions, status, how='left', left_on='symbol', right_on='symbol')
          positions['rate'] = round((positions['latest_price'] - positions['average_cost']) / positions['average_cost'], 2)
          positions['rate_inday'] = round((positions['Open'] - positions['Close']) / positions['Open'], 2)
          positions['market_value'] = round(positions['latest_price'] * positions['quantity'], 2)
        else:
          positions.rename(columns={'market_val': 'market_value', 'pl_ratio':'rate', 'nominal_price':'latest_price'}, inplace=True)
          positions['rate'] = round(positions['rate'] / 100, 2)
          positions['rate_inday'] = 0
          positions['latest_time'] = None
          
        # select columns
        result = positions[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'rate_inday', 'market_value', 'latest_time']].copy()

    except Exception as e:
      self.logger.exception(f'[erro]: can not get position summary: {e}')

    self.positions = result
    
  # get summary of assets
  def update_asset(self):
    try:
      ret_acc_list, acc_list = self.trade_client.get_acc_list()
      acc_idx = acc_list.query(f'trd_env == "{self.account_type}"').index
      if len(acc_idx) > 0:
        acc_id = acc_list.loc[acc_idx[0], 'acc_id']
      else:
        acc_id = self.account_type
      
      ret_assets, assets = self.trade_client.accinfo_query(trd_env=self.account_type)
      assets['account'] = acc_id
      self.assets = assets[['account', 'total_assets', 'market_val',  'cash', 'avl_withdrawal_cash', 'realized_pl', 'unrealized_pl']].rename(columns={'total_assets':'net_value', 'market_val': 'holding_value', 'avl_withdrawal_cash':'available_cash', 'realized_pl':'pnl', 'unrealized_pl':'holding_pnl'})
      
    except Exception as e:
      self.assets = None
      self.logger.exception(f'[erro]: can not gett asset summary: {e}')
    
    
class Tiger(Trader):
  
  def __init__(self, platform, account_type, config, logger_name=None):
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)  
    
    # get market status and trade time
    self.update_trade_time()

    
  
  # get client config
  def set_client_config(self):
    # initialize client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=self.user_info['sandbox_debug'])
    self.client_config.private_key = read_private_key(config['trader_path'] + self.user_info['private_key_name'])
    self.client_config.tiger_id = str(self.user_info['tiger_id'])
    self.client_config.language = Language.en_US
    self.client_config.account = self.account 
  
  # get quote client
  def open_quote_client(self):
    self.quote_client = QuoteClient(self.client_config)
    
  # exit current quote client
  def close_quote_client(self):
    if self.quote_client is not None:
      self.quote_client = None
  
  # get trade client
  def open_trade_client(self):
    self.trade_client = TradeClient(self.client_config)
    
  # exit current trade client
  def close_trade_client(self):
    if self.trade_client is not None:
      self.trade_client = None
  
  # get summary of positions
  def update_position(self, get_briefs=False):
    
    result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'rate_inday':[], 'market_value':[], 'latest_time':[]})

    try:
      # update positions, convert positions(list) to dataframe
      positions = self.trade_client.get_positions(account=self.client_config.account)
      if len(positions) > 0:
        result = {'symbol': [], 'quantity': [], 'average_cost': [], 'latest_price': []}
        for pos in positions:
          result['symbol'].append(pos.contract.symbol)
          result['quantity'].append(pos.quantity)
          result['average_cost'].append(pos.average_cost)
          result['latest_price'].append(pos.market_price)
        result = pd.DataFrame(result)

        # get briefs for stocks in positions
        if get_briefs:
          status = io_util.get_stock_briefs(symbols=[x.contract.symbol for x in positions], source='eod', period='1d', interval='1m', api_key=self.eod_api_key)
          if status.empty:
            status = pd.DataFrame({'symbol':[]})
          result = pd.merge(result, status, how='left', left_on='symbol', right_on='symbol')
          result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
          result['rate_inday'] = round((result['Open'] - result['Close']) / result['Open'], 2)
          result['market_value'] = round(result['latest_price'] * result['quantity'], 2)
        else:
          result['market_value'] = round(result['latest_price'] * result['quantity'], 2)
          result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
          result['rate_inday'] = 0
          result['latest_time'] = None

        # select columns
        result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'rate_inday', 'market_value', 'latest_time']]

    except Exception as e:
      self.logger.exception(f'[erro]: can not get position summary: {e}')

    self.positions = result
  
  # get summary of assets
  def update_asset(self):

    # update assets
    assets = self.trade_client.get_assets(account=self.client_config.account)
    asset = assets[0]
    result = {
      'account': [asset.account],
      'net_value': [asset.summary.net_liquidation],
      'holding_value': [asset.summary.gross_position_value],
      'cash': [asset.summary.cash],
      'available_cash': [asset.summary.available_funds],
      'pnl': [asset.summary.realized_pnl],
      'holding_pnl': [asset.summary.unrealized_pnl]
    }
    self.assets = pd.DataFrame(result)
  
  
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
    
  
