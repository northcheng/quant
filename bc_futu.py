# -*- coding: utf-8 -*-
"""
Utilities used for Futu Open API

:author: Beichen Chen
"""
import math
import logging

from quant import bc_util as util
from quant import bc_data_io as io_util
from futu import *


class Futu:


  # get logger
  defualt_logger = logging.getLogger('bc_futu_logger')

  # accont_type translate
  account_types = {'simulation_account': 'SIMULATE', 'global_account': 'REAL'}


  # init
  def __init__(self, account_type, config, market='US', host='127.0.0.1', port=11111, is_encrypt=False, logger_name=None):
        
    # get logger
    self.logger = Futu.defualt_logger if (logger_name is None) else logging.getLogger(logger_name)
        
    # read position record from local files
    self.account_type = account_type
    self.trd_env = Futu.account_types[account_type]
    self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='futu_position_record.json')

    # get quote and trade context, assets, positions
    self.open_quote_context(host=host, port=port, is_encrypt=is_encrypt)
    self.open_trade_context(market=market, host=host, port=port, is_encrypt=is_encrypt)
    ret_positions, self.positions = self.trade_context.position_list_query(trd_env=self.trd_env)
    ret_assets,self.assets = self.trade_context.accinfo_query(trd_env=self.trd_env)

    # copy position record for current account
    self.record = self.__position_record[account_type].copy()

    # initialize position record for symbols that not in position record
    init_cash = config['trade']['init_cash'][account_type]
    pool = config['selected_sec_list'][config['trade']['pool'][account_type]]  
    for symbol in pool:
      if symbol not in self.record.keys():
        self.record[symbol] = {'cash': init_cash, 'position': 0}

    # check position record with current positions
    if len(self.positions) > 0:
      code = [x.split('.')[1] for x in self.positions['code'].tolist()]
      qty = self.positions['qty'].tolist()
      position_dict = dict(zip(code, qty))
      
      for symbol in self.record.keys():
        
        # get record position and real position then compare to each other
        record_position = self.record[symbol]['position']
        current_position = 0 if (symbol not in position_dict.keys()) else position_dict[symbol]
        if current_position != record_position:
          if current_position > 0:
            self.record[symbol] = {'cash': 0, 'position': current_position}
          else:
            self.record[symbol] = {'cash': init_cash, 'position': 0}
          self.logger.error(f'[{account_type[:4]}]: {symbol} position({current_position}) not match with record ({record_position}), reset position record')

    self.logger.info(f'[init]: Futu instance created: {logger_name}')


  # initialize quote context
  def open_quote_context(self, host='127.0.0.1', port=11111, is_encrypt=False):
    self.quote_context = OpenQuoteContext(host=host, port=port, is_encrypt=is_encrypt)


  # exit current quote context
  def close_quote_context(self):
    self.quote_context.close()
    self.quote_context = None


  # initialize trade context
  def open_trade_context(self, market='US', host='127.0.0.1', port=11111, is_encrypt=False):
    self.market = market
    if market == 'US':
      self.trade_context = OpenUSTradeContext(host=host, port=port, is_encrypt=is_encrypt)
    elif market == 'HK':    
      self.trade_context = OpenHKTradeContext(host=host, port=port, is_encrypt=is_encrypt)
    else:
      print(f'Unknown market {market}')


  # exit current trade context
  def close_trade_context(self):
    self.trade_context.close()
    self.trade_context = None


  # finalize
  def finalize(self):
    self.close_quote_context()
    self.close_trade_context()


  # get user info
  def get_user_info(self):
    ret_acc_list, acc_list = self.trade_context.get_acc_list()
    return acc_list


  # get position record
  def get_position_record(self):
    return self.__position_record


  # get summary of positions
  def get_position_summary(self, get_briefs=True):

    ret_positions, self.positions = self.trade_context.position_list_query(trd_env=self.trd_env)
    result = self.positions.copy()

    if get_briefs and len(result) > 0:
      status = io_util.get_stock_briefs(symbols=[x.split('.')[1] for x in result.code.tolist()], source='yfinance', period='1d', interval='1m')
      status['symbol'] = f'{self.market}.' + status['symbol']
      result = pd.merge(result, status, how='left', left_on='code', right_on='symbol')
      result['rate'] = round((result['latest_price'] - result['cost_price']) / result['cost_price'], 2)
      result = result[['symbol', 'qty', 'cost_price', 'latest_price', 'rate', 'latest_time']]

    return result


  # get summary of assets
  def get_asset_summary(self, print_summary=False):

    ret_assets, self.assets = self.trade_context.accinfo_query(trd_env=self.trd_env)

    if print_summary:
      print(self.assets)

    return self.assets


  # get available money
  def get_available_cash(self):

    self.get_asset_summary()
    return self.assets.loc[0, 'cash']


  # get quantity of symbol currently in the position
  def get_in_position_quantity(self, symbol, get_briefs=False):

    # initialize affordable quantity
    quantity = 0
    symbol = f'{self.market}.{symbol}'

    # get position summary
    position = self.get_position_summary(get_briefs=get_briefs)
    if len(position) > 0:
      position = position.set_index('code')
      if symbol in position.index:
        quantity = position.loc[symbol, 'qty']

    return quantity


  # check whether it is affordable to buy certain amount of a stock
  def get_affordable_quantity(self, symbol, cash=None, trading_fee=3):

    # initialize affordable quantity and available cash
    quantity = 0
    available_cash = self.get_available_cash() if (cash is None) else cash

    # get latest price of stock
    stock_brief = io_util.get_stock_briefs(symbols=[symbol], source='yfinance', period='1d', interval='1m').set_index('symbol')
    latest_price = stock_brief.loc[symbol, 'latest_price']

    # check if it is affordable
    quantity = math.floor((available_cash-trading_fee)/latest_price)

    return quantity


  # update position for an account
  def update_position_record(self, config, init_cash=None, init_position=None, start_time=None, end_time=None):

    # set default values
    init_cash = config['trade']['init_cash'][self.account_type] if (init_cash is None) else init_cash
    init_position = 0 if (init_position is None) else init_position
    start_time = datetime.now().strftime(format="%Y-%m-%d") if (start_time is None) else start_time
    end_time = start_time if (end_time is None) else end_time

    try:
      # get today filled orders
      ret_orders, orders = self.trade_context.history_order_list_query(trd_env=self.trd_env, status_filter_list=[OrderStatus.FILLED_PART, OrderStatus.FILLED_ALL], start=start_time, end=end_time)

      # update position records
      for index, row in orders.iterrows():
        symbol = row['code'].split('.')[1] #order.contract.symbol
        action = row['trd_side'] #order.action
        quantity = row['dealt_qty'] #order.quantity - order.remaining
        commission = 3 # order.commission
        avg_fill_price = row['dealt_avg_price'] # order.avg_fill_price

        # init record if not exist
        if symbol not in self.record.keys():
          self.record[symbol] = {'cash': init_cash, 'position': init_position}
        
        # calculate new cash and position
        if action == 'BUY':
          cost = avg_fill_price * quantity + commission
          new_cash = self.record[symbol]['cash'] - cost
          new_position = self.record[symbol]['position'] + quantity
        
        elif action == 'SELL':
          acquire = avg_fill_price * quantity - commission
          new_cash = self.record[symbol]['cash'] + acquire
          new_position = self.record[symbol]['position'] - quantity

        else:
          new_cash = self.record[symbol]['cash']
          new_position = self.record[symbol]['position']

        # update record
        if new_cash >= 0 and new_position >= 0:
          self.record[symbol]['cash'] = new_cash
          self.record[symbol]['position'] = new_position

      # update __position_record
      self.__position_record[self.account_type] = self.record.copy()
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='futu_position_record.json')
      
    except Exception as e:
      self.logger.exception(f'[erro]: fail updating position records for {self.account_type}, {e}')
  