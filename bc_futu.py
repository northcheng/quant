# -*- coding: utf-8 -*-
"""
Utilities used for Futu Open API

:author: Beichen Chen
"""
import math
import logging
import datetime

from quant import bc_util as util
from quant import bc_data_io as io_util
from futu import *


class Futu:


  # get logger
  defualt_logger = logging.getLogger('bc_futu_logger')


  # init
  def __init__(self, account_type, config, market='US', is_encrypt=False, logger_name=None):
        
    # get logger
    self.logger = Futu.defualt_logger if (logger_name is None) else logging.getLogger(logger_name)
        
    # read user info, position record from local files
    self.__user_info = io_util.read_config(file_path=config['futu_path'], file_name='user_info.json')
    self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='futu_position_record.json')
    self.record = self.__position_record[account_type].copy()

    # set account type
    self.account_type = account_type
    
    # get quote and trade context, assets, positions
    self.open_quote_context(host=self.__user_info['host'], port=self.__user_info['port'], is_encrypt=is_encrypt)
    self.open_trade_context(market=market, host=self.__user_info['host'], port=self.__user_info['port'], is_encrypt=is_encrypt)
    if self.trade_context is not None:
      self.trade_context.unlock_trade(self.__user_info['unlock_pwd'])
      ret_positions, self.positions = self.trade_context.position_list_query(trd_env=account_type)
      ret_assets,self.assets = self.trade_context.accinfo_query(trd_env=account_type)
    else:
      self.positions = None
      self.assets = None
      self.logger.error('trade_context not available')
    
    # initialize position record for symbols that not in position record
    init_cash = config['trade']['init_cash'][account_type]
    pool = config['selected_sec_list'][config['trade']['pool'][account_type]]  
    for symbol in pool:
      if symbol not in self.record.keys():
        self.record[symbol] = {'cash': init_cash, 'position': 0}

    # get record position and real position then compare to each other
    record_conflicted = False
    position_dict = dict([(x[0].split('.')[1], x[1]) for x in self.positions[['code', 'qty']].values])
    for symbol in self.record.keys():

      if symbol in ['net_value', 'updated']:
        continue

      record_position = self.record[symbol]['position']
      current_position = 0 if (symbol not in position_dict.keys()) else position_dict[symbol]
      if current_position != record_position:
        record_conflicted = True
        if current_position > 0:
          self.record[symbol] = {'cash': 0, 'position': current_position}
        else:
          self.record[symbol] = {'cash': init_cash, 'position': 0}
        self.logger.error(f'[{account_type[:4]}]: {symbol} position({current_position}) not match with record ({record_position}), reset position record')

    # add record for symbol in position but not in record
    for symbol in [x for x in position_dict.keys() if x not in self.record.keys()]:
      record_conflicted = True
      self.record[symbol] = {'cash': 0, 'position': position_dict[symbol]}
      self.logger.error(f'[{account_type[:4]}]: {symbol} position({position_dict[symbol]}) not in record, reset position record')

    # update __position_record
    if record_conflicted:
      self.__position_record[self.account_type] = self.record.copy()
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='futu_position_record.json')

    self.logger.info(f'[futu]: Futu instance created: {logger_name}')


  # initialize quote context
  def open_quote_context(self, host=None, port=None, is_encrypt=None):
    try:
      host = self.__user_info['host'] if host is None else host
      port = self.__user_info['port'] if port is None else port
      is_encrypt = self.__user_info['is_encrypt'] if is_encrypt is None else is_encrypt
      self.quote_context = OpenQuoteContext(host=host, port=port, is_encrypt=is_encrypt)
    except Exception as e:
      self.quote_context = None
      self.logger.exception(f'[erro]: can not create quote context:{e}')


  # exit current quote context
  def close_quote_context(self):
    if self.quote_context is not None:
      self.quote_context.close()
      self.quote_context = None


  # initialize trade context
  def open_trade_context(self, market='US', host=None, port=None, is_encrypt=None):
    try:
      self.market = market
      host = self.__user_info['host'] if host is None else host
      port = self.__user_info['port'] if port is None else port
      is_encrypt = self.__user_info['is_encrypt'] if is_encrypt is None else is_encrypt

      if market == 'US':
        self.trade_context = OpenUSTradeContext(host=host, port=port, is_encrypt=is_encrypt)
      elif market == 'HK':    
        self.trade_context = OpenHKTradeContext(host=host, port=port, is_encrypt=is_encrypt)
      else:
        print(f'Unknown market {market}')
        self.trade_context = None
    except Exception as e:
      self.trade_context = None
      self.logger.exception(f'[erro]: can not create trade context:{e}')


  # exit current trade context
  def close_trade_context(self):
    if self.trade_context is not None:
      self.trade_context.close()
      self.trade_context = None


  # finalize
  def finalize(self):
    self.close_quote_context()
    self.close_trade_context()
    self.logger.info(f'[fin]: Futu instance finalized: {self.logger.name}')


  # get user info
  def get_user_info(self):
    try:
      ret_acc_list, acc_list = self.trade_context.get_acc_list()
    except Exception as e:
      acc_list = None
      self.logger.exception(f'[erro]: can not get user_info:{e}')

    return acc_list


  # get position record
  def get_position_record(self):
    return self.__position_record


  # update position for an account
  def update_position_record(self, config, init_cash=None, init_position=None, start_time=None, end_time=None):

    # set default values
    init_cash = config['trade']['init_cash'][self.account_type] if (init_cash is None) else init_cash
    init_position = 0 if (init_position is None) else init_position
    start_time = datetime.now().strftime(format="%Y-%m-%d") if (start_time is None) else start_time
    end_time = start_time if (end_time is None) else end_time

    try:
      assets = self.get_asset_summary()
      if len(assets) > 0:
        net_value = assets.loc[0, 'total_assets']
      else:
        net_value = -1
      self.record['net_value'] = net_value

      # get today filled orders
      ret_orders, orders = self.trade_context.history_order_list_query(trd_env=self.account_type, status_filter_list=[OrderStatus.FILLED_PART, OrderStatus.FILLED_ALL], start=start_time, end=end_time)

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
      self.record['updated'] = datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
      self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='futu_position_record.json')
      self.__position_record[self.account_type] = self.record.copy()
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='futu_position_record.json')
      
    except Exception as e:
      self.logger.exception(f'[erro]: fail updating position records for {self.account_type}, {e}')
  

  # get summary of positions
  def get_position_summary(self, get_briefs=False):

    try:
      ret_positions, self.positions = self.trade_context.position_list_query(trd_env=self.account_type)
      result = self.positions.copy()

      if get_briefs and len(result) > 0:
        status = io_util.get_stock_briefs(symbols=[x.split('.')[1] for x in result.code.tolist()], source='yfinance', period='1d', interval='1m')
        status['symbol'] = f'{self.market}.' + status['symbol']
        status.rename(columns={'symbol':'code'}, inplace=True)
        result = pd.merge(result, status, how='left', left_on='code', right_on='code')
        result['rate'] = round((result['latest_price'] - result['cost_price']) / result['cost_price'], 2) * 100
    except Exception as e:
      result = None
      self.logger.exception(f'[erro]: can not get position summary: {e}')

    return result


  # get summary of assets
  def get_asset_summary(self, print_summary=False):
    try:
      ret_assets, self.assets = self.trade_context.accinfo_query(trd_env=self.account_type)
      if print_summary:
        print(self.assets)
    except Exception as e:
      self.assets = None
      self.logger.exception(f'[erro]: can not gett asset summary: {e}')

    return self.assets


  # get available money
  def get_available_cash(self):
    try:
      self.get_asset_summary()
      avalialbe_cash = self.assets.loc[0, 'cash']
    except Exception as e:
      avalialbe_cash = 0
      self.logger.exception(f'[erro]: can not get available cash: {e}')
    
    return avalialbe_cash


  # get quantity of symbol currently in the position
  def get_in_position_quantity(self, symbol, get_briefs=False):

    # initialize affordable quantity
    quantity = 0
    symbol = f'{self.market}.{symbol}'

    # get position summary
    try:
      position = self.get_position_summary(get_briefs=get_briefs)
      if len(position) > 0:
        position = position.set_index('code')
        if symbol in position.index:
          quantity = position.loc[symbol, 'qty']
    except Exception as e:
      self.logger.exception(f'[erro]: can not get in position quantity for {symbol}: {e}')

    return quantity


  # check whether it is affordable to buy certain amount of a stock
  def get_affordable_quantity(self, symbol, cash=None, trading_fee=3):

    # initialize affordable quantity 
    quantity = 0

    try:
      # get available cash
      available_cash = self.get_available_cash() if (cash is None) else cash

      # get latest price of stock
      stock_brief = io_util.get_stock_briefs(symbols=[symbol], source='yfinance', period='1d', interval='1m').set_index('symbol')
      latest_price = stock_brief.loc[symbol, 'latest_price']

      # check if it is affordable
      quantity = math.floor((available_cash-trading_fee)/latest_price)
    except Exception as e:
      self.logger.exception(f'[erro]: can not get affordable quantity: {e}')

    return quantity


  # buy or sell stocks
  def trade(self, symbol, action, quantity, price=None, print_summary=True):

    trade_summary = ''
    try:

      # order type
      if price is None:
        order_type = OrderType.MARKET
        if action == 'BUY':
          price =  0.1
        elif action == 'SELL':
          price = 1000000

      else:
        order_type = OrderType.NORMAL

      # construct trade summary
      trade_summary += f'[{action}]: {symbol} X {quantity} ({order_type}-{price})\t'

      # place buy order if possible
      if action == 'BUY':
        trade_side = TrdSide.BUY
        affordable_quantity = self.get_affordable_quantity(symbol=symbol)
        if quantity <= affordable_quantity:
          ret_place_order, order_info = self.trade_context.place_order(price=price, qty=quantity, code=f'{self.market}.{symbol}', trd_side=trade_side, order_type=order_type, trd_env=self.account_type, remark=None)
          if ret_place_order == RET_OK:
            trade_summary += f'SUCCEED: {order_info.loc[0, "order_id"]}'
          else:
            trade_summary += f'FAILED: {order_info}'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity}/{quantity})'

      # place sell order if holding enough stocks
      elif action == 'SELL':
        trade_side = TrdSide.SELL
        in_position_quantity = self.get_in_position_quantity(symbol)
        if in_position_quantity >= quantity:
          ret_place_order, order_info = self.trade_context.place_order(price=price, qty=quantity, code=f'{self.market}.{symbol}', trd_side=trade_side, order_type=order_type, trd_env=self.account_type, remark=None)
          if ret_place_order == RET_OK:
            trade_summary += f'SUCCEED: {order_info.loc[0, "order_id"]}'
          else:
            trade_summary += f'FAILED: {order_info}'
        else:
          trade_summary += f'FAILED: Not enough stock to sell({in_position_quantity}/{quantity})'

      # other actions
      else:
        trade_summary += f'FAILED: Unknown action {action}'

    except Exception as e:
      trade_summary += f'FAILED: {e}'

    # print trade summary
    if print_summary: 
      self.logger.info(trade_summary)

    return trade_summary


  # auto trade according to signals
  def signal_trade(self, signal, money_per_sec, trading_fee=5, pool=None, according_to_record=True):    
    
    # set symbol to index
    if len(signal) > 0:
      signal = signal.rename(columns={'代码':'symbol', '交易信号':'action'})
      signal = signal.set_index('symbol')

      # filter sec with pool
      if pool is not None:
        filtered_list = [x for x in signal.index if x in pool]
        signal = signal.loc[filtered_list, signal.columns].copy()

    # if signal list is not empty
    if len(signal) > 0:
      # get latest price for signals
      signal_brief = io_util.get_stock_briefs(symbols=signal.index.tolist(), source='yfinance', period='1d', interval='1m').set_index('symbol')
      signal = pd.merge(signal, signal_brief[['latest_price']], how='left', left_index=True, right_index=True)

      # get in-position quantity and latest price for signals
      position = self.get_position_summary(get_briefs=False)
      if len(position) == 0:
        position = pd.DataFrame({'symbol':[], 'quantity':[]})
      else:
        position = position[['code', 'qty']].copy()
        position = position.rename(columns={'code':'symbol', 'qty':'quantity'})

      position['symbol'] = position['symbol'].apply(lambda x: x.split('.')[1])
      position = position.set_index('symbol')
      signal = pd.merge(signal, position[['quantity']], how='left', left_index=True, right_index=True).fillna(0)

      # sell
      # get sell signals
      sell_signal = signal.query('action == "s"')
      if len(sell_signal) > 0:
        # go through sell signals
        for symbol in sell_signal.index:
          # check whether symbol is in positions
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity > 0:
            trade_summary = self.trade(symbol=symbol, action='SELL', quantity=in_position_quantity, price=None, print_summary=False)
            self.logger.info(trade_summary)
          else:
            self.logger.info(f'[SELL]: {symbol} skipped (not in positions)')
      else:
        self.logger.info(f'[SELL]: no signal')

      # buy
      # get buy signals which not in posiitons yet
      default_money_per_sec = money_per_sec
      buy_signal = signal.query('action == "b"')
      if len(buy_signal) > 0:
        # go through buy signals
        for symbol in buy_signal.index:
          # check whether symbol is already in positions
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity == 0:
            # set money used to establish a new position
            if according_to_record:
              if (symbol in self.record.keys()) and (self.record[symbol]['position']==0):
                money_per_sec = self.record[symbol]['cash']
              else:
                money_per_sec = default_money_per_sec

            # check whether there is enough available money 
            available_cash = self.get_available_cash()
            money_per_sec = available_cash if (money_per_sec > available_cash) else money_per_sec

            # calculate quantity to buy
            quantity = math.floor((money_per_sec-trading_fee)/signal.loc[symbol, 'latest_price'])
            if quantity > 0:
              trade_summary = self.trade(symbol=symbol, action='BUY', quantity=quantity, price=None, print_summary=False)
              self.logger.info(trade_summary)
            else:
              self.logger.info(f'[BUY]: not enough money')
              continue
          else:
            self.logger.info(f'[BUY]: {symbol} skipped (already in positions:{in_position_quantity})')
            continue
      else:
       self.logger.info(f'[BUY]: no signal')
    else:
      self.logger.info(f'[SKIP]: no signal')
             

  def cash_out(self, stop_loss_rate=None, stop_profit_rate=None, clear_all=False, print_summary=True):
    
    # get current position with summary
    position = self.get_position_summary()
    
    if len(position) > 0:

      # set symbol as index
      position = position.set_index('code')
      position['pl_ratio'] = position['pl_ratio'] / 100

      # if clear all positions
      if clear_all:
        cash_out_list = position.index.tolist()
      else:
        stop_loss_list = [] if stop_loss_rate is None else position.query(f'pl_ratio < {stop_loss_rate}').index.tolist() 
        stop_profit_list = [] if stop_profit_rate is None else position.query(f'pl_ratio > {stop_profit_rate}').index.tolist() 
        cash_out_list = list(set(stop_loss_list + stop_profit_list))
        
      # cash out
      if len(cash_out_list) > 0:
        cash_out_position =  position.loc[cash_out_list, ].copy()
        for index, row in cash_out_position.iterrows():
          self.trade(symbol=index, action='SELL', quantity=row['can_sell_qty'], print_summary=print_summary)