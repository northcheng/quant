# -*- coding: utf-8 -*-
"""
Utilities used for Tiger Open API

:author: Beichen Chen
"""
import math
import pytz
import time
import datetime
import logging
import pandas as pd
from quant import bc_util as util
from quant import bc_data_io as io_util
from tigeropen.common.consts import (Language,  Market, BarPeriod, QuoteRight) # 语言, 市场, k线周期, 复权类型
from tigeropen.common.util.order_utils import (market_order, limit_order, stop_order, stop_limit_order, trail_order, order_leg) # 市价单, 限价单, 止损单, 限价止损单, 移动止损单, 附加订单
from tigeropen.common.util.contract_utils import (stock_contract, option_contract, future_contract) # 股票合约, 期权合约, 期货合约
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient

# get logger
defualt_logger = logging.getLogger('bc_tiger_logger')


class Tiger:

  # user info: dict
  __user_info = {}            
  __position_record = {}


  # init
  def __init__(self, account_type, config, sandbox_debug=False, logger_name=None):

    # read user info from local file
    self.__user_info = pd.read_csv(config['tiger_path'] + 'user_info.csv').loc[0,:].astype('str').to_dict()

    # get position record
    self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='tiger_position_record.json')

    # set client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=sandbox_debug)
    self.client_config.private_key = read_private_key(config['tiger_path'] + self.__user_info['private_key_name'])
    self.client_config.language = Language.en_US
    self.client_config.tiger_id = str(self.__user_info['tiger_id'])
    
    # get quote/trade client, assets and positions
    self.update_account(account_type=account_type)

    # get logger
    if logger_name is None:
      self.logger = defualt_logger
    else:
      self.logger = logging.getLogger(logger_name)

    self.logger.info(f'[init]: Tiger instance created: {logger_name}')


  # get quote/trade client, assets and positions for specified account
  def update_account(self, account_type):

    # update config, trade_client, quote_client
    self.client_config.account = str(self.__user_info[account_type])
    self.quote_client = QuoteClient(self.client_config)
    self.trade_client = TradeClient(self.client_config)

    # update asset and position
    self.positions = self.trade_client.get_positions(account=self.__user_info[account_type])
    self.assets = self.trade_client.get_assets(account=self.__user_info[account_type])

    # update position record
    self.record = self.__position_record[account_type]

    # update trade time
    self.get_trade_time()
    

  # get trade time
  def get_trade_time(self, market=Market.US, tz='Asia/Shanghai'):

    # get local timezone
    tz = pytz.timezone(tz)

    try:
      # get open_time
      status = self.quote_client.get_market_status(market=market)[0]
      market_status = status.status
      open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
      if status.status in ['Trading', 'Post-Market Trading']:
        open_time = open_time - datetime.timedelta(days=1)

      # get close time, pre_open_time, post_close_time
      close_time = open_time + datetime.timedelta(hours=6.5)
      pre_open_time = open_time - datetime.timedelta(hours=5.5)
      post_close_time = close_time + datetime.timedelta(hours=4)

    except Exception as e:
      self.logger.error(e)
      open_time = close_time = pre_open_time = post_close_time = market_status = None

    self.trade_time = {
      'status': status.status, 'tz': tz,
      'pre_open_time': pre_open_time, 'open_time': open_time,
      'close_time': close_time, 'post_close_time': post_close_time
    }


  # get summary of positions
  def get_position_summary(self, get_briefs=True):

    # update positions
    self.positions = self.trade_client.get_positions(account=self.client_config.account)

    # convert positions(list) to dataframe
    if len(self.positions) > 0:
      result = {'symbol': [], 'quantity': [], 'average_cost': [], 'market_price': []}
      for pos in self.positions:
        result['symbol'].append(pos.contract.symbol)
        result['quantity'].append(pos.quantity)
        result['average_cost'].append(pos.average_cost)
        result['market_price'].append(pos.market_price)
      result = pd.DataFrame(result)

      # get briefs for stocks in positions
      if get_briefs:
        # status = self.quote_client.get_stock_briefs(symbols=[x.contract.symbol for x in self.positions])

        status = io_util.get_stock_briefs(symbols=[x.contract.symbol for x in self.positions], source='yfinance', period='1d', interval='1m')
        result = pd.merge(result, status, how='left', left_on='symbol', right_on='symbol')
        result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
        result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'latest_time']]

    else:
      result = pd.DataFrame()

    return result


  # get summary of assets
  def get_asset_summary(self, print_summary=False):

    # update assets
    self.assets = self.trade_client.get_assets(account=self.client_config.account)
    asset = self.assets[0]
    result = {
      'account': [asset.account],
      'net_value': [asset.summary.net_liquidation],
      'holding_value': [asset.summary.gross_position_value],
      'cash': [asset.summary.cash],
      'available_casg': [asset.summary.available_funds],
      'pnl': [asset.summary.realized_pnl],
      'holding_pnl': [asset.summary.unrealized_pnl]
    }

    if print_summary:
      summary = f'''
      账户: {asset.account}({asset.summary.currency}):
      总资产： {asset.summary.net_liquidation}
      现金: {asset.summary.cash} (可用 {asset.summary.available_funds})
      持仓市值: {asset.summary.gross_position_value}
      日内交易次数: {asset.summary.day_trades_remaining}
      已实现盈亏: {asset.summary.realized_pnl}
      未实现盈亏: {asset.summary.unrealized_pnl}
      '''
      print(summary)

    return pd.DataFrame(result)


  # get available money
  def get_available_cash(self):

    # get available cash for real accounts
    self.assets = self.trade_client.get_assets(account=self.client_config.account)
    available_cash = self.assets[0].summary.available_funds
    
    # use cash rather than available_funds for simulation account
    if self.client_config.account == self.__user_info['simulation_account']:
      available_cash = self.assets[0].summary.cash

    return available_cash


  # check whether it is affordable to buy certain amount of a stock
  def get_affordable_quantity(self, symbol, cash=None, trading_fee=3):

    # initialize affordable quantity
    quantity = 0
    
    # get available cash
    available_cash = cash
    if cash is None:
      available_cash = self.get_available_cash()

    # get latest price of stock
    # stock_brief = self.quote_client.get_stock_briefs(symbols=[symbol]).set_index('symbol')
    stock_brief = io_util.get_stock_briefs(symbols=[symbol], source='yfinance', period='1d', interval='1m').set_index('symbol')
    latest_price = stock_brief.loc[symbol, 'latest_price']

    # check if it is affordable
    quantity = math.floor((available_cash-trading_fee)/latest_price)

    return quantity


  # get quantity of symbol currently in the position
  def get_in_position_quantity(self, symbol):

    # initialize affordable quantity
    quantity = 0

    # get position summary
    position = self.get_position_summary()
    if len(position) > 0:
      position = position.set_index('symbol')

      # if symbol in position, get the quantity
      if symbol in position.index:
        quantity = position.loc[symbol, 'quantity']

    return quantity


  # buy or sell stocks
  def trade(self, symbol, action, quantity, price=None, stop_loss=None, stop_profit=None, print_summary=True):

    trade_summary = ''
    try:

      # construct contract
      contract = stock_contract(symbol=symbol, currency='USD')

      # construct order
      if price is None:
        order_price = 'market'
        order = market_order(account=self.client_config.account, contract=contract, action=action, quantity=quantity)
      else:
        order_price = f'{price}'
        order = limit_order(account=self.client_config.account, contract=contract, action=action, quantity=quantity, limit_price=price)

      # construct trade summary
      trade_summary += f'[{action}]: {symbol} X {quantity} ({order_price})\t'

      # attach order legs
      order_legs = []
      if stop_loss is not None:
        stop_loss_order_leg = order_leg('LOSS', stop_loss, time_in_force='GTC') # 附加止损单
        order_legs.append(stop_loss_order_leg)
      if stop_profit is not None:
        stop_profit_order_leg = order_leg('PROFIT', stop_profit, time_in_force='GTC') # 附加止盈单
        orderlegs.append(stop_profit_order_leg)
      if len(order_legs)>0:
        order.order_legs = order_legs

      # place buy order if affordable
      if action == 'BUY':
        affordable_quantity = self.get_affordable_quantity(symbol=symbol)
        if quantity <= affordable_quantity:
          self.trade_client.place_order(order)
          trade_summary += f'SUCCEED: {order.id}'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity})'

      # place sell order if holding enough stocks
      elif action == 'SELL':    
        in_position_quantity = self.get_in_position_quantity(symbol)
        if in_position_quantity >= quantity:
          self.trade_client.place_order(order)
          trade_summary += f'SUCCEED: {order.id}'
        else:
          trade_summary += f'FAILED: Not enough stock to sell({in_position_quantity})'

      # other actions
      else:
        trade_summary += f'FAILED: Unknown action({action})'
      
    except Exception as e:
      trade_summary += f'FAILED: {e}'
      
    # print trade summary
    if print_summary: 
      self.logger.info(trade_summary)

    return trade_summary


  # auto trade according to signals
  def signal_trade(self, signal, money_per_sec, trading_fee=3, pool=None, according_to_record=True):

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
      # signal_brief = self.quote_client.get_stock_briefs(symbols=signal.index.tolist()).set_index('symbol')
      signal_brief = io_util.get_stock_briefs(symbols=signal.index.tolist(), source='yfinance', period='1d', interval='1m').set_index('symbol')
      signal = pd.merge(signal, signal_brief[['latest_price']], how='left', left_index=True, right_index=True)

      # get in-position quantity for signals
      position = self.get_position_summary()
      if len(position) == 0:
        position = pd.DataFrame({'symbol':[], 'quantity':[]})
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
              if (symbol in self.record.keys()) and (self.record[symbol]['position']) == 0:
                money_per_sec = self.record[symbol]['cash']
              else:
                money_per_sec = default_money_per_sec

            # check whether there is enough money 
            available_cash = self.get_available_cash()
            if available_cash >= (money_per_sec):
              quantity = math.floor((money_per_sec-trading_fee)/signal.loc[symbol, 'latest_price'])
              trade_summary = self.trade(symbol=symbol, action='BUY', quantity=quantity, price=None, print_summary=False)
              self.logger.info(trade_summary)
            else:
              self.logger.info(f'[BUY]: not enough money')
              break
          else:
            self.logger.info(f'[BUY]: {symbol} skipped (already in positions:{in_position_quantity})')
            continue
      else:
       self.logger.info(f'[BUY]: no signal')
    else:
      self.logger.info(f'[SKIP]: no signal')
          

  # stop loss or stop profit or clear all positions
  def cash_out(self, stop_loss_rate=None, stop_profit_rate=None, clear_all=False, print_summary=True):

    # get current position with summary
    position = self.get_position_summary(get_briefs=True)
    if len(position) > 0:

      # set symbol as index
      position = position.set_index('symbol')

      # if clear all positions
      if clear_all:
        cash_out_list = position.index.tolist()
      else:
        stop_loss_list = [] if stop_loss_rate is None else position.query(f'rate < {stop_loss_rate}').index.tolist() 
        stop_profit_list = [] if stop_profit_rate is None else position.query(f'rate > {stop_profit_rate}').index.tolist() 
        cash_out_list = list(set(stop_loss_list + stop_profit_list))
        
      # cash out
      if len(cash_out_list) > 0:
        cash_out_position =  position.loc[cash_out_list, ].copy()
        for index, row in cash_out_position.iterrows():
          self.trade(symbol=index, action='SELL', quantity=row['quantity'], print_summary=print_summary)
    #   else:
    #     print('empty cash out list')

    # else:
    #   print('empty position')      


  # update position for an account
  def update_position_record(self, account_type, config, init_cash=None, init_position=None, start_time=None, end_time=None):

    if init_cash is None:
      init_cash = config['trade']['init_cash'][account_type]
    if init_position is None:
      init_position = 0

    try:
      # get today filled orders
      if start_time is None:
        start_time = self.trade_time['pre_open_time'].strftime(format="%Y-%m-%d %H:%M:%S")
      if end_time is None:
        end_time = self.trade_time['post_close_time'].strftime(format="%Y-%m-%d %H:%M:%S")
      orders = self.trade_client.get_filled_orders(start_time=start_time, end_time=end_time)

      # update position records
      for order in orders:
        symbol = order.contract.symbol
        action = order.action
        
        avg_fill_price = order.avg_fill_price
        quantity = order.quantity - order.remaining
        commission = order.commission
        
        if symbol not in self.record.keys():
          self.record[symbol] = {'cash': init_cash, 'position': init_position}
        
        if action == 'BUY':
          cost = avg_fill_price * quantity + commission
          self.record[symbol]['cash'] -= cost
          self.record[symbol]['position'] += quantity
          
        if action == 'SELL':
          acquire = avg_fill_price * quantity - commission
          self.record[symbol]['cash'] += acquire
          self.record[symbol]['position'] -= quantity

      # update __position_record
      self.__position_record[account_type] = self.record

      # save updated config record
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='tiger_position_record.json')
      
    except Exception as e:
      self.logger.exception(f'[erro]: fail updating position records for {account_type}')
  

  # idle for specified time and check position in certain frequency
  def idle(self, target_time, check_frequency=600):
    """
    Sleep with a fixed frequency, until the target time

    :param target_time: the target time in datetime.datetime format
    :param check_frequency: the fixed sleep_time 
    :returns: none
    :raises: none
    """
    # get current time
    now = datetime.datetime.now()
    while now < target_time:

      # # get position summary
      # pos = self.get_position_summary()
      # self.logger.info(f'[rate]:----------------------------------------------\n{pos}\n')

      # get current time, calculate difference between current time and target time
      diff_time = (target_time - now).seconds
      sleep_time = (diff_time + 1) if (diff_time <= check_frequency) else check_frequency
      
      # sleep
      self.logger.info(f'[idle]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: sleep for {sleep_time} seconds')
      time.sleep(sleep_time)

      # update current time
      now = datetime.datetime.now()

    self.logger.info(f'[wake]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: exceed target time({target_time})')
