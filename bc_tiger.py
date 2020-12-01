# -*- coding: utf-8 -*-
"""
Utilities used for Tiger Open API

:author: Beichen Chen
"""
import math
import logging
import datetime

from quant import bc_util as util
from quant import bc_data_io as io_util

import pytz
import time
import pandas as pd

from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.common.consts import (Language,  Market, BarPeriod, QuoteRight) # 语言, 市场, k线周期, 复权类型
from tigeropen.common.util.contract_utils import (stock_contract, option_contract, future_contract) # 股票合约, 期权合约, 期货合约
from tigeropen.common.util.order_utils import (market_order, limit_order, stop_order, stop_limit_order, trail_order, order_leg) # 市价单, 限价单, 止损单, 限价止损单, 移动止损单, 附加订单


class Tiger:


  # default logger
  defualt_logger = logging.getLogger('bc_tiger_logger')
  

  # init
  def __init__(self, account_type, config, sandbox_debug=False, logger_name=None, open_time_adj=0, close_time_adj=0):

    # get logger
    self.logger = Tiger.defualt_logger if (logger_name is None) else logging.getLogger(logger_name)

    # read user info, position record from local files
    self.__user_info = io_util.read_config(file_path=config['tiger_path'], file_name='user_info.json')
    self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='tiger_position_record.json')
    self.record = self.__position_record[account_type].copy()

    # set account, account type
    self.account = self.__user_info[account_type]
    self.account_type = account_type
    
    # initialize client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=sandbox_debug)
    self.client_config.private_key = read_private_key(config['tiger_path'] + self.__user_info['private_key_name'])
    self.client_config.tiger_id = str(self.__user_info['tiger_id'])
    self.client_config.language = Language.en_US
    self.client_config.account = self.account 
    
    # get quote/trade clients, assets, positions
    self.quote_client = QuoteClient(self.client_config)
    self.trade_client = TradeClient(self.client_config)
    self.positions = self.trade_client.get_positions(account=self.account)
    self.assets = self.trade_client.get_assets(account=self.account)

    # get market status and trade time
    self.update_trade_time(open_time_adj=open_time_adj, close_time_adj=close_time_adj)

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
    for symbol in [x for x in position_dict.keys() if x not in self.record.keys()]:
      record_conflicted = True
      self.record[symbol] = {'cash': 0, 'position': position_dict[symbol]}
      self.logger.error(f'[{account_type[:4]}]: {symbol} position({position_dict[symbol]}) not in record, add record')

    # update __position_record
    if record_conflicted:
      self.__position_record[self.account_type] = self.record.copy()
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='tiger_position_record.json')

    self.logger.info(f'[tiger]: Tiger instance created: {logger_name}')


  # get user info
  def get_user_info(self):
    return self.__user_info


  # get position record
  def get_position_record(self):
    return self.__position_record


  # update position for an account
  def update_position_record(self, config, init_cash=None, init_position=None, start_time=None, end_time=None, is_print=True):

    # set default values
    init_cash = config['trade']['init_cash'][self.account_type] if (init_cash is None) else init_cash
    init_position = 0 if (init_position is None) else init_position
    start_time = self.trade_time['pre_open_time'].strftime(format="%Y-%m-%d %H:%M:%S") if (start_time is None) else start_time
    end_time = self.trade_time['post_close_time'].strftime(format="%Y-%m-%d %H:%M:%S") if (end_time is None) else end_time
    
    try:

      # get today filled orders
      orders = self.trade_client.get_filled_orders(start_time=start_time, end_time=end_time)

      # update position records
      for order in orders:
        symbol = order.contract.symbol
        action = order.action
        quantity = order.quantity - order.remaining
        commission = order.commission
        avg_fill_price = order.avg_fill_price

        # init record if not exist
        if symbol not in self.record.keys():
          self.record[symbol] = {'cash': init_cash, 'position': init_position}
        record_cash = self.record[symbol]['cash']
        record_position = self.record[symbol]['position']
        
        # calculate new cash and position
        if action == 'BUY':
          cost = avg_fill_price * quantity + commission
          new_cash = record_cash - cost
          new_position = record_position + quantity
        
        elif action == 'SELL':
          acquire = avg_fill_price * quantity - commission
          new_cash = record_cash + acquire
          new_position = record_position - quantity

        else:
          new_cash = record_cash
          new_position = record_position

        # update record
        if new_cash >= 0 and new_position >= 0:
          self.record[symbol]['cash'] = new_cash
          self.record[symbol]['position'] = new_position
          if is_print:
            self.logger.info(f'[{self.account_type[:4]}]: updating position record for {symbol} {record_cash, record_position} -> {new_cash, new_position}')

      # update __position_record
      # self.record['updated'] = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
      self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='tiger_position_record.json')
      self.__position_record[self.account_type] = self.record.copy()
      self.__position_record['updated'][self.account_type] = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
      io_util.create_config_file(config_dict=self.__position_record, file_path=config['config_path'], file_name='tiger_position_record.json')
      
    except Exception as e:
      self.logger.exception(f'[erro]: fail updating position records for {self.account_type}, {e}')
  

  # update portfolio for an account
  def update_portfolio_record(self, config, position_summary=None, is_print=True):

    # get position summary
    if position_summary is None:
      position_summary = self.get_position_summary(get_briefs=False)
    position_summary.set_index('symbol', inplace=True)
    position_summary = position_summary.round(2)

    # get asset summary
    net_value = 0
    market_value = 0
    cash = 0
    asset_summary = self.get_asset_summary()
    if len(asset_summary) > 0:
      net_value = asset_summary.loc[0, 'net_value']
      market_value = asset_summary.loc[0, 'holding_value']
      cash = asset_summary.loc[0, 'cash']

    # post process
    if market_value == float('inf'):
      market_value = position_summary['market_value'].sum().round(2)

    # load portfolio record
    portfolio_record = io_util.read_config(file_path=config['config_path'], file_name='portfolio.json')
    old_net_value = portfolio_record['tiger'][self.account_type].get('net_value')

    # update portfolio record for current account
    portfolio_record['tiger'][self.account_type]['portfolio'] = position_summary.to_dict()
    portfolio_record['tiger'][self.account_type]['market_value'] = market_value
    portfolio_record['tiger'][self.account_type]['net_value'] = net_value
    portfolio_record['tiger'][self.account_type]['cash'] = cash
    portfolio_record['tiger'][self.account_type]['updated'] = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
    io_util.create_config_file(config_dict=portfolio_record, file_path=config['config_path'], file_name='portfolio.json')

    # print
    if is_print:
      self.logger.info(f'[{self.account_type[:4]}]: net value {old_net_value} --> {net_value}')


  # get summary of positions
  def get_position_summary(self, get_briefs=True):

    try:
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
          status = io_util.get_stock_briefs(symbols=[x.contract.symbol for x in self.positions], source='yfinance', period='1d', interval='1m')
          result = pd.merge(result, status, how='left', left_on='symbol', right_on='symbol')
          result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
          result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'latest_time']]
        else:
          result.rename(columns={'market_price':'latest_price'}, inplace=True)
          result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
          result['latest_time'] = None

        # calculate market value
        result['market_value'] = result['quantity'] * result['latest_price']

      else:
        result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'market_value':[], 'latest_time':[]})
    
    except Exception as e:
      result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'market_value':[], 'latest_time':[]})
      self.logger.exception(f'[erro]: can not get position summary: {e}')

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
    available_cash = self.assets[0].summary.cash

    return available_cash


  # get quantity of symbol currently in the position
  def get_in_position_quantity(self, symbol, get_briefs=False):

    # initialize affordable quantity
    quantity = 0

    # get position summary
    position = self.get_position_summary(get_briefs=get_briefs)
    if len(position) > 0:
      position = position.set_index('symbol')
      if symbol in position.index:
        quantity = position.loc[symbol, 'quantity']

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
      diff_time = round((target_time - now).total_seconds())
      sleep_time = (diff_time + 1) if (diff_time <= check_frequency) else check_frequency
      
      # sleep
      self.logger.info(f'[idle]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: sleep for {sleep_time} seconds')
      time.sleep(sleep_time)

      # update current time
      now = datetime.datetime.now()

    self.logger.info(f'[wake]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: exceed target time({target_time})')


  # update trade time
  def update_trade_time(self, market=Market.US, tz='Asia/Shanghai', open_time_adj=0, close_time_adj=0):

    # get local timezone
    tz = pytz.timezone(tz)

    try:
      # get open_time
      status = self.quote_client.get_market_status(market=market)[0]
      current_status = status.status
      open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
      open_time = open_time + datetime.timedelta(hours=open_time_adj)

      # if program runs after market open, api will return trade time for next trade day, 
      # trade time for current trade day need to be calculated manually
      if status.status in ['Trading', 'Post-Market Trading']:
        if open_time.weekday() == 0:
          open_time = open_time - datetime.timedelta(days=3)
        else:
          open_time = open_time - datetime.timedelta(days=1)

      # calculate close time, pre_open_time, post_close_time
      close_time = open_time + datetime.timedelta(hours=6.5 + close_time_adj)
      pre_open_time = open_time - datetime.timedelta(hours=5.5)
      post_close_time = close_time + datetime.timedelta(hours=4)

    except Exception as e:
      self.logger.error(e)
      current_status = None
      open_time = None
      close_time = None
      pre_open_time = None
      post_close_time = None

    self.trade_time = {
      'status': current_status, 'tz': tz,
      'pre_open_time': pre_open_time, 'open_time': open_time,
      'close_time': close_time, 'post_close_time': post_close_time
    }


  # update market status
  def update_market_status(self, market=Market.US, return_str=False):

    try:
      # get market status
      status = self.quote_client.get_market_status(market=market)[0]
      self.trade_time['status'] = status.status

      if return_str:
        time_format = '%Y-%m-%d %H:%M'
        pre_open_time = self.trade_time['pre_open_time'].strftime(time_format)
        post_close_time = self.trade_time['post_close_time'].strftime(time_format)
        
        time_format = '%H:%M'
        open_time = self.trade_time['open_time'].strftime(time_format)
        close_time = self.trade_time['close_time'].strftime(time_format)
        
        time_str = f'<({pre_open_time}){open_time} -- {close_time}({post_close_time})>'

        return time_str
        
    except Exception as e:
      self.logger.error(e)


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
        order_price = float(f'{price}')
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
        order_legs.append(stop_profit_order_leg)
      if len(order_legs)>0:
        order.order_legs = order_legs

      # place buy order if affordable
      if action == 'BUY':
        affordable_quantity = self.get_affordable_quantity(symbol=symbol)
        if quantity <= affordable_quantity:
          self.trade_client.place_order(order)
          trade_summary += f'SUCCEED: {order.id}'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity}/{quantity})'

      # place sell order if holding enough stocks
      elif action == 'SELL':    
        in_position_quantity = self.get_in_position_quantity(symbol)
        if in_position_quantity >= quantity:
          self.trade_client.place_order(order)
          trade_summary += f'SUCCEED: {order.id}'
        else:
          trade_summary += f'FAILED: Not enough stock to sell({in_position_quantity}/{quantity})'

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
  def signal_trade(self, signal, money_per_sec, order_type='market', trading_fee=5, pool=None, according_to_record=True, minimum_position=None):

    # set symbol to index
    if len(signal) > 0:
      # signal = signal.rename(columns={'代码':'symbol', '交易信号':'action'})
      # signal = signal.set_index('symbol')

      # filter sec with pool
      if pool is not None:
        filtered_list = [x for x in signal.index if x in pool]
        signal = signal.loc[filtered_list, signal.columns].copy()

    # if signal list is not empty
    if len(signal) > 0:
      # get latest price for signals
      
      # if order_type == 'market':
      # signal_brief = self.quote_client.get_stock_briefs(symbols=signal.index.tolist()).set_index('symbol')
      # signal_brief = io_util.get_stock_briefs(symbols=signal.index.tolist(), source='yfinance', period='1d', interval='1m').set_index('symbol')
      # signal = pd.merge(signal, signal_brief[['latest_price']], how='left', left_index=True, right_index=True)

      # get in-position quantity and latest price for signals
      position = self.get_position_summary(get_briefs=False)
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
            if order_type == 'limit':
              price = signal.loc[symbol, 'latest_price']
            else:
              price = None
            trade_summary = self.trade(symbol=symbol, action='SELL', quantity=in_position_quantity, price=price, print_summary=False)
            self.logger.info(trade_summary)
          else:
            self.logger.info(f'[SELL]: {symbol} skipped (not in positions)')
      else:
        self.logger.info(f'[SELL]: no signal')

      # buy
      # get available cash, set minimum position
      available_cash = self.get_available_cash()
      if minimum_position is None:
        minimum_position = money_per_sec * 0.1

      # get buy signals which not in posiitons yet
      default_money_per_sec = money_per_sec
      buy_signal = signal.query('action == "b"')
      if len(buy_signal) > 0:
        # go through buy signals
        for symbol in buy_signal.index:

          # break when available cash is below 200
          if available_cash <= minimum_position:
            self.logger.info(f'[BUY]: Available cash is too low({available_cash}/{minimum_position}), stop buying')
            break

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
            money_per_sec = available_cash if (money_per_sec > available_cash) else money_per_sec

            # calculate quantity to buy
            quantity = math.floor((money_per_sec-trading_fee)/signal.loc[symbol, 'latest_price'])
            if quantity > 0:
              if order_type == 'limit':
                price = signal.loc[symbol, 'latest_price']
              else:
                price = None
              trade_summary = self.trade(symbol=symbol, action='BUY', quantity=quantity, price=price, print_summary=False)
              self.logger.info(trade_summary)

              # update available cash
              available_cash -= quantity * signal.loc[symbol, 'latest_price']
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


  