# -*- coding: utf-8 -*-
"""
Utilities used for Tiger Open API

:author: Beichen Chen
"""
import math
import pytz
import datetime
import logging
import pandas as pd
from quant import bc_util as util
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


  # init
  def __init__(self, account_type, info_path, sandbox_debug=False, logger_name=None):

    # read user info from local file
    self.__user_info = pd.read_csv(info_path + 'user_info.csv').loc[0,:].astype('str').to_dict()

    # set client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=sandbox_debug)
    self.client_config.private_key = read_private_key(info_path + self.__user_info['private_key_name'])
    self.client_config.language = Language.en_US
    self.client_config.tiger_id = str(self.__user_info['tiger_id'])
    
    # get quote/trade client, assets and positions
    self.update_account(account_type=account_type)

    # get trade time
    self.get_trade_time()

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
        status = self.quote_client.get_stock_briefs(symbols=[x.contract.symbol for x in self.positions])
        result = pd.merge(result, status, how='left', left_on='symbol', right_on='symbol')
        result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
        result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'status', 'latest_time']] #, 'pre_close', 'open', 'high', 'low', 'volume', 'ask_price', 'ask_size', 'bid_price', 'bid_size', 'market_price', ]]
        result['latest_time'] = result['latest_time'].apply(util.timestamp_2_time)

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
    stock_brief = self.quote_client.get_stock_briefs(symbols=[symbol]).set_index('symbol')
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
  def signal_trade(self, signal, money_per_sec, trading_fee=3, pool=None):

    if len(signal) > 0:
      signal = signal.rename(columns={'代码':'symbol', '交易信号':'action'})
      signal = signal.set_index('symbol')

      # filter sec with pool
      if pool is not None:
        signal = signal.loc[pool, signal.columns].copy()

      # get latest price for signals
      signal_brief = self.quote_client.get_stock_briefs(symbols=signal.index.tolist()).set_index('symbol')
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
      buy_signal = signal.query('action == "b"')
      if len(buy_signal) > 0:

        # go through buy signals
        for symbol in buy_signal.index:
          
          # check whether symbol is already in positions
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity == 0:
            
            # check whether there is enough money to establish a new position
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
      else:
        print('empty cash out list')

    else:
      print('empty position')      



