# -*- coding: utf-8 -*-
"""
Utilities used for Tiger Open API

:author: Beichen Chen
"""
import pandas as pd
import datetime
import math
import time
import pytz
from tigeropen.common.consts import (Language,  Market, BarPeriod, QuoteRight) # 语言, 市场, k线周期, 复权类型
from tigeropen.common.util.order_utils import (market_order, limit_order, stop_order, stop_limit_order, trail_order, order_leg) # 市价单, 限价单, 止损单, 限价止损单, 移动止损单, 附加订单
from tigeropen.common.util.contract_utils import (stock_contract, option_contract, future_contract) # 股票合约, 期权合约, 期货合约
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient

from quant import bc_util as util

class Tiger:

  # user info
  user_info = {}
  client_config = None
  account_type = None
  trade_client = None
  quote_client = None
  assets = None
  positions = None
  trade_time = {}
  orders = []

  # init
  def __init__(self, account_type, info_path, sandbox_debug=False):

    # read user info from local file
    self.user_info = pd.read_csv(info_path + 'user_info.csv').loc[0,:].astype('str').to_dict()

    # set client_config
    self.account_type = account_type
    self.client_config = TigerOpenClientConfig(sandbox_debug=sandbox_debug)
    self.client_config.language = Language.en_US
    self.client_config.tiger_id = str(self.user_info['tiger_id'])
    self.client_config.private_key = read_private_key(info_path + self.user_info['private_key_name'])
    
    # get quote/trade client, assets and positions
    self.update_account(account_type=account_type)

    # get trade time
    self.get_trade_time()

  # get quote/trade client, assets and positions for specified account
  def update_account(self, account_type):

    # update config, trade_client, quote_client
    self.account_type = account_type
    self.client_config.account = str(self.user_info[account_type])
    self.quote_client = QuoteClient(self.client_config)
    self.trade_client = TradeClient(self.client_config)

    # update asset and position
    self.assets = self.trade_client.get_assets(account=self.user_info[account_type])
    self.positions = self.trade_client.get_positions(account=self.user_info[account_type])

  # get trade time
  def get_trade_time(self, market=Market.US, tz='Asia/Shanghai'):
    # get local timezone
    tz = pytz.timezone(tz)

    try:
      # get market status
      status = self.quote_client.get_market_status(market=market)[0]
      
      # get opentime
      open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
      if status.status == 'Trading':
        open_time = open_time - datetime.timedelta(days=1)

      # get close time
      close_time = open_time + datetime.timedelta(hours=6.5)

      # get pre-market opentime and post-market closetime 
      pre_open_time = open_time - datetime.timedelta(hours=5.5)
      post_close_time = close_time + datetime.timedelta(hours=4)

    except Exception as e:
      print(e)
      open_time = close_time = pre_open_time = post_close_time = None

    self.trade_time = {
      'status': status.status,
      'pre_open_time': pre_open_time,
      'open_time': open_time,
      'close_time': close_time,
      'post_close_time': post_close_time,
      'tz': tz
    }

  # get summary of positions
  def get_position_summary(self, get_briefs=True):

    if len(self.positions) > 0:
      
      result = {'symbol': [], 'quantity': [], 'average_cost': [], 'market_price': []}
      for pos in self.positions:
        result['symbol'].append(pos.contract.symbol)
        result['quantity'].append(pos.quantity)
        result['average_cost'].append(pos.average_cost)
        result['market_price'].append(pos.market_price)
      result = pd.DataFrame(result)

      if get_briefs:
        status = self.quote_client.get_stock_briefs(symbols=[x.contract.symbol for x in self.positions])
        result = pd.merge(result, status, how='left', left_on='symbol', right_on='symbol')
        result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
        result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'status', 'ask_price', 'bid_price', 'latest_time']] #, 'pre_close', 'open', 'high', 'low', 'volume', 'ask_price', 'ask_size', 'bid_price', 'bid_size', 'market_price', ]]
        result['latest_time'] = result['latest_time'].apply(util.timestamp_2_time)

    else:
      result = pd.DataFrame()

    return result

  # get summary of assets
  def get_asset_summary(self, print_summary=False):

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

  # get summary of orders
  def get_order_summary(self):
    
    pass

  # check whether it is affordable to buy certain amount of a stock
  def get_affordable_quantity(self, symbol, cash=None, trading_fee=3):

    quantity = 0
    
    # get available cash
    available_cash = cash
    if cash is None:
      self.assets = self.trade_client.get_assets(account=self.client_config.account)
      available_cash = self.assets[0].summary.available_funds
      
      # use cash rather than available_funds for simulation account
      if self.client_config.account == self.user_info['simulation_account']:
        available_cash = self.assets[0].summary.cash

    # get latest price of stock
    stock_brief = self.quote_client.get_stock_briefs(symbols=[symbol]).set_index('symbol')
    latest_price = stock_brief.loc[symbol, 'latest_price']

    # check if it is affordable
    quantity = math.floor((available_cash-trading_fee)/latest_price)

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

      # attach order legs
      order_legs = []
      if stop_loss is not None:
        stop_loss_order_leg = order_leg('LOSS', stop_loss, time_in_force='GTC') # 附加止损单
        order_legs.append(stop_loss_order_leg)
      if stop_profit is not None:
        stop_profit_order_leg = order_leg('PROFIT', 400.0, time_in_force='GTC') # 附加止盈单
        orderlegs.append(stop_profit_order_leg)
      if len(order_legs)>0:
        order.order_legs = order_legs

      # construct trade summary
      trade_summary += f'[{action} {symbol} X {quantity} ({order_price})]\t'

      # place order if affordable
      affordable_quantity = self.get_affordable_quantity(symbol=symbol)
      if quantity <= affordable_quantity:
        self.trade_client.place_order(order)
        self.orders.append(order.id)
        trade_summary += f'SUCCEED: {order.id}'
        print(trade_summary)
      else:
        trade_summary += f'FAILED: Not affordable({affordable_quantity})'
      
    except Exception as e:
      trade_summary += f'FAILED: {e}'
      print(trade_summary)

    return trade_summary

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

  # sleep until specified time
  def sleep_until(self, target_time, check_frequency=3600, print_position_summary=True):
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

      # calculate sleeping time
      diff_time = (target_time - now).seconds
      sleep_time = diff_time+1
      if diff_time > check_frequency:
        sleep_time = check_frequency
      print(f'{now.strftime(format="%Y-%m-%d %H:%M:%S")}: sleep for {sleep_time} seconds', flush=True)

      # print position summary
      if print_position_summary:
        position = self.get_position_summary()
        if len(position)>0:
          print(position[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate']], end='\n\n')

      # sleep
      time.sleep(sleep_time)

      # update current time
      now = datetime.datetime.now()

    print(f'{now}: exceed target time({target_time})', flush=True)





