# -*- coding: utf-8 -*-
"""
Utilities used for Tiger Open API

:author: Beichen Chen
"""
import pandas as pd
import datetime
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
      'pre_open_time': pre_open_time,
      'open_time': open_time,
      'close_time': close_time,
      'post_close_time': post_close_time,
      'tz': tz
    }

  # calculate earning rate for each position
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
        result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'status', 'latest_time']] #, 'pre_close', 'open', 'high', 'low', 'volume', 'ask_price', 'ask_size', 'bid_price', 'bid_size', 'market_price', ]]
        result['latest_time'] = result['latest_time'].apply(util.timestamp_2_time)

    else:
      result = pd.DataFrame()

    return result

  # get summary of assets
  def get_asset_summary(self):

    asset = self.assets[0]
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

      # place order
      trade_summary += f'[{action} {symbol} X {quantity} ({order_price})]\t'
      self.trade_client.place_order(order)
      self.orders.append(order.id)
      trade_summary += f'SUCCEED: {order.id}'
      print(trade_summary)
      
    except Exception as e:
      trade_summary += f'FAILED: {e}'
      print(trade_summary)

    return trade_summary


  
# def get_user_info(info_path='drive/My Drive/tiger_quant/'):
#   """
#   Get user information stored in Google Drive file

#   :param info_path: the path where user information file stored in
#   :returns: user information in dictionary
#   :raises: none
#   """
#   user_info = pd.read_csv(info_path + 'user_info.csv')
#   return user_info.astype('str').loc[0,:].to_dict()


# def get_client_config(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
#   """
#   Get client config

#   :param account: which kind of account: global_account/standard_account/simulation_account
#   :param info_path: the path where user information file stored in
#   :param is_sandbox: whether to use sandbox mode
#   :returns: client config instance
#   :raises: none
#   """
#   # get user information
#   user_info = get_user_info(info_path=info_path)

#   # create client config
#   client_config = TigerOpenClientConfig(sandbox_debug=is_sandbox)
#   client_config.private_key = read_private_key(info_path + user_info['private_key_name'])
#   client_config.tiger_id = str(user_info['tiger_id'])
#   client_config.account = str(user_info[account])
#   client_config.language = Language.en_US

#   return client_config  
   

# def get_quote_client(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
#   """
#   Get quote client for querying purpose

#   :param account: which kind of account are you using
#   :returns: quote client instance
#   :raises: none
#   """
#   client_config = get_client_config(account=account, info_path=info_path, is_sandbox=is_sandbox)
#   quote_client = QuoteClient(client_config)

#   return quote_client


# def get_trade_client(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
#   """
#   Get trade client for trading purpose

#   :param account: which kind of account are you using
#   :returns: trade client instance
#   :raises: none
#   """
#   client_config = get_client_config(account=account, info_path=info_path, is_sandbox=is_sandbox)
#   trade_client = TradeClient(client_config)

#   return trade_client


# def get_account_info(account='global_account', info_path='drive/My Drive/tiger_quant/'):
#   """
#   Get account information for specific account

#   :param account: which kind of account are you using
#   :param info_path: the path where user information file stored in
#   :returns: account information in dictionary
#   :raises: none
#   """
#   user_info = get_user_info(info_path=info_path)
#   trade_client = get_trade_client(account=account, info_path=info_path)

#   try:
#     managed_account = trade_client.get_managed_accounts()
#   except Exception as e:
#     print(e)
#     managed_account = None  

#   position = trade_client.get_positions(account=user_info[account])
#   assets = trade_client.get_assets(account=user_info[account])
  
#   return{
#       'managed_account': managed_account,
#       'position': position,
#       'assets': assets,
#   }  


# def get_asset_summary(trade_client, account, info_path, is_print=True):
#   """
#   Get asset summary for specific account

#   :param trade_client: trade client instance
#   :param account: account instance
#   :param is_print: whether to print asset information
#   :returns: assets instance
#   :raises: none
#   """

#   user_info = get_user_info(info_path=info_path)

#   # get assets instance
#   assets = trade_client.get_assets(account=user_info[account])
  
#   try:
#     # print asset information for each asset in assets
#     for asset in assets:
#       if is_print:
#         print(
#         f'for {asset.account}:\n货币: {asset.summary.currency}\n\n总杠杆: {asset.summary.leverage}\n净杠杆: {asset.summary.net_leverage}\n总金额: {asset.summary.cash}\n购买力: {asset.summary.buying_power}\n可用资金: {asset.summary.available_funds}\n持仓市值: {asset.summary.gross_position_value}\n日内交易次数: {asset.summary.day_trades_remaining}')
#   except Exception as e:
#     print(e)

#   return assets


# def get_position_summary(trade_client, account, info_path):
#   """
#   Get position summary for specific account

#   :param trade_client: trade client instance
#   :param account: account instance
#   :returns: position instance
#   :raises: none
#   """

#   user_info = get_user_info(info_path=info_path)
  
#   # get positions
#   positions = trade_client.get_positions(account=user_info[account])

#   # calculate earning for each position
#   result = {
#     'sec_code': [],
#     'quantity': [],
#     'average_cost': [],
#     'market_price': []
#   }
#   for pos in positions:
#     result['sec_code'].append(pos.contract.symbol)
#     result['quantity'].append(pos.quantity)
#     result['average_cost'].append(pos.average_cost)
#     result['market_price'].append(pos.market_price)
  
#   result = pd.DataFrame(result)
#   # result['earning'] = (result['market_price'] - result['average_cost']) * result['quantity']
#   # result['earning_rate'] = round(((result['market_price'] - result['average_cost']) / result['average_cost']) * 100, ndigits=2)
  
#   return result   


# def get_trading_time(quote_client, market=Market.US, tz='Asia/Shanghai'):

#   # get local timezone
#   tz = pytz.timezone(tz)

#   try:
#     # get market status
#     status = quote_client.get_market_status(market=market)[0]
#     open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
#     if status.status == 'Trading':
#       open_time = open_time - datetime.timedelta(days=1)

#     close_time = open_time + datetime.timedelta(hours=6.5)
#     pre_open_time = open_time - datetime.timedelta(hours=5.5)
#     post_close_time = close_time + datetime.timedelta(hours=4)
#   except Exception as e:
#     print(e)
#     open_time = close_time = pre_open_time = post_close_time = None

#   return {
#     'pre_open_time': pre_open_time,
#     'open_time': open_time,
#     'close_time': close_time,
#     'post_close_time': post_close_time,
#     'tz': tz
#   }
