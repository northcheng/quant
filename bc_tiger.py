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
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient


def get_user_info(info_path='drive/My Drive/tiger_quant/'):
  """
  Get user information stored in Google Drive file

  :param info_path: the path where user information file stored in
  :returns: user information in dictionary
  :raises: none
  """
  user_info = pd.read_csv(info_path + 'user_info.csv')
  return user_info.astype('str').loc[0,:].to_dict()


def get_client_config(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
  """
  Get client config

  :param account: which kind of account: global_account/standard_account/simulation_account
  :param info_path: the path where user information file stored in
  :param is_sandbox: whether to use sandbox mode
  :returns: client config instance
  :raises: none
  """
  # get user information
  user_info = get_user_info(info_path=info_path)

  # create client config
  client_config = TigerOpenClientConfig(sandbox_debug=is_sandbox)
  client_config.private_key = read_private_key(info_path + user_info['private_key_name'])
  client_config.tiger_id = str(user_info['tiger_id'])
  client_config.account = str(user_info[account])
  client_config.language = Language.en_US

  return client_config  
   

def get_quote_client(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
  """
  Get quote client for querying purpose

  :param account: which kind of account are you using
  :returns: quote client instance
  :raises: none
  """
  client_config = get_client_config(account=account, info_path=info_path, is_sandbox=is_sandbox)
  quote_client = QuoteClient(client_config)

  return quote_client


def get_trade_client(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
  """
  Get trade client for trading purpose

  :param account: which kind of account are you using
  :returns: trade client instance
  :raises: none
  """
  client_config = get_client_config(account=account, info_path=info_path, is_sandbox=is_sandbox)
  trade_client = TradeClient(client_config)

  return trade_client


def get_account_info(account='global_account', info_path='drive/My Drive/tiger_quant/'):
  """
  Get account information for specific account

  :param account: which kind of account are you using
  :param info_path: the path where user information file stored in
  :returns: account information in dictionary
  :raises: none
  """
  user_info = get_user_info(info_path=info_path)
  trade_client = get_trade_client(account=account, info_path=info_path)

  try:
    managed_account = trade_client.get_managed_accounts()
  except Exception as e:
    print(e)
    managed_account = None  

  position = trade_client.get_positions(account=user_info[account])
  assets = trade_client.get_assets(account=user_info[account])
  
  return{
      'managed_account': managed_account,
      'position': position,
      'assets': assets,
  }  


def get_asset_summary(trade_client, account, info_path, is_print=True):
  """
  Get asset summary for specific account

  :param trade_client: trade client instance
  :param account: account instance
  :param is_print: whether to print asset information
  :returns: assets instance
  :raises: none
  """

  user_info = get_user_info(info_path=info_path)

  # get assets instance
  assets = trade_client.get_assets(account=user_info[account])
  
  try:
    # print asset information for each asset in assets
    for asset in assets:
      if is_print:
        print(
        f'for {asset.account}:\n货币: {asset.summary.currency}\n\n总杠杆: {asset.summary.leverage}\n净杠杆: {asset.summary.net_leverage}\n总金额: {asset.summary.cash}\n购买力: {asset.summary.buying_power}\n可用资金: {asset.summary.available_funds}\n持仓市值: {asset.summary.gross_position_value}\n日内交易次数: {asset.summary.day_trades_remaining}')
  except Exception as e:
    print(e)

  return assets


def get_position_summary(trade_client, account, info_path):
  """
  Get position summary for specific account

  :param trade_client: trade client instance
  :param account: account instance
  :returns: position instance
  :raises: none
  """

  user_info = get_user_info(info_path=info_path)
  
  # get positions
  positions = trade_client.get_positions(account=user_info[account])

  # calculate earning for each position
  result = {
    'sec_code': [],
    'quantity': [],
    'average_cost': [],
    'market_price': []
  }
  for pos in positions:
    result['sec_code'].append(pos.contract.symbol)
    result['quantity'].append(pos.quantity)
    result['average_cost'].append(pos.average_cost)
    result['market_price'].append(pos.market_price)
  
  result = pd.DataFrame(result)
  # result['earning'] = (result['market_price'] - result['average_cost']) * result['quantity']
  # result['earning_rate'] = round(((result['market_price'] - result['average_cost']) / result['average_cost']) * 100, ndigits=2)
  
  return result   


def get_trading_time(quote_client, market=Market.US, tz='Asia/Shanghai'):

  # get local timezone
  tz = pytz.timezone(tz)

  try:
    # get market status
    status = quote_client.get_market_status(market=market)[0]
    open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
    close_time = open_time + datetime.timedelta(hours=6.5)
    pre_open_time = open_time - datetime.timedelta(hours=5.5)
    post_close_time = close_time + datetime.timedelta(hours=4)
  except Exception as e:
    print(e)
    open_time = close_time = pre_open_time = post_close_time = None

  return {
    'pre_open_time': pre_open_time,
    'open_time': open_time,
    'close_time': close_time,
    'post_close_time': post_close_time,
    'tz': tz
  }




