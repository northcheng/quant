import pandas as pd
from tigeropen.common.consts import Language
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient


# 获取用户账户信息
def get_user_info(info_path='drive/My Drive/tiger_quant/'):
  user_info = pd.read_csv(info_path + 'user_info.csv')
  return user_info.astype('str').loc[0,:].to_dict()


# 获取用户配置
def get_client_config(account='global_account', info_path='drive/My Drive/tiger_quant/', is_sandbox=False):
  user_info = get_user_info(info_path=info_path)

  client_config = TigerOpenClientConfig(sandbox_debug=is_sandbox)
  client_config.private_key = read_private_key(info_path + user_info['private_key_name'])
  client_config.tiger_id = str(user_info['tiger_id'])
  client_config.account = str(user_info[account])
  client_config.language = Language.en_US

  return client_config  


  # 获取查询器    
def get_quote_client(account='global_account'):
  client_config = get_client_config(account=account)
  quote_client = QuoteClient(client_config)

  return quote_client


# 获取交易器            
def get_trade_client(account='global_account'):
  client_config = get_client_config(account=account)
  trade_client = TradeClient(client_config)

  return trade_client


# 获取账户信息
def get_account_info(account='global_account', info_path='drive/My Drive/tiger_quant/'):

  user_info = get_user_info(info_path=info_path)
  trade_client = get_trade_client(account=account)
  managed_account = trade_client.get_managed_accounts()  
  position = trade_client.get_positions(account=user_info[account])
  assets = trade_client.get_assets(account=user_info[account])
  
  return{
      'managed_account': managed_account,
      'position': position,
      'assets': assets,
  }  


def get_asset_summary(trade_client, account, is_print=True):

  # 获取资产
  assets = trade_client.get_assets(account=account)
  
  for asset in assets:

    if is_print:
      print(
      '''
      for account: %(account)s:
      货币: %(currency)s
 
      总杠杆: %(leverage)s
      净杠杆: %(net_leverage)s

      总金额: %(cash)s
      购买力: %(buying_power)s
      可用资金: %(available_funds)s
      持仓市值: %(gross_position_value)s
      日内交易次数: %(day_trades_remaining)s
      ''' % dict(
        account=asset.account,
        currency=asset.summary.currency,
        leverage=asset.summary.leverage,
        net_leverage=asset.summary.net_leverage,
        cash=asset.summary.cash,
        buying_power=asset.summary.buying_power,
        available_funds=asset.summary.available_funds,
        gross_position_value=asset.summary.gross_position_value,
        day_trades_remaining=asset.summary.day_trades_remaining,
        )
      )

  return assets


def get_position_summary(trade_client, account, is_print=False):

  # 获取持仓
  positions = trade_client.get_positions(account=account)

  # 计算持仓盈亏
  result = {
    'sec_code': [],
    'quantity': [],
    'average_cost': [],
    'market_price': []
  }
  for pos in positions:
  
    # 计算持仓盈亏
    result['sec_code'].append(pos.contract.symbol)
    result['quantity'].append(pos.quantity)
    result['average_cost'].append(pos.average_cost)
    result['market_price'].append(pos.market_price)
  
  result = pd.DataFrame(result)
  result['earning'] = (result['market_price'] - result['average_cost']) * result['quantity']
  result['earning_rate'] = round(((result['market_price'] - result['average_cost']) / result['average_cost']) * 100, ndigits=2)
  
  return result   

