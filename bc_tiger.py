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
def get_client_config(account='global_account', is_sandbox=False):
  user_info = get_user_info()

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
def get_account_info(account='glboal_account'):

  trade_client = get_trade_client(account=account)
  managed_account = trade_client.get_managed_accounts()  
  position = trade_client.get_positions(account=user_info[account])
  assets = trade_client.get_assets(account=user_info[account])
  
  return{
      'managed_account': managed_account,
      'position': position,
      'assets': assets,
  }  