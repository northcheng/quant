import panads as pd
import tushare as ts

# 获取用户信息
def get_user_info(info_path='drive/My Drive/tushare_quant/'):
  user_info = pd.read_csv(info_path + 'user_info.csv')
  return user_info.astype('str').loc[0,:].to_dict()

# 获取API
def get_ts_client(info_path='drive/My Drive/tushare_quant/'):
  user_info = get_user_info(info_path=info_path)
  ts.set_token(user_info['token'])
  tsp = ts.pro_api()

  return tsp