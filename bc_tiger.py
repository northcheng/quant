
# 获取用户账户信息
def get_user_info(info_path='drive/My Drive/tiger_quant/'):
  user_info = pd.read_csv(info_path + 'user_info.csv')
  return user_info.astype('str').loc[0,:].to_dict()

