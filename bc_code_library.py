# # 计算触发信号所需的累积涨跌
# def cal_expected_acc_rate(mean_reversion_df, window_size, times_std):
  
#   x = sympy.Symbol('x')
#   acc_rate = np.hstack((mean_reversion_df.tail(window_size-1).acc_rate.values, x))
#   ma = acc_rate.mean()
#   std = sympy.sqrt(sum((acc_rate - ma)**2)/window_size)
#   result = sympy.solve((x - ma)**2 - (times_std*std)**2, x)
  
#   return result