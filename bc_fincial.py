# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import math
# import sympy
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from quant import bc_util as util
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import mpl_finance as mpf
from matplotlib.pylab import date2num



#----------------------------- 股票池 -----------------------------------#
def get_symbols(remove_invalid=True, remove_not_fetched=True, not_fetched_list='drive/My Drive/probabilistic_model/yahoo_not_fetched_sec_code.csv'):

  try:
    symbols = get_nasdaq_symbols()
    symbols = symbols.loc[symbols['Test Issue'] == False,]
  except Exception as e:
    symbols = pd.read_table('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt', sep='|', index_col='Symbol').drop(np.NaN)
    symbols = symbols.loc[symbols['Test Issue'] == 'N',]

  sec_list = symbols.index.tolist()

  # 删除无效代码
  if remove_invalid:
    original_len = len(sec_list)
    sec_list = [x for x in sec_list if '$' not in x]
    sec_list = [x for x in sec_list if '.' not in x]
    current_len = len(sec_list)
    print('移除无效股票代码: ', original_len-current_len, '剩余长度: ', current_len)

  # 删除yahoo无法匹配的代码
  if remove_not_fetched:
    original_len = len(sec_list)
    yahoo_not_fetched_list = []
    try: 
      yahoo_not_fetched_list = pd.read_csv(not_fetched_list).sec_code.tolist()
    except Exception as e:
      print(e)
    sec_list = [x for x in sec_list if x not in yahoo_not_fetched_list]
    current_len = len(sec_list)
    print('移除无匹配股票代码: ', original_len-current_len, '剩余长度: ', current_len)

  return symbols.loc[sec_list, ]



#----------------------------- 蜡烛图 -----------------------------------#
# 为数据添加蜡烛图维度
def add_candle_dims_for_data(original_df):
  
  data = original_df.copy()
  
  # 影线范围
  data['shadow'] = (data['High'] - data['Low'])    
  
  # 实体范围
  data['entity'] = abs(data['Close'] - data['Open'])
  
  # 筛选涨跌
  up_idx = data.Open < data.Close
  down_idx = data.Open >= data.Close

  # 上影线/下影线
  data['upper_shadow'] = 0
  data['lower_shadow'] = 0
  data['candle_color'] = 0
  
  # 涨
  data.loc[up_idx, 'candle_color'] = 1
  data.loc[up_idx, 'upper_shadow'] = (data.loc[up_idx, 'High'] - data.loc[up_idx, 'Close'])
  data.loc[up_idx, 'lower_shadow'] = (data.loc[up_idx, 'Open'] - data.loc[up_idx, 'Low'])
  
  # 跌
  data.loc[down_idx, 'candle_color'] = -1
  data.loc[down_idx, 'upper_shadow'] = (data.loc[down_idx, 'High'] - data.loc[down_idx, 'Open'])
  data.loc[down_idx, 'lower_shadow'] = (data.loc[down_idx, 'Close'] - data.loc[down_idx, 'Low'])
  
  return data

# 画蜡烛图函数
def plot_candlestick(df, num_days=50, figsize=(15,5), title='', colors=('red', 'black')):
  
  # 取关键字段
  ohlc_timeseries_df = df[['Open', 'High', 'Low', 'Close']]

  # 转化数据
  data_list = []
  for dates,row in ohlc_timeseries_df.tail(num_days).iterrows():
   
    # 时间转化为float
    t = date2num(dates)
    open,high,low,close = row[:4]
    datas = (t,open,high,low,close)
    data_list.append(datas)

  # 创建子图
  fig, ax = plt.subplots(figsize=figsize)
  fig.subplots_adjust(bottom=0.2)
  fig.figsize = figsize
  #   ax.set_facecolor('white')
  
  # 设置x轴刻度为日期
  ax.xaxis_date()

  # x轴刻度文字倾斜45度
  plt.xticks(rotation=45)
  plt.xlabel('time')
  plt.ylabel('price')
  plt.title(title)

  # 绘制蜡烛图
  mpf.candlestick_ohlc(
    ax,
    data_list,
    width=0.8,
    colorup=colors[0], colordown=colors[1]
  )
  plt.grid(True)
  plt.show()
   


#----------------------------- 均值回归模型 -----------------------------------#
# 计算涨跌幅/累计涨跌幅
def cal_change_rate(original_df, dim, period=1, is_add_acc_rate=True):
  
  # 复制 dataframe
  df = original_df.copy()
  
  # 设置列名
  previous_dim = '%(dim)s-%(period)s' % dict(dim=dim, period=period)
  dim_rate = 'rate'
  dim_acc_rate = 'acc_rate'
  dim_acc_days = 'acc_days'
  
  # 计算涨跌率
  df[previous_dim] = df[dim].shift(period)
  df[dim_rate] = (df[dim] -  df[previous_dim]) / df[previous_dim] * 100
  
  # 添加累计维度列
  if is_add_acc_rate:
    
    df[dim_acc_rate] = 0
    df[dim_acc_days] = 1
  
    # 计算累计值
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current_rate = df.loc[current_idx, dim_rate]
      previous_acc_rate = df.loc[previous_idx, dim_acc_rate]
      previous_acc_days = df.loc[previous_idx, dim_acc_days]

      # 如果符号相同则累加, 否则重置
      if previous_acc_rate * current_rate > 0:
        df.loc[current_idx, dim_acc_rate] = current_rate + previous_acc_rate
        df.loc[current_idx, dim_acc_days] += previous_acc_days
      else:
        df.loc[current_idx, dim_acc_rate] = current_rate

    df.dropna(inplace=True) 
    df.drop(previous_dim, axis=1, inplace=True)

    return df

# 计算当前值与移动均值的差距离移动标准差的倍数
def cal_mean_reversion(df, dim, window_size=100, start_date=None, end_date=None):
  
  # 日收益率计算
  data = cal_change_rate(original_df=df,dim=dim)[start_date:end_date]
  
  # 计算变化率, 累计变化率, 累计天数的偏离均值距离
  for d in ['rate', 'acc_rate', 'acc_days']:
    
    # 计算累计变化率的移动平均及移动标准差
    tmp_mean = data[d].rolling(window_size).mean()
    tmp_std = data[d].rolling(window_size).std()
    
    # 计算偏差
    data[d+'_bias'] = (data[d] - tmp_mean) / (tmp_std)
  
  return data

# 画出均值回归偏差图
def plot_mean_reversion(df, times_std, window_size, start_date=None, end_date=None, is_save=False, img_info={'path': 'drive/My Drive/probabilistic_model/images/', 'name': 'untitled', 'format': '.png'}):
  
  # 需要绘出的维度
  plot_dims = ['rate_bias', 'acc_rate_bias', 'acc_days_bias']
    
  # 创建图片
  plt.figure()
  plot_data = df[plot_dims][:end_date].tail(window_size)
  plot_data['upper'] = times_std
  plot_data['lower'] = -times_std
  
  # 画出信号
  plot_data.plot(figsize=(20, 3))
  plt.legend(loc='best')
  
  # 保存图像
  if is_save:
    plot_name = img_info['path'] + img_info['name'] + '_' + end_date + '%s' %  img_info['format']
    plt.savefig(plot_name)

# # 计算触发信号所需的累积涨跌
# def cal_expected_acc_rate(mean_reversion_df, window_size, times_std):
  
#   x = sympy.Symbol('x')
#   acc_rate = np.hstack((mean_reversion_df.tail(window_size-1).acc_rate.values, x))
#   ma = acc_rate.mean()
#   std = sympy.sqrt(sum((acc_rate - ma)**2)/window_size)
#   result = sympy.solve((x - ma)**2 - (n*std)**2, x)
  
#   return result



#----------------------------- 均线模型 -----------------------------------#
# 计算移动平均信号
def cal_moving_average(df, dim, ma_windows=[3, 10], start_date=None, end_date=None):

  # 截取数据  
  df = df[start_date:end_date].copy()

  # 计算移动平均
  for mw in ma_windows:
    df[dim+'_ma_%s'% mw] = df[dim].rolling(mw).mean()
    
  return df

# 计算移动平均信号
def cal_moving_average_signal(ma_df, dim, short_ma, long_ma, start_date=None, end_date=None):
  
  ma_df = ma_df.copy()
  
  if long_ma not in ma_df.columns or short_ma not in ma_df.columns:
    print("%(short)s or %(long)s on %(dim)s not found" % dict(short=short_ma, long=long_ma, dim=dim))
    return pd.DataFrame()
  
  # 计算长短均线之差
  ma_df['ma_diff'] = ma_df[short_ma] - ma_df[long_ma]
  
  # 计算信号
  ma_df['signal'] = 'n'
  last_value = None
  for index, row in ma_df[start_date : end_date].iterrows():
    
    # 当前与之前的长短期均线差值
    current_value = row['ma_diff']
    
    if last_value is None:
      last_value = current_value
      continue
    
    # 短线从下方穿过长线, 买入
    if last_value < 0 and current_value > 0:
      ma_df.loc[index, 'signal'] = 'b'

    # 短线从上方穿过长线, 卖出
    elif last_value > 0 and current_value < 0:
      ma_df.loc[index, 'signal'] = 's'
      
    last_value = current_value
    
  return ma_df

# 画出移动平均图
def plot_moving_average(df, dim, short_ma, long_ma, window_size, start_date=None, end_date=None, is_save=False, img_info={'path': 'drive/My Drive/probabilistic_model/images/', 'name': 'untitled', 'format': '.png'}):
  
  plot_dims = ['Close']
  
  if long_ma not in df.columns or short_ma not in df.columns:
    print("%(short)s or %(long)s on %(dim)s not found" % dict(short=short_ma, long=long_ma, dim=dim))
    
  else:
    plot_dims += [long_ma, short_ma]
  
  # 创建图片
  plt.figure()
  plot_data = df[plot_dims][start_date:end_date].tail(window_size)
  
  # 画出信号
  plot_data.plot(figsize=(20, 3))
  plt.legend(loc='best')
  
  # 保存图像
  if is_save:
    plot_name = img_info['path'] + img_info['name'] + '_' + end_date + '%s' %  img_info['format']
    plt.savefig(plot_name)




#----------------------------- 回测工具 -----------------------------------#
# 回测
def back_test(signal, cash=0, stock=0, start_date=None, end_date=None, trading_fee=3, stop_profit=0.1, stop_loss=0.6, mode='earning', print_trading=True):
  
  # 获取指定期间的信号
  signal = signal[start_date:end_date]
  
  # 记录交易                           
  record = {
      'date': [],
      'action': [],
      'holding': [],
      'price': [],
      'cash': [],
      'total': []
  }
  
  # 以盈利模式进行回测
  if mode == 'earning':
    
    # 获取买入信号
    buy_signals = signal.query('signal == "b"').index.tolist()
    selling_date = signal.index.min()
  
    # 从第一次买入信号开始交易
    for date in buy_signals:
      
      # 信号的第二天开始操作
      tmp_data = signal[date:][1:]
      if (len(tmp_data) < 2) or  (date < selling_date):
        continue
      
      # 买入（开盘价）
      if stock == 0 and cash > 0:
        buying_date = tmp_data.index.min()
        buying_price = tmp_data.loc[buying_date, 'Open']
        stock = math.floor((cash-trading_fee) / buying_price)
        
        if stock > 0:
          cash = cash - stock * buying_price - trading_fee
          total = (cash + stock * buying_price)
          
          # 记录交易信息
          record['date'].append(buying_date.date())
          record['action'].append('b')
          record['holding'].append(stock)
          record['price'].append(buying_price)
          record['cash'].append(cash)
          record['total'].append(total)
          
          # 打印交易记录
          if print_trading:
            print(buying_date.date(), '买入 %(stock)s, 价格%(price)s, 流动资金%(cash)s, 总值%(total)s' % dict(stock=stock, price=buying_price, cash=cash, total=total))
        else: 
          print(buying_date.date(), '买入 %(stock)s' % dict(stock=stock))
      
      # 卖出（如果有持仓）
      if stock > 0:
        for index, row in tmp_data.iterrows():
          selling_date = index
          selling_price = row['Close']
          
          # 收益卖出(收盘价)
          if ((selling_price - buying_price)/ buying_price) > stop_profit:
            cash = cash + selling_price * stock - trading_fee
            stock = 0
            total = cash
            if print_trading:
              print(selling_date.date(), '止盈, 价格%(price)s, 流动资金%(cash)s, 总值%(total)s' % dict(price=selling_price, cash=cash, total=total))

            # 记录交易信息
            record['date'].append(selling_date.date())
            record['action'].append('s')
            record['holding'].append(stock)
            record['price'].append(selling_price)
            record['cash'].append(cash)
            record['total'].append(cash)
            break;

          # 止损卖出(收盘价)
          elif ((selling_price - buying_price)/ buying_price) < -stop_loss:
            cash = cash + selling_price * stock - trading_fee 
            stock = 0
            total = cash
            if print_trading:
              print(selling_date.date(), '止损, 价格%(price)s, 流动资金%(cash)s, 总值%(total)s' % dict(price=selling_price, cash=cash, total=total))

            # 记录交易信息
            record['date'].append(selling_date.date())
            record['action'].append('s')
            record['holding'].append(stock)
            record['price'].append(selling_price)
            record['cash'].append(cash)
            record['total'].append(total)
            break;

  # 以信号模式进行回测          
  elif mode == 'signal':
    
    # 去除冲突的信号
    buy_sell_signals = signal.query('signal != "n"')
    trading_signals = []
    last_signal = 'n'
    for index, row in buy_sell_signals.iterrows():
      current_signal = row['signal']  
      if current_signal == last_signal:
        continue
      else:
        trading_signals.append(index)
      last_signal = current_signal
    
    # 开始交易
    for date in trading_signals:
      
      if date == signal.index.max():
        print('信号于', date, '发出')
        break
      
      # 信号的第二天交易
      tmp_signal = signal.loc[date, 'signal']
      tmp_data = signal[date:][1:]
      trading_date = tmp_data.index.min()
      
      # 以开盘价买入
      if tmp_signal == 'b':
        buying_price = signal.loc[trading_date, 'Open']
        stock = math.floor((cash-trading_fee) / buying_price)
        if stock > 0:
          cash = cash - stock * buying_price - trading_fee
          total = (cash + stock * buying_price)
          if print_trading:
            print(trading_date.date(), '买入 %(stock)s, 价格%(price)s, 流动资金%(cash)s, 总值%(total)s' % dict(stock=stock, price=buying_price, cash=cash, total=total))

          # 记录交易信息
          record['date'].append(trading_date.date())
          record['action'].append('b')
          record['holding'].append(stock)
          record['price'].append(buying_price)
          record['cash'].append(cash)
          record['total'].append(total)
        else: 
          print(trading_date.date(), '买入 %(stock)s' % dict(stock=stock))

      # 以收盘价卖出
      elif tmp_signal == 's':
        if stock > 0:
          selling_price = signal.loc[trading_date, 'Close']
          cash = cash + selling_price * stock - trading_fee
          stock = 0
          total = cash + stock * selling_price
          if print_trading:
            print(trading_date.date(), '卖出, 价格%(price)s, 流动资金%(cash)s, 总值%(total)s' % dict(price=selling_price, cash=cash, total=total))

          # 记录交易信息
          record['date'].append(trading_date.date())
          record['action'].append('s')
          record['holding'].append(stock)
          record['price'].append(selling_price)
          record['cash'].append(cash)
          record['total'].append(total)
            
      else:
        print('invalid signal %s' % tmp_signal)
    
  # 未定义的模式
  else:
    print('mode [%s] not found' % mode)

  # 记录最新数据
  current_date = signal.index.max()
  current_price = signal.loc[current_date, 'Close']
  total = cash + stock * current_price
  record['date'].append(current_date.date())
  record['action'].append(signal.loc[current_date, 'signal'])
  record['holding'].append(stock)
  record['price'].append(current_price)
  record['cash'].append(cash)
  record['total'].append(total)
  if print_trading:
    print(current_date.date(), '当前, 价格%(price)s, 总值%(total)s' % dict(price=current_price, total=total))
  
  # 将记录转化为时序数据
  record = util.df_2_timeseries(pd.DataFrame(record), time_col='date')
  return record      



#----------------------------- 均值/方差模型 -----------------------------------#
# 风险溢价是超额收益的期望值(rate_premium = mean(excess_return)),
# 超额收益的标准差是其风险的测度(risk = std(excess_return))
# 计算持有期收益率(Holding Period Rate)
def cal_HPR(data, start, end, dim='Close', dividends=0):
  data = data[start:end][dim].tolist()
  HPR = (data[-1] - data[0]) / data[0]
  
  return HPR

# 计算有效年收益率(Effective Annual Rate)
def cal_EAR(data, start, end, dim='Close', dividends=0):
  # 计算期间内的收益率
  HPR = cal_HPR(data, start, end, dim, dividends) + 1
  # 计算期间的长度(年)
  period_in_year = num_year_between(start, end)
  # 计算有效年利率
  EAR = pow(HPR, 1/period_in_year) - 1
  
  return EAR

# 计算一段期间为多少年
def num_year_between(start, end):
  start=datetime.datetime.strptime(start,"%Y-%m-%d")
  end=datetime.datetime.strptime(end,"%Y-%m-%d")
  
  return (end-start).days / 365

# 计算年华百分比利率(Annual Percentile Rate)
def cal_APR(data, start, end, dim='Close', dividends=0):
  # 计算期间内的收益率
  HPR = cal_HPR(data, start, end, dim, dividends)
  # 计算期间的长度(年)
  period_in_year = num_year_between(start, end)
  # 计算有效年利率
  APR = HPR / period_in_year
  
  return APR

# 计算连续复利利率(Continuous Compounding Rate)
def cal_CCR(data, start, end, dim='Close', dividends=0):
  EAR = cal_EAR(data, start, end, dim, dividends)
  CCR = math.log((1+EAR), math.e)
  
  return CCR

# 计算期望收益率(Expected Return)
def cal_expected_rate(data, dim, start=None, end=None):
  ER = data[start : end][dim].mean()
    
  return ER

# 计算风险(方差)
def cal_standard_deviation(data, dim, start=None, end=None):
  STD = data[start : end][dim].std()
  
  return STD

# 计算风险溢价(Risk Premium)
def cal_rate_premium(expected_rate, risk_free_rate):
  RP = expected_rate - risl_free_rate
  
  return RP

# 计算超额收益(Excess Return)
def cal_excess_raturn(expected_rate, real_rate):
  ER = real_rate - expected_rate
  
  return ER

# 计算周期收益率(年/月)
def cal_period_rate(sec_data, by='month'):
  
  # 计算周期收益率
  start_date = sec_data.index.min().date()
  end_date = sec_data.index.max().date()
  
  # 构造周期列表
  periods = []

  # 年周期
  if by == 'year':
    for year in range(start_date.year, end_date.year+1):
      periods.append('%(year)s' % dict(year=year))
      
  # 月周期      
  elif by == 'month':
    for year in range(start_date.year, end_date.year+1):
      for month in range(1, 13):
        if year >= end_date.year and month > end_date.month:
          break
        p = '%(year)s-%(month)02d' % dict(year=year, month=month)
        periods.append(p)
  else:
    print('Invalid period')
  
  # 计算周期收益率
  period_rate = {
      'period': [],
      'rate': []
  } 
  for p in periods:
    tmp_data = sec_data[p:p]
    if len(tmp_data) == 0:
      continue
    else:
      period_rate['period'].append(p)
      period_rate['rate'].append(cal_HPR(data=tmp_data, start=None, end=None, dim='Close'))
  
  period_rate = pd.DataFrame(period_rate)
  period_rate = util.df_2_timeseries(df=period_rate, time_col='period')
  
  return period_rate

# 计算风险与收益
def cal_risk_and_rate(rate_df, risk_free_rate, A=0.5):
  
  # 算数平均利率
  mean_rate = rate_df.rate.mean()

  # 风险
  risk = rate_df.rate.std()

  # 风险溢价
  risk_premium = mean_rate - risk_free_rate

  # 夏普比率
  sharp_ratio = risk_premium / risk

  # 风险厌恶系数
  A = 0.5

  # 效用
  U = mean_rate - 0.5 * A * risk **2
  
  return {
      'mean_rate': mean_rate * 100,
      'risk': risk * 100,
      'risk_premium': risk_premium * 100,
      'sharp_ratio': sharp_ratio * 100,
      'U': U * 100
  }



#----------------------------- 技术分析 -----------------------------------#
# 画出以benchmark为界的柱状图, 上升为绿, 下降为红
def plot_indicator_around_benchmark(data, target_col, benchmark=0, title=None, start_date=None, end_date=None, color_mode='up_down', plot_close=True, figsize=(20, 5)):
  
  plot_data = data[start_date:end_date].copy()
  
  if color_mode == 'up_down':  
    plot_data['color'] = 'red'
    previous_target_col = 'previous_'+target_col
    plot_data[previous_target_col] = plot_data[target_col].shift(1)
    plot_data.loc[plot_data[target_col] > plot_data[previous_target_col], 'color'] = 'green'
  
  elif color_mode == 'zero':
    plot_data['color'] = 'red'
    plot_data.loc[plot_data[target_col] > benchmark, 'color'] = 'green'
    
  else:
    print('Unknown Color Mode')
    return None

  fig = plt.figure(figsize=figsize)
  ax1 = fig.add_subplot(111)
  ax1.bar(plot_data.index, height=plot_data[target_col], color=plot_data.color, alpha=0.5)
  ax1.set_title(title)

  if plot_close:
    ax2=ax1.twinx()
    ax2.plot(plot_data.Close, color='blue' )

# 画出指标与价格
def plot_indicator(data, target_col, title=None, start_date=None, end_date=None, plot_close=True, figsize=(20, 5)):
  
  plot_data = data[start_date:end_date].copy()

  fig = plt.figure(figsize=figsize)
  ax1 = fig.add_subplot(111)
  ax1.plot(plot_data[target_col], color='red', alpha=0.5)
  ax1.set_title(title)
  
  if plot_close:
    ax2=ax1.twinx()
    ax2.plot(plot_data.Close, color='blue' ) 
