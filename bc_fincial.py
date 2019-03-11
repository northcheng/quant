# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import mpl_finance as mpf
from matplotlib.pylab import date2num

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


#----------------------------- 均线模型 -----------------------------------#
# 计算移动平均信号
def cal_moving_average(df, dim, ma_windows=[3, 10], start_date=None, end_date=None):

  # 截取数据  
  df = df[start_date:end_date].copy()

  # 计算移动平均
  for mw in ma_windows:
    df[dim+'_ma_%s'% mw] = df[dim].rolling(mw).mean()
    
  return df


# 画出移动平均图
def plot_moving_average(df, dim, short_ma_window, long_ma_window, window_size, start_date=None, end_date=None, is_save=False, img_info={'path': 'drive/My Drive/probabilistic_model/images/', 'name': 'untitled', 'format': '.png'}):
  
  long_ma = dim + '_ma_%s' % long_ma_window
  short_ma = dim + '_ma_%s' % short_ma_window
  
  plot_dims = ['Close']
  
  if long_ma not in df.columns or short_ma not in df.columns:
    print("%(short)s or %(long)s MA on %(dim)s not found" % dict(short=short_ma_window, long=long_ma_window, dim=dim))
    
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
        buying_price = tmp_data.loc[buying_date, 'Open']
        stock = math.floor((cash-trading_fee) / buying_price)
        if stock > 0:
          buying_date = tmp_data.index.min()
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


#----------------------------- 资本资产定价模型 -----------------------------------#
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


#----------------------------- 概率模型 -----------------------------------#

# 计算特定列均值和上下N个标准差的范围
def cal_mean_std(df, dim, times_std, end_date=None, window_size=None):
 
  # 筛选数据
  if end_date is not None:
    df = df[:end_date]
  if window_size is not None:
    df = df[-window_size:]
    
  # 复制 dataframe
  df = df.copy()
  
  # 计算均值, 上下N倍标准差
  dim_mean = df[dim].mean()
  dim_std = df[dim].std()
  upper = dim_mean + times_std * dim_std
  lower = dim_mean - times_std * dim_std
  
  # 添加相应列
  df['mean'] = dim_mean
  df['std'] = dim_std
  df['upper'] = upper
  df['lower'] = lower

  return df


# 画出均值和上下N个标准差的范围
def plot_mean_std(df, dim, date, plot_info={'name': 'Untitled', 'data_length': 50, 'result_length':2}, is_save=False, img_info={'path': 'drive/My Drive/probabilistic_model/images/', 'format': '.png'}):
  
  # 需要绘出的维度
  plot_dims = ['upper', 'mean', 'lower', dim]
  
  # 构造图片名称
  title = '%(title)s [%(dim)s: %(dim_value).3f%%]\n[%(high).3f%%, %(avg).3f%%, %(low).3f%%]' % dict(
      title=plot_info['name'], 
      dim=dim,
      dim_value=df.loc[date, dim],
      avg=df.loc[date, 'mean'],
      high=df.loc[date, 'upper'],
      low=df.loc[date, 'lower']
  )
    
  # 创建图片
  plt.figure()
  plot_data = df[plot_dims].tail(plot_info['data_length'])
  
  # 画出信号
  signal_data = plot_data[:date]
  signal_data.plot(figsize=(20, 5), title=title)
  
  # 画出结果
  if plot_info['result_length'] > 0:
    result_idx = signal_data.index.tolist()[-1]
    result_data = plot_data[dim][result_idx:].head(plot_info['result_length']+1)
    plt.plot(result_data, '--oc', label='result', )
  
  plt.legend(loc='best')
  
  # 保存图像
  if is_save:
    plot_name = img_info['path'] + plot_info['name'] + '_' + date + '_' + '%s' % plot_info['result_length'] + img_info['format']
    plt.savefig(plot_name)
