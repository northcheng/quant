# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import math
import sympy
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from quant import bc_util as util
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.pylab import date2num



#----------------------------- 获取股票池 -------------------------------------#
def get_symbols(remove_invalid=True, remove_not_fetched=True, not_fetched_list='drive/My Drive/probabilistic_model/yahoo_not_fetched_sec_code.csv'):

  # 使用pandas_datareader下载股票列表
  try:
    symbols = get_nasdaq_symbols()
    symbols = symbols.loc[symbols['Test Issue'] == False,]
  
  # 直接从纳斯达克网站下载股票列表
  except Exception as e:
    symbols = pd.read_table('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt', sep='|', index_col='Symbol').drop(np.NaN)
    symbols = symbols.loc[symbols['Test Issue'] == 'N',]
  sec_list = symbols.index.tolist()

  # 删除无效代码
  if remove_invalid:
    original_len = len(sec_list)
    sec_list = [x for x in sec_list if '$' not in x]
    sec_list = [x for x in sec_list if '.' not in x]

  # 删除yahoo无法匹配的代码
  if remove_not_fetched:
    original_len = len(sec_list)
    yahoo_not_fetched_list = []
    try: 
      yahoo_not_fetched_list = pd.read_csv(not_fetched_list).sec_code.tolist()
    except Exception as e:
      print(e)
    sec_list = [x for x in sec_list if x not in yahoo_not_fetched_list]
  
  return symbols.loc[sec_list, ]



#----------------------------- 均值回归模型 -----------------------------------#
# 计算涨跌幅/累计涨跌幅
def cal_change_rate(df, dim, period=1, add_accumulation=True, add_prefix=False):
  
  # 复制 dataframe
  df = df.copy()
  
  # 设置列名
  previous_dim = '%(dim)s-%(period)s' % dict(dim=dim, period=period)
  rate_dim = 'rate'
  acc_rate_dim = 'acc_rate'
  acc_day_dim = 'acc_day'

  if add_prefix:
    rate_dim = dim + '_' + rate_dim
    acc_rate_dim = dim + '_' + acc_rate_dim
    acc_day_dim = dim + '_' + acc_day_dim
  
  # 计算涨跌率
  df[previous_dim] = df[dim].shift(period)
  df[rate_dim] = (df[dim] -  df[previous_dim]) / df[previous_dim]

  # 计算累计维度列
  if add_accumulation:
    
    df[acc_rate_dim] = 0
    # df[acc_day_dim] = 1
    df.loc[df['rate']>0, acc_day_dim] = 1
    df.loc[df['rate']<0, acc_day_dim] = -1
  
    # 计算累计值
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current_rate = df.loc[current_idx, rate_dim]
      previous_acc_rate = df.loc[previous_idx, acc_rate_dim]
      previous_acc_days = df.loc[previous_idx, acc_day_dim]

      # 如果符号相同则累加, 否则重置
      if previous_acc_rate * current_rate > 0:
        df.loc[current_idx, acc_rate_dim] = current_rate + previous_acc_rate
        df.loc[current_idx, acc_day_dim] += previous_acc_days
      else:
        df.loc[current_idx, acc_rate_dim] = current_rate

  df.dropna(inplace=True) 
  df.drop(previous_dim, axis=1, inplace=True)

  return df

# 计算当前值与移动均值的差距离移动标准差的倍数
def cal_mean_reversion(df, dim, window_size=100, start_date=None, end_date=None):
  
  # 日收益率计算
  original_columns = df.columns
  data = cal_change_rate(df=df, dim=dim, period=1, add_accumulation=True)[start_date:end_date]
  
  # 计算变化率, 累计变化率, 累计天数的偏离均值距离
  new_columns = [x for x in data.columns if x not in original_columns]
  for d in new_columns:
    
    # 计算累计变化率的移动平均及移动标准差
    tmp_mean = data[d].rolling(window_size).mean()
    tmp_std = data[d].rolling(window_size).std()
    
    # 计算偏差
    data[d+'_bias'] = (data[d] - tmp_mean) / (tmp_std)
  
  return data

# 计算均值回归信号
def cal_mean_reversion_signal(df, time_std=2, triger_dim=['rate_bias', 'acc_rate_bias', 'acc_day_bias'], triger_threshold=2, start_date=None, end_date=None):
  
  # 复制 dataframe
  mr_df = df.copy()

  # 选择包含 'bias' 的列
  target_dim = [x for x in mr_df.columns if 'bias' in x]
  for t in triger_dim:
    if t not in target_dim:
      print(t, 'not found in columns!')
      triger_dim = [x for x in triger_dim if x != t]
  
  #triger_threshold = len(triger_dim)

  # 初始化信号
  mr_df['signal'] = 0

  # 计算每种 bias 的信号
  for dim in triger_dim:
    signal_dim = dim.replace('bias', 'signal')
    mr_df[signal_dim] = 0
    
    # 超买信号
    mr_df.loc[mr_df[dim] > time_std, signal_dim] = 1
    
    # 超卖信号
    mr_df.loc[mr_df[dim] < -time_std, signal_dim] = -1

    # 综合信号
    mr_df['signal'] = mr_df['signal'] + mr_df[signal_dim]
  
  # 将信号从数字转化为字符  
  sell_signals = mr_df.loc[mr_df['signal'] >= triger_threshold, ].index
  buy_signals = mr_df.loc[mr_df['signal'] <= -triger_threshold, ].index
  mr_df['signal'] = 'n'
  mr_df.loc[sell_signals, 'signal'] = 's'
  mr_df.loc[buy_signals, 'signal'] = 'b'
  
  return mr_df

# 计算触发信号所需的累积涨跌
def cal_mean_reversion_expected_rate(df, rate_dim, window_size, time_std):
  
  x = sympy.Symbol('x')
  
  rate_data = np.hstack((df.tail(window_size-1)[rate_dim].values, x))
  ma = rate_data.mean()
  std = sympy.sqrt(sum((rate_data - ma)**2)/(window_size-1))
  result = sympy.solve(((x - ma)**2) - ((time_std*std)**2), x)
  
  return result

# 画出均值回归偏差图
def plot_mean_reversion(df, times_std, window_size, start_date=None, end_date=None):
  
  # 需要绘出的维度
  plot_dims = [x for x in df.columns if '_bias' in x]
    
  # 创建图片
  plt.figure()
  plot_data = df[plot_dims][:end_date].tail(window_size)
  plot_data['upper'] = times_std
  plot_data['lower'] = -times_std
  
  # 画出信号
  plot_data.plot(figsize=(20, 3))
  plt.legend(loc='best')
  


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
def plot_moving_average(df, dim, short_ma, long_ma, window_size, start_date=None, end_date=None):
  
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



#----------------------------- 回测工具 -----------------------------------#
# 回测
def back_test(signal, buy_price='Open', sell_price='Close', cash=0, stock=0, start_date=None, end_date=None, trading_fee=3, stop_profit=0.1, stop_loss=0.6, mode='earning', print_trading=True, plot_trading=True):
  
  # 获取指定期间的信号
  signal = signal[start_date:end_date]
  original_cash = cash
  
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
        buying_price = tmp_data.loc[buying_date, buy_price]
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
          selling_price = row[sell_price]
          
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
        buying_price = signal.loc[trading_date, buy_price]
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
          selling_price = signal.loc[trading_date, sell_price]
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

  # 画出回测图
  if plot_trading:
    buying_points = record.query('action == "b"')
    selling_points = record.query('action == "s"')
  
    f, ax = plt.subplots(figsize = (20, 3))
    plt.plot(signal[['Close']])
    plt.scatter(buying_points.index,buying_points.price, c='green')
    plt.scatter(selling_points.index,selling_points.price, c='red')
  
    total_value_data = pd.merge(signal[['Close']], record[['cash', 'holding', 'action']], how='left', left_index=True, right_index=True)
    total_value_data.fillna(method='ffill', inplace=True)
    total_value_data['original'] = original_cash
    total_value_data['total'] = total_value_data['Close'] * total_value_data['holding'] + total_value_data['cash']
    total_value_data[['total', 'original']].plot(figsize=(20, 3))
  
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
  start_date = util.time_2_string(data[start:end].index.min())
  end_date = util.time_2_string(data[start:end].index.max())
  period_in_year = util.num_days_between(start, end) / 365
  # 计算有效年利率
  EAR = pow(HPR, 1/period_in_year) - 1
  
  return EAR

# 计算年化百分比利率(Annual Percentile Rate)
def cal_APR(data, start, end, dim='Close', dividends=0):
  # 计算期间内的收益率
  HPR = cal_HPR(data, start, end, dim, dividends)
  # 计算期间的长度(年)
  start_date = util.time_2_string(data[start:end].index.min())
  end_date = util.time_2_string(data[start:end].index.max())
  period_in_year = util.num_days_between(start, end) / 365
  # 计算有效年利率
  APR = HPR / period_in_year
  
  return APR

# 计算连续复利利率(Continuous Compounding Rate)
def cal_CCR(data, start, end, dim='Close', dividends=0):
  EAR = cal_EAR(data, start, end, dim, dividends)
  CCR = math.log((1+EAR), math.e)
  
  return CCR

# 按日计算期望收益率(Expected Return)与风险(Risk)
def cal_rate_risk(data, dim, period, start=None, end=None):
  rate_df = cal_change_rate(df=data, dim=dim, period=period, add_accumulation=False, add_prefix=False)
  rate = rate_df[start : end]['rate'].mean()
  risk = rate_df[start : end]['rate'].std()

  return {'rate': rate, 'risk': risk}

# 计算风险溢价(Risk Premium)
def cal_risk_premium(expected_rate, risk_free_rate):
  RP = expected_rate - risk_free_rate
  
  return RP

# 计算超额收益(Excess Return)
def cal_excess_raturn(expected_rate, real_rate):
  ER = real_rate - expected_rate
  
  return ER

# 计算周期收益率(年/月)
def cal_period_rate(sec_data, by='month'):
  
  sec_data = cal_change_rate(df=sec_data, dim='Close', period=1, add_accumulation=False, add_prefix=False)

  # 计算周期收益率
  start_date = sec_data.index.min().date()
  end_date = sec_data.index.max().date()
  
  # 构造周期列表
  periods = []

  # 年周期
  if by == 'year':
    for year in range(start_date.year, end_date.year+1):
      p = '%(year)s' % dict(year=year)
      periods.append((p, p))
      
  # 月周期      
  elif by == 'month':
    for year in range(start_date.year, end_date.year+1):
      for month in range(1, 13):
        if year >= end_date.year and month > end_date.month:
          break
        p = '%(year)s-%(month)02d' % dict(year=year, month=month)
        periods.append((p, p))

  # 周周期
  elif by == 'week':
    week_start = start_date
    while week_start < end_date:
      week_end = week_start + datetime.timedelta(days=(6 - week_start.weekday()))
      periods.append((week_start, week_end))
      week_start = week_end + datetime.timedelta(days=1)
  else:
    print('Invalid period')
  
  # 计算周期收益率
  period_rate = {
      'period': [],
      'HPR': [],
      'EAR': [],
      'APR': [],
      'CCR': [],
      'daily_rate_mean': [],
      'daily_rate_std': []
  } 
  for p_pair in periods:
    tmp_data = sec_data[p_pair[0]:p_pair[1]]
    if len(tmp_data) == 0:
      continue
    else:
      period_rate['period'].append(p_pair[0])
      period_rate['HPR'].append(cal_HPR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['EAR'].append(cal_EAR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['APR'].append(cal_APR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['CCR'].append(cal_CCR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['daily_rate_mean'].append(tmp_data.rate.mean())
      period_rate['daily_rate_std'].append(tmp_data.rate.std())
  
  period_rate = pd.DataFrame(period_rate)
  period_rate = util.df_2_timeseries(df=period_rate, time_col='period')
  
  return period_rate

# 计算风险与收益
def cal_risk_and_rate(rate_df, risk_free_rate, window_size=10, A=0.5):
  
  result = rate_df.copy()

  # 算数平均利率
  result['mean_rate'] = rate_df.rate.rolling(window=window_size).mean()

  # 风险
  result['risk'] = rate_df.rate.rolling(window=window_size).std()

  # 风险溢价
  result['risk_premium'] = result.mean_rate - risk_free_rate

  # 夏普比率
  result['sharp_ratio'] = result.risk_premium / result.risk

  # 风险厌恶系数
  A = 0.5

  # 效用
  result['U'] = result.mean_rate - 0.5 * A * result.risk **2
  
  return result



#----------------------------- 画图 -----------------------------------#
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



#----------------------------- 蜡烛图维度 -----------------------------------#
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
