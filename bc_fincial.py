# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
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
  
  # 计算涨跌率
  df[previous_dim] = df[dim].shift(period)
  df[dim_rate] = (df[dim] -  df[previous_dim]) /df[previous_dim] * 100
  
  # 添加累计维度列
  if is_add_acc_rate:
    
    df[dim_acc_rate] = 0
  
    # 计算累计值
    idx = df.index.tolist()
    for i in range(1, len(df)):
      current_idx = idx[i]
      previous_idx = idx[i-1]
      current = df.loc[current_idx, dim_rate]
      previous = df.loc[previous_idx, dim_acc_rate]

      # 如果符号相同则累加, 否则重置
      if previous * current > 0:
        df.loc[current_idx, dim_acc_rate] = current + previous
      else:
        df.loc[current_idx, dim_acc_rate] = current
    
    df.dropna(inplace=True) 
    df.drop(previous_dim, axis=1, inplace=True)

    return df


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




