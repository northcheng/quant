# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import finance as mpf
from matplotlib.pylab import date2num

# 普通dataframe转时间序列数据
def df_2_timeseries(df, time_col='date'):
    df = df.set_index(time_col)
    df.index = pd.DatetimeIndex(df.index)
    return df

# 将时间字符串转化为时间对象
def string_2_time(string, date_format='%Y-%m-%d'):
    time_object = datetime.datetime.strptime(string, date_format)
    return time_object
 
# 将时间对象转化为时间字符串
def time_2_string(time_object, diff_days=0, date_format='%Y-%m-%d'):
    time_object = time_object + datetime.timedelta(days=diff_days)
    time_string = datetime.datetime.strftime(time_object, date_format)
    return time_string
 
# 直接在字符串上加减日期
def string_plus_day(string, diff_days, date_format='%Y-%m-%d'):
 
    # 字符串转日期, 加减天数
    time_object = string_2_time(string, date_format=date_format)
    time_string = time_2_string(time_object, diff_days, date_format=date_format)
    return time_string    

# 计算两个日期字符串之间的天数
def num_days_between(start_date, end_date, date_format='%Y-%m-%d'):
 
    # 将起止日期转为日期格式
    start_date = string_2_time(start_date, date_format)
    end_date = string_2_time(end_date, date_format)
    diff = end_date - start_date
    return diff.days

# 画多条折线图
def plot_data(selected_data, figsize=(15, 5)):
    
    # plot data
    plt.figure(figsize=figsize)
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    # 颜色设置
    cNorm = colors.Normalize(vmin=0, vmax=len(selected_data.columns))
    jet = cm = plt.get_cmap('jet')
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    for i in range(len(selected_data.columns)):
        col = selected_data.columns[i]
        
        plt.plot(selected_data.index, selected_data[col], 
                 label=col, color=scalarMap.to_rgba(i+1), linewidth=2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
    plt.grid(True)
    plt.xticks(rotation=90)   

# 逆转minmax_scale
def minmax_reverter(scaled_value, original_data, col):

  return scaled_value * (original_data[col].max() - original_data[col].min()) + original_data[col].min()

# 画蜡烛图函数
def plot_candlestick(df, num_days=50, figsize=(15,5), title=''):
  
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
    colorup='red', colordown='black'
  )
  plt.grid(True)
  plt.show()

# 计算涨跌幅/累计涨跌幅
# original_df: pandas.DataFrame, 原数据框
# dim: string, 要计算涨跌幅的列
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