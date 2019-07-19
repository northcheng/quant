# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# 将时间字符串转化为时间对象
def string_2_time(string, date_format='%Y-%m-%d'):
    time_object = datetime.datetime.strptime(string, date_format)
    return time_object
 

# 将时间对象转化为时间字符串
def time_2_string(time_object, diff_days=0, date_format='%Y-%m-%d'):
    time_object = time_object + datetime.timedelta(days=diff_days)
    time_string = datetime.datetime.strftime(time_object, date_format)
    return time_string


# 将时间戳转化为时间对象
def timestamp_2_time(timestamp, unit='ms', timezone='CN'):

    if timezone == 'CN':
        tz = pytz.timezone('Asia/Chongqing')
    else:
        tz = pytz.utc

    if unit == 'ms':
        timestamp = int(timestamp/1000)
    if unit == 'us':
        timestamp = int(timestamp/1000000)

    time_object = datetime.datetime.fromtimestamp(int(timestamp), tz)
    return time_object
 

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
  

# 普通dataframe转时间序列数据
def df_2_timeseries(df, time_col='date'):
    df = df.set_index(time_col)
    df.index = pd.DatetimeIndex(df.index)
    return df


# 画多条折线图
def plot_data(df, columns, start=None, end=None, figsize=(20, 5), colormap='tab10'):
    
    # 选择数据
    selected_data = df[start:end][columns]

    # 创建图像
    plt.figure(figsize=figsize)
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    # 颜色设置
    cNorm = colors.Normalize(vmin=0, vmax=len(selected_data.columns))
    cm = plt.get_cmap(colormap)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    for i in range(len(selected_data.columns)):
        col = selected_data.columns[i]
        
        plt.plot(selected_data.index, selected_data[col], 
                 label=col, color=scalarMap.to_rgba(i+1), linewidth=2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
    plt.grid(True)
    plt.xticks(rotation=90)   
