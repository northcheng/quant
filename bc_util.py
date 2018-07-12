# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime

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
def plot_data(selected_data):
    
    # plot data
    plt.figure(figsize=(15,4))
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

# # # 聚宽专用
# # 计算股票在指定期间内的涨跌幅
# def cal_price_change(stock_list, start_date, end_date):
#     stock_price = get_price(security=stock_list, start_date=start_date, end_date=end_date, fields='close').fillna(method='bfill')
#     stock_price_change = ((stock_price['close'].tail(1).values - stock_price['close'].head(1).values) / stock_price['close'].head(1).values) * 100
#     stock_price_change = pd.DataFrame(stock_price_change, columns=stock_price['close'].columns).T
#     stock_price_change.columns = ['price_change_pct']
    
#     return stock_price_change

# # 保存数据
# def df_2_csv(df, filename):
    
#     # 把 DataFrame 表保存到文件
#     write_file('%s.csv' % filename, df.to_csv(), append=False) #写到文件中
    
# # 获取指数成分股信息
# def get_index_stocks_info(index, all_sec_info):
    
#     index_stocks = get_index_stocks(index)
#     index_stocks_info = all_sec_info.loc[index_stocks]
    
#     return index_stocks_info