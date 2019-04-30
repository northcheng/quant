import os
import numpy as np
import pandas as pd
import time
import pandas_datareader.data as web 
from quant import bc_util as util
from google.colab import drive


# 挂载 Google drive
def mount_google_drive(destination_path='content/drive', force_remount=False):

  drive.mount('/content/drive', force_remount=force_remount)



# 下载股票数据
def download_stock_data(sec_code, source='yahoo', time_col='Date', start_date=None, end_date=None, file_path='drive/My Drive/stock_data_us/', file_format='.csv', is_return=False, is_print=True):
  
  # 构建股票数据文件名
  filename = file_path + sec_code + file_format
  
  # 下载开始
  stage = 'downloading_started'
  try:
    # 查看是否已存在下载好的文件, 若有则读取, 若没有则初始化
    stage = 'loading_existed_data'
    data = pd.DataFrame()
    if os.path.exists(filename):
      data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col=time_col)
    
    # 记录原始数据记录数, 更新下载的起始日期
    init_len = len(data)
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y-%m-%d')

    # 下载更新新下载的数据并保存
    stage = 'appending_new_data'
    tmp_data = web.DataReader(sec_code, source, start=start_date, end=end_date)
    if len(tmp_data) > 0:
      data = data.append(tmp_data, sort=False)

      # 保存数据
      stage = 'saving_data'
      data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
      data.to_csv(filename, index=False) 
      
    # 对比记录数量变化
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('%(sec_code)s: %(first_date)s - %(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
        diff_len=diff_len, final_len=final_len, first_date=data[time_col].min().date(), latest_date=data[time_col].max().date(), sec_code=sec_code))
  except Exception as e:
      print(sec_code, stage, e)

  # 返回数据
  if is_return:
    data = util.df_2_timeseries(data, time_col=time_col)
    return data 



# 从老虎API下载股票数据
def download_stock_data_from_tiger(sec_code, quote_client, start_date=None, end_date=None, download_limit=1200, time_col='time', file_path='drive/My Drive/stock_data_us/', file_format='.csv', is_return=False, is_print=True):
  
  # 构建股票数据文件名
  filename = file_path + sec_code + file_format
  
  # 下载开始
  stage = 'downloading_started'
  try:
    # 查看是否已存在下载好的文件, 若有则读取, 若没有则初始化
    stage = 'loading_existed_data'
    data = pd.DataFrame()
    if os.path.exists(filename):  
      data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col='Date')
      
    # 记录原始数据记录数, 更新下载起始日期
    init_len = len(data)  
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y-%m-%d')
      
    # 从老虎API下载数据
    stage = 'downloading_new_data'

    # 将开始结束时间转化为时间戳
    if start_date is not None:
      begin_time = round(time.mktime(util.string_2_time(start_date).timetuple()) * 1000)
    else:
      begin_time = 0
    if end_date is not None:
      end_time = round(time.mktime(util.string_2_time(end_date).timetuple()) * 1000)
    else:
      end_time = round(time.time() * 1000)

    # 开始下载数据
    tmp_len = download_limit
    new_data = pd.DataFrame()
    while tmp_len >= download_limit:  
      tmp_data = quote_client.get_bars([sec_code], begin_time=begin_time, end_time=end_time, limit=download_limit)
      tmp_len = len(tmp_data)
      new_data = tmp_data.append(new_data)
      end_time = int(tmp_data.time.min())
    
    # 处理下载的数据
    stage = 'processing_new_data'
    if len(new_data) > 0:
      new_data.drop('symbol', axis=1, inplace=True)
      new_data[time_col] = new_data[time_col].apply(lambda x: util.timestamp_2_time(x).date())
      new_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'time': 'Date'}, inplace=True)
      new_data['Adj Close'] = new_data['Close']
      time_col = 'Date'
      new_data = util.df_2_timeseries(df=new_data, time_col=time_col)
    
      # 附上已有数据
      data = data.append(new_data, sort=False)

      # 去重，保存数据
      stage = 'saving_data'
      data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
      data.sort_values(by=time_col,  )
      data.to_csv(filename, index=False) 
      
    # 对比记录数量变化
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('[From Tiger]%(sec_code)s: %(first_date)s - %(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
        diff_len=diff_len, final_len=final_len, first_date=data[time_col].min().date(), latest_date=data[time_col].max().date(), sec_code=sec_code))
      
  except Exception as e:
    print(sec_code, stage, e)   
    
  # 返回数据
  if is_return:
    data = util.df_2_timeseries(data, time_col=time_col)
    return data



# 读取股票数据
def read_stock_data(sec_code, time_col, file_path='drive/My Drive/stock_data_us/', file_format='.csv', source='google_drive', start_date=None, end_date=None, drop_cols=[], drop_na=False, sort_index=True):
  
  try:
    # 从 Google drive中读取股票数据
    if source == 'google_drive':
    
      # 构建文件名
      filename = file_path + sec_code + file_format
      if not os.path.exists(filename):
        print(filename, ' not exists')
        data = pd.DataFrame()

      else:
        # 读取数据
        stage = 'reading_from_google_drive'
        if file_format == '.csv':
          data = pd.read_csv(filename, encoding='utf8', engine='python')
        elif file_format == '.xlsx':
          data = pd.read_excel(filename)

        # 转化为时间序列
        stage = 'transforming_to_timeseries'
        data = util.df_2_timeseries(df=data, time_col=time_col)
        
        # 处理异常数据
        stage = 'handling_invalid_data'
        # 删除指定列
        data.drop(drop_cols, axis=1, inplace=True)
        # 删除NA列
        if drop_na:
          data.dropna(axis=1, inplace=True)
        # 重新排序index
        if sort_index:
          data.sort_index(inplace=True)

    # 从网络上下载股票数据
    elif source == 'web':

      # 下载数据
      stage = 'reading_from_pandas_datareader'
      data = web.DataReader(sec_code, 'yahoo', start=start_date, end=end_date)

    else:
      print('source %s not found' % source)
      data = pd.DataFrame()
  except Exception as e:
    print(sec_code, stage, e)
    data = pd.DataFrame()

  return data[start_date:end_date]

