import os
import numpy as np
import pandas as pd
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
    
    # 记录原始数据记录数
    init_len = len(data)
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y-%m-%d')

    # 下载数据
    stage = 'appending_new_data'
    tmp_data = web.DataReader(sec_code, source, start=start_date, end=end_date)
    data = data.append(tmp_data)

    # 保存数据
    stage = 'saving_data'
    data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
    data.to_csv(filename, index=False) 
    data = util.df_2_timeseries(data, time_col=time_col)

    # 对比记录数量变化
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('%(sec_code)s: 最新日期%(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
          diff_len=diff_len, final_len=final_len, latest_date=data.index.max().date(), sec_code=sec_code))
  except Exception as e:
      print(sec_code, stage, e)

  # 返回数据
  if is_return:
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
    init_len = len(data)
    if os.path.exists(filename):  
      old_data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col=time_col)
      init_len = len(old_data)  
    
    # 记录原始数据记录数
    if init_len > 0:
      start_date = util.time_2_string(old_data.index.max(), date_format='%Y-%m-%d')
      
    # 下载数据
    stage = 'appending_new_data'
    tmp_len = download_limit
    while tmp_len >= download_limit:  
      tmp_data = quote_client.get_bars([sec_code], begin_time=start_date, end_time=end_date, limit=download_limit)
      tmp_len = len(tmp_data)
      data = tmp_data.append(data)
      end_date = int(tmp_data.time.min())
      # if is_print:
      #   print(start_date, util.timestamp_2_time(end_date))
    
    # 处理下载的数据
    data.drop('symbol', axis=1, inplace=True)
    data.time = data.time.apply(lambda x: util.timestamp_2_time(x).date())
    data = util.df_2_timeseries(df=data, time_col='time')
    
    # 附上已有数据
    if init_len > 0:
      data = old_data.append(data)
    
    # 去重，保存数据
    stage = 'saving_data'
    data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
    data.sort_values(by=time_col)
    data.to_csv(filename, index=False) 
    data = util.df_2_timeseries(data, time_col=time_col)
    
    # 对比记录数量变化
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('%(sec_code)s: 最新日期%(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
          diff_len=diff_len, final_len=final_len, latest_date=data.index.max().date(), sec_code=sec_code))
      
  except Exception as e:
    print(sec_code, stage, e)   
    
  # 返回数据
  if is_return:
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

