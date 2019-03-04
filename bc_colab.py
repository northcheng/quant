import numpy as np
import pandas as pd
import os
import pandas_datareader.data as web 
from quant import bc_util as util
from google.colab import drive


# 挂载Google drive
def mount_google_drive(destination_path='content/drive', force_remount=False):

  drive.mount('/content/drive', force_remount=force_remount)


# 下载股票数据
def download_stock_data(sec_code, source, time_col='Date', start_date=None, end_date=None, file_path='drive/My Drive/stock_data_us/', filename=None, file_format='.csv', is_return=False, is_print=True):
  
  # 获取个股信息
  if filename is None:
    filename = file_path + sec_code + file_format
  else:
    filename = file_path + filename + file_format

  # 查看是否已存在下载好的文件, 若有则读取, 若没有则初始化
  try:
    data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col=time_col)
    # if os.path.exists(filename):
    #     data = util.df_2_timeseries(pd.read_csv(filename), time_col=time_col)
    #     start_date = util.time_2_string(data.index.max(), date_format='%Y%m%d')
    # else:
    #     data = pd.DataFrame()
  
    # 记录原始数据记录数
    init_len = len(data)
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y%m%d')

    # 下载最新数据
    tmp_data = web.DataReader(sec_code, source, start=start_date, end=end_date)
    data = data.append(tmp_data)

    # 保存数据
    date = data.reset_index().drop_duplicates(subset=time_col, keep='last')
    data.to_csv(filename, index=False) 

    # 对比记录数量变化
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('%(sec_code)s: 最新日期%(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
          diff_len=diff_len, final_len=final_len, latest_date=data[time_col].max().date(), sec_code=sec_code))
  except Exception as e:
      print(sec_code, e)

  if is_return:
      return util.df_2_timeseries(data, time_col='Date') 


# 读取股票数据
def read_stock_data(sec_code, file_path, file_format, time_col, drop_cols=[], drop_none_digit=False, drop_na=False, sort_index=True):
  
  # 构建文件名
  filename = file_path + sec_code + file_format
  
  # 读取文件
  if file_format == '.csv':
    data = pd.read_csv(filename)
  elif file_format == '.xlsx':
    data = pd.read_excel(filename)
    
  try:
    # 转化为时间序列
    data = util.df_2_timeseries(df=data, time_col=time_col)
    
    # [可选]删除非数值列
    if drop_none_digit:
      none_digit_cols = []
      for col in data.columns:
        none_digit_cols.append(not isinstance(data[col].values[0], (float, int)))
      none_digit_cols = data.columns[none_digit_cols].tolist()
      drop_cols += none_digit_cols
    
    # [可选]删除NA列
    if drop_na:
      data.dropna(axis=1, inplace=True)
    
    # [可选]删除指定列
    data.drop(drop_cols, axis=1, inplace=True)
    
    # [可选]重新排序index
    if sort_index:
      data.sort_index(inplace=True)
  except Exception as e:
    print(sec_code, e)
    data = pd.DataFrame()
    
  return data