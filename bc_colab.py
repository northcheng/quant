import os
import numpy as np
import pandas as pd
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

  stage = 'downloading_started'
  # 查看是否已存在下载好的文件, 若有则读取, 若没有则初始化
  try:
    stage = 'loading_downloaded_data'
    data = read_stock_data(sec_code, file_path=file_path, file_format=file_format, time_col=time_col)
    
    # 记录原始数据记录数
    init_len = len(data)
    if init_len > 0:
      start_date = util.time_2_string(data.index.max(), date_format='%Y%m%d')

    stage = 'downloading_new_data'
    # 下载最新数据
    tmp_data = web.DataReader(sec_code, source, start=start_date, end=end_date)
    data = data.append(tmp_data)

    stage = 'saving_new_data'
    # 保存数据
    data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
    data.to_csv(filename, index=False) 
    data = util.df_2_timeseries(data, time_col='Date')

    # 对比记录数量变化
    if is_print:
      final_len = len(data)
      diff_len = final_len - init_len
      print('%(sec_code)s: 最新日期%(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
          diff_len=diff_len, final_len=final_len, latest_date=data.index.max().date(), sec_code=sec_code))
  except Exception as e:
      print(sec_code, stage, e)

  if is_return:
      return data


# 读取股票数据
def read_stock_data(sec_code, file_path, file_format, time_col, source='google_drive', start_date=None, end_date=None, drop_cols=[], drop_none_digit=False, drop_na=False, sort_index=True):
  
  if source == 'google_drive':
    # 构建文件名
    filename = file_path + sec_code + file_format
  
    stage = 'reading_from_google_drive'
    try:

      stage = 'loading_downloaded_data'
      # 读取文件
      if file_format == '.csv':
        data = pd.read_csv(filename, encoding='utf8', engine='python')
      elif file_format == '.xlsx':
        data = pd.read_excel(filename)

      stage = 'transforming_to_timeseries'
      # 转化为时间序列
      data = util.df_2_timeseries(df=data, time_col=time_col)
      
      stage = 'handling_invalid_data'
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
      print(sec_code, stage, e)
      data = pd.DataFrame()

  elif source == 'web':
    stage = 'reading_from_pandas_datareader'
    try:
      data = web.DataReader(sec_code, 'yahoo', start=start_date, end=end_date)
    except Exception as e:
      print(sec_code, stage, e)
      data = pd.DataFrame()

  else:
    print('source %s not found' % source)
    data = pd.DataFrame()
    
  return data[start_date:end_date]

# # 下载列表中所有股票的数据
# def download_stock_list_data(sec_code_list, source, time_col='Date', start_date=None, end_date=None, file_path='drive/My Drive/stock_data_us/', file_format='.csv', is_print=True):
  
#   # 下载列表中的股票的数据
#   stock_data = web.DataReader(name=sec_code_list, data_source=source, start=start_date, end=end_date)
#   #   # 去重
#   #   stock_data = stock_data[stock_data.index.duplicated(keep='last')]
  
#   # 获取属性列表与代码列表
#   stock_attributes = stock_data.columns.levels[stock_data.columns.names.index('Attributes')].tolist()
#   stock_symbols = stock_data.columns.levels[stock_data.columns.names.index('Symbols')].tolist()
  
#   # 遍历每只股票:
#   for sec_code in stock_symbols:
    
#     # 处理最新数据
#     tmp_data = stock_data.loc[:, (tmp_attributes, sec_code)]
#     tmp_data.columns = tmp_attributes
  
#     # 查看是否已存在下载好的文件, 若有则读取, 若没有则初始化
#     filename = file_path + sec_code + file_format
#     try:
#       if os.path.exists(filename):
#           data = util.df_2_timeseries(pd.read_csv(filename), time_col=time_col)
#           start_date = util.time_2_string(data.index.max(), date_format='%Y%m%d')
#       else:
#           data = pd.DataFrame()
          
#       # 记录原始数据记录数
#       init_len = len(data)

#       # 添加新数据
#       data = data.append(tmp_data)

#       # 去重, 保存数据
#       data = data.reset_index().drop_duplicates(subset=time_col, keep='last')
#       data.to_csv(filename, index=False)

#       # 对比记录数量变化
#       if is_print:
#         final_len = len(data)
#         diff_len = final_len - init_len
#         print('%(sec_code)s: 最新日期%(latest_date)s, 新增记录 %(diff_len)s/%(final_len)s, ' % dict(
#             diff_len=diff_len, final_len=final_len, latest_date=data[time_col].max().date(), sec_code=sec_code))
#     except Exception as e:
#         print(sec_code, e)
  
# #@title 变量设置
# # 下载起始日期, 结束日期
# start_date = '2019-02-19' #@param {type:"date"}
# end_date = '2019-02-19' #@param {type:"date"}

# # 数据源
# download_source = 'yahoo'

# # 分批次下载
# download_len = len(download_list)
# batch_size = 500
# loops = round(download_len / batch_size)

# for i in range(loops):
#   print(i+1, '/', loops)
#   sec_code_list = download_list[batch_size*i : batch_size*(i+1)]
#   download_stock_list_data(sec_code_list=sec_code_list, source=download_source, start_date=start_date, end_date=end_date)  