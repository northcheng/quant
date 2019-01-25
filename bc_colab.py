import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quant import bc_util as util
from google.colab import drive

# 挂载Google drive
def mount_google_drive(destination_path='content/drive'):

  drive.mount('/content/drive')


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
    data = None
    
  return data