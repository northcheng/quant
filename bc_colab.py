import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quant import bc_util as util


# 读取股票数据
def read_stock_data(sec_code, file_path, file_format, time_col, drop_cols=[], drop_none_digit=False, drop_na=False, sort_index=True):
  
  # 构建文件名
  filename = file_path + sec_code + file_format
  
  # 读取文件
  if file_format == '.csv':
    data = pd.read_csv(filename)
  elif file_format == '.xlsx':
    data = pd.read_excel(filename)
    
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
  
  return data


# 转化为训练/测试/预测数据
def get_train_test_data(data, predict_dim, predict_date, start_date=None, end_date=None, is_classification=False, is_shuffle=True, test_size=0):
  
  # 定义输入和输出维度的长度
  input_dim = len(data.columns)
  output_dim = 1
  
  # 为数据添加ground truth
  for n in range(output_dim):
    data['next_%s'%(n+1)] = data[predict_dim].shift(-(n+1))
    data.fillna(method='ffill', inplace=True)
  
    # 如果是分类为问题, 则转化为涨跌(1/0)
    if is_classification:
      for n in range(output_dim):
        data['next_%s'%(n+1)] = (data['next_%s'%(n+1)] > data[predict_dim]).astype('int32')
      k = 2
    else:
      k = 0
      
  # 标准化
  scaler = MinMaxScaler()
  scaled_data = pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)
  
  if is_classification:
    for n in range(output_dim):
      scaled_data['next_%s'%(n+1)] = scaled_data['next_%s'%(n+1)].astype('int32')
      
  # 分为训练集与预测集
  train_data = scaled_data[start_date:end_date]
  predict_data = scaled_data.loc[predict_date]
  
  # 分为输入与输出
  x = train_data.values[:, :input_dim]
  y = train_data.values[:, -output_dim:]
  
  # 打乱顺序
  if is_shuffle:
    x, y = shuffle(x, y, random_state=0)
  
  # 分为训练集与测试集
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=0)
  predict_x = predict_data[:input_dim].values.reshape(-1, input_dim)
  
  print('训练数据: ', train_x.shape, train_y.shape)
  print('测试数据: ', test_x.shape, test_y.shape)
  print('预测数据: ', predict_x.shape)
  
  return{
      'input_dim': input_dim, 'output_dim': output_dim, 'k': k,
      'data': data, 'scaled_data': scaled_data, 
      'train_x': train_x, 'train_y': train_y, 
      'test_x': test_x, 'test_y': test_y, 
      'predict_x': predict_x
  }

