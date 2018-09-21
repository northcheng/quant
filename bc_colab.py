import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quant import bc_util as util


# 读取股票数据
def read_stock_data(sec_code, file_path, file_format, time_col, drop_cols=[], drop_none_digit=True, drop_na=True, sort_index=True):
  
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

# # 读取googole drive上的数据文件, 创建训练与测试数据
# def get_train_test_data(sec_code, prediction_date, drive_path='drive/My Drive', prediction_field='close', dims={'stock_data': 'index', 'stock_money_flow': 'date', 'stock_basic': 'trade_date'}, drop_col=['sec_code', 'ts_code'],  start_date=None, end_date=None, is_shuffle=True, is_classification=False):
  
#   # 读取原始数据文件
#   raw_data = {}
#   for d in dims.keys():
#     # 构建文件名
#     filename = drive_path + '/' + d + '/' + sec_code + '.csv'
#     # 获取时间列
#     time_col = dims[d]
#     # 读取数据, 转化为时间序列数据
#     raw_data[d] = util.df_2_timeseries(pd.read_csv(filename), time_col=time_col)
#     # 删除冗余列
#     for col in raw_data[d].columns:
#       if col in drop_col:
#         raw_data[d].drop(col, axis=1, inplace=True)
#     # 打印结果
#     print('读取', filename, ':', len(raw_data[d]))
  
#   # 合并数据
#   data = pd.DataFrame()
#   for rd in raw_data.keys():
#     data = pd.merge(data, raw_data[rd], how='outer', left_index=True, right_index=True)
  
#   # 预处理
#   data.dropna(axis=0, inplace=True)
#   data.sort_index(inplace=True)
  
#   # 定义输入和输出的维度数
#   input_dim = len(data.columns)  # 输入的维度: 所有的列
#   output_dim = 1                 # 输出的维度: n个, 表示未来1...n天的收盘价
  
#   # 为数据添加未来 n 日的收盘价格数据
#   for n in range(output_dim):
#     data['next_%s'%(n+1)] = data[prediction_field].shift(-(n+1))
#     data.fillna(method='ffill', inplace=True)

#     # 如果是分类问题则将输出转化为 0/1 即 跌/涨
#     if is_classification:
#       for n in range(output_dim):
#         data['next_%s'%(n+1)] = (data['next_%s'%(n+1)] > data[prediction_field]).astype('int32')
#       k = 2
#     else:
#       k = 0

#   # 标准化
#   scaler = MinMaxScaler()
#   scaled_data = pd.DataFrame(scaler.fit_transform(data.values),columns=data.columns, index=data.index)
  
#   if is_classification:
#     for n in range(output_dim):
#       scaled_data['next_%s'%(n+1)] = scaled_data['next_%s'%(n+1)].astype('int32')

#   # 分为训练集与测试集
#   train_data = scaled_data[start_date:end_date]
#   prediction_data = scaled_data.loc[prediction_date]
  
#   # 分开输入与输出
#   x = train_data.values[:, :input_dim]
#   y = train_data.values[:, -output_dim:]

#   if is_shuffle:
#     x, y = shuffle(x, y, random_state=0)
  
#   # # 分为训练集与测试集
#   train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0, random_state=0)
#   prediction_x = prediction_data[:input_dim].values.reshape(-1, input_dim)
  
#   print('训练数据: ', train_x.shape, train_y.shape)
#   print('测试数据: ', test_x.shape, test_y.shape)
#   print('预测数据: ', prediction_x.shape)
  
#   return {
#       'sec_code': sec_code, 'k': k,
#       'input_dim': input_dim, 'output_dim': output_dim,
#       'data': data, 'scaled_data': scaled_data, 
#       'train_x': train_x, 'train_y': train_y, 
#       'test_x': test_x, 'test_y': test_y, 
#       'prediction_x': prediction_x
#   }