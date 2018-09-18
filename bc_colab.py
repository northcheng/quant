import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quant import bc_util as util

# 读取googole drive上的数据文件, 创建训练与测试数据
def get_train_test_data(sec_code, prediction_date, drive_path='drive/My Drive', prediction_field='close', dims={'stock_data': 'index', 'stock_money_flow': 'date'}, drop_col=['sec_code'],  start_date=None, end_date=None, is_shuffle=True, is_classification=False):
  
  # 读取原始数据文件
  raw_data = {}
  for d in dims.keys():
    # 构建文件名
    filename = drive_path + '/' + d + '/' + sec_code + '.csv'
    # 获取时间列
    time_col = dims[d]
    # 读取数据, 转化为时间序列数据
    raw_data[d] = util.df_2_timeseries(pd.read_csv(filename), time_col=time_col)
    # 删除冗余列
    for col in raw_data[d].columns:
      if col in drop_col:
        raw_data[d].drop(col, axis=1, inplace=True)
    # 打印结果
    print('读取', filename, ':', len(raw_data[d]))
  
  # 合并数据
  data = pd.DataFrame()
  for rd in raw_data.keys():
    data = pd.merge(data, raw_data[rd], how='outer', left_index=True, right_index=True)
  
  # 预处理
  data.dropna(axis=0, inplace=True)
  data.sort_index(inplace=True)
  
  # 定义输入和输出的维度数
  input_dim = len(data.columns)  # 输入的维度: 所有的列
  output_dim = 1                 # 输出的维度: n个, 表示未来1...n天的收盘价
  
  # 为数据添加未来 n 日的收盘价格数据
  for n in range(output_dim):
    data['next_%s'%(n+1)] = data[prediction_field].shift(-(n+1))
    
    # 如果是分类问题则将输出转化为 0/1 即 跌/涨
    if is_classification:
      data['next_%s'%(n+1)] = (data['next_%s'%(n+1)] > data[prediction_field]).astype('int32')
      k = 2
    else:
      k = 0

  data.fillna(method='ffill', inplace=True)

  # 标准化
  scaler = MinMaxScaler()
  scaled_data = pd.DataFrame(scaler.fit_transform(data.values),columns=data.columns, index=data.index)
  
  # 分为训练集与测试集
  train_data = scaled_data[start_date:end_date]
  prediction_data = scaled_data.loc[prediction_date]
  
  # 分开输入与输出
  x = train_data.values[:, :input_dim]
  if is_classification:
    y = scaled_data.values[:, -output_dim:].astype('int32')
  else:
    y = train_data.values[:, -output_dim:]

  if is_shuffle:
    x, y = shuffle(x, y, random_state=0)
  
  # # 分为训练集与测试集
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0, random_state=0)
  prediction_x = prediction_data[:input_dim].values.reshape(-1, input_dim)
  
  print('训练数据: ', train_x.shape, train_y.shape)
  print('测试数据: ', test_x.shape, test_y.shape)
  print('预测数据: ', prediction_x.shape)
  
  return {
      'sec_code': sec_code,
      'input_dim': input_dim, 'output_dim': output_dim,
      'data': data, 'scaled_data': scaled_data, 
      'train_x': train_x, 'train_y': train_y, 
      'test_x': test_x, 'test_y': test_y, 
      'prediction_x': prediction_x
  }