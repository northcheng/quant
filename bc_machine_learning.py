# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from quant import bc_util as util

# 获取标准化器
def get_scaler(scale_method='StandardScaler', optional_args={}):

  scaler = None

  if scale_method == 'StandardScaler':
    scaler = preprocessing.StandardScaler()

  elif scale_method == 'MinMaxScaler':
    scaler = preprocessing.MinMaxScaler()

  elif scale_method == 'MaxAbsScaler':
    scaler = preprocessing.MaxAbsScaler()

  elif scale_method == 'RobustScaler':
    scaler = preprocessing.RobustScaler()  

  elif scale_method == 'QuantileTransformer':
    scaler = preprocessing.QuantileTransformer()

  elif scale_method == 'PowerTransformer':
    method = optional_args.get('method')
    standardize = optional_args.get('standardize')
    if method is None:
      method = 'box-cox'
    if standardize is None:
      standardize = False
    scaler = preprocessing.PowerTransformer(method=method, standardize=standardize)

  elif scale_method == 'Normalizer':
    scaler = preprocessing.Normalizer()

  else:
    print(scale_method, ' not found')
    
  return scaler


# 将已经标准化后的数据转化为训练/测试/预测集
def get_train_test_data(scaled_data, input_dim, output_dim, test_size=0.1, is_shuffle=True, start=None, end=None, predict_idx=[]):
    
  try:    
    
    # 预测数据
    predict_idx = [util.string_2_time(x) for x in predict_idx]
    predict_data = scaled_data.loc[predict_idx, :].copy()
    
    # 训练数据(删除预测数据)
    train_data = scaled_data[start:end].copy()
    for idx in predict_idx:
      train_data.drop(idx, inplace=True) 

    # 分为输入与输出
    x = train_data[input_dim].values
    y = train_data[output_dim].values
    
    # 分为训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=0, shuffle=is_shuffle)
    predict_x = predict_data[input_dim].values.reshape(-1, len(input_dim))
    predict_y = predict_data[output_dim].values.reshape(-1, len(output_dim))

    print('训练数据: ', train_x.shape, train_y.shape)
    print('测试数据: ', test_x.shape, test_y.shape)
    print('预测数据: ', predict_x.shape, predict_y.shape)

  except Exception as e:
    print(e)
  
  return{
      'scaled_data': scaled_data, 'input_dim': input_dim, 'output_dim': output_dim,
      'train_x': train_x, 'train_y': train_y, 
      'test_x': test_x, 'test_y': test_y, 
      'predict_x': predict_x, 'predict_y': predict_y
  }