# -*- coding: utf-8 -*-
"""
Utilities used for machine learning perpose 

:autohr: Beichen Chen
"""
from quant import bc_util as util
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_scaler(scale_method='StandardScaler'):
"""
Get different kinds of scalers from scikit-learn

:param scale_method: scale method
:returns: scaler instance
:raises: none
"""
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

  elif scale_method == 'Normalizer':
    scaler = preprocessing.Normalizer()

  else:
    print(scale_method, ' not found')
    
  return scaler


def get_scaled_data(df, scaler):
"""
Get data scaled with specific kind of scaler

:param df: dataframe to be scaled
:param scaler: scaler used to scale the data
:returns: scaled dataframe
:raises: none
"""
  scaled_data = scaler.fit_transform(df)
  scaled_data = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
  
  return scaled_data
  

def get_train_test_data(scaled_data, input_dim, output_dim, test_size=0.1, is_shuffle=True, start=None, end=None, predict_idx=[]):
"""
Split data into trian/valid/test datasets

:param scaled data: scaled dataframe
:param input_dim: input columns
:param output_dim: output columns
:param test_size: size of test dataset
:param is_shuffle: whether to shuffle the data
:param start: start row of the data
:param end: end row of the data
:param predict_idx: rows used as test data (to be predicted)
:returns: datasets in dictionary
:raises: none
"""
  try:
    # training set
    train_data = scaled_data[start:end].copy()

    # predicting set
    predict_data = pd.DataFrame()
    predict_x = pd.DataFrame()
    predict_y = pd.DataFrame()
    if len(predict_idx) > 0:
      predict_idx = [util.string_2_time(x) for x in predict_idx]
      predict_data = scaled_data.loc[predict_idx, :].copy()
      predict_x = predict_data[input_dim].values.reshape(-1, len(input_dim))
      predict_y = predict_data[output_dim].values.reshape(-1, len(output_dim))

    # remove predicting set from training set
    if len(predict_idx) > 0:
      for idx in predict_idx:
        train_data.drop(idx, inplace=True)
    
    # split input/output
    x = train_data[input_dim].values
    y = train_data[output_dim].values
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=0, shuffle=is_shuffle)
    
    print('Train   Size: ', train_x.shape, train_y.shape)
    print('Test    Size: ', test_x.shape, test_y.shape)
    print('Predict Size: ', predict_x.shape, predict_y.shape)

  except Exception as e:
    print(e)
  
  return{
      'scaled_data': scaled_data, 'input_dim': input_dim, 'output_dim': output_dim,
      'train_x': train_x, 'train_y': train_y, 
      'test_x': test_x, 'test_y': test_y, 
      'predict_x': predict_x, 'predict_y': predict_y
  }


def build_dense_network(hidden_layers, neuron_units, input_shape, output_shape, hidden_act_func, output_act_func, loss_func, optimizer, result_metrics, dropout=False, dropout_rate=0.3):
"""
Construct dense neural network

:param hidden_layers: number of hidden layers
:param neuron_units: number of neurons in ecah layer
:param input_shape: input shape
:param output_shape: output shape
:param hidden_act_func: activation function used in hidden layer
:param output_act_func: activation function used in output layer
:param loss_func: loss function used in optimizer
:param optimizer: optimizer
:param result_metrics: result metrics used for evaluation
:param dropout: whether to add dropout layers
:param dropout_rate: dropout rate
:returns: keras sequential model
:raises: none
"""
  # create sequetial model
  model = keras.models.Sequential()

  # input layer
  model.add(keras.layers.Dense(units=neuron_units, input_shape=input_shape, activation=hidden_act_func))

  # hidden layers
  for i in range(hidden_layers):
    model.add(keras.layers.Dense(units=neuron_units, activation=hidden_act_func))

    # Dropout layers
    if dropout:
      if i % 2 == 0:
        model.add(keras.layers.Dropout(dropout_rate))

  # output layer
  model.add(keras.layers.Dense(units=output_shape, activation=output_act_func))

  # construct model with layers, loss function, optimizer and metrics
  model.compile(loss=loss_func, optimizer=optimizer, metrics=result_metrics) 

  return model         

