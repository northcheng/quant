# -*- coding: utf-8 -*-
"""
Utilities used for machine learning perpose 

:autohr: Beichen Chen
"""
from quant import bc_util as util

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn, Tensor
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split


# dataset definition
class ClassificationDataset(Dataset):
  
  # load the dataset
  def __init__(self, path, feature_column=None, result_column=None, scaler=None):

    # load the csv file as a dataframe
    df = pd.read_csv(path)
    
    if feature_column is None:
      feature_column = df.columns[:-1]
    if result_column is None:
      result_column = df.columns[-1]
    
    # store the inputs and outputs
    self.X = df[feature_column].values
    self.y = df[result_column].values
    
    # ensure input data is floats
    self.X = self.X.astype('float32')
    
    if scaler is None:
      pass
    elif scaler == 'MinMax':
      self.X = preprocessing.MinMaxScaler().fit_transform(self.X)
    elif scaler == 'Standard':
      self.X = preprocessing.StandardScaler().fit_transform(self.X)
    else:
      pass

    # label encode target and ensure the values are floats
    le = preprocessing.LabelEncoder()
    self.y = le.fit_transform(self.y)
    print(le.classes_)

  # number of rows in the dataset
  def __len__(self):
    return len(self.X)

  # get a row at an index
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]

  # get indexes for train and test rows
  def get_splits(self, n_test=0.2):

    # determine sizes
    test_size = round(n_test * len(self.X))
    train_size = len(self.X) - test_size

    # calculate the split
    return random_split(self, [train_size, test_size])

      
class SimpleClassifier(nn.Module):
  
  def __init__(self, n_in, n_out):
    super(SimpleClassifier, self).__init__()
    
    # self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(n_in, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, n_out),
    )

  def forward(self, x):
    
    # x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    
    return logits
  

# prepare the dataset
def prepare_data(dataset):

  # calculate split
  train, test = dataset.get_splits()

  # prepare data loaders
  train_dl = DataLoader(train, batch_size=128, shuffle=True, drop_last=True)
  test_dl = DataLoader(test, batch_size=128, shuffle=True, drop_last=True) 

  return train_dl, test_dl


# train the model
def train_classifier(train_dl, model, epoch=100):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  # define the optimization
  criterion = nn.CrossEntropyLoss()

  # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = Adam(model.parameters())

  # enumerate epochs
  for epoch in range(epoch):

    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
      
      inputs = inputs.to(device)
      targets = targets.long().to(device)

      # clear the gradients
      optimizer.zero_grad()

      # compute the model output
      yhat = model(inputs)

      # calculate loss
      loss = criterion(yhat, targets)

      # credit assignment
      loss.backward()

      # update model weights
      optimizer.step()

    if epoch % 10 == 0:
      print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))


# evaluate the model
def evaluate_classifier(test_dl, model):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")
  
  predictions, actuals = [], []
  for i, (inputs, targets) in enumerate(test_dl):
    
    # evaluate the model on the test set
    inputs = inputs.to(device)
    yhat = model(inputs)
    
    # retrieve numpy array
    yhat = yhat.cpu().detach().numpy()
    actual = targets.numpy()
    
    # convert to class labels
    yhat = np.argmax(yhat, axis=1)
    
    # reshape for stacking
    actual = actual.reshape((len(actual), 1))
    yhat = yhat.reshape((len(yhat), 1))
    
    # store
    predictions.append(yhat)
    actuals.append(actual)
  
  predictions, actuals = np.vstack(predictions), np.vstack(actuals)
  
  # calculate accuracy
  acc = accuracy_score(actuals, predictions)
  
  return acc


# make a class prediction for one row of data
def predict(row, model):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  # print(f"Using {device} device")
  
  # convert row to data
  row = Tensor([row]).to(device)
  
  # make prediction
  pred = model(row)

  # get probability
  pred_probab = torch.nn.Softmax(dim=1)(pred).detach().cpu().numpy()

  # get label
  pred_label = np.argmax(pred.detach().cpu().numpy(), axis=1)

  # result
  label = pred_label[0]
  probability = pred_probab[0][label]
  
  return (label, probability)


# dataset definition
class RegressionDataset(Dataset):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")
  
  # load the dataset
  def __init__(self, path, feature_column=None, result_column=None, scaler=None):

    # load the csv file as a dataframe
    df = pd.read_csv(path)
    
    if feature_column is None:
      feature_column = data.columns[:-1]
    if result_column is None:
      result_column = data.columns[-1]
    
    # store the inputs and outputs
    self.X = df[feature_column].values
    self.y = df[result_column].values
    
    # ensure input data is floats
    self.X = self.X.astype('float32')
    
    if scaler is None:
      pass
    elif scaler == 'MinMax':
      self.X = preprocessing.MinMaxScaler().fit_transform(self.X)
    elif scaler == 'Standard':
      self.X = preprocessing.StandardScaler().fit_transform(self.X)
    else:
      pass

  # number of rows in the dataset
  def __len__(self):
    return len(self.X)

  # get a row at an index
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]

  # get indexes for train and test rows
  def get_splits(self, n_test=0.2):

    # determine sizes
    test_size = round(n_test * len(self.X))
    train_size = len(self.X) - test_size

    # calculate the split
    return random_split(self, [train_size, test_size])
  

class SimpleRegressor(nn.Module):
  
  def __init__(self, n_in, n_out):
    super(SimpleRegressor, self).__init__()
    
    # self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(n_in, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, n_out),
    )

  def forward(self, x):
    
    # x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    
    return logits  

  
# train the model
def train_regressor(train_dl, model, epoch=100):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")

  # define the optimization
  criterion = nn.MSELoss()

  # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = Adam(model.parameters())

  # enumerate epochs
  for epoch in range(epoch):

    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
      
      inputs = inputs.to(device)
      targets = targets.float().to(device)

      # clear the gradients
      optimizer.zero_grad()

      # compute the model output
      yhat = model(inputs)

      # calculate loss
      loss = criterion(yhat, targets)

      # credit assignment
      loss.backward()

      # update model weights
      optimizer.step()

    if epoch % 10 == 0:
      print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
      
      
# evaluate the model
def evaluate_regressor(test_dl, model):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  # predictions, actuals = [], []
  mse = []
  for i, (inputs, targets) in enumerate(test_dl):
    
    # evaluate the model on the test set
    inputs = inputs.to(device)
    yhat = model(inputs)
    targets = targets.float().to(device)
    
    # store
    mse.append(nn.functional.mse_loss(yhat, targets))
  
  return sum(mse)/len(mse)


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

  elif scale_method == 'PowerTransformer':
    scaler = preprocessing.PowerTransformer()

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

