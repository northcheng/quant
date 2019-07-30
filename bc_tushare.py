# -*- coding: utf-8 -*-
"""
Utilities used in tushre API

:author: Beichen Chen
"""
import pandas as pd
import tushare as ts


def get_user_info(info_path='drive/My Drive/tushare_quant/'):
  """
  Get user information

  :param info_path: the path user information file stored in
  :returns: user information in dictionary
  :raises: none
  """
  user_info = pd.read_csv(info_path + 'user_info.csv')
  return user_info.astype('str').loc[0,:].to_dict()


def get_ts_client(info_path='drive/My Drive/tushare_quant/'):
  """
  Get tushre client

  :param infor_path: the path user information file stored in
  :returns: tushare client
  :raises: none
  """
  user_info = get_user_info(info_path=info_path)
  ts.set_token(user_info['token'])
  tsp = ts.pro_api()

  return tsp
