# -*- coding: utf-8 -*-
"""
Generally used utilities

:author: Beichen Chen
"""
import pandas as pd
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


#----------------------- Date manipulation -----------------------#
def string_2_time(string, date_format='%Y-%m-%d'):
  """
  Convert date string to datetime instance

  :param string: date string
  :param date_format: the format of the date string
  :returns: datetime instance
  :raises: none
  """
  time_object = datetime.datetime.strptime(string, date_format)
  return time_object
 

def time_2_string(time_object, diff_days=0, date_format='%Y-%m-%d'):
  """
  Convert datetime instance to date string, with plus/minus certain days

  :param time_object: datetime instance
  :param diff_days: days need to be added or reduced
  :param date_format: the desired format of date string
  :returns: date string
  :raises: none
  """
  time_object = time_object + datetime.timedelta(days=diff_days)
  time_string = datetime.datetime.strftime(time_object, date_format)
  return time_string


def timestamp_2_time(timestamp, unit='ms', timezone='CN'):
  """
  Convert pytz timestamp instance to datetime instance

  :param timestamp: timestamp instance
  :param unit: unit of the timestamp
  :param timezone: timezone of the timestamp
  :returns: datetime instance
  :raises: none
  """
  if timezone == 'CN':
    tz = pytz.timezone('Asia/Chongqing')
  else:
    tz = pytz.utc

  if unit == 'ms':
    timestamp = int(timestamp/1000)
  if unit == 'us':
    timestamp = int(timestamp/1000000)

  time_object = datetime.datetime.fromtimestamp(int(timestamp), tz)
  return time_object


def string_plus_day(string, diff_days, date_format='%Y-%m-%d'):
  """
  Add or reduce days on a date string

  :param string: a date string
  :param diff_days: days that need to be added or reduced
  :param date_format: format of the date string
  :returns: date string
  :raises: none
  """
  # 字符串转日期, 加减天数
  time_object = string_2_time(string, date_format=date_format)
  time_string = time_2_string(time_object, diff_days, date_format=date_format)
  return time_string    


def num_days_between(start_date, end_date, date_format='%Y-%m-%d'):
  """
  Calculate the number of days between 2 date strings

  :param start_date: date string of start date
  :param end_date: date string of end date
  :date_format: format of date strings
  :returns: number of days between start/end date
  :raises: none
  """
  # convert date strings to datetime instances
  start_date = string_2_time(start_date, date_format)
  end_date = string_2_time(end_date, date_format)
  
  # calculate the difference between 2 date
  diff = end_date - start_date
  return diff.days


#----------------------- Dataframe manipulation --------------------#
def df_2_timeseries(df, time_col='date'):
  """
  Convert dataframe to timeseries-dataframe

  :param df: dataframe
  :param time_col: the column in the dataframe which contains time information
  :returns: timeseries dataframe with time_col as index
  :raises: none
  """
  df = df.set_index(time_col)
  df.index = pd.DatetimeIndex(df.index)
  return df


def remove_duplicated_index(df, keep='first'):
  """
  remove duplicated index with its row from dataframe

  :param df: dataframe
  :param keep: first or last duplicated index to keep
  :returns: dataframe without duplicated index
  :raises: Exception
  """
  try:
    df = df[~df.index.duplicated(keep=keep)].copy()

  except Exception as e:
    print(e)

  return df


#----------------------- Data visualization ----------------------#
def plot_data(df, columns, start=None, end=None, figsize=(20, 5), colormap='tab10'):
  """
  Plot chart for several different lines

  :param df: dataframe
  :param columns: columns that need to be plotted
  :param start: start row
  :param end: end row
  :param figsize: figsize of the plot
  :param colormap: colormap used for lines
  :returns: a figure
  :raises: none
  """
  # select target data that need to be plotted
  selected_data = df[start:end][columns]

  # create image
  plt.figure(figsize=figsize)
  plt.rcParams['axes.facecolor'] = 'whitesmoke'
  
  # set color
  cNorm = colors.Normalize(vmin=0, vmax=len(selected_data.columns))
  cm = plt.get_cmap(colormap)
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
  
  # plot columns one by one
  for i in range(len(selected_data.columns)):
    col = selected_data.columns[i]
    plt.plot(selected_data.index, selected_data[col], label=col, color=scalarMap.to_rgba(i+1), linewidth=2)

  # plot legend, grid and rotate xticks    
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
  plt.grid(True)
  plt.xticks(rotation=90)   


#----------------------- Print assistence ----------------------#
def print_when(condition, true_content='', false_content=None):
  if condition:
    if true_content is not None:
      print(true_content)

  else:
    if false_content is not None:
      print(false_content)

