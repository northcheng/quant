# -*- coding: utf-8 -*-
"""
Generally Utilities

:author: Beichen Chen
"""
import os
import time
import pytz
import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image



#----------------------- Date manipulation -----------------------#
def string_2_time(string, diff_days=0, date_format='%Y-%m-%d'):
  """
  Convert a date from string to datetime

  :param string: date in string format
  :param diff_days: days need to be added or reduced
  :param date_format: the format of the date string
  :returns: date in datetime format
  :raises: none
  """
  time_object = datetime.datetime.strptime(string, date_format)
  time_object = time_object + datetime.timedelta(days=diff_days)

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
  :param unit: unit of the timestamp, us or ms(default)
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
  time_object = string_2_time(string, date_format=date_format)
  time_string = time_2_string(time_object, diff_days=diff_days, date_format=date_format)

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
  start_date = string_2_time(start_date, date_format=date_format)
  end_date = string_2_time(end_date, date_format=date_format)
  diff = end_date - start_date

  return diff.days


#----------------------- Dataframe manipulation ------------------#
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


#----------------------- Print assistence ------------------------#
def print_when(condition, true_content='', false_content=None):
  """
  Print different content under different conditions

  :param condition: condition sentence, the result is a boolean value
  :param true_content: content to print when the condition is true
  :param false_content: content to print when the condition is false
  :returns: none
  :raises: none
  """
  if condition:
    if true_content is not None:
      print(true_content)

  else:
    if false_content is not None:
      print(false_content)


#----------------------- Process control -------------------------#
def sleep_until(target_time, description=None, check_frequency=3600):
  """
  Sleep with a fixed frequency, until the target time

  :param target_time: the target time in datetime.datetime format
  :param description: description of the target time
  :param check_frequency: the fixed sleep_time 
  :returns: none
  :raises: none
  """
  # get current time
  now = datetime.datetime.now()

  # sleep until target time
  while now < target_time:

    # check time difference in seconds
    diff_time = round((target_time - now).total_seconds())
    sleep_time = (diff_time+1) if (diff_time <= check_frequency) else check_frequency

    # print information then sleep
    description = '' if description is None else f'({description})'
    print(f'{now}: sleep for {sleep_time} seconds {description}')
    time.sleep(sleep_time)

    # update current time
    now = datetime.datetime.now()

  print(f'{now}: exceed target time({target_time})')


#----------------------- Image manipulation ----------------------#
def concate_image(image_list, adjust_size=False, save_name=None):
  """
  Concate images in the image list, save to save_name

  :param image_list: list of absolute path of images
  :param adjust_size: adjust images to the same size of the first image
  :param save_name: the absolute path of the concated image
  :returns: none
  :raises: none
  """

  # load images
  images = []
  for image in image_list:
    if os.path.exists(image):
      images.append(Image.open(image))

  # adjust image size 
  if adjust_size:
    ims = []
    for i in images:
      new_img = i.resize((1280, 1280), Image.BILINEAR)
      ims.append(new_img)
  else:
    ims = images

  if len(ims) > 0:
    # concate images
    width, height = ims[0].size
    result = Image.new(ims[0].mode, (width, height * len(ims)))
    for i, im in enumerate(ims):
      result.paste(im, box=(0, i * height))

    # save concated image
    if save_name is None:
      save_name = 'concated_image.png'

    result.save(save_name)
  

#----------------------- Script runner ---------------------------#
def run_script(cmd, retry=1, timeout=600):
  
  # try to run non_visual script, if failed, retry(10 times)
  retry_count = 0
  while retry_count < retry:

    # set retru count and current time
    retry_count += 1
    return_code = None

    try:
      return_code = subprocess.check_call(cmd, timeout=timeout)
      if return_code == 0:
        break              
    except Exception as e:
      print(f'[erro]: {type(e)}, {e}, retry({retry_count}/{retry})')
      continue

  return return_code