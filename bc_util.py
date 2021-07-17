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
import imageio as imio

# default arguments
default_date_format = '%Y-%m-%d'

#----------------------- Date manipulation -----------------------#
def string_2_time(string, diff_days=0, date_format=default_date_format):
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
 

def time_2_string(time_object, diff_days=0, date_format=default_date_format):
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


def string_plus_day(string, diff_days, date_format=default_date_format):
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


def num_days_between(start_date, end_date, date_format=default_date_format):
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


#----------------------- Http Request/Response -------------------#
def response_2_json(response, print_status=False):
  """
  extract json content from http response

  :param response: response from a http request
  :returns: json content if response is valid, else None
  :raises: None
  """
  # initialize json_content
  json_content = None
  
  # get status code of the response
  if response is not None:
    status = response.status_code

    # print status
    if print_status:
      print(f'Response status: {status}')
  
    # if the response is valid, get its json content
    if status == 200:
      json_content = response.json()
      
      # if json content is empty
      if len(json_content) == 0:
        json_content = None
        print(f'Empty json in response')
        
    # otherwise print error code of the response
    else:
      print(f'Error with code: {status}')
      
  return json_content


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
def concate_image(image_list, adjust_size=False, save_name=None, remove_old_image=True):
  """
  Concate images in the image list, save to a image named <save_name>

  :param image_list: list of absolute path of images
  :param adjust_size: adjust images to the same size of the first image
  :param save_name: the absolute path of the concated image
  :param remove_old_image: whether to remove old file with save_name
  :returns: none
  :raises: none
  """
  # remove old image before create new image
  if remove_old_image and (save_name is not None):
    if os.path.exists(save_name):
      os.remove(save_name)

  # load images
  images = []
  for image in image_list:
    if os.path.exists(image):
      images.append(Image.open(image))

  # adjust image size 
  if adjust_size:
    ims = []
    for i in images:
      new_img = i.resize((2500, 1200), Image.BILINEAR)
      ims.append(new_img)
  else:
    ims = images

  # concate images
  if len(ims) > 0:
    width, height = ims[0].size
    result = Image.new(ims[0].mode, (width, height * len(ims)))
    for i, im in enumerate(ims):
      result.paste(im, box=(0, i * height))

    # save concated image
    if save_name is None:
      save_name = 'concated_image.png'
    result.save(save_name)
  else:
    print(f'{save_name}: No image to concate')


def image_2_pdf(image_list, save_name=None, remove_old_pdf=True):
  """
  save images in the image list to a pdf file

  :param image_list: list of absolute path of images
  :param save_name: the absolute path of the concated image
  :param remove_old_pdf: whether to remove old file with save_name
  :returns: none
  :raises: none
  """
  # remove old image before create new image
  if remove_old_pdf and (save_name is not None):
    if os.path.exists(save_name):
      os.remove(save_name)

  # load images
  images = []
  for image in image_list:
    if os.path.exists(image):
      tmp_image = Image.open(image)
      tmp_image = tmp_image.convert('RGB')
      images.append(tmp_image)

  # save pdf
  if len(images) > 0:
    if save_name is None:
      save_name = 'pdf_from_image.png'
    images[0].save(save_name, save_all=True, append_images=images[1:])
  else:
    print('no images to convert to pdf')


def image_2_gif(image_list, save_name=None, remove_old_gif=True, fps=3):
  """
  save images in the image list to a gif file

  :param image_list: list of absolute path of images
  :param save_name: the absolute path of the concated image
  :param remove_old_gif: whether to remove old file with save_name
  :returns: none
  :raises: none
  """
  # remove old image before create new image
  if remove_old_gif and (save_name is not None):
    if os.path.exists(save_name):
      os.remove(save_name)

  # load images
  images = []
  for image in image_list:
    if os.path.exists(image):
      tmp_image = imio.imread(image, format='png')
      images.append(tmp_image)

  # save gif
  if len(images) > 0:
    if save_name is None:
      save_name = 'gif_from_image.gif'
    imio.mimsave(save_name, images, fps=fps)
  else:
    print('no images to convert to pdf')


#----------------------- Script runner ---------------------------#
def run_script(cmd, retry=1, timeout=1800):
  
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