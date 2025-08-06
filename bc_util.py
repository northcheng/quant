# -*- coding: utf-8 -*-
"""
Generally Utilities

:author: Beichen Chen
"""
import os
import time
import pytz
import shutil
import datetime
import subprocess
from typing import List, Optional, Union, Any, Dict, Tuple
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
def string_2_time(string: str, diff_days: int = 0, date_format: str = default_date_format) -> datetime.datetime:
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
 

def time_2_string(time_object: datetime.datetime, diff_days: int = 0, date_format: str = default_date_format) -> str:
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


def timestamp_2_time(timestamp: Union[int, float], unit: str = 'ms', timezone: str = 'CN') -> datetime.datetime:
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


def string_plus_day(string: str, diff_days: int, date_format: str = default_date_format) -> str:
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


def num_days_between(start_date: str, end_date: str, date_format: str = default_date_format) -> int:
  """
  Calculate the number of days between 2 date strings

  :param start_date: date string of start date
  :param end_date: date string of end date
  :param date_format: format of date strings
  :returns: number of days between start/end date
  :raises: none
  """
  start_date = string_2_time(start_date, date_format=date_format)
  end_date = string_2_time(end_date, date_format=date_format)
  diff = end_date - start_date

  return diff.days


def convert_timezone(time_object: datetime.datetime, from_tz: pytz.BaseTzInfo, to_tz: pytz.BaseTzInfo, keep_tzinfo: bool = False) -> datetime.datetime:
  """
  Convert a time object from one timezone to another

  :param time_object: datetime instance
  :param from_tz: original timezone
  :param to_tz: target timezone
  :param keep_tzinfo: whether to keep the tzinfo in the result
  :returns: datetime instance with timezone converted
  :raises: none
  """
  result = time_object.replace(tzinfo=from_tz).astimezone(to_tz)
  if not keep_tzinfo:
    result = result.replace(tzinfo=None)

  return result


#----------------------- File manipulation -----------------------#
def synchronize_file(local_folder: str, remote_folder: str, newer_only: bool = False, syn_file: bool = True, syn_folder: bool = True, file_type: Optional[List[str]] = None, folder_name: Optional[List[str]] = None, is_print: bool = True) -> None:
  """
  Synchronize files from remote folder to local folder

  :param local_folder: local folder path
  :param remote_folder: remote folder path
  :param newer_only: whether to synchronize the newer files only
  :param syn_file: whether to synchronize files
  :param syn_folder: whether to synchronize folders
  :param file_type: specify type of file to synchronize
  :param folder_name: specify name of folder to synchronize
  :param is_print: whether to print information
  :returns: none
  :raises: none
  """

  # check accessibility of local and remote folders
  if os.path.exists(remote_folder) and os.path.exists(local_folder):
    
    # get local files and folders
    local_files = os.listdir(local_folder)
    local_folders = [x for x in local_files if os.path.isdir(local_folder + x)]
    local_files = [x for x in local_files if not os.path.isdir(local_folder + x)]
    
    # get remote files and folders
    remote_files = os.listdir(remote_folder)
    remote_folders = [x for x in remote_files if os.path.isdir(remote_folder + x)]
    remote_files = [x for x in remote_files if not os.path.isdir(remote_folder + x)]

    # for files
    if syn_file:

      # remove local files(if exists), copy remote files
      for rf in remote_files:

        try:
        
          # get absolute path
          rf_abs_path = remote_folder + '/' + rf
          lf_abs_path = local_folder + '/' + rf

          # skip if file_type is not None
          if file_type is not None and type(file_type) == list:
            if rf.split('.')[-1] not in file_type:
              print(f'skip {remote_folder} (not in {file_type})')
              continue

          # skip if newer_only and local file is newer
          if newer_only and os.path.exists(lf_abs_path):
            if os.path.getmtime(rf_abs_path) < os.path.getmtime(lf_abs_path):
              print(f'skip {lf_abs_path} (newer)')
              continue
          
          # check file existence
          if os.path.exists(rf_abs_path):

            # remove local file if exists
            if os.path.exists(lf_abs_path):
              os.remove(lf_abs_path)
              print(f'remove {lf_abs_path}', end=', ')

            # copy remote file
            shutil.copyfile(rf_abs_path, lf_abs_path)
            print(f'copy {rf_abs_path}')

        except Exception as e:
          print(e, rf_abs_path, lf_abs_path)
          continue
      
    # for folders
    if syn_folder:
      
      # remove local folders(if exists), copy remote folders
      for fd in remote_folders:

        try:
        
          # get absolute path
          rfd_abs_path = remote_folder + '/' + fd
          lfd_abs_path = local_folder + '/' + fd
          
          # skip if folder_name is not None
          if folder_name is not None and type(folder_name) == list:
            if fd not in folder_name:
              print(f'skip {rfd_abs_path} (not in {folder_name})')
              continue

          # skip if newer_only and local folder is newer
          if newer_only:
            if os.path.getmtime(rfd_abs_path) < os.path.getmtime(lfd_abs_path):
              print(f'skip {lfd_abs_path} (newer)')
              continue

          # check folder existence
          if os.path.exists(rfd_abs_path) and os.path.isdir(rfd_abs_path):

            # remove local folder if exists
            if os.path.exists(lfd_abs_path) and os.path.isdir(lfd_abs_path):
              shutil.rmtree(lfd_abs_path)
              print(f'remove {lfd_abs_path}', end=', ')

            # copy remote folder
            shutil.copytree(rfd_abs_path, lfd_abs_path)
            print(f'copy {rfd_abs_path}')
      
        except Exception as e:
          print(e, rfd_abs_path, lfd_abs_path)
          continue
      
  else:
    print('please check existence of path')


def print_folder_tree(path: str, parent_is_last: int = 1, depth_limit: int = -1, tab_width: int = 1) -> List[str]:
  """
  Print folder and files in a tree structure
  :param path: target path
  :param tab_width: width of a tab
  :param depth_limit: number of depth to go through, -1 means go through all files
  :param parent_is_last: used for control output
  :return: list of all files in 'path'
  """
  files = []
  if len(str(parent_is_last)) - 1 == depth_limit:
    return files
  items = os.listdir(path)
  for index, i in enumerate(items):
    is_last = index == len(items) - 1
    i_path = path + "/" + i
    for k in str(parent_is_last)[1:]:
      if k == "0":
        print("│" + "\t" * tab_width, end="")
      if k == "1":
        print("\t" * tab_width, end="")
    if is_last:
      print("└── ", end="")
    else:
      print("├── ", end="")
    if os.path.isdir(i_path):
      print(i)
      files.extend(print_folder_tree(
        path=i_path, depth_limit=depth_limit, parent_is_last=(parent_is_last * 10 + 1) if is_last else (parent_is_last * 10)))
    else:
      print(i_path.split("/")[-1])
      files.append(i_path)
  return files


#----------------------- Dataframe manipulation ------------------#
def df_2_timeseries(df: pd.DataFrame, time_col: str = 'date') -> pd.DataFrame:
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


def remove_duplicated_index(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
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
def response_2_json(response: Any, print_status: bool = False) -> Optional[Dict[str, Any]]:
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
def plot_data(df: pd.DataFrame, columns: List[str], start: Optional[int] = None, end: Optional[int] = None, figsize: Tuple[int, int] = (20, 5), colormap: str = 'tab10') -> plt.Figure:
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
  return plt.gcf()


#----------------------- Print assistence ------------------------#
def print_when(condition: bool, true_content: str = '', false_content: Optional[str] = None) -> None:
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
def sleep_until(target_time: datetime.datetime, description: Optional[str] = None, check_frequency: int = 3600) -> None:
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
def concate_image(image_list: List[str], adjust_size: bool = False, save_name: Optional[str] = None, remove_old_image: bool = True) -> None:
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


def image_2_pdf(image_list: List[str], save_name: Optional[str] = None, remove_old_pdf: bool = True, is_print: bool = False) -> None:
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
    if is_print:
      print('no images to convert to gif')


def image_2_gif(image_list: List[str], save_name: Optional[str] = None, remove_old_gif: bool = True, fps: int = 3) -> None:
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
      tmp_image = Image.open(image)
      
      if len(images) == 0:
        target_size = tmp_image.size

      tmp_image = tmp_image.resize(target_size, Image.LANCZOS)
      images.append(tmp_image)

  # save gif
  if len(images) > 0:
    if save_name is None:
      save_name = 'gif_from_image.gif'
    # 保存为 GIF
    images[0].save(save_name, save_all=True, append_images=images[1:], duration=1000, loop=0)

  else:
    print('no images to convert to pdf')


#----------------------- Script runner ---------------------------#
def run_script(cmd: Union[List[str], str], retry: int = 1, timeout: int = 1800) -> Optional[int]:
  
  # try to run non_visual script, if failed, retry(10 times)
  retry_count = 0
  while retry_count < retry:

    # set retry count and current time
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