# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import math
from quant import bc_util as util



#----------------------------- 均值/方差模型 -----------------------------------#
# 风险溢价是超额收益的期望值(rate_premium = mean(excess_return)),
# 超额收益的标准差是其风险的测度(risk = std(excess_return))
# 计算持有期收益率(Holding Period Rate)
def cal_HPR(data, start, end, dim='Close', dividends=0):
  data = data[start:end][dim].tolist()
  HPR = (data[-1] - data[0]) / data[0]
  
  return HPR


# 计算有效年收益率(Effective Annual Rate)
def cal_EAR(data, start, end, dim='Close', dividends=0):
  # 计算期间内的收益率
  HPR = cal_HPR(data, start, end, dim, dividends) + 1
  # 计算期间的长度(年)
  start_date = util.time_2_string(data[start:end].index.min())
  end_date = util.time_2_string(data[start:end].index.max())
  period_in_year = util.num_days_between(start_date, end_date) / 365.0
  # 计算有效年利率
  EAR = pow(HPR, 1/period_in_year) - 1
  
  return EAR


# 计算年化百分比利率(Annual Percentile Rate)
def cal_APR(data, start, end, dim='Close', dividends=0):
  # 计算期间内的收益率
  HPR = cal_HPR(data, start, end, dim, dividends)
  # 计算期间的长度(年)
  start_date = util.time_2_string(data[start:end].index.min())
  end_date = util.time_2_string(data[start:end].index.max())
  period_in_year = util.num_days_between(start_date, end_date) / 365.0
  # 计算有效年利率
  APR = HPR / period_in_year
  
  return APR


# 计算连续复利利率(Continuous Compounding Rate)
def cal_CCR(data, start, end, dim='Close', dividends=0):
  EAR = cal_EAR(data, start, end, dim, dividends)
  CCR = math.log((1+EAR), math.e)
  
  return CCR


# 计算风险溢价(Risk Premium)
def cal_risk_premium(expected_rate, risk_free_rate):
  RP = expected_rate - risk_free_rate
  
  return RP


# 计算超额收益(Excess Return)
def cal_excess_raturn(expected_rate, real_rate):
  ER = real_rate - expected_rate
  
  return ER


# 计算周期收益率(年/月)
def cal_period_rate_risk(data, dim='Close', by='month'):
  
  # 计算每日的变化率
  data = cal_change_rate(df=data, dim=dim, period=1)

  # 获取开始/结束日期, 构造周期列表
  start_date = data.index.min().date()
  end_date = data.index.max().date()
  periods = []

  # 年周期
  if by == 'year':
    for year in range(start_date.year, end_date.year+1):
      p = '%(year)s' % dict(year=year)
      periods.append((p, p))
      
  # 月周期      
  elif by == 'month':
    for year in range(start_date.year, end_date.year+1):
      for month in range(1, 13):
        if year >= end_date.year and month > end_date.month:
          break
        p = '%(year)s-%(month)02d' % dict(year=year, month=month)
        periods.append((p, p))

  # 周周期
  elif by == 'week':
    week_start = start_date
    while week_start < end_date:
      week_end = week_start + datetime.timedelta(days=(6 - week_start.weekday()))
      periods.append((week_start, week_end))
      week_start = week_end + datetime.timedelta(days=1)
  else:
    print('Invalid period')
  
  # 计算周期收益率
  period_rate = {
      'period': [],
      'start': [],
      'end': [],
      'HPR': [],
      'EAR': [],
      'APR': [],
      'CCR': [],
      'daily_rate_mean': [],
      'daily_rate_std': []
  } 
  for p_pair in periods:
    tmp_data = data[p_pair[0]:p_pair[1]]
    if len(tmp_data) <= 1:
      continue
    else:
      period_rate['period'].append(p_pair[0])
      period_rate['start'].append(p_pair[0])
      period_rate['end'].append(p_pair[1])
      period_rate['HPR'].append(cal_HPR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['EAR'].append(cal_EAR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['APR'].append(cal_APR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['CCR'].append(cal_CCR(data=tmp_data, start=None, end=None, dim='Close'))
      period_rate['daily_rate_mean'].append(tmp_data.rate.mean())
      period_rate['daily_rate_std'].append(tmp_data.rate.std())
  
  period_rate = pd.DataFrame(period_rate)
  period_rate = util.df_2_timeseries(df=period_rate, time_col='period')
  
  return period_rate



