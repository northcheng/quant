# -*- coding: utf-8 -*-
"""
Utilities used for trade simulation

:authors: Beichen Chen
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
from quant import bc_util as util
from quant import bc_finance as finance_util
from quant import bc_technical_analysis as ta_util


def buy(money, price, trading_fee):
  """
  Buy stocks

  :param money: money used for buying stocks
  :param price: price of the stock
  :param trading_fee: trading_fee
  :returns: left money and bought stocks
  :raises: none
  """
  # calculate how many stocks could be bought, and how much money would left
  stock = math.floor((money-trading_fee) / price)
  if stock > 0:
    money = money - trading_fee - (price*stock) 
  else:
    stock = 0
    # print('Not enough money to buy')
  return {'money': money, 'stock': stock} 


def sell(stock, price, trading_fee):
  """
  Sell stocks

  :param stock: number of stock to sell
  :param price: price of the stock
  :param trading_fee: trading fee
  :returns: left stock and money of sold stock
  """
  # calculate how much the stock worthes 
  money = stock * price - trading_fee
  if money > 0:
    stock = 0
  else:
    money = 0
    # print('Not enough stock to sell')
  return {'money': money, 'stock': stock}


class FixedPositionTrader:

  sec_list = []
  start_cash = 0
  record = {}
  benchmark = None
  benchmark_return = 0

  cash = {}
  stock = {}
  value = {}
  holding_price = {}
  

  # init
  def __init__(self, sec_list, start_cash, data, recalculate_signal=False, start_date=None, end_date=None, benchmark='SPY'):

    # initialize stock list and start cash
    self.sec_list = sec_list.copy()
    self.start_cash = start_cash

    # initialize trading record
    signals = data['ta_data'].copy()
    for k in signals.keys():
      sec_code, interval = k.split('_')

      # recalculate ta trends and signals
      if recalculate_signal and (sec_code in self.sec_list or sec_code == benchmark):
        self.record[sec_code] = ta_util.calculate_ta_data(df=data['sec_data'][k][start_date:end_date], symbol=sec_code, interval=interval)
      else:
        self.record[sec_code] = signals[k].copy()

      # add extra columns 
      self.record[sec_code]['holding_price'] = 0
      self.record[sec_code]['holding_return'] = 0
      self.record[sec_code]['money'] = np.NaN
      self.record[sec_code]['stock'] = np.NaN
      self.record[sec_code]['value'] = np.NaN

    # set benchmark
    if benchmark in self.record.keys():
      self.benchmark = benchmark
      self.record['benchmark'] = self.record[benchmark].copy()
      self.record['benchmark']['signal'] = 'n'
      max_idx = self.record['benchmark'].index.max()
      min_idx = self.record['benchmark'].index.min()
      self.record['benchmark'].loc[min_idx,'signal'] = 'b'
      self.record['benchmark'].loc[max_idx,'signal'] = 's'

      # add benchmark into trade list
      self.sec_list.append('benchmark')
      self.start_cash = start_cash/len(sec_list) * len(self.sec_list)
    else:
      print(f'{benchmark} not in signals')

  # trade
  def trade(self, start_date, end_date, stop_profit=None, stop_loss=None):

    # evenly distribute money to each stock
    avg_position = self.start_cash / len(self.sec_list)
    for sec_code in self.sec_list:
      self.stock[sec_code] = 0
      self.cash[sec_code] = avg_position
      self.value[sec_code] = avg_position

    # set target sec list
    target_list = self.sec_list

    # start/end date
    start_dates = []
    end_dates = []
    for s in target_list:
      start_dates.append(self.record[s].index.min())
      end_dates.append(self.record[s].index.max())
    if start_date is None:
      start_date = util.time_2_string(min(start_dates)) if start_date is None else start_date
    if end_date is None:
      end_date =  util.time_2_string(max(end_dates)) if end_date is None else end_date

    # construct date list
    dates = []
    next_date = start_date
    while next_date <= end_date:
      dates.append(next_date)
      next_date = util.string_plus_day(next_date, 1)

    # go through each day
    for date in dates:
      # date_signal = []

      # go through each stock
      for sec_code in target_list:
        signal_data = self.record[sec_code]
        
        # if current date is trading day
        if date in signal_data.index:

          # get signal and price
          tmp_signal = signal_data.loc[date, 'signal']
          tmp_price = signal_data.loc[date, 'Close']

          # check if it is necessary to stop profit/loss
          if self.stock[sec_code] > 0:
            signal_data.loc[date, 'holding_price'] = self.holding_price[sec_code]
            signal_data.loc[date, 'holding_return'] = (tmp_price - signal_data.loc[date, 'holding_price']) / signal_data.loc[date, 'holding_price']
            if (stop_profit is not None and signal_data.loc[date, 'holding_return'] >= stop_profit) or (stop_loss is not None and signal_data.loc[date, 'holding_return'] <= stop_loss):
              tmp_signal = 's'

          if tmp_signal == 'b':
            trade_result = buy(money=self.cash[sec_code], price=tmp_price, trading_fee=3)
            self.cash[sec_code] = trade_result['money']
            self.stock[sec_code] += trade_result['stock']
            self.holding_price[sec_code] = tmp_price

          elif tmp_signal == 's':
            trade_result = sell(stock=self.stock[sec_code], price=tmp_price, trading_fee=3)
            self.cash[sec_code] += trade_result['money']
            self.stock[sec_code] = trade_result['stock']
            self.holding_price[sec_code] = 0

          else:
            pass

          self.value[sec_code] = self.cash[sec_code] + self.stock[sec_code] * tmp_price
          signal_data.loc[date, 'money'] = self.cash[sec_code]
          signal_data.loc[date, 'stock'] = self.stock[sec_code]
          signal_data.loc[date, 'value'] = self.value[sec_code]
                  
        # if current date is not trading day
        else:
          pass

    for sec_code in target_list:
      self.record[sec_code][['money', 'stock', 'value']] = self.record[sec_code][['money', 'stock', 'value']].fillna(method='bfill')

  # visualize
  def visualize(self, sec_code, start_date=None, end_date=None):

    # create image
    fig = plt.figure(figsize=(20, 5))  
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    trade_plot = plt.subplot(gs[0])
    money_plot = plt.subplot(gs[1], sharex=trade_plot)
    plt.setp(trade_plot.get_xticklabels(), visible=False)
    gs.update(wspace=0, hspace=0)
    
    # get plot data
    if sec_code == 'portfolio':
      record = self.record['benchmark'][['value']].copy()
      record['value'] = 0
      for k in self.record.keys():
        if k == 'benchmark':
          continue
        tmp_data = self.record[k][['value']].copy()
        record = record.add(tmp_data, fill_value=0)
      record['Close'] = np.NaN
      record['signal'] = 'n'
    else:
      record = self.record[sec_code].copy()
      
    record = record[start_date:end_date].copy()
    min_idx = record.index.min()
    max_idx = record.index.max()

    # plot trade signals
    buying_points = record.query('signal == "b"')
    selling_points = record.query('signal == "s"')
    trade_plot.plot(record.index, record[['Close']], label='Close', alpha=0.5)
    trade_plot.scatter(buying_points.index,buying_points.Close, c='green', marker='^', label='Buy')
    trade_plot.scatter(selling_points.index,selling_points.Close, c='red', marker='v', label='Sell')

    # plot money flow chart
    record.fillna(method='ffill', inplace=True)
    record['original'] = record.loc[min_idx, 'value']
    money_plot.plot(record.index, record.value, label='Value')
    money_plot.plot(record.index, record.original, label='Original')
    money_plot.fill_between(record.index, record.original, record.value, where=record.value > record.original, facecolor='green', interpolate=True, alpha=0.1)
    money_plot.fill_between(record.index, record.original, record.value, where=record.value < record.original, facecolor='red', interpolate=True, alpha=0.1)

    # plot benchmark money flowchart if benchmark exists
    if self.benchmark is not None:
      benchmark_record = self.record['benchmark'][start_date:end_date].copy()
      benchmark_min_idx = benchmark_record.index.min()
      benchmark_record.fillna(method='ffill', inplace=True)

      if sec_code == 'portfolio':
        benchmark_record['value'] = benchmark_record['value'] * (len(self.sec_list)-1)
      benchmark_record['original'] = benchmark_record.loc[benchmark_min_idx, 'value']
      money_plot.plot(benchmark_record.index, benchmark_record.value, label='benckmark', color='black', linestyle='--',)
      # money_plot.fill_between(benchmark_record.index, benchmark_record.original, benchmark_record.value, where=benchmark_record.value > benchmark_record.original, facecolor='green', interpolate=True, alpha=0.1)
      # money_plot.fill_between(benchmark_record.index, benchmark_record.original, benchmark_record.value, where=benchmark_record.value < benchmark_record.original, facecolor='red', interpolate=True, alpha=0.1)

    # set title and legend
    trade_plot.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.0) 
    trade_plot.set_title('Signals', rotation='vertical', x=-0.05, y=0.3)
    money_plot.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.0) 
    money_plot.set_title('Money', rotation='vertical', x=-0.05, y=0.3)

    hpr = finance_util.cal_HPR(data=record, start=min_idx.date(), end=max_idx.date(), dim='value', dividends=0)
    fig.suptitle(f'{sec_code}: {hpr*100:.2f}%', x=0.5, y=0.95, fontsize=20)

  # analysis
  def analyze(self, sort=True):

    # get records for self.sec_list
    records = dict((key,value) for key,value in self.record.items() if key in self.sec_list)
    
    # init dict for storing results
    analysis = {
        'sec_code': [],
        'start_date': [],
        'end_date': [],
        'start_money': [],
        'end_money': [],
        'EAR': [],
        'sharp_ratio': [],
        'max_drawndown': []
      }

    # go through each stock
    for sec_code in records.keys():
        
      # get record data
      record_data = records[sec_code]
      min_idx = record_data.index.min()
      max_idx = record_data.index.max()
      
      # analysis profit, hpr, ear, etc.
      analysis['sec_code'].append(sec_code)
      analysis['start_date'].append(util.time_2_string(min_idx.date()))
      analysis['end_date'].append(util.time_2_string(max_idx.date()))
      analysis['start_money'].append(record_data.loc[min_idx, 'value'])
      analysis['end_money'].append(record_data.loc[max_idx, 'value'])

      EAR = finance_util.cal_EAR(data=record_data, start=min_idx.date(), end=max_idx.date(), dim='value', dividends=0)
      analysis['EAR'].append(EAR)

      sharp_ratio = finance_util.cal_sharp_ratio(data=record_data, start=None, end=None, price_dim='value')
      analysis['sharp_ratio'].append(sharp_ratio)

      max_drawndown = finance_util.cal_max_drawndown(data=record_data)
      analysis['max_drawndown'].append(max_drawndown)

    # transform dict to dataframe
    analysis = pd.DataFrame(analysis).set_index('sec_code')
    if sort:
      analysis = analysis.sort_values('EAR', ascending=False)

    # calculate sum and mean for non benchmark stocks
    non_benchmark_list = [x for x in analysis.index.tolist() if x != 'benchmark']
    non_benchmark_analysis = analysis.loc[non_benchmark_list, analysis.columns].copy()
    if len(non_benchmark_analysis) > 1:

      # calculate sum and mean
      analysis_mean = non_benchmark_analysis.mean()
      analysis_sum = non_benchmark_analysis.sum()
      
      # calculate sum of the whole portfilo
      value_sum = None
      for symbol in records.keys():
        if symbol == 'benchmark':
          continue
        if value_sum is None:
          value_sum = records[symbol][['value']].copy()
        else:
          value_sum = value_sum.add(records[symbol][['value']], fill_value=0)
      value_sum['rate'] = value_sum.pct_change().fillna(0)
      total_ear = finance_util.cal_EAR(data=value_sum, start=None, end=None)
      total_max_drawndown = finance_util.cal_max_drawndown(data=value_sum)
      total_sharp_ratio = finance_util.cal_sharp_ratio(data=value_sum, start=None, end=None)

      # resort dataframe
      if self.benchmark is not None:
        right_order = [x for x in analysis.index if x != 'benchmark'] + ['benchmark'] 
        analysis = analysis.loc[right_order].copy()

      analysis = analysis.append(pd.DataFrame({'start_date': '', 'end_date': '', 'start_money': analysis_mean['start_money'], 'end_money':analysis_mean['end_money'], 'EAR':total_ear, 'sharp_ratio':total_sharp_ratio, 'max_drawndown':total_max_drawndown}, index=['mean']))
      analysis = analysis.append(pd.DataFrame({'start_date': '', 'end_date': '', 'start_money': analysis_sum['start_money'], 'end_money':analysis_sum['end_money'], 'EAR':total_ear, 'sharp_ratio':total_sharp_ratio, 'max_drawndown':total_max_drawndown}, index=['total']))

    # post process
    analysis['profit'] = analysis['end_money'] - analysis['start_money']
    analysis['HPR'] = analysis['profit'] / analysis['start_money']
    analysis = analysis[['start_date', 'end_date', 'start_money', 'end_money', 'profit', 'HPR', 'EAR', 'sharp_ratio', 'max_drawndown']].round(2)
    
    return analysis

