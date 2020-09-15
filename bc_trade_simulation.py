# -*- coding: utf-8 -*-
"""
Utilities used for trade simulation

:authors: Beichen Chen
"""
import pandas as pd
import numpy as np
import math
import datetime
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
  benchmark = None
  start_date = None
  end_date = None

  data = {}
  record = {}
  cash = {} 
  stock = {}
  value = {}
  holding_price = {}
  

  # init
  def __init__(self, data, start_date=None, end_date=None, num_days=365):

    # copy data
    self.data = data.copy()

    # set start_date/end_date
    if (start_date is not None) and (end_date is None):
      end_date = util.string_plus_day(string=start_date, diff_days=num_days)
    elif (start_date is None) and (end_date is not None):
      start_date = util.string_plus_day(string=end_date, diff_days=-num_days)
    elif (start_date is None) and (end_date is None):
      end_date = util.time_2_string(datetime.datetime.today().date())
      start_date = util.string_plus_day(string=end_date, diff_days=-num_days)
    self.start_date = start_date
    self.end_date = end_date

    # initialize record with sec_data
    ta_data = data['ta_data']
    for k in ta_data.keys():
      symbol = k.split('_')[0]
      self.record[symbol] = ta_data[k].copy()
    
    self.init_record()

  # set benchmark 
  def set_benchmark(self, benchmark, start_date=None, end_date=None): 
    
    # set start_date/end_date
    start_date = self.start_date if start_date is None else start_date
    end_date = self.end_date if end_date is None else end_date   

    # find benchmark data from sec_data
    benchmark_key = [x for x in self.data['sec_data'].keys() if benchmark in x]
    benchmark_num = len(benchmark_key)
    if (benchmark_num > 1) or (benchmark_num==0):
      print(f'{benchmark_num} benchmark data found')
      self.benchmark = None
    
    else:
      # set benchmark data from sec_data
      self.benchmark = benchmark
      self.record['benchmark'] = ta_util.cal_change_rate(df=self.data['sec_data'][benchmark_key[0]], target_col='Close').dropna()[start_date:end_date].copy()
      
      # set benchmark trading signals
      benchmark_idx = self.record['benchmark'].index
      self.record['benchmark']['signal'] = 'n'
      self.record['benchmark'].loc[benchmark_idx.min(),'signal'] = 'b'
      self.record['benchmark'].loc[benchmark_idx.max(),'signal'] = 's'

  # recalculate data
  def recalculate_data(self, sec_list, mode=None, start_date=None, end_date=None):

    # set start_date/end_date
    start_date = self.start_date if start_date is None else start_date
    end_date = self.end_date if end_date is None else end_date

    # copy sec_data, ta_data
    sec_data = self.data['sec_data'].copy()
    ta_data = self.data['ta_data'].copy()
    
    # recalculate lists
    copy_ta_data = []
    recalculate_trend = []
    recalculate_signal = []

    # set recalculate mode for each symbol
    for k in ta_data.keys():

      # skip symbols which not in sec_list
      symbol = k.split('_')[0]
      if symbol not in sec_list:
        continue

      # get data and its range
      tmp_data = ta_data[k]
      min_idx = util.time_2_string(tmp_data.index.min())
      max_idx = util.time_2_string(tmp_data.index.max())

      # for symbols which ta_data range covers start_date~end_date, process according to mode
      if (min_idx <= start_date) and (max_idx >= end_idx):
        if mode is None:
          copy_ta_data.append(k)
        elif mode == 'signal':
          recalculate_signal.append(k)
        elif mode == 'trend':
          recalculate_trend.append(k)
        else:
          print(f'Unknown mode: {mode}')
          copy_ta_data.append(k)

      # for symbols which ta_data range not covers start_date~end_date, recalculate from trend
      else:
        recalculate_trend.append(k)

    # for symbols just need to copy ta_data
    for k in copy_ta_data:
      symbol = k.split('_')[0]
      self.record[symbol] = ta_data[k][start_date:end_date].copy()

    # for symbols need to recalculate signals
    for k in recalculate_signal:
      symbol = k.split('_')[0]
      self.record[symbol] = ta_util.calculate_ta_signal(df=ta_data[k][start_date:end_date])

    # for symbols need to recalculate trend and signal
    for k in recalculate_trend:
      symbol, interval = k.split('_')
      self.record[symbol] = ta_util.calculate_ta_data(df=sec_data[k][start_date:end_date], symbol=symbol, interval=interval)

    # reset record
    self.init_record()

  # initialize record
  def init_record(self):
    # add extra columns for records
    for symbol in self.record.keys():
      self.record[symbol]['holding_price'] = 0
      self.record[symbol]['holding_return'] = 0
      self.record[symbol]['money'] = np.NaN
      self.record[symbol]['stock'] = np.NaN
      self.record[symbol]['value'] = np.NaN
      
  # trade
  def trade(self, sec_list, start_cash, start_date=None, end_date=None, stop_profit=None, stop_loss=None, benchmark='SPY'):
    
    # set start_date, end_date, sec_list, start_money, benchmark, record
    start_date = self.start_date if start_date is None else start_date
    end_date = self.end_date if end_date is None else end_date
    self.set_benchmark(benchmark=benchmark, start_date=start_date, end_date=end_date)
    self.sec_list = list(set(sec_list + ['benchmark']))
    self.start_cash = start_cash
    self.init_record()
    
    # initialize portfolio
    avg_position = start_cash #self.start_cash / len(self.sec_list)
    for symbol in self.sec_list:
      self.stock[symbol] = 0
      self.holding_price[symbol] = 0
      self.cash[symbol] = avg_position
      self.value[symbol] = avg_position

    # construct trading date list
    dates = []
    next_date = start_date
    while next_date <= end_date:
      dates.append(next_date)
      next_date = util.string_plus_day(next_date, 1)

    # go through each trading day
    for date in dates:

      # go through each stock
      for symbol in self.sec_list:
        signal_data = self.record[symbol]
        
        # if current date is trading day
        if date in signal_data.index:

          # get signal and price
          tmp_signal = signal_data.loc[date, 'signal']
          tmp_price = signal_data.loc[date, 'Close']

          # check if it is necessary to stop profit/loss
          if self.stock[symbol] > 0:
            signal_data.loc[date, 'holding_price'] = self.holding_price[symbol]
            signal_data.loc[date, 'holding_return'] = (tmp_price - signal_data.loc[date, 'holding_price']) / signal_data.loc[date, 'holding_price']
            if (stop_profit is not None and signal_data.loc[date, 'holding_return'] >= stop_profit) or (stop_loss is not None and signal_data.loc[date, 'holding_return'] <= stop_loss):
              tmp_signal = 's'

          # buy stock
          if tmp_signal == 'b':
            trade_result = buy(money=self.cash[symbol], price=tmp_price, trading_fee=3)
            self.cash[symbol] = trade_result['money']
            self.stock[symbol] += trade_result['stock']
            self.holding_price[symbol] = tmp_price

          # sell stock
          elif tmp_signal == 's':
            trade_result = sell(stock=self.stock[symbol], price=tmp_price, trading_fee=3)
            self.cash[symbol] += trade_result['money']
            self.stock[symbol] = trade_result['stock']
            self.holding_price[symbol] = 0

          else:
            pass
          
          # update stock, money, value
          self.value[symbol] = self.cash[symbol] + self.stock[symbol] * tmp_price
          signal_data.loc[date, 'money'] = self.cash[symbol]
          signal_data.loc[date, 'stock'] = self.stock[symbol]
          signal_data.loc[date, 'value'] = self.value[symbol]
          
        # if current date is not trading day
        else:
          pass

    for symbol in self.sec_list:
      self.record[symbol][['money', 'stock', 'value']] = self.record[symbol][['money', 'stock', 'value']].fillna(method='bfill')

    # calculate total value in portfolio
    total = self.record['benchmark'][['value']].copy()
    total['value'] = 0
    for k in self.record.keys():
      if k in ['benchmark', 'portfolio']:
        continue
      tmp_data = self.record[k][['value']].rename(columns={'value':k})
      total = pd.merge(total, tmp_data, how='left', left_index=True, right_index=True)
    total = total.fillna(method='bfill').fillna(0)
    total['value'] = total.sum(axis=1)
    total['Close'] = np.NaN
    total['signal'] = 'n'
    self.record['portfolio'] = total.copy()

  # visualize
  def visualize(self, symbol, start_date=None, end_date=None, is_return=False):

    # create image
    fig = plt.figure(figsize=(20, 5))  
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    trade_plot = plt.subplot(gs[0])
    money_plot = plt.subplot(gs[1], sharex=trade_plot)
    plt.setp(trade_plot.get_xticklabels(), visible=False)
    gs.update(wspace=0, hspace=0)
    
    # get plot data
    record = self.record[symbol].copy()  
    record = record[start_date:end_date].copy()
    if len(record) == 0:
      print(f'no record for {symbol}')
      return None
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
      benchmark_record = self.record['benchmark'][min_idx:max_idx].copy()
      benchmark_min_idx = benchmark_record.index.min()
      benchmark_record.fillna(method='ffill', inplace=True)

      if symbol == 'portfolio':
        benchmark_record['value'] = benchmark_record['value'] * (len(self.sec_list)-1)
      benchmark_record['original'] = benchmark_record.loc[benchmark_min_idx, 'value']
      money_plot.plot(benchmark_record.index, benchmark_record.value, label='benckmark', color='black', linestyle='--',)
    
    # set title and legend
    trade_plot.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.0) 
    trade_plot.set_title('Signals', rotation='vertical', x=-0.05, y=0.3)
    money_plot.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.0) 
    money_plot.set_title('Money', rotation='vertical', x=-0.05, y=0.3)
    hpr = finance_util.cal_HPR(data=record, start=min_idx.date(), end=max_idx.date(), dim='value', dividends=0)
    fig.suptitle(f'{symbol}: {hpr*100:.2f}%', x=0.5, y=0.95, fontsize=20)

    if is_return:
      return record

  # analysis
  def analyze(self, sort=True):

    # get records for self.sec_list
    records = dict((key,value) for key,value in self.record.items() if key in self.sec_list)
    
    # init dict for storing results
    analysis = {
        'symbol': [],
        'start_date': [],
        'end_date': [],
        'start_money': [],
        'end_money': [],
        'EAR': [],
        'sharp_ratio': [],
        'max_drawndown': []
      }

    # go through each stock
    for symbol in records.keys():
        
      # get record data
      record_data = records[symbol][self.start_date:self.end_date]      

      if len(record_data) == 0:
        print(f'no record for {symbol}')
        continue

      min_idx = record_data.index.min()
      max_idx = record_data.index.max()
      
      # analysis profit, hpr, ear, etc.
      analysis['symbol'].append(symbol)
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
    analysis = pd.DataFrame(analysis).set_index('symbol')
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
      value_sum = self.record['portfolio'].copy()
      value_sum['rate'] = value_sum['value'].pct_change().fillna(0)
      total_ear = finance_util.cal_EAR(data=value_sum, dim='value', start=None, end=None)
      total_max_drawndown = finance_util.cal_max_drawndown(data=value_sum, dim='value')
      total_sharp_ratio = finance_util.cal_sharp_ratio(data=value_sum, price_dim='value', rate_dim='rate', start=None, end=None)

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

