# -*- coding: utf-8 -*-
"""
Utilities used for trade simulation

:authors: Beichen Chen
"""
import pandas as pd
import math
import matplotlib.pyplot as plt
from quant import bc_util as util
from quant import bc_technical_analysis as ta_util


#----------------------------- Buy/Sell -------------------------------------#
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
  print('Not enough money to buy')

  return {'left_money': money, 'new_stock': stock} 


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
    print('Not enough stock to sell')

  return {'new_money': money, 'left_stock': stock}


def back_test(signal, buy_price='Open', sell_price='Close', money=0, stock=0, trading_fee=3, start_date=None, end_date=None, stop_profit=0.1, stop_loss=0.6, mode='signal', force_stop_loss=1, print_trading=True, plot_trading=True):  
  """
  Trade simulation with historical data

  :param signal: signal dataframe
  :param buy_price: column of buy price
  :param sell_price: column of sell price
  :param money: start money
  :param stock: start stock
  :param trading_fee: trading fee
  :param start_date: date start trading
  :param end_date: date trading stop
  :param stop_profit: the earning rate to stop profit
  :param stop_loss: the loss rate to stop loss
  :param mode: which mode to use: signal
  :param force_stop_loss: the loss rate to force stop loss in all modes
  :param print_trading: whether to print trading information
  :param plot_trading: whether to plot trading charts
  :returns: trading records in dataframe
  :raises: none
  """
  # get signal in specific period, and remove redundant(duplicated) signals
  signal = ta_util.remove_redundant_signal(signal[start_date:end_date])

  # initialization
  original_money = money
  last_total = total = money  

  # trading records
  record = {
    'date': [], 'action': [],
    'stock': [], 'price': [],
    'money': [], 'total': []
  }

  # simulate trading according to signals
  if mode == 'signal':

    # go through all trading dates
    date_list = signal.index.tolist()
    for i in range(len(date_list)-1):

      # get current date
      date = date_list[i]
      
      # trade in next day after the signal triggered
      next_date = date_list[i+1]
      
      # if force stop loss is triggered, sell all the stocks and stop trading today
      earning = (last_total - total)/last_total
      if earning >= force_stop_loss:
        print('stop loss at earning ', earning)
        price = signal.loc[next_date, sell_price]
        tmp_trading_result = sell(stock=stock, price=price, trading_fee=trading_fee)
        continue

      # trade according to today's signal
      action = signal.loc[date, 'signal']
      if action == 'n': # none
        continue
      
      elif action == 'b': # buy
        previous_money = money
        previous_stock = stock
        price = signal.loc[next_date, buy_price]
        tmp_trading_result = buy(money=money, price=price, trading_fee=trading_fee)
        money = tmp_trading_result.get('left_money')
        stock += tmp_trading_result.get('new_stock')
        if print_trading:
          print('%(date)s buy with price: %(price)s, stock: %(stock)s -> %(new_stock)s, money: %(money)s->%(new_money)s' % dict(
            date=date, price=price, 
            stock=previous_stock, new_stock=stock, 
            money=previous_money, money=money))

      elif action == 's': # sell
        previous_money = money
        previous_stock = stock
        price = signal.loc[next_date, sell_price]
        tmp_trading_result = sell(stock=stock, price=price, trading_fee=trading_fee)
        money += tmp_trading_result.get('new_money')
        stock = tmp_trading_result.get('left_stock')
        if print_trading:
          print('%(date)s sell with price: %(price)s, stock: %(stock)s -> %(new_stock)s, money: %(money)s->%(new_money)s' % dict(
            date=date, price=price, 
            stock=previous_stock, new_stock=stock, 
            money=previous_money, money=money))

      else: # others
        print('Invalid signal: ', action)
        tmp_result = None
      
      # update assets      
      last_total = total
      total = money + stock*price

      # record trading history
      record['date'].append(next_date)
      record['action'].append(action)
      record['price'].append(price)
      record['money'].append(money)
      record['stock'].append(stock)
      record['total'].append(total)

    # calculate the latest assets
    last_date = signal.index.max()
    record['date'].append(last_date)
    record['action'].append(signal.loc[last_date, 'signal'])
    record['price'].append(signal.loc[last_date, 'Close'])
    record['money'].append(money)
    record['stock'].append(stock)
    record['total'].append(money+stock*signal.loc[last_date, 'Close'])

    # transfer trading records to timeseries dataframe
    record = util.df_2_timeseries(pd.DataFrame(record), time_col='date')

    # plot trading charts
    if plot_trading:
      buying_points = record.query('action == "b"')
      selling_points = record.query('action == "s"')

      f, ax = plt.subplots(figsize = (20, 3))
      plt.plot(signal[['Close']])
      plt.scatter(buying_points.index,buying_points.price, c='green')
      plt.scatter(selling_points.index,selling_points.price, c='red')

      total_value_data = pd.merge(signal[['Close']], record[['money', 'stock', 'action']], how='left', left_index=True, right_index=True)
      total_value_data.fillna(method='ffill', inplace=True)
      total_value_data['original'] = original_money
      total_value_data['total'] = total_value_data['Close'] * total_value_data['stock'] + total_value_data['money']
      total_value_data[['total', 'original']].plot(figsize=(20, 3))

    return record


