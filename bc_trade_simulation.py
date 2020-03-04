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


def back_test(df, signal_col='signal', start_date=None, end_date=None, start_money=0, start_stock=0, trading_fee=3, stop_loss=-0.03, stop_earning=0.05, trade_on_next_day=False, print_trading=True, plot_trading=True):  
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
  :param mode: which mode to use: signal
  :param stop_loss: the rate to force stop loss 
  :param stop_earning: the rate to force stop earning
  :param print_trading: whether to print trading information
  :param plot_trading: whether to plot trading charts
  :returns: trading records in dataframe
  :raises: none
  """
  # get signal in specific period, and remove redundant(duplicated) signals
  signal = ta_util.remove_redundant_signal(df[start_date:end_date])
  date_list = signal.index.tolist()

  # initialization
  stock = start_stock
  money = start_money 
  last_total = total = money
  holding_price = 0
  holding_return = 0
  record = {'date': [], 'action': [], 'stock': [], 'price': [], 'money': [], 'total': [], 'holding_price': [], 'holding_return': []}

  # set buy/sell price
  if not trade_on_next_day:
    buy_price = 'Close'
    sell_price = 'Close'
  else:
    buy_price = 'Open'
    sell_price = 'Open'

  # print trading info
  def print_trading_info(date, action, price, previous_stock, stock, previous_money, money, holding_price, holding_return):
    action_name = {'b': 'buy', 's': 'sell', 'n': 'none', 'stop_earning': 'stop_earning', 'stop_loss': 'stop_loss'}[action]
    trading_info = f'[{date}] {action_name:<4}: {price:>7.2f}, stock: {previous_stock:>5} -> {stock:<5}, money: {previous_money:>8.1f} -> {money:<8.1f}, holding: {holding_price:>7.2f} | {holding_return:<4.2f}'
    print(trading_info)

  # go through all trading dates
  slse_triggered = False
  for i in range(len(date_list)-1):
    tmp_record = None
    date = date_list[i]
    trade_date = date
    action = signal.loc[date, 'signal']
    if trade_on_next_day:
      trade_date = date_list[i+1]
      
      # next_date = date_list[i+1]
    
    
    
    # check whether to stop loss or earning if holding stock
    if stock > 0 and holding_price > 0:
      current_price = signal.loc[date, sell_price]
      holding_return = (current_price - holding_price) / holding_price

      # if triggered stop loss or stop earning, sell all the stocks
      if ((stop_loss is not None) and (holding_return <= stop_loss)): 
        action = 's'
        slse_triggered = True
        if print_trading:
          print('[{date}]stop loss at: {holding_return:.4f}'.format(holding_return=holding_return, date=date))
      
      elif ((stop_earning is not None) and (holding_return >= stop_earning)):
        action = 's'
        slse_triggered = True
        if print_trading:  
          print('[{date}]stop earning at: {holding_return:.4f}'.format(holding_return=holding_return, date=date))

    # record money and stock
    previous_money = money
    previous_stock = stock

    # buy signal
    if action == 'b': 
      price = signal.loc[trade_date, buy_price]
      if money > price:
        tmp_record = buy(money=money, price=price, trading_fee=trading_fee)
        money = tmp_record.get('money')
        bought_stock = tmp_record.get('stock')
        if bought_stock > 0:
          holding_price = price
          holding_return = 0
        stock += bought_stock
        
    # sell single
    elif action == 's': 
      price = signal.loc[trade_date, sell_price]
      if stock > 0:
        tmp_record = sell(stock=stock, price=price, trading_fee=trading_fee)
        stock = tmp_record.get('stock')
        got_money = tmp_record.get('money')
        if got_money > 0:
          holding_return = (price - holding_price) / holding_price
          holding_price = 0
        money += got_money
        
    # others
    else: 
      price = signal.loc[trade_date, sell_price]

    if print_trading and tmp_record is not None:
      print_trading_info(date=date.date(), action=action, price=price, 
        previous_stock=previous_stock, stock=stock, previous_money=previous_money, money=money, 
        holding_price=holding_price, holding_return=holding_return)    

    # stop earning or stop loss
    if slse_triggered:
      signal['signal'] = 'n'
    
    # update total value
    last_total = total
    total = money + stock * price

    # record trading history
    record['date'].append(trade_date)
    record['action'].append(action)
    record['price'].append(price)
    record['money'].append(money)
    record['stock'].append(stock)
    record['total'].append(total)
    record['holding_price'].append(holding_price)
    record['holding_return'].append(holding_return)

  # transfer trading records to timeseries dataframe
  record = util.df_2_timeseries(pd.DataFrame(record), time_col='date')

  # plot trading charts
  if plot_trading:
    buying_points = record.query('action == "b"')
    selling_points = record.query('action == "s"')

    plt.subplots(figsize = (20, 3))
    plt.plot(signal[['Close']])
    plt.scatter(buying_points.index,buying_points.price, c='green')
    plt.scatter(selling_points.index,selling_points.price, c='red')

    total_value_data = pd.merge(signal[['Close']], record[['money', 'stock', 'action']], how='left', left_index=True, right_index=True)
    total_value_data.fillna(method='ffill', inplace=True)
    total_value_data['original'] = start_money
    total_value_data['total'] = total_value_data['Close'] * total_value_data['stock'] + total_value_data['money']
    total_value_data[['total', 'original']].plot(figsize=(20, 3))

  return record


