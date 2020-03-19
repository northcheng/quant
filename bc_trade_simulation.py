# -*- coding: utf-8 -*-
"""
Utilities used for trade simulation

:authors: Beichen Chen
"""
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

    # create image
    fig = plt.figure(figsize=(20, 5))  
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    trade_plot = plt.subplot(gs[0])
    money_plot = plt.subplot(gs[1], sharex=trade_plot)
    plt.setp(trade_plot.get_xticklabels(), visible=False)
    gs.update(wspace=0, hspace=0)
    
    # plot trade signals
    buying_points = record.query('action == "b"')
    selling_points = record.query('action == "s"')
    trade_plot.plot(signal.index, signal[['Close']], label='Close')
    trade_plot.scatter(buying_points.index,buying_points.price, c='green', label='Buy')
    trade_plot.scatter(selling_points.index,selling_points.price, c='red', label='Sell')

    # plot money flow chart
    total_value_data = pd.merge(signal[['Close']], record[['money', 'stock', 'action']], how='left', left_index=True, right_index=True)
    total_value_data.fillna(method='ffill', inplace=True)
    total_value_data['original'] = start_money
    total_value_data['total'] = total_value_data['Close'] * total_value_data['stock'] + total_value_data['money']
    money_plot.plot(total_value_data.index, total_value_data.total, label='Value')
    money_plot.plot(total_value_data.index, total_value_data.original, label='Principal')

    money_plot.fill_between(total_value_data.index, total_value_data.original, total_value_data.total, where=total_value_data.total > total_value_data.original, facecolor='green', interpolate=True, alpha=0.1)
    money_plot.fill_between(total_value_data.index, total_value_data.original, total_value_data.total, where=total_value_data.total < total_value_data.original, facecolor='red', interpolate=True, alpha=0.1)

    # set title and legend
    trade_plot.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.0) 
    trade_plot.set_title('Signals', rotation='vertical', x=-0.05, y=0.3)
    money_plot.legend(bbox_to_anchor=(1.02, 0.), loc=3, ncol=1, borderaxespad=0.0) 
    money_plot.set_title('Money', rotation='vertical', x=-0.05, y=0.3)

    fig.suptitle('Trade Simulation', x=0.5, y=0.95, fontsize=20)

  return record


class Trader:

  sec_list = []
  start_cash = 0
  record = {}

  cash = {}
  stock = {}
  holding_price = {}
  value = {}

  # init
  def __init__(self, sec_list, start_cash, signals):

    # initialize stock list and start cash
    self.sec_list = sec_list
    self.start_cash = start_cash

    # initialize trading record
    for k in signals.keys():
      sec_code = k.split('_')[0]
      self.record[sec_code] = signals[k][['signal', 'Close']].copy()
      self.record[sec_code]['holding_price'] = 0
      self.record[sec_code]['holding_return'] = 0
 
  # trade
  def trade_with_fixed_position(self, start_date, end_date, stop_profit=None, stop_loss=None):

    # 平均分配仓位资金
    avg_position = self.start_cash / len(self.sec_list)
    for sec_code in self.sec_list:
      self.stock[sec_code] = 0
      self.cash[sec_code] = avg_position
      self.value[sec_code] = avg_position

    # 构造日期列表
    dates = []
    next_date = start_date
    while next_date <= end_date:
      dates.append(next_date)
      next_date = util.string_plus_day(next_date, 1)

    # 遍历日期列表
    for date in dates:
      date_signal = []

      # 遍历每只股票的信号
      for sec_code in self.record.keys():
        signal_data = self.record[sec_code]
        
        # 如果是交易日
        if date in signal_data.index:

          # 获取交易信号与价格
          tmp_signal = signal_data.loc[date, 'signal']
          tmp_price = signal_data.loc[date, 'Close']

          # 计算持有收益, 查看是否需要止盈/止损
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
                  
        # 跳过非交易日
        else:
          pass


  def trade_with_dynamic_position(self, start_date, end_date, stop_profit=None, stop_loss=None, min_position=2000, max_position=10000):