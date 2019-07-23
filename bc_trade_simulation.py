# -*- coding: utf-8 -*-
import pandas as pd
import math
import matplotlib.pyplot as plt
from quant import bc_util as util
from quant import bc_technical_analysis as ta_util



#----------------------------- 买卖/回测 -------------------------------------#
# 买入
def buy(money, price, trading_fee):

	# 计算可买数量, 剩余的现金
	stock = math.floor((money-trading_fee) / price)
	if stock > 0:
		money = money - trading_fee - (price*stock) 
	else:
		stock = 0
		print('Not enough money to buy')

	return {'left_money': money, 'new_stock': stock} 


# 卖出
def sell(stock, price, trading_fee):

	# 计算卖出可获得金额，股票归零
	money = stock * price - trading_fee
	if money > 0:
		stock = 0
	else:
		money = 0
		print('Not enough stock to sell')

	return {'new_money': money, 'left_stock': stock}


# 回测
def back_test(signal, buy_price='Open', sell_price='Close', money=0, stock=0, trading_fee=3, start_date=None, end_date=None, stop_profit=0.1, stop_loss=0.6, mode='signal', force_stop_loss=1,print_trading=True, plot_trading=True):	
	
	# 获取指定期间的信号, 移除冗余信号
	signal = ta_util.remove_redundant_signal(signal[start_date:end_date])

	# 初始化
	original_money = money
	last_total = total = money	

	# 交易记录
	record = {
		'date': [], 'action': [],
		'stock': [], 'price': [],
		'money': [], 'total': []
	}

	# 以信号模式进行回测
	if mode == 'signal':

		# 遍历所有交易日
		date_list = signal.index.tolist()
		for i in range(len(date_list)-1):

			# 获取当前日期
			date = date_list[i]
			
			# 次日进行交易
			next_date = date_list[i+1]
			
			# 如果触发止损条件, 平仓之后不操作
			earning = (last_total - total)/last_total
			if earning >= force_stop_loss:
				print('stop loss at earning ', earning)
				price = signal.loc[next_date, sell_price]
				tmp_trading_result = sell(stock=stock, price=price, trading_fee=trading_fee)
				continue

			# 获取交易信号, 根据信号进行交易
			action = signal.loc[date, 'signal']
			if action == 'n': # 无
				continue
			
			elif action == 'b': # 买入
				price = signal.loc[next_date, buy_price]
				tmp_trading_result = buy(money=money, price=price, trading_fee=trading_fee)
				money = tmp_trading_result.get('left_money')
				stock += tmp_trading_result.get('new_stock')
			
			elif action == 's': # 卖出
				price = signal.loc[next_date, sell_price]
				tmp_trading_result = sell(stock=stock, price=price, trading_fee=trading_fee)
				money += tmp_trading_result.get('new_money')
				stock = tmp_trading_result.get('left_stock')

			else: # 其他
				print('Invalid signal: ', action)
				tmp_result = None
			
			# 更新结果  			
			last_total = total
			total = money + stock*price

			# 记录结果
			record['date'].append(next_date)
			record['action'].append(action)
			record['price'].append(price)
			record['money'].append(money)
			record['stock'].append(stock)
			record['total'].append(total)

		# 计算当前值
		last_date = signal.index.max()
		record['date'].append(last_date)
		record['action'].append(signal.loc[last_date, 'signal'])
		record['price'].append(signal.loc[last_date, 'Close'])
		record['money'].append(money)
		record['stock'].append(stock)
		record['total'].append(money+stock*signal.loc[last_date, 'Close'])

		# 将记录转化为时序数据
		record = util.df_2_timeseries(pd.DataFrame(record), time_col='date')

		# 画出回测图
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


