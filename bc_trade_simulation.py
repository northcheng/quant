# -*- coding: utf-8 -*-
import pandas as pd
import math

# 买入
def buy(money, price, trading_fee):

	# 计算可买数量
	stock = math.floor((money-trading_fee) / price)
	if stock > 0:
		money = money - trading_fee - (price*stock) 
	else:
		print('Not enough money to buy')

	return {'money': money, 'stock': stock} 

# 卖出
def sell(stock, price, trading_fee):
	
	# 计算卖出可获得金额
	if stock > 0:
		money = stock * price - trading_fee	
		stock = 0
	else:
		print('Not enough stock to sell')

	return { 'money': money, 'stock': stock}


# 去除冗余信号
def remove_redundant_signal(signal):

	# 获取非空信号
	buy_sell_signals = signal.query('signal != "n"')
	valid_signals = []
	last_signal = 'n'

	# 遍历信号数据 
	for index, row in buy_sell_signals.iterrows():
      		
  		# 获取当前信号
  		current_signal = row['signal']  
  		
  		# 如果当前信号与上一信号一致, 则移除信号
  		if current_signal == last_signal:
    		continue
  		else:
    		valid_signals.append(index)

    	# 更新
		last_signal = current_signal

	# 移除冗余的信号
	redundant_signals = [x for x in signal.index.tolist() if x not in valid_signals]

	signal.loc[redundant_signals, 'signal'] = 'n'

	return signal

# 回测
def back_test(signal, buy_price='Open', sell_price='Close', money=0, stock=0, trading_fee=3, start_date=None, end_date=None,  stop_profit=0.1, stop_loss=0.6, mode='signal', force_stop_loss=1, print_trading=True, plot_trading=True):	
	
	# 获取指定期间的信号
  	signal = signal[start_date:end_date]

  	# 记录原始现金数量
  	original_money = money	

	# 记录交易                           
	record = {
	  	'date': [], 'action': [],
      	'stock': [], 'price': [],
      	'money': [], 'total': []
	}

	# 以盈利模式进行回测
  	if mode == 'signal':
  		
  		# 去除冗余的信号
    	buy_sell_signals = signal.query('signal != "n"')
    	trading_date = []
    	last_signal = 'n'

    	# 遍历信号数据
    	for index, row in buy_sell_signals.iterrows():
      		
      		# 获取当前信号
      		current_signal = row['signal']  
      		
      		# 如果当前信号与上一信号一致, 则移除信号
      		if current_signal == last_signal:
        		continue
      		else:
        		trading_date.append(index)

        	# 更新
  			last_signal = current_signal
  
  		# 开始交易
  		for date in trading_date:

  			# 如果当前信号刚刚发出, 则无需交易, 立即结束
  			if date == signal.index.max():
  				print('信号于', date, '发出')
  				break

  			# 获取交易信号
  			tmp_signal = signal.loc[date, 'signal']

  			# 于信号的第二天开始交易
  			tmp_data = signal[date:][1:]

  			# 记录开始交易的日期
  			tmp_trading_date = tmp_data.index.min()

  			# 买入(开盘价)
  			if tmp_signal == 'b':

  				price = tmp_data.loc[tmp_trading_date, buy_price]
  				tmp_trading_result = buy(money=money, price=price, trading_fee=trading_fee)

  			# 卖出(收盘价)
  			elif tmp_signal == 's':

  				price = tmp_data.loc[tmp_trading_date, sell_price]
  				tmp_trading_result = sell(stock=stock, price=price, trading_fee=trading_fee)

  			# 其他信号
  			else:
  				print('Error signal')
  				tmp_result = None

			# 更新结果
			date = tmp_trading_date
			action = tmp_signal  			
			price = price	
			money = tmp_result['money']
			stock = tmp_result['stock']
			total = moeny + stock*price

			# 记录结果
          	record['date'].append(date)
          	record['action'].append(action)
          	record['price'].append(price)
          	record['money'].append(money)
          	record['stock'].append(stock)
	        record['total'].append(total)