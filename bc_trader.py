import math
import logging
import pytz
import time
import datetime
import pandas as pd

from quant import bc_data_io as io_util
from quant import bc_util as util

from futu import (OrderType, OrderStatus, TrdSide, RET_OK, RET_ERROR)
from futu import (OpenQuoteContext, OpenSecTradeContext, TrdMarket, SecurityFirm, Currency)
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.common.consts import (Language,  Market, BarPeriod, QuoteRight) # 语言, 市场, k线周期, 复权类型
from tigeropen.common.util.contract_utils import (stock_contract, option_contract, future_contract) # 股票合约, 期权合约, 期货合约
from tigeropen.common.util.order_utils import (market_order, limit_order, stop_order, stop_limit_order, trail_order, order_leg) # 市价单, 限价单, 止损单, 限价止损单, 移动止损单, 附加订单

ACCOUNT_GROUPS = {
  'tiger': {'global_account': 'real', 'simulation_account':'simu'},
  'futu': {'REAL': 'real', 'SIMULATE':'simu'}}

class Trader(object):
  
  # init
  def __init__(self, platform, account_type, config, logger_name=None):
    
    # get logger
    if logger_name is not None:
      self.logger = logging.getLogger(logger_name)
    else:
      logger_level = logging.INFO
      logger_name = f'{platform}_{account_type}'
      logger_format = logging.Formatter('[%(asctime)s] - %(message)s ', '%Y-%m-%d %H:%M:%S')
      console = logging.StreamHandler()
      console.setLevel(logger_level)
      console.setFormatter(logger_format)
      self.logger = logging.getLogger(logger_name)
      self.logger.setLevel(level = logger_level)
      for hdlr in self.logger.handlers:
        self.logger.removeHandler(hdlr)
      self.logger.addHandler(console)
    
    # read user info, position record from local files
    self.user_info = io_util.read_config(file_path=config['trader_path'], file_name='user_info.json').get(platform)
    self.position_record = io_util.read_config(file_path=config['config_path'], file_name=f'position.json')[platform]
    self.record = self.position_record[account_type].copy()
    self.eod_api_key = config['api_key']['eod']

    # set account, account type
    self.platform = platform
    self.account = self.user_info[account_type]
    self.account_type = account_type

    # get quote and trade context, position, asset
    self.set_client_config(config=config)
    self.open_quote_client()
    self.open_trade_client()
    self.update_position()
    self.update_asset()
    # update position record
    self.synchronize_position_record(config=config)
    self.logger.info(f'[{platform}]: instance created - {logger_name}')
  
  # get client config
  def set_client_config(self, config):
    pass
  
  # get quote client
  def open_quote_client(self):
    self.quote_client = None
  
  # close quote client
  def close_trade_client(self):
    pass
  
  # get trade client
  def open_trade_client(self):
    self.trade_client = None
  
  # close trade client
  def close_quote_client(self):
    pass
  
  # finalize trader
  def finalize(self):
    pass
  
  # get position
  def update_position(self):
    self.position = None
  
  # get asset
  def update_asset(self):
    self.asset = None
    
  # get available cash
  def get_available_cash(self):
    
    available_cash = 0
    
    if self.trade_client is None:
      pass
    else:
      self.update_asset()
      if len(self.asset) > 0:
        available_cash = self.asset.loc[0, 'cash']
      else:
        print('Not able to get available cash')
        
    return available_cash
  
  # get quantity of symbol currently in the position
  def get_in_position_quantity(self, symbol, get_briefs=False):

    # initialize affordable quantity
    quantity = 0

    # get position summary
    self.update_position(get_briefs=get_briefs)
    position = self.position.copy()
    if len(position) > 0:
      position = position.set_index('symbol')
      if symbol in position.index:
        quantity = position.loc[symbol, 'quantity']

    return quantity

  # check whether it is affordable to buy certain amount of a stock
  def get_affordable_quantity(self, symbol, cash=None, trading_fee=3):

    # initialize affordable quantity and available cash
    quantity = 0
    available_cash = self.get_available_cash() if (cash is None) else cash

    # get latest price of stock
    stock_brief = io_util.get_stock_briefs(symbols=[symbol], source='eod', api_key=self.eod_api_key).set_index('symbol')
    latest_price = stock_brief.loc[symbol, 'latest_price']

    # check if it is affordable
    quantity = math.floor((available_cash-trading_fee)/latest_price)

    return quantity

  # update position for an account
  def update_position_record(self, config, init_cash=None, init_position=None, start_time=None, end_time=None, is_print=True):
    
    # set default values
    account_group = ACCOUNT_GROUPS[self.platform][self.account_type]
    init_cash = config['trade']['init_cash'][self.platform][account_group] if (init_cash is None) else init_cash
    init_position = 0 if (init_position is None) else init_position

    try:      
      # get today filled orders
      orders = self.get_orders(start_time, end_time)

      # update position records
      for index, row in orders.iterrows():
        symbol = row['code'] # order.contract.symbol
        action = row['trd_side'] # order.action
        quantity = row['dealt_qty'] # order.quantity - order.remaining
        commission = 3 # order.commission
        avg_fill_price = row['dealt_avg_price']

        # init record if not exist
        if symbol not in self.record.keys():
          self.record[symbol] = {'cash': init_cash, 'position': init_position}
        record_cash = self.record[symbol]['cash']
        record_position = self.record[symbol]['position']

        # calculate new cash and position
        if action == 'BUY':
          cost = avg_fill_price * quantity + commission
          new_cash = record_cash - cost
          new_position = record_position + quantity
          print(action, symbol, cost, new_cash, new_position)
        
        elif action == 'SELL':
          acquire = avg_fill_price * quantity - commission
          new_cash = record_cash + acquire
          new_position = record_position - quantity
          print(action, symbol, cost, new_cash, new_position)

        else:
          new_cash = record_cash
          new_position = record_position
          print(action, symbol, new_cash, new_position)

        # update record
        # if new_cash >= 0 and new_position >= 0:
        self.record[symbol]['cash'] = new_cash
        self.record[symbol]['position'] = new_position
        if is_print:
          self.logger.info(f'[{self.account_type[:4]}]: updating position record for {symbol} {record_cash, record_position} -> {new_cash, new_position}')

      # update position_record
      self.position_record = io_util.read_config(file_path=config['config_path'], file_name='position.json')[self.platform]
      self.position_record[self.account_type] = self.record.copy()
      self.position_record['updated'][self.account_type] = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
      io_util.modify_config(self.platform, self.position_record, file_path=config['config_path'], file_name='position.json', is_print=False)
      # io_util.create_config_file(config_dict=self.position_record, file_path=config['config_path'], file_name='tiger_position_record.json')
      
    except Exception as e:
      self.logger.exception(f'[erro]: fail updating position records for {self.account_type}, {e}')
  
  # synchronize position record with real position status
  def synchronize_position_record(self, config):
    
    if self.position is None:
      pass
    
    else:
      account_group = ACCOUNT_GROUPS[self.platform][self.account_type]
      pool_name = config['trade']['pool'][self.platform][account_group]  
      pool = config['selected_sec_list'][pool_name]
      init_cash = config['trade']['init_cash'][self.platform][account_group]

      for symbol in pool:
        if symbol not in self.record.keys():
          self.record[symbol] = {'cash': init_cash, 'position': 0}

      # get record position and real position then compare to each other
      record_conflicted = False
      
      position_dict = {}
      self.update_position()
      if len(self.position) > 0:
        position_dict = dict(self.position[['symbol', 'quantity']].values)

      # compare position record with real position
      to_pop = []
      for symbol in self.record.keys():

        if symbol not in pool:
          to_pop.append(symbol)
          continue

        # update position in record
        record_position = self.record[symbol]['position']
        current_position = 0 if (symbol not in position_dict.keys()) else position_dict[symbol]
        if current_position != record_position:
          record_conflicted = True
          if current_position > 0:
            self.record[symbol] = {'cash': 0, 'position': current_position}
          else:
            self.record[symbol] = {'cash': init_cash, 'position': 0}
          self.logger.error(f'[{self.account_type[:4]}]: {symbol} position({current_position}) rather than ({record_position}), reset record')
      
      # pop symbols not in position
      for symbol in to_pop:
        self.record.pop(symbol)

      # add record for position that not recorded
      for symbol in [x for x in position_dict.keys() if (x in pool and x not in self.record.keys())]:
        record_conflicted = True
        self.record[symbol] = {'cash': 0, 'position': position_dict[symbol]}
        self.logger.error(f'[{self.account_type[:4]}]: {symbol} position({position_dict[symbol]}) not in record, add record')

      # update position_record
      if record_conflicted:
        self.position_record[self.account_type] = self.record.copy()
        io_util.modify_config(self.platform, self.position_record, file_path=config['config_path'], file_name='position.json', is_print=False)
        # io_util.create_config_file(config_dict=self.position_record, file_path=config['config_path'], file_name='_position_record.json')

  # update portfolio for an account
  def update_portfolio_record(self, config, position=None, get_briefs=True, is_print=True):

    # get position summary
    if position is None:
      self.update_position(get_briefs=get_briefs)
      position = self.position
    position.set_index('symbol', inplace=True)
    position = position.round(2)

    # get asset summary
    cash = 0
    net_value = 0
    market_value = 0
    self.update_asset()
    asset = self.asset
    if len(asset) > 0:
      net_value = asset.loc[0, 'net_value']
      market_value = asset.loc[0, 'holding_value']
      cash = asset.loc[0, 'cash']

    # post process
    if market_value == float('inf'):
      market_value = position['market_value'].sum().round(2)

    # load portfolio record
    portfolio_record = io_util.read_config(file_path=config['config_path'], file_name='portfolio.json')
    old_net_value = portfolio_record[self.platform][self.account_type].get('net_value')
    support = portfolio_record[self.platform][self.account_type].get('portfolio').get('support')
    resistant = portfolio_record[self.platform][self.account_type].get('portfolio').get('resistant')

    # update portfolio record for current account
    portfolio_record[self.platform][self.account_type]['portfolio'] = position.to_dict()
    portfolio_record[self.platform][self.account_type]['portfolio']['support'] = {}
    portfolio_record[self.platform][self.account_type]['portfolio']['resistant'] = {}

    quantity = portfolio_record[self.platform][self.account_type]['portfolio'].get('quantity')
    if quantity is not None:
      if support is not None:
        for symbol in quantity.keys():
          portfolio_record[self.platform][self.account_type]['portfolio']['support'][symbol] = support.get(symbol)

      if resistant is not None:
        for symbol in quantity.keys():
          portfolio_record[self.platform][self.account_type]['portfolio']['resistant'][symbol] = resistant.get(symbol)

    portfolio_record[self.platform][self.account_type]['market_value'] = market_value
    portfolio_record[self.platform][self.account_type]['net_value'] = net_value
    portfolio_record[self.platform][self.account_type]['cash'] = cash
    portfolio_record[self.platform][self.account_type]['updated'] = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
    io_util.create_config_file(config_dict=portfolio_record, file_path=config['config_path'], file_name='portfolio.json')

    # print
    if is_print:
      self.logger.info(f'[{self.account_type[:4]}]: net value {old_net_value} --> {net_value}')

  # auto trade according to signals
  def signal_trade(self, signal, money_per_sec, order_type='market', trading_fee=5, pool=None, according_to_record=True, minimum_position=None):    
    
    # set symbol to index
    if len(signal) > 0:

      # filter sec with pool
      if pool is not None:
        filtered_list = [x for x in signal.index if x in pool]
        signal = signal.loc[filtered_list, signal.columns].copy()

    # if signal list is not empty
    if len(signal) > 0:
      # # get latest price for signals
      # signal_brief = io_util.get_stock_briefs(symbols=signal.index.tolist(), source='eod', api_key=self.eod_api_key).set_index('symbol')
      # signal = pd.merge(signal, signal_brief[['latest_price']], how='left', left_index=True, right_index=True)

      # get in-position quantity and latest price for signals
      self.update_position(get_briefs=False)
      position = self.position
      if len(position) == 0:
        position = pd.DataFrame({'symbol':[], 'quantity':[]})
      else:
        position = position[['symbol', 'quantity']].copy()
      position = position.set_index('symbol')
      signal = pd.merge(signal, position[['quantity']], how='left', left_index=True, right_index=True).fillna(0)

      # sell
      # get sell signals
      sell_signal = signal.query('action == "s"')
      if len(sell_signal) > 0:
        # go through sell signals
        for symbol in sell_signal.index:
          # check whether symbol is in position
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity > 0:
            if order_type == 'limit':
              price = signal.loc[symbol, 'latest_price']
            else:
              price = None
            trade_summary = self.trade(symbol=symbol, action='SELL', quantity=in_position_quantity, price=price, print_summary=False)
            self.logger.info(trade_summary)
          else:
            self.logger.info(f'[SELL]: {symbol} skipped (not in position)')
      else:
        self.logger.info(f'[SELL]: no signal')

      # buy
      # get available cash, set minimum position
      available_cash = self.get_available_cash()
      if minimum_position is None:
        minimum_position = money_per_sec

      # get buy signals which not in posiitons yet
      default_money_per_sec = money_per_sec
      buy_signal = signal.query('action == "b"')
      if len(buy_signal) > 0:
        # go through buy signals
        for symbol in buy_signal.index:

          # break when available cash is below 200
          if available_cash <= minimum_position:
            self.logger.info(f'[BUY]: Available cash is too low({available_cash}/{minimum_position}), stop buying')
            break

          # check whether symbol is already in position
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity == 0:
            # set money used to establish a new position
            if according_to_record:
              if (symbol in self.record.keys()) and (self.record[symbol]['position']==0):
                money_per_sec = self.record[symbol]['cash']
              else:
                money_per_sec = default_money_per_sec

            # check whether there is enough available money 
            money_per_sec = available_cash if (money_per_sec > available_cash) else money_per_sec

            # calculate quantity to buy
            quantity = math.floor((money_per_sec-trading_fee)/signal.loc[symbol, 'latest_price'])
            if quantity > 0:
              if order_type == 'limit':
                price = signal.loc[symbol, 'latest_price']
              else:
                price = None
              trade_summary = self.trade(symbol=symbol, action='BUY', quantity=quantity, price=price, print_summary=False)
              self.logger.info(trade_summary)

              # update available cash
              available_cash -= quantity * signal.loc[symbol, 'latest_price']
            else:
              self.logger.info(f'[BUY]: not enough money')
              continue
          else:
            self.logger.info(f'[BUY]: {symbol} skipped (already in position:{in_position_quantity})')
            continue
      else:
       self.logger.info(f'[BUY]: no signal')
    else:
      self.logger.info(f'[SKIP]: no signal')
    
  # auto trader according to conditions
  def condition_trade(self, condition_df):

    # initialize return
    trade_summary = []
    
    # trade
    if len(condition_df) > 0:
      for symbol, row in condition_df.iterrows():
        try:

          # check condition 
          condition = row['condition'].replace('\'', '\"')
          condition = f'index == "{symbol}" and {condition}'
          check_result = condition_df.query(condition)
          
          # condition matched
          if len(check_result) == 1:
            
            # place order
            action = row['action']
            quantity = row['quantity']
            order_type = row['order_type']
            price = row['price'] if order_type in ['limit'] else None # 若为市价单, 则价格设为None
            tmp_trade_summary = self.trade(symbol=symbol, action=action, quantity=quantity, price=price, print_summary=False)
            trade_summary.append(tmp_trade_summary)
              
        except Exception as e:
          self.logger.error(f'Error when placing condition order for {symbol}:\n{row}', e)
          continue
      
    return trade_summary
  
  # stop loss or stop profit or clear all position
  def cash_out(self, stop_loss_rate=None, stop_profit_rate=None, stop_loss_rate_inday=None, stop_profit_rate_inday=None, clear_all=False, get_briefs=True, print_summary=True):
    
    # get current position with summary
    self.update_position(get_briefs=get_briefs)
    position = self.position.copy()
    
    if len(position) > 0:

      # set symbol as index
      position = position.set_index('symbol')
      cash_out_list = []

      # if clear all position
      if clear_all:
        cash_out_list = position.index.tolist()
      else:
        stop_loss_list = [] if stop_loss_rate is None else position.query(f'rate < {stop_loss_rate}').index.tolist() 
        stop_profit_list = [] if stop_profit_rate is None else position.query(f'rate > {stop_profit_rate}').index.tolist() 
        stop_loss_list_inday = [] if stop_loss_rate_inday is None else position.query(f'rate_inday < {stop_loss_rate_inday}').index.tolist() 
        stop_profit_list_inday = [] if stop_profit_rate_inday is None else position.query(f'rate_inday > {stop_profit_rate_inday}').index.tolist() 
        cash_out_list = list(set(stop_loss_list + stop_profit_list + stop_loss_list_inday + stop_profit_list_inday))
        
      # cash out
      if len(cash_out_list) > 0:
        cash_out_position =  position.loc[cash_out_list, ].copy()
        self.logger.info(f'[STOP]: LOSS: {stop_loss_list}, PROFIT: {stop_profit_list}')
        self.logger.info(f'[STOP]: LOSS_INDAY: {stop_loss_list}, PROFIT_INDAY: {stop_profit_list}')

        for index, row in cash_out_position.iterrows():
          self.trade(symbol=index, action='SELL', quantity=row['quantity'], print_summary=print_summary)
    
  

  
class Futu(Trader):
  
  def __init__(self, platform, account_type, config, logger_name=None):
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)
    
  # get quote client
  def open_quote_client(self):

    try:
      host = self.user_info['host']
      port = self.user_info['port']
      is_encrypt = self.user_info['is_encrypt']
      self.quote_client = OpenQuoteContext(host=host, port=port, is_encrypt=is_encrypt)
    except Exception as e:
      self.quote_client = None
      self.logger.exception(f'[erro]: can not create quote context:{e}')

  # exit current quote client
  def close_quote_client(self):
    if self.quote_client is not None:
      self.quote_client.close()
      self.quote_client = None
      
  # get trade context
  def open_trade_client(self, market='US'):

    try:
      self.market = market
      host = self.user_info['host']
      port = self.user_info['port']
      is_encrypt = self.user_info['is_encrypt']

      if market == 'US':
        # self.trade_client = OpenUSTradeContext(host=host, port=port, is_encrypt=is_encrypt)
        self.trade_client = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=host, port=port, is_encrypt=is_encrypt, security_firm=SecurityFirm.FUTUSECURITIES)
      elif market == 'HK':    
        # self.trade_client = OpenHKTradeContext(host=host, port=port, is_encrypt=is_encrypt)
        self.trade_client = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host=host, port=port, is_encrypt=is_encrypt, security_firm=SecurityFirm.FUTUSECURITIES)
      else:
        print(f'Unknown market {market}')
        self.trade_client = None
      # 综合账户不再区分市场
      
    except Exception as e:
      self.trade_client = None
      self.logger.exception(f'[erro]: can not create trade context:{e}')

  # exit current trade client
  def close_trade_client(self):
    if self.trade_client is not None:
      
      # 锁定实盘交易
      if self.account_type == 'real':
        ret, msg = self.trade_client.unlock_trade(is_unlock=False)
        if ret != RET_OK:
          self.logger.exception(f'[erro]: can not lock trade:{ret} - {msg}')
        else:
          self.logger.info(f'[futu]: lock trade')

      self.trade_client.close()
      self.trade_client = None

  # finalize trader
  def finalize(self):
    self.close_quote_client()
    self.close_trade_client()
    self.logger.info(f'[fin]: instance finalized - {self.logger.name}')
    
  # get summary of position
  def update_position(self, get_briefs=False):
    result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'rate_inday':[], 'market_value':[], 'latest_time':[]})
    
    # 若为实盘先解锁交易
    if self.account_type == 'real':
      ret, msg = self.trade_client.unlock_trade(password_md5=self.user_info['unlock_pwd'], is_unlock=True)
      if ret != RET_OK:
        self.logger.exception(f'[erro]: can not unlock trade:{ret} - {msg}')
      else:
        self.logger.info(f'[futu]: unlock trade')

    # 失败重试
    retry_count = 0
    while retry_count < 1:
      
      retry_count += 1
      try:
        
        # get positions
        _, position = self.trade_client.position_list_query(trd_env=self.account_type)
        if type(position) == pd.DataFrame and len(position) > 0:

          # rename columns, add extra columns
          position.rename(columns={'code':'symbol', 'qty':'quantity', 'cost_price': 'average_cost', 'nominal_price':'latest_price', 'pl_ratio':'rate', 'market_val': 'market_value'}, inplace=True)
          position['symbol'] = position['symbol'].apply(lambda x: x.split('.')[1])
          position['rate'] = round(position['rate'] / 100, 2)
          position['rate_inday'] = 0
          position['latest_time'] = None
          
          # get realtime data for stock in position
          if get_briefs:
            status = io_util.get_stock_briefs(symbols=position.symbol.tolist(), source='eod', api_key=self.eod_api_key)
            if len(status) > 0:

              # merge dataframes
              key_col = 'symbol'
              duplicated_col = [x for x in position.columns if (x in status.columns and x not in [key_col])]
              position.drop(duplicated_col, axis=1, inplace=True)
              position = pd.merge(position, status, how='left', left_on=key_col, right_on=key_col)
              position['rate'] = round((position['latest_price'] - position['average_cost']) / position['average_cost'], 2)
              position['rate_inday'] = round((position['Open'] - position['Close']) / position['Open'], 2)
              position['market_value'] = round(position['latest_price'] * position['quantity'], 2)

          # select columns
          result = position[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'rate_inday', 'market_value', 'latest_time']].copy()
          
          # break when finish
          break

      except Exception as e:
        self.logger.exception(f'[erro]: can not get position summary: {e}')
        time.sleep(5)
        continue

    self.position = result
    
  # get summary of asset
  def update_asset(self):

    # 若为实盘先解锁交易
    if self.account_type == 'real':
      ret, msg = self.trade_client.unlock_trade(password_md5=self.user_info['unlock_pwd'], is_unlock=True)
      if ret != RET_OK:
        self.logger.exception(f'[erro]: can not unlock trade:{ret} - {msg}')
      else:
        self.logger.info(f'[futu]: unlock trade')
      
    try:
      ret_acc_list, acc_list = self.trade_client.get_acc_list()
      acc_idx = acc_list.query(f'trd_env == "{self.account_type}"').index
      if len(acc_idx) > 0:
        acc_id = acc_list.loc[acc_idx[0], 'acc_id']
      else:
        acc_id = self.account_type
      
      ret_assets, asset = self.trade_client.accinfo_query(trd_env=self.account_type, currency=Currency.USD)
      if type(asset) != str:
        asset['account'] = acc_id
        self.asset = asset[['account', 'total_assets', 'market_val',  'cash', 'avl_withdrawal_cash', 'realized_pl', 'unrealized_pl']].rename(columns={'total_assets':'net_value', 'market_val': 'holding_value', 'avl_withdrawal_cash':'available_cash', 'realized_pl':'pnl', 'unrealized_pl':'holding_pnl'})
      else:
        self.logger.error(f'[erro]: get asset - {asset}')      
    except Exception as e:
      self.asset = None
      self.logger.exception(f'[erro]: can not get asset summary: {e}')

  # get orders
  def get_orders(self, start_time=None, end_time=None):

    # 若为实盘先解锁交易
    if self.account_type == 'real':
      ret, msg = self.trade_client.unlock_trade(password_md5=self.user_info['unlock_pwd'], is_unlock=True)
      if ret != RET_OK:
        self.logger.exception(f'[erro]: can not unlock trade:{ret} - {msg}')
      else:
        self.logger.info(f'[futu]: unlock trade')
    
    start_time = datetime.datetime.now().strftime(format="%Y-%m-%d") if (start_time is None) else start_time
    end_time = start_time if (end_time is None) else end_time
    
    ret, orders = self.trade_client.history_order_list_query(trd_env=self.account_type, status_filter_list=[OrderStatus.FILLED_PART, OrderStatus.FILLED_ALL], start=start_time, end=end_time)
    orders = orders[['code', 'trd_side', 'order_type', 'order_status', 'qty', 'price', 'dealt_qty', 'dealt_avg_price', 'order_id', 'create_time', 'updated_time']].copy()
    orders['code'] = orders['code'].apply(lambda x: x.split('.')[1]) 
    return orders
    
  # buy or sell stocks
  def trade(self, symbol, action, quantity, price=None, print_summary=True):

    # 若为实盘先解锁交易
    if self.account_type == 'real':
      ret, msg = self.trade_client.unlock_trade(password_md5=self.user_info['unlock_pwd'], is_unlock=True)
      if ret != RET_OK:
        self.logger.exception(f'[erro]: can not unlock trade:{ret} - {msg}')
      else:
        self.logger.info(f'[futu]: unlock trade')
        
    trade_summary = ''
    try:

      # set price according to order type
      if price is None:
        order_type = OrderType.MARKET
        if action == 'BUY':
          price =  0.1
        elif action == 'SELL':
          price = 1000000
      else:
        order_type = OrderType.NORMAL

      # construct trade summary
      trade_summary += f'[{action}]: {symbol} X {quantity} ({order_type}:{price}) | '

      # place buy order if possible
      if action == 'BUY':
        trade_side = TrdSide.BUY
        affordable_quantity = self.get_affordable_quantity(symbol=symbol)
        if quantity <= affordable_quantity:
          ret_place_order, order_info = self.trade_client.place_order(price=price, qty=quantity, code=f'{self.market}.{symbol}', trd_side=trade_side, order_type=order_type, trd_env=self.account_type, remark=None)
          if ret_place_order == RET_OK:
            trade_summary += f'SUCCEED: {order_info.loc[0, "order_id"]}'
          else:
            trade_summary += f'FAILED: {order_info}'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity}/{quantity})'

      # place sell order if holding enough stocks
      elif action == 'SELL':
        trade_side = TrdSide.SELL
        in_position_quantity = self.get_in_position_quantity(symbol)
        if in_position_quantity >= quantity:
          ret_place_order, order_info = self.trade_client.place_order(price=price, qty=quantity, code=f'{self.market}.{symbol}', trd_side=trade_side, order_type=order_type, trd_env=self.account_type, remark=None)
          if ret_place_order == RET_OK:
            trade_summary += f'SUCCEED: {order_info.loc[0, "order_id"]}'
          else:
            trade_summary += f'FAILED: {order_info}'
        else:
          trade_summary += f'FAILED: Not enough stock to sell({in_position_quantity}/{quantity})'

      # other actions
      else:
        trade_summary += f'FAILED: Unknown action {action}'

    except Exception as e:
      trade_summary += f'FAILED: {e}'

    # print trade summary
    if print_summary: 
      self.logger.info(trade_summary)

    return trade_summary
  



class Tiger(Trader):
  
  def __init__(self, platform, account_type, config, logger_name=None):
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)  
    
    # get market status and trade time
    self.update_trade_time()

  # get client config
  def set_client_config(self, config):
    # initialize client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=self.user_info['sandbox_debug'])
    self.client_config.private_key = read_private_key(config['trader_path'] + self.user_info['private_key_name'])
    self.client_config.tiger_id = str(self.user_info['tiger_id'])
    self.client_config.language = Language.en_US
    self.client_config.account = self.account 
  
  # get quote client
  def open_quote_client(self):
    self.quote_client = QuoteClient(self.client_config)
    
  # exit current quote client
  def close_quote_client(self):
    if self.quote_client is not None:
      self.quote_client = None
  
  # get trade client
  def open_trade_client(self):
    self.trade_client = TradeClient(self.client_config)
    
  # exit current trade client
  def close_trade_client(self):
    if self.trade_client is not None:
      self.trade_client = None
  
  # get summary of position
  def update_position(self, get_briefs=False):
    
    result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'rate_inday':[], 'market_value':[], 'latest_time':[]})

    # 失败重试
    retry_count = 0
    while retry_count < 1:
      
      retry_count += 1
      try:
        # get positions and convert to dataframe
        position = self.trade_client.get_positions(account=self.client_config.account)
        if len(position) > 0:
          result = {'symbol': [], 'quantity': [], 'average_cost': [], 'latest_price': []}
          for pos in position:
            result['symbol'].append(pos.contract.symbol)
            result['quantity'].append(pos.quantity)
            result['average_cost'].append(pos.average_cost)
            result['latest_price'].append(pos.market_price)
          result = pd.DataFrame(result)
          
          # add extra columns
          result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
          result['rate_inday'] = 0
          result['market_value'] = round(result['latest_price'] * result['quantity'], 2)
          result['latest_time'] = None

          # get realtime data for stock in position
          if get_briefs:
            status = io_util.get_stock_briefs(symbols=result['symbol'].tolist(), source='eod', api_key=self.eod_api_key)
            if len(status) > 0:
              # merge dataframes
              key_col = 'symbol'
              duplicated_col = [x for x in result.columns if (x in status.columns and x not in [key_col])]
              result.drop(duplicated_col, axis=1, inplace=True)
              result = pd.merge(result, status, how='left', left_on=key_col, right_on=key_col)
              result['rate'] = round((result['latest_price'] - result['average_cost']) / result['average_cost'], 2)
              result['rate_inday'] = round((result['Open'] - result['Close']) / result['Open'], 2)
              result['market_value'] = round(result['latest_price'] * result['quantity'], 2)       
          
          # select columns
          result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'rate_inday', 'market_value', 'latest_time']]

          # break when finish
          break

      except Exception as e:
        self.logger.exception(f'[erro]: can not get position summary: {e}')
        time.sleep(5)
        continue

    self.position = result
  
  # get summary of asset
  def update_asset(self):

    # update asset
    asset = self.trade_client.get_assets(account=self.client_config.account)
    asset = asset[0]
    result = {
      'account': [asset.account],
      'net_value': [asset.summary.net_liquidation],
      'holding_value': [asset.summary.gross_position_value],
      'cash': [asset.summary.cash],
      'available_cash': [asset.summary.available_funds],
      'pnl': [asset.summary.realized_pnl],
      'holding_pnl': [asset.summary.unrealized_pnl]
    }
    self.asset = pd.DataFrame(result)

  # get orders
  def get_orders(self, start_time=None, end_time=None):
    
    start_time = self.trade_time['pre_open_time'].strftime(format="%Y-%m-%d %H:%M:%S") if (start_time is None) else start_time
    end_time = self.trade_time['post_close_time'].strftime(format="%Y-%m-%d %H:%M:%S") if (end_time is None) else end_time

    # result initialization
    orders = []
    result = {
      'code': [], 
      'trd_side': [], 
      'order_type': [], 
      'order_status': [], 
      'qty': [], 
      'price': [], 
      'dealt_qty': [], 
      'dealt_avg_price': [], 
      'order_id': [], 
      'create_time': [], 
      'updated_time': []
    }

    # skip the 90 days limitation
    if util.num_days_between(start_time, end_time) > 90:
      
      tmp_start = start_time
      while tmp_start < end_time:
        tmp_end = util.string_plus_day(tmp_start, 90) 
        self.logger.info(f'[tiger]: getting orders from {tmp_start} to {tmp_end}')
        orders = orders + self.trade_client.get_filled_orders(start_time=tmp_start, end_time=tmp_end)
        tmp_start = tmp_end
        time.sleep(6)
    else:
      orders = self.trade_client.get_filled_orders(start_time=start_time, end_time=end_time)
    
    for ord in orders:
      result['code'].append(ord.contract.symbol)
      result['trd_side'].append(ord.action)
      result['order_type'].append(ord.order_type)
      result['order_status'].append(ord.status.name)
      result['qty'].append(ord.quantity)
      result['price'].append(ord.limit_price)
      result['dealt_qty'].append(ord.filled)
      result['dealt_avg_price'].append(ord.avg_fill_price)
      result['order_id'].append(ord.id)
      result['create_time'].append(ord.order_time)
      result['updated_time'].append(ord.trade_time)
    
    result_df = pd.DataFrame(result)
    for col in ['create_time', 'updated_time']:
      result_df[col] = result_df[col].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x/1000)))
    return result_df
    
  # buy or sell stocks
  def trade(self, symbol, action, quantity, price=None, stop_loss=None, stop_profit=None, print_summary=True):

    trade_summary = ''
    try:

      # construct contract
      contract = stock_contract(symbol=symbol, currency='USD')

      # construct order
      if price is None:
        order_price = 'market'
        order = market_order(account=self.client_config.account, contract=contract, action=action, quantity=quantity)
      else:
        order_price = float(f'{price}')
        order = limit_order(account=self.client_config.account, contract=contract, action=action, quantity=quantity, limit_price=price)

      # construct trade summary
      trade_summary += f'[{action}]: {symbol} X {quantity} ({order_price}) | '

      # attach order legs
      order_legs = []
      if stop_loss is not None:
        stop_loss_order_leg = order_leg('LOSS', stop_loss, time_in_force='GTC') # 附加止损单
        order_legs.append(stop_loss_order_leg)
      if stop_profit is not None:
        stop_profit_order_leg = order_leg('PROFIT', stop_profit, time_in_force='GTC') # 附加止盈单
        order_legs.append(stop_profit_order_leg)
      if len(order_legs)>0:
        order.order_legs = order_legs

      # place buy order if affordable
      if action == 'BUY':
        affordable_quantity = self.get_affordable_quantity(symbol=symbol)
        if quantity <= affordable_quantity:
          self.trade_client.place_order(order)
          trade_summary += f'SUCCEED: {order.id}'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity}/{quantity})'

      # place sell order if holding enough stocks
      elif action == 'SELL':    
        in_position_quantity = self.get_in_position_quantity(symbol)
        if in_position_quantity >= quantity:
          self.trade_client.place_order(order)
          trade_summary += f'SUCCEED: {order.id}'
        else:
          trade_summary += f'FAILED: Not enough stock to sell({in_position_quantity}/{quantity})'

      # other actions
      else:
        trade_summary += f'FAILED: Unknown action({action})'
      
    except Exception as e:
      trade_summary += f'FAILED: {e}'
      
    # print trade summary
    if print_summary: 
      self.logger.info(trade_summary)

    return trade_summary

  # update trade time
  def update_trade_time(self, market=Market, tz='Asia/Shanghai', open_time_adj=0, close_time_adj=0):

    # get local timezone
    tz = pytz.timezone(tz)
    now = datetime.datetime.now()
    
    # get US market trade time
    try:
      # get open_time
      status = self.quote_client.get_market_status(market=Market.US)[0]
      current_status = status.status
      open_time = status.open_time.astimezone(tz).replace(tzinfo=None)
      # open_time = open_time + datetime.timedelta(hours=open_time_adj)

      # if program runs after market open, api will return trade time for next trade day, 
      # trade time for current trade day need to be calculated manually
      if status.status in ['Trading']:
        if now.hour < 12:
          origin_date = now.date() - datetime.timedelta(days=1)
        else:
          origin_date = now.date()
        open_time = open_time.replace(year=origin_date.year, month=origin_date.month, day=origin_date.day) 

      elif status.status in ['Post-Market Trading']:
        if open_time.weekday() == 0:
          open_time = open_time - datetime.timedelta(days=3)
        else:
          open_time = open_time - datetime.timedelta(days=1)
      elif status.status in ['Pre-Market Trading', 'Closed', 'Not Yet Opened', 'Early Closed']:
        pass
      else:
        self.logger.error(f'No method for status [{status.status}]')

      # calculate close time, pre_open_time, post_close_time
      close_time = open_time + datetime.timedelta(hours=6.5) #  + close_time_adj
      pre_open_time = open_time - datetime.timedelta(hours=5.5)
      post_close_time = close_time + datetime.timedelta(hours=4)

    except Exception as e:
      self.logger.error(e)
      current_status = None
      open_time = None
      close_time = None
      pre_open_time = None
      post_close_time = None
      
    # get CN market trade time
    try:
      # get open_time
      cn_status = self.quote_client.get_market_status(market=Market.CN)[0]
      cn_current_status = cn_status.status
      cn_open_time = cn_status.open_time.astimezone(tz).replace(tzinfo=None)
      # cn_open_time = cn_open_time + datetime.timedelta(hours=open_time_adj)

      # if program runs after market open, api will return trade time for next trade day, 
      # trade time for current trade day need to be calculated manually 
      if cn_status.status in ['Trading']:
        origin_date = now.date()
        cn_open_time = cn_open_time.replace(year=origin_date.year, month=origin_date.month, day=origin_date.day) 

      elif (cn_status.status in ['Post-Market Trading']) or (cn_status.status in ['Closed', 'Noon Closed'] and pre_open_time < cn_open_time):
        if cn_open_time.weekday() == 0:
          cn_open_time = cn_open_time - datetime.timedelta(days=3)
        else:
          cn_open_time = cn_open_time - datetime.timedelta(days=1)

      # calculate close time, pre_open_time, post_close_time
      cn_close_time = cn_open_time + datetime.timedelta(hours=5.5) #  + close_time_adj

    except Exception as e:
      self.logger.error(e)
      cn_current_status = None
      cn_open_time = None
      cn_close_time = None

    self.trade_time = {
      'status': current_status, 'tz': tz,
      'pre_open_time': pre_open_time, 'open_time': open_time,
      'close_time': close_time, 'post_close_time': post_close_time,
      
      'a_status': cn_current_status,
      'a_open_time': cn_open_time, 'a_close_time': cn_close_time
    }

  # update market status
  def update_market_status(self, market=Market, return_str=False):
    
    try:
      # get market status
      status = self.quote_client.get_market_status(market=Market.US)[0]
      self.trade_time['status'] = status.status

      cn_status = self.quote_client.get_market_status(market=Market.CN)[0]
      self.trade_time['a_status'] = cn_status.status

      if return_str:
        time_format = '%Y-%m-%d %H:%M'
        pre_open_time = self.trade_time['pre_open_time'].strftime(time_format)
        post_close_time = self.trade_time['post_close_time'].strftime(time_format)
        a_open_time = self.trade_time['a_open_time'].strftime(time_format)
        a_close_time = self.trade_time['a_close_time'].strftime(time_format)

        time_format = '%H:%M'
        open_time = self.trade_time['open_time'].strftime(time_format)
        close_time = self.trade_time['close_time'].strftime(time_format)
        
        us_time_str = f'<Market US: ({pre_open_time}){open_time} -- {close_time}({post_close_time})>'
        cn_time_str = f'<Market CN: {a_open_time} -- {a_close_time}>'
        time_str = us_time_str + '\n' + cn_time_str

        return (us_time_str, cn_time_str)
        
    except Exception as e:
      self.logger.error(e)

  # idle for specified time and check position in certain frequency
  def idle(self, target_time, check_frequency=600):
    """
    Sleep with a fixed frequency, until the target time

    :param target_time: the target time in datetime.datetime format
    :param check_frequency: the fixed sleep_time 
    :returns: none
    :raises: none
    """
    # get current time
    now = datetime.datetime.now()
    while now < target_time:

      # # get position summary
      # self.update_position()
      # self.logger.info(f'[rate]:----------------------------------------------\n{self.position}\n')

      # get current time, calculate difference between current time and target time
      diff_time = round((target_time - now).total_seconds())
      sleep_time = (diff_time + 1) if (diff_time <= check_frequency) else check_frequency
      
      # sleep
      self.logger.info(f'[idle]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: sleep for {sleep_time} seconds')
      time.sleep(sleep_time)

      # update current time
      now = datetime.datetime.now()

    self.logger.info(f'[wake]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: exceed target time({target_time})')

