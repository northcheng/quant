# -*- coding: utf-8 -*-
"""
Utilities used for Futu Open API

:author: Beichen Chen
"""
import logging
from quant import bc_util as util
from quant import bc_data_io as io_util
from futu import *


class Futu:


  # get logger
  defualt_logger = logging.getLogger('bc_futu_logger')


  # init
  def __init__(self, account_type, config, host='127.0.0.1', port=11111, is_encrypt=False, logger_name=None):
        
    # get logger
    self.logger = Futu.defualt_logger if (logger_name is None) else logging.getLogger(logger_name)
        
    # read position record from local files
    self.__position_record = io_util.read_config(file_path=config['config_path'], file_name='tiger_position_record.json')

    # get quote and trade context, assets, positions
    self.open_quote_context(host=host, port=port, is_encrypt=is_encrypt)
    self.open_trade_context(host=host, port=port, is_encrypt=is_encrypt)
    self.positions = self.trade_context.position_list_query()[1]

    # # copy position record for current account
    # self.record = self.__position_record[account_type].copy()

    # # initialize position record for symbols that not in position record
    # init_cash = config['trade']['init_cash'][account_type]
    # pool = config['selected_sec_list'][config['trade']['pool'][account_type]]  
    # for symbol in pool:
    #   if symbol not in self.record.keys():
    #     self.record[symbol] = {'cash': init_cash, 'position': 0}

    # # check position record with current positions
    # if len(self.positions) > 0:
    #   code = [x.split('.')[1] for x in self.positions['code'].tolist()]
    #   qty = self.positions['qty'].tolist()
    #   position_dict = dict(zip(code, qty))
      
    #   for symbol in self.record.keys():
        
    #     # get record position and real position then compare to each other
    #     record_position = self.record[symbol]['position']
    #     current_position = 0 if (symbol not in position_dict.keys()) else position_dict[symbol]
    #     if current_position != record_position:
    #       if current_position > 0:
    #         self.record[symbol] = {'cash': 0, 'position': current_position}
    #       else:
    #         self.record[symbol] = {'cash': init_cash, 'position': 0}
    #       self.logger.error(f'[{account_type[:4]}]: {symbol} position({current_position}) not match with record ({record_position}), reset position record')

    # self.logger.info(f'[init]: Tiger instance created: {logger_name}')


  # initialize quote context
  def open_quote_context(self, host='127.0.0.1', port=11111, is_encrypt=False):
    self.quote_context = OpenQuoteContext(host=host, port=port, is_encrypt=is_encrypt)


  # exit current quote context
  def close_quote_context(self):
    self.quote_context.close()
    self.quote_context = None


  # initialize trade context
  def open_trade_context(self, market='US', host='127.0.0.1', port=11111, is_encrypt=False):
    if market == 'US':
      self.trade_context = OpenUSTradeContext(host=host, port=port, is_encrypt=is_encrypt)
    elif market == 'HK':    
      self.trade_context = OpenHKTradeContext(host=host, port=port, is_encrypt=is_encrypt)
    else:
      print(f'Unknown market {market}')


  # exit current trade context
  def close_trade_context(self):
    self.trade_context.close()
    self.trade_context = None


  # finalize
  def finalize(self):
    self.close_quote_context()
    self.close_trade_context()


  # get position record
  def get_position_record(self):
    return self.__position_record
