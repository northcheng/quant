# -*- coding: utf-8 -*-
"""
Utilities used for Futu Open API

:author: Beichen Chen
"""

from futu import *

# get logger
defualt_logger = logging.getLogger('bc_futu_logger')

class Futu:

    quote_context = None
    trade_context = None

    # init
    def __init__(self, host='127.0.0.1', port=11111, is_encrypt=False):
        self.open_quote_context(host=host, port=port, is_encrypt=is_encrypt)
        self.open_trade_context(host=host, port=port, is_encrypt=is_encrypt)

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