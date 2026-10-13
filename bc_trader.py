# -*- coding: utf-8 -*-
"""
Generally Utilities

:author: Beichen Chen
"""
import copy
import math
import logging
import os
import pytz
import time
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from quant import bc_data_io as io_util
from quant import bc_util as util

from futu import (OrderType, OrderStatus, TrdSide, RET_OK)
from futu import (OpenQuoteContext, OpenSecTradeContext, TrdMarket, SecurityFirm, Currency)
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.common.consts import (Language, Market) # 语言, 市场
from tigeropen.common.util.contract_utils import stock_contract # 股票合约
from tigeropen.common.util.order_utils import (market_order_by_amount, market_order, limit_order, order_leg) # 市价单, 限价单, 附加订单

# 账户分组契约 (single source of truth): platform -> {user_info 中的账户键 -> 账户分组}
# 三方约定全部锚定在这张表上, 新增平台/账户/分组时必须同步维护, 任何一侧改键名都会在这里报 KeyError:
#   1. 内层键   = trader/user_info.json 中该平台下的账户键名, Trader 用它取 self.account
#   2. 内层值   = 账户分组, 只有 'real'(实盘) / 'simu'(模拟) 两种:
#       - ta_config.json: trade.init_cash[platform][分组] 与 trade.pool[platform][分组] 按此索引
#       - automatic_trader.py: 命令行 --tiger/--futu 的取值 (simu/real/both) 与此对齐
#   3. 消费方   = bc_trader.Trader.update_position_record / synchronize_position_record,
#                automatic_trader.py 的交易器创建循环
# 注意: tiger 的 standard_account 纸面账户存在于 user_info.json 但故意不在此表中,
# 它不参与 init_cash/pool 分组, 若要启用需先在此登记并补齐 ta_config 对应配置
ACCOUNT_GROUPS = {
  'tiger': {'global_account': 'real', 'simulation_account': 'simu'},
  'futu': {'REAL': 'real', 'SIMULATE': 'simu'}}


# ============================ 下单结果契约 ============================
# trade()/buy_with_amount() 传统上只返回一段"给人看的字符串", 失败时**不抛异常**,
# 于是调用方无法判断成功与否 —— 曾经的后果是 signal_trade 无条件扣减 available_cash,
# 一次下单失败就吃掉一笔预算, 后续买单被系统性少买甚至直接 break.
#
# 这里统一成功标记的判定方式(见 trade_ok):
#   * 状态标记位于**最后一个 '|' 之后**的那一段 —— trade() 的构造顺序是
#     `trade_summary += f'[{action}]: {symbol} x {quantity} (...)| '` 然后 `+= 'SUCCEED: {id}'`
#     或 `+= 'FAILED: ...'`, 所以状态永远是最后一段;
#   * 判定顺序必须是**先看 FAILED** —— 失败详情里可能包含 'SUCCEED'(例如券商返回的报错
#     文本), 先匹配 SUCCEED 会把失败误判成成功;
#   * signal_trade 等调用方必须用 trade_ok(), 不要自己写 substring 判断.
_TRADE_SEP = '|'
_TRADE_OK_MARK = 'SUCCEED:'
_TRADE_FAIL_MARK = 'FAILED:'


def trade_ok(trade_summary) -> bool:
  """
  判断 trade()/buy_with_amount() 的返回串是否代表**下单成功**.

  返回串的形态是 `'[BUY]: AAPL x 10 (market) | SUCCEED: 12345'` —— 状态在**最后一段**
  (分隔符之后). 判定规则:
    1. 取最后一个 '|' 之后的那一段作为状态段(没有分隔符时取整串);
    2. 状态段含 'FAILED:' -> False(**优先判断**, 因为失败详情可能含有 'SUCCEED' 字样);
    3. 否则含 'SUCCEED:' -> True;
    4. 两者都没有 -> False(保守: 无法确认成功就不算成功).

  ⚠ 历史教训(2026-09): 本函数第一版把"状态段"取成了**第一个 '|' 之前**的前缀, 方向反了
  —— 于是它对**所有**返回串都返回 False, `signal_trade` 从此永不扣减预算(会把失败当成
  成功记账的反面: 成功也不记账), 下一单会超买. 是 tests 里那两条断言把它抓出来的.

  :param trade_summary: trade() 的返回值(字符串); None/其它类型一律视为失败
  :returns: True = 成功
  :raises: none
  """
  if not isinstance(trade_summary, str):
    return False
  status = trade_summary.rsplit(_TRADE_SEP, 1)[-1]
  if _TRADE_FAIL_MARK in status:
    return False
  return _TRADE_OK_MARK in status


# 幂等键开关: 打开后每笔订单的 remark 会带上 account+symbol+date+action 键,
# 便于在券商端/流水里去重(同一 bar 重跑不会重复下单的**基础**, 但仍需券商支持或本地台账配合).
# 默认关闭 —— 打开前请先确认所用券商对 remark 的语义, 不要盲目上线.
USE_IDEMPOTENT_REMARK = False

class Trader(object):
  
  # init
  def __init__(self, platform: str, account_type: str, config: dict, logger_name: str = None) -> None:
    
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
    self.record = copy.deepcopy(self.position_record[account_type])
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
  def set_client_config(self, config: dict) -> None:
    pass
  
  # get quote client
  def open_quote_client(self) -> None:
    self.quote_client = None
  
  # close trade client
  def close_trade_client(self) -> None:
    pass
  
  # get trade client
  def open_trade_client(self) -> None:
    self.trade_client = None
  
  # close quote client
  def close_quote_client(self) -> None:
    pass
  
  # finalize trader
  def finalize(self) -> None:
    pass
  
  # get position
  def update_position(self) -> None:
    self.position = None

  # whether the last update_position() actually got a usable answer from the broker.
  # 关键语义: "查询失败" 与 "确实无持仓" 必须能区分 —— 两者都可能是空 DataFrame,
  # 但前者绝不能被当成后者(否则止损单被静默跳过、本地台账被抹平).
  @property
  def position_known(self) -> bool:
    """:returns: True = 最近一次 update_position() 从券商拿到了有效答复(空表也算), False = 查询失败."""
    return bool(getattr(self, '_position_known', False))

  def idempotent_remark(self, symbol: str, action: str) -> str:
    """
    构造幂等键: `{platform}-{account_type}-{symbol}-{action}-{YYYYMMDD}`.

    用途: 传给券商的 remark 字段, 让"同一 bar 内重跑"(cron 重试/手工补跑)产生的重复订单
    在流水/对账时可以识别. 受模块级 `USE_IDEMPOTENT_REMARK` 控制(默认关闭).

    :param symbol: 标的
    :param action: BUY / SELL
    :returns: 幂等键字符串; 开关关闭时返回 None
    :raises: none
    """
    if not USE_IDEMPOTENT_REMARK:
      return None
    day = datetime.datetime.now().strftime('%Y%m%d')
    return f'{self.platform}-{self.account_type}-{symbol}-{action}-{day}'
  
  # get asset
  def update_asset(self) -> None:
    self.asset = None
    
  # get available cash
  def get_available_cash(self) -> float:
    """
    Return the cash actually usable for new orders.

    Why `available_cash` and not `cash`: both adapters put the broker's *usable*
    figure into an `available_cash` column (Futu: `avl_withdrawal_cash`,
    Tiger: `asset.summary.available_funds`, see update_asset), while `cash` is the
    gross cash balance. Sizing orders off the gross balance counts unsettled funds
    as spendable, so orders get rejected (or the account over-commits).

    :returns: usable cash; 0.0 when the broker query fails (and logs an error)
    :raises: none
    """

    available_cash = 0.0

    if self.trade_client is None:
      pass
    else:
      self.update_asset()
      if self.asset is not None and len(self.asset) > 0:
        if 'available_cash' in self.asset.columns:
          available_cash = self.asset.loc[0, 'available_cash']
        elif 'cash' in self.asset.columns:               # 兜底: adapter 未提供可用资金列
          available_cash = self.asset.loc[0, 'cash']
          self.logger.warning('[cash]: asset 无 available_cash 列, 回退用 cash(总现金) 估算')
        else:
          self.logger.error('Not able to get available cash (asset 无可用资金列)')
      else:
        self.logger.error('Not able to get available cash')

    # NaN / 负数一律视为不可用, 避免把 NaN 传进 sizing 计算
    try:
      available_cash = float(available_cash)
    except (TypeError, ValueError):
      self.logger.error(f'[cash]: available cash 非法({available_cash!r}), 视为 0')
      return 0.0
    if not np.isfinite(available_cash) or available_cash < 0:
      self.logger.error(f'[cash]: available cash 非法({available_cash!r}), 视为 0')
      return 0.0

    return available_cash
  
  # get quantity of symbol currently in the position
  def get_in_position_quantity(self, symbol: str, get_briefs: bool = False) -> int:
    """
    Quantity of `symbol` currently held.

    注意语义: 持仓查询失败时 `self.position` 是空表, 直接读会返回 0 —— 调用方若把它
    当成"确实没持仓", 止损/清仓单就会被静默跳过. 因此查询失败时这里记一条 error
    (由调用方自行决定是否下单), 而不是伪装成 0 而不留痕迹.

    :param symbol: target symbol
    :param get_briefs: 是否顺带刷新最新价(传给 update_position)
    :returns: holding quantity; 0 = 无持仓或查询失败
    :raises: none
    """

    # initialize affordable quantity
    quantity = 0

    # get position summary
    self.update_position(get_briefs=get_briefs)
    if not self.position_known:
      self.logger.error(f'[pos]: {symbol} 持仓查询失败, 数量按 0 处理(不可信)')
    position = self.position.copy()
    if len(position) > 0:
      position = position.set_index('symbol')
      if symbol in position.index:
        quantity = position.loc[symbol, 'quantity']

    return quantity

  # check whether it is affordable to buy certain amount of a stock
  def get_affordable_quantity(self, symbol: str, price: float = None, cash: float = None, trading_fee: float = 3) -> int:
    """
    Compute the maximum whole-share quantity affordable with available cash.

    Guard rails (why this matters):
      * `price` must be a *real* price. Callers that use market orders used to pass
        sentinel values (0.1 for BUY / 1e6 for SELL) as a placeholder, which made
        (cash - fee) / price explode to ~10x cash shares and defeated the check
        entirely. A non-positive or absurd price now returns 0 instead.
      * `trading_fee` is a per-order absolute amount, so quantity is floored after
        subtracting it. Available cash must be positive.
      * No lot-size (board lot) handling here yet -- see `min_lot` note below.

    :param symbol: target symbol
    :param price: real reference price; None = fetch the latest price
    :param cash: available cash override; None = query the broker
    :param trading_fee: per-order fee allowance (absolute amount)
    :returns: affordable whole-share quantity (0 when not affordable / bad inputs)
    :raises: none
    """

    # get latest price of stock
    if price is None:
      stock_brief = io_util.get_stock_briefs(symbols=[symbol], source='eod', api_key=self.eod_api_key).set_index('symbol')
      latest_price = stock_brief.loc[symbol, 'latest_price']
    else:
      latest_price = price

    # get available cash
    available_cash = self.get_available_cash() if (cash is None) else cash

    # bad price (NaN / <=0 / placeholder sentinel) -> not affordable, never "infinite"
    try:
      latest_price = float(latest_price)
    except (TypeError, ValueError):
      self.logger.error(f'[afford]: {symbol} invalid price {latest_price!r}, treat as not affordable')
      return 0
    if not np.isfinite(latest_price) or latest_price <= 0:
      self.logger.error(f'[afford]: {symbol} invalid price {latest_price!r}, treat as not affordable')
      return 0
    if available_cash is None or not np.isfinite(available_cash) or available_cash <= trading_fee:
      return 0

    # check if it is affordable
    quantity = math.floor((available_cash-trading_fee)/latest_price)

    return max(int(quantity), 0)

  # update position for an account
  def update_position_record(self, config: dict, init_cash: float = None, init_position: int = None, start_time: str = None, end_time: str = None, commission: float = 3, is_print: bool = True) -> None:
    
    # set default values
    account_group = ACCOUNT_GROUPS[self.platform][self.account_type]
    init_cash = config['trade']['init_cash'][self.platform][account_group] if (init_cash is None) else init_cash
    init_position = 0 if (init_position is None) else init_position

    dropped = []                                 # 未被入账的成交(负现金/负持仓), 结尾统一告警

    try:      
      # get today filled orders
      orders = self.get_orders(start_time, end_time)

      # update position records
      for index, row in orders.iterrows():
        symbol = row['code'] # order.contract.symbol
        action = row['trd_side'] # order.action
        quantity = row['dealt_qty'] # order.quantity - order.remaining
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
          self.logger.info(f'{action} {symbol} {cost} {new_cash} {new_position}')
        
        elif action == 'SELL':
          acquire = avg_fill_price * quantity - commission
          new_cash = record_cash + acquire
          new_position = record_position - quantity
          self.logger.info(f'{action} {symbol} {acquire} {new_cash} {new_position}')

        else:
          new_cash = record_cash
          new_position = record_position
          self.logger.info(f'{action} {symbol} {new_cash} {new_position}')

        # update record
        if new_cash >= 0 and new_position >= 0:
          self.record[symbol]['cash'] = new_cash
          self.record[symbol]['position'] = new_position
          if is_print:
            self.logger.info(f'[{self.account_type[:4]}]: updating position record for {symbol} {record_cash, record_position} -> {new_cash, new_position}')
        else:
          # 静默丢弃成交会让台账与真实持仓单向漂移且永不自愈 —— 必须显式记账并告警,
          # 让调用方能在本轮结束时发现"有成交没入账".
          dropped.append((symbol, action, new_cash, new_position))
          self.logger.error(f'[{self.account_type[:4]}]: skip updating record for {symbol}, negative cash/position {new_cash}/{new_position}')

      # update position_record
      if dropped:
        self.logger.error(f'[{self.account_type[:4]}]: {len(dropped)} 笔成交未入账(负现金/持仓), '
                          f'台账与真实持仓已不一致, 需人工对账: {dropped}')
      self.position_record = io_util.read_config(file_path=config['config_path'], file_name='position.json')[self.platform]
      self.position_record[self.account_type] = self.record.copy()
      self.position_record['updated'][self.account_type] = datetime.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
      io_util.modify_config(self.platform, self.position_record, file_path=config['config_path'], file_name='position.json', is_print=False)
      # io_util.create_config_file(config_dict=self.position_record, file_path=config['config_path'], file_name='tiger_position_record.json')
      
    except Exception as e:
      self.logger.exception(f'[erro]: fail updating position records for {self.account_type}, {e}')
  
  # synchronize position record with real position status
  def synchronize_position_record(self, config: dict) -> None:
    
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

      # 红线: 持仓查询失败时 position 是不可信的空表. 若照常往下走, "券商说没持仓"
      # 与 "我们没查到" 无法区分 —— 结果是每个池内标的都被判为 current_position(0) !=
      # record_position, 台账被改成 {'cash': init_cash, 'position': 0}: 一次瞬时 API
      # 抖动就抹掉本地持仓台账并把现金还原成初始值. 因此这里必须直接返回.
      if not self.position_known:
        self.logger.error(f'[{self.account_type[:4]}]: 持仓查询失败, 跳过台账同步(保留既有记录, 避免误清仓)')
        return
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
  def update_portfolio_record(self, config: dict, position: pd.DataFrame = None, get_briefs: bool = False, is_print: bool = True) -> None:

    # get position summary
    if position is None:
      self.update_position(get_briefs=get_briefs)
      # 持仓未知时不要把"空表"写进 portfolio.json(会覆盖真实持仓快照)
      if not self.position_known:
        self.logger.error('[pf]: 持仓查询失败, 跳过 portfolio 更新(避免用空持仓覆盖真实快照)')
        return
      position = self.position
    # 不要用 inplace=True: position is self.position 时, 就地改 index 会污染调用方的对象
    position = position.set_index('symbol').round(2)

    # get asset summary
    cash = 0
    net_value = 0
    market_value = 0
    self.update_asset()
    asset = self.asset
    if asset is None or len(asset) == 0:
      self.logger.error(f'[{self.account_type[:4]}]: not able to get asset summary, skip updating portfolio record')
      return
    net_value = asset.loc[0, 'net_value']
    market_value = asset.loc[0, 'holding_value']
    cash = asset.loc[0, 'cash']

    # post process
    if market_value == float('inf'):
      market_value = position['market_value'].sum().round(2)

    # load portfolio record, guard missing keys(read failure returns {}) so chained access won't crash
    portfolio_record = io_util.read_config(file_path=config['config_path'], file_name='portfolio.json')

    # if the file exists but is unreadable, skip updating rather than overwriting it with a partial record
    portfolio_file = Path(config['config_path']) / 'portfolio.json'
    if not portfolio_record and os.path.exists(portfolio_file) and os.path.getsize(portfolio_file) > 0:
      self.logger.error(f'[{self.account_type[:4]}]: portfolio.json exists but unreadable, skip updating portfolio record')
      return

    account_record = portfolio_record.setdefault(self.platform, {}).setdefault(self.account_type, {})
    old_net_value = account_record.get('net_value')
    support = (account_record.get('portfolio') or {}).get('support')
    resistant = (account_record.get('portfolio') or {}).get('resistant')

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
      self.logger.info(f'[{self.account_type[:4]}]: net value {old_net_value:.2f} --> {net_value:.2f}')

  # buy stock by amount
  def buy_with_amount(self, symbol: str, amount: float, print_summary: bool = True) -> str:
    print('method not yet implemented')
    return None

  # auto trade according to signals
  def signal_trade(self, signal: pd.DataFrame, money_per_sec: float, order_type: str = 'market', trading_fee: float = 5, pool: list = None, minimum_position: float = None) -> None:    
    
    # set symbol to index
    if len(signal) > 0:

      # filter sec with pool
      if pool is not None:
        filtered_list = [x for x in signal.index if x in pool]
        signal = signal.loc[filtered_list, signal.columns].copy()

    # if signal list is not empty
    if len(signal) > 0:

      if 'latest_price' not in signal.columns:
        signal['latest_price'] = np.nan

        signal_brief = io_util.get_stock_briefs(symbols=signal.index.tolist(), source='eod', api_key=self.eod_api_key).set_index('symbol')
        signal = pd.merge(signal, signal_brief[['latest_price']], how='left', left_index=True, right_index=True)

      # get in-position quantity and latest price for signals
      self.update_position(get_briefs=False)
      # 红线: 持仓未知时"不买"是安全的, "不卖"是危险的 —— 反之, 持仓未知时若继续,
      # signal 里的 quantity 会被填成 0, 所有卖出信号都被判"不在持仓"而跳过.
      # 因此这里直接中止本轮, 让人能看到并处理, 而不是静默只做一半.
      if not self.position_known:
        self.logger.error('[sgnl]: 持仓查询失败, 本轮不执行任何下单(持仓未知时买卖都不可信)')
        return
      position = self.position
      if len(position) == 0:
        position = pd.DataFrame({'symbol':[], 'quantity':[]})
      else:
        position = position[['symbol', 'quantity']].copy()
      position = position.set_index('symbol')
      signal = pd.merge(signal, position[['quantity']], how='left', left_index=True, right_index=True).fillna({'quantity': 0})

      # sell
      # get sell signals
      sell_signal = signal.query('action == "s"')
      skip_sell = []
      if len(sell_signal) > 0:
        # go through sell signals
        for symbol in sell_signal.index:
          # check whether symbol is in position
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity > 0:
            if order_type == 'limit':
              price = signal.loc[symbol, 'latest_price']
              if pd.isna(price) or price <= 0:
                self.logger.error(f'[SELL]: {symbol} skipped (invalid latest price: {price})')
                continue
            else:
              price = None
            trade_summary = self.trade(symbol=symbol, action='SELL', quantity=in_position_quantity, price=price, print_summary=False)
            self.logger.info(trade_summary)
          else:
            skip_sell.append(symbol)
            
        self.logger.info(f'[SELL]: {len(skip_sell)} symbols skipped (not in position)')
      else:
        self.logger.info(f'[SELL]: no signal')

      # buy
      # get available cash, set minimum position
      available_cash = self.get_available_cash()
      if minimum_position is None:
        minimum_position = money_per_sec

      # 每票预算: 全程不变 —— 之前把它写回 money_per_sec(在循环里被改小且不恢复),
      # 会让第 2 只之后的标的预算被前一只的余额永久压小.
      per_symbol_budget = money_per_sec

      # get buy signals which not in positions yet
      buy_signal = signal.query('action == "b"')
      skip_buy = []
      if len(buy_signal) > 0:
        # go through buy signals
        for symbol in buy_signal.index:

          # stop when available cash is below one symbol's budget
          if available_cash <= minimum_position:
            self.logger.info(f'[BUY]: Available cash is too low({available_cash}/{minimum_position}), stop buying')
            break

          # check whether symbol is already in position
          in_position_quantity = signal.loc[symbol, 'quantity']
          if in_position_quantity == 0:

            # skip when latest price is missing or invalid
            latest_price = signal.loc[symbol, 'latest_price']
            if pd.isna(latest_price) or latest_price <= 0:
              self.logger.error(f'[BUY]: {symbol} skipped (invalid latest price: {latest_price})')
              continue

            # 本票可用预算 = min(每票预算, 当前可用资金)
            budget = min(per_symbol_budget, available_cash)

            # calculate quantity to buy
            quantity = math.floor((budget-trading_fee)/latest_price)
            if quantity > 0:
              if order_type == 'limit':
                price = latest_price
              else:
                price = None
              trade_summary = self.trade(symbol=symbol, action='BUY', quantity=quantity, price=price, print_summary=False)
              self.logger.info(trade_summary)

              # 只有**确认成功**才扣减本地可用资金. 之前无条件扣减, 于是一次下单失败
              # 也会吃掉一笔预算 -> 本轮后续买单被系统性少买, 甚至因余额判定直接 break.
              # 判定统一走 trade_ok()(只看第一个分隔符前的前缀段, 避免误判).
              if trade_ok(trade_summary):
                available_cash -= quantity * latest_price
              else:
                self.logger.error(f'[BUY]: {symbol} 下单未成功, 不扣减预算: {trade_summary}')
            else:
              self.logger.info(f'[BUY]: {symbol} not enough money (budget {budget}, price {latest_price})')
              continue
          else:
            skip_buy.append(symbol)                      
            continue
        self.logger.info(f'[BUY]: {len(skip_buy)} symbols skipped (already in position)')
      else:
       self.logger.info(f'[BUY]: no signal')
    else:
      self.logger.info(f'[SKIP]: no signal')
    
  # auto trader according to conditions
  def condition_trade(self, condition_df: pd.DataFrame) -> list:

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
          self.logger.error(f'Error when placing condition order for {symbol}:\n{row}\n{e}')
          continue
      
    return trade_summary
  
  # stop loss or stop profit or clear all position
  @staticmethod
  def _norm_rate(value, name: str, logger) -> float:
    """
    Normalize a profit/loss threshold to a **decimal fraction** (e.g. -0.3 = -30%).

    Why: position['rate'] is round((price-cost)/cost, 2), i.e. a decimal fraction.
    ta_config.json historically mixed units -- stop_loss was -0.8 (decimal) while
    stop_profit was 10, which in decimal terms means "+1000%" and silently disabled
    take-profit. Values with |value| > 1 are therefore treated as percentages and
    divided by 100, with a warning, instead of being compared as-is.

    :param value: raw threshold from config
    :param name: field name, for the warning message
    :param logger: logger
    :returns: decimal-fraction threshold; None stays None (disabled)
    """
    if value is None:
      return None
    try:
      v = float(value)
    except (TypeError, ValueError):
      logger.error(f'[STOP]: {name}={value!r} 非法, 该保护条件已禁用')
      return None
    if abs(v) > 1.0:
      logger.warning(f'[STOP]: {name}={v} 疑似百分比写法, 已按 /100 归一为 {v/100.0:+g}')
      v = v / 100.0
    return v

  def cash_out(self, stop_loss_rate: float = None, stop_profit_rate: float = None, stop_loss_rate_inday: float = None, stop_profit_rate_inday: float = None, clear_all: bool = False, get_briefs: bool = False, print_summary: bool = True) -> None:
    """
    止盈/止损/日内止损统一出口.

    量纲约定: 所有 rate_* 阈值都是**小数比例**(rate 本身 = (price-cost)/cost).
    传入 |值|>1 的会被 `_norm_rate` 当百分比自动归一, 避免"stop_profit=10 等于不止盈"
    这类静默失效. 生效阈值每次都会打印, 便于确认保护是否真的开着.
    """

    # 量纲归一(见 _norm_rate 说明)
    stop_loss_rate = self._norm_rate(stop_loss_rate, 'stop_loss', self.logger)
    stop_profit_rate = self._norm_rate(stop_profit_rate, 'stop_profit', self.logger)
    stop_loss_rate_inday = self._norm_rate(stop_loss_rate_inday, 'stop_loss_inday', self.logger)
    stop_profit_rate_inday = self._norm_rate(stop_profit_rate_inday, 'stop_profit_inday', self.logger)

    # get current position with summary
    self.update_position(get_briefs=get_briefs)
    position = self.position.copy()

    # 持仓未知时绝不能把"空表"当成"没持仓"(同 synchronize_position_record 的理由)
    if not self.position_known:
      self.logger.error('[STOP]: 持仓查询失败, 本轮跳过止盈/止损判断')
      return

    if len(position) > 0:

      # set symbol as index
      position = position.set_index('symbol')
      cash_out_list = []
      stop_loss_list = []
      stop_profit_list = []
      stop_loss_list_inday = []
      stop_profit_list_inday = []

      # if clear all position
      if clear_all:
        cash_out_list = position.index.tolist()
      else:
        stop_loss_list = [] if stop_loss_rate is None else position.query(f'rate < {stop_loss_rate}').index.tolist() 
        stop_profit_list = [] if stop_profit_rate is None else position.query(f'rate > {stop_profit_rate}').index.tolist() 
        stop_loss_list_inday = [] if stop_loss_rate_inday is None else position.query(f'rate_inday < {stop_loss_rate_inday}').index.tolist() 
        stop_profit_list_inday = [] if stop_profit_rate_inday is None else position.query(f'rate_inday > {stop_profit_rate_inday}').index.tolist() 
        cash_out_list = list(dict.fromkeys(stop_loss_list + stop_profit_list + stop_loss_list_inday + stop_profit_list_inday))

        # 生效阈值显式打印: 之前配置写错(量纲)时"保护静默失效", 日志里看不出来.
        self.logger.info(f'[STOP]: 生效阈值 止损={stop_loss_rate} 止盈={stop_profit_rate} '
                         f'日内止损={stop_loss_rate_inday} 日内止盈={stop_profit_rate_inday}')

        # 日内阈值已设但 rate_inday 恒为 0 -> 日内保护实际不生效, 必须点出来.
        if (stop_loss_rate_inday is not None or stop_profit_rate_inday is not None) \
                and 'rate_inday' in position.columns \
                and float(position['rate_inday'].abs().max()) == 0.0:
          self.logger.warning('[STOP]: rate_inday 全为 0(需 get_briefs=True 才会计算), '
                              '日内止盈/止损本轮不会触发')

      # cash out
      if len(cash_out_list) > 0:
        cash_out_position =  position.loc[cash_out_list, ].copy()
        self.logger.info(f'[STOP]: LOSS: {stop_loss_list}, PROFIT: {stop_profit_list}')
        self.logger.info(f'[STOP]: LOSS_INDAY: {stop_loss_list_inday}, PROFIT_INDAY: {stop_profit_list_inday}')

        for index, row in cash_out_position.iterrows():
          self.trade(symbol=index, action='SELL', quantity=row['quantity'], print_summary=print_summary)
    
  

  
class Futu(Trader):
  
  def __init__(self, platform: str, account_type: str, config: dict, logger_name: str = None) -> None:
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)
    
  # get quote client
  def open_quote_client(self) -> None:

    try:
      host = self.user_info['host']
      port = self.user_info['port']
      is_encrypt = self.user_info['is_encrypt']
      self.quote_client = OpenQuoteContext(host=host, port=port, is_encrypt=is_encrypt)
    except Exception as e:
      self.quote_client = None
      self.logger.exception(f'[erro]: can not create quote context:{e}')

  # exit current quote client
  def close_quote_client(self) -> None:
    if self.quote_client is not None:
      self.quote_client.close()
      self.quote_client = None
      
  # get trade context
  def open_trade_client(self, market: str = 'US') -> None:

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
        self.logger.error(f'Unknown market {market}')
        self.trade_client = None
      # 综合账户不再区分市场
      
    except Exception as e:
      self.trade_client = None
      self.logger.exception(f'[erro]: can not create trade context:{e}')

  # exit current trade client
  def close_trade_client(self) -> None:
    if self.trade_client is not None:
      
      # 锁定实盘交易
      if self.account == 'REAL':
        ret, msg = self.trade_client.unlock_trade(is_unlock=False)
        if ret != RET_OK:
          self.logger.exception(f'[erro]: can not lock trade:{ret} - {msg}')
        else:
          self.logger.info(f'[futu]: lock trade')

      self.trade_client.close()
      self.trade_client = None

  # finalize trader
  def finalize(self) -> None:
    self.close_quote_client()
    self.close_trade_client()
    self.logger.info(f'[fin]: instance finalized - {self.logger.name}')
    
  # get summary of position
  def update_position(self, get_briefs: bool = False) -> None:
    result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'rate_inday':[], 'market_value':[], 'latest_time':[]})
    # 只要券商**返回了合法 DataFrame**(哪怕 0 行 = 确实无持仓), 就算查询成功.
    # 全部重试都异常才算失败 -> position_known=False, 上层不得据此做清仓/台账决策.
    answered = False

    # 失败重试
    retry_count = 0
    while retry_count < 3:
      
      retry_count += 1
      try:
        
        # get positions
        _, position = self.trade_client.position_list_query(trd_env=self.account_type)
        if type(position) == pd.DataFrame:
          answered = True
        if type(position) == pd.DataFrame and len(position) > 0:

          # remove duplicated columns
          col_to_drop = ['average_cost']
          for ctd in col_to_drop:
            if ctd in position.columns:
              position.drop('average_cost', axis=1, inplace=True)

          # rename columns, add extra columns
          position.rename(columns={'code':'symbol', 'qty':'quantity', 'cost_price': 'average_cost', 'nominal_price':'latest_price', 'pl_ratio':'rate', 'market_val': 'market_value'}, inplace=True)
          position['symbol'] = position['symbol'].apply(lambda x: x.split('.')[1])
          position['rate'] = round(position['rate'] / 100, 2)
          position['rate_inday'] = 0.0
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
              position['rate_inday'] = round((position['Close'] - position['Open']) / position['average_cost'], 2)
              position['market_value'] = round(position['latest_price'] * position['quantity'], 2)

          # select columns
          result = position[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'rate_inday', 'market_value', 'latest_time']].copy()
          
          # break when finish
          break

      except Exception as e:
        self.logger.exception(f'[erro]: can not get position summary: {e}')
        time.sleep(5)
        continue

    self._position_known = answered
    if not answered:
      self.logger.error('[erro]: 持仓查询失败(全部重试异常) —— 本次 position 不可信, 禁止据此清仓/同步台账')
    self.position = result
    
  # get summary of asset
  def update_asset(self) -> None:

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
        self.asset = None
    except Exception as e:
      self.asset = None
      self.logger.exception(f'[erro]: can not get asset summary: {e}')

  # get orders
  def get_orders(self, start_time: str = None, end_time: str = None) -> pd.DataFrame:

    start_time = datetime.datetime.now().strftime(format="%Y-%m-%d") if (start_time is None) else start_time
    end_time = start_time if (end_time is None) else end_time
    
    ret, orders = self.trade_client.history_order_list_query(trd_env=self.account_type, status_filter_list=[OrderStatus.FILLED_PART, OrderStatus.FILLED_ALL], start=start_time, end=end_time)
    if ret != RET_OK:
      self.logger.error(f'[erro]: get orders - {orders}')
      return pd.DataFrame()
    orders = orders[['code', 'trd_side', 'order_type', 'order_status', 'qty', 'price', 'dealt_qty', 'dealt_avg_price', 'order_id', 'create_time', 'updated_time']].copy()
    orders['code'] = orders['code'].apply(lambda x: x.split('.')[1]) 
    return orders
    
  # buy or sell stocks
  def trade(self, symbol: str, action: str, quantity: int, price: float = None, print_summary: bool = True) -> str:

    trade_summary = ''
    try:

      # set price according to order type
      # Futu 要求市价单也带一个 price: BUY 用极低价 / SELL 用极高价, 否则它会当成限价单.
      # 这两个哨兵值**只能用于下单**, 绝不能再拿去算可买数量 —— 否则
      # (cash-fee)/0.1 会得到 ~10 倍现金的股数, 资金校验被彻底绕过 (见 ref_price).
      ref_price = price                                       # 用于资金校验的真实参考价
      if price is None:
        order_type = OrderType.MARKET
        if action == 'BUY':
          price =  0.1
        elif action == 'SELL':
          price = 1000000
      else:
        order_type = OrderType.NORMAL

      # 市价单 / 未给参考价时, 现查一个真实最新价用于资金校验; 查不到则拒绝下单(宁可不买)
      if ref_price is None:
        try:
          brief = io_util.get_stock_briefs(symbols=[symbol], source='eod',
                                           api_key=self.eod_api_key).set_index('symbol')
          ref_price = float(brief.loc[symbol, 'latest_price'])
        except Exception as e:
          ref_price = None
          self.logger.warning(f'[order]: {symbol} 未取得参考价({e}), 将按未校验处理')

      # construct trade summary
      trade_summary += f'[{action}]: {symbol} x {quantity} ({order_type}:{price}) | '

      # 幂等键(默认关闭, 见 USE_IDEMPOTENT_REMARK); 打开后同一 bar 重跑可用流水去重
      remark = self.idempotent_remark(symbol=symbol, action=action)

      # place buy order if possible
      if action == 'BUY':
        trade_side = TrdSide.BUY
        if ref_price is None or not np.isfinite(ref_price) or ref_price <= 0:
          trade_summary += 'FAILED: no valid reference price for cash check'
          if print_summary:
            self.logger.info(trade_summary)
          return trade_summary
        affordable_quantity = self.get_affordable_quantity(symbol=symbol, price=ref_price)

        if quantity <= affordable_quantity:
          ret_place_order, order_info = self.trade_client.place_order(price=price, qty=quantity, code=f'{self.market}.{symbol}', trd_side=trade_side, order_type=order_type, trd_env=self.account_type, remark=remark)
          if ret_place_order == RET_OK:
            trade_summary += f'SUCCEED: {order_info.loc[0, "order_id"]}'
          else:
            trade_summary += f'FAILED: {order_info}'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity}/{quantity} @ {ref_price})'

      # place sell order if holding enough stocks
      elif action == 'SELL':
        trade_side = TrdSide.SELL
        in_position_quantity = self.get_in_position_quantity(symbol)
        if not self.position_known:
          # 持仓未知时返回的 0 不是"真的没有持仓". 若照常判 in_position >= quantity 为假,
          # 止损/清仓单会**静默跳过** —— 那是资金风险最大的失败模式, 必须显式失败.
          trade_summary += 'FAILED: position unknown (query failed), refuse to sell'
        elif in_position_quantity >= quantity:
          ret_place_order, order_info = self.trade_client.place_order(price=price, qty=quantity, code=f'{self.market}.{symbol}', trd_side=trade_side, order_type=order_type, trd_env=self.account_type, remark=remark)
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
  
  def __init__(self, platform: str, account_type: str, config: dict, logger_name: str = None) -> None:
    super().__init__(platform=platform, account_type=account_type, config=config, logger_name=logger_name)  
    
    # get market status and trade time
    self.update_trade_time()

  # get client config
  def set_client_config(self, config: dict) -> None:
    # initialize client_config
    self.client_config = TigerOpenClientConfig(sandbox_debug=self.user_info['sandbox_debug'])
    self.client_config.private_key = read_private_key(config['trader_path'] / self.user_info['private_key_name'])
    self.client_config.tiger_id = str(self.user_info['tiger_id'])
    self.client_config.language = Language.en_US
    self.client_config.account = self.account 
  
  # get quote client
  def open_quote_client(self) -> None:
    self.quote_client = QuoteClient(self.client_config)
    
  # exit current quote client
  def close_quote_client(self) -> None:
    if self.quote_client is not None:
      self.quote_client = None
  
  # get trade client
  def open_trade_client(self) -> None:
    self.trade_client = TradeClient(self.client_config)
    
  # exit current trade client
  def close_trade_client(self) -> None:
    if self.trade_client is not None:
      self.trade_client = None
  
  # get summary of position
  def update_position(self, get_briefs: bool = False) -> None:
    
    result = pd.DataFrame({'symbol':[], 'quantity':[], 'average_cost':[], 'latest_price':[], 'rate':[], 'rate_inday':[], 'market_value':[], 'latest_time':[]})
    # 同 Futu: get_positions 正常返回(含空列表)才算查询成功; 异常 -> position_known=False
    answered = False

    # 失败重试
    retry_count = 0
    while retry_count < 3:                       # 原为 <1, 注释却写"失败重试", 等于不重试
      
      retry_count += 1
      try:
        # get positions and convert to dataframe
        position = self.trade_client.get_positions(account=self.client_config.account)
        answered = True
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
          result['rate_inday'] = 0.0
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
              result['rate_inday'] = round((result['Close'] - result['Open']) / result['average_cost'], 2)
              result['market_value'] = round(result['latest_price'] * result['quantity'], 2)       
          
          # select columns
          result = result[['symbol', 'quantity', 'average_cost', 'latest_price', 'rate', 'rate_inday', 'market_value', 'latest_time']]

          # break when finish
          break

      except Exception as e:
        self.logger.exception(f'[erro]: can not get position summary: {e}')
        time.sleep(5)
        continue

    self._position_known = answered
    if not answered:
      self.logger.error('[erro]: 持仓查询失败(全部重试异常) —— 本次 position 不可信, 禁止据此清仓/同步台账')
    self.position = result
  
  # get summary of asset
  def update_asset(self) -> None:

    try:
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
    except Exception as e:
      self.asset = None
      self.logger.exception(f'[erro]: can not get asset summary: {e}')

  # get orders
  def get_orders(self, start_time: str = None, end_time: str = None) -> pd.DataFrame:
    
    # fall back to today when trade time is not available (e.g. market status request failed)
    pre_open_time = self.trade_time.get('pre_open_time')
    post_close_time = self.trade_time.get('post_close_time')
    # keep start_time/end_time in '%Y-%m-%d' format, so that util.num_days_between/string_plus_day can parse them
    if start_time is None:
      if pre_open_time is None:
        self.logger.error(f'[erro]: trade time is not ready, fall back to today for order query')
        start_time = datetime.datetime.now().strftime(format='%Y-%m-%d')
      else:
        start_time = pre_open_time.strftime(format='%Y-%m-%d')
    if end_time is None:
      if post_close_time is None:
        end_time = datetime.datetime.now().strftime(format='%Y-%m-%d')
      else:
        end_time = post_close_time.strftime(format='%Y-%m-%d')

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
        orders = orders + self.trade_client.get_filled_orders(start_time=tmp_start, end_time=tmp_end + ' 23:59:59')
        # 下一片必须从 tmp_end **次日**开始, 否则与上一片重叠一天(区间两端都是含端),
        # 那一天的成交会被查两次 -> 重复计入 update_position_record 的 cash/position.
        # 同时 tmp_end >= end_time 时已是最后一片, 直接结束, 避免末尾多查一次重复窗口.
        if tmp_end >= end_time:
          break
        tmp_start = util.string_plus_day(tmp_end, 1)
        time.sleep(6)
    else:
      # append time suffix so the end date itself is fully covered by the query
      orders = self.trade_client.get_filled_orders(start_time=start_time, end_time=end_time + ' 23:59:59')
    
    # 去重: 分片查询/券商重试都可能让同一笔订单出现多次, 而 update_position_record 会
    # 逐行累加 cash/position —— 重复一次就是台账被多记一次(双向偏差, 且不会自愈).
    # 这里按 order_id 去重(保留首现)。
    _seen_order_ids = set()
    _dup = 0
    for ord in orders:
      _oid = getattr(ord, 'id', None)
      if _oid is not None:
        if _oid in _seen_order_ids:
          _dup += 1
          continue
        _seen_order_ids.add(_oid)
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
    if _dup:
      self.logger.warning(f'[tiger]: 订单查询返回 {_dup} 条重复 order_id, 已去重'
                          f'(避免台账重复记账)')
    
    result_df = pd.DataFrame(result)
    for col in ['create_time', 'updated_time']:
      result_df[col] = result_df[col].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x/1000)))
    return result_df
    
  # buy or sell stocks
  def trade(self, symbol: str, action: str, quantity: int, price: float = None, stop_loss: float = None, stop_profit: float = None, print_summary: bool = True) -> str:

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
      trade_summary += f'[{action}]: {symbol} x {quantity} ({order_price}) | '

      # 幂等键(默认关闭, 见 USE_IDEMPOTENT_REMARK)
      remark = self.idempotent_remark(symbol=symbol, action=action)
      if remark is not None:
        # tigeropen 的 order 对象支持 remark 字段(订单备注), 用于流水去重识别
        try:
          order.remark = remark
        except Exception as e:
          self.logger.warning(f'[order]: 设置幂等 remark 失败({e}), 该笔订单不带去重键')

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
        affordable_quantity = self.get_affordable_quantity(symbol=symbol, price=price)
        if quantity <= affordable_quantity:
          # place_order returns None instead of raising when the request fails silently
          order_id = self.trade_client.place_order(order)
          if order_id is not None:
            trade_summary += f'SUCCEED: {order_id}'
          else:
            trade_summary += 'FAILED: no response from server, order status unknown'
        else:
          trade_summary += f'FAILED: Not affordable({affordable_quantity}/{quantity})'

      # place sell order if holding enough stocks
      elif action == 'SELL':    
        in_position_quantity = self.get_in_position_quantity(symbol)
        if not self.position_known:
          # 同 Futu: 持仓未知时返回的 0 不能当成"确实没持仓", 否则止损/清仓单静默跳过
          trade_summary += 'FAILED: position unknown (query failed), refuse to sell'
        elif in_position_quantity >= quantity:
          # place_order returns None instead of raising when the request fails silently
          order_id = self.trade_client.place_order(order)
          if order_id is not None:
            trade_summary += f'SUCCEED: {order_id}'
          else:
            trade_summary += 'FAILED: no response from server, order status unknown'
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

  # buy stock by amount
  def buy_with_amount(self, symbol: str, amount: float, stop_loss: float = None, stop_profit: float = None, print_summary: bool = True) -> str:
    trade_summary = ''
    try:

      # construct contract
      contract = stock_contract(symbol=symbol, currency='USD')

      available_cash = self.get_available_cash()
      if amount > available_cash:
        trade_summary += f'FAILED: Not enough cash({available_cash}/{amount})'
        return trade_summary
      else:
        
        order = market_order_by_amount(account=self.client_config.account, contract=contract, action='BUY', amount=amount)

        # construct trade summary
        trade_summary += f'[BUY]: {symbol} x ${amount} | '

        # 幂等键(默认关闭, 见 USE_IDEMPOTENT_REMARK)
        remark = self.idempotent_remark(symbol=symbol, action='BUY')
        if remark is not None:
          try:
            order.remark = remark
          except Exception as e:
            self.logger.warning(f'[order]: 设置幂等 remark 失败({e}), 该笔订单不带去重键')

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
        order_id = self.trade_client.place_order(order)
        if order_id is not None:
          trade_summary += f'SUCCEED: {order_id}'
        else:
          trade_summary += 'FAILED: no response from server, order status unknown'

    except Exception as e:
      trade_summary += f'FAILED: {e}'
      
    # print trade summary
    if print_summary: 
      self.logger.info(trade_summary)

    return trade_summary

  # update trade time
  def update_trade_time(self, market=Market, tz: str = 'Asia/Shanghai', open_time_adj: int = 0, close_time_adj: int = 0) -> None:

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

      elif status.status in ['Post-Market Trading', 'Post-Mkt']:
        if open_time.weekday() == 0:
          open_time = open_time - datetime.timedelta(days=3)
        else:
          open_time = open_time - datetime.timedelta(days=1)
      elif status.status in ['Pre-Market Trading', 'Pre-Mkt', 'Not Yet Opened']:
        # open_time from api is the next opening time, which is the current day's open
        pass
      elif status.status in ['Closed', 'Early Closed']:
        pass
        # # open_time from api is the next opening time; on non-trading days
        # # fall back to the most recent completed trading day, same as post-market
        # if open_time.weekday() == 0:
        #   open_time = open_time - datetime.timedelta(days=3)
        # else:
        #   open_time = open_time - datetime.timedelta(days=1)
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
      if cn_status.status in ['Trading', 'Noon Closed']:
        origin_date = now.date()
        cn_open_time = cn_open_time.replace(year=origin_date.year, month=origin_date.month, day=origin_date.day) 

      elif (cn_status.status in ['Post-Market Trading']) or (cn_status.status in ['Closed'] and pre_open_time < cn_open_time):
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
  def update_market_status(self, return_str: bool = False) -> tuple or None:
    
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
      time.sleep(12)

  # idle for specified time and check position in certain frequency
  def idle(self, target_time: datetime.datetime, check_frequency: int = 600) -> None:
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

