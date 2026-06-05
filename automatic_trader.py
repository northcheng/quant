#!/usr/bin/env python
# coding: utf-8
# 
# 时间逻辑(以下为美股相关时间为夏令时, 冬令时在原时间上+1h, A股相关时间固定不变):
# 1. 当前时间 ∈ [, US盘前交易开始时间(16:00)]
# 	1.1. 当前时间 ∈ [, A开盘时间(09:30)]
# 		1.1.1. A开盘时间 < US盘前交易时间(16:00): 
# 			睡眠至A中午休盘(12:00)
# 		1.1.2. 否则: 
# 			睡眠至US盘前交易开始后5分钟(16:05)
# 	1.2. 当前时间 ∈ [A开盘时间(09:30), ]
# 		1.2.1. 当前时间 ∈ [, A中午休盘(12:00)]: 
# 			睡眠至A中午休盘(12:00)
# 		1.2.2. 当前时间 ∈ [A中午休盘(12:00), A收盘前半小时(14:30)]: 
# 			睡眠至A收盘前半小时(14:30)
# 		1.2.3. 当前时间 ∈ [A收盘前半小时(14:30), A收盘时间(15:00)]: 
# 			运行脚本(更新/计算/可视化[a], 发送邮件), 睡眠至A收盘后5分钟(15:05)
# 		1.2.4. 当前时间 [A收盘时间(15:00), ]: 
# 			睡眠至US盘前交易开始后5分钟(16:05)
# 		1.2.5. 否则: 报错
# 2. 当前时间 ∈ [US盘前交易开始时间(16:00), US开盘时间(21:30)]: 
# 	运行脚本(更新[us], 更新/计算/可视化[company_300], 更新股票池)
# 3. 当前时间 ∈ [US开盘时间(21:30), US收盘时间(04:00):
# 	3.1. 当前时间 ∈ [, 睡前时间(23:15)]: 
# 		睡眠至睡前时间(23:15)
# 	3.2. 当前时间 ∈ [睡前时间, US收盘前10分钟(03:50)]: 
# 		运行脚本(更新/计算/可视化[us], 发送邮件), 睡眠至US收盘前10分钟(03:50) 
# 	3.3. 当前时间 ∈ [美股收盘前10分钟(03:50), ]: 
# 		运行脚本(更新/计算[universe], 自动交易), 睡眠至US收盘后2小时(6:00)
# 4. 当前时间 ∈ [US收盘时间(04:00), US盘后交易结束时间(08:00)]: 
# 	运行脚本(更新/计算/可视化[us], 发送邮件), 睡眠至US盘后交易结束后5分钟(08:05)
# 5. 当前时间 ∈ [US盘后交易结束时间(08:00), ]: 
# 	更新持仓记录, 同步github文件, 重启循环
#   如果当前为周六, 则以refresh模式更新所有标的数据(us-eod, cn-eod)
# 6. 否则: 报错
# 

# *******************************************************************************************************************
# ************************************************ 初始化 ***********************************************************
# *******************************************************************************************************************
# ================================================ 路径设置  ========================================================
import os
import sys
import platform
from pathlib import Path

# 根据操作系统设置初始路径
p = platform.system()
root_paths = {'home_path': Path.home()}
root_paths['git_path'] = root_paths['home_path']/'git'
sys.path.append(str(root_paths['git_path']))


# ================================================ 模块导入 =========================================================
# 常用模块
import time
import logging
import datetime
import argparse
import pandas as pd
from pathlib import Path

# quant 模块
from git import Repo
from quant import bc_util as util
from quant import bc_data_io as io_util
from quant import bc_trader as trader_util
from quant import bc_technical_analysis as ta_util


# ================================================ 配置读取 =========================================================
if 'config_initialization' > '':
  # 读取配置文件
  config = ta_util.load_config(root_paths)


# ================================================ 日志设置 =========================================================
if 'logger_initialization' > '':
  
  # logger 创建
  logger = logging.getLogger(__name__)
  logger_level = logging.INFO
  logger.setLevel(level = logger_level)

  # logger 文件设置
  logger_file_format = logging.Formatter('[%(asctime)s]- [%(levelname)s] - [%(name)s] - %(message)s ')
  handler = logging.FileHandler(config["log_path"]/f'automatic_trade_log_{datetime.datetime.now().date()}.txt')
  handler.setLevel(logger_level)
  handler.setFormatter(logger_file_format)
  logger.addHandler(handler)

  # logger 控制台设置
  logger_console_format = logging.Formatter('[%(asctime)s] - %(message)s ', '%Y-%m-%d %H:%M:%S')
  console = logging.StreamHandler()
  console.setLevel(logger_level)
  console.setFormatter(logger_console_format)
  logger.addHandler(console)


# ================================================ 参数设置 =========================================================
if 'parser_configuration' > '':
  
  # parser 参数配置
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, help='datasource for running TA scripts', default='eod')
  parser.add_argument('--tiger', type=str, help='which tiger account to use: <none>/<simu>/<real>/<both>', default='both')
  parser.add_argument('--futu', type=str, help='which futu account to use: <none>/<simu>/<real>/<both>', default='both')
  # parser.add_argument('--open_time_adj', type=int, help='adjustment for market open time(+)', default=0)
  # parser.add_argument('--close_time_adj', type=int, help='adjustment for market close time(+)', default=0)
  
  # 获取命令行参数
  terminal_args = parser.parse_args()
  trader_info = {}
  if terminal_args.tiger != 'none':
    trader_info['tiger'] = [terminal_args.tiger] if terminal_args.tiger != 'both' else ['simu', 'real']
  else:
    logger.error('[erro]: at least 1 tiger traders must be specified to be the main trader')
    exit(0)
  if terminal_args.futu != 'none':
    trader_info['futu'] = [terminal_args.futu] if terminal_args.futu != 'both' else ['simu', 'real']
  trader_platform = list(trader_info.keys())
  source = terminal_args.source
  # open_time_adj = terminal_args.open_time_adj
  # close_time_adj = terminal_args.close_time_adj


# ================================================ 交易器创建 =======================================================
if 'trader_initialization' > '':
  
  # 创建交易器: real(真实账户), simu(模拟账户)
  traders = {}
  main_trader = None
  for plfm in trader_info.keys():
    for acnt_grp in trader_info[plfm]:  
      for acnt in trader_util.ACCOUNT_GROUPS[plfm]:
        if trader_util.ACCOUNT_GROUPS[plfm][acnt] == acnt_grp:
          # print(f'eval(trader_util.{plfm.capitalize()}("{plfm}", "{acnt}", config=config, logger_name="{__name__}.{acnt}"))')
          traders[f'{plfm}_{acnt_grp}'] = eval(f'trader_util.{plfm.capitalize()}("{plfm}", "{acnt}", config=config, logger_name="{__name__}.{acnt}")')
          if main_trader is None and plfm == 'tiger':
            main_trader = traders[f'{plfm}_{acnt_grp}']

  # 初始化主交易器, 交易时间
  us_trade_time_str, a_trade_time_str = main_trader.update_market_status(return_str=True)

# 打印初始化完成信息
logger.info('')
logger.info(f'[init]: traders: {list(traders.keys())}')
logger.info(f'[time]: {us_trade_time_str}')
logger.info(f'[time]: {a_trade_time_str}')
# *******************************************************************************************************************
# *******************************************************************************************************************
# *******************************************************************************************************************


# *******************************************************************************************************************
# ************************************************ 循环主体 *********************************************************
# *******************************************************************************************************************
# 初始化计数器(counter), 目标时间(target_time), 目标时间描述(target_dscr), 检查频率(check_frequency)
counter = 0
target_time = None
target_dscr = None
check_frequency = 3600 # = 30min

# 为了打印美观
total_len = 67
level_1_symbol = '*'

# 循环开始:
# 主交易器(main_trader)用于控制整个循环的睡眠/唤醒
while main_trader is not None: 
  try:
    # 如果目标时间(target_time)已设置, 打印相关描述, 睡眠至目标时间
    # 达到目标时间: 检查持仓盈亏情况, 止盈止损(可选), 更新当前时间与市场状态
    if target_time is not None:
      logger.info('')
      logger.info(total_len * level_1_symbol)
      logger.info(f'[{counter}]: US Market - {main_trader.trade_time["status"]}, CN Market - {main_trader.trade_time["a_status"]}')
      logger.info(f'[{counter}]: Sleep until <{target_dscr}> ({target_time})')
      logger.info(total_len * level_1_symbol)
      
      # 获取当前时间
      now = datetime.datetime.now()

      # 睡眠至目标时间 # main_trader.idle(target_time=target_time, check_frequency=check_frequency)
      while now < target_time:

        try:
          # 遍历所有交易器，获取当前持仓标的价格，并对比止盈/止损点位
          for t in traders.keys():
            
            # 获取平台和账户类型
            plfm, acnt = t.split('_')

            # 老虎当前只检查真实账户
            if plfm == 'tiger' and acnt == 'simu':
              continue
            
            # 止盈止损阈值设定
            stop_loss_rate = config['trade']['stop_loss'][plfm][acnt]
            stop_profit_rate = config['trade']['stop_profit'][plfm][acnt]
            stop_loss_inday_rate = config['trade']['stop_loss_inday'][plfm][acnt]
            stop_profit_inday_rate = config['trade']['stop_profit_inday'][plfm][acnt]

            # 获取交易器
            tmp_trader = traders[t]

            # 若为futu实盘先解锁交易(每次时间触发后一次解锁, 就不用在方法中每次解锁)
            if plfm == 'futu' and acnt == 'real':
              ret, msg = tmp_trader.trade_client.unlock_trade(password_md5=tmp_trader.user_info['unlock_pwd'], is_unlock=True)
              if ret != trader_util.RET_OK:
                logger.exception(f'[erro]: can not unlock trade for {t}: {ret} - {msg}')
              else:
                logger.info(f'[futu]: unlock trade')

            tmp_trader.update_position(get_briefs=True)
            position = tmp_trader.position
            pos_info = 'EMPTY' if len(position)==0 else f'\n\n------------------- {t} -------------------\n{position}\n\n' 
            logger.info(f'[position]:{pos_info}')

            # 止盈(stop_loss_rate) 止损(stop_loss_rate), 上面更新过仓位信息(update_position), 此处不更新
            tmp_trader.cash_out(stop_loss_rate=stop_loss_rate, stop_profit_rate=stop_profit_rate, stop_loss_rate_inday=stop_loss_inday_rate, stop_profit_rate_inday=stop_profit_inday_rate, get_briefs=False)
            
            # 更新持仓记录, 上面更新过仓位信息(update_position), 此处不更新
            tmp_trader.update_portfolio_record(config=config, get_briefs=False) 
            
        except Exception as e:
          logger.error(f'[erro]: position check failed ({e})')

        # 计算当前时间与目标时间之间的时间差（需要睡眠的时间）
        diff_time = round((target_time - now).total_seconds())
        sleep_time = (diff_time + 1) if (diff_time <= check_frequency) else check_frequency
        
        # 睡眠并更新当前时间
        logger.info(f'[idle]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: sleep for {sleep_time} seconds')
        time.sleep(sleep_time)
        now = datetime.datetime.now()

      logger.info(f'[wake]: {now.strftime(format="%Y-%m-%d %H:%M:%S")}: exceed target time({target_time})')
      
      # --------------------------------------------------- 仓位检查 -----------------------------------------------
      # 遍历所有交易器
      for t in traders.keys():
        
        # 获取平台和账户类型
        plfm, acnt = t.split('_')

        # 老虎当前只检查真实账户
        if plfm == 'tiger' and acnt == 'simu':
          continue
        
        # 止盈止损阈值设定
        stop_loss_rate = config['trade']['stop_loss'][plfm][acnt]
        stop_profit_rate = config['trade']['stop_profit'][plfm][acnt]
        stop_loss_inday_rate = config['trade']['stop_loss_inday'][plfm][acnt]
        stop_profit_inday_rate = config['trade']['stop_profit_inday'][plfm][acnt]

        # 获取交易器
        tmp_trader = traders[t]

        # 若为futu实盘先解锁交易(每次时间触发后一次解锁, 就不用在方法中每次解锁)
        if plfm == 'futu' and acnt == 'real':
          ret, msg = tmp_trader.trade_client.unlock_trade(password_md5=tmp_trader.user_info['unlock_pwd'], is_unlock=True)
          if ret != trader_util.RET_OK:
            logger.exception(f'[erro]: can not unlock trade for {t}: {ret} - {msg}')
          else:
            logger.info(f'[futu]: unlock trade')

        tmp_trader.update_position()
        position = tmp_trader.position
        pos_info = 'EMPTY' if len(position)==0 else f'\n\n------------------- {t} -------------------\n{position}\n\n' 
        logger.info(f'[position]:{pos_info}')

        # 止盈(stop_loss_rate) 止损(stop_loss_rate), 上面更新过仓位信息(update_position), 此处不更新
        tmp_trader.cash_out(stop_loss_rate=stop_loss_rate, stop_profit_rate=stop_profit_rate, stop_loss_rate_inday=stop_loss_inday_rate, stop_profit_rate_inday=stop_profit_inday_rate, get_briefs=False)
        
        # 更新持仓记录, 上面更新过仓位信息(update_position), 此处不更新
        tmp_trader.update_portfolio_record(config=config, get_briefs=False) 

    # 更新当前时间与市场状态
    counter += 1
    now = datetime.datetime.now()
    main_trader.update_market_status()

    # ================================================ 未开盘 =======================================================
    # 如果当前时间 < 美股盘前交易开始时间(16:05[冬+1])
    if now < main_trader.trade_time['pre_open_time']:

      # 计算A股接近收盘时间(14:15), 中午休盘时间(12:00), 中午休盘结束时间(13:00)
      before_a_close = main_trader.trade_time['a_close_time'] - datetime.timedelta(minutes=45)
      cn_noon_break_start = main_trader.trade_time['a_open_time'] + datetime.timedelta(minutes=120)
      cn_noon_break_end = main_trader.trade_time['a_open_time'] + datetime.timedelta(minutes=210)

      # 如果当前时间 < A股开盘时间(9:30)
      # 发送邮件(上一日持仓情况记录)
      # # 如果当日A股开盘时间 < 美股盘前交易时间, 睡眠至中午休盘时间(12:00); 否则睡眠至美股盘前交易开始后五分钟
      if (now < main_trader.trade_time['a_open_time']):

        # ---------------------------------------------- 发送邮件 ---------------------------------------------------
        ret = io_util.send_result_by_email(
          config=config, 
          to_addr='northcheng@notme.com', from_addr='northcheng@qq.com', smtp_server='smtp.qq.com',  
          platform=trader_platform, signal_file_date=None, log_file_date=None
        )
        logger.info(f'[mail]: sending email ...{ret}')

        # ---------------------------------------------- 睡眠至目标时间 ---------------------------------------------
        # 如果A股开盘, 睡眠至中午休盘时间(12:00), 否则睡眠至美股盘前交易开始后五分钟
        if main_trader.trade_time["a_open_time"] > main_trader.trade_time["pre_open_time"]:
          target_dscr = 'US PREOPEN +5'
          target_time = main_trader.trade_time['pre_open_time'] + datetime.timedelta(minutes=5)
          check_frequency = 3600
          continue
        else:
          target_dscr = 'CN NOON BREAK START'
          target_time = cn_noon_break_start
          check_frequency = 3600 # = 60min
          continue

      # 如果当前时间在A股交易时段内(后半段) [12:00 ~ 15:00): 运行脚本(更新realtime数据, 计算, 可视化), 发送邮件
      # 未靠近收盘[12:00 ~ 14:15): 运行脚本(更新realtime数据, 计算, 可视化, 将[hs300]的potential更新为股票池[a]), 睡眠至收盘前(14:30)
      # 接近收盘[14:15 ~ 15:00): 运行脚本(更新realtime数据, 计算, 可视化), 发送邮件
      # 睡眠至收盘后5分钟(15:05)
      elif now >= main_trader.trade_time['a_open_time'] and now < main_trader.trade_time['a_close_time']:

        # 开盘后 - 中午休盘开始前 [09:30 ~ 11:30)
        if now < cn_noon_break_start:
          
          # 目标时间: A股中午休盘开始(12:00)
          target_dscr = 'CN NOON BREAK START'
          target_time = cn_noon_break_start
          check_frequency = 3600
          continue

        # 未靠近收盘 [11:30 ~ 14:15)
        elif now >= cn_noon_break_start and now < before_a_close:
          # ---------------------------------------------- 运行脚本 ---------------------------------------------------
          # 运行脚本(technical_analyst)更新历史与实盘数据(eod), 计算信号, 可视化, 运行时间限制: 120分钟, 失败重试次数: 1
          cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'),  '--update_mode', 'both', '--source', 'eod', '--pool', 'hs300']
          logger.info(f'[exec]: {" ".join(cmd)}')

          start_run = datetime.datetime.now()
          return_code = util.run_script(cmd, retry=1, timeout=7200)
          cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
          return_status = 'OK' if return_code==0 else 'Error'
          logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

          # ---------------------------------------------- 睡眠至目标时间 ---------------------------------------------
          # 目标时间: A股收盘前半小时(14:15)
          target_dscr = 'CN CLOSE - 30'
          target_time = before_a_close
          check_frequency = 300
          continue

        # 靠近收盘 [14:15 ~ 15:00)
        elif now >= before_a_close:
          # ---------------------------------------------- 运行脚本 ---------------------------------------------------
          # 运行脚本(technical_analyst)更新实时数据(realtime), 计算信号, 可视化, 运行时间限制: 25分钟, 失败重试次数: 5
          cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--update_mode', 'both', '--source', 'eod', '--pool', 'a', '--required_date', f'{util.time_2_string(main_trader.trade_time["a_open_time"].date())}']
          logger.info(f'[exec]: {" ".join(cmd)}')

          start_run = datetime.datetime.now()
          return_code = util.run_script(cmd, retry=5, timeout=1500)
          cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
          return_status = 'OK' if return_code==0 else 'Error'
          logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')
          
          # ---------------------------------------------- 发送邮件 ---------------------------------------------------
          signal_file_date = main_trader.trade_time['a_open_time'].date().strftime(format='%Y-%m-%d')
          ret = io_util.send_result_by_email(
            config=config, 
            to_addr='northcheng@notme.com', from_addr='northcheng@qq.com', smtp_server='smtp.qq.com', 
            platform=trader_platform, signal_file_date=signal_file_date, log_file_date=None, pool='a'
          )
          logger.info(f'[mail]: sending email ...{ret}')

          # ---------------------------------------------- 睡眠至目标时间 -------------------------------------------
          # 目标时间: A股收盘后5分钟(15:05)
          target_dscr = 'CN CLOSE +5'
          target_time = main_trader.trade_time['a_close_time'] + datetime.timedelta(minutes=5)
          check_frequency = 300
          continue

      # 如果A股已收盘 [15:00 ~ 16:00[冬+1])
      # 如果当日美股开盘, 则睡眠至美股盘前交易开始后5分钟(16:05[冬+1]), 否则睡眠至明日A股开盘
      elif (now >= main_trader.trade_time['a_close_time']):

        # 次日凌晨
        tomorrow_morning = main_trader.trade_time['a_close_time'] + datetime.timedelta(minutes=545)

        # 如果当日美股开盘
        if main_trader.trade_time["pre_open_time"] < tomorrow_morning:
          target_dscr = 'US PREOPEN +5'
          target_time = main_trader.trade_time['pre_open_time'] + datetime.timedelta(minutes=5)
          check_frequency = 3600
          continue

        # 如果当日美股不开盘
        else:
          target_dscr = 'TOMORROW MORNING'
          target_time = tomorrow_morning
          check_frequency = 3600

      # 其他情况
      else:
        logger.error(f'condition that not defined: {now}')

    # ================================================ 盘前交易 =====================================================
    # 如果当前时间在美股盘前交易时段内[16:00[冬+1] ~ 21:30[冬+1])
    # 运行脚本(更新eod数据)
    # 睡眠至美股开盘后5分钟
    elif now >= main_trader.trade_time['pre_open_time'] and now < main_trader.trade_time['open_time']:

      # ---------------------------------------------- 运行脚本 ---------------------------------------------------
      # 运行脚本(technical_analyst)更新(company_300)历史数据(eod), 跳过(计算, 可视化, 后处理), 失败重试次数: 5
      if config['selected_sec_list']['company_star_update_time'] < datetime.datetime.today().strftime(format='%Y-%m-%d'):
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', f'{source}', '--update_mode', 'eod', '--pool', 'company_300', '--skip_calculation', '--skip_visualization', '--skip_postprocess']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

      # 运行脚本(technical_analyst)更新(company_300)历史数据(eod), 计算, 可视化, 更新(company_star)股票池, 运行时间限制: 120分钟, 失败重试次数: 5
      if config['selected_sec_list']['company_star_update_time'] < datetime.datetime.today().strftime(format='%Y-%m-%d'):
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', f'{source}', '--update_mode', 'eod', '--pool', 'company_300']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

      # ---------------------------------------------- 运行脚本 -----------------------------------------------------
      # 运行脚本(technical_analyst)更新(us)历史数据(eod), 运行时间限制: 30分钟, 失败重试次数: 5
      cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', f'{source}', '--update_mode', 'eod','--pool', 'us', '--skip_calculation', '--skip_visualization', '--skip_postprocess']
      logger.info(f'[exec]: {" ".join(cmd)}')

      start_run = datetime.datetime.now()
      return_code = util.run_script(cmd, retry=5, timeout=1800)
      cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
      return_status = 'OK' if return_code==0 else 'Error'
      logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

      # ---------------------------------------------- 运行脚本 ---------------------------------------------------
      # 运行脚本(technical_analyst)更新(hs300)历史数据(eod), 计算, 可视化, 更新(a_company)股票池, 运行时间限制: 30分钟, 失败重试次数: 5
      if config['selected_sec_list']['a_company_update_time'] < datetime.datetime.today().strftime(format='%Y-%m-%d'):
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', 'eod', '--update_mode', 'eod', '--pool', 'hs300']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

      # ---------------------------------------------- 睡眠至目标时间 -----------------------------------------------
      # 目标时间: 美股开盘后5分钟(21:35[冬+1])
      target_dscr = 'US OPEN +5'
      target_time = main_trader.trade_time['open_time'] + datetime.timedelta(minutes=5)
      check_frequency = 3600
      continue
    
    # ================================================ 盘中交易 =====================================================
    # 如果当前时间在美股盘中交易时段内[21:30[冬+1] ~ 04:00[冬+1])
    # 午夜之前 [21:30[冬+1] ~ 23:15[冬+1]): 睡眠至午夜
    # 午夜后, 未接近收盘[23:15[冬+1] ~ 03:50[冬+1]): 运行脚本(更新realtime数据, 计算, 可视化), 发送邮件
    # 接近收盘[03:45[冬+1] ~ 04:00[冬+1]): 运行脚本(更新realtime数据, 计算), 自动交易
    # 睡眠至收盘后2小时(06:00[冬+1])
    elif now >= main_trader.trade_time['open_time'] and now < main_trader.trade_time['close_time']:

      # 计算午夜时间(开盘后1小时45分, 23:15[冬+1])
      h = 2
      m = 15 - main_trader.trade_time['open_time'].minute
      midnight = main_trader.trade_time['open_time'] + datetime.timedelta(hours=h, minutes=m)
      before_close = main_trader.trade_time['close_time'] - datetime.timedelta(minutes=15)

      # 如果当前时间 < 午夜时间 [21:30 ~ 23:15)
      # 睡眠至午夜
      if now < midnight:

        # ---------------------------------------------- 睡眠至目标时间 ---------------------------------------------
        # 目标时间: 开盘后1小时45分, (23:15[冬+1])
        target_dscr = 'MIDNIGHT'
        target_time = midnight
        check_frequency = 300
        continue

      # 如果当前时间在午夜后, 未接近收盘[23:15 ~ 03:45[冬+1])
      # 运行脚本(更新realtime数据, 计算, 可视化), 发送邮件
      # 睡眠至收盘前15分钟(03:45[冬+1])
      elif now >= midnight and now < before_close:

        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新实时数据(realtime), 计算, 可视化, 发送邮件, 运行时间限制: 30分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', f'eod', '--update_mode', 'realtime','--pool', 'us']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=1800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

        # ---------------------------------------------- 发送邮件 ---------------------------------------------------
        log_file_date = None
        signal_file_date = datetime.datetime.today().strftime(format='%Y-%m-%d') #main_trader.trade_time['open_time'].date().strftime(format='%Y-%m-%d')
        ret = io_util.send_result_by_email(
          config=config, 
          to_addr='northcheng@notme.com', from_addr='northcheng@qq.com', smtp_server='smtp.qq.com', 
          platform=trader_platform, signal_file_date=signal_file_date, log_file_date=log_file_date
        )
        logger.info(f'[mail]: sending email ...{ret}')

        # ---------------------------------------------- 睡眠至目标时间 ---------------------------------------------
        # 目标时间: 美股收盘前15分钟(03:45[冬+1])
        target_dscr = 'US CLOSE -15'
        target_time = before_close
        check_frequency = 300
        continue

      # 如果当前时间接近收盘[03:45[冬+1] ~ 04:00[冬+1])
      # 运行脚本(更新realtime数据, 计算), 自动交易
      # 睡眠至收盘后2小时(06:00[冬+1])
      elif now >= before_close:  
        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新实时数据(realtime), 计算, 运行时间限制: 10分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', f'{source}', '--update_mode', 'realtime','--pool', 'universe', '--required_date', f'{util.time_2_string(main_trader.trade_time["open_time"].date())}', '--skip_visualization']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=600)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

        # ---------------------------------------------- 自动交易 ---------------------------------------------------
        # 从excel文件中读取信号
        signal_file_date = datetime.datetime.now().date()
        signal_file = config["result_path"] / f'{signal_file_date}.xlsx'
        if os.path.exists(signal_file):
          
          signal = pd.read_excel(signal_file, sheet_name='signal', dtype={'代码': str})
          if len(signal) > 0:
            signal = signal.rename(columns={'signal':'action'}).set_index('symbol')
            signal_brief = io_util.get_stock_briefs(symbols=signal.index.tolist(), source=source, api_key=config['api_key'][source]).set_index('symbol')
            signal = pd.merge(signal, signal_brief[['latest_price']], how='left', left_index=True, right_index=True)
            logger.info(f'[signal]:----------------------------------------------------\n')
            logger.info(signal[['action', 'latest_price']])
                  
            # 设置订单类型
            order_type = 'market'
            if datetime.datetime.now() >= main_trader.trade_time['close_time']:
              order_type = 'limit'

            # 交易
            for t in traders.keys():
              # 获取平台和账户类型
              plfm, acnt = t.split('_')

              # 根据信号进行交易
              tmp_trader = traders[t]
              logger.info(f'[trade]:-------------------------------------- {t}')
              tmp_trader.signal_trade(signal=signal, money_per_sec=config['trade']['init_cash'][plfm][acnt], pool=config['selected_sec_list'][config['trade']['pool'][plfm][acnt]], order_type=order_type)
              logger.info(f'[done]:----------------------------------------------------\n')

          else:
            logger.info(f'[skip]: no signal in file ({signal_file})\n')
        else:
          logger.info(f'[erro]: signal file ({signal_file}) not found\n')

        # ---------------------------------------------- 条件交易 ---------------------------------------------------
        # 从json文件中读取信号
        signal = None
        trading_date = str(main_trader.trade_time["open_time"].date())
        condition_file = config['trader_path']/'trade_condition.json'
        if os.path.exists(condition_file):
          signal = pd.read_json(condition_file).T
          signal = signal.query(f'(index != "TEMPLATE") and (date == "{trading_date}")').copy()

          # 获取最新价格信息
          if len(signal) > 0:
            symbols = signal.index.tolist()
            signal_briefs = io_util.get_stock_briefs(symbols=symbols, source='eod', api_key=config['api_key']['eod'], batch_size=15)
            signal_briefs = signal_briefs.set_index('symbol')
            signal = pd.merge(signal, signal_briefs, how='left', left_index=True, right_index=True)
            logger.info(f'[signal]:----------------------------------------------------\n')
            logger.info(signal[['action', 'latest_price']])

            # 交易
            for t in traders.keys():
              # 获取平台和账户类型
              plfm, acnt = t.split('_')

              # 根据条件进行交易
              tmp_trader = traders[t]
              logger.info(f'[trade]:-------------------------------------- {t}')
              tmp_signal = signal.query(f'(plfm == "{plfm}") and (acnt == "{acnt}")').copy()
              tmp_trader.condition_trade(tmp_signal)
              logger.info(f'[done]:----------------------------------------------------\n')
          else:
            logger.info(f'[skip]: no valid symbol in condition file')
        else:
          logger.info(f'[erro]: condition file ({condition_file}) not found\n')

        # ---------------------------------------------- 睡眠至目标时间 ----------------------------------------------
        # 目标时间: 美股收盘后2小时(6:00[冬+1])
        target_dscr = 'US CLOSE +120'
        target_time = main_trader.trade_time["close_time"] + datetime.timedelta(minutes=120)
        check_frequency = 3600
        continue
    
    # ================================================ 盘后交易 =====================================================
    # 如果当前时间在美股盘后交易时段内[04:00[冬+1] ~ 08:00[冬+1])
    # 运行脚本(更新eod数据, 计算, 可视化), 发送邮件
    # 睡眠至盘后交易结束后5分钟(08:05[冬+1])
    elif now >= main_trader.trade_time['close_time'] and now < main_trader.trade_time['post_close_time']:

      # ---------------------------------------------- 运行脚本 ---------------------------------------------------
      # 运行脚本(technical_analyst)更新历史数据(eod), 计算, 可视化, 发送邮件, 运行时间限制: 30分钟, 失败重试次数: 5
      cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', f'{source}', '--update_mode', 'eod', '--pool', 'us', '--required_date', f'{util.time_2_string(main_trader.trade_time["open_time"].date())}']
      logger.info(f'[exec]: {" ".join(cmd)}')
      
      start_run = datetime.datetime.now()
      return_code = util.run_script(cmd, retry=5, timeout=1800)
      cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
      return_status = 'OK' if return_code==0 else 'Error'
      logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

      # ---------------------------------------------- 发送邮件 ---------------------------------------------------
      log_file_date = main_trader.trade_time['open_time'].date().strftime(format='%Y-%m-%d')
      signal_file_date = main_trader.trade_time['close_time'].date().strftime(format='%Y-%m-%d')
      ret = io_util.send_result_by_email(
        config=config, 
        to_addr='northcheng@notme.com', from_addr='northcheng@qq.com', smtp_server='smtp.qq.com', 
        platform=trader_platform, signal_file_date=signal_file_date, log_file_date=log_file_date
      )
      logger.info(f'[mail]: sending email ...{ret}')

      # ---------------------------------------------- 睡眠至目标时间 ---------------------------------------------
      # 目标时间: 美股盘后交易结束后5分钟(08:05[冬+1])
      target_dscr = 'US POSTCLOSE +5'
      target_time = main_trader.trade_time["post_close_time"] + datetime.timedelta(minutes=5)
      check_frequency = 3600
      continue

    # ================================================ 收盘后 =======================================================
    # 如果当前时间 > 盘后交易结束时间[08:00[冬+1])
    # 更新各平台持仓记录, 同步至github
    # 重启循环
    elif now >= main_trader.trade_time['post_close_time']:

      # ------------------------------------------------- 持仓记录更新 ---------------------------------------------
      # start_time = main_trader.trade_time['pre_open_time'].strftime(format="%Y-%m-%d %H:%M:%S")
      # end_time = main_trader.trade_time['post_close_time'].strftime(format="%Y-%m-%d %H:%M:%S")
      # for t in traders.keys():
      #   tmp_trader = traders[t]
        # tmp_trader.update_position_record(config=config, start_time=start_time, end_time=end_time)
        # tmp_trader.synchronize_position_record(config=config)

      # ---------------------------------------------- 持仓记录更新至github -----------------------------------------
      try:
        # 拉取服务器代码, 添加持仓记录, 提交至服务器
        repo_path = config['config_path']
        r = Repo(repo_path)
        r.git.pull()
        r.index.add([f'selected_sec_list.json', f'portfolio.json'])
        r.index.commit('update portfolio and selected_sec_list')
        origin = r.remote(name='origin')
        origin.push()
        logger.info('[sync]: portfolio and selected_sec_list synchronized')

      except Exception as e:
        logger.info(f'[sync]: portfolio and selected_sec_list synchronization failed: {e}')
        
      # ---------------------------------------------- 启动新的一轮循环 ---------------------------------------------
      logger.info('[done]: position record updated, loop will be restarted ...')

      # 更新新的交易时间
      main_trader.update_trade_time()
      us_trade_time_str, a_trade_time_str = main_trader.update_market_status(return_str=True)
      
      # 更新新的日志文件
      logger.removeHandler(handler)
      new_log_date = main_trader.trade_time['open_time'].date().strftime(format='%Y-%m-%d')
      handler = logging.FileHandler(config["log_path"]/f'automatic_trade_log_{new_log_date}.txt')
      handler.setLevel(logger_level)
      handler.setFormatter(logger_file_format)
      logger.addHandler(handler)
      logger.info(f'[init]: log file reset')
      
      # 重置循环参数
      counter = 0
      target_dscr = None
      target_time = None
      check_frequency = 3600
      logger.info(f'[time]: {us_trade_time_str}')
      logger.info(f'[time]: {a_trade_time_str}')

      # 如果为周六, 则重新下载数据
      current_weekday = datetime.datetime.today().weekday()
      if current_weekday == 5:
        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新(company_300)历史数据(refresh), 计算, 可视化, 更新(company_star)股票池, 运行时间限制: 120分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', 'eod', '--update_mode', 'refresh', '--pool', 'company_300']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新(etf_3x)历史数据(refresh), 计算, 可视化, 运行时间限制: 120分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', 'eod', '--update_mode', 'refresh', '--pool', 'etf_3x']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新(global)历史数据(refresh), 计算, 可视化, 运行时间限制: 120分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', 'eod', '--update_mode', 'refresh', '--pool', 'global']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新(a_etf)历史数据(refresh), 计算, 可视化, 运行时间限制: 120分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', 'eod', '--update_mode', 'refresh', '--pool', 'a_etf']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')
        
        # ---------------------------------------------- 运行脚本 ---------------------------------------------------
        # 运行脚本(technical_analyst)更新(hs300)历史数据(refresh), 计算, 可视化, 更新(a_company)股票池, 运行时间限制: 120分钟, 失败重试次数: 5
        cmd = [sys.executable, '-u', str(config["home_path"]/'technical_analyst.py'), '--source', 'eod', '--update_mode', 'refresh', '--pool', 'hs300']
        logger.info(f'[exec]: {" ".join(cmd)}')
        
        start_run = datetime.datetime.now()
        return_code = util.run_script(cmd, retry=5, timeout=10800)
        cost = round((datetime.datetime.now()-start_run).total_seconds() /60, 2)
        return_status = 'OK' if return_code==0 else 'Error'
        logger.info(f'[done]: {return_status}({return_code}) - cost {cost} mins')

      continue

    # ================================================ 其他情况 =====================================================
    # 记录到日志文件中
    else:
      logger.error(f'[erro]: unexpected situation ({now.strftime(format="%Y-%m-%d %H:%M:%S")}, market status - {main_trader.trade_time["status"]})')
      break

  except Exception as e:
    logger.exception(f'[erro]: Exception in the loop - {e}')
    continue

logger.info('[stop]: program terminated\n')

# *******************************************************************************************************************
# *******************************************************************************************************************
# *******************************************************************************************************************