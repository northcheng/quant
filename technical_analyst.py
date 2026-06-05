#!/usr/bin/env python
# coding: utf-8


# *******************************************************************************************************************
# ************************************************ 初始化开始 *******************************************************
# *******************************************************************************************************************
# ================================================ 根路径设置  ======================================================
import os
import sys
import platform
from pathlib import Path

# set root path according to platform
p = platform.system()
root_paths = {'home_path': Path.home()}
root_paths['git_path'] = root_paths['home_path'] / 'git'
sys.path.append(str(root_paths['git_path']))


# ================================================ 模块导入 =========================================================
# common modules
import gc
import time
import logging
import datetime
import argparse
import pandas as pd
from tqdm import tqdm

# quant modules
from quant import bc_util as util
from quant import bc_data_io as io_util
from quant import bc_technical_analysis as ta_util

# matplotlib initialization
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# ================================================ 基础配置 ========================================================
if 'basic_initialization' > '':

  # 初始化日期, 时间
  today = util.time_2_string(datetime.datetime.today().date())
  start_time = datetime.datetime.now()

  # 获取参数配置
  config = ta_util.load_config(root_paths)

  # 检查运行需要的文件
  folder_check_pass = True
  inexist_folder = []
  for p in ['git_path', 'config_path', 'quant_path', 'log_path', 'api_path', 'trader_path', 'data_path', 'result_path']:
    if not os.path.exists(config[p]):
      folder_check_pass = False
      inexist_folder.append(p)
  if not folder_check_pass:
    print(f'folder {inexist_folder} not exist')
    exit()
    

  # logger
  logger = logging.getLogger(__name__)
  logger.setLevel(level = logging.INFO)

  # 输出到文件
  handler = logging.FileHandler(config["log_path"] / f'ta_log_{start_time.date()}.txt')
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s]- [%(levelname)s] - [%(name)s] - %(message)s ')
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # 输出到控制台
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s] - %(message)s ', '%Y-%m-%d %H:%M:%S')
  console.setFormatter(formatter)
  logger.addHandler(console)


# ================================================ 命令行参数设置 ==================================================
if 'parser_configuration' > '':
  
  # parser 设置
  parser = argparse.ArgumentParser()
  parser.add_argument('--pool', type=str, help='symbol pool: defined in selected_sec_list.json', default='us')
  parser.add_argument('--source', type=str, help='data source: <yfinance>/<yahoo>/<eod>/<marketstack>', default='eod')
  parser.add_argument('--interval', type=str, help='interval of stock data for calculation: <day>/<week>/<month>', default='day')
  parser.add_argument('--update_mode', type=str, help='how to update stock data: <realtime>/<eod>/<both>/<refresh>', default='eod')
  parser.add_argument('--start_date', type=str, help='start date for visualization', default='')
  parser.add_argument('--required_date', type=str, help='required date of the latest stock data', default='')
  parser.add_argument('--query_benchmark', action='store_true', help='whether to query benchmark symbol to get the most current date')
  parser.add_argument('--skip_dataupdate', action='store_true', help='use locally stored OHLCV data instead of update from <source>')
  parser.add_argument('--skip_calculation', action='store_true', help='use locally stored TA data instead of recalculation')
  parser.add_argument('--skip_visualization', action='store_true', help='use locally stored images instead of revisualization')
  parser.add_argument('--skip_postprocess', action='store_true', help='skip postprocess including signal and image extraction')
  parser.add_argument('--recalculate_signal', action='store_true', help='recalculate signal instead of recalculate all')
  parser.add_argument('--send_email', action='store_true', help='send calculation results by email')
  parser.add_argument('--ml_signal_source', type=str, help='ML signal source: <auto>/<ml>/<total>/<off>. defaults to ta_config.json calculation.ml.signal_source', default='')
  parser.add_argument('--ml_horizon', type=int, help='ML forecast horizon (trading days). defaults to ta_config.json calculation.ml.horizon', default=0)
  parser.add_argument('--ml_pool', type=str, help='override the ML model pool tag (cross-pool model reuse, e.g. compute features for a_etf but use hs300 model). defaults to --pool', default='')

  # parse 参数获取
  args = parser.parse_args()
  pool = args.pool
  source = args.source
  update_mode = args.update_mode
  interval = args.interval
  start_date = args.start_date
  start_date = None if start_date == '' else start_date
  required_date = args.required_date
  required_date = None if required_date == '' else util.string_2_time(required_date)
  query_benchmark = args.query_benchmark
  skip_dataupdate = args.skip_dataupdate
  skip_calculation = args.skip_calculation
  skip_visualization = args.skip_visualization
  skip_postprocess = args.skip_postprocess
  recalculate_signal = args.recalculate_signal
  send_email = args.send_email

  # ML 参数设置 (signal_source / horizon): 优先用 CLI 参数, 缺省时读 ta_config.json
  ml_cfg = config.get('calculation', {}).get('ml', {}) if isinstance(config, dict) and 'calculation' in config else {}
  signal_source = args.ml_signal_source if args.ml_signal_source else ml_cfg.get('signal_source', 'auto')
  ml_horizon    = args.ml_horizon if args.ml_horizon > 0 else int(ml_cfg.get('horizon', 5))
  ml_pool       = args.ml_pool if args.ml_pool else pool  # 跨池模型复用, 默认与 pool 一致
  if ml_pool != pool:
    print(f'[ML cross-pool override]: data pool={pool}, model pool={ml_pool}')

  # 计算参数设置
  config['calculation']['save_sec_data'] = True
  config['calculation']['update_sec_data'] = False if skip_dataupdate else True
  config['calculation']['update_ta_data'] = False if skip_calculation else True
  config['calculation']['update_ta_signal'] = False if (skip_calculation and not recalculate_signal) else True
  config['calculation']['save_ta_data'] = (config['calculation']['update_ta_data'] or config['calculation']['update_ta_signal'])

  # 可视化参数设置
  config['visualization']['show_image'] = False
  config['visualization']['create_image'] = False if skip_visualization else True
  config['visualization']['save_image'] = config['visualization']['create_image']
  
  # 后处理参数设置
  config['postprocess']['save_result'] = True
  config['postprocess']['save_pdf'] = True
  config['postprocess']['send_email'] = True if send_email else False
  config['postprocess']['archive_old_xlsx'] = True

  # 流程参数
  DATAUPDATE = config['calculation']['update_sec_data']
  CALCULATION = (config['calculation']['update_ta_data'] or config['calculation']['update_ta_signal'])
  VISUALIZATION = config['visualization']['create_image']
  POSTPROCESS = ((config['postprocess']['save_result'] or config['postprocess']['save_pdf']) and not skip_postprocess)


# ================================================ 股票列表及数据设置 ==============================================
if 'stock_initialization' > '':

  # 获取目标列表
  target_list = {}
  if pool == 'us':
    for l in ['global', 'etf_3x', 'company_star']:
      target_list[l] = config['selected_sec_list'][l]
  elif pool == 'a':
    for l in ['a_etf', 'a_company']: #
      target_list[l] = config['selected_sec_list'][l]
  else:
    target_list = {
      pool: config['selected_sec_list'][pool]}
  target_list_description = f'['
  for target in target_list.keys():
    target_list_description = target_list_description + f'{target}({len(target_list[target])}), '
  target_list_description = target_list_description[:-2] + ']'
  
  # 设置默认数据源
  sources = {
      'us_eod': 'eod', 
      'us_realtime': 'eod', 
      'cn_eod': 'eod', 
      'cn_realtime': 'easyquotation', 
      'hk_eod': 'eod', 
      'hk_realtime': 'easyquotation'
    }
  
  # 根据指定source设置相关数据源
  if source in ['eod']:
    sources = {
      'us_eod': 'eod', 
      'us_realtime': 'eod', 
      'cn_eod': 'eod', 
      'cn_realtime': 'easyquotation', 
      'hk_eod': 'eod', 
      'hk_realtime': 'easyquotation'
    }

  # 读取本地数据: sec_data, ta_data, result
  data = ta_util.load_data(target_list=target_list, config=config, interval=interval, load_derived_data=skip_calculation, load_empty_data=False)


# ================================================ ML 参数辅助 =====================================================
if 'ml_helpers' > '':

  def _infer_market_for_target(target_name: str, sec_list: list) -> str:
    """
    Best-effort market tag ('us' / 'a') for a given target list.

    Rules (in order):
      1. name-based: target starts with 'a_' or equals 'hs300' / 'a_test'  -> 'a'
      2. symbol-based: first symbol whose first char is a digit            -> 'a'
      3. fallback: 'us'
    """
    if target_name.startswith('a_') or target_name in ('hs300', 'a_test'):
      return 'a'
    if sec_list:
      first = str(sec_list[0])
      if first and first[0].isdigit():
        return 'a'
    return 'us'

# 打印信息
logger.info(f'[init]: running on {p}, data from {source}({update_mode} mode)')
logger.info(f'[init]: pool {target_list_description}')
logger.info(f'[init]: load [sec_data({len(data["sec_data"])}), ta_data({len(data["ta_data"])}), result({len(data["result"])}), final_result({len(data["final_result"])})]\n')    


# ================================================ 核心流程 ========================================================
former_half_len = 27
latter_half_len = 38
total_len = 67
level_1_symbol = '='
level_2_symbol = '-'
tqdm_ncols = 91

stage = 'loop_start'
try:
  # 遍历每一个列表(e.g. etf, company)
  for target in target_list.keys():
    
    # 获取列表中的标的
    ti = f'{target}_{interval}'
    logger.info(total_len * level_1_symbol)
    logger.info(former_half_len*level_1_symbol + f' {ti} ' + (latter_half_len-len(ti))*level_1_symbol)

    # 如果列表为空则跳过
    sec_list = target_list[target]
    if len(sec_list) == 0: 
      logger.info(f'[skip]: [{target}] is an empty list')
      continue 
    
    # ============================================== 数据更新 ======================================================
    # 更新 OHLCV 数据, 失败重试次数: 5
    stage = 'dataupdate'
    logger.info('')
    logger.info(former_half_len*level_2_symbol + f' {stage} ' + (latter_half_len-len(stage))*level_2_symbol)
    if DATAUPDATE:
      
      # 创建容器储存下载数据, 失败重试: 5次
      new_data = {}
 
      # 下载数据, 完成后更新内存数据
      new_data = io_util.update_stock_data_new(symbols=sec_list, stock_data_path=config['data_path'], file_format='.csv', sources=sources, required_date=today, is_print=True, is_return=True, is_save=config['calculation']['save_sec_data'], api_keys=config['api_key'], update_mode=update_mode, query_benchmark=query_benchmark)
      for symbol in new_data.keys():
        sec_id = f'{symbol}_day'
        data['sec_data'][ti][sec_id] = new_data[symbol]
    else:
      logger.info(f'[skip]: <{stage}>')
      
    
    # ============================================== 计算 ==========================================================
    stage = 'calculation'
    logger.info('')
    logger.info(former_half_len*level_2_symbol + f' {stage} ' + (latter_half_len-len(stage))*level_2_symbol)
    if CALCULATION:
      
      # 初始化计算周期    
      data_start_date = start_date if start_date is not None else util.string_plus_day(string=today, diff_days=-config['calculation']['look_back_window'][interval])
      logger.info(f'[list]: {ti} <{data_start_date} - {today}>')
        
      # 遍历列表中的每个标的,保存最新的一行数据(tail(1))
      tmp_result = pd.DataFrame()
      for i in tqdm(range(len(sec_list)), ncols=tqdm_ncols, ascii=True, desc=f'[list]: {ti} '):
        
        # 查看数据是否存在, 若不存在则尝试从_day数据转化
        symbol = sec_list[i]
        sec_id = f'{symbol}_{interval}'

        sec_data = data['sec_data'][ti].get(sec_id)
        if sec_data is None:
          sec_data = data['sec_data'][ti].get(f'{symbol}_day')
        sec_data = io_util.switch_data_interval(df=sec_data, interval=interval)
          
        # 跳过无数据, 数据过少(<60), 未满足满足指定日期(required_date)的 # (len(sec_data[data_start_date:]) < 60) or
        if (sec_data is None) or ((required_date is not None) and (type(sec_data.index.max()) != float) and (sec_data.index.max() < required_date)):
          logger.info(f'[skip]: {sec_id} no data for calculation')
          continue
        
        # 获取现有 ta_data
        ta_data = data['ta_data'][ti].get(sec_id)

        # 重新计算 ta_features
        if config['calculation']['update_ta_data']:
          ta_data = ta_util.calculate_ta_feature(df=sec_data, start_date=data_start_date, symbol=symbol)

        # 重新计算 ta_signal
        if config['calculation']['update_ta_signal']:
          # 推断 market (us / a) 给 ML 模型查找用
          market_for_ml = _infer_market_for_target(target, sec_list)
          ta_data = ta_util.calculate_ta_signal(
            df=ta_data,
            market=market_for_ml,
            pool=target,
            horizon=ml_horizon,
            config=config,
            signal_source=signal_source,
            ml_pool=ml_pool,
          )

        # 保存当前计算结果  
        if ta_data is None or len(ta_data) == 0:
          print(f'{symbol}: get empty dataframe or none in ta_data calculation')
          continue
        else:
          data['ta_data'][ti][sec_id] = ta_data
          tmp_result = pd.concat([tmp_result, ta_data.tail(1)])

      # 保存当前target-interval最新数据
      data['result'][ti] = tmp_result

      # 保存计算结果
      if config['calculation']['save_ta_data']:
        io_util.pickle_dump_data(data=data['ta_data'][ti], file_path=config['quant_path'], file_name=f'{ti}_ta_data.pkl')
        io_util.pickle_dump_data(data=data['result'][ti], file_path=config['quant_path'], file_name=f'{ti}_result.pkl')
        logger.info(f'[save]: {ti} <ta_data, result>.pkl')
    else:
      logger.info(f'[skip]: <{stage}>')


    # ============================================== 可视化 ========================================================
    stage = 'visualization'
    logger.info('')
    logger.info(former_half_len*level_2_symbol + f' {stage} ' + (latter_half_len-len(stage))*level_2_symbol)
    if VISUALIZATION:

      # 根据周期设置可视化参数
      plot_start_date = start_date if start_date is not None else util.string_plus_day(string=today, diff_days=-config['visualization']['plot_window'][interval])
      
      # 设置图片保存路径, 若不存在则创建文件夹
      plot_save_path = config["result_path"] / target / interval
      if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

      # 如果需要保存图片, 先清空目标文件夹
      if config['visualization']['save_image']:
        for img in os.listdir(plot_save_path):
          os.remove(plot_save_path / img)

      # 遍历列表
      logger.info(f'[list]: {ti} <{plot_start_date} - {today}>')
      tmp_result = pd.DataFrame()
      for i in tqdm(range(len(sec_list)), ncols=tqdm_ncols, ascii=True, desc=f'[list]: {ti} '):
        symbol = sec_list[i]
        sec_id = f'{symbol}_{interval}'

        # 遍历列表中的每个标的
        if i % 10 == 0:
          plt.close('all')
          gc.collect()

        # 可视化数据, 保存最新记录
        plot_data = data['ta_data'][ti].get(sec_id)
        if plot_data is None:
          print(f'{symbol}: failed ploting as ta_data is None')
          continue
        else:
          if 'trade_info' not in globals():
            trade_info = None
          ta_util.visualization(df=plot_data, start=plot_start_date, interval=interval, title=f'{symbol}', save_path=plot_save_path, visualization_args=config['visualization'])
    else:
      logger.info(f'[skip]: <{stage}>')
    

    # # 完成当前列表的遍历
    logger.info('')    
    # logger.info('***********************************************************')
    logger.info('')
  
  # ================================================ 后处理 ========================================================
  stage = 'postprocess'
  
  logger.info(total_len * level_1_symbol)
  logger.info(former_half_len*level_1_symbol + f' {stage} ' + (latter_half_len-len(stage))*level_1_symbol)
  if POSTPROCESS:

    # 根据 pool 设定结果文件前缀
    prefix = '' if pool in ['us', 'universe'] else f'{pool}_'

    # 归集 signal
    signal = pd.DataFrame()
    for ti in data['result'].keys():

      # 解析目标列表
      target = ti.replace(f'_{interval}', '')      

      # 归集 signal
      data['final_result'][ti] = ta_util.postprocess(df=data['result'][ti], keep_columns=config['postprocess']['keep_columns'], drop_columns=config['postprocess']['drop_columns'], sec_names=config['visualization']['plot_args']['sec_name'], target_interval=ti)
      if len(data['final_result'][ti]) > 0: 
        
        # signal # pattern_score > 0 or pattern_score == 0 and 
        # signal = pd.concat([signal, data['final_result'][ti].query(f'(潜在信号 in ["up_1", "up", "down_up"])').copy()]) 
        signal = pd.concat([signal, data['final_result'][ti].query(f'(potential in ["up_1", "up", "down_up"])').copy()]) 
    
    sort_idx = ['trend_strength_symbol', 'position_score', 'potential_score', 'trigger_score']
    sort_order = [False, True, False, False]
    data['final_result']['signal'] = signal if len(signal) == 0 else signal.sort_values(by=sort_idx, ascending=sort_order)

    # 保存结果数据(final_result) 到excel
    if config['postprocess']['save_result']:
      file_name = f'{prefix}{today}.xlsx' if interval == 'day' else f'{prefix}{today}_{interval}.xlsx'
      io_util.dict_2_excel(dictionary=data['final_result'], file_path=config['result_path'], file_name=file_name)
      logger.info(f'[save]: {file_name}')

      # 更新持仓的支撑/阻挡位
      if pool in ['us', 'a']:
        io_util.update_portfolio_support_resistant(config=config, data=data, portfolio_file_name='portfolio.json', is_return=False)
        logger.info(f'[save]: portfolio support and resistant')
    
    # 保存符合条件的图片(signal, portfolio, index) 到PDF
    if config['postprocess']['save_pdf']:

      # 初始化图片列表
      images = {'signal':[], 'portfolio':[]}
      
      # 抽出 signal 图片
      for label in ['signal']:
        tmp_data = data['final_result'][label].head(20)
        if len(tmp_data) > 0:
          for index, row in tmp_data.iterrows():
            symbol = row['symbol']
            # print(symbol)
            ti_splited = row['ti'].split('_')
            t = '_'.join(ti_splited[:-1])
            i = ti_splited[-1]
            images[label].append(config['result_path'] / target / i / f'{symbol}.png')

      # 添加summary
      macro_path = config['result_path'] / 'macro' 
      if not os.path.exists(macro_path):
        os.mkdir(macro_path)
      summary_img = macro_path / f'{prefix}summary.png'
      ta_util.plot_summary(data, save_path=summary_img, config=config)
      images['signal'] = [summary_img] + images['signal']

      # 抽出 index，portfolio，global图片
      portfolio_list = []
      if pool=='us':
               
        # 抽出 index 图片
        idx_list_name = 'etf_3x'
        idx_list_symbol = data['result'][f'{idx_list_name}_{interval}'].sort_values(by=['potential_score', 'trend_strength_symbol', 'position_score', 'trigger_score'], ascending=[False, False, True, False])['symbol'].tolist()
        images['index'] = []
        index_path = config['result_path'] / idx_list_name / interval
        images['index'] += [index_path / f'{x}.png' for x in idx_list_symbol]

        # 抽出 global 图片
        global_list_name = 'global'
        global_list_symbol = data['result'][f'{global_list_name}_{interval}'].sort_values(by=['potential_score', 'trend_strength_symbol', 'position_score', 'trigger_score'], ascending=[False, False, True, False])['symbol'].tolist()
        images['global'] = []
        global_path = config['result_path'] / global_list_name / interval
        images['global'] += [global_path / f'{x}.png' for x in global_list_symbol]

        # 获取持仓股票列表
        portfolio_record = io_util.read_config(config['config_path'], 'portfolio.json')
        tiger_portfolio_list = list(portfolio_record['tiger']['global_account']['portfolio']['quantity'].keys()) 
        futu_portfolio_list = list(portfolio_record['futu']['REAL']['portfolio']['quantity'].keys())
        portfolio_list = list(set(tiger_portfolio_list + futu_portfolio_list))
      
      elif pool == 'a':

        # 抽出 index 图片        
        idx_list_name = 'a_etf'
        if len(config['selected_sec_list']['a_etf']) > 0:
          
          tmp_data = data['result'][f'{idx_list_name}_{interval}']
          idx_list_symbol = data['result'][f'{idx_list_name}_{interval}'].sort_values(by=['potential_score', 'trend_strength_symbol', 'position_score', 'trigger_score'], ascending=[False, False, True, False])['symbol'].tolist()
          images['index'] = []
          if len(idx_list_symbol) > 0:          
            index_path = config['result_path'] / idx_list_name / interval
            images['index'] += [index_path / f'{x}.png' for x in idx_list_symbol]

        # 获取持仓股票列表
        portfolio_record = io_util.read_config(config['config_path'], 'portfolio.json')
        snowball_portfolio_list = list(portfolio_record['pingan']['snowball']['portfolio']['quantity'].keys()) 
        portfolio_list = snowball_portfolio_list

      # 抽出 portfolio 图片
      portfolio_not_found = portfolio_list
      for target in target_list.keys():
        
        # 遍历目标列表
        ti = f'{target}_{interval}'
        if len(config['selected_sec_list'][target]) <= 0:
          continue

        # 查找 portfolio
        tmp_path = config['result_path'] / target / interval
        tmp_portfolio = data['final_result'][ti].query(f'symbol in {portfolio_list}')
        tmp_portfolio_symbols = [] if len(tmp_portfolio) == 0 else tmp_portfolio['symbol'].tolist()
        images['portfolio'] += [tmp_path / f'{symbol}.png' for symbol in tmp_portfolio_symbols]
        portfolio_not_found = [x for x in portfolio_not_found if x not in tmp_portfolio_symbols]

      # 合成 PDF 文件
      prefix = f'{pool}_' if pool != 'us' else ''
      for label in images.keys():

        # 如有同名旧文件存在, 则删除
        tmp_pdf_path = config['result_path'] / f'{prefix}{label}.pdf'
        if os.path.exists(tmp_pdf_path):
          os.remove(tmp_pdf_path)

        # 如相应图片存在, 则合成为pdf
        if len(images[label]) > 0:
          util.image_2_pdf(image_list=images[label], save_name=tmp_pdf_path)
          
        extra_info = f'(not found: {portfolio_not_found})' if (label == 'portfolio' and len(portfolio_not_found) > 0) else ''
        logger.info(f'[save]: {label} images {len(images[label])}{extra_info}') 

    # 基于大股票池的计算结果+持仓列表更新小股票池
    portfolio_record = io_util.read_config(config['config_path'], 'portfolio.json')
    for ti in data['result'].keys():
      if ti == f'hs300_{interval}':
        a_potential_symbol_list = data['result'][ti].query('(pattern_score > 0 or (pattern_score == 0 and trend not in ["down"]))').sort_values(by=['potential_score', 'trend_strength_symbol', 'position_score', 'trigger_score'], ascending=[False, False, True, False]).head(50)['symbol'].tolist()
        a_portfolio_symbol_list = list(portfolio_record['pingan']['snowball']['portfolio']['quantity'].keys()) 
        a_portfolio_symbol_list = [x for x in a_portfolio_symbol_list if x not in config['selected_sec_list']['a_etf']]

        new_a_company_list = a_potential_symbol_list + [x for x in a_portfolio_symbol_list if x not in a_potential_symbol_list]
        io_util.modify_config(config_key='a_company', config_value=new_a_company_list, file_path=config['config_path'], file_name='selected_sec_list.json', is_print=False)
        io_util.modify_config(config_key='a_company_update_time', config_value=today, file_path=config['config_path'], file_name='selected_sec_list.json', is_print=False)
        logger.info(f'[save]: updated pool [a_company] with {len(new_a_company_list)} symbols')

      if ti == f'company_300_{interval}':
        us_potential_symbol_list = data['result'][ti].query('(pattern_score > 0 or (pattern_score == 0 and trend not in ["down"]))').sort_values(by=['potential_score', 'trend_strength_symbol', 'position_score', 'trigger_score'], ascending=[False, False, True, False]).head(100)['symbol'].tolist()
        us_portfolio_symbol_list = list(set(list(portfolio_record['tiger']['global_account']['portfolio']['quantity'].keys()) + list(portfolio_record['futu']['REAL']['portfolio']['quantity'].keys())))
        us_portfolio_symbol_list = [x for x in us_portfolio_symbol_list if x not in config['selected_sec_list']['etf_3x'] and x not in config['selected_sec_list']['global']]
        watch_list = []# ['OMRNY']

        new_company_star_list = list(set(us_potential_symbol_list + us_portfolio_symbol_list + watch_list))
        io_util.modify_config(config_key='company_star', config_value=new_company_star_list, file_path=config['config_path'], file_name='selected_sec_list.json', is_print=False)
        io_util.modify_config(config_key='company_star_update_time', config_value=today, file_path=config['config_path'], file_name='selected_sec_list.json', is_print=False)
        logger.info(f'[save]: updated pool [company_star] with {len(new_company_star_list)} symbols')

    # 通过邮件发送计算结果
    if config['postprocess']['send_email']:
      signal_file_date = today
      ret = io_util.send_result_by_email(
        config=config, 
        to_addr='northcheng@notme.com', from_addr='northcheng@qq.com', 
        smtp_server='smtp.qq.com', platform=[], signal_file_date=signal_file_date, log_file_date=None, test=False, pool=pool
      )
      logger.info(f'[send]: email for signals calculated on {signal_file_date}')
  
    # 清理历史Excel文件
    if config['postprocess']['archive_old_xlsx']:

      # xlsx 列表
      xlsx_files = os.listdir(config['result_path'])
      xlsx_files = [x for x in xlsx_files if x.endswith('.xlsx')]

      # 按创建日期归集文件
      file_on_date = {}
      for f in xlsx_files:
        try:
          full_path = config['result_path'] / f
          tmp_ctime = os.path.getctime(full_path)
          tmp_localtime = time.localtime(tmp_ctime)
          tmp_cdate = time.strftime("%Y-%m-%d", tmp_localtime)

          if tmp_cdate not in file_on_date.keys():
            file_on_date[tmp_cdate] = [f]
          else:
            file_on_date[tmp_cdate].append(f)
        except Exception as e:
          logger.error(f'{e}')
          continue

      # 归档文件夹, 不存在则新建    
      xlsx_folder = config['result_path'] / 'xlsx'
      if not os.path.exists(xlsx_folder):
        os.mkdir(xlsx_folder)
      
      # 归档Excel
      archive_counter = 0
      for d in file_on_date:
        try:
          
          # 当天的excel除外
          if d == today:
            continue
          
          # 按月创建文件夹进行归档
          tmp_folder = xlsx_folder / d[:7]
          if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)

          # 若文件已存在则覆盖
          tmp_file = file_on_date[d]
          for tf in tmp_file:
            src = config['result_path'] / tf
            dst = tmp_folder / tf
            if os.path.exists(dst):
              os.remove(dst)
            os.rename(src, dst)
            archive_counter += 1

        except Exception as e:
          logger.error(f'{e}')
          continue
      
      if archive_counter > 0:
        logger.info(f'[cler]: archieved xlsx files {archive_counter}')

  else:
    logger.info(f'[skip]: <{stage}>')

except Exception as e:
  logger.error(f'[erro]: in {stage} - {e}')

end_time = datetime.datetime.now()
duration = (end_time - start_time).seconds
logger.info(f'[cost]: {round(duration / 60, 2)}mins')
logger.info('')
logger.info(total_len * level_1_symbol)