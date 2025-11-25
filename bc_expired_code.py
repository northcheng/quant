#20251125 updated
[
# 基础 (不需要重新计算)
'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split', 'symbol', 'rate', 

# candle: 基本元素 (需要重新计算-1)
'candle_color', 'candle_shadow', 'candle_entity', 'candle_entity_top', 'candle_entity_bottom',
'candle_upper_shadow_pct', 'candle_lower_shadow_pct', 'candle_entity_pct', 

# candle: 相对位置 (需要重新计算-2)
'相对candle位置', 'candle_position_score',

# candle: 窗口 (需要重新计算-2)
'candle_gap', 'candle_gap_color', 'candle_gap_top', 'candle_gap_bottom', 'candle_gap_distance', 

# ichimoku
'tankan', 'kijun', 'senkou_a', 'senkou_b', 'chikan', 

# kama
'kama_fast', 'kama_slow', 

# adx
'tr', 'atr', 'adx_value', 'adx_strength', 

# RSI
'rsi', 'rsi_signal', 

# ichimoku: distance
'ichimoku_distance', 'ichimoku_distance_day', '相对ichimoku位置',
'tankan_rate', 'tankan_day', 'kijun_rate', 'kijun_day',  

# kama: distance
'kama_distance', 'kama_distance_day', '相对kama位置', 
'kama_fast_rate', 'kama_fast_day',  'kama_slow_rate', 'kama_slow_day',  

# adx: basic
'adx_value_change', 'adx_direction', 'adx_direction_day', 'adx_direction_start', 
'adx_strength_change', 'adx_power', 'adx_power_day', 'adx_power_start', 
'adx_value_prediction', 'adx_value_pred_change', 
'adx_power_start_adx_value', 'prev_adx_duration', 
'adx_strong_day', 

# adx: distance
'adx_distance', 'adx_distance_change', 'adx_distance_day',
'adx_trend', 'adx_day',

# 相对位置
'position_score', 'position', 

# ichimoku-kama
'ki_distance', 

# candle: pattern
'shadow_trend', 'entity_trend', 
'长影线_trend', '长影线_day',
'窗口_trend', '窗口_day',
'十字星_trend', '十字星_day', 
'流星_trend', '流星_day', 
'锤子_trend', '锤子_day', 
'腰带_trend', '腰带_day', 
'平头_trend', '平头_day', 
'吞噬_trend', '吞噬_day',
'包孕_trend', '包孕_day', 
'穿刺_trend', '穿刺_day', 
'启明黄昏_trend', '启明黄昏_day', 
'candle_pattern_up_score', 'candle_pattern_up_description',
'candle_pattern_down_score', 'candle_pattern_down_description',
'candle_pattern_score', 

# support/resistant, break_up/break_down
'support_score', 'support_description', 
'resistant_score', 'resistant_description', 

'break_up_score', 'break_up_description', 
'break_down_score', 'break_down_description',

'kama_slow_break_up', 'kama_slow_break_down', 
'kama_slow_support', 'kama_slow_resistant', 
'candle_gap_bottom_support', 'candle_gap_top_resistant', 

'resistant', 'resistanter',
'support', 'supporter', 

'boundary_score', 
'break_score',

# trigger_score
'trigger_up_score', 'trigger_up_score_description',
'trigger_down_score', 'trigger_down_score_description',
'trigger_score', 'trigger_day', 

# adx/ichimoku/kama/aki
'adx_rate', 'adx_distance_status', 
'ichimoku_rate', 'ichimoku_distance_status', 
'kama_rate', 'kama_distance_status', 
'aki_rate', 'aki_status', 'aki_rate_day', 'aki_rate_change', 'aki_score', 'aki_score_change',

# position
'位置',

# trend (低位/高位, 触底/触顶, 上行/下行, 波动)
'trend_up_score', 'trend_down_score', 'trend_wave_score',
'trend_score', 'trend', 'trend_description', 'trend_day',

# pattern (超买, ichimoku/kama死叉, 长线阻挡)/(超卖, ichimoku/kama金叉, 长线支撑)
'pattern_up_score', 'pattern_down_score', 'pattern_score', 'pattern_description', 
'超买超卖', '关键交叉i', '关键交叉k', '长线边界', 

# signal
'signal', 'signal_day', 'signal_score', 'signal_score_change', 'signal_description'
]
