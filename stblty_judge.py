import pandas as pd

def calculate_stability(set_values, actual_values, threshold):
    """
    计算两组时间戳最接近的每个时刻的差值，并记录不稳定时间
    
    参数：
    set_values: 设备运行的设定值，DataFrame格式，索引为时间戳
    actual_values: 设备运行的实际值，DataFrame格式，索引为时间戳
    threshold: 阈值，差值超过设定值的20%时被记录为不稳定时间
    
    返回值：
    stability: 布尔值，表示设备运行是否稳定
    unstable_times: 不稳定时间戳列表
    """
    # 合并两组数据，保留相同时间戳的行
    merged_data = pd.merge(set_values, actual_values, left_index=True, right_index=True)
    
    # 初始化不稳定时间戳列表
    unstable_times = []
    
    # 遍历每个时刻
    for timestamp, row in merged_data.iterrows():
        # 计算差值
        diff = abs(row['设定值'] - row['实际值'])
        
        # 计算阈值
        threshold_value = row['设定值'] * threshold
        
        # 判断差值是否超过阈值
        if diff > threshold_value:
            unstable_times.append(timestamp)
    
    # 判断是否存在不稳定时间戳
    if len(unstable_times) > 0:
        stability = False
    else:
        stability = True
    
    return stability, unstable_times