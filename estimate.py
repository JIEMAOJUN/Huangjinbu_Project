
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
import pymssql

matplotlib.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
matplotlib.rcParams['axes.unicode_minus']=False       #显示负号

# 仅限设定值为阶跃扰动试验
def step_estimate(set_v,time_series,threshold,set_rate,stb_quality_indicators,set_stable_time):
    '''
    输入：
    set_v:设定值
    time_series:时间序列
    threshold:触发门槛
    set_rate:衰减率设定
    stb_quality_indicator:动态品质指标
    set_stable_time:稳定时间设定
    
    '''

    def detect_changes(set_v, threshold, time_window):
        start_time = None
        end_time = None
        results = []
        
        for i in range(len(set_v)):
            if start_time is None:
                start_time = i
            elif i - start_time >= time_window:
                if end_time is None or set_v[i] - set_v[start_time] >= threshold:
                    end_time = i
                else:
                    start_time = None
                    end_time = None
            elif set_v[i] - set_v[start_time] >= threshold:
                end_time = i
            
            if end_time is not None:
                results.append((start_time, end_time))
                start_time = None
                end_time = None

        return results

    def decay_rate_estimate(s,t,change_direction,set_rate):
        '''
        输入：
        s:设定值时间段
        t:实际值时间段
        time_d_max:最大扰动时间点
        set_rate:衰减率设定值
        '''
        # 对实际值曲线求极值点（从上面计算得到的实验开始时间起进行求值）
        x_greater = argrelextrema(t,np.greater) ##极大值点
        x_less = argrelextrema(t,np.less) ##极小值点
        # plt.figure()
        # plt.plot(t)
        # plt.plot(s)
        # plt.plot(x_greater[0],t[x_greater],'o', markersize=7) 
        # plt.plot(x_less[0],t[x_less],'+', markersize=10) 
        y_greater = t[x_greater] ##极大值
        y_less = t[x_less] ##极小值

        # 求衰减率
        # 依据幅值变化方向判断取值为极大值还是极小值

        if change_direction>0:
            M = abs(y_greater-s[x_greater])
            M1 = M[0]
            M2 = M[1]
        else:
            M = abs(y_less-s[x_less])
            M1 = M[0]
            M2 = M[1]   
        # print('衰减波峰时间值',M)
        decay_rate = (M1-M2)/M1
        if decay_rate>set_rate:
            Q1 = True
            print('衰减率：',decay_rate)
        else:
            Q1 = False
            print('衰减率：',decay_rate)
        return decay_rate,Q1
    
    def stable_time_estimate(s,t,change_direction,stb_quality_indicators,set_stable_time):
        '''
        输入：
        s:设定值时间段
        t:实际值时间段
        step_time:阶跃发生时间
        set_stable_time:稳定时间设定值
        stb_quality_indicators:稳态品质指标
        '''
        # 求稳定时间
        diff_y = abs(t-s) 
        stable_index = [] ##稳态区间索引

        for i in range(0,len(diff_y)):         
            if abs(diff_y[i])<stb_quality_indicators:         
                stable_index.append(i)
        # print('稳定区间索引',stable_index)

        x_greater = np.array(argrelextrema(t,np.greater)) ##极大值点
        x_less = np.array(argrelextrema(t,np.less)) ##极小值点

        if change_direction>0:
            extrema = x_greater[0][0]
        else:
            extrema = x_less[0][0]
        # print('实验时间',extrema)

        new_stable_index = []
        for i in range(len(stable_index)):
            if stable_index[i]>extrema:
                new_stable_index.append(stable_index[i])
        # print('实验后稳定区间',new_stable_index)

        stable_continue = []
        for i in range(0,len(new_stable_index)-1):
            if new_stable_index[i+1]-new_stable_index[i]==1:
                stable_continue.append(new_stable_index[i+1])
        # print('稳定连续时间',stable_continue)

        stable_start = stable_continue[0]       
        if stable_start <= set_stable_time:
            Q2 = True
            print('稳定时间：',stable_start)
        else:
            Q2 = False
            print('稳定时间：',stable_start)
        return stable_start,Q2


    results = detect_changes(set_v, threshold,5)
    indicators = []
    
    for start_time, end_time in results:
        time_range_s = set_v[start_time:start_time+100]
        time_range_t = time_series[start_time:start_time+100]
        change_direction = set_v[end_time]-set_v[start_time]
        indicator1,Q1 = decay_rate_estimate(time_range_s,time_range_t,change_direction,set_rate)
        indicator2,Q2 = stable_time_estimate(time_range_s,time_range_t,change_direction,stb_quality_indicators,set_stable_time)
        indicators.append((end_time, indicator1,Q1,indicator2,Q2))
    
    return indicators

    

   

    