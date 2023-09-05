import matplotlib
import pandas as pd
import numpy as np
import pymysql
import websocket
import json
from websocket import create_connection
from data_process import indicator_estimate



matplotlib.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
matplotlib.rcParams['axes.unicode_minus']=False       #显示负号




# 从MySQL接收数据,在函数内部调整数据库参数
def receive_data_from_mysql(SQL):
    # 连接MySQL数据库
    conn = pymysql.connect(host='localhost', user='ZZW', password='123456', db='hjb')
    cursor = conn.cursor()

    # 执行SQL查询语句
    cursor.execute(SQL)
    # dcs1_10lba10ct001         主蒸汽温度1
    # dcs1_10rtu01_01_cra       AGC指令
    # dcs1_ccmode               协调模式
    # dcs1_deh1_gv1pz1          GV1反馈
    # dcs1_deh1_gv2pz1          GV2反馈
    # dcs1_deh1_gv3pz1          GV3反馈
    # dcs1_deh1_gv4pz1          GV4反馈
    # dcs1_deh1_ws              转速
    # dcs1_gen_a_mw             实际负荷
    # dcs1_thrpress             实际主蒸汽压力
    # dcs1_thrprstp             主蒸汽压力设定值
    # dcs1_totfuel              实际总煤量
    # dcs2_20cff001_01_cra      AGC指令
    # dcs2_20lba10ct001         主蒸汽温度1
    # dcs2_ccmode               协调模式
    # dcs2_deh1_gv1pz1          GV1反馈
    # dcs2_deh1_gv2pz1          GV2反馈
    # dcs2_deh1_gv3pz1          GV3反馈
    # dcs2_deh1_gv4pz1          GV4反馈
    # dcs2_deh2_ws              转速
    # dcs2_gen_a_mw             实际负荷
    # dcs2_thrpress             实际主蒸汽压力
    # dcs2_thrprstp             主蒸汽压力设定值
    # dcs2_totfuel              实际总煤量

    # 获取查询结果
    data = cursor.fetchall()


    # 关闭数据库连接
    cursor.close()
    conn.close()

    return data


def process_data(set_v,data):
    processed_data = []   

    indicators = indicator_estimate(data,)  
    processed_data.append(indicators)

    return processed_data

# 将数据通过WebSocket发送到前端
def send_data_to_frontend(processed_data):
    # 连接到WebSocket服务器
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8000")

    # 将数据转换为JSON格式并发送到前端
    json_data = json.dumps(processed_data)
    ws.send(json_data)

    # 关闭WebSocket连接
    ws.close()


receive_data_from_mysql("SELECT * FROM dcs1_10lba10ct001")
