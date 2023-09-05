
import numpy as np
import pymysql
from vmdpy import VMD
from estimate import step_estimate
from DL_rec import rec
from stblty_judge import calculate_stability

def fetch_data_from_database(host, user, password, database, table):
    """
    从数据库获取数据，并将第一列作为数据的索引
    
    参数：
    host: 数据库主机名
    user: 数据库用户名
    password: 数据库密码
    database: 数据库名称
    table: 数据表名称
    
    返回值：
    data: 获取的数据，DataFrame格式
    """
    # 连接数据库
    connection = pymysql.connect(host=host, user=user, password=password, database=database)
    
    # 查询数据
    query = f"SELECT * FROM {table}"
    data = pd.read_sql(query, connection)
    
    # 将第一列作为索引
    data.set_index(data.columns[0], inplace=True)
    
    # 关闭数据库连接
    connection.close()
    
    return data


def indicator_estimate(set_v,data,K,alpha,set_change,set_rate,stb_quality_indicators,set_stable_time):
    u,_,_ = VMD(data, alpha, 0, K, 0, 1, 1e-6) 
    rec = np.sum(u[0:K//2],axis = 0)
    indicators= step_estimate(set_v,rec,set_change,set_rate,stb_quality_indicators,set_stable_time)
    return indicators
        

def stability_judge(set_v,actual_v):
    stability, unstable_times = calculate_stability(set_v,actual_v,)




def configure_parameters(params):
    indicator_estimate(**params)      
    # 配置参数
    parameters = {
        
    }



