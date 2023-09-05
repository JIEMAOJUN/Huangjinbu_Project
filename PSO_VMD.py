import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib import pyplot as plt
from vmdpy import VMD
from scipy.fftpack import hilbert
from scipy.signal import argrelextrema
from scipy.signal import hilbert

matplotlib.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
matplotlib.rcParams['axes.unicode_minus']=False       #显示负号

def PSO_VMD(result,freq):
    '''
    输入：
    result:数据库提取的回路时序
    freq:采样频率
    输出：
    N:采样点个数
    t:时间点
    K:分解模态个数
    '''
    # 格式转换
    sqldata = np.array(result)
    sqldata = sqldata.astype(float)
    sqldata = sqldata.T
    sqldata0 = sqldata[0]
    sqldata1 = sqldata[1]

    # 数据参数
    N = ((len(sqldata0))//100)*100
    f = sqldata0[:N]
    t = np.linspace(0,freq*N,N)
    sp = sqldata1[:N]

    # VMD参数
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-6

    # pso参数
    w_ini = 0.9
    w_end = 0.4
    c1 = 0.2                                # 学习因子
    c2 = 0.5                                # 学习因子
    n_iterations = 20                       # 迭代次数
    n_particles = 30                       # 种群规模
    low = [6, 100]
    up = [10, 3000]
    var_num = 2
    bound = (low,up)

    # 计算每个IMF分量的包络熵
    def calculate_entropy(imf):
        #每个分量的包络信号为env
        env = np.abs(hilbert(imf))
        # 将每个包络信号归一化到 [0, 1] 区间内
        env_norm = env / np.max(env)#在计算包络熵的过程中，需要对包络信号进行归一化处理，以确保不同幅度的包络信号具有可比性。
        # 将归一化后的包络信号作为概率分布
        p = env_norm / np.sum(env_norm)
        return -np.sum(p * np.log2(p))

    # 适应度函数
    def fitness_function(position):
        
        K = int(position[0])
        alpha = position[1]
        if K < bound[0][0]:
            K = bound[0][0]
        if K > bound[1][0]:
            K = bound[1][0]
            
        if alpha < bound[0][1]:
            alpha = bound[0][1]
        if alpha > bound[1][1]:
            alpha =bound[1][1]
    
        u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

        # 包络熵作为适应度函数
        num_modes = u.shape[0]
        entropy = np.zeros(num_modes)
        for i in range(num_modes):
            entropy[i] = calculate_entropy(u[i,:])
        
        # 找到最小的包络熵对应的模态
        min_entropy_index = np.argmin(entropy)
        min_entropy_mode = u[min_entropy_index]

        print("最小包络熵对应的模态：", min_entropy_index)
        # 返回最大包络熵值
        s = entropy[min_entropy_index]
        return s
        
    pop_x = np.zeros((n_particles,var_num))
    g_best = np.zeros(var_num)
    temp = -1
    for i in range(n_particles):
        for j in range(var_num):
            pop_x[i][j] = np.random.rand()*(bound[1][j]-bound[0][j])+bound[0][j]
        fit = fitness_function(pop_x[i])
    
        if fit > temp:
            g_best = pop_x[i]
            temp = fit
    pbest_position = pop_x
    pbest_fitness_value = np.zeros(n_particles)
    gbest_fitness_value = np.zeros(var_num)    
    gbest_position = g_best
    velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])
    iteration = 0

    while iteration < n_iterations:
        for i in range(n_particles):
            # print(pop_x[i])
            fitness_cadidate = fitness_function(pop_x[i])
            print("error of particle-", i, "is (training, test)", fitness_cadidate)
            print(" At (K, alpha): ",int(pop_x[i][0]),pop_x[i][1])
            # 更新粒子的最佳pos和fit
            if (pbest_fitness_value[i] > fitness_cadidate):
                pbest_fitness_value[i] = fitness_cadidate
                pbest_position[i] = pop_x[i]
            # 更新全局的最佳pos和fit
            elif (gbest_fitness_value[1] > fitness_cadidate):
                gbest_fitness_value[1] = fitness_cadidate
                gbest_position = pop_x[i]

            elif (gbest_fitness_value[0] < fitness_cadidate):
                gbest_fitness_value[0] = fitness_cadidate
                gbest_position = pop_x[i]

        for i in range(n_particles):
            W = w_end + (w_ini - w_end)*(n_iterations - iteration)/n_iterations
            new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                        pbest_position[i] - pop_x[i]) + (c2 * random.random()) * (
                                    gbest_position - pop_x[i])
            new_position = new_velocity + pop_x[i]
            pop_x[i] = new_position

        iteration = iteration + 1


    print("The best position is ", int(gbest_position[0]),gbest_position[1], "in iteration number", iteration, "with error (train, test):",
        fitness_function(gbest_position))

    K = int(gbest_position[0])
    alpha = int(gbest_position[1])


    return K,alpha








