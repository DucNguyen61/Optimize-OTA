from numpy.random import randint
from numpy.random import rand
import pandas as pd
import datetime as dt
import os
import numpy as np
import math

def test(particles, n_requirement):
    param_result = np.zeros((particles, n_requirement))

    with open("./APSO_results_opamp.txt") as f1:
        param = f1.readlines()
        param = np.array(param)
        param = param.astype(float)  # Chuyển đổi dữ liệu từ dạng chuỗi sang float
        param_result[:len(param)] = param.reshape(particles, n_requirement)  # Gán dữ liệu từ param vào param_result

    Cond = np.zeros(particles).astype(int)
    PM = np.zeros(particles).astype(float)
    Gain = np.zeros(particles).astype(float)
    GBW = np.zeros(particles).astype(float)
    Power = np.zeros(particles).astype(float)
    Itotal = np.zeros(particles).astype(float)
    CMRR = np.zeros(particles).astype(float)
    PSRR_n = np.zeros(particles).astype(float)
    PSRR_p = np.zeros(particles).astype(float)
    SR = np.zeros(particles).astype(float)
    Tan_60 = math.tan(math.radians(60))
    Cload = 1 * 1e-12
    fitness = np.zeros(particles).astype(float)

    for i in range(particles):
        Cond[i] = param_result[i,0]
        PM[i] = param_result[i,1]
        Gain[i] = param_result[i,2]
        CMRR[i] = param_result[i,3]
        GBW[i] = param_result[i,4] 
        Power[i] = param_result[i,5] 
        PSRR_n[i] = param_result[i,6]
        PSRR_p[i] = param_result[i,7]
        SR[i] = param_result[i,8] 
        Itotal[i] = Power[i]/1.2

        print("CMRR: ",CMRR[i])

        if Cond[i] == 0:
            fitness[i] = -1
        elif ( (PM[i] < 60.0) | (Gain[i] < 50.0) | (GBW[i] < 50) | (Power[i] > 250.0) | (CMRR[i] < 50) | (PSRR_p[i] < 50) | (SR[i] < 30) ):
            fitness[i] = 0
        else: 
            fitness[i] =  ((GBW[i] * 10e6 * Cload) * (PM[i]/Tan_60))/(Itotal[i] * 10e-6)

    return fitness,param_result

if __name__ == "__main__":
    particales = 16
    n_requirement = 9
    fitness,result = test(particales,n_requirement)

    print("fitness: ",fitness)
    print("result :", result)
