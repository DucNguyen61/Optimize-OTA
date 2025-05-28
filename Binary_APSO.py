from numpy.random import randint
from numpy.random import rand
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np
import shutil
import math


def calFitness(sol,particles, n_requirement): 
    
    #Gán các giá trị cho các biến của opamp
    print("Array : ",sol) 

    w01_1 = sol[0,0] * u
    w01_2 = sol[1,0] * u
    w01_3 = sol[2,0] * u
    w01_4 = sol[3,0] * u
    w01_5 = sol[4,0] * u
    w01_6 = sol[5,0] * u
    w01_7 = sol[6,0] * u
    w01_8 = sol[7,0] * u
    w01_9 = sol[8,0] * u
    w01_10 = sol[9,0] * u
    w01_11 = sol[10,0] * u
    w01_12 = sol[11,0] * u
    w01_13 = sol[12,0] * u
    w01_14 = sol[13,0] * u
    w01_15 = sol[14,0] * u
    w01_16 = sol[15,0] * u

    l01_1 = sol[0,1] * u
    l01_2 = sol[1,1] * u
    l01_3 = sol[2,1] * u
    l01_4 = sol[3,1] * u
    l01_5 = sol[4,1] * u
    l01_6 = sol[5,1] * u
    l01_7 = sol[6,1] * u
    l01_8 = sol[7,1] * u
    l01_9 = sol[8,1] * u
    l01_10 = sol[9,1] * u
    l01_11 = sol[10,1] * u
    l01_12 = sol[11,1] * u
    l01_13 = sol[12,1] * u
    l01_14 = sol[13,1] * u
    l01_15 = sol[14,1] * u
    l01_16 = sol[15,1] * u

    w23_1 = sol[0,2] * u
    w23_2 = sol[1,2] * u
    w23_3 = sol[2,2] * u
    w23_4 = sol[3,2] * u
    w23_5 = sol[4,2] * u
    w23_6 = sol[5,2] * u
    w23_7 = sol[6,2] * u
    w23_8 = sol[7,2] * u
    w23_9 = sol[8,2] * u
    w23_10 = sol[9,2] * u
    w23_11 = sol[10,2] * u
    w23_12 = sol[11,2] * u
    w23_13 = sol[12,2] * u
    w23_14 = sol[13,2] * u
    w23_15 = sol[14,2] * u
    w23_16 = sol[15,2] * u

    l23_1 = sol[0,3] * u
    l23_2 = sol[1,3] * u
    l23_3 = sol[2,3] * u
    l23_4 = sol[3,3] * u
    l23_5 = sol[4,3] * u
    l23_6 = sol[5,3] * u
    l23_7 = sol[6,3] * u
    l23_8 = sol[7,3] * u
    l23_9 = sol[8,3] * u
    l23_10 = sol[9,3] * u
    l23_11 = sol[10,3] * u
    l23_12 = sol[11,3] * u
    l23_13 = sol[12,3] * u
    l23_14 = sol[13,3] * u
    l23_15 = sol[14,3] * u
    l23_16 = sol[15,3] * u

    w47_1 = sol[0,4] * u
    w47_2 = sol[1,4] * u
    w47_3 = sol[2,4] * u
    w47_4 = sol[3,4] * u
    w47_5 = sol[4,4] * u
    w47_6 = sol[5,4] * u
    w47_7 = sol[6,4] * u
    w47_8 = sol[7,4] * u
    w47_9 = sol[8,4] * u
    w47_10 = sol[9,4] * u
    w47_11 = sol[10,4] * u
    w47_12 = sol[11,4] * u
    w47_13 = sol[12,4] * u
    w47_14 = sol[13,4] * u
    w47_15 = sol[14,4] * u
    w47_16 = sol[15,4] * u

    w5_1 = sol[0,5] * u
    w5_2 = sol[1,5] * u
    w5_3 = sol[2,5] * u
    w5_4 = sol[3,5] * u
    w5_5 = sol[4,5] * u
    w5_6 = sol[5,5] * u
    w5_7 = sol[6,5] * u
    w5_8 = sol[7,5] * u
    w5_9 = sol[8,5] * u
    w5_10 = sol[9,5] * u
    w5_11 = sol[10,5] * u
    w5_12 = sol[11,5] * u
    w5_13 = sol[12,5] * u
    w5_14 = sol[13,5] * u
    w5_15 = sol[14,5] * u
    w5_16 = sol[15,5] * u
    
    l457_1 = sol[0,6] * u
    l457_2 = sol[1,6] * u 
    l457_3 = sol[2,6] * u
    l457_4 = sol[3,6] * u
    l457_5 = sol[4,6] * u
    l457_6 = sol[5,6] * u
    l457_7 = sol[6,6] * u
    l457_8 = sol[7,6] * u 
    l457_9 = sol[8,6] * u
    l457_10 = sol[9,6] * u
    l457_11 = sol[10,6] * u
    l457_12 = sol[11,6] * u
    l457_13 = sol[12,6] * u
    l457_14 = sol[13,6] * u
    l457_15 = sol[14,6] * u
    l457_16 = sol[15,6] * u

    w6_1 = sol[0,7] * u
    w6_2 = sol[1,7] * u
    w6_3 = sol[2,7] * u
    w6_4 = sol[3,7] * u
    w6_5 = sol[4,7] * u
    w6_6 = sol[5,7] * u
    w6_7 = sol[6,7] * u
    w6_8 = sol[7,7] * u
    w6_9 = sol[8,7] * u
    w6_10 = sol[9,7] * u
    w6_11 = sol[10,7] * u
    w6_12 = sol[11,7] * u
    w6_13 = sol[12,7] * u
    w6_14 = sol[13,7] * u
    w6_15 = sol[14,7] * u
    w6_16 = sol[15,7] * u

    l6_1 = sol[0,8] * u
    l6_2 = sol[1,8] * u
    l6_3 = sol[2,8] * u
    l6_4 = sol[3,8] * u
    l6_5 = sol[4,8] * u
    l6_6 = sol[5,8] * u
    l6_7 = sol[6,8] * u
    l6_8 = sol[7,8] * u 
    l6_9 = sol[8,8] * u
    l6_10 = sol[9,8] * u
    l6_11 = sol[10,8] * u
    l6_12 = sol[11,8] * u
    l6_13 = sol[12,8] * u
    l6_14 = sol[13,8] * u
    l6_15 = sol[14,8] * u
    l6_16 = sol[15,8] * u

    Cc_1 = sol[0,9] * p
    Cc_2 = sol[1,9] * p
    Cc_3 = sol[2,9] * p
    Cc_4 = sol[3,9] * p
    Cc_5 = sol[4,9] * p
    Cc_6 = sol[5,9] * p
    Cc_7 = sol[6,9] * p
    Cc_8 = sol[7,9] * p
    Cc_9 = sol[8,9] * p
    Cc_10 = sol[9,9] * p
    Cc_11 = sol[10,9] * p
    Cc_12 = sol[11,9] * p
    Cc_13 = sol[12,9] * p
    Cc_14 = sol[13,9] * p
    Cc_15 = sol[14,9] * p
    Cc_16 = sol[15,9] * p


    ### Attach parameters to OCEAN File ### 
    filethongso = ("./APSO_params_opamp.txt")
    f = open(filethongso, 'w')

    f.write(  "%s"   %w01_1            )
    f.write(  "\n%s"   %w01_2            )
    f.write(  "\n%s"   %w01_3            )
    f.write(  "\n%s"   %w01_4            )
    f.write(  "\n%s"   %w01_5            )
    f.write(  "\n%s"   %w01_6            )
    f.write(  "\n%s"   %w01_7            )
    f.write(  "\n%s"   %w01_8            )
    f.write(  "\n%s"   %w01_9            )
    f.write(  "\n%s"   %w01_10            )
    f.write(  "\n%s"   %w01_11            )
    f.write(  "\n%s"   %w01_12            )
    f.write(  "\n%s"   %w01_13            )
    f.write(  "\n%s"   %w01_14            )
    f.write(  "\n%s"   %w01_15            )
    f.write(  "\n%s"   %w01_16            )

    f.write(  "\n%s"   %l01_1            )
    f.write(  "\n%s"   %l01_2            )
    f.write(  "\n%s"   %l01_3            )
    f.write(  "\n%s"   %l01_4            )
    f.write(  "\n%s"   %l01_5            )
    f.write(  "\n%s"   %l01_6            )
    f.write(  "\n%s"   %l01_7            )
    f.write(  "\n%s"   %l01_8            )
    f.write(  "\n%s"   %l01_9            )
    f.write(  "\n%s"   %l01_10            )
    f.write(  "\n%s"   %l01_11            )
    f.write(  "\n%s"   %l01_12            )
    f.write(  "\n%s"   %l01_13            )
    f.write(  "\n%s"   %l01_14            )
    f.write(  "\n%s"   %l01_15            )
    f.write(  "\n%s"   %l01_16            )

    f.write(  "\n%s"   %w23_1            )
    f.write(  "\n%s"   %w23_2            )
    f.write(  "\n%s"   %w23_3            )
    f.write(  "\n%s"   %w23_4            )
    f.write(  "\n%s"   %w23_5            )
    f.write(  "\n%s"   %w23_6            )
    f.write(  "\n%s"   %w23_7            )
    f.write(  "\n%s"   %w23_8            )
    f.write(  "\n%s"   %w23_9            )
    f.write(  "\n%s"   %w23_10            )
    f.write(  "\n%s"   %w23_11            )
    f.write(  "\n%s"   %w23_12            )
    f.write(  "\n%s"   %w23_13            )
    f.write(  "\n%s"   %w23_14            )
    f.write(  "\n%s"   %w23_15            )
    f.write(  "\n%s"   %w23_16            )

    f.write(  "\n%s"   %l23_1            )
    f.write(  "\n%s"   %l23_2            )
    f.write(  "\n%s"   %l23_3            )
    f.write(  "\n%s"   %l23_4            )
    f.write(  "\n%s"   %l23_5            )
    f.write(  "\n%s"   %l23_6            )
    f.write(  "\n%s"   %l23_7            )
    f.write(  "\n%s"   %l23_8            )
    f.write(  "\n%s"   %l23_9            )
    f.write(  "\n%s"   %l23_10            )
    f.write(  "\n%s"   %l23_11            )
    f.write(  "\n%s"   %l23_12            )
    f.write(  "\n%s"   %l23_13            )
    f.write(  "\n%s"   %l23_14            )
    f.write(  "\n%s"   %l23_15            )
    f.write(  "\n%s"   %l23_16            )

    f.write(  "\n%s"   %w47_1            )
    f.write(  "\n%s"   %w47_2            )
    f.write(  "\n%s"   %w47_3            )
    f.write(  "\n%s"   %w47_4            )
    f.write(  "\n%s"   %w47_5            )
    f.write(  "\n%s"   %w47_6            )
    f.write(  "\n%s"   %w47_7            )
    f.write(  "\n%s"   %w47_8            )
    f.write(  "\n%s"   %w47_9            )
    f.write(  "\n%s"   %w47_10            )
    f.write(  "\n%s"   %w47_11            )
    f.write(  "\n%s"   %w47_12            )
    f.write(  "\n%s"   %w47_13            )
    f.write(  "\n%s"   %w47_14            )
    f.write(  "\n%s"   %w47_15            )
    f.write(  "\n%s"   %w47_16            )

    f.write(  "\n%s"   %w5_1            )
    f.write(  "\n%s"   %w5_2            )
    f.write(  "\n%s"   %w5_3            )
    f.write(  "\n%s"   %w5_4            )
    f.write(  "\n%s"   %w5_5            )
    f.write(  "\n%s"   %w5_6            )
    f.write(  "\n%s"   %w5_7            )
    f.write(  "\n%s"   %w5_8            )
    f.write(  "\n%s"   %w5_9            )
    f.write(  "\n%s"   %w5_10            )
    f.write(  "\n%s"   %w5_11            )
    f.write(  "\n%s"   %w5_12            )
    f.write(  "\n%s"   %w5_13            )
    f.write(  "\n%s"   %w5_14            )
    f.write(  "\n%s"   %w5_15            )
    f.write(  "\n%s"   %w5_16            )

    f.write(  "\n%s"   %l457_1            )
    f.write(  "\n%s"   %l457_2            )
    f.write(  "\n%s"   %l457_3            )
    f.write(  "\n%s"   %l457_4            )
    f.write(  "\n%s"   %l457_5            )
    f.write(  "\n%s"   %l457_6            )
    f.write(  "\n%s"   %l457_7            )
    f.write(  "\n%s"   %l457_8            )
    f.write(  "\n%s"   %l457_9            )
    f.write(  "\n%s"   %l457_10            )
    f.write(  "\n%s"   %l457_11            )
    f.write(  "\n%s"   %l457_12            )
    f.write(  "\n%s"   %l457_13            )
    f.write(  "\n%s"   %l457_14            )
    f.write(  "\n%s"   %l457_15            )
    f.write(  "\n%s"   %l457_16            )

    f.write(  "\n%s"   %w6_1            )
    f.write(  "\n%s"   %w6_2            )
    f.write(  "\n%s"   %w6_3            )
    f.write(  "\n%s"   %w6_4            )
    f.write(  "\n%s"   %w6_5            )
    f.write(  "\n%s"   %w6_6            )
    f.write(  "\n%s"   %w6_7            )
    f.write(  "\n%s"   %w6_8            )
    f.write(  "\n%s"   %w6_9            )
    f.write(  "\n%s"   %w6_10            )
    f.write(  "\n%s"   %w6_11            )
    f.write(  "\n%s"   %w6_12            )
    f.write(  "\n%s"   %w6_13            )
    f.write(  "\n%s"   %w6_14            )
    f.write(  "\n%s"   %w6_15            )
    f.write(  "\n%s"   %w6_16            )

    f.write(  "\n%s"   %l6_1            )
    f.write(  "\n%s"   %l6_2            )
    f.write(  "\n%s"   %l6_3            )
    f.write(  "\n%s"   %l6_4            )
    f.write(  "\n%s"   %l6_5            )
    f.write(  "\n%s"   %l6_6            )
    f.write(  "\n%s"   %l6_7            )
    f.write(  "\n%s"   %l6_8            )
    f.write(  "\n%s"   %l6_9            )
    f.write(  "\n%s"   %l6_10            )
    f.write(  "\n%s"   %l6_11            )
    f.write(  "\n%s"   %l6_12            )
    f.write(  "\n%s"   %l6_13            )
    f.write(  "\n%s"   %l6_14            )
    f.write(  "\n%s"   %l6_15            )
    f.write(  "\n%s"   %l6_16            )

    f.write(  "\n%s"   %Cc_1            )
    f.write(  "\n%s"   %Cc_2            )
    f.write(  "\n%s"   %Cc_3            )
    f.write(  "\n%s"   %Cc_4            )
    f.write(  "\n%s"   %Cc_5            )
    f.write(  "\n%s"   %Cc_6            )
    f.write(  "\n%s"   %Cc_7            )
    f.write(  "\n%s"   %Cc_8            )
    f.write(  "\n%s"   %Cc_9            )
    f.write(  "\n%s"   %Cc_10            )
    f.write(  "\n%s"   %Cc_11            )
    f.write(  "\n%s"   %Cc_12            )
    f.write(  "\n%s"   %Cc_13            )
    f.write(  "\n%s"   %Cc_14            )
    f.write(  "\n%s"   %Cc_15            )
    f.write(  "\n%s"   %Cc_16            )



    f.close()

    ### Processing and Collecting the Result ###
    ### ------------------------------------ ###
    os.system("ocean -nograph -restore APSO_opamp.ocn")

    param_result = np.zeros((particles, n_requirement))

    with open("./APSO_results_opamp.txt") as f1:
        param = f1.readlines()
        param = np.array(param)
        param = param.astype(float)  # Chuyển đổi dữ liệu từ dạng chuỗi sang float
        param_result[:len(param)] = param.reshape(particles, n_requirement)  # Gán dữ liệu từ param vào param_result

    Cond = np.zeros(particles).astype(float)
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

        if Cond[i] == 0:#Khi Opamp không đạt SAT (Condition = 0) thì sẽ trả về hàm mục tiêu fitness = -1
            fitness[i] = -1
        elif ( (PM[i] < 60.0) | (Gain[i] < 50.0) | (GBW[i] < 50) | (Power[i] > 250.0) | (CMRR[i] < 50) | (PSRR_p[i] < 50) | (PSRR_n[i] < 120) | (SR[i] < 50) ):
            fitness[i] = 0 #Khi Opamp đã SAT ta so sánh các giá trị ràng buộc, nếu không thỏa tất cả ràng buộc thì hàm mục tiêu fitness = 0
        else: #Khi Opamp đã Sat và pass các ràng buộc thì hàm mục tiêu fitness sẽ tính theo công thức đã cho
            fitness[i] =  ((GBW[i] *10e6 * Cload) * (math.tan(math.radians(PM[i]))/Tan_60))/(Itotal[i]*10e-6)

    return fitness,param_result

def sigmoid(x): #hàm signoid trong công thức cảu Binary PSO 
    return 1 / (1 + np.exp(-x))

def decode(bounds, n_bits, bitstring): #đứa giá trị nhi phân thành giá trị thập phân
    decoded = list()
    largest = 2 ** n_bits - 1  # Correct the largest value calculation
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        value_rounded = np.round(value, 2)
        decoded.append(value_rounded)
    return decoded

def create_folder_path(g, b, result):
    # Gets the current date in the format DD-MM-YYYY
    now = dt.date.today()
    month = now.strftime("%b")
    day = now.strftime("%d")
    year = now.strftime("%Y")
     # Đặt tên file theo thời gian hiện tại DD-MM-YY và các giá trị khai báo và kết quả fitness + tên giải thuật
    formatted_date = f"{day}-{month}-{year}-gamma={g}-beta={b}-result={result:.2f}"
    # Join two paths together
    path1 = os.getcwd()
    path2 = formatted_date
    joined_path = os.path.join(path1, path2)
    return joined_path

class Particle:#function khai báo giá trị đầu tiên của quần thể
    def __init__(self, n_bits, n_particles, n_dimensions, position_file):
         #khởi tạo vị trí ban đầu (ở đây là lấy các giá trị ở trong 1 file .txt được khai báo bên dưới và bình thưởng vị trí ban đầu sẽ ramdom trong bound)
        self.position = self.read_positions_from_file(position_file, n_particles, n_bits * n_dimensions)
        self.velocity = np.random.uniform(-1, 1, size=(n_particles, n_bits * n_dimensions))
        self.best_position = self.position.copy()
        self.fitness = -float('inf')
        self.best_fitness = -float('inf')
        self.A_fitness = np.zeros(n_particles)

    def read_positions_from_file(self, position_file, n_particles, n_dimensions): #đọc fie .txt có giá trị nhị phân cho vị trí ban đầu
        positions = np.zeros((n_particles, n_dimensions), dtype=int)
        with open(position_file, 'r') as file:
            for i, line in enumerate(file):
                if i >= n_particles:
                    break
                positions[i] = np.array([int(x) for x in line.split()])
        return positions

def update_velocity(particle, global_best_position , n_iter):# Cập nhật vận tốc và vị trí của các hạt
    #khai báo các hằng số gia tốc ampha, beta , gamma cho thuật toán
    Epsilon = np.random.rand(*particle.position.shape)
    Gamma = 0.5
    Ampha0 = 1
    Ampha = Ampha0*(Gamma**n_iter)
    Beta = 0.3
    # Tính toán vận tốc mới
    velocity = particle.velocity + Beta * (global_best_position - particle.position) + Ampha*Epsilon

    return velocity

def update_position(particle): # Cập nhật vị trí mới theo vị trí tốt nhất
    probabilities = sigmoid(particle.velocity) #tính vị trị mới theo công thứ Signiod
    new_position = np.where(np.random.rand(*probabilities.shape) < probabilities, 1, 0)

    return new_position
    
def pso(bounds, n_particles, n_bits, n_dimensions ,n_requirement, n_iter, df):
    #Khởi tạo vị trí và vận tốc ban đầu cho các hạt
    particles = [Particle(n_bits, n_particles, n_dimensions,position_file) for _ in range(1)]
    #khai bao hàm
    global_best_fitness = -float('inf')
    global_best_position = None

    for i in range(n_iter):#số lần lặp
        begin_time = dt.datetime.now().strftime("%H:%M:%S")
        for particle in particles:
            decoded_positions = np.array([decode(bounds, n_bits, p) for p in particle.position]) #gọi vị trí đầu theo nhị phân qua hàm decoder 
            particle.A_fitness,param_result = calFitness(decoded_positions,n_particles,n_requirement)
            
            particle.fitness = np.max(particle.A_fitness) #tìm giá trị lớn nhất trong chuỗi fitness
            print("array fitness" ,i, " :",particle.A_fitness)

            print("position: ", particle.position)
            if particle.fitness > particle.best_fitness: #Nếu best fitness mới > best fitness cũ thì sẽ thay đổi
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness
            
            if particle.best_fitness > global_best_fitness:#Gọi cập nhập lại vị trí và vận tốc mới cho vòng lặp tiếp theo
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position.copy()
        
        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_position , i)
            particle.position = update_position(particle)

        results_xlsx = np.zeros(n_requirement)
        print("Array requiriment :",results_xlsx)
        print("global_best_fitness: (",i,") :", global_best_fitness)
        max_row_index = np.argmax(particle.A_fitness)
        global_best_chrom_params = decoded_positions[max_row_index]
        print("global_best_chrom_params: (",i,") :", global_best_chrom_params)
        for t in range(n_requirement): #giá trị được in trong excel
            results_xlsx[t] = param_result[max_row_index,t]
        new_row = [
            {
                "Lan chay": i+1,
                "Begin": begin_time,
                "End": dt.datetime.now().strftime("%H:%M:%S"),
                "Time (s)" : (dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"),'%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S')).total_seconds(),
                "Global Best Fitness"  : global_best_fitness,
                "Fitness" : np.max(particle.A_fitness),
                "Condition" : results_xlsx[0],
                "W01 (um)"  : global_best_chrom_params[0],
                "L01 (um)"  : global_best_chrom_params[1],
                "W23 (um)"  : global_best_chrom_params[2],
                "L23 (um)"  : global_best_chrom_params[3],
                "W47 (um)"  : global_best_chrom_params[4],
                "W5 (um)"  : global_best_chrom_params[5],
                "L457 (um)"  : global_best_chrom_params[6],
                "W6 (um)"  : global_best_chrom_params[7],
                "L6 (um)"  : global_best_chrom_params[8],
                "Cc (pF)"  : global_best_chrom_params[9],
                "PM (degree)" : results_xlsx[1],
                "DC Gain (dB)" : results_xlsx[2],
                "CMRR (dB)" : results_xlsx[3],
                "GBW (MHz)" : results_xlsx[4],
                "Power (uW)" : results_xlsx[5],
                "PSRR+ (dB)" : results_xlsx[6],
                "PSRR- (dB)" : results_xlsx[7],
                "Slew Race (V/us)" : results_xlsx[8],
            }]
        df = df.append(new_row, ignore_index = True, sort = False)

    
    return [global_best_chrom_params, global_best_fitness, df]

# Chạy thử với các tham số tùy chọn
if __name__ == "__main__":
    bounds = np.array([(0.85, 4), (0.23, 0.4), (0.7, 1), (0.06, 0.4), (2, 2.8), (20, 23), (0.1, 1), (16, 22), (0.25, 0.5), (0.3, 1)])  # Giới hạn cho từng biến
    position_file = 'Binary_APSO_first_params.txt'
    n_bits = 16 #số bit
    n_particles = 16  # Số lượng hạt trong quần thể
    n_dimensions = 10 # Số lượng biến
    n_iter = 10 #số vòng lặp
    n_requirement = 9 #số lượng ràng buộc
    u = 1e-6
    p = 1e-12
    
    column_names = ["Lan chay", "Begin", "End", "Time (s)","Global Best Fitness", "Fitness", "Condition", "W01 (um)", "L01 (um)", "W23 (um)", "L23 (um)" \
                    , "W47 (um)", "W5 (um)","L457 (um)", "W6 (um)", "L6 (um)", "Cc (pF)","PM (degree)", "DC Gain (dB)", "CMRR (dB)", "GBW (MHz)" \
                    ,"Power (uW)", "PSRR+ (dB)", "PSRR- (dB)", "Slew Race (V/us)"]
    
    df = pd.DataFrame(columns=column_names)
    positions,Best_fitness,df = pso(bounds, n_particles, n_bits, n_dimensions,n_requirement,n_iter,df)
    print('Done!')
    print(df)

    print("Best solution:", positions)
    print("Best fitness: ",Best_fitness)
    
    #giá trị in ra tên file folder cho dễ quan sát
    g = 0.5
    b = 0.3
    path = create_folder_path(g, b, Best_fitness)#tạo thư mục

    index = 1
    while os.path.exists(path): #hàm để các thư mục không bị trùng tên
        path = f"{create_folder_path(g,b,Best_fitness)}({index})"
        index += 1
    os.mkdir(path)
    # In kết quả tốt nhất
    # copy all the OCEAN Scripts to the created folder
    df.to_excel("./APSO_opamp_pop16_iter100.xlsx", sheet_name = "APSO_OPAMP", index = False)
    df.to_csv("./APSO_result.csv")

    #Các file copy vào trong thư mục
    shutil.copy2("./APSO_results_opamp.txt", path)
    shutil.copy2("./APSO_opamp.py", path)
    shutil.copy2("./APSO_params_opamp.txt", path)
    shutil.copy2("./APSO_opamp_pop16_iter100.xlsx",path)
    shutil.copy2("./APSO_result.csv",path)
    

   
    # change the current working directory to created folder’s directory
    os.chdir(path)
    

   

    