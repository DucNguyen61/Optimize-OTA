#including library
import random
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np
import shutil
import math

def calFitness(sol2,flag):
    sol2 = np.array(sol2)
    sizePos = np.prod(sol2.shape)
    sol2 = np.full((16,9), 2, dtype='f')

    if sizePos == 9:
        sol[0,:] = sol2
    else:
        sol = sol2
    #Attach values to variables#
    w01_1 = sol[0,0]
    w01_2 = sol[1,0]
    w01_3 = sol[2,0]
    w01_4 = sol[3,0]
    w01_5 = sol[4,0]
    w01_6 = sol[5,0]
    w01_7 = sol[6,0]
    w01_8 = sol[7,0]
    w01_9 = sol[8,0]
    w01_10 = sol[9,0]
    w01_11 = sol[10,0]
    w01_12 = sol[11,0]
    w01_13 = sol[12,0]
    w01_14 = sol[13,0]
    w01_15 = sol[14,0]
    w01_16 = sol[15,0]

    l01_1 = sol[0,1]
    l01_2 = sol[1,1]
    l01_3 = sol[2,1]
    l01_4 = sol[3,1]
    l01_5 = sol[4,1]
    l01_6 = sol[5,1]
    l01_7 = sol[6,1]
    l01_8 = sol[7,1]
    l01_9 = sol[8,1]
    l01_10 = sol[9,1]
    l01_11 = sol[10,1]
    l01_12 = sol[11,1]
    l01_13 = sol[12,1]
    l01_14 = sol[13,1]
    l01_15 = sol[14,1]
    l01_16 = sol[15,1]

    w23_1 = sol[1,2]
    w23_2 = sol[2,2]
    w23_3 = sol[3,2]
    w23_4 = sol[4,2]
    w23_5 = sol[5,2]
    w23_6 = sol[6,2]
    w23_7 = sol[7,2]
    w23_8 = sol[8,2]
    w23_9 = sol[9,2]
    w23_10 = sol[10,2]
    w23_11 = sol[11,2]
    w23_12 = sol[12,2]
    w23_13 = sol[13,2]
    w23_14 = sol[14,2]
    w23_15 = sol[15,2]
    w23_16 = sol[16,2]

    l23_1 = sol[0,3]
    l23_2 = sol[1,3]
    l23_3 = sol[2,3]
    l23_4 = sol[3,3]
    l23_5 = sol[4,3]
    l23_6 = sol[5,3]
    l23_7 = sol[6,3]
    l23_8 = sol[7,3]
    l23_9 = sol[8,3]
    l23_10 = sol[9,3]
    l23_11 = sol[10,3]
    l23_12 = sol[11,3]
    l23_13 = sol[12,3]
    l23_14 = sol[13,3]
    l23_15 = sol[14,3]
    l23_16 = sol[15,3]

    w47_1 = sol[0,4]
    w47_2 = sol[1,4]
    w47_3 = sol[2,4]
    w47_4 = sol[3,4]
    w47_5 = sol[4,4]
    w47_6 = sol[5,4]
    w47_7 = sol[6,4]
    w47_8 = sol[7,4]
    w47_9 = sol[8,4]
    w47_10 = sol[9,4]
    w47_11 = sol[10,4]
    w47_12 = sol[11,4]
    w47_13 = sol[12,4]
    w47_14 = sol[13,4]
    w47_15 = sol[14,4]
    w47_16 = sol[15,4]

    w5_1 = sol[0,5]
    w5_2 = sol[1,5]
    w5_3 = sol[2,5]
    w5_4 = sol[3,5]
    w5_5 = sol[4,5]
    w5_6 = sol[5,5]
    w5_7 = sol[6,5]
    w5_8 = sol[7,5]
    w5_9 = sol[8,5]
    w5_10 = sol[9,5]
    w5_11 = sol[10,5]
    w5_12 = sol[11,5]
    w5_13 = sol[12,5]
    w5_14 = sol[13,5]
    w5_15 = sol[14,5]
    w5_16 = sol[15,5]
    
    l457_1 = sol[0,6]
    l457_2 = sol[1,6]
    l457_3 = sol[2,6]
    l457_4 = sol[3,6]
    l457_5 = sol[4,6]
    l457_6 = sol[5,6]
    l457_7 = sol[6,6]
    l457_8 = sol[7,6]
    l457_9 = sol[8,6]
    l457_10 = sol[9,6]
    l457_11 = sol[10,6]
    l457_12 = sol[11,6]
    l457_13 = sol[12,6]
    l457_14 = sol[13,6]
    l457_15 = sol[14,6]
    l457_16 = sol[15,6]

    w6_1 = sol[0,7]
    w6_2 = sol[1,7]
    w6_3 = sol[2,7]
    w6_4 = sol[3,7]
    w6_5 = sol[4,7]
    w6_6 = sol[5,7]
    w6_7 = sol[6,7]
    w6_8 = sol[7,7]
    w6_9 = sol[8,7]
    w6_10 = sol[9,7]
    w6_11 = sol[10,7]
    w6_12 = sol[11,7]
    w6_13 = sol[12,7]
    w6_14 = sol[13,7]
    w6_15 = sol[14,7]
    w6_16 = sol[15,7]

    l6_1 = sol[0,8]
    l6_2 = sol[1,8]
    l6_3 = sol[2,8]
    l6_4 = sol[3,8]
    l6_5 = sol[4,8]
    l6_6 = sol[5,8]
    l6_7 = sol[6,8]
    l6_8 = sol[7,8]
    l6_9 = sol[8,8]
    l6_10 = sol[9,8]
    l6_11 = sol[10,8]
    l6_12 = sol[11,8]
    l6_13 = sol[12,8]
    l6_14 = sol[13,8]
    l6_15 = sol[14,8]
    l6_16 = sol[15,8]

    Cc_var = 1e-12


    ### Attach parameters to OCEAN File ### 
    filethongso = ("./APSO_thongso.txt") 
    f = open(filethongso, 'w')

    f.write(  "\n%s"   %w01_1            )
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

    f.write(  "\n%s"   %Cc_var            )


    f.close()

    ### Processing and Collecting the Result ###
    ### ------------------------------------ ###
    os.system("ocean -nograph -restore test_2_circuit.ocn")
    param_result = np.zeros((16, 5))

    with open("./FoM_result.txt") as f1:
        param = f1.readlines()
        param = np.array(param)
        param = param.astype(float)  # Chuyển đổi dữ liệu từ dạng chuỗi sang float
        param_result[:len(param)] = param.reshape(16, 5)  # Gán dữ liệu từ param vào param_result

    Cond = np.zeros(16)
    PM = np.zeros(16)
    Gain = np.zeros(16)
    GBW = np.zeros(16)
    Power = np.zeros(16)
    Itotal = np.zeros(16)
    Tan_60 = math.tan(math.radians(60))
    Cload = 1 * 1e-12
    fitness = np.zeros(16)

    for i in range(16):
        Cond[i] = param_result[i,0]
        PM[i] = param_result[i,1]
        Gain[i] = param_result[i,2]
        GBW[i] = param_result[i,3]
        Power[i] = param_result[i,4]
        Itotal[i] = Power[i]/1.2


    for i in range(16): 
        fitness[i] =  ((GBW[i] * Cload) * (PM[i]/Tan_60))/Itotal[i]

    # Ở đây, ta giả định rằng flag = 1 sẽ trả về một giá trị fitness ngẫu nhiên cho mỗi cá thể,
    # và flag = 0 sẽ tính toán giá trị fitness thực tế dựa trên giá trị giải mã của các biến
    fitness =  fitness.flatten()
    if flag == 1:
        fitness2 = fitness[0]
    else:
        # Giả sử ta chỉ trả về giá trị fitness của cá thể đầu tiên trong decoded
        fitness2 = fitness.tolist()
    return fitness2   
# function for create folder path

def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):                             
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits       
        substring = bitstring[start:end]                     
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])         
        # convert string to integer
        integer = int(chars, 2)                              
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded

def create_folder_path():
    # Gets the current date in the format DD-MM-YYYY
    now = dt.date.today()
    month = now.strftime("%b")
    day = now.strftime("%d")
    year = now.strftime("%Y")
    formatted_date = f"{day}-{month}-{year}"
    # Join two paths together
    path1 = os.getcwd()
    path2 = formatted_date
    joined_path = os.path.join(path1,path2)
    return joined_path

#--------------------Main Algorithm--------------------#
def pso(bounds, n_particles, n_dimensions, n_iter, df):
    # Khởi tạo vị trí và vận tốc ban đầu cho các hạt
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, n_dimensions))
    velocities = np.zeros((n_particles, n_dimensions))
    
    # Khởi tạo vị trí và giá trị tốt nhất cho cả quần thể
    best_positions = positions.copy()
    best_fitness = np.array([calFitness(pos, 0)[0] for pos in best_positions])
    
    # Lặp qua các thế hệ
    for _ in range(n_iter):
        # Cập nhật vận tốc và vị trí của các hạt
        inertia_weight = 0.5
        cognitive_weight = 0.5
        social_weight = 0.5
        
        # Tính toán vận tốc mới
        rp = np.random.rand(n_particles, n_dimensions)
        rg = np.random.rand(n_particles, n_dimensions)
        
        velocities = inertia_weight * velocities + \
                     cognitive_weight * rp * (best_positions - positions) + \
                     social_weight * rg * (best_positions[np.argmin(best_fitness)] - positions)
        
        # Cập nhật vị trí mới
        positions += velocities
        
        # Tính toán giá trị fitness cho các vị trí mới
        current_fitness = np.array([calFitness(pos, 0)[0] for pos in positions])
        
        # Cập nhật vị trí và giá trị tốt nhất cho cả quần thể
        mask = current_fitness < best_fitness
        best_positions[mask] = positions[mask]
        best_fitness[mask] = current_fitness[mask]
    
        # Ghi lại kết quả
        new_row = {
            "Lan chay": _,
            "Begin": dt.datetime.now().strftime("%H:%M:%S"),
            "End": dt.datetime.now().strftime("%H:%M:%S"),
            "Time (s)": 0,
            "Fitness": np.min(best_fitness),
            "W0 (um)": 1,
            "W1 (um)": best_positions[np.argmin(best_fitness)][0],
            "W2 (um)": best_positions[np.argmin(best_fitness)][1],
            "W3 (um)": best_positions[np.argmin(best_fitness)][2],
            "W4 (um)": 0,
            "W5 (um)": 0,
            "W6 (um)": 0,
            "W7 (um)": 0,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Trả về vị trí và giá trị tốt nhất sau khi tối ưu
    best_solution = best_positions[np.argmin(best_fitness)]
    best_fitness_value = np.min(best_fitness)
    return [best_solution, best_fitness_value, df]

if __name__ == "__main__":
    # Các thông số và tham số cho thuật toán PSO
    bounds = np.array([(0.85, 4), (0.23, 0.4), (0.7, 1), (0.06, 0.4), (2, 2.8), (18, 23), (0.1, 1), (16, 22), (0.25, 0.5)])  # Giới hạn cho từng biến
    n_particles = 16  # Số hạt trong bầy đàn
    n_dimensions = 9  # Số chiều của không gian tìm kiếm
    n_iter = 10  # Số thế hệ
    df = pd.DataFrame(columns=["Lan chay", "Begin", "End", "Time (s)", "Fitness", "W0 (um)", "W1 (um)", "W2 (um)", "W3 (um)", "W4 (um)", "W5 (um)", "W6 (um)", "W7 (um)"])  # DataFrame để lưu trữ kết quả

    # Chạy thuật toán PSO
    best_solution, best_fitness, df = pso(bounds, n_particles, n_dimensions, n_iter, df)

    index = 1
    while os.path.exists(path):
        path = f"{create_folder_path()}({index})"
        index += 1
    os.mkdir(path)
    # In kết quả tốt nhất
    # copy all the OCEAN Scripts to the created folder
    shutil.copy2("./FoM_result.txt", path)
    shutil.copy2("./run.py", path)
    shutil.copy2("./APSO_thongso.txt", path)
    

   
    # change the current working directory to created folder’s directory
    os.chdir(path)

    # In kết quả tốt nhất
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
    print("Dataframe:", df)