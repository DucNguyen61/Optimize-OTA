from numpy.random import randint
from numpy.random import rand
import pandas as pd
import datetime as dt
import os
import numpy as np
import shutil
def calFitness(position, particles, dimension):
    # Đây chỉ là một hàm giả định, bạn cần thay thế bằng hàm tính toán fitness thực tế của mình
    print("position " ,position)
    fitness = np.zeros(particles)

    fitness = np.sum(position,axis=1)

    Best_fitness = np.min(fitness)

    return Best_fitness

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

class Particle:
    def __init__(self, bounds, n_particles, n_dimensions):
        self.fisrt_param_result = np.zeros((n_particles, n_dimensions))
        with open("./APSO_first_param.txt") as f2:
            fisrt_param = f2.readlines()
            fisrt_param = np.array(fisrt_param)
            for i in range(n_particles):
                for j in range(n_dimensions):
                    self.fisrt_param_result[i,j] = fisrt_param[i + j*n_particles] 



        self.position = self.fisrt_param_result
        self.velocity = np.random.uniform(-1, 1, size=(n_particles, n_dimensions))
        self.best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')


def update_velocity(particle, global_best_position):
    r1 = np.random.rand()
    r2 = np.random.rand()
    w = 0.5
    c1 = 1.5
    c2 = 1.5

    velocity = w * particle.velocity + c1 * r1  * (particle.best_position - particle.position) + c2 * r2 * (global_best_position - particle.position)

    return velocity 

def update_position(particle, bounds):
    new_position = particle.position + particle.velocity
    new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])

    return new_position
    
def pso(bounds, n_particles, n_dimensions , n_iter, df):

    particles = [Particle(bounds, n_particles, n_dimensions)]
    global_best_fitness = float('inf')
    global_best_position = None

    for i in range(n_iter):
        begin_time = dt.datetime.now().strftime("%H:%M:%S")
        for particle in particles:
            particle.fitness = calFitness(particle.position,n_particles,n_dimensions)
            print("position: ", particle.position)
            if particle.fitness < particle.best_fitness:
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness
            
            if particle.best_fitness < global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position.copy()
        
        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_position)
            particle.position = update_position(particle, bounds)


        print("global_best_fitness: (",i,") :", global_best_fitness)
        min_row_index = np.argmin(np.sum(global_best_position, axis=1))
        global_best_chrom_params = particle.position[min_row_index]
        print("global_best_chrom_params: (",i,") :", global_best_chrom_params )
        new_row = [
            {
                "Lan chay": i+1,
                "Begin": begin_time,
                "End": dt.datetime.now().strftime("%H:%M:%S"),
                "Time (s)" : (dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"),'%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S')).total_seconds(),
                "Fitness"  : global_best_fitness,
                "W01 (um)"  : global_best_chrom_params[0],
                "L01 (um)"  : global_best_chrom_params[1],
                "C01 (um)"  : global_best_chrom_params[2],
            }]
        df = df._append(new_row, ignore_index = True)

    
    return [global_best_chrom_params, global_best_fitness, df]

# Chạy thử với các tham số tùy chọn
if __name__ == "__main__":
    bounds = np.array([(0.85, 4), (0.23, 0.4), (0.7, 1)])  # Phạm vi của các biến
    n_particles = 16  # Số lượng hạt trong quần thể
    n_dimensions = 3  # Số lượng biến
    n_iter = 10

    
    column_names = ["Lan chay", "Begin", "End", "Time (s)", "Fitness", "W01 (um)", "L01 (um)", "C01 (um)"]
    df = pd.DataFrame(columns=column_names)
    positions,Best_fitness,df = pso(bounds, n_particles, n_dimensions,n_iter,df)
    print('Done!')
    print(df)

    print("Best solution:", positions)
    print("Best fitness: ",Best_fitness)

    path = create_folder_path()
    index = 1
    while os.path.exists(path):
        path = f"{create_folder_path()}({index})"
        index += 1
    os.mkdir(path)
    # In kết quả tốt nhất
    # copy all the OCEAN Scripts to the created folder
    df.to_excel("./APSOmName_opamp_pop16_iter100.xlsx", sheet_name = "AlgorithmName_OPAMP", index = False)
    df.to_csv("./Optimize_result.csv")


    shutil.copy2("./FoM_result.txt", path)
    shutil.copy2("./run.py", path)
    shutil.copy2("./APSO_thongso.txt", path)
    shutil.copy2("./APSOmName_opamp_pop16_iter100.xlsx",path)
    

   
    # change the current working directory to created folder’s directory
    os.chdir(path)
    

   

    