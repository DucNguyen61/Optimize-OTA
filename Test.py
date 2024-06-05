from numpy.random import randint, rand
import pandas as pd
import datetime as dt
import numpy as np
import shutil

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calFitness(decoded_positions):
    fitness = np.sum(decoded_positions, axis=1)
    Best_fitness = np.max(fitness)
    return Best_fitness

def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        value_rounded = np.round(value, 2)
        decoded.append(value_rounded)
    return decoded

class Particle:
    def __init__(self, n_particles, n_bits, n_dimensions):
        self.position = np.random.randint(2, size=(n_particles, n_bits * n_dimensions))
        self.velocity = np.random.uniform(-1, 1, size=(n_particles, n_bits * n_dimensions))
        self.best_position = self.position.copy()
        self.fitness = float('-inf')
        self.best_fitness = float('-inf')

def update_velocity(particle, global_best_position):
    r1 = np.random.rand(*particle.position.shape)
    r2 = np.random.rand(*particle.position.shape)
    w = 0.5
    c1 = 1.5
    c2 = 1.5
    velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (global_best_position - particle.position)
    return velocity 

def update_position(particle):
    probabilities = sigmoid(particle.velocity)
    new_position = np.where(np.random.rand(*probabilities.shape) < probabilities, 1, 0)
    return new_position

def pso(bounds, n_particles, n_bits, n_dimensions, n_iter, df):
    particles = [Particle(n_particles, n_bits, n_dimensions)]
    global_best_fitness = float('-inf')
    global_best_position = None

    for i in range(n_iter):
        begin_time = dt.datetime.now().strftime("%H:%M:%S")
        for particle in particles:
            decoded_positions = np.array([decode(bounds, n_bits, p) for p in particle.position])
            particle.fitness = calFitness(decoded_positions)
            if particle.fitness > particle.best_fitness:
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness
            
            if particle.best_fitness > global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position.copy()
        
        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_position)
            particle.position = update_position(particle)

        print("global_best_fitness: (", i, ") :", global_best_fitness)
        best_row_index = np.argmax(np.sum(decoded_positions, axis=1))
        global_best_chrom_params = decoded_positions[best_row_index]
        print("global_best_chrom_params: (", i, ") :", global_best_chrom_params)
        new_row = [
            {
                "Lan chay": i + 1,
                "Begin": begin_time,
                "End": dt.datetime.now().strftime("%H:%M:%S"),
                "Fitness": global_best_fitness,
                "W01 (um)": global_best_chrom_params[0],
                "L01 (um)": global_best_chrom_params[1],
                "C01 (um)": global_best_chrom_params[2],
            }]
        df = df._append(new_row, ignore_index=True)

    return [global_best_chrom_params, global_best_fitness, df]

# Chạy thử với các tham số tùy chọn
if __name__ == "__main__":
    bounds = np.array([(0.85, 4), (0.23, 0.4), (0.7, 1)])  # Phạm vi của các biến
    n_bits = 8  # Số bit cho mỗi biến
    n_particles = 16  # Số lượng hạt trong quần thể
    n_dimensions = 3  # Số lượng biến
    n_iter = 10

    column_names = ["Lan chay", "Begin", "End", "Fitness", "W01 (um)", "L01 (um)", "C01 (um)"]
    df = pd.DataFrame(columns=column_names)
    positions, Best_fitness, df = pso(bounds, n_particles, n_bits, n_dimensions, n_iter, df)
    print('Done!')
    print(df)

    print("Best solution:", positions)
    print("Best fitness: ", Best_fitness)

    df.to_excel("./APSO_8bit_pop16_iter10.xlsx", sheet_name="BinaryPSO_8bit", index=False)
    df.to_csv("./Optimize_result_8bit.csv")
