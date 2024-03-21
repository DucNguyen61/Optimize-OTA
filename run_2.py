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

def Fun(D, sol):

    #Attach values to variables#
    w01_1 = 4e-06
    w01_2 = 4e-06
    w01_3 = 4e-06
    w01_4 = 4e-06
    w01_5 = 4e-06
    w01_6 = 4e-06
    w01_7 = 4e-06
    w01_8 = 4e-06
    w01_9 = 4e-06
    w01_10 = 4e-06
    w01_11 = 4e-06
    w01_12 = 4e-06
    w01_13 = 4e-06
    w01_14 = 4e-06
    w01_15 = 4e-06
    w01_16 = 4e-06

    l01_1 = 4e-07
    l01_2 = 4e-07
    l01_3 = 4e-07
    l01_4 = 4e-07
    l01_5 = 4e-07
    l01_6 = 4e-07
    l01_7 = 4e-07
    l01_8 = 4e-07
    l01_9 = 4e-07
    l01_10 = 4e-07
    l01_11 = 4e-07
    l01_12 = 4e-07
    l01_13 = 4e-07
    l01_14 = 4e-07
    l01_15 = 4e-07
    l01_16 = 4e-07

    w23_1 = 1e-06
    w23_2 = 1e-06
    w23_3 = 1e-06
    w23_4 = 1e-06
    w23_5 = 1e-06
    w23_6 = 1e-06
    w23_7 = 1e-06
    w23_8 = 1e-06
    w23_9 = 1e-06
    w23_10 = 1e-06
    w23_11 = 1e-06
    w23_12 = 1e-06
    w23_13 = 1e-06
    w23_14 = 1e-06
    w23_15 = 1e-06
    w23_16 = 1e-06

    l23_1 = 4e-07
    l23_2 = 4e-07
    l23_3 = 4e-07
    l23_4 = 4e-07
    l23_5 = 4e-07
    l23_6 = 4e-07
    l23_7 = 4e-07
    l23_8 = 4e-07
    l23_9 = 4e-07
    l23_10 = 4e-07
    l23_11 = 4e-07
    l23_12 = 4e-07
    l23_13 = 4e-07
    l23_14 = 4e-07
    l23_15 = 4e-07
    l23_16 = 4e-07

    w47_1 = 2.8e-06
    w47_2 = 2.8e-06
    w47_3 = 2.8e-06
    w47_4 = 2.8e-06
    w47_5 = 2.8e-06
    w47_6 = 2.8e-06
    w47_7 = 2.8e-06
    w47_8 = 2.8e-06
    w47_9 = 2.8e-06
    w47_10 = 2.8e-06
    w47_11 = 2.8e-06
    w47_12 = 2.8e-06
    w47_13 = 2.8e-06
    w47_14 = 2.8e-06
    w47_15 = 2.8e-06
    w47_16 = 2.8e-06

    w5_1 = 2.3e-05
    w5_2 = 2.3e-05
    w5_3 = 2.3e-05
    w5_4 = 2.3e-05
    w5_5 = 2.3e-05
    w5_6 = 2.3e-05
    w5_7 = 2.3e-05
    w5_8 = 2.3e-05
    w5_9 = 2.3e-05
    w5_10 = 2.3e-05
    w5_11 = 2.3e-05
    w5_12 = 2.3e-05
    w5_13 = 2.3e-05
    w5_14 = 2.3e-05
    w5_15 = 2.3e-05
    w5_16 = 2.3e-05
    
    l457_1 = 1e-06
    l457_2 = 1e-06
    l457_3 = 1e-06
    l457_4 = 1e-06
    l457_5 = 1e-06
    l457_6 = 1e-06
    l457_7 = 1e-06
    l457_8 = 1e-06
    l457_9 = 1e-06
    l457_10 = 1e-06
    l457_11 = 1e-06
    l457_12 = 1e-06
    l457_13 = 1e-06
    l457_14 = 1e-06
    l457_15 = 1e-06
    l457_16 = 1e-06

    w6_1 = 2.2e-05
    w6_2 = 2.2e-05
    w6_3 = 2.2e-05
    w6_4 = 2.2e-05
    w6_5 = 2.2e-05
    w6_6 = 2.2e-05
    w6_7 = 2.2e-05
    w6_8 = 2.2e-05
    w6_9 = 2.2e-05
    w6_10 = 2.2e-05
    w6_11 = 2.2e-05
    w6_12 = 2.2e-05
    w6_13 = 2.2e-05
    w6_14 = 2.2e-05
    w6_15 = 2.2e-05
    w6_16 = 2.2e-05

    l6_1 = 5e-07
    l6_2 = 5e-07
    l6_3 = 5e-07
    l6_4 = 5e-07
    l6_5 = 5e-07
    l6_6 = 5e-07
    l6_7 = 5e-07
    l6_8 = 5e-07
    l6_9 = 5e-07
    l6_10 = 5e-07
    l6_11 = 5e-07
    l6_12 = 5e-07
    l6_13 = 5e-07
    l6_14 = 5e-07
    l6_15 = 5e-07
    l6_16 = 5e-07

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
    param_result = np.array([])
    with open("./FoM_result.txt") as f1:
        param = f1.readlines()
        param = np.array(param)
        param_result = param.astype(float)

    ### Reform the result ###

    Power = [0] * 16
    Gain = [0] * 16
    GBW = [0] * 16
    PM = [0] * 16
    FoM = [0] * 16
    Tan_60 = math.tan(math.radians(60))
    Cload = 10 * 1e-12

    for i in range(D*3): 
        if i <= 19: 
            Gain[i] = param_result[i]
        elif 19 < i <= 39:
            GBW[i-20] = param_result[i]
        else: 
            PM[i-40] = math.tan(math.radians(param_result[i]))

    ### Calculate FoM function ### 

    for i in range(D): 
        FoM[i] =  ((GBW[i] * Cload) * (PM[i]/Tan_60))/Itt_result[i]

    return FoM, Gain

# function for create folder path
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
class BatAlgorithm():
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, bounds, function, df):
        self.D = D  #dimension of solutions
        self.NP = NP  #population size 
        self.N_Gen = N_Gen  #generations/Iterations
        self.A = A  #loudness
        self.r = r  #pulse rate
        self.Qmin = Qmin  #frequency min
        self.Qmax = Qmax  #frequency max
        
        self.FoM_max = [0.0]  #current max value fitness
        self.GainDC = [0] * self.NP #Gain DC of circuit
        self.GainofBest = [0.0] #Gain DC of the best solution

        self.Lb = bounds[0,:]   #lower bound
        self.Ub = bounds[1,:]   #upper bound
        self.Q = [0] * self.NP  #frequency for each Bat

        self.v = [[0 for i in range(self.D)] for j in range(self.NP)]  #velocity
        self.Sol = [[0 for i in range(self.D)] for j in range(self.NP)]  #population of solutions/main result of Algorithm
        self.Fitness = [0] * self.NP  #fitness FoM
        self.best = [0] * self.D  #best solution for Width of MOS devices of 1 circuit
        self.Fun = function #function use for finding fitness value
        self.df = df #Function use for output display

    #function find best fitness value
    def best_bat(self):
        i = 0
        j = 0
        for i in range(self.NP):
            if (self.Fitness[i] > self.Fitness[j]) and (self.GainDC[i] >= 30):
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.FoM_max = self.Fitness[j]
        self.GainofBest = self.GainDC[j]

    #function initiate the input value
    def init_bat(self):
        for i in range(self.NP):
            self.Q[i] = 0
            for j in range(self.D):
                rnd = np.random.uniform(0, 1)
                self.v[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]*3) * rnd
        self.Fitness, self.GainDC = self.Fun(self.NP, self.Sol)
        self.best_bat()

    def simplebounds(self, val, lower, upper):
        rnd = np.random.uniform(0, 1)
        if val < lower:
            val = lower + (upper - lower*5) * rnd
        if val > upper:
            val = lower + (upper - lower) * rnd
        return val

    def move_bat(self):
        #a temperature array for result of BAT algorithm
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)] 
        #a temperature vector of best fitness value
        Fnew = [0.0] * self.NP  
        #a temperature vector of Gain each iteration
        Gainew = [0.0] * self.NP 

        #initial value for the first iteration
        self.init_bat() 

        #loop for iterations
        for t in range(self.N_Gen):    
            begin_time = dt.datetime.now().strftime("%H:%M:%S")

            #loop for each Bats
            for i in range(self.NP):   
                rnd = np.random.uniform(0, 1)

                #Initial Frequency for each bats at current iteration
                self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * rnd 

                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] - self.best[j]) * self.Q[i]    
                    S[i][j] = self.Sol[i][j] + self.v[i][j]                                   
                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j])              

                rnd = np.random.random_sample()

                #fly randomly based on pulse rate and best current result (algorithm theory)
                if rnd > self.r:       
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.1 * random.gauss(0, 1)
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j])
            
            #calculate new fitness in this iteration of all bats  
            Fnew, Gainew = self.Fun(self.NP, S) 

            #change fitness if the conditions of new fitness or loudness are appropriate
            for i in range(self.NP):
                rnd = np.random.random_sample()             
                if ((Fnew[i] > self.Fitness[i]) and (Gainew[i] >= 30)) or (rnd < self.A):   
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew[i]
                    self.GainDC[i] = Gainew[i]

            #change the best fitness value and result in algorithm
            for i in range(self.NP):
                if (self.Fitness[i] > self.FoM_max) and (self.GainDC[i] >= 30):        
                    for j in range(self.D):
                        self.best[j] = self.Sol[i][j]
                    self.FoM_max = self.Fitness[i]
                    self.GainofBest = self.GainDC[i]

            new_row = [
                {
                    "Lan chay": t+1,
                    "Begin": begin_time,
                    "End": dt.datetime.now().strftime("%H:%M:%S"),
                    "Time (s)": (dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"),'%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S')).total_seconds(),
                    "Fitness" : self.FoM_max,
                    "Wa (um)" : self.best[0],
                    "Wb (um)" : self.best[1],
                    "Wc (um)" : self.best[2],
                    "Wd (um)" : self.best[2],
                    "We (um)" : self.best[3],
                    "Wf (um)" : self.best[4],
                    "Gain (dB)" : self.GainofBest, 
                }]
            self.df = self.df.append(new_row, ignore_index = True)
            print(self.df)

        return self.df



###-------main program--------###
if __name__ == "__main__":
    #define range for each Width of MOS device
    bounds = [[0.85, 4], [0.23, 0.4], [0.7, 1], [0.06, 0.4], [2, 2.8], [18, 23], [0.1, 1], [16, 22], [0.25, 0.5]]

    #creat folder to contain exported files
    path = create_folder_path()
    os.mkdir(path)

    #copy all the Ocean script to created folder
    shutil.copy2("./FoM.ocn", path)
    shutil.copy2("./Itotal.ocn", path)

    #change the director now working to the created folder
    os.chdir(path)

    #set up for output
    columns_name = ["Lan chay", "Begin", "End", "Time (s)", "Fitness","Wa (um)", 
                    "Wb (um)", "Wc (um)", "Wd (um)", "We (um)", "Wf (um)", "Gain (dB)"]
    df = pd.DataFrame(columns=columns_name)

    #call algorithm
    Algorithm = BatAlgorithm(5, 20, 100, 0.4, 0.5, 0.5, 1.5, bounds, Fun, df)
    df = Algorithm.move_bat()

    #output
    print('Done!')

    df.to_csv("./Optimize_result.csv")
    df.to_excel("./Optimize_result.xlsx", sheet_name = "BAT")
