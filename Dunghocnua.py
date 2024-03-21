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
    Wa_1_var   = sol[0][0] * 1e-6
    Wb_1_var   = sol[0][1] * 1e-6
    Wc_d_1_var = sol[0][2] * 1e-6
    We_1_var   = sol[0][3] * 1e-6
    Wf_1_var   = sol[0][4] * 1e-6
 
    Wa_2_var   = sol[1][0] * 1e-6
    Wb_2_var   = sol[1][1] * 1e-6
    Wc_d_2_var = sol[1][2] * 1e-6
    We_2_var   = sol[1][3] * 1e-6
    Wf_2_var   = sol[1][4] * 1e-6

    Wa_3_var   = sol[2][0] * 1e-6
    Wb_3_var   = sol[2][1] * 1e-6
    Wc_d_3_var = sol[2][2] * 1e-6
    We_3_var   = sol[2][3] * 1e-6
    Wf_3_var   = sol[2][4] * 1e-6

    Wa_4_var   = sol[3][0] * 1e-6
    Wb_4_var   = sol[3][1] * 1e-6
    Wc_d_4_var = sol[3][2] * 1e-6
    We_4_var   = sol[3][3] * 1e-6
    Wf_4_var   = sol[3][4] * 1e-6
 
    Wa_5_var   = sol[4][0] * 1e-6
    Wb_5_var   = sol[4][1] * 1e-6
    Wc_d_5_var = sol[4][2] * 1e-6
    We_5_var   = sol[4][3] * 1e-6
    Wf_5_var   = sol[4][4] * 1e-6

    Wa_6_var   = sol[5][0] * 1e-6
    Wb_6_var  = sol[5][1] * 1e-6
    Wc_d_6_var = sol[5][2] * 1e-6
    We_6_var   = sol[5][3] * 1e-6
    Wf_6_var   = sol[5][4] * 1e-6

    Wa_7_var   = sol[6][0] * 1e-6
    Wb_7_var   = sol[6][1] * 1e-6
    Wc_d_7_var = sol[6][2] * 1e-6
    We_7_var   = sol[6][3] * 1e-6
    Wf_7_var   = sol[6][4] * 1e-6

    Wa_8_var   = sol[7][0] * 1e-6
    Wb_8_var   = sol[7][1] * 1e-6
    Wc_d_8_var = sol[7][2] * 1e-6
    We_8_var   = sol[7][3] * 1e-6
    Wf_8_var   = sol[7][4] * 1e-6

    Wa_9_var   = sol[8][0] * 1e-6
    Wb_9_var   = sol[8][1] * 1e-6
    Wc_d_9_var = sol[8][2] * 1e-6
    We_9_var   = sol[8][3] * 1e-6
    Wf_9_var   = sol[8][4] * 1e-6

    Wa_10_var   = sol[9][0] * 1e-6
    Wb_10_var   = sol[9][1] * 1e-6
    Wc_d_10_var = sol[9][2] * 1e-6
    We_10_var   = sol[9][3] * 1e-6
    Wf_10_var   = sol[9][4] * 1e-6

    Wa_11_var   = sol[10][0] * 1e-6
    Wb_11_var   = sol[10][1] * 1e-6
    Wc_d_11_var = sol[10][2] * 1e-6
    We_11_var   = sol[10][3] * 1e-6
    Wf_11_var   = sol[10][4] * 1e-6
 
    Wa_12_var   = sol[11][0] * 1e-6
    Wb_12_var   = sol[11][1] * 1e-6
    Wc_d_12_var = sol[11][2] * 1e-6
    We_12_var   = sol[11][3] * 1e-6
    Wf_12_var   = sol[11][4] * 1e-6

    Wa_13_var   = sol[12][0] * 1e-6
    Wb_13_var   = sol[12][1] * 1e-6
    Wc_d_13_var = sol[12][2] * 1e-6
    We_13_var   = sol[12][3] * 1e-6
    Wf_13_var   = sol[12][4] * 1e-6

    Wa_14_var   = sol[13][0] * 1e-6
    Wb_14_var   = sol[13][1] * 1e-6
    Wc_d_14_var = sol[13][2] * 1e-6
    We_14_var   = sol[13][3] * 1e-6
    Wf_14_var   = sol[13][4] * 1e-6
 
    Wa_15_var   = sol[14][0] * 1e-6
    Wb_15_var   = sol[14][1] * 1e-6
    Wc_d_15_var = sol[14][2] * 1e-6
    We_15_var   = sol[14][3] * 1e-6
    Wf_15_var   = sol[14][4] * 1e-6

    Wa_16_var   = sol[15][0] * 1e-6
    Wb_16_var   = sol[15][1] * 1e-6
    Wc_d_16_var = sol[15][2] * 1e-6
    We_16_var   = sol[15][3] * 1e-6
    Wf_16_var   = sol[15][4] * 1e-6

    Wa_17_var   = sol[16][0] * 1e-6
    Wb_17_var   = sol[16][1] * 1e-6
    Wc_d_17_var = sol[16][2] * 1e-6
    We_17_var   = sol[16][3] * 1e-6
    Wf_17_var   = sol[16][4] * 1e-6

    Wa_18_var   = sol[17][0] * 1e-6
    Wb_18_var   = sol[17][1] * 1e-6
    Wc_d_18_var = sol[17][2] * 1e-6
    We_18_var   = sol[17][3] * 1e-6
    Wf_18_var   = sol[17][4] * 1e-6

    Wa_19_var   = sol[18][0] * 1e-6
    Wb_19_var   = sol[18][1] * 1e-6
    Wc_d_19_var = sol[18][2] * 1e-6
    We_19_var   = sol[18][3] * 1e-6
    Wf_19_var   = sol[18][4] * 1e-6

    Wa_20_var  = sol[19][0] * 1e-6
    Wb_20_var   = sol[19][1] * 1e-6
    Wc_d_20_var = sol[19][2] * 1e-6
    We_20_var   = sol[19][3] * 1e-6
    Wf_20_var   = sol[19][4] * 1e-6

    I5_var = 100 * 1e-6
    Vcm_var = 900 * 1e-3 
    Vdd_var = 2 
    Cc_var = 5 * 1e-12
    CL_var = 10 * 1e-12


    ### Attach parameters to OCEAN File ### 
    filethongso = ("./PSO_para.txt") 
    f = open(filethongso, 'w')
    f.write(  "\n%s"   %Wa_1_var            )
    f.write(  "\n%s"   %Wa_2_var            )
    f.write(  "\n%s"   %Wa_3_var            )
    f.write(  "\n%s"   %Wa_4_var            )
    f.write(  "\n%s"   %Wa_5_var            )
    f.write(  "\n%s"   %Wa_6_var            )
    f.write(  "\n%s"   %Wa_7_var            )
    f.write(  "\n%s"   %Wa_8_var            )
    f.write(  "\n%s"   %Wa_9_var            )
    f.write(  "\n%s"   %Wa_10_var           )
    f.write(  "\n%s"   %Wa_11_var           )
    f.write(  "\n%s"   %Wa_12_var           )
    f.write(  "\n%s"   %Wa_13_var           )
    f.write(  "\n%s"   %Wa_14_var           )
    f.write(  "\n%s"   %Wa_15_var           )
    f.write(  "\n%s"   %Wa_16_var           )
    f.write(  "\n%s"   %Wa_17_var           )
    f.write(  "\n%s"   %Wa_18_var           )
    f.write(  "\n%s"   %Wa_19_var           )
    f.write(  "\n%s"   %Wa_20_var           )
    f.write(  "\n%s"   %Wb_1_var            )
    f.write(  "\n%s"   %Wb_2_var            )
    f.write(  "\n%s"   %Wb_3_var            )
    f.write(  "\n%s"   %Wb_4_var            )
    f.write(  "\n%s"   %Wb_5_var            )
    f.write(  "\n%s"   %Wb_6_var            )
    f.write(  "\n%s"   %Wb_7_var            )
    f.write(  "\n%s"   %Wb_8_var            )
    f.write(  "\n%s"   %Wb_9_var            )
    f.write(  "\n%s"   %Wb_10_var           )
    f.write(  "\n%s"   %Wb_11_var           )
    f.write(  "\n%s"   %Wb_12_var           )
    f.write(  "\n%s"   %Wb_13_var           )
    f.write(  "\n%s"   %Wb_14_var           )
    f.write(  "\n%s"   %Wb_15_var           )
    f.write(  "\n%s"   %Wb_16_var           )
    f.write(  "\n%s"   %Wb_17_var           )
    f.write(  "\n%s"   %Wb_18_var           )
    f.write(  "\n%s"   %Wb_19_var           )
    f.write(  "\n%s"   %Wb_20_var           )
    f.write(  "\n%s"   %Wc_d_1_var            )
    f.write(  "\n%s"   %Wc_d_2_var            )
    f.write(  "\n%s"   %Wc_d_3_var            )
    f.write(  "\n%s"   %Wc_d_4_var            )
    f.write(  "\n%s"   %Wc_d_5_var            )
    f.write(  "\n%s"   %Wc_d_6_var            )
    f.write(  "\n%s"   %Wc_d_7_var            )
    f.write(  "\n%s"   %Wc_d_8_var            )
    f.write(  "\n%s"   %Wc_d_9_var            )
    f.write(  "\n%s"   %Wc_d_10_var           )
    f.write(  "\n%s"   %Wc_d_11_var           )
    f.write(  "\n%s"   %Wc_d_12_var           )
    f.write(  "\n%s"   %Wc_d_13_var           )
    f.write(  "\n%s"   %Wc_d_14_var           )
    f.write(  "\n%s"   %Wc_d_15_var           )
    f.write(  "\n%s"   %Wc_d_16_var           )
    f.write(  "\n%s"   %Wc_d_17_var           )
    f.write(  "\n%s"   %Wc_d_18_var           )
    f.write(  "\n%s"   %Wc_d_19_var           )
    f.write(  "\n%s"   %Wc_d_20_var           )
    f.write(  "\n%s"   %Wc_d_1_var            )
    f.write(  "\n%s"   %Wc_d_2_var            )
    f.write(  "\n%s"   %Wc_d_3_var            )
    f.write(  "\n%s"   %Wc_d_4_var            )
    f.write(  "\n%s"   %Wc_d_5_var            )
    f.write(  "\n%s"   %Wc_d_6_var            )
    f.write(  "\n%s"   %Wc_d_7_var            )
    f.write(  "\n%s"   %Wc_d_8_var            )
    f.write(  "\n%s"   %Wc_d_9_var            )
    f.write(  "\n%s"   %Wc_d_10_var          )
    f.write(  "\n%s"   %Wc_d_11_var           )
    f.write(  "\n%s"   %Wc_d_12_var           )
    f.write(  "\n%s"   %Wc_d_13_var           )
    f.write(  "\n%s"   %Wc_d_14_var           )
    f.write(  "\n%s"   %Wc_d_15_var           )
    f.write(  "\n%s"   %Wc_d_16_var           )
    f.write(  "\n%s"   %Wc_d_17_var           )
    f.write(  "\n%s"   %Wc_d_18_var           )
    f.write(  "\n%s"   %Wc_d_19_var           )
    f.write(  "\n%s"   %Wc_d_20_var           )
    f.write(  "\n%s"   %We_1_var            )
    f.write(  "\n%s"   %We_2_var            )
    f.write(  "\n%s"   %We_3_var            )
    f.write(  "\n%s"   %We_4_var            )
    f.write(  "\n%s"   %We_5_var            )
    f.write(  "\n%s"   %We_6_var            )
    f.write(  "\n%s"   %We_7_var            )
    f.write(  "\n%s"   %We_8_var            )
    f.write(  "\n%s"   %We_9_var            )
    f.write(  "\n%s"   %We_10_var           )
    f.write(  "\n%s"   %We_11_var           )
    f.write(  "\n%s"   %We_12_var           )
    f.write(  "\n%s"   %We_13_var           )
    f.write(  "\n%s"   %We_14_var           )
    f.write(  "\n%s"   %We_15_var           )
    f.write(  "\n%s"   %We_16_var           )
    f.write(  "\n%s"   %We_17_var           )
    f.write(  "\n%s"   %We_18_var           )
    f.write(  "\n%s"   %We_19_var           )
    f.write(  "\n%s"   %We_20_var           )
    f.write(  "\n%s"   %Wf_1_var            )
    f.write(  "\n%s"   %Wf_2_var            )
    f.write(  "\n%s"   %Wf_3_var            )
    f.write(  "\n%s"   %Wf_4_var            )
    f.write(  "\n%s"   %Wf_5_var            )
    f.write(  "\n%s"   %Wf_6_var            )
    f.write(  "\n%s"   %Wf_7_var            )
    f.write(  "\n%s"   %Wf_8_var            )
    f.write(  "\n%s"   %Wf_9_var            )
    f.write(  "\n%s"   %Wf_10_var           )
    f.write(  "\n%s"   %Wf_11_var           )
    f.write(  "\n%s"   %Wf_12_var           )
    f.write(  "\n%s"   %Wf_13_var           )
    f.write(  "\n%s"   %Wf_14_var           )
    f.write(  "\n%s"   %Wf_15_var           )
    f.write(  "\n%s"   %Wf_16_var           )
    f.write(  "\n%s"   %Wf_17_var           )
    f.write(  "\n%s"   %Wf_18_var           )
    f.write(  "\n%s"   %Wf_19_var           )
    f.write(  "\n%s"   %Wf_20_var           ) 
    f.write(  "\n%s"   %I5_var              )
    f.write(  "\n%s"   %Vcm_var             )    
    f.write(  "\n%s"   %Vdd_var             )
    f.write(  "\n%s"   %Cc_var              )
    f.write(  "\n%s"   %CL_var              )
    f.close()

    ### Processing and Collecting the Result ###
    ### ------------------------------------ ###
    os.system("ocean -nograph -restore FoM.ocn")
    param_result = np.array([])
    with open("./FoM_result.txt") as f1:
        param = f1.readlines()
        param = np.array(param)
        param_result = param.astype(float)

    ### ------------------------------------ ###
    os.system("ocean -nograph -restore Itotal.ocn")
    Itt_result = np.array([])
    with open("./Itotal_result.txt") as f2: 
        Itotal = f2.readlines()
        Itotal = np.array(Itotal)
        Itt_result = Itotal.astype(float)

    ### Reform the result ###

    Gain = [0] * 20
    GBW = [0] * 20
    PM = [0] * 20
    FoM = [0] * 20
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
class PSOAlgorithm():
    def __init__(self,D, NP, Max_T,function,df):
        self.NP = NP
        self.D = D
        self.Max_T = Max_T
        self.Fun = function
        self.df = df

        self.FoM_min = [0.0]
        self.GainDC = [0]*self.NP
        self.GainofBest = [0.0]
        
        self.position = [[0 for i in range(self.D)] for j in range(self.NP)]
        self.velocity = [[0 for i in range(self.D)] for j in range(self.NP)]
        self.fitness = [0]*self.NP
        self.best_position = [0]*self.D
        self.best_fitness = float('inf')

    #function find best fitness value   
    def best_PSO(self):
        i = 0 
        j = 0
        for i in range(self.NP):
            if(self.fitness[i] < self.fitness[j] and (self.GainDC[i] >= 30)):
                j = i
        for i in range(self.D):
            self.best_position = self.position[j][i]
        self.FoM_min = self.fitness[j]
        self.GainofBest = self.GainDC[j]
    
    def init_PSO(self):
        for i in range(self.NP):
            for j in range(self.D):
                self.velocity[i][j] = 0.0
                self.position[i][j] = 0.0 
        self.fitness, self.GainDC = self.Fun(self.NP, self.position)
        self.best_PSO()

        
    def Move_PSO(self):

        # Position Initialization
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]

        Fnew = [0.0]* self.NP

        Gainnew = [0.0]*self.NP

        r1 = np.random.random()
        r2 = np.random.random()
        w = 0.8
        c1 = 1.2
        c2 = 1.2

        #initial value
        self.init_PSO()
        #loop for iterations

        for t in range(self.Max_T):
            begin_time = dt.datetime.now().strftime("%H:%M:%S")

            #loop
            for i in range(self.NP):
                for j in range(self.D):
                    self.velocity[i][j] = w*self.velocity[i][j] + c1*r1*(self.best_position[i] - self.position[i][j]) + c2*r2*(self.best_fitness[i][j] - self.position[i][j])  
                    S[i][j] = self.position[i][j] + self.velocity[i][j]
            
            #calculate new fitness in this iteration if all swarm
            Fnew, Gainnew = self.Fun(self.NP, S)
            
            for i in range(self.NP):
                if ((Fnew[i] < self.fitness[i]) and (Gainnew[i] >= 30)):
                    for j in range(self.D):
                        self.position[i][j] = S[i][j]
                    self.fitness[i] = Fnew[i]
                    self.GainDC[i] = Gainnew[i]
            
            for i in range(self.NP):
                if (self.fitness[i] < self.FoM_min) and (self.GainDC[i] >=30):
                    for j in range(self.D):
                        self.best_position[j] = self.position[i][j]
                    self.FoM_min = self.fitness[i]
                    self.GainofBest = self.GainDC[i]
        
            new_row = [
                {
                    "Lan chay" : t+1,
                    "Begin": begin_time,
                    "End": dt.datetime.now().strftime("%H:%M:%S"),
                    "Time (s)": (dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"),'%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S')).total_seconds(),
                    "Fitness" : self.FoM_min,
                    "Wa (um)" : self.best_position[0],
                    "Wb (um)" : self.best_position[1],
                    "Wc (um)" : self.best_position[2],
                    "Wd (um)" : self.best_position[2],
                    "We (um)" : self.best_position[3],
                    "Wf (um)" : self.best_position[4],
                    "Gain (dB)" : self.GainofBest,
                }]   
            
            self.df = self.df.append(new_row, ignore_index = True)
            print(self.df)
        
        return self.df

###-------main program--------###
if __name__ == "__main__":
    #define range

    #create folder to contain exported files
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
    Algorithm = PSOAlgorithm(5, 20, 10, Fun, df)
    df = Algorithm.Move_PSO()

    #output
    print('Done!')

    df.to_csv("./Optimize_result.csv")
    df.to_excel("./Optimize_result.xlsx", sheet_name = "PSO")