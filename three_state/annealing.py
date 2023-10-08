import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Parameters
initial_temperature = 100
cooling = 0.95  
upper_bound = 1
lower_bound = 0
epsilon = 1e-6  # small value to exclude exact boundary values

y = np.linspace(lower_bound+epsilon,upper_bound-epsilon,num=100,endpoint=False)
z = np.linspace(lower_bound+epsilon,upper_bound-epsilon,num=100,endpoint=False)
x = y + z + epsilon
X,Y,Z = np.meshgrid(x,y,z)

def objective_function(solution, a, b):
    x, y, z = solution
    xyzLog = y * np.log(y) + z*np.log(z) + (x - y - z) * np.log(x - y - z) # >= 0
    result = -((2.18**2 + b**2)+(2.18 - 2.18**2)*x+(a - b**2)*y**2 + 
            2*(a - b**2)*(y**2 + z**2 + y*z - x*(y+z)))+((1.0 - x)*np.log((1.0 - x)/10.0)+
            xyzLog + 
             (2.18**2 * (1.0 - x) + b**2 * (1.0 - y**2 -z**2 - (x - y -z)**2)) / 2.0)
    return result
# Simulated Annealing function
def simulated_annealing(a, b):
    #best_x, best_y = x[0],y[0]
    #best_value = objective_function((best_x,best_y),a,b)
    #current_temperature = initial_temperature
    
    #while current_temperature > 0.1:
        new_value_grid = objective_function((X,Y,Z),a,b)
        new_value = np.nanmin(new_value_grid)
        #delta_value = new_value - best_value
        
        #if delta_value < 0 or np.random.random() < np.exp(delta_value / current_temperature):
        best_value = new_value
        best_x = x[np.where(new_value_grid==best_value)[2]][0]
        best_y = y[np.where(new_value_grid==best_value)[1]][0]
        best_z = z[np.where(new_value_grid==best_value)[0]][0]
        
        #current_temperature *= cooling

        return best_x, best_y, best_z, best_value


# Loop over a and b values and run Simulated Annealing for each case again with updated bounds
a_values = np.linspace(0, 5, 100)
b_values = np.linspace(0, 2.5, 100)

best_x_list=[]
best_y_list=[]
best_z_list=[]
best_value_list=[]
a_list = []
b_list = []

for a in tqdm(a_values):
    for b in b_values:
        best_x, best_y, best_z, best_value = simulated_annealing(a, b)
        best_x_list.append(best_x)
        best_y_list.append(best_y)
        best_z_list.append(best_z)
        best_value_list.append(best_value)
        a_list.append(a)
        b_list.append(b)