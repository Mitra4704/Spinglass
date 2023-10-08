import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
import time

# Parameters
initial_temperature = 100
cooling = 0.95  
x_upper_bound = 1
x_lower_bound = 0
y_upper_bound = 1
y_lower_bound = 0
z_upper_bound = 1
z_lower_bound = 0
epsilon = 1e-6  # small value to exclude exact boundary values

def objective_function(solution, a, b):
    x, y, z = solution
    xlog = y * np.log(y) + z*np.log(z) + (x - y - z) * np.log(x - y - z)
    print(xlog)
    return -((2.18**2 + b**2)+(2.18 - 2.18**2)*x+(a - b**2)*y**2 + 
             2*(a - b**2)*(y**2 + z**2 + y*z - x*(y+z)))+((1.0 - x)*np.log((1.0 - x)/10.0)+
             xlog + 
             (2.18**2 * (1.0 - x) + b**2 * (1.0 - y**2 -z**2 - (x - y -z)**2)) / 2.0)

# Update the neighbour generation to ensure it's within the new bounds and x > y
def neighbour(x, y, z):
    
    new_y = y + np.random.uniform(-1, 1)
    new_y = min(max(new_y, y_lower_bound + epsilon), new_y - epsilon)  
    
    new_z = z + np.random.uniform(-1, 1)
    new_z = min(max(new_z, z_lower_bound + epsilon), z_upper_bound - epsilon)
    
    new_x = x + np.random.uniform(-1, 1)
    new_x = min(max(new_x, x_lower_bound + epsilon), min((new_y + new_z + epsilon), x_upper_bound - epsilon))
    #new_z = min((new_y+new_z),(new_x-new_y)-epsilon)

    return new_x, new_y, new_z

# Simulated Annealing function
def simulated_annealing(a, b):
    x = np.random.uniform(x_lower_bound + epsilon, x_upper_bound - epsilon)
    y = np.random.uniform(y_lower_bound + epsilon, min(x - epsilon, y_upper_bound - epsilon)) # for x>y
    z = np.random.uniform(z_lower_bound + epsilon, min((x- y)-0.1*epsilon,z_upper_bound - epsilon))
    #z = min(max(z, z_lower_bound + epsilon), (x- y)-epsilon)
    if x < y + z:
        return "NA",-1,-1,-1,a,b
    best_x, best_y, best_z = x,y,z
    best_value = objective_function((best_x,best_y,best_z),a,b)
    current_temperature = initial_temperature
    prevValue = best_value
    for i in range(1,10000):
    #while current_temperature > 0.1:
        new_x, new_y, new_z = neighbour(x, y, z)
        new_value = objective_function((new_x,new_y,new_z),a,b)
        if not math.isnan(new_value):
            delta_value = new_value - best_value
            
            if delta_value < 0 or np.random.random() < np.exp(-i*delta_value / current_temperature):
                best_x, best_y, best_z, best_value = new_x, new_y, new_z, new_value
            
            #print(current_temperature,new_x,new_y,new_value,delta_value)
            current_temperature *= cooling
        # break
        else:
            return "NA",-1,-1,-1,a,b

    return best_x, best_y, best_z, best_value,a,b

# Loop over a and b values and run Simulated Annealing for each case again with updated bounds
a_values = np.linspace(0, 5, 100,endpoint=True)
b_values = np.linspace(0, 2.5, 100,endpoint=True)

best_x_list= np.empty(1)
best_y_list= np.empty(1)
best_z_list= np.empty(1)
best_value_list= np.empty(1)
a_list = np.empty(1)
b_list = np.empty(1)

# for a in tqdm(a_values):
#     for b in b_values:
#         best_x, best_y, best_z, best_value = simulated_annealing(a, b)
#         if best_x == "NA":
#             continue
#         best_x_list = np.append(best_x_list,best_x)
#         best_y_list = np.append(best_y_list,best_y)
#         best_z_list = np.append(best_z_list,best_z)
#         best_value_list = np.append(best_value_list,best_value)
#         a_list = np.append(a_list,a)
#         b_list = np.append(b_list,b)

def simulated_annealing_wrapper(args):
    return simulated_annealing(*args)
results = []
start_time = time.perf_counter()
with Pool() as pool:
        for a in a_values:
            # Create argument tuples for each b value
            args = [(a, b) for b in b_values]
            
            # Run the worker function in parallel for each b value
            results.extend(pool.map(simulated_annealing_wrapper, args))            
    
    # Filter out None results
results = [r for r in results if r[0]!="NA"]
    
    # Write results to a CSV file
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["best_x", "best_y", "best_z", "best_value", "a", "b"])  # Writing header
    writer.writerows(results)

end_time = time.perf_counter()

print(f" Total Time : {end_time-start_time}")
