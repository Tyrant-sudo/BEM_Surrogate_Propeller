import numpy as np

PC_path  = '/home/sh/WCY/auto_propeller/resource4/0_data/start_mesh.npy'

start_PC = np.load(PC_path) 
print(start_PC.shape, start_PC[:,0,2])

# 提取radius, chord, pitch, load_path 并放进BsplineProp.ini
# 提取sections放到1_model/sections

