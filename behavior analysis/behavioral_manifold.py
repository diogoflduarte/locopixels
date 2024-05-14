import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd
import pickle

os.chdir(r'C:\Users\User\Desktop\behavior_analysis')
from utils import inpaint_nans
from loco_tools import find_phase, get_cycles


######################### LOAD DATA & PRE-PROCESSING ##########################
# Load data
file_path = 'Y:\\data\\2022\\BATCH5\\processing\\VIV_23058\\S10\\VIV_23058_S10_behavioral_descriptor.csv'
data = pd.read_csv(file_path)

# Get forward locomotion
bw_loco_idx = np.where(data['wheel_speed'] < 0.1)[0] # adjust threshold
data_fw_loco = data.drop(bw_loco_idx)

# Work on 60' of data
sr_cam = 433
data_fw_loco = data_fw_loco[:60*60*sr_cam]

# Get paw position
paw_position = data_fw_loco.iloc[:, 14:26].values
paw_position = inpaint_nans(paw_position) # better to drop nans from the whole dataset
paw_position = paw_position - np.mean(paw_position, axis=0)


##################################### PCA #####################################
# Perform PCA
pca = PCA(n_components=3)
pca_fit = pca.fit(paw_position)
principal_components = pca.transform(paw_position)


################################# GLOBAL PHASE ################################
# Compute global phase with Phaser
global_phase_unwrapped = find_phase(paw_position.T)
global_phase = np.mod(global_phase_unwrapped[0], 2 * np.pi) # wrap phase


############################# AVERAGE TRAJECTORY ##############################
# Get cycles
cycles_interp, cycles, cycle_duration, edges = get_cycles(principal_components, global_phase, sr_cam)

# Get average trajectory 
avg_trajectory = np.mean(cycles_interp, axis = 0)


#################################### SAVE #####################################
# Rearrange data for saving
behav_manifold = pd.DataFrame({'PC1': principal_components[:, 0], 
                            'PC2': principal_components[:, 1],
                            'PC3': principal_components[:, 2],
                            'global phase': global_phase})    
             
cycle_timestamps = pd.DataFrame({'CycleOn': [False] * len(data_fw_loco), 'CycleOff': [False] * len(data_fw_loco)})
cycle_timestamps['CycleOn'].iloc[edges[0]] = True
cycle_timestamps['CycleOff'].iloc[edges[1]] = True                 
            
data_fw_loco.reset_index(inplace=True)     
df_loco = pd.concat([data_fw_loco['sessionwise_time'], data_fw_loco.iloc[:, 14:26], data_fw_loco.iloc[:, 35:], behav_manifold, cycle_timestamps], axis=1)
df_loco.drop(df_loco.columns[1], axis=1, inplace=True) # better to keep indices

save_path=r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\behavioral_manifold.csv'
df_loco.to_csv(save_path, index=False)
      
save_path=r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\cycles_interp.npy'
with open(save_path, 'wb') as file:
    pickle.dump(cycles_interp, file)
    
save_path=r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\cycles.npy'
with open(save_path, 'wb') as file:
    pickle.dump(cycles, file)