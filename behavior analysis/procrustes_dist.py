import numpy as np
import pickle
from scipy.spatial import procrustes

data_path = r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\cycles_interp.npy'
with open(data_path, 'rb') as file:
    cycles_interp = pickle.load(file)
    
# Get average trajectory
avg_trajectory = np.mean(cycles_interp, axis = 0)

# Compute Procrustes distance
procrustes_dist = np.zeros(len(cycles_interp))
rotated_cycles = [None] * len(cycles_interp)
for i, cycle in enumerate(cycles_interp):
    avg_trajectory_norm, r, d = procrustes(avg_trajectory, cycle)
    procrustes_dist[i] = d
    rotated_cycles[i] = r

# Compute pointwise Euclidean distance from average trajectory
euclidean_dist = [np.linalg.norm(cycle - avg_trajectory_norm, axis = 1) for cycle in rotated_cycles]