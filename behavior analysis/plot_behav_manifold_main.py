import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cmocean

os.chdir(r'C:\Users\User\Desktop\behavior_analysis')
import data_visualization as visual 

################################# LOAD DATA ###################################
behav_data_dir = r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold'
loco = pd.read_csv(os.path.join(behav_data_dir, 'behavioral_manifold.csv'))
principal_components = loco.iloc[:, 21:24].rolling(window=10, min_periods=1, center=True).mean().values
labels = loco.iloc[:, 1:13].columns
global_phase = loco.iloc[:, 24].values

with open(os.path.join(behav_data_dir, 'cycles_interp.npy'), 'rb') as file:
    cycles_interp = pickle.load(file)
avg_trajectory = np.mean(cycles_interp, axis = 0)
    
with open(os.path.join(behav_data_dir, 'pca_behav_obj.npy'), 'rb') as file:
    pca = pickle.load(file)
explained_variance = pca.explained_variance_ratio_
loadings = pca.components_[:3]

################################# ASSIGN SOME VARIABLES ###################################
color_paws = ['red', 'magenta', 'blue', 'cyan']
color_features = np.repeat(color_paws, 3)
epoch = [2000, 20000]


############################### PLOT #########################################
# Plot explained variance
visual.plot_data([range(1, len(explained_variance)+1), np.cumsum(explained_variance)],
          plottype='scatter',
          labels=['Principal components', 'VAF (ratio)'], 
          axlim=[[0, len(explained_variance)+1], [0, 1]], 
          color='k', s=45)

# Plot loadings
visual.plot_loadings(pca.components_[:3], labels, color = color_features)

# Map global phase on the behavioral manifold
colormap = cmocean.cm.phase  # Use the 'phase' colormap from cmocean
normalize = plt.Normalize(0, 2 * np.pi)
color_phase = colormap(normalize(global_phase[epoch[0]:epoch[1]]))
color_phase = [tuple(row) for row in color_phase]
ax = visual.plot_trajectory(principal_components[epoch[0]:epoch[1]], plot_dim='3D', plot_type='scatter', 
                            color=color_phase, s=1, marker='.') 

# Get average trajectory profile
ax = visual.plot_avg_trajectory(avg_trajectory, plot_type='factors', zorder=1, linewidth=3, color='crimson')
std_trajectory = np.nanstd(np.stack(cycles_interp, axis = 0), axis = 0)
for i in range(3):
    ax[i].fill_between(range(len(avg_trajectory[:, i])), avg_trajectory[:, i] - std_trajectory[:, i], 
                       avg_trajectory[:, i] + std_trajectory[:, i], color='crimson', alpha=0.2)

# Plot behavioral manifold with average trajectory
colormap = cmocean.cm.phase
color_phase = colormap(np.linspace(0, 1, len(avg_trajectory)))
for i in range(200):
ax = visual.plot_trajectory(cycles_interp[i][epoch[0]:epoch[1]], plot_dim='3D', plot_type='scatter', 
                      color = 'lightgray', alpha = 0.5, s=5, marker='.')
ax.scatter(avg_trajectory[:, 0], avg_trajectory[:, 1], avg_trajectory[:, 2], c = color_phase, s=20, marker='.')