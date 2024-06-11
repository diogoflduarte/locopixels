# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:40:47 2024

@author: User
"""

import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.colors as mcolors
import cmocean


def interpolate(X, n_samples):
    X_interp = []
    for x in X:
        size = x.shape
        idx = np.linspace(0, size[0] - 1, num=n_samples)
        if len(size) == 1:
            x_interp = interp1d(np.arange(size[0]), x)(idx)
        else:
            x_interp = np.column_stack([interp1d(np.arange(size[0]), x[:, col])(idx) for col in range(size[1])])
        X_interp.append(x_interp)
    return X_interp


## ##################### INITIALIZE SOME VARIABLES #############################
save_data = False
save_plot = True
load_data = False
# cell_type = 'Purkinje cells'
cell_type = 'Mossy Fibers'


## ############################## LOAD DATA ####################################
# Load neural data
neural_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels'
if cell_type == 'Purkinje Cells':
    selected_cells = ['time', 91, 111, 115, 151, 198, 226, 241, 246, 259, 400, 402,
                      415, 88, 209, 266, 274, 382] 
elif cell_type == 'Mossy Fibers':
    selected_cells = ['time', 317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467, 
                    479, 500, 501, 503, 507, 601, 602, 604, 611, 613] 
cols = [str(xx) for xx in selected_cells]
firing_rate = pd.read_csv(os.path.join(neural_data_path, 'sessionwise_firing_rates.csv'), usecols=cols)

# Load behavioral data and extract event onset and offset
behav_data_path = r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\behavioral_manifold.csv'
if os.getlogin() == 'diogo':
    behav_data_path = r'W:\Processed data\Behavioral manifold\behavioral_manifold.csv'

behavior = pd.read_csv(behav_data_path)
on = behavior.loc[behavior['CycleOn'], 'sessionwise_time'].values
off = behavior.loc[behavior['CycleOff'], 'sessionwise_time'].values
# on = behavior.loc[behavior['FR_StOn'], 'sessionwise_time'].values
# on = on[:-1]; off = on[1:]\
speed = behavior['wheel_speed']


################################# PETHs #######################################
# Get phase-locked PETH by cycle
peth_path = f'C:\\Users\\User\\Desktop\\Npx data analysis\\data\\{cell_type}_peth_global_phase.csv'
if os.path.exists(peth_path) and load_data:
    data = pd.read_csv(peth_path)
    print("File loaded successfully.")
else:
    print("File does not exist. Computing PETH...")
    ## TURN INTO A FUNCTION
    # n_cycles = len(off)
    n_cycles = 5000
    n_samples=300
    peth = pd.DataFrame()
    for i in range(n_cycles):
        mask = (firing_rate['time'] >= on[i]) & (firing_rate['time'] <= off[i])
        peth_i = firing_rate[mask].iloc[:, 1:].values
        peth_i = np.array(interpolate(peth_i.T, n_samples=n_samples)).T
        peth_i = pd.DataFrame(peth_i, columns=firing_rate.iloc[:, 1:].columns)
        peth = pd.concat([peth, peth_i], ignore_index=True)
trial_id = np.concatenate([np.ones(n_samples)*i for i in range(n_cycles)])

# Save
if save_data:
    peth.to_csv(f'C:\\Users\\User\\Desktop\\Npx data analysis\\data\\{cell_type}_peth_global_phase_cycles.csv')

# Standardize
peth = peth.loc[:, (peth != 0).any(axis=0)]
# peth_mean = peth.mean().values
# peth_std = peth.std().values
# peth_zs = (peth - peth_mean) / peth_std
peth_min = peth.min().values
peth_max = peth.max().values
peth_zs = (peth - peth_min) / (peth_max - peth_min)

# Get cycles average speed
# speed_mean = np.zeros(n_cycles)
# for i in range(n_cycles):
#         mask = (behavior['sessionwise_time'] >= on[i]) & (behavior['sessionwise_time'] <= off[i])
#         speed_mean[i] = np.mean(speed[mask].values)
    
# # Plot PETHs
# ## USE FUNCTION
# plt.figure()
# for unit in peth_zs.columns:
#     plt.plot(np.linspace(0, 100, 300), peth_zs[unit], linewidth=2.5)
#     plt.xlabel('Global phase (%)', fontsize=15)
#     plt.ylabel('Firing rate (z-score)', fontsize=15)    
#     plt.margins(0)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)


################################## PCA ########################################
# Perform PCA
n_components = 17
pca = PCA(n_components=n_components)
pca_fit = pca.fit(peth_zs)
loadings = pca.components_


# Plot explained variance
plt.figure()
cumulative_variance2 = np.cumsum(pca.explained_variance_ratio_)
plt.bar(np.arange(1, len(cumulative_variance2) + 1), cumulative_variance2, color='lightgray')
plt.xlabel('Principal Components', fontsize=15)
plt.ylabel('Cumulative Explained Variance', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Project each cycle onto PC space
selected_cycles = 30
pc_i = []
# speed_mean_i = np.zeros(selected_cycles)
# j=0
for i in range(1, n_cycles+1, round(n_cycles/selected_cycles)):
    peth_i = peth[trial_id == i]
    # peth_i_zs = (peth_i - peth_mean)/peth_std
    peth_i_zs = (peth_i - peth_min) / (peth_max - peth_min)
    # peth_i_zs = peth_i.apply(zscore)
    pc_i.append(pca.transform(peth_i_zs))
    # speed_mean_i[j] = speed_mean[i]
    # j=j+1
avg_pc = np.mean(np.stack(pc_i, axis=0), axis=0)

# Plot cycle-by-cycle PCA
cmap = cmocean.cm.phase
norm = mcolors.Normalize(vmin=0, vmax=1)
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(avg_pc[:, 0], avg_pc[:, 1], avg_pc[:, 2], c=np.linspace(0, 1, 300), cmap=cmap, norm=norm, s=45)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_zlabel('PC3', fontsize=15)
cbar = plt.colorbar(scatter, ax=ax, label='Global phase (%)', shrink=0.5, ticks=[0, 1], pad=0.15)
cbar.ax.yaxis.label.set_rotation(270)  # Rotate the colorbar label
cbar.ax.tick_params(labelsize=12)
cbar.ax.yaxis.labelpad = 20  # Increase the padding
cbar.ax.yaxis.label.set_fontsize(15)
cbar.set_ticklabels([0, 100])
plt.show()
# colormap = plt.cm.Reds
# color_speed = colormap(speed_mean_i)
for i in range(selected_cycles):
    scatter = ax.scatter(pc_i[i][:, 0], pc_i[i][:, 1], pc_i[i][:, 2], s=1, c='lightgray', alpha=0.2)
    # scatter = ax.scatter(pc_i[i][:, 0], pc_i[i][:, 1], pc_i[i][:, 2], s=1, c=color_speed[i])
    

# Plot individual components against phase and their loadings
fig, ax = plt.subplots(3, 2, figsize=(8, 8))
for i in range(3):
    ax[i, 0].plot(np.linspace(0, 100, 300), avg_pc[:, i], color='darkblue', linewidth=4)
    for j in range(selected_cycles):
        ax[i, 0].plot(np.linspace(0, 100, 300), pc_i[j][:, i], color='lightgray', linewidth=1, alpha=0.5)
    ax[i, 0].set_ylabel(f'PC{i+1}', fontsize=15)
    ax[i, 0].tick_params(axis='y', which='both', left=False, labelleft=False)
    ax[i, 0].margins(0)
    ax[i, 0].tick_params(labelsize=12)
    ax[i, 0].spines['top'].set_visible(False)
    ax[i, 0].spines['right'].set_visible(False)
    if i == 2:
        ax[i, 0].set_xlabel('Global phase (%)', fontsize=15)
    else:
        ax[i, 0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax[i, 1].bar(np.arange(loadings.shape[-1]), loadings[i, :], color='darkblue', edgecolor='black')
    ax[i, 1].margins(0)
    ax[i, 1].tick_params(labelsize=12)
    ax[i, 1].spines['top'].set_visible(False)
    ax[i, 1].spines['right'].set_visible(False)
    if i == 2:
        ax[i, 1].set_xlabel(cell_type, fontsize=15)
    ax[i, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tight_layout()
plt.show()