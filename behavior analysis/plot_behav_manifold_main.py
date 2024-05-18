import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import math


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
paws = ['FR', 'HR', 'FL', 'HL']


############################### PLOT #########################################
# Plot explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, len(explained_variance)+1), [0] + list(np.cumsum(explained_variance)),linewidth=2.5)
plt.scatter(np.arange(len(explained_variance)) + 1, np.cumsum(explained_variance), s=150, marker='.')
plt.xlabel('Principal Components', fontsize=15)
plt.ylabel('VAF (ratio)', fontsize=15)
plt.xticks([1, 2, 3], labels=[1, 2, 3], fontsize=12) 
plt.yticks(fontsize=12) 
plt.ylim(0, 1)  
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Plot 3D manifold
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n_cycles=30
selected_cycles = []
for i in range(1, len(cycles_interp), round(len(cycles_interp)/n_cycles)):
    cycle = cycles_interp[i]
    selected_cycles.append(cycle)
    ax.scatter(cycle[:, 0], cycle[:, 1], cycle[:, 2], color='lightgray', s=1, alpha=0.1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_zlabel('PC3', fontsize = 15)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.grid(False)
colormap = cmocean.cm.phase
color_phase = colormap(np.linspace(0, 1, len(avg_trajectory)))
avg_trajectory = np.mean(selected_cycles, axis = 0)
scatter = ax.scatter(avg_trajectory[:, 0], avg_trajectory[:, 1], avg_trajectory[:, 2], c = color_phase, s=55, marker='.')


# Plot cycles against phase & loadings
num_components = loadings.shape[0]
num_features = loadings.shape[1]
fig, ax = plt.subplots(num_components, 2, figsize=(9, 9), gridspec_kw={'width_ratios': [1, 2]})
num_samples = len(cycles_interp[0])
selected_cycles = []
phase = np.linspace(0, 100, len(cycles_interp[0]))
for pc in range(num_components):
    for i in range(0, len(cycles_interp), round(len(cycles_interp)/n_cycles)):
        cycle =  cycles_interp[i]
        selected_cycles.append(cycle)
        ax[pc, 0].plot(phase, cycle[:, pc], color='lightgray', linewidth=1, alpha=0.5)
        ax[pc, 1].bar(np.arange(num_features), loadings[pc], color=color_features)
        
    ax[pc, 0].set_ylabel(f'PC{pc+1}', fontsize = 15)
    ax[pc, 0].set_xticks([0, 49, 99])
    ax[pc, 0].set_xticklabels(['0', '50%', '100%'], fontsize = 15)
    ax[pc, 0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax[pc, 0].spines['right'].set_visible(False)
    ax[pc, 0].spines['top'].set_visible(False)
    ax[pc, 1].set_ylabel('Loadings', fontsize = 15)
    ax[pc, 1].tick_params(axis='y', labelsize=12)
    if pc < 2: 
        ax[pc, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    if pc == 1:
        ax[pc, 0].set_ylim(-40, 40)
    elif pc == 2:
        ax[pc, 0].set_ylim(-20, 20)
        
[ax[pc, 0].plot(phase, avg_trajectory[:, pc], color='crimson', linewidth=2.5) for pc in range(num_components)]
ax[num_components-1, 1].set_xticks(np.arange(num_features))
ax[num_components-1, 1].set_xticklabels(labels, fontsize = 15, rotation=90)

for pc in range(num_components):
    ax[pc, 1].spines['right'].set_visible(False)
    ax[pc, 1].spines['top'].set_visible(False)
    
plt.tight_layout()


# Stance onset distribution along the global phase
st_phase = {}
for paw in paws:
    st_phase[paw] = global_phase[np.where(loco[f'{paw}_StOn'] == True)[0]]
fig, ax = plt.subplots()
for p, paw in enumerate(paws):
    ax.hist(global_phase[np.where(loco[f'{paw}_StOn'] == True)[0]], bins=36, color=color_paws[p], histtype='step', linewidth=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks([])
ax.set_xlabel('Stance onset phase (%)', fontsize = 15)
ax.set_ylabel('Strides', fontsize = 15)
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0%', '50%', '100%'], fontsize = 15)


