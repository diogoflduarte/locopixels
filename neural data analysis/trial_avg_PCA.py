import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import cm


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


####################### INITIALIZE SOME VARIABLES #############################
save_data = True
save_plot = True
load_data = False
# cell_type = 'Purkinje cells'
cell_type = 'Mossy Fibers'


################################ LOAD DATA ####################################
# Load neural data
neural_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels'
if cell_type == 'Purkinje cells':
    selected_cells = ['time', 91, 111, 115, 151, 198, 226, 241, 246, 259, 400, 402,
                      415, 88, 209, 266, 274, 382] 
elif cell_type == 'Mossy Fibers':
    selected_cells = ['time', 317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467, 
                    479, 500, 501, 503, 507, 601, 602, 604, 611, 613] 
cols = [str(xx) for xx in selected_cells]
firing_rate = pd.read_csv(os.path.join(neural_data_path, 'sessionwise_firing_rates.csv'), usecols=cols)

# Load behavioral data and extract event onset and offset
behav_data_path = r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\behavioral_manifold.csv'
behavior = pd.read_csv(behav_data_path)
on = behavior.loc[behavior['CycleOn'], 'sessionwise_time'].values
off = behavior.loc[behavior['CycleOff'], 'sessionwise_time'].values
# on = behavior.loc[behavior['FR_StOn'], 'sessionwise_time'].values
# on = on[:-1]; off = on[1:]


################################# PETHs #######################################
# Get phase-locked PETH
peth_path = f'C:\\Users\\User\\Desktop\\Npx data analysis\\data\\{cell_type}_peth_global_phase.csv'
if os.path.exists(peth_path) and load_data:
    data = pd.read_csv(peth_path)
    print("File loaded successfully.")
else:
    print("File does not exist. Computing PETH...")
    ## TURN INTO A FUNCTION
    n_cycles = len(off)
    n_samples=300
    peth = np.zeros((n_samples, firing_rate.shape[-1]-1))
    for i in range(n_cycles):
        mask = (firing_rate['time'] > on[i]) & (firing_rate['time'] < off[i])
        fr_i = firing_rate[mask].iloc[:, 1:].values.T
        peth += np.array(interpolate(fr_i, n_samples=n_samples)).T
    peth /= n_cycles
    peth = pd.DataFrame(peth, columns=firing_rate.iloc[:, 1:].columns)

# Save
if save_data:
    peth.to_csv(f'C:\\Users\\User\\Desktop\\Npx data analysis\\data\\{cell_type}_peth_global_phase.csv')

# Z-score PETH
# peth_zs = peth.apply(zscore)
peth_zs = (peth - peth.mean())/ peth.std()

# Plot PETHs
## USE FUNCTION
plt.figure()
for unit in peth_zs.columns:
    plt.plot(np.linspace(0, 100, 300), peth_zs[unit], linewidth=2.5)
    plt.xlabel('Global phase (%)', fontsize=15)
    plt.ylabel('Firing rate (z-score)', fontsize=15)    
    plt.margins(0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Snakeplot of population PETHs
## TURN INTO A FUNCTION
preferred_phase = peth_zs.idxmax() # get preferred phase (index)
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(peth_zs.loc[:, preferred_phase.sort_values().index].T, cmap='viridis', ax=ax)
ax.yaxis.set_ticks([])
ax.set_xticks([0, 149, 299])
ax.set_xticklabels([0, 50, 100], fontsize=12, rotation = 0)
ax.set_xlabel('Global phase (%)', fontsize=15)
ax.set_ylabel(cell_type, fontsize=15)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(labelsize=12)
cbar.set_label('Firing rate (z-score)', fontsize=15)
plt.show()


################################## PCA ########################################
# Perform PCA
n_components = 5
pca = PCA(n_components=n_components)
pca_fit = pca.fit(peth_zs)
principal_components = pca.transform(peth_zs)
loadings = pca.components_

# Plot explained variance
## USE FUNCTION
plt.figure()
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.bar(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, color='lightgray')
plt.xlabel('Principal Components', fontsize=15)
plt.ylabel('Cumulative Explained Variance', fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Plot individual components against phase and their loadings
## USE FUNCTION
fig, ax = plt.subplots(n_components, 2, figsize=(8, 8))
for i in range(n_components):
    ax[i, 0].plot(np.linspace(0, 100, 300), principal_components[:, i], color='darkorange', linewidth=3)
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

    ax[i, 1].bar(np.arange(loadings.shape[-1]), loadings[i, :], color='blue')
    ax[i, 1].set_ylabel(f'Loadings PC{i+1}', fontsize=15)
    ax[i, 1].margins(0)
    ax[i, 1].tick_params(labelsize=12)
    ax[i, 1].spines['top'].set_visible(False)
    ax[i, 1].spines['right'].set_visible(False)
    if i == 2:
        ax[i, 1].set_xlabel(cell_type, fontsize=15)
    ax[i, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tight_layout()
plt.show()

# Plot 3D manifold
## USE FUNCTION
color_vector = np.linspace(0, 1, 300)
colors = cm.viridis(color_vector)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=colors, s=45)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_zlabel('PC3', fontsize=15)
cbar = plt.colorbar(scatter, ax=ax, label='Global phase (%)', shrink=0.5, ticks=[0, 1], pad=0.15)
cbar.ax.tick_params(labelsize=12)
cbar.ax.yaxis.label.set_fontsize(15)
cbar.set_ticklabels([0, 100])
plt.show()
