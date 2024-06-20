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
from tqdm import tqdm
from CareyConstants import CareyConstants
import CareyPlots
import CareyEphys
import seaborn as sns
import npyx
import cupy as cp

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
save_plot = False
load_data = False
cell_type = 'Purkinje Cells'
# cell_type = 'Mossy Fibers'
n_cycles = 5000     # locomotor cycles
n_samples = 100    # 300 # number of samples for the interpolate phase space
min_max_std = True
remove_zero_firing_entries = True  # todo: also check this one
how_to_get_firing_rates = 'compute' # 'load' or 'compute'
kernel = 0.005 # seconds

## ############################## LOAD DATA ####################################
# Load neural data
neural_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels'
if os.getlogin() == 'diogo':
    neural_data_path = r'X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels'
    dataset_folder = r'X:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\kilosort4_catgt_tshift'
firing_rates_file = os.path.join(neural_data_path, 'sessionwise_firing_rates_fast_20ms_kern.csv')


# Load behavioral data and extract event onset and offset
behav_data_path = r'C:\Users\User\Desktop\behavior_analysis\behavioral_manifold\behavioral_manifold.csv'
if os.getlogin() == 'diogo':
    behav_data_path = r"X:\data\2022\BATCH5\processing\VIV_23058\S10\Behavioral manifold\behavioral_manifold.csv"


if cell_type == 'Purkinje Cells':
    selected_cells = ['time', 91, 111, 115, 151, 198, 226, 241, 246, 259, 400, 402,
                      415, 88, 209, 266, 274, 382] 
elif cell_type == 'Mossy Fibers':
    selected_cells = ['time', 317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467,
                    479, 500, 501, 503, 507, 601, 602, 604, 611, 613]
mossy_and_purkinke = [91, 111, 115, 151, 198, 226, 241, 246, 259, 400, 402, 415, 88, 209, 266, 274, 382, # Pkj
                      317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467, 479, 500, 501, 503, 507, 601, 602, 604, 611, 613] # mossy

cols = [str(xx) for xx in mossy_and_purkinke]
print(f'Reading FRs: {firing_rates_file}')

if how_to_get_firing_rates == 'load':
    firing_rate = pd.read_csv(firing_rates_file, usecols=cols)
else:
    print('Compiling firing rates...')
    meta = npyx.read_metadata(dataset_folder)
    n_timepoints = int(meta['highpass']['binary_byte_size'] / meta['highpass']['n_channels_binaryfile'] / 2)
    time_array = cp.linspace(0, meta['recording_length_seconds'],
                             n_timepoints)  # really hoping maxime doesn't fix the parsing bug before I finish the PhD
    firing_rates = []
    for ii, unit in enumerate(tqdm(selected_cells[1:])):
        spike_indices = npyx.spk_t.trn(dataset_folder, unit)
        fr, t = CareyEphys.get_sessionwise_firingrate_singleunit_binning_fullgpu(spike_indices, time_array,
                                                                                 bwidth=10, gaussdev=kernel,
                                                                                 fs=CareyConstants.DEF_NPX_FS,
                                                                                 binnedoutput=True)
        if ii == 0:
            firing_rates.append(t)
        firing_rates.append(fr)
    data = cp.array(firing_rates).get().transpose()
    cp._default_memory_pool.free_all_blocks()
    firing_rate = pd.DataFrame(columns=selected_cells, data=data)
    print('Done')


print(f'Reading behavior file: {behav_data_path}')
behavior = pd.read_csv(behav_data_path)
on = behavior.loc[behavior['CycleOn'], 'sessionwise_time'].values
off = behavior.loc[behavior['CycleOff'], 'sessionwise_time'].values
# on = behavior.loc[behavior['FR_StOn'], 'sessionwise_time'].values
# on = on[:-1]; off = on[1:]\
speed = behavior['wheel_speed']

################################# PETHs #######################################
print('Computing population firing rates per locomotion cycle (PETHs) ')
peth = []
for i in tqdm(range(n_cycles)):
    mask = (firing_rate['time'] >= on[i]) & (firing_rate['time'] <= off[i])
    peth_i = firing_rate[mask].iloc[:, 1:].values
    peth_i = np.array(interpolate(peth_i.T, n_samples=n_samples)).T
    peth.append(peth_i)
peth = np.concatenate(peth, axis=0)
peth = pd.DataFrame(peth, columns=firing_rate.iloc[:, 1:].columns)

trial_id = np.concatenate([np.ones(n_samples)*i for i in range(n_cycles)])

# Save
if save_data:
    peth.to_csv(f'C:\\Users\\User\\Desktop\\Npx data analysis\\data\\{cell_type}_peth_global_phase_cycles.csv')

# Standardize
# todo: check if this meanmax standardization makes any difference

# speed_mean = np.zeros(n_cycles)
# for i in range(n_cycles):
#         mask = (behavior['sessionwise_time'] >= on[i]) & (behavior['sessionwise_time'] <= off[i])
#         speed_mean[i] = np.mean(speed[mask].values)

# # Plot PETHs
# ## USE FUNCTION
# plt.figure()
# for unit in peth_zs.columns:
if remove_zero_firing_entries:
    peth = peth.loc[:, (peth != 0).any(axis=0)]
if min_max_std:
    peth_min = peth.min().values
    peth_max = peth.max().values
    peth_zs = (peth - peth_min) / (peth_max - peth_min)
else:
    peth_zs = peth

# Get cycles average speed
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
scatter = ax.scatter(avg_pc[:, 0], avg_pc[:, 1], avg_pc[:, 2], c=np.linspace(0, 1, n_samples), cmap=cmap, norm=norm, s=45)
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
    ax[i, 0].plot(np.linspace(0, 100, n_samples), avg_pc[:, i], color='darkblue', linewidth=4)
    for j in range(selected_cycles):
        ax[i, 0].plot(np.linspace(0, 100, n_samples), pc_i[j][:, i], color='lightgray', linewidth=1, alpha=0.5)
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

##

total_number_of_cycles = int((peth.shape[0] / n_samples))
phase_array = np.tile(np.linspace(0, 1, n_samples), total_number_of_cycles).T

dat = np.hstack((peth, trial_id[:, None], phase_array[:, None])) # todo: missing sessionwise time here
neural = pd.DataFrame(columns=(selected_cells[1:] + ['locomotor_cycle', 'phase']), data=dat)

# run 6 dim pca on these neurons
if cell_type == 'Mossy Fibers':
    prefix = 'm'
elif cell_type == 'Purkinje Cells':
    prefix = 'p'

n_comps = 6
pc_names = [(prefix + 'PC' + str(ii)) for ii in np.arange(1, n_comps + 1)]
neural_pca = PCA(n_components=n_comps).fit(peth)
neural[pc_names] = neural_pca.transform(peth)

##
subset = neural[neural['locomotor_cycle'].isin(np.arange(0, total_number_of_cycles + 1, 50))]
ax = CareyPlots.multicoloredline_3d(subset, pc_names[0], pc_names[1], pc_names[2], 'phase', trials='locomotor_cycle', lw=0.5,
                                    cmap=cmocean.cm.phase, alpha=0.5)

##
app = CareyPlots.twinplots(subset, pc_names[0], pc_names[1], pc_names[2], pc_names[3], pc_names[4], pc_names[5],
                           colorby='phase', pop='locomotor_cycle', linewidth=0,
                           opacity=0.3, show_grid=True, show_background=False, DEF_SIZE=1, POP_SIZE=20)
