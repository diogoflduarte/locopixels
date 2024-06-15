import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'C:\Users\User\Desktop\neural data analysis')
from neuro_tools import peth_phase
from utils import normalize, image_saver
from neuro_visual import plot_peth, plot_peth_popul


# Load behavioral data and extract stride onset and offset
behav_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\Behavioral manifold\behavioral_manifold.csv'
behavioral_data = pd.read_csv(behav_data_path)
trial_boundaries = np.array([behavioral_data[behavioral_data['CycleOn']].iloc[:-1].index, 
                    behavioral_data[behavioral_data['CycleOff']].index]).T
global_phase = behavioral_data[['global phase', 'sessionwise_time']].values 
global_phase[:, 0] =  global_phase[:, 0] / (2*np.pi)
fs_cam = 430
wheel_speed = behavioral_data['wheel_speed']
wheel_speed_strides = np.array([np.mean(wheel_speed[start:end+1]) for start, end in trial_boundaries])
bins = np.percentile(wheel_speed_strides, np.arange(0, 110, 10))
bin_id = np.digitize(wheel_speed_strides, bins)-1
num_bins = len(bins)-1


# Load neural data
cell_type = 'Mossy Fibers'
neural_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels'
if cell_type == 'Purkinje cells':
    selected_cells = ['time', 91, 111, 115, 151, 198, 226, 241, 246, 259, 400, 402,
                      415, 88, 209, 266, 274, 382] 
elif cell_type == 'Mossy Fibers':
    selected_cells = ['time', 317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467, 
                    479, 500, 501, 503, 507, 601, 602, 604, 611, 613] 
neural_data = pd.read_csv(os.path.join(neural_data_path, 'sessionwise_firing_rates.csv'), usecols=[str(xx) for xx in selected_cells])


# Compute PETH in phase
bin_size = 0.05
phase_bins = np.arange(0, 1 + bin_size, bin_size)
peth, firing_rate_strides, global_phase_strides = peth_phase(neural_data, global_phase, trial_boundaries, fs_cam, phase_bins, verbose=True, selected_trials=None)


# Plot
plot_data = True
save_plot = True
num_neurons = len(selected_cells)-1
save_path = r'C:\Users\User\Desktop\peth_global_phase\speed'
if plot_data:
    # Single neurons by speed
    for i in range(3):
        ax = plot_peth(peth[i], peth_color=sns.color_palette("Reds", 5),
                      xlabel='Global phase (%)', ylabel=['Strides (sorted)', None], xticks=[0, 10, peth.shape[-1]], 
                      xticklabels=['0', '50', '100'], yticks=[0, peth.shape[1]], yticklabels=['Slow', 'Fast'],
                      sort_var=wheel_speed_strides, percent=np.arange(0, 120, 20), linewidth=3)
        if save_plot:        
            image_saver(save_path, f'{cell_type}', f'Unit{selected_cells[i+1]}')
            plt.close()

    # Population
    peth_mean = normalize(np.nanmean(peth, axis=1), norm_method='min-max') 
    peak_idx = np.argmax(peth_mean, axis=1)
    plot_peth_popul(peth_mean, xlabel='Global phase (%)', xticks=[0, 10, peth.shape[-1]], 
                    xticklabels=['0', '50', '100'], sort_var=peak_idx)  
    if save_plot:        
        image_saver(save_path, f'{cell_type}', 'Population')
        plt.close()
        
        
# Save
save_data = True
if save_data:
    data = {'PETH': peth, 'Neuron ID': selected_cells[1:], 'Firig rate strides': firing_rate_strides, 'Global phase strides interpolated': global_phase_strides}
    neural_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\peth_global_phase.npy'
    with open(neural_data_path, 'wb') as file:
        pickle.dump(data, file)