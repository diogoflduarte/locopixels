import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import npyx

os.chdir(r'C:\Users\User\Desktop\Ephys Visualization Tool')
import plotting_routine as pf


neural_data_path = r'Y:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25'
c4_data_path = r'Y:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\c4_pythonresults'
behav_data_path = r'C:\Users\User\Desktop\Ephys Visualization Tool\LocoNPX\VIV_23058_S10_behavioral_descriptor.csv'
save_path = r'C:\Users\User\Desktop\Npx data analysis\Figures\Stance-locked neural activity'
folder_name = ''

save_plot = True
unit_quality = 'good'
window = [-125, 125] #ms
epoch = [10*60, 15*60] #min
bin_size = 10
paws = ['FR', 'HR', 'FL', 'HL']
color_paws = ['red', 'magenta', 'blue', 'cyan']

meta = npyx.read_metadata(neural_data_path)
fs = meta['highpass']['sampling_rate']
units = npyx.get_units(neural_data_path, quality=unit_quality)

c4_classif = pf.plot_c4_classif(c4_data_path)

behav_data = pd.read_csv(behav_data_path)
ts_behav = behav_data['sessionwise_time'].values
ts_events = {}
for p, paw in enumerate(paws):
    align = f'{paw}_StOn'
    events = np.where(behav_data[align] == True)[0]
    ts = ts_behav[events]
    ts = ts[np.where((ts >= epoch[0]) & (ts <= epoch[1]))]
    ts_events[paw] = ts

for unit in units:
    spike_train = npyx.trn(neural_data_path, unit)
    # t_preprocessed = npyx.trn_filtered(neural_data_path, unit, plot_debug=False)
    spike_ts = spike_train/fs
    spike_ts = spike_ts[np.where((spike_ts >= epoch[0]) & (spike_ts <= epoch[1]))]
    
    if unit in c4_classif['cluster_id'].values:
        unit_classif = c4_classif[c4_classif['cluster_id'] == unit]
        cell_type = unit_classif['predicted_cell_type'].values[0]
        pred_prob = np.round(unit_classif['pred_probability'].values[0], 2)
        conf = np.round(unit_classif['confidence_ratio'].values[0], 2)
    else:
        cell_type = 'Other'
        pred_prob = ' '
        conf = ' '
    
    fig, ax = plt.subplots(2, 4, figsize=(4*3, 2*3))
    for p, paw in enumerate(paws):
        # RASTER
        pf.plot_raster(spike_ts, ts_events[paw], window=window, fs=fs, ax=ax[0, p], marker='.', s=1)
        npyx.mplp(ax=ax[0, p], xlabel="Time around stance (ms)", ylabel="Strides", ticklab_s=14, title_s=14, axlab_s=14)
        
        # PETH
        pf.plot_peth(spike_ts, ts_events[paw], bin_size=bin_size, window=window, color=color_paws[p], ax=ax[1, p], linewidth=2)   
        npyx.mplp(ax=ax[1, p], xlabel="Time around stance (ms)", ylabel="Firing Rate (Hz)", ticklab_s=14, title_s=14, axlab_s=14)
                
    plt.suptitle(f"Unit {unit} (cell type: {cell_type}; prob={pred_prob}; conf_ratio={conf})", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    if save_plot:
        pf.image_saver(save_path, cell_type, f'unit{unit}')
        plt.close()