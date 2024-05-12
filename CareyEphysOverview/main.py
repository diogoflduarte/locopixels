# Import libraries
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import matplotlib.pyplot as plt
import npyx

os.chdir(r'C:\Users\User\Desktop\CareyEphysOverview')
import visualization as visual
from load_data import load_c4, load_ks

# Set paths
# EDIT HERE!
ksdir =r'Y:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25'
behav_data_path = r'Y:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv'
c4dir = r'Y:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25\c4_pythonresults'
save_path = r'Y:\figures\neuron_overview'

# Set params
# EDIT HERE!
# To do: move to params.py
mouse_id = 'VIV_23058'
session = '10'
save_plot = True
align = 'FR_StOn'
unit_quality = 'good'
window = [-200, 200]
lags = 80
bin_size = 10
trials = 2000
color_celltypes = {'PkC_ss': 'green', 
                   'PkC_cs': 'darkviolet', 
                   'MFB': 'darkorange',
                   'MLI': 'hotpink',
                   'GoC': 'blue',
                   'GrC': 'salmon',
                   'good': 'black',
                   'mua': 'darkgray',
                   'noise': 'darkgray'}

# Load neural data
meta = npyx.read_metadata(ksdir)
fs = int(meta['highpass']['sampling_rate'])
units = npyx.get_units(ksdir, quality=unit_quality)
cluster_info, spike_clusters, spike_trains, amplitudes = load_ks(ksdir)
depth_clusters = np.histogram(cluster_info['depth'], bins=np.arange(0, 3900, 100))

# Load cell types
cell_types = load_c4(c4dir)

# Load behavioral data and extract events
behav_data = pd.read_csv(behav_data_path)
events =  np.where(behav_data[align] == True)[0]
event_ts = behav_data['sessionwise_time'][events].values

# Loop through units
for unit in units:
    # Get cell type for unit
    if unit in cell_types['cluster_id'].values:
        cell_type = cell_types[cell_types['cluster_id'] == unit]['predicted_cell_type'].values[0]
        pred_prob = np.round(cell_types[cell_types['cluster_id'] == unit]['pred_probability'].values[0], 2)
        conf_ratio = np.round(cell_types[cell_types['cluster_id'] == unit]['confidence_ratio'].values[0], 2)
    else:
        cell_type = cluster_info[cluster_info['cluster_id'] == unit]['group'].values[0]
        pred_prob = ' '
        conf_ratio = ' '
    
    # Get spike timestamps, spikes number, mean firing rate, amplitude, peak channel and depth for unit
    spike_train = spike_trains[spike_clusters==unit].ravel()
    spike_ts = spike_train/fs
    spikes_num = len(spike_train)
    mean_fr = round(cluster_info[cluster_info['cluster_id']==unit]['fr'].values[0])
    peak_channel = cluster_info[cluster_info['cluster_id']==unit]['ch'].values[0]
    amplitude = amplitudes[spike_clusters==unit]
    depth_unit = cluster_info[cluster_info['cluster_id'] == unit]['depth'].values
    
    # Plot depth
    fig = plt.figure(figsize=(12, 15))
    plt.suptitle(f'mouse: {mouse_id} | session: {session} | unit: {unit} | cell type: {cell_type} (prob = {pred_prob}; conf = {conf_ratio})', fontsize = 14)
    gs = GridSpec(4, 4)
    ax_depth = fig.add_subplot(gs[0:2, 0])
    visual.plot_depth(depth_clusters, depth_unit, color_unit=color_celltypes[cell_type], ax=ax_depth, color='lightgray')
    npyx.mplp(ax=ax_depth, xlabel="Count", ylabel="Depth of neuropixel probe (um)", ticklab_s=10, title_s=12, axlab_s=12)

    # Plot raster
    ax_raster = fig.add_subplot(gs[0:2, 1:3])
    visual.plot_raster(spike_ts, event_ts, window=window, trials=trials, fs=fs, ax=ax_raster, color='dimgray', marker='.', s=1)
    npyx.mplp(ax=ax_raster, xlabel="Time around stance (ms)", ylabel="Strides", ticklab_s=10, title_s=12, axlab_s=12) 
    
    # Plot PETH
    ax_peth = fig.add_subplot(gs[2, 1:3])
    visual.plot_peth(spike_ts, event_ts, bin_size=bin_size, window=window, ax=ax_peth, color='red', linewidth=2)   
    npyx.mplp(ax=ax_peth, xlabel="Time around event (ms)", ylabel="Firing Rate (Hz)", ticklab_s=10, title_s=12, axlab_s=12)
    
    # Plot waveform
    ax_wvf = fig.add_subplot(gs[0, 3])
    visual.plot_wvf(ksdir, unit, channel=peak_channel, fs=fs, n_wvf=300, color='k', ax=ax_wvf, linewidth=2)
    npyx.mplp(fig=fig, ax=ax_wvf, title=f"Channel {peak_channel}", xlabel="Time (ms)", ylabel="Voltage (uV)", ticklab_s=10, title_s=12, axlab_s=12)
    
    # Plot inter-spike interval
    ax_isi = fig.add_subplot(gs[1, 3])
    visual.plot_isi(spike_train, fs=fs, ax=ax_isi, color='dimgray', bins=100)
    npyx.mplp(fig=fig, ax=ax_isi, title=f'spikes = {spikes_num} \n firing rate = {mean_fr} Hz', xlabel="ISI (ms)", ylabel="Spike count", ticklab_s=10, title_s=12, axlab_s=12)
    
    # Plot auto-correlogram
    ax_acg = fig.add_subplot(gs[2, 3])
    visual.plot_acg(ksdir, unit, bin_size=bin_size/100, win_size=lags, fs=fs, normalize='Hertz', ax=ax_acg, color=color_celltypes[cell_type])
    npyx.mplp(fig=fig, ax=ax_acg, xlabel="Lag (ms)", ylabel="Firing rate (Hz)", ticklab_s=10, title_s=12, axlab_s=12)
    
    # Plot amplitude
    ax_raw = fig.add_subplot(gs[3, 0:3])
    visual.plot_raw(amplitude, spike_train, fs=fs, ax=ax_raw)
    npyx.mplp(fig=fig, ax=ax_raw, xlabel="Time (m)", ylabel="Voltage (uV)", ticklab_s=10, title_s=12, axlab_s=12)
    
    plt.tight_layout()
    visual.image_saver(save_path, f'{cell_type}', f'unit{unit}')
    plt.close()