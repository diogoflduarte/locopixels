'''
script to analyze and create the figures for fens 2024 poster
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import CareyPlots
import cupy as cp
import npyx
import CareyEphys
import CareyUtils
import CareyBehavior
from tqdm import tqdm
import scipy.signal
cp.cuda.set_allocator(None)

# import CareyPhaser.phaser_utils as phaser_utils
from CareyPhaser.phaser import Phaser
from sklearn.preprocessing import StandardScaler, RobustScaler

CONFIDENCE_THRESHOLD_FOR_ANALYSIS = 3

batch_5_loc = r'X:\data\2022\BATCH5'
if os.getlogin() == 'diogo':
    batch_5_loc = r'X:\data\2022\BATCH5'
dataset_folder = os.path.join(batch_5_loc, 'recordings', 'VIV_23058', 'VIV_23058_S10_g1', 'kilosort4_catgt_tshift')
processing_folder = os.path.join(batch_5_loc, 'processing', 'VIV_23058', 'S10')
locopixels_folder = os.path.join(processing_folder, 'locopixels')


c4_results = os.path.join(batch_5_loc, dataset_folder, 'c4_results')
c4_output_files = ['cluster_confidence_ratio.tsv', 'cluster_model_votes.tsv',
                   'cluster_pred_probability.tsv', 'cluster_predicted_cell_type.tsv']
dfs = [pd.read_csv(os.path.join(c4_results, ff), sep='\t') for ff in c4_output_files]
c4  = pd.concat(dfs).groupby('cluster_id').first().reset_index()
cell_types = ['PkC_ss', 'MFB', 'GoC', 'MLI', 'PkC_cs']
cluster_info = pd.read_csv(os.path.join(dataset_folder, 'cluster_info.tsv'), delimiter='\t')

##
total_number_of_cells = c4['predicted_cell_type'].shape[0]
threshold_depth         = 900 # 900
threshold_confidence    = 0 #3

cortex_only = c4
indices_to_drop = []
for ii in range(total_number_of_cells):
    if cluster_info[cluster_info['cluster_id']==c4['cluster_id'][ii]].depth.values[0] < threshold_depth or \
            c4.confidence_ratio[ii] < threshold_confidence:
        indices_to_drop.append(ii)
cortex_only = cortex_only.drop(indices_to_drop)
cortex_only['predicted_cell_type'] = pd.Categorical(cortex_only['predicted_cell_type'], cell_types)

# ----------------------  histogram  --------------------------------------
plt.figure()
cell_types_barplot = sns.histplot(data=cortex_only, x='predicted_cell_type', hue='predicted_cell_type',
                                  palette=[sns.color_palette('bright')[0], sns.color_palette('bright')[2],
                                           sns.color_palette('bright')[7], sns.color_palette('bright')[7],
                                           sns.color_palette('bright')[7]],
                                  shrink=0.9, legend=False)
plt.xlabel('Cell types')
plt.show()

## filter units by cell type and confidence

mossy_fibers =      cortex_only[np.logical_and(cortex_only.predicted_cell_type=='MFB',
                           cortex_only.confidence_ratio>CONFIDENCE_THRESHOLD_FOR_ANALYSIS)].cluster_id.values
purkinje_cells =    cortex_only[np.logical_and(cortex_only.predicted_cell_type=='PkC_ss',
                           cortex_only.confidence_ratio>CONFIDENCE_THRESHOLD_FOR_ANALYSIS)].cluster_id.values
meta = npyx.read_metadata(dataset_folder)
n_timepoints = int(meta['highpass']['binary_byte_size'] / meta['highpass']['n_channels_binaryfile']/2)
time_array = cp.linspace(0, meta['recording_length_seconds'], n_timepoints)  # really hoping maxime doesn't fix the parsing bug before I finish the PhD

neurons = np.hstack((mossy_fibers, purkinje_cells))
firing_rates = []
for ii, unit in enumerate(tqdm(neurons)):
    spike_indices = npyx.spk_t.trn(dataset_folder, unit)
    fr, t = CareyEphys.get_sessionwise_firingrate_singleunit_binning_fullgpu(spike_indices, time_array)
    if ii == 0:
        firing_rates.append(t)
    firing_rates.append(fr)
data = cp.array(firing_rates).get().transpose()
cp._default_memory_pool.free_all_blocks()
df = pd.DataFrame(columns=(['time'] + neurons.tolist()), data=data)


## read behavior dataset and get global phase
behav = pd.read_csv(os.path.join(processing_folder, 'VIV_23058_S10_behavioral_descriptor.csv'))
# paw_positions = behav[['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']]
# paw_positions = paw_positions.interpolate().values
# paw_positions = scipy.signal.detrend(paw_positions, axis=0)

# block = CareyUtils.find_segments_below_threshold(behav['wheel_speed'].values, 0.3, 50*342)[0][0]

# scaler = RobustScaler().fit(paw_positions[block,:])
# phaser = Phaser(y=paw_positions[block].transpose())
# global_phase = phaser.phaserEval(paw_positions.transpose()).squeeze()
# global_phase = (global_phase + np.pi) % (2 * np.pi)
# behav['global_phase'] = global_phase

# CareyUtils.profile_function(phaser.phaserEval, scaler.transform(paw_positions).transpose())
latents_lds = np.load(os.path.join(locopixels_folder, 'full_behav_lds_3dim.npy'))
main_sig = scipy.signal.detrend(latents_lds[:,0])

phaser = Phaser(y=latents_lds[2229642:2234575,:].transpose())
global_phase = phaser.phaserEval(latents_lds.transpose()).squeeze()
global_phase = (global_phase + np.pi) % (2 * np.pi)
behav['global_phase'] = global_phase
wrap_indices = np.where(np.diff(global_phase) < -np.pi)[0] + 1
locomotor_cycle = np.zeros_like(global_phase)
for i in range(len(wrap_indices)):
    start = wrap_indices[i-1] if i > 0 else 0
    end = wrap_indices[i]
    locomotor_cycle[start:end] = i+1
behav['locomotor_cycle'] = locomotor_cycle

##