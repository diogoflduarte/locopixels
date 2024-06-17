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
from CareyConstants import *
from tqdm import tqdm
import scipy.signal
import sklearn.decomposition
import sklearn.preprocessing
cp.cuda.set_allocator(None)

# import CareyPhaser.phaser_utils as phaser_utils
from CareyPhaser.phaser import Phaser
from sklearn.preprocessing import StandardScaler, RobustScaler

CONFIDENCE_THRESHOLD_FOR_ANALYSIS = 5
CYCLE_VAR = 'locomotor_cycle'
PHASE_VAR = 'global_phase'
RECOMPUTE_CYCLE_VAR = False
RECOMPUTE_FIRING_RATES_IN_PHASE = False
MIN_STRIDE_DURATION = 0.10 # s
MAX_STRIDE_DURATION = 0.5
PHASE_NPTS = 200
EXCLUDE_NEURONS = [189, 322, 469, 521, 523]

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
cell_types_barplot = sns.histplot(data=cortex_only, x='predicted_cell_type', hue='predicted_cell_type',
                                  palette=[sns.color_palette('bright')[0], sns.color_palette('bright')[2],
                                           sns.color_palette('bright')[7], sns.color_palette('bright')[7],
                                           sns.color_palette('bright')[7]],
                                  shrink=0.9, legend=False)
plt.xlabel('Cell types')
plt.show()

## filter units by cell type and confidence
print('filtering units by confidence')
mossy_fibers =      cortex_only[np.logical_and(cortex_only.predicted_cell_type=='MFB',
                           cortex_only.confidence_ratio>CONFIDENCE_THRESHOLD_FOR_ANALYSIS)].cluster_id.values
purkinje_cells =    cortex_only[np.logical_and(cortex_only.predicted_cell_type=='PkC_ss',
                           cortex_only.confidence_ratio>CONFIDENCE_THRESHOLD_FOR_ANALYSIS)].cluster_id.values
meta = npyx.read_metadata(dataset_folder)
n_timepoints = int(meta['highpass']['binary_byte_size'] / meta['highpass']['n_channels_binaryfile']/2)
time_array = cp.linspace(0, meta['recording_length_seconds'], n_timepoints)  # really hoping maxime doesn't fix the parsing bug before I finish the PhD

mossy_fibers = mossy_fibers[np.isin(mossy_fibers, EXCLUDE_NEURONS, invert=True)]
purkinje_cells = purkinje_cells[np.isin(purkinje_cells, EXCLUDE_NEURONS, invert=True)]
neurons = np.hstack((mossy_fibers, purkinje_cells))

print('Compiling firing rates...')
firing_rates = []
for ii, unit in enumerate(tqdm(neurons)):
    spike_indices = npyx.spk_t.trn(dataset_folder, unit)
    fr, t = CareyEphys.get_sessionwise_firingrate_singleunit_binning_fullgpu(spike_indices, time_array,
                                                                             bwidth=10, gaussdev=0.010,
                                                                             fs=CareyConstants.DEF_NPX_FS,
                                                                             binnedoutput=True)
    if ii == 0:
        firing_rates.append(t)
    firing_rates.append(fr)
data = cp.array(firing_rates).get().transpose()
cp._default_memory_pool.free_all_blocks()
df = pd.DataFrame(columns=(['time'] + neurons.tolist()), data=data)
print('Done')

## read behavior dataset and get global phase
print('Reading behavior dataset...')
behav = pd.read_csv(os.path.join(processing_folder, 'VIV_23058_S10_behavioral_descriptor.csv'))
print('Done')
if RECOMPUTE_CYCLE_VAR:
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

    __, behav['FR_SwPh'], behav['FR_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FR_SwOn'], behav['FR_StOn'], usegpu=0)
    __, behav['FR_StPh'], behav['FR_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FR_StOn'], behav['FR_SwOn'], usegpu=0)
    __, behav['HR_SwPh'], behav['HR_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HR_SwOn'], behav['HR_StOn'], usegpu=0)
    __, behav['HR_StPh'], behav['HR_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HR_StOn'], behav['HR_SwOn'], usegpu=0)
    __, behav['FL_SwPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FL_SwOn'], behav['FL_StOn'], usegpu=0)
    __, behav['FL_StPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FL_StOn'], behav['FL_SwOn'], usegpu=0)
    __, behav['HL_SwPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HL_SwOn'], behav['HL_StOn'], usegpu=0)
    __, behav['HL_StPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HL_StOn'], behav['HL_SwOn'], usegpu=0)

## interpolate neural data to behavior
print('interpolating neural data to behavior time...')
neural_data_time = df['time'].values
behaviorald_time = behav['sessionwise_time'].values
downsampled_neurons = []
for neuron in df.columns[1:]:
    downsampled_neurons.append(np.interp(behaviorald_time, neural_data_time, df[neuron]))
downsampled_neurons.insert(0, behaviorald_time)
df = pd.DataFrame(columns=df.columns, data=np.array(downsampled_neurons).transpose())
columns_to_copy = ['locomotor_cycle', 'global_phase', 'FR_SwPh', 'FR_Sw_Stride', 'FR_StPh', 'FR_St_Stride']
for col in columns_to_copy:
    df[col] = behav[col]
print('Done')
## reorganize by locomotor cycles opf global phase or FR paw or any other label that's valid in df
if RECOMPUTE_FIRING_RATES_IN_PHASE:
    print('Reorganizing data into strides by phase')
    phase_array = np.linspace(0, 1, PHASE_NPTS, endpoint=True)
    stride_counts = df[CYCLE_VAR].value_counts()
    dt = np.median(np.diff(df['time']))
    behavioral_columns = ['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz',
                          'wheel_distance', 'wheel_speed',
                          'FR_SwOn', 'FR_StOn', 'HR_SwOn', 'HR_StOn', 'FL_SwOn', 'FL_StOn', 'HL_SwOn', 'HL_StOn',
                          'global_phase', 'locomotor_cycle', 'FR_SwPh', 'FR_Sw_Stride', 'FR_StPh',
                          'FR_St_Stride', 'HR_SwPh', 'HR_Sw_Stride', 'HR_StPh', 'HR_St_Stride', 'FL_SwPh',
                          'FL_Sw_Stride', 'FL_StPh', 'HL_SwPh', 'HL_StPh']
    column_list = ['time'] + neurons.tolist() + behavioral_columns
    firing_rates_phase_sessionwise = pd.DataFrame(columns=column_list)
    for idx, stride in enumerate(tqdm(np.arange(1, df[CYCLE_VAR].max()+1))):
        # Filter groups by stride
        group = df[df[CYCLE_VAR] == stride]
        behav_group = behav[behav[CYCLE_VAR] == stride]
        # check if stride is within the right duration
        stride_duration = group.time.max()-group.time.min()
        if not (stride_duration >= MIN_STRIDE_DURATION and stride_duration <= MAX_STRIDE_DURATION) or\
                np.linalg.matrix_rank(df[df['locomotor_cycle']==stride][neurons].values) < len(neurons):
            continue

        # Interpolate firing rates if 'time' is within interval
        firing_rates = group[neurons].values
        firing_rates_along_phase = []
        phase_array_this_cycle = np.linspace(0, 1, firing_rates.shape[0], endpoint=True)
        for nn in range(firing_rates.shape[1]):
            firing_rates_along_phase.append(np.interp(phase_array, phase_array_this_cycle, firing_rates[:, nn]))
        # firing_rates_along_phase = np.array(firing_rates_along_phase).transpose()
        firing_rates_along_phase.insert(0, np.interp(phase_array, phase_array_this_cycle, group['time'].values))
        for feat in behavioral_columns:
            if behav[feat].dtype == 'bool':
                firing_rates_along_phase.append(
                    CareyUtils.boolean_interp(phase_array, phase_array_this_cycle, behav_group[feat].values))
            else:
                firing_rates_along_phase.append(np.interp(phase_array, phase_array_this_cycle, behav_group[feat].values))


        firing_rates_phase_sessionwise = firing_rates_phase_sessionwise.append(pd.DataFrame(
                                            columns=firing_rates_phase_sessionwise.columns,
                                            data=np.array(firing_rates_along_phase).transpose()),
                                            ignore_index=True)
else:
    print('loading firing_rates_phase_sessionwise...')
    firing_rates_phase_sessionwise = pd.read_csv(os.path.join(locopixels_folder, 'firing_rates_phase_sessionwise.csv'))
    print('done')

# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3',
#                                                             colorby='global_phase', pop='locomotor_cycle', linewidth=0,
#                                                             opacity=0.3)
##
behav_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(firing_rates_phase_sessionwise[['FRx', 'FRy', 'FRz',
                                                                                                    'HRx', 'HRy', 'HRz',
                                                                                                    'FLx', 'FLy', 'FLz',
                                                                                                    'HLx', 'HLy', 'HLz']])
firing_rates_phase_sessionwise[['bPCA1', 'bPCA2', 'bPCA3']] = behav_pca[:,:3]
mossy_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[mossy_fibers.astype(str)])
firing_rates_phase_sessionwise[['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']] = mossy_pca[:,:6]
pawcolors =  [[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]]
app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3',
                                                            colorby=['FR_StOn', 'HR_StOn', 'FL_StOn', 'HL_StOn'],
                                                            pop='locomotor_cycle', linewidth=0, opacity=1,
                                                            custom_colors=pawcolors)
