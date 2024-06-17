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
app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3', pop='global_phase',
                                                            colorby=['FR_StOn', 'HR_StOn', 'FL_StOn', 'HL_StOn'], linewidth=0,
                                                            custom_colors=pawcolors,
                                                            DEF_SIZE=5, POP_SIZE=10)
