'''
script to analyze and create the figures for fens 2024 poster
'''

import os
import numpy as np
import random
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
import cmcrameri
import cmocean
cp.cuda.set_allocator(None)
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

# import CareyPhaser.phaser_utils as phaser_utils
from CareyPhaser.phaser import Phaser
from sklearn.preprocessing import StandardScaler, RobustScaler

## ##################### INITIALIZE SOME VARIABLES #############################
save_data = False
save_plot = False
load_data = False
# cell_type = 'Purkinje Cells'
cell_type = 'Mossy Fibers'
n_cycles = 5000     # locomotor cycles
n_samples = 200    # 300 # number of samples for the interpolate phase space
min_max_std = True # True
remove_zero_firing_entries = True  # todo: also check this one
how_to_get_firing_rates = 'compute' # 'load' or 'compute'
kernel = 0.010 # seconds

PURKINJE_COLOR      = '#6F6AB0'
MOSSY_COLOR         = '#DA8529'
LOCOMOTION_COLOR    = '#38B666'

CONFIDENCE_THRESHOLD_FOR_ANALYSIS = 5
CYCLE_VAR = 'locomotor_cycle'
PHASE_VAR = 'FR_StPh'   # stance to stance, maybe this is the wrong naming. but they are from FR stance (to swing )to FR stance
RECOMPUTE_CYCLE_VAR = True
RECOMPUTE_FIRING_RATES_IN_PHASE = True
MIN_STRIDE_DURATION = 0.10 # s
MAX_STRIDE_DURATION = 0.2
PHASE_NPTS = 100
MIN_STRIDE_SPEED = 0.35

batch_5_loc = r'X:\data\2022\BATCH5'
if os.getlogin() == 'diogo':
    batch_5_loc = r'X:\data\2022\BATCH5'
dataset_folder = os.path.join(batch_5_loc, 'recordings', 'VIV_23058', 'VIV_23058_S10_g1', 'kilosort4_catgt_tshift')
processing_folder = os.path.join(batch_5_loc, 'processing', 'VIV_23058', 'S10')
locopixels_folder = os.path.join(processing_folder, 'locopixels')
behav_manifold_folder =  r"X:\data\2022\BATCH5\processing\VIV_23058\S10\Behavioral manifold"
figures_folder = r'G:\My Drive\careylab\conferences\FENS_2024\figures'

c4_results = os.path.join(batch_5_loc, dataset_folder, 'c4_results')
c4_output_files = ['cluster_confidence_ratio.tsv', 'cluster_model_votes.tsv',
                   'cluster_pred_probability.tsv', 'cluster_predicted_cell_type.tsv']
dfs = [pd.read_csv(os.path.join(c4_results, ff), sep='\t') for ff in c4_output_files]
c4  = pd.concat(dfs).groupby('cluster_id').first().reset_index()
cell_types = ['PkC_ss', 'MFB', 'GoC', 'MLI', 'PkC_cs']
cluster_info = pd.read_csv(os.path.join(dataset_folder, 'cluster_info.tsv'), delimiter='\t')


## read behavior dataset and get global phase
print('Reading behavior dataset...')
# behav = pd.read_csv(os.path.join(processing_folder, 'VIV_23058_S10_behavioral_descriptor.csv'))
behav = pd.read_hdf(os.path.join(behav_manifold_folder, 'behavioral_manifold.h5'))
print('Done')



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

# cortex_only[np.logical_and(cortex_only['predicted_cell_type']=='PkC_ss', cortex_only['confidence_ratio'] > 2.5)]['cluster_id'].values
purkinje_cells = [151, 185, 186, 198, 203, 213, 225, 226, 236, 243, 244, 245, 246,
                    263, 313, 400, 402, 408, 415, 423]
mossy_fibers = [317, 322, 351, 354, 367, 393, 421, 453, 457, 467,
                    479, 500, 501, 503, 507, 601, 602, 604, 611, 613]
print(f'Starting with {len(mossy_fibers)} mossy fibers and {len(purkinje_cells)} Purkinje cells.\n')

purkinje_cells = [item for item in purkinje_cells if item in cortex_only.cluster_id.values.tolist()]
mossy_fibers = [item for item in mossy_fibers if item in cortex_only.cluster_id.values.tolist()]
print(f'Cortex selected: {len(mossy_fibers)} mossy fibers and {len(purkinje_cells)} Purkinje cells.\n')

neurons = mossy_fibers + purkinje_cells

## ----------------------  histogram  --------------------------------------
cell_types_barplot = sns.histplot(data=cortex_only, x='predicted_cell_type', hue='predicted_cell_type',
                                  palette=['#6F6AB0', '#DA8529',
                                           sns.color_palette('bright')[7], sns.color_palette('bright')[7],
                                           sns.color_palette('bright')[7]],
                                  shrink=0.9, legend=False, edgecolor='k')
plt.xlabel('Cell types')
plt.show()

## filter units by cell type and confidence
meta = npyx.read_metadata(dataset_folder)
n_timepoints = int(meta['highpass']['binary_byte_size'] / meta['highpass']['n_channels_binaryfile']/2)
time_array = cp.linspace(0, meta['recording_length_seconds'], n_timepoints)  # really hoping maxime doesn't fix the parsing bug before I finish the PhD


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
df = pd.DataFrame(columns=(['time'] + neurons), data=data)
print('Done')


##

__, behav['FR_SwPh'], behav['FR_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FR_SwOn'], behav['FR_StOn'], usegpu=0)
__, behav['FR_StPh'], behav['FR_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FR_StOn'], behav['FR_SwOn'], usegpu=0)
__, behav['HR_SwPh'], behav['HR_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HR_SwOn'], behav['HR_StOn'], usegpu=0)
__, behav['HR_StPh'], behav['HR_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HR_StOn'], behav['HR_SwOn'], usegpu=0)
__, behav['FL_SwPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FL_SwOn'], behav['FL_StOn'], usegpu=0)
__, behav['FL_StPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['FL_StOn'], behav['FL_SwOn'], usegpu=0)
__, behav['HL_SwPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HL_SwOn'], behav['HL_StOn'], usegpu=0)
__, behav['HL_StPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav['HL_StOn'], behav['HL_SwOn'], usegpu=0)

# behav['locomotor_cycle'] = behav['FR_St_Stride']
behav['phase']           = behav[PHASE_VAR]


on = behav.loc[behav['CycleOn'], 'sessionwise_time'].values
off = behav.loc[behav['CycleOff'], 'sessionwise_time'].values
speed = behav['wheel_speed']

behav['locomotor_cycle'] = 0
for i in tqdm(range(n_cycles)):
    mask = (behav['sessionwise_time'] > on[i]) & (behav['sessionwise_time'] < off[i])
    behav['locomotor_cycle'][mask] = i+1



## interpolate neural data to behavior
print('interpolating neural data to behavior time...')
neural_data_time = df['time'].values
behaviorald_time = behav['sessionwise_time'].values
downsampled_neurons = []
for neuron in df.columns[1:]:
    downsampled_neurons.append(np.interp(behaviorald_time, neural_data_time, df[neuron]))
downsampled_neurons.insert(0, behaviorald_time)
df = pd.DataFrame(columns=df.columns, data=np.array(downsampled_neurons).transpose())
columns_to_copy = ['locomotor_cycle', PHASE_VAR, 'FR_SwPh', 'FR_Sw_Stride', 'FR_StPh', 'FR_St_Stride']
for col in columns_to_copy:
    df[col] = behav[col]
print('Done')
## reorganize by locomotor cycles opf global phase or FR paw or any other label that's valid in df
stride_durations = (df.groupby('locomotor_cycle').max()-df.groupby('locomotor_cycle').min()).time.values

if RECOMPUTE_FIRING_RATES_IN_PHASE:
    print('Reorganizing data into strides by phase')
    phase_array = np.linspace(0, 1, PHASE_NPTS, endpoint=False)
    stride_counts = df[CYCLE_VAR].value_counts()
    dt = np.median(np.diff(df['time']))
    behavioral_columns = ['FRx', 'HRx', 'FLx', 'HLx',
                          'wheel_speed',
                          'FR_SwOn', 'FR_StOn', 'HR_SwOn', 'HR_StOn', 'FL_SwOn', 'FL_StOn', 'HL_SwOn', 'HL_StOn',
                          'phase', 'locomotor_cycle', 'FR_SwPh', 'FR_Sw_Stride', 'FR_StPh',
                          'FR_St_Stride', 'HR_SwPh', 'HR_Sw_Stride', 'HR_StPh', 'HR_St_Stride', 'FL_SwPh',
                          'FL_Sw_Stride', 'FL_StPh', 'HL_SwPh', 'HL_StPh', 'PC1', 'PC2', 'PC3']
    column_list = ['time'] + neurons + behavioral_columns
    firing_rates_phase_sessionwise = pd.DataFrame(columns=column_list)
    stride_counter = 0
    enough_strides = 500 #500
    # with tqdm(total=df[CYCLE_VAR].max()) as pbar:  # enough_strides
    for idx, stride in enumerate(tqdm(np.arange(1, df[CYCLE_VAR].max()+1))):
        # Filter groups by stride
        group = df[df[CYCLE_VAR] == stride]
        behav_group = behav[behav[CYCLE_VAR] == stride]
        # check if stride is within the right duration
        stride_duration = group.time.max()-group.time.min()
        stride_speed    = behav_group.wheel_speed.mean()

        # pbar.update(idx)

        if (stride_speed < MIN_STRIDE_SPEED):
            continue
        if stride_counter >= n_cycles:
            continue
        if not (stride_duration >= MIN_STRIDE_DURATION and stride_duration <= MAX_STRIDE_DURATION):
            continue
        # if np.linalg.matrix_rank(df[df['locomotor_cycle']==stride][neurons].values) < len(neurons):
        #     continue
        stride_counter += 1
        # print(f'found {stride_counter} strides\n')
        # pbar.update(stride_counter)

        # Interpolate firing rates if 'time' is within interval
        firing_rates = group[neurons].values
        firing_rates_along_phase = []
        phase_array_this_cycle = np.linspace(0, 1, firing_rates.shape[0], endpoint=False)
        for nn in range(firing_rates.shape[1]):
            firing_rates_along_phase.append(np.interp(phase_array, phase_array_this_cycle, firing_rates[:, nn]))
        # firing_rates_along_phase = np.array(firing_rates_along_phase).transpose()
        firing_rates_along_phase.insert(0, np.interp(phase_array, phase_array_this_cycle, group['time'].values))
        for feat in behavioral_columns:
            if behav[feat].dtype == 'bool':
                firing_rates_along_phase.append(
                    CareyUtils.boolean_interp(phase_array, phase_array_this_cycle, behav_group[feat].values))
            else:
                if feat == 'phase':
                    firing_rates_along_phase.append(phase_array)
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
n_strides = np.unique(firing_rates_phase_sessionwise.locomotor_cycle).shape[0]
print(f'Found {n_strides} strides')
# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3',
#                                                             colorby='global_phase', pop='locomotor_cycle', linewidth=0,
#                                                             opacity=0.3)
## normalizing firing rates
neural_scaler = sklearn.preprocessing.StandardScaler().fit(firing_rates_phase_sessionwise[neurons].values)
normalized_neurons = np.char.add(np.array('n'), np.array(neurons, dtype=str))
firing_rates_phase_sessionwise[normalized_neurons] = neural_scaler.transform(firing_rates_phase_sessionwise[neurons].values)

##
print('filtering artifacts in the tracking and light smoothing')
for ff in tqdm(['FRx', 'HRx', 'FLx', 'HLx']):
    firing_rates_phase_sessionwise[ff] = scipy.signal.savgol_filter(scipy.signal.medfilt(firing_rates_phase_sessionwise[ff], 31), 15, 3)

behav_pca_transform = sklearn.decomposition.PCA(n_components=3).fit(firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']])
behav_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']])
firing_rates_phase_sessionwise[['bPCA1', 'bPCA2', 'bPCA3']] = behav_pca[:,:3]

mossy_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[mossy_fibers])
firing_rates_phase_sessionwise[['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']] = mossy_pca[:,:6]

mossy_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[mossy_fibers])
firing_rates_phase_sessionwise[['nmPCA1', 'nmPCA2', 'nmPCA3', 'nmPCA4', 'nmPCA5', 'nmPCA6']] = mossy_pca[:,:6]

purkinje_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[purkinje_cells])
firing_rates_phase_sessionwise[['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']] = purkinje_pca[:, :6]

pawcolors =  [[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]]
# firing_rates_phase_sessionwise.to_csv(r'X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\firing_rates_phase_sessionwise_FR_StSt_full.csv')
firing_rates_phase_sessionwise['locomotor_cycle'] = firing_rates_phase_sessionwise['locomotor_cycle'].convert_dtypes('int64')

## BEHAVIOR VS MOSSY FIBERS                         IN APP
num_strides_to_plot = 200
strides_to_plot = np.linspace(0, np.unique(firing_rates_phase_sessionwise.locomotor_cycle).shape[0] - 1, num_strides_to_plot, dtype=int)
firing_rates_to_plot = firing_rates_phase_sessionwise[firing_rates_phase_sessionwise['locomotor_cycle'].isin(strides_to_plot)]
subset_df = firing_rates_to_plot
##
app = CareyPlots.twinplots(firing_rates_to_plot,            'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3',
                                                            colorby='phase', pop='locomotor_cycle', linewidth=0,
                                                            opacity=0.3, show_grid=True, show_background=False, DEF_SIZE=1, POP_SIZE=20)
##  MOSSY FIBERS VS PURKINJE CELLS                  IN APP
app = CareyPlots.twinplots(firing_rates_to_plot, 'mPCA1', 'mPCA2', 'mPCA3', 'pPCA1', 'pPCA2', 'pPCA3',
                                                            colorby='phase', pop='locomotor_cycle', linewidth=0,
                                                            opacity=0.3, show_grid=True, show_background=False, DEF_SIZE=1, POP_SIZE=20)
# ##  MOSSY FIBERS VS normalized
## make manifold plots
trials_to_plot = random.choices(np.unique(firing_rates_phase_sessionwise.locomotor_cycle), k=20)
subset_df = firing_rates_phase_sessionwise[firing_rates_phase_sessionwise['locomotor_cycle'].isin(trials_to_plot)]
ax = CareyPlots.multicoloredline_3d(subset_df, 'mPCA1', 'mPCA2', 'mPCA3', 'phase', lw=0.5, cmap=cmocean.cm.phase, alpha=0.5)
ax = CareyPlots.multicoloredline_3d(subset_df, 'bPCA1', 'bPCA2', 'bPCA3', 'phase', lw=0.5, cmap=cmocean.cm.phase, alpha=0.5)

##

ax = CareyPlots.multicoloredline_3d(subset_df, 'bPCA1', 'bPCA2', 'bPCA3', 'phase', lw=0.5, cmap=cmocean.cm.phase, alpha=0.5)
ax = CareyPlots.multicoloredline_3d(subset_df, 'pPCA1', 'pPCA2', 'pPCA3', 'phase', lw=0.5, cmap=cmocean.cm.phase, alpha=0.5)
ax = CareyPlots.multicoloredline_3d(subset_df, 'pPCA1', 'pPCA2', 'pPCA3', 'phase', lw=0.5, cmap=cmocean.cm.phase, alpha=0.5)

# ## BEHAVIOR VS MOSSY FIBERS             STANCE              IN APP
# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3', pop='phase',
#                                                             colorby=['FR_StOn', 'HR_StOn', 'FL_StOn', 'HL_StOn'], linewidth=0,
#                                                             custom_colors=['red', 'blue', 'magenta', 'cyan'],
#                                                             DEF_SIZE=5, POP_SIZE=10, show_grid=True, show_background=False)
# ## MOSSY FIBERS VS PURKINJE CELLS       STANCE
# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'mPCA1', 'mPCA2', 'mPCA3', 'pPCA1', 'pPCA2', 'pPCA3', pop='phase',
#                                                             colorby=['FR_StOn', 'HR_StOn', 'FL_StOn', 'HL_StOn'], linewidth=0,
#                                                             custom_colors=['red', 'magenta', 'blue', 'cyan'],
#                                                             DEF_SIZE=2, POP_SIZE=10, show_grid=True, show_background=False)
# ## BEHAVIOR VS MOSSY FIBERS             SWING
# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3', pop='phase',
#                                                             colorby=['FR_SwOn', 'HR_SwOn', 'FL_SwOn', 'HL_SwOn'], linewidth=0,
#                                                             custom_colors=['red', 'blue', 'magenta', 'cyan'],
#                                                             DEF_SIZE=5, POP_SIZE=10, show_grid=True, show_background=False)
# ## MOSSY FIBERS VS PURKINJE CELLS       STANCE
# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'mPCA1', 'mPCA2', 'mPCA3', 'pPCA1', 'pPCA2', 'pPCA3', pop='phase',
#                                                             colorby=['FR_SwOn', 'HR_SwOn', 'FL_SwOn', 'HL_SwOn'], linewidth=0,
#                                                             custom_colors=['red', 'blue', 'magenta', 'cyan'],
#                                                             DEF_SIZE=5, POP_SIZE=10, show_grid=True, show_background=False)
# ##
# app = CareyPlots.twinplots(firing_rates_phase_sessionwise, 'bPCA1', 'bPCA2', 'bPCA3', 'mPCA1', 'mPCA2', 'mPCA3', pop='phase',
#                                                             colorby=['FR_StOn', 'HR_StOn', 'HL_StOn', 'HL_StOn'],
#                                                             custom_colors=['red', 'magenta', 'blue', 'cyan'], opacity=1,
#                                                             DEF_SIZE=1, POP_SIZE=10, show_grid=True, show_background=False)


## behavior figures
# subset_df
# firing_rates_phase_sessionwise
fig_behav3d, ax_behav_3d = CareyPlots.multicoloredline_3d(firing_rates_to_plot, 'PC1', 'PC2', 'PC3', trials='locomotor_cycle',
                                                          colorby='phase', cmap=cmcrameri.cm.corkO_r, lw=0.3, alpha=1)
ax_behav_3d.set_xlim([-100, 100])
ax_behav_3d.set_ylim([-25, 25])
ax_behav_3d.set_zlim([-25, 25])
ax_behav_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax_behav_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax_behav_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax_behav_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax_behav_3d.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax_behav_3d.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax_behav_3d.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax_behav_3d.view_init(37.45, 108.0)
fig_behav3d.savefig(os.path.join(figures_folder, 'PC1_PC2_PC3_behav_pca_FR_phase.svg'))

# all behavior pca plots
f_b_pc1_pc2, ax_behav_pc1_pc2 = CareyPlots.multicoloredline_2d(firing_rates_to_plot, 'PC1', 'PC2', 'phase', cmap=cmcrameri.cm.corkO_r, lw=0.3, alpha=1, trials='locomotor_cycle')
f_b_pc1_pc2.set_size_inches(10, 5)
CareyPlots.adjust_font_size(ax_behav_pc1_pc2, 2)
f_b_pc1_pc2.savefig(os.path.join(figures_folder, 'PC1_PC2_behav_pca_FR_phase.svg'))

f_b_pc1_pc3, ax_behav_pc1_pc3 = CareyPlots.multicoloredline_2d(firing_rates_to_plot, 'PC1', 'PC3', 'phase', cmap=cmcrameri.cm.corkO_r, lw=0.3, alpha=1, trials='locomotor_cycle')
f_b_pc1_pc3.set_size_inches(10, 5)
CareyPlots.adjust_font_size(ax_behav_pc1_pc3, 2)
f_b_pc1_pc2.savefig(os.path.join(figures_folder, 'PC1_PC3_behav_pca_FR_phase.svg'))

f_b_pc2_pc3, ax_behav_pc2_pc3 = CareyPlots.multicoloredline_2d(firing_rates_to_plot, 'PC2', 'PC3', 'phase', cmap=cmcrameri.cm.corkO_r, lw=0.3, alpha=1, trials='locomotor_cycle')
f_b_pc2_pc3.set_size_inches(10, 5)
CareyPlots.adjust_font_size(ax_behav_pc2_pc3, 2)
f_b_pc1_pc2.savefig(os.path.join(figures_folder, 'PC2_PC3_behav_pca_FR_phase.svg'))

f_bpca_explained_variance, ax_bpca_explained_variance = plt.subplots()
plt.plot(np.arange(1, 4), np.cumsum(behav_pca_transform.explained_variance_ratio_), 'k*-')
plt.xlabel('Num components')
plt.xticks([1, 2, 3])
plt.ylim([0.7, 1])
f_bpca_explained_variance.set_size_inches(3.84, 6.21)
f_b_pc1_pc2.savefig(os.path.join(figures_folder, 'explained_var_behav_pca_FR_phase.svg'))

##
## firing_rates_phase_sessionwise['Phase'] = firing_rates_phase_sessionwise['phase']
# black_cmap = plt.matplotlib.colors.ListedColormap(['black'])
# f_pc1_over_phase, ax_pc1_over_phase = CareyPlots.multicoloredline_2d(
#     firing_rates_phase_sessionwise, 'phase', 'bPCA1', 'phase',  trials='locomotor_cycle', cmap=black_cmap, lw=0.3, alpha=0.1, colorbar=False)
# CareyPlots.multicoloredline_2d(
#     firing_rates_phase_sessionwise.groupby('phase').median().reindex(), 'Phase', 'bPCA1', 'Phase', trials='locomotor_cycle',
#     cmap=cmcrameri.cm.corkO_r, lw=5, alpha=1, colorbar=False, fig=f_pc1_over_phase, ax=ax_pc1_over_phase)
# # sns.lineplot(data=firing_rates_phase_sessionwise.groupby('phase').mean().reindex(), x='phase', y='bPCA1')
# ax_pc1_over_phase.set_ylim(-100, 100)
##
# cycles = np.unique(firing_rates_phase_sessionwise['locomotor_cycle'])
# fig, ax = plt.subplots()
# plt.show()
# plt.draw()
# for ii in range(len(cycles)):
#     CareyPlots.multicoloredline_2d(
#         firing_rates_phase_sessionwise[firing_rates_phase_sessionwise['locomotor_cycle']==cycles[ii]],
#         'phase', 'bPCA1', 'phase', cmap=cmcrameri.cm.corkO_r, lw=0.3, alpha=1, fig=fig, ax=ax, colorbar=False);
#     plt.xlim([0, 1])
#     plt.pause(0.1)
#     plt.draw()


##
#
# import sklearn
# npts = firing_rates_phase_sessionwise.shape[0]
# train, test = sklearn.model_selection.train_test_split(np.arange(npts), test_size=0.2)
#
# mossy_to_behav = sklearn.linear_model.LinearRegression()
# scaler_mossy = sklearn.preprocessing.StandardScaler().fit(firing_rates_phase_sessionwise[mossy_fibers].iloc[train])
# mossy_to_behav.fit(scaler_mossy.transform(firing_rates_phase_sessionwise[mossy_fibers].iloc[train]),
#                    firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train])
# y_test_pred_mossy = mossy_to_behav.predict( scaler_mossy.transform(df[mossy_fibers]))
#
# pkc_to_behav = sklearn.linear_model.LinearRegression()
# scaler_pkc = sklearn.preprocessing.StandardScaler().fit(firing_rates_phase_sessionwise[purkinje_cells].iloc[train])
# pkc_to_behav.fit(scaler_pkc.transform(firing_rates_phase_sessionwise[purkinje_cells].iloc[train]),
#                    firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train])
# y_test_pred_pkc = pkc_to_behav.predict( scaler_pkc.transform(df[purkinje_cells]))
#
#
# plt.figure()
# plt.plot(behav['FRx'].values, color='gray')
# plt.plot(y_test_pred_mossy[:,0], color=CareyConstants.paw_colors_sns[0], linestyle='dashed')
# plt.plot(y_test_pred_pkc[:,0], color=CareyConstants.paw_colors_sns[0], linestyle='dotted')
# plt.legend(['FRx', 'mossy fiber pred', 'purkinje cell pred'])
#

## tiling

tiling_mossy = firing_rates_phase_sessionwise.groupby('phase').mean()[mossy_fibers].values
tiling_mossy = CareyUtils.zscore(tiling_mossy)
tiling_mossy = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(tiling_mossy)
peak_tuning = np.argmax(tiling_mossy, 0)
order = np.argsort(peak_tuning)

colors = ["#FFFFFF", MOSSY_COLOR]  # from white to the target color
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)
fig, ax = plt.subplots()
plt.imshow(tiling_mossy[:, order].T, cmap=cmap)
plt.colorbar()
ax.set_aspect('auto')
plt.xticks([0, 50, 100], np.linspace(0, 1, 3, endpoint=True))
CareyPlots.adjust_font_size(ax, 4)

tiling_purkinje = firing_rates_phase_sessionwise.groupby('phase').mean()[purkinje_cells].values
tiling_purkinje = CareyUtils.zscore(tiling_purkinje)
tiling_purkinje = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(tiling_purkinje)
peak_tuning = np.argmax(tiling_purkinje, 0)
order = np.argsort(peak_tuning)

colors = ["#FFFFFF", PURKINJE_COLOR]  # from white to the target color
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)
fig, ax = plt.subplots()
plt.imshow(tiling_purkinje[:, order].T, cmap=cmap)
plt.colorbar()
ax.set_aspect('auto')
plt.xticks([0, 50, 100], np.linspace(0, 1, 3, endpoint=True))
CareyPlots.adjust_font_size(ax, 4)

## distribution of stance points over the phase space

stances_FR = firing_rates_phase_sessionwise['phase'][firing_rates_phase_sessionwise['FR_StOn']==1]
stances_HR = firing_rates_phase_sessionwise['phase'][firing_rates_phase_sessionwise['HR_StOn']==1]
stances_FL = firing_rates_phase_sessionwise['phase'][firing_rates_phase_sessionwise['FL_StOn']==1]
stances_HL = firing_rates_phase_sessionwise['phase'][firing_rates_phase_sessionwise['HL_StOn']==1]
stances_df = stances_FR.append(stances_HR, ignore_index=True).append(stances_FL, ignore_index=True).append(stances_HL, ignore_index=True)
fig = plt.figure()
sns.kdeplot(data=stances_FR.to_frame(), x='phase', color=CareyConstants.paw_colors_sns[0])
sns.kdeplot(data=stances_HR.to_frame(), x='phase', color=CareyConstants.paw_colors_sns[1])
sns.kdeplot(data=stances_FL.to_frame(), x='phase', color=CareyConstants.paw_colors_sns[2])
sns.kdeplot(data=stances_HL.to_frame(), x='phase', color=CareyConstants.paw_colors_sns[3])

##
behav_fast = behav[behav['wheel_speed']>0.4]
# for ff in tqdm(['FRx', 'HRx', 'FLx', 'HLx']):
#     behav_fast[ff] = scipy.signal.savgol_filter(scipy.signal.medfilt(behav_fast[ff], 31), 15, 3)
behav_fast = behav_fast.interpolate()
behav_fast[['nPC1', 'nPC2', 'nPC3']] = sklearn.decomposition.PCA(n_components=3).fit_transform(behav_fast[['FRx', 'HRx', 'FLx', 'HLx']])


##behav_pca_transform = sklearn.decomposition.PCA(n_components=3).fit(firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']])
behav_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']])
firing_rates_phase_sessionwise[['bPCA1', 'bPCA2', 'bPCA3']] = behav_pca[:,:3]

mossy_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[mossy_fibers])
firing_rates_phase_sessionwise[['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']] = mossy_pca[:,:6]

mossy_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[mossy_fibers])
firing_rates_phase_sessionwise[['nmPCA1', 'nmPCA2', 'nmPCA3', 'nmPCA4', 'nmPCA5', 'nmPCA6']] = mossy_pca[:,:6]

purkinje_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[purkinje_cells])
firing_rates_phase_sessionwise[['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']] = purkinje_pca[:, :6]








##

mossy_pca = sklearn.decomposition.PCA(n_components=6).fit(firing_rates_phase_sessionwise[mossy_fibers])
fig, ax = plt.subplots(3,1)
for ii in range(3):
    ax[ii].bar(np.arange(1,21), mossy_pca.components_[ii, :], facecolor=MOSSY_COLOR ,edgecolor='k')
fig.set_size_inches([5.23, 5.68])
fig.savefig(os.path.join(figures_folder, 'mossy_pca_components.svg'))
##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(3,1)
for aa in range(3):
    for ii in strides:
        ax[aa].plot(firing_rates_to_plot['mPCA'+str(aa+1)][firing_rates_to_plot.locomotor_cycle==ii].values, lw=0.2, color=MOSSY_COLOR)
    ax[aa].plot(firing_rates_to_plot.groupby('phase').mean()['mPCA' + str(aa + 1)].values, lw=1, color='k')
    ax[aa].set_xticks([0, 50, 100])
    ax[aa].set_xticklabels([0, 0.5, 1])
fig.set_size_inches([7.38, 5.65])
fig.savefig(os.path.join(figures_folder, 'mossy_pc_averages_over_phase.svg'))
##
fig = px.scatter_3d(firing_rates_to_plot, x='mPCA1', y='mPCA2', z='mPCA3', color='phase', color_continuous_scale='phase')
fig.update_traces(marker_size = 1)
fig.update_scenes(aspectmode='cube')
fig.show()
##
mossy_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[mossy_fibers])
firing_rates_phase_sessionwise[['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']] = mossy_pca[:,:6]

fig, ax = CareyPlots.multicoloredline_3d(firing_rates_to_plot,
                                         'mPCA1', 'mPCA2', 'mPCA3', 'phase', trials='locomotor_cycle',
                                         cmap=cmocean.cm.phase, lw=1, alpha=0.5)
# PC1 by PC2
ax.azim = 90
ax.elev = 90
plt.draw()








##

purkinje_pca = sklearn.decomposition.PCA(n_components=6).fit(firing_rates_phase_sessionwise[purkinje_cells])
fig, ax = plt.subplots(3,1)
for ii in range(3):
    ax[ii].bar(np.arange(1,21), purkinje_pca.components_[ii, :], facecolor=PURKINJE_COLOR ,edgecolor='k')
fig.set_size_inches([5.23, 5.68])
fig.savefig(os.path.join(figures_folder, 'purkinje_pca_components.svg'))
##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(3,1)
for aa in range(3):
    for ii in strides:
        ax[aa].plot(firing_rates_to_plot['pPCA'+str(aa+1)][firing_rates_to_plot.locomotor_cycle==ii].values, lw=0.2, color=PURKINJE_COLOR)
    ax[aa].plot(firing_rates_to_plot.groupby('phase').mean()['pPCA' + str(aa + 1)].values, lw=1, color='k')
    ax[aa].set_xticks([0, 50, 100])
    ax[aa].set_xticklabels([0, 0.5, 1])
fig.set_size_inches([7.38, 5.65])
fig.savefig(os.path.join(figures_folder, 'purkinje_pc_averages_over_phase.svg'))
##
fig = px.scatter_3d(firing_rates_to_plot, x='pPCA1', y='pPCA2', z='pPCA3', color='phase', color_continuous_scale='phase')
fig.update_traces(marker_size = 1)
fig.update_scenes(aspectmode='cube')
fig.show()
##
purkinje_pca = sklearn.decomposition.PCA(n_components=6).fit_transform(firing_rates_phase_sessionwise[purkinje_cells])
firing_rates_phase_sessionwise[['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']] = purkinje_pca[:,:6]

fig, ax = CareyPlots.multicoloredline_3d(firing_rates_to_plot,
                                         'pPCA1', 'pPCA2', 'pPCA3', 'phase', trials='locomotor_cycle',
                                         cmap=cmocean.cm.phase, lw=1, alpha=0.5)
# PC1 by PC2
ax.azim = 90
ax.elev = 90
plt.draw()




##

behavior_pca = sklearn.decomposition.PCA(n_components=3).fit(firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']])
fig, ax = plt.subplots(3,1)
for ii in range(3):
    barlist = ax[ii].bar(np.arange(1,5), behavior_pca.components_[ii, :], facecolor='k',edgecolor='k')
    for ii, bb in enumerate(barlist):
        bb.set_color(CareyConstants.paw_colors_sns[ii])
fig.set_size_inches([5.23, 5.68])
fig.savefig(os.path.join(figures_folder, 'locomotion_pca_components.svg'))
## explained_variance
plt.figure()
plt.plot(np.arange(1, 4), np.cumsum(behavior_pca.explained_variance_ratio_), '*-', color=LOCOMOTION_COLOR)
plt.xlabel('Num components')
plt.xticks([1, 2, 3])
plt.ylim([0.85, 1])

##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(3, 1)
for aa in range(3):
    for ii in strides:
        ax[aa].plot(firing_rates_to_plot['bPCA'+str(aa+1)][firing_rates_to_plot.locomotor_cycle==ii].values, lw=0.2, color=LOCOMOTION_COLOR)
    ax[aa].plot(firing_rates_to_plot.groupby('phase').mean()['bPCA' + str(aa + 1)].values, lw=1, color='k')
    ax[aa].set_xticks([0, 50, 100])
    ax[aa].set_xticklabels([0, 0.5, 1])
fig.set_size_inches([7.38, 5.65])
fig.savefig(os.path.join(figures_folder, 'behavior_pc_averages_over_phase.svg'))
##
fig = px.scatter_3d(firing_rates_to_plot, x='bPCA1', y='bPCA2', z='bPCA3', color='phase', color_continuous_scale='phase')
fig.update_traces(marker_size = 1)
fig.update_scenes(aspectmode='cube')
fig.show()
##
# behavior_pca_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(firing_rates_phase_sessionwise[['FRx', 'HRx', 'FLx', 'HLx']])
# firing_rates_phase_sessionwise[['bPCA1', 'bPCA2', 'bPCA3']] = purkinje_pca[:,:3]

fig, ax = CareyPlots.multicoloredline_3d(firing_rates_to_plot,
                                         'bPCA1', 'bPCA2', 'bPCA3', 'phase', trials='locomotor_cycle',
                                         cmap=cmocean.cm.phase, lw=1, alpha=1)
# PC1 by PC2
ax.azim = 90
ax.elev = 0
plt.draw()

## dimensionality mossy vs purkinje

purkinje_pca = sklearn.decomposition.PCA(n_components=len(purkinje_cells)).fit(firing_rates_phase_sessionwise[purkinje_cells])
mossy_pca = sklearn.decomposition.PCA(n_components=len(mossy_fibers)).fit(firing_rates_phase_sessionwise[mossy_fibers])

plt.figure()
plt.plot(np.arange(1, len(purkinje_cells)+1), np.cumsum(purkinje_pca.explained_variance_ratio_), '*-', color=PURKINJE_COLOR)
plt.plot(np.arange(1, len(mossy_fibers)+1), np.cumsum(mossy_pca.explained_variance_ratio_), '*-', color=MOSSY_COLOR)

plt.xlabel('Num components')
plt.show()

## qunatifying information by checking how much of neural activity is explained by the behavior

npts = behav.shape[0]
train, test = sklearn.model_selection.train_test_split(np.arange(npts), test_size=0.2)

behav_to_mossy = sklearn.linear_model.LinearRegression()
behav_to_purkinje = sklearn.linear_model.LinearRegression()

df[mossy_fibers] = df[mossy_fibers].interpolate()
df[purkinje_cells] = df[purkinje_cells].interpolate()
behav[['FRx', 'HRx', 'FLx', 'HLx']] = behav[['FRx', 'HRx', 'FLx', 'HLx']].interpolate()

scaler_mossy = sklearn.preprocessing.StandardScaler().fit(df[mossy_fibers].iloc[train])
scaler_purkinje = sklearn.preprocessing.StandardScaler().fit(df[purkinje_cells].iloc[train])
scaler_behav = sklearn.preprocessing.StandardScaler().fit(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train])


behav_to_mossy.fit(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train]),
                   scaler_mossy.transform(df[purkinje_cells].iloc[train]))
behav_to_purkinje.fit(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train]),
                   scaler_mossy.transform(df[mossy_fibers].iloc[train]))


exp_var_mossy = sklearn.metrics.explained_variance_score(
                            scaler_mossy.transform(df[mossy_fibers].iloc[test]),
                            behav_to_mossy.predict(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[test])))
exp_var_purkinje = sklearn.metrics.explained_variance_score(
                            scaler_purkinje.transform(df[purkinje_cells].iloc[test]),
                            behav_to_purkinje.predict(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[test])))


# plt.figure()
# plt.plot(behav['FRx'].values, color='gray')
# plt.plot(y_test_pred_mossy[:,0], color=CareyConstants.paw_colors_sns[0], linestyle='dashed')
# plt.plot(y_test_pred_pkc[:,0], color=CareyConstants.paw_colors_sns[0], linestyle='dotted')
# plt.legend(['FRx', 'mossy fiber pred', 'purkinje cell pred'])

pca_mossy = sklearn.decomposition.PCA(n_components=3).fit_transform(df[mossy_fibers].values)
pca_purkinje = sklearn.decomposition.PCA(n_components=3).fit_transform(df[purkinje_cells].values)

scaler_mossy = sklearn.preprocessing.StandardScaler().fit(pca_mossy[train,:])
scaler_purkinje = sklearn.preprocessing.StandardScaler().fit(pca_purkinje[train,:])

behav_to_mossy = sklearn.linear_model.LinearRegression()
behav_to_purkinje = sklearn.linear_model.LinearRegression()

behav_to_mossy.fit(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train]),
                   scaler_mossy.transform(pca_mossy[train, :]))
behav_to_purkinje.fit(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[train]),
                   scaler_mossy.transform(pca_purkinje[train, :]))

exp_var_mossy = sklearn.metrics.r2_score(
                            scaler_mossy.transform(pca_mossy[test, :]),
                            behav_to_mossy.predict(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[test])))
exp_var_purkinje = sklearn.metrics.r2_score(
                            scaler_purkinje.transform(pca_purkinje[test, :]),
                            behav_to_purkinje.predict(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[test])))

exp_var_mossy = sklearn.metrics.explained_variance_score(
                            scaler_mossy.transform(pca_mossy[test, :]),
                            behav_to_mossy.predict(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[test])))
exp_var_purkinje = sklearn.metrics.explained_variance_score(
                            scaler_purkinje.transform(pca_purkinje[test, :]),
                            behav_to_purkinje.predict(scaler_behav.transform(behav[['FRx', 'HRx', 'FLx', 'HLx']].iloc[test])))


## TEST THE LEFTOVER VARIANCE OVER PHASE

canonical_stride_behav      = firing_rates_phase_sessionwise.groupby('phase').median()[['bPCA1', 'bPCA2', 'bPCA3']]
canonical_stride_behav['phase'] = canonical_stride_behav.index.values
canonical_stride_mossy      = firing_rates_phase_sessionwise.groupby('phase').median()[['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']]
canonical_stride_mossy['phase'] = canonical_stride_mossy.index.values
canonical_stride_purkinje   = firing_rates_phase_sessionwise.groupby('phase').median()[['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']]
canonical_stride_purkinje['phase'] = canonical_stride_purkinje.index.values

##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(3, 1)
for aa in range(3):
    for ii in strides:
        ax[aa].plot(firing_rates_to_plot['bPCA'+str(aa+1)][firing_rates_to_plot.locomotor_cycle==ii].values - canonical_stride_behav['bPCA' + str(aa + 1)].values,
                    lw=0.2, color=LOCOMOTION_COLOR)
##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(6, 1)
for aa in range(6):
    for ii in strides:
        ax[aa].plot(firing_rates_to_plot['mPCA'+str(aa+1)][firing_rates_to_plot.locomotor_cycle==ii].values - canonical_stride_mossy['mPCA' + str(aa + 1)].values,
                    lw=0.2, color=MOSSY_COLOR)
##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(6, 1)
for aa in range(6):
    for ii in strides:
        ax[aa].plot(firing_rates_to_plot['pPCA'+str(aa+1)][firing_rates_to_plot.locomotor_cycle==ii].values - canonical_stride_purkinje['pPCA' + str(aa + 1)].values,
                    lw=0.2, color=PURKINJE_COLOR)


##
canonical_centered_neural = firing_rates_to_plot
canonical_centered_neural['behav_dist'] = 0
canonical_centered_neural['mossy_dist'] = 0
canonical_centered_neural['purkinje_dist'] = 0
for ii in tqdm(np.unique(canonical_centered_neural.locomotor_cycle)):

    this_stride = canonical_centered_neural['locomotor_cycle']==ii

    canonical_centered_neural.loc[this_stride, ['bPCA1', 'bPCA2', 'bPCA3']] -= canonical_stride_behav[['bPCA1', 'bPCA2', 'bPCA3']].values
    behav_dist = (canonical_centered_neural.loc[this_stride, ['bPCA1', 'bPCA2', 'bPCA3']].values ** 2).sum(1)
    canonical_centered_neural.loc[this_stride, 'behav_dist'] = behav_dist

    canonical_centered_neural.loc[this_stride, ['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']] -= \
        canonical_stride_mossy[['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']].values
    mossy_dist = (canonical_centered_neural.loc[this_stride, ['mPCA1', 'mPCA2', 'mPCA3', 'mPCA4', 'mPCA5', 'mPCA6']].values ** 2).sum().sum()
    canonical_centered_neural.loc[this_stride, 'mossy_dist'] = mossy_dist

    canonical_centered_neural.loc[this_stride, ['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']] -= \
        canonical_stride_purkinje[['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']].values
    purkinje_dist = (canonical_centered_neural.loc[
                      this_stride, ['pPCA1', 'pPCA2', 'pPCA3', 'pPCA4', 'pPCA5', 'pPCA6']].values ** 2).sum().sum()
    canonical_centered_neural.loc[this_stride, 'purkinje_dist'] = purkinje_dist

##
plt.figure()
plt.plot(canonical_stride_mossy['phase'], canonical_stride_mossy['mPCA1'], 'k')
plt.plot(firing_rates_to_plot.loc[this_stride, 'phase'], firing_rates_to_plot.loc[this_stride, 'mPCA1'], 'orange')
plt.plot(canonical_centered_neural.loc[this_stride, 'phase'], canonical_centered_neural.loc[this_stride, 'mPCA1'], 'red')
plt.plot(firing_rates_to_plot.loc[this_stride, 'phase'], firing_rates_to_plot.loc[this_stride, 'mPCA1'].values-canonical_stride_mossy['mPCA1'].values, '--')

##
app = CareyPlots.twinplots(canonical_centered_neural, 'phase', 'mPCA1', 'mPCA2', 'phase', 'pPCA4', 'pPCA3',
                                                            colorby='behav_dist', pop='locomotor_cycle', linewidth=0,
                                                            opacity=0.3, colormap='viridis', DEF_SIZE=3, POP_SIZE=20)


##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(3, 1)
for aa in range(3):
    CareyPlots.multicoloredline_2d(firing_rates_to_plot, 'phase', ('mPCA'+str(aa+1)), colorby='behav_dist', trials='locomotor_cycle',
                                   cmap='viridis', ax=ax[aa], fig=fig, lw=0.3)
##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(6, 1)
for aa in range(6):
    CareyPlots.multicoloredline_2d(firing_rates_to_plot, 'phase', ('mPCA' + str(aa + 1)), colorby='behav_dist',
                                   trials='locomotor_cycle',
                                   cmap='viridis', ax=ax[aa], fig=fig, lw=0.5)
##
strides = np.unique(firing_rates_to_plot.locomotor_cycle)
fig, ax = plt.subplots(6, 1)
for aa in range(6):
    CareyPlots.multicoloredline_2d(firing_rates_to_plot, 'phase', ('pPCA' + str(aa + 1)), colorby='behav_dist',
                                   trials='locomotor_cycle',
                                   cmap='viridis', ax=ax[aa], fig=fig, lw=0.5)