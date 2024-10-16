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


batch_5_loc = r'X:\data\2022\BATCH5'
if os.getlogin() == 'diogo':
    batch_5_loc = r'X:\data\2022\BATCH5'
dataset_folder = os.path.join(batch_5_loc, 'recordings', 'VIV_23058', 'VIV_23058_S10_g1', 'kilosort4_catgt_tshift')
processing_folder = os.path.join(batch_5_loc, 'processing', 'VIV_23058', 'S10')
locopixels_folder = os.path.join(processing_folder, 'locopixels')
behav_manifold_folder =  os.path.join(batch_5_loc, 'processing\VIV_23058\S10\Behavioral manifold')

## read behavior dataset and get global phase
print('Reading behavior dataset...')
# behav = pd.read_csv(os.path.join(processing_folder, 'VIV_23058_S10_behavioral_descriptor.csv'))
behav = pd.read_csv(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv")
print('Done')

intervals = CareyUtils.find_speed_intervals_in_df(behav, 'wheel_speed', 'sessionwise_time', 0.4, constrain_to='trial')

for paw in CareyConstants.paw_labels:
    __, behav[f'{paw}_SwPh'], behav[f'{paw}_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav[f'{paw}_SwOn'], behav[f'{paw}_StOn'], usegpu=0)
    __, behav[f'{paw}_StPh'], behav[f'{paw}_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav[f'{paw}_StOn'], behav[f'{paw}_SwOn'], usegpu=0)

start, end = intervals.loc[intervals['duration'].argmax(), ['start_index', 'end_index']].astype(int)
fastloco = behav[start:end]

data = fastloco[['FR_StPh', 'HR_StPh', 'FL_StPh', 'HL_StPh']].values
# ani = CareyPlots.animate_signal(data,  nframes=10000, dt=50, colors=CareyConstants.paw_colors_sns, opacity=0.7, markersize=12)
# ani = CareyPlots.animate_signal((data-[0, 0.5, 0.5, 0]) % 1,  nframes=None, dt=50, colors=CareyConstants.paw_colors_sns, opacity=0.7, markersize=12)


##
pshift = lambda x: (x + 0.5) % 1 - 0.5
phase_shifted_data = (data-[0, 0.5, 0.5, 0]) % 1

phase_diff_FR_FL = ((data[:,0] - data[:,2]) % 1) # FR vs FL
fastloco['phase_diff_FR_FL'] = phase_diff_FR_FL

rel_to_FR = phase_shifted_data - np.tile(phase_shifted_data[:, 0].reshape(-1, 1), 4)
rel_to_FR = pshift(rel_to_FR)


phase_normalized_phase_shifted_data = pshift(phase_shifted_data)
# pmin = phase_normalized_phase_shifted_data.min(1)
# pmax = phase_normalized_phase_shifted_data.max(1)
pmin = phase_shifted_data.min(1)
pmax = phase_shifted_data.max(1)
fastloco['asym'] = CareyUtils.subtract_phases(pmin, pmax)

fastloco['phase_diff_FR_FL_binned'] = (fastloco['phase_diff_FR_FL'] // 0.05) * 0.05

