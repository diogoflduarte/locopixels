import os
import numpy as np
import npyx
import CareyEphys
import pandas as pd
import CareyConstants
import time
import cupy
from tqdm import tqdm

datadir = r'X:\data\2022\BATCH5'
neural  =  os.path.join(datadir, r'recordings\VIV_23058\VIV_23058_S10_g1\kilosort4_catgt_tshift')
new_proc = os.path.join(datadir, r'processing\VIV_23058\S10\locopixels')
dp = neural # for npyx readability


meta = npyx.read_metadata(dp)
cluster_info = pd.read_csv(os.path.join(neural, 'cluster_info.tsv'), delimiter='\t')

SAMP_RATE  = meta['highpass']['sampling_rate']
TOTAL_TIME = meta['highpass']['fileTimeSecs']

# load data
good_units = npyx.gl.get_good_units(dp)

# get unit types
cell_types = pd.read_csv(os.path.join(neural, 'c4_results', 'cluster_predicted_cell_type.tsv'), delimiter='\t')
confidence = pd.read_csv(os.path.join(neural, 'c4_results', 'cluster_confidence_ratio.tsv'), delimiter='\t')
c4 = pd.merge(cell_types, confidence, on='cluster_id')
c4 = pd.merge(c4, cluster_info, on='cluster_id')

MFs = c4.cluster_id[np.logical_and(c4.predicted_cell_type=='MFB', c4.confidence_ratio>3)].values

##
example_unit = MFs[0]
spikes = npyx.spk_t.trn(dp, example_unit)/ SAMP_RATE

start = time.clock()
fr = CareyEphys.estimate_gaussconv_based_FR(spikes, TOTAL_TIME, fs=SAMP_RATE, gaussdev=0.020)
stop = time.clock()-start


## iterate through all the good units and save them in a dataframe
# todo: should turn this into a function

t_arr = np.arange(0, TOTAL_TIME, 1 / 1000, dtype=float)
cupy_t_arr = cupy.array(t_arr)
cupy_t_full = cupy.arange(0, TOTAL_TIME, 1 / SAMP_RATE, dtype=float)
df = pd.DataFrame(data=t_arr, columns=['time'])

for neuron in tqdm(good_units):
    spikes = npyx.spk_t.trn(dp, neuron)/ SAMP_RATE
    fr = CareyEphys.estimate_gaussconv_based_FR(spikes, TOTAL_TIME, fs=SAMP_RATE, gaussdev=0.020)
    fr_downsampled = cupy.interp(cupy_t_arr, cupy_t_full, cupy.array(fr)).get()
    # df = pd.DataFrame(data=fr_downsamped, columns=[str(neuron)])
    # df.to_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'), mode='a', header=True, index=False)
    df[str(neuron)] = fr_downsampled

df.to_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'))