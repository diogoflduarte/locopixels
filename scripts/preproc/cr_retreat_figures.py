import os
import numpy as np
import npyx
import CareyEphys
import CareyBehavior
import CareyUtils
from tqdm import tqdm
import pandas as pd
from CareyConstants import CareyConstants
import CareyPlots
import time
import cupy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
import cmocean
import scipy.signal
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cmcrameri
import ssm

plot_behav = False
neural_manifold = 'load' # alternative is compute

if os.getlogin() == 'diogo':
    behav_manifold_folder   = r'W:\Processed data\Behavioral manifold'
    neural_manifold_folder  = r'X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels'

df      = pd.read_csv(os.path.join(behav_manifold_folder, 'behavioral_manifold.csv'))
cycles  = np.load(os.path.join(behav_manifold_folder, 'cycles_interp.npy'), allow_pickle=True)

## convert cycles to pandas dataframe
n_strides = len(cycles)
LIMIT_STRIDES_FOR_PCA_DATAFRAME = 200
strides = np.random.choice(n_strides, LIMIT_STRIDES_FOR_PCA_DATAFRAME)

strides = np.arange(100)
phase_pts, p_comp = cycles[0].shape
phase_array = np.linspace(0, 2*np.pi, phase_pts)
df_cycles = pd.DataFrame(data=None, columns=['stride', 'phase', 'PC1', 'PC2', 'PC3'])
for ss in tqdm(strides):
    data = np.hstack((np.ones_like(phase_array)[:,None]*ss,
                      phase_array[:,None],
                      cycles[ss]))
    df_cycles = df_cycles.append(pd.DataFrame(data=data,
                                              columns=['stride', 'phase', 'PC1', 'PC2', 'PC3']),
                                 ignore_index=True)
## plot PCA in cycles
if plot_behav:
    fig, ax = CareyPlots.plot3D(df_cycles, axis1='PC1', axis2='PC2', axis3='PC3', colorby='phase', colormap=cmcrameri.cm.romaO)
    for ii in range(len(ax)):
        npyx.mplp(fig, ax[ii])
        ax[0].zaxis.label.set_fontsize(ax[0].xaxis.label.get_fontsize())
        ax[0].zaxis.set_tick_params(labelsize=16)


## plot using the behavioral_manifold.csv
# todo


## neural manifold
neural = os.path.join(neural_manifold_folder, 'sessionwise_firing_rates.csv')
mossy = list(map(str, ['time',  317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467, 479, 500, 501, 503, 507, 601,
                                602, 604, 611, 613]))
mossy_list = mossy
mossy_list.pop(0)
n_mossy = len(mossy_list)
purkinje = list(map(str, ['time', 91, 111, 115, 151, 198, 226, 241, 246, 259, 400, 402, 415, 88, 209, 266, 274, 382]))
purkinje_list = purkinje
purkinje_list.pop(0)
n_purkinje = len(purkinje_list)

if neural_manifold == 'compute':

    mossy       = pd.read_csv(neural, usecols=mossy)
    purkinje    = pd.read_csv(neural, usecols=purkinje)
    indices     = np.logical_and( mossy.time >= df.sessionwise_time.min(), mossy.time <= df.sessionwise_time.max() )
    mossy       = mossy[indices]
    purkinje    = purkinje[indices]

    for m in mossy_list:
        mossy[m] = CareyUtils.zscore(mossy[m].values)
    for p in purkinje_list:
        purkinje[p] = CareyUtils.zscore(purkinje[p].values)


## reorganize by stride

    mossy['phase'] = cupy.interp( cupy.array(mossy.time),
                                  cupy.array(df.sessionwise_time),
                                  cupy.array(df['global phase'])).get()
    purkinje['phase'] = mossy['phase'].values
    mossy['cycle']      = CareyUtils.get_stride_indices_from_phase(mossy.phase.values)
    purkinje['cycle']   = CareyUtils.get_stride_indices_from_phase(purkinje.phase.values)

    strides = np.unique(mossy.cycle)
    n_strides_for_training = 1000
    mossy_by_phase      = np.zeros((n_strides_for_training, n_mossy,    phase_array.shape[0]))

    purkinje_by_phase   = np.zeros((n_strides_for_training, n_purkinje, phase_array.shape[0]))
    mossy_ph    =  pd.DataFrame(data=None, columns=mossy.columns)
    purkinje_ph =  pd.DataFrame(data=None, columns=purkinje.columns)
    for ii in tqdm(np.arange(1, n_strides_for_training)):
        for m in range(len(mossy_list)):
            mossy_by_phase[ii-1, m, :] = np.interp(     phase_array,
                                                        mossy.phase[mossy.cycle==ii],
                                                        mossy[mossy_list[m]][mossy.cycle==ii] )
            tmp_time = np.interp(phase_array, mossy.phase[mossy.cycle==ii], mossy['time'][mossy.cycle==ii] )
        tmp_data = np.hstack((tmp_time[:, None],
                              mossy_by_phase[ii-1, :, :].transpose(),
                              phase_array[:, None],
                              (np.ones_like(phase_array)*ii)[:, None]))
        mossy_ph = mossy_ph.append(pd.DataFrame(data=tmp_data, columns=mossy_ph.columns), ignore_index=True)
        for p in range(len(purkinje_list)):
            purkinje_by_phase[ii-1, p, :] = np.interp(  phase_array,
                                                        purkinje.phase[purkinje.cycle==ii],
                                                        purkinje[purkinje_list[p]][purkinje.cycle==ii] )
            tmp_time = np.interp(phase_array, purkinje.phase[purkinje.cycle == ii], purkinje['time'][purkinje.cycle == ii])
        tmp_data = np.hstack((tmp_time[:, None],
                              purkinje_by_phase[ii - 1, :, :].transpose(),
                              phase_array[:, None],
                              (np.ones_like(phase_array) * ii)[:, None]))
        purkinje_ph = purkinje_ph.append(pd.DataFrame(data=tmp_data, columns=purkinje_ph.columns), ignore_index=True)
elif neural_manifold == 'load':
    mossy_ph = pd.read_csv(os.path.join(neural_manifold_folder, 'mossy_by_phase.csv'))
    purkinje_ph = pd.read_csv(os.path.join(neural_manifold_folder, 'purkinje_by_phase.csv'))


## compute pca on neural firing rates

mossy_manifold = PCA(n_components=n_mossy)
mossy_manifold.fit(mossy_ph[mossy_list])
cols = CareyUtils.generatePCNames(len(mossy_list))
cols.append('phase')
cols.append('cycle')
cols.insert(0, 'time')

mossy_manifold_df = pd.DataFrame(columns=cols,
                                 data=np.hstack((   mossy_ph['time'].values[:, None],
                                                    mossy_manifold.transform(mossy_ph[mossy_list]),
                                                    mossy_ph['phase'].values[:, None],
                                                    mossy_ph['cycle'].values[:, None])) )

##
start = 10000
n_pts = 1000
stop = start + n_pts
fig, ax = CareyPlots.plot3D(mossy_manifold_df[start:stop], axis1='PC 1', axis2='PC 2', axis3='PC 3', colorby='phase')
for ii in range(len(ax)):
    npyx.mplp(fig, ax[ii])
    ax[0].zaxis.label.set_fontsize(ax[0].xaxis.label.get_fontsize())
    ax[0].zaxis.set_tick_params(labelsize=16)


## behavioral pca

behav_pca = PCA(n_components=3)
feat = ['FRx', 'HRx', 'FLx', 'HLx']

df[feat] = df[feat].interpolate()
behav_pca.fit(df[feat])
cols = CareyUtils.generatePCNames(3)
cols.append('phase')
cols.append('cycle')

behav_manifold_df = pd.DataFrame(columns=cols,
                                 data=np.hstack((   behav_pca.transform(df[feat]),
                                                    df['global phase'].values[:, None],
                                                    CareyUtils.get_stride_indices_from_phase(df['global phase'].values[:, None]))))
plt.plot(np.arange(1, 4), np.cumsum(behav_pca.explained_variance_ratio_), 'k*-')
plt.xlabel('Num components')
plt.xticks([1, 2, 3])
plt.ylim([0.7, 1])

## average signals over phase

mossy_and_purkinje = mossy_ph[['317', '322', '351', '354', '367', '393', '421', '453', '457', '467', '479', '500',
                               '501', '503', '507', '542', '601', '602', '604', '611', '613']]
mossy_and_purkinje[['88', '91', '111', '115', '151', '198', '209', '226', '241', '246', '259', '266', '274', '382', \
                    '400', '402', '415', 'phase', 'cycle']] = \
purkinje_ph[['88', '91', '111', '115', '151', '198', '209', '226', '241', '246', '259', '266', '274', '382', \
                    '400', '402', '415', 'phase', 'cycle']]
data = mossy_and_purkinje.groupby('phase').mean()[mossy_list + purkinje_list].values
data = CareyUtils.zscore(data)
sortmossy       = np.argsort(np.argmax(data[:, :len(mossy_list)], 0))
sortpurkinje    = np.argsort(np.argmax(data[:,  len(mossy_list):len(mossy_list)+len(purkinje_list)], 0))
sortpurkinje = sortpurkinje + np.max(sortmossy)
sort = np.hstack((sortmossy, sortpurkinje))
CareyPlots.ridgeplot(data[:, sort])

df = pd.melt(mossy_and_purkinje, id_vars=['phase'])
