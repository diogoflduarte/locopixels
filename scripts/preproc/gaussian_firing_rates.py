import os
import numpy as np
import npyx
import CareyEphys
import CareyBehavior
import CareyUtils
import pandas as pd
from CareyConstants import CareyConstants
import CareyPlots
import time
import cupy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
import scipy.signal
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import ssm
import joypy

datadir = r'X:\data\2022\BATCH5'
neural  =  os.path.join(datadir, r'recordings\VIV_23058\VIV_23058_S10_g1\kilosort4_catgt_tshift')
new_proc = os.path.join(datadir, r'processing\VIV_23058\S10\locopixels')
dp = neural # for npyx readability


recompute_firing_rates          = False
skip_to_reading_session_df      = True
run_neural_decoding             = False
organize_by_phase               = True
run_lds                         = False

print('loading neural data')
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

# start = time.clock()
# fr = CareyEphys.estimate_gaussconv_based_FR(spikes, TOTAL_TIME, fs=SAMP_RATE, gaussdev=0.020)
# stop = time.clock()-start


## iterate through all the good units and save them in a dataframe
# todo: should turn this into a function
if recompute_firing_rates:
    print('computing firing rates')
    t_arr = np.arange(0, TOTAL_TIME, 1 / 1000, dtype=float)
    cupy_t_arr = cupy.array(t_arr)
    cupy_t_full = cupy.arange(0, TOTAL_TIME, 1 / SAMP_RATE, dtype=float)
    df = pd.DataFrame(data=t_arr, columns=['time'])

    for neuron in tqdm(good_units):
        spikes = npyx.spk_t.trn(dp, neuron)/ SAMP_RATE
        fr = CareyEphys.estimate_gaussconv_based_FR(spikes, TOTAL_TIME, fs=SAMP_RATE, gaussdev=0.0001)
        fr_downsampled = cupy.interp(cupy_t_arr, cupy_t_full, cupy.array(fr)).get()
        # df = pd.DataFrame(data=fr_downsamped, columns=[str(neuron)])
        # df.to_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'), mode='a', header=True, index=False)
        df[str(neuron)] = fr_downsampled

    df.to_csv(os.path.join(new_proc, 'sessionwise_firing_rates_diracdelta.csv'))
else:
    if not skip_to_reading_session_df:
        print('reading firing rates')
        df = pd.read_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'))

## get stride info
if not skip_to_reading_session_df:
    print('loading stride info')
    behav = pd.read_csv(os.path.join(new_proc, '..', 'VIV_23058_S10_behavioral_descriptor.csv'),
                        usecols=['sessionwise_time', 'FRx', 'HRx', 'FLx', 'HLx', 'wheel_speed',
                                 'FR_SwOn', 'FR_StOn',
                                 'HR_SwOn', 'HR_StOn',
                                 'FL_SwOn', 'FL_StOn',
                                 'HL_SwOn', 'HL_StOn'])

    print('geting stride phase from events')
    start = time.clock()
    behav['FR_SwPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.FR_SwOn, behav.FR_StOn, usegpu=0)
    behav['FR_StPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.FR_StOn, behav.FR_SwOn, usegpu=0)
    behav['HR_SwPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.HR_SwOn, behav.HR_StOn, usegpu=0)
    behav['HR_StPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.HR_StOn, behav.HR_SwOn, usegpu=0)
    behav['FL_SwPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.FL_SwOn, behav.FL_StOn, usegpu=0)
    behav['FL_StPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.FL_StOn, behav.FL_SwOn, usegpu=0)
    behav['HL_SwPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.HL_SwOn, behav.HL_StOn, usegpu=0)
    behav['HL_StPh'], __, __ = CareyBehavior.get_stride_phase_from_events(behav.HL_StOn, behav.HL_SwOn, usegpu=0)
    stop_cpu = time.clock()-start
# plt.plot(behav.FR_SwOn.values)
# plt.plot(behav.FR_StOn.values)
# plt.plot(FRSwPhase)

## load dataframe of some mossy fibers
if not skip_to_reading_session_df:
    print('loading mossy fiber firing rates')
    selected_MFs = ['time', 317, 322, 351, 354, 367, 393, 421, 542, 453, 457, 467, 479, 500, 501, 503, 507, 601, 602, 604, 611, 613]
    cols = [str(xx) for xx in selected_MFs]
    # mossy = pd.read_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'), usecols=cols)
    # mossy = pd.read_csv(os.path.join(new_proc, "sessionwise_firing_rates_small.csv"), usecols=['time', '611'])
    mossy = df[cols]
    print('interpolate phases onto mossy fibers')
    mossy['FR_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FR_SwPh'].values)
    mossy['FR_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FR_StPh'].values)
    mossy['HR_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HR_SwPh'].values)
    mossy['HR_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HR_StPh'].values)
    mossy['FL_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FL_SwPh'].values)
    mossy['FL_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FL_StPh'].values)
    mossy['HL_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HL_SwPh'].values)
    mossy['HL_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HL_StPh'].values)

## do the mossy fibers tile the phase space?
# stable_time = np.logical_and(mossy['time'] >= 4000, mossy['time'] < 4060)
# subset = mossy[stable_time]
# subset.reset_index(inplace=True, drop=True)

# stride_indices = CareyUtils.get_stride_indices_from_phase(subset['FR_SwPh'].values, threshold=0.001)

# maxpeaks = scipy.signal.find_peaks(subset['FR_SwPh'])[0]
# minpeaks = scipy.signal.find_peaks(-subset['FR_SwPh'])[0]
# stride_indices = np.zeros_like(subset['FR_SwPh'].values)
# for idx in minpeaks:
#         stride_indices[idx:] += 1
# for idx in np.arange(1, minpeaks.shape[0]+1):
#     max_idx = subset['FR_SwPh'][stride_indices == idx - 1].index[subset['FR_SwPh'][stride_indices == idx - 1].argmax()]
#     next_stride = subset['FR_SwPh'][stride_indices==idx].index.min()
#     stride_indices[max_idx:next_stride] = 0
#
# # something is wrong here, we have over 9000 strides in 600s, that's more than 15 strides per second. should be around 5
# stride_indices = np.array(stride_indices, dtype=int)
# strides = np.unique(stride_indices)
# phase_array = np.linspace(0, 1, 100)
# firing_rates = np.tile(np.zeros_like(phase_array), (strides.shape[0], 1))
# for ii in strides:
#     firing_rates[ii, :] = np.interp(phase_array, subset['FR_SwPh'][stride_indices==ii], subset['611'][stride_indices==ii])
# plt.plot(phase_array, firing_rates.mean(0))

##
# mossy.index = pd.to_datetime(mossy.index) # all in nanosecs
# mossy['intphase'] = (mossy.FR_SwPh*200).astype(int)
# mossy['dtphase'] = pd.to_datetime(mossy.intphase, unit='s')
# mossy.groupby('dtphase').mean()
# sns.lineplot(data=mossy.groupby('dtphase').mean(), x='FR_SwPh', y='611')
# mossy.resample(rule='10s')




## firing rates of population over FR phase

# mossy = pd.read_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'), usecols=cols)
# mossy['FR_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FR_SwPh'].values)
# mossy['FR_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FR_StPh'].values)
# mossy['HR_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HR_SwPh'].values)
# mossy['HR_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HR_StPh'].values)
# mossy['FL_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FL_SwPh'].values)
# mossy['FL_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['FL_StPh'].values)
# mossy['HL_SwPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HL_SwPh'].values)
# mossy['HL_StPh'] = np.interp(mossy['time'].values, behav['sessionwise_time'].values, behav['HL_StPh'].values)

# stable_time = np.logical_and(mossy['time'] >= 4000, mossy['time'] < 4600)
# subset = mossy[stable_time]
# subset.reset_index(inplace=True, drop=True)
##
# stride_indices = CareyUtils.get_stride_indices_from_phase(subset['FR_SwPh'].values, threshold=0.001)
# maxpeaks = scipy.signal.find_peaks(subset['FR_SwPh'])[0]
# minpeaks = scipy.signal.find_peaks(-subset['FR_SwPh'])[0]
# stride_indices = np.zeros_like(subset['FR_SwPh'].values)
# for idx in minpeaks:
#         stride_indices[idx:] += 1
# for idx in np.arange(1, minpeaks.shape[0]+1):
#     max_idx = subset['FR_SwPh'][stride_indices == idx - 1].index[subset['FR_SwPh'][stride_indices == idx - 1].argmax()]
#     next_stride = subset['FR_SwPh'][stride_indices==idx].index.min()
#     stride_indices[max_idx:next_stride] = 0
#
# # something is wrong here, we have over 9000 strides in 600s, that's more than 15 strides per second. should be around 5
# stride_indices = np.array(stride_indices, dtype=int)
# strides = np.unique(stride_indices)
# phase_array = np.linspace(0, 1, 100)
# firing_rates = np.tile(np.zeros_like(phase_array), (strides.shape[0], 1))
# for ii in strides:
#     firing_rates[ii, :] = np.interp(phase_array, subset['FR_SwPh'][stride_indices==ii], subset['611'][stride_indices==ii])
# plt.plot(phase_array, firing_rates.mean(0))

# "Y:\Processed data\Behavioral manifold\behavioral_manifold.csv"

##
# contact neurobehav together
if not skip_to_reading_session_df:
    session = behav
    # session.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'filename_avi', 'filename_csv', 'trialwise_idx',
    #                     'syncpulse',  '0', '1', '2'], inplace=True)
    session['FR_SwStrides'] = CareyUtils.get_stride_indices_from_phase(session['FR_SwPh'].values)
    session['FR_StStrides'] = CareyUtils.get_stride_indices_from_phase(session['FR_StPh'].values)

    # append neural data to behavior by time interpolation
    selected_mossy_fibers = cols[1:]
    for neuron in tqdm(selected_mossy_fibers):
        session[neuron] = np.interp(session['sessionwise_time'], mossy['time'], mossy[neuron])

    neur = selected_mossy_fibers[0]
    phase_array = np.linspace(0, 1, 200)
    this_neuron = np.zeros((session.FR_SwStrides.max()-1, phase_array.shape[0]))
    for stride in tqdm(np.arange(1, session.FR_SwStrides.max())):
        this_neuron[stride-1, :] = np.interp(phase_array,
                                             session['FR_StPh'][session['FR_StStrides']==stride],
                                             session[neur][session['FR_StStrides']==stride])
# for stride in tqdm(np.arange(1, session.FR_SwStrides.max())):
#     this_neuron[stride-1, :] = np.interp(phase_array,
#                                          session['FR_SwPh'][session['FR_SwStrides']==stride],
#                                          session[neur][session['FR_SwStrides']==stride])
## cupy
# neur = selected_mossy_fibers[0]
# phase_array = cupy.linspace(0, 1, 200)
# this_neuron = cupy.zeros((session.FR_SwStrides.max()-1, phase_array.shape[0]))
# for stride in tqdm(np.arange(1, session.FR_SwStrides.max())):
#     this_neuron[stride-1, :] = cupy.interp( phase_array,
#                                             cupy.array(session['FR_SwPh'][session['FR_SwStrides']==stride]),
#                                             cupy.array(session[neur][session['FR_SwStrides']==stride]))





## neural decoding from mossy fibers
if run_neural_decoding:
    session[['FRx', 'HRx', 'FLx', 'HLx']] = \
        session[['FRx', 'HRx', 'FLx', 'HLx']].apply(lambda x: x*CareyConstants.NPXRIG_MM_PER_PX)

    train_idx = np.logical_and(session.sessionwise_time>4000, session.sessionwise_time<5000)
    test_idx  = np.logical_and(session.sessionwise_time>5000, session.sessionwise_time<6000)
    train = session[train_idx]
    train = train[['317', '322', '351', '354', '367', '393', '421', '542', '453', '457', '467', '479', '500', '501', '503',
                 '507', '601', '602', '604', '611', '613', 'FRx', 'HRx', 'FLx', 'HLx']]
    train_x = train[['317', '322', '351', '354', '367', '393', '421', '542', '453', '457', '467', '479', '500', '501', '503',
                   '507', '601', '602', '604', '611', '613']]
    train_y = train[['FRx', 'HRx', 'FLx', 'HLx']]
    train_y.interpolate(inplace=True)

    test = session[test_idx]
    test = test[['317', '322', '351', '354', '367', '393', '421', '542', '453', '457', '467', '479', '500', '501', '503',
                 '507', '601', '602', '604', '611', '613', 'FRx', 'HRx', 'FLx', 'HLx']]
    test_x = test[['317', '322', '351', '354', '367', '393', '421', '542', '453', '457', '467', '479', '500', '501', '503',
                   '507', '601', '602', '604', '611', '613']]
    test_y = test[['FRx', 'HRx', 'FLx', 'HLx']]
    test_y.interpolate(inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(train_x.values)
    y = train_y.values

    neural_decoder = LinearRegression().fit(X, y)
    pred_y = neural_decoder.predict(scaler.transform(test_x))

    sklearn.metrics.r2_score(test_y, pred_y, multioutput='raw_values')

    test_y['time'] = session['sessionwise_time'][test_y.index]

    comparison = test_y
    comparison['FRx_pred'] = pred_y[:, 0]
    comparison['HRx_pred'] = pred_y[:, 1]
    comparison['FLx_pred'] = pred_y[:, 2]
    comparison['HLx_pred'] = pred_y[:, 3]

    plt.figure()
    sns.lineplot(data=test_y, x='time', y='FRx', color=sns.color_palette()[3])
    sns.lineplot(data=test_y, x='time', y='FRx_pred', linestyle='--', color='black')
    plt.legend(['FRx', 'FRx pred'])

    fig, axs = plt.subplots(2,1)
    axs = axs.flatten()
    axs[0].plot(X.transpose())
    axs[1].plot(y.transpose())

## make dataframe by interpolating over phase
# loop for strides
loc_cycle = 1
#
# cupy.apply_along_axis(  cupy.interp,
#                         0,
#                         cupy_phase_array,
#                         cupy.array(session['FR_SwPh'][session['FR_SwStrides']==loc_cycle]),
#                         cupy.array(session['611'][session['FR_SwStrides']==loc_cycle]) )
# neural_activity_by_phase = np.zeros((phase_array.shape[0], len(selected_mossy_fibers)))
# for stride in tqdm(np.arange(1, np.max(session.FR_SwStrides)), position=0, desc='stride', leave=True):
#     for index, neuron in enumerate(tqdm(selected_mossy_fibers, leave=True, position=1, desc='neurons', disable=True)):
#         data_phase = session['FR_SwPh'][session['FR_SwStrides']==stride].values
#         neural_activity_by_phase[:,index] = np.interp(  phase_array,
#                                                         data_phase,
#                                                         session[neuron][session['FR_SwStrides']==stride].values)
##
if organize_by_phase:
    session = pd.read_csv(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\sessionwise_firing_rates.csv")
    n_strides = np.max(session.FR_SwStrides)-1
    neurons_by_phase = pd.DataFrame(data=None, columns=['time', 'phase', 'FR_SwStrides'] + selected_mossy_fibers)
    neural_activity_by_phase = np.zeros((phase_array.shape[0], len(selected_mossy_fibers), n_strides))
    time = np.zeros((phase_array.shape[0], n_strides))
    for stride in tqdm(np.arange(1, np.max(session.FR_SwStrides)), position=0, desc='stride', leave=True):
        for neuron_index, neuron in enumerate(tqdm(selected_mossy_fibers, leave=False, position=0, desc='neurons', disable=True)):
            data_phase = session['FR_SwPh'][session['FR_SwStrides']==stride].values
            neural_activity_by_phase[:, neuron_index, stride] = \
                np.interp(phase_array, data_phase, session[neuron][session['FR_SwStrides']==stride].values)
            time[:, stride] = np.interp(phase_array, data_phase, session['sessionwise_time'][session['FR_SwStrides']==stride].values)
    np.save(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\neural_activity_by_phase.npy", neural_activity_by_phase)
    np.save(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\time.npy", time)

##
neural_activity_by_phase = np.load(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\neural_activity_by_phase.npy")
avg_firing = neural_activity_by_phase.mean(2).transpose()
avg_firing = CareyUtils.zscore(avg_firing.transpose(), 0).transpose()
plt.figure()
order = np.argsort(np.argmax(avg_firing, 1))
plt.imshow(avg_firing[order,:], vmin=-2, vmax=2, cmap='bwr', aspect='auto')
plt.colorbar()

# data_for_df = np.ravel(avg_firing)
# data_for_df = np.vstack((data_for_df, np.tile(phase_array, len(selected_mossy_fibers)))).transpose()
# df_firing_rates_mossy = pd.DataFrame(data=data_for_df, columns=['FR', 'phase'])
# df_firing_rates_mossy['neuron'] = np.tile(np.array(selected_mossy_fibers), (phase_array.shape[0],1)).transpose().ravel()
# a = joypy.joyplot(df_firing_rates_mossy, column='FR', by='neuron', fill=False, linecolor='k')
# plt.show()
##
example_trial = neural_activity_by_phase[:,:, np.random.choice(neural_activity_by_phase.shape[2])]
plt.figure()
CareyPlots.ridgeplot(example_trial, multiplier=2.5, spacing=5)
plt.show()

##
# lds where A acts over neural data and C transforms it into behavior
if run_lds:
    session = pd.read_csv(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\session_neural_behavior.csv")
    units = ['317', '322', '351', '354', '367', '393', '421', '542', '453', '457', '467', '479', '500', '501', '503',
             '507', '601', '602', '604', '611', '613']
    N = len(units)
    D = 6
    y = session[units].values
    y = y[:np.int(y.shape[0]*0.10)]
    lds = ssm.LDS(N, D)
    lds.initialize(y, verbose=2)

    plt.figure()
    plt.imshow(lds.dynamics.A)
    plt.colorbar()

    q_mf_elbos, q_mf = lds.fit(y,
                               method="bbvi",
                               variational_posterior="mf",
                               num_iters=1000, stepsize=0.1,
                               initialize=False)
    q_mf_x = q_mf.mean[0]
    q_mf_y = lds.smooth(q_mf_x, y)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(q_mf_x[:3000, 0], q_mf_x[:3000, 1], q_mf_x[:3000, 2], s=0.1)

    session_subset = session[:y.shape[0]]
    session_subset['latent1'] = q_mf_x[:, 0]
    session_subset['latent2'] = q_mf_x[:, 1]
    session_subset['latent3'] = q_mf_x[:, 2]
    session_subset['latent4'] = q_mf_x[:, 3]
    session_subset['latent5'] = q_mf_x[:, 4]
    session_subset['latent6'] = q_mf_x[:, 5]
##
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.ion()
    plt.show()
    for strides in np.arange(1000, 1010):
        for_plotting = session_subset[session_subset['FR_SwStrides']==strides]
        ax.plot(for_plotting['latent2'], for_plotting['latent3'], for_plotting['latent4'], linewidth=1)
        plt.draw()
        plt.pause(1)
    plt.xlabel('1')
    plt.ylabel('2')
##
    fig = plt.figure()
    for_plotting = session_subset[np.logical_and(session_subset['FR_SwStrides']>1000, session_subset['FR_SwStrides']<1020)]
    sns.lineplot(data=for_plotting, x='latent1', y='latent5', hue='FR_SwStrides', estimator=None, sort=False, palette='Spectral')
    plt.show()

##
T = 300
latents = [1, 3]
plt.figure()
plt.plot(lds.sample(T)[0][:, latents[0]], lds.sample(T)[0][:, latents[1]])

T = 10000
latents = [1, 3]
plt.figure()
plt.plot(lds.sample(T)[0][:, latents[0]])
plt.plot(lds.sample(T)[0][:, latents[1]])