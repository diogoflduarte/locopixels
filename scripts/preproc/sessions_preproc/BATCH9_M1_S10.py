import os
import matplotlib
import timeit
import time
import plotext
plotext.theme('clear')

matplotlib.use('TkAgg')
import CareyPlots
import CareyLib
import CareyBehavior
import matplotlib.pyplot as plt
import numpy as np
import ibllib.io.spikeglx
import pandas as pd
import CareyUtils
import scipy.signal
import platform
import CareyEphys
import CareyFileProcessing
import subprocess
from tqdm import tqdm


## ..............................   SETTING PREPROCESSING PARAMETERS ...................................................
RUN_STRIDE_SEGMENTATION = True
DO_SYNC                 = True # cameras only
dt_bcam                 = 1/432
confThresh              = 0.05
tCov                    = 1.0
obsCov                  = 1.0
KALMAN_SMOOTH           = False
extract_npx             = 0
extract_nidq            = 0
extract_behav_videos    = 0
# ......................................................................................................................

# PATH SETUP
if platform.platform().__contains__('Windows'):
    base_path = r'X:\data\2022\BATCH9'
elif platform.platform().__contains__('Darwin'):
    base_path = '/Volumes/DiogoDuarte/data/2022/BATCH9'
else:
    base_path = '/home/diogo/megands1/data/2022/BATCH9'

nibin_path = os.path.join(base_path, 'recordings')
apbin_path = os.path.join(base_path, 'recordings')
mouse_list = ['VIV_27537']
session_list = ['S10']



mouse = mouse_list[0]
session = session_list[0]

basename = (mouse + '_' + session)
prefix = basename + '_'

# set the experiment variables
experiment = CareyLib.NeuropixelsExperiment()
experiment.mouse = mouse
experiment.session = session
experiment.nidq_bin = os.path.join(nibin_path, mouse,  basename + '_g0', basename + '_g0_t0.nidq.bin')
experiment.nidq_meta = os.path.join(nibin_path, mouse, basename + '_g0', basename + '_g0_t0.nidq.meta')
experiment.npx_apbin = os.path.join(apbin_path, mouse,
                                    basename + '_g0', basename + '_g0_imec0', basename + '_g0_t0.imec0.ap.bin')
experiment.npx_apmeta = os.path.join(apbin_path, mouse,
                                     basename + '_g0', basename + '_g0_imec0', basename + '_g0_t0.imec0.ap.meta')
experiment.behavior_videos = os.path.join(base_path, 'behavior', mouse, session)
experiment.processing_dir = os.path.join(base_path, 'processing', mouse, session)
if not os.path.exists(experiment.processing_dir):
    os.makedirs(experiment.processing_dir)

## B1) fix behavior filenames from bonsai timestamps to trial number
FILES_NEED_RENAMING = False
if FILES_NEED_RENAMING:
    CareyFileProcessing.standardizeBonsaiVideoFileNames(experiment.behavior_videos,
                                                        extensions=['.avi', '.csv'], mouse_name='VIV_27537',
                                                        session='S10', delete=True, log=True, dryrun=True)

## B2) Track with DLC
# DONE OUTSIDE

####################    SYNCHRONIZING DATA STREAMS #####################################################################

# NB1)  load syncpulse from the nibin file and save to a numpy file
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_apbin_sync.npy')):
    extract_npx     = 1
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_nidq_sync.npy')):
    extract_nidq    = 1
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_behavior_metadata.csv')):
    extract_behav_videos = 1
    experiment.extractSyncronizationStreams(extract_npx=extract_npx,
                                            extract_nidq=extract_nidq,
                                            extract_behav_videos=extract_behav_videos, sortby='trial')
    print('Metadata for behavior extracted, along with synchronization streams')


# generate corrected time file by synchronizing the ni_bin based on the ap_bin
if os.path.exists(os.path.join(experiment.processing_dir, basename + '_nidq_time_array.npy')):
    nidq_time = np.load(os.path.join(experiment.processing_dir, basename + '_nidq_time_array.npy'))
    npx_time  = np.load(os.path.join(experiment.processing_dir, basename + '_apbin_time_array.npy'))
else:
    nidq_time, npx_time = experiment.synchronizeDataStreams()


## estimate wheel speed from encoders because we'll need it for the stride segmentation
wheel_csv = os.path.join(experiment.processing_dir, basename + '_wheel.csv')
if not os.path.exists(wheel_csv):
    print('Estimating wheel speed from rotary encoders...')
    wheel_df = experiment.compile_wheel_speed(outfile=wheel_csv)
    print('Done')


## 3) correct the camera times for each frame
behavior_metadata   = os.path.join(experiment.processing_dir, basename + '_behavior_metadata.csv')
syncpulse_file      = os.path.join(experiment.processing_dir, basename + '_nidq_sync.npy')
metadata_corrected  = os.path.join(experiment.processing_dir, basename + '_behavior_metadata_corrected.csv')
if not os.path.exists(metadata_corrected):
    print('Synchronizing behavioral camera times based on neuropixels time via external sync pulse.')
    slopes, intercepts = experiment.correctCameraTime(behavior_metadata, syncpulse_file, align_to='npx', outputfile=metadata_corrected)
    plotext.scatter(np.diff(intercepts), slopes, marker='zero')
    plotext.plotsize(40, 12)
    plotext.show()
    plotext.clc()


## 4.1) add tracks
tracks_file = os.path.join(experiment.processing_dir, basename + '_DLC_tracks.csv')
tracks_df = experiment.compileSessionwiseTracks(experiment.behavior_videos, match='shuffle1', ext='h5', mouse_name=True)
tracks_df.to_csv(tracks_file)

## Optional: smooth based on likelihoods
if KALMAN_SMOOTH:
    featmap = [['FRx', 'FR_b_lik'], ['FRy', 'FR_b_lik'], ['FRz', 'FR_s_lik'],
               ['HRx', 'HR_b_lik'], ['HRy', 'HR_b_lik'], ['HRz', 'HR_s_lik'],
               ['FLx', 'FL_b_lik'], ['FLy', 'FL_b_lik'], ['FLz', 'FL_s_lik'],
               ['HLx', 'HL_b_lik'], ['HLy', 'HL_b_lik'], ['HLz', 'HL_s_lik'],
               ['Tail1x', 'Tail1_b_lik'], ['Tail1y', 'Tail1_b_lik'], ['Tail1z', 'Tail1_s_lik'],
               ['Tail2x', 'Tail2_b_lik'], ['Tail2y', 'Tail2_b_lik'], ['Tail2z', 'Tail2_s_lik']]
    print('Kalman smoothing features...')


    # feat = featmap[9]
    # thresh = np.quantile(tracks_df[feat[1]], 0.05)
    # tracks_df['threshold_met'] = np.where(tracks_df[feat[1]] >= thresh, 1, 0)
    # df_toplot = tracks_df[tracks_df['trial']==1]
    # # sns.scatterplot(data=df_toplot, x='frame', y=feat[0], hue='FRx')
    # CareyPlots.multicoloredline_2d(df_toplot, 'frame', feat[0], 'threshold_met', cmap='PiYG', lw=2)

    groups = tracks_df.groupby('trial')
    for trial, group in tqdm(groups):
        for ii, fm in enumerate(featmap):
            thresh = np.quantile(tracks_df[fm[1]], confThresh)
            tracks_df.loc[group.index, fm[0]] = \
                    CareyBehavior.kalman_smooth_low_confidence_tracks(
                        tracks_df.loc[group.index], fm[0], fm[1], dt=dt_bcam, confThresh=thresh, tCov=tCov,
                        obsCov=obsCov)
    kalman_smoothed_tracks_file = tracks_file = os.path.join(experiment.processing_dir, basename + '_DLC_tracks_smoothed.csv')
    tracks_df.to_csv(os.path.join(tracks_file))

## ADD CAMERA METADATA TO TRACKS FILE. THIS IS NOT OPTIONAL
df_metadata = pd.read_csv(metadata_corrected)
df_tracks   = pd.read_csv(tracks_file)
df = CareyLib.NeuropixelsExperiment.addTracksToMetadata(df_metadata, df_tracks)

## ADD WHEEL DISTANCE AND SPEED
wheel_df = pd.read_csv(wheel_csv)
df = CareyLib.NeuropixelsExperiment.addWheelDistanceAndSpeed(df,    wheel_df['speed'].values,
                                                                    wheel_df['distance'].values,
                                                                    wheel_df['time'])
tracks_df.to_csv(os.path.join(experiment.processing_dir, basename + '_behavioral_descriptor.csv'))

## TODO: re-run stride segmentation

