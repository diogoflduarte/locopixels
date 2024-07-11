import os
import matplotlib
import timeit
import time

matplotlib.use('TkAgg')
import CareyPlots
import CareyLib
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

RUN_STRIDE_SEGMENTATION = True

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
experiment.nidq_bin = os.path.join(nibin_path, mouse, basename + '_g0', basename + '_g0_t0.nidq.bin')
experiment.nidq_meta = os.path.join(nibin_path, mouse, session, basename + '_g0_t0.nidq.meta')
experiment.npx_apbin = os.path.join(apbin_path, mouse,
                                    basename + '_g0', basename + '_g0_imec0', basename + '_g0_t0.imec0.ap.bin')
experiment.npx_apmeta = os.path.join(apbin_path, mouse,
                                     basename + '_g0', basename + '_g0_imec0', basename + '_g0_t0.imec0.ap.meta')
experiment.behavior_videos = os.path.join(base_path, 'behavior', mouse, session)
experiment.processing_dir = os.path.join(base_path, 'processing', mouse, session)
if not os.path.exists(experiment.processing_dir):
    os.makedirs(experiment.processing_dir)

## get wheel speed from metadata
if not os.path.exists(os.path.join(experiment.processing_dir, 'wheel.h5')):
    print('Estimating wheel speed from rotary encoders...')
    experiment.compile_wheel_speed(outfile=os.path.join(experiment.processing_dir, 'wheel.h5'))
    print('Done')

## B1) fix behavior filenames from bonsai timestamps to trial number
FILES_NEED_RENAMING = False
if FILES_NEED_RENAMING:
    CareyFileProcessing.standardizeBonsaiVideoFileNames(experiment.behavior_videos,
                                                        extensions=['.avi', '.csv'], mouse_name='VIV_27537',
                                                        session='S10', delete=True, log=True, dryrun=True)

## B2) Track with DLC
# DONE OUTSIDE


## NB1)  load syncpulse from the nibin file and save to a numpy file
extract_npx             = 0
extract_nidq            = 0
extract_behav_videos    = 0
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_apbin_sync.npy')):
    extract_npx     = 1
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_nidq_sync.npy')):
    extract_nidq    = 1
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_behavior_metadata.csv')):
    extract_behav_videos = 1
experiment.extractSyncronizationStreams( extract_npx=extract_npx,
                                         extract_nidq=extract_nidq,
                                         extract_behav_videos=extract_behav_videos, sortby='trial')
print('Done')

## NB2) extract camera metadata into sessionwise file
if not os.path.exists(os.path.join(experiment.processing_dir, basename + '_behavior_metadata.csv')):
    CareyLib.NeuropixelsExperiment.compileSessionwiseMetadataBehavior(
        experiment.behavior_videos,
        save_to_file=os.path.join(experiment.processing_dir, basename + '_behavior_metadata.csv'),
        verbose=False)
