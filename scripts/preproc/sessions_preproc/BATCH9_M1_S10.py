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

nibin_path = os.path.join(base_path, 'wheel_enc')
apbin_path = os.path.join(base_path, 'recordings')
mouse_list = ['VIV_27537']
session_list = ['S10']

mouse = mouse_list[0]
session = session_list[0]

basename = (mouse + '_' + session)
prefix = basename + '_'

# set the experiment variables
VIV_27537_S10 = CareyLib.NeuropixelsExperiment()
VIV_27537_S10.mouse = mouse
VIV_27537_S10.session = session
VIV_27537_S10.nidq_bin = os.path.join(nibin_path, mouse, session,
                                      'VIV_27537_S10_g0_t0.nidq.bin')
VIV_27537_S10.nidq_meta = os.path.join(nibin_path, mouse, session,
                                       'VIV_27537_S10_g0_t0.nidq.meta')
VIV_27537_S10.npx_apbin = os.path.join(apbin_path, mouse,
                                       'VIV_27537_S10_g0', 'VIV_27537_S10_g0_imec0', 'VIV_27537_S10_g0_t0.imec0.ap.bin')
VIV_27537_S10.npx_apmeta = os.path.join(apbin_path, mouse,
                                        'VIV_27537_S10_g0', 'VIV_27537_S10_g0_imec0',
                                        'VIV_27537_S10_g0_t0.imec0.ap.meta')
VIV_27537_S10.behavior_videos = os.path.join(base_path, 'behavior', mouse, session)
VIV_27537_S10.processing_dir = os.path.join(base_path, 'processing', mouse, session)
if not os.path.exists(VIV_27537_S10.processing_dir):
    os.makedirs(VIV_27537_S10.processing_dir)

## get wheel speed from metadata
if not os.path.exists(os.path.join(VIV_27537_S10.processing_dir, 'wheel.h5')):
    print('Estimating wheel speed from rotary encoders...')
    VIV_27537_S10.compile_wheel_speed(outfile=os.path.join(VIV_27537_S10.processing_dir, 'wheel.h5'))
    print('Done')

## B1) fix behavior filenames from bonsai timestamps to trial number
FILES_NEED_RENAMING = False
if FILES_NEED_RENAMING:
    CareyFileProcessing.standardizeBonsaiVideoFileNames(VIV_27537_S10.behavior_videos,
                                                        extensions=['.avi', '.csv'], mouse_name='VIV_27537',
                                                        session='S1', delete=True, log=True, dryrun=True)

## B2) Track with DLC
# DONE OUTSIDE


## NB1)  load syncpulse from the nibin file and save to a numpy file
if not os.path.exists(os.path.join(VIV_27537_S10.processing_dir, 'VIV_27537_S10_apbin_sync.npy')): # TODO THIS IS WRONG
    print('Extracting sync pulse from nibin files...')
    VIV_27537_S10.extractSyncronizationStreams(extract_npx=1, extract_nidq=1, extract_behav_videos=1, sortby='trial') # todo: try this out with npyx
    print('Done')
else:
    print('Nibin sync extraction file found')


## NB2) extract camera metadata into sessionwise file
if not os.path.exists(os.path.join(VIV_27537_S10.processing_dir, 'VIV_27537_S1_behavior_metadata.h5')):
    CareyLib.NeuropixelsExperiment.compileSessionwiseMetadataBehavior(
        VIV_27537_S10.behavior_videos,
        save_to_file=os.path.join(VIV_27537_S10.processing_dir, 'VIV_27537_S1_behavior_metadata.h5'),
        verbose=False)
