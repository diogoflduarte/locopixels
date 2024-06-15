# import deeplabcut
import os
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import seaborn as sns
from CareyConstants import CareyConstants
import scipy.signal
import SwSt_Det_SlopeThres
import cupy
import scipy.linalg

def plot_tracks_from_file(file_from_DLC, coord='x', fps=430, filter=None, filtwidth=3):

    df = pd.read_hdf(file_from_DLC)
    # get scorer name for the multiindex
    scorer = df.columns[0][0]

    plot_tracks_from_DLC_df(df, coord=coord, filter=filter, filtwidth=filtwidth)

    return  df

def plot_tracks_from_DLC_df(df, coord='x', fps=430, filter=None, filtwidth=3):

    scorer = df.columns[0][0]
    time_array = np.linspace(0, df.shape[0] / fps, df.shape[0])
    df[scorer, 'general', 'time'] = time_array

    # iterate through paws
    paw_bottom_list = ['FR_bottom', 'HR_bottom', 'FL_bottom', 'HL_bottom']

    for paw in paw_bottom_list:

        paw_idx = CareyConstants.paw_idx[paw[:2]]

        this_paw = df[scorer][paw]
        this_paw['time'] = time_array

        if filter=='median':
            this_paw[coord] = scipy.signal.medfilt(this_paw[coord].values, filtwidth)

        sns.lineplot(data=this_paw, y=coord, x='time', color=CareyConstants.paw_colors_sns[paw_idx])

    pass

def import_essential_coords_from_DLC_file(file_from_DLC):
    iterables = [['FR_bottom', 'HR_bottom', 'FL_bottom', 'HL_bottom'], ['x', 'y']]
    iterables_z = [['FR_side', 'HR_side', 'FL_side', 'HL_side'], ['y']]
    columns_of_interest = pd.MultiIndex.from_product(iterables, names=['bodyparts', 'coords'])
    # TODO: complete this?

def SwingAndStanceDetection(tracksfile, strides_filename, CleanArtifs=True, FiltCutOff=60, Acq_Freq=430,
                                        SwSt_Outlier_Rej=True, Type_Experiment=1, graph=False, verbose=True,
                                        Belts_Dict=None):
    # TODO: this should call Jorge's function on one file, the iterator to all folders should be in NeuropixelsExperiment
    if verbose:
        print(f'Running swing and stance detection on: {tracksfile} ...')

    mismatch = Belts_Dict['Speed Left'].shape[0] - pd.read_hdf(tracksfile).shape[0]
    if mismatch == 1:
        Belts_Dict['Speed Left']    = Belts_Dict['Speed Left'][:-1]
        Belts_Dict['Speed Right']   = Belts_Dict['Speed Right'][:-1]
        Belts_Dict['Trial']         = Belts_Dict['Trial'][:-1]
        Belts_Dict['filename']      = Belts_Dict['filename'][:-1]

    if mismatch == -1:
        Belts_Dict['Speed Left']    = np.append(Belts_Dict['Speed Left'],    Belts_Dict['Speed Left'][-1])
        Belts_Dict['Speed Right']   = np.append(Belts_Dict['Speed Right'],   Belts_Dict['Speed Right'][-1])
        Belts_Dict['Trial']         = np.append(Belts_Dict['Trial'],         Belts_Dict['Trial'][-1])
        Belts_Dict['filename']      = np.append(Belts_Dict['filename'],      Belts_Dict['filename'][-1])

    # todo: compare beltdict with trackfile, if length is higher in trackfile interpolate, if not crop the dict

    SwSt_Det_SlopeThres.Pocket_SwSt_Det(tracksfile, CleanArtifs=CleanArtifs, FiltCutOff=FiltCutOff, Acq_Freq=Acq_Freq,
                                        SwSt_Outlier_Rej=SwSt_Outlier_Rej, Type_Experiment=Type_Experiment,
                                        graph=graph, save=strides_filename, Belts_Dict=Belts_Dict)
    if verbose:
        print('Done!')
        print(' ')

def get_stride_phase_from_events(FR_SwOn, FR_StOn, usegpu=True): # fix input bug
    # todo: bugged
    if usegpu:
        phase_continuous = cupy.zeros(FR_SwOn.shape)
        phase_inflexion  = cupy.zeros(FR_SwOn.shape)
        stride_indices   = np.zeros(FR_SwOn.shape, dtype=int)
    else:
        phase_continuous = np.zeros(FR_SwOn.shape)
        phase_inflexion  = np.zeros(FR_SwOn.shape)
        stride_indices   = np.zeros(FR_SwOn.shape, dtype=int)

    stride_count = 1

    # Find indices of the True values in FR_SwOn and FR_SwOff
    on_indices = np.where(FR_SwOn)[0]
    off_indices = np.where(FR_StOn)[0]

    # Iterate through the FR_SwOn indices to build the signal
    for i in range(len(on_indices) - 1):
        start = on_indices[i]
        next_start = on_indices[i + 1]

        # Find the off index between the current start and next start
        off_index = off_indices[(off_indices > start) & (off_indices < next_start)]

        if len(off_index) > 0:
            off_index = off_index[0]  # Use the first off_index in the interval

            # Create a linear increase from start to next_start (exclusive)
            if  usegpu:
                phase_continuous[start:next_start]      = cupy.linspace(0, 1, next_start - start, endpoint=False)
                phase_inflexion[start:off_index]        = cupy.linspace(0, 0.5, off_index - start, endpoint=False)
                phase_inflexion[off_index:next_start]   = cupy.linspace(0.5, 1, next_start - off_index, endpoint=False)
            else:
                phase_continuous[start:next_start]      = np.linspace(0, 1, next_start - start, endpoint=False)
                phase_inflexion[start:off_index]        = np.linspace(0, 0.5, off_index - start, endpoint=False)
                phase_inflexion[off_index:next_start]   = np.linspace(0.5, 1, next_start - off_index, endpoint=False)
            # Mark the stride index in the stride_indices array
            stride_indices[start:next_start] = stride_count
            stride_count += 1

    if usegpu:
        phase_continuous    = phase_continuous.get()
        phase_inflexion     = phase_inflexion.get()

    return phase_continuous, phase_inflexion, stride_indices
def anchored_relative_dist(A, B, anchor=0):
    '''
    computes the procrustes distance with alignment around one anchor point instead of mean
    :param A:
    :param B:
    :param anchor: 0 index by default
    :return:
    '''

    A -= A[0]
    B -= B[0]

    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)

    A /= normA
    B /= normA

    u, w, vt = scipy.linalg.svd(B.T.dot(A).T)
    R = u.dot(vt)
    scale = w.sum()

    B = np.dot(B, R.T) * scale
    disparity = np.sum((np.square(A - B)))

    return B, dist