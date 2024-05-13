# import deeplabcut
import os
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import seaborn as sns
from CareyConstants import CareyConstants
import scipy.signal
import SwSt_Det_SlopeThres

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
