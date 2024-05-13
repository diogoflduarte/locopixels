import CareyUtils
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import ibllib.io.spikeglx
import time
import CareyLib
from CareyConstants import CareyConstants
import scipy
import pandas as pd
import sklearn
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import jit
import npyx
try:
    import cupy
    gpu_avail = True
except ImportError:
    print('Cupy not installed, not available for gaussian smoothing firing rates in CareyEphys')
    gpu_avail = False

class Spikes:
    def __init__(self, ksDir):
        self.st      = []
        self.clu     = []


def loadKsDir(ksDir, fs=CareyConstants.DEF_NPX_FS):
    '''
    spiketimes, clusters, templates, kslabels, info = CareyEphys.loadKsDir(ksDir, fs=CareyConstants.DEF_NPX_FS)

    Parameters
    ----------
    ksDir
    fs

    Returns
    -------
     spiketimes
     clusters
     templates
     labels
    '''
    files_to_load = {}
    curated = True

    # st = spike times  [nSpikes]. timing of each spike in samples
    sp_file             = os.path.join(ksDir, 'spike_times.npy')
    # clu = clusters
    clu_file            = os.path.join(ksDir, 'spike_clusters.npy')
    # stemp = template identities [nsPikes]
    template_file       = os.path.join(ksDir, 'spike_templates.npy')
    cluster_group       = os.path.join(ksDir, 'cluster_group.tsv')

    cluster_info        = os.path.join(ksDir, 'cluster_info.tsv')

    if not os.path.exists(cluster_info):
        curated = False
        print('Electrophysiology data has not yet been curated')

    # files_to_load = {'spiketimes': sp_file,
    #                  'clusters':   clu_file,
    #                  'templates':  template_file}
    # for key, value in files_to_load.items():
    #     exec("%s = np.load(%s)" % (key, value))
    spiketimes  = (np.load(sp_file)/fs).squeeze()
    clusters    = (np.load(clu_file)).squeeze()
    templates   = (np.load(template_file)).squeeze()
    kslabels    = pd.read_csv(cluster_group, sep='\t')
    if curated:
        info        = pd.read_csv(cluster_info, sep='\t')
    else:
        info = None

    return spiketimes, clusters, templates, kslabels, info
def loadSpikeGLXMetada(ksDir):
    # step 1: find metadata file
    files = os.listdir(ksDir)
    # get the file containing '.ap.bin'
    index = [idx for idx, s in enumerate(files) if '.ap.meta' in s][0]
    full_file = os.path.join(ksDir, files[index])
    # finally read the metadata itself
    metadata = ibllib.io.spikeglx.read_meta_data(full_file)
    return metadata

def spikeRaster(spike_times, event_times,
                time_bounds = [-0.1, 0.1],
                sortby = None,
                markersiz=1,
                color='k'):
    # INPUTS:
    #   spike_times (this unit only)
    #   event_times
    # sort event by times, if no other option is specified
    if sortby is None:
        sortby = np.argsort(event_times)
    event_times = event_times[sortby]

    # validate event times within bounds

    et_idx_lowbound = event_times > (min(spike_times) + time_bounds[0])
    et_idx_highbound= event_times < (max(spike_times) + time_bounds[1])
    et_idx = np.logical_and(et_idx_lowbound, et_idx_highbound)
    event_times = event_times[et_idx];

    # ax = plt.axes() ### I commented this but I'm not sure

    for ii in range(event_times.shape[0]):
        these_bounds = [event_times[ii] + time_bounds[0],
                        event_times[ii] + time_bounds[1]]
        these_spikes = spike_times[np.logical_and(spike_times>these_bounds[0],
                                                  spike_times<these_bounds[1])]
        # centre spikes around zero
        this_line = these_spikes-event_times[ii]

        # ploty plot
        tl = plt.plot(this_line, np.ones(this_line.shape[0])*ii,'.k')
        plt.setp(tl, 'markersize', markersiz, 'color',color)
def estimateFiringRate(spike_times, output_time,
                       method='step_and_smooth',
                       fs=CareyConstants.DEF_NPX_FS, # sampling rate
                       cuttoff_freq=150,
                       verbose=0):  # in Hz

    spike_idx = np.zeros(output_time.shape)
    # assign ones in spike_idx where there is a spike

    # restrict spike times to the interval of choice
    spike_times_bin_idx = np.logical_and(spike_times>=output_time[0],
                                            spike_times<=output_time[-1])
    # convert to numerical indices
    spike_times_select_idx = np.where(spike_times_bin_idx)
    # convert tuple to array
    spike_times_select_idx = spike_times_select_idx[0]

    if spike_times_select_idx.shape[0]>0:
        # include the spike before the interval
        if spike_times_select_idx[0]>0:
            spike_times_select_idx = np.insert(spike_times_select_idx, 0, spike_times_select_idx[0]-1)
        # and the spike after the interval
        if spike_times_select_idx[-1]<spike_times.shape[0]-1:
            spike_times_select_idx = np.append(spike_times_select_idx, spike_times_select_idx[-1]+1)
            spike_times_select_idx = np.append(spike_times_select_idx, spike_times_select_idx[-1] + 1)

    spike_times_iter = spike_times[spike_times_select_idx]

    firing_rate = np.zeros(output_time.shape)
    if len(spike_times_iter)>=2:

        spike_times_iter = cleanupConsecutiveSpikes(spike_times_iter,
                                                    np.diff(output_time)[0],
                                                    verbose=1)

        if method=='step':
            # generate an accomodating time vector for the included spikes at the end
            estimation_time = np.arange(spike_times_iter[0],
                                        spike_times_iter[-1],
                                        1.0/fs)
            firing_rate_wide = np.zeros(estimation_time.shape)

            for ii in tqdm(range(spike_times_iter.shape[0]-1), disable=(not bool(verbose))):
                # find time equivalent to this spike and the next spike
                this_spike_idx = np.argmin(np.abs(estimation_time-spike_times_iter[ii]))
                next_spike_idx = np.argmin(np.abs(estimation_time-spike_times_iter[ii+1]))
                firing_rate_wide[this_spike_idx:next_spike_idx] = \
                    1/(estimation_time[next_spike_idx]-estimation_time[this_spike_idx])
            # the ends of the array (before the first spike and after the last) are incomplete
            # let's deal with that

            '''
            first_spike_idx = np.argmin(np.abs(output_time-spike_times_iter[0]))
            last_spike_idx  = np.argmin(np.abs(output_time-spike_times_iter[-1]))
            firing_rate[:first_spike_idx] = firing_rate[first_spike_idx+1]
            firing_rate[last_spike_idx:-1] = firing_rate[last_spike_idx-1]
            firing_rate[-1] = firing_rate[last_spike_idx-1]
            '''

            # interpolate to the output time
            firing_rate = np.interp(output_time, estimation_time, firing_rate_wide)
            aa=1

        elif method=='step_and_smooth':
            # method by step
            output_time, firing_rate = estimateFiringRate(spike_times, output_time, method='step')

            # define bessel filter params
            nyquist_freq = fs/2
            N = 3 # order of the filter
            Wn = cuttoff_freq / nyquist_freq # normalized order filter
            btype = 'low'
            filttype = False
            outputtype = 'ba'

            # instantiate the fiter
            b, a = scipy.signal.bessel(N, Wn, btype, filttype, outputtype)
            firing_rate = scipy.signal.filtfilt(b,a,firing_rate)

    return output_time, firing_rate
def estimate_isi_based_FR(spike_times, acquisition_time, fs=CareyConstants.DEF_NPX_FS, verbose=1):
    '''

    Parameters
    ----------
    spike_times
    output_time
    fs
    verbose

    Returns
    -------

    '''
    # acquisition_time = metadata['fileTimeSecs']
    t_arr = np.arange(0, acquisition_time, 1/fs, dtype=float)

    # restrict spikes to the interval of interest
    spike_times = spike_times[ np.logical_and(spike_times>=t_arr[0], spike_times<=t_arr[-1]) ]
    isi = np.diff(spike_times)
    isi_arr = np.zeros(t_arr.shape)

    # convert spike times to spike indices so that the assignment is faster
    spike_indices = np.round(spike_times * fs).astype(int)

    # inevitably iterate over spikes
    for ii in tqdm(range(len(spike_indices)), disable=(not bool(verbose))):

        if ii == 0:
            continue # the first spike doesn't really have a reference
        elif ii == 1:
            isi_arr[:spike_indices[ii]] = isi[ii-1]
        elif ii == len(spike_indices)-1:
            isi_arr[spike_indices[ii - 1]:] = isi[ii-1]
        else:
            isi_arr[spike_indices[ii - 1]:spike_indices[ii]] = isi[ii - 1]

    return 1/isi_arr

    # downsample to desired space
def estimate_gaussconv_based_FR(spike_times, acquisition_time, fs=CareyConstants.DEF_NPX_FS, gaussdev=0.001, usegpu=True):
    '''

    Parameters
    ----------
    spike_times
    acquisition_time
    fs
    verbose

    Returns
    -------

    '''

    # convert width of a gaussian filter from seconds to samples
    sigma = gaussdev * fs

    # acquisition_time = metadata['fileTimeSecs']
    t_arr = np.arange(0, acquisition_time, 1 / fs, dtype=float)
    fr_arr = np.zeros(t_arr.shape)

    # restrict spikes to the interval of interest
    spike_times = spike_times[np.logical_and(spike_times >= t_arr[0], spike_times <= t_arr[-1])]
    spike_indices = np.round(spike_times * fs).astype(int)

    # assign 1 where there's a spike
    fr_arr[spike_indices] = 1 * fs

    myconv = CareyUtils.gaussian_smoothing(fr_arr, sigma, usegpu=usegpu)

    # kernel = CareyUtils.gaussian(0, sigma, np.round(sigma*10))
    # kernel = kernel / np.sum(kernel)
    # in_sig = np.concatenate((np.flip(fr_arr[:kernel.shape[0]]), fr_arr, np.flip(fr_arr[-kernel.shape[0]:])), axis=0)
    # myconv =        np.convolve(in_sig, kernel)
    # myconv = myconv[int(np.round((kernel.shape[0]*1.5))):]
    # myconv = myconv[:fr_arr.shape[0]]
    # scipy_conv =    scipy.ndimage.gaussian_filter1d(fr_arr[:100000], sigma)

    # plt.plot(scipy_conv), plt.plot(scipy_conv)

    # smooth_fr = scipy.ndimage.gaussian_filter1d(fr_arr, sigma, axis=-1, order=0, output=None,
    #                                             mode='reflect', cval=0.0, truncate=4.0)
    return myconv

def cleanupConsecutiveSpikes(spike_times_array, delta_t, verbose=0):
    # due to some bug in kilosort, sometimes spikes may be consecutive (same time)
    # even for the same unit. this causes divisions by zero and positively biased
    # estimates of firing rates. This function is for eliminating repeated spikes
    time_differences = np.diff(spike_times_array)
    repeated = time_differences < delta_t
    if np.sum(repeated)>0:
        idx_to_keep = np.where(~repeated)[0] + 1
        spike_times_clean = spike_times_array[idx_to_keep]
        if verbose:
            print('Found repeated spikes')
    else:
        spike_times_clean = spike_times_array

    return spike_times_clean

def spikeTimesBinArray(spike_times, time_bounds, fs=30000):

    time_array = np.arange(time_bounds[0], time_bounds[1], 1.0/fs)
    spike_bin_array = np.zeros(time_array.shape)

    # restrict spike idx to the interval that we're interested in
    # spikes_to_keep = spike_times>=time_bounds[0] and spike_times<=time_bounds[1]
    spikes_to_keep = spike_times[np.logical_and(spike_times >= time_bounds[0],
                                    spike_times <= time_bounds[1])]

    # this will be unefficient. I'm sure there's a faster matrix based operation
    for ii in range(spikes_to_keep.shape[0]):
        # find the closest time
        this_time_idx = np.argmin(np.abs(time_array-spikes_to_keep[ii]))
        spike_bin_array[this_time_idx] = 1

    return time_array, spike_bin_array
def averageFiringRateInInterval(spike_times, time_bounds=None):
    # estimates the average firing rate in a given interval by
    # dividing the number of spikes in that interval by the total time
    if time_bounds is None:
        time_bounds = [spike_times[0], spike_times[-1]]
    spike_times = spike_times[np.logical_and(   spike_times >= time_bounds[0],
                                                spike_times <= time_bounds[1])]
    avgFR = len(spike_times) / (time_bounds[1]-time_bounds[0])
    return avgFR
def compilePopulationActivity(spike_times, units, event_times,
                              time_bounds = [-0.1, 0.1],
                              metric='smooth_fr_estimate',
                              subselect_units=None,
                              time_step=0.001,
                              verbose=0):
    # compiles smooth firing rates / binned spike counts for all
    # specified units over the time interval or around a behavioral
    # event
    # INPUTS:
    #   spike_times [N spikes]
    #   units [size is the same as spike times, value is the unit / cluster id]
    #

    # iterate through all units:
    if subselect_units is None:
        subselect_units = np.unique(units)
    else:
        spike_indices_of_interest = CareyUtils.findAny(units, subselect_units)
        spike_times = spike_times[spike_indices_of_interest]
        units       = units[spike_indices_of_interest]

    if metric == 'smooth_fr_estimate':
        # pre-allocate data matrix of UNITS by TIME by TRIALS
        n_units         = len(subselect_units)
        # n_timepoints    = len(np.arange(1+time_bounds[0], 1+time_bounds[1], time_step))
        n_timepoints    = int((np.max(time_bounds) - np.min(time_bounds)) / time_step)
        n_trials        = len(event_times)

        population_activity = np.zeros((n_units, n_timepoints, n_trials))

        for tt in range(len(event_times)):
            if verbose:
                print("Colecting data from trial %i" % (tt+1), end=" ")
            # set the time array
            time_interval_this_trial = np.arange(   event_times[tt]+time_bounds[0],
                                                    event_times[tt]+time_bounds[1],
                                                    1/CareyConstants.DEF_NPX_FS)
            downsampled_time_this_trial = np.linspace(  event_times[tt]+time_bounds[0],
                                                        event_times[tt]+time_bounds[1],
                                                        n_timepoints)
            for ii in range(len(subselect_units)):
                if verbose:
                    print(".", end=" ")

                # for each unit, get the activity estimate in the desired time interval
                spike_times_this_unit = spike_times[units==subselect_units[ii]]
                __, fr_this_trial = estimateFiringRate(spike_times_this_unit,
                                                   time_interval_this_trial,
                                                   method='step_and_smooth')
                # downsample
                fr_this_trial_downsampled = np.interp(downsampled_time_this_trial,
                                                      time_interval_this_trial,
                                                      fr_this_trial)

                population_activity[ii,:,tt] = fr_this_trial_downsampled

            print(" ")

    elif metric=='stepwise_fr_estimate':
        # pre-allocate data matrix of UNITS by TIME by TRIALS
        n_units         = len(subselect_units)
        n_timepoints    = len(np.arange(1+time_bounds[0], 1+time_bounds[1], time_step))
        n_trials        = len(event_times)

        population_activity = np.zeros((n_units, n_timepoints, n_trials))

        for tt in range(len(event_times)):
            if verbose:
                print("Colecting data from trial %i" % (tt+1), end=" ")
            # set the time array
            time_interval_this_trial = np.arange(   event_times[tt]+time_bounds[0],
                                                    event_times[tt]+time_bounds[1],
                                                    1/CareyConstants.DEF_NPX_FS)
            downsampled_time_this_trial = np.linspace(  event_times[tt]+time_bounds[0],
                                                        event_times[tt]+time_bounds[1],
                                                        n_timepoints)
            for ii in range(len(subselect_units)):
                if verbose:
                    print(".", end=" ")

                # for each unit, get the activity estimate in the desired time interval
                spike_times_this_unit = spike_times[units==subselect_units[ii]]
                __, fr_this_trial = estimateFiringRate(spike_times_this_unit,
                                                   time_interval_this_trial,
                                                   method='step')
                # downsample
                fr_this_trial_downsampled = np.interp(downsampled_time_this_trial,
                                                      time_interval_this_trial,
                                                      fr_this_trial)

                population_activity[ii,:,tt] = fr_this_trial_downsampled

            print(" ")

    elif metric=='binning':
        # pre-allocate data matrix of UNITS by TIME by TRIALS
        n_units         = len(subselect_units)
        # n_timepoints    = len(np.arange(1+time_bounds[0], 1+time_bounds[1], time_step))
        n_bins          = int((np.max(time_bounds) - np.min(time_bounds)) / time_step)
        n_trials        = len(event_times)

        population_activity = np.zeros((n_units, n_bins, n_trials))

        for tt in tqdm(range(len(event_times))):
            if verbose:
                print("Colecting data from trial %i" % (tt+1), end=" ")
            # set the time array
            bins_this_trial = np.linspace(event_times[tt]+time_bounds[0],
                                          event_times[tt]+time_bounds[1],
                                          n_bins+1)
            for ii in range(len(subselect_units)):
                if verbose:
                    print(".", end=" ")

                # for each unit, get the activity estimate in the desired time interval
                spike_times_this_unit = spike_times[units==subselect_units[ii]]
                spike_times_this_unit_this_trial = spike_times_this_unit[
                                                        np.logical_and(spike_times_this_unit>=np.min(bins_this_trial),
                                                        spike_times_this_unit<=np.max(bins_this_trial))]
                spike_counts_this_trial,__ = np.histogram(spike_times_this_unit, bins_this_trial)

                population_activity[ii,:,tt] = spike_counts_this_trial
            if verbose:
                print(" ")

    return population_activity

def compilePopulationActivity_Event2Event(spike_times, units, event_times_2d,
                              metric='smooth_fr_estimate',
                              subselect_units=None,
                              time_step=0.001,
                              verbose=0,
                              time_align=True):
    """compiles smooth firing rates between two events (one particular instance is
    swing and stance onsets of paws)
    event_times_2d should be a 2 dimensional array with dim0 = num_trials and dim1 = 2
    """

    # iterate through all units:
    if subselect_units is None:
        subselect_units = np.unique(units)
    else:
        spike_indices_of_interest = CareyUtils.findAny(units, subselect_units)
        spike_times = spike_times[spike_indices_of_interest]
        units = units[spike_indices_of_interest]

    # pre-allocate array for storing the number of points in each trial
    timepts_per_trial = np.zeros(event_times_2d.shape[0])

    if metric == 'smooth_fr_estimate':
        # pre-allocate data matrix of UNITS by TIME by TRIALS
        n_units = len(subselect_units)
        # number of timepoints is flexible, as it is variable in each instance / trial / stride
        # n_timepoints = int((np.max(time_bounds) - np.min(time_bounds)) / time_step)
        n_trials = event_times_2d.shape[0]

        # population_activity = np.zeros((n_units, n_timepoints, n_trials))
        population_activity = [None] * n_trials

        for tt in range(n_trials):
            if verbose:
                print("Colecting data from trial %i" % (tt + 1), end=" ")
            # set the time array

            # make a time array that's specific to this interval
            n_timepoints = int((event_times_2d[tt,1] - event_times_2d[tt,0]) / time_step)

            time_interval_this_trial = np.arange(event_times_2d[tt, 0],
                                                 event_times_2d[tt, 1],
                                                 1 / CareyConstants.DEF_NPX_FS)
            downsampled_time_this_trial = np.linspace(event_times_2d[tt, 0],
                                                      event_times_2d[tt, 1],
                                                      n_timepoints)

            all_units_this_trial = np.zeros((downsampled_time_this_trial.shape[0], n_units))

            timepts_per_trial[tt] = downsampled_time_this_trial.shape[0]

            for ii in range(n_units):
                if verbose:
                    print(".", end=" ")

                # for each unit, get the activity estimate in the desired time interval
                spike_times_this_unit = spike_times[units == subselect_units[ii]]
                __, fr_this_trial = estimateFiringRate(spike_times_this_unit,
                                                       time_interval_this_trial,
                                                       method='step_and_smooth')
                # downsample
                fr_this_trial_downsampled = np.interp(downsampled_time_this_trial,
                                                      time_interval_this_trial,
                                                      fr_this_trial)

                # population_activity[ii, :, tt] = fr_this_trial_downsampled
                all_units_this_trial[:, ii] = fr_this_trial_downsampled

            population_activity[tt] = all_units_this_trial

            print(" ")

    elif metric == 'smooth_fr_estimate':
        raise ValueError('Not implemented yet')

    # standardize time (interpolate) which leads to a phase alignment
    if isinstance(time_align, bool) and time_align is True:
        print('Aligning trials to be the same length in time')
        # get the median of all downsampled time trials. this will be the standard value
        final_timepts = int(np.median(timepts_per_trial))
    elif isinstance(time_align, int):
        final_timepts = time_align

        population_activity_3d = np.zeros((n_units, final_timepts, n_trials))

        for tt in range(n_trials):
            xp = np.linspace(0, 1, num=population_activity[tt].shape[0], endpoint=True)
            x  = np.linspace(0, 1, num=final_timepts, endpoint=True)
            for ii in range(n_units):
                population_activity_3d[ii,:,tt] = np.interp(x, xp, population_activity[tt][:,ii])

        return population_activity_3d
    else:
        return population_activity

def compilePopulationActivity_Parallel(spike_times, units, event_times,
                              time_bounds = [-0.1, 0.1],
                              metric='smooth_fr_estimate',
                              subselect_units=None,
                              time_step=0.001,
                              verbose=0):
    # compiles smooth firing rates / binned spike counts for all
    # specified units over the time interval or around a behavioral
    # event
    # INPUTS:
    #   spike_times [N spikes]
    #   units [size is the same as spike times, value is the unit / cluster id]
    #
    def compile_this_trial(tt, PAlist):
        if verbose:
            print("Colecting data from trial %i" % (tt + 1), end=" ")
        # set the time array
        time_interval_this_trial = np.arange(event_times[tt] + time_bounds[0],
                                             event_times[tt] + time_bounds[1],
                                             1 / CareyConstants.DEF_NPX_FS)
        downsampled_time_this_trial = np.linspace(event_times[tt] + time_bounds[0],
                                                  event_times[tt] + time_bounds[1],
                                                  n_timepoints)

        FR_this_trial_all_units = np.zeros((n_units, n_timepoints))

        for ii in range(len(subselect_units)):
            if verbose:
                print(".", end=" ")

            # for each unit, get the activity estimate in the desired time interval
            spike_times_this_unit = spike_times[units == subselect_units[ii]]
            __, fr_this_trial = estimateFiringRate(spike_times_this_unit,
                                                   time_interval_this_trial,
                                                   method='step_and_smooth')
            # downsample
            fr_this_trial_downsampled = np.interp(downsampled_time_this_trial,
                                                  time_interval_this_trial,
                                                  fr_this_trial)
            FR_this_trial_all_units[ii,:] = fr_this_trial_downsampled
            # population_activity[ii, :, tt] = fr_this_trial_downsampled
        PAlist[tt] = FR_this_trial_all_units

        return FR_this_trial_all_units
    def bin_this_trial(tt, PAlist, n_bins):
        bins_this_trial = np.linspace(event_times[tt] + time_bounds[0],
                                      event_times[tt] + time_bounds[1],
                                      n_bins + 1)
        bins_this_trial_all_units = np.zeros((n_units, n_bins))


        for ii in range(len(subselect_units)):
            # for each unit, get the activity estimate in the desired time interval
            spike_times_this_unit = spike_times[units == subselect_units[ii]]
            spike_times_this_unit_this_trial = spike_times_this_unit[
                np.logical_and(spike_times_this_unit >= np.min(bins_this_trial),
                               spike_times_this_unit <= np.max(bins_this_trial))]
            spike_counts_this_trial, __ = np.histogram(spike_times_this_unit, bins_this_trial)

            bins_this_trial_all_units[ii,:] = spike_counts_this_trial

        PAlist[tt] = bins_this_trial_all_units
        return bins_this_trial_all_units
    # ..........................................................................
    # iterate through all units:
    if subselect_units is None:
        subselect_units = np.unique(units)
    else:
        spike_indices_of_interest = CareyUtils.findAny(units, subselect_units)
        spike_times = spike_times[spike_indices_of_interest]
        units       = units[spike_indices_of_interest]

    if metric == 'smooth_fr_estimate':
        # pre-allocate data matrix of UNITS by TIME by TRIALS
        n_units         = len(subselect_units)
        # n_timepoints    = len(np.arange(1+time_bounds[0], 1+time_bounds[1], time_step))
        n_timepoints    = int((np.max(time_bounds) - np.min(time_bounds)) / time_step)
        n_trials        = len(event_times)

        population_activity = np.zeros((n_units, n_timepoints, n_trials))
        population_activity.flags.writeable = True

        PA = [None] * n_trials

        PA = Parallel(n_jobs=8, verbose=100)(
            delayed(compile_this_trial)(tt, PA) for tt in range(len(event_times)))

        print(" ")

        # convert PA from parallel computation to 3D matrix

        for tt in range(n_trials):
            population_activity[:,:,tt] = PA[tt]

        return population_activity
    if metric == 'binning':
        # pre-allocate data matrix of UNITS by TIME by TRIALS
        n_units         = len(subselect_units)
        # n_timepoints    = len(np.arange(1+time_bounds[0], 1+time_bounds[1], time_step))
        n_bins          = int((np.max(time_bounds) - np.min(time_bounds)) / time_step)
        n_trials        = len(event_times)

        population_activity = np.zeros((n_units, n_bins, n_trials))
        population_activity.flags.writeable = True

        PA = [None] * n_trials

        PA = Parallel(n_jobs=8, verbose=0)(
            delayed(bin_this_trial)(tt, PA, n_bins) for tt in tqdm(range(len(event_times))))

        # convert PA from parallel computation to 3D matrix

        for tt in range(n_trials):
            population_activity[:, :, tt] = PA[tt]

    return population_activity
# todo compile from smoothed population activity array or write a one liner on it

def standardizeByAxis(in_matrix, ax=0, method='std'):
    # standardizes values between 0 and 1 (minmax or std) along a preferred axis
    # de-mean
    m = np.mean(in_matrix, axis=ax)
    if method == 'std':
        sd = np.std(in_matrix, axis=ax)
        out_matrix = (in_matrix-m)/sd
    elif method == 'minmax':
        pass

    return out_matrix

def plotAverageOverAxis(population_activity, line_ax=0, average_ax=2,
                        x_axis='default',
                        metric='avg'):
    # if population_activity is a 3d matrix of units by time by trials
    # then the line_ax should be the axis for which a line will be plotted
    # (one per unit)
    # the average_ax is the axis along which the mean is computed (trials, in this case)
    # compute the mean for each unit

    # for ii in range(population_activity.shape[line_ax]):
    #     this_avg = np.mean(population_activity)
    if metric=='mean':
        mean_over_average_ax = np.mean(population_activity, axis=average_ax)
    elif metric=='median':
        mean_over_average_ax = np.median(population_activity, axis=average_ax)
    std_over_average_ax  = np.std(population_activity, axis=average_ax) # for the error bars

    # fig, axes = plt.subplots(14,4)
    plt.figure()
    axes = plt.gca()
    if line_ax==0:

        if isinstance(x_axis, str) and x_axis=='default':
            x = np.arange(population_activity.shape[1])
        else:
            x = x_axis

        for ii in range(population_activity.shape[line_ax]):

            plt.sca(axes) # plt.sca(axes[ii])
            sns.lineplot(ax=axes, x=x, y=mean_over_average_ax[ii,:])
            # plt.plot(mean_over_average_ax[ii,:])
            plt.draw()

def populationActivity_3D_to_2D(population_activity):
    # to be sure that the reshape is done correctly
    # 3d matrix is assumed to be UNITS by TIME by TRIALS
    # end shape will be TIME x TRIALS by UNITS

    n_units, n_timepoints, n_trials = population_activity.shape

    pa2d = np.zeros((n_timepoints*n_trials, n_units))

    for tt in range(n_trials):
        this_trial_start = tt*n_timepoints
        this_trial_end   = n_timepoints*(tt+1)
        pa2d[this_trial_start:this_trial_end, :] = population_activity[:,:,tt].transpose()

    return pa2d

def populationActivity_3D_to_DataFrame(population_activity, units, time_array):

    n_units, n_timepoints, n_trials = population_activity.shape

    column_names = generateUnitNames(units)
    column_names.append('trial')
    column_names.append('time')

    pop_act = pd.DataFrame(columns=column_names)

    for ii in range(n_trials):
        tmp_mat = np.concatenate((population_activity[:,:,ii].transpose().squeeze(), ii*np.ones((n_timepoints,1))),1)
        tmp_mat = np.hstack((tmp_mat, time_array[:,None]))
        tmp_df = pd.DataFrame(tmp_mat, columns=pop_act.columns)
        pop_act = pop_act.append(tmp_df, ignore_index=True)


    return pop_act



def generateUnitNames(units_array):
    unit_names = []
    for ii in range(len(units_array)):
        unit_names.append(str('unit %i' % (units_array[ii])))

    return unit_names
def generatePCNames(n_pcs, prefix='PC'):
    pc_names = []
    for ii in range(n_pcs):
        pc_names.append(str('%s %i' % (prefix, ii+1)))
    return pc_names

def PCA_matrix_to_DataFrame(pcamat, time_array):
    total_obs, n_dimensions = pcamat.shape
    n_timepoints = time_array.shape[0]
    n_trials = total_obs / n_timepoints

    if n_trials.is_integer():
        n_trials = int(n_trials)
    else:
        raise Exception('Number of trials doesnt match up')

    column_names = generatePCNames(n_dimensions)
    column_names.append('trial')
    column_names.append('time')

    df_pca = pd.DataFrame(columns=column_names)

    for ii in range(n_trials):
        trial_start = ii*time_array.shape[0]
        trial_end   = (ii+1)*time_array.shape[0]
        tmp_mat = np.concatenate((pcamat[trial_start:trial_end,:], ii*np.ones((n_timepoints,1))),1)
        tmp_mat = np.hstack((tmp_mat, time_array[:,None]))
        tmp_df = pd.DataFrame(tmp_mat, columns=df_pca.columns)
        df_pca = df_pca.append(tmp_df, ignore_index=True)

    return df_pca


    def scatter_3d(df_pca, hue=None, s=1, palette='bwr',
                             components=['PC 1', 'PC 2', 'PC 3']):
        sns.set()
        sns.set_style("whitegrid", {'axes.grid': False})

        plt.figure()
        plt.scatter
        x = df_pca[components[0]]
        y = df_pca[components[1]]
        z = df_pca[components[2]]
        sns.scatterplot(x, y, z)

def plotUnitResponseFR(dataframe_fr, unit_names, x='time', ci='sd',
                       color=sns.color_palette()[7],
                       showplot=0, saveplot='/home/diogo/careylab/figures',
                       title_str=''):
    # assumes data is organized in a dataframe
    current_backend = matplotlib.get_backend()

    if not showplot:
        # matplotlib.use('Agg')
        pass

    for ii in range(len(unit_names)):
        this_fig = plt.figure()
        this_title = str('%s %s' % (unit_names[ii], title_str))
        this_unit = unit_names[ii]
        sns.lineplot(data=dataframe_fr, y=this_unit, x='time', ci='sd',
                     color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Estim. of FR, err shade=stand. dev.')
        plt.title(this_title)

        if saveplot is not '':
            plt.savefig(os.path.join(saveplot, (this_title + '.png')),
                    dpi=300,
                    transparent=True,
                    format='png')
            plt.close(this_fig)

    # matplotlib.real_use(current_backend)

def lineplot(y):
    # making a wrapper function because seaborn is so picky
    plt.figure()
    sns.lineplot(x=np.arange(len(y)), y=y)

def trialwiseScatterPlot_3D(df, x='PC 1', y='PC 2', z='PC 3', hue='', linewidth=1, palette='YlOrRd'):
    # plots lines representing trials
    sns.lineplot(df, x='time', y='PC 1', linewidth=0.1, hue='trial', palette='YlOrRd')


def trialwise_FeatOverTime():
    pass

def compileSessionWise_FiringRates(ksDir, dt, units='good', method='isi_step', gaussdev=0.001):
    '''

    df = CareyEphys.compileSessionWise_FiringRates(ksDir, dt, units='good', method='gaussian_smoothing', gaussdev=0.001)

    Parameters
    ----------
    ksDir:      dirctory for the sorted and curated neuropixels data
    units:      the units of interest
    dt:         the timestep for the output (not related directly to the sampling rate)
    method:     ['isi_step', 'gaussian_smoothing']
    gaussdev:   width of the gaussian kernel (in seconds)

    Returns
    -------
    df:         dataframe containing columns for time and unit firing
    '''
    # ksDir = r'Z:\data\2022\BATCH5\recordings\VIV_23058\VIV_23058_S10_g1\VIV_23058_S10_g1_imec0_ks25'
    # units = info.cluster_id[info.group=='good'].values

    spiketimes, clusters, templates, kslabels, info = loadKsDir(ksDir, fs=CareyConstants.DEF_NPX_FS)
    metadata = loadSpikeGLXMetada(ksDir)

    # create the time array
    t_arr = np.arange(0, metadata['fileTimeSecs'], 1/metadata['imSampRate'], dtype=float)
    acquisition_time = metadata['fileTimeSecs']
    output_t_arr = np.arange(0, metadata['fileTimeSecs'], dt, dtype=float)
    if units == 'good':
        units = info.cluster_id[info.group=='good'].values

    df = pd.DataFrame(columns=['time'])
    df['time'] = output_t_arr

    for ii in tqdm(range(units.shape[0])):
        # get the spike times of the first unit
        these_spiketimes = spiketimes[clusters==units[ii]]

        if method == 'isi_step':
            this_fr_full = estimate_isi_based_FR(these_spiketimes, acquisition_time, fs=metadata['imSampRate'])
            pass
        elif method == 'gaussian_smoothing':
            this_fr_full = estimate_gaussconv_based_FR(these_spiketimes, acquisition_time, fs=CareyConstants.DEF_NPX_FS,
                                                       gaussdev=gaussdev)

        # interpolate to a lower frequency
        this_fr = np.interp(output_t_arr, t_arr, this_fr_full, left=None, right=None, period=None)

        df[('unit' + str(units[ii]))] = this_fr

    return df

def compileSessionWise_SpikeCounts(ksDir,units='good', binsize=0.02):
    '''

    bin_edges, df = CareyEphys.compileSessionWise_SpikeCounts(ksDir,units='good', binsize=0.02)

    Parameters
    ----------
    ksDir:      dirctory for the sorted and curated neuropixels data
    units:      the units of interest [optional]
    binsize:    bin width in seconds

    Returns
    -------
    bin_edges:  bin edges of dimension n_bins+1 (one more timepoint than the units)
    df:         dataframe containing columns for time and unit firing
    '''

    spiketimes, clusters, templates, kslabels, info = loadKsDir(ksDir, fs=CareyConstants.DEF_NPX_FS)
    metadata = loadSpikeGLXMetada(ksDir)

    # create the time array
    bin_edges = np.arange(0, metadata['fileTimeSecs'], binsize, dtype=float)
    acquisition_time = metadata['fileTimeSecs']
    if units == 'good':
        units = info.cluster_id[info.group == 'good'].values

    df = pd.DataFrame()

    for ii in tqdm(range(units.shape[0])):
        # get the spike times of the first unit
        these_spiketimes = spiketimes[clusters==units[ii]]
        these_counts = np.histogram(these_spiketimes, bin_edges)
        df[('unit' + str(units[ii]))] = these_counts[0]

    return bin_edges, df

def get_sessionwise_firingrate_singleunit(spike_times, time_array):
    # t_arr = np.arange(0, TOTAL_TIME, 1 / 1000, dtype=float)
    # padded_spike_train = np.zeros(time_array.shape[0])
    cupy_spike_train = cupy.zeros(time_array.shape[0])
    cupy_time_array = cupy.arange(0, TOTAL_TIME, 1 / 1000)
    cupy_spike_times = cupy.array(spike_times)
    for spike in tqdm(cupy_spike_times):
        cupy_spike_train[cupy.argmin(cupy.abs( spike-cupy_time_array ))] = 1

def get_and_save_downsampled_population_firing_rates(dp, downsamp_freq=1000, save_file_loc=None, good_units=None):
    if good_units is None:
        good_units = npyx.gl.get_good_units(dp)

    t_arr = np.arange(0, TOTAL_TIME, 1 / downsamp_freq, dtype=float)
    cupy_t_arr = cupy.array(t_arr)
    cupy_t_full = cupy.arange(0, TOTAL_TIME, 1 / SAMP_RATE, dtype=float)
    df = pd.DataFrame(data=t_arr, columns=['time'])

    for neuron in tqdm(good_units):
        spikes = npyx.spk_t.trn(dp, neuron) / SAMP_RATE
        fr = CareyEphys.estimate_gaussconv_based_FR(spikes, TOTAL_TIME, fs=SAMP_RATE, gaussdev=0.020)
        fr_downsampled = cupy.interp(cupy_t_arr, cupy_t_full, cupy.array(fr)).get()
        df[str(neuron)] = fr_downsampled
    if save_file_loc is not None:
        df.to_csv(os.path.join(new_proc, 'sessionwise_firing_rates.csv'))

