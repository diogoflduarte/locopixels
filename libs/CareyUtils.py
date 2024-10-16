import CareyUtils
import numpy as np
from numpy import matlib
import os
from scipy import signal
from scipy import stats
from matplotlib import pyplot as plt
import re
import scipy.signal
import scipy.stats
import datetime
from sklearn.decomposition import PCA
from numba import jit
import math
import pandas as pd
from tqdm import tqdm
import cProfile
import pstats
from io import StringIO
from pykalman import KalmanFilter
try:
    import cupy
    import cupyx.scipy.ndimage
    gpu_avail = True
except ImportError:
    print('Cupy not installed, not available for gaussian smoothing firing rates in CareyUtils')
    gpu_avail = False

def besselFilt(in_sig, sampling_freq, cuttoff_freq, order=2):

    nyq_freq = sampling_freq/2
    input_freq = cuttoff_freq / nyq_freq

    b, a = signal.bessel(order, input_freq, btype='bandpass', analog=False,
                            norm='phase')
    out_sig = signal.filtfilt(b, a, in_sig)

    return out_sig

def getFiles(in_dir, pattern):
    """gets files from first argument matching the pattern in the second
    argument. if the first argument is a list, then it should be a list of files"""
    if isinstance(in_dir, list):
        file_list = in_dir
    else:
        file_list = os.listdir(in_dir)

    out_files = []

    for ff in file_list:
        if ff.find(pattern)>1:
            out_files.append(ff)

    return out_files

def filterFilesFromList(file_list, pattern):
    """
    this way it can be used ietratively
    :param file_list:
    :param pattern:
    :return:
    """
    out_files = []

    for ff in file_list:
        if ff.find(pattern) > 1:
            out_files.append(ff)

    return out_files

def highlightSpikes(t, ephys_traces, spike_times=None, plot_range=None):
    # function to plot traces in light gray and highlight spikes in some
    # cool looking color, unlike all of that locomouse paw nonsense
    # based on np_ephys_traces(time_vec, m_in, varargin)

    # this function should be capable of highlighting spike times on multiple
    # traces
    # if there's more than one trace, 2nd dimension (dim 1) should be the number
    # of traces
    # plt.rcParams['figure.figsize'] = [16, 6]
    SPACING = 10; # relative spacing between plots
    SPIKE_WIN = 0.01


    if type(plot_range) is list:
        plot_range = np.asarray(plot_range)
        if plot_range.size!=2:
            print('Wrong time plot range')
            return 0
    elif type(plot_range)==np.ndarray:
        pass
    else:
        plot_range = np.array((0,np.max(t)))
    # make sure the limits are the same data type as time
    plot_range = plot_range.astype(type(t))

    ephys_traces = np.squeeze(ephys_traces)

    # how many traces are there
    if ephys_traces.ndim > 2:
        # not suppported
        print('wrong trace dimension')
        return 0
    else:
        n_timepts = ephys_traces.shape[0]
        if ephys_traces.ndim == 1:
            # only one trace
            num_traces = 1
            ephys_traces = np.expand_dims( ephys_traces, axis=1 )
        elif ephys_traces.ndim == 2:
            num_traces = ephys_traces.shape[1]

    # compute the offsets for several plots
    # ephys_traces = scipy.stats.zscore(ephys_traces, axis=0)
    # offsets = np.arange(1, num_traces*SPACING, SPACING)
    # ephys_traces = ephys_traces + np.matlib.repmat(offsets, ephys_traces.shape[0], 1)

    fig, axs = plt.subplots(num_traces, 1, squeeze=False)

    for ll in range(num_traces):
        #plt.sca(axs[ll,0])
        axs[ll,0].plot(t, ephys_traces[:,ll], color=[0.6, 0.6, 0.6])
        plt.xlim([plot_range[0], plot_range[1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Channel readout (a.u.)')
        # plt.setp(axs[ll,0], title=("unit " + str(cs_units[ii]['id']) +\
        #  " read on channel " + str(cs_units[ii]['ch'])));

        if type(spike_times)==np.ndarray:
            # if there are spike times, filter them so that only the ones
            # within the time limits are used
            keep_idx_bin = (spike_times>plot_range[0]+SPIKE_WIN) &\
                           (spike_times<plot_range[1]-SPIKE_WIN)
            spike_times = spike_times[keep_idx_bin==True]
            num_spikes = spike_times.shape[0]

        for s in range(num_spikes):
            # get the start and end indices for this spike highlight
            idx_start = np.argmin(np.abs(t-(spike_times[s]-SPIKE_WIN)))
            idx_end   = np.argmin(np.abs(t-(spike_times[s]+SPIKE_WIN)))

            #plt.sca(axs[ll,0])
            axs[ll,0].plot(t[idx_start:idx_end], ephys_traces[idx_start:idx_end,ll],
                        color=[0.75,0,0]);

    return axs

def getFileFromFolder(folder, pattern):

    files = os.listdir(folder)
    outfile = []
    for f in files:
        if re.search(pattern, f):
            outfile = f
            break
    return outfile

def getRisingLoweringEdges(in_sig, normalization='range'):
    """
    :param in_sig:
    :param normalization:
    :return: rising_edges, lowering_edges
    """
    if normalization=='range':
        in_sig = in_sig-np.min(in_sig)
        if np.max(in_sig)==np.min(in_sig):
            rising_edges = None
            lowering_edges = None
            return rising_edges, lowering_edges
        else:
            in_sig = in_sig/np.max(in_sig)

    # change datatype to int8 because it supports negative values
    in_sig = in_sig.astype('int8')

    # compute the diff
    diff_sig = np.diff(in_sig)
    # to make sure they like up, pad a zero at the beggining (because diff
    # removes one value from the array)
    diff_sig = np.concatenate((np.zeros(1).astype(in_sig.dtype), diff_sig))

    rising_edges    = np.squeeze(np.where(diff_sig>0))
    lowering_edges  = np.squeeze(np.where(diff_sig<0))

    return rising_edges, lowering_edges

def medfilt(in_sig):
    scipy.signal.medfilt(in_sig, kernel_size=3)

def getBonsaiDateTime(filename):
    filename = os.path.splitext(filename)[0]
    year    = int(filename[-19:-15])
    month   = int(filename[-14:-12])
    day     = int(filename[-11:-9])
    hour    = int(filename[-8:-6])
    minutes = int(filename[-5:-3])
    seconds = int(filename[-2:])

    creation_time = datetime.datetime(year, month, day, hour, minutes, seconds)

    return creation_time

def findAny(in_array, subselect_vals, method='iter'):
    if in_array.ndim > 1:
        raise Exception('array should be one dimensional')

    if method=='iter':
        idx = np.zeros(in_array.shape, dtype=bool)

        for ii in range(len(subselect_vals)):
            tmp_idx = (in_array==subselect_vals[ii])
            idx = np.logical_or(idx, tmp_idx)

    return idx

def convert_to_time(in_str, formatString='%Y-%m-%dT%H_%M_%S', idx=slice(-23,-4)):
    # idx = slice(-23,-4)
    # datetime.datetime.strptime('2021-09-02', '%Y-%m-%d')
    # datetime.datetime.strptime('16_54_22', '%H_%M_%S')
    # datetime.datetime.strptime('2021-09-02T16_54_22', '%Y-%m-%dT%H_%M_%S')

    dt = datetime.datetime.strptime(in_str[idx], formatString)
    return dt

def findMatchingFile_bonsai(fullpath_filename, folder='same', targetext='.avi',
                            formatString='%Y-%m-%dT%H_%M_%S', timestring_indices=slice(-23,-4),
                            tol=5):
    """finds the matching file in a folder created by bonsai outputs. In this case,
    bonsai creates a folder with videos and matching CSV files, but they don't always
    have the same time because one of the files might onluy close one or two seconds after.
    this function compares times from bonsai files, converts them to time and matches
    based on some tolerance 'tol' in seconds
    fullpath_filename should have the full path
    timestring_indices is time string indices array in the filename"""

    matchingFile = []

    if folder == 'same':
        folder = os.path.split(fullpath_filename)[0]
    filename = os.path.split(fullpath_filename)[1]

    if formatString == 'trial':
        trial_number = int(filename.split('_')[-1][0:-4])
        candidate_filelist = getFiles(folder, targetext)

        for ff in candidate_filelist:
            if int(ff.split('_')[-1][0:-4]) == trial_number:
                matchingFile = ff
                break


    else:
        this_dtime = convert_to_time(filename, formatString=formatString, idx=timestring_indices)

        candidate_filelist = getFiles(folder, targetext)

        # try to get a direct match first
        dateTimeString = filename[timestring_indices]
        single_candidate = getFiles(candidate_filelist, dateTimeString)
        if len(single_candidate)==1:
            matchingFile = os.path.join(folder, single_candidate[0])

        else:
            # list all files in folder and select those with an extension
            datetime_of_candidates = [None] * len(candidate_filelist)

            # pull datetime obj out of those files
            for fileidx, fil in enumerate(candidate_filelist):
                datetime_of_candidates[fileidx] = convert_to_time(fil, formatString=formatString, idx=timestring_indices)
                # if one of these files is within tol of the original file,
                # select and break the loop
                if np.abs(datetime_of_candidates[fileidx] - this_dtime) < datetime.timedelta(seconds=tol):
                    matchingFile = candidate_filelist[fileidx]
                    break

    return matchingFile



    # get a list or array of datetime objects for each of those files

    pass

def validateDimensions(in_var, ndim):
    if in_var.ndim is not ndim:
        raise ValueError('VALIDATION: Wrong number of dimensions')

def displayColor(in_color_array):
    plt.figure()
    circle1 = plt.Circle((0.5, 0.5), 0.2, color=in_color_array)
    ax = plt.gca()
    ax.add_patch(circle1)
    plt.axis('off')

def zscore(in_matrix, ax=0, method='std'):
    # standardizes values between 0 and 1 (minmax or std) along a preferred axis
    # de-mean
    m = np.mean(in_matrix, axis=ax)
    if method=='std':
        sd = np.std(in_matrix, axis=ax)
        out_matrix = (in_matrix-m)/sd
    elif method=='minmax':
        pass

    return out_matrix

def dimensionality_estimation_pca(Y, fraction_holdout=0.1, n_folds=10, n_recon_iter=100):
    """implementation of the dimensionality estimation by cross validation in PCA
    proposed in Alex William's website: http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
    Y is the data matrix [obs x feat]"""

    n_obs, n_feat = Y.shape
    total_pts = n_obs * n_feat

    # let's start by making a masking matrix M knowing the fraction of holdout
    number_of_holdout_pts = int(fraction_holdout * total_pts)
    holdout_pts = np.random.choice(total_pts, size=(number_of_holdout_pts,), replace=False)
    holdout_coordinates = np.unravel_index(holdout_pts, shape=Y.shape) # this is a tuple

    Y_standardized = scipy.stats.zscore(Y, axis=0) # now we have each column to be zero mean and 1 standard deviation.
    # this should help when assigning random values

    '''my personal twist is that I'll run steps i) and ii) iteratively by retaining 1 to n_feat components in step 2
    Here's the rationale: the new projected values and axis (U and V) will b estimated from Z, which will be 
    approximately the same as Y. If the information about the true held out values is contained in other datapoints (the
    assumption of dimensionality reduction), then by step ii), which is the reconstruction, the heldout value prediction
    should be close to the real value. There might be an optimal number of components for which this holds true, and
    under that number the prediction will be worse. Over it the remaining components are just adding noise
    '''

    reconstruction_error = np.zeros((n_folds, n_feat))
    for f in range(n_folds):
        # now let's build a matrix Z which is equal to Y_standardized everywhere except in holdout_coordinates
        # this is step 0
        Z = Y_standardized

        # Z is equal to Y except in holdout_coordinates. For a masking matrix M of size of Y, M is 0 in
        # holdout_coordinates and 1 everywhere else
        Z[holdout_coordinates] = np.random.rand(len(holdout_coordinates[0]))

        for c in range(n_feat):
            # Now we estimate the axes and projections via PCA on Z.
            # this is step i)

            Z_hat = Z
            # for ii in range(n_recon_iter):

            pca_model = PCA(n_components=c).fit(Z_hat)
            low_d_projection = pca_model.transform(Z_hat)

            # and we reconstruct the data constraining the masked values M to be the same
            Z_hat = pca_model.inverse_transform(low_d_projection)
            # Z_temp = Z_hat
            # Z_hat = Z
            # Z_hat[holdout_coordinates] = Z_temp[holdout_coordinates]

            # and we take the sum of squared errors only for the held out points
            err = np.sum((Y_standardized[holdout_coordinates] - Z_hat[holdout_coordinates])**2)

            reconstruction_error[f, c] = err

    return reconstruction_error

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

def gaussian(mu, sigma, pts):
    '''

    Parameters
    ----------
    mu:     mean
    sigma:  standard deviation
    pts:    number of points
    Returns

    -------

    '''
    pts = int(pts)
    x = np.linspace(-pts/2, pts/2, pts)

    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def gaussian_smoothing(sig, sigma, usegpu=False, asgpu=False):
    '''

    Parameters
    ----------
    sig
    sigma

    Returns
    -------
    smoothed_signal
    '''
    kernel = CareyUtils.gaussian(0, sigma, np.round(sigma * 10))
    kernel = kernel / np.sum(kernel)

    if gpu_avail is False:
        usegpu = False

    if usegpu is False:
        smoothed_signal = fft_convolution(sig, kernel)
    else:
        if type(sig) is np.ndarray:
            sig = cupy.array(sig)
        smoothed_signal = cupyx.scipy.ndimage.gaussian_filter1d(sig, sigma)
        if not asgpu:
            smoothed_signal = smoothed_signal.get()
    return smoothed_signal

@jit(nopython=True)
def gaussian_kernel_convolution(sig, kernel):
    in_sig = np.concatenate((np.flip(sig[:kernel.shape[0]]), sig, np.flip(sig[-kernel.shape[0]:])), axis=0)
    myconv = np.convolve(in_sig, kernel)
    myconv = myconv[int(np.round((kernel.shape[0] * 1.5))):]
    myconv = myconv[:sig.shape[0]]

    return myconv

def fft_convolution(sig, kernel):
    in_sig = np.concatenate((np.flip(sig[:kernel.shape[0]]), sig, np.flip(sig[-kernel.shape[0]:])), axis=0)
    myconv = scipy.signal.fftconvolve(in_sig, kernel, 'full')

    n_sig =     sig.shape[0]
    n_result =  myconv.shape[0]
    n_kernel =  kernel.shape[0]

    kp = int(np.round(n_kernel*1.5))

    myconv = myconv[kp:]
    myconv = myconv[:sig.shape[0]]

    return myconv

def stripExt(inp):
    return os.path.splitext(inp)[0]

def convert_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def get_stride_indices_from_phase(signal, threshold=0.001, min_stride_duration=50, max_stride_duration=1000, verbose=False):
    '''

    :param signal:
    :param threshold:
    :param min_stride_duration:
    :param max_stride_duration:
    :return:
    '''
    # # Create an array to hold the stride indices
    # stride_indices = np.zeros_like(signal, dtype=int)
    # # Compute the difference between consecutive elements
    # diff_signal = np.diff(signal)
    # # Find indices where the signal resets (negative difference)
    # reset_indices = np.where(diff_signal < -threshold)[0]
    # # Initialize stride index
    # stride_count = 1
    # start_idx = 0
    # for reset_idx in reset_indices:
    #     if (reset_idx-start_idx) < min_stride_duration or (reset_idx-start_idx) > max_stride_duration:
    #         continue
    #     else:
    #         stride_indices[start_idx:reset_idx + 1] = stride_count
    #         stride_count += 1
    #         start_idx = reset_idx + 1
    # # Assign the last stride index to the remaining part of the signal
    # stride_indices[start_idx:] = stride_count
    #
    # return stride_indices

    maxpeaks = scipy.signal.find_peaks(signal)[0]
    minpeaks = scipy.signal.find_peaks(-signal)[0]
    # stride_indices = np.zeros_like(signal, dtype=int)
    # for idx in tqdm(minpeaks, disable= not verbose):
    #     stride_indices[idx:] += 1
    # for idx in tqdm(np.arange(1, minpeaks.shape[0] + 1), disable= not verbose):
    #     max_idx = signal[stride_indices == idx - 1].argmax()
    #     next_stride = idx
    #     stride_indices[max_idx:next_stride] = 0
    # return stride_indices
    if minpeaks[0] > maxpeaks[0]:
        maxpeaks = maxpeaks[1:]
    if minpeaks[-1] > maxpeaks[-1]:
        minpeaks = minpeaks[:-1]
    stride_indices = np.zeros_like(signal, dtype=int)
    for ii in tqdm(range(minpeaks.shape[0]), disable = not verbose):
        stride_indices[minpeaks[ii]:maxpeaks[ii]] = ii + 1
    return  stride_indices

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Print the profiling results
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    print(stream.getvalue())

    return result


def find_segments_below_threshold(signal, threshold, min_block_length):
    # Step 1: Create a mask for points below threshold
    below_threshold_mask = signal < threshold

    # Step 2: Find boundaries between consecutive points in mask
    diff_mask = np.diff(below_threshold_mask.astype(int))

    # Ensure block_starts and block_ends include the start and end of the array if necessary
    block_starts = np.where(diff_mask == 1)[0] + 1
    block_ends = np.where(diff_mask == -1)[0] + 1

    # Handle cases where the signal starts or ends with a below-threshold block
    if below_threshold_mask[0]:
        block_starts = np.insert(block_starts, 0, 0)
    if below_threshold_mask[-1]:
        block_ends = np.append(block_ends, len(signal))

    # Step 3: Filter blocks by minimum length
    block_lengths = block_ends - block_starts
    valid_blocks_mask = block_lengths >= min_block_length

    # Check if there are no valid blocks
    if np.sum(valid_blocks_mask) == 0:
        return [], np.zeros_like(signal, dtype=bool)

    # Step 4: Extract segments of interest
    valid_block_starts = block_starts[valid_blocks_mask]
    valid_block_ends = block_ends[valid_blocks_mask]

    # Initialize mask with False (not of interest)
    mask = np.zeros_like(signal, dtype=bool)

    for start, end in zip(valid_block_starts, valid_block_ends):
        mask[start:end] = True

    # segments = list(zip(valid_block_starts, valid_block_ends))
    segments = [slice(start, end) for start, end in zip(valid_block_starts, valid_block_ends)]

    return segments, mask

def boolean_interp(x, xp, fp):
    """
    Interpolates boolean values from `fp` onto `x` based on closest `xp`.

    Parameters:
    - x : array_like
        Array representing the target boolean indices.
    - f : array_like
        Array representing the boolean values to be interpolated.
    - xp : array_like
        Array representing the original boolean indices.
    - fp : array_like
        Array representing the boolean values at `xp`.

    Returns:
    - array_like
        Interpolated boolean values for `x`.
    """
    # Initialize result array for x
    f = np.full_like(x, False)

    # Find indices where fp (time1) is True
    true_indices_fp = np.where(fp)[0]

    # Map True values from fp to closest xp in x
    for idx in true_indices_fp:
        xp_value = xp[idx]
        closest_idx = np.argmin(np.abs(x - xp_value))
        f[closest_idx] = True

    return f

def find_string_in_list(my_list, text):
    elements_containing_apple = [(element, idx) for idx, element in enumerate(my_list) if
                                 isinstance(element, str) and text in element]
    matching_elements = [element for element, idx in elements_containing_apple]
    matching_indices = [idx for element, idx in elements_containing_apple]
    return matching_elements, matching_indices


def kalman_smooth(signal, dt=1/432, tCov=0.01, obsCov=1.0):
    """
    Apply Kalman smoothing to a single coordinate signal with occlusions and artifacts.

    Parameters:
    - coordinate_signal: A (T,) array where T is the number of time points for a single coordinate (e.g., x, y, or z).
    - dt: Time step between observations. Default is 1/432 for 432Hz sampling rate.

    Returns:
    - smoothed_coordinate: A (T,) array of the smoothed signal for the single coordinate.
    """
    # Define the state transition matrix for constant velocity model
    transition_matrix = [[1, dt], [0, 1]]
    # Define the observation matrix
    observation_matrix = [[1, 0]]

    # Define the transition covariance (process noise)
    transition_covariance = tCov * np.eye(2)
    # Define the observation covariance (measurement noise)
    observation_covariance = obsCov

    # Initial state mean and covariance
    initial_state_mean = [np.ma.getdata(signal)[0], 0]
    initial_state_covariance = 1.0 * np.eye(2)

    # Initialize the KalmanFilter
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )

    # Fit and smooth the data
    smoothed_state_means, _ = kf.smooth(signal)

    # Return the smoothed position (first state component)
    return smoothed_state_means[:, 0], kf


def find_speed_intervals_in_df(data, speed_column, time_column, threshold, constrain_to=None):
    """
    Finds intervals of uninterrupted time where the speed is above a certain threshold.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the timeseries data.
    - speed_column (str): Name of the column containing speed values.
    - time_column (str): Name of the column containing time values.
    - threshold (float): The speed threshold.
    - constrain_to (str, optional): Column name to constrain intervals to. Intervals spanning multiple unique values in this column will be split.

    Returns:
    - intervals (pd.DataFrame): DataFrame containing the start and end times and indices of intervals where the speed is above the threshold.
    """
    df = data.copy()

    # Identify points above the threshold
    df['above_threshold'] = df[speed_column] > threshold

    # Find the start and end of each interval
    df['block'] = (df['above_threshold'] != df['above_threshold'].shift()).cumsum()

    # If constrain_to is provided, include it in the grouping
    if constrain_to:
        df['constraint_block'] = df[constrain_to].astype(str) + '_' + df['block'].astype(str)
        group_column = 'constraint_block'
    else:
        group_column = 'block'

    intervals = df[df['above_threshold']].groupby(group_column).agg(
        start_time=(time_column, 'first'),
        end_time=(time_column, 'last'),
        start_index=(time_column, lambda x: x.index[0]),
        end_index=(time_column, lambda x: x.index[-1])
    )

    # Remove blocks that do not meet the threshold
    intervals = intervals.reset_index(drop=True)

    # Calculate the duration of each interval
    intervals['duration'] = intervals['end_time'] - intervals['start_time']

    return intervals

def subtract_phases(phase1, phase2):
    """
    Subtract two phase values and return the result bounded between -0.5 and 0.5.

    Parameters:
    phase1 (float or np.ndarray): The first phase value (or array of phase values).
    phase2 (float or np.ndarray): The second phase value (or array of phase values).

    Returns:
    float or np.ndarray: The smallest amplitude difference between the phases, bounded between -0.5 and 0.5.
    """

    pshift = lambda x: (x + 0.5) % 1 - 0.5

    # Subtract the phases
    phase1 = pshift(phase1)
    phase2 = pshift(phase2)

    diff = phase1 - phase2

    # Normalize to the range -0.5 to 0.5
    diff = (diff + 0.5) % 1 - 0.5

    return diff

class phase():
    def subtract(a, b):
        a = phase.phase_to_radians(a)
        b = phase.phase_to_radians(b)
        c = (a-b) % (2*np.pi)-np.pi
        c = phase.radians_to_phase(c)
        return c
    def add(a, b):
        a = phase.phase_to_radians(a)
        b = phase.phase_to_radians(b)
        c = (a+b) % (2*np.pi)-np.pi
        c = phase.radians_to_phase(c)
        return c
    def phase_to_radians(a):
        '''
        if a is given between 0 and 1, return between  0 and 2pi
        '''
        return a*2*np.pi
    def radians_to_phase(a):
        return a/(2*np.pi)
    def rad2deg(a):
        return math.degrees(a)
    def deg2rad(a):
        return math.radians(a)
