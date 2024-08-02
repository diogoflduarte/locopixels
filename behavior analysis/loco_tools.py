import numpy as np
import os

os.chdir(r'C:\Users\User\Desktop\behavior_analysis')
from util import findpeaks, interpolate, detrend
import phaser

def find_phase(k):
    y = np.array(k)
    y = detrend(y.T).T
    phsr = phaser.Phaser(y=y)
    k[:] = phsr.phaserEval(y)[0,:]
    return k

def get_cycles(principal_components, global_phase, sr, peak_threshold=6, 
                       trough_threshold=0.5, interp_samples=200):  
    # Detect onset and offset of each cycle
    offset, onset = findpeaks(global_phase, peak_threshold=peak_threshold, trough_threshold=trough_threshold)
    # Interpolate cycles 
    cycles = []
    for i in range(len(offset)):
        cycles.append(principal_components[onset[i]:offset[i]+1, :])
    interp_cycles = interpolate(cycles, n_samples=interp_samples)
    # Compute cycle duration
    cycle_duration = np.diff(onset) * (1 / sr)
    return interp_cycles, cycles, cycle_duration, [onset, offset]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def process_tracking_data(tracking_data, threshold, interpolation_method='linear'):
    """
    Processes Deeplabcut tracking data by setting low-confidence samples to NaN and interpolating.

    Parameters:
    - tracking_data (np.array or pd.DataFrame): Input tracking data with features and likelihoods.
    - threshold (float): Likelihood threshold.
    - interpolation_method (str): Method for interpolation ('linear', 'quadratic', 'cubic').

    Returns:
    - pd.DataFrame: DataFrame with NaNs replaced by interpolated values.
    """
    # Convert to DataFrame if necessary
    if isinstance(tracking_data, np.ndarray):
        tracking_data = pd.DataFrame(tracking_data)

    # Set values below threshold to NaN in likelihood columns
    likelihood_columns = tracking_data.columns[1::2]
    tracking_data[likelihood_columns] = tracking_data[likelihood_columns].where(tracking_data[likelihood_columns] >= threshold)

    return tracking_data.interpolate(method=interpolation_method)

def compute_stride_phase(tracks, stride_events, phase_type='st-sw-st', sw_on_phase=0.5):
    """
    Computes stride phase from paw position data.
    
    Parameters:
    - tracks (pd.DataFrame): DataFrame in which each column represents the position of a paw, and each row represents a sample.
    - stride_events (dict of tuples): A dictionary of tuples with the stance onset, swing onset and swing offset for muliple strides for each paw.
      Each key represents a paw. Each row should contain three values: [st_on, sw_on, sw_off], which correspond to the start of the stance phase, 
      the start of the swing phase, and the end of the swing phase, respectively.
    - phase_type (str, optional): The type of stride phase calculation to perform. It can be 'st-sw-st' or 'st-st'. 
      - 'st-sw-st': Computes stride phase as a combination of stance and swing phases.
      - 'st-st': Computes stride phase as a linear progression from stance to swing.
      Default is 'st-sw-st'.
    - sw_on_phase (float, optional): The phase value at which the swing phase starts. Default is 0.5.
    
    Returns:
    - pd.DataFrame: A DataFrame of the stride phase for each paw.
    """
    
    stride_phase = np.full(tracks.shape, np.nan)
    for p, paw in enumerate(tracks):        
        for s in range(len(stride_events[paw])):
            st_on = stride_events[paw][s, 0]
            sw_on = stride_events[paw][s, 1]
            sw_off = stride_events[paw][s, 2]
            
            if phase_type == 'st-sw-st':
                len_st = sw_on - st_on
                len_sw = sw_off - sw_on
                stride_phase[st_on:sw_on+1, p] = np.linspace(0, sw_on_phase, len_st+1, endpoint=True)
                stride_phase[sw_on:sw_off+1, p] = np.linspace(sw_on_phase, 1, len_sw+1, endpoint=True)
                
            elif phase_type == 'st-st':
                len_stride = sw_off - st_on
                stride_phase[st_on:sw_off + 1, p] = np.linspace(0, 1, len_stride + 1, endpoint=True)
            
            else:
                raise ValueError(f"Unrecognized phase_type: {phase_type}")
    
    return pd.DataFrame(stride_phase, columns=tracks.columns)


def condition_on(data, threshold, min_duration, sampling_rate, operator='<'):
    """
    Identifies indices where the data meets a certain condition (based on the specified operator) 
    for a minimum duration.

    Parameters:
    - data (numpy array): 1-D array of data.
    - threshold (float): threshold value for comparison.
    - min_duration (float): minimum duration to be considered (in seconds).
    - sampling_rate (float): time interval between measurements (in seconds).
    - operator (str): comparison operator ('<', '<=', '>', '>=')

    Returns:
    - numpy array: indices of the data where the conditions are met.
    """
    
    # Map operator to corresponding comparison function
    operators = {
        '<': lambda x, t: x < t,
        '<=': lambda x, t: x <= t,
        '>': lambda x, t: x > t,
        '>=': lambda x, t: x >= t
    }
    
    if operator not in operators:
        raise ValueError(f"Invalid operator: {operator}. Must be one of {list(operators.keys())}.")
    
    # Compute the minimum number of samples for the given duration
    min_samples = int(min_duration / sampling_rate)
    condition_func = operators[operator]
    condition_met = condition_func(data, threshold)
    
    indices = []
    current_start = None

    # Iterate through the condition check results to find sustained periods
    for i, met in enumerate(condition_met):
        if met:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                if i - current_start >= min_samples:
                    indices.extend(range(current_start, i))
                current_start = None

    # Check if the last period reached the end of the array
    if current_start is not None and len(data) - current_start >= min_samples:
        indices.extend(range(current_start, len(data)))

    return np.array(indices)


def clip_strides(condition_met_indices, stride_boundaries):
    """
    Identifies indices of strides in which a given condition is met.

    Parameters:
    - condition_met_indices (list or array-like): A list or array of indices where a specific condition is met.
    - stride_boundaries (list of tuples): A list of tuples where each tuple defines the start and end indices of a stride. 

    Returns:
    - invalid_strides (list): A list of indices of the stride in which the condition is met.
    - invalid_indices (list): A list of indices that are part of the stride boundaries deemed invalid.
    """
    
    invalid_strides = []   
    invalid_indices = []   
    
    for i, (on, off) in enumerate(stride_boundaries):
        stride_indices = np.arange(on, off + 1)
        
        if np.any(np.isin(stride_indices, condition_met_indices)):
            # If the stride overlaps with `condition_met_indices`, it's invalid
            invalid_strides.append(i)  
            invalid_indices.extend(stride_indices)
    
    return np.array(invalid_strides), np.array(invalid_indices)


def compute_gait_metrics(tracks, stride_events, sr, paw1='FR', paw2='FL', metrics=None, plot_metrics=False):
    """
    Compute gait metrics and optionally plot their distributions.

    Parameters:
        tracks (dict): Dictionary with x position data for each paw.
        stride_events (dict): Dictionary of tuples with stance onset, swing onset, swing offset indices for each paw.
        sr (float): Sampling rate.
        paw1 (str): Key for the first paw in `tracks` and `stride_events`.
        paw2 (str): Key for the second paw in `tracks` and `stride_events`.
        metrics (list): List of metrics to compute. If None, all metrics are computed.
        plot_metrics (bool): If True, plots distribution of computed metrics.

    Returns:
        pd.DataFrame: DataFrame containing the computed metrics.
    """
    
    # Define valid metrics
    valid_metrics = {
        'stride duration', 'stance duration', 'swing duration', 'duty factor',
        'swing excursion', 'swing speed', 'step length', 'center of oscillation',
        'double support'
    }
    
    if metrics is not None:
        # Check if all metrics in the list are valid
        invalid_metrics = set(metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Invalid metrics specified: {', '.join(invalid_metrics)}")

    data = {}

    # Extract stride event indices
    st_on_paw1 = stride_events[paw1][:, 0]
    sw_on_paw1 = stride_events[paw1][:, 1]
    sw_off_paw1 = stride_events[paw1][:, 2]
    
    # st_on_paw2 = stride_events[paw2][:, 0]
    # sw_off_paw2 = stride_events[paw2][:, 2]
    
    # Extract position data 
    paw1_position = tracks[paw1]
    # paw2_position = tracks[paw2]
    
    # Compute gait metrics
    if metrics is None or 'stride duration' in metrics:
        stride_duration = (sw_off_paw1 - st_on_paw1) / sr
        data['stride duration'] = stride_duration

    if metrics is None or 'stance duration' in metrics:
        stance_duration = (sw_on_paw1 - st_on_paw1) / sr
        data['stance duration'] = stance_duration

    if metrics is None or 'swing duration' in metrics:
        swing_duration = (sw_off_paw1 - sw_on_paw1) / sr
        data['swing duration'] = swing_duration

    if metrics is None or 'duty factor' in metrics:
        stride_duration = data.get('stride duration', (sw_off_paw1 - st_on_paw1) / sr)
        stance_duration = data.get('stance duration', (sw_on_paw1 - st_on_paw1) / sr)
        duty_factor = (stance_duration / stride_duration) * 100
        data['duty factor'] = duty_factor

    if metrics is None or 'swing excursion' in metrics:
        swing_excursion = np.array([paw1_position[int(off)] - paw1_position[int(on)] for off, on in zip(sw_off_paw1, sw_on_paw1)])
        data['swing excursion'] = swing_excursion

    if metrics is None or 'swing speed' in metrics:
        swing_excursion = data.get('swing excursion', np.array([paw1_position[int(off)] - paw1_position[int(on)] for off, on in zip(sw_off_paw1, sw_on_paw1)]))
        stride_duration = data.get('stride duration', (sw_off_paw1 - st_on_paw1) / sr)
        swing_speed = swing_excursion / stride_duration
        data['swing speed'] = swing_speed

    if metrics is None or 'center of oscillation' in metrics:
        center_of_oscillation = np.nanmean([paw1_position[np.int64(st_on_paw1)], paw1_position[np.int64(sw_on_paw1)]], axis=0)
        data['center of oscillation'] = center_of_oscillation
    
    # TODO: match left-right stances for interlimb metrics!
    # if metrics is None or 'step length' in metrics:
    #     step_length = np.array([paw1_position[int(st_on1)] - paw2_position[int(st_on2)] for st_on1, st_on2 in zip(st_on_paw1, st_on_paw2)])
    #     data['step length'] = step_length

    # if metrics is None or 'double support' in metrics:
    #     start = np.maximum(st_on_paw1, st_on_paw2)
    #     end = np.minimum(sw_off_paw1, sw_off_paw2)
    #     double_support_time = np.maximum(0, (end - start) / sr)
    #     double_support_percentage = (double_support_time / stride_duration) * 100
    #     data['double support'] = double_support_percentage

    gait_metrics = pd.DataFrame(data)
    
    if plot_metrics and gait_metrics.shape[0] > 0:
        plot_gait_metrics(gait_metrics)

    return gait_metrics


def plot_gait_metrics(gait_metrics):
    """
    Plot distribution of gait metrics.

    Parameters:
        gait_metrics (pd.DataFrame): DataFrame containing the gait metrics.
    """

   # Create a figure with subplots
    num_metrics = len(gait_metrics.columns)
    num_rows = int(np.ceil(num_metrics / 2))  # Arrange plots in two columns
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, num_rows * 5), constrained_layout=True)
    if num_rows == 1:
        axes = np.array([axes])  # Ensure axes is a 2D array for consistency

    # Plot each metric
    for i, metric in enumerate(gait_metrics.columns):
        ax = axes[i // 2, i % 2]
        sns.histplot(gait_metrics[metric], kde=True, ax=ax, color='darkgreen')
        ax.set_xlabel(metric)
        ax.set_ylabel('freq.')
        
        # Remove grid
        ax.grid(False)
        
        # Calculate 5th and 95th percentiles
        lower_percentile = gait_metrics[metric].quantile(0.05)
        upper_percentile = gait_metrics[metric].quantile(0.95)
        
        # Shade regions below the 5th percentile and above the 95th percentile
        ax.axvspan(xmin=gait_metrics[metric].min(), xmax=lower_percentile, color='dimgray', alpha=0.1)
        ax.axvspan(xmin=upper_percentile, xmax=gait_metrics[metric].max(), color='dimgray', alpha=0.1)

    # Remove empty subplots if the number of metrics is odd
    if num_metrics % 2 != 0:
        fig.delaxes(axes[-1, -1])

    plt.show()


def filter_strides(stride_metrics, stride_metrics_thresholds):
    """
    Identify bad strides based on provided metrics.

    This function evaluates each column of the `stride_metrics` DataFrame against
    the thresholds provided in `stride_metrics_thresholds`. It returns a boolean Series
    indicating which rows have values outside the specified bounds.

    Parameters:
    - stride_metrics (pd.DataFrame): The DataFrame containing metrics (columns) for each stride (row) to be filtered.
    - stride_metrics_thresholds (dict of dict): A dictionary of dictionaries or integers or None.
        If None, default bounds will be used:
            - Lower bound: 5th percentile of the column data
            - Upper bound: 95th percentile of the column data


    Returns:
    - pd.Series: A boolean Series with the same index as `stride_metrics`. Each value is 
      `True` if the corresponding row has at least one column value outside the specified 
      bounds, and `False` otherwise.
    flag = pd.Series(False, index=stride_metrics.index)
    """
    flag_idx = pd.Series(False, index=stride_metrics.index)
    for col, bounds in stride_metrics_thresholds.items():
        if col in stride_metrics.columns:
            if bounds is None:
                lower_bound = stride_metrics[col].quantile(0.05)
                upper_bound = stride_metrics[col].quantile(0.95)
            else:
                lower_bound = bounds.get('lower', stride_metrics[col].min() - 1)  # default to min-1 if no lower bound
                upper_bound = bounds.get('upper', stride_metrics[col].max() + 1)  # default to max+1 if no upper bound
            
            flag_idx |= (stride_metrics[col] < lower_bound) | (stride_metrics[col] > upper_bound)
    
    return flag_idx
