import numpy as np
from scipy.interpolate import interp1d
    

def peth_phase(neural_data, behavioral_data, trial_boundaries, fs_behavior, bins, selected_trials=None, verbose=False):
    """
    Compute Peri-Event Time Histograms (PETHs) in phase for multiple neurons.

    Parameters:
    - neural_data (np.ndarray): 2D array of neural data where each row represents firing rates 
      and timestamps for a neuron.
    - behavioral_data (np.ndarray): 2D array of behavioral data where the first column represents behavioral 
      measurements (e.g.: stride phase) and the second column represents corresponding timestamps.
    - trial_boundaries (np.ndarray): 2D array specifying the start and end indices of each trial 
      within the behavioral data.
    - fs_behavior (int): Sampling frequency of the behavioral data.
    - bins (np.ndarray): Array of bin edges.
    - selected_trials (int, list, or np.ndarray, optional): Specifies which trials to include in the analysis. 
      If None, all trials are included. If int, every nth trial is selected. If a list or array, trials 
      are selected based on provided indices.
    - verbose (bool, optional): If True, prints progress during trial processing.

    Returns:
    - peth (np.ndarray): PETH for each trial and neuron, where each row corresponds to a trial, 
      and each column corresponds to a neuron's firing rates in the specified bins.
    - firing_rate_trials (list of np.ndarray): List of firing rates for each trial and neuron.
    - behavior_trials (list of np.ndarray): List of behavioral data for each trial.

    Example usage:
        bin_size = 0.05
        phase_bins = np.arange(0, 1 + bin_size, bin_size)
        peth, firing_rate_strides, behavior_strides = peth_phase(firing_rate, stride_phase, strides_boundaries, fs_cam, phase_bins, selected_trials=1000, verbose=True)
    """
    if selected_trials is None:
        trial_boundaries = trial_boundaries
    elif isinstance(selected_trials, int):
        trial_boundaries = trial_boundaries[::max(1, len(trial_boundaries)//selected_trials)]  # ensure at least one event per spike train
    elif isinstance(selected_trials, (list, np.ndarray)):
        trial_boundaries = trial_boundaries[selected_trials]
    else:
        raise ValueError("Invalid event type")
        
    num_trials = len(trial_boundaries)
    num_neurons = neural_data[:, 1:].shape[1]
    time_behavior = behavioral_data[:, 1]
    behavior = behavioral_data[:, 0]
    time_neural = neural_data[:, 0]
    firing_rate = neural_data[:, 1:]
    
    firing_rate_trials = [[] for _ in range(num_neurons)]
    behavior_trials = []
    peth = np.zeros((num_neurons, num_trials, len(bins)-1))
    
    for tr, (start_idx, end_idx) in enumerate(trial_boundaries):
        start_time = time_behavior[start_idx]
        end_time = time_behavior[end_idx]
        
        behav = behavior[start_idx:end_idx+1]
        behavior_trials.append(behav)
        
        mask = ((time_neural >= start_time) & (time_neural <= end_time))
        
        original_samples = len(behav)
        target_samples = np.sum(mask)
        trial_length = len(behav) / fs_behavior
        interpolator = interp1d(np.linspace(0, trial_length, original_samples), behav, kind='linear', fill_value='extrapolate')
        behavior_resampled = interpolator(np.linspace(0, trial_length, target_samples))
        bin_idx = np.digitize(behavior_resampled, bins) - 1

        fr = firing_rate[mask]
        for n in range(num_neurons):
            firing_rate_trials[n].append(fr[:, n])            

        fr_cumulative = np.zeros((num_neurons, len(bins)-1), dtype=np.float64)
        counts = np.zeros((num_neurons, len(bins)-1), dtype=np.int64)
        idx = (np.arange(num_neurons)[:, None], bin_idx)
        np.add.at(fr_cumulative, idx, fr.T)
        np.add.at(counts, idx, 1)
        peth[:, tr, :] = np.divide(fr_cumulative, counts, out=np.zeros_like(fr_cumulative), where=counts != 0)

        if verbose:
            if tr == 0 or tr == int(num_trials*(1/4)) or tr == int(num_trials*(1/2)) or tr == int(num_trials/(3/4)) or tr == num_trials-1:
                print(f'Trial {tr+1}/{num_trials}')

    return peth, firing_rate_trials, behavior_trials