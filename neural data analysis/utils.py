import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


def detrend(X):
    for k in range(np.shape(X)[1]):
        h = X[:,k]
        b, a = signal.butter(2, .01)
        h = signal.filtfilt(b, a, h.T).T
        X[:,k] -= h
    return X


def findpeaks(array, peak_threshold=None, trough_threshold=None):
    peaks = []
    troughs = []
    current_state = 'peak'  
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] > array[i + 1] and array[i] > peak_threshold:
            if current_state == 'trough':
                peaks.append(i)
                current_state = 'peak'
        elif array[i - 1] > array[i] < array[i + 1] and array[i] < trough_threshold:
            if current_state == 'peak':
                troughs.append(i)
                current_state = 'trough'
    return peaks, troughs


def interpolate(X, n_samples):
    X_interp = []
    for x in X:
        size = x.shape
        idx = np.linspace(0, size[0] - 1, num=n_samples)
        if len(size) == 1:
            x_interp = interp1d(np.arange(size[0]), x)(idx)
        else:
            x_interp = np.column_stack([interp1d(np.arange(size[0]), x[:, col])(idx) for col in range(size[1])])
        X_interp.append(x_interp)
    return X_interp


def image_saver(save_path, folder_name, file_name):
    if not os.path.exists(os.path.join(save_path, folder_name)):
        os.mkdir(os.path.join(save_path, folder_name))
    plt.savefig(os.path.join(save_path, folder_name + '\\', file_name + '.png'))
    
    
def inpaint_nans(A):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A


def normalize(X, norm_method='zscore'):
    if norm_method == 'zscore':
        X = (X - np.nanmean(X)) / np.nanstd(X)
    elif norm_method == 'mean centering':
        X -= np.nanmean(X)
    elif norm_method == 'min-max':
        X = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
    elif norm_method == 'max scaling':
        X /= np.nanmax(X)
    else:
        raise ValueError("Invalid normalization method. Choose from: 'zscore', 'mean centering', 'min-max', 'max scaling'")
    return X


def map_timestamps(t1, t2):
    """
    Maps each timestamp in t1 to its closest timestamp in t2.
    
    Params:
        t1 (array): array of timestamps.
        t2 (array): array of timestamps to which t1 will be mapped.
    
    Returns:
        ndarray: array of indices indicating the closest timestamp in t2 for each timestamp in t1.
    """
    return np.array([np.where(t2 == t2[np.abs(t2 - t).argmin()])[0][0] for t in t1])


def sort_by(a, b):
    """
    Sorts `a` based on the values in `b`.
    
    Params:
    a (list): list to be sorted.
    b (array): array which determines the sorting order of `a`.
    
    Returns:
        - sorted_a (list): sorted `a`.
        - sorted_a (array): sorted `b`.
        - idx: indices of the original positions of the sorted elements.
    """
    if len(a) != len(b):
        raise ValueError("Input data must be of the same length.")
        
    combined = sorted(zip(a, b, range(len(b))), key=lambda x: x[1])
    a_sorted, b_sorted, idx_sorted = zip(*combined)
    
    return list(a_sorted), np.array(b_sorted), np.array(idx_sorted)