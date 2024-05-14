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