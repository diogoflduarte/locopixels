import numpy as np
import npyx
import matplotlib.pyplot as plt
import os

def plot_ephys_summary(data_path, unit, event_idx, event_ts, save_plot=False, plot_data=True):
    plt.plot(np.arange(1, 20))
    print(f'{unit}')

def plot_depth(depth_all, depth_unit, color_unit='darkgreen', ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    bins = depth_all[1][:-1]
    values = depth_all[0]
    idx = np.digitize(depth_unit, bins)
    ax.barh(bins, values, height=bins[1]-bins[0], align='edge', **kwargs)
    ax.barh(bins[idx], values[idx], color=color_unit, height=bins[1]-bins[0], align='edge')
    ax.margins(0)

def plot_raster(spikes, events, window=[-1000, 1000], trials=None, fs=30000, ax=None, **kwargs):    
    if ax is None:
        fig, ax = plt.subplots()
    if trials:
        if isinstance(trials, int):
            events = events[:trials]
        else:
            events = events[trials[0]:trials[1]]
    x, y = [], []
    for i, event in enumerate(events):
        mask = (spikes > event + (window[0]/1000)) & (spikes < event + (window[1]/1000))
        ts = spikes[mask].astype(float)
        x += list(ts - event)
        y += list(ts*0 + i)
    x, y = npyx.npa(x)*1000, npyx.npa(y)
    ax.scatter(x, y, **kwargs)
    ax.margins(0)
    ax.axvline(x=0, linestyle='--', color='k')
    return ax

def plot_peth(spikes, events, bin_size=10, window=[-1000, 1000], trials=None, color='r', ax=None, **kwargs): 
    if ax is None:
        fig, ax = plt.subplots()
    if trials:
        if isinstance(trials, int):
            events = events[:trials]
        else:
            events = events[trials[0]:trials[1]]
    t, _, ifr, ifr_std = npyx.get_processed_ifr(spikes, events, b=bin_size, window=window)
    ax.plot(t, ifr, color=color, **kwargs)
    ax.fill_between(t, ifr+ifr_std, ifr-ifr_std, alpha=0.4, color=color)
    ax.axvline(x=0, linestyle='--', color='k')
    ax.margins(0)
    return ax

def plot_wvf(data_path, unit, channel=None, fs=30000, n_wvf=100, t_wvf=90, color='k', ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if channel is None:
        channel = npyx.get_peak_chan(data_path, unit)
    waveforms = npyx.wvf(data_path, unit, n_waveforms=n_wvf, t_waveforms=t_wvf)
    w_time = np.arange(waveforms.shape[1]) * 1000 / fs
    w_mean = np.squeeze(waveforms.mean(0)[:, channel])
    w_std = np.squeeze(waveforms.std(0)[:, channel])
    ax.fill_between(w_time, w_mean+w_std, w_mean-w_std, alpha=0.4, color=color)
    ax.plot(w_time, w_mean, color=color, **kwargs)
    ax.margins(0)
    return ax

def plot_isi(spikes, fs=30000, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    isi=npyx.compute_isi(spikes, fs=fs)*1000
    ax.hist(isi, **kwargs)  
    return ax

def plot_acg(data_path, unit, bin_size=0.2, win_size=80, normalize='Hertz', fs=30000, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    acg = npyx.acg(data_path, unit, bin_size=bin_size, win_size=win_size, fs=fs, normalize=normalize)
    lags = np.linspace(int(-win_size/2), int(win_size/2), len(acg))
    ax.bar(lags, acg, **kwargs)
    ax.margins(0)
    return ax

def plot_raw(amplitude, spike_train, fs=30000, color='dimgray', alpha=0.5, marker_size=2, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()      
    ax.scatter(spike_train[::100]/(60*fs), amplitude[::100], color=color, s=marker_size, alpha=alpha, **kwargs)
    ax.margins(0)
    divider = ax.inset_axes([1.05, 0, 0.1, 1], transform=ax.transAxes)
    divider.hist(amplitude, bins=150, orientation='horizontal', color=color)
    divider.axis('off')
    return ax

def image_saver(save_path, folder_name, file_name):
    if not os.path.exists(os.path.join(save_path, folder_name)):
        os.mkdir(os.path.join(save_path, folder_name))
    plt.savefig(os.path.join(save_path, folder_name + '\\', file_name + '.png'))