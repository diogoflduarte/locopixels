import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from CareyConstants import CareyConstants
import CareyUtils
import seaborn as sns
import cmocean
import cmcrameri
from matplotlib.collections import LineCollection
import scipy.stats
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import dash
from dash import dcc, html, Input, Output, ctx, callback, State
import dash_bootstrap_components as dbc
import dash_player as player
import plotly
import plotly.express as px
import plotly.graph_objects as go
import cv2
import base64
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
from matplotlib.animation import FuncAnimation
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def multiLinePlot(data, colors, linewidth=1, alpha=0.5, x=None):
    # plt.figure()
    if isinstance(data, np.ndarray):
        if x is None:
            plt.plot(data, color=colors, linewidth=linewidth, alpha=alpha)
        else:
            plt.plot(x, data, color=colors, linewidth=linewidth, alpha=alpha)
    elif isinstance(data, list):
        pass

def plotTracks(tracks, time=None, strides=None, stride_type='stance', strides_paw='FR', paws='all', framerate=330,
               linewidth=1):
    """plots the tracked mouse paws"""

    CareyUtils.validateDimensions(tracks, 2)
    n_pts, n_paws = tracks.shape

    if time is None:
        print(f'Creating time array with framerate = {framerate} fps.')
        final_time = n_pts / framerate
        time = np.linspace(0, final_time, n_pts)
    else:
        CareyUtils.validateDimensions(time, 1)

    ca = plt.gca()

    if paws == 'all':
        paws = [0, 1, 2, 3]
    elif isinstance(paws, list):
        # this should be a list of strings of paws
        paws_input = paws
        paws = None
        paw = [CareyConstants.paw_idx[p] for p in paws_input]

    for p in paws:
        plt.plot(time, tracks[:,p], linewidth=linewidth, color=CareyConstants.paw_colors_sns[p])


    # now add the tracks, if there are any
    if strides is not None:
        if stride_type == 'stance':
            pt_type = ['stance', 'swing', 'stance']
        elif stride_type == 'swing':
            pt_type = ['swing', 'stance', 'swing']

        markeredgecolor = CareyConstants.paw_colors_sns[CareyConstants.paw_idx[strides_paw]]

        for ss in range(strides.shape[0]):
            for pt in range(strides.shape[1]):
                if pt_type[pt] == 'stance':
                    markerfacecolor = markeredgecolor
                elif pt_type[pt] == 'swing':
                    markerfacecolor = 'w'

                plt.plot(time[strides[ss, pt]-1], tracks[strides[ss, pt]-1, CareyConstants.paw_idx[strides_paw]],
                         'o', markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)

def scatterPanels(datamat, cols=[0,1,2], lbl=['PC1', 'PC2', 'PC3'], color_by=None, s=0.1, alpha=0.2):
    """plots 3D data in 3 scatter panels of 2D projections"""
    plt.figure()
    plt.scatter(datamat[:, cols[0]], datamat[:, cols[1]], c=color_by, s=s, alpha=alpha,
                cmap=sns.color_palette("rocket", as_cmap=True))
    plt.colorbar()
    plt.xlabel(lbl[0]), plt.ylabel(lbl[1])

    plt.figure()
    plt.scatter(datamat[:, cols[1]], datamat[:, cols[2]], c=color_by, s=s, alpha=alpha,
                cmap=sns.color_palette("rocket", as_cmap=True))
    plt.colorbar()
    plt.xlabel(lbl[1]), plt.ylabel(lbl[2])

    plt.figure()
    plt.scatter(datamat[:, cols[0]], datamat[:, cols[2]], c=color_by, s=s, alpha=alpha,
                cmap=sns.color_palette("rocket", as_cmap=True))
    plt.colorbar()
    plt.xlabel(lbl[0]), plt.ylabel(lbl[2])

def plotSingleTrials_2D(x, y, data=None, trial=None, linewidth=1, color=None, alpha=0.2):
    """plots lines of single trials on a 2D space.
    INPUTS: x, y
    OPTIONAL:   data    if data exists, it should be a pandas dataframe
                trial   if trial exists and data exists, it should be the name of the column with the trial identifier
                        if trial exists but data doesn't, trial should be an array of the same size of x and y"""

    plt.figure()

    if data is None:
        if trial is None:
            trial = np.zeros(x.shape)

        trial_numbers = np.unique(trial)

        for tt in trial_numbers:
            sns.lineplot(x, y, size=linewidth, hue=color)

    else:
        if trial is None:
            trial = np.zeros(data[x].shape)

        trial_numbers = np.unique(data[trial].values)

        c = color

        for tt in trial_numbers:
            mini_df = data[data[trial]==tt]
            if isinstance(color, str) and any(color in s for s in data.columns.values):
                c = mini_df[color]
            # sns.lineplot(data=mini_df, x=x, y=y, linewidth=linewidth, hue=color, alpha=alpha)
            plt.plot(mini_df[x], mini_df[y], linewidth=linewidth, alpha=alpha, color=c)

def colormapLine(x, y, color_array, linewidth=1, cmap_lims=None, alpha=1, cmap=cmocean.cm.phase, forcelims=False):

    ax = plt.gca()

    if cmap_lims is None:
        cmap_lims = [np.min(color_array), np.max(color_array)]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(cmap_lims[0], cmap_lims[1]))
    lc.set_array(color_array)
    lc.set_linewidth(linewidth)
    lc.set_alpha(alpha)
    plt.gca().add_collection(lc)

    # now we'll have to be clever about this. autoscale() and autoscale_view() don't care about LineCollection, so
    # we'll check and update manually
    x_lowlim, x_highlim = ax.get_xlim()
    y_lowlim, y_highlim = ax.get_ylim()
    if x_lowlim > np.min(x)*1.1:
        x_lowlim = np.min(x)*1.1
    if x_highlim < np.max(x)*1.1:
        x_highlim = np.max(x)*1.1
    if y_lowlim > np.min(y)*1.1:
        y_lowlim = np.min(y)*1.1
    if y_highlim < np.max(y)*1.1:
        y_highlim = np.max(y)*1.1

    if forcelims:
        x_lowlim = np.min(x) * 1.1
        x_highlim = np.max(x) * 1.1
        y_lowlim = np.min(y) * 1.1
        y_highlim = np.max(y) * 1.1

    plt.xlim(x_lowlim, x_highlim)
    plt.ylim(y_lowlim, y_highlim)

    plt.show()

def populationRaster(spiketimes, spikeclusters, timebounds, samprate=30000, markersize=1, color='k'):
    ax = plt.axes()

    sc_array = np.unique(spikeclusters)
    # iterate through spikeclusters. each line for a spike cluster (unit / cell)
    for sc in range(len(sc_array)):
        spikes_thiscluster = spiketimes[spikeclusters==sc_array[sc]]
        these_spikes = spikes_thiscluster[np.logical_and(spikes_thiscluster>timebounds[0],
                                                         spikes_thiscluster<timebounds[1])]

        # plot the line
        tl = plt.plot(these_spikes, np.ones(these_spikes.shape[0]) * sc, '.')
        plt.setp(tl, 'markersize', markersize, 'color', color)

def plotTraces(rawdata, time=None, samprate=CareyConstants.DEF_NPX_FS, spacing=1, linewidth=1,
               channel_id=None, text_x=-0.1, fontsize=10, palette=None):

    n_timepoints, n_channels = rawdata.shape
    # normalize raw data so that loud and shy units speak the same volume
    rawdata = scipy.stats.zscore(rawdata, 0)
    if time is None:
        time = np.linspace(0, n_timepoints/samprate, n_timepoints)

    ax = plt.gca()
    fig = plt.gcf()

    for ii in range(n_channels):
        if palette==None:
            color='k'
        else:
            color=palette[ii]
        tl = plt.plot(time, rawdata[:,ii] + spacing*ii, linewidth=linewidth, color=color)
        plt.text(text_x, spacing*ii, str(channel_id[ii]), fontsize=fontsize)

    ax.get_yaxis().set_ticks([])

    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plotSingleTrialsColormapLine(data_df, x, y, trials, color_by, linewidth=1, alpha=1, cmap=cmocean.cm.phase):
    # get list of trials to iterate
    trial_list = np.unique(data_df[trials].values)

    cmap_lims = [ np.min(data_df[color_by]), np.max(data_df[color_by]) ]

    for tt in trial_list:
        if tt==0:
            forcelims=True
        else:
            forcelims=False

        this_trial = data_df[data_df[trials]==tt]
        x_this_trial = this_trial[x].values
        y_this_trial = this_trial[y].values
        color_array  = this_trial[color_by].values

        colormapLine(x_this_trial, y_this_trial, color_array, linewidth=linewidth,
                     cmap_lims=cmap_lims, alpha=alpha, cmap=cmap, forcelims=forcelims)

    plt.xlabel(x)
    plt.ylabel(y)

def plotEventAligned_multiline(df, x='sessionwise_time', y='FRx', event='FR_SwOn', bounds=[-0.2, 0.2], color='k',
                               plot=True, limit=False, linewidth=0.1):
    # find the number of events

    event_indices = np.where(df[event])[0]
    if limit is not False:
        event_indices = event_indices[np.random.permutation(event_indices.shape[0])[:limit]]
    n_events = event_indices.shape

    estimate_dt = np.median(np.diff(df[x]))
    estimate_timepoints = np.round(bounds[1]/estimate_dt)

    n_pts = int(np.round(estimate_timepoints * 2) + 1)
    data = np.empty((n_pts, 0))

    for event_num, event_idx in enumerate(tqdm(event_indices)):
        # sessionwise time at the start of the event
        t0 = df[x][event_idx]

        # let's check if t_before even exists (might be before the recording started or after ir ended)
        if event_idx - estimate_timepoints < 0 or event_idx + estimate_timepoints >= df.shape[0]:
            continue

        # now the second issue is that it might be between the dead second for camera save between trials
        t_before_real = t0 + bounds[0]
        dataframe_idx_before = np.argmin(np.abs(df[x]-t_before_real))
        t_before_from_idx = df[x][dataframe_idx_before]

        t_after_real = t0 + bounds[1]
        dataframe_idx_after = np.argmin(np.abs(df[x]-t_after_real))
        t_after_from_idx = df[x][dataframe_idx_after]

        if (np.abs(t_before_real - t_before_from_idx) > estimate_dt or
            np.abs(t_after_real - t_after_from_idx) > estimate_dt):
            continue

        # if all previous conditions are respected, let's compile the strides
        this_trial = df[y][dataframe_idx_before:dataframe_idx_after].values
        real_t_array = np.linspace(t_before_real, t_after_real, num=n_pts, endpoint=True)
        this_t_array = np.linspace(t_before_from_idx, t_after_from_idx, num=this_trial.shape[0], endpoint=True)
        this_trial_interp = np.interp(real_t_array, this_t_array, this_trial)

        data = np.append(data, this_trial_interp[:,None], axis=1)

    # finally do the plotting
    colors = color
    x = np.linspace(bounds[0], bounds[1], num=data.shape[0], endpoint=True)
    if plot:
        multiLinePlot(data, colors, linewidth=1, alpha=0.5, x=None)

    return data, x
    # todo: another version of this plot that iteratively plots the variable without computing necessarily.

def CSalignedSS(ss, cs, binsize=0.01, window=[-0.05, 0.05]):


    bin_edges = np.arange(window[0], window[1]+binsize/2, binsize)
    n_bins = bin_edges.shape[0]-1
    data = np.zeros((n_bins), dtype=int)
    data_bunch = np.empty(0)

    for c in cs:
        these_bins = bin_edges+c
        this_ss_subset = ss[np.logical_and(ss>c+window[0], ss<=c+window[1])]
        this_ss_bins = np.histogram(this_ss_subset, these_bins)[0]
        data = data + this_ss_bins

        # data for automated methods
        data_bunch = np.hstack((data_bunch, this_ss_subset-c))

    plt.hist(data_bunch, bin_edges)

def ridgeplot(data, x=None, y=None, by=None, linewidth=1, spacing=1.0, multiplier=1.0, color='k', alpha=None,
                    visibleframe=False, zscore=False):
    if isinstance(data, np.ndarray):
        if data.ndim is not 2:
            raise ValueError('Dimensions different from 2 not supported')
        n_xpts, n_lines = data.shape

        if zscore is True:
            data = CareyUtils.zscore(data)

        data = data*multiplier + np.tile(np.arange(n_lines)*spacing, (n_xpts, 1))
        if x is None:
            x = np.arange(n_xpts)

        ax = plt.gca()
        plt.plot(x, data, color=color, linewidth=linewidth, alpha=alpha);
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_visible(visibleframe)
        ax.axes.get_yaxis().set_visible(visibleframe)

def plot3D(df, axis1='PC1', axis2='PC2', axis3='PC3', colorby='phase', colormap=cmocean.cm.phase):
    ax = []
    MSIZE = 0.5
    fig = plt.figure(figsize=(12, 12))
    gs = plt.matplotlib.gridspec.GridSpec(3, 2, width_ratios=[2, 1])
    ax.append(fig.add_subplot(gs[:, 0], projection='3d'))
    ax[0].scatter(df[axis1], df[axis2], df[axis3], c=df['phase'], cmap=colormap,
                  s=MSIZE, rasterized=True)
    ax[0].set_xlabel(axis1)
    ax[0].set_ylabel(axis2)
    ax[0].set_zlabel(axis3)
    ax[0].set_facecolor((1, 1, 1, 0))
    ax[0].grid(False)

    ax.append(fig.add_subplot(gs[0, 1]))
    sc = sns.scatterplot(data=df, x=axis1, y=axis2,
                         hue='phase', palette=colormap, estimator=None, s=MSIZE, ax=ax[1], rasterized=True)
    elev = 130
    azim = -76
    ax[0].view_init(elev, azim)

    ax.append(fig.add_subplot(gs[1, 1]))
    sns.scatterplot(data=df, x=axis1, y=axis3,
                    hue='phase', palette=colormap, estimator=None, s=MSIZE, ax=ax[2], rasterized=True)

    ax.append(fig.add_subplot(gs[2, 1]))
    sns.scatterplot(data=df, x=axis2, y=axis3,
                    hue='phase', palette=colormap, estimator=None, s=MSIZE, ax=ax[3], rasterized=True)

    for ii in np.arange(1, len(ax)):
        ax[ii].legend_.remove()
        # ax[ii].set_xlim((-80, 80))
        # ax[ii].set_ylim((-80, 80))
        ax[ii].set_aspect('equal', adjustable='box')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig, ax
def plot3Dlines(df, axis1='PC1', axis2='PC2', axis3='PC3', colorby='phase', colormap=cmocean.cm.phase):
    # todo: complete this function
    ax = []
    MSIZE = 0.5
    fig = plt.figure(figsize=(12, 12))
    gs = plt.matplotlib.gridspec.GridSpec(3, 2, width_ratios=[2, 1])
    ax.append(fig.add_subplot(gs[:, 0], projection='3d'))
    ax[0].scatter(df[axis1], df[axis2], df[axis3], c=df['phase'], cmap=colormap,
                  s=MSIZE, rasterized=True)
    ax[0].set_xlabel(axis1)
    ax[0].set_ylabel(axis2)
    ax[0].set_zlabel(axis3)
    ax[0].set_facecolor((1, 1, 1, 0))
    ax[0].grid(False)

    ax.append(fig.add_subplot(gs[0, 1]))
    sc = sns.lineplot(data=df, x=axis1, y=axis2,
                         hue='phase', palette=colormap, estimator=None, s=MSIZE, ax=ax[1], rasterized=True, sort=False)
    sns.lineplot(data=behav_small[behav_small.FR_Sw_Stride == strides[5]], x='PC1', y='PC2', estimator=None, sort=False)
    elev = 130
    azim = -76
    ax[0].view_init(elev, azim)

    ax.append(fig.add_subplot(gs[1, 1]))
    sns.scatterplot(data=df, x=axis1, y=axis3,
                    hue='phase', palette=colormap, estimator=None, s=MSIZE, ax=ax[2], rasterized=True)

    ax.append(fig.add_subplot(gs[2, 1]))
    sns.scatterplot(data=df, x=axis2, y=axis3,
                    hue='phase', palette=colormap, estimator=None, s=MSIZE, ax=ax[3], rasterized=True)

    for ii in np.arange(1, len(ax)):
        ax[ii].legend_.remove()
        # ax[ii].set_xlim((-80, 80))
        # ax[ii].set_ylim((-80, 80))
        ax[ii].set_aspect('equal', adjustable='box')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig, ax

def create_colormap(rgb_color):
    colors = [(0, 0, 0), rgb_color]  # From white to the specified color
    n_bins = 100  # Discretizes the interpolation into 100 steps
    cmap_name = 'custom_cmap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
def multicoloredline_2d(df, x, y, colorby, trials=None, cmap=sns.cm.crest_r, lw=1, alpha=1.0, fig=None, ax=None, colorbar=True):
    """
    Plots PC1 vs PC2 as lines with color coding by the values of a specified column.

    Parameters:
    - df: pandas DataFrame containing the data
    - x: name of the column for PC1 values
    - y: name of the column for PC2 values
    - colorby: name of the column for color coding
    - rgb_color: a tuple of the RGB color to use for the colormap
    """
    # Create a colormap from black to the specified RGB color
    # colors = [(0, 0, 0), rgb_color]  # From black to the specified color
    # n_bins = 100  # Discretizes the interpolation into 100 steps
    # cmap_name = 'custom_cmap'
    # cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Extracting the relevant data
    pc1 = df[x].values
    pc2 = df[y].values
    var1 = df[colorby].values

    # Normalize the color column for coloring
    norm = plt.Normalize(var1.min(), var1.max())

    if fig is None:
        fig, ax = plt.subplots()
    plt.ion()

    if trials is None:
        df['trials'] = 0
        trials = 'trials'
    print( np.unique(df[trials]) )

    for tt, trial in enumerate(np.unique(df[trials])):
        this_trial_idx = df[trials]==trial
        pc1 = df[x][this_trial_idx].values
        pc2 = df[y][this_trial_idx].values
        var1 = df[colorby][this_trial_idx].values

        # Create line segments
        points = np.array([pc1, pc2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm, rasterized=True)
        lc.set_array(var1)
        lc.set_linewidth(lw)
        lc.set_alpha(alpha)
        plt.gca().add_collection(lc)
        plt.xlim(pc1.min(), pc1.max())
        plt.ylim(pc2.min(), pc2.max())

    if colorbar:
        cbar = plt.colorbar(lc)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

    return fig, ax
def multicoloredline_3d(df, x, y, z, colorby, trials=None, cmap=plt.cm.viridis, lw=1, alpha=1.0):
    """
    Plots PC1 vs PC2 vs PC3 as 3D lines with color coding by the values of a specified column.

    Parameters:
    - df: pandas DataFrame containing the data
    - x: name of the column for PC1 values
    - y: name of the column for PC2 values
    - z: name of the column for PC3 values
    - colorby: name of the column for color coding
    - trials: name of the column for trials (optional)
    - cmap: colormap to use for coloring
    - lw: line width
    """
    # Extracting the relevant data
    pc1     = df[x].values
    pc2     = df[y].values
    pc3     = df[z].values
    var1    = df[colorby].values

    # Normalize the color column for coloring
    norm = plt.Normalize(var1.min(), var1.max())

    if trials is None:
        df['trials'] = 0
        trials = 'trials'
    unique_trials = np.unique(df[trials])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for trial in unique_trials:
        this_trial_idx = df[trials] == trial
        pc1_trial = df[x][this_trial_idx].values
        pc2_trial = df[y][this_trial_idx].values
        pc3_trial = df[z][this_trial_idx].values
        var1_trial = df[colorby][this_trial_idx].values

        # Create line segments
        points = np.array([pc1_trial, pc2_trial, pc3_trial]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a Line3DCollection
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(var1_trial)
        lc.set_linewidth(lw)
        lc.set_alpha(alpha)
        ax.add_collection3d(lc, zs=pc3_trial, zdir='z')

    # Create a colorbar
    cbar = plt.colorbar(lc, ax=ax, pad=0.1)
    cbar.set_label(colorby)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.grid(False)

    ax.set_xlim([pc1.min(), pc1.max()])
    ax.set_ylim([pc2.min(), pc2.max()])
    ax.set_zlim([pc2.min(), pc2.max()])

    plt.show()

    return fig, ax
def plot_coefficients(coefficients, features=None, colors=None, metric='Principal Component'):
    """
    Plots barplots for the coefficients of principal components.

    Parameters:
    - coefficients: 2D numpy array of shape (3, 12) with coefficients
    - features: list of feature names
    - paw_colors: list of colors for each paw
    """
    num_pcs = coefficients.shape[0]
    num_features = coefficients.shape[1]

    if colors is None:
        colors = np.repeat(np.array(CareyConstants.paw_colors_sns), 3, axis=0)
    if features is None:
        features = [f"{paw}{axis}" for paw in CareyConstants.paw_labels for axis in ['x', 'y', 'z']]

    fig, axes = plt.subplots(num_pcs, 1 , sharex=True)
    for i, ax in enumerate(axes):
        coeff = coefficients[i,:]
        ax.bar(features, coeff, color=colors)
        ax.set_title(f'{metric} {i + 1}')
        ax.set_ylabel('Coefficient Value')
        if i == num_pcs - 1:
            ax.set_xticklabels(features, rotation=45)
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.show()
def rgb_and_opacity_to_rgba(rgb, opacities):
    rgb = np.array(rgb)
    opacities = np.array(opacities)

    if rgb.shape[0] != opacities.shape[0]:
        raise ValueError("The length of the RGB array must match the length of the opacity array.")

    if not np.all((0 <= opacities) & (opacities <= 1)):
        raise ValueError("All opacity values must be between 0 and 1.")

    # rgba_colors = [f'rgba({r}, {g}, {b}, {a})' for r, g, b, a in zip(rgb[:, 0], rgb[:, 1], rgb[:, 2], opacities)]
    rgba_colors = [f'rgba({r}, {g}, {b}, {a})' for r, g, b, a in zip(rgb[:, 0], rgb[:, 1], rgb[:, 2], opacities)]
    return rgba_colors
def map_values_to_colors(values, colormap=cmocean.cm.phase):
    # Get the colormap from matplotlib
    cmap = plt.get_cmap(colormap)

    # Normalize the values to be between 0 and 1
    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    normalized_values = norm(values)

    # Map the normalized values to the colormap
    colors = cmap(normalized_values)[:, :3]  # Extract RGB values

    # Convert colors to integer RGB values (0-255)
    rgb_colors = (colors * 255).astype(int)

    return rgb_colors
def map_values_to_colors_rgbstring(values, colormap=cmocean.cm.phase):
    # Get the colormap from matplotlib
    cmap = plt.get_cmap(colormap)

    # Normalize the values to be between 0 and 1
    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    normalized_values = norm(values)

    # Map the normalized values to the colormap
    colors = cmap(normalized_values)[:, :3]  # Extract RGB values

    # Convert colors to integer RGB values (0-255)
    rgb_colors = (colors * 255).astype(int)

    # Create the static RGB part of the RGBA string
    static_rgb = [f'rgba({r},{g},{b},' for r, g, b in rgb_colors]

    return static_rgb
def append_opacity_to_rgba(static_rgb, opacities):
    if len(static_rgb) != len(opacities):
        raise ValueError("The length of the static RGB array must match the length of the opacity array.")

    if not all(0 <= a <= 1 for a in opacities):
        raise ValueError("All opacity values must be between 0 and 1.")

    rgba_colors = [f'{rgb}{a:.1f})' for rgb, a in zip(static_rgb, opacities)]

    return rgba_colors
def create_paw_traces(df, paw_positions=['FR', 'HR', 'FL', 'HL'], phase_column='FR_SwPh', paw_colors=CareyConstants.paw_colors_sns_dict):
    traces = []
    for paw, color in paw_colors.items():
        trace = go.Scatter(
            x=df[phase_column],
            y=df[phase_column],
            mode='lines+markers',
            name=paw,
            line=dict(color=color),
            marker=dict(color=color)
        )
        traces.append(trace)
    return traces
def twinplots(df, b1, b2, b3, n1, n2, n3, colorby='phase', colormap='phase', pop='stride', DEF_SIZE=1, POP_SIZE=10, custom_colors=None, linewidth=0, opacity=1, hdatafields=None, show_grid=True, show_background=True):
    highlight_array = np.ones(len(df)) * DEF_SIZE
    highlight_array[0] = POP_SIZE
    alphas = None
    if custom_colors is not None:
        if len(custom_colors) == len(colorby):
            custom_colors.append('rgba(1,1,1,0.1)')
            alphas = np.ones(len(custom_colors))
            alphas[-1] = 0.1
        elif len(custom_colors) < len(colorby):
            raise ValueError("Length of custom_colors must be at least as long as colorby")

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot-left', style={'height': '90vh'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            dcc.Graph(id='scatter-plot-right', style={'height': '90vh'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ])

    def get_color_codes(df, colorby, custom_colors):
        color_code = np.zeros((len(df)), dtype='<U7')
        color_code[:] = '0'
        idx, vals = np.where(np.array(df[colorby].values))
        color_code[idx] = np.array(colorby)[vals]
        return color_code.astype(str)

    if isinstance(colorby, list):
        color_discrete_map = {(colorby + ['0'])[i]: str(custom_colors[i]) for i in range(len(custom_colors))}
        color_code = get_color_codes(df, colorby, custom_colors)
        fig_left = px.scatter_3d(df, x=b1, y=b2, z=b3, size=highlight_array, color=color_code, size_max=POP_SIZE, color_discrete_map=color_discrete_map, hover_data=hdatafields)
        fig_right = px.scatter_3d(df, x=n1, y=n2, z=n3, size=highlight_array, color=color_code, size_max=POP_SIZE, color_discrete_map=color_discrete_map, hover_data=hdatafields)
    elif isinstance(colorby, str):
        fig_left = px.scatter_3d(df, x=b1, y=b2, z=b3, size=highlight_array, color=colorby, size_max=POP_SIZE, color_continuous_scale=colormap, hover_data=hdatafields)
        fig_right = px.scatter_3d(df, x=n1, y=n2, z=n3, size=highlight_array, color=colorby, size_max=POP_SIZE, color_continuous_scale=colormap, hover_data=hdatafields)
    else:
        fig_left = px.scatter_3d(df, x=b1, y=b2, z=b3, size=highlight_array, size_max=POP_SIZE, hover_data=hdatafields)
        fig_right = px.scatter_3d(df, x=n1, y=n2, z=n3, size=highlight_array, size_max=POP_SIZE)

    fig_left.update_traces(marker=dict(size=highlight_array, line=dict(width=linewidth), opacity=opacity))
    fig_right.update_traces(marker=dict(size=highlight_array, line=dict(width=linewidth), opacity=opacity))

    scene_update = dict(
        xaxis=dict(showbackground=show_background, showgrid=show_grid, showline=True, zeroline=True, linecolor='black', linewidth=1),
        yaxis=dict(showbackground=show_background, showgrid=show_grid, showline=True, zeroline=True, linecolor='black', linewidth=1),
        zaxis=dict(showbackground=show_background, showgrid=show_grid, showline=True, zeroline=True, linecolor='black', linewidth=1)
    )

    fig_left.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=scene_update)
    fig_right.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=scene_update)

    @app.callback(
        Output('scatter-plot-left', 'figure'),
        Output('scatter-plot-right', 'figure'),
        Input('scatter-plot-left', 'clickData'),
        Input('scatter-plot-right', 'clickData'),
        State('scatter-plot-left', 'relayoutData'),
        State('scatter-plot-right', 'relayoutData')
    )
    def update_plots(clickDataLeft, clickDataRight, relayoutDataLeft, relayoutDataRight):
        nonlocal highlight_array

        plot_clicked = ctx.triggered_id
        if plot_clicked is None:
            return fig_left, fig_right

        camera_left = relayoutDataLeft.get('scene.camera') if relayoutDataLeft else None
        camera_right = relayoutDataRight.get('scene.camera') if relayoutDataRight else None

        if plot_clicked == 'scatter-plot-left':
            selected_index = clickDataLeft['points'][0]['pointNumber']
        else:
            selected_index = clickDataRight['points'][0]['pointNumber']

        selected_value = df[pop].iloc[selected_index]
        pop_indices = np.where(df[pop].values == selected_value)
        highlight_array[:] = DEF_SIZE
        highlight_array[pop_indices] = POP_SIZE

        fig_left.update_traces(marker=dict(size=highlight_array, line=dict(width=linewidth), opacity=opacity))
        fig_right.update_traces(marker=dict(size=highlight_array, line=dict(width=linewidth), opacity=opacity))

        if camera_left:
            fig_left.update_layout(scene_camera=camera_left)
        if camera_right:
            fig_right.update_layout(scene_camera=camera_right)

        return fig_left, fig_right

    app.run_server(debug=True)

    return app
def adjust_font_size(ax=plt.gca(), increment=0):
    """
    Adjust all font sizes in the current axes by a specified increment.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to modify.
    increment (int): The number to add to the current font sizes.
    """
    # Get the current font sizes
    title_size = ax.title.get_size()
    label_size = ax.xaxis.label.get_size()
    tick_size = ax.xaxis.get_ticklabels()[
        0].get_size() if ax.xaxis.get_ticklabels() else 10  # default tick size if no ticks
    legend = ax.get_legend()
    legend_size = legend.get_texts()[0].get_size() if legend else 10  # default legend size if no legend

    # Adjust font sizes by the specified increment
    ax.title.set_size(title_size + increment)
    ax.xaxis.label.set_size(label_size + increment)
    ax.yaxis.label.set_size(label_size + increment)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(tick_size + increment)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(tick_size + increment)
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(legend_size + increment)
# def animate_multicoloredline_3d(df, x, y, z, colorby, trials=None, cmap=plt.cm.viridis, lw=1, alpha=1.0, frames=360, interval=20, output=None):
#     fig, ax = multicoloredline_3d(df, x, y, z, colorby, trials, cmap, lw, alpha)
#
#     def rotate(angle):
#         ax.view_init(elev=30, azim=angle)
#         print(f"Animating frame {angle:.1f}")
#         return ax
#
#     anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 360/frames), interval=interval)
#
#     if output:
#         anim.save(output, writer=PillowWriter())
#         print(f"Animation saved as {output}")
#
#     plt.show()
#     return fig, ax
# def dummyf():
#     pass