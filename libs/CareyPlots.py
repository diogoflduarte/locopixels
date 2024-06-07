import numpy as np
from matplotlib import pyplot as plt
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
from dash import dcc, html, Input, Output, ctx, callback
import plotly.express as px
import pandas as pd

def multiLinePlot(data, colors, linewidth=1, alpha=0.5, x=None):
    plt.figure()
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

def multicoloredline_2d(df, x, y, colorby, trials=None, cmap=sns.cm.crest_r, lw=1):
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

    plt.figure()
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
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(var1)
        lc.set_linewidth(lw)

        plt.gca().add_collection(lc)
        plt.xlim(pc1.min(), pc1.max())
        plt.ylim(pc2.min(), pc2.max())

    # Plotting
    # plt.figure(figsize=(10, 6))

    # Create a colorbar
    cbar = plt.colorbar(lc)
    # cbar.set_label(colorby)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
def multicoloredline_3d(df, x, y, z, colorby, trials=None, cmap=plt.cm.viridis, lw=1):
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
def twinplots():
    DEF_SIZE = 0.1
    highlight_array = np.ones((5))*DEF_SIZE
    # Sample DataFrame
    df = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'y1': [10, 11, 12, 13, 14],
        'x2': [1, 2, 3, 4, 5],
        'y2': [14, 13, 12, 11, 10],
        'highlight': highlight_array
    })

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot-left')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='scatter-plot-right')
        ], style={'width': '48%', 'display': 'inline-block'})
    ])

    fig_left  = px.scatter(df, x='x1', y='y1', size='highlight', size_max=20)
    fig_right = px.scatter(df, x='x2', y='y2', size='highlight', size_max=20)

    @app.callback(
        Output('scatter-plot-left', 'figure'),
        Output('scatter-plot-right', 'figure'),
        Input('scatter-plot-left', 'clickData'),
        Input('scatter-plot-right', 'clickData'),
    )
    def update_plots(clickDataLeft, clickDataRight):

        plot_clicked = ctx.triggered_id
        if plot_clicked is None:
            return fig_left, fig_right

        if plot_clicked == 'scatter-plot-left':
            clicked = 'L'
            selected_index = clickDataLeft['points'][0]['pointIndex']
        else:
            selected_index = clickDataRight['points'][0]['pointIndex']
            clicked = 'R'
        print(clicked + ' ' + str(selected_index))

        # if selected_index is not None:
        highlight_array[:] = DEF_SIZE
        highlight_array[selected_index] = 2
        fig_left.data[0].marker.size  = highlight_array
        fig_right.data[0].marker.size = highlight_array

        clickDataLeft = None
        clickDataRight = None

        return fig_left, fig_right

    app.run_server(debug=True)

    return app
