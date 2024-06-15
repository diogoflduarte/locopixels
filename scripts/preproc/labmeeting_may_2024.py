import os
import numpy as np
import npyx
import CareyEphys
import CareyBehavior
import CareyUtils
from tqdm import tqdm
import pandas as pd
from CareyConstants import CareyConstants
import CareyPlots
import time
import cupy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
import cmocean
import scipy.signal
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cmcrameri
import ssm
from CareyUtils import phase
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'


## define dirs

datadir = r'X:\data\2022\BATCH5'
neural  =  os.path.join(datadir, r'recordings\VIV_23058\VIV_23058_S10_g1\kilosort4_catgt_tshift')
new_proc = os.path.join(datadir, r'processing\VIV_23058\S10\locopixels')

run_FA  = False
run_LDS = False

## load behav
behav = pd.read_csv(os.path.join(new_proc, '..', 'VIV_23058_S10_behavioral_descriptor.csv'))

## get continuous phase
__, behav['FR_SwPh'], behav['FR_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.FR_SwOn, behav.FR_StOn, usegpu=0)
__, behav['FR_StPh'], behav['FR_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.FR_StOn, behav.FR_SwOn, usegpu=0)
__, behav['HR_SwPh'], behav['HR_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.HR_SwOn, behav.HR_StOn, usegpu=0)
__, behav['HR_StPh'], behav['HR_St_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.HR_StOn, behav.HR_SwOn, usegpu=0)
__, behav['FL_SwPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.FL_SwOn, behav.FL_StOn, usegpu=0)
__, behav['FL_StPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.FL_StOn, behav.FL_SwOn, usegpu=0)
__, behav['HL_SwPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.HL_SwOn, behav.HL_StOn, usegpu=0)
__, behav['HL_StPh'], behav['FL_Sw_Stride'] = CareyBehavior.get_stride_phase_from_events(behav.HL_StOn, behav.HL_SwOn, usegpu=0)

# ## interpolate nans or missing values from tracking
# behav['FR_Sw_Stride#'][behav['FR_Sw_Stride#']==0] = np.nan
# behav['FR_St_Stride#'][behav['FR_St_Stride#']==0] = np.nan
# behav['HR_Sw_Stride#'][behav['HR_Sw_Stride#']==0] = np.nan
# behav['HR_St_Stride#'][behav['HR_St_Stride#']==0] = np.nan
# behav['FL_Sw_Stride#'][behav['HR_St_Stride#']==0] = np.nan
# behav['FL_Sw_Stride#'][behav['HR_St_Stride#']==0] = np.nan
# behav['FL_Sw_Stride#'][behav['HR_St_Stride#']==0] = np.nan
# behav['FL_Sw_Stride#'][behav['HR_St_Stride#']==0] = np.nan

## tracks need to be filtered
feat = ['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']
for ff in tqdm(feat):
    behav[ff] = scipy.signal.medfilt(behav[ff], 7)


## phase difference between FR and FL
behav['PhDiff'] = phase.subtract(behav['FR_SwPh'], behav['FL_SwPh'])
behav['PhDiff'] = scipy.signal.medfilt(behav.PhDiff, 93) # needs filtering
# behav_small = behav[behav['FR_Sw_Stride#']<1000]
# plt.plot(behav_small.sessionwise_time, CareyUtils.zscore(behav_small.FRx), color=CareyConstants.paw_colors_sns[0])
# plt.plot(behav_small.sessionwise_time, CareyUtils.zscore(behav_small.FLx), color=CareyConstants.paw_colors_sns[2])
# plt.plot(behav_small.sessionwise_time, behav_small.PhDiff, 'k')

# run PCA and color code by stride-average phase difference
behav_pca = PCA(n_components=3)
feat = ['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']
for f in feat:
    behav[feat] = behav[feat].interpolate()
behav_pcs = behav_pca.fit_transform(CareyUtils.zscore(behav[['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz',
                                           'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']].values))
behav['PC1'] = behav_pcs[:, 0]
behav['PC2'] = behav_pcs[:, 1]
behav['PC3'] = behav_pcs[:, 2]

behav_small_idx = np.logical_and(behav['FR_Sw_Stride'] > 9000, behav['FR_Sw_Stride'] < 10500)
behav_small = behav[behav_small_idx]
## recompute pca on small dataset
behav_pca = PCA(n_components=3)
behav_pcs = behav_pca.fit_transform(CareyUtils.zscore(behav_small[['FRx', 'HRx', 'FLx', 'HLx']].values))
behav_small['PC1'] = behav_pcs[:,0]
behav_small['PC2'] = behav_pcs[:,1]
behav_small['PC3'] = behav_pcs[:,2]
##
fig = px.scatter_3d(behav_small[:], x='PC1', y='PC2', z='PC3', color='FR_SwPh', color_continuous_scale ='phase')
fig.update_traces(marker_size = 1)
fig.update_scenes(aspectmode='cube')
fig.show()
## by wheel speed
fig = px.scatter_3d(behav_small[:], x='PC1', y='PC2', z='PC3', color='wheel_speed')
fig.update_traces(marker_size = 1)
fig.show()
## by phase difference
groups = behav_small[:].groupby('FR_Sw_Stride')
norm = plt.Normalize(behav_small['PhDiff'].min(), behav_small['PhDiff'].max())

fig, axs = plt.subplots(2, 2)
axs = axs.flatten()
for ii, paw in enumerate(CareyConstants.paw_labels):
    plt.sca(axs[CareyConstants.subplot_topview_order[ii]-1])
    cmap = CareyPlots.create_colormap(CareyConstants.paw_colors_sns[ii])
    for name, group in groups:
        avg_phdiff = group['PhDiff'].mean()
        feat = (paw + 'x')
        plt.plot(group['FR_SwPh'], group[feat], label=f'Stride {name}', color=cmap(norm(avg_phdiff)), linewidth=0.5)
        plt.xlabel('FR_SwPhase')
        plt.ylabel('forward exc. (px)')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Average PhDiff')

## same but for principal comps
groups = behav_small[:].groupby('FR_Sw_Stride')
norm = plt.Normalize(behav_small['PhDiff'].min(), behav_small['PhDiff'].max())

fig, axs = plt.subplots(2, 2)
axs = axs.flatten()
for ii, pc in enumerate(['PC1', 'PC2', 'PC3']):
    plt.sca(axs[ii])
    cmap = CareyPlots.create_colormap(np.array([1, 0.5, 0.0]))
    for name, group in groups:
        avg_phdiff = group['PhDiff'].mean()
        feat = pc
        plt.plot(group['FR_SwPh'], group[feat], label=f'Stride {name}', color=cmap(norm(avg_phdiff)), linewidth=0.5)
        plt.xlabel('FR_SwPhase')
        plt.ylabel(pc)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Average PhDiff')

axs[3] = fig.add_subplot(2,2,4, projection='3d')
plt.sca(axs[3])
for name, group in groups:
    avg_phdiff = group['PhDiff'].mean()
    plt.plot(group['PC1'], group['PC2'], group['PC3'], label=f'Stride {name}', color=cmap(norm(avg_phdiff)), linewidth=0.5)
    # ax[3].scatter(df[axis1], df[axis2], df[axis3], c=df['phase'], cmap=colormap,
    #               s=MSIZE, rasterized=True)
    axs[3].set_xlabel('PC1')
    axs[3].set_ylabel('PC2')
    axs[3].set_zlabel('PC3')
    axs[3].set_facecolor((1, 1, 1, 0))
    axs[3].grid(False)
## lines
strides = np.unique(behav_small.FR_Sw_Stride)
fig =  px.line_3d(behav_small[behav_small.FR_Sw_Stride==strides[1000]],  x='PC1', y='PC2', z='PC3')
fig.update_scenes(aspectmode='cube')
fig.show()

## let's try to smooth via LDS
if run_LDS:
    feat = ['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']
    y = behav[feat].values
    y_train = y[:-1000000,:]
    scaler = sklearn.preprocessing.StandardScaler()
    y_train = scaler.fit_transform(y_train)

    N = len(feat)
    D = 3
    lds = ssm.LDS(N, D)
    lds.initialize(y_train, verbose=2)



    plt.figure()
    plt.imshow(lds.dynamics.A)
    plt.colorbar()

    q_mf_elbos, q_mf = lds.fit(y_train,
                               method="bbvi",
                               variational_posterior="mf",
                               num_iters=10, stepsize=0.1,
                               initialize=False)
    q_mf_x = q_mf.mean[0]
    q_mf_y = lds.smooth(q_mf_x, y)

    elbo, unseen_posterior_mean = lds.approximate_posterior(y[behav_small.index.values,:])
    smoothed_y_unseen = lds.smooth(unseen_posterior_mean.mean[0][1], y[behav_small.index.values,:])
##
plt.figure()
plt.plot(CareyUtils.zscore(behav_small.FRx.values), color=CareyConstants.paw_colors_sns[0])
plt.plot(CareyUtils.zscore(smoothed_y_unseen[:,0]), '--', color=CareyConstants.paw_colors_sns[0])

# now let's plot the latents
behav_small['latent1'] = unseen_posterior_mean.mean[0][1][:, 0]
behav_small['latent2'] = unseen_posterior_mean.mean[0][1][:, 1]
behav_small['latent3'] = unseen_posterior_mean.mean[0][1][:, 2]

behav_tiny = behav_small[behav_small.FR_Sw_Stride>10000]
behav_tiny.latent1 = CareyUtils.zscore(behav_tiny.latent1)
behav_tiny.latent2 = CareyUtils.zscore(behav_tiny.latent2)
behav_tiny.latent3 = CareyUtils.zscore(behav_tiny.latent3)
CareyPlots.multicoloredline_3d(behav_tiny, 'latent1', 'latent2', 'latent3', 'FR_SwPh', lw=0.5, cmap=cmcrameri.cm.romaO)

CareyPlots.plot_coefficients()
##
if run_FA:
    fa_model = sklearn.decomposition.FactorAnalysis(n_components = 3)
    behav_factors = fa_model.fit_transform(behav_small[['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz',
                                               'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']].values)
    behav_small['FA1'] = behav_factors[:, 0]
    behav_small['FA2'] = behav_factors[:, 1]
    behav_small['FA3'] = behav_factors[:, 2]
##
strides = np.unique(behav_small.FR_Sw_Stride)
fig = px.line_3d(behav_small[behav_small.FR_Sw_Stride==strides[1000]],  x='FA1', y='FA2', z='FA3')
fig.update_scenes(aspectmode='cube')
fig.show()

##
feat = ['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']
fa_small = fa_model.transform( behav_small[feat].values )
behav_small['FA1'] = fa_small[:,0]
behav_small['FA2'] = fa_small[:,1]
behav_small['FA3'] = fa_small[:,2]

##
fig = px.scatter_3d(behav_small, x='FA1', y='FA2', z='FA3', color='FR_SwPh', color_continuous_scale ='phase')
fig.update_traces(marker_size=1)
fig.update_scenes(aspectmode='cube')
fig.show()
##
fig = px.scatter_3d(behav_small, x='FA1', y='FA2', z='FA3', color='wheel_speed', color_continuous_scale ='darkmint')
fig.update_traces(marker_size = 1)
fig.update_scenes(aspectmode='cube')
fig.show()


## PCA VS FA

plt.plot(behav_small['sessionwise_time'], CareyUtils.zscore(behav_small['PC1'])+10, color=sns.color_palette('deep')[3])
plt.plot(behav_small['sessionwise_time'], CareyUtils.zscore(behav_small['FA1'])+10, '--', color=sns.color_palette('deep')[3])
plt.plot(behav_small['sessionwise_time'], CareyUtils.zscore(behav_small['PC2'])+5, color=sns.color_palette('deep')[6])
plt.plot(behav_small['sessionwise_time'], CareyUtils.zscore(behav_small['FA2'])+5, '--', color=sns.color_palette('deep')[6])
plt.plot(behav_small['sessionwise_time'], CareyUtils.zscore(behav_small['PC3']), color=sns.color_palette('deep')[1])
plt.plot(behav_small['sessionwise_time'], CareyUtils.zscore(behav_small['FA3']), '--', color=sns.color_palette('deep')[1])

##
onestride = behav_small[behav_small.FR_Sw_Stride==10000]
CareyPlots.multicoloredline_2d(onestride, 'FA1', 'FA2', 'FR_SwPh')