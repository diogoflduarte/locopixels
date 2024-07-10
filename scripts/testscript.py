import CareyPlots
import pandas as pd
import importlib
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri
from matplotlib.colors import LinearSegmentedColormap
from CareyConstants import CareyConstants
from tqdm import tqdm
import scipy
import ssm

##
df = pd.read_csv(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\session_neural_behavior.csv")
df = df[np.logical_and(df['FR_SwStrides'] > 9000, df['FR_SwStrides'] < 9500)] # 10500
feat = ['FRx', 'HRx', 'FLx', 'HLx']

# for ff in tqdm(feat):
#     df[ff] = scipy.signal.medfilt(df[ff], 7) #11

behav_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(df[['FRx', 'HRx', 'FLx', 'HLx']].interpolate())
neural_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(df[['317', '322', '351', '354', '367', '393',
                                                                         '421', '542', '453', '457', '467', '479',
                                                                         '500', '501', '503', '507', '601', '602',
                                                                         '604', '611', '613']].interpolate())
df[['bPC1', 'bPC2', 'bPC3']] = behav_pca
df[['nPC1', 'nPC2', 'nPC3']] = neural_pca

##
# app = CareyPlots.twinplots(df[:5000], 'bPC1', 'bPC2', 'bPC3', 'nPC1', 'nPC2', 'nPC3', colorby='FR_SwPh',
#                                         pop='FR_SwStrides', DEF_SIZE=1, POP_SIZE=20, linewidth=0.1, opacity=0.5,
#                                         hdatafields=['FR_SwStrides', 'sessionwise_time'])
# app = CareyPlots.twinplots_with_paw_tracks(df[:5000], 'bPC1', 'bPC2', 'bPC3', 'phase', ['FRx', 'HRx', 'FLx', 'HLx'],
#                                             colorby='FR_SwPh', pop='stride', hdatafields=['FR_SwStrides', 'sessionwise_time'])

##

# ##
# minidf = df[:5000]
# strides = np.unique(minidf.FR_SwStrides)
# ll_FR = []
# ll_HR = []
# ll_FL = []
# ll_FL = []
# plt.figure()
# for ii, stride in enumerate(strides):
#     ll_FR = sns.lineplot(data=minidf[minidf['FR_SwStrides']==stride], x='FR_SwPh', y='FRx', color=CareyConstants.paw_colors_sns[0])
#     ll_HR = sns.lineplot(data=minidf[minidf['FR_SwStrides']==stride], x='FR_SwPh', y='HRx', color=CareyConstants.paw_colors_sns[1])
#     ll_FL = sns.lineplot(data=minidf[minidf['FR_SwStrides']==stride], x='FR_SwPh', y='FLx', color=CareyConstants.paw_colors_sns[2])
#     ll_HL = sns.lineplot(data=minidf[minidf['FR_SwStrides']==stride], x='FR_SwPh', y='HLx', color=CareyConstants.paw_colors_sns[3])
#
# ##
# behav = pd.read_csv(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.csv")
# feat = ['FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz']
# y = behav[feat].interpolate().values
# scaler = sklearn.preprocessing.StandardScaler()
# y = scaler.fit_transform(y)
#
# N = len(feat)
# D = 3
# lds = ssm.LDS(N, D)
# lds.initialize(y, verbose=2)
#
# q_mf_elbos, q_mf = lds.fit(y,
#                            method="bbvi",
#                            variational_posterior="mf",
#                            num_iters=1000, stepsize=0.1,
#                            initialize=False)
# q_mf_x = q_mf.mean[0]
# q_mf_y = lds.smooth(q_mf_x, y)
#
# # df[['bLT1', 'bLT2', 'bLT3']] = q_mf_x
# np.save(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\full_behav_lds_3dim.npy", q_mf_x)
# np.save(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\full_behav_lds_smoothed.npy", q_mf_y)
# ##
# app = CareyPlots.twinplots(df[:5000], 'bPC1', 'bPC2', 'bPC3', 'bLT1', 'bLT2', 'bLT3', colorby='FR_SwPh',
#                                         pop='FR_SwStrides', DEF_SIZE=1, POP_SIZE=20, linewidth=0.1, opacity=0.5,
#                                         hdatafields=['FR_SwStrides', 'sessionwise_time'])

behav_file = r"X:\data\2022\BATCH5\behavior\VIV_23058\S10\VIV_23058_BodyCamVideo2022-03-24_S10_1DLC_resnet50_npx_cage_bodycamFeb8shuffle1_500000.h5"
behav = pd.read_hdf(behav_file)

behav[('DLC_resnet50_npx_cage_bodycamFeb8shuffle1_500000', 'time', 'time')] = np.linspace(0, 25972 / 432.0, behav.shape[0])

FRx = behav.columns[3]
HRx = behav.columns[9]
FLx = behav.columns[6]
HLx = behav.columns[12]

FRx_lik = behav.columns[5]
HRx_lik = behav.columns[11]
FLx_lik = behav.columns[8]
HLx_lik = behav.columns[14]

timex = ('DLC_resnet50_npx_cage_bodycamFeb8shuffle1_500000', 'time', 'time')

plt.figure()
badpoints = behav[FRx_lik] < 0.8
sns.lineplot(data=behav, x=timex, y=FRx, linewidth=1, color='r')
sns.scatterplot(data=behav[badpoints], x=timex, y=FRx)

# plt.figure()
# badpoints = behav[FRx_lik] < 0.65
# sns.lineplot(data=behav, x=timex, y=FRx, linewidth=1, color='r')
# sns.scatterplot(data=behav[badpoints], x=timex, y=FRx)
##
mycmap = plt.cm.colors.ListedColormap(cmcrameri.cm.vik(np.linspace(0, 1, 256)) ** 0.5)
CareyPlots.multicoloredline_2d(behav, timex, FRx, colorby=FRx_lik, lw=5, cmap=mycmap)
##
plt.figure()
badpoints = behav[FRx_lik] < np.quantile(behav[FRx_lik], 0.05)
sns.lineplot(data=behav, x=timex, y=FRx, linewidth=1, color='r')
sns.scatterplot(data=behav[badpoints], x=timex, y=FRx, color='k')
##
plt.figure()
badpoints = behav[HRx_lik] < np.quantile(behav[HRx_lik], 0.05)
sns.lineplot(data=behav, x=timex, y=HRx, linewidth=1, color='magenta')
sns.scatterplot(data=behav[badpoints], x=timex, y=HRx, color='k')
HRx_filt = HRx
HRx_filt = HRx_filt[2] + '_filt'
behav[HRx_filt] = behav[HRx]
behav.loc[badpoints, HRx_filt] = np.nan
behav[HRx_filt] = behav[HRx_filt].interpolate(method='polynomial', order=3)
sns.scatterplot(data=behav[badpoints], x=timex, y=HRx_filt, color='magenta')
##
plt.figure()
badpoints = behav[FLx_lik] < np.quantile(behav[FLx_lik], 0.02)
sns.lineplot(data=behav, x=timex, y=FLx, linewidth=1, color='blue')
sns.scatterplot(data=behav[badpoints], x=timex, y=FLx, color='k')
FLx_filt = FLx
FLx_filt[2] + '_filt'
behav[FLx_filt][badpoints] = np.nan
behav[FLx_filt].interpolate()
sns.scatterplot(data=behav[badpoints], x=timex, y=FLx_filt, color='blue')
##
plt.figure()
badpoints = behav[HLx_lik] < np.quantile(behav[HLx_lik], 0.05)
sns.lineplot(data=behav, x=timex, y=HLx, linewidth=1, color='cyan')
sns.scatterplot(data=behav[badpoints], x=timex, y=HLx, color='k')
##
import matplotlib.pyplot as plt
import numpy as np

# # Sample data
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
#
# # Sample confidence values between 0 and 1
# confidence = np.abs(np.cos(x))  # example confidence values
#
# # Create a new figure
# plt.figure(figsize=(10, 6))
#
# # Plot each segment of the line with varying alpha
# for i in range(len(x) - 1):
#     plt.plot(x[i:i+2], y[i:i+2], color='blue', alpha=confidence[i])
#
# # Add labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line Plot with Transparency Encoding Confidence')
#
# # Show the plot
# plt.show()
##
behav = pd.read_hdf(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\VIV_23058_S10_behavioral_descriptor.h5")

##
plt.figure()
badpoints = behav[HRx_lik] < np.quantile(behav[HRx_lik], 0.05)
sns.lineplot(data=behav, x=timex, y=HRx, linewidth=1, color='magenta')
sns.scatterplot(data=behav[badpoints], x=timex, y=HRx, color='k')
HRx_filt = HRx
HRx_filt = HRx_filt[2] + '_filt'
behav[HRx_filt] = behav[HRx]
behav.loc[badpoints, HRx_filt] = np.nan
behav[HRx_filt] = behav[HRx_filt].interpolate(method='polynomial', order=3)
sns.scatterplot(data=behav[badpoints], x=timex, y=HRx_filt, color='magenta')