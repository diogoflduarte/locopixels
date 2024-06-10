import CareyPlots
import pandas as pd
import importlib
import sklearn
import numpy as np

df = pd.read_csv(r"X:\data\2022\BATCH5\processing\VIV_23058\S10\locopixels\session_neural_behavior.csv")
df = df[np.logical_and(df['FR_SwStrides'] > 9000, df['FR_SwStrides'] < 9500)] # 10500

behav_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(df[['FRx', 'HRx', 'FLx', 'HLx']].interpolate())
neural_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(df[['317', '322', '351', '354', '367', '393',
                                                                         '421', '542', '453', '457', '467', '479',
                                                                         '500', '501', '503', '507', '601', '602',
                                                                         '604', '611', '613']].interpolate())
df[['bPC1', 'bPC2', 'bPC3']] = behav_pca
df[['nPC1', 'nPC2', 'nPC3']] = neural_pca
app = CareyPlots.twinplots(df[:5000], 'bPC1', 'bPC2', 'bPC3', 'nPC1', 'nPC2', 'nPC3', colorby='FR_SwPh',
                                        pop='FR_SwStrides', DEF_SIZE=1, POP_SIZE=20, linewidth=0.1, opacity=0.5)

