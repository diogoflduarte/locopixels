import CareyUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

t1 = r"X:\data\2022\BATCH5\behavior\VIV_23058\S10\VIV_23058_BodyCamVideo2022-03-24_S10_1DLC_resnet50_npx_cage_bodycamFeb8shuffle1_500000.h5"
df = pd.read_hdf(t1)
badpoints = df.iloc[:,5] < np.quantile(df.iloc[:,5], 0.005)
signal = df.iloc[:,3].values
masked_signal = ma.asarray(signal)
masked_signal[badpoints.values] = ma.masked

x, kf = CareyUtils.kalman_smooth(masked_signal, dt=1/432, tCov=1.0, obsCov=1.0)

plt.figure(figsize=(19, 4.8))
plt.plot(signal)
plt.plot(x)
plt.show()