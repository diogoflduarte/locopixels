import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import npyx
import pandas as pd

import seaborn as sns

batch_5_loc = r'X:\data\2022\BATCH5'
if os.getlogin() == 'diogo':
    batch_5_loc = r'X:\data\2022\BATCH5'
dataset_folder = os.path.join(batch_5_loc, 'recordings', 'VIV_23058', 'VIV_23058_S10_g1', 'kilosort4_catgt_tshift')

cluster_info_file = os.path.join(dataset_folder, 'cluster_info.tsv')
cluster_info = pd.read_csv(cluster_info_file, delimiter='\t')

sbdepth = cluster_info.groupby('depth').sum()
sum_by_depth = cluster_info.groupby('depth').sum().reset_index()
sns.lineplot(data=sum_by_depth, x='depth', y='Amplitude'), plt.show()

plt.figure(), sns.barplot(data=sum_by_depth, y='Amplitude', x='depth'), plt.show()
plt.figure(), sns.barplot(data=cluster_info, y='Amplitude', x='depth', estimator='sum'), plt.show()

bin_width = 100
max_depth = cluster_info['depth'].max()
bins = range(int(cluster_info['depth'].min()), int(cluster_info['depth'].max()) + bin_width, bin_width)
cluster_info['depth_binned'] = pd.cut(cluster_info['depth'], bins=bins)
grouped_data = cluster_info.groupby('depth_binned')['Amplitude'].sum().reset_index()
grouped_data['inverted_depth_binned'] = grouped_data['depth_binned'].apply(
    lambda x: f"{int(max_depth - x.left)}-{int(max_depth - x.right)}")
grouped_data['midpoint_depth'] = grouped_data['depth_binned'].apply(lambda x: (x.left + x.right) / 2)
plt.figure(), sns.barplot(data=grouped_data, y='midpoint_depth', x='Amplitude', width=1, ), plt.gca().invert_yaxis(), plt.show()