'''
script to analyze and create the figures for fens 2024 poster
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import CareyPlots

CONFIDENCE_THRESHOLD_FOR_ANALYSIS = 3

batch_5_loc = r'X:\data\2022\BATCH5'
if os.getlogin() == 'diogo':
    batch_5_loc = r'X:\data\2022\BATCH5'
dataset_folder = os.path.join(batch_5_loc, 'recordings\VIV_23058\VIV_23058_S10_g1\kilosort4_catgt_tshift')

c4_results = os.path.join(batch_5_loc, dataset_folder, 'c4_results')
c4_output_files = ['cluster_confidence_ratio.tsv', 'cluster_model_votes.tsv',
                   'cluster_pred_probability.tsv', 'cluster_predicted_cell_type.tsv']
dfs = [pd.read_csv(os.path.join(c4_results, ff), sep='\t') for ff in c4_output_files]
c4  = pd.concat(dfs).groupby('cluster_id').first().reset_index()
cell_types = ['PkC_ss', 'MFB', 'GoC', 'MLI', 'PkC_cs']
cluster_info = pd.read_csv(os.path.join(dataset_folder, 'cluster_info.tsv'), delimiter='\t')

##
total_number_of_cells = c4['predicted_cell_type'].shape[0]
threshold_depth         = 900 # 900
threshold_confidence    = 0 #3

cortex_only = c4
indices_to_drop = []
for ii in range(total_number_of_cells):
    if cluster_info[cluster_info['cluster_id']==c4['cluster_id'][ii]].depth.values[0] < threshold_depth or \
            c4.confidence_ratio[ii] < threshold_confidence:
        indices_to_drop.append(ii)
cortex_only = cortex_only.drop(indices_to_drop)
cortex_only['predicted_cell_type'] = pd.Categorical(cortex_only['predicted_cell_type'], cell_types)

# ----------------------  histogram  --------------------------------------
plt.figure()
cell_types_barplot = sns.histplot(data=cortex_only, x='predicted_cell_type', hue='predicted_cell_type',
                                  palette=[sns.color_palette('bright')[0], sns.color_palette('bright')[2],
                                           sns.color_palette('bright')[7], sns.color_palette('bright')[7],
                                           sns.color_palette('bright')[7]],
                                  shrink=0.9, legend=False)
plt.xlabel('Cell types')
plt.show()

## filter units by cell type and confidence

mossy_fibers =      cortex_only[np.logical_and(cortex_only.predicted_cell_type=='MFB',
                           cortex_only.confidence_ratio>CONFIDENCE_THRESHOLD_FOR_ANALYSIS)].cluster_id.values
purkinje_cells =    cortex_only[np.logical_and(cortex_only.predicted_cell_type=='PkC_ss',
                           cortex_only.confidence_ratio>CONFIDENCE_THRESHOLD_FOR_ANALYSIS)].cluster_id.values

