import numpy as np
import os
import pandas as pd

def load_c4(c4dir):
    conf_ratio = pd.read_csv(os.path.join(c4dir, 'cluster_confidence_ratio.tsv'), sep='\t')
    predicted_celltype = pd.read_csv(os.path.join(c4dir, 'cluster_predicted_cell_type.tsv'), sep='\t')
    pred_prob = pd.read_csv(os.path.join(c4dir, 'cluster_pred_probability.tsv'), sep='\t')
    cell_types = pd.concat([predicted_celltype, pred_prob['pred_probability'], conf_ratio['confidence_ratio']], axis=1)
    return cell_types

def load_ks(ksdir):
    cluster_info = pd.read_csv(os.path.join(ksdir, 'cluster_info.tsv'), delimiter='\t')
    spike_clusters = np.load(os.path.join(ksdir, 'spike_clusters.npy'))
    spike_trains = np.load(os.path.join(ksdir, 'spike_times.npy'))
    amplitudes = np.load(os.path.join(ksdir, 'amplitudes.npy'))
    return cluster_info, spike_clusters, spike_trains, amplitudes