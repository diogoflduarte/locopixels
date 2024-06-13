import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

os.chdir(r'C:\Users\User\Desktop\neural data analysis')
from utils import sort_by


def plot_peth(data, peth=None, peth_std=None, xlabel=None, ylabel=[None, None], xticks=None, yticks=None, yticklabels=None, xticklabels=None, ax=None, sort_var=None, percent=None, peth_color=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    
    if sort_var is not None:
        data_sorted = np.array(sort_by(data, sort_var)[0])
        sns.heatmap(data_sorted, cmap='viridis', cbar=False, ax=ax[0])
        bin_edges = np.percentile(sort_var, percent)
        bins = np.digitize(sort_var, bin_edges)-1
        num_bins = len(bin_edges)-1
        if peth_color is None or peth_color == 1:
            peth_color = sns.color_palette("Reds", num_bins)
    else:
        sns.heatmap(data, cmap='viridis', cbar=False, ax=ax[0])
    
    if peth is not None and peth_std is not None:
        if len(peth) > 1:
            for i in range(len(peth)):
                ax[1].plot(peth[i], color=peth_color, **kwargs)
                # ax[1].fill_between(range(len(peth)), peth[i] - peth_std[i], peth[i] + peth_std[i], color=peth_color, alpha=0.3)
        else:
            ax[1].plot(peth, **kwargs)
            # ax[1].fill_between(range(len(peth)), peth - peth_std, peth + peth_std, alpha=0.3, **kwargs)
    
    elif sort_var is not None:
        for i in range(num_bins):
            peth = np.mean(data[bins==i], axis=0)
            peth_std = np.std(data[bins==i], axis=0)
            ax[1].plot(peth, color=peth_color[i], **kwargs)
            # ax[1].fill_between(range(len(peth)), peth - peth_std, peth + peth_std, alpha=0.3, color=peth_color[i], **kwargs)            
    
    else:
        peth = np.mean(data, axis=0)
        # peth = np.std(data, axis=0)
        ax[1].plot(peth, **kwargs)
    
    if xlabel:
        ax[0].set_xlabel(xlabel, fontsize=15)
        ax[1].set_xlabel(xlabel, fontsize=15)
    else:
        ax[0].set_xlabel('Time (s)', fontsize=15)
        ax[1].set_xlabel('Time (s)', fontsize=15)
    if ylabel[0]:
        ax[0].set_ylabel(ylabel[0], fontsize=15)
    else:
        ax[0].set_ylabel('Trials', fontsize=15)
    if ylabel[1]:
        ax[1].set_ylabel(ylabel[1], fontsize=15)
    else:
        ax[1].set_ylabel('Firing rate (Hz)', fontsize=15) 
    ax[1].margins(0)
    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    if xticks:
        ax[0].set_xticks(xticks)
        ax[0].set_xticklabels(xticklabels, fontsize=15)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels(xticklabels, fontsize=15)
    if yticks:
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(yticklabels, fontsize=15, rotation=90)
    else:
        ax[0].set_yticks([0, len(data)])
        ax[0].set_yticklabels([1, len(data)], fontsize=15, rotation=90)   
        
    return ax


def plot_peth_popul(data, xlabel=None, ylabel=None, xticks=None, yticks=None, yticklabels=None, xticklabels=None, ax=None, sort_var=None, cbar_label=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots()
    
    if sort_var is not None:
        data = np.array(sort_by(data, sort_var)[0])
        
    hm = sns.heatmap(data, cmap='viridis', ax=ax)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    else:
        ax.set_xlabel('Time (s)', fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel[0], fontsize=15)
    else:
        ax.set_ylabel('Neurons', fontsize=15)
    ax.tick_params(labelsize=12)
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=15)
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=15, rotation=90)
    else:
        ax.set_yticks([0, len(data)])
        ax.set_yticklabels([1, len(data)], fontsize=15, rotation=90)   
    
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  
    if cbar_label:
        cbar.set_label('Label', fontsize=15) 
        
    return ax