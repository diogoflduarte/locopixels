import numpy as np
import os

os.chdir(r'C:\Users\User\Desktop\behavior_analysis')
from util import findpeaks, interpolate, detrend
import phaser

def find_phase(k):
    y = np.array(k)
    y = detrend(y.T).T
    phsr = phaser.Phaser(y=y)
    k[:] = phsr.phaserEval(y)[0,:]
    return k

def get_cycles(principal_components, global_phase, sr, peak_threshold=6, 
                       trough_threshold=0.5, interp_samples=200):  
    # Detect onset and offset of each cycle
    offset, onset = findpeaks(global_phase, peak_threshold=peak_threshold, trough_threshold=trough_threshold)
    # Interpolate cycles 
    cycles = []
    for i in range(len(offset)):
        cycles.append(principal_components[onset[i]:offset[i]+1, :])
    interp_cycles = interpolate(cycles, n_samples=interp_samples)
    # Compute cycle duration
    cycle_duration = np.diff(onset) * (1 / sr)
    return interp_cycles, cycles, cycle_duration, [onset, offset]