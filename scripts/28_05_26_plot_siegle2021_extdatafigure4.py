"""Reproduce Siegle et al. 2021 Extended Data Figure 4 from histograms.npz.

This is a sanity-check plot - the output PNG should look like Fig 4 panels
(line histograms of 12 QC metrics across cortex/thalamus/hippocampus/midbrain).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / 'data' / 'extdatafigure4' / 'histograms.npz'
OUT = ROOT / 'data' / 'extdatafigure4' / 'fig4_reproduction.png'

METRICS = [
    'firing_rate', 'presence_ratio', 'max_drift', 'waveform_amplitude',
    'waveform_spread', 'waveform_duration', 'isi_violations', 'snr',
    'isolation_distance', 'd_prime', 'amplitude_cutoff', 'nn_hit_rate',
]
LABELS = {
    'firing_rate': 'Overall firing rate (Hz)',
    'presence_ratio': 'Presence ratio',
    'max_drift': 'Maximum drift (um)',
    'waveform_amplitude': 'Waveform amplitude (uV)',
    'waveform_spread': 'Waveform spread (um)',
    'waveform_duration': 'Waveform duration (ms)',
    'isi_violations': 'ISI violations',
    'snr': 'SNR',
    'isolation_distance': 'Isolation distance',
    'd_prime': "d'",
    'amplitude_cutoff': 'Amplitude cutoff',
    'nn_hit_rate': 'Nearest-neighbors hit rate',
}
COLORS = ['#08858C', '#FC6B6F', '#7ED04B', '#FC9DFE']

d = np.load(NPZ, allow_pickle=True)
area_names = list(d['area_names'])
unit_counts = d['unit_counts']

fig, axes = plt.subplots(4, 3, figsize=(13, 12))
axes = axes.ravel()

for i, metric in enumerate(METRICS):
    ax = axes[i]
    bins = d[f'bin_edges_{metric}']
    hist = d[f'hist_{metric}']
    x = bins[:-1]
    for ai, area in enumerate(area_names):
        y = gaussian_filter1d(hist[ai], 1)
        ax.plot(x, y, color=COLORS[ai], lw=2.0, label=area)
    ax.set_xlabel(LABELS[metric])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    if metric == 'firing_rate':
        ax.set_xticks([-2, -1, 0, 1])
        ax.set_xticklabels(['0.01', '0.1', '1', '10'])
    elif metric == 'isi_violations':
        ax.set_xticks([-4.632, -3, -1, 1])
        ax.set_xticklabels(['0', '0.001', '0.1', '10'])

handles = [plt.Line2D([0], [0], color=COLORS[i], lw=2,
                       label=f'{area_names[i]} (N = {unit_counts[i]:,})')
           for i in range(len(area_names))]
axes[-1].legend(handles=handles, loc='center', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches='tight')
print(f'wrote {OUT}')
