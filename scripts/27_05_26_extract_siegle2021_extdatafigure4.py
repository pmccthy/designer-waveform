"""Reproduce the data underlying Siegle et al. 2021, Extended Data Figure 4.

This mirrors the data path of
https://github.com/AllenInstitute/neuropixels_platform_paper/blob/master/ExtDataFigure4/ExtDataFigure4.py
but reads from the locally-cached `data/unit_analysis_metrics.pkl` (a
dict of {session_id: {unit_id: {col: val}}} produced by the
`06_05_26_extract_unit_analysis_metrics.ipynb` notebook), so no AllenSDK
download is required.

Outputs (written next to this script under ../data/extdatafigure4/):
    - units_long.csv            One row per unit, with all 12 metrics +
                                quality + ecephys_structure_acronym +
                                area_group ('cortex'/'thalamus'/...).
    - histograms.npz            Per (metric, area_group) density histograms
                                using the *exact* bin edges from the published
                                figure code. Arrays:
                                  bin_edges_<metric>  shape (nbins+1,)
                                  hist_<metric>       shape (n_area_groups, nbins)
                                  area_names          shape (n_area_groups,)
                                  unit_counts         shape (n_area_groups,)
                                Pre/post-bar adjustments for `isi_violations`
                                and `amplitude_cutoff` are NOT applied to the
                                raw histograms - they are a plotting trick. The
                                CSV has the raw values so you can re-bin freely.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration - copied verbatim from ExtDataFigure4.py
# ---------------------------------------------------------------------------

AREA_LIST = [
    ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal', 'VISmma', 'VISmmp', 'VISli'],
    ['LGd', 'LD', 'LP', 'VPM', 'TH', 'MGm', 'MGv', 'MGd', 'PO', 'LGv', 'VL',
     'VPL', 'POL', 'Eth', 'PoT', 'PP', 'PIL', 'IntG', 'IGL', 'SGN', 'VPL', 'PF', 'RT'],
    ['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POST', 'PRE', 'ProS', 'HPF'],
    ['MB', 'SCig', 'SCiw', 'SCsg', 'SCzo', 'PPT', 'APN', 'NOT', 'MRN', 'OP', 'LT', 'RPF', 'CP'],
]
AREA_NAMES = ['cortex', 'thalamus', 'hippocampus', 'midbrain']

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

BINS = {
    'firing_rate':        np.linspace(-3, 2, 100),
    'presence_ratio':     np.linspace(0, 1, 50),
    'max_drift':          np.linspace(0, 120, 100),
    'waveform_amplitude': np.linspace(0, 500, 100),
    'waveform_spread':    np.linspace(0, 200, 50),
    'waveform_duration':  np.linspace(0, 1.15, 80),
    'isi_violations':     np.linspace(-5, 3, 100),
    'snr':                np.linspace(0, 8, 100),
    'isolation_distance': np.linspace(0, 200, 100),
    'd_prime':            np.linspace(0, 12, 80),
    'amplitude_cutoff':   np.linspace(0, 0.5, 100),
    'nn_hit_rate':        np.linspace(0.01, 1, 100),
}

USE_LOG = {
    'firing_rate': True, 'isi_violations': True,
    'presence_ratio': False, 'max_drift': False, 'waveform_amplitude': False,
    'waveform_spread': False, 'waveform_duration': False, 'snr': False,
    'isolation_distance': False, 'd_prime': False, 'amplitude_cutoff': False,
    'nn_hit_rate': False,
}

# Plotting jitters in the published code - applied to match Fig 4 exactly.
# Set rng seed so re-runs are reproducible.
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / 'data' / 'unit_analysis_metrics.pkl'
OUT_DIR = ROOT / 'data' / 'extdatafigure4'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: flatten pkl -> long DataFrame
# ---------------------------------------------------------------------------

def load_units() -> pd.DataFrame:
    with open(PKL_PATH, 'rb') as f:
        sessions = pickle.load(f)

    frames = []
    for session_id, unit_dict in sessions.items():
        df = pd.DataFrame.from_dict(unit_dict, orient='index')
        df.index.name = 'unit_id'
        df = df.reset_index()
        df['session_id'] = session_id
        frames.append(df)

    units = pd.concat(frames, ignore_index=True)

    # Assign each unit to one of the four area groups (NaN if not in any).
    area_lookup = {acro: name
                   for areas, name in zip(AREA_LIST, AREA_NAMES)
                   for acro in areas}
    units['area_group'] = units['ecephys_structure_acronym'].map(area_lookup)

    return units


# ---------------------------------------------------------------------------
# Step 2: build per-area histograms matching the published bin edges
# ---------------------------------------------------------------------------

def build_histograms(units: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    out = {'area_names': np.array(AREA_NAMES)}

    unit_counts = np.zeros(len(AREA_NAMES), dtype=int)

    for metric in METRICS:
        bins = BINS[metric]
        hist = np.zeros((len(AREA_NAMES), len(bins) - 1))

        for ai, area in enumerate(AREA_NAMES):
            sel = (units['area_group'] == area) & (units['quality'] == 'good')
            D = units.loc[sel, metric].dropna().to_numpy(dtype=float)

            # match the published jitters
            if metric == 'waveform_duration':
                D = D + rng.random(len(D)) * 0.02
            elif metric == 'waveform_spread':
                D = D + rng.random(len(D)) * 10

            if USE_LOG[metric]:
                vals = np.log10(D + 1e-4)
            else:
                vals = D

            h, _ = np.histogram(vals, bins=bins, density=True)
            hist[ai] = h

            if metric == METRICS[0]:
                unit_counts[ai] = len(D)

        out[f'bin_edges_{metric}'] = bins
        out[f'hist_{metric}'] = hist

    out['unit_counts'] = unit_counts
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f'Loading {PKL_PATH} ...')
    units = load_units()
    print(f'  {len(units):,} units across {units["session_id"].nunique()} sessions')

    keep_cols = (['session_id', 'unit_id', 'ecephys_structure_acronym',
                  'area_group', 'quality'] + METRICS)
    long = units[keep_cols].copy()
    csv_path = OUT_DIR / 'units_long.csv'
    long.to_csv(csv_path, index=False)
    print(f'  wrote {csv_path}  ({len(long):,} rows, {len(keep_cols)} cols)')

    # Quick area_group / quality coverage report
    cov = (long.dropna(subset=['area_group'])
                 .groupby(['area_group', 'quality'])
                 .size()
                 .unstack(fill_value=0))
    print('\nUnit counts by area_group x quality:')
    print(cov.to_string())

    print('\nBuilding histograms ...')
    hists = build_histograms(units)
    npz_path = OUT_DIR / 'histograms.npz'
    np.savez(npz_path, **hists)
    print(f'  wrote {npz_path}')
    print(f'  unit_counts (good only, per Fig 4): {dict(zip(AREA_NAMES, hists["unit_counts"].tolist()))}')


if __name__ == '__main__':
    main()
