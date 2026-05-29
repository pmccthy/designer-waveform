"""
Extract VISp spike times (all layers) to natural scenes.
Saves raw spike times (s re stim onset) so bin size can be chosen freely at analysis time.

Usage:
    python scripts/28_04_26_extract_spiketimes_alllayers.py
"""
import subprocess
import sys

# ── pynwb/hdmf compatibility check (must be first — allensdk imports pynwb) ──
try:
    import pynwb
    from packaging.version import Version
    needs_fix = Version(pynwb.__version__) >= Version('2.6.0')
    version_str = pynwb.__version__
except ImportError:
    needs_fix = True
    version_str = 'unimportable'

if needs_fix:
    print(f'pynwb ({version_str}) incompatible with allensdk — installing pynwb<2.6 + hdmf<4...')
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install',
         'pynwb>=2.3,<2.6', 'hdmf>=3.5,<4', '--quiet'],
        check=True,
    )
    print('Done — please re-run this script for the change to take effect.')
    sys.exit(0)

print(f'pynwb {pynwb.__version__} OK')

import pickle
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache

T_PRE  = 0.05   # seconds before stim onset to retain
T_POST = 0.35   # seconds after stim onset to retain

CACHE_DIR = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_neuropixels_longwindow_5ms_bins')
OUT_DIR   = CACHE_DIR / 'spike_times_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

cache = EcephysProjectCache.from_warehouse(manifest=str(CACHE_DIR / 'manifest.json'))
sessions = cache.get_session_table()
has_visp = ['VISp' in str(areas) for areas in sessions.ecephys_structure_acronyms]
v1_ns_sessions = sessions[
    (sessions.session_type == 'brain_observatory_1.1') & has_visp
]
n_sessions = len(v1_ns_sessions)
print(f'{n_sessions} sessions with VISp + natural scenes')

resolution = 10
rsc = ReferenceSpaceCache(
    resolution=resolution,
    reference_space_key='annotation/ccf_2017',
    manifest=str(CACHE_DIR / 'reference_space_manifest.json'),
)
annot, _ = rsc.get_annotation_volume()
id_to_acronym = {s['id']: s['acronym'] for s in rsc.get_structure_tree().nodes()}

session_durations = []

for session_num, session_id in enumerate(v1_ns_sessions.index, start=1):
    out_path = OUT_DIR / f'{session_id}_alllayers_spiketimes.pkl'
    if out_path.exists():
        print(f'Session {session_id}: already extracted, skipping.')
        continue

    print(f'\n[{session_num}/{n_sessions}] Processing session {session_id} ...')
    t0 = time.time()
    session = cache.get_session_data(session_id)

    ns_table = session.get_stimulus_table('natural_scenes')
    ns_valid = ns_table[ns_table.frame >= 0].copy()

    v1_units = session.units[session.units.ecephys_structure_acronym == 'VISp'].copy()
    coords   = v1_units[['anterior_posterior_ccf_coordinate',
                          'dorsal_ventral_ccf_coordinate',
                          'left_right_ccf_coordinate']].values
    voxels   = (coords / resolution).astype(int)
    v1_units['layer'] = [id_to_acronym.get(sid, 'unknown')
                          for sid in annot[voxels[:, 0], voxels[:, 1], voxels[:, 2]]]

    print(f'  VISp units by layer:\n{v1_units["layer"].value_counts().to_string()}')

    if len(v1_units) == 0:
        print('  No VISp units — skipping.')
        nwb_path = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'
        if nwb_path.exists():
            nwb_path.unlink()
        continue

    unit_ids     = v1_units.index.values
    frame_counts = ns_valid.frame.astype(int).value_counts().sort_index()
    num_stim     = len(frame_counts)
    num_trials   = int(frame_counts.min())
    num_units    = len(v1_units)

    # index lookups: trial_id -> (stim_idx, trial_idx), unit_id -> unit_idx
    trial_id_to_idx = {}
    for i, frame_id in enumerate(frame_counts.index):
        for j, tid in enumerate(ns_valid[ns_valid.frame == frame_id].index.values[:num_trials]):
            trial_id_to_idx[int(tid)] = (i, j)
    unit_id_to_idx = {int(uid): k for k, uid in enumerate(unit_ids)}
    all_trial_ids  = np.array(list(trial_id_to_idx.keys()))

    # build sorted spike-time arrays per unit for fast searchsorted windowing
    unit_spike_times = {int(uid): np.sort(session.spike_times[uid]) for uid in unit_ids}

    # 3D object array: spikes[stim_idx, trial_idx, unit_idx] = float32 array of spike times
    spikes = np.empty((num_stim, num_trials, num_units), dtype=object)
    for idx in np.ndindex(spikes.shape):
        spikes[idx] = np.array([], dtype=np.float32)

    for trial_id, (stim_idx, trial_idx) in trial_id_to_idx.items():
        t_onset = ns_valid.loc[trial_id, 'start_time']
        t_lo    = t_onset - T_PRE
        t_hi    = t_onset + T_POST
        for k, uid in enumerate(unit_ids):
            st   = unit_spike_times[int(uid)]
            i_lo = np.searchsorted(st, t_lo)
            i_hi = np.searchsorted(st, t_hi, side='right')
            if i_hi > i_lo:
                spikes[stim_idx, trial_idx, k] = (st[i_lo:i_hi] - t_onset).astype(np.float32)

    print(f'  spikes shape: {spikes.shape}  '
          f'non-empty: {sum(s.size > 0 for s in spikes.flat)}/{spikes.size}')

    trial_start_times = np.array([
        ns_valid[ns_valid.frame == fid].start_time.values[:num_trials]
        for fid in frame_counts.index
    ])

    with open(out_path, 'wb') as f:
        pickle.dump({
            'session_id':        session_id,
            'frame_ids':         frame_counts.index.values,
            'unit_ids':          unit_ids,
            'layer':             v1_units['layer'].values,
            'T_PRE':             T_PRE,
            'T_POST':            T_POST,
            'spikes':            spikes,            # (num_stim, num_trials, num_units) object array
            'trial_start_times': trial_start_times,
            'unit_info':         v1_units[['probe_vertical_position',
                                           'anterior_posterior_ccf_coordinate',
                                           'dorsal_ventral_ccf_coordinate',
                                           'left_right_ccf_coordinate',
                                           'layer']].to_dict('list'),
        }, f)
    print(f'  Saved → {out_path}')

    elapsed = time.time() - t0
    session_durations.append(elapsed)
    mean_dur  = sum(session_durations) / len(session_durations)
    remaining = n_sessions - session_num
    print(f'  Time: {timedelta(seconds=int(elapsed))}  |  '
          f'ETA ({remaining} remaining): {timedelta(seconds=int(mean_dur * remaining))}')

    nwb_path = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'
    if nwb_path.exists():
        nwb_path.unlink()
        print(f'  Deleted {nwb_path}')
