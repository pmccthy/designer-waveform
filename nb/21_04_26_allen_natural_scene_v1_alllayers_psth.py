"""
Extract VISp responses (all layers) to natural scenes from Allen Visual Coding Neuropixels dataset
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import pickle
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache

# PSTH parameters
BIN_SIZE = 0.005   # 5 ms
T_PRE    = 0.05
T_POST   = 0.35

CACHE_DIR = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_neuropixels_longwindow_5ms_bins')
OUT_DIR   = CACHE_DIR / 'all_layers'
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = CACHE_DIR / 'manifest.json'
cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

sessions = cache.get_session_table()
has_visp = ['VISp' in str(areas) for areas in sessions.ecephys_structure_acronyms]
v1_ns_sessions = sessions[
    (sessions.session_type == 'brain_observatory_1.1') &
    has_visp
]
n_sessions = len(v1_ns_sessions)
print(f'{n_sessions} sessions with VISp + natural scenes')

resolution = 10
rsc = ReferenceSpaceCache(
    resolution=resolution,
    reference_space_key='annotation/ccf_2017',
    manifest=str(CACHE_DIR / 'reference_space_manifest.json')
)
annot, _ = rsc.get_annotation_volume()
id_to_acronym = {s['id']: s['acronym'] for s in rsc.get_structure_tree().nodes()}

bin_edges = np.arange(0, T_POST + BIN_SIZE, BIN_SIZE)

session_durations = []

for session_num, session_id in enumerate(v1_ns_sessions.index, start=1):

    out_path = OUT_DIR / f'{session_id}_alllayers_psth_responses.pkl'
    if out_path.exists():
        print(f'Session {session_id}: already extracted, skipping.')
        continue

    print(f'\n[{session_num}/{n_sessions}] Processing session {session_id} ...')
    session_start = time.time()
    session = cache.get_session_data(session_id)

    ns_table = session.get_stimulus_table('natural_scenes')
    ns_valid = ns_table[ns_table.frame >= 0].copy()

    # label all VISp units with their CCF layer
    v1_units = session.units[session.units.ecephys_structure_acronym == 'VISp'].copy()
    coords = v1_units[['anterior_posterior_ccf_coordinate',
                        'dorsal_ventral_ccf_coordinate',
                        'left_right_ccf_coordinate']].values
    voxels = (coords / resolution).astype(int)
    structure_ids = annot[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    v1_units['layer'] = [id_to_acronym.get(sid, 'unknown') for sid in structure_ids]

    layer_counts = v1_units['layer'].value_counts()
    print(f'  VISp units by layer:\n{layer_counts.to_string()}')

    if len(v1_units) == 0:
        print('  No VISp units — skipping.')
        nwb_path = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'
        if nwb_path.exists():
            nwb_path.unlink()
            print(f'  Deleted {nwb_path}')
        continue

    unit_ids     = v1_units.index.values
    frame_counts = ns_valid.frame.astype(int).value_counts().sort_index()
    num_stim     = len(frame_counts)
    num_trials   = int(frame_counts.min())
    num_units    = len(v1_units)
    num_bins     = len(bin_edges) - 1

    responses = np.zeros((num_stim, num_trials, num_bins, num_units))

    for i, frame_id in enumerate(frame_counts.index):
        trial_ids = ns_valid[ns_valid.frame == frame_id].index.values[:num_trials]
        r = session.presentationwise_spike_counts(
            stimulus_presentation_ids=trial_ids,
            bin_edges=bin_edges,
            unit_ids=unit_ids,
        ).values  # (num_trials, num_bins, num_units)
        responses[i, :, :, :] = r

    print(f'  responses shape (stim × trials × bins × units): {responses.shape}')

    trial_start_times = np.array([
        ns_valid[ns_valid.frame == fid].start_time.values[:num_trials]
        for fid in frame_counts.index
    ])

    with open(out_path, 'wb') as f:
        pickle.dump({
            'session_id':        session_id,
            'frame_ids':         frame_counts.index.values,
            'unit_ids':          unit_ids,
            'bin_edges':         bin_edges,
            'responses':         responses,
            'trial_start_times': trial_start_times,
            'layer':             v1_units['layer'].values,      # (num_units,) — CCF layer for each unit
            'unit_info':         v1_units[['probe_vertical_position',
                                           'anterior_posterior_ccf_coordinate',
                                           'dorsal_ventral_ccf_coordinate',
                                           'left_right_ccf_coordinate',
                                           'layer']].to_dict('list'),
        }, f)
    print(f'  Saved → {out_path}')

    elapsed = time.time() - session_start
    session_durations.append(elapsed)
    mean_duration = sum(session_durations) / len(session_durations)
    sessions_remaining = n_sessions - session_num
    eta = timedelta(seconds=int(mean_duration * sessions_remaining))
    print(f'  Session time: {timedelta(seconds=int(elapsed))}  |  '
          f'Mean: {timedelta(seconds=int(mean_duration))}  |  '
          f'ETA ({sessions_remaining} remaining): {eta}')

    nwb_path = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'
    if nwb_path.exists():
        nwb_path.unlink()
        print(f'  Deleted {nwb_path}')
