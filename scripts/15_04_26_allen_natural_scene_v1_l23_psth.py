"""
Extract V1 L2/3 responses to natural scenes from Allen Visual Coding Neuropixels dataset
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import os
import pickle
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache

# define parameters for PSTH computation
BIN_SIZE = 0.005   # 10 ms
T_PRE    = 0.05 
T_POST   = 0.35  

# Load data
# initialise cache
CACHE_DIR = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_neuropixels_longwindow_5ms_bins') 
CACHE_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = CACHE_DIR / 'manifest.json'
cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

# load session table
sessions = cache.get_session_table()

# identify sessions which contain natural scene stimuli and V1 neurons
has_visp = ['VISp' in str(areas) for areas in sessions.ecephys_structure_acronyms] # boolean indicating presence of VISp in area list

# get relevant sessions 
v1_ns_sessions = sessions[
    (sessions.session_type == 'brain_observatory_1.1') & # "brain_observatory_1.1" sessions contain natural scene stimuli
    has_visp
] 

n_sessions = len(v1_ns_sessions)
print(f'{n_sessions} sessions with VISp + natural scenes')

# load CCF annotation volume once (cached after first download, ~1 GB)
resolution = 10  # microns per voxel
rsc = ReferenceSpaceCache(
    resolution=resolution,
    reference_space_key='annotation/ccf_2017',
    manifest=str(CACHE_DIR / 'reference_space_manifest.json')
)
annot, _ = rsc.get_annotation_volume()  # shape (1320, 800, 1140), uint32
id_to_acronym = {s['id']: s['acronym'] for s in rsc.get_structure_tree().nodes()}

bin_edges = np.arange(0, T_POST + BIN_SIZE, BIN_SIZE)  # T_PRE=0, no pre-stim (ISI ≈ 250 ms)

session_durations = []

for session_num, session_id in enumerate(v1_ns_sessions.index, start=1):

    out_path = CACHE_DIR / f'{session_id}_l23_psth_responses.pkl'
    if out_path.exists():
        print(f'Session {session_id}: already extracted, skipping.')
        continue

    print(f'\n[{session_num}/{n_sessions}] Processing session {session_id} ...')
    session_start = time.time()
    session = cache.get_session_data(session_id)

    ## Natural scene stimulus table
    ns_table = session.get_stimulus_table('natural_scenes')
    ns_valid = ns_table[ns_table.frame >= 0].copy()  # exclude blank (frame == -1)

    ## Find V1 L2/3 units via CCF layer lookup
    v1_units = session.units[session.units.ecephys_structure_acronym == 'VISp'].copy()
    coords = v1_units[['anterior_posterior_ccf_coordinate',
                        'dorsal_ventral_ccf_coordinate',
                        'left_right_ccf_coordinate']].values
    voxels = (coords / resolution).astype(int)
    structure_ids = annot[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    v1_units['layer'] = [id_to_acronym.get(sid, 'unknown') for sid in structure_ids]

    l23_units = v1_units[v1_units.layer.isin(['VISp2/3', 'VISp2/3a', 'VISp2/3b'])]
    print(f'  {len(l23_units)} L2/3 units')

    if len(l23_units) == 0:
        print('  No L2/3 units — skipping.')
        # delete NWB to free space
        nwb_path = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'
        if nwb_path.exists():
            nwb_path.unlink()
            print(f'  Deleted {nwb_path}')
        continue

    ## Build response matrix: (num_stim, num_trials, num_bins, num_units)
    unit_ids     = l23_units.index.values
    frame_counts = ns_valid.frame.astype(int).value_counts().sort_index()
    num_stim     = len(frame_counts)
    num_trials   = int(frame_counts.min())   # use min so every stimulus has equal trials
    num_units    = len(l23_units)
    num_bins     = len(bin_edges) - 1

    responses = np.zeros((num_stim, num_trials, num_bins, num_units))

    for i, frame_id in enumerate(frame_counts.index):
        trial_ids = ns_valid[ns_valid.frame == frame_id].index.values[:num_trials]
        r = session.presentationwise_spike_counts(
            stimulus_presentation_ids=trial_ids,
            bin_edges=bin_edges,
            unit_ids=unit_ids,
        ).values  # (num_trials, num_bins, num_units)
        responses[i, : , :, :] = r

    print(f'  responses shape (stim × trials × bins × units): {responses.shape}')

    ## Save
    trial_start_times = np.array([
        ns_valid[ns_valid.frame == fid].start_time.values[:num_trials]
        for fid in frame_counts.index
    ])  # shape (num_stim, num_trials)

    with open(out_path, 'wb') as f:
        pickle.dump({
            'session_id':        session_id,
            'frame_ids':         frame_counts.index.values,   # (num_stim,) — indexes into natural scene images 0-117
            'unit_ids':          unit_ids,                    # (num_units,) — indexes into session.units
            'bin_edges':         bin_edges,                   # (num_bins+1,) — time axis in seconds relative to stim onset
            'responses':         responses,                   # (num_stim, num_trials, num_bins, num_units)
            'trial_start_times': trial_start_times,           # (num_stim, num_trials) — absolute onset times in seconds
            'unit_info':         l23_units[['probe_vertical_position',
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

    ## Delete NWB file to free disk space (~14 GB per session)
    nwb_path = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'
    if nwb_path.exists():
        nwb_path.unlink()
        print(f'  Deleted {nwb_path}')

