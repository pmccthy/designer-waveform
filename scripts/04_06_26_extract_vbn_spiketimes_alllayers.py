"""
Extract VISp spike times (all layers) to natural scenes from the
Allen Visual Behavior Neuropixels dataset.

Mirrors 28_05_26_extract_spiketimes_alllayers.py (passive dataset), adapted
for the VisualBehaviorNeuropixelsProjectCache API.

Key differences vs. the passive dataset:
  - Cache class  : VisualBehaviorNeuropixelsProjectCache  (not EcephysProjectCache)
  - Session load : cache.get_ecephys_session(ecephys_session_id=...)
  - Units        : session.get_units()  (method, not .units property)
  - Stimuli      : session.stimulus_presentations  filtered on
                   stimulus_name == 'natural_images', grouped by image_name
  - image_index  : integer 0-117, equivalent to 'frame' in the passive dataset
  - 'active'     : bool column saved per trial (True = active behaviour block,
                   False = passive replay block)

Output per session  (saved to OUT_DIR):
  {session_id}_vbn_alllayers_spiketimes.pkl
  Keys:
    session_id        : int
    image_names       : (num_stim,)  str array — image filename stem
    image_indices     : (num_stim,)  int array — 0-based index (equiv. to 'frame')
    unit_ids          : (num_units,) int array
    layer             : (num_units,) str array — CCF layer acronym
    T_PRE, T_POST     : float, seconds
    spikes            : (num_stim, num_trials, num_units) object array of
                        float32 spike-time arrays (seconds re stimulus onset)
    trial_start_times : (num_stim, num_trials) float64 array — absolute onset (s)
    active            : (num_stim, num_trials) bool array — True = active block
    unit_info         : dict of lists (probe_vertical_position, CCF coords, layer)

Usage:
    python scripts/04_06_26_extract_vbn_spiketimes_alllayers.py
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

import glob
import pickle
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
from allensdk.core.reference_space_cache import ReferenceSpaceCache

# ── parameters ────────────────────────────────────────────────────────────────
T_PRE  = 0.05   # seconds before stimulus onset to retain
T_POST = 0.35   # seconds after stimulus onset to retain

CACHE_DIR = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_behavior_neuropixels')
OUT_DIR   = CACHE_DIR / 'spike_times'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── initialise cache ──────────────────────────────────────────────────────────
# from_s3_cache downloads files from the Allen S3 bucket on first access
# and caches them locally in CACHE_DIR.
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))

# ── identify sessions with VISp units ─────────────────────────────────────────
# get_unit_table() is a lightweight metadata download (no NWB files required)
print('Fetching unit table to identify VISp sessions …')
unit_table = cache.get_unit_table()
visp_session_ids = set(
    unit_table[unit_table.ecephys_structure_acronym == 'VISp']
    .ecephys_session_id.unique()
)

sessions = cache.get_ecephys_session_table()
vbn_visp_sessions = sessions.loc[sessions.index.isin(visp_session_ids)]
n_sessions = len(vbn_visp_sessions)
print(f'{n_sessions} ecephys sessions with VISp units')

# ── load CCF annotation volume once (cached after first download, ~1 GB) ──────
resolution = 10  # microns per voxel
rsc = ReferenceSpaceCache(
    resolution=resolution,
    reference_space_key='annotation/ccf_2017',
    manifest=str(CACHE_DIR / 'reference_space_manifest.json'),
)
annot, _ = rsc.get_annotation_volume()          # (1320, 800, 1140) uint32
id_to_acronym = {s['id']: s['acronym'] for s in rsc.get_structure_tree().nodes()}

# ── main extraction loop ───────────────────────────────────────────────────────
session_durations = []

for session_num, session_id in enumerate(vbn_visp_sessions.index, start=1):

    out_path = OUT_DIR / f'{session_id}_vbn_alllayers_spiketimes.pkl'
    if out_path.exists():
        print(f'Session {session_id}: already extracted, skipping.')
        continue

    print(f'\n[{session_num}/{n_sessions}] Processing session {session_id} …')
    t0 = time.time()

    session = cache.get_ecephys_session(ecephys_session_id=session_id)

    # ── stimulus table: natural images only ───────────────────────────────────
    stim = session.stimulus_presentations
    ns_table = stim[stim.stimulus_name == 'natural_images'].copy()

    if len(ns_table) == 0:
        print('  No natural_images presentations — skipping.')
        continue

    # image_index: integer 0-117 (equivalent to 'frame' in the passive dataset)
    # image_name:  string, e.g. 'im015_r'
    # active:      True during active behaviour block, False during passive replay
    if 'image_index' not in ns_table.columns:
        # Fallback: derive from image_name sort order
        unique_names = sorted(ns_table.image_name.unique())
        name_to_idx  = {n: i for i, n in enumerate(unique_names)}
        ns_table['image_index'] = ns_table.image_name.map(name_to_idx)

    # ── VISp units + CCF layer assignment ─────────────────────────────────────
    # get_units() returns all units; filter to VISp
    all_units = session.get_units()
    v1_units  = all_units[all_units.ecephys_structure_acronym == 'VISp'].copy()

    coords = v1_units[['anterior_posterior_ccf_coordinate',
                        'dorsal_ventral_ccf_coordinate',
                        'left_right_ccf_coordinate']].values
    voxels = (coords / resolution).astype(int)
    # clamp to annotation volume bounds
    voxels[:, 0] = np.clip(voxels[:, 0], 0, annot.shape[0] - 1)
    voxels[:, 1] = np.clip(voxels[:, 1], 0, annot.shape[1] - 1)
    voxels[:, 2] = np.clip(voxels[:, 2], 0, annot.shape[2] - 1)
    v1_units['layer'] = [
        id_to_acronym.get(int(annot[vox[0], vox[1], vox[2]]), 'unknown')
        for vox in voxels
    ]
    print(f'  VISp units by layer:\n{v1_units["layer"].value_counts().to_string()}')

    if len(v1_units) == 0:
        print('  No VISp units — skipping.')
        continue

    # ── build sorted spike-time arrays per unit ───────────────────────────────
    unit_ids         = v1_units.index.values
    unit_spike_times = {int(uid): np.sort(session.spike_times[uid])
                        for uid in unit_ids}

    # ── group trials by image ─────────────────────────────────────────────────
    # Use image_index as the primary grouper (consistent with passive dataset)
    image_idx_counts = ns_table.image_index.astype(int).value_counts().sort_index()
    num_stim   = len(image_idx_counts)
    num_trials = int(image_idx_counts.min())   # equal trials per stimulus
    num_units  = len(v1_units)

    # Ordered list of unique image_names matching sorted image_index
    idx_to_name = (
        ns_table[['image_index', 'image_name']]
        .drop_duplicates()
        .set_index('image_index')['image_name']
        .sort_index()
    )
    image_indices = image_idx_counts.index.values          # (num_stim,) int
    image_names   = idx_to_name.loc[image_indices].values  # (num_stim,) str

    # ── allocate output arrays ────────────────────────────────────────────────
    spikes            = np.empty((num_stim, num_trials, num_units), dtype=object)
    trial_start_times = np.zeros((num_stim, num_trials), dtype=np.float64)
    active_flags      = np.zeros((num_stim, num_trials), dtype=bool)

    for idx in np.ndindex(spikes.shape):
        spikes[idx] = np.array([], dtype=np.float32)

    # ── extract windowed spike times ──────────────────────────────────────────
    for i, img_idx in enumerate(image_indices):
        rows = ns_table[ns_table.image_index == img_idx].iloc[:num_trials]
        for j, (trial_id, row) in enumerate(rows.iterrows()):
            t_onset = row.start_time
            t_lo    = t_onset - T_PRE
            t_hi    = t_onset + T_POST
            trial_start_times[i, j] = t_onset
            active_flags[i, j]      = bool(row.get('active', False))
            for k, uid in enumerate(unit_ids):
                st   = unit_spike_times[int(uid)]
                i_lo = np.searchsorted(st, t_lo)
                i_hi = np.searchsorted(st, t_hi, side='right')
                if i_hi > i_lo:
                    spikes[i, j, k] = (st[i_lo:i_hi] - t_onset).astype(np.float32)

    non_empty = sum(s.size > 0 for s in spikes.flat)
    print(f'  spikes shape: {spikes.shape}  non-empty: {non_empty}/{spikes.size}')

    # ── save ──────────────────────────────────────────────────────────────────
    with open(out_path, 'wb') as f:
        pickle.dump({
            'session_id':        session_id,
            'image_names':       image_names,        # (num_stim,) str — image filename stem
            'image_indices':     image_indices,      # (num_stim,) int — equiv. to 'frame' in passive
            'unit_ids':          unit_ids,            # (num_units,)
            'layer':             v1_units['layer'].values,
            'T_PRE':             T_PRE,
            'T_POST':            T_POST,
            'spikes':            spikes,              # (num_stim, num_trials, num_units) object array
            'trial_start_times': trial_start_times,  # (num_stim, num_trials) absolute onset times (s)
            'active':            active_flags,        # (num_stim, num_trials) bool — True = active block
            'unit_info':         v1_units[[
                                     'probe_vertical_position',
                                     'anterior_posterior_ccf_coordinate',
                                     'dorsal_ventral_ccf_coordinate',
                                     'left_right_ccf_coordinate',
                                     'layer',
                                 ]].to_dict('list'),
        }, f)
    print(f'  Saved → {out_path}')

    elapsed = time.time() - t0
    session_durations.append(elapsed)
    mean_dur  = sum(session_durations) / len(session_durations)
    remaining = n_sessions - session_num
    print(f'  Time: {timedelta(seconds=int(elapsed))}  |  '
          f'ETA ({remaining} remaining): {timedelta(seconds=int(mean_dur * remaining))}')

    # ── free disk space: delete NWB file (~several GB per session) ────────────
    nwb_files = glob.glob(
        str(CACHE_DIR / '**' / f'*{session_id}*.nwb'), recursive=True
    )
    for nwb_path in nwb_files:
        Path(nwb_path).unlink()
        print(f'  Deleted {nwb_path}')
