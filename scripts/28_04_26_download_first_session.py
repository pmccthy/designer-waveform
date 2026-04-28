"""
Download the NWB for the first V1 natural-scenes session to the notebook cache dir.
Run once so 15_04_26_allen_natural_scene_v1_l23_psth.ipynb can load the session
interactively (DataArray exploration, etc.).

Pre-computed responses (pkl) are already saved — this only ensures the NWB is present.

Usage:
    python scripts/28_04_26_download_first_session.py
"""
import time
from datetime import timedelta
from pathlib import Path

import h5py
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

CACHE_DIR = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_neuropixels')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

cache = EcephysProjectCache.from_warehouse(manifest=str(CACHE_DIR / 'manifest.json'))

sessions    = cache.get_session_table()
has_visp    = ['VISp' in str(areas) for areas in sessions.ecephys_structure_acronyms]
v1_ns_sessions = sessions[
    (sessions.session_type == 'brain_observatory_1.1') & has_visp
]
session_id = v1_ns_sessions.index[0]
nwb_path   = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'

# check if a valid (non-truncated) NWB already exists
if nwb_path.exists():
    try:
        with h5py.File(nwb_path, 'r'):
            pass
        size_gb = nwb_path.stat().st_size / 1e9
        print(f'NWB already valid ({size_gb:.2f} GB): {nwb_path}')
        raise SystemExit(0)
    except OSError:
        size_mb = nwb_path.stat().st_size / 1e6
        print(f'NWB truncated ({size_mb:.0f} MB) — deleting and re-downloading...')
        nwb_path.unlink()

print(f'Downloading NWB for session {session_id}...')
t0 = time.time()
cache.get_session_data(session_id)
print(f'Done in {timedelta(seconds=int(time.time() - t0))}')
print(f'NWB saved at: {nwb_path}')
