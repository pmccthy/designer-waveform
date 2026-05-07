"""
Download the NWB for the first V1 natural-scenes session to the notebook cache dir.
Uses curl for a reliable download and pins pynwb+hdmf to versions compatible with allensdk.

Usage:
    python scripts/28_04_26_download_first_session.py
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

# ── remaining imports (safe now that pynwb/hdmf are compatible) ───────────────
import time
from datetime import timedelta
from pathlib import Path

import h5py
import requests
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_neuropixels')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

cache = EcephysProjectCache.from_warehouse(manifest=str(CACHE_DIR / 'manifest.json'))
sessions       = cache.get_session_table()
has_visp       = ['VISp' in str(areas) for areas in sessions.ecephys_structure_acronyms]
v1_ns_sessions = sessions[(sessions.session_type == 'brain_observatory_1.1') & has_visp]
session_id     = v1_ns_sessions.index[0]
nwb_path       = CACHE_DIR / f'session_{session_id}' / f'session_{session_id}.nwb'

# ── check existing file ───────────────────────────────────────────────────────
if nwb_path.exists():
    try:
        with h5py.File(nwb_path, 'r'):
            pass
        size_gb = nwb_path.stat().st_size / 1e9
        print(f'NWB already valid ({size_gb:.2f} GB): {nwb_path}')
        sys.exit(0)
    except OSError:
        size_mb = nwb_path.stat().st_size / 1e6
        print(f'NWB truncated ({size_mb:.0f} MB) — deleting...')
        nwb_path.unlink()

# ── resolve download URL via Allen RMA API ────────────────────────────────────
print(f'Resolving download URL for session {session_id}...')
rma_url = (
    f'https://api.brain-map.org/api/v2/data/query.json'
    f"?criteria=model::WellKnownFile"
    f",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
    f"[attachable_type$eq'EcephysSession']"
    f'[attachable_id$eq{session_id}]&num_rows=1'
)
resp = requests.get(rma_url, timeout=15)
resp.raise_for_status()
download_link = resp.json()['msg'][0]['download_link']
full_url = f'https://api.brain-map.org{download_link}'
print(f'URL: {full_url}')

# ── download with curl ────────────────────────────────────────────────────────
nwb_path.parent.mkdir(parents=True, exist_ok=True)
print(f'Downloading to {nwb_path} ...')
t0 = time.time()
subprocess.run(
    ['curl', '-L', '--progress-bar', '-o', str(nwb_path), full_url],
    check=True,
)
elapsed = timedelta(seconds=int(time.time() - t0))

# ── verify ────────────────────────────────────────────────────────────────────
try:
    with h5py.File(nwb_path, 'r') as f:
        size_gb = nwb_path.stat().st_size / 1e9
        keys    = list(f.keys())
    print(f'Download complete in {elapsed} ({size_gb:.2f} GB)')
    print(f'Top-level NWB groups: {keys}')
except OSError as e:
    print(f'Download appears truncated: {e}')
    print('Re-run the script to try again.')
