"""
Compute grand-average target PSTH using only stimulus-increasing units.

For each (session × stimulus) pair, a unit is "increasing" if its trial-averaged
mean firing rate in the stimulus window (0–250 ms) exceeds its mean baseline
firing rate (−50–0 ms). Only increasing units contribute to the average for that
stimulus. Session-level traces are the mean across stimuli; the grand average is
then the cross-session mean.

This is purely a post-processing step on top of the existing spike-time pickles
produced by scripts/28_05_26_extract_spiketimes_alllayers.py — no re-download
or re-extraction from Allen is needed.

Output
------
OUT_PATH  (pkl)
  A dict with keys:
    'meta'        : recording parameters, time axis, thresholds
    'grand_avg'   : dict with keys 'mean_hz', 'sem_hz', 'n_sessions', 't_ms'
    'per_session' : list of dicts (one per session), each with:
                      'session_id', 'n_stim_total', 'n_stim_contributing',
                      'mean_n_increasing_units', 'mean_hz' (time series)

Usage
-----
    python scripts/05_06_26_extract_increasing_units_psth.py
"""

from pathlib import Path
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── parameters ────────────────────────────────────────────────────────────────
BIN_SIZE_S = 0.005          # 5 ms bins
T_STIM_ON  = 0.0            # stim onset (s, relative to stim onset — always 0)
T_STIM_OFF = 0.250          # stim offset / end of classification window (s)

# A unit is "increasing" for stimulus s if:
#   peak_fr(stim_window) > mean_fr(baseline_window)
# where stim_window = [STIM_WIN_LO, STIM_WIN_HI] and
# baseline_window   = [BASELINE_LO, BASELINE_HI]
STIM_WIN_LO   = 0.050        # s
STIM_WIN_HI   = 0.175        # s  (core response window, avoids onset transient edge & offset contamination)
BASELINE_LO   = -0.050       # s  (pre-stim onset)
BASELINE_HI   = 0.0          # s

# Unimodality filter: require no local minima in the stim window (0–250 ms)
# on the Gaussian-smoothed PSTH.  This removes units with bump-dip-bump responses.
# Set REQUIRE_UNIMODAL = False to disable and revert to peak-only criterion.
REQUIRE_UNIMODAL  = True
SMOOTH_SIGMA_MS   = 15.0     # Gaussian smoothing σ before unimodality check (ms)
                              # Larger → more lenient (small dips smoothed away)

SPIKES_DIR = Path('/Users/pmccarthy/Documents/experimental_data/'
                  'allen_visual_neuropixels_longwindow_5ms_bins/spike_times_v2')
OUT_PATH   = SPIKES_DIR.parent / 'increasing_units_grand_avg_psth.pkl'


# ── helpers ───────────────────────────────────────────────────────────────────

def bin_session_spikes(spikes_obj: np.ndarray,
                       bin_edges: np.ndarray) -> np.ndarray:
    """
    Convert the raw spike-time object array to a binned PSTH array.

    Parameters
    ----------
    spikes_obj : ndarray, shape (num_stim, num_trials, num_units), dtype=object
        Each element is a float32 array of spike times (s re onset).
    bin_edges  : ndarray, shape (num_bins + 1,)

    Returns
    -------
    psth : ndarray, shape (num_stim, num_units, num_bins), float32
        Trial-averaged firing rate in Hz for each (stim, unit).
    """
    num_stim, num_trials, num_units = spikes_obj.shape
    num_bins = len(bin_edges) - 1
    psth = np.zeros((num_stim, num_units, num_bins), dtype=np.float32)

    for si in range(num_stim):
        for k in range(num_units):
            counts = np.zeros(num_bins, dtype=np.float32)
            for t in range(num_trials):
                st = spikes_obj[si, t, k]
                if len(st) > 0:
                    counts += np.histogram(st, bins=bin_edges)[0].astype(np.float32)
            psth[si, k, :] = counts / (num_trials * BIN_SIZE_S)  # → Hz

    return psth


def compute_session_target(psth: np.ndarray,
                            t_ms: np.ndarray) -> dict:
    """
    Given a trial-averaged PSTH (num_stim, num_units, num_bins) in Hz,
    select stimulus-increasing units per stimulus and compute the
    session-level target PSTH.

    Returns a dict with:
      'mean_hz'                 : (num_bins,) grand mean over stimuli
      'n_stim_contributing'     : int — number of stim with ≥1 increasing unit
      'mean_n_increasing_units' : float — mean #increasing units per contributing stim
      'stim_targets'            : (n_stim_contributing, num_bins) — per-stim averages
    """
    baseline_mask = (t_ms >= BASELINE_LO * 1000) & (t_ms < BASELINE_HI * 1000)
    stim_win_mask = (t_ms >= STIM_WIN_LO * 1000) & (t_ms <= STIM_WIN_HI * 1000)

    num_stim = psth.shape[0]

    # baseline mean and peak response in classification window: (num_stim, num_units)
    # Using peak rather than mean avoids a window-edge selection artifact: mean-based
    # classification biases towards early-responding units (they dominate the window
    # mean), which produces an artificial step in the population average at exactly
    # STIM_WIN_LO.  Peak-based classification selects any unit that responds anywhere
    # in the window, regardless of when, giving a smooth population-level rise.
    baseline_mean = psth[:, :, baseline_mask].mean(axis=2)
    stim_peak     = psth[:, :, stim_win_mask].max(axis=2)

    # increasing mask: (num_stim, num_units)
    increasing = stim_peak > baseline_mean

    if REQUIRE_UNIMODAL:
        # Smooth along the time axis then check for local minima in the full
        # stim window (0–250 ms).  A local minimum means the response dips then
        # recovers — the bump-dip-bump pattern.  Units with any such dip are
        # excluded, keeping only units with a single clean response peak.
        sigma_bins  = SMOOTH_SIGMA_MS / (BIN_SIZE_S * 1000)
        psth_smooth = gaussian_filter1d(psth.astype(np.float32), sigma=sigma_bins, axis=2)

        stim_full_mask = (t_ms >= T_STIM_ON * 1000) & (t_ms <= T_STIM_OFF * 1000)
        p        = psth_smooth[:, :, stim_full_mask]          # (stim, units, t)
        inner    = p[:, :, 1:-1]
        has_dip  = ((inner < p[:, :, :-2]) & (inner < p[:, :, 2:])).any(axis=2)  # (stim, units)
        increasing = increasing & ~has_dip

    stim_targets   = []
    n_inc_per_stim = []

    for si in range(num_stim):
        inc_mask = increasing[si, :]          # (num_units,)
        n_inc    = int(inc_mask.sum())
        if n_inc == 0:
            continue                          # skip: no unit increased for this stim
        stim_targets.append(psth[si, inc_mask, :].mean(axis=0))   # (num_bins,)
        n_inc_per_stim.append(n_inc)

    if not stim_targets:
        return None

    stim_targets = np.stack(stim_targets)    # (n_contributing, num_bins)
    return {
        'mean_hz':                 stim_targets.mean(axis=0),
        'n_stim_contributing':     len(stim_targets),
        'mean_n_increasing_units': float(np.mean(n_inc_per_stim)),
        'stim_targets':            stim_targets,
    }


# ── main ──────────────────────────────────────────────────────────────────────

pkl_files = sorted(SPIKES_DIR.glob('*_alllayers_spiketimes.pkl'))
print(f'Found {len(pkl_files)} session pkl files in {SPIKES_DIR}')

if not pkl_files:
    raise FileNotFoundError(f'No spike-time pickles found in {SPIKES_DIR}')

# Build time axis from first session's metadata
with open(pkl_files[0], 'rb') as f:
    _s0 = pickle.load(f)
T_PRE  = _s0['T_PRE']
T_POST = _s0['T_POST']
bin_edges = np.arange(-T_PRE, T_POST + BIN_SIZE_S, BIN_SIZE_S)
t_ms      = (bin_edges[:-1] + bin_edges[1:]) / 2 * 1000       # bin centres in ms
num_bins  = len(bin_edges) - 1
print(f'Window: {-T_PRE*1000:.0f} – {T_POST*1000:.0f} ms  |  '
      f'bin size: {BIN_SIZE_S*1000:.0f} ms  |  {num_bins} bins')

session_results = []

for pkl_path in pkl_files:
    session_id = int(pkl_path.stem.split('_')[0])
    print(f'\nSession {session_id} ...', end=' ', flush=True)

    with open(pkl_path, 'rb') as f:
        s = pickle.load(f)

    spikes    = s['spikes']      # (num_stim, num_trials, num_units) object
    num_units = spikes.shape[2]

    if num_units == 0:
        print('0 units — skipped.')
        continue

    # bin → (num_stim, num_units, num_bins) Hz
    psth = bin_session_spikes(spikes, bin_edges)

    result = compute_session_target(psth, t_ms)
    if result is None:
        print('no increasing units found in any stim — skipped.')
        continue

    session_results.append({
        'session_id':              session_id,
        'n_stim_total':            spikes.shape[0],
        'n_stim_contributing':     result['n_stim_contributing'],
        'mean_n_increasing_units': result['mean_n_increasing_units'],
        'mean_hz':                 result['mean_hz'],
    })
    print(f'{result["n_stim_contributing"]}/{spikes.shape[0]} stim contributing  |  '
          f'mean {result["mean_n_increasing_units"]:.1f} increasing units/stim')

print(f'\n{len(session_results)} sessions contributed.')

if not session_results:
    raise RuntimeError('No sessions contributed — check spike-time pickle directory.')

# Grand average: one trace per session (mean over contributing stimuli), then mean ± SEM
session_traces = np.stack([r['mean_hz'] for r in session_results])  # (n_sessions, num_bins)
grand_mean     = session_traces.mean(axis=0)
grand_sem      = session_traces.std(axis=0) / np.sqrt(session_traces.shape[0])

payload = {
    'meta': {
        'bin_size_ms':    float(BIN_SIZE_S * 1000),
        'T_PRE_ms':       float(T_PRE  * 1000),
        'T_POST_ms':      float(T_POST * 1000),
        'stim_win_lo_ms':    float(STIM_WIN_LO * 1000),
        'stim_win_hi_ms':    float(STIM_WIN_HI * 1000),
        'baseline_lo_ms':    float(BASELINE_LO * 1000),
        'baseline_hi_ms':    float(BASELINE_HI * 1000),
        'require_unimodal':  REQUIRE_UNIMODAL,
        'smooth_sigma_ms':   float(SMOOTH_SIGMA_MS) if REQUIRE_UNIMODAL else None,
        't_ms':           t_ms,
        'n_sessions_total':       len(pkl_files),
        'n_sessions_contributing': len(session_results),
    },
    'grand_avg': {
        'mean_hz':   grand_mean,
        'sem_hz':    grand_sem,
        'n_sessions': len(session_results),
        't_ms':      t_ms,
    },
    'per_session': session_results,
}

with open(OUT_PATH, 'wb') as f:
    pickle.dump(payload, f)

print(f'\nSaved → {OUT_PATH}')
print(f'  {len(session_results)} sessions  |  '
      f'peak grand avg: {grand_mean.max():.2f} Hz  |  '
      f'bin size: {BIN_SIZE_S*1000:.0f} ms')
