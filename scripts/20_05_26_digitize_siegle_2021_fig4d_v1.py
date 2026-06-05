"""
Save a visually-digitized version of the V1 / VISp population response from
Siegle et al. 2021, Fig. 4d middle panel ("Change" and "Pre-change" traces).

⚠ APPROXIMATE — these points were read off the published figure by eye.
Expect ~±0.5 spikes/s on y, ~±10 ms on x. For a serious optimisation target,
re-digitize with WebPlotDigitizer (https://automeris.io/wpd) or contact the
authors for the source data.

Output: results/allen/siegle_2021_fig4d_v1.pkl
Structure matches results/allen/visp_psths.pkl so the optim notebook's load
cell can read it the same way.

Usage:
    python scripts/20_05_26_digitize_siegle_2021_fig4d_v1.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH  = REPO_ROOT / 'results' / 'allen' / 'siegle_2021_fig4d_v1.pkl'

# ─── Hand-digitized points (t in ms, response in spikes/s) ─────────────────
# Middle panel of Fig 4d in Siegle et al. 2021. Sampled roughly every 10 ms
# along each trace.

CHANGE_POINTS = np.array([
    # (t_ms, hz)
    (  0,  1.5),
    ( 10,  1.5),
    ( 20,  1.5),
    ( 30,  2.0),
    ( 40,  4.0),
    ( 50,  8.0),
    ( 60, 12.0),
    ( 70, 13.5),
    ( 80, 13.5),
    ( 90, 12.0),
    (100, 10.0),
    (110,  8.5),
    (120,  7.0),
    (130,  6.0),
    (140,  5.5),
    (150,  5.0),
    (160,  4.8),
    (170,  4.5),
    (180,  4.5),
    (190,  4.5),
    (200,  4.5),
    (210,  4.5),
    (220,  4.5),
    (230,  4.5),
    (240,  4.5),
    (250,  4.5),
    (260,  6.0),
    (270,  8.0),
    (280, 10.5),
    (290, 12.0),
    (300, 11.5),
    (310,  9.0),
    (320,  6.5),
    (330,  4.5),
    (340,  3.0),
    (350,  2.0),
])

PRE_CHANGE_POINTS = np.array([
    (  0,  1.0),
    ( 10,  1.0),
    ( 20,  1.0),
    ( 30,  1.5),
    ( 40,  3.5),
    ( 50,  6.0),
    ( 60,  8.0),
    ( 70,  9.0),
    ( 80,  9.0),
    ( 90,  8.0),
    (100,  6.5),
    (110,  5.5),
    (120,  4.5),
    (130,  4.0),
    (140,  4.0),
    (150,  4.0),
    (160,  4.0),
    (170,  4.0),
    (180,  4.0),
    (190,  4.0),
    (200,  4.0),
    (210,  4.0),
    (220,  4.0),
    (230,  4.0),
    (240,  4.5),
    (250,  5.0),
    (260,  7.0),
    (270,  9.0),
    (280, 11.0),
    (290, 11.5),
    (300, 11.0),
    (310,  8.0),
    (320,  5.0),
    (330,  3.0),
    (340,  2.0),
    (350,  1.5),
])


def resample(points: np.ndarray, bin_size_ms: float = 10.0,
             t_start: float = 0.0, t_end: float = 350.0) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate the hand-digitized points onto a regular bin grid."""
    t_centres = np.arange(t_start + bin_size_ms / 2, t_end, bin_size_ms)
    hz        = np.interp(t_centres, points[:, 0], points[:, 1])
    return t_centres, hz


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    bin_size_ms = 10.0
    t_centres, change_hz     = resample(CHANGE_POINTS,     bin_size_ms)
    _,         pre_change_hz = resample(PRE_CHANGE_POINTS, bin_size_ms)

    # Match the schema used by results/allen/visp_psths.pkl so the optim
    # notebook can read it via the same code path. We treat "change" as the
    # 'filtered' bucket and "pre_change" as the 'unfiltered' bucket, both under
    # an 'all' key (no per-layer breakdown is available from the figure).
    payload = {
        'meta': {
            'source':           'Siegle et al. 2021, Nature, Fig. 4d middle panel (V1 / VISp)',
            'method':           'Hand-digitized by visual inspection of the published figure.',
            'precision_note':   'Approximate; expect ~±0.5 spikes/s on y and ~±10 ms on x.',
            'bin_size_ms':      bin_size_ms,
            'T_PRE_ms':         0.0,
            'T_POST_ms':        float(t_centres[-1] + bin_size_ms / 2),
            'P_RF_THRESHOLD':   0.01,  # matches Siegle's primary criterion
            'time_ms':          t_centres,
            'layer_order':      ['all'],
            'n_sessions_total': None,
            'stim_onset_ms':    0.0,
            'stim_offset_ms':   250.0,
        },
        # 'filtered' = Change trace (this is what gets used when the optim
        # notebook has USE_RF_FILTER=True)
        'filtered': {
            'all': {
                'mean_hz':       change_hz,
                'sem_hz':        np.full_like(change_hz, np.nan),  # not extractable from figure
                'n_sessions':    None,
                'n_units_total': None,
                'label':         'Change',
            },
        },
        # 'unfiltered' = Pre-change trace
        'unfiltered': {
            'all': {
                'mean_hz':       pre_change_hz,
                'sem_hz':        np.full_like(pre_change_hz, np.nan),
                'n_sessions':    None,
                'n_units_total': None,
                'label':         'Pre-change',
            },
        },
    }

    with open(OUT_PATH, 'wb') as f:
        pickle.dump(payload, f)

    print(f'Saved → {OUT_PATH.resolve()}')
    print(f'  bin size:  {bin_size_ms:.1f} ms')
    print(f'  time grid: {t_centres[0]:.1f} → {t_centres[-1]:.1f} ms ({len(t_centres)} bins)')
    print(f'  Change      peak: {change_hz.max():.1f} Hz at t={t_centres[change_hz.argmax()]:.0f} ms')
    print(f'  Pre-change  peak: {pre_change_hz.max():.1f} Hz at t={t_centres[pre_change_hz.argmax()]:.0f} ms')


if __name__ == '__main__':
    main()
