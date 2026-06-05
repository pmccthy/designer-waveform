"""
Build a per-session mask of unit_ids passing the receptive-field p-value filter
(p_value_rf < 0.01), the primary unit-selection criterion in Siegle et al. 2021.

Operates entirely on already-extracted data:
    - data/unit_analysis_metrics.pkl  (from 06_05_26_extract_unit_analysis_metrics.ipynb)
    - spike_times_v2/<sid>_alllayers_spiketimes.pkl  (from 28_04_26_extract_spiketimes_alllayers.py)

Nothing is re-downloaded. No AllenSDK call.

Output: data/units_rf_pvalue_lt_001.pkl with the structure
    {
        'threshold':          0.01,
        'created_at':         iso timestamp,
        'passing_unit_ids':   {session_id (int): np.ndarray[int] of passing unit_ids},
        'summary':            pd.DataFrame indexed by session_id with columns
                              [total_units, rf_notna, passing, visp_units,
                               visp_passing],
    }

Usage:
    python scripts/20_05_26_filter_units_rf_pvalue.py
    python scripts/20_05_26_filter_units_rf_pvalue.py --threshold 0.005
    python scripts/20_05_26_filter_units_rf_pvalue.py --visp-only
"""
import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parent.parent
METRICS    = REPO_ROOT / 'data' / 'unit_analysis_metrics.pkl'
CACHE_DIR  = Path('/Users/pmccarthy/Documents/experimental_data/allen_visual_neuropixels_longwindow_5ms_bins')
SPIKES_DIR = CACHE_DIR / 'spike_times_v2'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--threshold', type=float, default=0.01,
                   help='p_value_rf upper bound, exclusive (default: 0.01)')
    p.add_argument('--metrics', type=Path, default=METRICS,
                   help=f'path to unit_analysis_metrics.pkl (default: {METRICS})')
    p.add_argument('--spikes-dir', type=Path, default=SPIKES_DIR,
                   help=f'path to per-session spike-time pickles (default: {SPIKES_DIR})')
    p.add_argument('--out', type=Path, default=None,
                   help='output pickle path (default: data/units_rf_pvalue_lt_<thr>.pkl)')
    p.add_argument('--visp-only', action='store_true',
                   help='restrict the mask to unit_ids that appear in the spike_times_v2 VISp pickles')
    return p.parse_args()


def load_metrics(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f'metrics file not found: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def visp_unit_ids_by_session(spikes_dir: Path) -> dict[int, np.ndarray]:
    """Map session_id -> array of unit_ids already extracted in spike_times_v2."""
    out: dict[int, np.ndarray] = {}
    if not spikes_dir.exists():
        return out
    for pkl in sorted(spikes_dir.glob('*_alllayers_spiketimes.pkl')):
        sid = int(pkl.stem.split('_')[0])
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        out[sid] = np.asarray(d['unit_ids']).astype(int)
    return out


def build_mask(metrics: dict[int, dict],
               threshold: float,
               visp_by_session: dict[int, np.ndarray],
               visp_only: bool) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    passing: dict[int, np.ndarray] = {}
    rows = []
    for sid, units in metrics.items():
        df = pd.DataFrame.from_dict(units, orient='index')
        df.index = df.index.astype(int)
        df.index.name = 'unit_id'

        pass_mask = df['p_value_rf'] < threshold
        pass_ids  = df.index[pass_mask].to_numpy()

        visp_ids = visp_by_session.get(sid, np.array([], dtype=int))
        visp_passing = np.intersect1d(visp_ids, pass_ids, assume_unique=False)

        # Choose what to write out per session
        passing[sid] = visp_passing if visp_only else pass_ids

        rows.append({
            'session_id':    sid,
            'total_units':   len(df),
            'rf_notna':      int(df['p_value_rf'].notna().sum()),
            'passing':       int(len(pass_ids)),
            'visp_units':    int(len(visp_ids)),
            'visp_passing':  int(len(visp_passing)),
        })

    summary = pd.DataFrame(rows).set_index('session_id').sort_index()
    return passing, summary


def main() -> None:
    args = parse_args()

    thr = args.threshold
    out_path = args.out or (REPO_ROOT / 'data' / f'units_rf_pvalue_lt_{str(thr).replace(".", "")}.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading metrics from {args.metrics} ...')
    metrics = load_metrics(args.metrics)
    print(f'  {len(metrics)} sessions, '
          f'{sum(len(u) for u in metrics.values()):,} units')

    print(f'Indexing VISp spike-time pickles in {args.spikes_dir} ...')
    visp_by_session = visp_unit_ids_by_session(args.spikes_dir)
    if visp_by_session:
        print(f'  {len(visp_by_session)} sessions with extracted VISp spike times '
              f'({sum(len(v) for v in visp_by_session.values()):,} VISp units total)')
    else:
        print('  (none found — VISp cross-reference will be empty)')
        if args.visp_only:
            print('  WARNING: --visp-only set but no spike-time pickles found; '
                  'the mask will be empty.')

    print(f'Applying filter: p_value_rf < {thr}'
          + (' (restricted to extracted VISp units)' if args.visp_only else ''))
    passing, summary = build_mask(metrics, thr, visp_by_session, args.visp_only)

    payload = {
        'threshold':        thr,
        'visp_only':        args.visp_only,
        'created_at':       datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'metrics_source':   str(args.metrics),
        'spikes_dir':       str(args.spikes_dir),
        'passing_unit_ids': passing,
        'summary':          summary,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    # ── Report ──────────────────────────────────────────────────────────────
    total_passing = sum(len(v) for v in passing.values())
    print('\nPer-session summary:')
    print(summary.to_string())
    print(f'\nTotal across {len(summary)} sessions:')
    print(f'  units in metrics             : {summary.total_units.sum():>8,}')
    print(f'  with non-NaN p_value_rf      : {summary.rf_notna.sum():>8,}')
    print(f'  passing p_value_rf < {thr:<5g}   : {summary.passing.sum():>8,}')
    if summary.visp_units.sum() > 0:
        print(f'  extracted VISp units         : {summary.visp_units.sum():>8,}')
        print(f'  VISp units passing filter    : {summary.visp_passing.sum():>8,}  '
              f'({100*summary.visp_passing.sum()/max(summary.visp_units.sum(),1):.1f}%)')
    print(f'\nWrote mask of {total_passing:,} unit_ids → {out_path}')


if __name__ == '__main__':
    main()
