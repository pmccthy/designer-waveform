#!/usr/bin/env python3
"""
Compare optimised-waveform multirun PSTHs for C1V1 and ChRmine.

Shows cross-run variability (individual traces + mean ± SEM) for the model,
and the target mean only — no target SEM band.

Run from the repo root:
    python scripts/12_06_26_compare_opsin_psth.py
"""
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
plt.style.use(ROOT / 'configs' / 'mpl.mplstyle')

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS = {
    'C1V1': ROOT / 'results/optimised_waveforms/increasing_l23/l23/increasing_l23_l23_optimisation_result.pkl',
    'ChRmine': ROOT / 'results/optimised_waveforms/increasing_l23/l23_chrmine/increasing_l23_l23_optimisation_result.pkl',
}

COLORS   = {'C1V1': 'tomato', 'ChRmine': 'mediumpurple'}
OUT_DIR  = ROOT / 'results' / 'opsin_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
data = {}
for label, path in RESULTS.items():
    if not path.exists():
        print(f'  ⚠  {label} result not found — skipping: {path}')
        continue
    with open(path, 'rb') as f:
        data[label] = pickle.load(f)
    d = data[label]
    n = d.get('n_runs', d['multirun_psth_hz'].shape[0])
    print(f'  {label}: {n} runs, bin={d["psth_bin_ms"]:.0f} ms, '
          f'peak target={d["target_hz"].max():.2f} Hz')

if not data:
    raise RuntimeError('No result files found.')

# ── FIGURE ───────────────────────────────────────────────────────────────────
n_panels = len(data)
fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5), sharey=False)
if n_panels == 1:
    axes = [axes]

for ax, (label, d) in zip(axes, data.items()):
    col         = COLORS[label]
    bin_ms      = float(d['psth_bin_ms'])
    t_psth      = d.get('t_psth_ms',
                         np.arange(d['multirun_psth_hz'].shape[1]) * bin_ms + bin_ms / 2)
    t_target    = d['target_t_ms']
    target_hz   = d['target_hz']

    # multirun array: rows = runs, cols = bins  (already in Hz)
    runs_hz     = d['multirun_psth_hz']          # shape (N_runs, n_bins)
    mean_hz     = d['multirun_mean_hz']
    sem_hz      = d['multirun_sem_hz']

    # individual runs (thin, transparent)
    for run in runs_hz:
        ax.plot(t_psth, run, color=col, lw=0.6, alpha=0.15)

    # mean ± SEM band
    ax.fill_between(t_psth, mean_hz - sem_hz, mean_hz + sem_hz,
                    color=col, alpha=0.35)

    # mean line
    ax.plot(t_psth, mean_hz, color=col, lw=2.0,
            label=f'{label} mean ± SEM  (n={runs_hz.shape[0]})')

    # target — mean only, no band
    ax.plot(t_target, target_hz, color='k', lw=1.8, ls='--', label='Target (Allen mean)')

    ax.set_xlabel('Time from stim onset (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(label)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

source = next(iter(data.values())).get('source', '')
fig.suptitle(f'{source}  —  optimised waveform multirun PSTH', y=1.01)
fig.tight_layout()

out_path = OUT_DIR / 'opsin_comparison_multirun_psth.png'
fig.savefig(out_path, dpi=150)
print(f'\nSaved → {out_path}')

# ── OVERLAY (both opsins on one panel) ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

_target_plotted = False
for label, d in data.items():
    col      = COLORS[label]
    bin_ms   = float(d['psth_bin_ms'])
    t_psth   = d.get('t_psth_ms',
                      np.arange(d['multirun_psth_hz'].shape[1]) * bin_ms + bin_ms / 2)
    runs_hz  = d['multirun_psth_hz']
    mean_hz  = d['multirun_mean_hz']
    sem_hz   = d['multirun_sem_hz']

    for run in runs_hz:
        ax2.plot(t_psth, run, color=col, lw=0.6, alpha=0.12)
    ax2.fill_between(t_psth, mean_hz - sem_hz, mean_hz + sem_hz, color=col, alpha=0.3)
    ax2.plot(t_psth, mean_hz, color=col, lw=2.0,
             label=f'{label} (n={runs_hz.shape[0]})')

    if not _target_plotted:
        ax2.plot(d['target_t_ms'], d['target_hz'], color='k', lw=1.8, ls='--',
                 label='Target (Allen mean)')
        _target_plotted = True

ax2.set_xlabel('Time from stim onset (ms)')
ax2.set_ylabel('Firing rate (Hz)')
ax2.set_title(f'{source}  —  C1V1 vs ChRmine')
ax2.legend(frameon=False, fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)
fig2.tight_layout()

out_path2 = OUT_DIR / 'opsin_comparison_overlay.png'
fig2.savefig(out_path2, dpi=150)
print(f'Saved → {out_path2}')
print('Done.')
