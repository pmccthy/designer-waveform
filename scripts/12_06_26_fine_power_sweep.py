#!/usr/bin/env python3
"""
Fine low-power sweep — optimised waveform vs energy-matched comparisons.

Sweeps FINE_MIN_POWER_MW → FINE_MAX_POWER_MW with FINE_N_LEVELS levels.
Saves to the same results directory as the main power sweep script so both
result sets live together.

Run from the repo root:
    python scripts/12_06_26_fine_power_sweep.py
"""
# ── CONFIG ───────────────────────────────────────────────────────────────────
OPT_RESULT_PATH = 'results/optimised_waveforms/increasing_l23/l23/increasing_l23_l23_optimisation_result.pkl'

OPSIN = 'c1v1'   # 'c1v1' or 'chrmine'

FINE_MIN_POWER_MW = 0.01
FINE_MAX_POWER_MW = 0.5
FINE_N_LEVELS     = 15

PRE_SILENCE_MS  = 20.0
POST_SILENCE_MS = 100.0
BIN_SIZE_MS     = 10.0

N_RUNS    = 10
SEED_BASE = 1000
VARY_INIT_V       = True
VARY_CONNECTIVITY = True
VARY_WEIGHTS      = True

CMP_PT_FREQ_HZ      = 25.0
CMP_PT_PULSE_DUR_MS = 10.0

# ── IMPORTS ──────────────────────────────────────────────────────────────────
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

from designer_waveform.models import RandomEINetwork, load_config
from designer_waveform.optics import OpticsConfig, SigmoidPowerCurve
from designer_waveform.waveforms import RectangularPulseWaveform, PulseTrainWaveform

# ── LOAD WAVEFORM ─────────────────────────────────────────────────────────────
print(f'Loading waveform: {OPT_RESULT_PATH}')
with open(ROOT / OPT_RESULT_PATH, 'rb') as f:
    _opt = pickle.load(f)

_inner_wf      = _opt['opt_waveform']
_orig_stim_dur = float(_opt['stim_dur_ms'])
_source        = _opt['source']
_layer         = _opt['target_layer']

_t_eval           = np.linspace(0, _orig_stim_dur, 20_000)
_w_eval           = np.clip(_inner_wf(_t_eval), 0, None)
_peak_native      = float(_w_eval.max())
_energy_native    = float(np.trapz(_w_eval, _t_eval))
_avg_power_native = _energy_native / _orig_stim_dur

STIM_DUR_MS = PRE_SILENCE_MS + _orig_stim_dur + POST_SILENCE_MS
print(f'  {_source} / {_layer}  |  inner dur {_orig_stim_dur:.1f} ms  '
      f'peak {_peak_native:.4f} mW')

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
print(f'\nBuilding model (opsin: {OPSIN})')
cfg = load_config(ROOT / 'configs' / 'random_ei.json')
cfg.t_stim_ms   = STIM_DUR_MS
cfg.psth_bin_ms = BIN_SIZE_MS

_optics = OpticsConfig.from_file(ROOT / 'data' / 'optics_params.json')
_curve  = {'c1v1': SigmoidPowerCurve.c1v1, 'chrmine': SigmoidPowerCurve.chrmine}[OPSIN]()
model   = RandomEINetwork(cfg, optics=_optics, power_curve=_curve,
                          normalization='max_expression')
print(f'  N_exc={cfg.N_exc}, N_inh={cfg.N_inh}  |  '
      f'opsin mean {model._stim_dist_pA.mean():.1f} pA')

OUTPUT_DIR = ROOT / 'results' / 'power_sweep' / f'{_source}_{_layer.replace("/","_")}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'  output dir: {OUTPUT_DIR}')

# ── FACTORY FUNCTIONS ─────────────────────────────────────────────────────────
_pt_unit = PulseTrainWaveform(
    onset_ms=0.0, pulse_duration_ms=CMP_PT_PULSE_DUR_MS,
    frequency_hz=CMP_PT_FREQ_HZ, train_duration_ms=_orig_stim_dur, amplitude=1.0,
)
_pt_unit_energy = float(np.trapz(np.clip(_pt_unit(_t_eval), 0, None), _t_eval))

def _make_waveform(p_mw):
    _s = p_mw / _peak_native
    def _wf(t, _w=_inner_wf, _p=PRE_SILENCE_MS, _d=_orig_stim_dur, _sc=_s):
        t = np.asarray(t, dtype=float)
        mask = (t >= _p) & (t < _p + _d)
        out  = np.zeros_like(t)
        if mask.any():
            out[mask] = np.clip(_w(t[mask] - _p) * _sc, 0, None)
        return out
    return _wf

def _make_rect(p_mw):
    _amp  = _avg_power_native * (p_mw / _peak_native)
    _rect = RectangularPulseWaveform(onset_ms=0.0, duration_ms=_orig_stim_dur, amplitude=_amp)
    def _wf(t, _w=_rect, _p=PRE_SILENCE_MS, _d=_orig_stim_dur):
        t = np.asarray(t, dtype=float)
        mask = (t >= _p) & (t < _p + _d)
        out  = np.zeros_like(t)
        if mask.any():
            out[mask] = _w(t[mask] - _p)
        return out
    return _wf

def _make_pt(p_mw):
    _amp = _energy_native * (p_mw / _peak_native) / _pt_unit_energy
    _pt  = PulseTrainWaveform(
        onset_ms=0.0, pulse_duration_ms=CMP_PT_PULSE_DUR_MS,
        frequency_hz=CMP_PT_FREQ_HZ, train_duration_ms=_orig_stim_dur, amplitude=_amp,
    )
    def _wf(t, _w=_pt, _p=PRE_SILENCE_MS, _d=_orig_stim_dur):
        t = np.asarray(t, dtype=float)
        mask = (t >= _p) & (t < _p + _d)
        out  = np.zeros_like(t)
        if mask.any():
            out[mask] = _w(t[mask] - _p)
        return out
    return _wf

_makers = [('opt', _make_waveform), ('rect', _make_rect), ('pt', _make_pt)]
_t_wf   = np.linspace(0, STIM_DUR_MS, int(STIM_DUR_MS / 0.2))

# ── RUN SWEEP ─────────────────────────────────────────────────────────────────
fine_power_levels = np.linspace(FINE_MIN_POWER_MW, FINE_MAX_POWER_MW, FINE_N_LEVELS)
print(f'\nFine sweep: {FINE_N_LEVELS} levels  '
      f'{fine_power_levels[0]:.3f} – {fine_power_levels[-1]:.3f} mW')

t_psth_ms = model.run(_make_waveform(fine_power_levels[-1]))['t_psth_ms']

fine_sweep_results = []
for idx, p_mw in enumerate(fine_power_levels):
    print(f'  [{idx+1}/{FINE_N_LEVELS}] {p_mw:.4f} mW ...', flush=True)
    entry = {'power_mw': p_mw}
    for cname, maker in _makers:
        wf   = maker(p_mw)
        runs = []
        for _i in range(N_RUNS):
            _r = model.run(wf, seed=SEED_BASE + _i,
                           vary_init_v=VARY_INIT_V,
                           vary_connectivity=VARY_CONNECTIVITY,
                           vary_weights=VARY_WEIGHTS)
            runs.append(_r['psth_exc'] / (BIN_SIZE_MS / 1000.0))
        arr = np.stack(runs)
        entry[cname] = {
            'mean_hz': arr.mean(0),
            'sem_hz':  arr.std(0) / np.sqrt(N_RUNS),
            'peak_hz': float(arr.mean(0).max()),
        }
    fine_sweep_results.append(entry)
    print(f'       peak — opt: {entry["opt"]["peak_hz"]:.1f}  '
          f'rect: {entry["rect"]["peak_hz"]:.1f}  '
          f'pt: {entry["pt"]["peak_hz"]:.1f} Hz')

# ── OVERLAY PLOT ──────────────────────────────────────────────────────────────
cmap   = plt.cm.plasma
p_norm = plt.Normalize(fine_power_levels.min(), fine_power_levels.max())

_cond_info = [
    ('opt',  'Optimised',                   _make_waveform),
    ('rect', 'Energy-matched square pulse', _make_rect),
    ('pt',   f'Energy-matched pulse train\n({CMP_PT_FREQ_HZ:.0f} Hz, {CMP_PT_PULSE_DUR_MS:.0f} ms pulses)', _make_pt),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 6),
                          gridspec_kw={'height_ratios': [1, 2.5]},
                          sharex='col')
for col, (ckey, ctitle, maker) in enumerate(_cond_info):
    ax_wf, ax_psth = axes[0, col], axes[1, col]
    for p_mw in fine_power_levels:
        c = cmap(p_norm(p_mw))
        ax_wf.plot(_t_wf, maker(p_mw)(_t_wf), color=c, lw=1.2)
    ax_wf.set_xlim(0, STIM_DUR_MS)
    ax_wf.set_ylabel('Power (mW)')
    ax_wf.set_title(ctitle)
    ax_wf.tick_params(bottom=False, labelbottom=False)
    for res in fine_sweep_results:
        c = cmap(p_norm(res['power_mw']))
        d = res[ckey]
        ax_psth.fill_between(t_psth_ms, d['mean_hz'] - d['sem_hz'],
                             d['mean_hz'] + d['sem_hz'], color=c, alpha=0.15)
        ax_psth.plot(t_psth_ms, d['mean_hz'], color=c, lw=1.5)
    ax_psth.set_xlim(0, STIM_DUR_MS)
    ax_psth.set_xlabel('Time from stim onset (ms)')
axes[1, 0].set_ylabel('Firing rate (Hz)')
fig.suptitle(f'{_source} / {_layer}  —  fine low-power sweep  ({N_RUNS} runs each)', y=1.01)
fig.tight_layout()
fig.subplots_adjust(right=0.87)
cax = fig.add_axes([0.89, 0.1, 0.015, 0.8])
sm  = plt.cm.ScalarMappable(cmap=cmap, norm=p_norm)
sm.set_array([])
fig.colorbar(sm, cax=cax, label='Peak source power (mW)')
fig.savefig(OUTPUT_DIR / 'fine_power_sweep_overlay.png', dpi=150)
plt.close(fig)
print('  fine_power_sweep_overlay.png saved')

# ── HEATMAP + IO PLOT ─────────────────────────────────────────────────────────
_cond_cols = [
    ('opt',  'Optimised',                       'tomato'),
    ('rect', 'Energy-matched\nsquare pulse',     'steelblue'),
    ('pt',   f'Energy-matched\npulse train ({CMP_PT_FREQ_HZ:.0f} Hz)', 'seagreen'),
]

fig, axes = plt.subplots(1, 4, figsize=(22, 4),
                          gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
for ax, (ckey, ctitle, _) in zip(axes[:3], _cond_cols):
    heatmap = np.stack([r[ckey]['mean_hz'] for r in fine_sweep_results])
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   extent=[t_psth_ms[0], t_psth_ms[-1],
                           fine_power_levels[0], fine_power_levels[-1]],
                   cmap='inferno')
    fig.colorbar(im, ax=ax, label='Firing rate (Hz)', shrink=0.9)
    ax.set_xlabel('Time from stim onset (ms)')
    ax.set_ylabel('Peak source power (mW)')
    ax.set_title(ctitle)
ax = axes[3]
for ckey, ctitle, col in _cond_cols:
    peak_arr = np.array([r[ckey]['peak_hz'] for r in fine_sweep_results])
    ax.plot(fine_power_levels, peak_arr, 'o-', color=col, lw=1.8,
            label=ctitle.replace('\n', ' '))
ax.set_xlabel('Peak source power (mW)')
ax.set_ylabel('Peak firing rate (Hz)')
ax.set_title('Input–output (fine range)')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fine_power_sweep_heatmap.png', dpi=150)
plt.close(fig)
print('  fine_power_sweep_heatmap.png saved')

# ── SAVE ─────────────────────────────────────────────────────────────────────
_save = {
    'source':               _source,
    'target_layer':         _layer,
    'opsin':                OPSIN,
    'stim_dur_ms':          STIM_DUR_MS,
    'orig_stim_dur_ms':     _orig_stim_dur,
    'bin_size_ms':          BIN_SIZE_MS,
    'n_runs':               N_RUNS,
    'native_peak_mw':       _peak_native,
    'native_energy_mwms':   _energy_native,
    'native_avg_power_mw':  _avg_power_native,
    'pt_unit_energy_mwms':  _pt_unit_energy,
    't_psth_ms':            t_psth_ms,
    'fine_power_levels_mw': fine_power_levels,
    'fine_sweep_results':   fine_sweep_results,
}
_save_path = OUTPUT_DIR / 'fine_power_sweep_results.pkl'
with open(_save_path, 'wb') as f:
    pickle.dump(_save, f)
print(f'\nSaved → {_save_path}')
print('Done.')
