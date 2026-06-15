"""
generate_chrmine_comparison_waveforms.py

Loads the optimised ChRmine waveform from the l23_chrmine result pkl and
produces three comparison waveforms, saved to data/template_waveforms.pkl
so they can be used directly in the simulate_template_responses notebook
(set WAVEFORM_SOURCE = 'template' and pick the relevant TEMPLATE_NAME).

All three waveforms are referenced to the 1 mW-peak rescaled version.

Waveforms generated
-------------------
  chrmine_opt_1mw_peak
      Same shape as the optimised waveform, rescaled so peak power = 1 mW.
      Duration: 250 ms.

  chrmine_opt_peak_matched_square
      Rectangular pulse with peak = 1 mW and duration chosen so that total
      energy equals the 1 mW waveform's energy.
          amplitude = 1 mW
          duration  = energy_1mw / 1.0 mW

  chrmine_opt_energy_dur_matched_square
      Rectangular pulse with the same duration (250 ms) and the same total
      energy as the 1 mW waveform, achieved via a lower flat amplitude.
          amplitude = energy_1mw / 250 ms
          duration  = 250 ms

All energy calculations use 1 kHz sampling (dt = 1 ms).
CSV columns: time_ms, power_mW
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
OPT_PKL       = (SCRIPT_DIR
                 / 'results/optimised_waveforms/increasing_l23/l23_chrmine'
                 / 'increasing_l23_l23_optimisation_result.pkl')
TEMPLATES_PKL = SCRIPT_DIR / 'data' / 'template_waveforms.pkl'

sys.path.insert(0, str(SCRIPT_DIR))
from designer_waveform.waveforms import (
    AsymBaselineSplitGaussianWaveform,
    RectangularPulseWaveform,
)

# ── Load optimised result ──────────────────────────────────────────────────
print(f'Loading optimised waveform from:\n  {OPT_PKL}')
with open(OPT_PKL, 'rb') as f:
    _opt = pickle.load(f)

inner_wf = _opt['opt_waveform']
stim_dur = float(_opt['stim_dur_ms'])   # 250 ms
print(f'  Waveform : {inner_wf}')
print(f'  Stim dur : {stim_dur:.0f} ms')

# ── Characterise native waveform at 1 kHz (dt = 1 ms) ─────────────────────
dt_ms   = 1.0
t_eval  = np.arange(int(round(stim_dur / dt_ms))) * dt_ms   # 0, 1, …, 249 ms
w_eval  = np.clip(inner_wf(t_eval), 0.0, None)

energy_native = float(np.trapz(w_eval, t_eval))   # mW·ms
peak_native   = float(w_eval.max())               # mW

print(f'\nNative waveform metrics (1 kHz):')
print(f'  peak power : {peak_native:.6f} mW')
print(f'  energy     : {energy_native:.4f} mW·ms')

# ── 1. Rescaled to 1 mW peak ──────────────────────────────────────────────
scale_1mw  = 1.0 / peak_native
energy_1mw = energy_native * scale_1mw   # mW·ms (scales linearly)

wf_1mw = AsymBaselineSplitGaussianWaveform(
    amplitude     = inner_wf.amplitude     * scale_1mw,
    mu            = inner_wf.mu,
    sigma_rise    = inner_wf.sigma_rise,
    sigma_fall    = inner_wf.sigma_fall,
    baseline_rise = inner_wf.baseline_rise * scale_1mw,
    baseline_fall = inner_wf.baseline_fall * scale_1mw,
)

_peak1 = float(np.clip(wf_1mw(t_eval), 0, None).max())
print(f'\n[1] Rescaled to 1 mW peak')
print(f'    {wf_1mw}')
print(f'    Verified peak   : {_peak1:.6f} mW  (target 1.000000)')
print(f'    Energy          : {energy_1mw:.4f} mW·ms')

# ── 2. Peak-matched square: amplitude = 1 mW, onset = 40 ms ──────────────
#    Duration chosen so energy matches the 1 mW waveform: dur = energy_1mw / 1.0
SQUARE_ONSET_MS = 40.0
peak_dur_ms     = energy_1mw / 1.0   # active pulse duration (ms)

wf_peak_sq = RectangularPulseWaveform(
    onset_ms    = SQUARE_ONSET_MS,
    duration_ms = peak_dur_ms,
    amplitude   = 1.0,
)

_e_sq2 = float(np.trapz(np.clip(wf_peak_sq(t_eval), 0, None), t_eval))
print(f'\n[2] Peak-matched square')
print(f'    {wf_peak_sq}')
print(f'    Peak   : 1.000000 mW  (onset at {SQUARE_ONSET_MS:.0f} ms)')
print(f'    Energy : {_e_sq2:.4f} mW·ms  (target {energy_1mw:.4f} mW·ms)')

# ── 3. Energy- and duration-matched square: onset = 40 ms, runs to 250 ms ─
#    Active window = stim_dur - SQUARE_ONSET_MS; amplitude raised accordingly
active_dur_ms = stim_dur - SQUARE_ONSET_MS   # 210 ms
mean_1mw      = energy_1mw / active_dur_ms   # mW — higher than stim_dur average

wf_endur_sq = RectangularPulseWaveform(
    onset_ms    = SQUARE_ONSET_MS,
    duration_ms = active_dur_ms,
    amplitude   = mean_1mw,
)

_e_sq3 = float(np.trapz(np.clip(wf_endur_sq(t_eval), 0, None), t_eval))
print(f'\n[3] Energy- and duration-matched square')
print(f'    {wf_endur_sq}')
print(f'    Peak   : {mean_1mw:.6f} mW  (onset at {SQUARE_ONSET_MS:.0f} ms, active for {active_dur_ms:.0f} ms)')
print(f'    Energy : {_e_sq3:.4f} mW·ms  (target {energy_1mw:.4f} mW·ms)')

# ── Build template entries ─────────────────────────────────────────────────
new_entries = {
    'chrmine_opt_1mw_peak': {
        'waveform'    : wf_1mw,
        'stim_dur_ms' : stim_dur,
        'description' : (
            f'Optimised ChRmine waveform rescaled to 1 mW peak '
            f'(scale={scale_1mw:.4f}x native; duration={stim_dur:.0f} ms)'
        ),
    },
    'chrmine_opt_peak_matched_square': {
        'waveform'    : wf_peak_sq,
        'stim_dur_ms' : stim_dur,
        'description' : (
            f'Peak-matched square: 1 mW, onset={SQUARE_ONSET_MS:.0f} ms, '
            f'pulse duration={peak_dur_ms:.2f} ms — same peak and total energy '
            f'as chrmine_opt_1mw_peak; zero before {SQUARE_ONSET_MS:.0f} ms'
        ),
    },
    'chrmine_opt_energy_dur_matched_square': {
        'waveform'    : wf_endur_sq,
        'stim_dur_ms' : stim_dur,
        'description' : (
            f'Energy+duration matched square: {mean_1mw:.4f} mW, '
            f'onset={SQUARE_ONSET_MS:.0f} ms, active for {active_dur_ms:.0f} ms — '
            f'same total energy and window as chrmine_opt_1mw_peak, '
            f'zero before {SQUARE_ONSET_MS:.0f} ms'
        ),
    },
}

# ── Save to template pkl ───────────────────────────────────────────────────
TEMPLATES_PKL.parent.mkdir(parents=True, exist_ok=True)
if TEMPLATES_PKL.exists():
    with open(TEMPLATES_PKL, 'rb') as f:
        templates = pickle.load(f)
    print(f'\nLoaded existing templates: {list(templates.keys())}')
else:
    templates = {}
    print('\nNo existing template file — creating new one.')

templates.update(new_entries)
with open(TEMPLATES_PKL, 'wb') as f:
    pickle.dump(templates, f)
print(f'\nSaved {len(new_entries)} waveforms to:\n  {TEMPLATES_PKL}')
print(f'All templates: {list(templates.keys())}')

# ── Export CSVs at 1 kHz (columns: time_ms, power_mW) ────────────────────
CSV_DIR = SCRIPT_DIR / 'results' / 'optimised_waveforms' / 'increasing_l23' / 'l23_chrmine'
CSV_DIR.mkdir(parents=True, exist_ok=True)

csv_waveforms = [
    ('chrmine_opt_1mw_peak',                 wf_1mw),
    ('chrmine_opt_peak_matched_square',       wf_peak_sq),
    ('chrmine_opt_energy_dur_matched_square', wf_endur_sq),
]

print('\nWriting CSVs (1 kHz, columns: time_ms [ms], power_mW [mW]):')
for name, wf in csv_waveforms:
    t_csv = np.arange(0, stim_dur, dt_ms)   # full 0–249 ms window
    p_csv = wf(t_csv)
    # Append trailing zero at stim_dur to show any falling edge
    t_csv = np.append(t_csv, stim_dur)
    p_csv = np.append(p_csv, 0.0)
    csv_path = CSV_DIR / f'{name}.csv'
    # Header comment makes units explicit; second row is column names
    with open(csv_path, 'w') as fh:
        fh.write('# Units: time_ms = milliseconds, power_mW = milliwatts\n')
        fh.write('time_ms,power_mW\n')
        for t, p in zip(t_csv, p_csv):
            fh.write(f'{t:.6f},{p:.6f}\n')
    print(f'  {csv_path.name}  ({len(t_csv)} samples, incl. trailing zero)')

# ── Sanity plot ────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _style = SCRIPT_DIR / 'configs' / 'mpl.mplstyle'
    if _style.exists():
        plt.style.use(_style)

    _t = np.linspace(0, stim_dur, int(stim_dur / 0.1))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(_t, wf_1mw(_t),      color='steelblue', lw=2,
            label='Rescaled to 1 mW peak')
    ax.plot(_t, wf_peak_sq(_t),  color='tomato',    lw=2,
            label=f'Peak-matched square (onset {SQUARE_ONSET_MS:.0f} ms, {peak_dur_ms:.1f} ms pulse, 1 mW)')
    ax.plot(_t, wf_endur_sq(_t), color='seagreen',  lw=2,
            label=f'Energy+dur matched square (onset {SQUARE_ONSET_MS:.0f} ms, {mean_1mw:.4f} mW)')
    ax.set_xlabel('Time from waveform onset (ms)')
    ax.set_ylabel('Source power (mW)')
    ax.set_title('ChRmine comparison waveforms (all referenced to 1 mW peak)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    plot_path = CSV_DIR / 'comparison_waveforms.png'
    fig.savefig(plot_path, dpi=150)
    print(f'\nPlot saved to:\n  {plot_path}')
except Exception as _e:
    print(f'\n(Plot skipped: {_e})')

print('\nDone. In the simulate notebook, set:')
print('  WAVEFORM_SOURCE = "template"')
print('  TEMPLATE_NAME   = "chrmine_opt_1mw_peak"')
print('                    "chrmine_opt_peak_matched_square"')
print('                    "chrmine_opt_energy_dur_matched_square"')
