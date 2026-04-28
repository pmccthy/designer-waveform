"""
Lognormal istim simulation with membrane parameters and opsin distribution
matched to O'Rawe et al. 2023 (corticalSuppressionRepo):
  - C_m = 200 pF  (vs 900 pF in current scripts)
  - tau_r = 5 ms  (vs 2 ms)
  - lognormal fit params from 2p data: s=0.6324, loc=0.0061, scale=0.2387

Pulse structure kept the same as the other scripts (3 x 200 ms pulses).

Results are saved to: results/random_ei_snn_pulses_lognormal_istim_orawe_params/<timestamp>/
"""

import shutil
import numpy as np
import scipy.stats as stats
import brian2 as b2
from brian2 import ms, mV, nS, pA
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = (Path(__file__).parents[1] / "results"
               / "random_ei_snn_pulses_lognormal_istim_orawe_params" / timestamp)
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
seed = 20250319
rng = np.random.default_rng(seed)
np.random.seed(seed)
b2.seed(seed)

# Network
N_exc = 8000
N_inh = 2000
p_conn = 0.02

# Membrane — matched to O'Rawe et al. 2023
C_m = 200.0 * b2.pfarad
g_L = 10.0 * nS
E_L = -60 * mV
V_reset = -60 * mV
V_thresh = -50 * mV
tau_r = 5 * ms

# Synapses
E_inh = -80 * mV
tau_exc = 5 * ms
tau_inh = 10 * ms
w_ee_mean = 0.4 * nS;  w_ee_var = 0.4 * nS
w_ei_mean = 0.8 * nS;  w_ei_var = 0.8 * nS
w_ii_mean = 4.0 * nS;  w_ii_var = 4.0 * nS
w_ie_mean = 5.0 * nS;  w_ie_var = 5.0 * nS
I_bg_exc = 260  # pA
I_bg_inh = 140  # pA

# Opto: lognormal fit params from O'Rawe et al. 2023 2p data
#   scipy.stats.lognorm parameterisation: lognorm(s, loc, scale)
I_stim_lognormal_s     = 0.6324531774890443
I_stim_lognormal_loc   = 0.006100072101138047
I_stim_lognormal_scale = 0.23865410791827243
I_stim_sparsity = 0.168
I_max = 1700  # pA

# Simulation timing
dt = 0.1 * ms
t_pre    = 500 * ms
t_pulse  = 200 * ms
t_gap    = 200 * ms
n_pulses = 3
t_post   = 400 * ms
t_total  = t_pre + n_pulses * t_pulse + (n_pulses - 1) * t_gap + t_post

n_ts_total = int(t_total / dt)
n_ts_pre   = int(t_pre   / dt)
n_ts_pulse = int(t_pulse / dt)
n_ts_gap   = int(t_gap   / dt)

# ---------------------------------------------------------------------------
# Build per-neuron opto strength distribution (O'Rawe et al. 2023)
# ---------------------------------------------------------------------------
stim_dist = stats.lognorm.rvs(
    I_stim_lognormal_s,
    loc=I_stim_lognormal_loc,
    scale=I_stim_lognormal_scale,
    size=N_exc,
)
stim_dist = np.clip(stim_dist, 0, 1)

n_zeros_existing = int(np.sum(stim_dist == 0))
n_zeros_needed = int(N_exc * I_stim_sparsity) - n_zeros_existing
if n_zeros_needed > 0:
    nonzero_idx = np.where(stim_dist > 0)[0]
    silenced = np.random.choice(nonzero_idx, size=n_zeros_needed, replace=False)
    stim_dist[silenced] = 0.0

stim_dist_pA = stim_dist * I_max

print(f"Opto distribution — mean: {stim_dist_pA.mean():.1f} pA, "
      f"max: {stim_dist_pA.max():.1f} pA, "
      f"frac zero: {(stim_dist_pA == 0).mean():.3f}")

# ---------------------------------------------------------------------------
# Build timed_input array  (n_ts_total x (N_exc + N_inh))  in pA
# ---------------------------------------------------------------------------
timed_input = np.ones((n_ts_total, N_exc + N_inh))
timed_input[:, :N_exc] *= I_bg_exc
timed_input[:, N_exc:] *= I_bg_inh

cursor = n_ts_pre
for _ in range(n_pulses):
    timed_input[cursor : cursor + n_ts_pulse, :N_exc] += stim_dist_pA
    cursor += n_ts_pulse + n_ts_gap

# ---------------------------------------------------------------------------
# Build Brian2 network
# ---------------------------------------------------------------------------
b2.start_scope()
b2.defaultclock.dt = dt

bgcurrent = b2.TimedArray(timed_input * pA, dt=dt)

eqs = """
dv/dt = (-g_L*(v - E_L) - g_exc*v - g_inh*(v - E_inh) + bgcurrent(t, i)) / C_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
"""

neurons = b2.NeuronGroup(
    N=N_exc + N_inh,
    model=eqs,
    threshold="v > V_thresh",
    reset="v = V_reset",
    refractory=tau_r,
    method="euler",
    name="neurons",
)

neurons_exc = neurons[:N_exc]
neurons_inh = neurons[N_exc:]
neurons_exc.v = "E_L + (rand() - 0.5) * 10*mV"
neurons_inh.v = "E_inh + (rand() - 0.5) * 10*mV"
neurons.g_exc = 0 * nS
neurons.g_inh = 0 * nS

# exc -> exc
syn_ee = b2.Synapses(neurons_exc, neurons_exc, model="w_ee : siemens",
                     on_pre="g_exc += w_ee", method="euler", name="syn_ee")
syn_ee.connect(p=p_conn)
syn_ee.w_ee = np.clip(rng.normal(w_ee_mean/nS, w_ee_var/nS, len(syn_ee)), 0, None) * nS

# exc -> inh
syn_ei = b2.Synapses(neurons_exc, neurons_inh, model="w_ei : siemens",
                     on_pre="g_exc += w_ei", method="euler", name="syn_ei")
syn_ei.connect(p=p_conn)
syn_ei.w_ei = np.clip(rng.normal(w_ei_mean/nS, w_ei_var/nS, len(syn_ei)), 0, None) * nS

# inh -> inh
syn_ii = b2.Synapses(neurons_inh, neurons_inh, model="w_ii : siemens",
                     on_pre="g_inh += w_ii", method="euler", name="syn_ii")
syn_ii.connect(p=p_conn)
syn_ii.w_ii = np.clip(rng.normal(w_ii_mean/nS, w_ii_var/nS, len(syn_ii)), 0, None) * nS

# inh -> exc
syn_ie = b2.Synapses(neurons_inh, neurons_exc, model="w_ie : siemens",
                     on_pre="g_inh += w_ie", method="euler", name="syn_ie")
syn_ie.connect(p=p_conn)
syn_ie.w_ie = np.clip(rng.normal(w_ie_mean/nS, w_ie_var/nS, len(syn_ie)), 0, None) * nS

# Monitors
spike_mon = b2.SpikeMonitor(neurons, name="spike_mon")
state_mon = b2.StateMonitor(neurons, ["v", "g_exc", "g_inh"], record=True,
                             name="state_mon", dt=1 * ms)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print("Running simulation...")
b2.run(t_total, report="text")
print(f"Simulation complete. Total simulated time: {b2.defaultclock.t / ms:.1f} ms")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.savez(
    results_dir / "spike_monitor.npz",
    spike_times=spike_mon.t / ms,
    spike_indices=np.array(spike_mon.i),
)

np.savez(
    results_dir / "state_monitor.npz",
    t=state_mon.t / ms,
    V=state_mon.v / mV,
    g_exc=state_mon.g_exc / nS,
    g_inh=state_mon.g_inh / nS,
)

np.savez(
    results_dir / "stim_waveform.npz",
    timed_input_pA=timed_input,
    dt_ms=dt / ms,
    n_ts_total=n_ts_total,
    t_pre_ms=t_pre / ms,
    t_pulse_ms=t_pulse / ms,
    t_gap_ms=t_gap / ms,
    n_pulses=n_pulses,
    t_post_ms=t_post / ms,
    t_total_ms=t_total / ms,
)

np.savez(
    results_dir / "params.npz",
    N_exc=N_exc,
    N_inh=N_inh,
    p_conn=p_conn,
    seed=seed,
    C_m_pF=200.0,
    g_L_nS=10.0,
    tau_r_ms=5.0,
    I_stim_lognormal_s=I_stim_lognormal_s,
    I_stim_lognormal_scale=I_stim_lognormal_scale,
    I_stim_lognormal_loc=I_stim_lognormal_loc,
    I_stim_sparsity=I_stim_sparsity,
    I_max_pA=I_max,
)

np.savez(
    results_dir / "stim_dist.npz",
    stim_dist=stim_dist,
    stim_dist_pA=stim_dist_pA,
)

shutil.copy2(__file__, results_dir / Path(__file__).name)
print(f"Results saved to: {results_dir}")
