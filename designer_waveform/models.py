"""
Neural population models for waveform optimisation.

Each model exposes a ``run(waveform)`` method that accepts a
:class:`~designer_waveform.waveforms.Waveform` instance and returns a dict
containing at minimum a ``psth_exc`` array that can be used as the objective
target.

Config files are flat JSON.  Load them with :func:`load_config`, which
returns a :class:`~types.SimpleNamespace` so parameters are accessible with
dot notation (e.g. ``cfg.N_exc``).  Individual values can be overridden
directly::

    cfg = load_config("configs/random_ei_orawe_params.json")
    cfg.N_exc = 2000
    cfg.t_pre_ms = 200.0
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.stats as stats

from designer_waveform.waveforms import Waveform


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a JSON config file and return a nested :class:`~types.SimpleNamespace`.

    Nested dicts are converted recursively so all parameter groups are
    accessible with dot notation, e.g. ``cfg.membrane.C_m_pF``.

    Args:
        path: Path to the JSON config file.

    Returns:
        Nested SimpleNamespace mirroring the JSON structure.
    """
    with open(path) as f:
        data = json.load(f)
    return _dict_to_namespace(data)


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


class RandomEINetwork:
    """Random E/I spiking neural network with optogenetic stimulation.

    Membrane parameters and opsin distribution are matched to O'Rawe et al.
    2023.  The per-neuron opsin strength distribution is sampled once on
    construction and reused across all calls to :meth:`run`, so the network
    topology and expression pattern are held fixed while the waveform shape
    is varied during optimisation.

    Each call to :meth:`run` rebuilds the Brian2 network from scratch with
    the same seed, ensuring fully deterministic, comparable simulations.

    The config is a flat :class:`~types.SimpleNamespace` produced by
    :func:`load_config`.  Expected keys mirror the JSON config file — see
    ``configs/random_ei_orawe_params.json`` for the full list.

    Args:
        config: Nested SimpleNamespace from :func:`load_config`.

    Example:
        >>> cfg = load_config("configs/random_ei_orawe_params.json")
        >>> model = RandomEINetwork(cfg)
        >>> result = model.run(waveform)
        >>> psth = result["psth_exc"]
    """

    def __init__(self, config: SimpleNamespace):
        self.cfg = config
        self._stim_dist_pA = self._build_stim_dist()

    def _build_stim_dist(self) -> np.ndarray:
        """Sample per-neuron opsin strength distribution (pA).

        Returns:
            Array of shape ``(N_exc,)`` with per-neuron opto currents in pA.
        """
        c = self.cfg
        rng = np.random.default_rng(c.seed)

        dist = stats.lognorm.rvs(
            c.lognormal_s,
            loc=c.lognormal_loc,
            scale=c.lognormal_scale,
            size=c.N_exc,
            random_state=rng,
        )
        dist = np.clip(dist, 0, 1)

        n_zeros_existing = int(np.sum(dist == 0))
        n_zeros_needed = int(c.N_exc * c.sparsity) - n_zeros_existing
        if n_zeros_needed > 0:
            nonzero_idx = np.where(dist > 0)[0]
            silenced = rng.choice(nonzero_idx, size=n_zeros_needed, replace=False)
            dist[silenced] = 0.0

        return dist * c.I_max_pA

    def run(self, waveform: Waveform) -> dict:
        """Run one simulation with the given stimulation waveform.

        The waveform is evaluated on ``[0, t_stim_ms]`` and used as a
        dimensionless temporal envelope.  The actual current injected into
        excitatory neuron *i* at time *t* (relative to stimulus onset) is::

            I_opto(t, i) = waveform(t) * stim_dist_pA[i]

        The network is rebuilt from scratch with a fixed seed on every call,
        so results are deterministic and comparable across waveforms.

        Args:
            waveform: Waveform instance defining the stimulus envelope.

        Returns:
            dict with keys:

            - ``psth_exc`` — mean excitatory PSTH over the stimulation window,
              in spikes per neuron per bin.
            - ``t_psth_ms`` — bin centres relative to stimulus onset (ms).
            - ``spike_times_ms`` — all spike times (ms).
            - ``spike_indices`` — neuron index of each spike.
            - ``t_stim_ms`` — time axis used to evaluate the waveform (ms,
              relative to stimulus onset).
            - ``stim_vals`` — waveform values at each stim timestep.
        """
        import brian2 as b2
        from brian2 import ms, mV, nS, pA

        c = self.cfg

        dt = c.dt_ms * ms
        t_total = (c.t_pre_ms + c.t_stim_ms + c.t_post_ms) * ms

        n_ts_total = round((c.t_pre_ms + c.t_stim_ms + c.t_post_ms) / c.dt_ms)
        n_ts_pre = round(c.t_pre_ms / c.dt_ms)
        n_ts_stim = round(c.t_stim_ms / c.dt_ms)

        t_stim_arr = np.linspace(0.0, c.t_stim_ms, n_ts_stim)
        stim_vals = np.asarray(waveform(t_stim_arr), dtype=float)

        # Build timed-input array (n_ts_total × N_neurons) in pA
        timed_input = np.empty((n_ts_total, c.N_exc + c.N_inh), dtype=np.float32)
        timed_input[:, : c.N_exc] = c.I_bg_exc_pA
        timed_input[:, c.N_exc :] = c.I_bg_inh_pA
        timed_input[n_ts_pre : n_ts_pre + n_ts_stim, : c.N_exc] += np.outer(
            stim_vals, self._stim_dist_pA
        ).astype(np.float32)

        b2.start_scope()
        b2.seed(c.seed)
        b2.defaultclock.dt = dt

        rng = np.random.default_rng(c.seed)
        bgcurrent = b2.TimedArray(timed_input * pA, dt=dt)

        eqs = """
        dv/dt = (-g_L*(v - E_L) - g_exc*v - g_inh*(v - E_inh) + bgcurrent(t, i)) / C_m : volt (unless refractory)
        dg_exc/dt = -g_exc / tau_exc : siemens
        dg_inh/dt = -g_inh / tau_inh : siemens
        """

        C_m      = c.C_m_pF * b2.pfarad
        g_L      = c.g_L_nS * nS
        E_L      = c.E_L_mV * mV
        V_reset  = c.V_reset_mV * mV
        V_thresh = c.V_thresh_mV * mV
        tau_r    = c.tau_r_ms * ms
        E_inh    = c.E_inh_mV * mV
        tau_exc  = c.tau_exc_ms * ms
        tau_inh  = c.tau_inh_ms * ms

        neurons = b2.NeuronGroup(
            N=c.N_exc + c.N_inh,
            model=eqs,
            threshold="v > V_thresh",
            reset="v = V_reset",
            refractory=tau_r,
            method="euler",
        )
        neurons[: c.N_exc].v = "E_L + (rand() - 0.5) * 10*mV"
        neurons[c.N_exc :].v = "E_inh + (rand() - 0.5) * 10*mV"
        neurons.g_exc = 0 * nS
        neurons.g_inh = 0 * nS

        def _w(mean, var, n):
            return np.clip(rng.normal(mean, var, n), 0, None) * nS

        syn_ee = b2.Synapses(neurons[: c.N_exc], neurons[: c.N_exc],
                             model="w_ee : siemens", on_pre="g_exc += w_ee")
        syn_ee.connect(p=c.p_conn)
        syn_ee.w_ee = _w(c.w_ee_mean_nS, c.w_ee_var_nS, len(syn_ee))

        syn_ei = b2.Synapses(neurons[: c.N_exc], neurons[c.N_exc :],
                             model="w_ei : siemens", on_pre="g_exc += w_ei")
        syn_ei.connect(p=c.p_conn)
        syn_ei.w_ei = _w(c.w_ei_mean_nS, c.w_ei_var_nS, len(syn_ei))

        syn_ii = b2.Synapses(neurons[c.N_exc :], neurons[c.N_exc :],
                             model="w_ii : siemens", on_pre="g_inh += w_ii")
        syn_ii.connect(p=c.p_conn)
        syn_ii.w_ii = _w(c.w_ii_mean_nS, c.w_ii_var_nS, len(syn_ii))

        syn_ie = b2.Synapses(neurons[c.N_exc :], neurons[: c.N_exc],
                             model="w_ie : siemens", on_pre="g_inh += w_ie")
        syn_ie.connect(p=c.p_conn)
        syn_ie.w_ie = _w(c.w_ie_mean_nS, c.w_ie_var_nS, len(syn_ie))

        spike_mon = b2.SpikeMonitor(neurons)
        b2.run(t_total, report=None)

        spike_times_ms = np.array(spike_mon.t / ms)
        spike_indices = np.array(spike_mon.i)

        # PSTH for excitatory population over the stimulation window
        exc_times = spike_times_ms[spike_indices < c.N_exc]
        bin_edges = np.arange(
            c.t_pre_ms, c.t_pre_ms + c.t_stim_ms + c.psth_bin_ms, c.psth_bin_ms
        )
        counts, _ = np.histogram(exc_times, bins=bin_edges)

        return {
            "psth_exc": counts / c.N_exc,
            "t_psth_ms": 0.5 * (bin_edges[:-1] + bin_edges[1:]) - c.t_pre_ms,
            "spike_times_ms": spike_times_ms,
            "spike_indices": spike_indices,
            "t_stim_ms": t_stim_arr,
            "stim_vals": stim_vals,
        }
