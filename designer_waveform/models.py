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
from typing import Literal

import numpy as np
import scipy.stats as stats

from designer_waveform.optics import OpticsConfig, PowerToCurrentCurve
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

    def __init__(
        self,
        config: SimpleNamespace,
        optics: OpticsConfig | None = None,
        power_curve: PowerToCurrentCurve | None = None,
        normalization: Literal["max_expression", "pop_mean"] = "max_expression",
    ):
        """
        Args:
            config: Network and stimulus parameters from :func:`load_config`.
            optics: Optional :class:`~designer_waveform.optics.OpticsConfig`
                defining the optical path attenuation.  When provided together
                with ``power_curve``, the waveform is interpreted as source
                power (mW) and converted to per-neuron current via the full
                optical pipeline.  When ``None``, the waveform is treated as a
                dimensionless amplitude envelope (legacy behaviour).
            power_curve: Optional :class:`~designer_waveform.optics.PowerToCurrentCurve`
                mapping tissue irradiance (mW/mm²) to photocurrent (pA).
                Must be provided alongside ``optics`` to activate the optical
                pipeline.
            normalization: How the power-to-current curve relates to the
                per-neuron opsin expression distribution.

                ``"max_expression"`` — the curve gives the current received by
                a maximally-expressing neuron.  Per-neuron current is scaled by
                ``stim_dist_pA / I_max_pA``, which lives in [0, 1].

                ``"pop_mean"`` — the curve gives the population-mean current.
                Per-neuron current is scaled by
                ``stim_dist_pA / stim_dist_pA.mean()``, so above-mean neurons
                receive more than the curve value.  An explicit per-neuron clip
                at ``I_max_pA`` is applied to prevent unphysical values.
        """
        self.cfg = config
        self.optics = optics
        self.power_curve = power_curve
        self.normalization = normalization
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

    def _stim_to_current(self, stim_vals: np.ndarray) -> np.ndarray:
        """Convert waveform values to a (n_ts × N_exc) current matrix (pA).

        When ``optics`` and ``power_curve`` are both set, ``stim_vals`` is
        treated as source power (mW) and converted through the full optical
        pipeline.  Otherwise ``stim_vals`` is the dimensionless amplitude
        envelope and the original linear scaling is used.

        Returns:
            float32 array of shape ``(len(stim_vals), N_exc)`` in pA.
        """
        c = self.cfg
        if self.optics is not None and self.power_curve is not None:
            irradiance = self.optics.source_power_to_irradiance(stim_vals)
            mean_current = self.power_curve(irradiance)  # (n_ts,) pA
            if self.normalization == "max_expression":
                # curve = current at full opsin expression; scale by [0,1] fraction
                expr_weights = self._stim_dist_pA / c.I_max_pA
                out = np.outer(mean_current, expr_weights)
            else:  # pop_mean
                # curve = population-mean current; scale by relative expression
                rel_weights = self._stim_dist_pA / self._stim_dist_pA.mean()
                out = np.outer(mean_current, rel_weights)
                np.clip(out, 0, c.I_max_pA, out=out)
        else:
            out = np.outer(stim_vals, self._stim_dist_pA)
        return out.astype(np.float32)

    def run(
        self,
        waveform: Waveform,
        seed: int | None = None,
        vary_init_v: bool = True,
        vary_connectivity: bool = True,
        vary_weights: bool = True,
    ) -> dict:
        """Run one simulation with the given stimulation waveform.

        The waveform is evaluated on ``[0, t_stim_ms]``.  Its interpretation
        depends on whether an optical pipeline was configured at construction:

        *Dimensionless mode* (no ``optics``/``power_curve``): the waveform is
        a temporal envelope in [0, 1] and the current injected into neuron *i*
        at time *t* is::

            I_opto(t, i) = waveform(t) * stim_dist_pA[i]

        *Power mode* (``optics`` and ``power_curve`` both set): the waveform
        gives source optical power in mW.  The current is::

            irradiance(t)  = waveform(t) × total_transmission / area_mm2
            I_opto(t, i)   = power_curve(irradiance(t)) × expr_weight[i]

        where ``expr_weight`` is ``stim_dist_pA / I_max_pA`` for
        ``normalization="max_expression"`` or
        ``stim_dist_pA / stim_dist_pA.mean()`` (clipped at ``I_max_pA``)
        for ``normalization="pop_mean"``.

        The network is rebuilt from scratch on every call.  By default the
        seed from the config is used, making the call deterministic.  Pass a
        different ``seed`` to get an independent stochastic realisation.

        Each source of randomness can be independently toggled.  When a source
        is disabled it always uses ``cfg.seed``, so only the enabled sources
        differ across runs.

        Args:
            waveform: Waveform instance defining the stimulus envelope.
            seed: Base random seed.  ``None`` uses ``cfg.seed`` (fully
                deterministic).  Each source derives its own seed from this
                value so the three sources are independent.
            vary_init_v: If ``False``, initial membrane voltages are the same
                across all runs (uses ``cfg.seed``).
            vary_connectivity: If ``False``, synaptic connectivity pattern is
                the same across all runs.
            vary_weights: If ``False``, synaptic weight values are the same
                across all runs.

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
        timed_input[n_ts_pre : n_ts_pre + n_ts_stim, : c.N_exc] += (
            self._stim_to_current(stim_vals)
        )

        _base = c.seed if seed is None else seed
        # Each source gets an independent seed derived from the base.
        # Fixed sources always use c.seed so they don't vary across runs.
        _seed_v    = _base * 3     if vary_init_v      else c.seed
        _seed_conn = _base * 3 + 1 if vary_connectivity else c.seed
        _seed_w    = _base * 3 + 2 if vary_weights      else c.seed

        b2.start_scope()
        b2.seed(_seed_conn)   # Brian2 RNG: only used for syn.connect(p=...)
        b2.defaultclock.dt = dt

        rng_v = np.random.default_rng(_seed_v)
        rng_w = np.random.default_rng(_seed_w)
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
        neurons[: c.N_exc].v = (c.E_L_mV   + rng_v.uniform(-5, 5, c.N_exc)) * mV
        neurons[c.N_exc :].v = (c.E_inh_mV + rng_v.uniform(-5, 5, c.N_inh)) * mV
        neurons.g_exc = 0 * nS
        neurons.g_inh = 0 * nS

        def _w(mean, var, n):
            return np.clip(rng_w.normal(mean, var, n), 0, None) * nS

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
