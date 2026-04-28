"""
Neural population models for waveform optimisation.

Each model exposes a ``run(waveform)`` method that accepts a
:class:`~designer_waveform.waveforms.Waveform` instance and returns a dict
containing at minimum a ``psth_exc`` array that can be used as the objective
target.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.stats as stats

from designer_waveform.waveforms import Waveform


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NetworkParams:
    """Size and connectivity of the random E/I network.

    Attributes:
        N_exc: Number of excitatory neurons.
        N_inh: Number of inhibitory neurons.
        p_conn: All-to-all connection probability.
    """

    N_exc: int = 8000
    N_inh: int = 2000
    p_conn: float = 0.02


@dataclass
class MembraneParams:
    """Single-compartment LIF membrane parameters.

    Attributes:
        C_m_pF: Membrane capacitance (pF).
        g_L_nS: Leak conductance (nS).
        E_L_mV: Leak reversal potential (mV).
        V_reset_mV: Post-spike reset voltage (mV).
        V_thresh_mV: Spike threshold (mV).
        tau_r_ms: Refractory period (ms).
    """

    C_m_pF: float = 200.0
    g_L_nS: float = 10.0
    E_L_mV: float = -60.0
    V_reset_mV: float = -60.0
    V_thresh_mV: float = -50.0
    tau_r_ms: float = 5.0


@dataclass
class SynapseParams:
    """Synaptic parameters and background drive.

    Attributes:
        E_inh_mV: Inhibitory reversal potential (mV).
        tau_exc_ms: Excitatory conductance decay time constant (ms).
        tau_inh_ms: Inhibitory conductance decay time constant (ms).
        w_ee_mean_nS: E→E mean synaptic weight (nS).
        w_ee_var_nS: E→E weight standard deviation (nS).
        w_ei_mean_nS: E→I mean synaptic weight (nS).
        w_ei_var_nS: E→I weight standard deviation (nS).
        w_ii_mean_nS: I→I mean synaptic weight (nS).
        w_ii_var_nS: I→I weight standard deviation (nS).
        w_ie_mean_nS: I→E mean synaptic weight (nS).
        w_ie_var_nS: I→E weight standard deviation (nS).
        I_bg_exc_pA: Tonic background current to excitatory neurons (pA).
        I_bg_inh_pA: Tonic background current to inhibitory neurons (pA).
    """

    E_inh_mV: float = -80.0
    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0
    w_ee_mean_nS: float = 0.4
    w_ee_var_nS: float = 0.4
    w_ei_mean_nS: float = 0.8
    w_ei_var_nS: float = 0.8
    w_ii_mean_nS: float = 4.0
    w_ii_var_nS: float = 4.0
    w_ie_mean_nS: float = 5.0
    w_ie_var_nS: float = 5.0
    I_bg_exc_pA: float = 260.0
    I_bg_inh_pA: float = 140.0


@dataclass
class OptoParams:
    """Optogenetic stimulation parameters.

    The per-neuron opsin expression level is drawn from a lognormal
    distribution fit to 2-photon data (O'Rawe et al. 2023), clipped to
    [0, 1], then scaled by ``I_max_pA``.  A fraction ``sparsity`` of neurons
    are silenced (weight set to zero) to match the measured expression
    sparsity.

    The waveform passed to :meth:`RandomEINetwork.run` acts as a
    dimensionless temporal envelope: actual current to neuron *i* at time *t*
    is ``waveform(t) * stim_dist_pA[i]``.  An envelope amplitude of 1.0
    therefore corresponds to a peak drive of ``I_max_pA`` for the
    most-strongly-coupled neuron.

    Attributes:
        lognormal_s: Shape parameter of the lognormal (scipy ``s``).
        lognormal_loc: Location parameter of the lognormal.
        lognormal_scale: Scale parameter of the lognormal.
        sparsity: Fraction of excitatory neurons with zero opsin expression.
        I_max_pA: Peak opto current for a neuron with relative weight 1 (pA).
    """

    lognormal_s: float = 0.6324531774890443
    lognormal_loc: float = 0.006100072101138047
    lognormal_scale: float = 0.23865410791827243
    sparsity: float = 0.168
    I_max_pA: float = 1700.0


@dataclass
class SimulationParams:
    """Timing and numerical integration parameters.

    Attributes:
        dt_ms: Integration timestep (ms).
        t_pre_ms: Pre-stimulus period (ms).
        t_stim_ms: Stimulation window duration (ms).
        t_post_ms: Post-stimulus period (ms).
        psth_bin_ms: PSTH bin width (ms).
        seed: Master random seed for network construction and simulation.
    """

    dt_ms: float = 0.1
    t_pre_ms: float = 500.0
    t_stim_ms: float = 200.0
    t_post_ms: float = 400.0
    psth_bin_ms: float = 5.0
    seed: int = 20250319


@dataclass
class RandomEINetworkConfig:
    """Full configuration for :class:`RandomEINetwork`.

    Attributes:
        network: Network size and connectivity.
        membrane: Membrane biophysics.
        synapses: Synaptic weights and background drive.
        opto: Optogenetic stimulation distribution.
        simulation: Timing and numerical integration.
    """

    network: NetworkParams = field(default_factory=NetworkParams)
    membrane: MembraneParams = field(default_factory=MembraneParams)
    synapses: SynapseParams = field(default_factory=SynapseParams)
    opto: OptoParams = field(default_factory=OptoParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)

    @classmethod
    def from_toml(cls, path: str | Path) -> RandomEINetworkConfig:
        """Load config from a TOML file.

        Args:
            path: Path to the TOML file.

        Returns:
            Populated :class:`RandomEINetworkConfig` instance.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            network=NetworkParams(**data.get("network", {})),
            membrane=MembraneParams(**data.get("membrane", {})),
            synapses=SynapseParams(**data.get("synapses", {})),
            opto=OptoParams(**data.get("opto", {})),
            simulation=SimulationParams(**data.get("simulation", {})),
        )


# ---------------------------------------------------------------------------
# Network model
# ---------------------------------------------------------------------------


class RandomEINetwork:
    """Random E/I spiking neural network with optogenetic stimulation.

    Membrane parameters and opsin distribution are matched to O'Rawe et al.
    2023.  The per-neuron opsin strength distribution is sampled once on
    construction and reused across all calls to :meth:`run`, so the network
    topology and expression pattern are held fixed while the waveform shape
    is varied during optimisation.

    Each call to :meth:`run` rebuilds the Brian2 network from scratch with
    the same seed, ensuring fully deterministic, comparable simulations.

    Args:
        config: :class:`RandomEINetworkConfig` instance.  Defaults to the
            O'Rawe et al. 2023 parameters if not provided.

    Example:
        >>> from pathlib import Path
        >>> cfg = RandomEINetworkConfig.from_toml(
        ...     Path("configs/random_ei_orawe_params.toml")
        ... )
        >>> model = RandomEINetwork(cfg)
        >>> result = model.run(waveform)
        >>> psth = result["psth_exc"]
    """

    def __init__(self, config: RandomEINetworkConfig | None = None):
        self.cfg = config or RandomEINetworkConfig()
        self._stim_dist_pA = self._build_stim_dist()

    def _build_stim_dist(self) -> np.ndarray:
        """Sample per-neuron opsin strength distribution.

        Returns:
            Array of shape ``(N_exc,)`` with per-neuron opto currents in pA.
        """
        opto = self.cfg.opto
        net = self.cfg.network
        rng = np.random.default_rng(self.cfg.simulation.seed)

        dist = stats.lognorm.rvs(
            opto.lognormal_s,
            loc=opto.lognormal_loc,
            scale=opto.lognormal_scale,
            size=net.N_exc,
            random_state=rng,
        )
        dist = np.clip(dist, 0, 1)

        n_zeros_existing = int(np.sum(dist == 0))
        n_zeros_needed = int(net.N_exc * opto.sparsity) - n_zeros_existing
        if n_zeros_needed > 0:
            nonzero_idx = np.where(dist > 0)[0]
            silenced = rng.choice(nonzero_idx, size=n_zeros_needed, replace=False)
            dist[silenced] = 0.0

        return dist * opto.I_max_pA

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

            - ``psth_exc`` (ndarray): Mean excitatory PSTH over the
              stimulation window, in spikes per neuron per bin.
            - ``t_psth_ms`` (ndarray): Bin centres relative to stimulus
              onset (ms).
            - ``spike_times_ms`` (ndarray): All spike times (ms).
            - ``spike_indices`` (ndarray): Neuron index of each spike.
            - ``t_stim_ms`` (ndarray): Time axis used to evaluate the
              waveform (ms, relative to stimulus onset).
            - ``stim_vals`` (ndarray): Waveform values at each stim timestep.
        """
        import brian2 as b2
        from brian2 import ms, mV, nS, pA

        cfg = self.cfg
        net = cfg.network
        mem = cfg.membrane
        syn = cfg.synapses
        sim = cfg.simulation

        dt = sim.dt_ms * ms
        t_pre = sim.t_pre_ms * ms
        t_stim = sim.t_stim_ms * ms
        t_post = sim.t_post_ms * ms
        t_total = t_pre + t_stim + t_post

        n_ts_total = round(float(t_total / ms) / sim.dt_ms)
        n_ts_pre = round(sim.t_pre_ms / sim.dt_ms)
        n_ts_stim = round(sim.t_stim_ms / sim.dt_ms)

        # Evaluate waveform on the stimulation time axis
        t_stim_arr = np.linspace(0.0, sim.t_stim_ms, n_ts_stim)
        stim_vals = np.asarray(waveform(t_stim_arr), dtype=float)

        # Build full timed-input array  (n_ts_total × N_neurons)  in pA
        # Using float32 to halve memory relative to float64.
        timed_input = np.empty((n_ts_total, net.N_exc + net.N_inh), dtype=np.float32)
        timed_input[:, : net.N_exc] = syn.I_bg_exc_pA
        timed_input[:, net.N_exc :] = syn.I_bg_inh_pA

        # Add opto: outer product of temporal envelope and per-neuron strengths
        opto_slice = np.outer(stim_vals, self._stim_dist_pA).astype(np.float32)
        timed_input[n_ts_pre : n_ts_pre + n_ts_stim, : net.N_exc] += opto_slice

        # ------------------------------------------------------------------
        # Build Brian2 network
        # ------------------------------------------------------------------
        b2.start_scope()
        b2.seed(sim.seed)
        b2.defaultclock.dt = dt

        rng = np.random.default_rng(sim.seed)

        bgcurrent = b2.TimedArray(timed_input * pA, dt=dt)

        C_m = mem.C_m_pF * b2.pfarad
        g_L = mem.g_L_nS * nS
        E_L = mem.E_L_mV * mV
        V_reset = mem.V_reset_mV * mV
        V_thresh = mem.V_thresh_mV * mV
        tau_r = mem.tau_r_ms * ms
        E_inh = syn.E_inh_mV * mV
        tau_exc = syn.tau_exc_ms * ms
        tau_inh = syn.tau_inh_ms * ms

        eqs = """
        dv/dt = (-g_L*(v - E_L) - g_exc*v - g_inh*(v - E_inh) + bgcurrent(t, i)) / C_m : volt (unless refractory)
        dg_exc/dt = -g_exc / tau_exc : siemens
        dg_inh/dt = -g_inh / tau_inh : siemens
        """

        neurons = b2.NeuronGroup(
            N=net.N_exc + net.N_inh,
            model=eqs,
            threshold="v > V_thresh",
            reset="v = V_reset",
            refractory=tau_r,
            method="euler",
        )

        neurons_exc = neurons[: net.N_exc]
        neurons_inh = neurons[net.N_exc :]
        neurons_exc.v = "E_L + (rand() - 0.5) * 10*mV"
        neurons_inh.v = "E_inh + (rand() - 0.5) * 10*mV"
        neurons.g_exc = 0 * nS
        neurons.g_inh = 0 * nS

        def _weights(mean_nS, var_nS, n):
            return np.clip(rng.normal(mean_nS, var_nS, n), 0, None) * nS

        syn_ee = b2.Synapses(
            neurons_exc, neurons_exc, model="w_ee : siemens", on_pre="g_exc += w_ee"
        )
        syn_ee.connect(p=net.p_conn)
        syn_ee.w_ee = _weights(syn.w_ee_mean_nS, syn.w_ee_var_nS, len(syn_ee))

        syn_ei = b2.Synapses(
            neurons_exc, neurons_inh, model="w_ei : siemens", on_pre="g_exc += w_ei"
        )
        syn_ei.connect(p=net.p_conn)
        syn_ei.w_ei = _weights(syn.w_ei_mean_nS, syn.w_ei_var_nS, len(syn_ei))

        syn_ii = b2.Synapses(
            neurons_inh, neurons_inh, model="w_ii : siemens", on_pre="g_inh += w_ii"
        )
        syn_ii.connect(p=net.p_conn)
        syn_ii.w_ii = _weights(syn.w_ii_mean_nS, syn.w_ii_var_nS, len(syn_ii))

        syn_ie = b2.Synapses(
            neurons_inh, neurons_exc, model="w_ie : siemens", on_pre="g_inh += w_ie"
        )
        syn_ie.connect(p=net.p_conn)
        syn_ie.w_ie = _weights(syn.w_ie_mean_nS, syn.w_ie_var_nS, len(syn_ie))

        spike_mon = b2.SpikeMonitor(neurons)

        b2.run(t_total, report=None)

        spike_times_ms = np.array(spike_mon.t / ms)
        spike_indices = np.array(spike_mon.i)

        # ------------------------------------------------------------------
        # PSTH for excitatory population over the stimulation window
        # ------------------------------------------------------------------
        stim_onset_ms = sim.t_pre_ms
        stim_end_ms = sim.t_pre_ms + sim.t_stim_ms

        exc_mask = spike_indices < net.N_exc
        exc_times = spike_times_ms[exc_mask]

        bin_edges = np.arange(
            stim_onset_ms, stim_end_ms + sim.psth_bin_ms, sim.psth_bin_ms
        )
        counts, _ = np.histogram(exc_times, bins=bin_edges)
        psth_exc = counts / net.N_exc  # spikes per neuron per bin

        # Time axis relative to stimulus onset
        t_psth_ms = 0.5 * (bin_edges[:-1] + bin_edges[1:]) - stim_onset_ms

        return {
            "psth_exc": psth_exc,
            "t_psth_ms": t_psth_ms,
            "spike_times_ms": spike_times_ms,
            "spike_indices": spike_indices,
            "t_stim_ms": t_stim_arr,
            "stim_vals": stim_vals,
        }
