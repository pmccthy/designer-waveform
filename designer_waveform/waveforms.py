"""
Classes for parameterised waveforms and optimisation logic.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from abc import ABC, abstractmethod

class Waveform(ABC):
    """
    Abstract base class for parameterised waveforms.

    Subclasses must implement:
      - __call__(t)       : evaluate waveform at time points t
      - to_params()       : return current parameters as a flat numpy array
      - from_params(params): classmethod — construct instance from flat param array

    The optimise() method handles simulation-based optimisation: it wraps an
    arbitrary objective callable (waveform -> scalar) and passes it to
    scipy.optimize, keeping simulation logic entirely outside this class.
    """

    @abstractmethod
    def __call__(self, t: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def to_params(self) -> np.ndarray:
        """Return current parameters as a flat numpy array (initial guess for optimiser)."""
        ...

    @classmethod
    @abstractmethod
    def from_params(cls, params: np.ndarray):
        """Construct an instance from a flat parameter array."""
        ...

    def optimise(
        self,
        objective_fn,
        method="Nelder-Mead",
        bounds=None,
        verbose=True,
        log_every=10,
        patience=None,
        min_improvement=1e-4,
        **kwargs,
    ):
        """Optimise waveform parameters against a simulation-based objective.

        Args:
            objective_fn: Callable ``(waveform) -> float``.  Takes a Waveform
                instance, runs the simulation, and returns a scalar loss (e.g.
                MSE between simulated PSTH and target PSTH).
            method: ``scipy.optimize.minimize`` method.  ``'Nelder-Mead'``
                works well for gradient-free black-box objectives.  Use
                ``'L-BFGS-B'`` with bounds for differentiable proxies, or
                ``differential_evolution`` for a global search.
            bounds: Sequence of ``(min, max)`` pairs, one per parameter.
                Required for bounded methods.
            verbose: If ``True``, print a progress line every ``log_every``
                objective calls plus a summary on completion.
            log_every: Number of objective calls between progress lines.
            patience: Stop after this many consecutive objective calls with no
                relative improvement greater than ``min_improvement``.  ``None``
                disables early stopping and defers entirely to the solver.
            min_improvement: Minimum relative improvement in best loss required
                to reset the patience counter.  Ignored when ``patience`` is
                ``None``.
            **kwargs: Forwarded to ``scipy.optimize.minimize``.

        Returns:
            tuple:
                - **result_waveform** (*Waveform*) — new instance with the
                  best parameters seen (not necessarily the final iterate).
                - **opt_result** (*OptimizeResult*) — full scipy result object
                  (check ``.success``, ``.fun``, ``.nit``).
        """
        import sys
        from scipy.optimize import OptimizeResult

        _state = {
            "n_calls": 0,
            "best_loss": float("inf"),
            "best_params": None,
            "calls_since_improvement": 0,
        }

        class _EarlyStop(Exception):
            pass

        def _objective(params):
            wf = self.from_params(params)
            loss = float(objective_fn(wf))
            _state["n_calls"] += 1

            rel_improvement = (_state["best_loss"] - loss) / (abs(_state["best_loss"]) + 1e-12)
            if loss < _state["best_loss"]:
                _state["best_params"] = params.copy()
                _state["best_loss"] = loss
            if rel_improvement > min_improvement:
                _state["calls_since_improvement"] = 0
            else:
                _state["calls_since_improvement"] += 1

            if patience is not None and _state["calls_since_improvement"] >= patience:
                if verbose:
                    print(
                        f"\n  Early stop at call {_state['n_calls']}: "
                        f"no improvement > {min_improvement:.1e} for {patience} calls",
                        flush=True,
                    )
                raise _EarlyStop()

            if verbose and _state["n_calls"] % log_every == 0:
                print(
                    f"  call {_state['n_calls']:4d} | "
                    f"best loss {_state['best_loss']:.6e} | "
                    f"{self.from_params(_state['best_params'])}",
                    flush=True,
                )
            return loss

        if verbose:
            print(f"Starting {method} optimisation ({len(self.to_params())} params)")
            print(f"  Initial waveform: {self}")
            sys.stdout.flush()

        try:
            opt_result = minimize(
                _objective, self.to_params(), method=method, bounds=bounds, **kwargs
            )
        except _EarlyStop:
            opt_result = OptimizeResult(
                x=_state["best_params"],
                fun=_state["best_loss"],
                success=False,
                message="Early stop: patience exceeded",
                nit=_state["n_calls"],
                nfev=_state["n_calls"],
            )

        if verbose:
            status = "converged" if opt_result.success else "stopped"
            print(
                f"\n{status} after {_state['n_calls']} calls ({opt_result.nit} iters) | "
                f"loss {opt_result.fun:.6e} | success={opt_result.success}"
            )
            if not opt_result.success:
                print(f"  scipy message: {opt_result.message}")
            sys.stdout.flush()

        return self.from_params(_state["best_params"]), opt_result

    def plot(self, t: np.ndarray, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(t, self(t), **kwargs)
        return ax
    
class SplitGaussianWaveform(Waveform):
    """
    Asymmetric Gaussian with independent rise and fall timescales.

        y(t) = amplitude * exp(-0.5 * ((t - mu) / sigma_rise)^2) + baseline,  t < mu
               amplitude * exp(-0.5 * ((t - mu) / sigma_fall)^2) + baseline,  t >= mu

    Parameters
    ----------
    amplitude : float
        Peak amplitude at t = mu.
    mu : float
        Peak time.
    sigma_rise : float > 0
        Width of the rising (left) half.
    sigma_fall : float > 0
        Width of the falling (right) half.
    baseline : float
        Additive offset applied uniformly.

    Notes
    -----
    When amplitude >= 0 and baseline >= 0 the waveform is everywhere non-negative.
    Suggested bounds for optimise():
        amplitude > 0, sigma_rise > 0, sigma_fall > 0, baseline >= 0.
    """

    def __init__(self, amplitude=1.0, mu=0.0, sigma_rise=1.0, sigma_fall=1.0, baseline=0.0):
        self.amplitude = amplitude
        self.mu = mu
        self.sigma_rise = sigma_rise
        self.sigma_fall = sigma_fall
        self.baseline = baseline

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        sigma = np.where(t < self.mu, self.sigma_rise, self.sigma_fall)
        return self.amplitude * np.exp(-0.5 * ((t - self.mu) / sigma) ** 2) + self.baseline

    def to_params(self):
        return np.array([self.amplitude, self.mu, self.sigma_rise, self.sigma_fall, self.baseline])

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (f"SplitGaussianWaveform(amplitude={self.amplitude:.3g}, mu={self.mu:.3g}, "
                f"sigma_rise={self.sigma_rise:.3g}, sigma_fall={self.sigma_fall:.3g}, "
                f"baseline={self.baseline:.3g})")


class AsymBaselineSplitGaussianWaveform(Waveform):
    """Asymmetric Gaussian with independent rise/fall timescales and independent
    pre-peak / post-peak baselines.

    The waveform is defined piecewise around the peak time ``mu``:

        y(t) = (amplitude - baseline_rise) * exp(-0.5 * ((t - mu) / sigma_rise)**2)
                + baseline_rise,   t <  mu

        y(t) = (amplitude - baseline_fall) * exp(-0.5 * ((t - mu) / sigma_fall)**2)
                + baseline_fall,   t >= mu

    Unlike :class:`SplitGaussianWaveform`, ``amplitude`` here is the *absolute
    peak value* of the waveform rather than an additive bump on top of a single
    baseline. Each half decays from this shared peak down to its own asymptotic
    level, so the waveform is continuous at ``t = mu`` even when the two
    baselines differ. As ``t -> -inf`` the waveform tends to ``baseline_rise``;
    as ``t -> +inf`` it tends to ``baseline_fall``. This lets a single
    envelope represent e.g. a low pre-stim drive, a transient peak, and a
    different post-stim plateau.

    Args:
        amplitude: Peak value at ``t = mu``. Should normally satisfy
            ``amplitude >= max(baseline_rise, baseline_fall)`` so the shape is
            genuinely peaked rather than inverted.
        mu: Peak time.
        sigma_rise: Width of the rising (left) half. Strictly positive.
        sigma_fall: Width of the falling (right) half. Strictly positive.
        baseline_rise: Asymptotic level as ``t -> -inf`` (pre-stim offset).
        baseline_fall: Asymptotic level as ``t -> +inf`` (post-stim plateau).

    Notes:
        Reduces to the standard split Gaussian (with a uniform baseline) when
        ``baseline_rise == baseline_fall``. Suggested bounds for
        :meth:`Waveform.optimise`: ``amplitude > 0``, ``sigma_rise > 0``,
        ``sigma_fall > 0``, with ``baseline_rise`` and ``baseline_fall``
        typically ``>= 0`` for opto envelopes.
    """

    def __init__(
        self,
        amplitude=1.0,
        mu=0.0,
        sigma_rise=1.0,
        sigma_fall=1.0,
        baseline_rise=0.0,
        baseline_fall=0.0,
    ):
        self.amplitude = amplitude
        self.mu = mu
        self.sigma_rise = sigma_rise
        self.sigma_fall = sigma_fall
        self.baseline_rise = baseline_rise
        self.baseline_fall = baseline_fall

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        sigma = np.where(t < self.mu, self.sigma_rise, self.sigma_fall)
        baseline = np.where(t < self.mu, self.baseline_rise, self.baseline_fall)
        return (self.amplitude - baseline) * np.exp(
            -0.5 * ((t - self.mu) / sigma) ** 2
        ) + baseline

    def to_params(self):
        return np.array(
            [
                self.amplitude,
                self.mu,
                self.sigma_rise,
                self.sigma_fall,
                self.baseline_rise,
                self.baseline_fall,
            ]
        )

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (
            f"AsymBaselineSplitGaussianWaveform("
            f"amplitude={self.amplitude:.3g}, mu={self.mu:.3g}, "
            f"sigma_rise={self.sigma_rise:.3g}, sigma_fall={self.sigma_fall:.3g}, "
            f"baseline_rise={self.baseline_rise:.3g}, "
            f"baseline_fall={self.baseline_fall:.3g})"
        )


class GaussianExpFallWaveform(Waveform):
    """Gaussian rise, single-exponential fall — a skewed bump with a heavy tail.

    The rising half is the left side of a Gaussian; the falling half is a pure
    exponential decay rather than a Gaussian. This gives a *heavy* (exponential)
    tail instead of the super-exponential decay of a Gaussian, so it tracks the
    long, slow decay of a sensory PSTH far better while still allowing a sharp
    peak:

        y(t) = (amplitude - baseline_rise) * exp(-0.5 * ((t - mu) / sigma_rise)**2)
                + baseline_rise,                                    t <  mu

        y(t) = (amplitude - baseline_fall) * exp(-(t - mu) / tau_fall)
                + baseline_fall,                                    t >= mu

    Both halves equal ``amplitude`` at ``t = mu``, so the waveform is continuous
    at the peak even when the two baselines differ. As ``t -> -inf`` it tends to
    ``baseline_rise``; as ``t -> +inf`` it tends to ``baseline_fall``.

    This is the minimal change from
    :class:`AsymBaselineSplitGaussianWaveform` (same six-parameter footprint,
    ``sigma_fall`` simply becomes the decay constant ``tau_fall``) and is the
    recommended drop-in when the *rise/peak* already fits but the *tail* decays
    too quickly.

    Args:
        amplitude: Peak value at ``t = mu``.
        mu: Peak time.
        sigma_rise: Width of the rising (left) Gaussian half. Strictly positive.
        tau_fall: Decay time constant of the falling exponential. Strictly
            positive; larger values give a longer, heavier tail.
        baseline_rise: Asymptotic level as ``t -> -inf`` (pre-stim offset).
        baseline_fall: Asymptotic level as ``t -> +inf`` (post-stim plateau).

    Notes:
        Suggested bounds for :meth:`Waveform.optimise`: ``amplitude > 0``,
        ``sigma_rise > 0``, ``tau_fall > 0``, with ``baseline_rise`` and
        ``baseline_fall`` typically ``>= 0`` for opto envelopes.
    """

    def __init__(
        self,
        amplitude=1.0,
        mu=0.0,
        sigma_rise=1.0,
        tau_fall=1.0,
        baseline_rise=0.0,
        baseline_fall=0.0,
    ):
        self.amplitude = amplitude
        self.mu = mu
        self.sigma_rise = sigma_rise
        self.tau_fall = tau_fall
        self.baseline_rise = baseline_rise
        self.baseline_fall = baseline_fall

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        dt = t - self.mu
        rise = (self.amplitude - self.baseline_rise) * np.exp(
            -0.5 * (dt / self.sigma_rise) ** 2
        ) + self.baseline_rise
        fall = (self.amplitude - self.baseline_fall) * np.exp(
            -dt / self.tau_fall
        ) + self.baseline_fall
        return np.where(t < self.mu, rise, fall)

    def to_params(self):
        return np.array(
            [
                self.amplitude,
                self.mu,
                self.sigma_rise,
                self.tau_fall,
                self.baseline_rise,
                self.baseline_fall,
            ]
        )

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (
            f"GaussianExpFallWaveform("
            f"amplitude={self.amplitude:.3g}, mu={self.mu:.3g}, "
            f"sigma_rise={self.sigma_rise:.3g}, tau_fall={self.tau_fall:.3g}, "
            f"baseline_rise={self.baseline_rise:.3g}, "
            f"baseline_fall={self.baseline_fall:.3g})"
        )


class GaussianBiExpFallWaveform(Waveform):
    """Gaussian rise, *bi-exponential* fall — captures two decay timescales.

    The falling half is the weighted sum of a fast and a slow exponential, which
    reproduces the common sensory-PSTH shape of a rapid post-peak drop followed
    by a slowly-decaying sustained component (and the gentle "shoulder" their
    sum produces in between):

        y(t) = (amplitude - baseline_rise) * exp(-0.5 * ((t - mu) / sigma_rise)**2)
                + baseline_rise,                                    t <  mu

        y(t) = (amplitude - baseline_fall) * [ w * exp(-(t - mu) / tau_fast)
                + (1 - w) * exp(-(t - mu) / tau_slow) ]
                + baseline_fall,                                    t >= mu

    where ``w = weight_fast``. Because the two exponentials each equal 1 at
    ``t = mu`` and the weights sum to 1, the falling half equals ``amplitude``
    at the peak, so the waveform is continuous at ``mu``.

    Args:
        amplitude: Peak value at ``t = mu``.
        mu: Peak time.
        sigma_rise: Width of the rising (left) Gaussian half. Strictly positive.
        tau_fast: Fast decay constant (the rapid post-peak drop). Strictly
            positive.
        tau_slow: Slow decay constant (the sustained tail). Strictly positive;
            normally ``tau_slow > tau_fast``.
        weight_fast: Fraction of the falling amplitude carried by the fast
            component, in ``[0, 1]``. ``w = 1`` recovers a single fast
            exponential; ``w = 0`` recovers a single slow exponential.
        baseline_rise: Asymptotic level as ``t -> -inf`` (pre-stim offset).
        baseline_fall: Asymptotic level as ``t -> +inf`` (post-stim plateau).

    Notes:
        Suggested bounds for :meth:`Waveform.optimise`: ``amplitude > 0``,
        ``sigma_rise > 0``, ``tau_fast > 0``, ``tau_slow > 0``,
        ``0 <= weight_fast <= 1``, ``baseline_rise, baseline_fall >= 0``. To keep
        ``tau_fast`` and ``tau_slow`` from swapping roles during optimisation,
        give them non-overlapping bounds (e.g. ``tau_fast in (2, 30)``,
        ``tau_slow in (30, 300)``).
    """

    def __init__(
        self,
        amplitude=1.0,
        mu=0.0,
        sigma_rise=1.0,
        tau_fast=1.0,
        tau_slow=10.0,
        weight_fast=0.5,
        baseline_rise=0.0,
        baseline_fall=0.0,
    ):
        self.amplitude = amplitude
        self.mu = mu
        self.sigma_rise = sigma_rise
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.weight_fast = weight_fast
        self.baseline_rise = baseline_rise
        self.baseline_fall = baseline_fall

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        dt = t - self.mu
        w = self.weight_fast
        rise = (self.amplitude - self.baseline_rise) * np.exp(
            -0.5 * (dt / self.sigma_rise) ** 2
        ) + self.baseline_rise
        decay = w * np.exp(-dt / self.tau_fast) + (1.0 - w) * np.exp(
            -dt / self.tau_slow
        )
        fall = (self.amplitude - self.baseline_fall) * decay + self.baseline_fall
        return np.where(t < self.mu, rise, fall)

    def to_params(self):
        return np.array(
            [
                self.amplitude,
                self.mu,
                self.sigma_rise,
                self.tau_fast,
                self.tau_slow,
                self.weight_fast,
                self.baseline_rise,
                self.baseline_fall,
            ]
        )

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (
            f"GaussianBiExpFallWaveform("
            f"amplitude={self.amplitude:.3g}, mu={self.mu:.3g}, "
            f"sigma_rise={self.sigma_rise:.3g}, tau_fast={self.tau_fast:.3g}, "
            f"tau_slow={self.tau_slow:.3g}, weight_fast={self.weight_fast:.3g}, "
            f"baseline_rise={self.baseline_rise:.3g}, "
            f"baseline_fall={self.baseline_fall:.3g})"
        )


class LogNormalWaveform(Waveform):
    """Log-normal bump — intrinsically right-skewed with a heavy tail.

    A single-component, naturally asymmetric pulse: a sharp rise after an onset
    delay followed by a long, heavy tail. Defined for ``t > t0`` (and held at
    ``baseline`` before that), which also lets it reproduce a response latency
    where the PSTH sits flat before rising:

        x    = (t - t0) / tau
        y(t) = (amplitude - baseline) * exp(-0.5 * (ln(x) / sigma)**2) + baseline,
                                                                   t > t0
        y(t) = baseline,                                           t <= t0

    The bump peaks at ``t = t0 + tau`` with value ``amplitude`` (since the
    exponent is zero when ``x = 1``). ``sigma`` controls skew/tail heaviness:
    larger ``sigma`` gives a more skewed shape with a longer tail.

    Args:
        amplitude: Peak value (attained at ``t = t0 + tau``).
        t0: Onset time; the waveform is flat at ``baseline`` for ``t <= t0``.
        tau: Time from onset to the peak (sets the peak location). Strictly
            positive.
        sigma: Log-scale width / shape. Strictly positive; larger values give a
            heavier right tail.
        baseline: Additive offset / asymptotic level outside the bump.

    Notes:
        Suggested bounds for :meth:`Waveform.optimise`: ``amplitude > 0``,
        ``tau > 0``, ``sigma > 0``, ``baseline >= 0``. ``t0`` may be negative
        (onset before the stim window) if the rise has no measurable latency.
    """

    def __init__(self, amplitude=1.0, t0=0.0, tau=1.0, sigma=0.5, baseline=0.0):
        self.amplitude = amplitude
        self.t0 = t0
        self.tau = tau
        self.sigma = sigma
        self.baseline = baseline

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        active = t > self.t0
        # Evaluate ln(x) only where active; use a safe placeholder elsewhere.
        x = np.where(active, (t - self.t0) / self.tau, 1.0)
        bump = (self.amplitude - self.baseline) * np.exp(
            -0.5 * (np.log(x) / self.sigma) ** 2
        ) + self.baseline
        return np.where(active, bump, self.baseline)

    def to_params(self):
        return np.array([self.amplitude, self.t0, self.tau, self.sigma, self.baseline])

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (
            f"LogNormalWaveform(amplitude={self.amplitude:.3g}, t0={self.t0:.3g}, "
            f"tau={self.tau:.3g}, sigma={self.sigma:.3g}, "
            f"baseline={self.baseline:.3g})"
        )


class RectangularPulseWaveform(Waveform):
    """Single rectangular (square-wave) pulse.

        y(t) = amplitude   for onset_ms <= t < onset_ms + duration_ms
               baseline    otherwise

    Args:
        onset_ms: Start of the pulse relative to stimulus onset.
        duration_ms: Width of the pulse.
        amplitude: Pulse height.  Should be in [0, 1] for opsin-drive envelopes.
        baseline: Value outside the pulse window.
    """

    def __init__(self, onset_ms=0.0, duration_ms=50.0, amplitude=1.0, baseline=0.0):
        self.onset_ms = onset_ms
        self.duration_ms = duration_ms
        self.amplitude = amplitude
        self.baseline = baseline

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        active = (t >= self.onset_ms) & (t < self.onset_ms + self.duration_ms)
        return np.where(active, self.amplitude, self.baseline)

    def to_params(self):
        return np.array([self.onset_ms, self.duration_ms, self.amplitude, self.baseline])

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (f"RectangularPulseWaveform(onset_ms={self.onset_ms:.3g}, "
                f"duration_ms={self.duration_ms:.3g}, amplitude={self.amplitude:.3g}, "
                f"baseline={self.baseline:.3g})")


class PulseTrainWaveform(Waveform):
    """Periodic rectangular pulse train.

    Pulses of width ``pulse_duration_ms`` repeat at ``frequency_hz``,
    starting at ``onset_ms`` and running for ``train_duration_ms``.
    For well-defined pulses: ``pulse_duration_ms < 1000 / frequency_hz``.

    Args:
        onset_ms: Start of the train relative to stimulus onset.
        pulse_duration_ms: Width of each individual pulse.
        frequency_hz: Pulse repetition rate.
        train_duration_ms: Total duration of the train.
        amplitude: Amplitude during each pulse.
        baseline: Amplitude between pulses and outside the train.
    """

    def __init__(
        self,
        onset_ms=0.0,
        pulse_duration_ms=20.0,
        frequency_hz=25.0,
        train_duration_ms=1000.0,
        amplitude=1.0,
        baseline=0.0,
    ):
        self.onset_ms = onset_ms
        self.pulse_duration_ms = pulse_duration_ms
        self.frequency_hz = frequency_hz
        self.train_duration_ms = train_duration_ms
        self.amplitude = amplitude
        self.baseline = baseline

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        period_ms = 1000.0 / self.frequency_hz
        t_rel = t - self.onset_ms
        in_train = (t_rel >= 0) & (t_rel < self.train_duration_ms)
        in_pulse = (t_rel % period_ms) < self.pulse_duration_ms
        return np.where(in_train & in_pulse, self.amplitude, self.baseline)

    def to_params(self):
        return np.array([
            self.onset_ms, self.pulse_duration_ms, self.frequency_hz,
            self.train_duration_ms, self.amplitude, self.baseline,
        ])

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    def __repr__(self):
        return (
            f"PulseTrainWaveform(onset_ms={self.onset_ms:.3g}, "
            f"pulse_duration_ms={self.pulse_duration_ms:.3g}, "
            f"frequency_hz={self.frequency_hz:.3g}, "
            f"train_duration_ms={self.train_duration_ms:.3g}, "
            f"amplitude={self.amplitude:.3g}, baseline={self.baseline:.3g})"
        )