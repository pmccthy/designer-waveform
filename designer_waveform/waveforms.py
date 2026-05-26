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