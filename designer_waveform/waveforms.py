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
            **kwargs: Forwarded to ``scipy.optimize.minimize``.

        Returns:
            tuple:
                - **result_waveform** (*Waveform*) — new instance with
                  optimised parameters.
                - **opt_result** (*OptimizeResult*) — full scipy result object
                  (check ``.success``, ``.fun``, ``.nit``).
        """
        import sys

        _state = {"n_calls": 0, "best_loss": float("inf"), "best_params": None}

        def _objective(params):
            wf = self.from_params(params)
            loss = float(objective_fn(wf))
            _state["n_calls"] += 1
            if loss < _state["best_loss"]:
                _state["best_loss"] = loss
                _state["best_params"] = params.copy()
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

        opt_result = minimize(
            _objective, self.to_params(), method=method, bounds=bounds, **kwargs
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

        return self.from_params(opt_result.x), opt_result

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