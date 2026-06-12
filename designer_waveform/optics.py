"""
Optical stimulation pipeline: source power → tissue irradiance → photocurrent.

The conversion chain is:

    P_source [mW]
      × (window_transmission × tissue_transmission) / area_mm2
      → irradiance at tissue [mW/mm²]
      → PowerToCurrentCurve *or* SigmoidPowerCurve (irradiance)
      → photocurrent for a maximally-expressing neuron [pA]

Two curve types are provided:

* :class:`PowerToCurrentCurve` — piecewise-linear interpolation of
  measured (irradiance, current) data loaded from a ``.npz`` file.
* :class:`SigmoidPowerCurve` — parametric Hill (sigmoid) function::

      I(E) = i_max_pA × Eⁿ / (K½ⁿ + Eⁿ)

  where ``K½`` is the half-saturation irradiance and ``n`` is the Hill
  coefficient.  Both curves share the same ``__call__`` interface so they
  are interchangeable in :class:`~designer_waveform.models.RandomEINetwork`.

Per-neuron current is then::

    I(t, i) = curve(irradiance(t)) × expr_frac[i]

where ``expr_frac[i] = stim_dist_pA[i] / I_max_pA`` ∈ [0, 1] is the
normalised opsin expression level drawn from the lognormal distribution.
The curve's ``i_max_pA`` therefore sets the absolute current scale for a
maximally-expressing neuron, and expression heterogeneity scales it down.

Default physical setup parameters are stored in ``data/optics_params.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class OpticsConfig:
    """Physical parameters of the optical stimulation path.

    Args:
        window_transmission: Fractional power transmitted through the
            cranial window (dimensionless, 0–1).
        tissue_transmission: Fractional power transmitted through tissue
            to the depth of interest (dimensionless, 0–1).
        area_mm2: Illuminated area at the tissue surface (mm²).

    Example:
        >>> cfg = OpticsConfig.from_file("data/optics_params.json")
        >>> irr = cfg.source_power_to_irradiance(1.0)   # mW → mW/mm²
    """

    window_transmission: float = 0.92
    tissue_transmission: float = 0.30
    area_mm2: float = 12.56

    @property
    def total_transmission(self) -> float:
        """Combined window × tissue transmission factor."""
        return self.window_transmission * self.tissue_transmission

    def source_power_to_irradiance(
        self, power_mW: float | np.ndarray
    ) -> float | np.ndarray:
        """Convert source power (mW) to irradiance at the tissue surface (mW/mm²).

        Args:
            power_mW: Source optical power in milliwatts.

        Returns:
            Irradiance in mW/mm² at the depth of interest.
        """
        return np.asarray(power_mW) * self.total_transmission / self.area_mm2

    @classmethod
    def from_file(cls, path: str | Path) -> OpticsConfig:
        """Load parameters from a JSON file.

        Args:
            path: Path to a JSON file with keys ``window_transmission``,
                ``tissue_transmission``, and ``area_mm2``.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str | Path) -> None:
        """Save parameters to a JSON file.

        Args:
            path: Destination path for the JSON file.
        """
        with open(path, "w") as f:
            json.dump(
                {
                    "window_transmission": self.window_transmission,
                    "tissue_transmission": self.tissue_transmission,
                    "area_mm2": self.area_mm2,
                },
                f,
                indent=2,
            )


class PowerToCurrentCurve:
    """Interpolating map from tissue irradiance (mW/mm²) to photocurrent (pA).

    Values outside the measured irradiance range are clamped to the
    curve endpoints (no extrapolation).

    Args:
        irradiance_mW_mm2: Measured irradiance values (mW/mm²), must be
            monotonically increasing.
        current_pA: Corresponding photocurrent values (pA).

    Example:
        >>> curve = PowerToCurrentCurve.from_file("data/power_to_current_curve.npz")
        >>> current = curve(0.05)   # irradiance in mW/mm² → current in pA
    """

    def __init__(
        self,
        irradiance_mW_mm2: np.ndarray,
        current_pA: np.ndarray,
    ) -> None:
        self.irradiance = np.asarray(irradiance_mW_mm2, dtype=float)
        self.current = np.asarray(current_pA, dtype=float)
        self._interp = interp1d(
            self.irradiance,
            self.current,
            kind="linear",
            bounds_error=False,
            fill_value=(self.current[0], self.current[-1]),
        )

    def __call__(self, irradiance: float | np.ndarray) -> np.ndarray:
        """Interpolate photocurrent at the given irradiance.

        Args:
            irradiance: Tissue irradiance in mW/mm².

        Returns:
            Photocurrent in pA (same shape as input).
        """
        return self._interp(np.asarray(irradiance, dtype=float))

    @classmethod
    def from_file(cls, path: str | Path) -> PowerToCurrentCurve:
        """Load a measured curve from an ``.npz`` file.

        The file must contain arrays ``irradiance_mW_mm2`` and
        ``current_pA``.

        Args:
            path: Path to the ``.npz`` file.
        """
        data = np.load(path)
        return cls(data["irradiance_mW_mm2"], data["current_pA"])

    def to_file(self, path: str | Path) -> None:
        """Save curve data to an ``.npz`` file.

        Args:
            path: Destination path (will be saved as ``.npz``).
        """
        np.savez(
            path,
            irradiance_mW_mm2=self.irradiance,
            current_pA=self.current,
        )

    @classmethod
    def placeholder(
        cls,
        i_max_pA: float = 1700.0,
        max_irradiance_mW_mm2: float = 0.1,
    ) -> PowerToCurrentCurve:
        """Linear placeholder curve — replace with real measured data.

        Creates a two-point linear curve from (0, 0) to
        (``max_irradiance_mW_mm2``, ``i_max_pA``).

        Args:
            i_max_pA: Current at saturating irradiance (pA).
            max_irradiance_mW_mm2: Irradiance at which current saturates
                (mW/mm²).
        """
        irr = np.array([0.0, max_irradiance_mW_mm2])
        cur = np.array([0.0, i_max_pA])
        return cls(irr, cur)


class SigmoidPowerCurve:
    """Parametric Hill (sigmoid) irradiance-to-current curve.

    Models the opsin photocurrent response as a Hill function::

        I(E) = i_max_pA × Eⁿ / (K½ⁿ + Eⁿ)

    where *E* is irradiance (mW/mm²), *K½* is the half-saturation
    irradiance, and *n* is the Hill coefficient.

    This is the recommended alternative to :class:`PowerToCurrentCurve`
    when you have parameters rather than a measured lookup table.  Both
    share the same ``__call__`` interface and are interchangeable in
    :class:`~designer_waveform.models.RandomEINetwork`.

    The ``i_max_pA`` attribute sets the absolute current scale for a
    maximally-expressing neuron; per-neuron expression heterogeneity in
    the network model scales this down.

    Named presets are available via classmethods:

    * :meth:`c1v1` — C1V1-A (default parameters, ~1175 pA peak, saturates
      at ~0.1 mW/mm²).
    * :meth:`chrmine` — ChRmine (red-shifted, ~4600 pA peak, saturates
      at ~0.7 mW/mm²).

    Args:
        i_max_pA: Saturating photocurrent for a maximally-expressing
            neuron (pA).
        half_sat_mW_mm2: Irradiance at which current reaches half its
            maximum (mW/mm²).  Note: this is tissue irradiance, not
            source power — :class:`OpticsConfig` handles the source→tissue
            conversion upstream.
        hill_n: Hill coefficient (dimensionless).  ``n=1`` gives a
            first-order (Michaelis-Menten) response; larger values
            produce a steeper sigmoid.

    Example:
        >>> curve = SigmoidPowerCurve.c1v1()
        >>> curve(0.0015)   # → ~587.5 pA  (half-max)
        >>> curve = SigmoidPowerCurve.chrmine()
        >>> curve(0.037)    # → ~2300 pA   (half-max)
    """

    def __init__(
        self,
        i_max_pA: float = 1175.0,
        half_sat_mW_mm2: float = 0.0015,
        hill_n: float = 1.0,
    ) -> None:
        self.i_max_pA = float(i_max_pA)
        self.half_sat_mW_mm2 = float(half_sat_mW_mm2)
        self.hill_n = float(hill_n)

    def __call__(self, irradiance: float | np.ndarray) -> np.ndarray:
        """Evaluate photocurrent at the given tissue irradiance.

        Args:
            irradiance: Tissue irradiance in mW/mm².  Negative values
                are clamped to zero.

        Returns:
            Photocurrent in pA (same shape as input).
        """
        E = np.clip(np.asarray(irradiance, dtype=float), 0.0, None)
        En = E ** self.hill_n
        Kn = self.half_sat_mW_mm2 ** self.hill_n
        return self.i_max_pA * En / (Kn + En)

    def __repr__(self) -> str:
        return (
            f"SigmoidPowerCurve("
            f"i_max_pA={self.i_max_pA}, "
            f"half_sat_mW_mm2={self.half_sat_mW_mm2}, "
            f"hill_n={self.hill_n})"
        )

    @classmethod
    def c1v1(cls) -> SigmoidPowerCurve:
        """C1V1-A preset: ~1175 pA peak, half-sat ~0.0015 mW/mm²."""
        return cls(i_max_pA=1175.0, half_sat_mW_mm2=0.0015, hill_n=1.0)

    @classmethod
    def chrmine(cls) -> SigmoidPowerCurve:
        """ChRmine preset: ~4600 pA peak, half-sat ~0.037 mW/mm².

        Parameters are approximate — ChRmine saturates near 0.7 mW/mm²
        (95% saturation at K½ × 19 ≈ 0.7 mW/mm² → K½ ≈ 0.037 mW/mm²).
        Refine ``half_sat_mW_mm2`` if you have measured data.
        """
        return cls(i_max_pA=4600.0, half_sat_mW_mm2=0.037, hill_n=1.0)
