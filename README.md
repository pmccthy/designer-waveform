# designer-waveform

A framework for optimising stimulation waveforms to evoke naturalistic population activity in models of cortical circuits.

## Overview

Typical optogenetic stimulation protocols do not produce the same population-level activity as natural sensory input, partally due to strong synchronous stimulation. This repository provides tools to close that gap: given a target population response (e.g. a PSTH recorded during natural stimulation), it searches the space of parametric waveforms to find the stimulus shape that drives a neural population model towards that target.

The core design separates waveform parameterisation from simulation logic. Any callable that maps a waveform to a scalar loss can be used as the objective, making it straightforward to swap in different neural population models (Brian2, rate models, etc.) without modifying the waveform or optimisation code.

## Structure

```
designer_waveform/    # Installable package
    waveforms.py      # Waveform base class and concrete implementations
    optimisation.py   # (reserved for optimisation utilities)
    models.py         # (reserved for population model wrappers)
experiment_nb/        # Jupyter notebooks for experiments
experimental_data/    # Experimental data (not tracked by git)
results/              # Saved results (not tracked by git)
```

## Installation

Requires Python ≥ 3.12. Install into a mamba/conda environment:

```bash
mamba activate designer_waveform
pip install -e .
```

## Usage

```python
from designer_waveform.waveforms import SplitGaussianWaveform

# Define a waveform
wf = SplitGaussianWaveform(amplitude=1.0, mu=0.05, sigma_rise=0.01, sigma_fall=0.03)

# Define an objective: takes a Waveform, returns a scalar loss
def objective(waveform):
    # run your neural population simulation with waveform(t) as the stimulus
    # return MSE (or similar) against your target PSTH
    ...

# Optimise
best_wf, result = wf.optimise(objective, method='Nelder-Mead')
print(best_wf)
```

## Dependencies

- [Brian2](https://brian2.readthedocs.io/) — spiking neural network simulation
- NumPy, SciPy, Matplotlib
