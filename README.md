# Gillespie: Stochastic Simulation of Markov Population Processes

A Python toolkit for exact stochastic simulation of continuous-time Markov chains (CTMCs), with a focus on queueing systems and population processes.

## Overview

This package implements **Gillespie's Stochastic Simulation Algorithm (SSA)** [1], also known as the direct method, for simulating Markov population processes. The simulator is generic and can handle any process specified by:

- **Jump vectors**: State changes when events occur
- **Rate functions**: Transition rates as functions of current state
- **Event labels**: Descriptive names for tracking event types

The implementation is exact (not approximate) and generates sample paths from the true stochastic process.

## Features

- **Generic MPP simulator**: Works with any Markov population process specification
- **Gillespie's direct method**: Exact stochastic simulation algorithm
- **Performance metrics**: Time-weighted state averages and event throughput tracking
- **Analytical validation**: Closed-form formulas for M/M/1 and M/M/c queues
- **Pedagogical code**: Well-documented with mathematical explanations
- **Reproducible**: Explicit random seed support

## Installation

```bash
# Clone the repository
git clone https://github.com/mirco-tribastone/qn-ssa.git
cd qn-ssa

# Install the package in development mode
pip install -e .
```

## Quick Start

### M/M/1 Queue Example

```python
import numpy as np
from qn_ssa.models import MarkovPopulationProcess
from qn_ssa.simulators import GillespieSimulator

# Create M/M/1 queue: single server with Poisson arrivals and exponential service
mpp = MarkovPopulationProcess(state_dim=1)

# Arrivals: X → X+1 with rate λ
mpp.add_event(
    jump=np.array([1]),
    rate_fn=lambda x: 0.8,  # λ = 0.8
    label="ARRIVAL"
)

# Departures: X → X-1 with rate μ (only if customers present)
mpp.add_event(
    jump=np.array([-1]),
    rate_fn=lambda x: 1.0 if x[0] > 0 else 0.0,  # μ = 1.0
    label="DEPARTURE"
)

# Run simulation
simulator = GillespieSimulator()
results = simulator.simulate(
    mpp=mpp,
    initial_state=np.array([0]),
    time_horizon=10000.0,
    n_runs=100,
    seed=42
)

print(f"Mean queue length: {results.mean_state[0]:.3f}")
print(f"Throughput: {results.event_throughputs['DEPARTURE']:.3f}")
```

### M/M/c Queue Example

```python
# M/M/c queue: c parallel servers
c = 5
lambda_rate = 4.0
mu_rate = 1.0

mpp = MarkovPopulationProcess(state_dim=1)

# Arrivals
mpp.add_event(
    jump=np.array([1]),
    rate_fn=lambda x: lambda_rate,
    label="ARRIVAL"
)

# Departures with c parallel servers
# Rate = μ * min(X, c)
mpp.add_event(
    jump=np.array([-1]),
    rate_fn=lambda x: mu_rate * min(x[0], c),
    label="DEPARTURE"
)
```

## Running Examples

The `examples/` directory contains complete validation scripts:

```bash
# M/M/1 queue simulation and validation
python examples/mm1_queue_example.py

# M/M/c queue simulation and validation
python examples/mmc_queue_example.py
```

Both examples compare simulation results against analytical closed-form formulas.

## Project Structure

```
qn-ssa/
├── src/
│   └── qn_ssa/                     # Main package
│       ├── models/
│       │   └── markov_population.py    # MPP specification class
│       ├── simulators/
│       │   └── gillespie_ssa.py        # Gillespie algorithm implementation
│       ├── analysis/
│       │   └── metrics.py              # Performance metrics tracking
│       └── utils/
│           ├── validation.py           # Analytical formulas for validation
│           └── phase_type.py           # Phase-type distribution utilities
├── examples/
│   ├── mm1_queue_example.py            # M/M/1 queue validation
│   ├── mmc_queue_example.py            # M/M/c queue validation
│   └── mcox2_1_queue_example.py        # M/Cox2/1 queue validation
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Algorithm Details

The Gillespie algorithm (direct method) simulates exact trajectories of continuous-time Markov chains:

1. **Compute propensities**: For state X(t), compute rates λⱼ(X(t)) for all events j
2. **Sample time to next event**: τ ~ Exp(∑ⱼ λⱼ(X(t)))
3. **Select event**: Choose event j with probability λⱼ(X(t)) / ∑ᵢ λᵢ(X(t))
4. **Update state**: X(t+τ) = X(t) + νⱼ where νⱼ is the jump vector
5. **Repeat** until time horizon reached

This method is exact because it exploits the memoryless property of exponential distributions and the fact that the minimum of independent exponentials is exponential with rate equal to the sum of individual rates.

## Validation Results

### M/M/1 Queue (λ=0.8, μ=1.0)

| Metric              | Analytical | Simulated | Error  |
| ------------------- | ---------- | --------- | ------ |
| Mean queue length   | 4.000000   | 3.937823  | 1.55%  |
| Mean response time  | 5.000000   | 4.921805  | 1.56%  |
| Throughput          | 0.800000   | 0.800077  | 0.01%  |

**Status**: ✓ VALIDATION PASSED

### M/M/c Queue (λ=4.0, μ=1.0, c=5)

| Metric              | Analytical | Simulated | Error  |
| ------------------- | ---------- | --------- | ------ |
| Mean in queue (Lq)  | 2.216450   | 2.205231  | 0.51%  |
| Mean in system (L)  | 6.216450   | 6.205231  | 0.18%  |
| Mean response time  | 1.554113   | 1.550801  | 0.21%  |
| Throughput          | 4.000000   | 4.001308  | 0.03%  |

**Status**: ✓ VALIDATION PASSED

Both validations use 100 independent runs with time horizon T=10,000.

## Mathematical Background

### Markov Population Processes

A Markov population process is a CTMC on state space ℤ^d where transitions occur via jump vectors:

- **State**: X(t) ∈ ℤ^d (often ℤ₊^d for population counts)
- **Events**: Collection of (νⱼ, λⱼ) pairs where:
  - νⱼ ∈ ℤ^d is the jump vector
  - λⱼ: ℤ^d → ℝ₊ is the rate function
- **Generator**: Q(x, x+ν) = λⱼ(x) if ν = νⱼ for some j

### Queueing Theory

For M/M/1 queue (λ < μ):
- **Utilization**: ρ = λ/μ
- **Mean in system**: L = ρ/(1-ρ)
- **Mean response time**: W = 1/(μ-λ)
- **Little's Law**: L = λW

For M/M/c queue (λ < cμ):
- **Utilization**: ρ = λ/(cμ)
- **Erlang-C**: C(c, λ/μ) = P(wait)
- Uses Erlang-C formula for performance metrics

## Future Extensions

Planned enhancements include:

- **Confidence intervals**: Using t-distribution for metric estimates
- **Batch means method**: For efficient steady-state analysis
- **Warm-up period detection**: Automatic transient removal
- **Stopping criteria**: Relative precision-based termination
- **Tau-leaping**: Approximate accelerated simulation
- **Phase-type distributions**: Non-exponential timing support
- **Network models**: Multi-node queueing networks

## References

[1] Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *The Journal of Physical Chemistry*, 81(25), 2340-2361.

[2] Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*. Wiley.

[3] Allen, A. O. (1990). *Probability, Statistics, and Queueing Theory*. Academic Press.
