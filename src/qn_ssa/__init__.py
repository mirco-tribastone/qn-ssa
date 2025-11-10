"""
qn-ssa: Stochastic Simulation Algorithm for Queueing Networks

A Python toolkit for exact stochastic simulation of continuous-time Markov chains (CTMCs),
with a focus on queueing systems and population processes using Gillespie's SSA.

This package provides tools for simulating and analyzing continuous-time
Markov chains, with a focus on queueing systems and population processes.
"""

__version__ = "0.1.0"
__author__ = "Mirco Tribastone"
__package_name__ = "qn-ssa"

# Core simulation components
from qn_ssa.models import MarkovPopulationProcess
from qn_ssa.simulators import GillespieSimulator
from qn_ssa.analysis import MetricTracker, SimulationResults

# Utility functions for validation
from qn_ssa.utils import (
    # M/M/1 queue
    mm1_mean_queue_length,
    mm1_mean_response_time,
    mm1_mean_waiting_time,
    mm1_utilization,
    # M/M/c queue
    erlang_c,
    mmc_mean_queue_length,
    mmc_mean_response_time,
    mmc_mean_waiting_time,
    mmc_utilization,
    # M/G/1 queue
    mg1_mean_queue_length,
    mg1_mean_system_length,
    mg1_mean_response_time,
    mg1_mean_waiting_time,
    # Phase-type distributions
    coxian2_from_moments,
    hyperexponential2_from_moments,
    CoxianPhaseParameters,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__package_name__",
    # Core classes
    "MarkovPopulationProcess",
    "GillespieSimulator",
    "MetricTracker",
    "SimulationResults",
    # M/M/1 validation
    "mm1_mean_queue_length",
    "mm1_mean_response_time",
    "mm1_mean_waiting_time",
    "mm1_utilization",
    # M/M/c validation
    "erlang_c",
    "mmc_mean_queue_length",
    "mmc_mean_response_time",
    "mmc_mean_waiting_time",
    "mmc_utilization",
    # M/G/1 validation
    "mg1_mean_queue_length",
    "mg1_mean_system_length",
    "mg1_mean_response_time",
    "mg1_mean_waiting_time",
    # Phase-type utilities
    "coxian2_from_moments",
    "hyperexponential2_from_moments",
    "CoxianPhaseParameters",
]
