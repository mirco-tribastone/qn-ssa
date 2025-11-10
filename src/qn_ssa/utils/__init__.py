"""Utility functions for validation and analysis."""

from .validation import (
    mm1_mean_queue_length,
    mm1_mean_response_time,
    mm1_mean_waiting_time,
    mm1_utilization,
    erlang_c,
    mmc_mean_queue_length,
    mmc_mean_response_time,
    mmc_mean_waiting_time,
    mmc_utilization,
    mg1_mean_queue_length,
    mg1_mean_system_length,
    mg1_mean_response_time,
    mg1_mean_waiting_time,
)

from .phase_type import (
    coxian2_from_moments,
    hyperexponential2_from_moments,
    CoxianPhaseParameters,
)

__all__ = [
    # M/M/1 queue
    "mm1_mean_queue_length",
    "mm1_mean_response_time",
    "mm1_mean_waiting_time",
    "mm1_utilization",
    # M/M/c queue
    "erlang_c",
    "mmc_mean_queue_length",
    "mmc_mean_response_time",
    "mmc_mean_waiting_time",
    "mmc_utilization",
    # M/G/1 queue (Pollaczek-Khinchine)
    "mg1_mean_queue_length",
    "mg1_mean_system_length",
    "mg1_mean_response_time",
    "mg1_mean_waiting_time",
    # Phase-type distributions
    "coxian2_from_moments",
    "hyperexponential2_from_moments",
    "CoxianPhaseParameters",
]
