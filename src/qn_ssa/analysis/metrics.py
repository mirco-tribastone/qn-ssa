"""
Performance metrics for continuous-time stochastic simulations.

This module provides utilities for tracking time-weighted averages and
event counts in continuous-time Markov chain simulations.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimulationResults:
    """
    Container for simulation results with performance metrics.

    Attributes
    ----------
    mean_state : np.ndarray
        Time-averaged state vector E[X(t)] over the simulation horizon
    event_counts : Dict[str, float]
        Average number of occurrences for each event type across runs
    event_throughputs : Dict[str, float]
        Average throughput (events per unit time) for each event type
    time_horizon : float
        Total simulation time for each run
    n_runs : int
        Number of independent simulation runs
    """

    mean_state: np.ndarray
    event_counts: Dict[str, float]
    event_throughputs: Dict[str, float]
    time_horizon: float
    n_runs: int

    def __str__(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"Simulation Results ({self.n_runs} runs, T={self.time_horizon})",
            f"  Mean state: {self.mean_state}",
            f"  Event counts:",
        ]
        for event, count in self.event_counts.items():
            throughput = self.event_throughputs[event]
            lines.append(f"    {event:15s}: {count:10.2f} (λ={throughput:.4f})")
        return "\n".join(lines)


class MetricTracker:
    """
    Track time-weighted performance metrics during a simulation run.

    For continuous-time simulations, we compute time-weighted averages:
        E[X] = (1/T) ∫₀ᵀ X(t) dt

    This is approximated by summing X(t) * Δt over all state transitions.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space
    event_labels : List[str]
        Labels for tracking event occurrences

    Examples
    --------
    >>> tracker = MetricTracker(state_dim=2, event_labels=["ARRIVAL", "DEPARTURE"])
    >>> # Simulate: state [1, 0] for 0.5 time units
    >>> tracker.update(np.array([1, 0]), 0.5)
    >>> tracker.record_event("ARRIVAL")
    >>> # State [2, 0] for 0.3 time units
    >>> tracker.update(np.array([2, 0]), 0.3)
    >>> mean_state = tracker.get_mean_state(total_time=0.8)
    """

    def __init__(self, state_dim: int, event_labels: List[str]):
        """
        Initialize a metric tracker.

        Parameters
        ----------
        state_dim : int
            Dimension of the state space
        event_labels : List[str]
            List of event labels to track
        """
        self.state_dim = state_dim
        self.event_labels = event_labels

        # Accumulator for time-weighted state sum
        # Stores ∑ X(t) * Δt over the simulation
        self._state_time_integral = np.zeros(state_dim)

        # Event occurrence counters
        self._event_counts = {label: 0 for label in event_labels}

    def update(self, state: np.ndarray, time_delta: float) -> None:
        """
        Update time-weighted state accumulator.

        Call this method after each state transition to accumulate
        the contribution X(t) * Δt to the time integral.

        Parameters
        ----------
        state : np.ndarray
            Current state X(t)
        time_delta : float
            Time duration Δt spent in this state (until next event)
        """
        if time_delta < 0:
            raise ValueError(f"time_delta must be non-negative, got {time_delta}")

        # Accumulate X(t) * Δt for time-weighted average
        self._state_time_integral += state * time_delta

    def record_event(self, event_label: str) -> None:
        """
        Record occurrence of an event.

        Parameters
        ----------
        event_label : str
            Label of the event that occurred
        """
        if event_label not in self._event_counts:
            raise ValueError(f"Unknown event label: {event_label}")

        self._event_counts[event_label] += 1

    def get_mean_state(self, total_time: float) -> np.ndarray:
        """
        Compute time-averaged state E[X(t)].

        Parameters
        ----------
        total_time : float
            Total simulation time T

        Returns
        -------
        np.ndarray
            Time-averaged state (1/T) ∫₀ᵀ X(t) dt
        """
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")

        return self._state_time_integral / total_time

    def get_event_counts(self) -> Dict[str, int]:
        """
        Get event occurrence counts.

        Returns
        -------
        Dict[str, int]
            Number of times each event occurred during simulation
        """
        return self._event_counts.copy()

    def get_event_throughputs(self, total_time: float) -> Dict[str, float]:
        """
        Compute event throughputs (events per unit time).

        For a queueing system, throughput represents the rate at which
        events occur: λ_observed = N_events / T

        Parameters
        ----------
        total_time : float
            Total simulation time T

        Returns
        -------
        Dict[str, float]
            Throughput (events/time) for each event type
        """
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")

        return {
            label: count / total_time for label, count in self._event_counts.items()
        }

    def reset(self) -> None:
        """Reset all accumulators for a new simulation run."""
        self._state_time_integral = np.zeros(self.state_dim)
        self._event_counts = {label: 0 for label in self.event_labels}
