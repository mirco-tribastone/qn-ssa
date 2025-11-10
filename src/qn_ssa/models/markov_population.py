"""
Markov Population Process (MPP) model specification.

A Markov population process is a continuous-time Markov chain (CTMC) where:
- State: X(t) ∈ ℤ^d (d-dimensional non-negative integer vector)
- Transitions: State jumps by vector ν_j with rate λ_j(X(t))
- Applications: Queueing networks, epidemic models, chemical reactions

This module provides a flexible specification framework for MPPs that can
be simulated using the Gillespie algorithm (stochastic simulation algorithm).
"""

from typing import Callable, List, Optional
import numpy as np


class MarkovPopulationProcess:
    """
    Specification of a Markov population process via jump vectors and rates.

    The process is defined by a collection of possible events, where each event:
    - Changes the state by a jump vector ν ∈ ℤ^d
    - Occurs at rate λ(x) when the state is x ∈ ℤ^d
    - Is labeled with a string describing the event type

    Attributes
    ----------
    state_dim : int
        Dimension d of the state space ℤ^d
    jump_vectors : List[np.ndarray]
        List of jump vectors ν_j, each of shape (d,)
    rate_functions : List[Callable[[np.ndarray], float]]
        List of rate functions λ_j(x) mapping state to transition rate
    event_labels : List[str]
        List of descriptive labels for each event type
    n_events : int
        Number of event types in the process

    Examples
    --------
    >>> # M/M/1 queue with arrival rate λ=0.8, service rate μ=1.0
    >>> mpp = MarkovPopulationProcess(state_dim=1)
    >>> mpp.add_event(np.array([1]), lambda x: 0.8, "ARRIVAL")
    >>> mpp.add_event(np.array([-1]), lambda x: 1.0 if x[0] > 0 else 0, "DEPARTURE")
    """

    def __init__(self, state_dim: int):
        """
        Initialize an empty Markov population process.

        Parameters
        ----------
        state_dim : int
            Dimension of the state space, must be positive
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        self.state_dim = state_dim
        self.jump_vectors: List[np.ndarray] = []
        self.rate_functions: List[Callable[[np.ndarray], float]] = []
        self.event_labels: List[str] = []

    @property
    def n_events(self) -> int:
        """Number of event types in the process."""
        return len(self.jump_vectors)

    def add_event(
        self,
        jump: np.ndarray,
        rate_fn: Callable[[np.ndarray], float],
        label: str,
    ) -> None:
        """
        Add an event type to the process.

        Parameters
        ----------
        jump : np.ndarray
            Jump vector ν of shape (state_dim,) specifying state change
        rate_fn : Callable[[np.ndarray], float]
            Function λ(x) computing the rate at which this event occurs in state x
            Must return a non-negative float
        label : str
            Descriptive label for this event type (e.g., "ARRIVAL", "DEPARTURE")

        Raises
        ------
        ValueError
            If jump vector has wrong dimension
        """
        jump = np.asarray(jump)

        # Validate jump vector dimension
        if jump.shape != (self.state_dim,):
            raise ValueError(
                f"Jump vector must have shape ({self.state_dim},), "
                f"got {jump.shape}"
            )

        self.jump_vectors.append(jump)
        self.rate_functions.append(rate_fn)
        self.event_labels.append(label)

    def get_propensities(self, state: np.ndarray) -> np.ndarray:
        """
        Compute propensities (rates) for all events at a given state.

        In the Gillespie algorithm, propensities determine:
        1. Time to next event: τ ~ Exp(sum of propensities)
        2. Which event occurs: probability ∝ propensity

        Parameters
        ----------
        state : np.ndarray
            Current state x ∈ ℤ^d of shape (state_dim,)

        Returns
        -------
        np.ndarray
            Array of propensities [λ_1(x), λ_2(x), ..., λ_n(x)]
            Shape: (n_events,)

        Notes
        -----
        Propensities must be non-negative. A propensity of zero means
        the corresponding event cannot occur in the current state.
        """
        state = np.asarray(state)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"State must have shape ({self.state_dim},), got {state.shape}"
            )

        # Evaluate all rate functions at current state
        propensities = np.array([rate_fn(state) for rate_fn in self.rate_functions])

        # Validate non-negativity
        if np.any(propensities < 0):
            raise ValueError(
                f"Rate functions must return non-negative values, "
                f"got {propensities}"
            )

        return propensities

    def validate(self) -> None:
        """
        Validate the process specification for consistency.

        Checks:
        - At least one event type is defined
        - All components (jumps, rates, labels) have same length

        Raises
        ------
        ValueError
            If the process specification is invalid
        """
        if self.n_events == 0:
            raise ValueError("Process must have at least one event type")

        if not (
            len(self.jump_vectors) == len(self.rate_functions) == len(self.event_labels)
        ):
            raise ValueError(
                f"Inconsistent event specification: "
                f"{len(self.jump_vectors)} jumps, "
                f"{len(self.rate_functions)} rate functions, "
                f"{len(self.event_labels)} labels"
            )

    def __repr__(self) -> str:
        """String representation of the process."""
        return (
            f"MarkovPopulationProcess(state_dim={self.state_dim}, "
            f"n_events={self.n_events})"
        )

    def __str__(self) -> str:
        """Human-readable description of the process."""
        lines = [
            f"Markov Population Process",
            f"  State dimension: {self.state_dim}",
            f"  Number of events: {self.n_events}",
        ]

        if self.n_events > 0:
            lines.append("  Events:")
            for i, (jump, label) in enumerate(zip(self.jump_vectors, self.event_labels)):
                lines.append(f"    {i}. {label:15s} jump={jump}")

        return "\n".join(lines)
