"""
Gillespie's Stochastic Simulation Algorithm (Direct Method).

The Gillespie algorithm (Gillespie, 1977) is an exact Monte Carlo method
for simulating continuous-time Markov chains, particularly Markov population
processes arising in chemical kinetics, epidemic models, and queueing systems.

References
----------
Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
reactions. The Journal of Physical Chemistry, 81(25), 2340-2361.
"""

from typing import Optional
import numpy as np

from ..models.markov_population import MarkovPopulationProcess
from ..analysis.metrics import MetricTracker, SimulationResults


class GillespieSimulator:
    """
    Exact stochastic simulation of Markov population processes.

    The Gillespie algorithm simulates the exact trajectory of a continuous-time
    Markov chain by:
    1. Computing propensities (transition rates) for all possible events
    2. Sampling time to next event from Exp(sum of propensities)
    3. Selecting which event occurs with probability proportional to propensities
    4. Updating the state and repeating

    The algorithm is exact in the sense that it generates sample paths from
    the true stochastic process (not an approximation).

    Examples
    --------
    >>> # M/M/1 queue simulation
    >>> mpp = MarkovPopulationProcess(state_dim=1)
    >>> mpp.add_event(np.array([1]), lambda x: 0.8, "ARRIVAL")
    >>> mpp.add_event(np.array([-1]), lambda x: 1.0 if x[0] > 0 else 0, "DEPARTURE")
    >>>
    >>> simulator = GillespieSimulator()
    >>> results = simulator.simulate(
    ...     mpp=mpp,
    ...     initial_state=np.array([0]),
    ...     time_horizon=1000.0,
    ...     n_runs=100,
    ...     seed=42
    ... )
    >>> print(f"Mean queue length: {results.mean_state[0]:.3f}")
    """

    def simulate(
        self,
        mpp: MarkovPopulationProcess,
        initial_state: np.ndarray,
        time_horizon: float,
        n_runs: int = 1,
        seed: Optional[int] = None,
    ) -> SimulationResults:
        """
        Run Gillespie simulation for a Markov population process.

        Parameters
        ----------
        mpp : MarkovPopulationProcess
            The process to simulate (jump vectors and rate functions)
        initial_state : np.ndarray
            Initial state X(0) ∈ ℤ^d
        time_horizon : float
            Simulation time horizon T (stop when t ≥ T)
        n_runs : int, optional
            Number of independent simulation runs (default: 1)
        seed : int, optional
            Random seed for reproducibility (default: None)

        Returns
        -------
        SimulationResults
            Container with time-averaged metrics and event statistics

        Raises
        ------
        ValueError
            If parameters are invalid or inconsistent
        """
        # Validate inputs
        mpp.validate()
        initial_state = np.asarray(initial_state)

        if initial_state.shape != (mpp.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({mpp.state_dim},), "
                f"got {initial_state.shape}"
            )

        if time_horizon <= 0:
            raise ValueError(f"time_horizon must be positive, got {time_horizon}")

        if n_runs <= 0:
            raise ValueError(f"n_runs must be positive, got {n_runs}")

        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(seed)

        # Accumulators for averaging across runs
        total_mean_state = np.zeros(mpp.state_dim)
        total_event_counts = {label: 0 for label in mpp.event_labels}

        # Run multiple independent replications
        for run_idx in range(n_runs):
            # Run single simulation trajectory
            mean_state, event_counts = self._single_run(
                mpp=mpp,
                initial_state=initial_state.copy(),
                time_horizon=time_horizon,
                rng=rng,
            )

            # Accumulate results across runs
            total_mean_state += mean_state
            for label, count in event_counts.items():
                total_event_counts[label] += count

        # Compute averages over all runs
        avg_mean_state = total_mean_state / n_runs
        avg_event_counts = {
            label: count / n_runs for label, count in total_event_counts.items()
        }
        avg_event_throughputs = {
            label: count / time_horizon for label, count in avg_event_counts.items()
        }

        return SimulationResults(
            mean_state=avg_mean_state,
            event_counts=avg_event_counts,
            event_throughputs=avg_event_throughputs,
            time_horizon=time_horizon,
            n_runs=n_runs,
        )

    def _single_run(
        self,
        mpp: MarkovPopulationProcess,
        initial_state: np.ndarray,
        time_horizon: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Execute a single Gillespie simulation run.

        This implements the direct method of the Gillespie algorithm:
        1. Initialize state X(0) and time t = 0
        2. While t < T:
           a. Compute propensities a_j = λ_j(X(t)) for all events j
           b. Sample time to next event: τ ~ Exp(a_0) where a_0 = Σa_j
           c. Select event j with probability a_j / a_0
           d. Update state: X(t+τ) = X(t) + ν_j
           e. Update time: t ← t + τ

        Parameters
        ----------
        mpp : MarkovPopulationProcess
            Process specification
        initial_state : np.ndarray
            Initial state (will be modified in-place)
        time_horizon : float
            Simulation time horizon
        rng : np.random.Generator
            NumPy random number generator

        Returns
        -------
        mean_state : np.ndarray
            Time-averaged state over [0, T]
        event_counts : Dict[str, int]
            Number of occurrences of each event type
        """
        # Initialize state and time
        state = initial_state  # Current state X(t)
        time = 0.0  # Current simulation time t

        # Create metric tracker for this run
        tracker = MetricTracker(
            state_dim=mpp.state_dim, event_labels=mpp.event_labels
        )

        # Gillespie algorithm main loop
        while time < time_horizon:
            # Step 1: Compute propensities (rates) for all possible events
            # a_j = λ_j(X(t)) for j = 1, ..., n_events
            propensities = mpp.get_propensities(state)
            total_propensity = np.sum(propensities)

            # Check if any event can occur
            if total_propensity == 0:
                # No events possible: system is "stuck" (absorbing state)
                # Accumulate current state until time horizon
                tracker.update(state, time_horizon - time)
                break

            # Step 2: Sample time to next event
            # τ ~ Exponential(a_0) where a_0 = sum of all propensities
            # This follows from the memoryless property of exponential distributions
            # and the fact that min of exponentials is exponential with rate = sum of rates
            time_to_next_event = rng.exponential(1.0 / total_propensity)

            # Check if next event occurs within time horizon
            if time + time_to_next_event > time_horizon:
                # Next event would occur after T: stop simulation
                # Accumulate current state for remaining time
                tracker.update(state, time_horizon - time)
                break

            # Accumulate current state for the time until next event
            tracker.update(state, time_to_next_event)

            # Advance time to next event
            time += time_to_next_event

            # Step 3: Select which event occurs
            # Event j is chosen with probability a_j / a_0
            # This is equivalent to sampling from a categorical distribution
            probabilities = propensities / total_propensity
            event_idx = rng.choice(mpp.n_events, p=probabilities)

            # Step 4: Update state by the jump vector
            # X(t) ← X(t) + ν_j
            state = state + mpp.jump_vectors[event_idx]

            # Record event occurrence
            tracker.record_event(mpp.event_labels[event_idx])

        # Extract metrics from tracker
        mean_state = tracker.get_mean_state(time_horizon)
        event_counts = tracker.get_event_counts()

        return mean_state, event_counts
