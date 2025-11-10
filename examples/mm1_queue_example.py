"""
M/M/1 Queue Simulation and Validation Example.

This example demonstrates:
1. How to specify an M/M/1 queue as a Markov population process
2. Running Gillespie simulation
3. Validating results against analytical formulas

M/M/1 Queue Specification
-------------------------
- State: X(t) = number of customers in system (queue + service)
- Events:
  * ARRIVAL: X → X+1 with rate λ (constant)
  * DEPARTURE: X → X-1 with rate μ·𝟙(X>0) (only if customers present)
- Steady-state exists when ρ = λ/μ < 1

Analytical Results (Kleinrock, 1975)
------------------------------------
- Utilization: ρ = λ/μ
- Mean number in system: L = ρ/(1-ρ) = λ/(μ-λ)
- Mean response time: W = 1/(μ-λ)
- Mean waiting time: Wq = λ/(μ(μ-λ))

These follow from the geometric steady-state distribution:
    P(N=n) = (1-ρ)ρⁿ for n = 0, 1, 2, ...
"""

import numpy as np
from qn_ssa.models import MarkovPopulationProcess
from qn_ssa.simulators import GillespieSimulator
from qn_ssa.utils import (
    mm1_mean_queue_length,
    mm1_mean_response_time,
    mm1_mean_waiting_time,
    mm1_utilization,
)


def create_mm1_queue(lambda_rate: float, mu_rate: float) -> MarkovPopulationProcess:
    """
    Create an M/M/1 queue as a Markov population process.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ (customers per unit time)
    mu_rate : float
        Service rate μ (customers per unit time)

    Returns
    -------
    MarkovPopulationProcess
        MPP specification with ARRIVAL and DEPARTURE events
    """
    # State vector has dimension 1: [number in system]
    mpp = MarkovPopulationProcess(state_dim=1)

    # Event 1: ARRIVAL - adds one customer to the system
    # Rate: λ (constant, independent of state)
    mpp.add_event(
        jump=np.array([1]),  # Increment state by 1
        rate_fn=lambda x: lambda_rate,  # Constant arrival rate
        label="ARRIVAL",
    )

    # Event 2: DEPARTURE - removes one customer from the system
    # Rate: μ if X > 0, else 0 (can only depart if customers present)
    mpp.add_event(
        jump=np.array([-1]),  # Decrement state by 1
        rate_fn=lambda x: mu_rate if x[0] > 0 else 0.0,  # Service rate when busy
        label="DEPARTURE",
    )

    return mpp


def main():
    """Run M/M/1 queue simulation and compare with analytical results."""

    print("=" * 70)
    print("M/M/1 Queue Simulation - Gillespie Algorithm")
    print("=" * 70)

    # Queue parameters
    lambda_rate = 0.8  # Arrival rate
    mu_rate = 1.0  # Service rate
    rho = mm1_utilization(lambda_rate, mu_rate)

    print(f"\nQueue Parameters:")
    print(f"  Arrival rate (λ):        {lambda_rate}")
    print(f"  Service rate (μ):        {mu_rate}")
    print(f"  Utilization (ρ = λ/μ):   {rho:.3f}")
    print(f"  Stability: {'✓ Stable (ρ < 1)' if rho < 1 else '✗ Unstable (ρ ≥ 1)'}")

    # Analytical results
    analytical_l = mm1_mean_queue_length(lambda_rate, mu_rate)
    analytical_w = mm1_mean_response_time(lambda_rate, mu_rate)
    analytical_wq = mm1_mean_waiting_time(lambda_rate, mu_rate)

    print(f"\nAnalytical Results (Closed-form Formulas):")
    print(f"  Mean number in system (L):     {analytical_l:.6f}")
    print(f"  Mean response time (W):        {analytical_w:.6f}")
    print(f"  Mean waiting time (Wq):        {analytical_wq:.6f}")
    print(f"  Arrival throughput (λ):        {lambda_rate:.6f}")

    # Create M/M/1 queue model
    mpp = create_mm1_queue(lambda_rate, mu_rate)
    print(f"\n{mpp}")

    # Simulation parameters
    time_horizon = 10000.0  # Long simulation for steady-state
    n_runs = 100  # Multiple runs for averaging
    seed = 42  # Reproducibility

    print(f"\nSimulation Parameters:")
    print(f"  Time horizon (T):        {time_horizon}")
    print(f"  Number of runs:          {n_runs}")
    print(f"  Random seed:             {seed}")

    # Run simulation
    print(f"\nRunning Gillespie simulation...")
    simulator = GillespieSimulator()
    results = simulator.simulate(
        mpp=mpp,
        initial_state=np.array([0]),  # Start empty
        time_horizon=time_horizon,
        n_runs=n_runs,
        seed=seed,
    )

    # Extract simulation results
    simulated_l = results.mean_state[0]
    simulated_throughput = results.event_throughputs["DEPARTURE"]
    # Use Little's Law to compute response time from simulation
    simulated_w = simulated_l / simulated_throughput if simulated_throughput > 0 else 0
    simulated_wq = simulated_w - 1.0 / mu_rate

    print(f"\nSimulation Results:")
    print(f"  Mean number in system (L):     {simulated_l:.6f}")
    print(f"  Mean response time (W):        {simulated_w:.6f}")
    print(f"  Mean waiting time (Wq):        {simulated_wq:.6f}")
    print(f"  Departure throughput:          {simulated_throughput:.6f}")
    print(f"  Arrival count (avg):           {results.event_counts['ARRIVAL']:.1f}")
    print(f"  Departure count (avg):         {results.event_counts['DEPARTURE']:.1f}")

    # Validation: compare simulation with analytical results
    print(f"\n" + "=" * 70)
    print("Validation: Simulation vs. Analytical")
    print("=" * 70)

    l_error = abs(simulated_l - analytical_l) / analytical_l * 100
    w_error = abs(simulated_w - analytical_w) / analytical_w * 100
    wq_error = abs(simulated_wq - analytical_wq) / analytical_wq * 100
    throughput_error = abs(simulated_throughput - lambda_rate) / lambda_rate * 100

    print(f"\nMetric                     Analytical    Simulated     Error")
    print(f"-" * 70)
    print(
        f"Mean queue length (L):     {analytical_l:10.6f}  {simulated_l:10.6f}  "
        f"{l_error:6.2f}%"
    )
    print(
        f"Mean response time (W):    {analytical_w:10.6f}  {simulated_w:10.6f}  "
        f"{w_error:6.2f}%"
    )
    print(
        f"Mean waiting time (Wq):    {analytical_wq:10.6f}  {simulated_wq:10.6f}  "
        f"{wq_error:6.2f}%"
    )
    print(
        f"Throughput (λ):            {lambda_rate:10.6f}  "
        f"{simulated_throughput:10.6f}  {throughput_error:6.2f}%"
    )

    # Check if validation is successful (errors < 5%)
    threshold = 5.0
    all_errors = [l_error, w_error, wq_error, throughput_error]

    print(f"\nValidation threshold: {threshold}%")
    if all(error < threshold for error in all_errors):
        print("✓ VALIDATION PASSED: All metrics within threshold")
    else:
        print("✗ VALIDATION WARNING: Some metrics exceed threshold")
        print("  (This may indicate insufficient simulation time or runs)")

    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    main()
