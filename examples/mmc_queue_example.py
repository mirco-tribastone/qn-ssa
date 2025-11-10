"""
M/M/c Queue Simulation and Validation Example.

This example demonstrates:
1. How to specify an M/M/c queue (c parallel servers) as a Markov population process
2. Running Gillespie simulation
3. Validating results against analytical formulas using Erlang-C

M/M/c Queue Specification
-------------------------
- State: X(t) = number of customers in system (queue + service)
- Number of servers: c
- Events:
  * ARRIVAL: X → X+1 with rate λ (constant)
  * DEPARTURE: X → X-1 with rate μ·min(X, c) (c parallel servers)
- Steady-state exists when ρ = λ/(c·μ) < 1

Analytical Results (Kleinrock, 1975)
------------------------------------
- Utilization: ρ = λ/(c·μ)
- Erlang-C formula: C(c, a) = probability of waiting
- Mean in queue: Lq = C(c, a)·ρ/(1-ρ)
- Mean in system: L = Lq + λ/μ
- Mean response time: W = L/λ (Little's Law)
- Mean waiting time: Wq = Lq/λ

where a = λ/μ is the offered load (average number of busy servers).
"""

import numpy as np
from qn_ssa.models import MarkovPopulationProcess
from qn_ssa.simulators import GillespieSimulator
from qn_ssa.utils import (
    erlang_c,
    mmc_mean_queue_length,
    mmc_mean_response_time,
    mmc_mean_waiting_time,
    mmc_utilization,
)


def create_mmc_queue(
    lambda_rate: float, mu_rate: float, c: int
) -> MarkovPopulationProcess:
    """
    Create an M/M/c queue as a Markov population process.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ (customers per unit time)
    mu_rate : float
        Service rate per server μ (customers per unit time)
    c : int
        Number of parallel servers

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
    # Rate: μ·min(X, c) where X is number in system
    # - If X ≤ c: all customers are in service, rate = μ·X
    # - If X > c: all c servers busy, rate = μ·c
    mpp.add_event(
        jump=np.array([-1]),  # Decrement state by 1
        rate_fn=lambda x: mu_rate * min(x[0], c),  # c parallel servers
        label="DEPARTURE",
    )

    return mpp


def main():
    """Run M/M/c queue simulation and compare with analytical results."""

    print("=" * 70)
    print("M/M/c Queue Simulation - Gillespie Algorithm")
    print("=" * 70)

    # Queue parameters
    lambda_rate = 4.0  # Arrival rate
    mu_rate = 1.0  # Service rate per server
    c = 5  # Number of servers
    rho = mmc_utilization(lambda_rate, mu_rate, c)
    offered_load = lambda_rate / mu_rate

    print(f"\nQueue Parameters:")
    print(f"  Arrival rate (λ):             {lambda_rate}")
    print(f"  Service rate per server (μ):  {mu_rate}")
    print(f"  Number of servers (c):        {c}")
    print(f"  Offered load (a = λ/μ):       {offered_load:.3f}")
    print(f"  Utilization (ρ = λ/(c·μ)):    {rho:.3f}")
    print(f"  Stability: {'✓ Stable (ρ < 1)' if rho < 1 else '✗ Unstable (ρ ≥ 1)'}")

    # Analytical results
    erlang_c_val = erlang_c(c, rho)
    analytical_l = mmc_mean_queue_length(lambda_rate, mu_rate, c)
    analytical_w = mmc_mean_response_time(lambda_rate, mu_rate, c)
    analytical_wq = mmc_mean_waiting_time(lambda_rate, mu_rate, c)
    analytical_lq = analytical_l - offered_load  # Mean in queue only

    print(f"\nAnalytical Results (Erlang-C Formulas):")
    print(f"  Erlang-C (prob. of waiting):   {erlang_c_val:.6f}")
    print(f"  Mean number in queue (Lq):     {analytical_lq:.6f}")
    print(f"  Mean number in system (L):     {analytical_l:.6f}")
    print(f"  Mean response time (W):        {analytical_w:.6f}")
    print(f"  Mean waiting time (Wq):        {analytical_wq:.6f}")
    print(f"  Throughput (λ):                {lambda_rate:.6f}")

    # Create M/M/c queue model
    mpp = create_mmc_queue(lambda_rate, mu_rate, c)
    print(f"\n{mpp}")

    # Simulation parameters
    time_horizon = 10000.0  # Long simulation for steady-state
    n_runs = 100  # Multiple runs for averaging
    seed = 42  # Reproducibility

    print(f"\nSimulation Parameters:")
    print(f"  Time horizon (T):              {time_horizon}")
    print(f"  Number of runs:                {n_runs}")
    print(f"  Random seed:                   {seed}")

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
    simulated_lq = simulated_l - offered_load

    print(f"\nSimulation Results:")
    print(f"  Mean number in queue (Lq):     {simulated_lq:.6f}")
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

    lq_error = abs(simulated_lq - analytical_lq) / analytical_lq * 100 if analytical_lq > 0 else 0
    l_error = abs(simulated_l - analytical_l) / analytical_l * 100
    w_error = abs(simulated_w - analytical_w) / analytical_w * 100
    wq_error = (
        abs(simulated_wq - analytical_wq) / analytical_wq * 100 if analytical_wq > 0 else 0
    )
    throughput_error = abs(simulated_throughput - lambda_rate) / lambda_rate * 100

    print(f"\nMetric                     Analytical    Simulated     Error")
    print(f"-" * 70)
    print(
        f"Mean in queue (Lq):        {analytical_lq:10.6f}  {simulated_lq:10.6f}  "
        f"{lq_error:6.2f}%"
    )
    print(
        f"Mean in system (L):        {analytical_l:10.6f}  {simulated_l:10.6f}  "
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
    all_errors = [lq_error, l_error, w_error, wq_error, throughput_error]

    print(f"\nValidation threshold: {threshold}%")
    if all(error < threshold for error in all_errors):
        print("✓ VALIDATION PASSED: All metrics within threshold")
    else:
        print("✗ VALIDATION WARNING: Some metrics exceed threshold")
        print("  (This may indicate insufficient simulation time or runs)")

    print(f"\n" + "=" * 70)

    # Additional insights
    print("\nQueue Behavior Insights:")
    print(f"  Average busy servers: {min(offered_load, c):.2f} out of {c}")
    print(f"  Probability of waiting: {erlang_c_val:.3f}")
    if erlang_c_val < 0.01:
        print("  → Very low congestion: customers rarely wait")
    elif erlang_c_val < 0.1:
        print("  → Low congestion: most customers served immediately")
    elif erlang_c_val < 0.5:
        print("  → Moderate congestion: some queueing occurs")
    else:
        print("  → High congestion: significant queueing occurs")

    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    main()
