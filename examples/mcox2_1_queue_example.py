"""
M/Cox2/1 Queue Simulation and Validation Example.

This example demonstrates:
1. How to specify an M/Cox2/1 queue (Coxian-2 service) as a Markov population process
2. 2D state space to track queue length and service phase
3. Validation against Pollaczek-Khinchine (P-K) formula for M/G/1 queues

M/Cox2/1 Queue Specification
-----------------------------
- Arrivals: Poisson process with rate λ
- Service: Coxian-2 distribution (2-phase sequential service)
  * Phase 1: exponential with rate μ₁
  * With probability p: continue to phase 2 (rate μ₂)
  * With probability (1-p): depart after phase 1
- State: [queue_length, service_phase]
  * queue_length: total customers in system
  * service_phase: {0=idle, 1=phase1, 2=phase2}

Analytical Validation (Pollaczek-Khinchine Formula)
---------------------------------------------------
Since arrivals are Poisson and service is general (Coxian-2 is a phase-type
distribution), the M/Cox2/1 queue is a special case of M/G/1.

The P-K formula gives exact performance metrics based on:
- Mean service time: E[S] = 1/μ₁ + p/μ₂
- Service SCV: c²ₛ = Var(S)/E[S]²

Key Results:
- Mean waiting time: Wq = (λ · E[S²]) / (2(1 - ρ))
- Mean queue length: Lq = (ρ² · (1 + c²ₛ)) / (2(1 - ρ))
- Mean in system: L = Lq + ρ
- Mean response time: W = Wq + E[S]

where ρ = λ · E[S] is utilization.
"""

import numpy as np
from qn_ssa.models import MarkovPopulationProcess
from qn_ssa.simulators import GillespieSimulator
from qn_ssa.utils import (
    coxian2_from_moments,
    mg1_mean_system_length,
    mg1_mean_response_time,
    mg1_mean_waiting_time,
    mg1_mean_queue_length,
)


def create_mcox2_1_queue(
    lambda_rate: float, mu1: float, mu2: float, p_cont: float
) -> MarkovPopulationProcess:
    """
    Create an M/Cox2/1 queue as a Markov population process.

    State space is 2-dimensional:
    - state[0] = n: total number of customers in system (≥ 0)
    - state[1] = p: service phase indicator {0, 1}
      * p = 0: customer (if any) in phase 1, or idle if n = 0
      * p = 1: customer in phase 2 (implies n > 0)

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ
    mu1 : float
        Phase 1 service rate μ₁
    mu2 : float
        Phase 2 service rate μ₂
    p_cont : float
        Probability of continuing to phase 2 after phase 1 (0 < p_cont ≤ 1)

    Returns
    -------
    MarkovPopulationProcess
        MPP specification with 2D state space and 4 events
    """
    # State vector: [n, p] where n=total in system, p=phase indicator
    mpp = MarkovPopulationProcess(state_dim=2)

    # Event 1: ARRIVAL
    # Customer arrives (Poisson process)
    # Increment n, phase unchanged
    mpp.add_event(
        jump=np.array([1, 0]),
        rate_fn=lambda x: lambda_rate,
        label="ARRIVAL",
    )

    # Event 2: PHASE1_TO_PHASE2
    # Complete phase 1, continue to phase 2
    # n unchanged, phase: 0 → 1
    # Only possible when n > 0 and currently in phase 1 (p=0)
    mpp.add_event(
        jump=np.array([0, 1]),
        rate_fn=lambda x: p_cont * mu1 if (x[0] > 0 and x[1] == 0) else 0.0,
        label="PHASE1_TO_PHASE2",
    )

    # Event 3: PHASE1_DEPART
    # Complete phase 1 and depart (skip phase 2)
    # Decrement n, phase stays 0 (next customer starts phase 1 if any)
    # Only possible when n > 0 and currently in phase 1 (p=0)
    mpp.add_event(
        jump=np.array([-1, 0]),
        rate_fn=lambda x: (1 - p_cont) * mu1 if (x[0] > 0 and x[1] == 0) else 0.0,
        label="PHASE1_DEPART",
    )

    # Event 4: PHASE2_DEPART
    # Complete phase 2 and depart
    # Decrement n, phase: 1 → 0 (next customer starts phase 1 if any)
    # Only possible when n > 0 and currently in phase 2 (p=1)
    mpp.add_event(
        jump=np.array([-1, -1]),
        rate_fn=lambda x: mu2 if (x[0] > 0 and x[1] == 1) else 0.0,
        label="PHASE2_DEPART",
    )

    return mpp


def main():
    """Run M/Cox2/1 queue simulation and compare with P-K formula."""

    print("=" * 70)
    print("M/Cox2/1 Queue Simulation - Coxian-2 Service Distribution")
    print("=" * 70)

    # Target service parameters: high SCV to see impact of variability
    target_mean = 1.0
    target_scv = 4.0
    p_continue = 0.2  # Probability of continuing to phase 2

    print(f"\nTarget Service Distribution:")
    print(f"  Mean service time:         {target_mean}")
    print(f"  Service SCV (c²):          {target_scv}")
    print(f"  Continuation prob (p):     {p_continue}")

    # Compute Coxian-2 parameters
    cox2_params = coxian2_from_moments(target_mean, target_scv, p_continue)
    print(f"\n{cox2_params}")

    # Queue parameters
    lambda_rate = 0.8  # Same as M/M/1 example for comparison
    rho = lambda_rate * cox2_params.mean

    print(f"\nQueue Parameters:")
    print(f"  Arrival rate (λ):          {lambda_rate}")
    print(f"  Utilization (ρ = λ·E[S]):  {rho:.3f}")
    print(f"  Stability: {'✓ Stable (ρ < 1)' if rho < 1 else '✗ Unstable (ρ ≥ 1)'}")

    # Analytical results using Pollaczek-Khinchine formula
    analytical_lq = mg1_mean_queue_length(lambda_rate, cox2_params.mean, cox2_params.scv)
    analytical_l = mg1_mean_system_length(lambda_rate, cox2_params.mean, cox2_params.scv)
    analytical_wq = mg1_mean_waiting_time(lambda_rate, cox2_params.mean, cox2_params.scv)
    analytical_w = mg1_mean_response_time(lambda_rate, cox2_params.mean, cox2_params.scv)

    print(f"\nAnalytical Results (Pollaczek-Khinchine Formula for M/G/1):")
    print(f"  Mean in queue (Lq):        {analytical_lq:.6f}")
    print(f"  Mean in system (L):        {analytical_l:.6f}")
    print(f"  Mean waiting time (Wq):    {analytical_wq:.6f}")
    print(f"  Mean response time (W):    {analytical_w:.6f}")

    # Create M/Cox2/1 queue model
    mpp = create_mcox2_1_queue(
        lambda_rate=lambda_rate,
        mu1=cox2_params.mu1,
        mu2=cox2_params.mu2,
        p_cont=cox2_params.p,
    )
    print(f"\n{mpp}")

    # Simulation parameters
    time_horizon = 10000.0
    n_runs = 100
    seed = 42

    print(f"\nSimulation Parameters:")
    print(f"  Time horizon (T):          {time_horizon}")
    print(f"  Number of runs:            {n_runs}")
    print(f"  Random seed:               {seed}")

    # Run simulation
    print(f"\nRunning Gillespie simulation...")
    simulator = GillespieSimulator()
    results = simulator.simulate(
        mpp=mpp,
        initial_state=np.array([0, 0]),  # Start empty and idle
        time_horizon=time_horizon,
        n_runs=n_runs,
        seed=seed,
    )

    # Extract simulation results
    # state[0] = n (total customers in system)
    # state[1] = p (phase indicator: 0 or 1)
    simulated_l = results.mean_state[0]  # This is already total in system!

    # Compute departure throughput
    phase1_departs = results.event_throughputs.get("PHASE1_DEPART", 0)
    phase2_departs = results.event_throughputs.get("PHASE2_DEPART", 0)
    simulated_throughput = phase1_departs + phase2_departs

    # Use Little's Law to compute response time from simulation
    simulated_w = simulated_l / simulated_throughput if simulated_throughput > 0 else 0
    simulated_wq = simulated_w - cox2_params.mean
    simulated_lq = simulated_l - rho

    print(f"\nSimulation Results:")
    print(f"  Mean in queue (Lq):        {simulated_lq:.6f}")
    print(f"  Mean in system (L):        {simulated_l:.6f}")
    print(f"  Mean waiting time (Wq):    {simulated_wq:.6f}")
    print(f"  Mean response time (W):    {simulated_w:.6f}")
    print(f"  Departure throughput:      {simulated_throughput:.6f}")
    print(f"  Phase 1 departures/s:      {phase1_departs:.6f}")
    print(f"  Phase 2 departures/s:      {phase2_departs:.6f}")
    print(f"  Arrival count (avg):       {results.event_counts['ARRIVAL']:.1f}")

    # Validation: compare simulation with P-K analytical results
    print(f"\n" + "=" * 70)
    print("Validation: Simulation vs. Pollaczek-Khinchine (M/G/1)")
    print("=" * 70)

    lq_error = abs(simulated_lq - analytical_lq) / analytical_lq * 100
    l_error = abs(simulated_l - analytical_l) / analytical_l * 100
    wq_error = abs(simulated_wq - analytical_wq) / analytical_wq * 100
    w_error = abs(simulated_w - analytical_w) / analytical_w * 100
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
        f"Mean waiting time (Wq):    {analytical_wq:10.6f}  {simulated_wq:10.6f}  "
        f"{wq_error:6.2f}%"
    )
    print(
        f"Mean response time (W):    {analytical_w:10.6f}  {simulated_w:10.6f}  "
        f"{w_error:6.2f}%"
    )
    print(
        f"Throughput (λ):            {lambda_rate:10.6f}  "
        f"{simulated_throughput:10.6f}  {throughput_error:6.2f}%"
    )

    # Check if validation is successful (errors < 5%)
    threshold = 5.0
    all_errors = [lq_error, l_error, wq_error, w_error, throughput_error]

    print(f"\nValidation threshold: {threshold}%")
    if all(error < threshold for error in all_errors):
        print("✓ VALIDATION PASSED: All metrics within threshold")
    else:
        print("✗ VALIDATION WARNING: Some metrics exceed threshold")
        print("  (This may indicate insufficient simulation time or runs)")

    # Display phase analysis
    print(f"\n" + "=" * 70)
    print("Service Phase Analysis")
    print("=" * 70)

    phase1_frac = phase1_departs / simulated_throughput if simulated_throughput > 0 else 0
    phase2_frac = phase2_departs / simulated_throughput if simulated_throughput > 0 else 0

    print(f"\n  Expected phase 1 only: {1-p_continue:.3f}")
    print(f"  Simulated phase 1 only: {phase1_frac:.3f}")
    print(f"\n  Expected phase 2 (both phases): {p_continue:.3f}")
    print(f"  Simulated phase 2 (both phases): {phase2_frac:.3f}")

    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    main()
