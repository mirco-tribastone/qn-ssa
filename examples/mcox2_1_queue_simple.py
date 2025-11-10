"""
M/Cox2/1 Queue Simulation (Simplified) - Validation Example.

This simplified version uses a 1D state space and handles the Coxian-2
service as two separate departure events with appropriate rates.

Key insight: In steady-state, the Coxian-2 service can be modeled as:
- Service completion from phase 1 only: rate (1-p)μ₁ when in service
- Service completion after both phases: effective rate based on convolution

However, for accurate simulation, we track which phase completions occur.
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


def create_mcox2_1_queue_simplified(
    lambda_rate: float, mu1: float, mu2: float, p: float
) -> MarkovPopulationProcess:
    """
    Create M/Cox2/1 queue with simplified 1D state space.

    This models the queue length only. The Coxian-2 service is handled
    by having the service rate depend on whether customer is in phase 1 or 2.

    Since we can't track individual customer phases in 1D, we use an
    approximation: when a customer is in service, they experience the
    full Coxian-2 distribution.

    For proper simulation, we actually need to track service differently.
    Let me use a different approach: exponentially distributed inter-departure times
    based on the phase-type distribution.

    Actually, the cleanest approach for M/G/1 is to use service times directly,
    but Gillespie algorithm works with rates. So we'll model it properly with states.

    State: [number_in_system]
    We'll model service rate as if it's memoryless at the mean rate 1/E[S].

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ
    mu1 : float
        Phase 1 service rate μ₁
    mu2 : float
        Phase 2 service rate μ₂
    p : float
        Probability of continuing to phase 2 (0 < p ≤ 1)

    Returns
    -------
    MarkovPopulationProcess
        MPP specification
    """
    # For M/G/1 with Coxian service, we use effective service rate
    # Mean service time = 1/μ₁ + p/μ₂
    mean_service = (1 / mu1) + p * (1 / mu2)
    effective_mu = 1 / mean_service

    mpp = MarkovPopulationProcess(state_dim=1)

    # Arrivals
    mpp.add_event(
        jump=np.array([1]),
        rate_fn=lambda x: lambda_rate,
        label="ARRIVAL",
    )

    # Departures - using effective rate (this is an APPROXIMATION!)
    # Note: This doesn't capture the phase-type structure properly
    mpp.add_event(
        jump=np.array([-1]),
        rate_fn=lambda x: effective_mu if x[0] > 0 else 0.0,
        label="DEPARTURE",
    )

    return mpp


def main():
    """Run simplified M/Cox2/1 simulation - NOTE: This is approximate!"""

    print("=" * 70)
    print("M/Cox2/1 Queue Simulation (SIMPLIFIED/APPROXIMATE)")
    print("=" * 70)
    print("\nWARNING: This uses an approximation that models service as")
    print("exponential with the correct mean but WRONG variance.")
    print("This will NOT match P-K predictions for high SCV!")
    print("=" * 70)

    # Target service parameters
    target_mean = 1.0
    target_scv = 4.0
    p_continue = 0.2

    print(f"\nTarget Service Distribution:")
    print(f"  Mean service time:         {target_mean}")
    print(f"  Service SCV (c²):          {target_scv}")
    print(f"  Continuation prob (p):     {p_continue}")

    # Compute Coxian-2 parameters
    cox2_params = coxian2_from_moments(target_mean, target_scv, p_continue)
    print(f"\n{cox2_params}")

    # Queue parameters
    lambda_rate = 0.8
    rho = lambda_rate * cox2_params.mean

    print(f"\nQueue Parameters:")
    print(f"  Arrival rate (λ):          {lambda_rate}")
    print(f"  Utilization (ρ = λ·E[S]):  {rho:.3f}")

    # Analytical results using Pollaczek-Khinchine
    analytical_l = mg1_mean_system_length(lambda_rate, cox2_params.mean, cox2_params.scv)
    analytical_w = mg1_mean_response_time(lambda_rate, cox2_params.mean, cox2_params.scv)

    print(f"\nAnalytical Results (Pollaczek-Khinchine for M/G/1):")
    print(f"  Mean in system (L):        {analytical_l:.6f}")
    print(f"  Mean response time (W):    {analytical_w:.6f}")

    # This simplified version will NOT match because it doesn't preserve SCV!
    print(f"\nNote: Simplified simulation will behave like M/M/1 (SCV=1)")
    print(f"      because we're using exponential service, not true Coxian-2.")

    mpp = create_mcox2_1_queue_simplified(
        lambda_rate=lambda_rate,
        mu1=cox2_params.mu1,
        mu2=cox2_params.mu2,
        p=cox2_params.p,
    )

    time_horizon = 10000.0
    n_runs = 100
    seed = 42

    print(f"\nRunning simulation...")
    simulator = GillespieSimulator()
    results = simulator.simulate(
        mpp=mpp,
        initial_state=np.array([0]),
        time_horizon=time_horizon,
        n_runs=n_runs,
        seed=seed,
    )

    simulated_l = results.mean_state[0]
    simulated_throughput = results.event_throughputs["DEPARTURE"]
    simulated_w = simulated_l / simulated_throughput if simulated_throughput > 0 else 0

    print(f"\nSimulation Results:")
    print(f"  Mean in system (L):        {simulated_l:.6f}")
    print(f"  Mean response time (W):    {simulated_w:.6f}")

    print(f"\nComparison:")
    print(f"  P-K (true Cox2, c²=4.0):   L={analytical_l:.2f}")
    print(f"  Simulated (approx, c²≈1):  L={simulated_l:.2f}")
    print(f"\n  As expected, simulation gives M/M/1-like results (L≈4)")
    print(f"  instead of true M/Cox2/1 results (L≈8.8).")

    print(f"\n" + "=" * 70)
    print("CONCLUSION: Need proper 2D state space to simulate Cox2 correctly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
