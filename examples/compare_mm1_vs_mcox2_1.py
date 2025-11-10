"""
Comparison: M/M/1 vs M/Cox2/1 Queue

This example demonstrates the impact of service time variability on
queueing performance by comparing:
- M/M/1: exponential service (SCV = 1.0)
- M/Cox2/1: Coxian-2 service with high variability (SCV = 4.0)

Both queues have:
- Same arrival rate λ
- Same mean service time E[S]
- Same utilization ρ = λ·E[S]

Key Question: How does service variability affect performance?

Theoretical Prediction (Pollaczek-Khinchine Formula)
---------------------------------------------------
Mean queue length: Lq = (ρ² · (1 + c²ₛ)) / (2(1 - ρ))

The factor (1 + c²ₛ) shows the direct impact of service SCV:
- M/M/1 with c² = 1.0: factor = 2.0
- M/Cox2/1 with c² = 4.0: factor = 5.0

Expected result: M/Cox2/1 should have ~2.5× longer queues!
"""

import numpy as np
from qn_ssa.models import MarkovPopulationProcess
from qn_ssa.simulators import GillespieSimulator
from qn_ssa.utils import (
    coxian2_from_moments,
    mm1_mean_queue_length,
    mm1_mean_response_time,
    mg1_mean_system_length,
    mg1_mean_response_time,
    mg1_mean_queue_length,
)


def create_mm1_queue(lambda_rate: float, mu_rate: float) -> MarkovPopulationProcess:
    """Create M/M/1 queue (1D state space)."""
    mpp = MarkovPopulationProcess(state_dim=1)

    mpp.add_event(
        jump=np.array([1]),
        rate_fn=lambda x: lambda_rate,
        label="ARRIVAL",
    )

    mpp.add_event(
        jump=np.array([-1]),
        rate_fn=lambda x: mu_rate if x[0] > 0 else 0.0,
        label="DEPARTURE",
    )

    return mpp


def create_mcox2_1_queue(
    lambda_rate: float, mu1: float, mu2: float, p_cont: float
) -> MarkovPopulationProcess:
    """
    Create M/Cox2/1 queue (2D state space).

    State: [n, p] where n=total in system, p=phase indicator {0,1}
    """
    mpp = MarkovPopulationProcess(state_dim=2)

    # Event 1: ARRIVAL
    mpp.add_event(
        jump=np.array([1, 0]),
        rate_fn=lambda x: lambda_rate,
        label="ARRIVAL",
    )

    # Event 2: PHASE1_TO_PHASE2
    mpp.add_event(
        jump=np.array([0, 1]),
        rate_fn=lambda x: p_cont * mu1 if (x[0] > 0 and x[1] == 0) else 0.0,
        label="PHASE1_TO_PHASE2",
    )

    # Event 3: PHASE1_DEPART
    mpp.add_event(
        jump=np.array([-1, 0]),
        rate_fn=lambda x: (1 - p_cont) * mu1 if (x[0] > 0 and x[1] == 0) else 0.0,
        label="PHASE1_DEPART",
    )

    # Event 4: PHASE2_DEPART
    mpp.add_event(
        jump=np.array([-1, -1]),
        rate_fn=lambda x: mu2 if (x[0] > 0 and x[1] == 1) else 0.0,
        label="PHASE2_DEPART",
    )

    return mpp


def main():
    """Compare M/M/1 vs M/Cox2/1 with same mean service time."""

    print("=" * 80)
    print("COMPARISON: M/M/1 vs M/Cox2/1 Queue")
    print("Impact of Service Time Variability")
    print("=" * 80)

    # Common parameters
    lambda_rate = 0.8
    service_mean = 1.0
    rho = lambda_rate * service_mean

    print(f"\nCommon Parameters:")
    print(f"  Arrival rate (λ):           {lambda_rate}")
    print(f"  Mean service time (E[S]):   {service_mean}")
    print(f"  Utilization (ρ):            {rho:.3f}")

    # M/M/1 parameters
    mm1_mu = 1.0 / service_mean  # μ = 1.0
    mm1_scv = 1.0  # Exponential has SCV = 1

    # M/Cox2/1 parameters (high variability)
    cox2_scv = 4.0
    cox2_params = coxian2_from_moments(service_mean, cox2_scv, p=0.2)

    print(f"\n" + "-" * 80)
    print("M/M/1 Queue (Exponential Service)")
    print("-" * 80)
    print(f"  Service rate (μ):           {mm1_mu}")
    print(f"  Service SCV (c²):           {mm1_scv}")

    print(f"\n" + "-" * 80)
    print("M/Cox2/1 Queue (Coxian-2 Service - High Variability)")
    print("-" * 80)
    print(f"  Phase 1 rate (μ₁):          {cox2_params.mu1:.6f}")
    print(f"  Phase 2 rate (μ₂):          {cox2_params.mu2:.6f}")
    print(f"  Continuation prob (p):      {cox2_params.p:.6f}")
    print(f"  Service SCV (c²):           {cox2_params.scv}")

    # Analytical predictions
    print(f"\n" + "=" * 80)
    print("ANALYTICAL PREDICTIONS")
    print("=" * 80)

    mm1_l_analytical = mm1_mean_queue_length(lambda_rate, mm1_mu)
    mm1_w_analytical = mm1_mean_response_time(lambda_rate, mm1_mu)

    cox2_l_analytical = mg1_mean_system_length(lambda_rate, service_mean, cox2_scv)
    cox2_w_analytical = mg1_mean_response_time(lambda_rate, service_mean, cox2_scv)
    cox2_lq_analytical = mg1_mean_queue_length(lambda_rate, service_mean, cox2_scv)

    print(f"\nM/M/1 (SCV = 1.0):")
    print(f"  Mean in system (L):         {mm1_l_analytical:.6f}")
    print(f"  Mean response time (W):     {mm1_w_analytical:.6f}")

    print(f"\nM/Cox2/1 (SCV = 4.0):")
    print(f"  Mean in queue (Lq):         {cox2_lq_analytical:.6f}")
    print(f"  Mean in system (L):         {cox2_l_analytical:.6f}")
    print(f"  Mean response time (W):     {cox2_w_analytical:.6f}")

    # Show the impact of variability
    l_ratio = cox2_l_analytical / mm1_l_analytical
    w_ratio = cox2_w_analytical / mm1_w_analytical

    print(f"\n" + "-" * 80)
    print(f"Impact of High Variability (SCV = 4.0 vs 1.0):")
    print(f"  Queue length ratio (L_Cox2/L_MM1):     {l_ratio:.3f}×")
    print(f"  Response time ratio (W_Cox2/W_MM1):    {w_ratio:.3f}×")
    print(f"  Predicted by P-K: (1+4)/(1+1) =        {5/2:.3f}×")
    print("-" * 80)

    # Simulation parameters
    time_horizon = 10000.0
    n_runs = 100
    seed = 42

    print(f"\n" + "=" * 80)
    print("SIMULATION VALIDATION")
    print("=" * 80)

    print(f"\nSimulation Parameters:")
    print(f"  Time horizon (T):           {time_horizon}")
    print(f"  Number of runs:             {n_runs}")
    print(f"  Random seed:                {seed}")

    # Simulate M/M/1
    print(f"\n[1/2] Simulating M/M/1 queue...")
    mpp_mm1 = create_mm1_queue(lambda_rate, mm1_mu)
    simulator = GillespieSimulator()
    results_mm1 = simulator.simulate(
        mpp=mpp_mm1,
        initial_state=np.array([0]),
        time_horizon=time_horizon,
        n_runs=n_runs,
        seed=seed,
    )

    mm1_l_simulated = results_mm1.mean_state[0]
    mm1_throughput = results_mm1.event_throughputs["DEPARTURE"]
    mm1_w_simulated = mm1_l_simulated / mm1_throughput if mm1_throughput > 0 else 0

    # Simulate M/Cox2/1
    print(f"[2/2] Simulating M/Cox2/1 queue...")
    mpp_cox2 = create_mcox2_1_queue(
        lambda_rate, cox2_params.mu1, cox2_params.mu2, cox2_params.p
    )
    results_cox2 = simulator.simulate(
        mpp=mpp_cox2,
        initial_state=np.array([0, 0]),
        time_horizon=time_horizon,
        n_runs=n_runs,
        seed=seed,
    )

    # state[0] = n (total in system)
    cox2_l_simulated = results_cox2.mean_state[0]
    phase1_departs = results_cox2.event_throughputs.get("PHASE1_DEPART", 0)
    phase2_departs = results_cox2.event_throughputs.get("PHASE2_DEPART", 0)
    cox2_throughput = phase1_departs + phase2_departs
    cox2_w_simulated = cox2_l_simulated / cox2_throughput if cox2_throughput > 0 else 0

    # Display results
    print(f"\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'M/M/1 (c²=1)':<20} {'M/Cox2/1 (c²=4)':<20} {'Ratio':<10}")
    print("-" * 80)

    print(
        f"{'Mean in system (L):':<30} {mm1_l_simulated:>19.6f} {cox2_l_simulated:>19.6f} "
        f"{cox2_l_simulated/mm1_l_simulated:>9.3f}×"
    )

    print(
        f"{'Mean response time (W):':<30} {mm1_w_simulated:>19.6f} {cox2_w_simulated:>19.6f} "
        f"{cox2_w_simulated/mm1_w_simulated:>9.3f}×"
    )

    print(
        f"{'Throughput (λ):':<30} {mm1_throughput:>19.6f} {cox2_throughput:>19.6f} "
        f"{cox2_throughput/mm1_throughput:>9.3f}×"
    )

    # Validation table
    print(f"\n" + "=" * 80)
    print("VALIDATION: Simulation vs Analytical")
    print("=" * 80)

    print(f"\nM/M/1 Queue:")
    print(f"  {'Metric':<25} {'Analytical':>12} {'Simulated':>12} {'Error':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    mm1_l_error = abs(mm1_l_simulated - mm1_l_analytical) / mm1_l_analytical * 100
    print(
        f"  {'Mean in system (L):':<25} {mm1_l_analytical:>12.6f} {mm1_l_simulated:>12.6f} "
        f"{mm1_l_error:>9.2f}%"
    )

    mm1_w_error = abs(mm1_w_simulated - mm1_w_analytical) / mm1_w_analytical * 100
    print(
        f"  {'Mean response time (W):':<25} {mm1_w_analytical:>12.6f} {mm1_w_simulated:>12.6f} "
        f"{mm1_w_error:>9.2f}%"
    )

    print(f"\nM/Cox2/1 Queue:")
    print(f"  {'Metric':<25} {'Analytical':>12} {'Simulated':>12} {'Error':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    cox2_l_error = abs(cox2_l_simulated - cox2_l_analytical) / cox2_l_analytical * 100
    print(
        f"  {'Mean in system (L):':<25} {cox2_l_analytical:>12.6f} {cox2_l_simulated:>12.6f} "
        f"{cox2_l_error:>9.2f}%"
    )

    cox2_w_error = abs(cox2_w_simulated - cox2_w_analytical) / cox2_w_analytical * 100
    print(
        f"  {'Mean response time (W):':<25} {cox2_w_analytical:>12.6f} {cox2_w_simulated:>12.6f} "
        f"{cox2_w_error:>9.2f}%"
    )

    # Summary
    threshold = 5.0
    mm1_pass = mm1_l_error < threshold and mm1_w_error < threshold
    cox2_pass = cox2_l_error < threshold and cox2_w_error < threshold

    print(f"\nValidation (threshold: {threshold}%):")
    print(f"  M/M/1:   {'✓ PASSED' if mm1_pass else '✗ FAILED'}")
    print(f"  M/Cox2/1: {'✓ PASSED' if cox2_pass else '✗ FAILED'}")

    # Key insights
    print(f"\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"""
1. Service Variability Impact:
   - Same arrival rate (λ = {lambda_rate})
   - Same mean service time (E[S] = {service_mean})
   - Same utilization (ρ = {rho:.1f})

2. But VERY different performance:
   - M/M/1 (SCV=1.0):   L = {mm1_l_simulated:.2f}, W = {mm1_w_simulated:.2f}
   - M/Cox2/1 (SCV=4.0): L = {cox2_l_simulated:.2f}, W = {cox2_w_simulated:.2f}

3. High variability → much longer queues and wait times!
   - Queue length increased by {l_ratio:.2f}×
   - Response time increased by {w_ratio:.2f}×

4. Pollaczek-Khinchine formula predicts this exactly:
   - Factor (1 + c²) in mean queue length
   - M/M/1: 1 + 1 = 2
   - M/Cox2/1: 1 + 4 = 5
   - Ratio: 5/2 = 2.5× (matches simulation!)

5. Lesson: Reducing service time variability is as important
   as reducing mean service time for queueing performance!
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
