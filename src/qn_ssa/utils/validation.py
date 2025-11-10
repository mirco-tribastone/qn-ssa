"""
Analytical formulas for queueing systems validation.

This module provides closed-form solutions for classical queueing models
to validate simulation results. These formulas assume steady-state conditions
and Poisson arrivals.

References
----------
- Kleinrock, L. (1975). Queueing Systems, Volume 1: Theory.
- Allen, A. O. (1990). Probability, Statistics, and Queueing Theory.
- Gross, D., & Harris, C. M. (1998). Fundamentals of Queueing Theory.
"""

import numpy as np
from scipy.special import factorial


def mm1_utilization(lambda_rate: float, mu_rate: float) -> float:
    """
    Compute utilization ρ for M/M/1 queue.

    The utilization (traffic intensity) is the fraction of time the server
    is busy in steady state.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate μ > 0

    Returns
    -------
    float
        Utilization ρ = λ/μ

    Notes
    -----
    For stability (steady state to exist), we require ρ < 1.
    """
    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("Rates must be positive")

    return lambda_rate / mu_rate


def mm1_mean_queue_length(lambda_rate: float, mu_rate: float) -> float:
    """
    Compute mean number of customers in M/M/1 queue system.

    This includes customers in queue and in service: L = E[N].

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate μ > λ (stability condition)

    Returns
    -------
    float
        Mean number in system L = λ/(μ - λ) = ρ/(1 - ρ)

    Raises
    ------
    ValueError
        If ρ = λ/μ ≥ 1 (unstable system)

    Notes
    -----
    The formula is derived from the geometric steady-state distribution:
        P(N = n) = (1 - ρ)ρⁿ for n = 0, 1, 2, ...
        L = Σ n·P(N = n) = ρ/(1 - ρ)
    """
    rho = mm1_utilization(lambda_rate, mu_rate)

    if rho >= 1:
        raise ValueError(
            f"System is unstable: ρ = {rho:.3f} ≥ 1. "
            f"Require λ < μ for steady state."
        )

    return rho / (1 - rho)


def mm1_mean_response_time(lambda_rate: float, mu_rate: float) -> float:
    """
    Compute mean response time (time in system) for M/M/1 queue.

    Response time W is the total time a customer spends in the system
    (waiting + service).

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate μ > λ (stability condition)

    Returns
    -------
    float
        Mean response time W = 1/(μ - λ)

    Notes
    -----
    This follows from Little's Law: L = λW
        W = L/λ = [ρ/(1-ρ)] / λ = 1/(μ - λ)
    """
    rho = mm1_utilization(lambda_rate, mu_rate)

    if rho >= 1:
        raise ValueError(
            f"System is unstable: ρ = {rho:.3f} ≥ 1. "
            f"Require λ < μ for steady state."
        )

    return 1.0 / (mu_rate - lambda_rate)


def mm1_mean_waiting_time(lambda_rate: float, mu_rate: float) -> float:
    """
    Compute mean waiting time (time in queue) for M/M/1 queue.

    Waiting time Wq is the time a customer spends waiting before service begins.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate μ > λ (stability condition)

    Returns
    -------
    float
        Mean waiting time Wq = λ/(μ(μ - λ)) = ρ/(μ - λ)

    Notes
    -----
    Response time = Waiting time + Service time
        W = Wq + 1/μ
        Wq = W - 1/μ = 1/(μ - λ) - 1/μ = λ/(μ(μ - λ))
    """
    rho = mm1_utilization(lambda_rate, mu_rate)

    if rho >= 1:
        raise ValueError(
            f"System is unstable: ρ = {rho:.3f} ≥ 1. "
            f"Require λ < μ for steady state."
        )

    return lambda_rate / (mu_rate * (mu_rate - lambda_rate))


def erlang_c(c: int, rho: float) -> float:
    """
    Compute Erlang C formula (probability of waiting in M/M/c queue).

    The Erlang C formula gives the probability that an arriving customer
    must wait (all c servers are busy).

    Parameters
    ----------
    c : int
        Number of servers, c ≥ 1
    rho : float
        Total offered load ρ = λ/(c·μ), must satisfy ρ < 1 for stability

    Returns
    -------
    float
        Probability of waiting C(c, a) where a = c·ρ is the traffic intensity

    Notes
    -----
    The formula is:
        C(c, a) = [aᶜ/(c!(1-ρ))] / [Σₖ₌₀ᶜ⁻¹ aᵏ/k! + aᶜ/(c!(1-ρ))]

    where a = λ/μ is the offered load (average number of busy servers).
    """
    if c < 1:
        raise ValueError(f"Number of servers must be at least 1, got {c}")

    if rho >= 1:
        raise ValueError(
            f"System is unstable: ρ = {rho:.3f} ≥ 1. "
            f"Require ρ = λ/(c·μ) < 1 for steady state."
        )

    if rho <= 0:
        raise ValueError(f"Utilization must be positive, got {rho}")

    # Offered load a = c·ρ (average number of busy servers)
    a = c * rho

    # Compute denominator: sum of P(0) normalization
    # Sum from k=0 to c-1: aᵏ/k!
    sum_terms = sum(a**k / factorial(k) for k in range(c))

    # Last term: aᶜ/(c!(1-ρ))
    last_term = (a**c) / (factorial(c) * (1 - rho))

    denominator = sum_terms + last_term

    # Erlang C formula
    erlang_c_value = last_term / denominator

    return erlang_c_value


def mmc_utilization(lambda_rate: float, mu_rate: float, c: int) -> float:
    """
    Compute utilization ρ for M/M/c queue.

    The utilization is the average fraction of servers that are busy.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate per server μ > 0
    c : int
        Number of servers c ≥ 1

    Returns
    -------
    float
        Utilization ρ = λ/(c·μ)

    Notes
    -----
    For stability, require ρ < 1, i.e., λ < c·μ.
    The total service capacity is c·μ.
    """
    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("Rates must be positive")

    if c < 1:
        raise ValueError(f"Number of servers must be at least 1, got {c}")

    return lambda_rate / (c * mu_rate)


def mmc_mean_queue_length(lambda_rate: float, mu_rate: float, c: int) -> float:
    """
    Compute mean number of customers in M/M/c queue system.

    This includes customers in queue and in service: L = E[N].

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate per server μ > 0
    c : int
        Number of servers c ≥ 1, with λ < c·μ (stability)

    Returns
    -------
    float
        Mean number in system L

    Notes
    -----
    The formula uses the Erlang C formula:
        Lq = C(c, a) · ρ/(1 - ρ)  (mean in queue)
        L = Lq + a                 (mean in system)

    where a = λ/μ is offered load and ρ = λ/(c·μ) is utilization.
    """
    rho = mmc_utilization(lambda_rate, mu_rate, c)

    if rho >= 1:
        raise ValueError(
            f"System is unstable: ρ = {rho:.3f} ≥ 1. "
            f"Require λ < c·μ for steady state."
        )

    # Offered load (average number of busy servers if no queueing)
    a = lambda_rate / mu_rate

    # Mean number waiting in queue (not in service)
    erlang_c_val = erlang_c(c, rho)
    lq = erlang_c_val * rho / (1 - rho)

    # Mean number in system = waiting + in service
    l = lq + a

    return l


def mmc_mean_response_time(lambda_rate: float, mu_rate: float, c: int) -> float:
    """
    Compute mean response time (time in system) for M/M/c queue.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate per server μ > 0
    c : int
        Number of servers c ≥ 1, with λ < c·μ (stability)

    Returns
    -------
    float
        Mean response time W

    Notes
    -----
    By Little's Law: W = L/λ
    """
    l = mmc_mean_queue_length(lambda_rate, mu_rate, c)
    return l / lambda_rate


def mmc_mean_waiting_time(lambda_rate: float, mu_rate: float, c: int) -> float:
    """
    Compute mean waiting time (time in queue) for M/M/c queue.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    mu_rate : float
        Service rate per server μ > 0
    c : int
        Number of servers c ≥ 1, with λ < c·μ (stability)

    Returns
    -------
    float
        Mean waiting time Wq

    Notes
    -----
    Wq = W - 1/μ (response time - service time)
    """
    w = mmc_mean_response_time(lambda_rate, mu_rate, c)
    return w - 1.0 / mu_rate


# ============================================================================
# M/G/1 Queue with General Service Distribution (Pollaczek-Khinchine)
# ============================================================================


def mg1_mean_waiting_time(
    lambda_rate: float, service_mean: float, service_scv: float
) -> float:
    """
    Compute mean waiting time in queue for M/G/1 queue (Pollaczek-Khinchine).

    The M/G/1 queue has Poisson arrivals and general service time distribution.
    The Pollaczek-Khinchine (P-K) formula gives exact steady-state performance
    metrics based on the first two moments of service time.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    service_mean : float
        Mean service time E[S] > 0
    service_scv : float
        Squared coefficient of variation of service time c²ₛ = Var(S)/E[S]² ≥ 0

    Returns
    -------
    float
        Mean waiting time in queue Wq

    Raises
    ------
    ValueError
        If ρ = λ·E[S] ≥ 1 (unstable system)

    Notes
    -----
    The Pollaczek-Khinchine formula for mean waiting time:
        Wq = (λ · E[S²]) / (2(1 - ρ))
           = (λ · E[S]² · (1 + c²ₛ)) / (2(1 - ρ))

    where:
    - ρ = λ · E[S] is the utilization
    - E[S²] = E[S]² · (1 + c²ₛ) is the second moment
    - c²ₛ = Var(S)/E[S]² is the SCV of service time

    Key insight: Higher service variability (larger c²ₛ) increases waiting time.
    - M/M/1: c²ₛ = 1 (exponential service)
    - M/D/1: c²ₛ = 0 (deterministic service, minimal waiting)
    - High variability: c²ₛ > 1 (e.g., Coxian, hyperexponential)

    Reference: Kleinrock (1975), Queueing Systems Vol. 1, Section 5.3
    """
    if lambda_rate <= 0:
        raise ValueError(f"Arrival rate must be positive, got {lambda_rate}")

    if service_mean <= 0:
        raise ValueError(f"Service mean must be positive, got {service_mean}")

    if service_scv < 0:
        raise ValueError(f"Service SCV must be non-negative, got {service_scv}")

    # Utilization ρ = λ · E[S]
    rho = lambda_rate * service_mean

    if rho >= 1:
        raise ValueError(
            f"System is unstable: ρ = {rho:.3f} ≥ 1. "
            f"Require λ · E[S] < 1 for steady state."
        )

    # Pollaczek-Khinchine formula for mean waiting time
    # Wq = (λ · E[S²]) / (2(1 - ρ))
    # where E[S²] = E[S]² · (1 + c²ₛ)
    numerator = lambda_rate * (service_mean**2) * (1 + service_scv)
    denominator = 2 * (1 - rho)

    wq = numerator / denominator

    return wq


def mg1_mean_queue_length(
    lambda_rate: float, service_mean: float, service_scv: float
) -> float:
    """
    Compute mean number in queue for M/G/1 queue (Pollaczek-Khinchine).

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    service_mean : float
        Mean service time E[S] > 0
    service_scv : float
        Squared coefficient of variation of service time c²ₛ ≥ 0

    Returns
    -------
    float
        Mean number in queue Lq (not including those in service)

    Notes
    -----
    By Little's Law applied to the queue:
        Lq = λ · Wq

    The P-K formula gives:
        Lq = (ρ² · (1 + c²ₛ)) / (2(1 - ρ))

    where ρ = λ · E[S] is the utilization.
    """
    wq = mg1_mean_waiting_time(lambda_rate, service_mean, service_scv)
    return lambda_rate * wq


def mg1_mean_system_length(
    lambda_rate: float, service_mean: float, service_scv: float
) -> float:
    """
    Compute mean number in system for M/G/1 queue.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    service_mean : float
        Mean service time E[S] > 0
    service_scv : float
        Squared coefficient of variation of service time c²ₛ ≥ 0

    Returns
    -------
    float
        Mean number in system L (queue + service)

    Notes
    -----
    L = Lq + ρ

    where:
    - Lq is mean in queue (from P-K formula)
    - ρ = λ · E[S] is mean in service
    """
    lq = mg1_mean_queue_length(lambda_rate, service_mean, service_scv)
    rho = lambda_rate * service_mean
    return lq + rho


def mg1_mean_response_time(
    lambda_rate: float, service_mean: float, service_scv: float
) -> float:
    """
    Compute mean response time (time in system) for M/G/1 queue.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ > 0
    service_mean : float
        Mean service time E[S] > 0
    service_scv : float
        Squared coefficient of variation of service time c²ₛ ≥ 0

    Returns
    -------
    float
        Mean response time W (waiting + service)

    Notes
    -----
    By Little's Law:
        W = L/λ = Wq + E[S]

    Response time = waiting time + service time.
    """
    wq = mg1_mean_waiting_time(lambda_rate, service_mean, service_scv)
    return wq + service_mean
