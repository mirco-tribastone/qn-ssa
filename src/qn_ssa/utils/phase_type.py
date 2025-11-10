"""
Phase-Type Distribution Utilities.

This module provides utilities for constructing phase-type distributions
(Erlang, Hypoexponential, Hyperexponential, Coxian) from target moments.

Based on: "Moment Fitting with Phase-Type Distributions:
Closed-Form Constructions and SCV Validity Ranges"

References
----------
Phase-type distributions are continuous distributions representing the
time to absorption in a finite-state continuous-time Markov chain.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class CoxianPhaseParameters:
    """
    Parameters for a Coxian-2 phase-type distribution.

    The Coxian-2 distribution has two sequential phases:
    - Phase 1: exponential with rate μ₁
    - Phase 2: exponential with rate μ₂ (entered with probability p)

    Service time X = T₁ + J·T₂ where:
    - T₁ ~ Exp(μ₁)
    - T₂ ~ Exp(μ₂)
    - J ~ Bernoulli(p)

    Attributes
    ----------
    mu1 : float
        Rate parameter for phase 1 (μ₁ > 0)
    mu2 : float
        Rate parameter for phase 2 (μ₂ > 0)
    p : float
        Probability of continuing to phase 2 (0 < p ≤ 1)
    mean : float
        Mean of the distribution E[X] = 1/μ₁ + p/μ₂
    scv : float
        Squared coefficient of variation c² = Var(X)/E[X]²
    """

    mu1: float
    mu2: float
    p: float
    mean: float
    scv: float

    def __str__(self) -> str:
        """Human-readable description."""
        return (
            f"Coxian-2 Distribution:\n"
            f"  Phase 1 rate (μ₁): {self.mu1:.6f}\n"
            f"  Phase 2 rate (μ₂): {self.mu2:.6f}\n"
            f"  Continuation prob (p): {self.p:.6f}\n"
            f"  Mean: {self.mean:.6f}\n"
            f"  SCV (c²): {self.scv:.6f}"
        )


def coxian2_from_moments(
    mean: float, scv: float, p: float = 0.5
) -> CoxianPhaseParameters:
    """
    Construct Coxian-2 distribution parameters from target mean and SCV.

    The Coxian-2 distribution can represent any SCV ≥ 1/2. It consists of
    two sequential exponential phases with optional completion after phase 1.

    Parameters
    ----------
    mean : float
        Target mean E[X] > 0
    scv : float
        Target squared coefficient of variation c² ≥ 1/2
    p : float, optional
        Continuation probability (0 < p ≤ 1), default 0.5
        Smaller p allows larger SCV values

    Returns
    -------
    CoxianPhaseParameters
        Object containing μ₁, μ₂, p and verified mean and SCV

    Raises
    ------
    ValueError
        If parameters are invalid or SCV is out of range

    Notes
    -----
    For SCV < 1 (underdispersion):
        Set p = 1 and use hypoexponential fit:
        s = sqrt(2c² - 1)
        a = m/2 * (1 - s), b = m/2 * (1 + s)
        μ₁ = 1/a, μ₂ = 1/b

    For SCV ≥ 1 (overdispersion):
        Choose p ∈ (0, 1] and compute:
        b = (m/2) * (1 + sqrt(1 - 2(1-c²)/p))
        a = m - p*b
        μ₁ = 1/a, μ₂ = 1/b

    The minimum achievable SCV is 1/2; for c² < 1/2 use more phases.

    Examples
    --------
    >>> # High variability service (SCV = 4.0, mean = 1.0)
    >>> params = coxian2_from_moments(mean=1.0, scv=4.0, p=0.2)
    >>> print(f"μ₁={params.mu1:.3f}, μ₂={params.mu2:.3f}")
    """
    if mean <= 0:
        raise ValueError(f"Mean must be positive, got {mean}")

    if scv < 0.5:
        raise ValueError(
            f"Coxian-2 cannot achieve SCV < 0.5, got {scv}. "
            "Use more phases or different distribution."
        )

    if not (0 < p <= 1):
        raise ValueError(f"Continuation probability p must be in (0,1], got {p}")

    # Underdispersion case: 0.5 ≤ c² < 1
    if scv < 1.0:
        # Set p = 1 (always continue to phase 2) and use hypoexponential fit
        p = 1.0
        s = np.sqrt(2 * scv - 1)
        a = (mean / 2) * (1 - s)
        b = (mean / 2) * (1 + s)
        mu1 = 1.0 / a
        mu2 = 1.0 / b

    # Overdispersion case: c² ≥ 1
    else:
        # Check feasibility for c² < 1
        if scv < 1.0 and p < 2 * (1 - scv):
            raise ValueError(
                f"For SCV={scv}, need p ≥ {2*(1-scv):.3f}, got p={p}"
            )

        # Compute phase mean times using closed-form formula
        discriminant = 1 - 2 * (1 - scv) / p

        if discriminant < 0:
            raise ValueError(
                f"Invalid parameter combination: discriminant {discriminant:.3f} < 0. "
                f"For SCV={scv}, try smaller p"
            )

        b = (mean / 2) * (1 + np.sqrt(discriminant))
        a = mean - p * b

        if a <= 0:
            raise ValueError(
                f"Invalid parameter combination: phase 1 mean a={a:.3f} ≤ 0. "
                f"For SCV={scv}, try smaller p"
            )

        mu1 = 1.0 / a
        mu2 = 1.0 / b

    # Verify moments match target
    actual_mean = (1 / mu1) + p * (1 / mu2)
    actual_var = (1 / mu1**2) + p * (2 - p) / (mu2**2)
    actual_scv = actual_var / (actual_mean**2)

    # Check for numerical errors
    mean_error = abs(actual_mean - mean) / mean
    scv_error = abs(actual_scv - scv) / scv if scv > 0 else abs(actual_scv - scv)

    if mean_error > 1e-6 or scv_error > 1e-6:
        raise ValueError(
            f"Moment matching failed: "
            f"target mean={mean:.6f}, actual={actual_mean:.6f} (error={mean_error:.2e}); "
            f"target SCV={scv:.6f}, actual={actual_scv:.6f} (error={scv_error:.2e})"
        )

    return CoxianPhaseParameters(
        mu1=mu1, mu2=mu2, p=p, mean=actual_mean, scv=actual_scv
    )


def hyperexponential2_from_moments(mean: float, scv: float) -> Tuple[float, float, float]:
    """
    Construct Hyperexponential-2 (H2) distribution using Balanced Rates method.

    H2 is a mixture of two exponentials: rate μ₁ with probability p,
    rate μ₂ with probability (1-p). Can represent any SCV > 1.

    Parameters
    ----------
    mean : float
        Target mean E[X] > 0
    scv : float
        Target squared coefficient of variation c² > 1

    Returns
    -------
    p : float
        Mixing probability for first exponential
    mu1 : float
        Rate of first exponential
    mu2 : float
        Rate of second exponential

    Raises
    ------
    ValueError
        If SCV ≤ 1 (hyperexponential requires high variability)

    Notes
    -----
    Balanced Rates formula (valid for all c² > 1):
        γ = sqrt((c² - 1)/(c² + 1))
        p = (1 + γ)/2
        μ₁ = (1 + γ)/m
        μ₂ = (1 - γ)/m

    Then E[X] = p/μ₁ + (1-p)/μ₂ = m and c² matches target.
    """
    if mean <= 0:
        raise ValueError(f"Mean must be positive, got {mean}")

    if scv <= 1:
        raise ValueError(
            f"Hyperexponential requires SCV > 1, got {scv}. "
            "Use hypoexponential or Coxian for SCV ≤ 1."
        )

    # Balanced Rates construction
    gamma = np.sqrt((scv - 1) / (scv + 1))
    p = (1 + gamma) / 2
    mu1 = (1 + gamma) / mean
    mu2 = (1 - gamma) / mean

    return p, mu1, mu2
