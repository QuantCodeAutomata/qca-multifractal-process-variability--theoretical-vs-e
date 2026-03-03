"""
Synthetic lognormal random cascade generation on DWT dyadic trees.

This module implements the generation of multifractal time series using
a random multiplicative cascade on a wavelet (DAUB12) dyadic tree structure.
"""

import numpy as np
import pywt
from typing import Tuple


def compute_lognormal_parameters(sigma_w: float) -> Tuple[float, float]:
    """
    Compute lognormal multiplicator parameters.
    
    Uses the constraint mu_W = -0.5 * sigma_W^2 to ensure proper cascade normalization.
    
    Parameters:
    -----------
    sigma_w : float
        Standard deviation of log|W| distribution
        
    Returns:
    --------
    mu_w : float
        Mean of log|W| distribution
    sigma_w : float
        Standard deviation (echoed for clarity)
    """
    mu_w = -0.5 * sigma_w**2
    return mu_w, sigma_w


def theoretical_tau_q(q_values: np.ndarray, sigma_w: float) -> np.ndarray:
    """
    Compute theoretical scaling function tau(q) for lognormal cascade.
    
    For lognormal multiplicators with parameters (mu_W, sigma_W):
    tau(q) = -1 - (mu_W * q + 0.5 * sigma_W^2 * q^2) / ln(2)
    
    With mu_W = -0.5 * sigma_W^2, this simplifies to:
    tau(q) = -1 + (0.5 * sigma_W^2 * q - 0.5 * sigma_W^2 * q^2) / ln(2)
    
    Parameters:
    -----------
    q_values : np.ndarray
        Array of moment orders
    sigma_w : float
        Lognormal parameter
        
    Returns:
    --------
    tau : np.ndarray
        Theoretical tau(q) values
    """
    mu_w, _ = compute_lognormal_parameters(sigma_w)
    tau = -1.0 - (mu_w * q_values + 0.5 * sigma_w**2 * q_values**2) / np.log(2)
    return tau


def generate_cascade_series_direct(N: int, sigma_w: float, seed: int = None) -> np.ndarray:
    """
    Generate multifractal series via direct multiplicative cascade.
    
    Implements binary tree cascade:
    - Start with root value 1
    - At each level, split each interval in two and multiply by lognormal weights
    - Continue for log2(N) levels
    
    Parameters:
    -----------
    N : int
        Series length (must be power of 2)
    sigma_w : float
        Lognormal parameter
    seed : int, optional
        Random seed
        
    Returns:
    --------
    series : np.ndarray
        Multifractal time series
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    mu_w, _ = compute_lognormal_parameters(sigma_w)
    n = int(np.log2(N))
    
    # Initialize series with all ones
    series = np.ones(N)
    
    # Cascade down levels
    for level in range(n):
        block_size = N // (2**(level + 1))
        num_blocks = 2**(level + 1)
        
        for block in range(num_blocks):
            # Draw lognormal multiplicator
            log_w = rng.normal(mu_w, sigma_w)
            w = np.exp(log_w)
            
            # Apply to this block
            start_idx = block * block_size
            end_idx = (block + 1) * block_size
            series[start_idx:end_idx] *= w
    
    return series


def generate_multifractal_series(N: int, sigma_w: float = 0.3, seed: int = None) -> np.ndarray:
    """
    Generate a synthetic multifractal time series via lognormal cascade.
    
    Uses direct multiplicative cascade approach.
    
    Parameters:
    -----------
    N : int
        Length of time series (must be a power of 2)
    sigma_w : float, default=0.3
        Lognormal cascade parameter
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    series : np.ndarray
        Synthetic multifractal time series of length N
        
    Raises:
    -------
    ValueError
        If N is not a power of 2
    """
    # Check N is power of 2
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"N must be a power of 2, got N={N}")
    
    # Generate series directly via cascade
    series = generate_cascade_series_direct(N, sigma_w, seed)
    
    return series


def verify_cascade_parameters(sigma_w: float) -> dict:
    """
    Verify and report cascade parameters.
    
    Parameters:
    -----------
    sigma_w : float
        Lognormal parameter
        
    Returns:
    --------
    params : dict
        Dictionary with mu_w, sigma_w, and E[W^2]
    """
    mu_w, _ = compute_lognormal_parameters(sigma_w)
    
    # For lognormal: E[W^2] = exp(2*mu_w + 2*sigma_w^2)
    expected_w2 = np.exp(2*mu_w + 2*sigma_w**2)
    
    params = {
        'mu_w': mu_w,
        'sigma_w': sigma_w,
        'E[W^2]': expected_w2,
        'ln(2)': np.log(2)
    }
    
    return params
