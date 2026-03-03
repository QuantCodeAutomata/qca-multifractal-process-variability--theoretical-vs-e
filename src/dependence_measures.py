"""
Linear and nonlinear dependence measures.

Implements:
- Autocorrelation function (ACF)
- Auto-mutual information using KSG estimator
"""

import numpy as np
from scipy.special import digamma
from scipy.spatial import KDTree
from typing import Tuple


def compute_acf(signal_data: np.ndarray, max_lag: int = 21) -> np.ndarray:
    """
    Compute normalized autocorrelation function.
    
    ACF(eta) = Cov(s(t), s(t+eta)) / Var(s(t))
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    max_lag : int, default=21
        Maximum lag to compute
        
    Returns:
    --------
    acf : np.ndarray
        ACF values for lags 0, 1, ..., max_lag
    """
    N = len(signal_data)
    s_centered = signal_data - np.mean(signal_data)
    
    # Compute autocorrelation via FFT (efficient)
    acf_full = np.correlate(s_centered, s_centered, mode='full')
    
    # Extract positive lags and normalize
    acf = acf_full[N-1:N+max_lag+1]
    acf = acf / acf[0]  # Normalize so ACF(0) = 1
    
    return acf


def ksg_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Estimate mutual information using Kraskov-Stoegbauer-Grassberger (KSG) estimator.
    
    I(X; Y) = psi(k) - <psi(nx+1) + psi(ny+1)> + psi(N)
    
    where:
    - psi is the digamma function
    - nx, ny are neighbor counts in marginal spaces
    - <> denotes average over all points
    
    Uses Chebyshev (max-norm) distance.
    
    Parameters:
    -----------
    x : np.ndarray
        First variable (1D)
    y : np.ndarray
        Second variable (1D)
    k : int, default=5
        Number of nearest neighbors
        
    Returns:
    --------
    mi : float
        Mutual information estimate in nats
    """
    N = len(x)
    
    if N != len(y):
        raise ValueError("x and y must have same length")
    
    if N < k + 1:
        return 0.0
    
    # Construct 2D joint space
    data_2d = np.column_stack([x, y])
    
    # Compute pairwise Chebyshev distances manually
    nx_values = np.zeros(N)
    ny_values = np.zeros(N)
    
    for i in range(N):
        # Compute Chebyshev distances in 2D joint space
        diff_2d = data_2d - data_2d[i]
        dist_2d = np.max(np.abs(diff_2d), axis=1)
        
        # Sort and get k-th nearest neighbor distance (excluding self at index 0)
        sorted_dists = np.sort(dist_2d)
        epsilon = sorted_dists[k]  # k-th nearest (0-indexed, so k gives k+1-th including self)
        
        # Add small tolerance
        epsilon_tol = epsilon + 1e-10
        
        # Count neighbors in marginal spaces
        dist_x = np.abs(x - x[i])
        dist_y = np.abs(y - y[i])
        
        # Count points within epsilon (excluding self)
        nx = np.sum(dist_x < epsilon_tol) - 1
        ny = np.sum(dist_y < epsilon_tol) - 1
        
        nx_values[i] = nx
        ny_values[i] = ny
    
    # KSG estimator
    mi = digamma(k) - np.mean(digamma(nx_values + 1) + digamma(ny_values + 1)) + digamma(N)
    
    return max(0.0, mi)  # MI should be non-negative


def compute_auto_mi(signal_data: np.ndarray, max_lag: int = 21, k: int = 5) -> np.ndarray:
    """
    Compute auto-mutual information I(s(t); s(t+eta)) for multiple lags.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    max_lag : int, default=21
        Maximum lag to compute
    k : int, default=5
        Number of nearest neighbors for KSG estimator
        
    Returns:
    --------
    mi : np.ndarray
        Auto-MI values for lags 1, 2, ..., max_lag
        (lag 0 is infinite, not computed)
    """
    N = len(signal_data)
    mi_values = np.zeros(max_lag)
    
    for lag in range(1, max_lag + 1):
        if N - lag < 10:
            mi_values[lag - 1] = 0.0
            continue
        
        # Construct lagged pairs
        x = signal_data[:N-lag]
        y = signal_data[lag:N]
        
        # Compute MI
        mi = ksg_mutual_information(x, y, k=k)
        mi_values[lag - 1] = mi
    
    return mi_values


def compute_acf_ensemble(signals: list, max_lag: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ACF statistics across ensemble of signals.
    
    Parameters:
    -----------
    signals : list
        List of time series
    max_lag : int, default=21
        Maximum lag
        
    Returns:
    --------
    mean_acf : np.ndarray
        Mean ACF across ensemble
    lo_acf : np.ndarray
        2.5th percentile
    hi_acf : np.ndarray
        97.5th percentile
    """
    n_signals = len(signals)
    acf_matrix = np.zeros((n_signals, max_lag + 1))
    
    for i, signal in enumerate(signals):
        acf_matrix[i, :] = compute_acf(signal, max_lag)
    
    mean_acf = np.mean(acf_matrix, axis=0)
    lo_acf = np.percentile(acf_matrix, 2.5, axis=0)
    hi_acf = np.percentile(acf_matrix, 97.5, axis=0)
    
    return mean_acf, lo_acf, hi_acf


def compute_mi_ensemble(signals: list, max_lag: int = 21, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute auto-MI statistics across ensemble of signals.
    
    Parameters:
    -----------
    signals : list
        List of time series
    max_lag : int, default=21
        Maximum lag
    k : int, default=5
        KSG parameter
        
    Returns:
    --------
    mean_mi : np.ndarray
        Mean MI across ensemble
    lo_mi : np.ndarray
        2.5th percentile
    hi_mi : np.ndarray
        97.5th percentile
    """
    n_signals = len(signals)
    mi_matrix = np.zeros((n_signals, max_lag))
    
    for i, signal in enumerate(signals):
        mi_matrix[i, :] = compute_auto_mi(signal, max_lag, k)
    
    mean_mi = np.mean(mi_matrix, axis=0)
    lo_mi = np.percentile(mi_matrix, 2.5, axis=0)
    hi_mi = np.percentile(mi_matrix, 97.5, axis=0)
    
    return mean_mi, lo_mi, hi_mi
