"""
Surrogate generation methods for multifractal time series.

Implements four methods:
1. Multifractal cascade surrogates (proposed method)
2. IAFT (Iteratively Amplitude-Adjusted Fourier Transform)
3. Wavelet coefficient permutation
4. Wavelet coefficient cyclic rotation
"""

import numpy as np
import pywt
from typing import List


def multifractal_cascade_surrogate(signal_data: np.ndarray, wavelet: str = 'db6',
                                   mode: str = 'periodization', seed: int = None) -> np.ndarray:
    """
    Generate multifractal cascade surrogate via wavelet coefficient manipulation.
    
    Algorithm:
    1. Compute DWT of input signal
    2. Keep levels j=0,1 frozen (coarsest scales)
    3. For j>=2: compute multiplicators M[j] = C[j] / C[j-1] (parent)
    4. Randomly permute multiplicators at each level
    5. Reconstruct coefficients via cascade
    6. Apply amplitude adjustment per scale (rank-based remapping)
    7. Inverse DWT to get surrogate
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    wavelet : str, default='db6'
        Wavelet type (DAUB12 = db6)
    mode : str, default='periodization'
        Boundary mode
    seed : int, optional
        Random seed
        
    Returns:
    --------
    surrogate : np.ndarray
        Surrogate time series
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    N = len(signal_data)
    n = int(np.log2(N))
    
    # Compute DWT
    # pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    coeffs_orig = pywt.wavedec(signal_data, wavelet, mode=mode, level=n-1)
    
    # Convert to list indexed by paper's convention
    # coeffs_list[0] = approximation (coarsest)
    # coeffs_list[j] = detail level j (j=1 is next coarsest)
    coeffs_list = [coeffs_orig[0]]  # Approximation
    for j in range(1, len(coeffs_orig)):
        coeffs_list.append(coeffs_orig[j])
    
    n_levels = len(coeffs_list)
    
    # Freeze levels 0 and 1
    coeffs_surr = [coeffs_list[0].copy(), coeffs_list[1].copy()]
    
    # Process levels j >= 2
    for j in range(2, n_levels):
        parent_coeffs = coeffs_list[j-1]
        current_coeffs = coeffs_list[j]
        n_coeffs = len(current_coeffs)
        n_parents = len(parent_coeffs)
        
        # Compute multiplicators
        # Each coefficient has parent at index k//2
        multiplicators = np.zeros(n_coeffs)
        for k in range(n_coeffs):
            parent_idx = k // 2
            if parent_idx < n_parents:
                parent_val = parent_coeffs[parent_idx]
                if np.abs(parent_val) > 1e-12:
                    multiplicators[k] = current_coeffs[k] / parent_val
                else:
                    multiplicators[k] = 0.0
        
        # Random permutation of multiplicators
        perm_multiplicators = rng.permutation(multiplicators)
        
        # Cascade reconstruction
        provisional_coeffs = np.zeros(n_coeffs)
        surr_parent = coeffs_surr[j-1]
        for k in range(n_coeffs):
            parent_idx = k // 2
            if parent_idx < len(surr_parent):
                provisional_coeffs[k] = perm_multiplicators[k] * surr_parent[parent_idx]
        
        # Amplitude adjustment (rank-based remapping)
        # Sort indices by modulus
        orig_order = np.argsort(np.abs(current_coeffs), kind='stable')
        prov_order = np.argsort(np.abs(provisional_coeffs), kind='stable')
        
        # Remap: position prov_order[rank] gets value current_coeffs[orig_order[rank]]
        final_coeffs = np.zeros(n_coeffs)
        for rank in range(n_coeffs):
            final_coeffs[prov_order[rank]] = current_coeffs[orig_order[rank]]
        
        coeffs_surr.append(final_coeffs)
    
    # Convert back to pywt format
    coeffs_pywt = [coeffs_surr[0]]  # Approximation
    for j in range(1, n_levels):
        coeffs_pywt.append(coeffs_surr[j])
    
    # Inverse DWT
    surrogate = pywt.waverec(coeffs_pywt, wavelet, mode=mode)
    
    # Handle length mismatch
    if len(surrogate) > N:
        surrogate = surrogate[:N]
    
    return surrogate


def iaft_surrogate(signal_data: np.ndarray, n_iterations: int = 100, 
                   seed: int = None) -> np.ndarray:
    """
    Generate IAFT (Iteratively Amplitude-Adjusted Fourier Transform) surrogate.
    
    Algorithm:
    1. Start with random permutation of signal
    2. Iterate:
       a. FFT, replace amplitudes with target spectrum, keep phases
       b. IFFT
       c. Rank-remap to match target amplitude distribution
    3. Repeat for n_iterations
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    n_iterations : int, default=100
        Number of iterations
    seed : int, optional
        Random seed
        
    Returns:
    --------
    surrogate : np.ndarray
        IAFT surrogate
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    N = len(signal_data)
    
    # Target amplitude distribution (sorted values)
    target_amplitudes = np.sort(signal_data)
    
    # Target power spectrum
    signal_fft = np.fft.fft(signal_data)
    target_power = np.abs(signal_fft)
    
    # Initialize with random permutation
    surrogate = rng.permutation(signal_data).astype(float)
    
    # Iterative adjustment
    for iteration in range(n_iterations):
        # FFT
        surr_fft = np.fft.fft(surrogate)
        
        # Replace amplitudes, keep phases
        phases = np.angle(surr_fft)
        adjusted_fft = target_power * np.exp(1j * phases)
        
        # IFFT
        surrogate = np.real(np.fft.ifft(adjusted_fft))
        
        # Rank-based amplitude adjustment
        sorted_indices = np.argsort(np.argsort(surrogate))
        surrogate = target_amplitudes[sorted_indices]
    
    return surrogate


def wavelet_permutation_surrogate(signal_data: np.ndarray, wavelet: str = 'db6',
                                  mode: str = 'periodization', seed: int = None) -> np.ndarray:
    """
    Generate surrogate by permuting wavelet coefficients within each scale.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    wavelet : str, default='db6'
        Wavelet type
    mode : str, default='periodization'
        Boundary mode
    seed : int, optional
        Random seed
        
    Returns:
    --------
    surrogate : np.ndarray
        Surrogate with permuted wavelet coefficients
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    N = len(signal_data)
    n = int(np.log2(N))
    
    # Compute DWT
    coeffs = pywt.wavedec(signal_data, wavelet, mode=mode, level=n-1)
    
    # Permute each level independently
    coeffs_perm = []
    for level_coeffs in coeffs:
        coeffs_perm.append(rng.permutation(level_coeffs))
    
    # Inverse DWT
    surrogate = pywt.waverec(coeffs_perm, wavelet, mode=mode)
    
    # Handle length mismatch
    if len(surrogate) > N:
        surrogate = surrogate[:N]
    
    return surrogate


def wavelet_rotation_surrogate(signal_data: np.ndarray, wavelet: str = 'db6',
                               mode: str = 'periodization', seed: int = None) -> np.ndarray:
    """
    Generate surrogate by cyclically rotating wavelet coefficients within each scale.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    wavelet : str, default='db6'
        Wavelet type
    mode : str, default='periodization'
        Boundary mode
    seed : int, optional
        Random seed
        
    Returns:
    --------
    surrogate : np.ndarray
        Surrogate with rotated wavelet coefficients
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    N = len(signal_data)
    n = int(np.log2(N))
    
    # Compute DWT
    coeffs = pywt.wavedec(signal_data, wavelet, mode=mode, level=n-1)
    
    # Rotate each level independently
    coeffs_rot = []
    for level_coeffs in coeffs:
        shift = rng.randint(0, len(level_coeffs))
        coeffs_rot.append(np.roll(level_coeffs, shift))
    
    # Inverse DWT
    surrogate = pywt.waverec(coeffs_rot, wavelet, mode=mode)
    
    # Handle length mismatch
    if len(surrogate) > N:
        surrogate = surrogate[:N]
    
    return surrogate


def generate_surrogate_ensemble(signal_data: np.ndarray, method: str, 
                                n_surrogates: int = 1000, 
                                base_seed: int = None, **kwargs) -> List[np.ndarray]:
    """
    Generate ensemble of surrogates using specified method.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    method : str
        One of: 'multifractal', 'iaft', 'permutation', 'rotation'
    n_surrogates : int, default=1000
        Number of surrogates to generate
    base_seed : int, optional
        Base random seed
    **kwargs : dict
        Additional arguments for surrogate method
        
    Returns:
    --------
    surrogates : List[np.ndarray]
        List of surrogate time series
    """
    method_map = {
        'multifractal': multifractal_cascade_surrogate,
        'iaft': iaft_surrogate,
        'permutation': wavelet_permutation_surrogate,
        'rotation': wavelet_rotation_surrogate
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
    
    surrogate_func = method_map[method]
    surrogates = []
    
    # Generate seeds deterministically from base_seed
    if base_seed is not None:
        rng = np.random.RandomState(base_seed)
        seeds = rng.randint(0, 2**31 - 1, size=n_surrogates)
    else:
        seeds = [None] * n_surrogates
    
    for i in range(n_surrogates):
        surr = surrogate_func(signal_data, seed=seeds[i], **kwargs)
        surrogates.append(surr)
    
    return surrogates
