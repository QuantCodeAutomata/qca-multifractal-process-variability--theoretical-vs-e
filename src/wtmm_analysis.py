"""
Wavelet Transform Modulus Maxima (WTMM) method for tau(q) estimation.

Implements:
- Continuous wavelet transform with complex Morlet wavelet
- Local modulus maxima detection
- Maxima line linking across scales
- Partition function computation
- tau(q) estimation via linear regression
"""

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from typing import Tuple, List, Dict


def morlet_wavelet(t: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """
    Complex Morlet wavelet.
    
    psi(t) = pi^{-1/4} * exp(i*omega_0*t) * exp(-t^2/2)
    
    Parameters:
    -----------
    t : np.ndarray
        Time array
    omega0 : float, default=6.0
        Central frequency (standard choice)
        
    Returns:
    --------
    psi : np.ndarray
        Complex wavelet values
    """
    norm = np.pi**(-0.25)
    psi = norm * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2.0)
    return psi


def cwt_morlet(signal_data: np.ndarray, scales: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """
    Continuous wavelet transform using complex Morlet wavelet.
    
    Uses FFT-based convolution for efficiency.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series of length N
    scales : np.ndarray
        Array of scales to compute CWT
    omega0 : float, default=6.0
        Morlet wavelet central frequency
        
    Returns:
    --------
    W : np.ndarray
        CWT coefficients, shape (len(scales), len(signal_data))
    """
    N = len(signal_data)
    n_scales = len(scales)
    
    # Pad to next power of 2 for FFT efficiency
    N_padded = int(2**np.ceil(np.log2(N)))
    signal_padded = np.zeros(N_padded)
    signal_padded[:N] = signal_data - np.mean(signal_data)  # Remove mean
    
    # FFT of signal
    signal_fft = np.fft.fft(signal_padded)
    
    # Frequency array
    freqs = 2 * np.pi * np.fft.fftfreq(N_padded)
    
    W = np.zeros((n_scales, N), dtype=complex)
    
    for i, scale in enumerate(scales):
        # Wavelet in Fourier domain: psi_hat(scale*omega)
        # For Morlet: psi_hat(omega) ~ exp(-(omega - omega0)^2 / 2)
        psi_hat = np.exp(-0.5 * (scale * freqs - omega0)**2)
        psi_hat[freqs < 0] = 0  # Analytic wavelet (positive frequencies only)
        
        # Normalization
        psi_hat = psi_hat * np.sqrt(scale)
        
        # Convolution via FFT
        W_scale = np.fft.ifft(signal_fft * psi_hat)
        W[i, :] = W_scale[:N]
    
    return W


def detect_local_maxima(modulus: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Detect local maxima in 1D array.
    
    Parameters:
    -----------
    modulus : np.ndarray
        1D array of wavelet modulus values
    order : int, default=1
        Number of points on each side to compare
        
    Returns:
    --------
    maxima_positions : np.ndarray
        Indices of local maxima
    """
    # Use maximum_filter to find local maxima
    local_max = maximum_filter(modulus, size=2*order+1, mode='constant', cval=0)
    maxima_mask = (modulus == local_max) & (modulus > 0)
    maxima_positions = np.where(maxima_mask)[0]
    
    return maxima_positions


def link_maxima_lines(W: np.ndarray, scales: np.ndarray, 
                      max_gap: int = 1) -> List[Dict]:
    """
    Link modulus maxima across scales to form maxima lines.
    
    Tracks from fine scales (small a) to coarse scales (large a).
    
    Parameters:
    -----------
    W : np.ndarray
        CWT coefficients, shape (n_scales, N)
    scales : np.ndarray
        Scale values
    max_gap : int, default=1
        Maximum scale gap allowed in linking
        
    Returns:
    --------
    maxima_lines : List[Dict]
        List of maxima lines, each containing:
        - 'scales': array of scales
        - 'positions': array of positions
        - 'modulus': array of modulus values
    """
    n_scales, N = W.shape
    modulus = np.abs(W)
    
    # Detect maxima at each scale
    maxima_at_scale = []
    for i in range(n_scales):
        positions = detect_local_maxima(modulus[i, :])
        maxima_at_scale.append(positions)
    
    # Link maxima from fine to coarse
    # Start from finest scale (index n_scales-1)
    maxima_lines = []
    
    # Initialize lines from finest scale
    if len(maxima_at_scale[-1]) > 0:
        for pos in maxima_at_scale[-1]:
            line = {
                'scale_indices': [n_scales - 1],
                'positions': [pos],
                'scales': [scales[-1]],
                'modulus': [modulus[n_scales - 1, pos]]
            }
            maxima_lines.append(line)
    
    # Track from fine to coarse (decreasing scale index)
    for scale_idx in range(n_scales - 2, -1, -1):
        current_maxima = maxima_at_scale[scale_idx]
        current_scale = scales[scale_idx]
        
        # Try to extend each existing line
        for line in maxima_lines:
            if line['scale_indices'][-1] <= scale_idx + max_gap + 1:
                # Line is still active, try to link
                last_pos = line['positions'][-1]
                
                # Search window: within +/- scale/2 of last position
                window = int(max(1, current_scale / 2))
                search_min = max(0, last_pos - window)
                search_max = min(N, last_pos + window + 1)
                
                # Find nearest maximum in search window
                candidates = current_maxima[(current_maxima >= search_min) & 
                                           (current_maxima < search_max)]
                
                if len(candidates) > 0:
                    # Choose nearest
                    distances = np.abs(candidates - last_pos)
                    nearest_idx = np.argmin(distances)
                    nearest_pos = candidates[nearest_idx]
                    
                    # Extend line
                    line['scale_indices'].append(scale_idx)
                    line['positions'].append(nearest_pos)
                    line['scales'].append(current_scale)
                    line['modulus'].append(modulus[scale_idx, nearest_pos])
    
    # Convert lists to arrays and filter lines that survive to coarse scales
    valid_lines = []
    for line in maxima_lines:
        if len(line['scales']) > 3:  # At least 4 scale points
            line['scales'] = np.array(line['scales'])
            line['positions'] = np.array(line['positions'])
            line['modulus'] = np.array(line['modulus'])
            line['scale_indices'] = np.array(line['scale_indices'])
            valid_lines.append(line)
    
    return valid_lines


def compute_partition_function(maxima_lines: List[Dict], scales: np.ndarray, 
                               q: float, epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute partition function Z(q, a) from maxima lines.
    
    Z(q, a) = sum_l |W(a, t_l)|^q
    
    Parameters:
    -----------
    maxima_lines : List[Dict]
        Maxima lines from link_maxima_lines
    scales : np.ndarray
        Scale values
    q : float
        Moment order
    epsilon : float, default=1e-10
        Threshold for small modulus values (avoid numerical issues for q<0)
        
    Returns:
    --------
    scales_valid : np.ndarray
        Scales with valid partition function
    Z_values : np.ndarray
        Partition function values at each scale
    """
    n_scales = len(scales)
    Z = np.zeros(n_scales)
    
    for scale_idx in range(n_scales):
        scale = scales[scale_idx]
        
        # Sum contributions from all maxima lines at this scale
        for line in maxima_lines:
            # Find if this line has a point at this scale
            mask = line['scale_indices'] == scale_idx
            if np.any(mask):
                idx = np.where(mask)[0][0]
                mod_val = line['modulus'][idx]
                
                # Apply threshold for negative q
                if q < 0 and mod_val < epsilon:
                    continue
                
                Z[scale_idx] += mod_val**q
    
    # Filter out scales with zero partition function
    valid_mask = Z > 0
    scales_valid = scales[valid_mask]
    Z_values = Z[valid_mask]
    
    return scales_valid, Z_values


def estimate_tau_q(signal_data: np.ndarray, q_values: np.ndarray, 
                   scale_range: Tuple[float, float] = None) -> np.ndarray:
    """
    Estimate scaling function tau(q) via WTMM method.
    
    Steps:
    1. Compute CWT with Morlet wavelet
    2. Detect and link modulus maxima
    3. Compute partition function Z(q, a) for each q
    4. Estimate tau(q) as slope of log Z vs log a
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Input time series
    q_values : np.ndarray
        Array of moment orders to estimate
    scale_range : Tuple[float, float], optional
        (a_min, a_max) for regression fit
        If None, uses (4, N/16)
        
    Returns:
    --------
    tau : np.ndarray
        Estimated tau(q) values
    """
    N = len(signal_data)
    
    # Generate log-spaced scale grid
    # From a_min=2 to a_max=N/8 with 32 voices per octave
    a_min = 2.0
    a_max = N / 8.0
    n_octaves = np.log2(a_max / a_min)
    n_voices = 32
    n_scales = int(n_octaves * n_voices) + 1
    scales = np.logspace(np.log2(a_min), np.log2(a_max), n_scales, base=2.0)
    
    # Compute CWT
    W = cwt_morlet(signal_data, scales)
    
    # Link maxima lines
    maxima_lines = link_maxima_lines(W, scales)
    
    if len(maxima_lines) == 0:
        # No maxima lines found, return NaN
        return np.full_like(q_values, np.nan, dtype=float)
    
    # Set scale range for regression
    if scale_range is None:
        fit_a_min = 4.0
        fit_a_max = N / 16.0
    else:
        fit_a_min, fit_a_max = scale_range
    
    # Estimate tau(q) for each q
    tau = np.zeros(len(q_values))
    
    for i, q in enumerate(q_values):
        if q == 0:
            # tau(0) = -1 by definition (partition function is number of maxima)
            tau[i] = -1.0
            continue
        
        # Compute partition function
        scales_z, Z = compute_partition_function(maxima_lines, scales, q)
        
        if len(scales_z) < 3:
            tau[i] = np.nan
            continue
        
        # Filter to fit range
        fit_mask = (scales_z >= fit_a_min) & (scales_z <= fit_a_max)
        if np.sum(fit_mask) < 3:
            tau[i] = np.nan
            continue
        
        scales_fit = scales_z[fit_mask]
        Z_fit = Z[fit_mask]
        
        # Linear regression: log Z vs log a
        log_a = np.log(scales_fit)
        log_Z = np.log(Z_fit)
        
        # tau(q) is the slope
        coeffs = np.polyfit(log_a, log_Z, 1)
        tau[i] = coeffs[0]
    
    return tau
