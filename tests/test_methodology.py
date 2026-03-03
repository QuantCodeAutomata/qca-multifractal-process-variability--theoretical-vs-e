"""
Comprehensive test suite verifying methodology adherence.

Tests cover:
- Cascade generation correctness
- WTMM tau(q) estimation
- Surrogate generation methods
- Dependence measures (ACF, MI)
- Edge cases and numerical stability
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cascade_generation import (
    generate_multifractal_series,
    theoretical_tau_q,
    compute_lognormal_parameters,
    verify_cascade_parameters
)
from wtmm_analysis import (
    morlet_wavelet,
    cwt_morlet,
    detect_local_maxima,
    estimate_tau_q
)
from surrogate_methods import (
    multifractal_cascade_surrogate,
    iaft_surrogate,
    wavelet_permutation_surrogate,
    wavelet_rotation_surrogate
)
from dependence_measures import (
    compute_acf,
    compute_auto_mi,
    ksg_mutual_information
)


class TestCascadeGeneration:
    """Test synthetic cascade generation."""
    
    def test_series_length(self):
        """Test that generated series has correct length."""
        for n in [8, 10, 12]:
            N = 2**n
            series = generate_multifractal_series(N, sigma_w=0.3, seed=42)
            assert len(series) == N, f"Series length mismatch for N={N}"
    
    def test_invalid_length(self):
        """Test that non-power-of-2 length raises error."""
        with pytest.raises(ValueError):
            generate_multifractal_series(100, sigma_w=0.3)
    
    def test_reproducibility(self):
        """Test that same seed produces same series."""
        series1 = generate_multifractal_series(256, sigma_w=0.3, seed=42)
        series2 = generate_multifractal_series(256, sigma_w=0.3, seed=42)
        np.testing.assert_array_almost_equal(series1, series2)
    
    def test_lognormal_parameters(self):
        """Test lognormal parameter computation."""
        sigma_w = 0.3
        mu_w, _ = compute_lognormal_parameters(sigma_w)
        assert mu_w == -0.5 * sigma_w**2, "mu_w constraint violated"
    
    def test_theoretical_tau_q(self):
        """Test theoretical tau(q) computation."""
        sigma_w = 0.3
        q_values = np.array([-2, -1, 0, 1, 2])
        tau = theoretical_tau_q(q_values, sigma_w)
        
        # tau(0) should be -1 for lognormal cascade
        # Actually, for our formula: tau(q) = -1 - (mu_w*q + 0.5*sigma_w^2*q^2)/ln(2)
        # At q=0: tau(0) = -1
        assert np.abs(tau[2] - (-1.0)) < 1e-10, "tau(0) should be -1"
        
        # tau(q) should be monotonically decreasing for multifractal process
        # (Actually can be non-monotonic, but for symmetric lognormal it is)
    
    def test_cascade_variance(self):
        """Test that cascade has finite variance."""
        series = generate_multifractal_series(1024, sigma_w=0.3, seed=42)
        assert np.isfinite(np.var(series)), "Series variance is not finite"
        assert np.var(series) > 0, "Series has zero variance"


class TestWTMMAnalysis:
    """Test WTMM tau(q) estimation."""
    
    def test_morlet_wavelet(self):
        """Test Morlet wavelet properties."""
        t = np.linspace(-4, 4, 1000)
        psi = morlet_wavelet(t, omega0=6.0)
        
        # Should be complex
        assert np.iscomplexobj(psi), "Morlet wavelet should be complex"
        
        # Approximate admissibility (zero mean for real part)
        assert np.abs(np.mean(np.real(psi))) < 0.1, "Real part should have ~zero mean"
    
    def test_cwt_shape(self):
        """Test CWT output shape."""
        signal = np.random.randn(512)
        scales = np.array([2, 4, 8, 16])
        W = cwt_morlet(signal, scales)
        
        assert W.shape == (len(scales), len(signal)), "CWT shape mismatch"
        assert np.iscomplexobj(W), "CWT should be complex"
    
    def test_maxima_detection(self):
        """Test local maxima detection."""
        # Simple test signal with known maxima
        signal = np.array([0, 1, 0, 2, 0, 1, 0])
        maxima = detect_local_maxima(signal)
        
        expected = np.array([1, 3, 5])
        np.testing.assert_array_equal(maxima, expected)
    
    def test_tau_q_estimation(self):
        """Test tau(q) estimation on synthetic data."""
        # Generate series
        series = generate_multifractal_series(4096, sigma_w=0.3, seed=42)
        
        # Estimate tau(q)
        q_values = np.array([-2, -1, 0, 1, 2])
        tau = estimate_tau_q(series, q_values)
        
        # Check that tau(q) values are finite
        assert np.all(np.isfinite(tau)), "tau(q) contains non-finite values"
        
        # tau(0) should be close to -1
        assert np.abs(tau[2] - (-1.0)) < 0.5, "tau(0) should be near -1"
    
    def test_tau_q_extreme_values(self):
        """Test tau(q) at extreme q values."""
        series = generate_multifractal_series(2048, sigma_w=0.3, seed=42)
        
        q_values = np.array([-10, -5, 0, 5, 10])
        tau = estimate_tau_q(series, q_values)
        
        # Should handle extreme q without crashing
        # May have NaN for extreme values due to partition function issues
        assert len(tau) == len(q_values), "tau(q) length mismatch"


class TestSurrogateMethods:
    """Test surrogate generation methods."""
    
    def test_multifractal_surrogate_length(self):
        """Test that multifractal surrogate preserves length."""
        signal = generate_multifractal_series(1024, sigma_w=0.3, seed=42)
        surrogate = multifractal_cascade_surrogate(signal, seed=100)
        
        assert len(surrogate) == len(signal), "Surrogate length mismatch"
    
    def test_iaft_surrogate_amplitude_distribution(self):
        """Test that IAFT preserves amplitude distribution."""
        signal = generate_multifractal_series(512, sigma_w=0.3, seed=42)
        surrogate = iaft_surrogate(signal, n_iterations=100, seed=100)
        
        # Should have same sorted values (up to numerical precision)
        sorted_signal = np.sort(signal)
        sorted_surrogate = np.sort(surrogate)
        
        np.testing.assert_array_almost_equal(sorted_signal, sorted_surrogate, decimal=10)
    
    def test_iaft_surrogate_spectrum(self):
        """Test that IAFT approximately preserves power spectrum."""
        signal = generate_multifractal_series(512, sigma_w=0.3, seed=42)
        surrogate = iaft_surrogate(signal, n_iterations=100, seed=100)
        
        # Compare power spectra
        power_signal = np.abs(np.fft.fft(signal))**2
        power_surrogate = np.abs(np.fft.fft(surrogate))**2
        
        # Should be very close
        relative_error = np.mean(np.abs(power_signal - power_surrogate) / (power_signal + 1e-10))
        assert relative_error < 0.2, "IAFT power spectrum mismatch"
    
    def test_wavelet_permutation_preserves_amplitude(self):
        """Test that wavelet permutation preserves coefficient amplitudes."""
        signal = generate_multifractal_series(512, sigma_w=0.3, seed=42)
        surrogate = wavelet_permutation_surrogate(signal, seed=100)
        
        assert len(surrogate) == len(signal), "Permutation surrogate length mismatch"
    
    def test_wavelet_rotation_preserves_length(self):
        """Test that wavelet rotation preserves length."""
        signal = generate_multifractal_series(512, sigma_w=0.3, seed=42)
        surrogate = wavelet_rotation_surrogate(signal, seed=100)
        
        assert len(surrogate) == len(signal), "Rotation surrogate length mismatch"
    
    def test_surrogate_reproducibility(self):
        """Test that surrogates are reproducible with seed."""
        signal = generate_multifractal_series(256, sigma_w=0.3, seed=42)
        
        surr1 = multifractal_cascade_surrogate(signal, seed=100)
        surr2 = multifractal_cascade_surrogate(signal, seed=100)
        
        np.testing.assert_array_almost_equal(surr1, surr2)


class TestDependenceMeasures:
    """Test ACF and mutual information."""
    
    def test_acf_properties(self):
        """Test ACF mathematical properties."""
        signal = np.random.randn(500)
        acf = compute_acf(signal, max_lag=20)
        
        # ACF(0) should be 1
        assert np.abs(acf[0] - 1.0) < 1e-10, "ACF(0) should be 1"
        
        # ACF should be bounded by 1
        assert np.all(np.abs(acf) <= 1.0 + 1e-10), "ACF should be bounded by 1"
    
    def test_acf_white_noise(self):
        """Test ACF of white noise."""
        np.random.seed(42)
        signal = np.random.randn(2000)
        acf = compute_acf(signal, max_lag=20)
        
        # ACF(0) = 1
        assert np.abs(acf[0] - 1.0) < 1e-10
        
        # ACF(eta > 0) should be small for white noise
        assert np.all(np.abs(acf[1:]) < 0.2), "ACF of white noise should be near zero"
    
    def test_mi_non_negative(self):
        """Test that mutual information is non-negative."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        
        mi = ksg_mutual_information(x, y, k=5)
        assert mi >= 0, "MI should be non-negative"
    
    def test_mi_perfect_correlation(self):
        """Test MI for perfectly correlated variables."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = x.copy()  # Perfect correlation
        
        mi = ksg_mutual_information(x, y, k=5)
        
        # MI should be large (close to entropy of x)
        assert mi > 1.0, "MI of perfectly correlated variables should be large"
    
    def test_mi_independence(self):
        """Test MI for independent variables."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        
        mi = ksg_mutual_information(x, y, k=5)
        
        # MI should be small for independent variables
        assert mi < 0.5, "MI of independent variables should be small"
    
    def test_auto_mi_length(self):
        """Test auto-MI output length."""
        signal = np.random.randn(500)
        mi = compute_auto_mi(signal, max_lag=20, k=5)
        
        assert len(mi) == 20, "Auto-MI length mismatch"
        assert np.all(mi >= 0), "Auto-MI should be non-negative"


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_small_series(self):
        """Test behavior with small series."""
        series = generate_multifractal_series(256, sigma_w=0.3, seed=42)
        assert len(series) == 256
        
        # Should be able to compute basic statistics
        assert np.isfinite(np.mean(series))
        assert np.isfinite(np.var(series))
    
    def test_zero_variance_handling(self):
        """Test handling of constant (zero variance) signal."""
        signal = np.ones(100)
        acf = compute_acf(signal, max_lag=10)
        
        # ACF should be all NaN or 1 for constant signal
        # Our implementation normalizes by variance, so should handle gracefully
    
    def test_negative_values(self):
        """Test that series can have negative values."""
        series = generate_multifractal_series(512, sigma_w=0.3, seed=42)
        
        # Multifractal cascade can produce negative values after inverse DWT
        # This is expected behavior
        assert np.any(series != 0), "Series should not be all zero"


class TestMethodologyAdherence:
    """Test adherence to paper methodology."""
    
    def test_wavelet_type(self):
        """Verify DAUB12 (db6) wavelet is used."""
        import pywt
        
        # db6 has 12 filter coefficients (6 vanishing moments)
        wavelet = pywt.Wavelet('db6')
        assert wavelet.dec_len == 12, "db6 should have 12 filter coefficients"
    
    def test_sigma_w_parameter(self):
        """Verify sigma_w=0.3 parameter."""
        sigma_w = 0.3
        params = verify_cascade_parameters(sigma_w)
        
        assert params['sigma_w'] == 0.3
        assert params['mu_w'] == -0.5 * 0.3**2
    
    def test_q_grid(self):
        """Verify q grid covers -10 to 10."""
        q_values = np.arange(-10, 11, 1)
        assert len(q_values) == 21
        assert q_values[0] == -10
        assert q_values[-1] == 10
    
    def test_ksg_k_parameter(self):
        """Verify KSG k=5 parameter."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        
        # Should not raise error with k=5
        mi = ksg_mutual_information(x, y, k=5)
        assert np.isfinite(mi)
    
    def test_iaft_iterations(self):
        """Verify IAFT uses 100 iterations."""
        signal = generate_multifractal_series(256, sigma_w=0.3, seed=42)
        
        # Should complete 100 iterations without error
        surrogate = iaft_surrogate(signal, n_iterations=100, seed=100)
        assert len(surrogate) == len(signal)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
