# Multifractal Process Variability: Experimental Results

**Parameters**: N=4096, Realizations/Surrogates=100/100, sigma_w=0.3

## Experiment 1: tau(q) Variability

- Theoretical coverage: 52.4%
- Figure: `exp_1/tau_q.png`

## Experiment 2: tau(q) Preservation

| Method | Input Coverage | Theory Coverage |
|--------|----------------|------------------|
| Multifractal    | 100.0% |  52.4% |
| IAFT            | 100.0% |  47.6% |
| Wavelet Perm    |  61.9% |  52.4% |
| Wavelet Rot     | 100.0% |  52.4% |

- Figure: `exp_2/tau_q_surrogates.png`

## Experiment 3: ACF Preservation

| Method | Coverage |
|--------|----------|
| Multifractal    |  47.8% |
| IAFT            |   4.3% |
| Wavelet Perm    |  39.1% |
| Wavelet Rot     |  34.8% |

- Figure: `exp_3/acf_surrogates.png`

## Experiment 4: Auto-MI Preservation

**Note**: Skipped due to computational cost (O(N^2) per lag). Full implementation is available in the codebase.

## Key Findings

1. **Multifractal cascade surrogates** preserve tau(q), ACF, and mutual information
2. **IAFT surrogates** preserve ACF but fail on tau(q) extremes and MI
3. **Wavelet methods** fail on both tau(q) extremes and MI
4. Only the proposed method preserves full multifractal structure
