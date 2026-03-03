# Multifractal Process Variability: Theoretical vs Empirical Analysis

This repository implements a comprehensive study of multifractal scaling function tau(q) estimation and surrogate data validation using the Wavelet Transform Modulus Maxima (WTMM) method.

## Overview

The project validates a novel multifractal surrogate generation method based on random cascades on wavelet dyadic trees, comparing it against alternative surrogate approaches (IAFT, wavelet permutation, wavelet rotation) through:

1. **Experiment 1**: Establishing baseline variability of tau(q) estimation across 1000 independent realizations
2. **Experiment 2**: Testing tau(q) preservation across four surrogate methods
3. **Experiment 3**: Verifying autocorrelation function (ACF) preservation
4. **Experiment 4**: Testing nonlinear dependence preservation via auto-mutual information

## Key Features

- Synthetic lognormal random cascade generation on DWT dyadic trees (DAUB12 wavelets)
- WTMM-based tau(q) estimation with Morlet wavelet CWT
- Four surrogate generation methods implementation
- KSG mutual information estimator for nonlinear dependence testing
- Comprehensive visualization and statistical analysis

## Repository Structure

```
.
├── src/
│   ├── cascade_generation.py    # Synthetic multifractal cascade generation
│   ├── wtmm_analysis.py          # WTMM tau(q) estimation
│   ├── surrogate_methods.py      # Four surrogate generation methods
│   ├── dependence_measures.py    # ACF and mutual information
│   ├── exp_1.py                  # Experiment 1: tau(q) variability
│   ├── exp_2.py                  # Experiment 2: tau(q) preservation
│   ├── exp_3.py                  # Experiment 3: ACF preservation
│   └── exp_4.py                  # Experiment 4: Mutual information
├── tests/
│   └── test_methodology.py       # Comprehensive test suite
├── results/
│   └── RESULTS.md                # Summary metrics and findings
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:

```bash
python -m src.exp_1
python -m src.exp_2
python -m src.exp_3
python -m src.exp_4
```

Run tests:

```bash
pytest tests/ -v
```

## Methodology

### Cascade Generation
- Uses DAUB12 (db6) wavelets with periodic boundary conditions
- Lognormal multiplicators: sigma_W=0.3, mu_W=-0.5*sigma_W^2
- Dyadic tree recursion with scale-by-scale construction

### WTMM Analysis
- Complex Morlet wavelet (omega_0=6) for CWT
- Log-spaced scale grid: 32 voices per octave
- Maxima line linking with gap tolerance
- Linear regression of log Z(q,a) vs log(a) for tau(q)

### Surrogate Methods
1. **Multifractal cascade**: Random permutation of multiplicators with amplitude adjustment
2. **IAFT**: Iterative amplitude-adjusted Fourier transform (100 iterations)
3. **Wavelet permutation**: Independent random permutation per scale
4. **Wavelet rotation**: Independent cyclic rotation per scale

### Dependence Measures
- **ACF**: Normalized autocorrelation for lags 1-21
- **Mutual Information**: KSG estimator with k=5 nearest neighbors

## Results

All results including figures and metrics are saved to `results/` directory. See `results/RESULTS.md` for detailed findings.

## License

MIT License

## References

Based on the methodology from Arneodo et al. and related multifractal analysis literature.
