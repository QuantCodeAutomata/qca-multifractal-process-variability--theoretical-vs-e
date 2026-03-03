"""
Experiment 4: Auto-Mutual Information Preservation

Test nonlinear dependence preservation via auto-mutual information using KSG estimator.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
from tqdm import tqdm

from cascade_generation import generate_multifractal_series
from dependence_measures import compute_auto_mi, compute_mi_ensemble
from surrogate_methods import generate_surrogate_ensemble


def run_experiment_4(N: int = 65536, n_surrogates: int = 1000, 
                     sigma_w: float = 0.3, input_seed: int = 42,
                     max_lag: int = 21, k: int = 5, n_jobs: int = -1) -> dict:
    """
    Run Experiment 4 for given sample size N.
    
    Parameters:
    -----------
    N : int
        Sample size (power of 2)
    n_surrogates : int
        Number of surrogates per method
    sigma_w : float
        Lognormal cascade parameter
    input_seed : int
        Seed for input realization
    max_lag : int
        Maximum lag for MI
    k : int
        KSG parameter (number of neighbors)
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    results : dict
        Contains MI for input and all surrogate methods
    """
    print(f"\n=== Experiment 4: Auto-MI Preservation (N={N}) ===")
    
    # Generate input realization
    print(f"Generating input series (seed={input_seed})...")
    input_series = generate_multifractal_series(N, sigma_w, input_seed)
    
    # Compute input MI
    print("Computing input auto-MI...")
    mi_input = compute_auto_mi(input_series, max_lag, k)
    
    # Surrogate methods
    methods = {
        'multifractal': 'Multifractal Cascade',
        'iaft': 'IAFT',
        'permutation': 'Wavelet Permutation',
        'rotation': 'Wavelet Rotation'
    }
    
    results = {
        'N': N,
        'n_surrogates': n_surrogates,
        'sigma_w': sigma_w,
        'input_seed': input_seed,
        'max_lag': max_lag,
        'k': k,
        'mi_input': mi_input
    }
    
    # For each surrogate method
    for method_key, method_name in methods.items():
        print(f"\n--- Method: {method_name} ---")
        
        # Generate surrogates
        print(f"Generating {n_surrogates} {method_name} surrogates...")
        surrogates = generate_surrogate_ensemble(
            input_series, 
            method=method_key, 
            n_surrogates=n_surrogates,
            base_seed=1000 + list(methods.keys()).index(method_key)
        )
        
        # Compute MI for each surrogate
        print(f"Computing auto-MI for {method_name} surrogates...")
        mean_mi, lo_mi, hi_mi = compute_mi_ensemble(surrogates, max_lag, k)
        
        # Check coverage
        lags = np.arange(1, max_lag + 1)
        coverage = np.zeros(len(lags))
        for i in range(len(lags)):
            if lo_mi[i] <= mi_input[i] <= hi_mi[i]:
                coverage[i] = 1.0
        
        print(f"Input MI coverage: {np.mean(coverage)*100:.1f}%")
        
        results[method_key] = {
            'mean_mi': mean_mi,
            'lo_mi': lo_mi,
            'hi_mi': hi_mi,
            'coverage': coverage
        }
    
    return results


def plot_results(results: dict, save_dir: str = None):
    """Plot auto-MI comparison for all methods."""
    methods = {
        'multifractal': 'Multifractal Cascade',
        'iaft': 'IAFT',
        'permutation': 'Wavelet Permutation',
        'rotation': 'Wavelet Rotation'
    }
    
    max_lag = results['max_lag']
    lags = np.arange(1, max_lag + 1)
    mi_input = results['mi_input']
    N = results['N']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (method_key, method_name) in enumerate(methods.items()):
        ax = axes[idx]
        
        method_results = results[method_key]
        mean_mi = method_results['mean_mi']
        lo_mi = method_results['lo_mi']
        hi_mi = method_results['hi_mi']
        
        # Input MI
        ax.plot(lags, mi_input, 'ro', markersize=5, label='Input', zorder=3)
        
        # Surrogate mean
        ax.plot(lags, mean_mi, 'b-', linewidth=1, label='Surrogate Mean', zorder=2)
        
        # 95% band
        ax.fill_between(lags, lo_mi, hi_mi, alpha=0.3, color='blue',
                       label='95% Band', zorder=1)
        
        ax.set_xlabel('Lag (eta)', fontsize=11)
        ax.set_ylabel('Mutual Information (nats)', fontsize=11)
        ax.set_title(f'{method_name}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(np.max(hi_mi)*1.1, np.max(mi_input)*1.1)])
    
    plt.suptitle(f'Exp 4: Auto-MI Preservation (N={N}, k={results["k"]})', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'mi_surrogates_N{N}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Run Experiment 4 for both N=65536 and N=4096."""
    os.makedirs('results/exp_4', exist_ok=True)
    
    # N = 65536
    print("\n" + "="*60)
    print("Running Experiment 4: N=65536")
    print("="*60)
    results_65536 = run_experiment_4(N=65536, n_surrogates=1000, sigma_w=0.3, 
                                     input_seed=42, max_lag=21, k=5)
    
    # Save results
    np.savez('results/exp_4/results_N65536.npz', **results_65536)
    print("Saved: results/exp_4/results_N65536.npz")
    
    # Plot
    plot_results(results_65536, 'results/exp_4')
    
    # N = 4096
    print("\n" + "="*60)
    print("Running Experiment 4: N=4096")
    print("="*60)
    results_4096 = run_experiment_4(N=4096, n_surrogates=1000, sigma_w=0.3, 
                                    input_seed=42, max_lag=21, k=5)
    
    # Save results
    np.savez('results/exp_4/results_N4096.npz', **results_4096)
    print("Saved: results/exp_4/results_N4096.npz")
    
    # Plot
    plot_results(results_4096, 'results/exp_4')
    
    # Summary
    print("\n" + "="*60)
    print("Experiment 4 Summary")
    print("="*60)
    for N, res in [('65536', results_65536), ('4096', results_4096)]:
        print(f"\nN={N}:")
        for method in ['multifractal', 'iaft', 'permutation', 'rotation']:
            coverage = np.mean(res[method]['coverage']) * 100
            print(f"  {method:15s}: MI Coverage = {coverage:5.1f}%")
    
    return results_65536, results_4096


if __name__ == '__main__':
    main()
