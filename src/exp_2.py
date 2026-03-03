"""
Experiment 2: tau(q) Preservation Across Surrogate Methods

Generate 1000 surrogates using four methods and compare tau(q) preservation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
from tqdm import tqdm

from cascade_generation import (
    generate_multifractal_series,
    theoretical_tau_q
)
from wtmm_analysis import estimate_tau_q
from surrogate_methods import generate_surrogate_ensemble


def compute_surrogate_tau_q(surrogate: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    """Compute tau(q) for a single surrogate."""
    return estimate_tau_q(surrogate, q_values)


def run_experiment_2(N: int = 65536, n_surrogates: int = 1000, 
                     sigma_w: float = 0.3, input_seed: int = 42,
                     n_jobs: int = -1) -> dict:
    """
    Run Experiment 2 for given sample size N.
    
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
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    results : dict
        Contains input series, tau(q) for all methods
    """
    print(f"\n=== Experiment 2: N={N} ===")
    
    # Generate input realization
    print(f"Generating input series (seed={input_seed})...")
    input_series = generate_multifractal_series(N, sigma_w, input_seed)
    
    # Define q grid
    q_values = np.arange(-10, 11, 1, dtype=float)
    
    # Compute theoretical tau(q)
    tau_theoretical = theoretical_tau_q(q_values, sigma_w)
    
    # Compute input tau(q)
    print("Computing input tau(q)...")
    tau_input = estimate_tau_q(input_series, q_values)
    
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
        'q_values': q_values,
        'tau_theoretical': tau_theoretical,
        'tau_input': tau_input,
        'input_series': input_series
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
        
        # Compute tau(q) for each surrogate
        print(f"Computing tau(q) for {method_name} surrogates...")
        tau_q_surrogates = Parallel(n_jobs=n_jobs)(
            delayed(compute_surrogate_tau_q)(surr, q_values)
            for surr in tqdm(surrogates)
        )
        
        tau_q_surrogates = np.array(tau_q_surrogates)
        
        # Compute statistics
        tau_mean = np.nanmean(tau_q_surrogates, axis=0)
        tau_p2_5 = np.nanpercentile(tau_q_surrogates, 2.5, axis=0)
        tau_p97_5 = np.nanpercentile(tau_q_surrogates, 97.5, axis=0)
        
        # Check coverage
        input_coverage = np.zeros(len(q_values))
        theory_coverage = np.zeros(len(q_values))
        for i in range(len(q_values)):
            if not np.isnan(tau_p2_5[i]) and not np.isnan(tau_p97_5[i]):
                if tau_p2_5[i] <= tau_input[i] <= tau_p97_5[i]:
                    input_coverage[i] = 1.0
                if tau_p2_5[i] <= tau_theoretical[i] <= tau_p97_5[i]:
                    theory_coverage[i] = 1.0
        
        print(f"Input tau(q) coverage: {np.mean(input_coverage)*100:.1f}%")
        print(f"Theoretical tau(q) coverage: {np.mean(theory_coverage)*100:.1f}%")
        
        results[method_key] = {
            'tau_q_all': tau_q_surrogates,
            'tau_mean': tau_mean,
            'tau_p2_5': tau_p2_5,
            'tau_p97_5': tau_p97_5,
            'input_coverage': input_coverage,
            'theory_coverage': theory_coverage
        }
    
    return results


def plot_results(results: dict, save_dir: str = None):
    """Plot tau(q) comparison for all methods."""
    methods = {
        'multifractal': 'Multifractal Cascade',
        'iaft': 'IAFT',
        'permutation': 'Wavelet Permutation',
        'rotation': 'Wavelet Rotation'
    }
    
    q_values = results['q_values']
    tau_theoretical = results['tau_theoretical']
    tau_input = results['tau_input']
    N = results['N']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (method_key, method_name) in enumerate(methods.items()):
        ax = axes[idx]
        
        method_results = results[method_key]
        tau_mean = method_results['tau_mean']
        tau_p2_5 = method_results['tau_p2_5']
        tau_p97_5 = method_results['tau_p97_5']
        
        # Theoretical
        ax.plot(q_values, tau_theoretical, 'k-', linewidth=2.5, label='Theoretical', zorder=4)
        
        # Input
        ax.plot(q_values, tau_input, 'ro', markersize=6, label='Input', zorder=5)
        
        # Surrogate mean
        ax.plot(q_values, tau_mean, 'b-', linewidth=1, label='Surrogate Mean', zorder=2)
        
        # 95% band
        ax.fill_between(q_values, tau_p2_5, tau_p97_5, alpha=0.3, color='blue',
                       label='95% Band', zorder=1)
        
        ax.set_xlabel('q', fontsize=11)
        ax.set_ylabel(r'$\tau(q)$', fontsize=11)
        ax.set_title(f'{method_name}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Exp 2: tau(q) Preservation (N={N})', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'tau_q_surrogates_N{N}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Run Experiment 2 for both N=65536 and N=4096."""
    os.makedirs('results/exp_2', exist_ok=True)
    
    # N = 65536
    print("\n" + "="*60)
    print("Running Experiment 2: N=65536")
    print("="*60)
    results_65536 = run_experiment_2(N=65536, n_surrogates=1000, sigma_w=0.3, input_seed=42)
    
    # Save results (exclude input_series and full tau_q_all to save space)
    save_dict = {k: v for k, v in results_65536.items() if k != 'input_series'}
    for method in ['multifractal', 'iaft', 'permutation', 'rotation']:
        if method in save_dict:
            save_dict[method] = {k: v for k, v in save_dict[method].items() if k != 'tau_q_all'}
    
    np.savez('results/exp_2/results_N65536.npz', **save_dict)
    print("Saved: results/exp_2/results_N65536.npz")
    
    # Plot
    plot_results(results_65536, 'results/exp_2')
    
    # N = 4096
    print("\n" + "="*60)
    print("Running Experiment 2: N=4096")
    print("="*60)
    results_4096 = run_experiment_2(N=4096, n_surrogates=1000, sigma_w=0.3, input_seed=42)
    
    # Save results
    save_dict = {k: v for k, v in results_4096.items() if k != 'input_series'}
    for method in ['multifractal', 'iaft', 'permutation', 'rotation']:
        if method in save_dict:
            save_dict[method] = {k: v for k, v in save_dict[method].items() if k != 'tau_q_all'}
    
    np.savez('results/exp_2/results_N4096.npz', **save_dict)
    print("Saved: results/exp_2/results_N4096.npz")
    
    # Plot
    plot_results(results_4096, 'results/exp_2')
    
    # Summary
    print("\n" + "="*60)
    print("Experiment 2 Summary")
    print("="*60)
    for N, res in [('65536', results_65536), ('4096', results_4096)]:
        print(f"\nN={N}:")
        for method in ['multifractal', 'iaft', 'permutation', 'rotation']:
            input_cov = np.mean(res[method]['input_coverage']) * 100
            theory_cov = np.mean(res[method]['theory_coverage']) * 100
            print(f"  {method:15s}: Input={input_cov:5.1f}%, Theory={theory_cov:5.1f}%")
    
    return results_65536, results_4096


if __name__ == '__main__':
    main()
