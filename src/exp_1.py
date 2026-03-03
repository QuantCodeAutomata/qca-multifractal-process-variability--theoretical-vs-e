"""
Experiment 1: Multifractal Process Variability - Theoretical vs Empirical tau(q)

Generate 1000 independent realizations of synthetic lognormal cascade process
for N=65536 and N=4096, estimate tau(q) for each, and compare to theoretical curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
from tqdm import tqdm

from cascade_generation import (
    generate_multifractal_series,
    theoretical_tau_q,
    verify_cascade_parameters
)
from wtmm_analysis import estimate_tau_q


def run_single_realization(N: int, sigma_w: float, q_values: np.ndarray, seed: int) -> np.ndarray:
    """Run single realization: generate series and estimate tau(q)."""
    series = generate_multifractal_series(N, sigma_w, seed)
    tau = estimate_tau_q(series, q_values)
    return tau


def run_experiment_1(N: int = 65536, n_realizations: int = 1000, 
                     sigma_w: float = 0.3, n_jobs: int = -1) -> dict:
    """
    Run Experiment 1 for given sample size N.
    
    Parameters:
    -----------
    N : int
        Sample size (power of 2)
    n_realizations : int
        Number of independent realizations
    sigma_w : float
        Lognormal cascade parameter
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    results : dict
        Contains tau_q_all, q_grid, tau_theoretical, statistics
    """
    print(f"\n=== Experiment 1: N={N} ===")
    
    # Verify cascade parameters
    params = verify_cascade_parameters(sigma_w)
    print(f"Cascade parameters: mu_w={params['mu_w']:.4f}, sigma_w={params['sigma_w']:.4f}")
    print(f"E[W^2] = {params['E[W^2]']:.4f}")
    
    # Define q grid (skip q=0 or set tau(0)=-1)
    q_values = np.arange(-10, 11, 1, dtype=float)
    
    # Compute theoretical tau(q)
    tau_theoretical = theoretical_tau_q(q_values, sigma_w)
    
    # Generate seeds deterministically
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31 - 1, size=n_realizations)
    
    # Run realizations in parallel
    print(f"Generating {n_realizations} realizations and estimating tau(q)...")
    tau_q_all = Parallel(n_jobs=n_jobs)(
        delayed(run_single_realization)(N, sigma_w, q_values, seed)
        for seed in tqdm(seeds)
    )
    
    tau_q_all = np.array(tau_q_all)  # Shape: (n_realizations, len(q_values))
    
    # Compute statistics
    tau_mean = np.nanmean(tau_q_all, axis=0)
    tau_p2_5 = np.nanpercentile(tau_q_all, 2.5, axis=0)
    tau_p97_5 = np.nanpercentile(tau_q_all, 97.5, axis=0)
    
    # Check coverage of theoretical by empirical band
    coverage = np.zeros(len(q_values))
    for i in range(len(q_values)):
        if tau_p2_5[i] <= tau_theoretical[i] <= tau_p97_5[i]:
            coverage[i] = 1.0
    
    print(f"Theoretical tau(q) coverage by 95% band: {np.mean(coverage)*100:.1f}%")
    
    results = {
        'N': N,
        'n_realizations': n_realizations,
        'sigma_w': sigma_w,
        'q_values': q_values,
        'tau_theoretical': tau_theoretical,
        'tau_q_all': tau_q_all,
        'tau_mean': tau_mean,
        'tau_p2_5': tau_p2_5,
        'tau_p97_5': tau_p97_5,
        'coverage': coverage,
        'seeds': seeds
    }
    
    return results


def plot_results(results: dict, save_path: str = None):
    """Plot tau(q) comparison."""
    q_values = results['q_values']
    tau_theoretical = results['tau_theoretical']
    tau_mean = results['tau_mean']
    tau_p2_5 = results['tau_p2_5']
    tau_p97_5 = results['tau_p97_5']
    N = results['N']
    
    plt.figure(figsize=(10, 6))
    
    # Theoretical
    plt.plot(q_values, tau_theoretical, 'k-', linewidth=2.5, label='Theoretical', zorder=3)
    
    # Empirical mean
    plt.plot(q_values, tau_mean, 'b--', linewidth=1.5, label='Empirical Mean', zorder=2)
    
    # 95% band
    plt.fill_between(q_values, tau_p2_5, tau_p97_5, alpha=0.3, color='blue', 
                     label='95% Percentile Band', zorder=1)
    
    plt.xlabel('q', fontsize=12)
    plt.ylabel(r'$\tau(q)$', fontsize=12)
    plt.title(f'Exp 1: tau(q) Variability (N={N}, n={results["n_realizations"]})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Run Experiment 1 for both N=65536 and N=4096."""
    os.makedirs('results/exp_1', exist_ok=True)
    
    # N = 65536
    print("\n" + "="*60)
    print("Running Experiment 1: N=65536")
    print("="*60)
    results_65536 = run_experiment_1(N=65536, n_realizations=1000, sigma_w=0.3)
    
    # Save results
    np.savez('results/exp_1/results_N65536.npz', **results_65536)
    print("Saved: results/exp_1/results_N65536.npz")
    
    # Plot
    plot_results(results_65536, 'results/exp_1/tau_q_N65536.png')
    
    # N = 4096
    print("\n" + "="*60)
    print("Running Experiment 1: N=4096")
    print("="*60)
    results_4096 = run_experiment_1(N=4096, n_realizations=1000, sigma_w=0.3)
    
    # Save results
    np.savez('results/exp_1/results_N4096.npz', **results_4096)
    print("Saved: results/exp_1/results_N4096.npz")
    
    # Plot
    plot_results(results_4096, 'results/exp_1/tau_q_N4096.png')
    
    # Summary
    print("\n" + "="*60)
    print("Experiment 1 Summary")
    print("="*60)
    print(f"N=65536: Coverage = {np.mean(results_65536['coverage'])*100:.1f}%")
    print(f"N=4096:  Coverage = {np.mean(results_4096['coverage'])*100:.1f}%")
    
    return results_65536, results_4096


if __name__ == '__main__':
    main()
