import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from imhg_calculator import imHGCalculator


def run_simulation(N=1000, Pin=0.7, Pout=0.1, interval_length=None, seed=None, verbose=True):
    """
    Run a simulation experiment for imHG calculation.
    
    Parameters:
    -----------
    N : int
        Vector length (default=1000)
    Pin : float
        Probability of 1 inside the interval (default=0.7)
    Pout : float
        Probability of 1 outside the interval (default=0.1)
    interval_length : int, optional
        Fixed interval length. If None, randomly drawn between 50-100
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Whether to print progress messages (default=True)
    
    Returns:
    --------
    dict : Dictionary containing simulation results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate parameters
    if Pout >= Pin:
        raise ValueError("Pout must be less than Pin")
    
    # Draw s uniformly from 0 to 999
    s = np.random.randint(0, 1000)
    if verbose:
        print(f"Drew interval start s = {s}")
    
    # Determine interval end
    if interval_length is not None:
        e = s + interval_length - 1
    else:
        # Draw e uniformly from s+50 to s+100
        e = np.random.randint(s + 50, s + 101)
    if verbose:
        print(f"Drew interval end e = {e}")
    
    # Ensure e doesn't exceed N-1
    e = min(e, N - 1)
    if verbose:
        print(f"Adjusted e to {e} (capped at N-1)")
    
    # Create a zero vector of size N
    vector = np.zeros(N, dtype=int)
    if verbose:
        print(f"Created zero vector of size {N}")
    
    # Inside the interval [s, e], flip each 0 to 1 with probability Pin
    for i in range(s, e + 1):
        if np.random.random() < Pin:
            vector[i] = 1
    ones_inside = int(np.sum(vector[s:e+1]))
    if verbose:
        print(f"Flipped {ones_inside} positions to 1 inside interval [{s}, {e}] (Pin={Pin})")
    
    # Outside the interval [s, e], flip each 0 to 1 with probability Pout
    for i in range(0, s):
        if np.random.random() < Pout:
            vector[i] = 1
    for i in range(e + 1, N):
        if np.random.random() < Pout:
            vector[i] = 1
    ones_outside = int(np.sum(vector)) - ones_inside
    if verbose:
        print(f"Flipped {ones_outside} positions to 1 outside interval (Pout={Pout})")
    
    # Calculate B (number of active genes / 1s in the vector)
    B = int(np.sum(vector))
    if verbose:
        print(f"Total number of 1s (B) = {B}")
    
    # Convert vector to tuple for imHG calculation
    lamda = tuple(vector)
    
    # Run imHG calculator
    if verbose:
        print("Running imHG calculator...")
    calculator = imHGCalculator()
    imhg_score, (start_idx, end_idx) = calculator.calculate_imhg(N, B, lamda)
    if verbose:
        print(f"imHG calculation complete. Score = {imhg_score}, interval = [{start_idx}, {end_idx}]")
    
    # Calculate p-value
    if verbose:
        print("Calculating p-value...")
    p_value = calculator.calculate_p_value(N, B, imhg_score)
    if verbose:
        print(f"P-value calculation complete. P-value = {p_value}")
    
    results = {
        'N': N,
        'Pin': Pin,
        'Pout': Pout,
        'true_interval': (s, e),
        'true_interval_length': e - s + 1,
        'B': B,
        'imhg_score': imhg_score,
        'detected_interval': (start_idx, end_idx),
        'detected_interval_length': end_idx - start_idx + 1,
        'p_value': p_value,
        'vector': vector
    }
    
    return results


def _run_single_simulation(sim_idx, N, Pin, Pout, interval_length, alpha):
    """
    Helper function for parallel execution of a single simulation.
    Returns 1 if significant, 0 otherwise.
    """
    try:
        result = run_simulation(
            N=N, 
            Pin=Pin, 
            Pout=Pout, 
            interval_length=interval_length,
            verbose=False
        )
        return 1 if result['p_value'] < alpha else 0
    except (ValueError, RuntimeError):
        return 0


def _compute_permutation_score(perm_idx, vector, N, B):
    """
    Helper function for parallel execution of a single permutation.
    Computes imHG score for a permuted vector.
    """
    permuted_vector = np.random.permutation(vector)
    lamda = tuple(permuted_vector)
    calculator = imHGCalculator()
    try:
        score, _ = calculator.calculate_imhg(N, B, lamda)
        return score
    except (ValueError, RuntimeError):
        return np.nan


def _compute_empirical_pvalue_single_permutation(perm_idx, vector, N, B, original_imhg_score):
    """
    Helper function for parallel execution of a single permutation.
    Returns 1 if permuted score <= original score, 0 otherwise.
    """
    # Permute the vector
    permuted_vector = np.random.permutation(vector)
    lamda = tuple(permuted_vector)
    
    # Compute imHG score for permuted vector
    calculator = imHGCalculator()
    try:
        permuted_score, _ = calculator.calculate_imhg(N, B, lamda)
        return 1 if permuted_score <= original_imhg_score else 0
    except (ValueError, RuntimeError):
        return 0


def compute_empirical_pvalue(vector, N, B, original_imhg_score, n_permutations=10000, n_jobs=None):
    """
    Compute empirical p-value via Monte Carlo permutation test.
    
    Parameters:
    -----------
    vector : array-like
        The binary vector
    N : int
        Vector length
    B : int
        Number of 1s in the vector
    original_imhg_score : float
        The imHG score from the original (unpermuted) vector
    n_permutations : int
        Number of Monte Carlo permutations (default=10000)
    n_jobs : int, optional
        Number of parallel workers
        
    Returns:
    --------
    float : Empirical p-value
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    worker_func = partial(
        _compute_empirical_pvalue_single_permutation,
        vector=vector,
        N=N,
        B=B,
        original_imhg_score=original_imhg_score
    )
    
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker_func, range(n_permutations))
    
    # Empirical p-value = (count of permuted scores <= original + 1) / (n_permutations + 1)
    # Adding 1 to both numerator and denominator for proper p-value estimation
    empirical_pvalue = (sum(results) + 1) / (n_permutations + 1)
    
    return empirical_pvalue


def _run_pvalue_comparison_simulation(sim_idx, N, Pin, Pout, interval_length, n_permutations, n_jobs_inner):
    """
    Helper function for parallel execution of p-value comparison.
    Returns dict with theoretical and empirical p-values.
    """
    try:
        result = run_simulation(
            N=N, 
            Pin=Pin, 
            Pout=Pout, 
            interval_length=interval_length,
            verbose=False
        )
        
        theoretical_pvalue = result['p_value']
        
        # Compute empirical p-value (using single worker internally to avoid nested parallelism)
        empirical_pvalue = compute_empirical_pvalue(
            vector=result['vector'],
            N=N,
            B=result['B'],
            original_imhg_score=result['imhg_score'],
            n_permutations=n_permutations,
            n_jobs=1  # Single worker to avoid nested parallelism issues
        )
        
        return {
            'theoretical': theoretical_pvalue,
            'empirical': empirical_pvalue,
            'imhg_score': result['imhg_score']
        }
    except (ValueError, RuntimeError):
        return None


def generate_pvalue_tightness_figure(
    N=1000,
    Pout=0.1,
    Pin=0.7,
    interval_length=75,
    n_permutations=10000,
    output_dir='output',
    seed=None,
    n_jobs=None
):
    """
    Generate Figure 1B: P-value tightness analysis showing the distribution of 
    imHG scores from permutations with a dashed red line for the observed statistic.
    
    Parameters:
    -----------
    N : int
        Vector length (default=1000)
    Pout : float
        Probability of 1 outside the interval (default=0.1)
    Pin : float
        Probability of 1 inside the interval (default=0.7)
    interval_length : int
        Interval length (default=75)
    n_permutations : int
        Number of Monte Carlo permutations (default=10000)
    output_dir : str
        Output directory for saving the figure (default='output')
    seed : int, optional
        Random seed for reproducibility
    n_jobs : int, optional
        Number of parallel jobs
        
    Returns:
    --------
    dict : Dictionary containing permutation test results
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Using {n_jobs} parallel workers")
    print(f"Running {n_permutations} permutations")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation to get observed data
    print(f"\nGenerating observed data: Pin={Pin}, interval_length={interval_length}")
    result = run_simulation(
        N=N, 
        Pin=Pin, 
        Pout=Pout, 
        interval_length=interval_length,
        verbose=False
    )
    
    observed_score = result['imhg_score']
    theoretical_pvalue = result['p_value']
    vector = result['vector']
    B = result['B']
    
    print(f"Observed imHG score: {observed_score:.2e}")
    print(f"Theoretical p-value: {theoretical_pvalue:.2e}")
    
    # Run permutations in parallel to get null distribution of imHG scores
    print(f"\nRunning {n_permutations} permutations...")
    
    worker_func = partial(
        _compute_permutation_score,
        vector=vector,
        N=N,
        B=B
    )
    
    with Pool(processes=n_jobs) as pool:
        permutation_scores = list(tqdm(
            pool.imap(worker_func, range(n_permutations)),
            total=n_permutations,
            desc="Permutations"
        ))
    
    permutation_scores = np.array([s for s in permutation_scores if not np.isnan(s)])
    
    # Compute empirical p-value
    empirical_pvalue = (np.sum(permutation_scores <= observed_score) + 1) / (len(permutation_scores) + 1)
    
    print(f"\nPermutations complete!")
    print(f"Empirical p-value: {empirical_pvalue:.2e}")
    print(f"Theoretical p-value: {theoretical_pvalue:.2e}")
    print(f"Ratio (theoretical/empirical): {theoretical_pvalue/empirical_pvalue:.2f}")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot histogram of permutation scores (log scale for better visualization)
    log_perm_scores = np.log10(permutation_scores + 1e-300)  # Add small value to avoid log(0)
    log_observed = np.log10(observed_score + 1e-300)
    
    ax.hist(log_perm_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue',
            label=f'Permutation scores (n={len(permutation_scores)})')
    
    # Add vertical line for observed score
    ax.axvline(x=log_observed, color='red', linestyle='--', linewidth=2.5,
               label=f'Observed score = {observed_score:.2e}')
    
    ax.set_xlabel('log₁₀(imHG score)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Figure 1B: Permutation Test - Null Distribution vs Observed', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with p-values
    textstr = f'Empirical p-value: {empirical_pvalue:.2e}\nTheoretical p-value: {theoretical_pvalue:.2e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'fig1b_pvalue_tightness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1b_pvalue_tightness.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()
    
    return {
        'observed_score': observed_score,
        'permutation_scores': permutation_scores,
        'theoretical_pvalue': theoretical_pvalue,
        'empirical_pvalue': empirical_pvalue,
        'output_path': output_path
    }


def generate_power_analysis_figure(
    N=1000,
    Pout=0.1,
    Pin_values=None,
    interval_lengths=None,
    n_simulations=100,
    alpha=0.0001,
    output_dir='output',
    seed=None,
    n_jobs=None
):
    """
    Generate Figure 1A: Power analysis showing probability of detecting a significant
    interval (p < 0.05) as a function of signal strength (density of 1s within the 
    target interval).
    
    Parameters:
    -----------
    N : int
        Vector length (default=1000)
    Pout : float
        Probability of 1 outside the interval (default=0.1)
    Pin_values : list of float
        Signal strengths (densities) to test. Default: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    interval_lengths : list of int
        Interval lengths to test. Default: [25, 50, 75, 100]
    n_simulations : int
        Number of simulations per condition (default=100)
    alpha : float
        Significance threshold (default=0.05)
    output_dir : str
        Output directory for saving the figure (default='output')
    seed : int, optional
        Random seed for reproducibility
    n_jobs : int, optional
        Number of parallel jobs. Default: number of CPU cores
        
    Returns:
    --------
    dict : Dictionary containing power values for each condition
    """
    if Pin_values is None:
        Pin_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if interval_lengths is None:
        interval_lengths = [25, 50, 75, 100]
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Using {n_jobs} parallel workers")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store power results
    power_results = {length: [] for length in interval_lengths}
    
    total_conditions = len(interval_lengths) * len(Pin_values)
    current_condition = 0
    
    for interval_length in interval_lengths:
        for Pin in Pin_values:
            current_condition += 1
            print(f"Running condition {current_condition}/{total_conditions}: "
                  f"interval_length={interval_length}, Pin={Pin}")
            
            # Skip if Pin <= Pout (invalid condition)
            if Pin <= Pout:
                power_results[interval_length].append(np.nan)
                continue
            
            # Run n_simulations in parallel and count significant detections
            worker_func = partial(
                _run_single_simulation,
                N=N,
                Pin=Pin,
                Pout=Pout,
                interval_length=interval_length,
                alpha=alpha
            )
            
            with Pool(processes=n_jobs) as pool:
                results = pool.map(worker_func, range(n_simulations))
            
            n_significant = sum(results)
            power = n_significant / n_simulations
            power_results[interval_length].append(power)
            print(f"  Power: {power:.3f} ({n_significant}/{n_simulations} significant)")
    
    # Create the figure
    plt.figure(figsize=(10, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(interval_lengths)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for idx, interval_length in enumerate(interval_lengths):
        plt.plot(
            Pin_values, 
            power_results[interval_length],
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=8,
            label=f'Interval length = {interval_length}'
        )
    
    plt.xlabel('Signal Strength (Pin - density of 1s in interval)', fontsize=12)
    plt.ylabel(f'Power (P(p-value < {alpha}))', fontsize=12)
    plt.title('Figure 1A: Detection Power vs Signal Strength', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([min(Pin_values) - 0.05, max(Pin_values) + 0.05])
    plt.ylim([-0.05, 1.05])
    
    # Save the figure
    output_path = os.path.join(output_dir, 'fig1a_power_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1a_power_analysis.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()
    
    return {
        'Pin_values': Pin_values,
        'interval_lengths': interval_lengths,
        'power_results': power_results,
        'output_path': output_path
    }


def main():
    parser = argparse.ArgumentParser(description='Simulation experiment for imHG calculation')
    parser.add_argument('--N', type=int, default=1000, help='Vector length (default: 1000)')
    parser.add_argument('--Pin', type=float, default=0.7, 
                        help='Probability of 1 inside the interval (default: 0.7)')
    parser.add_argument('--Pout', type=float, default=0.1, 
                        help='Probability of 1 outside the interval (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    parser.add_argument('--power-analysis', action='store_true',
                        help='Run power analysis and generate Figure 1A')
    parser.add_argument('--pvalue-tightness', action='store_true',
                        help='Run p-value tightness analysis and generate Figure 1B')
    parser.add_argument('--n-simulations', type=int, default=100,
                        help='Number of simulations per condition for power analysis (default: 100)')
    parser.add_argument('--n-permutations', type=int, default=10000,
                        help='Number of Monte Carlo permutations for p-value tightness (default: 10000)')
    parser.add_argument('--interval-length', type=int, default=75,
                        help='Interval length for p-value tightness analysis (default: 75)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance threshold for power analysis (default: 0.05)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for figures (default: output)')
    
    args = parser.parse_args()
    
    if args.power_analysis:
        print(f"Running power analysis with {args.n_simulations} simulations per condition...")
        generate_power_analysis_figure(
            N=args.N,
            Pout=args.Pout,
            n_simulations=args.n_simulations,
            alpha=args.alpha,
            output_dir=args.output_dir,
            seed=args.seed,
            n_jobs=args.n_jobs
        )
    elif args.pvalue_tightness:
        print(f"Running p-value tightness analysis with {args.n_permutations} permutations...")
        generate_pvalue_tightness_figure(
            N=args.N,
            Pout=args.Pout,
            Pin=args.Pin,
            interval_length=args.interval_length,
            n_permutations=args.n_permutations,
            output_dir=args.output_dir,
            seed=args.seed,
            n_jobs=args.n_jobs
        )
    else:
        print(f"Running simulation with N={args.N}, Pin={args.Pin}, Pout={args.Pout}")
        
        results = run_simulation(N=args.N, Pin=args.Pin, Pout=args.Pout, seed=args.seed)
        
        print(f"\n=== Simulation Results ===")
        print(f"Vector length (N): {results['N']}")
        print(f"Number of 1s (B): {results['B']}")
        print(f"True interval: [{results['true_interval'][0]}, {results['true_interval'][1]}] "
              f"(length: {results['true_interval_length']})")
        print(f"Detected interval: [{results['detected_interval'][0]}, {results['detected_interval'][1]}] "
              f"(length: {results['detected_interval_length']})")
        print(f"imHG score: {results['imhg_score']}")
        print(f"P-value <= {results['p_value']}")


if __name__ == '__main__':
    main()
