import numpy as np
import argparse
from imhg_calculator import imHGCalculator


def run_simulation(N=1000, Pin=0.7, Pout=0.1, seed=None):
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
    seed : int, optional
        Random seed for reproducibility
    
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
    print(f"Drew interval start s = {s}")
    
    # Draw e uniformly from s+50 to s+100
    e = np.random.randint(s + 50, s + 101)  # +101 because randint is exclusive on upper bound
    print(f"Drew interval end e = {e}")
    
    # Ensure e doesn't exceed N-1
    e = min(e, N - 1)
    print(f"Adjusted e to {e} (capped at N-1)")
    
    # Create a zero vector of size N
    vector = np.zeros(N, dtype=int)
    print(f"Created zero vector of size {N}")
    
    # Inside the interval [s, e], flip each 0 to 1 with probability Pin
    for i in range(s, e + 1):
        if np.random.random() < Pin:
            vector[i] = 1
    ones_inside = int(np.sum(vector[s:e+1]))
    print(f"Flipped {ones_inside} positions to 1 inside interval [{s}, {e}] (Pin={Pin})")
    
    # Outside the interval [s, e], flip each 0 to 1 with probability Pout
    for i in range(0, s):
        if np.random.random() < Pout:
            vector[i] = 1
    for i in range(e + 1, N):
        if np.random.random() < Pout:
            vector[i] = 1
    ones_outside = int(np.sum(vector)) - ones_inside
    print(f"Flipped {ones_outside} positions to 1 outside interval (Pout={Pout})")
    
    # Calculate B (number of active genes / 1s in the vector)
    B = int(np.sum(vector))
    print(f"Total number of 1s (B) = {B}")
    
    # Convert vector to tuple for imHG calculation
    lamda = tuple(vector)
    
    # Run imHG calculator
    print("Running imHG calculator...")
    calculator = imHGCalculator()
    imhg_score, (start_idx, end_idx) = calculator.calculate_imhg(N, B, lamda)
    print(f"imHG calculation complete. Score = {imhg_score}, interval = [{start_idx}, {end_idx}]")
    
    # Calculate p-value
    print("Calculating p-value...")
    p_value = calculator.calculate_p_value(N, B, imhg_score)
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


def main():
    parser = argparse.ArgumentParser(description='Simulation experiment for imHG calculation')
    parser.add_argument('--N', type=int, default=1000, help='Vector length (default: 1000)')
    parser.add_argument('--Pin', type=float, default=0.7, 
                        help='Probability of 1 inside the interval (default: 0.7)')
    parser.add_argument('--Pout', type=float, default=0.1, 
                        help='Probability of 1 outside the interval (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
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
    print(f"P-value: {results['p_value']}")


if __name__ == '__main__':
    main()
