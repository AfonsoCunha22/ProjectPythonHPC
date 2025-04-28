import sys
import numpy as np
from os.path import join
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt


def load_data(load_dir, bid):
    ''' Load the floor plan and interior mask for a given building ID. '''
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    '''Jacobi iteration for solving the heat equation.'''
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    '''Performs summary statistics on the interior of the domain.'''
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

def process_floorplans(args):
    '''Process a list of floor plans.'''
    bids, max_iter, abs_tol, load_dir = args
    results = []
    for bid in bids:
        u0, interior_mask = load_data(load_dir, bid)
        u = jacobi(u0, interior_mask, max_iter, abs_tol)
        stats = summary_stats(u, interior_mask)
        results.append((bid, stats))
    return results

if __name__ == '__main__':
    t0 = time.perf_counter() # Start timer for serial execution
    
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
        
    # Limit the number of floorplans to process to 100
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:min(N, 100)]

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Measurement of speed-up
    max_workers = min(cpu_count(), len(building_ids))
    times_summary = []
    worker_counts = list(range(1, max_workers + 1))
    t1 = time.perf_counter() # End timer for serial execution

    for num_workers in worker_counts:
        t2 = time.perf_counter() # Start timer for serial execution of each worker count
        k, m = divmod(len(building_ids), num_workers)
        chunks = [building_ids[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_workers)]
        args = [(chunk, MAX_ITER, ABS_TOL, LOAD_DIR) for chunk in chunks]
        t3 = time.perf_counter() # End timer for serial execution each worker count
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_floorplans, args)
        t4 = time.perf_counter()
        
        serial_time = (t3 - t2) + (t1 - t0) # Time taken for serial execution
        parallel_time = t4 - t3 # Time taken for parallel execution
        total_time = (t4 - t2) + (t1 - t0) # Total time taken for execution
        
        times_summary.append({
            'workers': num_workers,
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'total_time': total_time
        })
        
    #Estimating parallel fraction using Amdahl's law 
    print("\n--- Benchmark Results ---")
    T_1 = times_summary[0]['total_time']
    for entry in times_summary:
        p = entry['workers']
        T_p = entry['total_time']
        speedup = T_1 / T_p
        # Estimate parallel fraction F using rearranged Amdahl's Law:
        # speedup = 1 / ((1 - F) + F / p)
        if p > 1:
            F = (speedup - 1) / (speedup * (1 - 1/p))
        else:
            F = 0.0
        print(f"{p} worker(s): Time = {T_p:.2f}s, Speedup = {speedup:.2f}, Estimated Parallel Fraction = {F:.4f}")
            
    #Compute speedups
    speedups = [times_summary[0]['total_time']/entry['total_time'] for entry in times_summary]
    
    # Plot speed-up
    plt.figure()
    plt.plot(worker_counts, speedups, marker='o')
    plt.xticks(worker_counts)
    plt.xlabel('Number of Workers')
    plt.ylabel('Speed-up')
    plt.title('Parallel Speed-up')
    plt.grid(True)
    plt.savefig('speedup_plot_75_123.png')
    plt.show()