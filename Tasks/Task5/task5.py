import sys
import numpy as np
from os.path import join
from multiprocessing import Pool, cpu_count


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

    # Static scheduling: split building_ids into chunks for each worker
    num_workers = min(cpu_count(), len(building_ids))
    k, m = divmod(len(building_ids), num_workers) # k = chunk size, m = remainder
    
    # Create chunks of building_ids for each worker
    chunks = [building_ids[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_workers)]
    args = [(chunk, MAX_ITER, ABS_TOL, LOAD_DIR) for chunk in chunks]

    # Process floor plans in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_floorplans, args)
        
    # Flatten results and print CSV
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for chunk in results:
        for bid, stats in chunk:
            print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))