from os.path import join
import sys
import numpy as np
from numba import jit


def load_data(load_dir, bid):
    ''' Load the floor plan and interior mask for a given building ID. '''
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter, atol=1e-6):
    '''Jacobi iteration for solving the heat equation using Numba JIT on the CPU.'''
    SIZE = u.shape[0] - 2 # Excluding the boundary
    for it in range(max_iter):
        delta = 0.0
        u_new = u.copy()
        for i in range(1, SIZE + 1):
            for j in range(1, SIZE + 1):
                if interior_mask[i-1, j-1]:
                    val = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                    diff = abs(u[i, j] - val)
                    if diff > delta:
                        delta = diff
                    u_new[i, j] = val
        u = u_new
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


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Dummy run to compile the JIT function
    dummy_u = np.zeros((514, 514))
    dummy_mask = np.zeros((512, 512), dtype='bool')
    jacobi_jit(dummy_u, dummy_mask, 1, 1e-4)

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_jit(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))