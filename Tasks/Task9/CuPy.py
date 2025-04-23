import sys
from os.path import join
import numpy as np
import cupy as cp

def load_data_gpu(load_dir, bid):
    SIZE = 512
    # Create a CuPy array instead of NumPy array for the padded temperature grid
    u = cp.zeros((SIZE + 2, SIZE + 2), dtype=cp.float32)
    # Load domain npy files containing the temperature grid into CPU
    domain = np.load(join(load_dir, f"{bid}_domain.npy"))
    # cp.asarray makes sure we transfer the temperature grid from CPU memory to GPU memory
    u[1:-1, 1:-1] = cp.asarray(domain)
    # Load interior mask on CPU, then transfer to GPU
    interior_mask = cp.asarray(np.load(join(load_dir, f"{bid}_interior.npy")), dtype=bool)
    # u and interior mask are now CuPy arrays on the GPU, so what gets returned are two Python Objects cupy.ndarray 
    # which contains pointers to data in GPU memory and hte object itself like metadata, shape, dtype is still managed in CPU memory
    # The data itself is handled on the GPU
    return u, interior_mask

# GPU-accelerated Jacobi using CuPy
# u is a CuPy array with current temperatures, interior_mask is a CuPy boolean mask
# max_iter and atol as before
def jacobi_gpu(u, interior_mask, max_iter, atol=1e-6):
    # Make a copy on GPU to avoid modifying the original
    u = u.copy()

    for i in range(max_iter):
        # Compute neighbor average entirely on GPU
        u_new = 0.25 * (
            u[1:-1, :-2] +  # left
            u[1:-1, 2:] +   # right
            u[:-2, 1:-1] +  # up
            u[2:, 1:-1]     # down
        )
        # Select only interior points
        u_new_interior = u_new[interior_mask]
        # Compute maximum change for convergence check
        old_vals = u[1:-1, 1:-1][interior_mask]
        delta = cp.max(cp.abs(old_vals - u_new_interior))
        # Update interior values
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        # Synchronize and check convergence on GPU
        if delta < atol:
            break
    return u

# Compute summary statistics by transferring needed data back to CPU
def summary_stats_gpu(u, interior_mask):
    # Python on the CPU instructs u (which is on the GPU) to slice up the array and apply a boolean mask. CuPy translates the instructions
    # to GPU compatible operations
    u_interior = u[1:-1, 1:-1][interior_mask]
    # Transfer to CPU for numpy operations
    u_cpu = cp.asnumpy(u_interior)

    mean_temp = u_cpu.mean()
    std_temp = u_cpu.std()
    pct_above_18 = np.sum(u_cpu > 18) / u_cpu.size * 100
    pct_below_15 = np.sum(u_cpu < 15) / u_cpu.size * 100

    return {
        'mean_temp': float(mean_temp),
        'std_temp': float(std_temp),
        'pct_above_18': float(pct_above_18),
        'pct_below_15': float(pct_below_15),
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # Opens file called building_ids.txt, containing list of building IDs, one per line
    # Output will be something like building_ids = ['10000', '10334', '10786', ...]
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # Default to 1 building if no argument is given
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    # Creates a list with the first N building IDs
    building_ids = building_ids[:N]

    # DIFFERENT: instead of creating pre-sized arrays via np.empty() we instead dynamically expand arrays
    # This is because GPUs typically have less memory than CPUs and Numpy -> CuPy transfers happen ONE ARRAY AT A TIME.
    # So even if we did allocate the entire 3d building data to the GPU to prepare for workload, we can still only transfer 1 floorplan at a time
    # Preallocate GPU data arrays for initial grids and masks
    all_u0 = []
    all_masks = []

    # We iterate again through the building_ids
    for bid in building_ids:
        # We load the temperature grid and the corresponding masks and transfer it from the CPU memory to the GPU memory
        u0, mask = load_data_gpu(LOAD_DIR, bid)
        # Store each cupy array objects which are pointers to the grid temperature and mask 2d arrays on the GPU
        all_u0.append(u0)
        all_masks.append(mask)

    MAX_ITER = 20000
    ABS_TOL = 1e-4

    # Run Jacobi on GPU for each floorplan
    results = []
    # start and end are CUDA events that work like GPU stopwatches
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    # Tells CUDA to start the timer
    start.record()
    # Go through each pair of GPU arrays
    for u0, mask in zip(all_u0, all_masks):
        # Get the final temperature grid for each instance
        u_final = jacobi_gpu(u0, mask, MAX_ITER, ABS_TOL)
        # Add the final temperature grid and mask to the results array, which contains CuPy objects with pointers to the grids on the GPU
        results.append((u_final, mask))
    end.record()
    # Forces the CPU to wait until the GPU has finished ALL its operations
    end.synchronize()
    gpu_time_ms = cp.cuda.get_elapsed_time(start, end)

    # Print timing comparison header
    print(f"GPU elapsed time: {gpu_time_ms/1000:.3f} seconds for {N} buildings")

    # Print CSV results
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    # 
    for bid, (u_final, mask) in zip(building_ids, results):
        stats = summary_stats_gpu(u_final, mask)
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
