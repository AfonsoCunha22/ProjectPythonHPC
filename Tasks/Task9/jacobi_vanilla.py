from os.path import join
import sys
import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

# Function to repeatedly update the interior temperature values of a grid by averaging values of the four neighbors
# u is a 2d array with current temperature values of a specific floorplan grid, interior mask: 2d boolean array where True is where we should update
# max iterations to run and atol to stop early if updates get smalelr than this value
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    # creates a copy of the grid
    u = np.copy(u)

    for i in range(max_iter):
        # For each element in the floorplan grid, take the value to the left, right, up and down and divide by 4.
        u_new = 0.25 * (
            u[1:-1, :-2] +  # left
            u[1:-1, 2:] +   # right
            u[:-2, 1:-1] +  # up
            u[2:, 1:-1]     # down
        )
        # Apply the mask to u_new, which returns an array reflecting u_new only with the masked values labelled true
        u_new_interior = u_new[interior_mask]
        # Compares the previous iteration's values with the current one and compares each element's difference in value
        # and returns the element with the biggest change in value
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        # Update the interior grid to the new values
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        # Go out of the loop if we have converged 
        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # Opens file called building_ids.txt, containing list of building IDs, one per line
    # Output will be something like building_ids = ['10000', '10334', '10786', ...]
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # Default N = 1
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    
    # Creates a list with the first N building IDs
    building_ids = building_ids[:N]

    # Load floor plans, Store the initial temperature grids so we store N number of 514 x 514 grids 
    # because the 512 x 512 simulation is padded by 1 on each side
    all_u0 = np.empty((N, 514, 514))
    # Stores the interior masks, which highlight the outsides and the walls
    all_interior_mask = np.empty((N, 512, 512), dtype=bool)
    # Loop over building IDs
    for i, bid in enumerate(building_ids):
        # Load the two npy files for building floors
        # u0 514 x 514 array grid with temperatures like 25 (inside walls), 5 (load bearing walls), 0 (interior space)
        # interior_mask is where to update the values
        u0, interior_mask = load_data(LOAD_DIR, bid)
        # Holds the simulation grid for building i
        all_u0[i] = u0
        # holds the mask for building i
        all_interior_mask[i] = interior_mask

    # Max number of iterations before it stops
    MAX_ITER = 20_000
    # Absolute total: Stop early if difference between iterations gets smaller than this value
    ABS_TOL = 1e-4
    # Store results of jacobi of each floor plan in all_u
    all_u = np.empty_like(all_u0)

    # Loop through each floorplan and its subsequent masks and run the jacobi algorithm to calculate temperatures of each element in grid
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # Print CSV results
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
