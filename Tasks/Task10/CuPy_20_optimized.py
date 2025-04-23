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
    # cp.asarray makes sure we transfer the temperature grid from CPU memory to GPU memory, we also convert to cp.float32 because we didn't need float64 saving space
    u[1:-1,1:-1] = cp.asarray(domain, dtype=cp.float32)
    # Load interior mask on CPU, then transfer to GPU
    interior_mask = cp.asarray(np.load(join(load_dir, f"{bid}_interior.npy")), dtype=bool)
    # u and interior mask are now CuPy arrays on the GPU, so what gets returned are two Python Objects cupy.ndarray 
    # which contains pointers to data in GPU memory and hte object itself like metadata, shape, dtype is still managed in CPU memory
    # The data itself is handled on the GPU
    return u, interior_mask

def jacobi_gpu(u, interior_mask, max_iter, atol=1e-4, check_every=100):
    """
    - 1 CuPy kernel to compute u_new
    - 1 CuPy kernel to merge u_new back into u (only interior points) via cp.where
    - 1 D→H reduction every `check_every` iters instead of every iter
    """
    u = u.copy()
    mask = interior_mask  # shape (512,512)
    # pad mask to 514×514 so we can apply it directly to the padded view
    mask_pad = cp.pad(mask, pad_width=1, constant_values=False)

    for i in range(max_iter):
        # 1) compute neighbor average in one kernel
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +
            u[:-2, 1:-1] + u[2:, 1:-1]
        )

        # 2) write back only interior points in one kernel
        #    u_inner = u[1:-1,1:-1];  u_inner[mask] = u_new[mask]
        #  → use cp.where on the padded view for a single fused kernel
        interior_view = u[1:-1, 1:-1]  # view of shape (512,512)
        # cp.where broadcasts over same‐shape arrays:
        interior_view = cp.where(mask, u_new, interior_view)
        u[1:-1, 1:-1] = interior_view

        # 3) every `check_every` iterations, do one reduction + host sync
        if (i % check_every) == 0:
            # get the max change on the GPU, then one D→H copy
            delta = float(cp.max(cp.abs(interior_view - u_new)).get())
            if delta < atol:
                break

    return u

def summary_stats_gpu(u, interior_mask):
    # pull back only the interior values one final time
    u_cpu = cp.asnumpy(u[1:-1,1:-1][interior_mask])
    return {
        'mean_temp':   float(u_cpu.mean()),
        'std_temp':    float(u_cpu.std()),
        'pct_above_18': float((u_cpu>18).sum() / u_cpu.size * 100),
        'pct_below_15': float((u_cpu<15).sum() / u_cpu.size * 100),
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # read IDs
    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        building_ids = f.read().splitlines()
    N = int(sys.argv[1]) if len(sys.argv)>1 else 1
    building_ids = building_ids[:N]

    # load & transfer all plans
    all_u0 = []
    all_masks = []
    for bid in building_ids:
        u0, mask = load_data_gpu(LOAD_DIR, bid)
        all_u0.append(u0)
        all_masks.append(mask)

    # run
    MAX_ITER = 20_000
    ABS_TOL  = 1e-4
    results = []
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for u0, mask in zip(all_u0, all_masks):
        u_final = jacobi_gpu(u0, mask, MAX_ITER, atol=ABS_TOL, check_every=100)
        results.append((u_final, mask))
    end.record(); end.synchronize()
    print(f"GPU time: {cp.cuda.get_elapsed_time(start,end)/1000:.3f}s for {N}")

    # print CSV
    print("building_id,mean,std,%>18,%<15")
    for bid,(u,mask) in zip(building_ids, results):
        stats = summary_stats_gpu(u, mask)
        print(f"{bid},{stats['mean_temp']:.4f},{stats['std_temp']:.4f},"
              f"{stats['pct_above_18']:.1f},{stats['pct_below_15']:.1f}")
