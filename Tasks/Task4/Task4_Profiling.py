from simulate import load_data, jacobi

load_dir = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
building_id = "10000"

u, interior_mask = load_data(load_dir, building_id)
jacobi(u, interior_mask, max_iter=500, atol=1e-6)
