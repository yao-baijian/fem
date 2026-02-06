"""
Smoke test: export full QUBO matrix and solve with SBSolver.

Uses the same problem setup as test_refactor_branch_qubo.py but routes
through export_placement_qubo -> SBSolver.solve_qubo -> decode_qubo_solution.
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    SBSolver,
)
from fem_placer.objectives import (
    export_placement_qubo,
    decode_qubo_solution,
    get_site_distance_matrix,
)
from fem_placer.utils import parse_fpga_design

# Configuration - matches test_refactor_branch_qubo.py
dev = 'cpu'
dcp_file = './vivado/output_dir/post_impl.dcp'
output_file = 'optimized_placement_sb_qubo.dcp'

# =============================================================================
# 1. Load design (same as test_refactor_branch_qubo.py)
# =============================================================================
print("=" * 80)
print("Smoke Test: export_placement_qubo -> SBSolver -> decode_qubo_solution")
print("=" * 80)

print("\nINFO: Initializing FPGA placer...")
fpga_placer = FpgaPlacer(utilization_factor=0.3)

print(f"INFO: Loading FPGA design from {dcp_file}...")
fpga_placer.init_placement(dcp_file=dcp_file, dcp_output=output_file)

print("\nINFO: Parsing FPGA design...")
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)

area_length = fpga_placer.bbox['area_length']
area_width = fpga_placer.bbox['area_width']

print(f"INFO: Number of instances (m): {num_inst}")
print(f"INFO: Number of sites (n): {area_length ** 2}")
print(f"INFO: Area dimensions: {area_length} x {area_length}")

# Site coordinates - same grid as test_refactor_branch_qubo.py
logic_site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32),
)
n = logic_site_coords.shape[0]

print(f"INFO: QUBO variable count: m*n + n = {num_inst * n + n}")
print(f"INFO: Q matrix size: {num_inst * n + n} x {num_inst * n + n}")

# =============================================================================
# 2. Export full QUBO
# =============================================================================
print("\nINFO: Exporting full QUBO matrix...")

# Use same constraint weight as test_refactor_branch_qubo.py
lam = num_inst / 2.0   # one-hot constraint weight
mu = num_inst / 2.0     # at-most-one constraint weight

Q, metadata = export_placement_qubo(J, logic_site_coords, lam=lam, mu=mu)
m, n = metadata['m'], metadata['n']

print(f"INFO: Q shape: {Q.shape}")
print(f"INFO: Q is symmetric: {torch.allclose(Q, Q.T, atol=1e-6)}")
print(f"INFO: Q diagonal range: [{Q.diagonal().min():.2f}, {Q.diagonal().max():.2f}]")
print(f"INFO: Q off-diagonal range: [{Q[~torch.eye(Q.shape[0], dtype=bool)].min():.2f}, "
      f"{Q[~torch.eye(Q.shape[0], dtype=bool)].max():.2f}]")

# =============================================================================
# 3. Solve with SBSolver
# =============================================================================
print("\nINFO: Solving with SBSolver...")
solver = SBSolver(mode='discrete', heated=True, device=dev)

solution, energy = solver.solve_qubo(
    Q,
    agents=10,
    max_steps=10000,
    best_only=True,
)

print(f"INFO: Solution shape: {solution.shape}")
print(f"INFO: QUBO energy: {energy.item():.2f}")
print(f"INFO: Solution binary stats: {solution.sum().item():.0f} ones out of {solution.numel()}")

# =============================================================================
# 4. Decode solution
# =============================================================================
print("\nINFO: Decoding QUBO solution...")
site_indices, coords = decode_qubo_solution(solution, m, n, logic_site_coords)

print(f"INFO: Site indices shape: {site_indices.shape}")
print(f"INFO: Coords shape: {coords.shape}")

# Validate: check one-hot constraint
x = solution[:m * n].reshape(m, n)
row_sums = x.sum(dim=1)
valid_onehot = (row_sums == 1).all()
print(f"INFO: One-hot constraint satisfied: {valid_onehot.item()}")
if not valid_onehot:
    print(f"  WARNING: Row sums: min={row_sums.min():.0f}, max={row_sums.max():.0f}, "
          f"mean={row_sums.float().mean():.2f}")
    # Count violations
    violations = (row_sums != 1).sum().item()
    print(f"  WARNING: {violations}/{m} instances violate one-hot constraint")

# Check site uniqueness
col_sums = x.sum(dim=0)
max_overlap = col_sums.max().item()
print(f"INFO: Max site overlap: {max_overlap:.0f}")

# Compute actual HPWL for the decoded placement
D = get_site_distance_matrix(logic_site_coords)
hpwl = 0.0
for i in range(m):
    for j in range(i + 1, m):
        if J[i, j] > 0:
            hpwl += J[i, j] * D[site_indices[i], site_indices[j]]
print(f"INFO: Decoded placement HPWL: {hpwl.item():.2f}")

# =============================================================================
# 5. Convert to real FPGA coordinates and legalize
# =============================================================================
print("\nINFO: Converting to real FPGA coordinates...")
logic_grid = fpga_placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(coords)
print(f"INFO: Real coordinates shape: {real_coords.shape}")

print("\nINFO: Legalizing placement...")
legalizer = Legalizer(placer=fpga_placer, device=dev)
logic_ids = torch.arange(num_inst)

placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    real_coords, logic_ids,
    io_coords=None,
    io_ids=None,
    include_io=False
)

print(f"INFO: Overlaps resolved: {overlap}")
print(f"INFO: HPWL before legalization: {hpwl_before}")
print(f"INFO: HPWL after legalization: {hpwl_after}")

# =============================================================================
# 6. Route and visualize
# =============================================================================
print("\nINFO: Routing connections...")
router = Router(fpga_placer.bbox)
routes = router.route_connections(J, placement_legalized[0])

drawer = PlacementDrawer(placer=fpga_placer, num_subplots=2, debug_mode=False)
drawer.draw_place_and_route(
    logic_coords=placement_legalized[0],
    routes=routes,
    io_coords=None,
    include_io=False,
    iteration=0,
    title_suffix='SB QUBO Export - Final'
)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Smoke Test Complete!")
print("=" * 80)
print(f"\nResults:")
print(f"  - QUBO energy (z^T Q z):     {energy.item():.2f}")
print(f"  - Decoded HPWL:              {hpwl.item():.2f}")
print(f"  - HPWL before legalization:  {hpwl_before}")
print(f"  - HPWL after legalization:   {hpwl_after}")
print(f"  - Overlaps resolved:         {overlap}")
print(f"  - One-hot satisfied:         {valid_onehot.item()}")
