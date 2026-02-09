"""
Test script for xw/refactor branch FPGA placement using CyclicExpansion.

This script solves the same placement problem as test_refactor_branch_qubo.py
but uses the CyclicExpansion algorithm (arXiv:2312.15467) which iteratively
solves small 2-cycle swap sub-problems instead of one large QUBO.

2-cycles naturally preserve permutation feasibility — no constraint weights needed.
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    Legalizer,
    Router,
    PlacementDrawer,
    solve_placement_cyclic,
)
from fem_placer.utils import parse_fpga_design

# Configuration
max_iters = 200
k = 60          # instances per iteration
k_u = 30        # unbound sites per iteration
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dcp_file = './vivado/output_dir/post_impl.dcp'
output_file = 'optimized_placement_cyclic.dcp'

# Initialize FPGA placer
print("=" * 80)
print("Testing xw/refactor Branch FPGA Placement (CyclicExpansion)")
print("=" * 80)

print("\nINFO: Initializing FPGA placer...")
fpga_placer = FpgaPlacer(utilization_factor=0.3)

print(f"INFO: Loading FPGA design from {dcp_file}...")
fpga_placer.init_placement(
    dcp_file=dcp_file,
    dcp_output=output_file
)

# Parse FPGA design to get coupling matrix
print("\nINFO: Parsing FPGA design...")
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)

# Get layout dimensions
area_length = fpga_placer.bbox['area_length']
area_width = fpga_placer.bbox['area_width']
area_size = fpga_placer.bbox['area_size']

print(f"INFO: Number of instances: {num_inst}")
print(f"INFO: Number of sites: {num_site}")
print(f"INFO: Area dimensions: {area_length} x {area_width} = {area_size}")
print(f"INFO: Grid size: {area_length} x {area_length} = {area_length ** 2} positions")

# Create site coordinates matrix
print("\nINFO: Creating site coordinates matrix for entire grid...")
logic_site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)
print(f"INFO: Site coordinates matrix shape: {logic_site_coords.shape}")

print(f"\nINFO: CyclicExpansion params: k={k}, k_u={k_u}, max_iters={max_iters}")

# Solve with CyclicExpansion
print("\nINFO: Solving placement with CyclicExpansion...")
site_indices, grid_coords, energy, meta = solve_placement_cyclic(
    J, logic_site_coords,
    k=k, k_u=k_u, max_iters=max_iters,
    seed=42, verbose=True,
)

# Check feasibility (CyclicExpansion guarantees unique sites by construction)
n_unique = len(torch.unique(site_indices))
print(f"\nINFO: Unique sites used: {n_unique} / {num_inst} instances")
if n_unique < num_inst:
    print(f"WARNING: Only {n_unique} distinct sites — {num_inst - n_unique} instances overlap")

print(f"\nINFO: QAP cost: {energy:.4f}")
print(f"INFO: Iterations: {meta['iterations']}")
print(f"INFO: Grid coordinates shape: {grid_coords.shape}")

# Convert to real FPGA coordinates
print("INFO: Converting to real FPGA coordinates...")
logic_grid = fpga_placer.get_grid('logic')
coords = logic_grid.to_real_coords_tensor(grid_coords)
print(f"INFO: Real coordinates shape: {coords.shape}")
print(f"DEBUG: First 10 real coords: {coords[:10]}")
print(f"DEBUG: Min/Max coords: ({coords.min(dim=0).values}, {coords.max(dim=0).values})")

# Set up visualization
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Legalize placement
print("\nINFO: Legalizing placement...")
legalizer = Legalizer(placer=fpga_placer, device=dev)
logic_ids = torch.arange(num_inst)

placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    coords, logic_ids,
    io_coords=None,
    io_ids=None,
    include_io=False
)

print(f"INFO: Overlaps resolved: {overlap}")
print(f"INFO: HPWL before legalization: {hpwl_before}")
print(f"INFO: HPWL after legalization: {hpwl_after}")

# Route connections
print("\nINFO: Routing connections...")
router = Router(fpga_placer.bbox)
routes = router.route_connections(J, placement_legalized[0])

# Visualize final placement
print("\nINFO: Generating visualization...")
drawer.draw_place_and_route(
    logic_coords=placement_legalized[0],
    routes=routes,
    io_coords=None,
    include_io=False,
    iteration=0,
    title_suffix='xw/refactor Branch - CyclicExpansion - Final'
)

print("\n" + "=" * 80)
print("xw/refactor Branch FPGA Placement Test (CyclicExpansion) Complete!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  - QAP cost: {energy:.4f}")
print(f"  - Iterations: {meta['iterations']}")
print(f"  - HPWL before legalization: {hpwl_before}")
print(f"  - HPWL after legalization: {hpwl_after}")
print(f"  - Overlaps resolved: {overlap}")
print(f"  - Output DCP: {output_file}")
