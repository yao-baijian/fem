"""
Test script for FPGA placement using discrete Simulated Bifurcation (dSB).

This script solves the placement problem by constructing a full QUBO matrix
with one-hot and at-most-one constraints, then solving it with the
simulated-bifurcation library (heated discrete mode).

Uses the master branch FpgaPlacer API (net_manager for coupling matrix).
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    Legalizer,
    Router,
    PlacementDrawer,
    solve_placement_sb,
    decode_qubo_solution,
)

# Configuration
agents = 128
max_steps = 10000
lam = 50.0        # one-hot constraint weight
mu = 50.0         # at-most-one constraint weight
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dcp_file = './vivado/output_dir/post_impl.dcp'
output_file = 'optimized_placement_dsb.dcp'

# Initialize FPGA placer
print("=" * 80)
print("Testing FPGA Placement (dSB)")
print("=" * 80)

print("\nINFO: Initializing FPGA placer...")
fpga_placer = FpgaPlacer(utilization_factor=0.3)

print(f"INFO: Loading FPGA design from {dcp_file}...")
vivado_hpwl, opti_insts_num, site_net_num, total_net_num = fpga_placer.init_placement(
    dcp_file=dcp_file,
    dcp_output=output_file
)

print(f"INFO: Vivado HPWL: {vivado_hpwl}")
print(f"INFO: Optimizable instances: {opti_insts_num}")
print(f"INFO: Site net number: {site_net_num}")
print(f"INFO: Total net number: {total_net_num}")

# Get coupling matrix from net_manager
J = fpga_placer.net_manager.insts_matrix
num_inst = fpga_placer.opti_insts_num

# Get grid information
logic_grid = fpga_placer.get_grid('logic')
print(f"\nINFO: Logic grid size: {logic_grid.width} x {logic_grid.height} = {logic_grid.area}")
print(f"INFO: Number of instances: {num_inst}")

# Create site coordinates matrix
print("\nINFO: Creating site coordinates matrix for entire grid...")
logic_site_coords = torch.cartesian_prod(
    torch.arange(logic_grid.width, dtype=torch.float32),
    torch.arange(logic_grid.height, dtype=torch.float32)
)
print(f"INFO: Site coordinates matrix shape: {logic_site_coords.shape}")

# QUBO size warning
n_sites = logic_site_coords.shape[0]
qubo_dim = num_inst * n_sites + n_sites
qubo_memory_gb = qubo_dim ** 2 * 4 / 1024 ** 3
print(f"\nINFO: QUBO matrix size: {qubo_dim} x {qubo_dim}")
print(f"INFO: QUBO memory: ~{qubo_memory_gb:.2f} GB")

print(f"\nINFO: dSB params: agents={agents}, max_steps={max_steps}, lam={lam}, mu={mu}")

# Solve with dSB
print("\nINFO: Solving placement with dSB...")
site_indices, grid_coords, energy, meta = solve_placement_sb(
    J, logic_site_coords,
    lam=lam, mu=mu,
    agents=agents, max_steps=max_steps,
    best_only=True,
)

# Check feasibility
n_unique = len(torch.unique(site_indices))
print(f"\nINFO: Unique sites used: {n_unique} / {num_inst} instances")
if n_unique < num_inst:
    print(f"WARNING: Only {n_unique} distinct sites — {num_inst - n_unique} instances overlap")

print(f"\nINFO: QUBO energy: {energy:.4f}")
print(f"INFO: Grid coordinates shape: {grid_coords.shape}")

# Convert to real FPGA coordinates
print("INFO: Converting to real FPGA coordinates...")
coords = logic_grid.to_real_coords_tensor(grid_coords)
print(f"INFO: Real coordinates shape: {coords.shape}")
print(f"DEBUG: First 10 real coords: {coords[:10]}")
print(f"DEBUG: Min/Max coords: ({coords.min(dim=0).values}, {coords.max(dim=0).values})")

# Set up visualization
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Get instance IDs
logic_ids, io_ids = fpga_placer.get_ids()

# Legalize placement
print("\nINFO: Legalizing placement...")
legalizer = Legalizer(placer=fpga_placer, device=dev)

if fpga_placer.with_io():
    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        coords, logic_ids,
        io_coords=None, io_ids=io_ids,
        include_io=True
    )
else:
    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        coords, logic_ids
    )

print(f"INFO: Overlaps resolved: {overlap}")
print(f"INFO: HPWL before legalization: {hpwl_before}")
print(f"INFO: HPWL after legalization: {hpwl_after}")

# Route connections
print("\nINFO: Routing connections...")
router = Router(placer=fpga_placer)
all_coords = torch.cat([placement_legalized[0], placement_legalized[1]], dim=0) \
    if fpga_placer.with_io() else placement_legalized[0]
routes = router.route_connections(fpga_placer.net_manager.insts_matrix, all_coords)

# Visualize final placement
print("\nINFO: Generating visualization...")
drawer.draw_place_and_route(
    placement_legalized[0],
    routes,
    io_coords=placement_legalized[1] if fpga_placer.with_io() else None,
    include_io=fpga_placer.with_io(),
    title_suffix='dSB - Final'
)

print("\n" + "=" * 80)
print("FPGA Placement Test (dSB) Complete!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  - QUBO energy: {energy:.4f}")
print(f"  - HPWL before legalization: {hpwl_before}")
print(f"  - HPWL after legalization: {hpwl_after}")
print(f"  - Overlaps resolved: {overlap}")
print(f"  - Output DCP: {output_file}")
