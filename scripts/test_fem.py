"""
Test script for FPGA placement using QUBO + FEM optimization.

This script tests the FPGA placement implementation using the QUBO
(Quadratic Unconstrained Binary Optimization) formulation with FPGAPlacementOptimizer.

Uses the master branch FpgaPlacer API (net_manager for coupling matrix).
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
    Legalizer,
    Router,
    PlacementDrawer
)

# Configuration
num_trials = 10
num_steps = 2000
dev = 'cpu'
dcp_file = './vivado/output_dir/post_impl.dcp'
output_file = 'optimized_placement_qubo.dcp'

# Initialize FPGA placer
print("=" * 80)
print("Testing FPGA Placement (QUBO Algorithm)")
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
num_fixed = fpga_placer.fixed_insts_num

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
print(f"INFO: Grid covers all {logic_grid.width}x{logic_grid.height}={logic_site_coords.shape[0]} positions")

# Memory comparison
qubo_memory_mb = num_trials * num_inst * logic_grid.area * 4 / 1024 / 1024
print(f"\nMemory usage:")
print(f"  QUBO distribution: ~{qubo_memory_mb:.2f} MB")

# Set up visualization
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Create QUBO optimizer
print("\nINFO: Setting up QUBO optimizer...")
constraint_alpha = num_inst * 7.0
print(f"INFO: Constraint weight (alpha): {constraint_alpha}")

optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_fixed_inst=num_fixed,
    num_site=logic_site_coords.shape[0],
    num_fixed_site=fpga_placer.grids['io'].area_width,
    logic_grid_width=logic_grid.width,
    coupling_matrix=J,
    site_coords_matrix=logic_site_coords,
    constraint_alpha=constraint_alpha,
    num_trials=num_trials,
    num_steps=num_steps,
    dev=dev,
    betamin=0.01,
    betamax=0.5,
    anneal='inverse',
    optimizer='adam',
    learning_rate=0.1,
    h_factor=0.01,
    seed=1,
    dtype=torch.float32,
)

# Optimize
print("\nINFO: Starting FEM optimization with QUBO formulation...")
config, result = optimizer.optimize()

# Find optimal solution
best_idx = torch.argmin(result)
print(f"\nINFO: Best trial index: {best_idx}")
print(f"INFO: Optimizer HPWL (expected over soft probabilities): {result[best_idx]:.2f}")

# Extract best solution
print("\nINFO: Extracting best coordinates...")
grid_coords = config[best_idx]  # Shape: [num_inst, 2]
print(f"INFO: Grid coordinates shape: {grid_coords.shape}")

# Convert to real FPGA coordinates
print("INFO: Converting to real FPGA coordinates...")
coords = logic_grid.to_real_coords_tensor(grid_coords)
print(f"INFO: Real coordinates shape: {coords.shape}")
print(f"DEBUG: First 10 real coords: {coords[:10]}")
print(f"DEBUG: Min/Max coords: ({coords.min(dim=0).values}, {coords.max(dim=0).values})")

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
    title_suffix='QUBO - Final'
)

print("\n" + "=" * 80)
print("FPGA Placement Test (QUBO) Complete!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  - Optimizer HPWL (expected/soft): {result[best_idx]:.2f}")
print(f"  - HPWL before legalization: {hpwl_before}")
print(f"  - HPWL after legalization: {hpwl_after}")
print(f"  - Overlaps resolved: {overlap}")
print(f"  - Output DCP: {output_file}")
