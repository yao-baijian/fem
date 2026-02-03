"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    FPGAPlacementOptimizer
)
from fem_placer.utils import parse_fpga_design


# Configuration
num_trials = 10
num_steps = 2000
dev = 'cpu'

# Initialize FPGA placer
print("=" * 80)
print("Testing FPGA Placement Optimizer on Real FPGA Design")
print("=" * 80)

print("\nINFO: Loading FPGA design...")
fpga_wrapper = FpgaPlacer()
fpga_wrapper.init_placement('./vivado/output_dir/post_impl.dcp', 'optimized_placement.dcp')

# Parse FPGA design to get coupling matrix
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)

# Get layout dimensions
area_length = fpga_wrapper.bbox['area_length']

print(f"INFO: Number of instances: {num_inst}")
print(f"INFO: Number of sites: {num_site}")
print(f"INFO: Area length: {area_length}")
print(f"INFO: Grid size: {area_length} × {area_length} = {area_length ** 2} positions")

# Create site coordinates matrix for all grid positions
print("\nINFO: Creating site coordinates matrix...")
site_coords_matrix = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)
print(f"INFO: Site coordinates shape: {site_coords_matrix.shape}")

# Set up visualization
global_drawer = PlacementDrawer(placer=fpga_wrapper, num_subplots=5, debug_mode=False)

# Create optimizer
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=site_coords_matrix.shape[0],
    coupling_matrix=J,
    site_coords_matrix=site_coords_matrix,
    drawer=None,
    visualization_steps=[0, 250, 500, 750, 999],
    constraint_weight=1.0
)

# Solve with optimizer
print("\nINFO: Starting FEM optimization...")
config, result = optimizer.optimize(
    num_trials=num_trials,
    num_steps=num_steps,
    dev=dev,
    area_width=area_length,
    betamin=0.01,
    betamax=0.5,
    anneal='inverse',
    optimizer='adam',
    learning_rate=0.1,
    h_factor=0.01,
    seed=1,
    dtype=torch.float32
)

# Find optimal solution
optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
print(f"\nINFO: Optimal indices: {optimal_inds.tolist()} with min HPWL: {result.min():.2f}")
grid_coords = config[optimal_inds[0]]  # Shape: [num_inst, 2]

# Convert grid coordinates to real FPGA coordinates
print("INFO: Converting grid coordinates to real coordinates...")
logic_grid = fpga_wrapper.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(grid_coords)

# Legalize placement
print("INFO: Legalizing placement...")
legalizer = Legalizer(placer=fpga_wrapper, device=dev)
logic_ids = torch.arange(num_inst)
placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    real_coords, logic_ids
)

# Calculate final HPWL
print(f"INFO: HPWL before legalization: {hpwl_before['hpwl']:.2f}")
print(f"INFO: HPWL after legalization: {hpwl_after['hpwl']:.2f}")
print(f"INFO: Overlap violations: {overlap}")

# Route connections
print("INFO: Routing connections...")
router = Router(placer=fpga_wrapper)
routes = router.route_connections(J, placement_legalized[0])

# Visualize final result
global_drawer.draw_complete_placement(
    placement_legalized[0],
    routes,
    num_steps,
    title_suffix="Final Placement"
)

print("\nINFO: FPGA placement complete!")
