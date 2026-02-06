"""
Test script for xw/refactor branch FPGA placement.

This script tests the refactored FPGA placement implementation using the QUBO
(Quadratic Unconstrained Binary Optimization) formulation. The refactor should
produce identical results to the master branch.
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
from fem_placer.utils import parse_fpga_design

num_trials = 10
num_steps = 2000
dev = 'cpu'
dcp_file = './vivado/output_dir/post_impl.dcp'
output_file = 'optimized_placement_refactor.dcp'

# Initialize FPGA placer
print("=" * 80)
print("Testing xw/refactor Branch FPGA Placement (QUBO Algorithm)")
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
print(f"INFO: Area dimensions: {area_length} × {area_width} = {area_size}")
print(f"INFO: Grid size: {area_length} × {area_length} = {area_length ** 2} positions")

print("\nINFO: Creating site coordinates matrix for entire grid...")
logic_site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),  # x: [0, 18]
    torch.arange(area_length, dtype=torch.float32)   # y: [0, 18]
)
print(f"INFO: Site coordinates matrix shape: {logic_site_coords.shape}")
print(f"INFO: Grid covers all {area_length}×{area_length}={logic_site_coords.shape[0]} positions")

# Memory comparison
qubo_memory_mb = num_trials * num_inst * (area_length ** 2) * 4 / 1024 / 1024
print(f"\nMemory usage:")
print(f"  QUBO distribution: ~{qubo_memory_mb:.2f} MB")

# Set up visualization
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Create QUBO optimizer
print("\nINFO: Setting up QUBO optimizer...")
constraint_weight = num_inst / 2.0
print(f"INFO: Constraint weight (alpha): {constraint_weight}")

optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=logic_site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=logic_site_coords,
    drawer=drawer,
    visualization_steps=[],
    constraint_weight=constraint_weight
)

# Optimize
print("\nINFO: Starting FEM optimization with QUBO formulation...")
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
best_idx = torch.argmin(result)
print(f"\nINFO: Best trial index: {best_idx}")
print(f"INFO: Optimizer HPWL (expected over soft probabilities): {result[best_idx]:.2f}")

# Extract best solution
print("\nINFO: Extracting best coordinates...")
grid_coords = config[best_idx]  # Shape: [num_inst, 2] - grid coordinates (0-18)!

print(f"INFO: Grid coordinates shape: {grid_coords.shape}")

# Convert to real FPGA coordinates (like master does)
print("INFO: Converting to real FPGA coordinates...")
logic_grid = fpga_placer.get_grid('logic')
coords = logic_grid.to_real_coords_tensor(grid_coords)
print(f"INFO: Real coordinates shape: {coords.shape}")
print(f"DEBUG Refactor: First 10 real coords: {coords[:10]}")
print(f"DEBUG Refactor: Min/Max coords: ({coords.min(dim=0).values}, {coords.max(dim=0).values})")

# Legalize placement (using master's legalizer API)
print("\nINFO: Legalizing placement...")
legalizer = Legalizer(placer=fpga_placer, device=dev)

# Get instance IDs
logic_ids = torch.arange(num_inst)

# Legalize with master's API
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
    iteration=num_steps,
    title_suffix='xw/refactor Branch - QUBO - Final'
)

print("\n" + "=" * 80)
print("xw/refactor Branch FPGA Placement Test (QUBO) Complete!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  - Optimizer HPWL (expected/soft): {result[best_idx]:.2f}")
print(f"  - HPWL before legalization: {hpwl_before}")
print(f"  - HPWL after legalization: {hpwl_after}")
print(f"  - Overlaps resolved: {overlap}")
print(f"  - Output DCP: {output_file}")
