"""
Test script for joint XY probability distribution optimizer.

This script uses the same FPGA design as test_fpga_placement.py
to test and compare joint optimizer.
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    FPGAPlacementOptimizerJoint
)
from fem_placer.utils import parse_fpga_design


# Configuration
num_trials = 10
num_steps = 2000
dev = 'cpu'

# Initialize FPGA placer
print("=" * 80)
print("Testing Joint XY Optimizer on Real FPGA Design")
print("=" * 80)

print("\nINFO: Loading FPGA design...")
fpga_wrapper = FpgaPlacer()
fpga_wrapper.init_placement('./vivado/output_dir/post_impl.dcp', 'optimized_placement_joint.dcp')

# Parse FPGA design to get coupling matrix
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)

# Get layout dimensions
area_length = fpga_wrapper.bbox['area_length']

print(f"INFO: Number of instances: {num_inst}")
print(f"INFO: Number of sites: {num_site}")
print(f"INFO: Area length: {area_length}")
print(f"INFO: Grid size: {area_length} × {area_length} = {area_length ** 2} positions")

# Memory comparison
joint_memory_mb = num_trials * num_inst * (area_length ** 2) * 4 / 1024 / 1024
separate_memory_mb = num_trials * num_inst * area_length * 2 * 4 / 1024 / 1024
print(f"\nMemory usage comparison:")
print(f"  Joint distribution: ~{joint_memory_mb:.2f} MB")
print(f"  Separate XY: ~{separate_memory_mb:.2f} MB")
print(f"  Ratio (joint/separate): {joint_memory_mb / separate_memory_mb:.2f}x")

# Set up visualization
global_drawer = PlacementDrawer(bbox=fpga_wrapper.bbox)

# Create joint optimizer
optimizer_joint = FPGAPlacementOptimizerJoint(
    num_inst=num_inst,
    coupling_matrix=J,
    drawer=None,
    visualization_steps=[0, 250, 500, 750, 999],
    constraint_weight=1.0
)

# Solve with joint optimizer
print("\nINFO: Starting FEM optimization with joint distribution...")
config_joint, result_joint = optimizer_joint.optimize(
    num_trials=num_trials,
    num_steps=num_steps,
    dev=dev,
    grid_width=area_length,
    grid_height=area_length
)

# Find optimal solution
optimal_inds_joint = torch.argwhere(result_joint == result_joint.min()).reshape(-1)
print(f"\nINFO: Optimal indices: {optimal_inds_joint.tolist()} with min HPWL: {result_joint.min():.2f}")
best_config_joint = config_joint[optimal_inds_joint[0]]

# Legalize placement
print("INFO: Legalizing placement...")
legalizer = Legalizer(fpga_wrapper.bbox)
placement_legalized_joint = legalizer.legalize_placement(best_config_joint, max_attempts=100)

# Calculate final HPWL
hpwl_joint = fpga_wrapper.estimate_solver_hpwl(
    placement_legalized_joint,
    io_coords=None,
    include_io=False
)
print(f"INFO: Total HPWL after legalization: {hpwl_joint:.2f}")

# Route connections
print("INFO: Routing connections...")
router = Router(fpga_wrapper.bbox)
routes_joint = router.route_connections(J, placement_legalized_joint.unsqueeze(0))[0]

# Visualize final result
global_drawer.draw_complete_placement(
    placement_legalized_joint,
    routes_joint,
    1000,
    title_suffix="Joint Optimizer - Final Placement"
)

print("\nINFO: FPGA placement with joint optimizer complete!")
