"""
Test script for master branch FPGA placement.

This script follows the master branch API as documented in MIGRATION_GUIDE.md
to test FPGA placement using the FEM interface.
"""

import sys
sys.path.insert(0, '.')

import torch
from FEM.interface import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.legalizer import Legalizer
from FEM.placement.router import Router
from FEM.placement.drawer import PlacementDrawer
from FEM.placement.config import PlaceType, GridType

# Configuration
num_trials = 10
num_steps = 2000
dev = 'cpu'
dcp_file = './vivado/output_dir/post_impl.dcp'
output_file = 'optimized_placement_master.dcp'

# Initialize FPGA placer
print("=" * 80)
print("Testing Master Branch FPGA Placement")
print("=" * 80)

print("\nINFO: Initializing FPGA placer...")
fpga_placer = FpgaPlacer(
    place_orientation=PlaceType.CENTERED,
    grid_type=GridType.SQUARE,
    utilization_factor=0.3,  # Lower utilization to ensure enough sites
    device=dev
)

print(f"INFO: Loading FPGA design from {dcp_file}...")
vivado_hpwl, opti_insts_num, site_net_num, total_net_num = fpga_placer.init_placement(
    dcp_file=dcp_file,
    dcp_output=output_file
)

print(f"INFO: Vivado HPWL: {vivado_hpwl}")
print(f"INFO: Optimizable instances: {opti_insts_num}")
print(f"INFO: Site net number: {site_net_num}")
print(f"INFO: Total net number: {total_net_num}")

# Get grid information
logic_grid = fpga_placer.get_grid('logic')
print(f"\nINFO: Logic grid size: {logic_grid.width} × {logic_grid.height} = {logic_grid.area}")

# Set up visualization
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Create FEM problem from FPGA placer
print("\nINFO: Setting up FEM problem...")
case = FEM.from_file(
    problem_type='fpga_placement',
    filename='fpga_design',
    fpga_wrapper=fpga_placer,
    epsilon=0.03,
    q=logic_grid.area,
    hyperedges=None,
    map_type='normal'
)

# Setup solver
print("INFO: Setting up solver...")
case.set_up_solver(
    num_trials=num_trials,
    num_steps=num_steps,
    betamin=0.01,
    betamax=0.5,
    anneal='inverse',
    optimizer='adam',
    learning_rate=0.1,
    dev=dev,
    dtype=torch.float32,
    seed=1,
    q=logic_grid.area,  # Must match number of sites!
    manual_grad=False,
    h_factor=0.01,
    sparse=False,
    drawer=drawer
)

# Solve
print("\nINFO: Starting FEM optimization...")
config, result = case.solve()

# Find optimal solution
best_idx = torch.argmin(result)
print(f"\nINFO: Best trial index: {best_idx} with min HPWL: {result[best_idx]:.2f}")

# Get IDs
logic_ids, io_ids = fpga_placer.get_ids()
print(f"INFO: Number of logic instances: {len(logic_ids)}")
print(f"INFO: Number of IO instances: {len(io_ids)}")

# Convert solver output to real coordinates
print("\nINFO: Converting to real coordinates...")
if fpga_placer.with_io():
    logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[0][best_idx])
    io_coords = fpga_placer.get_grid('io').to_real_coords_tensor(config[1][best_idx])
    print(f"INFO: Logic coordinates shape: {logic_coords.shape}")
    print(f"INFO: IO coordinates shape: {io_coords.shape}")
else:
    logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[best_idx])
    print(f"INFO: Logic coordinates shape: {logic_coords.shape}")
    print(f"DEBUG Master: First 10 coords: {logic_coords[:10]}")
    print(f"DEBUG Master: Min/Max coords: ({logic_coords.min(dim=0).values}, {logic_coords.max(dim=0).values})")
    io_coords = None

# Legalize placement
print("\nINFO: Legalizing placement...")
legalizer = Legalizer(placer=fpga_placer, device=dev)

if fpga_placer.with_io():
    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        logic_coords, logic_ids,
        io_coords, io_ids,
        include_io=True
    )
else:
    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        logic_coords, logic_ids
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

# Visualize
print("\nINFO: Drawing final placement...")
drawer.draw_place_and_route(
    placement_legalized[0],
    routes,
    io_coords=placement_legalized[1] if fpga_placer.with_io() else None,
    include_io=fpga_placer.with_io(),
    title_suffix='Master Branch - Final'
)

print("\n" + "=" * 80)
print("Master Branch FPGA Placement Test Complete!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  - Optimizer HPWL: {result[best_idx]:.2f}")
print(f"  - HPWL before legalization: {hpwl_before}")
print(f"  - HPWL after legalization: {hpwl_after}")
print(f"  - Overlaps resolved: {overlap}")
print(f"  - Output DCP: {output_file}")
