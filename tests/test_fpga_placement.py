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
num_steps = 1000
dev = 'cpu'

# Initialize FPGA placer
fpga_wrapper = FpgaPlacer()
fpga_wrapper.init_placement('./vivado/output_dir/post_impl.dcp', 'optimized_placement.dcp')

# Parse FPGA design to get coupling matrix
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)

# Get layout dimensions
area_length = fpga_wrapper.bbox['area_length']

print(f"INFO: Number of instances: {num_inst}")
print(f"INFO: Number of sites: {num_site}")
print(f"INFO: Area length: {area_length}")

# Set up visualization
global_drawer = PlacementDrawer(bbox=fpga_wrapper.bbox)

# Create FPGA placement optimizer with visualization support
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    coupling_matrix=J,
    drawer=global_drawer,
    visualization_steps=[0, 250, 500, 750, 999]
)

# Solve
print("INFO: Starting FEM optimization...")
config, result = optimizer.optimize(
    num_trials=num_trials,
    num_steps=num_steps,
    dev=dev,
    q=area_length  # Grid size for each coordinate dimension
)

# Find optimal solution
optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
print(f"INFO: Optimal indices: {optimal_inds.tolist()} with min HPWL: {result.min():.2f}")
best_config = config[optimal_inds[0]]

# Legalize placement
print("INFO: Legalizing placement...")
legalizer = Legalizer(fpga_wrapper.bbox)
placement_legalized = legalizer.legalize_placement(best_config, max_attempts=100)

# Calculate final HPWL
hpwl_after_legalization = fpga_wrapper.estimate_solver_hpwl(
    placement_legalized,
    io_coords=None,
    include_io=False
)
print(f"INFO: Total HPWL after legalization: {hpwl_after_legalization:.2f}")

# Route connections
print("INFO: Routing connections...")
router = Router(fpga_wrapper.bbox)
routes = router.route_connections(J, placement_legalized.unsqueeze(0))[0]

# Visualize final result
global_drawer.draw_complete_placement(
    placement_legalized,
    routes,
    1000,
    title_suffix="Final Placement with Routing"
)

print("INFO: FPGA placement complete!")
