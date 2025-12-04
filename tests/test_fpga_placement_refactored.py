import sys
sys.path.append('.')

import torch
from FEM import FEM
from fpga_placement import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    expected_fpga_placement_xy,
    infer_placements_xy
)
from fpga_placement.utils import parse_fpga_design


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


# Define customize functions for FEM
def customize_expected_func(coupling_matrix, p_list):
    """
    Custom expectation function for FPGA placement

    Args:
        coupling_matrix: Instance connectivity matrix (not used directly here)
        p_list: [p_x, p_y] - probability distributions for X and Y coordinates

    Returns:
        Total loss (HPWL + constraints)
    """
    p_x, p_y = p_list
    return expected_fpga_placement_xy(coupling_matrix, p_x, p_y)


def customize_infer_func(coupling_matrix, p_list):
    """
    Custom inference function for FPGA placement

    Args:
        coupling_matrix: Instance connectivity matrix
        p_list: [p_x, p_y] - probability distributions

    Returns:
        config: Inferred placement coordinates
        result: HPWL value
    """
    p_x, p_y = p_list
    return infer_placements_xy(coupling_matrix, p_x, p_y)


# Create FEM problem using customize interface
# Note: We use 'customize' instead of adding a new problem type to FEM
case_placements = FEM.from_couplings(
    'customize',  # Use FEM's customize interface
    num_inst,
    num_inst * (num_inst - 1) // 2,  # num_interactions
    J,  # coupling matrix
    customize_expected_func=customize_expected_func,
    customize_infer_func=customize_infer_func
)

# Set up visualization
global_drawer = PlacementDrawer(bbox=fpga_wrapper.bbox)

# Set up solver
# Note: q parameter represents grid dimensions for each coordinate
case_placements.set_up_solver(
    num_trials,
    num_steps,
    dev=dev,
    q=area_length,  # Grid size for each coordinate dimension
    manual_grad=False,
    drawer=global_drawer
)

# Solve
print("INFO: Starting FEM optimization...")
config, result = case_placements.solve()

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
