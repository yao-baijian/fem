"""
FPGA Placement Package

This package provides FPGA placement functionality using the FEM framework.
It uses FEM's customize interface to implement FPGA-specific objectives and constraints.
"""

from .placer import FpgaPlacer
from .drawer import PlacementDrawer
from .legalizer import Legalizer
from .router import Router
from .optimizer import FPGAPlacementOptimizerJoint
from .timer import Timer
from .solver_sb import SBSolver, SBPlacementSolver

# Joint distribution approach (grid_width/grid_height based)
from .objectives import (
    expected_fpga_placement_joint,
    infer_placements_joint,
    get_hpwl_loss_joint,
    get_constraints_loss_joint,
    get_grid_coords_joint,
    get_placements_from_joint_st,
)

# QUBO approach (site_coords_matrix based)
from .objectives import (
    get_inst_coords_from_index,
    get_io_coords_from_index,
    get_site_distance_matrix,
    get_expected_placements_from_index,
    get_hard_placements_from_index,
    get_placements_from_index_st,
    get_hpwl_loss_qubo,
    get_hpwl_loss_qubo_with_io,
    get_constraints_loss,
    get_constraints_loss_with_io,
    expected_fpga_placement,
    expected_fpga_placement_with_io,
    infer_placements,
    infer_placements_with_io,
    get_loss_history,
    get_placement_history,
    clear_history,
    manual_grad_hpwl_loss,
    manual_grad_constraint_loss,
    manual_grad_placement,
)

# Hypergraph balanced min-cut
from .hyper_bmincut import (
    balance_constrain,
    balance_constrain_softplus,
    balance_constrain_relu,
    infer_hyperbmincut,
    expected_hyperbmincut,
    expected_hyperbmincut_expected_nodes_temped,
    expected_hyperbmincut_max_expected_nodes,
    expected_hyperbmincut_all_comb,
    expected_hyperbmincut_expected_crossing_simplified,
    manual_grad_hyperbmincut,
)

__all__ = [
    # Core classes
    'FpgaPlacer',
    'PlacementDrawer',
    'Legalizer',
    'Router',
    'FPGAPlacementOptimizerJoint',
    'Timer',
    'SBSolver',
    'SBPlacementSolver',

    # Joint distribution functions
    'expected_fpga_placement_joint',
    'infer_placements_joint',
    'get_hpwl_loss_joint',
    'get_constraints_loss_joint',
    'get_grid_coords_joint',
    'get_placements_from_joint_st',

    # QUBO functions
    'get_inst_coords_from_index',
    'get_io_coords_from_index',
    'get_site_distance_matrix',
    'get_expected_placements_from_index',
    'get_hard_placements_from_index',
    'get_placements_from_index_st',
    'get_hpwl_loss_qubo',
    'get_hpwl_loss_qubo_with_io',
    'get_constraints_loss',
    'get_constraints_loss_with_io',
    'expected_fpga_placement',
    'expected_fpga_placement_with_io',
    'infer_placements',
    'infer_placements_with_io',
    'get_loss_history',
    'get_placement_history',
    'clear_history',
    'manual_grad_hpwl_loss',
    'manual_grad_constraint_loss',
    'manual_grad_placement',

    # Hypergraph balanced min-cut
    'balance_constrain',
    'balance_constrain_softplus',
    'balance_constrain_relu',
    'infer_hyperbmincut',
    'expected_hyperbmincut',
    'expected_hyperbmincut_expected_nodes_temped',
    'expected_hyperbmincut_max_expected_nodes',
    'expected_hyperbmincut_all_comb',
    'expected_hyperbmincut_expected_crossing_simplified',
    'manual_grad_hyperbmincut',
]
