"""
FPGA Placement Package

This package provides FPGA placement functionality using the FEM framework.
It uses FEM's customize interface to implement FPGA-specific objectives and constraints.
"""

from .placer import FpgaPlacer
from .drawer import PlacementDrawer
from .legalizer import Legalizer
from .router import Router
from .optimizer import FPGAPlacementOptimizer
from .objectives import (
    expected_fpga_placement_xy,
    infer_placements_xy,
    get_hpwl_loss_xy_simple,
    get_constraints_loss_xy
)

__all__ = [
    'FpgaPlacer',
    'PlacementDrawer',
    'Legalizer',
    'Router',
    'FPGAPlacementOptimizer',
    'expected_fpga_placement_xy',
    'infer_placements_xy',
    'get_hpwl_loss_xy_simple',
    'get_constraints_loss_xy',
]
