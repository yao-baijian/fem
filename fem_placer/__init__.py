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
from .objectives import (
    expected_fpga_placement_joint,
    infer_placements_joint,
    get_hpwl_loss_joint,
    get_constraints_loss_joint
)

__all__ = [
    'FpgaPlacer',
    'PlacementDrawer',
    'Legalizer',
    'Router',
    'FPGAPlacementOptimizerJoint',
    'expected_fpga_placement_joint',
    'infer_placements_joint',
    'get_hpwl_loss_joint',
    'get_constraints_loss_joint',
]
