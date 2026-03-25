import time
from typing import Dict, Any

import torch

from fem_placer import Legalizer, FPGAPlacementOptimizer
from fem_placer.config import PlaceType


def run_logic_placement(
    fpga_placer,
    alpha: float,
    beta: float = 0.0,
    io_factor: float = 1.0,
    num_trials: int = 5,
    num_steps: int = 200,
    dev: str = "cpu",
    manual_grad: bool = False,
    anneal: str = "inverse",
    place_type: PlaceType = PlaceType.CENTERED,
) -> Dict[str, Any]:
    """Run a single logic placement optimization + legalization.

    This centralizes the common optimizer/legalizer flow used by
    test_train_alpha.py and test_ml_prediction_comparison.py.

    Returns a dict with placement, overlap, HPWL dictionaries, and runtime.
    """
    # Clear grid state before each run
    fpga_placer.grids["logic"].clear_all()
    if place_type == PlaceType.IO:
        fpga_placer.grids["io"].clear_all()

    fpga_placer.set_alpha(alpha)
    if place_type == PlaceType.IO:
        fpga_placer.set_beta(beta)

    optimizer = FPGAPlacementOptimizer(
        num_inst=fpga_placer.instances["logic"].num,
        num_fixed_inst=fpga_placer.instances["io"].num,
        num_site=fpga_placer.get_grid("logic").area,
        num_fixed_site=fpga_placer.get_grid("io").area,
        coupling_matrix=fpga_placer.net_manager.insts_matrix,
        site_coords_matrix=fpga_placer.logic_site_coords,
        io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
        io_site_coords=fpga_placer.io_site_coords,
        constraint_alpha=fpga_placer.constraint_alpha,
        # For IO placements, beta can be set separately via fpga_placer,
        # but for logic-only CENTERED placement we mirror existing usage.
        constraint_beta=fpga_placer.constraint_alpha,
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev,
        betamin=0.01,
        betamax=0.5,
        anneal=anneal,
        optimizer="adam",
        learning_rate=0.1,
        h_factor=0.01,
        io_factor=io_factor,
        seed=1,
        dtype=torch.float32,
        with_io=(place_type == PlaceType.IO),
        manual_grad=manual_grad,
    )

    t0 = time.time()
    config, result = optimizer.optimize()
    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
    legalizer = Legalizer(placer=fpga_placer, device=dev)
    logic_ids, io_ids = fpga_placer.get_ids()

    if place_type == PlaceType.IO:
        real_logic_coords = config[0][optimal_inds[0]]
        real_io_coords = config[1][optimal_inds[0]]
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
            real_logic_coords, logic_ids, real_io_coords, io_ids, include_io=True
        )
    else:
        real_logic_coords = config[optimal_inds[0]]
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
            real_logic_coords, logic_ids
        )

    t1 = time.time()

    return {
        "placement_legalized": placement_legalized,
        "overlap": overlap,
        "fem_hpwl_initial": fem_hpwl_initial,
        "fem_hpwl_final": fem_hpwl_final,
        "time": t1 - t0,
    }
