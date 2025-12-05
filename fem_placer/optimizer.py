"""
FPGA Placement Optimizer with Visualization Support

This module provides a clean interface for FPGA placement optimization
using FEM principles, with built-in visualization support.

The optimizer implements a custom FEM iteration loop specifically designed
for FPGA placement, which requires separate probability distributions for
X and Y coordinates.
"""

import torch
from math import log
from typing import Optional, List, Tuple
from .objectives import expected_fpga_placement_xy, infer_placements_xy
from .drawer import PlacementDrawer


def entropy_q(p):
    """
    Calculate entropy for q-dimensional probability distributions.

    Args:
        p: Probabilities [batch, N, q]

    Returns:
        Entropy values [batch]
    """
    return -(p * torch.log(p + 1e-10)).sum(2).sum(1)


def get_site_coordinates_from_px_py(p_x: torch.Tensor, p_y: torch.Tensor) -> torch.Tensor:
    """
    Convert probability distributions to site coordinates.

    Args:
        p_x: Probability distribution over X coordinates [num_trials, num_nodes, q]
        p_y: Probability distribution over Y coordinates [num_trials, num_nodes, q]

    Returns:
        Coordinates tensor [num_trials, num_nodes, 2]
    """
    num_trials, num_nodes, q = p_x.shape

    # Get expected coordinates for each site
    coords_x = torch.sum(p_x * torch.arange(q, device=p_x.device).view(1, 1, -1), dim=2)
    coords_y = torch.sum(p_y * torch.arange(q, device=p_y.device).view(1, 1, -1), dim=2)

    # Stack into coordinate tensor
    coords = torch.stack([coords_x, coords_y], dim=2)

    return coords


class FPGAPlacementOptimizer:
    """
    FPGA Placement optimizer using FEM principles with visualization support.

    This optimizer implements a custom FEM-based iteration loop specifically
    designed for FPGA placement. It maintains separate probability distributions
    for X and Y coordinates and supports real-time visualization.

    Example:
        >>> optimizer = FPGAPlacementOptimizer(
        ...     num_inst, J, drawer=global_drawer,
        ...     visualization_steps=[0, 250, 500, 750, 999]
        ... )
        >>> config, result = optimizer.optimize(
        ...     num_trials=10, num_steps=1000, q=area_length
        ... )
    """

    def __init__(
            self,
            num_inst: int,
            coupling_matrix: torch.Tensor,
            drawer: Optional[PlacementDrawer] = None,
            visualization_steps: Optional[List[int]] = None
        ):
        """
        Initialize the FPGA placement optimizer.

        Args:
            num_inst: Number of instances to place
            coupling_matrix: Instance connectivity matrix
            drawer: Optional PlacementDrawer for visualization
            visualization_steps: Steps at which to visualize (default: [0, 250, 500, 750, 999])
        """
        self.num_inst = num_inst
        self.coupling_matrix = coupling_matrix
        self.drawer = drawer
        self.visualization_steps = visualization_steps or [0, 250, 500, 750, 999]

    def _initialize_potentials(self, num_trials, q, dev, dtype, h_factor, seed):
        """
        Initialize separate potentials for X and Y coordinates.

        Args:
            num_trials: Number of parallel trials
            q: Grid dimension size
            dev: Device ('cpu' or 'cuda')
            dtype: Torch dtype
            h_factor: Initialization scale factor
            seed: Random seed

        Returns:
            h_x, h_y: Initialized potentials for X and Y
        """
        torch.manual_seed(seed)

        h_x = h_factor * torch.randn(
            [num_trials, self.num_inst, q],
            device=dev, dtype=dtype
        )
        h_y = h_factor * torch.randn(
            [num_trials, self.num_inst, q],
            device=dev, dtype=dtype
        )

        h_x.requires_grad = True
        h_y.requires_grad = True

        return h_x, h_y

    def _setup_optimizer(self, params, optimizer_name, learning_rate):
        """Set up the torch optimizer."""
        if optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(
                params, lr=learning_rate, alpha=0.98, eps=1e-08,
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError("Unknown optimizer, valid choices are ['adam', 'rmsprop'].")

    def _setup_betas(self, num_steps, betamin, betamax, anneal, dev, dtype):
        """Set up temperature schedule."""
        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax), num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(1/betamax, 1/betamin, num_steps)
        else:
            raise ValueError(f"Unknown anneal schedule: {anneal}")

        return betas.to(dtype).to(dev)

    def iterate_placement(
            self, num_trials, num_steps, q, dev, dtype,
            betamin, betamax, anneal, optimizer_name, learning_rate,
            h_factor, seed
        ):
        """
        FEM optimization loop for FPGA placement with separate X/Y coordinates.

        Uses free energy minimization with HPWL and constraint losses.
        """
        # Initialize
        h_x, h_y = self._initialize_potentials(
            num_trials, q, dev, dtype, h_factor, seed
        )
        opt = self._setup_optimizer([h_x, h_y], optimizer_name, learning_rate)
        betas = self._setup_betas(num_steps, betamin, betamax, anneal, dev, dtype)

        # Iterate
        for step in range(num_steps):
            # Convert potentials to probabilities
            p_x = torch.softmax(h_x, dim=2)
            p_y = torch.softmax(h_y, dim=2)

            opt.zero_grad()

            # Calculate losses
            hpwl_loss, constrain_loss = expected_fpga_placement_xy(
                self.coupling_matrix, p_x, p_y
            )

            # Calculate free energy
            free_energy = hpwl_loss + constrain_loss - \
                (entropy_q(p_x) + entropy_q(p_y)) / betas[step]

            # Backpropagate
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

            # Visualization at specified steps
            if self.drawer is not None and step in self.visualization_steps:
                coords = get_site_coordinates_from_px_py(p_x, p_y)
                self.drawer.add_placement(coords[0].detach(), step)
                print(f"INFO: Step {step}, HPWL loss = {[f'{x:.2f}' for x in hpwl_loss.tolist()]}, "
                      f"Constraint loss = {[f'{x:.2f}' for x in constrain_loss.tolist()]}")

        # Draw multi-step visualization
        if self.drawer is not None and len(self.drawer.placement_history) > 0:
            self.drawer.draw_multi_step_placement()

        return p_x, p_y

    def optimize(
            self,
            num_trials: int = 10,
            num_steps: int = 1000,
            dev: str = 'cpu',
            q: Optional[int] = None,
            betamin: float = 0.01,
            betamax: float = 0.5,
            anneal: str = 'inverse',
            optimizer: str = 'adam',
            learning_rate: float = 0.1,
            h_factor: float = 0.01,
            seed: int = 1,
            dtype: torch.dtype = torch.float32,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run FPGA placement optimization.

        Args:
            num_trials: Number of parallel optimization trials
            num_steps: Number of optimization steps
            dev: Device to run on ('cpu' or 'cuda')
            q: Grid size for each coordinate dimension
            betamin: Minimum inverse temperature
            betamax: Maximum inverse temperature
            anneal: Annealing schedule ('lin', 'exp', or 'inverse')
            optimizer: Optimizer type ('adam' or 'rmsprop')
            learning_rate: Learning rate for optimization
            h_factor: Initialization scale factor
            seed: Random seed
            dtype: Torch dtype

        Returns:
            config: Optimized placement configurations [num_trials, num_inst, 2]
            result: HPWL values for each trial [num_trials]
        """
        if q is None:
            raise ValueError("q (grid size) must be specified for FPGA placement")

        # Run FEM iteration
        p_x, p_y = self.iterate_placement(
            num_trials, num_steps, q, dev, dtype,
            betamin, betamax, anneal, optimizer, learning_rate,
            h_factor, seed
        )

        # Inference
        config, result = infer_placements_xy(self.coupling_matrix, p_x, p_y)

        return config, result
