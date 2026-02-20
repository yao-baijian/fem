import torch
from math import log
from typing import Optional, List, Tuple
from .objectives import expected_fpga_placement, infer_placements
from .drawer import PlacementDrawer


def entropy_q(p):
    """
    Calculate entropy for q-dimensional probability distributions.
    NOTE: Matches master branch exactly - no epsilon added.

    Args:
        p: Probabilities [batch, N, q]

    Returns:
        Entropy values [batch]
    """
    return -(p * torch.log(p)).sum(2).sum(1)


class FPGAPlacementOptimizer:
    """
    FPGA Placement optimizer using QUBO formulation with site coordinates.

    This matches the algorithm from the master branch, using pre-computed
    site coordinate matrices for HPWL calculation.

    Example:
        >>> optimizer = FPGAPlacementOptimizer(
        ...     num_inst, num_site, J, site_coords_matrix,
        ...     drawer=global_drawer,
        ...     visualization_steps=[0, 250, 500, 750, 999]
        ... )
        >>> config, result = optimizer.optimize(
        ...     num_trials=10,
        ...     num_steps=1000
        ... )
    """

    def __init__(
            self,
            num_inst: int,
            num_site: int,
            coupling_matrix: torch.Tensor,
            site_coords_matrix: torch.Tensor,
            drawer: Optional[PlacementDrawer] = None,
            visualization_steps: Optional[List[int]] = None,
            constraint_weight: float = 1.0
        ):
        """
        Initialize the FPGA placement optimizer with QUBO formulation.

        Args:
            num_inst: Number of instances to place
            num_site: Number of available placement sites
            coupling_matrix: Instance connectivity matrix [num_inst, num_inst]
            site_coords_matrix: Site coordinates [num_site, 2]
            drawer: Optional PlacementDrawer for visualization
            visualization_steps: Steps at which to visualize
            constraint_weight: Weight for constraint loss (alpha parameter)
        """
        self.num_inst = num_inst
        self.num_site = num_site
        self.coupling_matrix = coupling_matrix
        self.site_coords_matrix = site_coords_matrix
        self.drawer = drawer
        self.visualization_steps = visualization_steps or [0, 250, 500, 750, 999]
        self.constraint_weight = constraint_weight

    def _initialize_potentials(self, num_trials, dev, dtype, h_factor, seed):
        """
        Initialize potentials for site assignments.

        Args:
            num_trials: Number of parallel trials
            dev: Device ('cpu' or 'cuda')
            dtype: Torch dtype
            h_factor: Initialization scale factor
            seed: Random seed

        Returns:
            h: Initialized potentials [num_trials, num_inst, num_site]
        """
        torch.manual_seed(seed)

        h = h_factor * torch.randn(
            [num_trials, self.num_inst, self.num_site],
            device=dev, dtype=dtype
        )

        h.requires_grad = True

        return h

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
        """Set up temperature schedule (matches master branch)."""
        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax), num_steps))
        elif anneal == 'inverse':
            # IMPORTANT: Master uses betamax, betamin (swapped order)
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        else:
            raise ValueError(f"Unknown anneal schedule: {anneal}")

        return betas.to(dtype).to(dev)

    def iterate_placement(
            self, num_trials, num_steps, dev, dtype,
            betamin, betamax, anneal, optimizer_name, learning_rate,
            h_factor, seed, area_width
        ):
        """
        FEM optimization loop for FPGA placement with QUBO formulation.

        Uses free energy minimization with HPWL and constraint losses,
        matching the master branch algorithm.
        """
        # Initialize
        h = self._initialize_potentials(
            num_trials, dev, dtype, h_factor, seed
        )
        opt = self._setup_optimizer([h], optimizer_name, learning_rate)
        betas = self._setup_betas(num_steps, betamin, betamax, anneal, dev, dtype)

        # Iterate
        for step in range(num_steps):
            # Convert potentials to probabilities
            p = torch.softmax(h, dim=2)

            opt.zero_grad()

            # Calculate total loss using QUBO formulation (matches master branch)
            total_loss = expected_fpga_placement(
                self.coupling_matrix, p, self.site_coords_matrix,
                step, area_width, self.constraint_weight
            )

            # Calculate free energy
            free_energy = total_loss - entropy_q(p) / betas[step]

            # Backpropagate
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

            # Visualization at specified steps
            if self.drawer is not None and step in self.visualization_steps:
                from .objectives import get_hard_placements_from_index
                coords = get_hard_placements_from_index(p, self.site_coords_matrix)
                self.drawer.add_placement(coords[0].detach(), step)
                print(f"INFO: Step {step}, Total loss = {[f'{x:.2f}' for x in total_loss.tolist()]}, "
                      f"Free energy = {[f'{x:.2f}' for x in free_energy.tolist()]}")

        # Draw multi-step visualization
        if self.drawer is not None and len(self.drawer.placement_history) > 0:
            self.drawer.draw_multi_step_placement()

        return p

    def optimize(
            self,
            num_trials: int = 10,
            num_steps: int = 1000,
            dev: str = 'cpu',
            area_width: Optional[int] = None,
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
        Run FPGA placement optimization with QUBO formulation.

        Args:
            num_trials: Number of parallel optimization trials
            num_steps: Number of optimization steps
            dev: Device to run on ('cpu' or 'cuda')
            area_width: Grid width (required for constraint loss calculation)
            betamin: Minimum inverse temperature
            betamax: Maximum inverse temperature
            anneal: Annealing schedule ('lin', 'exp', or 'inverse')
            optimizer: Optimizer type ('adam' or 'rmsprop')
            learning_rate: Learning rate for optimization
            h_factor: Initialization scale factor
            seed: Random seed
            dtype: Torch dtype

        Returns:
            config: Optimized placement probabilities [num_trials, num_inst, num_site]
            result: HPWL values for each trial [num_trials]
        """
        if area_width is None:
            # Infer from site coordinates
            area_width = int(self.site_coords_matrix[:, 0].max().item()) + 1

        # Run FEM iteration
        p = self.iterate_placement(
            num_trials, num_steps, dev, dtype,
            betamin, betamax, anneal, optimizer, learning_rate,
            h_factor, seed, area_width
        )

        # Inference
        config, result = infer_placements(self.coupling_matrix, p, area_width, self.site_coords_matrix)

        return config, result
