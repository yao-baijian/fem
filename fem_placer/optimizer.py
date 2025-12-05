import torch
from math import log
from typing import Optional, List, Tuple
from .objectives import expected_fpga_placement_joint, infer_placements_joint
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


def get_site_coordinates_from_p_joint(p: torch.Tensor, grid_width: int) -> torch.Tensor:
    """
    Convert joint probability distribution to site coordinates.

    Args:
        p: Joint probability distribution [num_trials, num_nodes, grid_width * grid_height]
        grid_width: Width of the grid

    Returns:
        Coordinates tensor [num_trials, num_nodes, 2]
    """
    num_trials, num_nodes, num_positions = p.shape

    # Get position with highest probability for each node
    position_indices = torch.argmax(p, dim=2)  # [num_trials, num_nodes]

    # Convert flattened position index to (x, y) coordinates
    x_coords = position_indices % grid_width
    y_coords = position_indices // grid_width

    # Stack into coordinate tensor
    coords = torch.stack([x_coords, y_coords], dim=2)

    return coords.float()


class FPGAPlacementOptimizerJoint:
    """
    FPGA Placement optimizer using joint XY probability distribution.

    This is a simpler, more direct approach compared to the separate XY version.
    It uses a single probability distribution over all 2D grid positions.

    Example:
        >>> optimizer = FPGAPlacementOptimizerJoint(
        ...     num_inst, J, drawer=global_drawer,
        ...     visualization_steps=[0, 250, 500, 750, 999]
        ... )
        >>> config, result = optimizer.optimize(
        ...     num_trials=10, 
        ...     num_steps=1000, 
        ...     grid_width=50, 
        ...     grid_height=50
        ... )
    """

    def __init__(
            self,
            num_inst: int,
            coupling_matrix: torch.Tensor,
            drawer: Optional[PlacementDrawer] = None,
            visualization_steps: Optional[List[int]] = None,
            constraint_weight: float = 1.0
        ):
        """
        Initialize the FPGA placement optimizer.

        Args:
            num_inst: Number of instances to place
            coupling_matrix: Instance connectivity matrix
            drawer: Optional PlacementDrawer for visualization
            visualization_steps: Steps at which to visualize (default: [0, 250, 500, 750, 999])
            hpwl_weight: Weight for HPWL loss (default: 1.0)
            constraint_weight: Weight for constraint loss (default: 1.0)
        """
        self.num_inst = num_inst
        self.coupling_matrix = coupling_matrix
        self.drawer = drawer
        self.visualization_steps = visualization_steps or [0, 250, 500, 750, 999]
        self.constraint_weight = constraint_weight

    def _initialize_potentials(self, num_trials, grid_width, grid_height, dev, dtype, h_factor, seed):
        """
        Initialize joint potentials for all 2D positions.

        Args:
            num_trials: Number of parallel trials
            grid_width: Grid width
            grid_height: Grid height
            dev: Device ('cpu' or 'cuda')
            dtype: Torch dtype
            h_factor: Initialization scale factor
            seed: Random seed

        Returns:
            h: Initialized potentials [num_trials, num_inst, grid_width * grid_height]
        """
        torch.manual_seed(seed)

        num_positions = grid_width * grid_height
        h = h_factor * torch.randn(
            [num_trials, self.num_inst, num_positions],
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
            self, num_trials, num_steps, grid_width, grid_height, dev, dtype,
            betamin, betamax, anneal, optimizer_name, learning_rate,
            h_factor, seed
        ):
        """
        FEM optimization loop for FPGA placement with joint XY distribution.

        Uses free energy minimization with HPWL and constraint losses.
        """
        # Initialize
        h = self._initialize_potentials(
            num_trials, grid_width, grid_height, dev, dtype, h_factor, seed
        )
        opt = self._setup_optimizer([h], optimizer_name, learning_rate)
        betas = self._setup_betas(num_steps, betamin, betamax, anneal, dev, dtype)

        # Iterate
        for step in range(num_steps):
            # Convert potentials to probabilities
            p = torch.softmax(h, dim=2)

            opt.zero_grad()

            # Calculate losses
            hpwl_loss, constrain_loss = expected_fpga_placement_joint(self.coupling_matrix, p, grid_width, grid_height)

            # Calculate free energy with weighted losses
            free_energy = hpwl_loss + self.constraint_weight * constrain_loss - entropy_q(p) / betas[step]

            # Backpropagate
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

            # Visualization at specified steps
            if self.drawer is not None and step in self.visualization_steps:
                coords = get_site_coordinates_from_p_joint(p, grid_width)
                self.drawer.add_placement(coords[0].detach(), step)
                print(f"INFO: Step {step},\n"
                      f"HPWL loss = {[f'{x:.2f}' for x in hpwl_loss.tolist()]},\n"
                      f"Constraint loss = {[f'{x:.2f}' for x in (constrain_loss*self.constraint_weight).tolist()]},\n"
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
            grid_width: Optional[int] = None,
            grid_height: Optional[int] = None,
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
        Run FPGA placement optimization with joint distribution.

        Args:
            num_trials: Number of parallel optimization trials
            num_steps: Number of optimization steps
            dev: Device to run on ('cpu' or 'cuda')
            grid_width: Grid width (required)
            grid_height: Grid height (defaults to grid_width if not specified)
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
        if grid_width is None:
            raise ValueError("grid_width must be specified for FPGA placement")

        if grid_height is None:
            grid_height = grid_width

        # Run FEM iteration
        p = self.iterate_placement(
            num_trials, num_steps, grid_width, grid_height, dev, dtype,
            betamin, betamax, anneal, optimizer, learning_rate,
            h_factor, seed
        )

        # Inference
        config, result = infer_placements_joint(self.coupling_matrix, p, grid_width, grid_height)

        return config, result
