import torch
from math import log
from typing import Tuple
from .objectives import *
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

    """

    def __init__(
            self,
            num_inst: int,
            num_fixed_inst: int,
            num_site: int,
            coupling_matrix: torch.Tensor,
            site_coords_matrix: torch.Tensor,
            io_site_connect_matrix: torch.Tensor = None,
            io_site_coords: torch.Tensor = None,
            constraint_weight: float = 1.0,
            num_trials: int = 10,
            num_steps: int = 1000,
            dev: str = 'cpu',
            betamin: float = 0.01,
            betamax: float = 0.5,
            anneal: str = 'inverse',
            optimizer: str = 'adam',
            learning_rate: float = 0.1,
            h_factor: float = 0.01,
            seed: int = 1,
            dtype: torch.dtype = torch.float32,
            with_io: bool = False,
            manual_grad: bool = False,
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
        self.fixed_insts_num = num_fixed_inst
        self.num_site = num_site
        self.coupling_matrix = coupling_matrix
        self.site_coords_matrix = site_coords_matrix
        self.io_site_connect_matrix = io_site_connect_matrix
        self.io_site_coords = io_site_coords
        
        # self.net_sites_tensor = self.fpga_wrapper.net_manager.net_tensor      # Net to slice sites mapping tensor
        # self.io_site_connect_matrix = self.fpga_wrapper.net_manager.io_insts_matrix
        # self.site_coords = self.fpga_wrapper.logic_site_coords
        # self.io_site_coords = self.fpga_wrapper.io_site_coords
        # self.bbox_length = self.fpga_wrapper.grids['logic'].area_length
        # self.constraint_weight = self.fpga_wrapper.constraint_weight
        
        self.constraint_weight = constraint_weight
        
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.dev = dev

        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax),num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        self.betas = betas.to(self.dtype).to(self.dev) 
        
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.h_factor = h_factor
        self.seed = seed
        self.dtype = dtype
        self.with_io = with_io
        self.manual_grad = manual_grad

    def _initialize(self):

        torch.manual_seed(self.seed)
        
        if self.with_io:
            h_logic = self.h_factor * torch.randn(
                [self.num_trials, self.num_inst, self.num_site], 
                device=self.dev, dtype=self.dtype
            )
            
            h_io = self.h_factor * torch.randn(
                [self.num_trials, self.fixed_insts_num, self.fixed_insts_num], 
                device=self.dev, dtype=self.dtype
            )

            if not self.manual_grad:
                h_logic.requires_grad=True
                h_io.requires_grad=True

            return h_logic, h_io

        h = self.h_factor * torch.randn(
            [self.num_trials, self.num_inst, self.num_site],
            device=self.dev, dtype=self.dtype
        )

        if not self.manual_grad:
            h.requires_grad = True

        return h

    def _setup_optimizer(self, params):
        """Set up the torch optimizer."""
        if self.optimizer == 'adam':
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(
                params, lr=self.learning_rate, alpha=0.98, eps=1e-08,
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError("Unknown optimizer, valid choices are ['adam', 'rmsprop'].")

    def iterate_placement(self, area_width):
        h = self._initialize()
        opt = self._setup_optimizer([h], self.optimizer, self.learning_rate)

        for step in range(self.num_steps):
            p = torch.softmax(h, dim=2)
            opt.zero_grad()

            loss = expected_fpga_placement(
                self.coupling_matrix, p, self.site_coords_matrix,
                step, area_width, self.constraint_weight
            )

            free_energy = loss - entropy_q(p) / self.betas[step]
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

        return p
    
    def iterate_placement_with_io(self):
        h_logic, h_io = self._initialize()
        opt = self._setup_optimizer([h_logic, h_io], self.optimizer, self.learning_rate)

        for step in range(self.num_steps):
            p_logic = torch.softmax(h_logic, dim=2)
            p_io = torch.softmax(h_io, dim=2)
            opt.zero_grad()

            loss = expected_fpga_placement_with_io(
                self.coupling_matrix, self.io_site_connect_matrix, p_logic, p_io, self.site_coords_matrix, self.io_site_coords, self.constraint_weight)

            free_energy = loss - (entropy_q(p_logic) + entropy_q(p_io)) / self.betas[step]
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

        return p_logic, p_io

    def optimize(self, area_width) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.with_io:
            p = self.iterate_placement_with_io()
            config, result = infer_placements_with_io(self.coupling_matrix, self.io_site_connect_matrix, p[0],  p[1], self.bbox_length, self.site_coords, self.io_site_coords)
            return config, result
        else:
            p = self.iterate_placement(area_width)
            config, result = infer_placements(self.coupling_matrix, p, area_width, self.site_coords_matrix)
            return config, result
