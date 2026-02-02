"""
Simulated Bifurcation solver wrapper for FPGA placement.

Uses: https://github.com/bqth29/simulated-bifurcation-algorithm
"""

from typing import Tuple, Literal, Optional, Callable
import torch
import simulated_bifurcation as sb


class SBSolver:
    """
    SB-based solver for FPGA placement QUBO problems.

    Usage:
        >>> solver = SBSolver(mode='discrete', heated=True)
        >>> # User provides QUBO conversion
        >>> Q = convert_connectivity_to_qubo(J)  # User implements
        >>> spins, energy = solver.solve(Q, agents=10)
        >>> # User converts spins to coordinates
        >>> coords = convert_spins_to_coords(spins, grid)  # User implements
    """

    def __init__(
        self,
        mode: Literal['ballistic', 'discrete'] = 'discrete',
        heated: bool = False,
        device: str = 'cpu',
    ):
        """
        Initialize SB solver.

        Args:
            mode: 'ballistic' (bSB, faster) or 'discrete' (dSB, more accurate)
            heated: Enable heated variants for better exploration
            device: 'cpu' or 'cuda'
        """
        self.mode = mode
        self.heated = heated
        self.device = device

    def solve(
        self,
        Q: torch.Tensor,
        agents: int = 10,
        max_steps: int = 10000,
        domain: Literal['spin', 'binary'] = 'spin',
        early_stopping: bool = True,
        best_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve QUBO/Ising problem.

        Args:
            Q: QUBO matrix [n, n] - USER PROVIDES THIS
            agents: Number of parallel trials
            max_steps: Maximum optimization steps
            domain: 'spin' for {-1,+1}, 'binary' for {0,1}
            early_stopping: Stop when converged
            best_only: Return only best solution

        Returns:
            solution: [n] or [agents, n] depending on best_only
            energy: scalar or [agents]
        """
        solution, energy = sb.minimize(
            Q,
            domain=domain,
            agents=agents,
            max_steps=max_steps,
            device=self.device,
            best_only=best_only,
            mode=self.mode,
            heated=self.heated,
            early_stopping=early_stopping,
        )
        return solution, energy

    def solve_ising(self, J: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve Ising: minimize s^T J s, s in {-1,+1}^n"""
        return self.solve(J, domain='spin', **kwargs)

    def solve_qubo(self, Q: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve QUBO: minimize x^T Q x, x in {0,1}^n"""
        return self.solve(Q, domain='binary', **kwargs)


class SBPlacementSolver:
    """
    High-level SB solver for FPGA placement with coordinate conversion.

    The QUBO formulation and coordinate conversion are left as interfaces
    for the user to implement based on their specific problem setup.
    """

    def __init__(
        self,
        mode: Literal['ballistic', 'discrete'] = 'discrete',
        heated: bool = True,
        device: str = 'cpu',
        qubo_converter: Optional[Callable] = None,
        coord_converter: Optional[Callable] = None,
    ):
        """
        Args:
            mode: SB algorithm mode
            heated: Enable heated variants
            device: Compute device
            qubo_converter: Function(J) -> Q to convert connectivity to QUBO
            coord_converter: Function(spins, grid_info) -> coords to convert solution
        """
        self.sb_solver = SBSolver(mode=mode, heated=heated, device=device)
        self.qubo_converter = qubo_converter
        self.coord_converter = coord_converter
        self.device = device

    def solve(
        self,
        connectivity_matrix: torch.Tensor,
        grid_info: dict = None,
        agents: int = 10,
        max_steps: int = 10000,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve placement and return coordinates.

        Args:
            connectivity_matrix: J matrix (insts_matrix)
            grid_info: Grid parameters for coordinate conversion
            agents: Number of parallel trials
            max_steps: Max optimization steps

        Returns:
            coords: Grid coordinates [num_inst, 2] if coord_converter provided
            energy: Objective value

        Raises:
            ValueError: If qubo_converter not set
        """
        if self.qubo_converter is None:
            raise ValueError("qubo_converter must be set. Use: solver.qubo_converter = your_function")

        # Convert connectivity to QUBO
        Q = self.qubo_converter(connectivity_matrix)

        # Solve with SB
        spins, energy = self.sb_solver.solve(Q, agents=agents, max_steps=max_steps, **kwargs)

        # Convert to coordinates if converter provided
        if self.coord_converter is not None and grid_info is not None:
            coords = self.coord_converter(spins, grid_info)
            return coords, energy

        return spins, energy
