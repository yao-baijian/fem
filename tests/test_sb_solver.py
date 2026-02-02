import pytest
import torch
import sys
from pathlib import Path

# Add fem_placer to path to import solver_sb directly
sys.path.insert(0, str(Path(__file__).parent.parent / "fem_placer"))
from solver_sb import SBSolver, SBPlacementSolver


class TestSBSolver:

    def test_solve_ising(self):
        """Test Ising problem solving."""
        J = torch.randn(10, 10)
        J = (J + J.T) / 2

        solver = SBSolver(mode='discrete')
        spins, energy = solver.solve_ising(J, agents=5, max_steps=500)

        assert spins.shape == (10,)
        assert torch.all((spins == -1) | (spins == 1))

    def test_solve_qubo(self):
        """Test QUBO problem solving."""
        Q = torch.randn(10, 10)
        Q = (Q + Q.T) / 2

        solver = SBSolver()
        bits, energy = solver.solve_qubo(Q, agents=5, max_steps=500)

        assert bits.shape == (10,)
        assert torch.all((bits == 0) | (bits == 1))

    def test_modes(self):
        """Test all algorithm modes."""
        J = torch.randn(8, 8)
        J = (J + J.T) / 2

        for mode in ['ballistic', 'discrete']:
            for heated in [False, True]:
                solver = SBSolver(mode=mode, heated=heated)
                spins, _ = solver.solve_ising(J, agents=2, max_steps=100)
                assert spins.shape == (8,)


class TestSBPlacementSolver:

    def test_with_custom_converter(self):
        """Test with user-provided QUBO converter."""
        # Simple identity converter for testing
        def identity_converter(J):
            return J

        solver = SBPlacementSolver(
            mode='discrete',
            qubo_converter=identity_converter
        )

        J = torch.randn(10, 10)
        J = (J + J.T) / 2

        spins, energy = solver.solve(J, agents=5, max_steps=200)
        assert spins.shape == (10,)

    def test_missing_converter_raises(self):
        """Test error when converter not set."""
        solver = SBPlacementSolver()
        J = torch.randn(5, 5)

        with pytest.raises(ValueError, match="qubo_converter"):
            solver.solve(J)
