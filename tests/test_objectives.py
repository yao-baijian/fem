"""
Unit tests for fem_placer.objectives module.

Tests verify that the refactored objective functions produce results
matching the master branch implementation.
"""

import pytest
import torch
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem_placer import objectives


@pytest.fixture
def objectives_data():
    """Load test fixtures generated from master branch."""
    fixtures_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'objectives_data.pt')
    if not os.path.exists(fixtures_path):
        pytest.skip(f"Test fixtures not found at {fixtures_path}")
    return torch.load(fixtures_path, weights_only=False)


class TestCoordinateFunctions:
    """Test coordinate conversion functions."""

    def test_get_inst_coords_from_index(self, objectives_data):
        """Test instance coordinate conversion from indices."""
        inst_indices = objectives_data['inst_indices']
        area_width = objectives_data['area_width']
        expected = objectives_data['inst_coords_from_index']

        result = objectives.get_inst_coords_from_index(inst_indices, area_width)

        assert torch.allclose(result, expected, atol=1e-5), \
            f"get_inst_coords_from_index mismatch: max diff = {(result - expected).abs().max()}"

    def test_get_site_distance_matrix(self, objectives_data):
        """Test site distance matrix calculation."""
        site_coords_matrix = objectives_data['site_coords_matrix']
        expected = objectives_data['site_distance_matrix']

        result = objectives.get_site_distance_matrix(site_coords_matrix)

        assert torch.allclose(result, expected, atol=1e-5), \
            f"get_site_distance_matrix mismatch: max diff = {(result - expected).abs().max()}"

    def test_get_expected_placements_from_index(self, objectives_data):
        """Test expected placement calculation."""
        p = objectives_data['p']
        site_coords_matrix = objectives_data['site_coords_matrix']
        expected = objectives_data['expected_placements']

        result = objectives.get_expected_placements_from_index(p, site_coords_matrix)

        assert torch.allclose(result, expected, atol=1e-5), \
            f"get_expected_placements_from_index mismatch: max diff = {(result - expected).abs().max()}"

    def test_get_hard_placements_from_index(self, objectives_data):
        """Test hard placement calculation."""
        p = objectives_data['p']
        site_coords_matrix = objectives_data['site_coords_matrix']
        expected = objectives_data['hard_placements']

        result = objectives.get_hard_placements_from_index(p, site_coords_matrix)

        assert torch.allclose(result, expected, atol=1e-5), \
            f"get_hard_placements_from_index mismatch: max diff = {(result - expected).abs().max()}"

    def test_get_placements_from_index_st(self, objectives_data):
        """Test straight-through placement calculation."""
        p = objectives_data['p']
        site_coords_matrix = objectives_data['site_coords_matrix']
        expected = objectives_data['st_placements']

        result = objectives.get_placements_from_index_st(p, site_coords_matrix)

        assert torch.allclose(result, expected, atol=1e-5), \
            f"get_placements_from_index_st mismatch: max diff = {(result - expected).abs().max()}"


class TestHPWLFunctions:
    """Test HPWL loss functions."""

    def test_get_hpwl_loss_qubo(self, objectives_data):
        """Test QUBO HPWL loss calculation."""
        J = objectives_data['J']
        p = objectives_data['p']
        site_coords_matrix = objectives_data['site_coords_matrix']
        expected = objectives_data['hpwl_qubo']

        result = objectives.get_hpwl_loss_qubo(J, p, site_coords_matrix)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"get_hpwl_loss_qubo mismatch: max diff = {(result - expected).abs().max()}"

    def test_get_hpwl_loss_qubo_with_io(self, objectives_data):
        """Test QUBO HPWL loss with IO."""
        J_LL = objectives_data['J_LL']
        J_LI = objectives_data['J_LI']
        p_logic = objectives_data['p_logic']
        p_io = objectives_data['p_io']
        logic_site_coords = objectives_data['site_coords_matrix']
        io_site_coords = objectives_data['io_site_coords']
        expected = objectives_data['hpwl_with_io']

        result = objectives.get_hpwl_loss_qubo_with_io(
            J_LL, J_LI, p_logic, p_io, logic_site_coords, io_site_coords
        )

        assert torch.allclose(result, expected, atol=1e-4), \
            f"get_hpwl_loss_qubo_with_io mismatch: max diff = {(result - expected).abs().max()}"


class TestConstraintFunctions:
    """Test constraint loss functions."""

    def test_get_constraints_loss(self, objectives_data):
        """Test constraint loss calculation."""
        p = objectives_data['p']
        expected = objectives_data['constraints']

        result = objectives.get_constraints_loss(p)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"get_constraints_loss mismatch: max diff = {(result - expected).abs().max()}"

    def test_get_constraints_loss_with_io(self, objectives_data):
        """Test constraint loss with IO."""
        p_logic = objectives_data['p_logic']
        p_io = objectives_data['p_io']
        expected = objectives_data['constraints_with_io']

        result = objectives.get_constraints_loss_with_io(p_logic, p_io)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"get_constraints_loss_with_io mismatch: max diff = {(result - expected).abs().max()}"


class TestExpectedPlacementFunctions:
    """Test expected placement loss functions."""

    def test_expected_fpga_placement(self, objectives_data):
        """Test expected placement loss calculation."""
        J = objectives_data['J']
        p = objectives_data['p']
        site_coords_matrix = objectives_data['site_coords_matrix']
        area_width = objectives_data['area_width']
        alpha = objectives_data['alpha']
        expected = objectives_data['expected_placement']

        # Clear history before test
        objectives.clear_history()

        result = objectives.expected_fpga_placement(
            J, p, site_coords_matrix, step=0, area_width=area_width, alpha=alpha
        )

        assert torch.allclose(result, expected, atol=1e-4), \
            f"expected_fpga_placement mismatch: max diff = {(result - expected).abs().max()}"

    def test_expected_fpga_placement_with_io(self, objectives_data):
        """Test expected placement loss with IO."""
        J_LL = objectives_data['J_LL']
        J_LI = objectives_data['J_LI']
        p_logic = objectives_data['p_logic']
        p_io = objectives_data['p_io']
        logic_site_coords = objectives_data['site_coords_matrix']
        io_site_coords = objectives_data['io_site_coords']
        expected = objectives_data['expected_placement_with_io']

        result = objectives.expected_fpga_placement_with_io(
            J_LL, J_LI, p_logic, p_io, logic_site_coords, io_site_coords
        )

        assert torch.allclose(result, expected, atol=1e-4), \
            f"expected_fpga_placement_with_io mismatch: max diff = {(result - expected).abs().max()}"


class TestInferenceFunctions:
    """Test inference functions."""

    def test_infer_placements(self, objectives_data):
        """Test placement inference."""
        J = objectives_data['J']
        p = objectives_data['p']
        area_width = objectives_data['area_width']
        site_coords_matrix = objectives_data['site_coords_matrix']
        expected_coords = objectives_data['inferred_coords']
        expected_hpwl = objectives_data['inferred_hpwl']

        coords, hpwl = objectives.infer_placements(J, p, area_width, site_coords_matrix)

        assert torch.allclose(coords, expected_coords, atol=1e-5), \
            f"infer_placements coords mismatch: max diff = {(coords - expected_coords).abs().max()}"
        assert torch.allclose(hpwl, expected_hpwl, atol=1e-4), \
            f"infer_placements hpwl mismatch: max diff = {(hpwl - expected_hpwl).abs().max()}"

    def test_infer_placements_with_io(self, objectives_data):
        """Test placement inference with IO."""
        J_LL = objectives_data['J_LL']
        J_LI = objectives_data['J_LI']
        p_logic = objectives_data['p_logic']
        p_io = objectives_data['p_io']
        area_width = objectives_data['area_width']
        logic_site_coords = objectives_data['site_coords_matrix']
        io_site_coords = objectives_data['io_site_coords']
        expected_logic_coords = objectives_data['inferred_coords_logic']
        expected_io_coords = objectives_data['inferred_coords_io']
        expected_hpwl = objectives_data['inferred_hpwl_with_io']

        coords, hpwl = objectives.infer_placements_with_io(
            J_LL, J_LI, p_logic, p_io, area_width, logic_site_coords, io_site_coords
        )

        assert torch.allclose(coords[0], expected_logic_coords, atol=1e-5), \
            f"infer_placements_with_io logic coords mismatch"
        assert torch.allclose(coords[1], expected_io_coords, atol=1e-5), \
            f"infer_placements_with_io io coords mismatch"
        assert torch.allclose(hpwl, expected_hpwl, atol=1e-4), \
            f"infer_placements_with_io hpwl mismatch"


class TestHistoryFunctions:
    """Test history tracking functions."""

    def test_history_tracking(self, objectives_data):
        """Test that history functions work correctly."""
        objectives.clear_history()

        J = objectives_data['J']
        p = objectives_data['p']
        site_coords_matrix = objectives_data['site_coords_matrix']
        area_width = objectives_data['area_width']
        alpha = objectives_data['alpha']

        # Run a few iterations to populate history
        for step in range(5):
            objectives.expected_fpga_placement(
                J, p, site_coords_matrix, step=step, area_width=area_width, alpha=alpha
            )

        history = objectives.get_loss_history()
        placements = objectives.get_placement_history()

        assert len(history['hpwl_losses']) == 5
        assert len(history['constrain_losses']) == 5
        assert len(history['total_losses']) == 5
        assert isinstance(placements, list)

        # Clear and verify
        objectives.clear_history()
        history = objectives.get_loss_history()
        assert len(history['hpwl_losses']) == 0


class TestExportPlacementQUBO:
    """Test QUBO export and decode functions."""

    @pytest.fixture
    def small_problem(self):
        """Create a small 2-instance, 4-site placement problem."""
        m, n = 2, 4
        # Flow matrix: instances 0 and 1 are connected
        F = torch.tensor([[0.0, 1.0],
                          [1.0, 0.0]])
        # 2x2 grid of sites
        site_coords = torch.tensor([[0.0, 0.0],
                                     [1.0, 0.0],
                                     [0.0, 1.0],
                                     [1.0, 1.0]])
        return m, n, F, site_coords

    def test_qubo_shape(self, small_problem):
        """Test that export_placement_qubo returns correct Q shape."""
        m, n, F, site_coords = small_problem
        Q, meta = objectives.export_placement_qubo(F, site_coords, lam=1.0, mu=1.0)

        expected_size = m * n + n
        assert Q.shape == (expected_size, expected_size), \
            f"Expected Q shape ({expected_size}, {expected_size}), got {Q.shape}"
        assert meta['m'] == m
        assert meta['n'] == n
        assert torch.equal(meta['site_coords'], site_coords)

    def test_qubo_symmetry(self, small_problem):
        """Test that Q_full is symmetric."""
        m, n, F, site_coords = small_problem
        Q, _ = objectives.export_placement_qubo(F, site_coords, lam=2.0, mu=3.0)
        assert torch.allclose(Q, Q.T, atol=1e-6), "Q_full should be symmetric"

    def test_qubo_energy_matches_hpwl_loss(self, small_problem):
        """Test that z^T Q z matches the full objective up to a constant offset.

        The QUBO matrix drops the constant λ·m from ‖Ax-1‖² since it doesn't
        affect the argmin. So: z^T Q z = objective - λ·m.
        """
        m, n, F, site_coords = small_problem
        lam, mu = 2.0, 3.0
        Q, _ = objectives.export_placement_qubo(F, site_coords, lam=lam, mu=mu)

        # Feasible solution: instance 0 -> site 0, instance 1 -> site 3
        x = torch.zeros(m * n)
        x[0] = 1.0   # instance 0 picks site 0
        x[7] = 1.0   # instance 1 picks site 3

        # Matching slack: s[j] = 1 if site j is used
        s = torch.zeros(n)
        s[0] = 1.0
        s[3] = 1.0

        z = torch.cat([x, s])
        qubo_energy = z @ Q @ z

        # Compute expected energy manually:
        D = objectives.get_site_distance_matrix(site_coords)
        FkD = torch.kron(F, D)
        hpwl_term = x @ FkD @ x

        x_mat = x.reshape(m, n)
        row_sums = x_mat.sum(dim=1)
        onehot_penalty = lam * ((row_sums - 1) ** 2).sum()

        col_sums = x_mat.sum(dim=0)
        atmost_penalty = mu * ((col_sums - s) ** 2).sum()

        # Q drops constant λ·m from the one-hot constraint expansion
        constant_offset = lam * m
        expected_energy = hpwl_term + onehot_penalty + atmost_penalty - constant_offset
        assert torch.allclose(qubo_energy, expected_energy, atol=1e-4), \
            f"QUBO energy {qubo_energy.item():.4f} != expected {expected_energy.item():.4f}"

    def test_qubo_energy_feasible_zero_penalty(self, small_problem):
        """Test that feasible solutions differ only by the constant offset."""
        m, n, F, site_coords = small_problem
        lam, mu = 10.0, 10.0
        Q, _ = objectives.export_placement_qubo(F, site_coords, lam=lam, mu=mu)

        # Feasible: each instance picks distinct site, s matches
        x = torch.zeros(m * n)
        x[1] = 1.0  # inst 0 -> site 1
        x[6] = 1.0  # inst 1 -> site 2
        s = torch.zeros(n)
        s[1] = 1.0
        s[2] = 1.0
        z = torch.cat([x, s])

        qubo_energy = z @ Q @ z

        # HPWL term minus constant offset λ·m
        D = objectives.get_site_distance_matrix(site_coords)
        FkD = torch.kron(F, D)
        hpwl_only = x @ FkD @ x
        expected = hpwl_only - lam * m

        assert torch.allclose(qubo_energy, expected, atol=1e-4), \
            f"Feasible energy: {qubo_energy.item():.4f} != expected {expected.item():.4f}"

    def test_decode_roundtrip(self, small_problem):
        """Test decode_qubo_solution roundtrip."""
        m, n, _, site_coords = small_problem

        # Encode: instance 0 -> site 2, instance 1 -> site 1
        x = torch.zeros(m * n)
        x[2] = 1.0   # inst 0 -> site 2
        x[5] = 1.0   # inst 1 -> site 1
        s = torch.zeros(n)
        s[2] = 1.0
        s[1] = 1.0
        z = torch.cat([x, s])

        site_indices, coords = objectives.decode_qubo_solution(z, m, n, site_coords)

        assert site_indices[0].item() == 2
        assert site_indices[1].item() == 1
        assert torch.allclose(coords[0], site_coords[2])
        assert torch.allclose(coords[1], site_coords[1])

    def test_decode_with_soft_solution(self, small_problem):
        """Test decode handles non-binary (soft) solutions via argmax."""
        m, n, _, site_coords = small_problem

        z = torch.zeros(m * n + n)
        # Soft assignment for instance 0: mostly site 0
        z[0] = 0.8
        z[1] = 0.2
        # Soft assignment for instance 1: mostly site 3
        z[6] = 0.1
        z[7] = 0.9

        site_indices, coords = objectives.decode_qubo_solution(z, m, n, site_coords)
        assert site_indices[0].item() == 0
        assert site_indices[1].item() == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
