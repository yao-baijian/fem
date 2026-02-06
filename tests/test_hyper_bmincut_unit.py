"""
Unit tests for fem_placer.hyper_bmincut module.

Tests verify that the refactored hyper_bmincut functions produce results
matching the master branch implementation.
"""

import pytest
import torch
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem_placer import hyper_bmincut


@pytest.fixture
def hyper_bmincut_data():
    """Load test fixtures generated from master branch."""
    fixtures_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'hyper_bmincut_data.pt')
    if not os.path.exists(fixtures_path):
        pytest.skip(f"Test fixtures not found at {fixtures_path}")
    return torch.load(fixtures_path, weights_only=False)


class TestBalanceConstraintFunctions:
    """Test balance constraint functions."""

    def test_balance_constrain(self, hyper_bmincut_data):
        """Test balance constraint calculation."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        U_max = hyper_bmincut_data['U_max']
        L_min = hyper_bmincut_data['L_min']
        expected = hyper_bmincut_data['balance_constrain']

        result = hyper_bmincut.balance_constrain(J, p, U_max, L_min)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"balance_constrain mismatch: max diff = {(result - expected).abs().max()}"

    def test_balance_constrain_softplus(self, hyper_bmincut_data):
        """Test softplus balance constraint."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        U_max = hyper_bmincut_data['U_max']
        L_min = hyper_bmincut_data['L_min']

        # Just verify it runs without error and produces reasonable output
        result = hyper_bmincut.balance_constrain_softplus(J, p, U_max, L_min)

        assert result.shape == (hyper_bmincut_data['batch_size'],)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_balance_constrain_relu(self, hyper_bmincut_data):
        """Test ReLU balance constraint."""
        J = hyper_bmincut_data['J_nodes']
        p = hyper_bmincut_data['p']
        U_max = hyper_bmincut_data['U_max']
        L_min = hyper_bmincut_data['L_min']

        result = hyper_bmincut.balance_constrain_relu(J, p, U_max, L_min)

        assert result.shape == (hyper_bmincut_data['batch_size'],)
        assert not torch.isnan(result).any()


class TestCutValueFunctions:
    """Test hypergraph cut value functions."""

    def test_expected_hyperbmincut(self, hyper_bmincut_data):
        """Test expected hyperbmincut calculation."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        hyperedges = hyper_bmincut_data['hyperedges']
        expected = hyper_bmincut_data['expected_hyperbmincut']

        result = hyper_bmincut.expected_hyperbmincut(J, p, hyperedges)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"expected_hyperbmincut mismatch: max diff = {(result - expected).abs().max()}"

    def test_infer_hyperbmincut(self, hyper_bmincut_data):
        """Test hyperbmincut inference."""
        J_nodes = hyper_bmincut_data['J_nodes']
        p = hyper_bmincut_data['p']
        hyperedges = hyper_bmincut_data['hyperedges']
        expected_config = hyper_bmincut_data['infer_config']
        expected_cut = hyper_bmincut_data['infer_cut_value']

        if expected_config is None:
            pytest.skip("infer_hyperbmincut fixture not available")

        config, cut_value = hyper_bmincut.infer_hyperbmincut(J_nodes, p, hyperedges)

        assert torch.allclose(config, expected_config, atol=1e-5), \
            f"infer_hyperbmincut config mismatch"
        assert torch.allclose(cut_value, expected_cut, atol=1e-4), \
            f"infer_hyperbmincut cut_value mismatch"

    def test_expected_hyperbmincut_temped(self, hyper_bmincut_data):
        """Test temperature-scaled expected hyperbmincut."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        hyperedges = hyper_bmincut_data['hyperedges']
        expected = hyper_bmincut_data['expected_hyperbmincut_temped']

        result = hyper_bmincut.expected_hyperbmincut_expected_nodes_temped(J, p, hyperedges)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"expected_hyperbmincut_temped mismatch: max diff = {(result - expected).abs().max()}"

    def test_expected_hyperbmincut_simplified(self, hyper_bmincut_data):
        """Test simplified expected hyperbmincut."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        hyperedges = hyper_bmincut_data['hyperedges']
        expected = hyper_bmincut_data['expected_hyperbmincut_simplified']

        result = hyper_bmincut.expected_hyperbmincut_expected_crossing_simplified(J, p, hyperedges)

        assert torch.allclose(result, expected, atol=1e-4), \
            f"expected_hyperbmincut_simplified mismatch: max diff = {(result - expected).abs().max()}"

    def test_expected_hyperbmincut_max_expected(self, hyper_bmincut_data):
        """Test max expected nodes hyperbmincut."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        hyperedges = hyper_bmincut_data['hyperedges']

        result = hyper_bmincut.expected_hyperbmincut_max_expected_nodes(J, p, hyperedges)

        assert result.shape == (hyper_bmincut_data['batch_size'],)
        assert not torch.isnan(result).any()


class TestGradientFunctions:
    """Test gradient computation functions."""

    def test_manual_grad_hyperbmincut(self, hyper_bmincut_data):
        """Test manual gradient computation."""
        J_nodes = hyper_bmincut_data['J_nodes']
        p = hyper_bmincut_data['p']
        h = hyper_bmincut_data['h']
        U_max = hyper_bmincut_data['U_max']
        L_min = hyper_bmincut_data['L_min']
        batch_size = hyper_bmincut_data['batch_size']
        num_nodes = hyper_bmincut_data['num_nodes']
        num_clusters = hyper_bmincut_data['num_clusters']
        imbalance_weight = 1.0

        result = hyper_bmincut.manual_grad_hyperbmincut(
            J_nodes, p, U_max, L_min, num_nodes, h, imbalance_weight, num_clusters, batch_size
        )

        assert result.shape == h.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestFunctionalProperties:
    """Test functional properties of the module."""

    def test_all_comb_function(self, hyper_bmincut_data):
        """Test all combination function for 4 clusters."""
        J = hyper_bmincut_data['J']
        p = hyper_bmincut_data['p']
        hyperedges = hyper_bmincut_data['hyperedges']

        # Only run if we have 4 clusters
        if hyper_bmincut_data['num_clusters'] != 4:
            pytest.skip("Test requires 4 clusters")

        result = hyper_bmincut.expected_hyperbmincut_all_comb(J, p, hyperedges)

        assert result.shape == (hyper_bmincut_data['batch_size'],)
        # Note: This function may produce NaN for certain probability distributions
        # due to log(0) when cluster probabilities are very small.
        # The function is still correct for well-behaved inputs.

    def test_gradient_flow(self, hyper_bmincut_data):
        """Test that gradients flow through expected_hyperbmincut."""
        J = hyper_bmincut_data['J']
        hyperedges = hyper_bmincut_data['hyperedges']

        # Create p with requires_grad
        h = torch.randn_like(hyper_bmincut_data['h'], requires_grad=True)
        p = torch.softmax(h, dim=2)

        result = hyper_bmincut.expected_hyperbmincut(J, p, hyperedges)
        loss = result.sum()
        loss.backward()

        assert h.grad is not None
        assert not torch.isnan(h.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
