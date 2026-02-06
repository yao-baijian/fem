"""
Unit tests for fem_placer.timer module.

Tests verify basic functionality of the Timer class.
"""

import pytest
import torch
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem_placer.timer import Timer


class TestTimerBasics:
    """Test basic Timer functionality."""

    def test_timer_initialization(self):
        """Test Timer can be initialized."""
        timer = Timer()

        assert timer.timing_library == {}
        assert timer.cell_delays == {}
        assert timer.net_delays == {}
        assert timer.timing_paths == []

    def test_get_expected_placements_from_index(self):
        """Test expected placement calculation."""
        timer = Timer()

        batch_size = 2
        num_instances = 10
        num_sites = 25  # 5x5 grid
        area_width = 5

        p = torch.softmax(torch.randn(batch_size, num_instances, num_sites), dim=2)

        result = timer.get_expected_placements_from_index(p, area_width)

        assert result.shape == (batch_size, num_instances, 2)
        assert not torch.isnan(result).any()

        # Verify coordinates are within valid range
        assert result[:, :, 0].min() >= 0  # x >= 0
        assert result[:, :, 0].max() < area_width  # x < area_width
        assert result[:, :, 1].min() >= 0  # y >= 0
        assert result[:, :, 1].max() < area_width  # y < area_width

    def test_estimate_congestion(self):
        """Test congestion estimation."""
        timer = Timer()

        batch_size = 2
        num_instances = 10
        num_sites = 25
        area_width = 5

        p = torch.softmax(torch.randn(batch_size, num_instances, num_sites), dim=2)

        result = timer.estimate_congestion(p, area_width)

        assert result.shape == (batch_size, area_width, area_width)
        assert not torch.isnan(result).any()
        assert result.min() >= 0  # Congestion should be non-negative

    def test_calculate_path_congestion(self):
        """Test path congestion calculation."""
        timer = Timer()

        # Create a simple congestion map
        congestion_map = torch.rand(5, 5)

        start = torch.tensor([0, 0])
        end = torch.tensor([4, 4])

        result = timer.calculate_path_congestion(start, end, congestion_map)

        assert isinstance(result, float)
        assert result >= 0


class TestTimingAnalysis:
    """Test timing analysis functionality."""

    def test_calculate_timing_violation_loss_no_paths(self):
        """Test timing violation with no paths."""
        timer = Timer()

        batch_size = 2
        num_instances = 10
        num_sites = 25
        area_width = 5

        p = torch.softmax(torch.randn(batch_size, num_instances, num_sites), dim=2)

        result = timer.calculate_timing_violation_loss(p, area_width)

        assert result == 0.0  # No paths means no violations

    def test_analyze_timing_closure_no_paths(self):
        """Test timing closure analysis with no paths."""
        timer = Timer()

        batch_size = 2
        num_instances = 10
        num_sites = 25
        area_width = 5

        p = torch.softmax(torch.randn(batch_size, num_instances, num_sites), dim=2)

        result = timer.analyze_timing_closure(p, area_width)

        assert result['total_paths'] == 0
        assert result['violating_paths'] == 0
        assert result['worst_slack'] == float('inf')


class TestCongestionAware:
    """Test congestion-aware functions."""

    def test_calculate_congestion_weights(self):
        """Test congestion weight calculation."""
        timer = Timer()

        batch_size = 2
        num_instances = 5
        area_width = 5

        expected_coords = torch.rand(batch_size, num_instances, 2) * (area_width - 1)
        congestion_map = torch.rand(batch_size, area_width, area_width)

        result = timer.calculate_congestion_weights(expected_coords, congestion_map)

        assert result.shape == (batch_size, num_instances, num_instances)
        assert result.min() >= 1.0  # Weights are 1.0 + congestion

    def test_calculate_congestion_aware_hpwl(self):
        """Test congestion-aware HPWL calculation."""
        timer = Timer()

        batch_size = 2
        num_instances = 10
        num_sites = 25
        area_width = 5

        J = torch.rand(num_instances, num_instances)
        J = (J + J.T) / 2  # Make symmetric

        p = torch.softmax(torch.randn(batch_size, num_instances, num_sites), dim=2)

        result = timer.calculate_congestion_aware_hpwl(J.numpy(), p, area_width)

        assert not torch.isnan(result)


class TestComprehensiveEnergy:
    """Test comprehensive energy function."""

    def test_comprehensive_energy_function(self):
        """Test comprehensive energy function."""
        timer = Timer()

        batch_size = 2
        num_instances = 10
        num_sites = 25
        area_width = 5

        J = torch.rand(num_instances, num_instances)
        J = (J + J.T) / 2

        p = torch.softmax(torch.randn(batch_size, num_instances, num_sites), dim=2)

        total_energy, losses = timer.comprehensive_energy_function(J.numpy(), p, area_width)

        assert 'hpwl_loss' in losses
        assert 'timing_loss' in losses
        assert 'congestion_loss' in losses


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
