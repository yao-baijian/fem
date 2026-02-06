# Testing Guide

Comprehensive testing documentation for FPGA Placement FEM.

## Test Suite Overview

The package includes 50+ unit tests covering:

- **Objective functions** (HPWL, constraints)
- **Optimization algorithms** (FEM, SB solver)
- **Legalization** (overlap detection and resolution)
- **Timing analysis** (Timer functionality)
- **ML components** (parameter prediction)
- **Integration tests** (complete placement pipeline)

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_objectives.py -v

# Run specific test
pytest tests/test_objectives.py::test_hpwl_basic -v

# Run tests matching pattern
pytest tests/ -k "hpwl" -v
```

### With Coverage

```bash
# Run with coverage report
pytest tests/ --cov=fem_placer --cov-report=html

# View coverage report
open htmlcov/index.html

# Terminal coverage report
pytest tests/ --cov=fem_placer --cov-report=term-missing
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto
```

## Test Structure

### tests/test_objectives.py

Tests for QUBO objective functions.

```python
def test_hpwl_basic():
    """Test basic HPWL computation"""
    # Simple 2x2 grid, 2 instances
    J = torch.tensor([[0, 1], [1, 0]])  # Connectivity
    p = torch.eye(2).unsqueeze(0)  # Hard placement at different sites
    site_coords = torch.tensor([[0., 0.], [1., 0.]])

    hpwl = get_hpwl_loss_qubo(J, p, site_coords)

    assert hpwl > 0  # Should have non-zero wirelength

def test_constraints_satisfied():
    """Test constraint loss for valid placement"""
    # Valid: each instance at exactly one site
    p = torch.tensor([[[1., 0.], [0., 1.]]])

    loss = get_constraints_loss(p)

    assert loss < 1e-6  # Should be nearly zero

def test_constraints_violated():
    """Test constraint loss for invalid placement"""
    # Invalid: both instances partially at same site
    p = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])

    loss = get_constraints_loss(p)

    # Will have some constraint violation
    assert loss > 0
```

### tests/test_timer.py

Tests for timing-aware placement.

```python
def test_timing_based_hpwl():
    """Test timing-weighted HPWL calculation"""
    timer = Timer()

    J = torch.randn(10, 10)
    p = torch.softmax(torch.randn(1, 10, 20), dim=-1)
    timing_criticality = torch.ones_like(J)

    hpwl = timer.calculate_timing_based_hpwl(
        J, p, area_width=5, timing_criticality=timing_criticality
    )

    assert not torch.isnan(hpwl)
    assert hpwl >= 0
```

### tests/test_sb_solver.py

Tests for Simulated Bifurcation solver.

```python
def test_sb_solver_ising():
    """Test SB solver on Ising problem"""
    solver = SBSolver(mode='discrete', heated=True, device='cpu')

    # Small Ising problem
    J = torch.tensor([[0., -1.], [-1., 0.]])

    spins, energy = solver.solve_ising(J, agents=5, max_steps=1000)

    assert spins.shape[0] == 2  # Two spins
    assert isinstance(energy, float)
```

### tests/test_fpga_placement.py

Integration tests for complete pipeline.

```python
def test_placement_pipeline():
    """Test complete placement pipeline"""
    # This would use a small test DCP file
    # For testing without actual FPGA design:

    num_inst = 20
    num_site = 40

    # Create random connectivity
    J = torch.randint(0, 2, (num_inst, num_inst)).float()
    J = (J + J.t()) / 2  # Symmetric

    # Create site coordinates
    area_length = 7
    site_coords = torch.cartesian_prod(
        torch.arange(area_length, dtype=torch.float32),
        torch.arange(area_length, dtype=torch.float32)
    )

    # Create optimizer
    optimizer = FPGAPlacementOptimizer(
        num_inst=num_inst,
        num_site=site_coords.shape[0],
        coupling_matrix=J,
        site_coords_matrix=site_coords
    )

    # Optimize
    configs, energies = optimizer.optimize(
        num_trials=2,
        num_steps=100,
        dev='cpu',
        area_width=area_length
    )

    # Verify results
    assert configs.shape == (2, num_inst, site_coords.shape[0])
    assert energies.shape == (2,)
    assert torch.all(energies >= 0)
```

## Writing New Tests

### Test Structure

```python
import pytest
import torch
from fem_placer import your_function

class TestYourFeature:
    """Group related tests in a class"""

    def test_basic_functionality(self):
        """Test basic functionality"""
        result = your_function(input_data)
        assert result.shape == expected_shape

    def test_edge_case(self):
        """Test edge cases"""
        with pytest.raises(ValueError):
            your_function(invalid_input)

    @pytest.mark.parametrize("size,expected", [
        (10, 100),
        (20, 400),
    ])
    def test_parametrized(self, size, expected):
        """Test with different parameters"""
        result = your_function(size)
        assert len(result) == expected
```

### Fixtures

Use fixtures for reusable test data:

```python
@pytest.fixture
def sample_connectivity():
    """Create sample connectivity matrix"""
    return torch.randint(0, 2, (10, 10)).float()

@pytest.fixture
def sample_placement():
    """Create sample placement probabilities"""
    return torch.softmax(torch.randn(1, 10, 20), dim=-1)

def test_with_fixtures(sample_connectivity, sample_placement):
    """Test using fixtures"""
    result = compute_hpwl(sample_connectivity, sample_placement)
    assert result > 0
```

### Testing with GPU

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_functionality():
    """Test GPU-specific functionality"""
    device = 'cuda'
    data = torch.randn(100, 100, device=device)
    result = compute(data)
    assert result.device.type == 'cuda'
```

### Mocking External Dependencies

For tests that would use RapidWright:

```python
from unittest.mock import Mock, patch

def test_placer_init():
    """Test placer initialization without actual DCP file"""
    with patch('fem_placer.placer.Design.readCheckpoint') as mock_design:
        # Mock the design
        mock_design.return_value = Mock()

        placer = FpgaPlacer()
        placer.init_placement('mock.dcp', 'output.dcp')

        assert mock_design.called
```

## Test Coverage Goals

Aim for >80% coverage:

```bash
pytest tests/ --cov=fem_placer --cov-report=term

# Should show something like:
# Name                              Stmts   Miss  Cover
# -----------------------------------------------------
# fem_placer/__init__.py               15      0   100%
# fem_placer/objectives.py            120     15    88%
# fem_placer/optimizer.py              95     10    89%
# fem_placer/legalizer.py              80      8    90%
# -----------------------------------------------------
# TOTAL                               310     33    89%
```

## Continuous Integration

Tests run automatically on GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install torch rapidwright pytest pytest-cov

      - name: Run tests
        run: pytest tests/ --cov=fem_placer

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Performance Testing

### Benchmarking

```python
import time

def test_optimization_performance():
    """Benchmark optimization speed"""
    # Setup
    optimizer = create_optimizer()

    # Measure time
    start = time.time()
    configs, energies = optimizer.optimize(num_trials=10, num_steps=1000)
    elapsed = time.time() - start

    # Should complete in reasonable time
    assert elapsed < 60  # 1 minute for small problem

def test_legalization_performance():
    """Benchmark legalization speed"""
    legalizer = Legalizer(placer, 'cpu')
    coords = create_test_coords(1000)  # 1000 instances

    start = time.time()
    legal_coords, overlaps, _, _ = legalizer.legalize_placement(
        coords, torch.arange(1000)
    )
    elapsed = time.time() - start

    assert elapsed < 10  # Should be fast
```

## Debugging Tests

### Verbose Output

```bash
# Show print statements
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l

# Enter debugger on failure
pytest tests/ --pdb
```

### Selective Testing

```bash
# Only failed tests
pytest tests/ --lf

# Stop on first failure
pytest tests/ -x

# Run last failed, then all
pytest tests/ --lf --ff
```

## Best Practices

1. **Test naming**: Use descriptive names starting with `test_`
2. **One assertion per test**: Focus tests on single functionality
3. **Independence**: Tests should not depend on each other
4. **Reproducibility**: Use fixed random seeds
5. **Speed**: Keep tests fast (<1s each if possible)
6. **Documentation**: Add docstrings explaining what's tested

## Common Issues

### Floating Point Comparison

```python
# Bad: Direct comparison
assert result == 1.0

# Good: Use tolerance
assert abs(result - 1.0) < 1e-6

# Better: Use pytest.approx
assert result == pytest.approx(1.0, rel=1e-6)

# Or torch.allclose
assert torch.allclose(result, expected, rtol=1e-6)
```

### Random Tests

```python
# Set seed for reproducibility
torch.manual_seed(42)

# Or use fixtures
@pytest.fixture
def rng():
    torch.manual_seed(42)
    return torch.Generator().manual_seed(42)
```

### Cleanup

```python
# Use fixtures for setup/teardown
@pytest.fixture
def temp_file():
    f = create_temp_file()
    yield f
    cleanup(f)  # Runs after test
```

## Resources

- **pytest documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Testing Best Practices**: https://docs.python-guide.org/writing/tests/
