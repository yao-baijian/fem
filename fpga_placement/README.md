# FPGA Placement Package

This package provides FPGA placement functionality as an **independent extension** to the FEM framework. It uses FEM's `customize` interface without modifying any FEM core files.

## Architecture

### Design Principles

1. **Clean Separation**: FEM framework remains completely unmodified
2. **Use Customize Interface**: Leverage FEM's built-in extensibility mechanism
3. **Independent Package**: All FPGA logic is self-contained in `fpga_placement/`
4. **Easy Maintenance**: Can easily sync with upstream FEM updates

### Directory Structure

```
fpga_placement/
├── __init__.py           # Package exports
├── placer.py             # FpgaPlacer class (RapidWright interface)
├── objectives.py         # FPGA-specific objective functions (HPWL, constraints)
├── drawer.py             # Visualization tools
├── legalizer.py          # Placement legalization
├── router.py             # Connection routing
└── utils.py              # Utility functions (design parsing, etc.)
```

## Usage

### Basic Example

```python
from FEM import FEM
from fpga_placement import (
    FpgaPlacer,
    expected_fpga_placement_xy,
    infer_placements_xy
)
from fpga_placement.utils import parse_fpga_design

# Initialize FPGA placer
fpga_wrapper = FpgaPlacer()
fpga_wrapper.init_placement('./design.dcp', 'output.dcp')

# Parse design
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)

# Define customize functions
def customize_expected_func(coupling_matrix, p_list):
    p_x, p_y = p_list
    return expected_fpga_placement_xy(coupling_matrix, p_x, p_y)

def customize_infer_func(coupling_matrix, p_list):
    p_x, p_y = p_list
    return infer_placements_xy(coupling_matrix, p_x, p_y)

# Create FEM problem using customize interface
fem_problem = FEM.from_couplings(
    'customize',  # Use FEM's customize type
    num_inst,
    num_inst * (num_inst - 1) // 2,
    J,
    customize_expected_func=customize_expected_func,
    customize_infer_func=customize_infer_func
)

# Solve
fem_problem.set_up_solver(num_trials=10, num_steps=1000, dev='cpu')
config, result = fem_problem.solve()
```

## Key Features

### 1. X-Y Coordinate Separation

FPGA placement uses separate probability distributions for X and Y coordinates:
- `p_x`: `[batch, num_instances, grid_width]`
- `p_y`: `[batch, num_instances, grid_height]`

This allows efficient modeling of 2D placement space.

### 2. Objective Functions

- **HPWL (Half-Perimeter Wirelength)**: Minimizes total wire length
- **Site Constraints**: Ensures valid placements without overlaps

### 3. Post-Processing

- **Legalization**: Resolves overlaps and snaps to valid sites
- **Routing**: Computes actual wire paths for visualization

## Advantages of This Approach

1. **No FEM Pollution**: FEM core remains pristine and can be updated from upstream
2. **Clear Boundaries**: FPGA logic is isolated in its own package
3. **Reusable Pattern**: Other domain-specific problems can follow the same structure
4. **Maintainable**: Changes to FPGA logic don't affect FEM or other problems

## Comparison with Previous Approach

### ❌ Old Approach (Invasive)
- Added `fpga_placement` as a new problem type in FEM core
- Modified `problem.py`, `solver_fem.py`, `interface.py` with special cases
- Tightly coupled FPGA logic with FEM internals
- Difficult to sync with upstream FEM updates

### ✅ New Approach (Clean)
- Use FEM's `customize` interface (already exists!)
- All FPGA code in separate `fpga_placement/` package
- Zero modifications to FEM core files
- Easy to maintain and extend

## Testing

Run the refactored test:

```bash
python tests/test_fpga_placement_refactored.py
```

## Dependencies

- FEM framework (from https://github.com/Fanerst/FEM)
- RapidWright (for FPGA design handling)
- PyTorch
- Matplotlib (for visualization)
