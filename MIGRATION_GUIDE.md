# API Migration Guide: Master → xw/refactor

This guide documents the API changes between the `master` branch and the `xw/refactor` branch for the FPGA placement toolkit.

## Overview of Changes

| Aspect | Master Branch | xw/refactor Branch |
|--------|---------------|-------------------|
| Entry Point | `FEM` class interface | Direct `FpgaPlacer` + `FPGAPlacementOptimizer` |
| Problem Setup | `FEM.from_file()` / `FEM.from_couplings()` | `parse_fpga_design()` utility |
| Solver | `Solver` class via `FEM.set_up_solver()` | `FPGAPlacementOptimizer.optimize()` |
| Grid Management | `Grid` dataclass with full API | `Grid` class + bbox dictionary |
| Net Management | `NetManager` class | `NetManager` class (same as master) |
| Algorithm | QUBO formulation | QUBO formulation (identical to master) |
| New Features | - | `SBSolver`, `Timer`, hypergraph functions |

---

## 1. Initialization

### Master Branch
```python
from FEM.interface import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.config import PlaceType, GridType

# Initialize with configuration
fpga_placer = FpgaPlacer(
    place_orientation=PlaceType.CENTERED,  # or PlaceType.IO
    grid_type=GridType.SQUARE,
    utilization_factor=0.3,
    debug=True,
    device='cuda'
)

# Load design
vivado_hpwl, opti_insts_num, site_net_num, total_net_num = fpga_placer.init_placement(
    dcp_file='post_impl.dcp',
    edf_file='optimized.edf',
    dcp_output='output.dcp'
)
```

### xw/refactor Branch
```python
from fem_placer import FpgaPlacer
from fem_placer.utils import parse_fpga_design

# Initialize (simplified constructor)
fpga_placer = FpgaPlacer(utilization_factor=0.95)

# Load design
fpga_placer.init_placement(
    dcp_file='post_impl.dcp',
    dcp_output='output.dcp'
)

# Parse design to get matrices (replaces NetManager functionality)
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)
```

### Migration Notes
- `PlaceType` and `GridType` enums removed - IO handling now via `include_io` flag
- `debug` parameter removed from constructor
- `device` parameter removed from constructor (passed to optimizer/legalizer instead)
- `edf_file` parameter removed from `init_placement()`
- Use `parse_fpga_design()` to get coupling matrices instead of accessing `net_manager`

---

## 2. Problem Setup & Solver

### Master Branch
```python
from FEM.interface import FEM

# Create FEM problem from file
case = FEM.from_file(
    problem_type='fpga_placement',
    filename='design_name',
    fpga_wrapper=fpga_placer,
    epsilon=0.03,
    q=fpga_placer.grids['logic'].area,  # Number of placement sites
    hyperedges=None,
    map_type='normal'
)

# Or from couplings directly
case = FEM.from_couplings(
    problem_type='fpga_placement',
    num_nodes=num_instances,
    num_interactions=num_nets,
    couplings=coupling_matrix,
    fpga_wrapper=fpga_placer
)

# Setup solver
case.set_up_solver(
    num_trials=5,
    num_steps=200,
    betamin=0.001,
    betamax=0.5,
    anneal='inverse',      # 'lin', 'exp', 'inverse'
    optimizer='adam',
    learning_rate=0.1,
    dev='cuda',
    dtype=torch.float32,
    seed=1,
    q=2,
    manual_grad=False,
    h_factor=0.01,
    sparse=False,
    drawer=drawer
)

# Solve
config, result = case.solve()
```

### xw/refactor Branch
```python
from fem_placer import FPGAPlacementOptimizer
from fem_placer.utils import parse_fpga_design

# Get coupling matrix and site coordinates
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)

# Create site coordinates matrix for all grid positions
area_length = fpga_placer.bbox['area_length']
logic_site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

# Create optimizer with QUBO formulation (matches master)
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=logic_site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=logic_site_coords,
    drawer=drawer,                    # Optional
    visualization_steps=[100, 500],   # Optional
    constraint_weight=num_inst / 2.0  # Matches master
)

# Optimize (solver setup + solve combined)
config, result = optimizer.optimize(
    num_trials=10,
    num_steps=1000,
    dev='cpu',
    area_width=area_length,
    betamin=0.01,
    betamax=0.5,
    anneal='inverse',
    optimizer='adam',
    learning_rate=0.1,
    h_factor=0.01,
    seed=1,
    dtype=torch.float32
)
```

### Migration Notes
- `FEM` class eliminated - use `FPGAPlacementOptimizer` directly
- Uses QUBO formulation with site coordinates (same algorithm as master)
- `q` parameter replaced by `num_site` and `site_coords_matrix`
- `sparse` and `manual_grad` parameters removed
- `constraint_weight` now in constructor, not solver setup
- `set_up_solver()` + `solve()` combined into single `optimize()` call
- Produces identical results to master branch

---

## 3. Grid & Coordinate Management

### Master Branch
```python
# Access grids
logic_grid = fpga_placer.get_grid('logic')
io_grid = fpga_placer.get_grid('io')
clock_grid = fpga_placer.get_grid('clock')

# Grid properties
area = logic_grid.area                    # grid_width * grid_height
width = logic_grid.width
height = logic_grid.height
center = (logic_grid.center_x, logic_grid.center_y)

# Coordinate conversion
grid_coords = logic_grid.to_grid_coords(real_x, real_y)
real_coords = logic_grid.to_real_coords(grid_x, grid_y)
coords_tensor = logic_grid.to_real_coords_tensor(coord_tensor)

# Instance placement on grid
logic_grid.place_instance(instance_id, x, y)
logic_grid.move_instance(instance_id, new_x, new_y, swap_allowed=True)
logic_grid.remove_instance(instance_id)

# Query grid state
pos = logic_grid.get_instance_position(instance_id)
occupant = logic_grid.get_position_occupant(x, y)
is_empty = logic_grid.is_position_empty(x, y)

# Find empty positions
empty_positions = logic_grid.find_nearest_empty(x, y, max_radius=10, k=5)

# Tensor operations
coords = logic_grid.to_coords_tensor(num_instances)
logic_grid.from_coords_tensor(coords, instance_ids)

# Get instance IDs
logic_ids, io_ids = fpga_placer.get_ids()
```

### xw/refactor Branch
```python
# Bounding box dictionary (replaces Grid class)
bbox = fpga_placer.bbox
# bbox = {
#     'start_x': int, 'end_x': int,
#     'start_y': int, 'end_y': int,
#     'area_length': int, 'area_width': int,
#     'area_size': int,
#     'estimated_sites': int,
#     'utilization': float
# }

# IO and clock areas
io_area = fpga_placer.io_area      # {'start_x', 'end_x', 'start_y', 'end_y', ...}
clock_area = fpga_placer.clock_area

# Grid dimensions for optimizer
grid_width = fpga_placer.bbox['area_length']
grid_height = fpga_placer.bbox['area_length']  # or 'area_width'

# Instance IDs (direct attributes)
num_logic = len(fpga_placer.optimizable_insts)
num_io = len(fpga_placer.fixed_insts)
logic_ids = torch.arange(num_logic)
io_ids = torch.arange(num_io)

# Coordinate conversion via objective functions
from fem_placer import get_inst_coords_from_index

# QUBO approach (only approach in refactor)
coords = get_inst_coords_from_index(inst_indices, area_width)

# Or extract from grid coordinates
logic_grid = fpga_placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(grid_coords)
```

### Migration Notes
- `Grid` class now available via `get_grid()` method (matches master)
- `get_grid('logic')` → access `fpga_placer.bbox` directly
- `get_grid('io')` → access `fpga_placer.io_area`
- Grid manipulation methods (place, move, remove) moved to `Legalizer`
- `get_ids()` removed - use `torch.arange(len(fpga_placer.optimizable_insts))`
- Coordinate conversion now via objective functions, not grid methods

---

## 4. Network & Connectivity

### Master Branch
```python
# Access NetManager
net_manager = fpga_placer.net_manager

# Connectivity matrices
insts_matrix = net_manager.insts_matrix        # Logic-to-logic [N, N]
io_insts_matrix = net_manager.io_insts_matrix  # Logic-to-IO [N, M]
net_tensor = net_manager.net_tensor            # [num_nets, num_instances]

# Net mappings
net_to_sites = net_manager.net_to_sites        # Net name -> sites
site_to_nets = net_manager.site_to_nets        # Site -> nets

# HPWL analysis
hpwl_dict = net_manager.analyze_solver_hpwl(coords, io_coords, include_io=True)
total_hpwl, hpwl_no_io = net_manager.analyze_design_hpwl(design)
single_hpwl = net_manager.get_single_instance_net_hpwl(inst_id, logic_coords, io_coords, True)

# Debug output
net_manager.save_net_debug_info('debug.txt')
net_manager.save_solver_hpwl_debug(coords, net_to_sites, 'hpwl_debug.txt')
```

### xw/refactor Branch
```python
from fem_placer.utils import parse_fpga_design

# Get coupling matrices via utility
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)
# J: Logic-to-logic connectivity [num_inst, num_inst]
# J_extend: Extended with IO [num_inst+num_io, num_inst+num_io]

# Direct connectivity access on placer
site_to_site = fpga_placer.site_to_site_connectivity  # Dict[str, Dict[str, int]]
io_to_site = fpga_placer.io_to_site_connectivity      # Dict[str, Dict[str, int]]
net_to_sites = fpga_placer.net_to_sites               # Dict[str, List[str]]

# HPWL estimation
total_hpwl, hpwl_no_io = fpga_placer.estimate_hpwl(design)
solver_hpwl = fpga_placer.estimate_solver_hpwl(coords, io_coords, include_io=True)

# HPWL loss via objective functions
from fem_placer import get_hpwl_loss_qubo, get_hpwl_loss_qubo_with_io

# QUBO approach (only approach in refactor)
hpwl_loss = get_hpwl_loss_qubo(J, p, site_coords_matrix)
hpwl_loss = get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io, logic_coords, io_coords)
```

### Migration Notes
- `NetManager` class now matches master implementation (uses same class)
- `net_manager.insts_matrix` → `J` from `parse_fpga_design()`
- HPWL calculation uses QUBO formulation

---

## 5. Legalization

### Master Branch
```python
from FEM.placement.legalizer import Legalizer

legalizer = Legalizer(placer=fpga_placer, device='cuda')

# Get IDs
logic_ids, io_ids = fpga_placer.get_ids()

# Convert solver output to real coordinates
if fpga_placer.with_io():
    logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[0][best_idx])
    io_coords = fpga_placer.get_grid('io').to_real_coords_tensor(config[1][best_idx])

    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        logic_coords, logic_ids,
        io_coords, io_ids,
        include_io=True
    )
else:
    logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[best_idx])

    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        logic_coords, logic_ids
    )

# Result: placement_legalized = [legalized_logic_coords, legalized_io_coords]
```

### xw/refactor Branch
```python
from fem_placer import Legalizer

legalizer = Legalizer(placer=fpga_placer, device='cpu')

# Get best solution
best_idx = torch.argmin(result)
best_config = config[best_idx]  # Shape: [num_inst, grid_width * grid_height]

# Convert to coordinates using objective functions
from fem_placer import infer_placements

# Infer placements using QUBO formulation
coords, hpwl_value = infer_placements(J, best_p, area_width, site_coords_matrix)
coords = coords.squeeze(0)  # Remove batch dim, shape: [num_inst, 2]

# Convert grid coordinates to real FPGA coordinates
logic_grid = fpga_placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(coords)

# Instance IDs
logic_ids = torch.arange(num_inst)

# Legalize
placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    real_coords, logic_ids,
    io_coords=None,
    io_ids=None,
    include_io=False
)

# Result: placement_legalized = [legalized_logic_coords, legalized_io_coords]
# hpwl_before, hpwl_after are dicts: {'hpwl': float, 'hpwl_no_io': float}
```

### Migration Notes
- Same `Legalizer` class interface
- Coordinate conversion now via objective functions instead of Grid methods
- `with_io()` method removed - use `include_io` parameter directly
- `get_ids()` removed - create IDs manually with `torch.arange()`
- HPWL return values changed from floats to dictionaries

---

## 6. Routing & Visualization

### Master Branch
```python
from FEM.placement.router import Router
from FEM.placement.drawer import PlacementDrawer

# Setup drawer
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Route connections
router = Router(placer=fpga_placer)
all_coords = torch.cat([placement_legalized[0], placement_legalized[1]], dim=0) \
    if fpga_placer.with_io() else placement_legalized[0]
routes = router.route_connections(fpga_placer.net_manager.insts_matrix, all_coords)

# Visualize
drawer.draw_place_and_route(
    placement_legalized[0],
    routes,
    io_coords=placement_legalized[1] if fpga_placer.with_io() else None,
    include_io=fpga_placer.with_io(),
    scale=1000,
    title_suffix='Final'
)
drawer.plot_fpga_placement_loss('loss.png')
drawer.draw_multi_step_placement('steps.png')
```

### xw/refactor Branch
```python
from fem_placer import Router, PlacementDrawer

# Setup drawer
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5, debug_mode=False)

# Route connections
router = Router(placer=fpga_placer)
routes = router.route_connections(J, placement_legalized[0])

# Visualize
drawer.draw_complete_placement(
    coords=placement_legalized[0],
    routes=routes,
    step=num_steps,
    title_suffix='Final'
)
drawer.draw_multi_step_placement()
```

### Migration Notes
- Same basic interface for `Router` and `PlacementDrawer`
- Use `J` matrix instead of `net_manager.insts_matrix`
- `draw_place_and_route()` renamed to `draw_complete_placement()`
- `plot_fpga_placement_loss()` removed (tracking done in optimizer)
- `scale` and `include_io` parameters removed from draw methods

---

## 7. New Features in xw/refactor

### Simulated Bifurcation Solver
```python
from fem_placer import SBSolver, SBPlacementSolver

# Low-level solver
solver = SBSolver(mode='discrete', heated=True, device='cpu')
spins, energy = solver.solve_ising(J, agents=10, max_steps=5000)
bits, energy = solver.solve_qubo(Q, agents=10, max_steps=5000)

# High-level placement solver (requires user-provided converters)
def my_qubo_converter(J):
    # Convert connectivity to QUBO formulation
    return Q

def my_coord_converter(spins, grid_info):
    # Convert solution spins to coordinates
    return coords

solver = SBPlacementSolver(
    mode='discrete',
    heated=True,
    qubo_converter=my_qubo_converter,
    coord_converter=my_coord_converter
)
coords, energy = solver.solve(J, grid_info={'width': w, 'height': h}, agents=10)
```

### Timer (Timing-Aware Placement)
```python
from fem_placer import Timer

timer = Timer()
timer.setup_timing_analysis(design, timing_library)

# Timing-based HPWL
timing_hpwl = timer.calculate_timing_based_hpwl(J_extend, p, area_width, timing_criticality)

# Congestion-aware placement
congestion_hpwl = timer.calculate_congestion_aware_hpwl(J_extend, p, area_width, congestion_map)

# Comprehensive optimization
p_optimized, final_cost = timer.optimize_with_all_constraints(J_extend, area_width)

# Timing closure analysis
closure_report = timer.analyze_timing_closure(p, area_width)
```

### Hypergraph Balanced Min-Cut
```python
from fem_placer import (
    expected_hyperbmincut,
    infer_hyperbmincut,
    balance_constrain,
    balance_constrain_softplus
)

# Define hyperedges
hyperedges = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]  # List of node lists

# Calculate expected cut
cut_value = expected_hyperbmincut(J, p, hyperedges)

# Balance constraints
balance_loss = balance_constrain(J, p, U_max=0.6, L_min=0.4)
balance_loss = balance_constrain_softplus(J, p, U_max=0.6, L_min=0.4, beta=1.0)

# Infer partition
partition = infer_hyperbmincut(J, p, hyperedges)
```

---

## 8. Complete Migration Example

### Master Branch (Before)
```python
from FEM.interface import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.legalizer import Legalizer
from FEM.placement.router import Router
from FEM.placement.drawer import PlacementDrawer
from FEM.placement.config import PlaceType, GridType
import torch

# 1. Initialize
fpga_placer = FpgaPlacer(
    place_orientation=PlaceType.CENTERED,
    grid_type=GridType.SQUARE,
    utilization_factor=0.4,
    device='cuda'
)
fpga_placer.init_placement(dcp_file='design.dcp', dcp_output='output.dcp')

# 2. Setup FEM
drawer = PlacementDrawer(placer=fpga_placer)
case = FEM.from_file(
    problem_type='fpga_placement',
    filename='design',
    fpga_wrapper=fpga_placer,
    q=fpga_placer.grids['logic'].area
)
case.set_up_solver(
    num_trials=5,
    num_steps=200,
    betamin=0.001,
    betamax=0.5,
    anneal='inverse',
    dev='cuda',
    drawer=drawer
)

# 3. Solve
config, result = case.solve()
best_idx = torch.argmin(result)

# 4. Legalize
legalizer = Legalizer(placer=fpga_placer, device='cuda')
logic_ids, io_ids = fpga_placer.get_ids()
logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[best_idx])
placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    logic_coords, logic_ids
)

# 5. Route & Draw
router = Router(placer=fpga_placer)
routes = router.route_connections(fpga_placer.net_manager.insts_matrix, placement_legalized[0])
drawer.draw_place_and_route(placement_legalized[0], routes, None, False, 1000, 'Final')
```

### xw/refactor Branch (After)
```python
from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
    Legalizer,
    Router,
    PlacementDrawer,
    infer_placements,
    get_inst_coords_from_index
)
from fem_placer.utils import parse_fpga_design
import torch

# 1. Initialize
fpga_placer = FpgaPlacer(utilization_factor=0.4)
fpga_placer.init_placement(dcp_file='design.dcp', dcp_output='output.dcp')

# 2. Parse design & create optimizer
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)
drawer = PlacementDrawer(placer=fpga_placer)

optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    coupling_matrix=J,
    drawer=drawer,
    constraint_weight=1.0
)

# 3. Optimize
grid_size = fpga_placer.bbox['area_length']
config, result = optimizer.optimize(
    num_trials=5,
    num_steps=200,
    dev='cpu',
    grid_width=grid_size,
    grid_height=grid_size,
    betamin=0.001,
    betamax=0.5,
    anneal='inverse'
)

# 4. Extract best solution using QUBO inference
best_idx = torch.argmin(result)
grid_coords = config[best_idx]  # Grid coordinates
logic_grid = fpga_placer.get_grid('logic')
coords = logic_grid.to_real_coords_tensor(grid_coords)  # Convert to real coordinates

# 5. Legalize
legalizer = Legalizer(placer=fpga_placer, device='cpu')
logic_ids = torch.arange(num_inst)
placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    coords, logic_ids
)

# 6. Route & Draw
router = Router(placer=fpga_placer)
routes = router.route_connections(J, placement_legalized[0])
drawer.draw_complete_placement(placement_legalized[0], routes, 200, 'Final')
```

---

## 9. Quick Reference Table

| Master Branch | xw/refactor Branch |
|--------------|-------------------|
| `FEM.from_file(problem_type='fpga_placement', ...)` | `FPGAPlacementOptimizer(num_inst, coupling_matrix, ...)` |
| `case.set_up_solver(...)` | Parameters passed to `optimizer.optimize(...)` |
| `case.solve()` | `optimizer.optimize(...)` |
| `fpga_placer.grids['logic']` | `fpga_placer.bbox` |
| `fpga_placer.get_grid('logic').area` | `fpga_placer.bbox['area_size']` |
| `fpga_placer.net_manager.insts_matrix` | `J` from `parse_fpga_design()` |
| `fpga_placer.with_io()` | `include_io` parameter |
| `fpga_placer.get_ids()` | `torch.arange(len(fpga_placer.optimizable_insts))` |
| `grid.to_real_coords_tensor(config)` | `grid.to_real_coords_tensor(config)` (same) |
| `drawer.draw_place_and_route(...)` | `drawer.draw_complete_placement(...)` |
| `PlaceType.CENTERED` / `PlaceType.IO` | Removed (use `include_io` flag) |
| `GridType.SQUARE` / `GridType.RECTAN` | Removed |
| N/A | `SBSolver`, `SBPlacementSolver` |
| N/A | `Timer` |
| N/A | Hypergraph min-cut functions |

---

## 10. Import Statement Changes

### Master Branch
```python
from FEM.interface import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.legalizer import Legalizer
from FEM.placement.router import Router
from FEM.placement.drawer import PlacementDrawer
from FEM.placement.grid import Grid
from FEM.placement.net import NetManager
from FEM.placement.config import PlaceType, GridType
from FEM.placement.logger import INFO, DEBUG, WARNING, ERROR
from FEM.customized_problem.fpga_placement import (
    get_hpwl_loss_qubo,
    get_constraints_loss,
    expected_fpga_placement,
    infer_placements
)
```

### xw/refactor Branch
```python
from fem_placer import (
    # Core classes
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    FPGAPlacementOptimizer,
    Timer,
    SBSolver,
    SBPlacementSolver,

    # QUBO functions (only approach in refactor)
    get_inst_coords_from_index,
    get_hpwl_loss_qubo,
    get_hpwl_loss_qubo_with_io,
    get_constraints_loss,
    get_constraints_loss_with_io,
    expected_fpga_placement,
    infer_placements,

    # Hypergraph functions
    expected_hyperbmincut,
    infer_hyperbmincut,
    balance_constrain,
)
from fem_placer.utils import parse_fpga_design
```
