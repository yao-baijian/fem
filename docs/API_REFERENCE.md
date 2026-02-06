# API Reference

Complete API documentation for the FPGA Placement FEM package.

## Table of Contents

- [Core Classes](#core-classes)
  - [FpgaPlacer](#fpgaplacer)
  - [FPGAPlacementOptimizer](#fpgaplacementoptimizer)
  - [Legalizer](#legalizer)
  - [Router](#router)
  - [PlacementDrawer](#placementdrawer)
  - [Timer](#timer)
- [Objective Functions](#objective-functions)
  - [Coordinate Functions](#coordinate-functions)
  - [HPWL Functions](#hpwl-functions)
  - [Constraint Functions](#constraint-functions)
  - [Placement Functions](#placement-functions)
- [Hypergraph Functions](#hypergraph-functions)
- [Utility Functions](#utility-functions)

---

## Core Classes

### FpgaPlacer

Main interface for loading and managing FPGA designs using RapidWright.

#### Constructor

```python
FpgaPlacer(utilization_factor=0.95)
```

**Parameters:**
- `utilization_factor` (float, optional): Target utilization for the placement area. Default: 0.95

**Attributes:**
- `design`: RapidWright design object
- `optimizable_insts`: List of logic instances to be placed
- `fixed_insts`: List of fixed instances (IO, clock, etc.)
- `bbox` (dict): Bounding box information
  - `'start_x'` (int): Starting X coordinate
  - `'end_x'` (int): Ending X coordinate
  - `'start_y'` (int): Starting Y coordinate
  - `'end_y'` (int): Ending Y coordinate
  - `'area_length'` (int): Grid width/height
  - `'area_size'` (int): Total grid area
  - `'estimated_sites'` (int): Estimated available sites
  - `'utilization'` (float): Actual utilization
- `io_area` (dict): IO region bounding box
- `clock_area` (dict): Clock region bounding box
- `site_to_site_connectivity` (dict): Logic-to-logic connectivity map
- `io_to_site_connectivity` (dict): IO-to-logic connectivity map
- `net_to_sites` (dict): Net name to sites mapping

#### Methods

##### init_placement()

```python
init_placement(dcp_file: str, dcp_output: str) -> None
```

Load an FPGA design from a DCP file.

**Parameters:**
- `dcp_file` (str): Path to input DCP file
- `dcp_output` (str): Path for output DCP file

**Example:**
```python
placer = FpgaPlacer(utilization_factor=0.4)
placer.init_placement('design.dcp', 'output.dcp')
```

##### get_grid()

```python
get_grid(grid_type: str) -> Grid
```

Get a grid manager for coordinate conversion.

**Parameters:**
- `grid_type` (str): Type of grid - 'logic', 'io', or 'clock'

**Returns:**
- `Grid`: Grid object for coordinate conversion

**Example:**
```python
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(grid_coords)
```

##### estimate_hpwl()

```python
estimate_hpwl(design) -> Tuple[float, float]
```

Estimate HPWL from the design.

**Returns:**
- `tuple`: (total_hpwl, hpwl_without_io)

##### estimate_solver_hpwl()

```python
estimate_solver_hpwl(logic_coords: torch.Tensor,
                     io_coords: torch.Tensor = None,
                     include_io: bool = False) -> Dict[str, float]
```

Estimate HPWL from solver coordinates.

**Parameters:**
- `logic_coords` (torch.Tensor): Logic instance coordinates [N, 2]
- `io_coords` (torch.Tensor, optional): IO instance coordinates
- `include_io` (bool): Whether to include IO connections

**Returns:**
- `dict`: {'hpwl': total_hpwl, 'hpwl_no_io': hpwl_without_io}

---

### FPGAPlacementOptimizer

Free energy minimization optimizer for FPGA placement using QUBO formulation.

#### Constructor

```python
FPGAPlacementOptimizer(
    num_inst: int,
    num_site: int,
    coupling_matrix: torch.Tensor,
    site_coords_matrix: torch.Tensor,
    drawer: PlacementDrawer = None,
    visualization_steps: List[int] = None,
    constraint_weight: float = None
)
```

**Parameters:**
- `num_inst` (int): Number of instances to place
- `num_site` (int): Number of placement sites
- `coupling_matrix` (torch.Tensor): Connectivity matrix [N, N]
- `site_coords_matrix` (torch.Tensor): Site coordinates [num_site, 2]
- `drawer` (PlacementDrawer, optional): Visualization tool
- `visualization_steps` (list, optional): Steps at which to visualize
- `constraint_weight` (float, optional): Weight for constraint violations. Default: num_inst / 2.0

#### Methods

##### optimize()

```python
optimize(
    num_trials: int = 10,
    num_steps: int = 1000,
    dev: str = 'cpu',
    area_width: int = None,
    betamin: float = 0.01,
    betamax: float = 0.5,
    anneal: str = 'inverse',
    optimizer: str = 'adam',
    learning_rate: float = 0.1,
    h_factor: float = 0.01,
    seed: int = 1,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]
```

Run placement optimization.

**Parameters:**
- `num_trials` (int): Number of independent trials
- `num_steps` (int): Optimization steps per trial
- `dev` (str): Device - 'cpu' or 'cuda'
- `area_width` (int): Grid width for coordinate conversion
- `betamin` (float): Initial inverse temperature
- `betamax` (float): Final inverse temperature
- `anneal` (str): Annealing schedule - 'lin', 'exp', or 'inverse'
- `optimizer` (str): Optimizer type - 'adam', 'sgd', etc.
- `learning_rate` (float): Learning rate
- `h_factor` (float): Weight for local fields
- `seed` (int): Random seed
- `dtype` (torch.dtype): Data type for tensors

**Returns:**
- `tuple`: (configs, energies)
  - `configs` (torch.Tensor): Placement probabilities [num_trials, num_inst, num_site]
  - `energies` (torch.Tensor): Final energies [num_trials]

**Example:**
```python
optimizer = FPGAPlacementOptimizer(
    num_inst=100,
    num_site=200,
    coupling_matrix=J,
    site_coords_matrix=site_coords,
    constraint_weight=1.0
)

configs, energies = optimizer.optimize(
    num_trials=10,
    num_steps=1000,
    dev='cpu',
    area_width=20,
    betamin=0.01,
    betamax=0.5,
    anneal='inverse'
)

best_idx = torch.argmin(energies)
best_config = configs[best_idx]
```

---

### Legalizer

Resolves placement overlaps and legalizes the placement solution.

#### Constructor

```python
Legalizer(placer: FpgaPlacer, device: str = 'cpu')
```

**Parameters:**
- `placer` (FpgaPlacer): FPGA placer instance
- `device` (str): Device for tensor operations

#### Methods

##### legalize_placement()

```python
legalize_placement(
    logic_coords: torch.Tensor,
    logic_ids: torch.Tensor,
    io_coords: torch.Tensor = None,
    io_ids: torch.Tensor = None,
    include_io: bool = False
) -> Tuple[List[torch.Tensor], int, Dict[str, float], Dict[str, float]]
```

Legalize placement by resolving overlaps.

**Parameters:**
- `logic_coords` (torch.Tensor): Logic instance coordinates [N, 2]
- `logic_ids` (torch.Tensor): Logic instance IDs [N]
- `io_coords` (torch.Tensor, optional): IO coordinates
- `io_ids` (torch.Tensor, optional): IO instance IDs
- `include_io` (bool): Whether to include IO in HPWL calculation

**Returns:**
- `tuple`:
  - `placement_legalized` (list): [logic_coords, io_coords] (legalized)
  - `overlap` (int): Number of overlapping instances
  - `hpwl_before` (dict): HPWL before legalization
  - `hpwl_after` (dict): HPWL after legalization

**Example:**
```python
legalizer = Legalizer(placer=fpga_placer, device='cpu')
logic_ids = torch.arange(num_inst)

placement, overlaps, hpwl_before, hpwl_after = legalizer.legalize_placement(
    logic_coords=coords,
    logic_ids=logic_ids
)

print(f"Overlaps: {overlaps}")
print(f"HPWL after: {hpwl_after['hpwl_no_io']:.2f}")
```

---

### Router

Provides routing functionality for visualization.

#### Constructor

```python
Router(placer: FpgaPlacer)
```

**Parameters:**
- `placer` (FpgaPlacer): FPGA placer instance

#### Methods

##### route_connections()

```python
route_connections(
    coupling_matrix: torch.Tensor,
    coords: torch.Tensor
) -> List[List[Tuple[int, int]]]
```

Route connections using Manhattan routing.

**Parameters:**
- `coupling_matrix` (torch.Tensor): Connectivity matrix [N, N]
- `coords` (torch.Tensor): Instance coordinates [N, 2]

**Returns:**
- `list`: List of routes, each route is a list of (x, y) waypoints

**Example:**
```python
router = Router(placer=fpga_placer)
routes = router.route_connections(J, placement_legalized[0])
```

---

### PlacementDrawer

Visualization tools for FPGA placement.

#### Constructor

```python
PlacementDrawer(
    placer: FpgaPlacer,
    num_subplots: int = 5,
    debug_mode: bool = False
)
```

**Parameters:**
- `placer` (FpgaPlacer): FPGA placer instance
- `num_subplots` (int): Number of subplots for multi-step visualization
- `debug_mode` (bool): Enable debug output

#### Methods

##### draw_place_and_route()

```python
draw_place_and_route(
    logic_coords: torch.Tensor,
    routes: List,
    io_coords: torch.Tensor = None,
    include_io: bool = False,
    iteration: int = 0,
    title_suffix: str = ''
) -> None
```

Draw placement and routing visualization.

**Parameters:**
- `logic_coords` (torch.Tensor): Logic instance coordinates [N, 2]
- `routes` (list): List of routes from Router
- `io_coords` (torch.Tensor, optional): IO coordinates
- `include_io` (bool): Whether to include IO instances
- `iteration` (int): Current iteration number
- `title_suffix` (str): Additional title text

##### draw_multi_step_placement()

```python
draw_multi_step_placement(output_file: str = 'placement_steps.png') -> None
```

Draw multi-step placement visualization.

**Parameters:**
- `output_file` (str): Output filename

**Example:**
```python
drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5)

drawer.draw_place_and_route(
    logic_coords=placement_legalized[0],
    routes=routes,
    io_coords=None,
    include_io=False,
    iteration=1000,
    title_suffix='Final'
)

drawer.draw_multi_step_placement('steps.png')
```

---

### Timer

Timing-aware and congestion-aware placement analysis.

#### Constructor

```python
Timer()
```

#### Methods

##### setup_timing_analysis()

```python
setup_timing_analysis(design, timing_library) -> None
```

Setup timing analysis with design and timing library.

##### calculate_timing_based_hpwl()

```python
calculate_timing_based_hpwl(
    J: torch.Tensor,
    p: torch.Tensor,
    area_width: int,
    timing_criticality: torch.Tensor
) -> torch.Tensor
```

Calculate timing-weighted HPWL.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Placement probabilities
- `area_width` (int): Grid width
- `timing_criticality` (torch.Tensor): Timing criticality weights

**Returns:**
- `torch.Tensor`: Timing-weighted HPWL

##### calculate_congestion_aware_hpwl()

```python
calculate_congestion_aware_hpwl(
    J: torch.Tensor,
    p: torch.Tensor,
    area_width: int,
    congestion_map: torch.Tensor
) -> torch.Tensor
```

Calculate congestion-weighted HPWL.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Placement probabilities
- `area_width` (int): Grid width
- `congestion_map` (torch.Tensor): Congestion penalties per site

**Returns:**
- `torch.Tensor`: Congestion-weighted HPWL

##### optimize_with_all_constraints()

```python
optimize_with_all_constraints(
    J: torch.Tensor,
    area_width: int,
    timing_criticality: torch.Tensor = None,
    congestion_map: torch.Tensor = None,
    num_steps: int = 1000
) -> Tuple[torch.Tensor, float]
```

Optimize placement with timing and congestion constraints.

**Returns:**
- `tuple`: (optimized_probabilities, final_cost)

---

## Objective Functions

### Coordinate Functions

##### get_inst_coords_from_index()

```python
get_inst_coords_from_index(
    indices: torch.Tensor,
    area_width: int
) -> torch.Tensor
```

Convert site indices to coordinates.

**Parameters:**
- `indices` (torch.Tensor): Site indices [N]
- `area_width` (int): Grid width

**Returns:**
- `torch.Tensor`: Coordinates [N, 2]

##### get_site_distance_matrix()

```python
get_site_distance_matrix(
    site_coords_matrix: torch.Tensor
) -> torch.Tensor
```

Compute pairwise distance matrix between sites.

**Parameters:**
- `site_coords_matrix` (torch.Tensor): Site coordinates [num_site, 2]

**Returns:**
- `torch.Tensor`: Distance matrix [num_site, num_site]

---

### HPWL Functions

##### get_hpwl_loss_qubo()

```python
get_hpwl_loss_qubo(
    J: torch.Tensor,
    p: torch.Tensor,
    site_coords_matrix: torch.Tensor
) -> torch.Tensor
```

Calculate expected HPWL using QUBO formulation.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix [N, N]
- `p` (torch.Tensor): Placement probabilities [batch, N, num_site]
- `site_coords_matrix` (torch.Tensor): Site coordinates [num_site, 2]

**Returns:**
- `torch.Tensor`: Expected HPWL [batch]

##### get_hpwl_loss_qubo_with_io()

```python
get_hpwl_loss_qubo_with_io(
    J_LL: torch.Tensor,
    J_LI: torch.Tensor,
    p_logic: torch.Tensor,
    p_io: torch.Tensor,
    logic_coords: torch.Tensor,
    io_coords: torch.Tensor
) -> torch.Tensor
```

Calculate HPWL including IO connections.

**Parameters:**
- `J_LL` (torch.Tensor): Logic-to-logic connectivity
- `J_LI` (torch.Tensor): Logic-to-IO connectivity
- `p_logic` (torch.Tensor): Logic placement probabilities
- `p_io` (torch.Tensor): IO placement probabilities
- `logic_coords` (torch.Tensor): Logic site coordinates
- `io_coords` (torch.Tensor): IO site coordinates

**Returns:**
- `torch.Tensor`: Total expected HPWL

---

### Constraint Functions

##### get_constraints_loss()

```python
get_constraints_loss(p: torch.Tensor) -> torch.Tensor
```

Calculate constraint violation loss (one instance per site, one site per instance).

**Parameters:**
- `p` (torch.Tensor): Placement probabilities [batch, N, num_site]

**Returns:**
- `torch.Tensor`: Constraint loss [batch]

---

### Placement Functions

##### expected_fpga_placement()

```python
expected_fpga_placement(
    J: torch.Tensor,
    p: torch.Tensor,
    site_coords_matrix: torch.Tensor,
    constraint_weight: float = 1.0
) -> torch.Tensor
```

Calculate total placement energy (HPWL + constraints).

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Placement probabilities
- `site_coords_matrix` (torch.Tensor): Site coordinates
- `constraint_weight` (float): Weight for constraint violations

**Returns:**
- `torch.Tensor`: Total energy

##### infer_placements()

```python
infer_placements(
    J: torch.Tensor,
    p: torch.Tensor,
    area_width: int,
    site_coords_matrix: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Infer hard placement from probabilities.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Placement probabilities [batch, N, num_site]
- `area_width` (int): Grid width
- `site_coords_matrix` (torch.Tensor, optional): Site coordinates

**Returns:**
- `tuple`: (coordinates [batch, N, 2], hpwl [batch])

**Example:**
```python
# After optimization
best_idx = torch.argmin(energies)
best_p = configs[best_idx]

# Infer hard placement
coords, hpwl = infer_placements(J, best_p, area_width, site_coords_matrix)
```

---

## Hypergraph Functions

##### expected_hyperbmincut()

```python
expected_hyperbmincut(
    J: torch.Tensor,
    p: torch.Tensor,
    hyperedges: List[List[int]]
) -> torch.Tensor
```

Calculate expected hypergraph min-cut.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Node probabilities [batch, N, 2] (2 partitions)
- `hyperedges` (list): List of hyperedges, each is a list of node indices

**Returns:**
- `torch.Tensor`: Expected cut value

##### infer_hyperbmincut()

```python
infer_hyperbmincut(
    J: torch.Tensor,
    p: torch.Tensor,
    hyperedges: List[List[int]]
) -> Tuple[torch.Tensor, float]
```

Infer partition from probabilities.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Node probabilities
- `hyperedges` (list): List of hyperedges

**Returns:**
- `tuple`: (partition assignment, cut value)

##### balance_constrain()

```python
balance_constrain(
    J: torch.Tensor,
    p: torch.Tensor,
    U_max: float = 0.6,
    L_min: float = 0.4
) -> torch.Tensor
```

Calculate balance constraint loss.

**Parameters:**
- `J` (torch.Tensor): Connectivity matrix
- `p` (torch.Tensor): Node probabilities
- `U_max` (float): Upper bound for partition ratio
- `L_min` (float): Lower bound for partition ratio

**Returns:**
- `torch.Tensor`: Balance constraint loss

**Example:**
```python
# Define hypergraph
hyperedges = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]

# Calculate expected cut
cut = expected_hyperbmincut(J, p, hyperedges)

# Add balance constraint
balance_loss = balance_constrain(J, p, U_max=0.6, L_min=0.4)

# Total objective
total_loss = cut + balance_loss
```

---

## Utility Functions

##### parse_fpga_design()

```python
parse_fpga_design(
    placer: FpgaPlacer
) -> Tuple[int, int, torch.Tensor, torch.Tensor]
```

Parse FPGA design to extract connectivity matrices.

**Parameters:**
- `placer` (FpgaPlacer): Initialized placer with loaded design

**Returns:**
- `tuple`:
  - `num_inst` (int): Number of logic instances
  - `num_site` (int): Number of available sites
  - `J` (torch.Tensor): Logic-to-logic connectivity [N, N]
  - `J_extend` (torch.Tensor): Extended with IO [N+M, N+M]

**Example:**
```python
from fem_placer.utils import parse_fpga_design

placer = FpgaPlacer()
placer.init_placement('design.dcp', 'output.dcp')

num_inst, num_site, J, J_extend = parse_fpga_design(placer)
```

---

## Constants and Enumerations

### Annealing Schedules

- `'lin'`: Linear annealing
- `'exp'`: Exponential annealing
- `'inverse'`: Inverse annealing (recommended)

### Optimizers

- `'adam'`: Adam optimizer (recommended)
- `'sgd'`: Stochastic gradient descent
- `'adamw'`: Adam with weight decay

### SB Modes

- `'discrete'`: Discrete dynamics
- `'ballistic'`: Ballistic dynamics

---

## Type Definitions

### Common Tensor Shapes

- Connectivity matrix `J`: `[num_inst, num_inst]`
- Extended connectivity `J_extend`: `[num_inst + num_io, num_inst + num_io]`
- Placement probabilities `p`: `[batch, num_inst, num_site]`
- Coordinates: `[num_inst, 2]`
- Site coordinates matrix: `[num_site, 2]`

### Return Value Structures

**HPWL Dictionary:**
```python
{
    'hpwl': float,        # Total HPWL including IO
    'hpwl_no_io': float   # HPWL without IO connections
}
```

**Bounding Box Dictionary:**
```python
{
    'start_x': int,
    'end_x': int,
    'start_y': int,
    'end_y': int,
    'area_length': int,
    'area_width': int,
    'area_size': int,
    'estimated_sites': int,
    'utilization': float
}
```

---

## Error Handling

Most functions will raise standard Python exceptions:

- `ValueError`: Invalid parameter values or incompatible tensor shapes
- `RuntimeError`: Runtime errors during optimization or CUDA operations
- `FileNotFoundError`: Missing DCP or design files
- `KeyError`: Missing required dictionary keys

Always wrap file operations in try-except blocks:

```python
try:
    placer = FpgaPlacer()
    placer.init_placement('design.dcp', 'output.dcp')
except FileNotFoundError as e:
    print(f"Design file not found: {e}")
except Exception as e:
    print(f"Error loading design: {e}")
```

---

## Performance Tips

1. **Use CUDA when available:**
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   optimizer.optimize(dev=device, ...)
   ```

2. **Batch multiple trials:**
   ```python
   # Run 10 trials in parallel
   optimizer.optimize(num_trials=10, ...)
   ```

3. **Adjust num_steps based on design size:**
   - Small designs (<100 instances): 500-1000 steps
   - Medium designs (100-500): 1000-2000 steps
   - Large designs (>500): 2000-5000 steps

4. **Use appropriate constraint weights:**
   - Default: `num_inst / 2.0`
   - Increase for better legality
   - Decrease for better HPWL

---

## Version History

- **v0.1.0**: Initial release with QUBO formulation, SB solver, and timer
