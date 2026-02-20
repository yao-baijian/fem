# Code Explained

Deep dive into the implementation of FPGA Placement FEM.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
3. [QUBO Implementation](#qubo-implementation)
4. [Optimization Loop](#optimization-loop)
5. [Legalization Algorithm](#legalization-algorithm)
6. [Design Patterns](#design-patterns)
7. [Performance Considerations](#performance-considerations)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────┐
│  DCP File   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│            FpgaPlacer (placer.py)               │
│  - Load design via RapidWright                  │
│  - Extract instances and sites                  │
│  - Build connectivity graph                     │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│         parse_fpga_design (utils.py)            │
│  - Extract coupling matrix J                    │
│  - Create extended matrix with IO               │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│    FPGAPlacementOptimizer (optimizer.py)        │
│  - Initialize probability distribution          │
│  - Run FEM optimization loop                    │
│  - Return best configuration                    │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│          Legalizer (legalizer.py)               │
│  - Detect overlaps                              │
│  - Resolve conflicts                            │
│  - Return legal placement                       │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│      Router & Drawer (router.py, drawer.py)    │
│  - Route connections                            │
│  - Visualize placement                          │
└─────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. placer.py - FPGA Design Interface

**Purpose**: Bridge between RapidWright and our optimizer.

**Key Class: FpgaPlacer**

```python
class FpgaPlacer:
    def __init__(self, utilization_factor=0.95):
        self.utilization_factor = utilization_factor
        self.design = None
        self.optimizable_insts = []  # Logic instances
        self.fixed_insts = []         # IO, clock, etc.
```

**Design Loading Process**:

```python
def init_placement(self, dcp_file: str, dcp_output: str):
    # 1. Load DCP using RapidWright
    self.design = Design.readCheckpoint(dcp_file)

    # 2. Categorize instances
    for cell in self.design.getCells():
        if is_logic_cell(cell):
            self.optimizable_insts.append(cell)
        else:
            self.fixed_insts.append(cell)

    # 3. Calculate bounding box
    self.bbox = self._calculate_placement_area()

    # 4. Build connectivity
    self._build_connectivity_graph()
```

**Bounding Box Calculation**:

```python
def _calculate_placement_area(self):
    # Get device dimensions
    total_sites = count_available_sites(self.design)

    # Calculate required area based on utilization
    needed_sites = len(self.optimizable_insts) / self.utilization_factor

    # Create square grid
    area_length = int(math.sqrt(needed_sites))

    return {
        'area_length': area_length,
        'area_size': area_length * area_length,
        'utilization': len(self.optimizable_insts) / needed_sites
    }
```

**Connectivity Graph**:

```python
def _build_connectivity_graph(self):
    # site_to_site: {site_name: {connected_site: weight}}
    self.site_to_site_connectivity = {}

    for net in self.design.getNets():
        sites = [pin.getSite() for pin in net.getPins()]

        # Create pairwise connections
        for i, site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                if site1 not in self.site_to_site_connectivity:
                    self.site_to_site_connectivity[site1] = {}

                # Increment connection weight
                self.site_to_site_connectivity[site1][site2] = \
                    self.site_to_site_connectivity[site1].get(site2, 0) + 1
```

---

### 2. objectives.py - QUBO Formulation

**Purpose**: Implement differentiable objective functions for optimization.

#### Coordinate Conversion

```python
def get_inst_coords_from_index(indices: torch.Tensor, area_width: int):
    """
    Convert 1D site indices to 2D grid coordinates.

    indices: [N] - site index for each instance
    area_width: grid width

    Returns: [N, 2] - (x, y) coordinates
    """
    x = indices % area_width
    y = indices // area_width
    return torch.stack([x, y], dim=-1)
```

**Example**:
```
Grid (area_width=4):
  0  1  2  3
  4  5  6  7
  8  9 10 11
 12 13 14 15

indices = [5, 10]
→ x = [5%4, 10%4] = [1, 2]
→ y = [5//4, 10//4] = [1, 2]
→ coords = [[1,1], [2,2]]
```

#### Expected Coordinates

```python
def get_expected_placements_from_index(
    p: torch.Tensor,           # [batch, N, num_site]
    site_coords_matrix: torch.Tensor  # [num_site, 2]
):
    """
    Calculate expected coordinates from probability distribution.

    E[coords_i] = Σ_s p[i,s] * site_coords[s]

    This is just a matrix multiplication:
    expected_coords = p @ site_coords_matrix
    """
    return torch.matmul(p, site_coords_matrix)
```

**Intuition**: Each instance has a probability distribution over sites. The expected coordinate is the weighted average of all site coordinates.

**Example**:
```python
# Instance has 50% chance at site (0,0) and 50% at site (2,2)
p = torch.tensor([[0.5, 0, 0, 0.5]])  # Probabilities for 4 sites
site_coords = torch.tensor([[0,0], [1,0], [0,1], [2,2]])

expected = p @ site_coords  # [0.5*[0,0] + 0.5*[2,2]] = [1,1]
```

#### HPWL Calculation

```python
def get_hpwl_loss_qubo(
    J: torch.Tensor,              # [N, N] connectivity
    p: torch.Tensor,              # [batch, N, num_site] probabilities
    site_coords_matrix: torch.Tensor  # [num_site, 2]
):
    """
    Calculate expected HPWL using QUBO formulation.

    E[HPWL] = Σ_{i,j} J[i,j] * E[distance(i,j)]
            ≈ Σ_{i,j} J[i,j] * ||E[coords_i] - E[coords_j]||
    """
    # Expected coordinates: [batch, N, 2]
    expected_coords = torch.matmul(p, site_coords_matrix)

    # Pairwise coordinate differences
    # expected_coords: [batch, N, 2]
    # unsqueeze(-2): [batch, N, 1, 2]
    # unsqueeze(-3): [batch, 1, N, 2]
    # Result: [batch, N, N, 2]
    coord_diff = expected_coords.unsqueeze(-2) - expected_coords.unsqueeze(-3)

    # L2 distances: [batch, N, N]
    distances = torch.sqrt((coord_diff ** 2).sum(dim=-1) + 1e-8)

    # Weighted by connectivity: [batch]
    hpwl = (J.unsqueeze(0) * distances).sum(dim=(-2, -1))

    return hpwl
```

**Step-by-Step Example**:

```python
# 2 instances, 4 sites
N = 2
num_site = 4

J = torch.tensor([[0, 1],   # Connectivity matrix
                  [1, 0]])

p = torch.tensor([[[0.8, 0.1, 0.1, 0],    # Instance 0 probabilities
                   [0, 0.2, 0.3, 0.5]]])  # Instance 1 probabilities

site_coords = torch.tensor([[0, 0],  # Site coordinates
                           [1, 0],
                           [0, 1],
                           [1, 1]])

# Step 1: Expected coordinates
expected_coords = p @ site_coords
# Instance 0: 0.8*[0,0] + 0.1*[1,0] + 0.1*[0,1] = [0.1, 0.1]
# Instance 1: 0.2*[1,0] + 0.3*[0,1] + 0.5*[1,1] = [0.7, 0.8]
# expected_coords = [[0.1, 0.1], [0.7, 0.8]]

# Step 2: Coordinate differences
# coord_diff[0,1] = [0.1,0.1] - [0.7,0.8] = [-0.6, -0.7]
# coord_diff[1,0] = [0.7,0.8] - [0.1,0.1] = [0.6, 0.7]

# Step 3: Distances
# distance[0,1] = sqrt(0.6^2 + 0.7^2) = 0.922

# Step 4: Weighted sum
# hpwl = J[0,1] * distance[0,1] + J[1,0] * distance[1,0]
#      = 1 * 0.922 + 1 * 0.922 = 1.844
```

#### Constraint Loss

```python
def get_constraints_loss(p: torch.Tensor):
    """
    Calculate constraint violations.

    Two constraints:
    1. Each instance assigned to exactly one site: Σ_s p[i,s] = 1
    2. Each site has at most one instance: Σ_i p[i,s] ≤ 1
    """
    # Constraint 1: sum over sites for each instance should be 1
    # p: [batch, N, num_site]
    # p.sum(dim=-1): [batch, N] - sum of probabilities for each instance
    inst_constraint = ((p.sum(dim=-1) - 1) ** 2).sum(dim=-1)  # [batch]

    # Constraint 2: sum over instances for each site should be ≤ 1
    # p.sum(dim=-2): [batch, num_site] - occupation probability of each site
    site_occupation = p.sum(dim=-2)  # [batch, num_site]
    site_constraint = torch.relu(site_occupation - 1).pow(2).sum(dim=-1)  # [batch]

    return inst_constraint + site_constraint
```

**Example**:
```python
# Valid placement: each instance at one site
p_valid = torch.tensor([[[1, 0, 0], [0, 1, 0]]])
constraint_loss = get_constraints_loss(p_valid)
# Instance sums: [1, 1] → (1-1)^2 + (1-1)^2 = 0
# Site sums: [1, 1, 0] → relu(1-1)^2 + relu(1-1)^2 + relu(0-1)^2 = 0
# Total: 0

# Invalid placement: both instances partially at same site
p_invalid = torch.tensor([[[0.5, 0.5, 0], [0.5, 0.5, 0]]])
constraint_loss = get_constraints_loss(p_invalid)
# Instance sums: [1, 1] → still 0 (constraint 1 satisfied)
# Site sums: [1, 1, 0] → still 0, but conceptually wrong
# Actually: [0.5+0.5, 0.5+0.5, 0] = [1, 1, 0] → no violation detected!
# This is why we need legalization afterwards
```

---

### 3. optimizer.py - FEM Optimization Loop

**Purpose**: Implement the Free Energy Minimization algorithm.

**Key Class: FPGAPlacementOptimizer**

```python
class FPGAPlacementOptimizer:
    def __init__(
        self,
        num_inst: int,
        num_site: int,
        coupling_matrix: torch.Tensor,
        site_coords_matrix: torch.Tensor,
        constraint_weight: float = None
    ):
        self.num_inst = num_inst
        self.num_site = num_site
        self.J = coupling_matrix
        self.site_coords = site_coords_matrix

        # Default constraint weight
        self.alpha = constraint_weight or (num_inst / 2.0)
```

**Optimization Loop**:

```python
def optimize(
    self,
    num_trials: int = 10,
    num_steps: int = 1000,
    dev: str = 'cpu',
    area_width: int = None,
    betamin: float = 0.01,
    betamax: float = 0.5,
    anneal: str = 'inverse',
    optimizer: str = 'adam',
    learning_rate: float = 0.1,
    **kwargs
):
    """Run multiple optimization trials."""

    # Storage for results
    all_configs = []
    all_energies = []

    # Run multiple trials
    for trial in range(num_trials):
        config, energy = self._single_trial(
            num_steps, dev, area_width, betamin, betamax,
            anneal, optimizer, learning_rate, **kwargs
        )
        all_configs.append(config)
        all_energies.append(energy)

    return torch.stack(all_configs), torch.tensor(all_energies)
```

**Single Trial Implementation**:

```python
def _single_trial(
    self,
    num_steps: int,
    dev: str,
    area_width: int,
    betamin: float,
    betamax: float,
    anneal: str,
    optimizer_name: str,
    learning_rate: float,
    **kwargs
):
    # 1. Initialize probability distribution randomly
    # Shape: [num_inst, num_site]
    logits = torch.randn(
        self.num_inst,
        self.num_site,
        device=dev,
        requires_grad=True
    )

    # 2. Create optimizer
    if optimizer_name == 'adam':
        opt = torch.optim.Adam([logits], lr=learning_rate)
    elif optimizer_name == 'sgd':
        opt = torch.optim.SGD([logits], lr=learning_rate)

    # 3. Optimization loop
    for step in range(num_steps):
        # 3a. Calculate current temperature (inverse temperature β)
        beta = self._get_beta(step, num_steps, betamin, betamax, anneal)

        # 3b. Convert logits to probabilities using temperature
        p = torch.softmax(logits * beta, dim=-1)  # [num_inst, num_site]
        p = p.unsqueeze(0)  # [1, num_inst, num_site] - add batch dim

        # 3c. Calculate loss
        hpwl_loss = get_hpwl_loss_qubo(self.J, p, self.site_coords)
        constraint_loss = get_constraints_loss(p)
        total_loss = hpwl_loss + self.alpha * constraint_loss

        # 3d. Gradient descent
        opt.zero_grad()
        total_loss.backward()
        opt.step()

    # 4. Final probabilities
    with torch.no_grad():
        p_final = torch.softmax(logits * betamax, dim=-1)

    # 5. Calculate final energy
    p_final = p_final.unsqueeze(0)
    final_energy = (
        get_hpwl_loss_qubo(self.J, p_final, self.site_coords) +
        self.alpha * get_constraints_loss(p_final)
    ).item()

    return p_final.squeeze(0), final_energy
```

**Temperature Annealing**:

```python
def _get_beta(self, step, num_steps, betamin, betamax, anneal):
    """
    Calculate inverse temperature β at current step.

    β starts low (high temperature, more exploration)
    β ends high (low temperature, more exploitation)
    """
    t = step / num_steps  # Normalized time [0, 1]

    if anneal == 'lin':
        # Linear: β = βmin + (βmax - βmin) * t
        beta = betamin + (betamax - betamin) * t

    elif anneal == 'exp':
        # Exponential: β = βmin * (βmax/βmin)^t
        beta = betamin * (betamax / betamin) ** t

    elif anneal == 'inverse':
        # Inverse (recommended): β = βmin*βmax / (βmin + (βmax-βmin)*t)
        beta = betamin * betamax / (betamin + (betamax - betamin) * t)

    return beta
```

**Why Temperature Annealing?**

```
High temperature (low β):
  p = softmax(logits / T) with large T
  → probabilities are more uniform
  → more exploration, less commitment

Low temperature (high β):
  p = softmax(logits * β) with large β
  → probabilities are more peaked
  → less exploration, more exploitation
  → converges to hard assignment

Example:
logits = [1.0, 1.5, 0.8]

β = 0.1 (high temp):
  p = softmax([0.1, 0.15, 0.08]) = [0.32, 0.35, 0.33]  # uniform

β = 10 (low temp):
  p = softmax([10, 15, 8]) = [0.006, 0.993, 0.001]  # peaked
```

---

### 4. legalizer.py - Overlap Resolution

**Purpose**: Convert soft placement to legal hard placement.

**Algorithm Overview**:

```python
class Legalizer:
    def legalize_placement(self, logic_coords, logic_ids, **kwargs):
        """
        Main legalization algorithm.

        Strategy:
        1. Detect overlaps (multiple instances at same site)
        2. For each overlapping instance:
           a. Find nearest empty site
           b. Move instance there
        3. Repeat until no overlaps
        """
```

**Overlap Detection**:

```python
def _detect_overlaps(self, coords):
    """
    Find sites with multiple instances.

    coords: [N, 2] - instance coordinates
    Returns: dict {site: [inst_ids]}
    """
    site_to_instances = {}

    for inst_id, (x, y) in enumerate(coords):
        site = (int(x.item()), int(y.item()))

        if site not in site_to_instances:
            site_to_instances[site] = []
        site_to_instances[site].append(inst_id)

    # Return sites with >1 instance
    overlaps = {
        site: insts
        for site, insts in site_to_instances.items()
        if len(insts) > 1
    }

    return overlaps
```

**Nearest Empty Site Search**:

```python
def _find_nearest_empty(self, current_site, occupied_sites, max_radius=50):
    """
    Find nearest unoccupied site using BFS.

    current_site: (x, y) tuple
    occupied_sites: set of occupied (x, y) tuples

    Returns: (x, y) tuple of nearest empty site
    """
    from collections import deque

    queue = deque([current_site])
    visited = {current_site}

    while queue:
        x, y = queue.popleft()

        # Check 4-connected neighbors
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            # Check bounds
            if not self._in_bounds(nx, ny):
                continue

            if neighbor in visited:
                continue

            # Found empty site!
            if neighbor not in occupied_sites:
                return neighbor

            visited.add(neighbor)
            queue.append(neighbor)

    # No empty site found, return random
    return self._find_random_empty(occupied_sites)
```

**Complete Legalization**:

```python
def _legalize_greedy(self, coords, logic_ids):
    """
    Greedy legalization algorithm.
    """
    coords = coords.clone()  # Don't modify input

    max_iterations = 100
    for iteration in range(max_iterations):
        # Detect overlaps
        overlaps = self._detect_overlaps(coords)

        if not overlaps:
            break  # Done!

        # Get set of occupied sites
        occupied = set(
            (int(x.item()), int(y.item()))
            for x, y in coords
        )

        # Resolve each overlap
        for site, inst_list in overlaps.items():
            # Keep first instance at site, move others
            for inst_id in inst_list[1:]:
                # Find new site
                new_site = self._find_nearest_empty(site, occupied)

                # Move instance
                coords[inst_id] = torch.tensor(new_site, dtype=coords.dtype)

                # Update occupied set
                occupied.remove(site)  # One less at old site
                occupied.add(new_site)  # One more at new site

    return coords
```

---

### 5. router.py - Connection Routing

**Purpose**: Route connections for visualization.

**Manhattan Routing**:

```python
class Router:
    def route_connections(self, J, coords):
        """
        Route connections using Manhattan routing.

        J: [N, N] connectivity matrix
        coords: [N, 2] instance coordinates

        Returns: List of routes, each is list of (x,y) waypoints
        """
        routes = []

        # For each connection
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                if J[i, j] > 0:  # If connected
                    route = self._manhattan_route(
                        coords[i], coords[j]
                    )
                    routes.append(route)

        return routes

    def _manhattan_route(self, start, end):
        """
        Create L-shaped route from start to end.

        start: [x1, y1]
        end: [x2, y2]

        Returns: [(x1,y1), (x2,y1), (x2,y2)]
        """
        x1, y1 = start.tolist()
        x2, y2 = end.tolist()

        # L-route: horizontal first, then vertical
        return [
            (x1, y1),  # Start
            (x2, y1),  # Turn point
            (x2, y2)   # End
        ]
```

---

### 6. drawer.py - Visualization

**Purpose**: Visualize placement and routing.

**Drawing Placement**:

```python
class PlacementDrawer:
    def draw_place_and_route(
        self,
        logic_coords,
        routes,
        io_coords=None,
        include_io=False,
        iteration=0,
        title_suffix=''
    ):
        """Draw placement with routing."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Draw instances
        x = logic_coords[:, 0].cpu().numpy()
        y = logic_coords[:, 1].cpu().numpy()
        ax.scatter(x, y, c='blue', s=100, label='Logic', zorder=3)

        # 2. Draw IO if included
        if include_io and io_coords is not None:
            io_x = io_coords[:, 0].cpu().numpy()
            io_y = io_coords[:, 1].cpu().numpy()
            ax.scatter(io_x, io_y, c='red', s=100,marker='s', label='IO', zorder=3)

        # 3. Draw routes
        for route in routes:
            xs = [point[0] for point in route]
            ys = [point[1] for point in route]
            ax.plot(xs, ys, 'gray', alpha=0.3, linewidth=0.5, zorder=1)

        # 4. Formatting
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'FPGA Placement - {title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.show()
```

---

## Design Patterns

### 1. Separation of Concerns

- **placer.py**: RapidWright interface only
- **objectives.py**: Pure mathematical functions
- **optimizer.py**: Optimization algorithm only
- **legalizer.py**: Post-processing only

### 2. Functional Core, Imperative Shell

Most functions in `objectives.py` are pure:

```python
# Pure function - no side effects
def get_hpwl_loss_qubo(J, p, site_coords):
    return (J * distances).sum()
```

Side effects isolated to:
- File I/O in `placer.py`
- Visualization in `drawer.py`

### 3. Composition Over Inheritance

No deep inheritance hierarchies. Functionality composed via function calls:

```python
# Compose objectives
total_loss = (
    get_hpwl_loss_qubo(J, p, coords) +
    alpha * get_constraints_loss(p)
)
```

---

## Performance Considerations

### 1. Vectorization

**Bad** (loop-based):
```python
hpwl = 0
for i in range(N):
    for j in range(N):
        dist = torch.norm(coords[i] - coords[j])
        hpwl += J[i,j] * dist
```

**Good** (vectorized):
```python
coord_diff = coords.unsqueeze(1) - coords.unsqueeze(0)
distances = torch.norm(coord_diff, dim=-1)
hpwl = (J * distances).sum()
```

### 2. Memory Layout

Keep tensors contiguous:

```python
# After transpose or view, make contiguous
p = p.transpose(0, 1).contiguous()
```

### 3. GPU Utilization

Transfer to GPU once, compute there:

```python
# Good
J = J.to(device)
coords = coords.to(device)
result = compute(J, coords)

# Bad - repeated transfers
for step in range(1000):
    J_cpu = J.cpu()
    result = compute(J_cpu)
    result = result.to(device)
```

### 4. Mixed Precision

For large designs:

```python
with torch.cuda.amp.autocast():
    loss = compute_loss(J, p)
```

---

This documentation explains the actual implementation. For usage examples, see [User Guide](USER_GUIDE.md). For mathematical background, see [Algorithm](ALGORITHM.md).
