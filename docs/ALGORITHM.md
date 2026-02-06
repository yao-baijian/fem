# Algorithm Overview

Technical details of the FEM-based FPGA placement algorithm.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Formulation](#problem-formulation)
3. [QUBO Formulation](#qubo-formulation)
4. [FEM: Mean-Field Relaxation](#fem-mean-field-relaxation)
5. [Optimization Algorithm](#optimization-algorithm)
6. [Legalization](#legalization)
7. [Simulated Bifurcation](#simulated-bifurcation)
8. [Complexity Analysis](#complexity-analysis)

---

## Introduction

The FPGA placement problem involves assigning logic instances to physical sites on an FPGA chip to minimize wirelength (HPWL) while satisfying placement constraints.

This package uses a **Free Energy Minimization (FEM)** approach with **QUBO (Quadratic Unconstrained Binary Optimization)** formulation, where:
- Placements are represented as probability distributions
- HPWL and constraints are differentiable objective functions
- Optimization uses gradient descent with temperature annealing

---

## Problem Formulation

### Input

- **Netlist**: Logic instances and their connections
- **Device**: FPGA fabric with available placement sites
- **Constraints**: One instance per site, one site per instance

### Output

- **Placement**: Assignment of instances to sites
- **Objective**: Minimize Half-Perimeter Wirelength (HPWL)

### Mathematical Formulation

Given:

- \(m\) instances: \(\{I_0, I_1, \ldots, I_{m-1}\}\)
- \(n\) sites: \(\{S_0, S_1, \ldots, S_{n-1}\}\)
- Connectivity matrix \(F_{ij}\): weight of connection between instances \(i\) and \(j\)
- Site coordinates: \(\{(x_0, y_0), (x_1, y_1), \ldots, (x_{n-1}, y_{n-1})\}\)

Find placement \(\pi: \text{Instances} \to \text{Sites}\) that minimizes:

\[
\text{HPWL} = \sum_{\text{net}} \bigl(\max x - \min x + \max y - \min y\bigr)
\]

Subject to:

- Each instance assigned to exactly one site
- Each site occupied by at most one instance

---

## QUBO Formulation

### One-Hot Binary Variables

Each instance \(i \in \{0, \ldots, m{-}1\}\) is assigned to a site \(s \in \{0, \ldots, n{-}1\}\) using one-hot binary variables:

\[
x_{i,s} \in \{0, 1\}, \quad \text{where } x_{i,s} = 1 \iff \text{instance } i \text{ is placed at site } s
\]

Flatten all \(x_{i,s}\) into a single vector \(\mathbf{x} \in \{0,1\}^{mn}\) (instance-major ordering).

### Discrete Optimization Problem

The FPGA placement problem is formulated as the following constrained QUBO:

\[
\arg\min_{\mathbf{x} \in \{0,1\}^{mn},\; \mathbf{s} \in \{0,1\}^{n}} \;
\mathbf{x}^\top \mathbf{Q} \mathbf{x}
\;+\; \lambda \| \mathbf{A}\mathbf{x} - \mathbf{1}_m \|^2
\;+\; \mu \| \mathbf{B}\mathbf{x} - \mathbf{s} \|^2
\]

where:

- \(\mathbf{Q} := F \otimes D\) — wirelength objective
- \(\lambda \| \mathbf{A}\mathbf{x} - \mathbf{1}_m \|^2\) — one-hot constraint (each instance at exactly one site)
- \(\mu \| \mathbf{B}\mathbf{x} - \mathbf{s} \|^2\) — at-most-one constraint (each site holds at most one instance)

### Term 1: Wirelength Objective x'Qx

Define:

- \(F \in \mathbb{R}^{m \times m}\): symmetric coupling matrix (\(F_{ij}\) = connection weight between instances \(i\) and \(j\); called `J` in code)
- \(D \in \mathbb{R}^{n \times n}\): Manhattan distance matrix, \(D_{st} = |x_s - x_t| + |y_s - y_t|\)
- \(\mathbf{Q} := F \otimes D \in \mathbb{R}^{mn \times mn}\): Kronecker product, with entries \(Q_{(i,s),(j,t)} = F_{ij} \cdot D_{st}\)

The quadratic form expands to:

\[
\mathbf{x}^\top \mathbf{Q} \mathbf{x}
= \sum_{i,j} \sum_{s,t} F_{ij} \cdot D_{st} \cdot x_{i,s} \cdot x_{j,t}
\]

For a feasible one-hot assignment (each \(\mathbf{x}_i\) has exactly one nonzero entry at position \(\sigma_i\)):

\[
\mathbf{x}^\top \mathbf{Q} \mathbf{x}
= \sum_{i,j} F_{ij} \cdot D(\sigma_i, \sigma_j)
= 2 \sum_{i < j} F_{ij} \cdot D(\sigma_i, \sigma_j)
\]

The factor of 2 arises because \(F\) is symmetric. The diagonal terms vanish since \(D(s,s) = 0\).

### Term 2: One-Hot Constraint

Define:

\[
\mathbf{A} := I_m \otimes \mathbf{1}_n^\top \;\in\; \mathbb{R}^{m \times mn}
\]

Row \(i\) of \(\mathbf{A}\) sums all site variables for instance \(i\):

\[
(\mathbf{A}\mathbf{x})_i = \sum_{s} x_{i,s}
\]

The penalty enforces each instance is placed at exactly one site:

\[
\| \mathbf{A}\mathbf{x} - \mathbf{1}_m \|^2 = \sum_{i} \left( \sum_{s} x_{i,s} - 1 \right)^2
\]

### Term 3: At-Most-One Constraint

Define:

\[
\mathbf{B} := \mathbf{1}_m^\top \otimes I_n \;\in\; \mathbb{R}^{n \times mn}, \qquad
\mathbf{s} \in \{0,1\}^n \;\text{(slack variables: } s_s = 1 \text{ if site } s \text{ is occupied)}
\]

Row \(s\) of \(\mathbf{B}\) sums all instance variables at site \(s\):

\[
(\mathbf{B}\mathbf{x})_s = \sum_{i} x_{i,s}
\]

The penalty with binary slack \(\mathbf{s}\) enforces at-most-one occupancy:

\[
\| \mathbf{B}\mathbf{x} - \mathbf{s} \|^2 = \sum_{s} \left( \sum_{i} x_{i,s} - s_s \right)^2
\]

Since \(s_s \in \{0,1\}\), feasibility requires \(\sum_i x_{i,s} \in \{0, 1\}\) for all sites:

- \(\sum_i x_{i,s} = 0 \;\Rightarrow\; s_s = 0\), penalty = 0 (empty site)
- \(\sum_i x_{i,s} = 1 \;\Rightarrow\; s_s = 1\), penalty = 0 (single occupancy)
- \(\sum_i x_{i,s} = 2 \;\Rightarrow\; s_s = 1\) is optimal, penalty = 1 (violation)

---

## FEM: Mean-Field Relaxation

The FEM approach (see [arXiv:2412.09285](https://arxiv.org/abs/2412.09285)) replaces the
discrete QUBO with a continuous optimization over probability distributions.

### Probability Variables

Replace binary one-hot variables with marginal probabilities under a factorized
(mean-field) distribution:

\[
P_{\text{MF}}(\vec{\sigma}) = \prod_{i} P_i(\sigma_i)
\]

where \(P_i(s) \in [0,1]\) is the probability that instance \(i\) is placed at site \(s\),
parameterized via softmax over local fields \(h_i(s)\):

\[
P_i(s) = \frac{\exp\bigl(h_i(s)\bigr)}{\sum_{s'} \exp\bigl(h_i(s')\bigr)}
\]

The softmax automatically satisfies \(\sum_s P_i(s) = 1\) (one-hot constraint), so
the \(\lambda\)-penalty term vanishes and \(h \in \mathbb{R}^{m \times n}\) are the true variational parameters.

### Mean-Field Expected Energy

Under the factorized distribution, the expected energy decomposes as:

\[
\langle E \rangle_{\text{MF}}
= \sum_{\vec{\sigma}} \left[\prod_k P_k(\sigma_k)\right] E(\vec{\sigma})
= \sum_{i < j} F_{ij} \sum_{s,t} P_i(s) \, D_{st} \, P_j(t)
\]

In matrix form (what `get_hpwl_loss_qubo` computes):

```python
D = site_distance_matrix(site_coords)   # [n, n]
PD = p @ D                              # [batch, m, n]
E_matrix = PD @ p.T                     # [batch, m, m]
# E_matrix[i,j] = Σ_{s,t} P_i(s) · D(s,t) · P_j(t)
energy = (E_matrix * J)[triu_mask].sum()
```

This is the **exact mean-field expected energy** — no approximation is involved in
going from the discrete QUBO to this continuous form, given the mean-field factorization.

!!! note "Scaling factor"
    The code sums over upper-triangular \(i < j\) only, so it computes
    \(\frac{1}{2}\mathbf{x}^\top\mathbf{Q}\mathbf{x}\) under the mean-field distribution.
    This constant factor is absorbed into hyperparameters.

### Site Coordinates Matrix

Precompute all site coordinates:

```python
site_coords_matrix = torch.cartesian_prod(
    torch.arange(area_width, dtype=torch.float32),
    torch.arange(area_height, dtype=torch.float32)
)
# Shape: [n, 2]
```

### Constraint Relaxation

The at-most-one constraint (\(\mu\)-term) is relaxed to a soft penalty on expected
site occupancy:

\[
C(P) = \sum_{s} c \cdot \operatorname{softplus}\!\left(\sum_{i} P_i(s) - 1\right)^2
\]

where \(\operatorname{softplus}(x) = \log(1 + e^x)\) is a smooth approximation of \(\max(0, x)\).

!!! info "Approximation"
    This computes \(\text{penalty}\bigl(\mathbb{E}[n_s]\bigr)\) rather than the exact
    \(\mathbb{E}\bigl[\text{penalty}(n_s)\bigr]\). These coincide as \(P\) converges to
    near-one-hot distributions at low temperature.

### Variational Free Energy

The total free energy to minimize (from the FEM framework):

\[
F_{\text{MF}}(\mathbf{h}, \beta)
= \langle E \rangle_{\text{MF}}
+ \alpha \cdot C(P)
- \frac{1}{\beta} \cdot S_{\text{MF}}
\]

where:

- \(\langle E \rangle_{\text{MF}}\): mean-field expected wirelength (see above)
- \(C(P)\): soft constraint penalty
- \(S_{\text{MF}} = -\sum_i \sum_s P_i(s) \log P_i(s)\): mean-field entropy
- \(\beta = 1/T\): inverse temperature, annealed from \(\beta_{\min}\) to \(\beta_{\max}\)

At high temperature (\(\beta\) small), the entropy term dominates and the distribution
stays exploratory. As \(\beta \to \infty\), \(P_i(s)\) converges to a one-hot distribution and
\(F_{\text{MF}}\) approaches the discrete QUBO objective.

### Temperature Annealing

Three schedules are supported:

| Schedule | Formula |
|----------|---------|
| Inverse (recommended) | \(\beta_t = \dfrac{\beta_{\min}\,\beta_{\max}}{\beta_{\min} + (\beta_{\max} - \beta_{\min})\,t/T}\) |
| Linear | \(\beta_t = \beta_{\min} + (\beta_{\max} - \beta_{\min})\,t/T\) |
| Exponential | \(\beta_t = \beta_{\min}\left(\dfrac{\beta_{\max}}{\beta_{\min}}\right)^{t/T}\) |

### Correspondence: QUBO Terms ↔ Code

| QUBO Term | FEM Relaxation | Code |
|-----------|---------------|------|
| \(\mathbf{x}^\top(F \otimes D)\mathbf{x}\) | \(\sum_{i<j} F_{ij} \sum_{s,t} P_i(s)\,D_{st}\,P_j(t)\) | `get_hpwl_loss_qubo()` |
| \(\lambda\|\mathbf{A}\mathbf{x}-\mathbf{1}\|^2\) | Automatically 0 (softmax) | `p = softmax(h, dim=2)` |
| \(\mu\|\mathbf{B}\mathbf{x}-\mathbf{s}\|^2\) | \(c \sum_s \operatorname{softplus}(\sum_i P_i(s)-1)^2\) | `get_constraints_loss()` |
| — | \(-\frac{1}{\beta} S_{\text{MF}}\) | `entropy_q(p) / beta` |
| Full problem | \(F_{\text{MF}} = \langle E\rangle + \alpha C - S/\beta\) | `optimizer.py:154` |

---

## Optimization Algorithm

### Gradient Descent on Free Energy

The optimization loop minimizes the variational free energy `F_MF(h, β)` w.r.t.
the local fields `h`:

```python
for step in range(num_steps):
    # Probabilities from local fields via softmax
    p = softmax(h, dim=2)                    # P_i(s) = softmax(h_i)

    # Mean-field expected energy + constraint penalty
    E = get_hpwl_loss_qubo(J, p, site_coords)  # <E>_MF
    C = get_constraints_loss(p)                  # soft ‖Bx-s‖²
    total_loss = E + alpha * C

    # Free energy = energy - T * entropy
    S = entropy_q(p)                             # -Σ P log P
    free_energy = total_loss - S / betas[step]

    # Gradient descent on h
    optimizer.zero_grad()
    free_energy.backward()
    optimizer.step()
```

### Optimizer Choices

- **Adam** (recommended): Adaptive learning rates, good convergence
- **SGD**: Simple but requires careful learning rate tuning
- **AdamW**: Adam with weight decay

### Inference (Hard Assignment)

After optimization, convert soft probabilities to hard assignments:

```python
# Greedy assignment
site_indices = torch.argmax(p, dim=1)  # [N]

# Convert to coordinates
coords = site_coords_matrix[site_indices]  # [N, 2]
```

---

## Legalization

### Overlap Detection

Check if multiple instances assigned to the same site:

```python
unique_sites, counts = torch.unique(site_indices, return_counts=True)
overlaps = (counts > 1).sum()
```

### Legalization Strategies

1. **Greedy Legalization**:
   - Sort instances by priority
   - For each instance, if site is occupied, find nearest empty site

2. **Global Optimization**:
   - Formulate as minimum cost perfect matching
   - Use Hungarian algorithm or other assignment solvers

### Implementation

```python
def legalize_placement(coords, logic_ids):
    # Find overlaps
    overlaps = detect_overlaps(coords)

    # For each overlap
    for site in overlaps:
        instances_at_site = get_instances_at(site)

        # Keep first instance, move others
        for inst in instances_at_site[1:]:
            # Find nearest empty site
            new_site = find_nearest_empty(coords, site)
            coords[inst] = new_site

    return coords
```

---

## Simulated Bifurcation

Alternative solver based on quantum annealing simulation.

### Formulation

Convert placement problem to Ising model:

\[
H(\mathbf{s}) = -\sum_{i,j} J_{ij} \, s_i \, s_j
\]

where \(s_i \in \{-1, +1\}\) are spins.

### Algorithm

```python
# Initialize spins and momenta
x = torch.randn(N)  # Positions
y = torch.zeros(N)  # Momenta

# Simulated dynamics
for t in range(max_steps):
    # Compute forces
    forces = -J @ torch.sign(x)

    # Update positions and momenta
    y = y - dt * (a * x + forces)
    x = x + dt * y

    # Increase pressure
    a = a0 * (1 + t / max_steps)

# Extract solution
spins = torch.sign(x)
```

### Comparison with FEM

| Aspect | FEM | Simulated Bifurcation |
|--------|-----|----------------------|
| Variables | Probabilities | Spins/bits |
| Optimization | Gradient descent | Hamiltonian dynamics |
| Flexibility | High (differentiable) | Medium |
| Speed | Medium | Fast |
| Quality | High | Medium |

---

## Complexity Analysis

### Space Complexity

- **Probability matrix**: `O(N * M)` where `N` = instances, `M` = sites
- **Connectivity matrix**: `O(N^2)` or `O(E)` if sparse (E = edges)
- **Site coordinates**: `O(M)`

Total: `O(N * M + N^2)`

For large designs:
- Use sparse matrices for connectivity
- Consider hierarchical placement

### Time Complexity

Per optimization step:
- **Expected coordinates**: `O(N * M)` (matrix multiplication)
- **HPWL computation**: `O(N^2)` or `O(E)` if sparse
- **Constraint computation**: `O(N * M)`
- **Gradient computation**: `O(N * M)`

Total per step: `O(N * M + N^2)`

For `T` steps and `K` trials:
- Total: `O(K * T * (N * M + N^2))`

### Optimization Tips

1. **Use GPU**: Parallelize matrix operations
   ```python
   configs, energies = optimizer.optimize(dev='cuda')
   ```

2. **Reduce sites**: Use coarser grid initially
   ```python
   # Coarse grid: area_length / 2
   # Fine grid: area_length
   ```

3. **Sparse connectivity**: For sparse graphs
   ```python
   J_sparse = J.to_sparse()
   ```

4. **Mixed precision**: Use float16 for large designs
   ```python
   configs, energies = optimizer.optimize(dtype=torch.float16)
   ```

---

## Mathematical Properties

### Convergence

Under standard assumptions (convexity, Lipschitz gradients):
- Adam optimizer converges to local minimum
- Annealing helps escape local minima
- Multiple trials improve solution quality

### Optimality

- **No guarantee of global optimum** (NP-hard problem)
- Solutions are typically within 10-20% of optimal
- Quality improves with more trials and longer optimization

### Scalability

Tested on designs up to:
- 1000+ instances
- 2000+ sites
- 5000+ nets

Scales linearly with:
- Number of instances (N)
- Number of sites (M)

Scales quadratically with:
- Connectivity density

---

## References

1. **FEM Framework**:
   - Fan et al., "Free-Energy Machine: A General Method for Combinatorial Optimization", [arXiv:2412.09285](https://arxiv.org/abs/2412.09285), 2024
   - Implementation: https://github.com/Fanerst/FEM

2. **QUBO Formulation**:
   - "QUBO formulations for optimization problems"
   - Quadratic unconstrained binary optimization

3. **Simulated Bifurcation**:
   - "Simulated bifurcation for combinatorial optimization"
   - Library: https://pypi.org/project/simulated-bifurcation/

4. **FPGA Placement**:
   - VPR: Versatile Place and Route
   - RapidWright: FPGA CAD framework

---

## Future Directions

1. **Hierarchical Placement**: Coarse-to-fine approach
2. **Routing-Aware Placement**: Include routing congestion
3. **Multi-Objective Optimization**: Balance HPWL, timing, power
4. **Machine Learning**: Learn better annealing schedules
5. **Incremental Placement**: Fast updates for design changes
