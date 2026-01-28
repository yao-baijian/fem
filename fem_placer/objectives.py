"""
Objective functions for FPGA placement optimization.

This module provides loss functions for:
1. Joint distribution approach - where each cell has a probability distribution over all 2D grid positions
2. QUBO-based approach - using coupling matrices and site coordinate matrices

The QUBO-based functions use site_coords_matrix while joint functions use grid_width/grid_height.
"""

import torch
import torch.nn.functional as Func

# Global history tracking for optimization visualization
_hpwl_loss_history = []
_constrain_loss_history = []
_total_loss_history = []
_placement_history = []

show_steps = [50, 100, 150, 199]


def get_loss_history():
    """Get the loss history from optimization."""
    return {
        'hpwl_losses': _hpwl_loss_history.copy(),
        'constrain_losses': _constrain_loss_history.copy(),
        'total_losses': _total_loss_history.copy()
    }


def get_placement_history():
    """Get the placement history from optimization."""
    return _placement_history.copy()


def clear_history():
    """Clear the loss and placement history."""
    global _hpwl_loss_history, _constrain_loss_history, _total_loss_history, _placement_history
    _hpwl_loss_history = []
    _constrain_loss_history = []
    _total_loss_history = []
    _placement_history = []


# =============================================================================
# Coordinate Functions (QUBO approach)
# =============================================================================

def get_inst_coords_from_index(inst_indices, area_width):
    """
    Convert instance indices to coordinates.

    Args:
        inst_indices: Instance placement indices [batch_size, num_instances]
        area_width: Width of the placement area

    Returns:
        coords: Instance coordinates [batch_size, num_instances, 2]
    """
    x_coords = inst_indices % area_width
    y_coords = inst_indices // area_width
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()


def get_io_coords_from_index(inst_indices):
    """
    Convert IO instance indices to coordinates (IO on left edge).

    Args:
        inst_indices: IO instance indices [batch_size, num_io]

    Returns:
        coords: IO coordinates [batch_size, num_io, 2]
    """
    x_coords = torch.full_like(inst_indices, 0, dtype=torch.int32)
    y_coords = inst_indices
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()


def get_site_distance_matrix(coords):
    """
    Calculate Manhattan distance matrix between all pairs of sites.

    Args:
        coords: Site coordinates [num_sites, 2]

    Returns:
        distances: Distance matrix [num_sites, num_sites]
    """
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    distances = torch.sum(torch.abs(coords_i - coords_j), dim=2)
    return distances


def get_expected_placements_from_index(p, site_coords_matrix):
    """
    Get expected placement coordinates from probability distribution.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        expected_coords: Expected coordinates [batch_size, num_instances, 2]
    """
    expected_coords = torch.matmul(p, site_coords_matrix)
    return expected_coords


def get_hard_placements_from_index(p, site_coords_matrix):
    """
    Get hard (discrete) placement coordinates using argmax.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        hard_coords: Hard placement coordinates [batch_size, num_instances, 2]
    """
    site_indices = torch.argmax(p, dim=2)
    hard_coords = site_coords_matrix[site_indices]
    return hard_coords


def get_placements_from_index_st(p, site_coords_matrix):
    """
    Get placements using straight-through estimator.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        straight_coords: Coordinates with straight-through gradient [batch_size, num_instances, 2]
    """
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
        hard_coords = site_coords_matrix[site_indices]

    expected_coords = torch.matmul(p, site_coords_matrix)
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()

    return straight_coords


# =============================================================================
# HPWL Loss Functions (QUBO approach)
# =============================================================================

def get_hpwl_loss_qubo(J, p, site_coords_matrix):
    """
    Calculate HPWL loss using QUBO formulation.

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        total_wirelength: Total wirelength for each batch [batch_size]
    """
    batch_size, num_instances, num_sites = p.shape

    # Distance matrix between sites
    coords_i = site_coords_matrix.unsqueeze(1)
    coords_j = site_coords_matrix.unsqueeze(0)
    D = torch.sum(torch.abs(coords_i - coords_j), dim=2)

    # Batch matrix multiplication: (p @ D) @ p^T
    PD = torch.matmul(p, D)
    P_transposed = p.transpose(1, 2)
    E_matrix = torch.bmm(PD, P_transposed)

    # Upper triangular mask
    triu_mask = torch.triu(torch.ones(num_instances, num_instances, device=p.device), diagonal=1).bool()

    # Weight and sum
    weighted_E = E_matrix * J.unsqueeze(0)
    weighted_E_triu = weighted_E[:, triu_mask]
    total_wirelength = torch.sum(weighted_E_triu, dim=1)

    return total_wirelength


def get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io,
                               logic_site_coords_matrix, io_site_coords_matrix):
    """
    Calculate HPWL loss including IO connections.

    Args:
        J_LL: Logic-logic coupling matrix [num_logic, num_logic]
        J_LI: Logic-IO coupling matrix [num_logic, num_io]
        p_logic: Logic probability distribution [batch_size, num_logic, num_logic_sites]
        p_io: IO probability distribution [batch_size, num_io, num_io_sites]
        logic_site_coords_matrix: Logic site coordinates [num_logic_sites, 2]
        io_site_coords_matrix: IO site coordinates [num_io_sites, 2]

    Returns:
        total_wl: Total wirelength for each batch [batch_size]
    """
    batch_size, n_logic, _ = p_logic.shape
    device = p_logic.device

    x_logic = logic_site_coords_matrix[:, 0]
    y_logic = logic_site_coords_matrix[:, 1]

    Dx_LL = torch.abs(x_logic.unsqueeze(1) - x_logic.unsqueeze(0))
    Dy_LL = torch.abs(y_logic.unsqueeze(1) - y_logic.unsqueeze(0))
    D_LL = Dx_LL + Dy_LL

    x_io = io_site_coords_matrix[:, 0]
    y_io = io_site_coords_matrix[:, 1]

    Dx_LI = torch.abs(x_logic.unsqueeze(1) - x_io.unsqueeze(0))
    Dy_LI = torch.abs(y_logic.unsqueeze(1) - y_io.unsqueeze(0))
    D_LI = Dx_LI + Dy_LI

    total_wl = torch.zeros(batch_size, device=device)

    # Logic-logic wirelength
    PD = torch.matmul(p_logic, D_LL)
    p_logic_T = p_logic.transpose(1, 2)
    E = torch.bmm(PD, p_logic_T)
    triu_mask = torch.triu(torch.ones(n_logic, n_logic, device=device), diagonal=1)
    wl_LL = torch.sum(E * J_LL.unsqueeze(0) * triu_mask.unsqueeze(0), dim=(1, 2))
    total_wl += wl_LL

    # Logic-IO wirelength
    PD_LI = torch.matmul(p_logic, D_LI)
    p_io_T = p_io.transpose(1, 2)
    E_LI = torch.bmm(PD_LI, p_io_T)
    wl_LI = torch.sum(E_LI * J_LI.unsqueeze(0), dim=(1, 2))
    total_wl += wl_LI

    return total_wl


# =============================================================================
# Constraint Loss Functions (QUBO approach)
# =============================================================================

def get_constraints_loss(p):
    """
    Calculate site usage constraint loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]

    Returns:
        site_constraint: Constraint loss for each batch [batch_size]
    """
    _, num_instance, num_sites = p.shape

    site_usage = torch.sum(p, dim=1)

    expected_usage_per_site = float(num_instance) / num_sites
    site_constraint = torch.sum(30 * Func.softplus(site_usage - 1)**2, dim=1)
    return site_constraint


def get_constraints_loss_with_io(p_logic, p_io):
    """
    Calculate constraint loss for both logic and IO placements.

    Args:
        p_logic: Logic probability distribution [batch_size, num_logic, num_logic_sites]
        p_io: IO probability distribution [batch_size, num_io, num_io_sites]

    Returns:
        constraint_loss: Total constraint loss [batch_size]
    """
    coeff_1 = p_logic.shape[1] / 2
    logic_site_usage = torch.sum(p_logic, dim=1)
    logic_constraint = torch.sum(coeff_1 * Func.softplus(logic_site_usage - 1)**2, dim=1)

    coeff_2 = p_io.shape[1] / 2
    io_site_usage = torch.sum(p_io, dim=1)
    io_constraint = torch.sum(coeff_2 * Func.softplus(io_site_usage - 1)**2, dim=1)

    return logic_constraint + io_constraint


# =============================================================================
# Manual Gradient Functions
# =============================================================================

def manual_grad_hpwl_loss(p, W, D):
    """
    Compute manual gradient of HPWL loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        W: Weight matrix [num_instances, num_instances]
        D: Distance matrix [num_sites, num_sites]

    Returns:
        h_grad: Gradient [batch_size, num_instances, num_sites]
    """
    batch_size, _, _ = p.shape

    PD = torch.matmul(p, D)
    W_batch = W.unsqueeze(0).expand(batch_size, -1, -1)
    h_grad = torch.bmm(W_batch, PD)

    return h_grad


def manual_grad_constraint_loss(p, lambda_constraint=30.0):
    """
    Compute manual gradient of constraint loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        lambda_constraint: Constraint weight

    Returns:
        h_grad: Gradient [batch_size, num_instances, num_sites]
    """
    site_occupancy = torch.sum(p, dim=1)
    excess = site_occupancy - 1.0

    softplus_val = Func.softplus(excess)
    sigmoid_val = torch.sigmoid(excess)

    site_grad = 2 * lambda_constraint * softplus_val * sigmoid_val
    h_grad = site_grad.unsqueeze(1).repeat(1, p.shape[1], 1)

    return h_grad


def manual_grad_placement(p, J, site_coords_matrix, lambda_constraint=30.0):
    """
    Compute manual gradient of combined placement loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        J: Coupling matrix [num_instances, num_instances]
        site_coords_matrix: Site coordinates [num_sites, 2]
        lambda_constraint: Constraint weight

    Returns:
        dE_dh: Gradient [batch_size, num_instances, num_sites]
    """
    batch_size, N, M = p.shape

    # Distance matrix
    coords_i = site_coords_matrix.unsqueeze(1)
    coords_j = site_coords_matrix.unsqueeze(0)
    D = torch.sum(torch.abs(coords_i - coords_j), dim=2)

    # HPWL gradient
    PD = torch.matmul(p, D)
    mask = torch.triu(torch.ones(N, N, device=p.device), diagonal=1).bool()
    J_upper = J * mask
    J_batch = J_upper.unsqueeze(0).expand(batch_size, -1, -1)
    dE_hpwl_dp = torch.bmm(J_batch, PD)

    # Constraint gradient
    site_occupancy = torch.sum(p, dim=1)
    excess = site_occupancy - 1.0
    softplus_val = Func.softplus(excess)
    sigmoid_val = torch.sigmoid(excess)
    site_grad = 2 * lambda_constraint * softplus_val * sigmoid_val
    dE_constraint_dp = site_grad.unsqueeze(1).expand(-1, N, -1)

    dE_dp = dE_hpwl_dp + dE_constraint_dp

    # Compute softmax gradient
    sum_term = torch.sum(dE_dp * p, dim=2, keepdim=True)
    dE_dh = dE_dp * p - p * sum_term

    return dE_dh


# =============================================================================
# Expected Placement Loss Functions (QUBO approach)
# =============================================================================

def expected_fpga_placement(J, p, site_coords_matrix, step, area_width, alpha):
    """
    Calculate expected placement loss (HPWL + constraints).

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]
        step: Current optimization step (for history tracking)
        area_width: Width of the placement area
        alpha: Constraint weight

    Returns:
        total_loss: Combined loss [batch_size]
    """
    global _hpwl_loss_history, _constrain_loss_history, _total_loss_history, _placement_history

    current_hpwl = get_hpwl_loss_qubo(J, p, site_coords_matrix)
    constrain_loss = get_constraints_loss(p)

    hpwl_val = current_hpwl
    constrain_val = alpha * constrain_loss
    total_val = hpwl_val + constrain_val

    _hpwl_loss_history.append(hpwl_val.mean().item())
    _constrain_loss_history.append(constrain_val.mean().item())
    _total_loss_history.append(total_val.mean().item())

    if step in show_steps:
        inst_indices = torch.argmax(p, dim=2)
        inst_coords = get_inst_coords_from_index(inst_indices, area_width)
        _placement_history.append(inst_coords)

    return current_hpwl + alpha * constrain_loss


def expected_fpga_placement_with_io(J_LL, J_LI, p_logic, p_io, logic_site_coords, io_site_coords):
    """
    Calculate expected placement loss including IO connections.

    Args:
        J_LL: Logic-logic coupling matrix
        J_LI: Logic-IO coupling matrix
        p_logic: Logic probability distribution
        p_io: IO probability distribution
        logic_site_coords: Logic site coordinates
        io_site_coords: IO site coordinates

    Returns:
        total_loss: Combined loss [batch_size]
    """
    hpwl_weight, constrain_weight = 1, 20
    current_hpwl = get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io, logic_site_coords, io_site_coords)
    constrain_loss = get_constraints_loss_with_io(p_logic, p_io)
    return hpwl_weight * current_hpwl + constrain_weight * constrain_loss


# =============================================================================
# Inference Functions (QUBO approach)
# =============================================================================

def infer_placements(J, p, area_width, site_coords_matrix):
    """
    Infer final placements from probability distribution.

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Probability distribution [batch_size, num_instances, num_sites]
        area_width: Width of the placement area
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        inst_coords: Instance coordinates [batch_size, num_instances, 2]
        hpwl: HPWL values [batch_size]
    """
    inst_indices = torch.argmax(p, dim=2)
    inst_coords = get_inst_coords_from_index(inst_indices, area_width)
    result = get_hpwl_loss_qubo(J, p, site_coords_matrix)
    return inst_coords, result


def infer_placements_with_io(J_LL, J_LI, p_logic, p_io, area_width, logic_site_coords_matrix, io_site_coords_matrix):
    """
    Infer final placements including IO from probability distributions.

    Args:
        J_LL: Logic-logic coupling matrix
        J_LI: Logic-IO coupling matrix
        p_logic: Logic probability distribution
        p_io: IO probability distribution
        area_width: Width of the placement area
        logic_site_coords_matrix: Logic site coordinates
        io_site_coords_matrix: IO site coordinates

    Returns:
        coords: List of [logic_coords, io_coords]
        hpwl: HPWL values [batch_size]
    """
    logic_inst_indices = torch.argmax(p_logic, dim=2)
    io_inst_indices = torch.argmax(p_io, dim=2)
    logic_inst_coords = get_inst_coords_from_index(logic_inst_indices, area_width)
    io_inst_coords = get_io_coords_from_index(io_inst_indices)
    result = get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io, logic_site_coords_matrix, io_site_coords_matrix)
    return [logic_inst_coords, io_inst_coords], result


# =============================================================================
# Joint Distribution Functions (Original xw/refactor approach)
# =============================================================================

def get_grid_coords_joint(grid_width: int, grid_height: int, device='cpu'):
    """
    Get coordinate matrix for all grid positions.

    Args:
        grid_width: Grid width
        grid_height: Grid height
        device: Torch device

    Returns:
        coords: [grid_width * grid_height, 2] coordinate matrix
    """
    num_positions = grid_width * grid_height

    # Create coordinate matrix
    x_coords = torch.arange(grid_width, device=device, dtype=torch.float32)
    y_coords = torch.arange(grid_height, device=device, dtype=torch.float32)

    # Create meshgrid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Flatten and stack
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [num_positions, 2]

    return coords


def get_placements_from_joint_st(p, grid_width, grid_height):
    """
    Convert joint probability distribution to placements using straight-through estimator.

    Args:
        p: Joint probability [batch_size, num_instances, num_positions]
        grid_width: Grid width
        grid_height: Grid height

    Returns:
        coords: Placement coordinates [batch_size, num_instances, 2]
    """
    # Get coordinate matrix
    coord_matrix = get_grid_coords_joint(grid_width, grid_height, device=p.device)  # [num_positions, 2]

    # Hard placement (discrete, for forward pass)
    with torch.no_grad():
        position_indices = torch.argmax(p, dim=2)  # [batch_size, num_instances]
        hard_coords = coord_matrix[position_indices]  # [batch_size, num_instances, 2]

    # Soft placement (continuous, for gradient)
    expected_coords = torch.matmul(p, coord_matrix)  # [batch_size, num_instances, 2]

    # Straight-through: forward uses hard, backward uses soft
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()

    return straight_coords


def get_hpwl_loss_joint(J, p, grid_width, grid_height):
    """
    Calculate HPWL loss from joint probability distribution.

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Joint probability distribution [batch_size, num_instances, num_positions]
        grid_width: Grid width
        grid_height: Grid height

    Returns:
        hpwl_loss: HPWL loss for each trial [batch_size]
    """
    batch_size, num_instances, num_positions = p.shape

    # Get expected placements using straight-through estimator
    expected_coords = get_placements_from_joint_st(p, grid_width, grid_height)

    # Calculate pairwise Manhattan distances
    coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
    coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)  # [batch_size, num_instances, num_instances]

    # Weight by coupling matrix
    weighted_dist = manhattan_dist * J.unsqueeze(0)  # [batch_size, num_instances, num_instances]

    # Only count upper triangular (avoid double counting)
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]  # [batch_size, num_pairs]

    # Sum to get total wirelength
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)  # [batch_size]

    return total_wirelength


def get_constraints_loss_joint(p, grid_width, grid_height):
    """
    Calculate constraint loss to prevent overlapping placements.

    Args:
        p: Joint probability distribution [batch_size, num_instances, num_positions]
        grid_width: Grid width
        grid_height: Grid height

    Returns:
        constraint_loss: Constraint loss for each trial [batch_size]
    """
    batch_size, num_instances, num_positions = p.shape

    # Calculate position usage: sum probability across all instances
    position_usage = torch.sum(p, dim=1)  # [batch_size, num_positions]

    # Target usage: each position should have on average num_instances / num_positions usage
    target_usage = num_instances / num_positions

    # Penalize positions that are over-utilized
    # Using softplus to make it differentiable and smooth
    excess_usage = position_usage - target_usage
    constraint_loss = torch.sum(Func.softplus(10 * excess_usage ** 2), dim=1)  # [batch_size]

    return constraint_loss


def expected_fpga_placement_joint(J: torch.Tensor, p: torch.Tensor, grid_width: int, grid_height: int):
    """
    Calculate expected placement loss (HPWL + constraints).

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Joint probability distribution [batch_size, num_instances, num_positions]
        grid_width: Grid width
        grid_height: Grid height

    Returns:
        hpwl_loss: HPWL loss [batch_size]
        constraint_loss: Constraint loss [batch_size]
    """
    hpwl_loss = get_hpwl_loss_joint(J, p, grid_width, grid_height)
    constraint_loss = get_constraints_loss_joint(p, grid_width, grid_height)

    return hpwl_loss, constraint_loss


def infer_placements_joint(J, p, grid_width, grid_height):
    """
    Infer final placement from joint probability distribution.

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Joint probability distribution [batch_size, num_instances, num_positions]
        grid_width: Grid width
        grid_height: Grid height

    Returns:
        coords: Placement coordinates [batch_size, num_instances, 2]
        hpwl: HPWL values [batch_size]
    """
    # Get hard placements (argmax)
    position_indices = torch.argmax(p, dim=2)  # [batch_size, num_instances]

    # Convert to coordinates
    x_coords = (position_indices % grid_width).float()
    y_coords = (position_indices // grid_width).float()
    coords = torch.stack([x_coords, y_coords], dim=2)  # [batch_size, num_instances, 2]

    # Calculate HPWL for these placements
    hpwl = get_hpwl_loss_joint(J, p, grid_width, grid_height)

    return coords, hpwl
