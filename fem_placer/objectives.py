"""
Objective functions for joint XY probability distribution.

This module provides loss functions for the joint distribution approach,
where each cell has a single probability distribution over all 2D grid positions.
"""

import torch
import torch.nn.functional as Func

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
