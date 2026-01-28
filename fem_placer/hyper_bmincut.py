"""
Hypergraph Balanced Min-Cut Functions.

This module provides functions for hypergraph partitioning with balanced constraints,
including loss functions for optimization and inference functions for final partitioning.
"""

import torch
import torch.nn.functional as Func
import numpy as np


def balance_constrain(J, p, U_max, L_min):
    """
    Calculate balance constraint loss.

    Args:
        J: Coupling matrix (unused, kept for API compatibility)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        U_max: Upper bound on cluster size
        L_min: Lower bound on cluster size

    Returns:
        balance_loss: Balance constraint loss [batch_size]
    """
    S_k = p.sum(dim=1)  # [batch, n_clusters]

    upper_violation = torch.relu(S_k - U_max)
    lower_violation = torch.relu(L_min - S_k)

    balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)
    return balance_loss


def balance_constrain_softplus(J, p, U_max, L_min, beta=1.0):
    """
    Calculate balance constraint loss using softplus activation.

    Args:
        J: Coupling matrix (unused, kept for API compatibility)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        U_max: Upper bound on cluster size
        L_min: Lower bound on cluster size
        beta: Temperature parameter for softplus

    Returns:
        balance_loss: Balance constraint loss [batch_size]
    """
    S_k = p.sum(dim=1)

    upper_violation = Func.softplus(beta * (S_k - U_max))
    lower_violation = Func.softplus(beta * (L_min - S_k))

    balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)
    return balance_loss


def balance_constrain_relu(J, p, U_max, L_min, hyperedges=None):
    """
    Calculate balance constraint loss using ReLU with hard assignments.

    Args:
        J: Coupling matrix [num_nodes, num_nodes]
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        U_max: Upper bound on cluster size
        L_min: Lower bound on cluster size
        hyperedges: Optional hyperedge list

    Returns:
        balance_loss: Balance constraint loss [batch_size]
    """
    batch_size, n_nodes, n_clusters = p.shape

    probabilities = torch.softmax(p, dim=2)
    assignments = torch.argmax(probabilities, dim=2)
    one_hot = Func.one_hot(assignments, num_classes=n_clusters)
    S_k = one_hot.sum(dim=1).float()

    upper_violation = torch.relu(S_k - U_max)
    lower_violation = torch.relu(L_min - S_k)
    balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)

    return balance_loss


def infer_hyperbmincut(J, p, hyperedges):
    """
    Infer hypergraph partition from probability distribution.

    Args:
        J: Node-node coupling matrix [num_nodes, num_nodes]
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        hyperedges: List of hyperedges, each hyperedge is a list of node indices

    Returns:
        config: One-hot encoded configuration [batch_size, num_nodes, num_clusters]
        cut_value: Cut value for each configuration [batch_size]
    """
    config = Func.one_hot(
        p.view(-1, J.shape[0], p.shape[-1]).argmax(dim=2),
        num_classes=p.shape[-1]
    ).to(J.dtype)
    return config, expected_hyperbmincut(J, config, hyperedges) / 2


def expected_hyperbmincut(J, p, hyperedges):
    """
    Calculate expected hypergraph cut value.

    Uses a continuous relaxation where for each hyperedge, we estimate
    the expected number of clusters it spans.

    Args:
        J: Coupling matrix (unused in current implementation)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        hyperedges: List of hyperedges, each hyperedge is a list of node indices

    Returns:
        total_cut_value: Expected cut value [batch_size]
    """
    total_cut_value = 0.0

    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)
        m = p.shape[2]

        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)  # [batch, m]

        # Continuous mapping: crossing = m - (m-1) * max_ratio
        max_ratio = torch.max(expected_nodes_per_cluster, dim=1)[0] / k
        expected_crossing = m * (1 - max_ratio)

        total_cut_value += expected_crossing

    return total_cut_value


def expected_hyperbmincut_expected_nodes_temped(J, p, hyperedges, temperature=0.1):
    """
    Calculate expected hypergraph cut with temperature-scaled softmax.

    Args:
        J: Coupling matrix (unused)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        hyperedges: List of hyperedges
        temperature: Softmax temperature

    Returns:
        total_cut_value: Expected cut value [batch_size]
    """
    total_cut_value = 0.0

    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)

        he_probs = p[:, he, :]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)

        weights = torch.softmax(expected_nodes_per_cluster / temperature, dim=1)
        weighted_max = torch.sum(weights * expected_nodes_per_cluster, dim=1)

        cut_value = 1 - (weighted_max / k)
        total_cut_value = total_cut_value + cut_value * weight

    return total_cut_value


def expected_hyperbmincut_max_expected_nodes(J, p, hyperedges):
    """
    Calculate expected hypergraph cut using max expected nodes.

    Args:
        J: Coupling matrix (unused)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        hyperedges: List of hyperedges

    Returns:
        total_cut_value: Expected cut value [batch_size]
    """
    total_cut_value = 0.0

    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)

        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)  # [batch, m]
        max_expected_nodes = torch.max(expected_nodes_per_cluster, dim=1)[0]
        cut_value = 1 - (max_expected_nodes / k)
        total_cut_value = total_cut_value + cut_value

    return total_cut_value


def expected_hyperbmincut_all_comb(J, p, hyperedges):
    """
    Calculate expected hypergraph cut using all cluster combinations.

    This version computes exact probabilities for 4 clusters.

    Args:
        J: Coupling matrix (unused)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        hyperedges: List of hyperedges

    Returns:
        total_cut_value: Expected cut value [batch_size]
    """
    total_cut_value = 0.0
    m = 4  # Number of clusters

    # Pre-define all combinations
    pair_masks = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float32, device=p.device)

    triple_masks = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ], dtype=torch.float32, device=p.device)

    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)

        he_probs = p[:, he, :]  # [batch, k, m]
        batch_size = he_probs.shape[0]

        # P(crossing = 1) - vectorized
        prob_single_cluster = torch.sum(torch.prod(he_probs, dim=1), dim=1)

        # P(crossing = 2) - vectorized
        pair_masks_expanded = pair_masks.view(1, 6, 1, 4).expand(batch_size, 6, k, 4)
        he_probs_expanded = he_probs.unsqueeze(1).expand(batch_size, 6, k, 4)

        pair_probs = torch.prod(
            torch.sum(he_probs_expanded * pair_masks_expanded, dim=3),
            dim=2
        )
        sum_2comb = torch.sum(pair_probs, dim=1)
        prob_2_clusters = sum_2comb - 2 * prob_single_cluster

        # P(crossing = 3) - vectorized
        triple_masks_expanded = triple_masks.view(1, 4, 1, 4).expand(batch_size, 4, k, 4)
        he_probs_expanded_triple = he_probs.unsqueeze(1).expand(batch_size, 4, k, 4)

        triple_probs = torch.prod(
            torch.sum(he_probs_expanded_triple * triple_masks_expanded, dim=3),
            dim=2
        )
        sum_3comb = torch.sum(triple_probs, dim=1)
        prob_3_clusters = sum_3comb - 2 * sum_2comb + 3 * prob_single_cluster

        # P(crossing = 4)
        prob_4_clusters = 1 - prob_single_cluster - prob_2_clusters - prob_3_clusters

        epsilon = 1e-3
        prob_2 = torch.log(prob_2_clusters + epsilon)
        prob_3 = torch.log(prob_3_clusters + epsilon)
        prob_4 = torch.log(prob_4_clusters + epsilon)

        total_cut_value += prob_2 + 2 * prob_3 + 3 * prob_4

    return total_cut_value


def expected_hyperbmincut_expected_crossing_simplified(J, p, hyperedges):
    """
    Calculate expected hypergraph cut using simplified crossing estimation.

    Args:
        J: Coupling matrix (unused)
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        hyperedges: List of hyperedges

    Returns:
        total_cut_value: Expected cut value [batch_size]
    """
    total_cut_value = 0.0

    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)

        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)
        e = expected_nodes_per_cluster / k  # Normalized expected node count
        p_used = 1 - torch.exp(-k * e)

        # Expected crossing
        expected_crossing = torch.sum(p_used, dim=1)

        cut_value = torch.relu(expected_crossing - 1) * weight
        total_cut_value = total_cut_value + cut_value

    return total_cut_value


def manual_grad_hyperbmincut(J, p, U_max, L_min, n, h, imbalance_weight, q, batch_size):
    """
    Compute manual gradient for hyperbmincut optimization.

    Args:
        J: Coupling matrix [num_nodes, num_nodes]
        p: Probability distribution [batch_size, num_nodes, num_clusters]
        U_max: Upper bound on cluster size
        L_min: Lower bound on cluster size
        n: Number of nodes
        h: Logits [batch_size, num_nodes, num_clusters]
        imbalance_weight: Weight for imbalance penalty
        q: Number of clusters
        batch_size: Batch size

    Returns:
        total_grad: Gradient [batch_size, num_nodes, num_clusters]
    """
    group_sizes = p.sum(dim=1)  # [batch, q]

    temperature = 0.1
    indicator_upper = torch.sigmoid((group_sizes - U_max) / temperature)
    indicator_lower = torch.sigmoid((L_min - group_sizes) / temperature)

    balance_grad = imbalance_weight * (indicator_upper - indicator_lower)

    # Expand to each node
    balance_grad_expanded = balance_grad.unsqueeze(1).expand(-1, n, -1)

    cut_grad = torch.zeros_like(h)
    for k in range(q):
        p_k = p[:, :, k]  # [batch, n_nodes]
        for b in range(batch_size):
            cut_grad[b, :, k] = torch.matmul(J, 1 - 2 * p_k[b])

    total_grad = cut_grad + balance_grad_expanded

    return total_grad
