import torch
import torch.nn.functional as Func

_hpwl_loss_history = []
_constrain_loss_history = []
_total_loss_history = []
_placement_history = []

show_steps = [50, 100, 150, 199]

# coords functions
def get_inst_coords_from_index(inst_indices, area_width):
    x_coords = inst_indices % area_width
    y_coords = inst_indices // area_width
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()

def get_io_coords_from_index(inst_indices):
    x_coords = torch.full_like(inst_indices, 0, dtype=torch.int32)
    y_coords = inst_indices
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()

def get_site_distance_matrix(coords):
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    distances = torch.sum(torch.abs(coords_i - coords_j), dim=2)
    return distances

def get_expected_placements_from_index(p, site_coords_matrix):
    expected_coords = torch.matmul(p, site_coords_matrix)
    return expected_coords

def get_hard_placements_from_index(p, site_coords_matrix):
    site_indices = torch.argmax(p, dim=2)
    hard_coords = site_coords_matrix[site_indices]  # [batch_size, num_instances, 2]
    return hard_coords

def get_placements_from_index_st(p, site_coords_matrix):
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
        hard_coords = site_coords_matrix[site_indices]
    
    expected_coords = torch.matmul(p, site_coords_matrix)
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()
    
    return straight_coords

# hpwl loss
def get_hpwl_loss_qubo(J, p, site_coords_matrix):
    batch_size, num_instances, num_sites = p.shape
    
    # 1. 距离矩阵
    coords_i = site_coords_matrix.unsqueeze(1)  # [num_sites, 1, 2]
    coords_j = site_coords_matrix.unsqueeze(0)  # [1, num_sites, 2]
    D = torch.sum(torch.abs(coords_i - coords_j), dim=2)  # [num_sites, num_sites]
    
    # 2. 批处理矩阵乘法: (p @ D) @ p^T
    PD = torch.matmul(p, D)  # [batch_size, num_instances, num_sites]
    P_transposed = p.transpose(1, 2)  # [batch_size, num_sites, num_instances]
    E_matrix = torch.bmm(PD, P_transposed)  # [batch_size, num_instances, num_instances]
    
    # 3. 上三角掩码
    triu_mask = torch.triu(torch.ones(num_instances, num_instances, device=p.device), diagonal=1).bool()
    
    # 4. 加权并求和
    weighted_E = E_matrix * J.unsqueeze(0)  # 广播J到每个batch
    weighted_E_triu = weighted_E[:, triu_mask]  # [batch_size, num_pairs]
    total_wirelength = torch.sum(weighted_E_triu, dim=1)  # [batch_size]
    
    return total_wirelength

def get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io,
                           logic_site_coords_matrix, io_site_coords_matrix):
    batch_size, n_logic, _ = p_logic.shape
    
    device = p_logic.device
    
    D_LL = None
    D_LI = None
    
    x_logic = logic_site_coords_matrix[:, 0]  # [L]
    y_logic = logic_site_coords_matrix[:, 1]  # [L]
    
    Dx_LL = torch.abs(x_logic.unsqueeze(1) - x_logic.unsqueeze(0))  # [L, L]
    Dy_LL = torch.abs(y_logic.unsqueeze(1) - y_logic.unsqueeze(0))  # [L, L]
    D_LL = Dx_LL + Dy_LL  # [L, L]
    
    x_io = io_site_coords_matrix[:, 0]  # [M]
    y_io = io_site_coords_matrix[:, 1]  # [M]
    
    Dx_LI = torch.abs(x_logic.unsqueeze(1) - x_io.unsqueeze(0))  # [L, M]
    Dy_LI = torch.abs(y_logic.unsqueeze(1) - y_io.unsqueeze(0))  # [L, M]
    D_LI = Dx_LI + Dy_LI  # [L, M]
    
    total_wl = torch.zeros(batch_size, device=device)
    
    # 1. logic-logic
    PD = torch.matmul(p_logic, D_LL)
    p_logic_T = p_logic.transpose(1, 2)  # [B, L, N]
    E = torch.bmm(PD, p_logic_T)  # [B, N, N]
    triu_mask = torch.triu(torch.ones(n_logic, n_logic, device=device), diagonal=1)
    wl_LL = torch.sum(E * J_LL.unsqueeze(0) * triu_mask.unsqueeze(0), dim=(1, 2))
    total_wl += wl_LL
    
    # 2. logic-io
    PD_LI = torch.matmul(p_logic, D_LI)  # [B, N, M]
    p_io_T = p_io.transpose(1, 2)  # [B, M, M_io]
    E_LI = torch.bmm(PD_LI, p_io_T)  # [B, N, M_io]
    wl_LI = torch.sum(E_LI * J_LI.unsqueeze(0), dim=(1, 2))
    total_wl += wl_LI
    
    return total_wl

# position constraints loss
def get_constraints_loss(p):
    _, num_instance, num_sites = p.shape
    # topk_values, _ = torch.topk(p, k=5, dim=1)  # [batch, k, site]
    
    site_usage = torch.sum(p, dim=1)
    # site_usage = torch.sum(topk_values, dim=1)  # [batch, site]

    expected_usage_per_site = float(num_instance) / num_sites
    site_constraint = torch.sum(30 * Func.softplus(site_usage - 1)**2, dim=1)
    return site_constraint

def get_constraints_loss_with_io(p_logic, p_io):
    coeff_1 = p_logic.shape[1] / 2
    logic_site_usage = torch.sum(p_logic, dim=1)
    logic_constraint = torch.sum(coeff_1 * Func.softplus(logic_site_usage - 1)**2, dim=1)
    
    coeff_2 = p_io.shape[1] / 2
    io_site_usage = torch.sum(p_io, dim=1)
    io_constraint = torch.sum(coeff_2 * Func.softplus(io_site_usage - 1)**2, dim=1)

    return logic_constraint + io_constraint

def manual_grad_hpwl_loss(p, W, D):
    batch_size, _, _ = p.shape
    
    # 计算 PD = P @ D
    PD = torch.matmul(p, D)  # [batch, N, M]
    W_batch = W.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, N, N]
    h_grad = torch.bmm(W_batch, PD)  # [batch, N, M]
    
    return h_grad

def manual_grad_constraint_loss(p, lambda_constraint=30.0):
    site_occupancy = torch.sum(p, dim=1)  # [batch, M]
    excess = site_occupancy - 1.0
    
    # 您的autograd版本：loss = sum(λ * softplus(excess)^2)
    # 梯度：dL/dp_ij = λ * 2 * softplus(excess_j) * sigmoid(excess_j)
    
    softplus_val = Func.softplus(excess)
    sigmoid_val = torch.sigmoid(excess)
    
    site_grad = 2 * lambda_constraint * softplus_val * sigmoid_val  # [batch, M]
    h_grad = site_grad.unsqueeze(1).repeat(1, p.shape[1], 1)  # [batch, N, M]
    
    return h_grad

def manual_grad_placement(p, J, site_coords_matrix, lambda_constraint=30.0):
    batch_size, N, M = p.shape
    
    # 1. 距离矩阵
    coords_i = site_coords_matrix.unsqueeze(1)
    coords_j = site_coords_matrix.unsqueeze(0)
    D = torch.sum(torch.abs(coords_i - coords_j), dim=2)
    
    # 2. 计算 dF/dP
    PD = torch.matmul(p, D)
    mask = torch.triu(torch.ones(N, N, device=p.device), diagonal=1).bool()
    J_upper = J * mask
    J_batch = J_upper.unsqueeze(0).expand(batch_size, -1, -1)
    dE_hpwl_dp = torch.bmm(J_batch, PD)
    
    site_occupancy = torch.sum(p, dim=1)
    excess = site_occupancy - 1.0
    softplus_val = Func.softplus(excess)
    sigmoid_val = torch.sigmoid(excess)
    site_grad = 2 * lambda_constraint * softplus_val * sigmoid_val
    dE_constraint_dp = site_grad.unsqueeze(1).expand(-1, N, -1)
    
    dE_dp = dE_hpwl_dp + dE_constraint_dp

    # 计算 Σ_k grad_p_ik * p_ik 对于每个i
    sum_term = torch.sum(dE_dp * p, dim=2, keepdim=True)  # [batch, N, 1]
    
    # 计算 grad_h
    dE_dh = dE_dp * p - p * sum_term  # [batch, N, M]
    return dE_dh

# expected placement loss
def expected_fpga_placement(J, p, site_coords_matrix, step, area_width, alpha):
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

def get_loss_history():
    return {
        'hpwl_losses': _hpwl_loss_history.copy(),
        'constrain_losses': _constrain_loss_history.copy(),
        'total_losses': _total_loss_history.copy()
    }
    
def get_placment_history():
    return _placement_history.copy()

def expected_fpga_placement_with_io(J_LL, J_LI,  p_logic, p_io, logic_site_coords, io_site_coords):
    hpwl_weight, constrain_weight = 1, 20
    current_hpwl = get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io, logic_site_coords, io_site_coords)
    constrain_loss = get_constraints_loss_with_io(p_logic, p_io)
    return hpwl_weight * current_hpwl + constrain_weight * constrain_loss

# infer placment
def infer_placements(J, p, area_width, site_coords_matrix):
    inst_indices = torch.argmax(p, dim=2)
    inst_coords = get_inst_coords_from_index(inst_indices, area_width)
    result = get_hpwl_loss_qubo(J, p, site_coords_matrix)
    return inst_coords, result

def infer_placements_with_io(J_LL, J_LI, p_logic, p_io, area_width, logic_site_coords_matrix, io_site_coords_matrix):
    logic_inst_indices = torch.argmax(p_logic, dim=2)
    io_inst_indices = torch.argmax(p_io, dim=2)
    logic_inst_coords = get_inst_coords_from_index(logic_inst_indices, area_width)
    io_inst_coords = get_io_coords_from_index(io_inst_indices)
    result = get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io, logic_site_coords_matrix, io_site_coords_matrix)
    return [logic_inst_coords, io_inst_coords], result