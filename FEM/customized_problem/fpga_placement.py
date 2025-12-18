import torch
import torch.nn.functional as Func

# coords functions
def get_site_coordinates_from_index(expected_site_index, area_width):
    x_coords = expected_site_index % area_width
    y_coords = expected_site_index // area_width
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()

def get_site_coordinates_from_px_py(p_x, p_y):
    x = torch.argmax(p_x, dim=2)
    y = torch.argmax(p_y, dim=2)
    return torch.stack([x, y], dim=2)

def get_site_distance_matrix(coords):
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    distances = torch.sum(torch.abs(coords_i - coords_j), dim=2)
    return distances

def get_expected_placements_from_index(p, site_coords_matrix):
    expected_coords = torch.matmul(p, site_coords_matrix)
    return expected_coords

def get_site_coords_all(num_locations, area_width):
    indices = torch.arange(num_locations, dtype=torch.float32)
    x_coords = indices % area_width
    y_coords = indices // area_width
    return torch.stack([x_coords, y_coords], dim=1)

def get_grid_coords_xy(area_width):
    grid_x_coords = torch.arange(area_width, dtype=torch.float32)
    grid_y_coords = torch.arange(area_width, dtype=torch.float32)
    return grid_x_coords, grid_y_coords

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

def get_placements_from_xy_st(p_x, p_y, grid_x_coords, grid_y_coords):
    with torch.no_grad():
        x_hard = grid_x_coords[torch.argmax(p_x, dim=2)] 
        y_hard = grid_y_coords[torch.argmax(p_y, dim=2)] 
        hard_coords = torch.stack([x_hard, y_hard], dim=2)
    
    expected_x = torch.matmul(p_x, grid_x_coords.unsqueeze(1)).squeeze(2) 
    expected_y = torch.matmul(p_y, grid_y_coords.unsqueeze(1)).squeeze(2)
    expected_coords = torch.stack([expected_x, expected_y], dim=2)
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()
    return straight_coords

# hpwl loss
def get_hpwl_loss(J, p, site_coords_matrix):
    # expected_coords = get_placements_from_index_st(p, site_coords_matrix)
    expected_coords = get_expected_placements_from_index(p, site_coords_matrix)
    
    coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
    coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    
    weighted_dist = manhattan_dist * J.unsqueeze(0)  # [batch_size, num_instances, num_instances]
    
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]  # [batch_size, num_pairs]
    
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)  # [batch_size]
    return total_wirelength

# use LSE model to estimate wire length
def get_hpwl_loss_from_net_tensor_lse(p, net_tensor, site_coords_matrix, gamma=0.1):
    batch_size, num_instances, num_sites = p.shape
    num_nets = net_tensor.shape[0]
    
    # 计算期望坐标
    expected_coords = torch.matmul(p, site_coords_matrix)
    expected_x = expected_coords[..., 0]
    expected_y = expected_coords[..., 1]
    
    # 扩展用于批量计算
    net_tensor_expanded = net_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    net_x = expected_x.unsqueeze(1).expand(-1, num_nets, -1)
    net_y = expected_y.unsqueeze(1).expand(-1, num_nets, -1)
    
    small_value = -1e6
    
    def compute_boundaries(coords, net_mask):
        coords_masked = torch.where(net_mask, coords, torch.tensor(small_value, device=p.device))
        coords_neg_masked = torch.where(net_mask, -coords, torch.tensor(small_value, device=p.device))
        
        coord_max = gamma * torch.logsumexp(coords_masked / gamma, dim=2)
        coord_min = -gamma * torch.logsumexp(coords_neg_masked / gamma, dim=2)
        
        return coord_max, coord_min
    
    # 计算x和y方向的边界
    x_max, x_min = compute_boundaries(net_x, net_tensor_expanded)
    y_max, y_min = compute_boundaries(net_y, net_tensor_expanded)
    
    # 计算HPWL
    net_hpwl = (x_max - x_min) + (y_max - y_min)
    total_hpwl = torch.sum(net_hpwl, dim=1)
    
    return total_hpwl



    # batch_size, num_instance, num_sites = p.shape
    # penalty_strength = 2
    # site_usage = torch.sum(p, dim=1)
    # overuse = torch.relu(site_usage - num_instance / num_sites)
    # penalty = torch.sum(torch.exp(overuse * penalty_strength) - 1.0, dim=1)
    # # penalty = torch.sum(overuse ** 3, dim=1)
    # return penalty

    # _, num_instance, num_sites = p.shape
    # site_usage = torch.sum(p, dim=1)
    # log_penalty = torch.sum(torch.log1p(Func.relu(site_usage - num_instance / num_sites)), dim=1)
    # return log_penalty

# use WA model to estimate wire length
def get_hpwl_loss_from_net_tensor_wa_vectorized(p, net_tensor, site_coords_matrix, gamma=0.03):

    # print(net_tensor)

    batch_size, num_instances, num_sites = p.shape
    num_nets = net_tensor.shape[0]

    # print(net_tensor.shape)
    
    # 1. 计算期望坐标
    expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
    expected_x = expected_coords[..., 0]  # [batch_size, num_instances]
    expected_y = expected_coords[..., 1]  # [batch_size, num_instances]
    
    # 2. 对坐标进行归一化（关键步骤！）
    # 假设你的网格尺寸是 area_width × area_height
    # 或者使用每个batch的统计信息
    x_min = expected_x.min(dim=1, keepdim=True)[0]
    x_max = expected_x.max(dim=1, keepdim=True)[0]
    y_min = expected_y.min(dim=1, keepdim=True)[0]
    y_max = expected_y.max(dim=1, keepdim=True)[0]
    
    # 归一化到 [0, 1] 范围
    x_norm = (expected_x - x_min) / (x_max - x_min + 1e-10)
    y_norm = (expected_y - y_min) / (y_max - y_min + 1e-10)
    
    net_tensor_expanded = net_tensor.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nets, num_instances]    
    total_hpwl = torch.zeros(batch_size, device=p.device)
    
    for coord_norm in [x_norm, y_norm]:  # 使用归一化后的坐标
        # 扩展坐标张量
        coord_expanded = coord_norm.unsqueeze(1).expand(-1, num_nets, -1)  # [batch_size, num_nets, num_instances]
        coord_masked = coord_expanded * net_tensor_expanded

        coord_for_max = coord_masked.clone()
        zero_mask_max = (coord_masked == 0) & (net_tensor_expanded == 0)
        coord_for_max[zero_mask_max] = -float('inf')
        max_vals = coord_for_max.max(dim=2, keepdim=True)[0]
        
        # 对于WA_min：将0替换为+inf
        coord_for_min = coord_masked.clone()
        zero_mask_min = (coord_masked == 0) & (net_tensor_expanded == 0)
        coord_for_min[zero_mask_min] = float('inf')
        min_vals = coord_for_min.min(dim=2, keepdim=True)[0]
        
        weight_pos_safe = torch.exp((coord_masked - max_vals) / gamma)
        numerator_pos = torch.sum(coord_masked * weight_pos_safe * net_tensor_expanded, dim=2)
        denominator_pos = torch.sum(weight_pos_safe * net_tensor_expanded, dim=2)
        
        weight_neg_safe = torch.exp(-(coord_masked - min_vals) / gamma)
        numerator_neg = torch.sum(coord_masked * weight_neg_safe * net_tensor_expanded, dim=2)
        denominator_neg = torch.sum(weight_neg_safe * net_tensor_expanded, dim=2)
        
        # 避免除零
        denominator_pos = torch.where(denominator_pos == 0, torch.tensor(1e-10, device=p.device), denominator_pos)
        denominator_neg = torch.where(denominator_neg == 0, torch.tensor(1e-10, device=p.device), denominator_neg)
        
        # 计算加权平均值
        weighted_avg_pos = numerator_pos / denominator_pos
        weighted_avg_neg = numerator_neg / denominator_neg
        
        # 当前坐标方向的wirelength
        coord_wirelength = weighted_avg_pos - weighted_avg_neg
        
        # 将归一化的长度转换回原始尺度
        # coord_range = x_max - x_min if coord_tensor is x_norm else y_max - y_min
        # coord_wirelength_original = coord_wirelength * coord_range.squeeze(1)
        
        total_hpwl += torch.sum(coord_wirelength, dim=1)
    
    return total_hpwl

def get_hpwl_loss_xy_simple(J, p_x, p_y):
    
    batch_size, num_instances, num_x = p_x.shape

    grid_x_coords, grid_y_coords = get_grid_coords_xy(num_x)

    expected_coords = get_placements_from_xy_st(p_x, p_y, grid_x_coords, grid_y_coords)

    coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
    coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    
    weighted_dist = manhattan_dist * J.unsqueeze(0)
    
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]
    
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)
    return total_wirelength

def get_hpwl_loss_topk_simple(J, p, site_coords_matrix, k=3):
    batch_size, num_instances, num_sites = p.shape
    
    # 获取topk
    topk_values, topk_indices = torch.topk(p, k=k, dim=2)
    
    # 重新归一化topk概率
    topk_probs = topk_values / (topk_values.sum(dim=2, keepdim=True) + 1e-10)
    
    # 获取topk位置的坐标
    topk_coords = site_coords_matrix[topk_indices]
    
    # 计算加权平均坐标
    expected_coords = torch.sum(topk_probs.unsqueeze(-1) * topk_coords, dim=2)
    
    # 计算HPWL
    coords_i = expected_coords.unsqueeze(2)
    coords_j = expected_coords.unsqueeze(1)
    
    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    weighted_dist = manhattan_dist * J.unsqueeze(0)
    
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]
    
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)
    
    return total_wirelength

# timing loss
def get_timing_loss(design, timer):
    wirelength = get_hpwl_loss(design)
    return max(0, 1000 - wirelength / 1000)

# position constraints loss
def get_constraints_loss(p):
    _, num_instance, num_sites = p.shape
    # p_normalized = Func.softmax(p, dim=-1)
    # topk_values, _ = torch.topk(p, k=5, dim=1)  # [batch, k, site]
    
    site_usage = torch.sum(p, dim=1)
    # site_usage = torch.sum(topk_values, dim=1)  # [batch, site]

    expected_usage_per_site = float(num_instance) / num_sites
    site_constraint_softplus = torch.sum((30 * Func.relu(site_usage - expected_usage_per_site))**2, dim=1)
    return site_constraint_softplus

def get_constraints_loss_diff(p, site_coords_matrix):
    """
    智能约束损失：区分不同违反情况
    """
    batch_size, num_instance, num_sites = p.shape
    
    site_usage = torch.sum(p, dim=1)
    # expected_usage = (num_instance / num_sites) * 1.5

    expected_usage = 0.5
    
    # 计算竞争指标
    max_prob = torch.max(p, dim=1)[0]  # 每个位置的最大概率
    prob_variance = torch.var(p, dim=1)  # 每个位置的概率方差
    
    # 竞争强度 = (1 - 最大概率) * (1 + 方差)
    # 最大概率小且方差大 -> 竞争激烈
    competition = (1 - max_prob) * (1 + prob_variance)
    # print()
    # 基础违反量
    # usage_violation = Func.relu(site_usage - expected_usage)
    
    # 最终违反程度 = 基础违反量 * (1 + 竞争强度)
    # 竞争越激烈，惩罚越重
    # final_violation = usage_violation * (1 + competition)
    final_violation = (competition)
    # 损失
    loss = torch.sum((3 * final_violation)**2, dim=1)

    return loss

def get_constraints_loss_rpt(p, site_coords_matrix):
    batch_size, num_instance, num_sites = p.shape
    
    site_usage = torch.sum(p, dim=1)
    hard_placements = get_hard_placements_from_index(p, site_coords_matrix)
    
    for b in range(batch_size):
        if b > 0:
            break
        
        batch_placements = hard_placements[b]
        
        # 使用字典记录每个位置的实例索引
        position_dict = {}
        for inst_idx, (x, y) in enumerate(batch_placements):
            key = (x.item(), y.item())
            if key not in position_dict:
                position_dict[key] = []
            position_dict[key].append(inst_idx)
        
        # 检查重复
        for (x, y), inst_indices in position_dict.items():
            if len(inst_indices) > 1:
                # 找到对应的site索引
                site_idx = None
                for idx in range(num_sites):
                    site_coord = site_coords_matrix[idx]
                    if torch.allclose(site_coord, torch.tensor([x, y])):
                        site_idx = idx
                        break
                
                if site_idx is not None:
                    usage = site_usage[b, site_idx].item()
                    print(f"Batch {b}: poz ({x:.0f}, {y:.0f}), instance {len(inst_indices)}, usage {usage:.4f}")

                    # avg_prob = sum(p[b, i, site_idx].item() for i in inst_indices) / len(inst_indices)
                    # print(f"  平均概率: {avg_prob:.4f}")
    
    expected_usage_per_site = (num_instance / num_sites) * 3
    # site_constraint_softplus = torch.sum(Func.softplus(30 * (site_usage - expected_usage_per_site))**2, dim=1)

    site_constraint_softplus = torch.sum((30 * Func.relu(site_usage - expected_usage_per_site))**2, dim=1)
    
    return site_constraint_softplus

def get_constraints_loss_topk(p, site_coords_matrix, k=2):
    """
    考虑top-k个最高概率的实例
    """
    batch_size, num_instance, num_sites = p.shape
    
    site_usage = torch.sum(p, dim=1)
    
    # 获取每个位置top-k的概率
    topk_values, _ = torch.topk(p, k=k, dim=1)  # [batch, k, site]
    
    # 方案1：top-k概率之和作为竞争指标
    topk_sum = torch.sum(topk_values, dim=1)  # [batch, site]
    
    # 方案2：top-k概率的方差（衡量是否均匀）
    topk_variance = torch.var(topk_values, dim=1)  # [batch, site]
    
    expected_usage = (num_instance / num_sites) * 1.5
    usage_violation = Func.relu(site_usage - expected_usage)
    
    # 如果top-k概率之和很大且方差很小（均匀分布），说明竞争激烈
    competition_score = topk_sum * (1 - topk_variance)
    violation_severity = usage_violation * (1 + competition_score)
    
    loss = torch.sum((30 * violation_severity)**2, dim=1)
    
    return loss

def get_constraints_loss_electric_field(p, site_coords_matrix, push_strength=10.0):
    """
    电场力式约束：基于密度梯度产生推力
    """
    batch_size, num_instances, num_sites = p.shape
    site_usage = torch.sum(p, dim=1)  # [batch_size, num_sites]
    
    # 创建密度图 [batch_size, layout_size, layout_size]
    layout_size = int(num_sites ** 0.5)
    density_map = site_usage.reshape(batch_size, layout_size, layout_size)
    
    # 计算密度梯度（电场方向）
    # 使用Sobel算子计算密度梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=p.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=p.device).view(1, 1, 3, 3)
    
    # 计算密度梯度 [batch_size, 2, layout_size, layout_size]
    density_grad_x = Func.conv2d(density_map.unsqueeze(1), sobel_x, padding=1)
    density_grad_y = Func.conv2d(density_map.unsqueeze(1), sobel_y, padding=1)
    density_grad = torch.stack([density_grad_x.squeeze(1), density_grad_y.squeeze(1)], dim=1)
    
    # 计算每个实例受到的推力
    expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
    
    # 将连续坐标映射到网格索引
    grid_coords = (expected_coords * (layout_size - 1)).long()  # [batch_size, num_instances, 2]
    
    total_push_force = torch.zeros(batch_size, device=p.device)
    
    for b in range(batch_size):
        for i in range(num_instances):
            x, y = grid_coords[b, i, 0], grid_coords[b, i, 1]
            
            # 确保在边界内
            x = torch.clamp(x, 0, layout_size-1)
            y = torch.clamp(y, 0, layout_size-1)
            
            # 获取该位置的密度梯度（推力方向）
            force_direction = density_grad[b, :, y, x]  # [2]
            
            # 推力大小与密度成正比
            local_density = density_map[b, y, x]
            force_magnitude = local_density * push_strength
            
            # 推力损失：实例应该沿着梯度反方向移动（从高密度到低密度）
            # 我们惩罚实例不沿着推力方向移动的程度
            instance_force = force_magnitude * torch.norm(force_direction)
            total_push_force[b] += instance_force
    
    return total_push_force

def get_constraints_loss_max_prob(p, threshold=0.7):
    batch_size, num_instances, num_sites = p.shape
    max_probs, max_indices = torch.max(p, dim=2)
    max_prob_mask = torch.zeros_like(p)
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_instances)
    instance_indices = torch.arange(num_instances).unsqueeze(0).expand(batch_size, -1)
    max_prob_mask[batch_indices, instance_indices, max_indices] = max_probs
    
    site_usage = torch.sum(max_prob_mask, dim=1) 
    target_usage = num_instances / num_sites
    excess_usage = Func.relu(site_usage - target_usage)
    constraint_loss = torch.sum(excess_usage ** 2, dim=1)
    return constraint_loss

def get_constraints_loss_hard_exclusion_st(p, site_coords_matrix, exclusion_radius=1.0, strength=50.0):
    batch_size, num_instances, num_sites = p.shape
    
    # 硬坐标（用于前向计算）
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
        if site_coords_matrix.dim() == 2:
            hard_coords = site_coords_matrix[site_indices]
        else:
            batch_indices = torch.arange(batch_size, device=p.device).unsqueeze(1).expand(-1, num_instances)
            hard_coords = site_coords_matrix[batch_indices, site_indices, :]
    
    # 软坐标（用于梯度）
    expected_coords = torch.matmul(p, site_coords_matrix)
    
    # 硬重叠计算（前向）
    hard_overlap_penalty = torch.zeros(batch_size, device=p.device)
    for b in range(batch_size):
        overlap_count = 0
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = torch.norm(hard_coords[b, i] - hard_coords[b, j])
                if distance < exclusion_radius:
                    overlap_count += 1
                    overlap_penalty = (exclusion_radius - distance) ** 2
                    hard_overlap_penalty[b] += overlap_penalty
        
        if overlap_count > 0:
            hard_overlap_penalty[b] += overlap_count * 10.0
    
    # 软重叠计算（梯度）
    soft_overlap_penalty = torch.zeros(batch_size, device=p.device)
    for b in range(batch_size):
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = torch.norm(expected_coords[b, i] - expected_coords[b, j])
                if distance < exclusion_radius:
                    overlap_penalty = (exclusion_radius - distance) ** 2
                    soft_overlap_penalty[b] += overlap_penalty
    
    # Straight-Through
    final_penalty = soft_overlap_penalty + (hard_overlap_penalty - soft_overlap_penalty).detach()
    
    return final_penalty * strength

def get_constraints_loss_centering(p, site_coords_matrix, center_strength=5.0):
    batch_size, num_instances, num_sites = p.shape

    expected_coords = torch.matmul(p, site_coords_matrix)
    center = torch.mean(site_coords_matrix, dim=0)
    
    total_centering_loss = torch.zeros(batch_size, device=p.device)
    
    for b in range(batch_size):
        # 计算所有实例到中心的平均距离
        distances_to_center = torch.norm(expected_coords[b] - center, dim=1)
        avg_distance = torch.mean(distances_to_center)
        total_centering_loss[b] = avg_distance * center_strength
    
    return total_centering_loss

def get_constraints_loss_site_exclusivity_st(p, strength=10.0):
    batch_size, num_instances, num_sites = p.shape
    
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
    
    total_overlap_penalty = torch.zeros(batch_size)
    
    for b in range(batch_size):
        # 使用bincount统计每个站点的实例数量
        site_counts = torch.bincount(site_indices[b], minlength=num_sites)
        
        # 惩罚有多个实例的站点
        overlapping_sites = site_counts > 1
        overlap_degree = torch.sum(site_counts[overlapping_sites] - 1)
        total_overlap_penalty[b] = overlap_degree * strength
    
    site_usage = torch.sum(p, dim=1)  # [batch_size, num_sites]
    
    soft_overlap = Func.softplus(site_usage - num_instances / num_sites)
    soft_overlap_penalty = torch.sum(soft_overlap, dim=1) * strength
    
    final_penalty = soft_overlap_penalty + (total_overlap_penalty - soft_overlap_penalty).detach()
    
    return final_penalty

def get_constraints_loss_enhanced(p, bin_sizes=[4, 8, 16], weights=[1.0, 0.5, 0.2]):
    batch_size, num_instances, num_sites = p.shape
    site_usage = torch.sum(p, dim=1)  # [batch_size, num_sites]
    
    # 假设布局是方形的
    layout_size = int(num_sites ** 0.5)
    density_map = site_usage.view(batch_size, layout_size, layout_size)
    
    total_constraint = 0.0
    
    for bin_size, weight in zip(bin_sizes, weights):
        # 使用平均池化计算局部密度
        kernel = torch.ones(1, 1, bin_size, bin_size, device=p.device) / (bin_size * bin_size)
        local_density = Func.conv2d(
            density_map.unsqueeze(1), 
            kernel, 
            padding=bin_size//2, 
            stride=1
        ).squeeze(1)
        
        # 更强的惩罚：目标密度更低，惩罚系数更高
        target_density = num_instances / num_sites * 0.6  # 更严格的目标
        density_excess = Func.relu(local_density - target_density)
        
        # 使用更激进的惩罚函数
        bin_constraint = torch.sum(density_excess ** 3, dim=(1, 2))  # 三次方惩罚
        total_constraint += weight * bin_constraint
    
    return total_constraint

def get_constraints_loss_xy(p_x, p_y):
    batch_size, num_instances, num_x = p_x.shape
    _, _, num_y = p_y.shape

    usage = num_instances / (num_x * num_y)
    p_x_expanded = p_x.unsqueeze(3)  # [batch_size, num_instances, num_x, 1]
    p_y_expanded = p_y.unsqueeze(2)  # [batch_size, num_instances, 1, num_y]
    p_instance_xy = p_x_expanded * p_y_expanded  # [batch_size, num_instances, num_x, num_y]

    p_position_usage = torch.sum(p_instance_xy, dim=1)  # [batch_size, num_x, num_y]
    p_position_usage_flat = p_position_usage.flatten(start_dim=1)  # [batch_size, num_x * num_y]
    constraint_loss = torch.sum(Func.softplus(10 * (p_position_usage_flat - usage) ** 2 ), dim=1)  # [batch_size, num_x, num_y]
    return constraint_loss



def get_constraints_loss_expected_coords(p, site_coords_matrix, grid_spacing=1.0):
    """
    新的约束损失：基于期望坐标
    1. 期望坐标之间至少间隔grid_spacing
    2. 期望坐标尽量靠近整数网格点
    """
    batch_size, num_instance, num_sites = p.shape
    
    # 计算期望坐标
    expected_coords = torch.matmul(p, site_coords_matrix)  # [batch, instance, 2]
    
    # 约束1：实例间最小距离约束
    distance_loss = get_minimum_distance_constraint(expected_coords, grid_spacing)
    
    # 约束2：期望坐标靠近整数点约束
    integer_loss = get_integer_alignment_constraint(expected_coords)
    # total_constraint = distance_loss + integer_loss
    
    total_constraint = distance_loss
    
    return total_constraint

def get_minimum_distance_constraint(coords, min_distance=1.0):
    """
    确保任意两个实例的期望坐标至少相隔min_distance
    """
    batch_size, num_instance, _ = coords.shape
    
    # 计算所有实例对之间的距离
    coords_i = coords.unsqueeze(2)  # [batch, instance, 1, 2]
    coords_j = coords.unsqueeze(1)  # [batch, 1, instance, 2]
    
    # 欧氏距离
    # distances = torch.sqrt(torch.sum((coords_i - coords_j)**2, dim=-1) + 1e-8)
    # 或者曼哈顿距离
    distances = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    
    # 创建mask排除对角线（自己对自己的距离）
    mask = torch.eye(num_instance, device=coords.device).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 获取所有实例对的距离（排除对角线）
    pair_distances = distances[~mask].view(batch_size, num_instance, num_instance-1)
    
    # 计算违反最小距离的程度
    violation = 5 * Func.relu(min_distance - pair_distances)  # 距离小于min_distance的部分
    
    # 惩罚违反（平方惩罚更严厉）
    loss = torch.sum(violation**2, dim=(1, 2))
    # print(f'site distance loss {loss}')
    return loss

def get_integer_alignment_constraint(coords, weight=5.0):
    # integer_coords = torch.round(coords)
    # alignment_loss = torch.sum((coords - integer_coords)**2, dim=(1, 2))
    
    pi = torch.tensor(3.1415926535, device=coords.device)
    alignment_loss = torch.sum(torch.sin(pi * coords)**2, dim=(1, 2))

    # print(f'site distance loss {alignment_loss}')

    return weight * alignment_loss


# expected placement loss
def expected_fpga_placement(J, p, io_site_connect_matrix, site_coords_matrix, net_sites_tensor, best_hpwl):
    total_energy = 0
    hpwl_weight, timing_weight, constrain_weight = 1, 1, 20

    current_hpwl = get_hpwl_loss(J, p, site_coords_matrix)
    # current_hpwl = get_hpwl_loss_topk_simple(J, p, site_coords_matrix, k = 30)
    # current_hpwl = get_hpwl_loss_from_net_tensor_wa_vectorized(p, net_sites_tensor, site_coords_matrix)
    # current_hpwl = get_hpwl_loss_from_net_tensor_lse(p, net_sites_tensor, site_coords_matrix)

    # improved_mask = current_hpwl < best_hpwl
    # best_hpwl[improved_mask] = current_hpwl[improved_mask]

    # constrain_loss = get_constraints_loss(p)
    # constrain_loss = get_reward_based_constraints_loss(p, current_hpwl, best_hpwl, constrain_weight)
    constrain_loss = get_constraints_loss_expected_coords(p, site_coords_matrix)
    # constrain_loss = get_constraints_loss_topk(p, site_coords_matrix)
    # constrain_loss = get_constraints_loss_site_exclusivity_st(p) + get_constraints_loss_centering(p, site_coords_matrix)
    # constrain_loss = get_constraints_loss_combined_fast(p)
    
    # total_energy += hpwl_weight *  hpwl_loss + constrain_weight * constrain_loss
    
    return hpwl_weight * current_hpwl, constrain_weight * constrain_loss


def expected_fpga_placement_v2(p, site_coords_matrix, net_sites_tensor):
    hpwl_weight, constrain_weight = 1, 2
    current_hpwl = get_hpwl_loss_from_net_tensor_wa_vectorized(p, net_sites_tensor, site_coords_matrix)
    # constrain_loss = get_constraints_loss(p)

    constrain_loss = get_constraints_loss_rpt(p, site_coords_matrix)
    
    return hpwl_weight * current_hpwl, constrain_weight * constrain_loss

def expected_fpga_placement_xy(J, p_x, p_y):
    hpwl_weight, constrain_weight = 1, 1
    current_hpwl = get_hpwl_loss_xy_simple(J, p_x, p_y)
    constrain_loss = get_constraints_loss_xy(p_x, p_y)
    return hpwl_weight * current_hpwl, constrain_weight * constrain_loss


# infer placment
def infer_site_coordinates(expected_site_index, area_width):
    x_coords = expected_site_index % area_width
    y_coords = expected_site_index // area_width
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()

def infer_placements(J, p, area_width, site_coords_matrix, net_sites_tensor):
    site_indices = torch.argmax(p, dim=2)
    site_coords = get_site_coordinates_from_index(site_indices, area_width)
    print(f'argmax {site_coords[0]}')
    expected_coords = get_expected_placements_from_index(p, site_coords_matrix)
    # print(f'expected {expected_coords[0]}')

    integer_coords = torch.round(expected_coords)
    # print(f'integer rounded {integer_coords[0]}')

    result = get_hpwl_loss(J, p, site_coords_matrix)


    # result = get_hpwl_loss_from_net_tensor_wa_vectorized(p, net_sites_tensor, site_coords_matrix)
    return site_coords, result

def infer_placements_xy(J, p_x, p_y):
    site_coords = get_site_coordinates_from_px_py(p_x, p_y)
    result = get_hpwl_loss_xy_simple(J, p_x, p_y)
    return site_coords, result