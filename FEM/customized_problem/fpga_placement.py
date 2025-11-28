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
    expected_coords = get_placements_from_index_st(p, site_coords_matrix)

    # expected_coords = get_expected_placements_from_index(p, site_coords_matrix)
    
    coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
    coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    
    weighted_dist = manhattan_dist * J.unsqueeze(0)  # [batch_size, num_instances, num_instances]
    
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]  # [batch_size, num_pairs]
    
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)  # [batch_size]
    return total_wirelength

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

# timing loss
def get_timing_loss(design, timer):
    wirelength = get_hpwl_loss(design)
    return max(0, 1000 - wirelength / 1000)

# position constraints loss
def get_constraints_loss(p):
    _, num_instance, num_sites = p.shape
    # p_normalized = Func.softmax(p, dim=-1)
    site_usage = torch.sum(p, dim=1)
    site_constraint_softplus = torch.sum(Func.softplus(10 * (site_usage - num_instance / num_sites))**2, dim=1)
    return site_constraint_softplus

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

# expected placement loss
def expected_fpga_placement(J, p, io_site_connect_matrix, site_coords_matrix, net_sites_tensor, best_hpwl):
    total_energy = 0
    hpwl_weight, timing_weight, constrain_weight = 1, 1, 10

    current_hpwl = get_hpwl_loss(J, p, site_coords_matrix)
    # current_hpwl = get_hpwl_loss_lse(J, p, bbox_length, site_coords_matrix)

    # current_hpwl = get_hpwl_loss_from_net_tensor_lse(p, net_sites_tensor, site_coords_matrix)

    improved_mask = current_hpwl < best_hpwl
    best_hpwl[improved_mask] = current_hpwl[improved_mask]

    # constrain_loss = get_reward_based_constraints_loss(p, current_hpwl, best_hpwl, constrain_weight)
    # constrain_loss = get_constraints_loss(p)

    constrain_loss = get_constraints_loss_site_exclusivity_st(p) + get_constraints_loss_centering(p, site_coords_matrix)
    
    # total_energy += hpwl_weight *  hpwl_loss + constrain_weight * constrain_loss
    
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

def infer_placements(J, p, area_width, site_coords_matrix):
    site_indices = torch.argmax(p, dim=2)
    site_coords = get_site_coordinates_from_index(site_indices, area_width)
    result = get_hpwl_loss(J, p, site_coords_matrix)
    return site_coords, result

def infer_placements_xy(J, p_x, p_y):
    site_coords = get_site_coordinates_from_px_py(p_x, p_y)
    result = get_hpwl_loss_xy_simple(J, p_x, p_y)
    return site_coords, result