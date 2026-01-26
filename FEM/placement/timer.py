import torch

class Timer:
    def __init__(self):
        pass

    def setup_timing_analysis(self, design, timing_library):
        self.timing_library = timing_library
        self.cell_delays = self.extract_cell_delays(design)
        self.net_delays = self.extract_net_delays(design)
        self.timing_paths = self.extract_timing_paths(design)
        
    def extract_cell_delays(self, design):
        cell_delays = {}
        for cell in design.getCells():
            cell_type = cell.getType()
            # 从时序库中获取单元延迟
            if cell_type in self.timing_library:
                cell_delays[cell.getName()] = self.timing_library[cell_type]
            else:
                # 默认延迟
                cell_delays[cell.getName()] = {
                    'min_delay': 0.1,
                    'max_delay': 0.2,
                    'setup_time': 0.05,
                    'hold_time': 0.02
                }
        return cell_delays

    def extract_net_delays(self, design):
        net_delays = {}
        for net in design.getNets():
            net_length = self.estimate_net_length(net)
            net_delays[net.getName()] = {
                'unit_delay': 0.01,
                'estimated_delay': net_length * 0.01
            }
        return net_delays

    def extract_timing_paths(self, design):
        timing_paths = []
        
        for cell in design.getCells():
            if 'FD' in cell.getType():  # 触发器
                timing_paths.append({
                    'start_cell': cell.getName(),
                    'end_cell': self.find_connected_flipflop(cell),
                    'required_time': 10.0,  # 要求的时序
                    'criticality': 1.0
                })
        
        return timing_paths

    def calculate_timing_based_hpwl(self, J_extended, p, area_width, timing_criticality):
        batch_size = p.shape[0]
        expected_coords = self.get_expected_placements_from_index(p, area_width)
        
        J = torch.tensor(J_extended, dtype=torch.float32, device=p.device)
        
        # 应用时序关键性权重
        timing_weights = self.calculate_timing_weights(J_extended, timing_criticality)
        weighted_J = J * timing_weights
        
        # 计算加权 HPWL
        coords_i = expected_coords.unsqueeze(2)
        coords_j = expected_coords.unsqueeze(1)
        manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
        
        weighted_dist = manhattan_dist * weighted_J.unsqueeze(0)
        triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
        weighted_dist_triu = weighted_dist[:, triu_mask]
        
        total_wirelength = torch.sum(weighted_dist_triu, dim=1)
        
        return torch.mean(total_wirelength) / batch_size

    def calculate_timing_weights(self, J_extended, timing_criticality):
        """计算时序关键性权重"""
        num_instances = J_extended.shape[0]
        timing_weights = np.ones_like(J_extended)
        
        for path in self.timing_paths:
            start_idx = self.site_to_index.get(path['start_cell'], -1)
            end_idx = self.site_to_index.get(path['end_cell'], -1)
            
            if start_idx != -1 and end_idx != -1:
                # 关键路径上的连接获得更高权重
                criticality = path['criticality'] * timing_criticality
                timing_weights[start_idx, end_idx] += criticality
                timing_weights[end_idx, start_idx] += criticality
        
        return torch.tensor(timing_weights, dtype=torch.float32)

    def calculate_path_delays(self, p, area_width):
        """
        计算时序路径延迟
        """
        batch_size = p.shape[0]
        expected_coords = self.get_expected_placements_from_index(p, area_width)
        
        path_delays = []
        
        for path in self.timing_paths:
            start_idx = self.site_to_index.get(path['start_cell'], -1)
            end_idx = self.site_to_index.get(path['end_cell'], -1)
            
            if start_idx != -1 and end_idx != -1:
                # 计算路径长度
                start_coords = expected_coords[:, start_idx, :]
                end_coords = expected_coords[:, end_idx, :]
                path_length = torch.sum(torch.abs(start_coords - end_coords), dim=1)
                
                # 计算总延迟（单元延迟 + 线延迟）
                cell_delay = (self.cell_delays[path['start_cell']]['max_delay'] + 
                            self.cell_delays[path['end_cell']]['max_delay'])
                wire_delay = path_length * self.net_delays[path['start_cell'] + '_to_' + path['end_cell']]['unit_delay']
                
                total_delay = cell_delay + wire_delay
                path_delays.append({
                    'delay': total_delay,
                    'required_time': path['required_time'],
                    'slack': path['required_time'] - total_delay,
                    'criticality': path['criticality']
                })
        
        return path_delays

    def calculate_timing_violation_loss(self, p, area_width):
        """
        计算时序违例损失
        """
        path_delays = self.calculate_path_delays(p, area_width)
        
        if not path_delays:
            return 0.0
        
        total_violation = 0.0
        critical_path_count = 0
        
        for path_info in path_delays:
            slack = path_info['slack']
            if slack < 0:  # 时序违例
                violation = -slack * path_info['criticality']
                total_violation += violation
                critical_path_count += 1
        
        return total_violation / len(path_delays) if path_delays else 0.0

    def calculate_congestion_aware_hpwl(self, J_extended, p, area_width, congestion_map=None):
        """
        考虑拥塞的 HPWL 计算
        """
        if congestion_map is None:
            congestion_map = self.estimate_congestion(p, area_width)
        
        batch_size = p.shape[0]
        expected_coords = self.get_expected_placements_from_index(p, area_width)
        
        J = torch.tensor(J_extended, dtype=torch.float32, device=p.device)
        
        # 计算基础 HPWL
        coords_i = expected_coords.unsqueeze(2)
        coords_j = expected_coords.unsqueeze(1)
        manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
        
        # 应用拥塞权重
        congestion_weights = self.calculate_congestion_weights(expected_coords, congestion_map)
        weighted_J = J * congestion_weights.unsqueeze(0)
        
        weighted_dist = manhattan_dist * weighted_J.unsqueeze(0)
        triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
        weighted_dist_triu = weighted_dist[:, triu_mask]
        
        total_wirelength = torch.sum(weighted_dist_triu, dim=1)
        
        return torch.mean(total_wirelength) / batch_size

    def estimate_congestion(self, p, area_width):
        """估计布局拥塞"""
        batch_size = p.shape[0]
        
        # 计算每个位置的期望使用率
        site_usage = torch.sum(p, dim=1)  # [batch_size, num_locations]
        
        # 将使用率映射到网格
        congestion_map = torch.zeros(batch_size, area_width, area_width, device=p.device)
        
        for b in range(batch_size):
            for loc_idx in range(site_usage.shape[1]):
                x = loc_idx % area_width
                y = loc_idx // area_width
                if x < area_width and y < area_width:
                    congestion_map[b, y, x] = site_usage[b, loc_idx]
        
        return congestion_map

    def calculate_congestion_weights(self, expected_coords, congestion_map):
        """计算拥塞权重"""
        batch_size, num_instances, _ = expected_coords.shape
        congestion_weights = torch.ones(batch_size, num_instances, num_instances, 
                                    device=expected_coords.device)
        
        for b in range(batch_size):
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    coord_i = expected_coords[b, i].long()
                    coord_j = expected_coords[b, j].long()
                    
                    # 计算路径上的平均拥塞
                    path_congestion = self.calculate_path_congestion(
                        coord_i, coord_j, congestion_map[b]
                    )
                    
                    # 高拥塞区域增加权重
                    congestion_weights[b, i, j] = 1.0 + path_congestion
                    congestion_weights[b, j, i] = 1.0 + path_congestion
        
        return congestion_weights

    def calculate_path_congestion(self, start, end, congestion_map):
        """计算两点路径上的平均拥塞"""
        # 简化的曼哈顿路径拥塞计算
        x_path = torch.arange(min(start[0], end[0]), max(start[0], end[0]) + 1)
        y_path = torch.arange(min(start[1], end[1]), max(start[1], end[1]) + 1)
        
        total_congestion = 0.0
        point_count = 0
        
        for x in x_path:
            for y in y_path:
                if x < congestion_map.shape[1] and y < congestion_map.shape[0]:
                    total_congestion += congestion_map[y, x]
                    point_count += 1
        
        return total_congestion / point_count if point_count > 0 else 0.0

    def comprehensive_energy_function(self, J_extended, p, area_width, weights=None):
        """
        综合能量函数，包含所有约束
        """
        if weights is None:
            weights = {
                'hpwl': 1.0,
                'timing': 5.0,
                'congestion': 2.0,
                'site_constraint': 10.0,
                'type_constraint': 5.0
            }
        
        total_energy = 0.0
        
        # 基础 HPWL
        hpwl_loss = self.calculate_matrix_based_hpwl(J_extended, p, area_width)
        total_energy += weights['hpwl'] * hpwl_loss
        
        # 时序约束
        timing_loss = self.calculate_timing_violation_loss(p, area_width)
        total_energy += weights['timing'] * timing_loss
        
        # 拥塞感知 HPWL
        congestion_loss = self.calculate_congestion_aware_hpwl(J_extended, p, area_width)
        total_energy += weights['congestion'] * congestion_loss
        
        # 布局约束
        constraints = self.calculate_parallel_constraints(p)
        total_energy += weights['site_constraint'] * constraints['site_constraint']
        total_energy += weights['type_constraint'] * constraints['type_constraint']
        
        return total_energy, {
            'hpwl_loss': hpwl_loss.item(),
            'timing_loss': timing_loss.item(),
            'congestion_loss': congestion_loss.item(),
            'constraint_loss': constraints['site_constraint'].item()
        }

    def optimize_with_all_constraints(self, J_extended, area_width, num_iterations=1000):
        """考虑所有约束的优化"""
        num_instances = len(self.optimizable_sites)
        num_locations = len(self.available_target_sites)
        
        # 初始化参数
        h = torch.randn(1, num_instances, num_locations, requires_grad=True)
        optimizer = torch.optim.Adam([h], lr=0.005)
        
        best_energy = float('inf')
        best_p = None
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            p = torch.softmax(h, dim=2)
            
            # 计算综合能量
            total_energy, losses = self.comprehensive_energy_function(J_extended, p, area_width)
            
            # 反向传播
            total_energy.backward()
            optimizer.step()
            
            # 记录最佳解
            if total_energy.item() < best_energy:
                best_energy = total_energy.item()
                best_p = p.detach().clone()
            
            if iteration % 100 == 0:
                print(f"Iter {iteration}: "
                    f"Total={total_energy.item():.3f}, "
                    f"HPWL={losses['hpwl_loss']:.3f}, "
                    f"Timing={losses['timing_loss']:.3f}, "
                    f"Congestion={losses['congestion_loss']:.3f}")
        
        return best_p, best_energy

    def analyze_timing_closure(self, p, area_width):
        """分析时序收敛情况"""
        path_delays = self.calculate_path_delays(p, area_width)
        
        timing_info = {
            'total_paths': len(path_delays),
            'violating_paths': 0,
            'worst_slack': float('inf'),
            'total_slack': 0.0
        }
        
        for path in path_delays:
            if path['slack'] < 0:
                timing_info['violating_paths'] += 1
            timing_info['worst_slack'] = min(timing_info['worst_slack'], path['slack'])
            timing_info['total_slack'] += path['slack']
        
        timing_info['avg_slack'] = timing_info['total_slack'] / len(path_delays) if path_delays else 0
        
        return timing_info