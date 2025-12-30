import torch
from typing import Dict, Any, List, Tuple
from .grid import Grid
from .logger import INFO, WARNING, ERROR

class Legalizer:
    
    def __init__(self, bbox: Dict[str, Any], placer):
        self.bbox = bbox
        self.area_width = bbox["area_length"]
        self.area_height = bbox["area_length"]
        self.placer = placer
        
        self.grid_manager = Grid(
            width=self.area_width,
            height=self.area_height,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def legalize_placement(self, coordinates, io_coords=None, include_io=False, max_attempts=100):
        INFO(f"Stage 1: solve overlap ")
        self.grid_manager.from_coords_tensor(coordinates, clear_existing=True)
        stage1_result = self._resolve_overlaps(coordinates, io_coords, include_io, max_attempts)
        
        hpwl_before = stage1_result['hpwl_before']
        hpwl_after = stage1_result['hpwl_after']
        legalized_coords = stage1_result['legalized_coords']
        overlap = stage1_result['moved_instances']
        
        INFO(f"Hpwl {hpwl_before:.2f} -> {hpwl_after:.2f}, moved {overlap} instances")
        INFO(f"Stage 2: global optimization")
        
        stage2_result = self._global_hpwl_optimization(legalized_coords, io_coords, include_io)
        
        hpwl_optimized = stage2_result['hpwl_optimized']
        final_coords = stage2_result['optimized_coords']
        
        INFO(f"Optimized hpwl {hpwl_optimized:.2f}, improve {hpwl_after - hpwl_optimized:.2f}")
        
        return final_coords, overlap, stage1_result, stage2_result
    
    def _resolve_overlaps(self, coords, io_coords, include_io, max_attempts):
        hpwl_before, hpwl_before_no_io = self.placer.net_manager.analyze_solver_hpwl(coords, io_coords, include_io)
        
        conflict_groups = self._find_conflict_groups()
        moved_count = 0
        
        for conflict_pos, conflict_instances in conflict_groups.items():
            if len(conflict_instances) <= 1:
                continue
            
            success, num_moved = self._resolve_conflict_group_optimal(
                conflict_pos, conflict_instances, coords, io_coords, include_io
            )
            
            if success:
                moved_count += num_moved
        
        legalized_coords = self.grid_manager.to_coords_tensor(coords.shape[0])
        hpwl_after, hpwl_after_no_io = self.placer.net_manager.analyze_solver_hpwl(legalized_coords, io_coords, include_io)
        
        return {
            'legalized_coords': legalized_coords,
            'hpwl_before': hpwl_before_no_io,
            'hpwl_after': hpwl_after_no_io,
            'moved_instances': moved_count
        }
    
    def _find_conflict_groups(self) -> Dict[Tuple[int, int], List[int]]:
        position_to_instances = {}
        conflict_groups = {}
        
        for instance_id in self.grid_manager.instance_positions:
            pos = self.grid_manager.get_instance_position(instance_id)
            if pos:
                pos_tuple = tuple(pos)
                if pos_tuple in position_to_instances:
                    position_to_instances[pos_tuple].append(instance_id)
                else:
                    position_to_instances[pos_tuple] = [instance_id]
        
        for pos, instances in position_to_instances.items():
            if len(instances) > 1:
                conflict_groups[pos] = instances
        
        return conflict_groups
    
    def _resolve_conflict_group_optimal(self, conflict_pos, conflict_instances, 
                                       coords, io_coords, include_io):
        conflict_pos_x, conflict_pos_y = conflict_pos
        
        empty_positions = []
        
        for radius in range(1, 5): 
            radius_empty = self.grid_manager.find_empty_positions_in_radius(
                conflict_pos_x, conflict_pos_y, radius=radius
            )
            empty_positions.extend([(x, y, dist) for x, y, dist in radius_empty])
            
            if len(empty_positions) >= len(conflict_instances) + 2:
                break
        
        empty_positions.insert(0, (conflict_pos_x, conflict_pos_y, 0))
        
        m = len(conflict_instances)
        n = min(len(empty_positions), m + 5)
        
        cost_matrix = torch.zeros((m, n), device=self.grid_manager.device)
        
        for i, instance_id in enumerate(conflict_instances):
            # 获取当前位置的局部HPWL
            current_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(instance_id, coords, io_coords, include_io)
            
            for j in range(n):
                pos_x, pos_y, _ = empty_positions[j]
                
                # 创建临时坐标，移动实例到新位置
                temp_coords = coords.clone()
                temp_coords[instance_id] = torch.tensor([pos_x, pos_y], 
                                                       device=temp_coords.device)
                
                # 获取新位置的局部HPWL
                new_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
                    instance_id, temp_coords, io_coords, include_io)
                
                hpwl_change = new_hpwl - current_hpwl
                dist = abs(pos_x - conflict_pos_x) + abs(pos_y - conflict_pos_y)
                distance_penalty = dist * 0.1
                
                cost_matrix[i, j] = hpwl_change + distance_penalty
        
        # 使用贪婪分配
        best_assignment = self._greedy_assignment(cost_matrix)
        moved_count = 0
        
        for i, j in enumerate(best_assignment):
            instance_id = conflict_instances[i]
            target_x, target_y, _ = empty_positions[j]
            
            current_pos = self.grid_manager.get_instance_position(instance_id)
            if current_pos and (current_pos[0] != target_x or current_pos[1] != target_y):
                success, swapped_with, _ = self.grid_manager.move_instance(
                    instance_id, target_x, target_y, swap_allowed=True
                )
                if success:
                    moved_count += 1
        
        return True, moved_count
    
    def _greedy_assignment(self, cost_matrix):
        m, n = cost_matrix.shape
        assigned_positions = set()
        assignment = [-1] * m
        for i in range(m):
            min_cost = float('inf')
            min_j = -1
            
            for j in range(n):
                if j not in assigned_positions and cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    min_j = j
            
            if min_j != -1:
                assignment[i] = min_j
                assigned_positions.add(min_j)
        
        return assignment
    
    def _resolve_conflict_group_priority(self, conflict_pos, conflict_instances):
        priorities = []
        for instance_id in conflict_instances:
            priority = self._compute_instance_priority(instance_id)
            priorities.append((priority, instance_id))
        
        # 按优先级降序排序
        priorities.sort(reverse=True)
        
        # 优先级高的实例优先选择位置
        moved_count = 0
        occupied_positions = set([conflict_pos])
        
        for _, instance_id in priorities:
            current_pos = self.grid_manager.get_instance_position(instance_id)
            
            # 如果当前位置已被其他高优先级实例占据，需要移动
            if tuple(current_pos) in occupied_positions:
                # 寻找最近的空位
                empty_positions = self.grid_manager.find_nearest_empty(
                    current_pos[0], current_pos[1], max_radius=5, k=3
                )
                
                for _, new_x, new_y in empty_positions:
                    if (new_x, new_y) not in occupied_positions:
                        success, _, _ = self.grid_manager.move_instance(
                            instance_id, new_x, new_y, swap_allowed=False
                        )
                        if success:
                            occupied_positions.add((new_x, new_y))
                            moved_count += 1
                            break
            else:
                # 保持当前位置
                occupied_positions.add(tuple(current_pos))
        
        return moved_count > 0, moved_count
    
    def _compute_instance_priority(self, instance_id):
        net_tensor = self.placer.net_manager.net_tensor
        if net_tensor is not None and instance_id < net_tensor.shape[1]:
            # 连接度作为优先级
            connectivity = net_tensor[:, instance_id].sum().item()
            return connectivity
        
        # 默认优先级
        return 0
    
    def _global_hpwl_optimization(self, coords, io_coords, include_io):
        current_hpwl, current_hpwl_no_io = self.placer.net_manager.analyze_solver_hpwl(coords, io_coords, include_io)

        self.grid_manager.from_coords_tensor(coords, clear_existing=True)
        critical_instances = self._get_critical_instances(top_k=15)
        
        optimized_coords = coords.clone()
        best_hpwl = current_hpwl

        for iteration in range(3):
            improved = False
            
            for instance_id in critical_instances:
                success, improvement = self._optimize_instance_position(
                    instance_id, optimized_coords, io_coords, include_io
                )
                
                if success and improvement > 0:
                    optimized_coords = self.grid_manager.to_coords_tensor(coords.shape[0])
                    best_hpwl -= improvement
                    improved = True
            
            if not improved:
                break
        
        hpwl_optimized, hpwl_optimized_no_io = self.placer.net_manager.analyze_solver_hpwl(optimized_coords, io_coords, include_io)
        return {
            'optimized_coords': optimized_coords,
            'hpwl_optimized': hpwl_optimized_no_io
        }
    
    def _get_critical_instances(self, top_k: int = 15) -> List[int]:
        net_tensor = self.placer.net_manager.net_tensor
        if net_tensor is not None:
            connectivity = net_tensor.sum(dim=0)
            _, indices = torch.topk(connectivity, min(top_k, len(connectivity)))
            return indices.tolist()
        
        all_instances = list(self.grid_manager.instance_positions.keys())
        return all_instances[:min(top_k, len(all_instances))]
    
    def _optimize_instance_position(self, instance_id: int, current_coords, io_coords, include_io) -> Tuple[bool, float]:
        # 获取当前位置的局部HPWL
        current_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
            instance_id, current_coords, io_coords, include_io)
        
        # 获取当前位置
        current_pos = self.grid_manager.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0
        
        best_pos = current_pos
        best_hpwl = current_hpwl
        
        # 搜索周围的空位
        search_radius = 3
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                new_x = current_pos[0] + dx
                new_y = current_pos[1] + dy
                
                # 检查边界
                if not (0 <= new_x < self.grid_manager.width and 
                        0 <= new_y < self.grid_manager.height):
                    continue
                
                # 检查位置是否可用
                # 1. 使用is_position_empty检查是否空位
                # 2. 如果不空，使用get_position_occupants检查是否是当前实例
                occupants = self.grid_manager.get_position_occupants(new_x, new_y)
                
                # 位置为空，或者只有当前实例
                if not occupants or (len(occupants) == 1 and occupants[0] == instance_id):
                    # 位置可用
                    temp_coords = current_coords.clone()
                    temp_coords[instance_id] = torch.tensor([new_x, new_y], 
                                                        device=temp_coords.device)
                    
                    new_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
                        instance_id, temp_coords, io_coords, include_io)
                    
                    if new_hpwl < best_hpwl:
                        best_hpwl = new_hpwl
                        best_pos = (new_x, new_y)
        
        # 如果有改进，移动实例
        if best_hpwl < current_hpwl and best_pos != tuple(current_pos):
            # 使用move_instance移动实例（swap_allowed=False表示不允许交换）
            success, swapped_with, old_pos = self.grid_manager.move_instance(
                instance_id, best_pos[0], best_pos[1], swap_allowed=False
            )
            
            if success:
                improvement = current_hpwl - best_hpwl
                return True, improvement
        
        return False, 0.0