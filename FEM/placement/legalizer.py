import torch
from typing import Dict, Any, List, Tuple, Optional
from .grid import Grid
from .placer import FpgaPlacer
from .logger import INFO, WARNING, ERROR

class Legalizer:
    
    def __init__(self, placer, device):
        self.placer: FpgaPlacer = placer
        self.logic_grid: Grid = self.placer.get_grid('logic')
        self.io_grid: Grid = self.placer.get_grid('io')
        
        self.device = device

        self.logic_instance_ids = set()
        self.io_instance_ids = set()
    
    def legalize_placement(self, coords, ids, io_coords=None, io_ids=None, include_io=False):
        INFO(f"Stage 1: solve overlap")
        self._load_coords_to_grids(self.logic_grid, coords, ids)
        logic_moved = self._resolve_grid_overlaps(self.logic_grid, coords, io_coords, False)
        io_moved = 0
        if include_io:
            self._load_coords_to_grids(self.io_grid, io_coords, io_ids)
            io_moved = self._resolve_grid_overlaps(self.io_grid, coords, io_coords, True)
            
        moved_count = logic_moved + io_moved
        
        legalized_logic = self.logic_grid.to_coords_tensor(coords.shape[0])
        legalized_io = self.io_grid.to_coords_tensor(io_coords.shape[0]) if include_io else None
        
        hpwl_before= self.placer.net_manager.analyze_solver_hpwl(coords, io_coords, include_io)
        hpwl_legalized = self.placer.net_manager.analyze_solver_hpwl(legalized_logic, legalized_io, include_io)
        
        if include_io:
            INFO(f"Hpwl {hpwl_before['hpwl']:.2f} -> {hpwl_legalized['hpwl']:.2f}, moved {moved_count} instances")
        else:
            INFO(f"Hpwl no IO {hpwl_before['hpwl_no_io']:.2f} -> {hpwl_legalized['hpwl_no_io']:.2f}, moved {moved_count} instances")
            
        INFO(f"Stage 2: global optimization")
        
        optimized_logic, optimized_io = self._global_optimization(
            legalized_logic, legalized_io, include_io, iteration=3
        )
        hpwl_opt = self.placer.net_manager.analyze_solver_hpwl(
            optimized_logic, optimized_io, include_io
        )
        
        if include_io:
            INFO(f"Optimized Hpwl {hpwl_opt['hpwl']:.2f}, improve {hpwl_opt['hpwl'] - hpwl_legalized['hpwl']:.2f}")
        else:
            INFO(f"Optimized hpwl no IO {hpwl_opt['hpwl_no_io']:.2f}, improve {hpwl_opt['hpwl_no_io'] - hpwl_legalized['hpwl_no_io']:.2f}")
        
        return [optimized_logic, optimized_io], moved_count, hpwl_legalized, hpwl_opt
    
    def _load_coords_to_grids(self, grid: Grid, coords: torch.Tensor, ids: torch.Tensor):
        grid.clear_all()
        grid.from_coords_tensor(coords, ids)
        INFO(f"Loaded {len(ids)} instance to grid")
    
    def _resolve_grid_overlaps(self, grid: Grid, logic_coords: torch.Tensor, 
                              io_coords: Optional[torch.Tensor], include_io: bool) -> int:
        moved_count = 0
        
        conflict_groups = {}
        for instance_id, poz in grid.instance_positions.items():
            pos_tuple = tuple(poz)
            if pos_tuple in conflict_groups:
                conflict_groups[pos_tuple].append(instance_id)
            else:
                conflict_groups[pos_tuple] = [instance_id]
        
        conflict_groups = {pos: insts for pos, insts in conflict_groups.items() if len(insts) > 1}
        sorted_conflicts = sorted(conflict_groups.items(), key=lambda x: len(x[1]), reverse=True)

        for conflict_pos, conflict_instances in sorted_conflicts:
            if len(conflict_instances) <= 1:
                continue
            
            success, num_moved = self._resolve_conflict_in_grid(
                grid, conflict_pos, conflict_instances, logic_coords, io_coords, include_io
            )
            
            if success:
                moved_count += num_moved
                
        remaining_conflicts = self._check_remaining_overlaps(grid)

        if remaining_conflicts > 0:
            WARNING(f'{remaining_conflicts} conflicts are not resolved')
        
        return moved_count
    
    def _check_remaining_overlaps(self, grid: Grid) -> int:
        position_count = {}
        for _, pos in grid.instance_positions.items():
            pos_tuple = tuple(pos)
            position_count[pos_tuple] = position_count.get(pos_tuple, 0) + 1
        
        remaining = [(pos, count) for pos, count in position_count.items() if count > 1]
        
        if remaining:
            WARNING(f" remain overlapped opsition: {remaining}")
        
        return len(remaining)
    
    def _resolve_conflict_in_grid(self, grid: Grid, conflict_pos, conflict_instances, 
                                 logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor], 
                                 include_io: bool) -> Tuple[bool, int]:
        conflict_x, conflict_y = conflict_pos
        
        # 搜索空位置
        empty_positions = []
        for radius in range(1, 5):
            radius_empty = grid.find_empty_positions_in_radius(conflict_x, conflict_y, radius)
            empty_positions.extend([(x, y, dist) for x, y, dist in radius_empty])
            
            if len(empty_positions) >= len(conflict_instances) + 2:
                break
        
        empty_positions.insert(0, (conflict_x, conflict_y, 0))
        
        m = len(conflict_instances)
        n = min(len(empty_positions), m + 5)

        cost_matrix = torch.zeros((m, n), device=self.device)
        
        for i, instance_id in enumerate(conflict_instances):
            current_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
                instance_id, logic_coords, io_coords, include_io
            )
            
            for j in range(n):
                pos_x, pos_y, dist = empty_positions[j]
                
                if grid is self.logic_grid:
                    temp_logic = logic_coords.clone()
                    if instance_id < temp_logic.shape[0]:
                        temp_logic[instance_id] = torch.tensor([pos_x, pos_y], device=self.device)
                    temp_io = io_coords
                else:
                    temp_logic = logic_coords
                    if io_coords is not None and instance_id >= logic_coords.shape[0]:
                        io_idx = instance_id - logic_coords.shape[0]
                        temp_io = io_coords.clone()
                        temp_io[io_idx] = torch.tensor([pos_x, pos_y], device=self.device)
                    else:
                        temp_io = io_coords
                
                new_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
                    instance_id, temp_logic, temp_io, include_io
                )
                
                hpwl_change = new_hpwl - current_hpwl
                distance_penalty = dist * 0.1
                cost_matrix[i, j] = hpwl_change + distance_penalty
        
        assignment = self._greedy_assignment(cost_matrix)
        moved_count = 0
        
        for i, j in enumerate(assignment):
            if j < 0:
                continue
                
            instance_id = conflict_instances[i]
            target_x, target_y, _ = empty_positions[j]
            
            current_pos = grid.get_instance_position(instance_id)
            if current_pos and (current_pos[0] != target_x or current_pos[1] != target_y):
                success, swapped_with, _ = grid.move_instance(
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
    
    def _global_optimization(self, logic_coords: torch.Tensor, 
                                     io_coords: Optional[torch.Tensor], 
                                     include_io: bool,
                                     iteration: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        for iter in range(iteration):
            improved = False

            logic_improved = self._optimize_grid_instances(
                self.logic_grid, logic_coords.shape[0], logic_coords, io_coords, include_io
            )
            if logic_improved:
                improved = True
            
            if include_io:
                io_improved = self._optimize_grid_instances(
                    self.io_grid, io_coords.shape[0], logic_coords, io_coords, include_io
                )
                if io_improved:
                    improved = True
            
            if not improved:
                break
        
        optimized_logic = self.logic_grid.to_coords_tensor(logic_coords.shape[0])
        optimized_io = self.io_grid.to_coords_tensor(io_coords.shape[0]) if include_io else None
        
        return optimized_logic, optimized_io
    
    def _optimize_grid_instances(self, grid: Grid, num_instances: int, 
                                logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor], 
                                include_io: bool) -> bool:
        improved = False

        critical_instances = self._select_critical_instances_for_grid(
            grid, num_instances, logic_coords, io_coords, include_io
        )
        
        for instance_id in critical_instances:
            success, improvement = self._optimize_instance_in_grid(
                grid, instance_id, logic_coords, io_coords, include_io
            )
            if success and improvement > 0:
                improved = True
        
        return improved
    
    def _select_critical_instances_for_grid(self, grid: Grid, num_instances: int,
                                           logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                           include_io: bool) -> List[int]:
        """为指定网格选择关键实例"""
        # 如果网格中没有实例，返回空列表
        if not grid.instance_positions:
            return []
        
        # 根据连接度选择关键实例
        net_tensor = self.placer.net_manager.net_tensor
        if net_tensor is not None:
            # 获取该网格中所有实例的连接度
            instances_in_grid = list(grid.instance_positions.keys())
            connectivity_scores = []
            
            for inst_id in instances_in_grid:
                if inst_id < net_tensor.shape[1]:
                    connectivity = net_tensor[:, inst_id].sum().item()
                    connectivity_scores.append((connectivity, inst_id))
            
            # 按连接度降序排序
            connectivity_scores.sort(reverse=True)
            top_k = min(10, len(connectivity_scores))
            return [inst_id for _, inst_id in connectivity_scores[:top_k]]
        
        # 如果没有网络信息，选择网格中心附近的实例
        center_x, center_y = grid.area_length // 2, grid.area_width // 2
        instances_in_grid = list(grid.instance_positions.keys())[:num_instances]
        
        sorted_instances = sorted(instances_in_grid,
            key=lambda inst_id: abs(grid.instance_positions[inst_id][0] - center_x) + 
                                abs(grid.instance_positions[inst_id][1] - center_y))
        
        return sorted_instances[:min(10, len(sorted_instances))]
    
    def _optimize_instance_in_grid(self, grid: Grid, instance_id: int,
                                  logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                  include_io: bool) -> Tuple[bool, float]:
        """在指定网格中优化单个实例位置"""
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0
        
        # 获取当前位置的HPWL
        current_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
            instance_id, logic_coords, io_coords, include_io
        )
        
        best_pos = current_pos
        best_hpwl = current_hpwl
        
        # 根据网格类型设置搜索半径
        search_radius = 2 if grid is self.logic_grid else 1
        
        # 搜索邻域
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
                
                if not grid.is_within_bounds(new_x, new_y):
                    continue
                
                # 检查位置是否可用
                if grid.is_position_empty(new_x, new_y) or \
                   (grid.get_position_occupants(new_x, new_y) and 
                    grid.get_position_occupants(new_x, new_y)[0] == instance_id):
                    
                    # 创建临时坐标
                    if grid is self.logic_grid:
                        temp_logic = logic_coords.clone()
                        if instance_id < temp_logic.shape[0]:
                            temp_logic[instance_id] = torch.tensor([new_x, new_y], device=self.device)
                        temp_io = io_coords
                    else:
                        temp_logic = logic_coords
                        if io_coords is not None and instance_id >= logic_coords.shape[0]:
                            io_idx = instance_id - logic_coords.shape[0]
                            temp_io = io_coords.clone()
                            temp_io[io_idx] = torch.tensor([new_x, new_y], device=self.device)
                        else:
                            temp_io = io_coords
                    
                    new_hpwl = self.placer.net_manager.get_single_instance_net_hpwl(
                        instance_id, temp_logic, temp_io, include_io
                    )
                    
                    if new_hpwl < best_hpwl:
                        best_hpwl = new_hpwl
                        best_pos = (new_x, new_y)
        
        # 如果找到更好的位置
        if best_hpwl < current_hpwl and best_pos != current_pos:
            improvement = current_hpwl - best_hpwl
            success, _, _ = grid.move_instance(
                instance_id, best_pos[0], best_pos[1], swap_allowed=True
            )
            return success, improvement
        
        return False, 0.0