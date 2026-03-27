import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .grid import Grid
from .placer import FpgaPlacer
from .logger import INFO, WARNING, ERROR

class Legalizer:

    def __init__(self,
                 placer,
                 device,
                 overlap_solver: str = 'greedy', # greedy, hungarian, 
                 hungarian_distance_weight: float = 0.1,
                 hungarian_max_empty_sites: Optional[int] = None,
                 enable_importance_based_swapping: bool = False,
                 fast_first_improvement: bool = True):
        self.placer: FpgaPlacer = placer
        self.logic_grid: Grid = self.placer.get_grid('logic')
        self.io_grid: Grid = self.placer.get_grid('io')

        self.device = device

        self.logic_instance_ids = set()
        self.io_instance_ids = set()
        self.overlap_solver = overlap_solver
        self.hungarian_distance_weight = hungarian_distance_weight
        self.hungarian_max_empty_sites = hungarian_max_empty_sites
        self.enable_importance_based_swapping = enable_importance_based_swapping
        self.fast_first_improvement = fast_first_improvement

    def legalize_placement(self, coords, ids, io_coords=None, io_ids=None, include_io=False):
        INFO(f"Stage 1: solve overlap")
        self._load_coords_to_grids(self.logic_grid, coords, ids)
        logic_moved = self._resolve_grid_overlaps(self.logic_grid, coords, io_coords, False)
        io_moved = 0
        if include_io:
            self._load_coords_to_grids(self.io_grid, io_coords, io_ids)
            io_moved = self._resolve_grid_overlaps(self.io_grid, coords, io_coords, True)

        legalized_logic = self.logic_grid.to_coords_tensor(coords.shape[0])
        legalized_io = self.io_grid.to_coords_tensor(io_coords.shape[0]) if include_io else None

        hpwl_before= self.placer.net_manager.analyze_solver_hpwl(coords, io_coords, include_io)
        hpwl_legalized = self.placer.net_manager.analyze_solver_hpwl(legalized_logic, legalized_io, include_io)

        if include_io:
            INFO(f"Hpwl {hpwl_before['hpwl']:.2f} -> {hpwl_legalized['hpwl']:.2f}, logic move: {logic_moved}, io move: {io_moved}")
        else:
            INFO(f"Hpwl no IO {hpwl_before['hpwl_no_io']:.2f} -> {hpwl_legalized['hpwl_no_io']:.2f}, moved {logic_moved} instances")

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

        return [optimized_logic, optimized_io], logic_moved+io_moved, hpwl_legalized, hpwl_opt

    def _load_coords_to_grids(self, grid: Grid, coords: torch.Tensor, ids: torch.Tensor):
        grid.clear_all()
        grid.from_coords_tensor(coords, ids)
        INFO(f"Loaded {len(ids)} instance to grid")

    def _resolve_grid_overlaps(self, grid: Grid, logic_coords: torch.Tensor,
                              io_coords: Optional[torch.Tensor], include_io: bool) -> int:
        if self.overlap_solver == 'hungarian':
            return self._resolve_grid_overlaps_hungarian(
                grid, logic_coords, io_coords, include_io
            )

        moved_count = 0

        conflict_groups = self._collect_conflict_groups(grid)
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

    def _collect_conflict_groups(self, grid: Grid) -> Dict[Tuple[int, int], List[int]]:
        conflict_groups: Dict[Tuple[int, int], List[int]] = {}
        for instance_id, poz in grid.instance_positions.items():
            pos_tuple = tuple(poz)
            if pos_tuple in conflict_groups:
                conflict_groups[pos_tuple].append(instance_id)
            else:
                conflict_groups[pos_tuple] = [instance_id]
        return {pos: insts for pos, insts in conflict_groups.items() if len(insts) > 1}

    def _resolve_grid_overlaps_hungarian(self,
                                         grid: Grid,
                                         logic_coords: torch.Tensor,
                                         io_coords: Optional[torch.Tensor],
                                         include_io: bool) -> int:
        conflict_groups = self._collect_conflict_groups(grid)
        if not conflict_groups:
            return 0

        conflict_instances: List[int] = []
        for instances in conflict_groups.values():
            if len(instances) > 1:
                conflict_instances.extend(instances[1:])

        if not conflict_instances:
            return 0

        empty_positions: List[Tuple[int, int]] = list(grid._empty_positions)
        if not empty_positions:
            ERROR(f"Grid '{grid.name}' has no empty place")
            return 0

        if self.hungarian_max_empty_sites is not None and self.hungarian_max_empty_sites > 0 and len(empty_positions) > self.hungarian_max_empty_sites:
            conflict_center_x = sum(pos[0] for pos in conflict_groups.keys()) / max(1, len(conflict_groups))
            conflict_center_y = sum(pos[1] for pos in conflict_groups.keys()) / max(1, len(conflict_groups))
            empty_positions = sorted(
                empty_positions,
                key=lambda pos: abs(pos[0] - conflict_center_x) + abs(pos[1] - conflict_center_y)
            )[:self.hungarian_max_empty_sites]

        if len(empty_positions) < len(conflict_instances):
            WARNING(
                f"Grid '{grid.name}' empty sites ({len(empty_positions)}) are fewer than conflict instances ({len(conflict_instances)}), applying partial matching."
            )

        candidate_xy = [(x, y) for x, y in empty_positions]
        num_instances = len(conflict_instances)
        num_candidates = len(candidate_xy)

        if num_instances == 0 or num_candidates == 0:
            return 0

        cost_matrix = np.zeros((num_instances, num_candidates), dtype=np.float32)

        for i, instance_id in enumerate(conflict_instances):
            current_pos = grid.get_instance_position(instance_id)
            if current_pos is None:
                cost_matrix[i, :] = 1e6
                continue

            current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                instance_id, logic_coords, io_coords, include_io
            )

            hpwl_candidates = self.placer.net_manager.compute_instance_move_hpwl_batch(
                instance_id,
                logic_coords,
                io_coords,
                include_io,
                candidate_xy
            )

            distance_penalty = np.fromiter(
                (
                    (abs(candidate_x - current_pos[0]) + abs(candidate_y - current_pos[1])) * self.hungarian_distance_weight
                    for candidate_x, candidate_y in candidate_xy
                ),
                dtype=np.float32,
                count=num_candidates,
            )

            hpwl_delta = np.asarray(hpwl_candidates, dtype=np.float32) - float(current_hpwl)
            cost_matrix[i, :] = hpwl_delta + distance_penalty

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        moved_count = 0

        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            instance_id = conflict_instances[row_idx]
            target_x, target_y = candidate_xy[col_idx]
            current_pos = grid.get_instance_position(instance_id)
            if current_pos is None or (current_pos[0] == target_x and current_pos[1] == target_y):
                continue

            success, _, _ = grid.move_instance(
                instance_id,
                target_x,
                target_y,
                swap_allowed=False,
            )
            if success:
                moved_count += 1

        remaining_conflicts = self._check_remaining_overlaps(grid)
        if remaining_conflicts > 0:
            WARNING(f'{remaining_conflicts} conflicts are not resolved after Hungarian stage')

        return moved_count

    def _check_remaining_overlaps(self, grid: Grid) -> int:
        position_count = {}
        for _, pos in grid.instance_positions.items():
            pos_tuple = tuple(pos)
            position_count[pos_tuple] = position_count.get(pos_tuple, 0) + 1

        remaining = [(pos, count) for pos, count in position_count.items() if count > 1]

        if remaining:
            INFO(f" remain overlapped position: {remaining}")

        return len(remaining)

    def _resolve_conflict_in_grid(self, grid: Grid, conflict_pos, conflict_instances,
                                 logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                 include_io: bool) -> Tuple[bool, int]:
        conflict_x, conflict_y = conflict_pos
        
        needed_positions = len(conflict_instances) + 1
        empty_positions = grid.find_empty_positions_nearby(conflict_x, conflict_y, needed_positions)

        if len(empty_positions) < needed_positions - 1:
            ERROR(f"Grid '{grid.name}' has no empty place")
            return False, 0
        
        empty_positions.insert(0, (conflict_x, conflict_y, 0))
        m = len(conflict_instances)
        n = min(len(empty_positions), m + 3)
        candidate_positions = empty_positions[:n]
        candidate_xy = [(pos_x, pos_y) for pos_x, pos_y, _ in candidate_positions]
        cost_matrix = torch.zeros((m, n), device=self.device)

        for i, instance_id in enumerate(conflict_instances):
            current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                instance_id, logic_coords, io_coords, include_io
            )

            hpwl_candidates = self.placer.net_manager.compute_instance_move_hpwl_batch(
                instance_id,
                logic_coords,
                io_coords,
                include_io,
                candidate_xy
            )

            for j in range(n):
                _, _, dist = candidate_positions[j]
                new_hpwl = hpwl_candidates[j]
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

        if self.enable_importance_based_swapping:
            # 重要性感知优化（较慢但质量更好）
            critical_instances = self._select_critical_instances_for_grid(
                grid, num_instances, logic_coords, io_coords, include_io
            )
            for instance_id in critical_instances:
                success, improvement = self._optimize_instance_in_grid_importance_aware(
                    grid, instance_id, logic_coords, io_coords, include_io
                )
                if success and improvement > 0:
                    improved = True
        else:
            # 快速贪心优化（速度快）
            for instance_id in list(grid.instance_positions.keys())[:num_instances]:
                success, improvement = self._optimize_instance_in_grid_fast(
                    grid, instance_id, logic_coords, io_coords, include_io
                )
                if success and improvement > 0:
                    improved = True
                    if self.fast_first_improvement:
                        break

        return improved

    def _compute_instance_connectivity(self, instance_id: int) -> float:
        """计算实例的连接度得分"""
        net_tensor = self.placer.net_manager.net_tensor
        if net_tensor is None or instance_id >= net_tensor.shape[1]:
            return 0.0
        return net_tensor[:, instance_id].sum().item()

    def _select_critical_instances_for_grid(self, grid: Grid, num_instances: int,
                                           logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                           include_io: bool) -> List[int]:
        """为指定网格选择关键实例 (top 20% by connectivity)"""
        # 如果网格中没有实例，返回空列表
        if not grid.instance_positions:
            return []

        # 根据连接度选择关键实例
        instances_in_grid = list(grid.instance_positions.keys())
        connectivity_scores = []

        for inst_id in instances_in_grid:
            connectivity = self._compute_instance_connectivity(inst_id)
            connectivity_scores.append((connectivity, inst_id))

        if not connectivity_scores:
            return []

        # 按连接度降序排序
        connectivity_scores.sort(reverse=True)
        
        # 选择top 20%作为关键实例
        top_k = max(1, len(connectivity_scores) // 5)
        top_k = min(top_k, len(connectivity_scores))
        
        return [inst_id for _, inst_id in connectivity_scores[:top_k]]

    def _optimize_instance_in_grid_fast(self, grid: Grid, instance_id: int,
                                       logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                       include_io: bool) -> Tuple[bool, float]:
        """快速贪心优化: 仅考虑邻域空位, 不做交换(批量HPWL评估)"""
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0

        # 获取当前位置的HPWL
        current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
            instance_id, logic_coords, io_coords, include_io
        )

        best_pos = current_pos
        best_hpwl = current_hpwl

        # 根据网格类型设置搜索半径
        search_radius = 2 if grid is self.logic_grid else 1

        # 搜索邻域并收集空位候选
        candidate_xy: List[Tuple[int, int]] = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                new_x, new_y = current_pos[0] + dx, current_pos[1] + dy

                if not grid.is_within_bounds(new_x, new_y):
                    continue

                if grid.is_position_empty(new_x, new_y):
                    candidate_xy.append((new_x, new_y))

        if not candidate_xy:
            return False, 0.0

        hpwl_candidates = self.placer.net_manager.compute_instance_move_hpwl_batch(
            instance_id,
            logic_coords,
            io_coords,
            include_io,
            candidate_xy,
        )

        if not hpwl_candidates:
            return False, 0.0

        hpwl_candidates_np = np.asarray(hpwl_candidates, dtype=np.float32)
        best_idx = int(np.argmin(hpwl_candidates_np))
        best_hpwl = float(hpwl_candidates_np[best_idx])
        best_pos = candidate_xy[best_idx]

        # 如果找到更好的位置
        if best_hpwl < current_hpwl and best_pos != current_pos:
            improvement = current_hpwl - best_hpwl
            success, _, _ = grid.move_instance(
                instance_id, best_pos[0], best_pos[1], swap_allowed=False
            )
            return success, improvement

        return False, 0.0

    def _optimize_instance_in_grid_importance_aware(self, grid: Grid, instance_id: int,
                                  logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                  include_io: bool) -> Tuple[bool, float]:
        """重要性感知优化：支持与非关键实例交换
        
        策略：
        1. 优先查找并移到空位
        2. 其次考虑与低重要性实例交换
        3. 只在总HPWL改进时执行交换
        """
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0

        # 获取当前位置的HPWL
        current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
            instance_id, logic_coords, io_coords, include_io
        )

        best_pos = current_pos
        best_hpwl = current_hpwl
        best_swap_candidate = None
        best_improvement = 0.0

        # 获取该实例的连接度
        instance_connectivity = self._compute_instance_connectivity(instance_id)

        # 根据网格类型设置搜索半径 (Manhattan distance)
        search_radius = 3 if grid is self.logic_grid else 1

        # 收集邻域内的所有位置（空位或其他实例）
        neighbor_positions = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                # 使用Manhattan距离判断
                if abs(dx) + abs(dy) > search_radius:
                    continue

                new_x, new_y = current_pos[0] + dx, current_pos[1] + dy

                if not grid.is_within_bounds(new_x, new_y):
                    continue

                neighbor_positions.append((new_x, new_y))

        # 首先评估所有邻域位置中的空位（优先级高）
        empty_positions = []
        occupied_positions = []
        
        for new_x, new_y in neighbor_positions:
            occupants = grid.get_position_occupants(new_x, new_y)
            if not occupants:
                empty_positions.append((new_x, new_y))
            elif occupants[0] != instance_id:
                occupied_positions.append((new_x, new_y, occupants[0]))

        # 评估空位（优先级最高）
        for new_x, new_y in empty_positions:
            new_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                instance_id,
                logic_coords,
                io_coords,
                include_io,
                candidate_pos=(new_x, new_y)
            )

            improvement = current_hpwl - new_hpwl
            if improvement > best_improvement:
                best_improvement = improvement
                best_hpwl = new_hpwl
                best_pos = (new_x, new_y)
                best_swap_candidate = None

        # 如果没有找到改进的空位，才考虑交换（只与非关键实例）
        if best_swap_candidate is None and best_improvement <= 0:
            for new_x, new_y, swap_instance_id in occupied_positions:
                # 只考虑与低重要性实例的交换
                swap_connectivity = self._compute_instance_connectivity(swap_instance_id)
                
                # 交换的实例不能是关键实例（连接度不能更高）
                if swap_connectivity >= instance_connectivity:
                    continue

                # 计算交换后的HPWL变化
                swap_current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                    swap_instance_id, logic_coords, io_coords, include_io
                )
                
                # 当前实例移到新位置
                instance_new_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                    instance_id,
                    logic_coords,
                    io_coords,
                    include_io,
                    candidate_pos=(new_x, new_y)
                )
                
                # 交换的实例移到当前位置
                swap_new_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                    swap_instance_id,
                    logic_coords,
                    io_coords,
                    include_io,
                    candidate_pos=current_pos
                )
                
                # 总的HPWL变化
                total_hpwl_after_swap = instance_new_hpwl + swap_new_hpwl
                total_hpwl_before_swap = current_hpwl + swap_current_hpwl
                improvement = total_hpwl_before_swap - total_hpwl_after_swap
                
                # 如果交换能改进且改进最好，记录
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_hpwl = instance_new_hpwl
                    best_pos = (new_x, new_y)
                    best_swap_candidate = swap_instance_id

        # 如果找到更好的位置或交换
        if best_improvement > 0 and best_pos != current_pos:
            if best_swap_candidate is not None:
                # 执行交换
                success, _, _ = grid.move_instance(
                    instance_id, best_pos[0], best_pos[1], swap_allowed=True
                )
                if success:
                    return success, best_improvement
            else:
                # 仅移动到空位
                success, _, _ = grid.move_instance(
                    instance_id, best_pos[0], best_pos[1], swap_allowed=False
                )
                if success:
                    return success, best_improvement

        return False, 0.0