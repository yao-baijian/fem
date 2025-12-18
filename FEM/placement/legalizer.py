import torch
import heapq
from typing import Dict, Any, Optional, List, Tuple
from .grid import Grid

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
        print("=== Legalizer Stage 1: Solve Overlap ===")
        
        # Stage 1: 初始化网格并解决冲突
        self.grid_manager.from_coords_tensor(coordinates, clear_existing=True)
        stage1_result = self._resolve_overlaps(coordinates, io_coords, include_io, max_attempts)
        
        hpwl_before = stage1_result['hpwl_before']
        hpwl_after = stage1_result['hpwl_after']
        legalized_coords = stage1_result['legalized_coords']
        
        print(f"    INFO: HPWL {hpwl_before:.2f} -> {hpwl_after:.2f}, "
              f"Moved {stage1_result['moved_instances']} instances")
        
        print("=== Legalizer Stage 2: Global Optimization ===")
        
        # Stage 2: 全局优化
        stage2_result = self._global_hpwl_optimization(legalized_coords, io_coords, include_io)
        
        hpwl_optimized = stage2_result['hpwl_optimized']
        final_coords = stage2_result['optimized_coords']
        
        print(f"    INFO: optimized hpwl {hpwl_optimized:.2f}, improve {hpwl_after - hpwl_optimized:.2f}")
        
        return final_coords
    
    def _resolve_overlaps(self, coords, io_coords, include_io, max_attempts):
        hpwl_before = self.placer.estimate_solver_hpwl(coords, io_coords, include_io)
        conflict_instances = self._find_conflict_instances()
        
        moved_count = 0
        for instance_id in conflict_instances:
            if self._resolve_single_conflict(instance_id, max_attempts):
                moved_count += 1

        legalized_coords = self.grid_manager.to_coords_tensor(coords.shape[0])
        hpwl_after = self.placer.estimate_solver_hpwl(legalized_coords, io_coords, include_io)
        
        return {
            'legalized_coords': legalized_coords,
            'hpwl_before': hpwl_before,
            'hpwl_after': hpwl_after,
            'moved_instances': moved_count
        }
    
    def _find_conflict_instances(self) -> List[int]:
        position_counts = {}
        conflict_ids = []
        
        for instance_id in self.grid_manager.instance_positions:
            pos = self.grid_manager.get_instance_position(instance_id)
            if pos:
                if pos in position_counts:
                    position_counts[pos].append(instance_id)
                else:
                    position_counts[pos] = [instance_id]
        
        for pos, instances in position_counts.items():
            if len(instances) > 1:
                conflict_ids.extend(instances)
        
        return list(set(conflict_ids))
    
    def _resolve_single_conflict(self, instance_id: int, max_attempts: int) -> bool:
        """解决单个实例的冲突 - 适配list版本"""
        current_pos = self.grid_manager.get_instance_position(instance_id)
        if not current_pos:
            return False
        
        # 获取当前位置的所有占用者
        occupants = self.grid_manager.get_position_occupants(current_pos[0], current_pos[1])
        if not occupants or len(occupants) <= 1:
            return False  # 没有冲突
        
        # 查找最近的空位
        empty_positions = self.grid_manager.find_nearest_empty(
            current_pos[0], current_pos[1], max_radius=max_attempts, k=5
        )
        
        # 尝试每个空位
        for _, new_x, new_y in empty_positions:
            # 注意：move_instance现在返回三元组 (success, swapped_with, old_pos)
            result = self.grid_manager.move_instance(
                instance_id, new_x, new_y, swap_allowed=False
            )
            if result[0]:  # success是第一个元素
                return True
        
        return False
    
    def _global_hpwl_optimization(self, coords, io_coords, include_io):
        current_hpwl = self.placer.estimate_solver_hpwl(coords, io_coords, include_io)
        
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
        
        hpwl_optimized = self.placer.estimate_solver_hpwl(optimized_coords, io_coords, include_io)
        return {
            'optimized_coords': optimized_coords,
            'hpwl_optimized': hpwl_optimized
        }
    
    def _get_critical_instances(self, top_k: int = 15) -> List[int]:
        if hasattr(self.placer, 'net_to_slice_sites_tensor'):
            net_tensor = self.placer.net_to_slice_sites_tensor
            if net_tensor is not None:
                # 计算每个实例的连接度
                connectivity = net_tensor.sum(dim=0)
                
                # 获取连接度最高的实例
                _, indices = torch.topk(connectivity, min(top_k, len(connectivity)))
                return indices.tolist()
        
        # 备选方案：选择随机实例
        all_instances = list(self.grid_manager.instance_positions.keys())
        return all_instances[:min(top_k, len(all_instances))]
    
    def _optimize_instance_position(self, instance_id: int, current_coords, io_coords, include_io) -> Tuple[bool, float]:
        current_pos = self.grid_manager.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0
        
        # 计算当前位置的HPWL
        current_hpwl = self.placer.estimate_solver_hpwl(current_coords, io_coords, include_io)
        
        # 查找附近的候选位置
        candidate_positions = self.grid_manager.find_empty_positions_in_radius(
            current_pos[0], current_pos[1], radius=3
        )
        
        best_improvement = 0.0
        best_position = None
        
        # 测试每个候选位置
        for new_x, new_y, _ in candidate_positions[:8]:  # 最多测试8个位置
            # 临时保存当前网格状态
            temp_coords_backup = self.grid_manager.to_coords_tensor(current_coords.shape[0])
            
            # 尝试移动实例
            success, swapped_with, _ = self.grid_manager.move_instance(
                instance_id, new_x, new_y, swap_allowed=True
            )
            
            if success:
                # 计算新HPWL
                new_coords = self.grid_manager.to_coords_tensor(current_coords.shape[0])
                new_hpwl = self.placer.estimate_solver_hpwl(new_coords, io_coords, include_io)
                improvement = current_hpwl - new_hpwl
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_position = (new_x, new_y)
                
                # 恢复网格状态
                self.grid_manager.from_coords_tensor(temp_coords_backup, clear_existing=True)
        
        # 应用最佳移动（如果有）
        if best_improvement > 0 and best_position:
            self.grid_manager.move_instance(
                instance_id, best_position[0], best_position[1], swap_allowed=True
            )
            return True, best_improvement
        
        return False, 0.0