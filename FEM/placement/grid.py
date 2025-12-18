import torch
import heapq

class Grid:

    def __init__(self, width: int, height: int, device='cpu'):
        self.width = width
        self.height = height
        self.device = device
        self.grid = [[[] for _ in range(height)] for _ in range(width)]
        self.instance_positions = {}
    
    def place_instance(self, instance_id: int, x: int, y: int, force=False):
        self.remove_instance(instance_id)
        
        # 添加到新位置
        self.grid[x][y].append(instance_id)
        self.instance_positions[instance_id] = (x, y)
        
        return True
    
    def move_instance(self, instance_id: int, new_x: int, new_y: int, swap_allowed=False):
        """移动实例到新位置"""
        if instance_id not in self.instance_positions:
            raise KeyError(f"实例{instance_id}不存在")
        
        old_x, old_y = self.instance_positions[instance_id]
        
        # 检查新位置边界
        if not self._is_within_bounds(new_x, new_y):
            return False, None, (old_x, old_y)
        
        # 检查新位置是否可用
        occupants = self.get_position_occupants(new_x, new_y)
        
        if not occupants:  # 位置为空
            # 直接移动
            self.remove_instance(instance_id)
            self.place_instance(instance_id, new_x, new_y)
            return True, None, (old_x, old_y)
        
        elif instance_id not in occupants and swap_allowed:
            # 尝试与第一个占用者交换
            other_instance_id = occupants[0]
            
            # 交换位置
            self.remove_instance(instance_id)
            self.remove_instance(other_instance_id)
            
            self.place_instance(instance_id, new_x, new_y)
            self.place_instance(other_instance_id, old_x, old_y)
            
            return True, other_instance_id, (old_x, old_y)
        
        else:
            # 位置被占用且不允许交换，或实例已经在那个位置
            return False, None, (old_x, old_y)
    
    def find_nearest_empty(self, x: int, y: int, max_radius=10, k=5):
        
        heap = []
        visited = set()
        queue = [(x, y, 0)]
        
        while queue and len(heap) < k:
            cx, cy, dist = queue.pop(0)
            
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            # 检查是否空位（list版本）
            if self._is_within_bounds(cx, cy) and not self.grid[cx][cy]:
                heapq.heappush(heap, (dist, cx, cy))
            
            if dist >= max_radius:
                continue
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    queue.append((nx, ny, dist + 1))
        
        return [(cx, cy, dist) for (dist, cx, cy) in heap]
    
    def find_empty_positions_in_radius(self, x: int, y: int, radius: int):
        """在半径内查找所有空位置 - list版本"""
        empty_positions = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                
                # 修改：list索引是 grid[x][y]，检查是否为空列表
                if (self._is_within_bounds(nx, ny) and 
                    not self.grid[nx][ny]):  # 空列表表示位置为空
                    distance = abs(dx) + abs(dy)
                    empty_positions.append((nx, ny, distance))
        
        empty_positions.sort(key=lambda pos: pos[2])
        return empty_positions
    
    def get_instance_position(self, instance_id: int):
        return self.instance_positions.get(instance_id)
    
    def get_position_occupant(self, x: int, y: int):
        """获取位置上的第一个实例（如果有多个，返回第一个）"""
        if self._is_within_bounds(x, y) and self.grid[x][y]:
            return self.grid[x][y][0]  # 返回第一个实例
        return -1  # 或者返回None
    
    def get_position_occupants(self, x: int, y: int):
        """获取位置上的所有实例"""
        if self._is_within_bounds(x, y):
            return self.grid[x][y].copy()  # 返回副本
        return []
    
    def is_position_empty(self, x: int, y: int):
        return self._is_within_bounds(x, y) and len(self.grid[x][y]) == 0
    
    def get_all_placed_instances(self):
        return list(self.instance_positions.keys())
    
    def get_all_occupied_positions(self):
        """获取所有被占用的位置 - list版本"""
        occupied = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y]:  # 非空列表表示有实例
                    for instance_id in self.grid[x][y]:
                        occupied.append((x, y, instance_id))
        return occupied
    
    def clear_position(self, x: int, y: int):
        """清空位置 - list版本"""
        if self._is_within_bounds(x, y):
            # 移除这个位置的所有实例
            for instance_id in self.grid[x][y][:]:  # 复制列表遍历
                if instance_id in self.instance_positions:
                    del self.instance_positions[instance_id]
            self.grid[x][y] = []  # 清空列表
    
    def remove_instance(self, instance_id: int):
        """移除实例"""
        if instance_id in self.instance_positions:
            x, y = self.instance_positions[instance_id]
            
            # 从网格中移除（安全地）
            if instance_id in self.grid[x][y]:
                self.grid[x][y].remove(instance_id)
            
            # 从位置映射中移除
            del self.instance_positions[instance_id]

    def _is_within_bounds(self, x: int, y: int):
        return 0 <= x < self.width and 0 <= y < self.height
    
    def to_coords_tensor(self, num_instances: int, dtype=torch.float32):
        """将网格状态转换为坐标张量"""
        coords = torch.zeros((num_instances, 2), dtype=dtype, device=self.device)
        
        for instance_id, (x, y) in self.instance_positions.items():
            if instance_id < num_instances:
                coords[instance_id] = torch.tensor([float(x), float(y)], 
                                                dtype=dtype, device=self.device)
        
        return coords
    
    def from_coords_tensor(self, coords: torch.Tensor, clear_existing=True):
        """从坐标张量初始化网格 - 适配list版本"""
        if clear_existing:
            # 清空现有放置（list版本）
            self.grid = [[[] for _ in range(self.height)] for _ in range(self.width)]
            self.instance_positions.clear()
        
        for i in range(coords.shape[0]):
            x = int(round(coords[i, 0].item()))
            y = int(round(coords[i, 1].item()))
            
            if self._is_within_bounds(x, y):
                # list版本的放置方法
                self.grid[x][y].append(i)
                self.instance_positions[i] = (x, y)