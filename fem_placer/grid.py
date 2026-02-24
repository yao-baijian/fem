import torch
import heapq
import bisect
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from .logger import INFO, WARNING, ERROR

@dataclass
class Grid:
    name: str
    start_x: int = 0
    start_y: int = 0
    area_length: int = 0
    area_width: int = 0
    device: str = 'cpu'
    
    _grid: List[List[List[int]]] = field(default_factory=list)
    _instance_positions: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    _empty_positions: List[Tuple[int, int]] = field(default_factory=list) 
    
    @property
    def end_x(self) -> int:
        return self.start_x + self.area_length
    
    @property
    def end_y(self) -> int:
        return self.start_y + self.area_width
    
    @property
    def center_x(self) -> int:
        return (self.start_x + self.end_x) // 2
    
    @property
    def center_y(self) -> int:
        return (self.start_y + self.end_y) // 2
    
    @property
    def width(self) -> int:
        return self.area_length
    
    @property
    def height(self) -> int:
        return self.area_width
    
    @property
    def area(self) -> int:
        return self.area_length * self.area_width

    def __post_init__(self):
        self._grid = [[[] for _ in range(self.area_width)] for _ in range(self.area_length)]
        for x in range(self.start_x, self.end_x + 1):
            for y in range(self.start_y, self.end_y + 1):
                if self.is_position_empty(x, y):
                    self._empty_positions.append((x, y))
        self._empty_positions.sort()

    def _position_to_key(self, x: int, y: int) -> int:
        return x * 10000 + y  # 假设y不会超过10000

    def find_empty_positions_nearby(self, start_x: int, start_y: int, needed_count: int) -> List[Tuple[int, int, int]]:
        """
        二分查找最近的空位置，自动延展直到找到足够数量或遍历完所有空位置
        """
        if not self._empty_positions:
            return []
        
        result = []
        
        # 二分查找插入位置
        import bisect
        pos = bisect.bisect_left(self._empty_positions, (start_x, start_y))
        
        # 左右指针向两边扩展，直到找到足够数量或遍历完
        left = pos - 1
        right = pos
        
        while len(result) < needed_count and (left >= 0 or right < len(self._empty_positions)):
            left_dist = float('inf')
            right_dist = float('inf')
            
            if left >= 0:
                left_x, left_y = self._empty_positions[left]
                left_dist = abs(left_x - start_x) + abs(left_y - start_y)
            
            if right < len(self._empty_positions):
                right_x, right_y = self._empty_positions[right]
                right_dist = abs(right_x - start_x) + abs(right_y - start_y)
            
            # 选择距离更近的
            if left_dist <= right_dist:
                x, y = self._empty_positions[left]
                if (x, y) != (start_x, start_y):
                    result.append((x, y, left_dist))
                left -= 1
            else:
                x, y = self._empty_positions[right]
                if (x, y) != (start_x, start_y):
                    result.append((x, y, right_dist))
                right += 1
        
        return result
        
    def update_empty_on_place(self, x: int, y: int):
        pos = (x, y)
        if pos in self._empty_positions:
            idx = bisect.bisect_left(self._empty_positions, pos)
            if idx < len(self._empty_positions) and self._empty_positions[idx] == pos:
                del self._empty_positions[idx]
    
    def update_empty_on_remove(self, x: int, y: int):
        if self.is_position_empty(x, y):
            pos = (x, y)
            if pos not in self._empty_positions:
                bisect.insort(self._empty_positions, pos)

    def to_grid_coords(self, real_x: int, real_y: int) -> Tuple[int, int]:
        grid_x = real_x - self.start_x
        grid_y = real_y - self.start_y
        return grid_x, grid_y
    
    def to_real_coords(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        real_x = grid_x + self.start_x
        real_y = grid_y + self.start_y
        return real_x, real_y
    
    def to_real_coords_tensor(self, coord_tensor: torch.Tensor) -> torch.Tensor:
        return coord_tensor + torch.tensor([self.start_x, self.start_y], device=coord_tensor.device)
    
    def is_within_bounds(self, x: int, y: int) -> bool:
        return (self.start_x <= x < self.end_x and 
                self.start_y <= y < self.end_y)
    
    def place_instance(self, instance_id: int, x: int, y: int, force=False) -> bool:
        self.remove_instance(instance_id)
        grid_x, grid_y = self.to_grid_coords(x, y)
        
        if not (0 <= grid_x < self.area_length and 0 <= grid_y < self.area_width):
            return False
        
        self._grid[grid_x][grid_y].append(instance_id)
        self._instance_positions[instance_id] = (x, y)
        self.update_empty_on_place(x, y)
        return True
    
    def move_instance(self, instance_id: int, new_x: int, new_y: int, 
                     swap_allowed=False) -> Tuple[bool, Optional[int], Tuple[int, int]]:
        if instance_id not in self._instance_positions:
            ERROR(f'{instance_id} is not in instance position list')
            return False, None, (None, None)
        
        old_x, old_y = self._instance_positions[instance_id]
        
        if not self.is_within_bounds(new_x, new_y):
            return False, None, (old_x, old_y)
        
        occupants = self.get_position_occupants(new_x, new_y)
        
        if not occupants:
            self.remove_instance(instance_id)
            self.place_instance(instance_id, new_x, new_y)
            return True, None, (old_x, old_y)
        
        elif instance_id not in occupants and swap_allowed:
            other_instance_id = occupants[0]
            
            self.remove_instance(instance_id)
            self.remove_instance(other_instance_id)
            
            self.place_instance(instance_id, new_x, new_y)
            self.place_instance(other_instance_id, old_x, old_y)
            
            return True, other_instance_id, (old_x, old_y)
        
        return False, None, (old_x, old_y)
    
    def remove_instance(self, instance_id: int):
        if instance_id in self._instance_positions:
            x, y = self._instance_positions[instance_id]
            grid_x, grid_y = self.to_grid_coords(x, y)

            if instance_id in self._grid[grid_x][grid_y]:
                self._grid[grid_x][grid_y].remove(instance_id)
                del self._instance_positions[instance_id]
                self.update_empty_on_remove(x, y)
            
    def find_nearest_empty(self, x: int, y: int, max_radius=10, k=5) -> List[Tuple[int, int, int]]:
        heap = []
        visited = set()
        grid_x, grid_y = self.to_grid_coords(x, y)
        queue = [(grid_x, grid_y, 0)]
        
        while queue and len(heap) < k:
            cx, cy, dist = queue.pop(0)
            
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            real_x, real_y = self.to_real_coords(cx, cy)
            if self.is_within_bounds(real_x, real_y) and not self._grid[cx][cy]:
                heapq.heappush(heap, (dist, real_x, real_y))
            
            if dist >= max_radius:
                continue
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    queue.append((nx, ny, dist + 1))
        
        return [(cx, cy, dist) for (dist, cx, cy) in heap]
    
    def find_empty_positions_in_radius(self, x: int, y: int, radius: int) -> List[Tuple[int, int, int]]:
        empty_positions = []
        grid_x, grid_y = self.to_grid_coords(x, y)
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = grid_x + dx, grid_y + dy
                
                if (0 <= nx < self.area_length and 
                    0 <= ny < self.area_width and 
                    not self._grid[nx][ny]): 
                    
                    real_x, real_y = self.to_real_coords(nx, ny)
                    distance = abs(dx) + abs(dy)
                    empty_positions.append((real_x, real_y, distance))
        
        empty_positions.sort(key=lambda pos: pos[2])
        return empty_positions
    
    def get_instance_position(self, instance_id: int) -> Optional[Tuple[int, int]]:
        return self._instance_positions.get(instance_id)
    
    def get_position_occupant(self, x: int, y: int):
        if self.is_within_bounds(x, y):
            grid_x, grid_y = self.to_grid_coords(x, y)
            if self._grid[grid_x][grid_y]:
                return self._grid[grid_x][grid_y][0] 
        return -1
    
    def get_position_occupants(self, x: int, y: int) -> List[int]:
        if not self.is_within_bounds(x, y):
            return []
        
        grid_x, grid_y = self.to_grid_coords(x, y)
        return self._grid[grid_x][grid_y].copy()
    
    def is_position_empty(self, x: int, y: int) -> bool:
        if not self.is_within_bounds(x, y):
            return False
        
        grid_x, grid_y = self.to_grid_coords(x, y)
        return len(self._grid[grid_x][grid_y]) == 0
    
    def get_all_placed_instances(self) -> List[int]:
        return list(self._instance_positions.keys())
    
    def clear_all(self):
        self._grid = [[[] for _ in range(self.area_width)] 
                     for _ in range(self.area_length)]
        self._instance_positions.clear()
        # Reset and repopulate _empty_positions
        self._empty_positions.clear()
        for x in range(self.start_x, self.end_x):
            for y in range(self.start_y, self.end_y):
                self._empty_positions.append((x, y))
        self._empty_positions.sort()
    
    def to_coords_tensor(self, num_instances: int, dtype=torch.float32) -> torch.Tensor:
        coords = torch.zeros((num_instances, 2), dtype=dtype, device=self.device)
        
        for instance_id, (x, y) in self._instance_positions.items():
            if instance_id < num_instances:
                coords[instance_id] = torch.tensor([float(x), float(y)], 
                                                  dtype=dtype, device=self.device)
        return coords
    
    def from_coords_tensor(self, coords: torch.Tensor, instance_ids: Optional[torch.Tensor] = None, 
                          clear_existing=True):
        if clear_existing:
            self.clear_all()
        
        if instance_ids is None:
            instance_ids = torch.arange(coords.shape[0], device=self.device)
        
        for idx, instance_id in enumerate(instance_ids):
            x = int(round(coords[idx, 0].item()))
            y = int(round(coords[idx, 1].item()))
            
            if self.is_within_bounds(x, y):
                self.place_instance(instance_id.item(), x, y)
            else:
                WARNING(f"Instance {instance_id.item()} with coords ({x}, {y}) is out of bounds and cannot be placed.")
    
    @property
    def instance_positions(self) -> Dict[int, Tuple[int, int]]:
        return self._instance_positions.copy()
    
    @property
    def grid_data(self) -> List[List[List[int]]]:
        return [row[:] for row in self._grid]
    
    def __str__(self) -> str:
        return (f"Grid(name='{self.name}', start=({self.start_x}, {self.start_y}), "
                f"size={self.area_length}x{self.area_width}, "
                f"instances={len(self._instance_positions)})")
    
    def __repr__(self) -> str:
        return self.__str__()