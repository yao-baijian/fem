import bisect
import heapq
import torch
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from .logger import INFO, WARNING, ERROR
from .grid import Grid

@dataclass
class HollowGrid(Grid):
    """
    A grid representing a hollow region, specifically for bounding box IO modes. 
    In this type of grid, instances can only be placed on the boundary perimeter cells.
    """
    boundary_thickness: int = 1

    def is_within_bounds(self, x: int, y: int) -> bool:
        """
        Check if the real coordinate (x,y) is within the hollow boundary region.
        """
        if not super().is_within_bounds(x, y):
            return False
            
        # Is on boundary if distance to any of the 4 edges is less than boundary_thickness
        dist_left = x - self.start_x
        dist_right = self.end_x - 1 - x
        dist_bottom = y - self.start_y
        dist_top = self.end_y - 1 - y
        
        min_dist_x = min(dist_left, dist_right)
        min_dist_y = min(dist_bottom, dist_top)
        
        return min_dist_x < self.boundary_thickness or min_dist_y < self.boundary_thickness

    def clear_all(self):
        self._grid = [[[] for _ in range(self.area_width)] 
                     for _ in range(self.area_length)]
        self._instance_positions.clear()
        self._empty_positions.clear()
        
        for x in range(self.start_x, self.end_x):
            for y in range(self.start_y, self.end_y):
                if self.is_within_bounds(x, y):
                    if super().is_position_empty(x, y):
                        self._empty_positions.append((x, y))
                        
        self._empty_positions.sort()

    def __post_init__(self):
        super().__post_init__()
        self.clear_all()  # Recalculate empty positions to exclude hollow center
        
    @property
    def area(self) -> int:
        """
        Return the exact number of sites available in the perimeter.
        """
        if self.area_length == 0 or self.area_width == 0:
            return 0
        if self.boundary_thickness * 2 >= self.area_length or self.boundary_thickness * 2 >= self.area_width:
            return self.area_length * self.area_width
            
        inner_length = self.area_length - 2 * self.boundary_thickness
        inner_width = self.area_width - 2 * self.boundary_thickness
        return (self.area_length * self.area_width) - (inner_length * inner_width)
    
    @property
    def thick(self) -> int:
        return self.boundary_thickness

    def is_position_empty(self, x: int, y: int) -> bool:
        if not self.is_within_bounds(x, y):
            return False
            
        grid_x, grid_y = self.to_grid_coords(x, y)
        return len(self._grid[grid_x][grid_y]) == 0
