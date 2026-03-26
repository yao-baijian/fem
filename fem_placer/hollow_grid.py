import bisect
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from .logger import INFO, WARNING, ERROR
from .grid import Grid

@dataclass
class HollowGrid(Grid):
    """
    A grid representing a hollow region, specifically for bounding box IO modes. 
    In this type of grid, instances can only be placed on the boundary perimeter cells.
    """
    boundary_thickness: int = 1

    def __post_init__(self):
        self._boundary_layers: List[List[Tuple[int, int]]] = []
        self._boundary_lookup: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._empty_order_keys: List[Tuple[int, int]] = []
        super().__post_init__()
        self._build_boundary_layers()
        self.clear_all()

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

    def _build_boundary_layers(self):
        self._boundary_layers = []
        self._boundary_lookup = {}

        if self.area_length <= 0 or self.area_width <= 0:
            return

        max_layers = self.boundary_thickness
        for layer_offset in range(max_layers):
            min_x = self.start_x + layer_offset
            max_x = self.end_x - layer_offset - 1
            min_y = self.start_y + layer_offset
            max_y = self.end_y - layer_offset - 1

            if min_x > max_x or min_y > max_y:
                break

            layer_positions: List[Tuple[int, int]] = []

            # Top edge
            for x in range(min_x, max_x + 1):
                layer_positions.append((x, min_y))

            # Right edge (excluding corners already added)
            for y in range(min_y + 1, max_y):
                layer_positions.append((max_x, y))

            # Bottom edge (if different from top)
            if max_y != min_y:
                for x in range(max_x, min_x - 1, -1):
                    layer_positions.append((x, max_y))

            # Left edge (if different from right)
            if max_x != min_x:
                for y in range(max_y - 1, min_y, -1):
                    layer_positions.append((min_x, y))

            if not layer_positions:
                continue

            # Deduplicate corners (can appear twice when width/height == 1)
            seen: Set[Tuple[int, int]] = set()
            unique_positions: List[Tuple[int, int]] = []
            for pos in layer_positions:
                if pos in seen:
                    continue
                seen.add(pos)
                unique_positions.append(pos)

            layer_id = len(self._boundary_layers)
            self._boundary_layers.append(unique_positions)
            for idx, pos in enumerate(unique_positions):
                self._boundary_lookup[pos] = (layer_id, idx)

    def clear_all(self):
        self._grid = [[[] for _ in range(self.area_width)] 
                     for _ in range(self.area_length)]
        self._instance_positions.clear()
        self._empty_positions.clear()
        self._empty_order_keys = []

        if not self._boundary_layers:
            return

        for layer_id, layer in enumerate(self._boundary_layers):
            for idx, pos in enumerate(layer):
                self._empty_positions.append(pos)
                self._empty_order_keys.append((layer_id, idx))

    def update_empty_on_place(self, x: int, y: int):
        pos = (x, y)
        order_key = self._boundary_lookup.get(pos)
        if order_key is None:
            return

        idx = bisect.bisect_left(self._empty_order_keys, order_key)
        if idx < len(self._empty_positions) and self._empty_positions[idx] == pos:
            del self._empty_positions[idx]
            del self._empty_order_keys[idx]

    def update_empty_on_remove(self, x: int, y: int):
        if not self.is_within_bounds(x, y):
            return

        if not self.is_position_empty(x, y):
            return

        pos = (x, y)
        order_key = self._boundary_lookup.get(pos)
        if order_key is None:
            return

        idx = bisect.bisect_left(self._empty_order_keys, order_key)
        if idx < len(self._empty_positions) and self._empty_positions[idx] == pos:
            return

        self._empty_positions.insert(idx, pos)
        self._empty_order_keys.insert(idx, order_key)

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
        
    @thick.setter
    def thick(self, value: int):
        self.boundary_thickness = value
        if hasattr(self, '_boundary_layers'):
            self._build_boundary_layers()
            self.clear_all()

    def is_position_empty(self, x: int, y: int) -> bool:
        if not self.is_within_bounds(x, y):
            return False
            
        grid_x, grid_y = self.to_grid_coords(x, y)
        return len(self._grid[grid_x][grid_y]) == 0

    def find_empty_positions_nearby(self, start_x: int, start_y: int, needed_count: int) -> List[Tuple[int, int, int]]:
        order_info = self._boundary_lookup.get((start_x, start_y))
        if order_info is None or not self._boundary_layers:
            return super().find_empty_positions_nearby(start_x, start_y, needed_count)

        result: List[Tuple[int, int, int]] = []
        layer_id, pos_idx = order_info

        self._collect_layer_neighbors(layer_id, pos_idx, start_x, start_y, needed_count, result)

        if len(result) >= needed_count:
            return result

        for other_layer in range(len(self._boundary_layers)):
            if other_layer == layer_id:
                continue
            self._collect_layer_neighbors(other_layer, None, start_x, start_y, needed_count, result)
            if len(result) >= needed_count:
                break

        return result

    def _collect_layer_neighbors(self,
                                 layer_id: int,
                                 start_index: int,
                                 start_x: int,
                                 start_y: int,
                                 needed_count: int,
                                 result: List[Tuple[int, int, int]]):
        if needed_count <= len(result):
            return

        if layer_id < 0 or layer_id >= len(self._boundary_layers):
            return

        layer = self._boundary_layers[layer_id]
        if not layer:
            return

        visited: Set[Tuple[int, int]] = set()

        if start_index is not None and 0 <= start_index < len(layer):
            step = 1
            while len(result) < needed_count and step < len(layer):
                for direction in (-1, 1):
                    idx = (start_index + direction * step) % len(layer)
                    candidate = layer[idx]
                    if candidate in visited:
                        continue
                    visited.add(candidate)
                    if not self.is_position_empty(*candidate):
                        continue
                    distance = abs(candidate[0] - start_x) + abs(candidate[1] - start_y)
                    result.append((candidate[0], candidate[1], distance))
                    if len(result) >= needed_count:
                        return
                step += 1
            return

        ordered = sorted(
            layer,
            key=lambda pos: abs(pos[0] - start_x) + abs(pos[1] - start_y)
        )

        for candidate in ordered:
            if candidate in visited:
                continue
            visited.add(candidate)
            if not self.is_position_empty(*candidate):
                continue
            distance = abs(candidate[0] - start_x) + abs(candidate[1] - start_y)
            result.append((candidate[0], candidate[1], distance))
            if len(result) >= needed_count:
                return
