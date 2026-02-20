"""
Grid class for FPGA placement.
Manages placement state and provides grid operations for legalizer and drawer.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any


class Grid:
    """Grid class for managing placement state on a 2D grid."""

    def __init__(self, name: str, bbox_dict: Dict[str, Any]):
        """
        Initialize grid from bounding box dictionary.

        Args:
            name: Grid identifier (e.g., 'logic', 'io')
            bbox_dict: Dictionary containing grid bounds with keys:
                start_x, end_x, start_y, end_y, area_length, area_width
        """
        self.name = name
        self.start_x = bbox_dict.get('start_x', 0)
        self.end_x = bbox_dict.get('end_x', 0)
        self.start_y = bbox_dict.get('start_y', 0)
        self.end_y = bbox_dict.get('end_y', 0)
        self.area_length = bbox_dict.get('area_length', self.end_x - self.start_x + 1)
        self.area_width = bbox_dict.get('area_width', self.end_y - self.start_y + 1)

        # Track placement state: {(x, y): [instance_ids]}
        self.grid_state: Dict[Tuple[int, int], List[int]] = {}
        # Track instance positions: {instance_id: (x, y)}
        self._instance_positions: Dict[int, Tuple[int, int]] = {}

    def is_within_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return (self.start_x <= x <= self.end_x and
                self.start_y <= y <= self.end_y)

    def clear_all(self) -> None:
        """Clear all placements from grid."""
        self.grid_state = {}
        self._instance_positions = {}

    def is_position_empty(self, x: int, y: int) -> bool:
        """Check if position is empty."""
        return (x, y) not in self.grid_state or len(self.grid_state[(x, y)]) == 0

    def place_instance(self, instance_id: int, x: int, y: int, force: bool = False) -> bool:
        """
        Place an instance at given coordinates.

        Args:
            instance_id: Instance identifier
            x, y: Target coordinates
            force: If True, replace existing instance; if False, append

        Returns:
            True if position was empty, False if there was a conflict
        """
        if not self.is_within_bounds(x, y):
            return False

        # Remove from old position if exists
        if instance_id in self._instance_positions:
            old_pos = self._instance_positions[instance_id]
            if old_pos in self.grid_state and instance_id in self.grid_state[old_pos]:
                self.grid_state[old_pos].remove(instance_id)
                if not self.grid_state[old_pos]:
                    del self.grid_state[old_pos]

        # Place at new position
        if (x, y) not in self.grid_state:
            self.grid_state[(x, y)] = []

        was_empty = len(self.grid_state[(x, y)]) == 0

        if force:
            self.grid_state[(x, y)] = [instance_id]
        else:
            if instance_id not in self.grid_state[(x, y)]:
                self.grid_state[(x, y)].append(instance_id)

        self._instance_positions[instance_id] = (x, y)
        return was_empty

    def get_position_occupants(self, x: int, y: int) -> List[int]:
        """Get list of instances at position."""
        return self.grid_state.get((x, y), []).copy()

    def remove_instance(self, instance_id: int) -> bool:
        """Remove instance from grid."""
        if instance_id not in self._instance_positions:
            return False

        pos = self._instance_positions[instance_id]
        if pos in self.grid_state and instance_id in self.grid_state[pos]:
            self.grid_state[pos].remove(instance_id)
            if not self.grid_state[pos]:
                del self.grid_state[pos]

        del self._instance_positions[instance_id]
        return True

    def get_instance_position(self, instance_id: int) -> Optional[Tuple[int, int]]:
        """
        Get position of an instance.

        Args:
            instance_id: Instance identifier

        Returns:
            (x, y) tuple or None if instance not found
        """
        return self._instance_positions.get(instance_id)

    def move_instance(self, instance_id: int, target_x: int, target_y: int,
                      swap_allowed: bool = True) -> Tuple[bool, Optional[int], Optional[Tuple[int, int]]]:
        """
        Move an instance to a new position.

        Args:
            instance_id: Instance to move
            target_x, target_y: Target coordinates
            swap_allowed: If True and target is occupied by single instance, swap positions

        Returns:
            (success, swapped_instance_id, old_position)
            - success: Whether move was successful
            - swapped_instance_id: ID of instance that was swapped (if any)
            - old_position: Original position of the moved instance
        """
        if not self.is_within_bounds(target_x, target_y):
            return False, None, None

        if instance_id not in self._instance_positions:
            return False, None, None

        old_pos = self._instance_positions[instance_id]
        target_pos = (target_x, target_y)

        # If target is same as current, nothing to do
        if old_pos == target_pos:
            return True, None, old_pos

        # Check target position
        target_occupants = self.get_position_occupants(target_x, target_y)

        if len(target_occupants) == 0:
            # Target is empty, simple move
            self.place_instance(instance_id, target_x, target_y)
            return True, None, old_pos

        elif len(target_occupants) == 1 and swap_allowed:
            # Single occupant, try to swap
            other_id = target_occupants[0]

            # Remove both from their positions
            self.grid_state[old_pos].remove(instance_id)
            if not self.grid_state[old_pos]:
                del self.grid_state[old_pos]
            self.grid_state[target_pos].remove(other_id)
            if not self.grid_state[target_pos]:
                del self.grid_state[target_pos]

            # Place at swapped positions
            if old_pos not in self.grid_state:
                self.grid_state[old_pos] = []
            self.grid_state[old_pos].append(other_id)
            self._instance_positions[other_id] = old_pos

            if target_pos not in self.grid_state:
                self.grid_state[target_pos] = []
            self.grid_state[target_pos].append(instance_id)
            self._instance_positions[instance_id] = target_pos

            return True, other_id, old_pos
        else:
            # Target has multiple occupants or swap not allowed
            return False, None, None

    def find_empty_positions_in_radius(self, x: int, y: int, radius: int) -> List[Tuple[int, int, int]]:
        """
        Find empty positions within a given radius.

        Args:
            x, y: Center coordinates
            radius: Search radius (Manhattan distance)

        Returns:
            List of (x, y, distance) tuples for empty positions, sorted by distance
        """
        candidates = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                distance = abs(dx) + abs(dy)
                if distance == 0 or distance > radius:
                    continue

                new_x, new_y = x + dx, y + dy
                if self.is_within_bounds(new_x, new_y) and self.is_position_empty(new_x, new_y):
                    candidates.append((new_x, new_y, distance))

        return sorted(candidates, key=lambda c: c[2])

    def find_nearest_empty(self, x: int, y: int, max_radius: int = 10, k: int = 5) -> List[Tuple[int, int, int]]:
        """
        Find k nearest empty positions.

        Args:
            x, y: Center coordinates
            max_radius: Maximum search radius
            k: Number of positions to find

        Returns:
            List of (x, y, distance) tuples for empty positions
        """
        candidates = []
        for r in range(1, max_radius + 1):
            radius_empty = self.find_empty_positions_in_radius(x, y, r)
            for pos in radius_empty:
                if pos not in candidates:
                    candidates.append(pos)
            if len(candidates) >= k:
                break
        return sorted(candidates, key=lambda c: c[2])[:k]

    def from_coords_tensor(self, coords: torch.Tensor,
                          instance_ids: Optional[torch.Tensor] = None,
                          clear_existing: bool = True) -> None:
        """
        Load placements from coordinate tensor.

        Args:
            coords: Coordinate tensor [num_instances, 2]
            instance_ids: Optional tensor of instance IDs [num_instances]
            clear_existing: Whether to clear existing placements first
        """
        from .logger import WARNING

        if clear_existing:
            self.clear_all()

        num_instances = coords.shape[0]
        placed_count = 0
        out_of_bounds_count = 0

        for i in range(num_instances):
            x = int(coords[i][0].item()) if torch.is_tensor(coords[i][0]) else int(coords[i][0])
            y = int(coords[i][1].item()) if torch.is_tensor(coords[i][1]) else int(coords[i][1])

            if instance_ids is not None:
                instance_id = int(instance_ids[i].item()) if torch.is_tensor(instance_ids[i]) else int(instance_ids[i])
            else:
                instance_id = i

            success = self.place_instance(instance_id, x, y, force=False)
            if success or (x, y) in self.grid_state:
                placed_count += 1
            else:
                out_of_bounds_count += 1

        if out_of_bounds_count > 0:
            WARNING(f"Grid '{self.name}': {out_of_bounds_count}/{num_instances} instances out of bounds (range: x=[{self.start_x},{self.end_x}], y=[{self.start_y},{self.end_y}])")

    def to_coords_tensor(self, num_instances: int, device: str = 'cpu') -> torch.Tensor:
        """
        Export placements to coordinate tensor.

        Args:
            num_instances: Expected number of instances
            device: Torch device for output tensor

        Returns:
            Coordinate tensor [num_instances, 2]
        """
        coords = torch.zeros((num_instances, 2), dtype=torch.float32, device=device)

        for instance_id, (x, y) in self._instance_positions.items():
            if instance_id < num_instances:
                coords[instance_id, 0] = x
                coords[instance_id, 1] = y

        return coords

    def to_real_coords_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Convert grid coordinates to real FPGA coordinates.

        Args:
            coords: Grid coordinates [batch, 2] or [2]

        Returns:
            Real FPGA coordinates with start_x and start_y offset applied
        """
        offset = torch.tensor([self.start_x, self.start_y],
                             device=coords.device, dtype=coords.dtype)
        return coords + offset

    @property
    def instance_positions(self) -> Dict[int, Tuple[int, int]]:
        """Return dict mapping instance_id -> (x, y)."""
        return self._instance_positions.copy()
