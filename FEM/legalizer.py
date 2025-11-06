import torch

class Legalizer:
    
    def __init__(self, bbox):
        self.bbox = bbox
        self.area_width = bbox["area_length"]
        self.area_height = bbox["area_length"]
        self.grid = torch.full((self.area_width, self.area_height), -1, dtype=torch.long)
        
    def legalize_placement(self, coordinates, max_attempts=100):
        batch_conflicts = self._resolve_conflicts_batch(coordinates, max_attempts)
        conflict_count = batch_conflicts['conflict_count']
        moved_instances = batch_conflicts['moved_instances']
        
        if conflict_count > 0:
            print(f"Warning: Legalizer found {conflict_count} conflicts, {moved_instances} instances are re-placed")
        
        return batch_conflicts['legalized_coords']
    
    def _resolve_conflicts_batch(self, coords, max_attempts):
        num_instances = coords.shape[0]
        
        conflict_count = 0
        moved_instances = 0
        legalized_coords = coords.clone()
        
        conflicts = []
        for i in range(num_instances):
            x, y = int(coords[i][0]), int(coords[i][1])
            if 0 <= x <= self.area_width and 0 <= y <= self.area_height:
                if self.grid[x, y] == -1:
                    self.grid[x, y] = i
                else:
                    conflicts.append(i)
                    conflict_count += 1
            else:
                conflicts.append(i)
                conflict_count += 1
        
        for instance_idx in conflicts:
            original_x, original_y = int(coords[instance_idx][0]), int(coords[instance_idx][1])
            new_pos = self._find_nearest_available_position(
                original_x, original_y, max_attempts
            )
            
            if new_pos:
                new_x, new_y = new_pos
                legalized_coords[instance_idx] = torch.tensor([new_x, new_y], dtype=coords.dtype)
                self.grid[new_x, new_y] = instance_idx
                moved_instances += 1
                # print(f"INFO: instance {instance_idx} moved from ({original_x}, {original_y}) to ({new_x}, {new_y})")
            else:
                print(f"ERROR: can not find avaliable position for instance {instance_idx}")
        
        return {
            'legalized_coords': legalized_coords,
            'conflict_count': conflict_count,
            'moved_instances': moved_instances
        }
    
    def _find_nearest_available_position(self, x, y, max_attempts):
        for radius in range(1, max_attempts):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) == radius:
                        new_x, new_y = x + dx, y + dy
                        if (0 <= new_x < self.area_width and 
                            0 <= new_y < self.area_height and 
                            self.grid[new_x, new_y] == -1):
                            return (new_x, new_y)
        return None