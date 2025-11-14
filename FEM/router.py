import torch

# TODO need better routing algorithm or vivado routing API

class Router:

    def __init__(self, bbox):
        self.area_width = bbox["area_length"]
        self.area_height = bbox["area_length"]
        
    def route_connections(self, site_to_site_connect_matrix, coordinates):
        batch_size, num_instances, _ = coordinates.shape
        J = torch.tensor(site_to_site_connect_matrix, dtype=torch.float32)
        
        all_routes = []
        
        for b in range(batch_size):
            batch_routes = []
            batch_coords = coordinates[b]  # [num_instances, 2]
            
            rows, cols = torch.nonzero(J, as_tuple=True)
            
            for i, j in zip(rows, cols):
                if i >= num_instances or j >= num_instances:
                    continue
                    
                if i < j:
                    route = self._manhattan_route(
                        batch_coords[i], batch_coords[j], connection_weight=J[i, j].item()
                    )
                    if route:
                        batch_routes.append(route)
            
            all_routes.append(batch_routes)
        
        return all_routes
    
    def _manhattan_route(self, start, end, connection_weight=1.0):
        start_x, start_y = start[0].item(), start[1].item()
        end_x, end_y = end[0].item(), end[1].item()
        
        route_segments = []
        
        if start_x != end_x:
            route_segments.append({
                'type': 'horizontal',
                'start': (start_x, start_y),
                'end': (end_x, start_y),
                'weight': connection_weight
            })
        
        # Y方向布线
        if start_y != end_y:
            route_segments.append({
                'type': 'vertical', 
                'start': (end_x, start_y),
                'end': (end_x, end_y),
                'weight': connection_weight
            })
        
        return {
            'start_instance': (start_x, start_y),
            'end_instance': (end_x, end_y),
            'segments': route_segments,
            'total_length': abs(start_x - end_x) + abs(start_y - end_y),
            'weight': connection_weight
        }

    