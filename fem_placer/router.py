import torch
from typing import Tuple, List, Dict
from .grid import Grid

class Router:
    def __init__(self, placer):
        self.placer = placer

    def route_connections(self, connect_matrix, all_coords):
        J = torch.tensor(connect_matrix, dtype=torch.float32, 
                device=all_coords.device)
        rows, cols = torch.nonzero(J, as_tuple=True)

        routes = []

        for i, j in zip(rows, cols):
            if i < j:
                route = self._manhattan_route(
                    all_coords[i],
                    all_coords[j],
                    J[i, j].item()
                )
                if route:
                    routes.append(route)

        return routes

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