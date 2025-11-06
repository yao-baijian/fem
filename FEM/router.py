import torch

# TODO need better routing algorithm or vivado routing API

class Router:

    def __init__(self, area_width, area_height):
        self.area_width = area_width
        self.area_height = area_height
        
    def route_connections(self, J_extended, coordinates):
        batch_size, num_instances, _ = coordinates.shape
        J = torch.tensor(J_extended, dtype=torch.float32)
        
        all_routes = []
        
        for b in range(batch_size):
            batch_routes = []
            batch_coords = coordinates[b]  # [num_instances, 2]
            
            rows, cols = torch.nonzero(J, as_tuple=True)
            
            for i, j in zip(rows, cols):
                if i >= num_instances or j >= num_instances:
                    continue
                    
                if i < j:  # 避免重复
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

# 扩展绘图器支持布线显示
class PlacementDrawer:
    def __init__(self, bbox, area_width, area_height=None):
        # ... 原有代码 ...
        self.wire_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
    def draw_routing(self, routes, alpha=0.7):
        """绘制布线"""
        if not routes:
            return
            
        for route in routes:
            color = self.wire_colors[int(route['weight'] * 10) % len(self.wire_colors)]
            linewidth = max(1.0, route['weight'] * 3)
            
            for segment in route['segments']:
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']
                
                # 绘制线段
                self.ax.plot([start_x, end_x], [start_y, end_y], 
                           color=color, linewidth=linewidth, alpha=alpha,
                           linestyle='-', marker='')
                
                # 绘制端点
                self.ax.scatter([start_x, end_x], [start_y, end_y], 
                              color=color, s=30, alpha=alpha)
    
    def draw_complete_placement(self, hard_placements, routes, iteration, title_suffix=""):
        """绘制完整的布局和布线"""
        self.draw_hard_placement(hard_placements, iteration, title_suffix)
        self.draw_routing(routes)
        
        # 添加布线统计
        if routes:
            total_length = sum(route['total_length'] for route in routes)
            avg_length = total_length / len(routes)
            
            routing_text = f'Routes: {len(routes)}\nTotal Length: {total_length:.1f}\nAvg Length: {avg_length:.1f}'
            self.ax.text(0.02, 0.15, routing_text, transform=self.ax.transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))