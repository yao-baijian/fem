import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class PlacementDrawer:

    def __init__(self, bbox, num_subplots=5, debug_mode=False):
        self.bbox = bbox
        self.area_width = bbox['area_length']
        self.area_height = bbox['area_length']
        self.num_subplots = num_subplots
        self.debug_mode = debug_mode  # 新增：debug模式标志
        
        self.site_colors = {
            'empty_internal': {'face': 'white', 'edge': 'black'},
            'placed_internal': {'face': 'lightblue', 'edge': 'darkblue'},
            'empty_boundary': {'face': 'yellow', 'edge': 'black'}, 
            'placed_boundary': {'face': 'lightgreen', 'edge': 'green'},
            'text': 'black'
        }

        self.wire_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        self.placement_history = []

        # 关键修改：只有当需要时才创建多子图窗口
        self.fig = plt.figure(figsize=(8, 6))  # 创建图形但不显示
        self.ax = self.fig.add_subplot(111)    # 手动添加子图
        
        # 多子图窗口延迟创建
        self.figs = None
        self.axes = None
        
        # 初始化debug接口（如果需要）
        if debug_mode:
            self._init_debug_interface()
    
    def _init_debug_interface(self):
        print("Debug mode enabled - interface to be implemented")
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _on_key_press(self, event):
        if event.key == 'd' or event.key == 'D':
            print("Debug command received")
    
    def add_placement(self, site_coords, step):
        placement_data = {
            'site_coords': site_coords,
            'step': step
        }
        self.placement_history.append(placement_data)
    
    def setup_plot(self, ax):
        ax.set_xlim(-1, self.area_width)
        ax.set_ylim(-1, self.area_height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    def is_boundary_site(self, x, y):
        return (x == 0 or x == self.area_width - 1 or 
                y == 0 or y == self.area_height - 1)
    
    def draw_placement_step(self, p, iteration):
        self.ax.clear()
        self.setup_plot(self.ax)
        p_softmax = torch.softmax(p, dim=-1)
        
        batch_size, num_instances, num_locations = p_softmax.shape
        
        for loc_idx in range(num_locations):
            x = loc_idx % self.area_width
            y = loc_idx // self.area_width
            
            if y >= self.area_height:
                continue
            
            site_usage = torch.sum(p_softmax[:, :, loc_idx]).item()
            is_placed = site_usage > 0.1
            
            if self.is_boundary_site(x, y):
                color_config = (self.site_colors['placed_boundary'] if is_placed 
                              else self.site_colors['empty_boundary'])
            else:
                color_config = (self.site_colors['placed_internal'] if is_placed
                              else self.site_colors['empty_internal'])
            
            rect = patches.Rectangle(
                (x - 0.4, y - 0.4), 0.8, 0.8,
                linewidth=2,
                edgecolor=color_config['edge'],
                facecolor=color_config['face'],
                alpha=min(1.0, site_usage * 2)
            )
            self.ax.add_patch(rect)
            
            if site_usage > 0.05:
                self.ax.text(x, y, f'{site_usage:.2f}', 
                           ha='center', va='center', 
                           fontsize=8, color=self.site_colors['text'])
        
        stats_text = (f'Iteration: {iteration}\n'
                     f'Instances: {num_instances}\n')
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.add_legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def add_legend(self):
        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='black', label='Empty Internal'),
            patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Placed Internal'),
            patches.Patch(facecolor='yellow', edgecolor='black', label='Empty Boundary'), 
            patches.Patch(facecolor='lightgreen', edgecolor='green', label='Placed Boundary')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.0, 0.7))
    
    def draw_hard_placement(self, site_coords, iteration, title_suffix=""):
        self.ax.clear()
        self.setup_plot(self.ax)
        
        num_instances, dim = site_coords.shape
        
        for loc_idx in range(self.area_width * self.area_height):
            x = loc_idx % self.area_width
            y = loc_idx // self.area_width
            
            if y >= self.area_height:
                continue
            
            color_config = (self.site_colors['empty_boundary'] if self.is_boundary_site(x, y)
                          else self.site_colors['empty_internal'])
            
            rect = patches.Rectangle(
                (x - 0.4, y - 0.4), 0.8, 0.8,
                linewidth=1, edgecolor=color_config['edge'],
                facecolor=color_config['face'], alpha=0.3
            )
            self.ax.add_patch(rect)
        

        for i in range(num_instances):
            x = site_coords[i][0]
            y = site_coords[i][1]
            
            if y < self.area_height:
                if self.is_boundary_site(x, y):
                    color_config = self.site_colors['placed_boundary']
                else:
                    color_config = self.site_colors['placed_internal']
                
                rect = patches.Rectangle(
                    (x - 0.3, y - 0.3), 0.6, 0.6,
                    linewidth=3, edgecolor=color_config['edge'],
                    facecolor=color_config['face'], alpha=0.8
                )
                self.ax.add_patch(rect)
                
                self.ax.text(x, y, f'I{i}', 
                            ha='center', va='center', fontsize=7,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        title = f'Hard Placement - Iteration {iteration}'
        if title_suffix:
            title += f' - {title_suffix}'
        self.ax.set_title(title)
    
    def save_figure(self, filename):
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    def draw_routing(self, routes, alpha=0.7):
        if not routes:
            return
        
        weights = [route['weight'] for route in routes]
        if weights:
            max_weight = max(weights)
            min_weight = min(weights)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        else:
            max_weight = 1.0
            min_weight = 0.0
            weight_range = 1.0
        
        for route in routes:
            normalized_weight = (route['weight'] - min_weight) / weight_range
            normalized_weight = max(0.2, min(1.0, normalized_weight)) 
            
            weight_alpha = 0.3 + normalized_weight * 0.5  # [0.3, 0.8]
            
            base_linewidth = 1.2
            linewidth = base_linewidth + normalized_weight * 0.8  # [1.2, 2.0]
            
            color_idx = int(route['weight'] * 5) % len(self.wire_colors)
            color = self.wire_colors[color_idx]
            
            for segment in route['segments']:
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']
                
                self.ax.plot([start_x, end_x], [start_y, end_y], 
                        color=color, linewidth=linewidth, alpha=weight_alpha,
                        linestyle='-', marker='', zorder=1) 
                self.ax.scatter([start_x, end_x], [start_y, end_y], 
                            color=color, s=15, alpha=weight_alpha * 0.8, zorder=2)

    def draw_complete_placement(self, site_coords, routes, iteration, title_suffix=""):
        self.draw_hard_placement(site_coords, iteration, title_suffix)
        self.draw_routing(routes)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(10)
        
        total_length = sum(route['total_length'] for route in routes)
        avg_length = total_length / len(routes)
        
        routing_text = f'Routes: {len(routes)}\nTotal Length: {total_length:.1f}\nAvg Length: {avg_length:.1f}'
        self.ax.text(0.02, 0.15, routing_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def draw_base_grid(self, ax):
        for y in range(self.area_height):
            for x in range(self.area_width):
                if self.is_boundary_site(x, y):
                    color_config = self.site_colors['empty_boundary']
                else:
                    color_config = self.site_colors['empty_internal']
                
                rect = patches.Rectangle(
                    (x - 0.4, y - 0.4), 0.8, 0.8,
                    linewidth=1, edgecolor=color_config['edge'],
                    facecolor=color_config['face'], alpha=0.3
                )
                ax.add_patch(rect)

    def draw_multi_step_placement(self):
        """按需创建多子图窗口，避免冲突"""
        if not self.placement_history:
            print("No placement history to show")
            return
        
        # 按需创建多子图窗口（同样使用 plt.figure() 避免自动显示）
        if self.figs is None or self.axes is None:
            self.figs = plt.figure(figsize=(5*self.num_subplots, 5))
            self.axes = []
            for i in range(self.num_subplots):
                ax = self.figs.add_subplot(1, self.num_subplots, i+1)
                self.setup_plot(ax)
                self.axes.append(ax)
        
        for ax in self.axes:
            self.setup_plot(ax)
        
        for idx in range(self.num_subplots):
            ax = self.axes[idx]
            placement_data = self.placement_history[idx]
            site_coords = placement_data['site_coords']
            step = placement_data['step']
            num_instances, _ = site_coords.shape
            
            self.draw_base_grid(ax)
            
            # 检测重叠的实例
            coord_dict = {}
            overlapped_coords = set()
            
            # 第一遍：统计每个位置的实例数量
            for i in range(num_instances):
                coord = site_coords[i]
                x, y = coord[0].item(), coord[1].item()
                
                if 0 <= x <= self.area_width and 0 <= y <= self.area_height:
                    coord_key = (x, y)
                    if coord_key in coord_dict:
                        coord_dict[coord_key].append(i)
                        overlapped_coords.add(coord_key)
                    else:
                        coord_dict[coord_key] = [i]
            
            # 第二遍：绘制实例，重叠的用红色
            for i in range(num_instances):
                coord = site_coords[i]
                x, y = coord[0].item(), coord[1].item()
                
                if 0 <= x <= self.area_width and 0 <= y <= self.area_height:
                    coord_key = (x, y)
                    
                    # 检查这个位置是否有重叠
                    if coord_key in overlapped_coords:
                        # 重叠的实例使用红色
                        rect = patches.Rectangle(
                            (x - 0.3, y - 0.3), 0.6, 0.6,
                            linewidth=2, edgecolor='red',
                            facecolor='red', alpha=0.8
                        )
                    else:
                        # 正常的颜色
                        if self.is_boundary_site(x, y):
                            color_config = self.site_colors['placed_boundary']
                        else:
                            color_config = self.site_colors['placed_internal']
                        
                        rect = patches.Rectangle(
                            (x - 0.3, y - 0.3), 0.6, 0.6,
                            linewidth=2, edgecolor=color_config['edge'],
                            facecolor=color_config['face'], alpha=0.8
                        )
                    
                    ax.add_patch(rect)
                    
                    # 可以在重叠的位置上添加数字显示重叠数量
                    if coord_key in overlapped_coords and coord_dict[coord_key][0] == i:
                        overlap_count = len(coord_dict[coord_key])
                        if overlap_count > 1:
                            ax.text(x, y, str(overlap_count), 
                                ha='center', va='center', 
                                fontsize=8, fontweight='bold', 
                                color='white')
            
            title = f'Step {step}'
            ax.set_title(title, fontsize=10)
        
        self.figs.tight_layout()
        self.figs.canvas.draw()
        plt.show(block=False)  # 非阻塞显示
        plt.pause(5)