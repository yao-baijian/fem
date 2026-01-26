import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
from .grid import Grid
from .logger import INFO, WARNING, ERROR
from ..customized_problem.fpga_placement import get_loss_history, get_placment_history

RECT_WIDTH = 0.6
RECT_HEIGHT = 0.6
INSTANCE_WIDTH = 0.5
INSTANCE_HEIGHT = 0.5
        
class PlacementDrawer:

    def __init__(self, placer, num_subplots=5, debug_mode=False):
        self.placer = placer
        
        self.logic_grid: Grid = placer.get_grid('logic')
        self.io_grid: Grid  = placer.get_grid('io')
        self._calculate_overall_bbox(False)
        self.debug_mode = debug_mode
        
        self.site_colors = {
            'logic_empty': {'face': "#B8B8B8", 'edge': "#555454"},
            'logic_placed': {'face': '#64B5F6', 'edge': '#1976D2'}, 
            'io_empty': {'face': '#FFF176', 'edge': '#FFB300'},
            'io_placed': {'face': '#81C784', 'edge': '#388E3C'},
            'text': "#282727"                                      
        }

        self.wire_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        
        self.figs = None
        self.axes = None
    
        if debug_mode:
            self._init_debug_interface()
        
        plt.rcParams['font.family'] = 'Calibri'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['axes.edgecolor'] = "#C9C4C4"
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['grid.alpha'] = 0.7
    
    def _calculate_overall_bbox(self, include_io):
        all_grids = [self.logic_grid]
        if include_io:
            all_grids.append(self.io_grid)
        
        min_x = min([grid.start_x for grid in all_grids])
        max_x = max([grid.end_x for grid in all_grids])
        min_y = min([grid.start_y for grid in all_grids])
        max_y = max([grid.end_y for grid in all_grids])
        
        self.overall_width = max_x - min_x
        self.overall_height = max_y - min_y
        self.offset_x = -min_x
        self.offset_y = -min_y
    
        self.grid_bounds = {}
        for grid in all_grids:
            self.grid_bounds[grid.name] = {
                'start_x': grid.start_x + self.offset_x,
                'end_x': grid.end_x + self.offset_x,
                'start_y': grid.start_y + self.offset_y,
                'end_y': grid.end_y + self.offset_y
            }
    
    def _normalize_coords(self, x, y):
        return x + self.offset_x, y + self.offset_y
    
    def _get_grid_for_position(self, x, y):
        real_x, real_y = x - self.offset_x, y - self.offset_y

        if self.logic_grid.is_within_bounds(real_x, real_y):
            return 'logic'
        
        if self.io_grid and self.io_grid.is_within_bounds(real_x, real_y):
            return 'io'
        
        return None
    
    def setup_plot(self, ax):
        padding = 2
        ax.set_xlim(-padding, self.overall_width + padding)
        ax.set_ylim(-padding, self.overall_height + padding)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # self._draw_grid_boundaries(ax)
    
    def _draw_grid_boundaries(self, ax):
        for grid_name, bounds in self.grid_bounds.items():
            width = bounds['end_x'] - bounds['start_x'] + 1
            height = bounds['end_y'] - bounds['start_y'] + 1
            
            rect = patches.Rectangle(
                (bounds['start_x'] - 0.5, bounds['start_y'] - 0.5),
                width, height,
                linewidth=3, linestyle='--',
                edgecolor=self.site_colors[f'{grid_name}_empty']['edge'],
                facecolor='none', alpha=0.5
            )
            ax.add_patch(rect)
            
            ax.text(
                bounds['start_x'] + width/2,
                bounds['start_y'] + height/2,
                grid_name.upper(),
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color=self.site_colors[f'{grid_name}_empty']['edge'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
    
    def draw_placement(self, logic_coords, io_coords=None, include_io=False, iteration=0, title_suffix=""):
        self.ax.clear()
        self.setup_plot(self.ax)
        self._draw_all_base_grids(self.ax, include_io)
        self._draw_instances(self.ax, logic_coords, 'logic', False)
        
        if include_io:
            self._draw_instances(self.ax, io_coords, 'io', False)
        
        title = f'Placement - Iteration {iteration}'
        if title_suffix:
            title += f' - {title_suffix}'
        self.ax.set_title(title, fontsize=14)
        
        self._add_legend()
        
    def _draw_single_grid_base(self, ax, grid, grid_type):
        for x in range(grid.start_x, grid.end_x + 1):
            for y in range(grid.start_y, grid.end_y + 1):
                plot_x, plot_y = self._normalize_coords(x, y)
                color_config = self.site_colors[f'{grid_type}_empty']
                
                rect = patches.Rectangle(
                    (plot_x - RECT_WIDTH/2, plot_y - RECT_HEIGHT/2), 
                    RECT_WIDTH, RECT_HEIGHT,
                    linewidth= 0.5,
                    edgecolor=color_config['edge'],
                    facecolor=color_config['face'],
                    alpha=0.3
                )
                ax.add_patch(rect)
    
    def _draw_all_base_grids(self, ax, include_io):
        self._draw_single_grid_base(ax, self.logic_grid, 'logic')
        
        if include_io:
            self._draw_single_grid_base(ax, self.io_grid, 'io')
        pass
            
    def _draw_instances(self, ax, coords, grid_type, label=False):
        num_instances = coords.shape[0]
        grid = getattr(self, f'{grid_type}_grid')
        
        coord_dict = {}
        overlapped_coords = set()
        
        for i in range(num_instances):
            x = int(coords[i][0].item()) if torch.is_tensor(coords[i][0]) else int(coords[i][0])
            y = int(coords[i][1].item()) if torch.is_tensor(coords[i][1]) else int(coords[i][1])
            
            if grid.is_within_bounds(x, y):
                coord_key = (x, y)
                if coord_key in coord_dict:
                    coord_dict[coord_key].append(i)
                    overlapped_coords.add(coord_key)
                else:
                    coord_dict[coord_key] = [i]
            else:
                WARNING(f'instance {i} with coords {x, y} out of bounds')
        
        for i in range(num_instances):
            x = int(coords[i][0].item()) if torch.is_tensor(coords[i][0]) else int(coords[i][0])
            y = int(coords[i][1].item()) if torch.is_tensor(coords[i][1]) else int(coords[i][1])
            
            if grid.is_within_bounds(x, y):
                plot_x, plot_y = self._normalize_coords(x, y)
                coord_key = (x, y)
            
                if coord_key in overlapped_coords:
                    rect = patches.Rectangle(
                        (plot_x - INSTANCE_WIDTH/2, plot_y - INSTANCE_HEIGHT/2), 
                        INSTANCE_WIDTH, INSTANCE_HEIGHT,
                        linewidth=1, edgecolor='red',
                        facecolor='red', alpha=0.8
                    )
                else:
                    color_config = self.site_colors[f'{grid_type}_placed']
                    rect = patches.Rectangle(
                        (plot_x - INSTANCE_WIDTH/2, plot_y - INSTANCE_HEIGHT/2), 
                        INSTANCE_WIDTH, INSTANCE_HEIGHT,
                        linewidth=1, edgecolor=color_config['edge'],
                        facecolor=color_config['face'], alpha=0.8
                    )
                
                ax.add_patch(rect)
                
                if label:
                    label = f'{grid_type[0].upper()}{i}'
                    ax.text(plot_x, plot_y, label,
                        ha='center', va='center', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                if coord_key in overlapped_coords and coord_dict[coord_key][0] == i:
                    overlap_count = len(coord_dict[coord_key])
                    if overlap_count > 1:
                        ax.text(plot_x, plot_y, str(overlap_count),
                               ha='center', va='center',
                               fontsize=8, fontweight='bold',
                               color='white')
    
    def _add_legend(self):
        legend_elements = [
            patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Logic'),
            patches.Patch(facecolor='lightgreen', edgecolor='green', label='IO'),
            patches.Patch(facecolor='red', edgecolor='darkred', label='Overlap')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(0.95, 0.95), fontsize=9)
    
    def draw_place_and_route(self, logic_coords, routes, io_coords=None, 
                             include_io = False, iteration=0, title_suffix=""):
        self.draw_placement(logic_coords, io_coords, include_io, iteration, title_suffix)
        self.draw_routing(routes)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
        plt.pause(20) 
        plt.savefig(f'final_placement.png', dpi=150, bbox_inches='tight')
                
    def draw_routing(self, routes, alpha=0.7):        
        weights = [route['weight'] for route in routes]
        max_weight = max(weights)
        min_weight = min(weights)
        greens_cmap = cm.Greens
        
        for route in routes:
            if max_weight > min_weight:
                normalized_weight = (route['weight'] - min_weight) / (max_weight - min_weight)
            else:
                normalized_weight = 0.5
            color = greens_cmap(0.3 + 0.6 * normalized_weight)
            
            linewidth = 0.8 + normalized_weight * 1.2
            
            for segment in route['segments']:
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']
                
                norm_start_x, norm_start_y = self._normalize_coords(start_x, start_y)
                norm_end_x, norm_end_y = self._normalize_coords(end_x, end_y)
                
                self.ax.plot([norm_start_x, norm_end_x], [norm_start_y, norm_end_y], 
                        color=color, 
                        linewidth=linewidth, 
                        alpha=0.5 * alpha,
                        linestyle='-', 
                        zorder=0)
                
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

    def draw_multi_step_placement(self, save_path=None):
        step_labels = ['250', '500', '750', '1000']
        placement_history = get_placment_history()
        # Create figure with subplots
        num_plots = len(step_labels)
        self.figs = plt.figure(figsize=(5 * num_plots, 5))
        self.axes = []
        
        for plot_idx, step_label in enumerate(step_labels):
            ax = self.figs.add_subplot(1, num_plots, plot_idx + 1)
            self.axes.append(ax)
            site_coords = placement_history[plot_idx][0]
            # Setup plot
            
            real_logic_coords = self.logic_grid.to_real_coords_tensor(site_coords)
            self.setup_plot(ax)
            self._draw_all_base_grids(ax, include_io=False)
            self._draw_instances(ax, real_logic_coords, 'logic', label=False)
            ax.set_title(f'Step {step_label}', fontsize=12, fontweight='bold')
        
        self.figs.tight_layout()
        self.figs.canvas.draw()
        plt.show(block=False)
        plt.pause(5)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    def plot_fpga_placement_loss(self, save_path=None):
        loss_data = get_loss_history()
        steps = list(range(len(loss_data['hpwl_losses'])))
        colors = ["#53D5F9", "#86FAD8", "#DF6FFA"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(steps, loss_data['hpwl_losses'], 
                color=colors[0], linewidth=2, label='HPWL Loss')
        ax.plot(steps, loss_data['constrain_losses'], 
                color=colors[1], linewidth=2, label='Constraint Loss')
        ax.plot(steps, loss_data['total_losses'], 
                color=colors[2], linewidth=2, label='Total Loss')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(False)
        
        plt.tight_layout()
        plt.pause(5)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')