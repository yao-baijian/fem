import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class PlacementDrawer:
    def __init__(self, bbox):
        self.bbox = bbox
        self.area_width = bbox['area_length']
        self.area_height = bbox['area_length']
        
        self.colors = {
            'empty_internal': {'face': 'white', 'edge': 'black'},
            'placed_internal': {'face': 'lightblue', 'edge': 'darkblue'},
            'empty_boundary': {'face': 'yellow', 'edge': 'black'}, 
            'placed_boundary': {'face': 'lightgreen', 'edge': 'green'},
            'text': 'black'
        }
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.setup_plot()
    
    def setup_plot(self):
        self.ax.set_xlim(-1, self.area_width)
        self.ax.set_ylim(-1, self.area_height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('FEM Placement Optimization', fontsize=14)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
    
    def is_boundary_site(self, x, y):
        return (x == 0 or x == self.area_width - 1 or 
                y == 0 or y == self.area_height - 1)
    
    def draw_placement_step(self, p, iteration):
        self.ax.clear()
        self.setup_plot()
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
                color_config = (self.colors['placed_boundary'] if is_placed 
                              else self.colors['empty_boundary'])
            else:
                color_config = (self.colors['placed_internal'] if is_placed
                              else self.colors['empty_internal'])
            
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
                           fontsize=8, color=self.colors['text'])
        
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
    
    def draw_hard_placement(self, site_indices, iteration, title_suffix=""):
        self.ax.clear()
        self.setup_plot()
        
        num_instances, dim = site_indices.shape
        
        for loc_idx in range(self.area_width * self.area_height):
            x = loc_idx % self.area_width
            y = loc_idx // self.area_width
            
            if y >= self.area_height:
                continue
            
            color_config = (self.colors['empty_boundary'] if self.is_boundary_site(x, y)
                          else self.colors['empty_internal'])
            
            rect = patches.Rectangle(
                (x - 0.4, y - 0.4), 0.8, 0.8,
                linewidth=1, edgecolor=color_config['edge'],
                facecolor=color_config['face'], alpha=0.3
            )
            self.ax.add_patch(rect)
        

        for i in range(num_instances):
            x = site_indices[i][0]
            y = site_indices[i][1]
            
            if y < self.area_height:
                if self.is_boundary_site(x, y):
                    color_config = self.colors['placed_boundary']
                else:
                    color_config = self.colors['placed_internal']
                
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
        
        plt.tight_layout()
        plt.draw()
        plt.pause(50)
    
    def save_figure(self, filename):
        plt.savefig(filename, dpi=150, bbox_inches='tight')