import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def drawPreset():
    plt.rcParams['font.size'] = 17
    path = '/usr/share/fonts/opentype/linux-libertine/LinLibertine_RI.otf'  
    prop = fm.FontProperties(fname=path)
    plt.rcParams['font.family'] = prop.get_name()

def visualize_two_point_constraint():
    """
    Visualize the constraint function with two points,
    showing both spacing term and alignment term
    """
    # Apply custom font settings
    drawPreset()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Parameters
    d_min = 2.0  # minimum spacing
    lambda_align = 0.5  # alignment weight
    
    # Define two point coordinates
    point1 = np.array([1.2, 0.8])
    point2 = np.array([3.5, 2.3])
    
    # Calculate constraint terms
    # 1. Spacing term
    l1_distance = np.sum(np.abs(point1 - point2))
    spacing_violation = max(0, d_min - l1_distance)
    spacing_penalty = spacing_violation ** 2
    
    # 2. Alignment term
    def alignment_penalty(point):
        return lambda_align * np.sum(np.sin(np.pi * point) ** 2)
    
    align_penalty1 = alignment_penalty(point1)
    align_penalty2 = alignment_penalty(point2)
    total_align_penalty = align_penalty1 + align_penalty2
    
    # Total constraint
    total_constraint = spacing_penalty + total_align_penalty
    
    # Figure 1: Spacing term visualization
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot two points
    ax1.scatter([point1[0], point2[0]], [point1[1], point2[1]], 
               s=300, c=['red', 'blue'], alpha=0.8, edgecolors='black', linewidth=2)
    ax1.text(point1[0], point1[1] - 0.2, 'Point 1', ha='center', va='top', 
            fontsize=14, fontweight='bold')
    ax1.text(point2[0], point2[1] - 0.2, 'Point 2', ha='center', va='top', 
            fontsize=14, fontweight='bold')
    
    # Draw L1 distance (Manhattan distance) path
    # Horizontal line
    ax1.plot([point1[0], point2[0]], [point1[1], point1[1]], 
            'black', linestyle='-', linewidth=2, alpha=0.5)
    # Vertical line
    ax1.plot([point2[0], point2[0]], [point1[1], point2[1]], 
            'black', linestyle='-', linewidth=2, alpha=0.5)
    
    # Annotate L1 distance components
    horizontal_dist = abs(point1[0] - point2[0])
    vertical_dist = abs(point1[1] - point2[1])
    
    # Horizontal distance annotation
    mid_x = (point1[0] + point2[0]) / 2
    ax1.text(mid_x, point1[1] + 0.1, f'Δx = {horizontal_dist:.2f}', 
            ha='center', va='bottom', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Vertical distance annotation
    ax1.text(point2[0] + 0.1, (point1[1] + point2[1]) / 2, f'Δy = {vertical_dist:.2f}', 
            ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Calculate and display L1 total distance
    l1_text = f'L1 Distance = |Δx| + |Δy| \n= {l1_distance:.2f}'
    ax1.text(0.5, 2.8, l1_text, ha='left', va='top', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Display spacing constraint information
    spacing_info = f'Minimum spacing d_min = {d_min:.1f}\n'
    if l1_distance < d_min:
        spacing_info += f'Violation: {d_min:.1f} - {l1_distance:.2f} = {spacing_violation:.2f}\n'
        spacing_info += f'Penalty: ({spacing_violation:.2f})² = {spacing_penalty:.4f}'
        color = 'red'
    else:
        spacing_info += f'Satisfied: {l1_distance:.2f} ≥ {d_min:.1f}\n'
        spacing_info += f'Penalty: 0'
        color = 'green'
    
    ax1.text(4.5, 4.2, spacing_info, ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))
    
    ax1.set_xlabel('X coordinate', fontsize=14)
    ax1.set_ylabel('Y coordinate', fontsize=14)
    ax1.set_title('Spacing Term: max(0, $d_{min}$ - ||c(σ₁) - c(σ₂)||₁)²', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Figure 2: Alignment term visualization
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Draw discrete grid points (integer coordinates)
    grid_points = []
    for x in range(0, 6):  # integers from 0 to 5
        for y in range(0, 5):  # integers from 0 to 4
            grid_points.append((x, y))
    
    grid_x = [p[0] for p in grid_points]
    grid_y = [p[1] for p in grid_points]
    ax2.scatter(grid_x, grid_y, s=120, c='gray', alpha=0.4, marker='s', 
               label='Ideal grid points', edgecolors='black', linewidth=0.5)
    
    # Draw two points
    ax2.scatter([point1[0], point2[0]], [point1[1], point2[1]], 
               s=300, c=['red', 'blue'], alpha=0.8, edgecolors='black', linewidth=2,
               label='Actual points')
    ax2.text(point1[0], point1[1] - 0.2, 'Point 1', ha='center', va='top', 
            fontsize=14, fontweight='bold')
    ax2.text(point2[0], point2[1] - 0.2, 'Point 2', ha='center', va='top', 
            fontsize=14, fontweight='bold')
    
    # Annotate coordinates for each point
    ax2.text(point1[0] + 0.3, point1[1] + 0.1, 
            f'({point1[0]:.2f}, {point1[1]:.2f})', 
            ha='left', va='bottom', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
    
    ax2.text(point2[0] + 0.3, point2[1] + 0.1, 
            f'({point2[0]:.2f}, {point2[1]:.2f})', 
            ha='left', va='bottom', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
    
    # Calculate distance to nearest integer point
    def distance_to_nearest_grid(point):
        nearest_x = round(point[0])
        nearest_y = round(point[1])
        return (nearest_x, nearest_y), np.abs(point[0] - nearest_x) + np.abs(point[1] - nearest_y)
    
    # Draw connections to nearest integer points
    colors = ['red', 'blue']
    for i, point in enumerate([point1, point2]):
        nearest_grid, dist = distance_to_nearest_grid(point)
        
        # Draw connection line
        ax2.plot([point[0], nearest_grid[0]], [point[1], nearest_grid[1]], 
                colors[i], linestyle=':', linewidth=2, alpha=0.7)
        
        # Annotate distance
        mid_x = (point[0] + nearest_grid[0]) / 2
        mid_y = (point[1] + nearest_grid[1]) / 2
        ax2.text(mid_x, mid_y, f'{dist:.2f}', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.3))
        
        # Calculate alignment penalty
        sin_term1 = np.sin(np.pi * point[0]) ** 2
        sin_term2 = np.sin(np.pi * point[1]) ** 2
        point_penalty = lambda_align * (sin_term1 + sin_term2)
        align_text = f'sin²(π×{point[0]:.2f}) = {sin_term1:.3f}\n'
        align_text += f'sin²(π×{point[1]:.2f}) = {sin_term2:.3f}\n'
        align_text += f'Penalty = {point_penalty:.4f}'
        
        ax2.text(0.5 + i*2, 1.5, align_text, ha='left', va='bottom', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=colors[i], alpha=0.2))
    
    # Display total alignment term information
    total_align_info = f'λ_align = {lambda_align:.1f}\n'
    total_align_info += f'Total alignment penalty:\n'
    total_align_info += f'Point 1: {align_penalty1:.4f}\n'
    total_align_info += f'Point 2: {align_penalty2:.4f}\n'
    total_align_info += f'Total: {total_align_penalty:.4f}'
    
    ax2.text(4.5, 3.8, total_align_info, ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="purple", alpha=0.2))
    
    ax2.set_xlabel('X coordinate', fontsize=14)
    ax2.set_ylabel('Y coordinate', fontsize=14)
    ax2.set_title('Alignment Term: λ_align Σ sin²(πc(σ_i))', 
                 fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=12)
    
    # Display total constraint value
    fig.suptitle(f'Total Constraint: {spacing_penalty:.4f} (spacing) + {total_align_penalty:.4f} (alignment) = {total_constraint:.4f}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('two_point_constraint_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("="*70)
    print("CONSTRAINT FUNCTION ANALYSIS")
    print("="*70)
    print(f"Point 1 coordinates: ({point1[0]:.2f}, {point1[1]:.2f})")
    print(f"Point 2 coordinates: ({point2[0]:.2f}, {point2[1]:.2f})")
    print(f"\n1. SPACING TERM ANALYSIS:")
    print(f"   L1 distance = |{point1[0]:.2f}-{point2[0]:.2f}| + |{point1[1]:.2f}-{point2[1]:.2f}| = {l1_distance:.2f}")
    print(f"   Minimum spacing d_min = {d_min:.1f}")
    print(f"   Spacing violation = max(0, {d_min:.1f} - {l1_distance:.2f}) = {spacing_violation:.2f}")
    print(f"   Spacing penalty = ({spacing_violation:.2f})² = {spacing_penalty:.4f}")
    
    print(f"\n2. ALIGNMENT TERM ANALYSIS (λ_align = {lambda_align:.1f}):")
    for i, point in enumerate([point1, point2]):
        print(f"   Point {i+1}:")
        print(f"     sin²(π×{point[0]:.2f}) = {np.sin(np.pi * point[0])**2:.4f}")
        print(f"     sin²(π×{point[1]:.2f}) = {np.sin(np.pi * point[1])**2:.4f}")
        print(f"     Alignment penalty = {lambda_align:.1f} × ({np.sin(np.pi * point[0])**2:.4f} + {np.sin(np.pi * point[1])**2:.4f}) = {alignment_penalty(point):.4f}")
    
    print(f"\n3. TOTAL CONSTRAINT:")
    print(f"   Total constraint = {spacing_penalty:.4f} + {total_align_penalty:.4f} = {total_constraint:.4f}")
    print("="*70)

# Run visualization
if __name__ == "__main__":
    visualize_two_point_constraint()