import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 设置随机种子以便复现
torch.manual_seed(42)

# 模拟数据
batch_size = 2
num_instances = 5  # 5个instance
num_sites = 10     # 10个可能的site位置
num_nets = 3       # 3个网络
gamma = 0.03

# 1. 概率分布p [batch_size, num_instances, num_sites]
p = torch.randn(batch_size, num_instances, num_sites)
p = torch.softmax(p, dim=-1)  # 确保是概率分布
print(f"1. Probability distribution p shape: {p.shape}")
print(f"p[0]:\n{p[0]}\n")

# 2. net_tensor [num_nets, num_instances] - 布尔矩阵，表示哪些instance属于哪些net
net_tensor = torch.tensor([
    [1, 0, 1, 0, 1],  # Net 0: 连接instance 0, 2, 4
    [0, 1, 1, 0, 1],  # Net 1: 连接instance 1, 2, 4
    [1, 1, 0, 1, 0],  # Net 2: 连接instance 0, 1, 3
], dtype=torch.bool)
print(f"2. Net tensor shape: {net_tensor.shape}")
print(f"net_tensor:\n{net_tensor}\n")

# 3. site坐标矩阵 [num_sites, 2]
# 假设site在100x100的网格上
site_coords_matrix = torch.rand(num_sites, 2) * 100
print(f"3. Site coordinates matrix shape: {site_coords_matrix.shape}")
print(f"site_coords_matrix:\n{site_coords_matrix}\n")

def debug_get_hpwl_loss_from_net_tensor_wa_vectorized(p, net_tensor, site_coords_matrix, gamma=0.03):
    batch_size, num_instances, num_sites = p.shape
    num_nets = net_tensor.shape[0]
    
    print("="*60)
    print("STEP 1: 计算期望坐标")
    print("="*60)
    
    # 1. 计算期望坐标
    expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
    expected_x = expected_coords[..., 0]  # [batch_size, num_instances]
    expected_y = expected_coords[..., 1]  # [batch_size, num_instances]
    
    print(f"expected_coords shape: {expected_coords.shape}")
    print(f"Batch 0 - Expected coordinates for each instance:")
    for i in range(num_instances):
        print(f"  Instance {i}: ({expected_x[0, i]:.2f}, {expected_y[0, i]:.2f})")
    print()
    
    print("="*60)
    print("STEP 2: 坐标归一化")
    print("="*60)
    
    # 2. 坐标归一化
    x_min = expected_x.min(dim=1, keepdim=True)[0]
    x_max = expected_x.max(dim=1, keepdim=True)[0]
    y_min = expected_y.min(dim=1, keepdim=True)[0]
    y_max = expected_y.max(dim=1, keepdim=True)[0]
    
    print(f"Batch 0 - X range: [{x_min[0, 0]:.2f}, {x_max[0, 0]:.2f}]")
    print(f"Batch 0 - Y range: [{y_min[0, 0]:.2f}, {y_max[0, 0]:.2f}]")
    
    x_norm = (expected_x - x_min) / (x_max - x_min + 1e-10)
    y_norm = (expected_y - y_min) / (y_max - y_min + 1e-10)
    
    print(f"\nBatch 0 - Normalized X coordinates:")
    for i in range(num_instances):
        print(f"  Instance {i}: {x_norm[0, i]:.4f}")
    print()
    
    net_tensor_expanded = net_tensor.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nets, num_instances]
    print(f"net_tensor_expanded shape: {net_tensor_expanded.shape}")
    
    total_hpwl = torch.zeros(batch_size, device=p.device)
    
    print("="*60)
    print("STEP 3: 对每个坐标方向(X和Y)计算wirelength")
    print("="*60)
    
    # 对X方向进行详细计算展示
    print("\n>>> 详细计算X方向:")
    
    # 扩展坐标张量
    coord_norm = x_norm  # 先看X方向
    coord_expanded = coord_norm.unsqueeze(1).expand(-1, num_nets, -1)  # [batch_size, num_nets, num_instances]
    
    print(f"coord_expanded shape: {coord_expanded.shape}")
    print(f"Batch 0, Net 0 - Normalized X coordinates for connected instances:")
    print(f"  Connected instances: {torch.where(net_tensor[0])[0].tolist()}")
    print(f"  Their normalized X: {[coord_expanded[0, 0, i].item() for i in torch.where(net_tensor[0])[0]]}")
    
    coord_masked = coord_expanded * net_tensor_expanded
    print(f"\ncoord_masked shape: {coord_masked.shape}")
    print(f"Batch 0, Net 0 - After masking (non-connected set to 0):")
    print(f"  {coord_masked[0, 0]}")
    
    # 计算max值
    print("\n>>> 计算max值:")
    coord_for_max = coord_masked.clone()
    zero_mask_max = (coord_masked == 0) & (net_tensor_expanded == 0)
    coord_for_max[zero_mask_max] = -float('inf')
    max_vals = coord_for_max.max(dim=2, keepdim=True)[0]
    print(f"max_vals shape: {max_vals.shape}")
    print(f"Batch 0, Net 0 - Max value: {max_vals[0, 0, 0]:.4f}")
    
    # 计算min值
    print("\n>>> 计算min值:")
    coord_for_min = coord_masked.clone()
    zero_mask_min = (coord_masked == 0) & (net_tensor_expanded == 0)
    coord_for_min[zero_mask_min] = float('inf')
    min_vals = coord_for_min.min(dim=2, keepdim=True)[0]
    print(f"min_vals shape: {min_vals.shape}")
    print(f"Batch 0, Net 0 - Min value: {min_vals[0, 0, 0]:.4f}")
    
    # 计算正方向的加权平均
    print("\n>>> 计算正方向加权平均 (weighted_avg_pos):")
    weight_pos_safe = torch.exp((coord_masked - max_vals) / gamma)
    print(f"weight_pos_safe shape: {weight_pos_safe.shape}")
    print(f"Batch 0, Net 0 - Weights: {weight_pos_safe[0, 0]}")
    
    numerator_pos = torch.sum(coord_masked * weight_pos_safe * net_tensor_expanded, dim=2)
    denominator_pos = torch.sum(weight_pos_safe * net_tensor_expanded, dim=2)
    
    print(f"Batch 0, Net 0 - Numerator_pos: {numerator_pos[0, 0]:.6f}")
    print(f"Batch 0, Net 0 - Denominator_pos: {denominator_pos[0, 0]:.6f}")
    
    denominator_pos = torch.where(denominator_pos == 0, torch.tensor(1e-10, device=p.device), denominator_pos)
    weighted_avg_pos = numerator_pos / denominator_pos
    print(f"Batch 0, Net 0 - Weighted_avg_pos: {weighted_avg_pos[0, 0]:.6f}")
    
    # 计算负方向的加权平均
    print("\n>>> 计算负方向加权平均 (weighted_avg_neg):")
    weight_neg_safe = torch.exp(-(coord_masked - min_vals) / gamma)
    print(f"Batch 0, Net 0 - Weights (neg): {weight_neg_safe[0, 0]}")
    
    numerator_neg = torch.sum(coord_masked * weight_neg_safe * net_tensor_expanded, dim=2)
    denominator_neg = torch.sum(weight_neg_safe * net_tensor_expanded, dim=2)
    
    print(f"Batch 0, Net 0 - Numerator_neg: {numerator_neg[0, 0]:.6f}")
    print(f"Batch 0, Net 0 - Denominator_neg: {denominator_neg[0, 0]:.6f}")
    
    denominator_neg = torch.where(denominator_neg == 0, torch.tensor(1e-10, device=p.device), denominator_neg)
    weighted_avg_neg = numerator_neg / denominator_neg
    print(f"Batch 0, Net 0 - Weighted_avg_neg: {weighted_avg_neg[0, 0]:.6f}")
    
    # 计算当前坐标方向的wirelength
    coord_wirelength = weighted_avg_pos - weighted_avg_neg
    print(f"\nBatch 0, Net 0 - X-direction wirelength: {coord_wirelength[0, 0]:.6f}")
    
    # 对Y方向进行同样计算（简化展示）
    print("\n>>> 计算Y方向 (简化):")
    for batch_idx in range(batch_size):
        for net_idx in range(num_nets):
            connected_instances = torch.where(net_tensor[net_idx])[0]
            if len(connected_instances) > 0:
                y_coords = [y_norm[batch_idx, i].item() for i in connected_instances]
                print(f"Batch {batch_idx}, Net {net_idx}: Y coords {y_coords}")
    
    # 最终计算总和
    print("\n" + "="*60)
    print("STEP 4: 计算总wirelength")
    print("="*60)
    
    for coord_norm, coord_name in [(x_norm, "X"), (y_norm, "Y")]:
        coord_expanded = coord_norm.unsqueeze(1).expand(-1, num_nets, -1)
        coord_masked = coord_expanded * net_tensor_expanded
        
        coord_for_max = coord_masked.clone()
        zero_mask_max = (coord_masked == 0) & (net_tensor_expanded == 0)
        coord_for_max[zero_mask_max] = -float('inf')
        max_vals = coord_for_max.max(dim=2, keepdim=True)[0]
        
        coord_for_min = coord_masked.clone()
        zero_mask_min = (coord_masked == 0) & (net_tensor_expanded == 0)
        coord_for_min[zero_mask_min] = float('inf')
        min_vals = coord_for_min.min(dim=2, keepdim=True)[0]
        
        weight_pos_safe = torch.exp((coord_masked - max_vals) / gamma)
        numerator_pos = torch.sum(coord_masked * weight_pos_safe * net_tensor_expanded, dim=2)
        denominator_pos = torch.sum(weight_pos_safe * net_tensor_expanded, dim=2)
        
        weight_neg_safe = torch.exp(-(coord_masked - min_vals) / gamma)
        numerator_neg = torch.sum(coord_masked * weight_neg_safe * net_tensor_expanded, dim=2)
        denominator_neg = torch.sum(weight_neg_safe * net_tensor_expanded, dim=2)
        
        denominator_pos = torch.where(denominator_pos == 0, torch.tensor(1e-10, device=p.device), denominator_pos)
        denominator_neg = torch.where(denominator_neg == 0, torch.tensor(1e-10, device=p.device), denominator_neg)
        
        weighted_avg_pos = numerator_pos / denominator_pos
        weighted_avg_neg = numerator_neg / denominator_neg
        
        coord_wirelength = weighted_avg_pos - weighted_avg_neg
        
        print(f"\n{coord_name}-direction wirelength per net (Batch 0):")
        for net_idx in range(num_nets):
            print(f"  Net {net_idx}: {coord_wirelength[0, net_idx]:.6f}")
        
        total_hpwl += torch.sum(coord_wirelength, dim=1)
    
    print(f"\nFinal total HPWL per batch:")
    for batch_idx in range(batch_size):
        print(f"  Batch {batch_idx}: {total_hpwl[batch_idx]:.6f}")
    
    print("\n" + "="*60)
    print("WA近似 vs 精确HPWL对比")
    print("="*60)

    # 为每个batch和net计算精确HPWL
    for batch_idx in range(batch_size):
        print(f"\nBatch {batch_idx}:")
        
        # 获取该batch所有instance的期望坐标
        batch_coords = expected_coords[batch_idx]  # [num_instances, 2]
        
        batch_exact_total = 0.0
        batch_wa_total = 0.0
        
        for net_idx in range(num_nets):
            # 该net连接的instance索引
            connected = torch.where(net_tensor[net_idx])[0]
            
            if len(connected) < 2:
                continue
                
            # 精确HPWL
            net_coords = batch_coords[connected]
            x_exact = net_coords[:, 0].max() - net_coords[:, 0].min()
            y_exact = net_coords[:, 1].max() - net_coords[:, 1].min()
            exact_hpwl = x_exact + y_exact
            
            # WA近似HPWL
            # 从之前计算的coord_wirelength中获取（需要记录之前的结果）
            # 这里简化，直接重新计算
            
            batch_exact_total += exact_hpwl.item()
        
        # 该batch的WA总HPWL
        wa_total = total_hpwl[batch_idx].item()
        
        print(f"  精确总HPWL: {batch_exact_total:.4f}")
        print(f"  WA近似总HPWL: {wa_total:.4f}")
        print(f"  绝对误差: {abs(batch_exact_total - wa_total):.4f}")
        print(f"  相对误差: {abs(batch_exact_total - wa_total)/batch_exact_total*100:.2f}%")

    return total_hpwl

def drawPreset():
    plt.rcParams['font.size'] = 17
    path = '/usr/share/fonts/opentype/linux-libertine/LinLibertine_RI.otf'  
    prop = fm.FontProperties(fname=path)
    plt.rcParams['font.family'] = prop.get_name()

def visualize_hpwl_models():
    """
    Visualize two HPWL models: Manhattan-distance-based and Weighted-average-based
    """
    # Apply custom font settings
    drawPreset()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Model A: Manhattan-Distance-Based HPWL
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Model A: Manhattan-Distance-Based HPWL', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Create a net with 4 instances
    net_points = np.array([
        [2, 3],
        [5, 7],
        [7, 2],
        [8, 5]
    ])
    
    # Plot instances
    ax1.scatter(net_points[:, 0], net_points[:, 1], 
               s=300, c='blue', alpha=0.7, edgecolors='black', linewidth=2,
               label='Instances')
    
    # Label instances
    for i, (x, y) in enumerate(net_points):
        ax1.text(x, y + 0.3, f'I{i+1}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Draw Manhattan distance paths (steiner tree approximation)
    # Connect all points to approximate center
    center = np.mean(net_points, axis=0)
    
    # Draw connections from center to each point
    for point in net_points:
        # Horizontal then vertical (L-shaped path)
        ax1.plot([center[0], point[0]], [center[1], center[1]], 
                'red', linestyle='-', linewidth=1.5, alpha=0.5)
        ax1.plot([point[0], point[0]], [center[1], point[1]], 
                'red', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Highlight the center
    ax1.scatter([center[0]], [center[1]], s=150, c='red', 
               alpha=0.8, marker='x', linewidth=3, label='Center point')
    
    # Calculate HPWL (Manhattan distance to center)
    hpwl_a = 0
    for point in net_points:
        l1_dist = np.abs(point[0] - center[0]) + np.abs(point[1] - center[1])
        hpwl_a += l1_dist
    
    # Show HPWL calculation
    formula_text = r'$HPWL^{(A)} = \sum w_{ij} \|c_i - c_j\|_1$'
    ax1.text(1, 9.5, formula_text, ha='left', va='top', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    calc_text = 'Example calculation:\n'
    for i, point in enumerate(net_points):
        dx = abs(point[0] - center[0])
        dy = abs(point[1] - center[1])
        calc_text += f'I{i+1}: |{point[0]:.1f}-{center[0]:.1f}| + |{point[1]:.1f}-{center[1]:.1f}| = {dx+dy:.1f}\n'
    calc_text += f'Total HPWL = {hpwl_a:.1f}'
    
    ax1.text(1, 7.5, calc_text, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.3))
    
    ax1.set_xlabel('X coordinate', fontsize=14)
    ax1.set_ylabel('Y coordinate', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    
    # 2. Model B: Weighted-Average-Based HPWL
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Model B: Weighted-Average-Based HPWL', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Create another net with 4 instances (different distribution)
    net_points_b = np.array([
        [2, 2],
        [3, 7],
        [6, 3],
        [9, 6]
    ])
    
    # Plot instances
    ax2.scatter(net_points_b[:, 0], net_points_b[:, 1], 
               s=300, c='green', alpha=0.7, edgecolors='black', linewidth=2,
               label='Instances')
    
    # Label instances
    for i, (x, y) in enumerate(net_points_b):
        ax2.text(x, y + 0.3, f'I{i+1}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Calculate softmax-weighted averages
    gamma = 1.0  # temperature parameter
    
    # For x-coordinate
    x_coords = net_points_b[:, 0]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    # WA+ (softmax weighted towards max)
    exp_pos = np.exp((x_coords - x_max) / gamma)
    wa_plus_x = np.sum(x_coords * exp_pos) / np.sum(exp_pos)
    
    # WA- (softmax weighted towards min)
    exp_neg = np.exp(-(x_coords - x_min) / gamma)
    wa_minus_x = np.sum(x_coords * exp_neg) / np.sum(exp_neg)
    
    # For y-coordinate
    y_coords = net_points_b[:, 1]
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    exp_pos_y = np.exp((y_coords - y_max) / gamma)
    wa_plus_y = np.sum(y_coords * exp_pos_y) / np.sum(exp_pos_y)
    
    exp_neg_y = np.exp(-(y_coords - y_min) / gamma)
    wa_minus_y = np.sum(y_coords * exp_neg_y) / np.sum(exp_neg_y)
    
    # Calculate HPWL
    hpwl_b = (wa_plus_x - wa_minus_x) + (wa_plus_y - wa_minus_y)
    
    # Draw bounding box based on weighted averages
    ax2.plot([wa_minus_x, wa_plus_x, wa_plus_x, wa_minus_x, wa_minus_x],
             [wa_minus_y, wa_minus_y, wa_plus_y, wa_plus_y, wa_minus_y],
             'purple', linewidth=2.5, linestyle='--', alpha=0.8,
             label='Weighted bounding box')
    
    # Mark weighted average points
    ax2.scatter([wa_plus_x], [wa_plus_y], s=150, c='red', 
               alpha=0.8, marker='^', linewidth=2, label='WA⁺ (weighted max)')
    ax2.scatter([wa_minus_x], [wa_minus_y], s=150, c='blue', 
               alpha=0.8, marker='v', linewidth=2, label='WA⁻ (weighted min)')
    
    # Show formulas
    formula1 = r'$WA^+ = \frac{\sum x_i \exp(\frac{x_i-x_{\max}}{\gamma})}{\sum \exp(\frac{x_i-x_{\max}}{\gamma})}$'
    formula2 = r'$HPWL^{(B)} = \sum [(WA^+_x - WA^-_x) + (WA^+_y - WA^-_y)]$'
    
    ax2.text(1, 9.5, formula1, ha='left', va='top', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    ax2.text(1, 8.5, formula2, ha='left', va='top', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Show calculation details
    calc_text_b = f'γ = {gamma:.1f}\n'
    calc_text_b += f'X-coordinates: {x_coords}\n'
    calc_text_b += f'WA⁺_x = {wa_plus_x:.2f}, WA⁻_x = {wa_minus_x:.2f}\n'
    calc_text_b += f'Y-coordinates: {y_coords}\n'
    calc_text_b += f'WA⁺_y = {wa_plus_y:.2f}, WA⁻_y = {wa_minus_y:.2f}\n'
    calc_text_b += f'HPWL = ({wa_plus_x:.2f}-{wa_minus_x:.2f}) + ({wa_plus_y:.2f}-{wa_minus_y:.2f})\n'
    calc_text_b += f'     = {hpwl_b:.2f}'
    
    ax2.text(1, 6.0, calc_text_b, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.3))
    
    ax2.set_xlabel('X coordinate', fontsize=14)
    ax2.set_ylabel('Y coordinate', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hpwl_models_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("="*70)
    print("HPWL MODELS VISUALIZATION")
    print("="*70)
    
    print("\nMODEL A: Manhattan-Distance-Based HPWL")
    print("-"*40)
    print(f"Instances coordinates:")
    for i, point in enumerate(net_points):
        print(f"  I{i+1}: ({point[0]:.1f}, {point[1]:.1f})")
    print(f"\nCenter point: ({center[0]:.1f}, {center[1]:.1f})")
    print("\nManhattan distances to center:")
    total = 0
    for i, point in enumerate(net_points):
        dist = abs(point[0]-center[0]) + abs(point[1]-center[1])
        total += dist
        print(f"  I{i+1}: |{point[0]:.1f}-{center[0]:.1f}| + |{point[1]:.1f}-{center[1]:.1f}| = {dist:.1f}")
    print(f"Total HPWL(A) = {total:.1f}")
    
    print("\n" + "="*70)
    print("MODEL B: Weighted-Average-Based HPWL")
    print("-"*40)
    print(f"Instances coordinates:")
    for i, point in enumerate(net_points_b):
        print(f"  I{i+1}: ({point[0]:.1f}, {point[1]:.1f})")
    print(f"\nTemperature parameter γ = {gamma:.1f}")
    print(f"\nX-coordinate analysis:")
    print(f"  x_min = {x_min:.1f}, x_max = {x_max:.1f}")
    print(f"  WA⁺_x = {wa_plus_x:.2f} (weighted toward max)")
    print(f"  WA⁻_x = {wa_minus_x:.2f} (weighted toward min)")
    print(f"  X-span = WA⁺_x - WA⁻_x = {wa_plus_x-wa_minus_x:.2f}")
    print(f"\nY-coordinate analysis:")
    print(f"  y_min = {y_min:.1f}, y_max = {y_max:.1f}")
    print(f"  WA⁺_y = {wa_plus_y:.2f}")
    print(f"  WA⁻_y = {wa_minus_y:.2f}")
    print(f"  Y-span = WA⁺_y - WA⁻_y = {wa_plus_y-wa_minus_y:.2f}")
    print(f"\nTotal HPWL(B) = {hpwl_b:.2f}")
    print("="*70)

# Run visualization
if __name__ == "__main__":
    visualize_hpwl_models()
    
# 执行带调试信息的函数
# hpwl = debug_get_hpwl_loss_from_net_tensor_wa_vectorized(p, net_tensor, site_coords_matrix, gamma)