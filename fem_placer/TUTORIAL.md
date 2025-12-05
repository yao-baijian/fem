# FEM Placer 详解文档

## 目录
1. [概述](#概述)
2. [核心概念](#核心概念)
3. [架构设计](#架构设计)
4. [工作流程](#工作流程)
5. [关键组件详解](#关键组件详解)
6. [优化算法原理](#优化算法原理)
7. [使用示例](#使用示例)

---

## 概述

**FEM Placer** 是一个基于 **FEM (Field Effect Method)** 框架的 FPGA 布局工具。它使用物理场效应方法来解决 FPGA 芯片上的逻辑单元布局问题，目标是最小化线长（HPWL - Half-Perimeter Wirelength）并满足布局约束。

### 什么是 FPGA 布局问题？

FPGA（现场可编程门阵列）布局问题是指：给定一个 FPGA 设计（包含大量逻辑单元和它们之间的连接关系），需要将这些逻辑单元放置在 FPGA 芯片的可用位置上，使得：

1. **线长最小化**：连接单元之间的走线总长度尽可能短
2. **约束满足**：每个单元必须放置在兼容的位置上，且不能重叠
3. **性能优化**：考虑时序、功耗等其他因素

这是一个典型的 NP 难组合优化问题。

---

## 核心概念

### 1. 概率分布表示

FEM Placer 使用**分离的 X 和 Y 坐标概率分布**来表示每个单元的位置：

- `p_x[i, j]`：单元 `i` 在 X 方向位置 `j` 的概率
- `p_y[i, j]`：单元 `i` 在 Y 方向位置 `j` 的概率

这种设计允许我们独立地优化 X 和 Y 坐标，大大简化了优化过程。

### 2. 耦合矩阵 (Coupling Matrix)

耦合矩阵 `J` 表示单元之间的连接强度：

```python
J[i, j] = 连接单元 i 和单元 j 的网线数量
```

如果两个单元之间有多个网线连接，它们的耦合强度就更高，优化算法会倾向于将它们放置得更近。

### 3. 自由能 (Free Energy)

FEM 优化的目标是最小化自由能：

```
F = E - T * S
```

其中：
- **E**：能量项（HPWL 损失 + 约束损失）
- **T**：温度参数（控制优化过程的探索性）
- **S**：熵（概率分布的分散程度）

随着优化进行，温度逐渐降低（退火），系统从探索状态逐渐收敛到确定状态。

---

## 架构设计

FEM Placer 采用模块化设计，主要包含以下组件：

```
fem_placer/
├── placer.py          # FpgaPlacer: RapidWright 接口，处理 FPGA 设计文件
├── optimizer.py        # FPGAPlacementOptimizer: FEM 优化器核心
├── objectives.py       # 目标函数：HPWL 损失、约束损失
├── legalizer.py        # Legalizer: 布局合法化（解决重叠）
├── router.py           # Router: 连接布线（用于可视化）
├── drawer.py           # PlacementDrawer: 可视化工具
└── utils.py            # 工具函数：设计解析、矩阵构建
```

### 设计原则

1. **与 FEM 框架解耦**：使用 FEM 的 `customize` 接口，不修改 FEM 核心代码
2. **模块化**：每个组件职责单一，易于维护和扩展
3. **可扩展性**：支持自定义目标函数和约束

---

## 工作流程

完整的 FPGA 布局流程包括以下步骤：

```
1. 初始化阶段
   ├── 读取 FPGA 设计文件 (.dcp)
   ├── 解析逻辑单元和连接关系
   ├── 构建耦合矩阵 J
   └── 初始化概率分布 p_x, p_y

2. 优化阶段
   ├── 迭代优化（FEM 算法）
   ├── 计算 HPWL 损失
   ├── 计算约束损失
   ├── 更新概率分布
   └── 可视化（可选）

3. 后处理阶段
   ├── 从概率分布推断硬位置
   ├── 合法化（解决重叠）
   └── 布线（用于可视化）

4. 输出阶段
   ├── 生成最终布局
   └── 可视化结果
```

---

## 关键组件详解

### 1. FpgaPlacer (`placer.py`)

**职责**：与 RapidWright 交互，处理 FPGA 设计文件

**主要功能**：

- **设计解析**：
  - 读取 `.dcp` 文件（Vivado 设计检查点）
  - 识别可优化单元（SLICE）和固定单元（IO）
  - 提取网线连接关系

- **区域估计**：
  - 根据单元数量估计布局区域大小
  - 计算边界框（bbox）

- **连接性记录**：
  - 构建 `site_to_site_connectivity`：单元之间的连接
  - 构建 `io_to_site_connectivity`：IO 与单元之间的连接

**关键数据结构**：

```python
class FpgaPlacer:
    optimizable_insts: List[SiteInst]      # 可优化的逻辑单元
    fixed_insts: List[SiteInst]            # 固定的 IO 单元
    available_sites: List[Site]            # 可用的放置位置
    site_to_site_connectivity: Dict       # 单元间连接关系
    bbox: Dict                            # 布局区域边界
```

### 2. FPGAPlacementOptimizer (`optimizer.py`)

**职责**：执行 FEM 优化算法

**核心算法**：

```python
def iterate_placement(...):
    # 1. 初始化势函数 h_x, h_y
    h_x, h_y = initialize_potentials(...)
    
    # 2. 迭代优化
    for step in range(num_steps):
        # 2.1 将势函数转换为概率分布
        p_x = softmax(h_x, dim=2)
        p_y = softmax(h_y, dim=2)
        
        # 2.2 计算损失
        hpwl_loss, constraint_loss = expected_fpga_placement_xy(J, p_x, p_y)
        
        # 2.3 计算自由能
        free_energy = hpwl_loss + constraint_loss - entropy / beta[step]
        
        # 2.4 反向传播和更新
        free_energy.backward()
        optimizer.step()
    
    return p_x, p_y
```

**关键参数**：

- `num_trials`：并行试验次数（探索多个初始条件）
- `num_steps`：优化迭代步数
- `q`：网格大小（X 和 Y 方向的离散化粒度）
- `betamin, betamax`：温度参数范围（控制退火过程）
- `anneal`：退火策略（'lin', 'exp', 'inverse'）

### 3. 目标函数 (`objectives.py`)

#### 3.1 HPWL 损失 (`get_hpwl_loss_xy_simple`)

**HPWL (Half-Perimeter Wirelength)** 是衡量线长的标准指标：

```
HPWL(net) = (max_x - min_x) + (max_y - min_y)
```

对于每个网线，计算其所有连接单元的边界框的半周长。

**实现**：

```python
def get_hpwl_loss_xy_simple(J, p_x, p_y):
    # 1. 从概率分布计算期望坐标
    expected_coords = get_placements_from_xy_st(p_x, p_y)
    
    # 2. 计算曼哈顿距离
    coords_i = expected_coords.unsqueeze(2)  # [batch, N, 1, 2]
    coords_j = expected_coords.unsqueeze(1)   # [batch, 1, N, 2]
    manhattan_dist = |coords_i - coords_j|
    
    # 3. 加权求和（权重来自耦合矩阵 J）
    weighted_dist = manhattan_dist * J
    total_wirelength = sum(weighted_dist)
    
    return total_wirelength
```

#### 3.2 约束损失 (`get_constraints_loss_xy`)

**目标**：防止多个单元放置在同一位置

**实现**：

```python
def get_constraints_loss_xy(p_x, p_y):
    # 1. 计算每个位置的使用概率
    p_instance_xy = p_x ⊗ p_y  # 外积
    p_position_usage = sum(p_instance_xy, dim=1)  # 每个位置的总使用概率
    
    # 2. 计算目标使用率
    target_usage = num_instances / (num_x * num_y)
    
    # 3. 惩罚超出目标使用率的位置
    excess = softplus(10 * (p_position_usage - target_usage)^2)
    constraint_loss = sum(excess)
    
    return constraint_loss
```

### 4. Legalizer (`legalizer.py`)

**职责**：将优化得到的连续坐标转换为合法的离散位置，解决重叠问题

**算法**：

```python
def legalize_placement(coordinates):
    # 1. 检测冲突
    conflicts = find_overlaps(coordinates)
    
    # 2. 对每个冲突的单元，找到最近的可用位置
    for conflict_unit in conflicts:
        new_pos = find_nearest_available_position(conflict_unit)
        coordinates[conflict_unit] = new_pos
    
    return legalized_coordinates
```

**策略**：使用螺旋搜索（spiral search）从原始位置向外搜索最近的可用位置。

### 5. Router (`router.py`)

**职责**：计算单元之间的走线路径（主要用于可视化）

**算法**：使用简单的曼哈顿布线（Manhattan routing）

```python
def _manhattan_route(start, end):
    # 先走 X 方向，再走 Y 方向
    route = [
        horizontal_segment(start_x -> end_x, y=start_y),
        vertical_segment(start_y -> end_y, x=end_x)
    ]
    return route
```

### 6. PlacementDrawer (`drawer.py`)

**职责**：可视化布局过程和结果

**功能**：

- **多步骤可视化**：显示优化过程中的布局变化
- **布线可视化**：显示单元之间的连接
- **密度图**：显示位置使用情况

---

## 优化算法原理

### FEM 方法的核心思想

FEM 将离散优化问题转化为连续优化问题：

1. **离散问题**：每个单元必须放在某个离散的位置上
2. **连续化**：用概率分布表示单元的位置不确定性
3. **优化**：通过梯度下降优化概率分布
4. **离散化**：最终从概率分布中采样或取最大概率位置

### 自由能最小化

自由能公式：

```
F = E - T * S
```

- **能量项 E**：
  - HPWL 损失：鼓励连接紧密的单元靠近
  - 约束损失：防止重叠

- **熵项 S**：
  - 衡量概率分布的分散程度
  - 高熵 = 不确定性高（探索阶段）
  - 低熵 = 确定性高（收敛阶段）

- **温度 T**：
  - 控制熵的权重
  - 高温：更注重探索（高熵）
  - 低温：更注重利用（低熵）

### 退火策略

温度参数 `beta = 1/T` 随迭代步数变化：

- **线性退火**：`beta = linspace(betamin, betamax, steps)`
- **指数退火**：`beta = exp(linspace(log(betamin), log(betamax), steps))`
- **逆退火**：`beta = 1 / linspace(1/betamax, 1/betamin, steps)`

逆退火通常效果最好，因为它在早期快速升温（探索），后期缓慢降温（收敛）。

### Straight-Through Estimator

在计算 HPWL 损失时，使用 Straight-Through Estimator 来处理离散采样：

```python
def get_placements_from_xy_st(p_x, p_y, grid_x, grid_y):
    # 硬坐标（前向传播，用于计算损失）
    with torch.no_grad():
        x_hard = grid_x[argmax(p_x)]
        y_hard = grid_y[argmax(p_y)]
        hard_coords = [x_hard, y_hard]
    
    # 软坐标（反向传播，用于梯度计算）
    expected_x = matmul(p_x, grid_x)
    expected_y = matmul(p_y, grid_y)
    expected_coords = [expected_x, expected_y]
    
    # Straight-Through：前向用硬坐标，反向用软坐标
    return expected_coords + (hard_coords - expected_coords).detach()
```

这样可以在保持梯度可导的同时，使用离散的硬坐标计算损失。

---

## 使用示例

### 基本使用流程

```python
import torch
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    FPGAPlacementOptimizer
)
from fem_placer.utils import parse_fpga_design

# 1. 初始化 FPGA 布局器
fpga_wrapper = FpgaPlacer()
fpga_wrapper.init_placement(
    './vivado/output_dir/post_impl.dcp',  # 输入设计文件
    'optimized_placement.dcp'              # 输出文件
)

# 2. 解析设计，获取耦合矩阵
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)
area_length = fpga_wrapper.bbox['area_length']

# 3. 设置可视化
drawer = PlacementDrawer(bbox=fpga_wrapper.bbox)

# 4. 创建优化器
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    coupling_matrix=J,
    drawer=drawer,
    visualization_steps=[0, 250, 500, 750, 999]
)

# 5. 执行优化
config, result = optimizer.optimize(
    num_trials=10,      # 10 个并行试验
    num_steps=1000,     # 1000 步迭代
    dev='cpu',          # 使用 CPU
    q=area_length       # 网格大小
)

# 6. 选择最优解
optimal_idx = torch.argmin(result)
best_config = config[optimal_idx]

# 7. 合法化
legalizer = Legalizer(fpga_wrapper.bbox)
placement_legalized = legalizer.legalize_placement(
    best_config, 
    max_attempts=100
)

# 8. 计算最终 HPWL
hpwl_final = fpga_wrapper.estimate_solver_hpwl(
    placement_legalized,
    io_coords=None,
    include_io=False
)

# 9. 布线和可视化
router = Router(fpga_wrapper.bbox)
routes = router.route_connections(J, placement_legalized.unsqueeze(0))[0]

drawer.draw_complete_placement(
    placement_legalized,
    routes,
    1000,
    title_suffix="Final Placement with Routing"
)
```

### 参数调优建议

1. **网格大小 `q`**：
   - 太小：精度不够，可能找不到最优解
   - 太大：计算开销大，优化慢
   - 建议：设置为 `area_length`（布局区域的边长）

2. **迭代步数 `num_steps`**：
   - 太少：可能未收敛
   - 太多：浪费时间
   - 建议：1000-5000 步，根据问题规模调整

3. **并行试验数 `num_trials`**：
   - 更多试验可以探索更多初始条件
   - 建议：10-50 个

4. **温度参数**：
   - `betamin`：控制最终收敛温度（建议 0.01-0.1）
   - `betamax`：控制初始探索温度（建议 0.5-2.0）
   - `anneal`：建议使用 'inverse'

---

## 总结

FEM Placer 是一个强大的 FPGA 布局工具，它：

1. **使用物理场方法**：将组合优化问题转化为连续优化
2. **模块化设计**：易于维护和扩展
3. **完整流程**：从设计解析到最终布局的完整解决方案
4. **可视化支持**：帮助理解优化过程

通过概率分布建模和自由能最小化，FEM Placer 能够在合理的时间内找到高质量的布局方案。

---

## 参考资料

- [FEM Framework](https://github.com/Fanerst/FEM)
- [RapidWright](https://www.rapidwright.io/)
- FPGA Placement 相关论文和算法

