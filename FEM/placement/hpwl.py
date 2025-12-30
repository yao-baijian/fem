import torch
from .config import *
from .logger import INFO, WARNING, ERROR

class HPWLCalculator:

    def __init__(self, debug=False):
        self.name = "HPWL-Calculator"
        self.net_hpwl = {}
        self.net_bbox = {}
        self.net_hpwl_no_io = {}
        self.net_bbox_no_io = {}
        
        self.total_hpwl = 0.0
        self.total_hpwl_no_io = 0.0
        
        self.nets = []
        self.net_names = []
        
        self.debug = debug
        pass
    
    def get_hpwl(self):
        return self.total_hpwl, self.total_hpwl_no_io
    
    def clear(self):
        self.total_hpwl = 0.0
        self.net_hpwl.clear()
        self.net_bbox.clear()
        
        self.total_hpwl_no_io = 0.0
        self.net_hpwl_no_io.clear()
        self.net_bbox_no_io.clear()
    
    def compute_net_hpwl_rapidwright(self, net, net_name, include_io=False):
        if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
            return 0.0, {}
        
        pins = net.getPins()
        if len(pins) < 2:
            return 0.0, {}
        
        coordinates_set = set()
        for pin in pins:
            if not include_io:
                if pin.getSiteInst().getSiteTypeEnum() in IO_SITE_ENUM:
                    continue
            
            site_inst = pin.getSiteInst()
            coord = (site_inst.getInstanceX(), site_inst.getInstanceY())
            coordinates_set.add(coord)
        
        coordinates = list(coordinates_set)
        
        if len(coordinates) < 2:
            return 0.0, {}
        
        hpwl, bbox = self._compute_hpwl_from_coordinates(coordinates)
    
        if include_io:
            self.net_hpwl[net_name] = hpwl
            self.net_bbox[net_name] = bbox
            self.total_hpwl += hpwl
        else:
            self.net_hpwl_no_io[net_name] = hpwl
            self.net_bbox_no_io[net_name] = bbox
            self.total_hpwl_no_io += hpwl
    
    def compute_net_hpwl(self, net_name, connected_sites, instance_coords, include_io=False):
        coordinates = []
        
        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])
            else:
                # WARNING(f"Site {site_name} not found in instance coordinates for net {net_name}")
                pass
        
        if len(coordinates) < 2:
            return 0.0, {}
        
        hpwl, bbox = self._compute_hpwl_from_coordinates(coordinates)
    
        if include_io:
            self.net_hpwl[net_name] = hpwl
            self.net_bbox[net_name] = bbox
            self.total_hpwl += hpwl
        else:
            self.net_hpwl_no_io[net_name] = hpwl
            self.net_bbox_no_io[net_name] = bbox
            self.total_hpwl_no_io += hpwl
        
    def compute_single_instance_hpwl(self, connected_sites, instance_coords):
        coordinates = []
        
        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])
        
        if len(coordinates) < 2:
            return 0.0, {}
        
        return self._compute_hpwl_from_coordinates(coordinates)
    
    def _compute_hpwl_from_coordinates(self, coordinates):
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        hpwl = (max_x - min_x) + (max_y - min_y)
        bbox = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'num_pins': len(coordinates)
        }
        
        return hpwl, bbox
    
    def compute_wa_hpwl(self,
                       p: torch.Tensor,
                       site_coords_matrix: torch.Tensor,
                       net_tensor: torch.Tensor,
                       gamma: float = 0.03) -> torch.Tensor:
        batch_size, num_instances, num_sites = p.shape
        num_nets = net_tensor.shape[0]
        
        # 计算期望坐标
        expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
        expected_x = expected_coords[..., 0]  # [batch_size, num_instances]
        expected_y = expected_coords[..., 1]  # [batch_size, num_instances]
        
        # 归一化坐标（WA方法需要归一化）
        x_min = expected_x.min(dim=1, keepdim=True)[0]
        x_max = expected_x.max(dim=1, keepdim=True)[0]
        y_min = expected_y.min(dim=1, keepdim=True)[0]
        y_max = expected_y.max(dim=1, keepdim=True)[0]
        
        x_norm = (expected_x - x_min) / (x_max - x_min + 1e-10)
        y_norm = (expected_y - y_min) / (y_max - y_min + 1e-10)
        
        # 扩展net_tensor
        net_tensor_expanded = net_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        if net_tensor_expanded.dtype == torch.bool:
            net_tensor_expanded = net_tensor_expanded.float()
        
        total_hpwl = torch.zeros(batch_size, device=p.device)
        
        # 分别处理x和y坐标
        for coord_norm in [x_norm, y_norm]:
            coord_expanded = coord_norm.unsqueeze(1).expand(-1, num_nets, -1)  # [batch_size, num_nets, num_instances]
            coord_masked = coord_expanded * net_tensor_expanded  # 过滤非net实例
            
            # 关键修复：正确计算max和min（忽略非net实例）
            large_neg = -1e10
            large_pos = 1e10
            
            # 对于WA_max计算：将非net坐标替换为large_neg
            zero_mask_max = (coord_masked == 0) & (net_tensor_expanded == 0)
            coord_for_max = torch.where(zero_mask_max, 
                                       torch.tensor(large_neg, device=p.device), 
                                       coord_masked)
            max_vals = coord_for_max.max(dim=2, keepdim=True)[0]
            
            # 对于WA_min计算：将非net坐标替换为large_pos
            zero_mask_min = (coord_masked == 0) & (net_tensor_expanded == 0)
            coord_for_min = torch.where(zero_mask_min,
                                       torch.tensor(large_pos, device=p.device),
                                       coord_masked)
            min_vals = coord_for_min.min(dim=2, keepdim=True)[0]
            
            # 计算加权平均
            weight_pos = torch.exp((coord_masked - max_vals) / gamma)
            numerator_pos = torch.sum(coord_masked * weight_pos * net_tensor_expanded, dim=2)
            denominator_pos = torch.sum(weight_pos * net_tensor_expanded, dim=2)
            
            weight_neg = torch.exp(-(coord_masked - min_vals) / gamma)
            numerator_neg = torch.sum(coord_masked * weight_neg * net_tensor_expanded, dim=2)
            denominator_neg = torch.sum(weight_neg * net_tensor_expanded, dim=2)
            
            # 避免除零
            denominator_pos = torch.where(denominator_pos == 0, 
                                         torch.tensor(1e-10, device=p.device), 
                                         denominator_pos)
            denominator_neg = torch.where(denominator_neg == 0,
                                         torch.tensor(1e-10, device=p.device),
                                         denominator_neg)
            
            wa_max = numerator_pos / denominator_pos
            wa_min = numerator_neg / denominator_neg
            
            coord_hpwl = wa_max - wa_min
            total_hpwl += torch.sum(coord_hpwl, dim=1)
        
        return total_hpwl
    
    def compute_jmatrix_hpwl(self,
                            p: torch.Tensor,
                            site_coords_matrix: torch.Tensor,
                            J: torch.Tensor) -> torch.Tensor:
        expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
        coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
        coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]
        
        manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
        
        # 加权距离
        weighted_dist = manhattan_dist * J.unsqueeze(0)  # [batch_size, num_instances, num_instances]
        
        # 只取上三角部分
        triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
        weighted_dist_triu = weighted_dist[:, triu_mask]  # [batch_size, num_pairs]
        
        # 总wirelength
        total_wirelength = torch.sum(weighted_dist_triu, dim=1)  # [batch_size]
        
        return total_wirelength