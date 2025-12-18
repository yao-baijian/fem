# hpwl_calculator.py
import torch

class HPWLCalculator:

    def __init__(self, slice_site_enum=None, io_site_enum=None):
        self.slice_site_enum = slice_site_enum
        self.io_site_enum = io_site_enum
        
        # 存储原始设计中的HPWL数据
        self.net_hpwl = {}
        self.net_bbox = {}
        self.net_hpwl_no_io = {}
        self.net_bbox_no_io = {}
        
        self.total_hpwl = 0.0
        self.total_hpwl_no_io = 0.0
        
        self.nets = []
        self.net_names = []
        
        self.debug = True
        pass
    
    def analyze_design_hpwl(self, design, save_debug=True):
        self.total_hpwl = 0.0
        self.net_hpwl.clear()
        self.net_bbox.clear()
        
        self.total_hpwl_no_io = 0.0
        self.net_hpwl_no_io.clear()
        self.net_bbox_no_io.clear()
        
        self.nets = list(design.getNets())
        self.net_names = [net.getName() for net in self.nets]
        
        # 计算包含IO的HPWL（注意：原始代码中这部分计算为空）
        for net in self.nets:
            net_name = net.getName()
            hpwl = 0  # 原始代码中这部分为0
            bbox = {}
            
            self.net_hpwl[net_name] = hpwl
            self.net_bbox[net_name] = bbox
            self.total_hpwl += hpwl
        
        # 计算不包含IO的HPWL
        skipped_count = 0
        for net in self.nets:
            net_name = net.getName()
            hpwl, bbox = self._calculate_net_hpwl_rapidwright(net, False)
            
            self.net_hpwl_no_io[net_name] = hpwl
            self.net_bbox_no_io[net_name] = bbox
            self.total_hpwl_no_io += hpwl
            
            if hpwl == 0.0:
                skipped_count += 1
        
        print(f"HPWLCalculator: Analyzed {len(self.nets)} nets")
        print(f"  Total HPWL (without IO): {self.total_hpwl_no_io:.2f}")
        print(f"  Skipped {skipped_count} nets with zero HPWL")
        
        # 保存debug信息
        if save_debug:
            self.save_net_debug_info()
        
        return self.total_hpwl, self.total_hpwl_no_io
    
    def _calculate_net_hpwl_rapidwright(self, net, include_io=True):
        """
        计算单个网络的HPWL，替代原来的_calculate_net_hpwl_rapidwright方法
        """
        if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
            return 0.0, {}
        
        pins = net.getPins()
        if len(pins) < 2:
            return 0.0, {}
        
        coordinates_set = set()
        for pin in pins:
            if not include_io and self.io_site_enum:
                if pin.getSiteInst().getSiteTypeEnum() in self.io_site_enum:
                    continue
            
            site_inst = pin.getSiteInst()
            coord = (site_inst.getInstanceX(), site_inst.getInstanceY())
            coordinates_set.add(coord)
        
        coordinates = list(coordinates_set)
        
        if len(coordinates) < 2:
            return 0.0, {}
        
        return self._calculate_hpwl_from_coordinates(coordinates)
    
    def _calculate_hpwl_from_coordinates(self, coordinates):
        """
        从坐标计算HPWL
        """
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
    
    def save_net_debug_info(self, output_path='result/net_debug_info.txt'):
        """
        保存网络debug信息
        """
        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name\tHPWL\tSite_Count\tSites_Info\n")
            for idx, net in enumerate(self.nets):
                net_name = net.getName()
                hpwl = self.net_hpwl_no_io.get(net_name, 0.0)
                
                # 收集所有pin的详细信息
                sites_set = set()
                if not (net.isClockNet() or net.isVCCNet() or net.isGNDNet()):
                    pins = net.getPins()
                    
                    for pin in pins:
                        site_inst = pin.getSiteInst()
                        if site_inst:
                            site_name = site_inst.getName()
                            site_x = site_inst.getInstanceX()
                            site_y = site_inst.getInstanceY()
                            site_key = f"{site_name}({site_x},{site_y})"
                            sites_set.add(site_key)
                
                sites_list = sorted(list(sites_set))
                site_count = len(sites_list)
                sites_str = " | ".join(sites_list)
                
                if hpwl > 0.0:
                    f.write(f"{idx}\t{net_name}\t{hpwl:.2f}\t{site_count}\t{sites_str}\n")
    
    def calculate_solver_hpwl(self, placements, fixed_placements, instance_mapping, 
                             fixed_instance_mapping, net_to_sites, include_io=False):
        """
        计算求解器生成的placement的HPWL
        placements: 优化实例的坐标 [num_instances, 2]
        fixed_placements: 固定实例的坐标 [num_fixed, 2]
        """
        instance_coords = {}
        
        # 添加优化实例的坐标
        for instance_id, coord in enumerate(placements):
            site_name = self._get_site_name_by_id(instance_id, instance_mapping, is_fixed=False)
            if site_name:
                instance_coords[site_name] = coord
        
        # 添加固定实例的坐标（如果需要）
        if include_io:
            for fixed_id, coord in enumerate(fixed_placements):
                site_name = self._get_site_name_by_id(fixed_id, fixed_instance_mapping, is_fixed=True)
                if site_name:
                    instance_coords[site_name] = coord
        
        total_hpwl = 0.0
        skipped = 0
        
        for net_name, connected_sites in net_to_sites.items():
            hpwl, _ = self._calculate_net_hpwl_from_instance_coords(
                net_name, connected_sites, instance_coords, include_io)
            
            if hpwl == 0.0:
                skipped += 1
                continue
            
            total_hpwl += hpwl
        
        if self.debug:
            print(f"HPWLCalculator: total {len(net_to_sites)} nets, skipped {skipped} nets.")
            print(f"  Solver HPWL: {total_hpwl:.2f}")
        
        # 保存solver HPWL debug信息
        self.save_solver_hpwl_debug(instance_coords, net_to_sites)
        
        return total_hpwl
    
    def _calculate_net_hpwl_from_instance_coords(self, net_name, connected_sites, 
                                                 instance_coords, include_io=True):
        """
        从实例坐标计算网络HPWL
        """
        coordinates = []
        
        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])
        
        if len(coordinates) < 2:
            return 0.0, {}
        
        return self._calculate_hpwl_from_coordinates(coordinates)
    
    def _get_site_name_by_id(self, instance_id, mapping, is_fixed=False):
        """
        根据ID获取站点名称
        """
        if is_fixed:
            # 固定实例的映射
            for site_name, idx in mapping.items():
                if idx == instance_id:
                    return site_name
        else:
            # 优化实例的映射
            for site_name, idx in mapping.items():
                if idx == instance_id:
                    return site_name
        return None
    
    def save_solver_hpwl_debug(self, instance_coords, net_to_sites, 
                              output_path='result/solver_hpwl_debug.txt'):
        """
        保存求解器HPWL的debug信息
        """
        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name\tHPWL\tInstance_Count\tInstances_Info\n")
            
            for idx, (net_name, connected_sites) in enumerate(net_to_sites.items()):
                hpwl, _ = self._calculate_net_hpwl_from_instance_coords(
                    net_name, connected_sites, instance_coords, include_io=True)
                
                if hpwl == 0.0:
                    continue
                
                instances_info = []
                for site_name in connected_sites:
                    if site_name in instance_coords:
                        coord = instance_coords[site_name]
                        instance_info = f"{site_name}[({coord[0]:.2f},{coord[1]:.2f})]"
                        instances_info.append(instance_info)
                
                instance_count = len(instances_info)
                instances_str = " | ".join(instances_info)
                
                f.write(f"{idx}\t{net_name}\t{hpwl:.2f}\t{instance_count}\t{instances_str}\n")
                
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