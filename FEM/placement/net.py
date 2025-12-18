# net_manager.py
import torch
import rapidwright
from .hpwl import HPWLCalculator
from typing import Dict, Set
from com.xilinx.rapidwright.design import Design, Net

class NetManager:
    
    def __init__(self, slice_site_enum=None, io_site_enum=None):
        self.slice_site_enum = slice_site_enum
        self.io_site_enum = io_site_enum
    
        self.nets = []  
        self.net_names = [] 
        
        self.net_to_sites = {} 
        self.site_to_nets = {} 
        self.site_to_site_connectivity = {} 
        self.io_to_site_connectivity = {} 
        
        self.net_tensor = None  # [num_nets, num_sites] 
        self.connection_matrix = None  # [num_sites, num_sites] 

        self.debug_src_root = "result/net_manager_debug.txt"
        
        self.hpwl_calculator = HPWLCalculator()
        
    def analyze_nets(self, design: Design, instance_mapping: Dict[str, int]):
        self.nets = list(design.getNets())
        self.net_names = [net.getName() for net in self.nets]
        
        valid_net_idx = 0
        
        for net in self.nets:
            net_name = net.getName()
            
            # 跳过时钟、电源、地网络
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue
            
            # 收集网络中的站点
            sites_in_net = set()
            logic_sites = set()
            io_sites = set()
            
            for pin in net.getPins():
                site_inst = pin.getSiteInst()
                site_name = site_inst.getName()
                site_type = site_inst.getSiteTypeEnum()
                
                sites_in_net.add(site_name)
                
                # 分类站点
                if self.slice_site_enum and site_type in self.slice_site_enum:
                    logic_sites.add(site_name)
                elif self.io_site_enum and site_type in self.io_site_enum:
                    io_sites.add(site_name)
            
            # 至少有两个逻辑站点才记录为有效网络
            if len(logic_sites) >= 2:
                self.net_to_sites[valid_net_idx] = {
                    'name': net_name,
                    'all_sites': list(sites_in_net),
                    'logic_sites': list(logic_sites),
                    'io_sites': list(io_sites)
                }
                
                # 更新站点到网络的映射
                for site_name in sites_in_net:
                    if site_name not in self.site_to_nets:
                        self.site_to_nets[site_name] = []
                    self.site_to_nets[site_name].append(valid_net_idx)
                
                # 记录连接关系
                self._record_connectivity(valid_net_idx, logic_sites, io_sites)
                
                valid_net_idx += 1
        
        # 创建张量表示
        self._create_tensor_representations(instance_mapping)

        instance_count = len(instance_mapping)
        self.save_tensor_debug_info(instance_count=instance_count)
        self.save_connectivity_debug_info()

        print(f"NetManager: Processed {valid_net_idx} valid nets out of {len(self.nets)} total nets")
        print(f"      Total {len(self.site_to_site_connectivity)} site-to-site routes")
        print(f"      Total {len(self.io_to_site_connectivity)} io-to-site routes")
        print(f"      Total {len(self.net_to_sites)} inter-tile routes")
        
        return self
    
    def _record_connectivity(self, net_idx: int, logic_sites: Set[str], io_sites: Set[str]):
        logic_sites_list = list(logic_sites)
        io_sites_list = list(io_sites)
        
        # 1. 记录IO站点与逻辑站点之间的连接（双向）
        for i in range(len(io_sites_list)):
            for j in range(len(logic_sites_list)):
                io_inst1, inst2 = io_sites_list[i], logic_sites_list[j]
                
                # IO到逻辑站点的连接
                if io_inst1 not in self.io_to_site_connectivity:
                    self.io_to_site_connectivity[io_inst1] = {}
                if inst2 not in self.io_to_site_connectivity[io_inst1]:
                    self.io_to_site_connectivity[io_inst1][inst2] = 0
                self.io_to_site_connectivity[io_inst1][inst2] += 1
                
                # 逻辑站点到IO的连接（双向记录）
                if inst2 not in self.io_to_site_connectivity:
                    self.io_to_site_connectivity[inst2] = {}
                if io_inst1 not in self.io_to_site_connectivity[inst2]:
                    self.io_to_site_connectivity[inst2][io_inst1] = 0
                self.io_to_site_connectivity[inst2][io_inst1] += 1
        
        # 2. 记录逻辑站点之间的连接
        # 注意：原始代码有weight计算，但实际使用的是 += 1
        # weight = (len(logic_sites_list) - 1) ** 2  # 这个weight在原始代码中计算了但没有使用
        
        for i in range(len(logic_sites_list)):
            for j in range(i + 1, len(logic_sites_list)):
                inst1, inst2 = logic_sites_list[i], logic_sites_list[j]
                
                # 注意：原始代码中有重复的 += 1，实际上每个连接加了两次
                # 第一次：self.site_to_site_connectivity[inst1][inst2] += 1
                # 第二次：self.site_to_site_connectivity[inst1][inst2] += 1 (重复)
                
                # 双向记录
                if inst1 not in self.site_to_site_connectivity:
                    self.site_to_site_connectivity[inst1] = {}
                if inst2 not in self.site_to_site_connectivity[inst1]:
                    self.site_to_site_connectivity[inst1][inst2] = 0
                self.site_to_site_connectivity[inst1][inst2] += 1  # 第一次加1
                
                if inst2 not in self.site_to_site_connectivity:
                    self.site_to_site_connectivity[inst2] = {}
                if inst1 not in self.site_to_site_connectivity[inst2]:
                    self.site_to_site_connectivity[inst2][inst1] = 0
                self.site_to_site_connectivity[inst2][inst1] += 1  # 对称加1
                
                # 原始代码中有重复的 += 1，这里我们只加一次
                # self.site_to_site_connectivity[inst1][inst2] += 1  # 原始代码重复的第二次加1
    
    def _create_tensor_representations(self, instance_mapping: Dict[str, int]):
        """创建张量表示的网络连接矩阵"""
        num_nets = len(self.net_to_sites)
        num_instances = len(instance_mapping)
        
        # 创建net_tensor
        self.net_tensor = torch.zeros((num_nets, num_instances), dtype=torch.bool)
        
        for net_idx, net_data in self.net_to_sites.items():
            for site_name in net_data['logic_sites']:
                if site_name in instance_mapping:
                    inst_idx = instance_mapping[site_name]
                    self.net_tensor[net_idx, inst_idx] = True
        
        # 创建连接矩阵J
        self.connection_matrix = torch.zeros((num_instances, num_instances))
        
        for inst1_name, connections in self.site_to_site_connectivity.items():
            if inst1_name in instance_mapping:
                inst1_idx = instance_mapping[inst1_name]
                for inst2_name, weight in connections.items():
                    if inst2_name in instance_mapping:
                        inst2_idx = instance_mapping[inst2_name]
                        self.connection_matrix[inst1_idx, inst2_idx] += weight
                        self.connection_matrix[inst2_idx, inst1_idx] += weight
        
        print(f"Created tensor representations: "
              f"net_tensor shape {self.net_tensor.shape}, "
              f"connection_matrix shape {self.connection_matrix.shape}")
    
    def get_net_statistics(self) -> Dict:
        """获取网络统计信息"""
        if self.net_tensor is None:
            return {}
        
        stats = {
            'num_nets': self.net_tensor.shape[0],
            'num_instances': self.net_tensor.shape[1],
            'avg_instances_per_net': self.net_tensor.sum(dim=1).float().mean().item(),
            'max_instances_per_net': self.net_tensor.sum(dim=1).max().item(),
            'min_instances_per_net': self.net_tensor.sum(dim=1).min().item(),
            'avg_nets_per_instance': self.net_tensor.sum(dim=0).float().mean().item(),
        }
        return stats
    
    def validate_net_tensor(self) -> Dict:
        """验证net_tensor的有效性"""
        if self.net_tensor is None:
            return {'is_valid': False, 'error': 'net_tensor is None'}
        
        issues = []
        
        # 检查空网络
        empty_nets = (self.net_tensor.sum(dim=1) == 0).sum().item()
        if empty_nets > 0:
            issues.append(f"{empty_nets} empty nets")
        
        # 检查单实例网络
        single_instance_nets = (self.net_tensor.sum(dim=1) == 1).sum().item()
        if single_instance_nets > 0:
            issues.append(f"{single_instance_nets} single-instance nets")
        
        # 检查孤立实例
        isolated_instances = (self.net_tensor.sum(dim=0) == 0).sum().item()
        if isolated_instances > 0:
            issues.append(f"{isolated_instances} isolated instances")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'empty_nets': empty_nets,
            'single_instance_nets': single_instance_nets,
            'isolated_instances': isolated_instances
        }
    
    def save_debug_info(self):
        with open(self.debug_src_root, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("NetManager Debug Information\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write("Basic Information:\n")
            f.write(f"  Total nets processed: {len(self.net_to_sites)}\n")
            
            if self.net_tensor is not None:
                f.write(f"  net_tensor shape: {self.net_tensor.shape}\n")
                f.write(f"  connection_matrix shape: {self.connection_matrix.shape}\n\n")
            
            # 网络详情
            f.write("Network Details (first 10 nets):\n")
            for net_idx in range(min(10, len(self.net_to_sites))):
                net_data = self.net_to_sites.get(net_idx, {})
                f.write(f"  Net {net_idx} ({net_data.get('name', 'N/A')}):\n")
                f.write(f"    Logic sites: {len(net_data.get('logic_sites', []))}\n")
                f.write(f"    IO sites: {len(net_data.get('io_sites', []))}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"NetManager debug info saved to {self.debug_src_root}")

    def save_tensor_debug_info(self, output_path='result/net_to_slice_sites_tensor_debug.txt', instance_count=None):
        if self.net_tensor is None:
            print("Warning: net_tensor is None, skipping debug info save")
            return
        
        num_nets = self.net_tensor.shape[0]
        if instance_count is None:
            instance_count = self.net_tensor.shape[1]
        
        with open(output_path, 'w') as f:
            # 写入标题行
            f.write("Net_IDX\tNet_Name")
            for instance_idx in range(instance_count):
                f.write(f"\t{instance_idx}")
            f.write("\n")
            
            # 写入每个网络的信息
            for net_idx in range(num_nets):
                # 获取网络名称
                net_name = self.net_to_sites.get(net_idx, {}).get('name', f'Net_{net_idx}')
                f.write(f"{net_idx}\t{net_name}")
                
                for instance_idx in range(instance_count):
                    # 检查索引是否在有效范围内
                    if instance_idx < self.net_tensor.shape[1]:
                        value = 1 if self.net_tensor[net_idx, instance_idx] else 0
                    else:
                        value = 0
                    f.write(f"\t{value}")
                f.write("\n")
            
            # 添加汇总信息
            f.write(f"\n=== Summary ===\n")
            f.write(f"Total Nets: {num_nets}\n")
            f.write(f"Total Instances: {instance_count}\n")
            f.write(f"Tensor Shape: {self.net_tensor.shape}\n")
            
            # 统计每个网络的连接数
            f.write(f"\n=== Connections per Net ===\n")
            f.write("Net_IDX\tNet_Name\tConnections\tConnected_Instances\n")
            for net_idx in range(num_nets):
                if instance_idx < self.net_tensor.shape[1]:
                    connections = self.net_tensor[net_idx].sum().item()
                else:
                    connections = 0
                    
                net_name = self.net_to_sites.get(net_idx, {}).get('name', f'Net_{net_idx}')
                connected_instances = []
                
                for instance_idx in range(instance_count):
                    if instance_idx < self.net_tensor.shape[1] and self.net_tensor[net_idx, instance_idx]:
                        connected_instances.append(str(instance_idx))
                        
                instances_str = ", ".join(connected_instances) if connected_instances else "None"
                f.write(f"{net_idx}\t{net_name}\t{int(connections)}\t{instances_str}\n")
            
            # 统计每个instance被多少个网络连接
            f.write(f"\n=== Connections per Instance ===\n")
            f.write("Instance_ID\tConnections\tConnected_Nets\n")
            for instance_idx in range(instance_count):
                if instance_idx < self.net_tensor.shape[1]:
                    connections = self.net_tensor[:, instance_idx].sum().item()
                else:
                    connections = 0
                    
                connected_nets = []
                for net_idx in range(num_nets):
                    if instance_idx < self.net_tensor.shape[1] and self.net_tensor[net_idx, instance_idx]:
                        connected_nets.append(str(net_idx))
                        
                nets_str = ", ".join(connected_nets) if connected_nets else "None"
                f.write(f"{instance_idx}\t{int(connections)}\t{nets_str}\n")
        
        print(f"Net tensor debug info saved to {output_path}")

    def save_connectivity_debug_info(self, output_path='result/connectivity_debug.txt'):
        """保存连接关系debug信息"""
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Site-to-Site Connectivity\n")
            f.write("=" * 60 + "\n")
            
            total_connections = 0
            for site1, connections in self.site_to_site_connectivity.items():
                for site2, weight in connections.items():
                    total_connections += weight
                    f.write(f"{site1} <-> {site2}: {weight}\n")
            
            f.write(f"\nTotal site-to-site connections: {total_connections}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("IO-to-Site Connectivity\n")
            f.write("=" * 60 + "\n")
            
            io_connections = 0
            for io_site, connections in self.io_to_site_connectivity.items():
                for logic_site, weight in connections.items():
                    io_connections += weight
                    f.write(f"{io_site} -> {logic_site}: {weight}\n")
            
            f.write(f"\nTotal IO-to-site connections: {io_connections}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Network Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total nets analyzed: {len(self.net_to_sites)}\n")
            f.write(f"Total inter-tile routes: {len(self.net_to_sites)}\n")
        