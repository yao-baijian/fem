import sys
import os
import torch
import rapidwright
import numpy as np
from typing import Dict, Set, Tuple
from com.xilinx.rapidwright.design import Design, Net
from .hpwl import HPWLCalculator
from .config import *
from .logger import INFO, WARNING, ERROR

class NetManager:

    def __init__(self,
                 get_site_inst_id_by_name_func=None,
                 get_site_inst_name_by_id_func=None,
                 map_coords_to_instance_func=None,
                 debug=False,
                 device = 'cpu'):

        self.debug = debug
        self.device = device
        self.site_to_site_connectivity = {}
        self.io_to_site_connectivity = {}
        self.net_to_sites = {}
        self.site_to_nets = {}
        self.nets = []
        self.net_names = []
        self.net_tensor = None
        self.insts_matrix = None
        self.io_insts_matrix = None
        self.logic_depth = 1.0  # Store estimated logic depth
        self.max_degree = 0 
        self.avg_degree = 0.0

        self.get_site_inst_id_by_name_func = get_site_inst_id_by_name_func
        self.get_site_inst_name_by_id_func = get_site_inst_name_by_id_func
        self.map_coords_to_instance_func = map_coords_to_instance_func

        self.debug_src_root = "result"
        self.hpwl_calculator = HPWLCalculator(device, debug=debug)
    
    def set_debug_path(self, result_dir='result', instance_name=None):
        self.debug_src_root = os.path.join(result_dir, instance_name) if instance_name else result_dir

    def analyze_design_hpwl(self, design):
        self.hpwl_calculator.clear()
        self.nets = design.getNets()
        self.net_names = [net.getName() for net in self.nets]

        for net in self.nets:
            net_name = net.getName()
            self.hpwl_calculator.compute_net_hpwl_rapidwright(net, net_name, True)

        for net in self.nets:
            net_name = net.getName()
            self.hpwl_calculator.compute_net_hpwl_rapidwright(net, net_name, False)

        hpwl = self.hpwl_calculator.get_hpwl()
        total_hpwl = hpwl['hpwl']
        total_hpwl_no_io = hpwl['hpwl_no_io']
        INFO(f"Nets num: {len(self.nets)}, total hpwl: {total_hpwl:.2f}, without io: {total_hpwl_no_io:.2f} ")
        
        # Estimate logic depth from the design
        self._estimate_logic_depth()
        self._calculate_net_degrees()
        
        if self.debug:
            self.save_net_debug_info()

        return total_hpwl, total_hpwl_no_io
    
    def get_net_degrees(self) -> tuple[int, float]:
        return self.max_degree, self.avg_degree

    # Calculate network degree statistics for placer ML
    def _calculate_net_degrees(self):
        """
        Calculate network degree statistics.
        Returns:
            (max_degree, avg_degree)
        """
        if not self.nets or len(self.nets) == 0:
            return 0, 0.0
        degrees = []
        for net in self.nets:
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue
            pins = net.getPins()
            # Count unique sites that this net connects to
            sites_in_net = set()
            for pin in pins:
                site_inst = pin.getSiteInst()
                if site_inst:
                    sites_in_net.add(site_inst.getName())
            degree = len(sites_in_net)
            if degree >= 2:  # Only count nets that connect at least 2 sites
                degrees.append(degree)

        self.max_degree = max(degrees)
        self.avg_degree = sum(degrees) / len(degrees)

    def _estimate_logic_depth(self):
        """
        Estimate the logic depth of the design by analyzing the netlist.
        Uses a heuristic based on net connectivity to estimate critical path depth.
        
        Stores result in self.logic_depth: ratio > 1.0 means deep logic
        """
        try:
            if not self.nets or len(self.nets) == 0:
                self.logic_depth = 1.0
                return
            
            # Count connectivity degree per site
            site_connectivity = {}
            for net in self.nets:
                if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                    continue
                
                pins = net.getPins()
                sites_in_net = set()
                
                for pin in pins:
                    site_inst = pin.getSiteInst()
                    if site_inst:
                        site_name = site_inst.getName()
                        sites_in_net.add(site_name)
                
                # Update connectivity degree
                for site in sites_in_net:
                    site_connectivity[site] = site_connectivity.get(site, 0) + len(sites_in_net)
            
            # Calculate average connectivity degree
            if not site_connectivity:
                self.logic_depth = 1.0
                return
            
            avg_connectivity = sum(site_connectivity.values()) / len(site_connectivity)
            total_sites = len(site_connectivity)
            
            # Heuristic: deep logic has higher connectivity per site and lower site count
            # depth_factor correlates with (avg_connectivity / sqrt(total_sites))
            depth_factor = avg_connectivity / max(1.0, np.sqrt(total_sites))
            
            # Normalize to a reasonable range [0.5, 2.0]
            self.logic_depth = min(2.0, max(0.5, depth_factor / 10.0))
            
            INFO(f"Estimated logic depth factor: {self.logic_depth:.3f} (avg_connectivity: {avg_connectivity:.2f}, sites: {total_sites})")
            
        except Exception as e:
            WARNING(f"Failed to estimate logic depth: {e}, using default value 1.0")
            self.logic_depth = 1.0

    def analyze_solver_hpwl(self, coords, io_coords=None, include_io=False):
        self.hpwl_calculator.clear()
        instance_coords = self.map_coords_to_instance_func(coords, io_coords, include_io)
        for net_name, connected_sites in self.net_to_sites.items():
            self.hpwl_calculator.compute_net_hpwl(net_name, connected_sites, instance_coords, include_io=include_io)
        return self.hpwl_calculator.get_hpwl()

    def save_net_debug_info(self, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.debug_src_root, 'net_debug_info.txt')
        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name\tHPWL\tSite_Count\tSites_Info\n")
            for idx, net in enumerate(self.nets):
                net_name = net.getName()
                hpwl = self.hpwl_calculator.net_hpwl_no_io.get(net_name, 0.0)

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

    def save_solver_hpwl_debug(self, instance_coords, net_to_sites, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.debug_src_root, 'solver_hpwl_debug.txt')
        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name\tHPWL\tInstance_Count\tInstances_Info\n")

            for idx, (net_name, connected_sites) in enumerate(net_to_sites.items()):
                hpwl, _ = self.hpwl_calculator.compute_net_hpwl(net_name, connected_sites, instance_coords)

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

    def analyze_nets(self, optimizable_insts_num=0, available_sites_num=0, fixed_insts_num=0):

        self.net_names = [net.getName() for net in self.nets]
        sites_net_list = []
        valid_net_num = 0

        for net in self.nets:
            net_name = net.getName()
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue

            sites_in_net = set()
            logic_sites = set()
            io_sites = set()

            for pin in net.getPins():
                site_inst = pin.getSiteInst()
                site_name = site_inst.getName()
                site_type = site_inst.getSiteTypeEnum()

                sites_in_net.add(site_name)

                if site_type in SLICE_SITE_ENUM:
                    logic_sites.add(site_name)
                elif site_type in IO_SITE_ENUM:
                    io_sites.add(site_name)

            if len(logic_sites) + len(io_sites) >= 2:
                self.net_to_sites[net_name] = list(logic_sites) + list(io_sites)

                for site_name in sites_in_net:
                    if site_name not in self.site_to_nets:
                        self.site_to_nets[site_name] = []
                    self.site_to_nets[site_name].append(net_name)
                self._record_connectivity(logic_sites, io_sites)
            else:
                # WARNING(f'Net {net_name} skipped, logic: {logic_sites}, io: {io_sites}')
                pass

            if len(logic_sites) >= 2:
                valid_net_num += 1
                sites_net_list.append(logic_sites)

        self._create_net_tensor(valid_net_num, sites_net_list, optimizable_insts_num)
        self._create_net_matrix(optimizable_insts_num, fixed_insts_num)
        self.save_tensor_debug_info(instance_count=optimizable_insts_num)
        INFO(f"Processed {valid_net_num} nets, total {len(self.nets)} nets",
              f" {len(self.site_to_site_connectivity)} site-to-site routes",
              f" {len(self.io_to_site_connectivity)} io-to-site routes",
              f" {len(self.net_to_sites)} inter-tile routes")

        return len(self.site_to_site_connectivity), len(self.nets)

    def _record_connectivity(self, logic_sites: Set[str], io_sites: Set[str]):
        logic_sites_list = list(logic_sites)
        io_sites_list = list(io_sites)

        # 1. IO to logic
        for i in range(len(io_sites_list)):
            for j in range(len(logic_sites_list)):
                io_inst1, inst2 = io_sites_list[i], logic_sites_list[j]

                if io_inst1 not in self.io_to_site_connectivity:
                    self.io_to_site_connectivity[io_inst1] = {}
                if inst2 not in self.io_to_site_connectivity[io_inst1]:
                    self.io_to_site_connectivity[io_inst1][inst2] = 0
                self.io_to_site_connectivity[io_inst1][inst2] += 1

                if inst2 not in self.io_to_site_connectivity:
                    self.io_to_site_connectivity[inst2] = {}
                if io_inst1 not in self.io_to_site_connectivity[inst2]:
                    self.io_to_site_connectivity[inst2][io_inst1] = 0
                self.io_to_site_connectivity[inst2][io_inst1] += 1

        # 2. Logic to logic
        # weight = (len(logic_sites_list) - 1) ** 2

        for i in range(len(logic_sites_list)):
            for j in range(i + 1, len(logic_sites_list)):
                inst1, inst2 = logic_sites_list[i], logic_sites_list[j]

                if inst1 not in self.site_to_site_connectivity:
                    self.site_to_site_connectivity[inst1] = {}
                if inst2 not in self.site_to_site_connectivity[inst1]:
                    self.site_to_site_connectivity[inst1][inst2] = 0
                self.site_to_site_connectivity[inst1][inst2] += 1

                if inst2 not in self.site_to_site_connectivity:
                    self.site_to_site_connectivity[inst2] = {}
                if inst1 not in self.site_to_site_connectivity[inst2]:
                    self.site_to_site_connectivity[inst2][inst1] = 0
                self.site_to_site_connectivity[inst2][inst1] += 1

    def _create_net_tensor(self, valid_net_num, sites_net_list, optimizable_insts_num):
        self.net_tensor = torch.zeros(valid_net_num, optimizable_insts_num, dtype=torch.bool)

        for net_idx, sites in enumerate(sites_net_list):
            site_idx = []
            for site_name in sites:
                instance_idx = self.get_site_inst_id_by_name_func(site_name)
                site_idx.append(instance_idx)
                self.net_tensor[net_idx, instance_idx] = True

        INFO(f"Net tensor shape {self.net_tensor.shape[0]} x {self.net_tensor.shape[1]}")

    def _create_net_matrix(self, optimizable_insts_num, fixed_insts_num):
        n = optimizable_insts_num
        k = fixed_insts_num
        self.insts_matrix = torch.zeros((n, n), device=self.device)

        for source_site, connections in self.site_to_site_connectivity.items():
            source_id = self.get_site_inst_id_by_name_func(source_site)
            if source_id is not None:
                for target_site, connection_count in connections.items():
                    target_id = self.get_site_inst_id_by_name_func(target_site)
                    if target_id is not None:
                        self.insts_matrix[source_id, target_id] += connection_count
                    else:
                        ERROR(f'Cannot find site target id {target_id}')
            else:
                ERROR(f'Cannot find site source id {source_site}')

        self.io_insts_matrix_all = torch.zeros((n + k, n + k), device=self.device)

        for source_site, connections in self.io_to_site_connectivity.items():
            source_id = self.get_site_inst_id_by_name_func(source_site)
            if source_id is not None:
                for target_site, connection_count in connections.items():
                    target_id = self.get_site_inst_id_by_name_func(target_site)
                    if target_id is not None:
                        self.io_insts_matrix_all[source_id, target_id] += connection_count
                        # INFO(f'Connecting IO {source_site} (id {source_id}) to site {target_site} (id {target_id}), count {connection_count}')
                    else:
                        ERROR(f'Cannot find site target id {target_id}')
            else:
                ERROR(f'Cannot find site source id {source_site}')

        self.io_insts_matrix =  self.io_insts_matrix_all[0:n, n:n+k]

        INFO(f'Site matrix {n} x {n}, io site matrix {n + k} x {n + k}')

        original_options = np.get_printoptions()
        # Set print options to show full matrix without truncation
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        matrix_debug_path = os.path.join(self.debug_src_root, 'matrix_debug.txt')
        with open(matrix_debug_path, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            
            print("=" * 50)
            print("insts_matrix ({} x {}):".format(n, n))
            print("=" * 50)
            print(self.insts_matrix.cpu().numpy())
            print("\n")
            
            print("=" * 50)
            print("io_insts_matrix_all ({} x {}):".format(n + k, n + k))
            print("=" * 50)
            print(self.io_insts_matrix_all.cpu().numpy())
            print("\n")
            
            print("=" * 50)
            print("io_insts_matrix ({} x {}):".format(n, k))
            print("=" * 50)
            print(self.io_insts_matrix.cpu().numpy())
            print("\n")
            
            # Restore stdout
            sys.stdout = original_stdout
        
        # Restore original numpy print options
        np.set_printoptions(**original_options)

    def save_tensor_debug_info(self, output_path=None, instance_count=None):
        if output_path is None:
            output_path = os.path.join(self.debug_src_root, 'net_to_slice_sites_tensor_debug.txt')
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

    def get_single_instance_net_hpwl(self, instance_id: int,
                                     coords: Dict[str, Tuple[float, float]],
                                     io_coords: Dict[str, Tuple[float, float]],
                                     include_io: bool = True) -> float:
        instance_hpwl = 0.0
        site_name = self.get_site_inst_name_by_id_func(instance_id)
        net_group = self.site_to_nets[site_name]
        instance_nets_connection = []
        instance_coords = self.map_coords_to_instance_func(coords, io_coords, include_io)
        for net_name in net_group:
            instance_nets_connection.append(self.net_to_sites[net_name])
        for connected_sites in instance_nets_connection:
            hpwl, _ = self.hpwl_calculator.compute_single_instance_hpwl(connected_sites, instance_coords)
            instance_hpwl += hpwl

        return instance_hpwl