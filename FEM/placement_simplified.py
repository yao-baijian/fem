import sys
sys.path.append('.')
from utils import *
import random
import rapidwright

from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import Device, Site, SiteTypeEnum


class FpgaPlacer:

    def __init__(self, utilization_factor = 0.9):

        self.slice_site_enum = [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]
        self.io_site_enum = [SiteTypeEnum.HPIOB, 
                             SiteTypeEnum.HRIO, 
                             SiteTypeEnum.BITSLICE_COMPONENT_RX_TX]

        self.utilization_factor = utilization_factor
        self.len_optimizable_sites = 0

        self.num_of_instance = 0
        self.num_of_sites = 0

        self.available_sites = []
        self.optimizable_sites = []
        self.fixed_sites = []
        self.cells = []
        self.area_info = {}
        self.bbox = {}
        self.unfixed_placements = {}
        self.fixed_placements = {}

        self.site_to_site_connectivity = {}
        self.io_to_site_connectivity = {}
        self.net_to_sites = {}

        self.optimizable_site_inst_to_id = {} 
        self.available_site_to_id = {}    
        self.optimizable_id_to_site_inst = {}
        self.available_id_to_site = {}   

        self.fixed_site_to_id = {}    
        self.fixed_id_to_site = {}  

        pass

    def get_site_inst_id_by_name(self, site_name):
        if site_name in self.optimizable_site_inst_to_id:
            return self.optimizable_site_inst_to_id.get(site_name)
        elif site_name in self.fixed_site_to_id:
            return self.fixed_site_to_id.get(site_name) + self.len_optimizable_sites
        else:
            return None
        
    def get_site_inst_name_by_id(self, id):
        if id in self.optimizable_id_to_site_inst:
            return self.optimizable_id_to_site_inst.get(id)
        elif (id - self.len_optimizable_sites) in self.fixed_id_to_site:
            return self.fixed_id_to_site.get(id - self.len_optimizable_sites)
        else:
            return None

    def get_site_id_by_name(self, site_name):
        return self.available_site_to_id.get(site_name)
    
    def get_site_name_by_id(self, id):
        return self.available_id_to_site.get(id)

    def get_optimizable_sites(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()

            if site_type in self.slice_site_enum:
                self.optimizable_sites.append(site)
            # else:
            #     print(f'site: {site.getName()}, type: {site.getSiteTypeEnum()}')
    
    def get_fixed_sites(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()
            
            if site_type in self.io_site_enum:
                self.fixed_sites.append(site)

    def get_available_target_sites(self, device):
        
        for site in device.getAllSites():
            site_x = site.getInstanceX()
            site_y = site.getInstanceY()

            # print(f'site: {site.getName()}, type: {site.getSiteTypeEnum()}, x: {site_x}, y: {site_y}')
            
            if (self.bbox['start_x'] <= site_x <= self.bbox['end_x'] and 
                self.bbox['start_y'] <= site_y <= self.bbox['end_y'] ):
                
                site_type = site.getSiteTypeEnum()

                if site_type in self.slice_site_enum:
                    self.available_sites.append(site)
        
        if len(self.available_sites) < len(self.optimizable_sites):
            print(f"Warning: available sites({len(self.available_sites)}) less than optimizable sites({len(self.optimizable_sites)})")
            # TODO add sites here

    def map_site_to_id(self):

        # self.optimizable_name_to_id.clear()
        # self.available_name_to_id.clear()
        # self.optimizable_id_to_name.clear()
        # self.available_id_to_name.clear()

        for idx, site_inst in enumerate(self.optimizable_sites): 
            site_name = site_inst.getName()
            self.optimizable_site_inst_to_id[site_name] = idx
            self.optimizable_id_to_site_inst[idx] = site_name
        
        self.len_optimizable_sites = len(self.optimizable_sites)

        for idx, site_inst in enumerate(self.fixed_sites): 
            site_name = site_inst.getName()
            self.fixed_site_to_id[site_name] = idx
            self.fixed_id_to_site[idx] = site_name
        
        for idx, site in enumerate(self.available_sites):
            site_name = site.getName()
            self.available_site_to_id[site_name] = idx
            self.available_id_to_site[idx] = site_name

    def estimate_place_areas(self, design):
        
        total_site_insts = len(design.getSiteInsts())
        other_sites = total_site_insts - (len(self.optimizable_sites) + len(self.fixed_sites))

        print(f"Estimate Sites Number: {len(self.optimizable_sites)} slice sites, {len(self.fixed_sites)} fixed sites, {other_sites} other sites, total {total_site_insts} sites.")
        
        side_length = int(np.ceil(np.sqrt(total_site_insts / self.utilization_factor)))
        side_width = 0
        
        min_x = min(site.getInstanceX() for site in self.available_slices_ml)
        max_x = max(site.getInstanceX() for site in self.available_slices_ml)
        min_y = min(site.getInstanceY() for site in self.available_slices_ml)
        max_y = max(site.getInstanceY() for site in self.available_slices_ml)

        device_center_x = (min_x + max_x) // 2
        device_center_y = (min_y + max_y) // 2
        
        start_x = device_center_x - side_length // 2
        end_x = device_center_x + side_length // 2
        start_y = device_center_y - side_length // 2
        end_y = device_center_y + side_length // 2
        
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(max_x, end_x)
        end_y = min(max_y, end_y)
        
        self.bbox = {
            'start_x': start_x,
            'end_x': end_x,
            'start_y': start_y,
            'end_y': end_y,
            'estimated_sites': total_site_insts,
            'area_length': side_length,
            'area_width': side_width,
            'area_size': side_length * side_length,
            'utilization': total_site_insts / (side_length * side_length) if side_length > 0 else 0
        }
        
        print(f"Estimate Area: {side_length} x {side_length} , start=({start_x},{start_y}), end=({end_x},{end_y})")

    def random_initial_placement(self, design):
    
        design.unplaceDesign()

        self.num_of_instance = len(self.optimizable_sites)
        self.num_of_sites = len(self.available_sites)
        
        print(f"INFO: number of optimizable instance: {self.num_of_instance}, number of available sites: {self.num_of_sites}")

        random.shuffle(self.available_sites)

        placed_count = 0
        for i, site in enumerate(self.optimizable_sites):
            if i < len(self.available_sites):
                target_site = self.available_sites[i]
                
                if self.is_site_compatible(site, target_site):
                    # self.place_site(site, target_site)
                    
                    self.unfixed_placements[site.getName()] = {
                        'target_site': target_site,
                        'source_site': site,
                        'target_x': target_site.getInstanceX(),
                        'target_y': target_site.getInstanceY(),
                        'bel': self.suggest_bel_for_site(site, target_site)
                    }
                    placed_count += 1
                else:
                    self.unfixed_placements[site.getName()] = {
                        'target_site': None,
                        'source_site': None,
                        'target_x': -1,
                        'target_y': -1,
                        'bel': None,
                    }
                    print(f"ERROR: site {site.getName()}({site.getSiteTypeEnum().name}) is not compatible {target_site.getName()}({target_site.getSiteTypeEnum().name})")
            else:
                self.unfixed_placements[site.getName()] = {
                    'target_site': None,
                    'source_site': None,
                    'target_x': -1,
                    'target_y': -1,
                    'bel': None,
                }
                print(f"ERROR: No more site for {site.getName()}")

        # Fixed sites
        for fixed_site in self.fixed_sites:
            self.fixed_placements[fixed_site.getName()] = {
                'target_site': None,
                'target_site': fixed_site,
                'target_x': fixed_site.getInstanceX(),
                'target_y': fixed_site.getInstanceY(),
                'bel': None,
            }

        print(f"Initial Placement: {placed_count}/{len(self.optimizable_sites)} for total {len(self.cells)} sites done, {len(self.fixed_placements)} remains.")

    def is_site_compatible(self, source_site, target_site):
        source_type = source_site.getSiteTypeEnum()
        target_type = target_site.getSiteTypeEnum()
        
        if source_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]:
            return target_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]
        
        if source_type in [SiteTypeEnum.IOB33, SiteTypeEnum.IOB33M]:
            return target_type in [SiteTypeEnum.IOB33, SiteTypeEnum.IOB33M]
        
        if source_type in [SiteTypeEnum.IOB18, SiteTypeEnum.IOB18M]:
            return target_type in [SiteTypeEnum.IOB18, SiteTypeEnum.IOB18M]
        
        return source_type == target_type

    # TODO need decide whether this is needed
    def suggest_bel_for_site(self, source_site, target_site):
        site_type = target_site.getSiteTypeEnum()
        
        if site_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]:
            available_bels = ['A6LUT', 'B6LUT', 'C6LUT', 'D6LUT']
            return available_bels[hash(source_site.getName()) % len(available_bels)]
        
        elif site_type in [SiteTypeEnum.IOB33, SiteTypeEnum.IOB33M, 
                        SiteTypeEnum.IOB18, SiteTypeEnum.IOB18M]:
            return 'IOB'
        
        return 'UNKNOWN'
  
    def record_site_connectivity(self, design):

        for net in design.getNets():
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue
                
            net_name = net.getName()

            logic_sites_in_net = set()  # 逻辑站点 (SLICE, DSP, BRAM等)
            io_sites_in_net = set()     # IO 站点 (IOB, HPIOB等)

            for pin in net.getPins():
                site_inst = pin.getSiteInst()
                site_inst_type = site_inst.getSiteTypeEnum()
                # Slice site - Slice site
                if site_inst_type in self.slice_site_enum:
                    logic_site_name = site_inst.getName()
                    logic_sites_in_net.add(logic_site_name)
                else:
                    # IO site - IO site | IO site - Slice site
                    io_site_name = site_inst.getName()
                    io_sites_in_net.add(io_site_name)

            total_site_num = len(io_sites_in_net) + len(logic_sites_in_net)
            if total_site_num >= 2:
                self.net_to_sites[net_name] = list(io_sites_in_net) + list(logic_sites_in_net)

                io_sites_list = list(io_sites_in_net)
                logic_sites_list = list(logic_sites_in_net)

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
        
        print(f"Process Network: total {len(self.site_to_site_connectivity)} site to site connections, {len(self.io_to_site_connectivity)} io to site connections, {len(self.net_to_sites)} networks processed.")

    def init_placement(self, dcp_file = '', edf_file='', dcp_output=''):

        design = Design.readCheckpoint(dcp_file)
        device = design.getDevice()

        self.available_slices_ml = list(device.getAllCompatibleSites(SiteTypeEnum.SLICEL))
        self.available_slices_ml.extend(list(device.getAllCompatibleSites(SiteTypeEnum.SLICEM)))

        self.cells = design.getCells()

        self.record_site_connectivity(design)
        self.get_optimizable_sites(design)      # SLICEL
        self.get_fixed_sites(design)            # HPIOB, BITSLICE_COMPONENT_RX_TX
        self.estimate_place_areas(design)
        self.get_available_target_sites(device)

        self.map_site_to_id()
        self.random_initial_placement(design)

