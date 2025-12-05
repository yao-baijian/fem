from .utils import *
import random
import rapidwright
import torch
from enum import Enum
import numpy as np

class PlaceType(Enum):
    CENTERED = 1
    IO = 2
    OTHER = 3

from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import Device, Site, SiteTypeEnum
from com.xilinx.rapidwright.rwroute import RWRoute

class FpgaPlacer:

    def __init__(self, utilization_factor = 0.95):

        self.slice_site_enum = [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]

        self.io_site_enum = [SiteTypeEnum.HPIOB, 
                             SiteTypeEnum.HRIO, 
                             SiteTypeEnum.BITSLICE_COMPONENT_RX_TX,
                             SiteTypeEnum.BUFGCE]
        
        self.clock_site_enum = [SiteTypeEnum.BUFGCE] # NOT USED NOW

        self.utilization_factor = utilization_factor
        self.num_optimizable_insts = 0

        self.num_of_instance = 0
        self.num_of_sites = 0

        self.available_sites = []
        self.optimizable_insts = []
        self.fixed_insts = []
        self.clock_insts = []

        self.cells = []
        self.bbox = {}
        self.io_area = {}
        self.clock_area = {}

        self.unfixed_placements = {}
        self.fixed_placements = {}

        self.site_to_site_connectivity = {}
        self.io_to_site_connectivity = {}
        self.net_to_sites = {}
        self.net_to_slice_sites_tensor = None

        self.optimizable_site_inst_to_id = {} 
        self.available_site_to_id = {}    
        self.optimizable_id_to_site_inst = {}
        self.available_id_to_site = {}   

        self.fixed_site_to_id = {}    
        self.fixed_id_to_site = {}  

        self.nets = None

        self.total_hpwl = 0.0
        self.net_hpwl = {}
        self.net_bbox = {}

        self.total_hpwl_no_io = 0.0
        self.net_hpwl_no_io = {}
        self.net_bbox_no_io = {}

        self.place_orientation = PlaceType.IO

        pass

    def get_site_inst_id_by_name(self, site_name):
        if site_name in self.optimizable_site_inst_to_id:
            return self.optimizable_site_inst_to_id.get(site_name)
        elif site_name in self.fixed_site_to_id:
            return self.fixed_site_to_id.get(site_name) + self.num_optimizable_insts
        else:
            return None
        
    def get_site_inst_name_by_id(self, id):
        if id in self.optimizable_id_to_site_inst:
            return self.optimizable_id_to_site_inst.get(id)
        elif (id - self.num_optimizable_insts) in self.fixed_id_to_site:
            return self.fixed_id_to_site.get(id - self.num_optimizable_insts)
        else:
            return None

    def get_site_id_by_name(self, site_name):
        return self.available_site_to_id.get(site_name)
    
    def get_site_name_by_id(self, id):
        return self.available_id_to_site.get(id)

    def get_optimizable_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()

            if site_type in self.slice_site_enum:
                self.optimizable_insts.append(site)
    
    def get_fixed_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()
            
            if site_type in self.io_site_enum:
                self.fixed_insts.append(site)

    def get_clock_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()
            
            if site_type in self.clock_site_enum:
                # self.fixed_insts.append(site)
                continue

            elif site not in self.optimizable_insts and site not in self.fixed_insts:
                print(f'Warning: cannot orient site type, site: {site.getName()}, type: {site.getSiteTypeEnum()}')

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
        
        if len(self.available_sites) < len(self.optimizable_insts):
            print(f"Warning: available sites({len(self.available_sites)}) less than optimizable sites({len(self.optimizable_insts)})")
            # TODO add sites here

    def map_site_to_id(self):

        for idx, site_inst in enumerate(self.optimizable_insts): 
            site_name = site_inst.getName()
            self.optimizable_site_inst_to_id[site_name] = idx
            self.optimizable_id_to_site_inst[idx] = site_name
        
        self.num_optimizable_insts = len(self.optimizable_insts)

        for idx, site_inst in enumerate(self.fixed_insts): 
            site_name = site_inst.getName()
            self.fixed_site_to_id[site_name] = idx
            self.fixed_id_to_site[idx] = site_name
        
        for idx, site in enumerate(self.available_sites):
            site_name = site.getName()
            self.available_site_to_id[site_name] = idx
            self.available_id_to_site[idx] = site_name

    def init_place_areas(self, design):
        
        total_site_insts = len(design.getSiteInsts())
        other_sites = total_site_insts - (len(self.optimizable_insts) + len(self.fixed_insts))

        print(f"INFO, estimate sites number: {len(self.optimizable_insts)} slice sites, {len(self.fixed_insts)} fixed sites, {other_sites} other sites, total {total_site_insts} sites.")
        
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
        
        print(f"INFO: estimate area {side_length} x {side_length} , start=({start_x},{start_y}), end=({end_x},{end_y})")

    def _init_io_area(self):
        num_pins = len(self.fixed_insts)
        
        io_width = 1
        io_height = num_pins
        
        io_start_x = self.bbox['start_x'] - io_width - 2
        io_end_x = io_start_x + io_width
        
        bbox_center_y = ( self.bbox['start_y'] + self.bbox['end_y'] ) // 2
        io_start_y = bbox_center_y - io_height // 2
        io_end_y = io_start_y + io_height
        
        self.io_area = {
            'type': 'left_io',
            'start_x': io_start_x,
            'end_x': io_end_x,
            'start_y': io_start_y,
            'end_y': io_end_y,
            'width': io_width,
            'height': io_height,
            'center_y': (io_start_y + io_end_y) // 2,
            'num_pins': num_pins
        }
        
        print(f"INFO: Left IO area - position: ({io_start_x}, {io_start_y}) to ({io_end_x}, {io_end_y})")

    def _init_clock_buffer_area(self):
        num_clk_buf = len(self.clock_insts)
        
        clock_width = 1
        clock_height = num_clk_buf
        
        clock_start_x = self.bbox['start_x'] - 1
        clock_end_x = clock_start_x + clock_width
        
        bbox_center_y = ( self.bbox['start_y'] + self.bbox['end_y'] ) // 2
        clock_start_y = bbox_center_y - clock_height // 2
        clock_end_y = clock_start_y + clock_height
        
        self.clock_area = {
            'type': 'clock_buffers',
            'start_x': clock_start_x,
            'end_x': clock_end_x,
            'start_y': clock_start_y,
            'end_y': clock_end_y,
            'width': clock_width,
            'height': clock_height,
            'center_y': (clock_start_y + clock_end_y) // 2,
            'num_pins': num_clk_buf,
        }
        
        print(f"INFO: Clock buffer area - position: ({clock_start_x}, {clock_start_y}) to ({clock_end_x}, {clock_end_y})")

    def random_initial_placement(self, design):
    
        design.unplaceDesign()

        self.num_of_instance = len(self.optimizable_insts)
        self.num_of_sites = len(self.available_sites)
        
        print(f"INFO: number of optimizable instance: {self.num_of_instance}, number of available sites: {self.num_of_sites}")

        random.shuffle(self.available_sites)

        placed_count = 0
        for i, site in enumerate(self.optimizable_insts):
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
        for fixed_site in self.fixed_insts:
            self.fixed_placements[fixed_site.getName()] = {
                'target_site': None,
                'target_site': fixed_site,
                'target_x': fixed_site.getInstanceX(),
                'target_y': fixed_site.getInstanceY(),
                'bel': None,
            }

        print(f"INFO: initial placement {placed_count} for total {len(self.cells)} sites done, {len(self.fixed_placements)} remains.")

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
  
    def record_site_connectivity(self):
        sites_net_num = 0
        sites_net_list = []
        for net in self.nets:
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue
                
            net_name = net.getName()
            logic_sites_in_net = set()  # 逻辑站点 (SLICE, DSP, BRAM等)
            io_sites_in_net = set()     # IO 站点 (IOB, HPIOB等)

            for pin in net.getPins():
                site_inst = pin.getSiteInst()
                site_inst_type = site_inst.getSiteTypeEnum()
                # print(f'Net: {net_name}, Pin Site: {site_inst.getName()}, Type: {site_inst_type}')
                # Slice site - Slice site
                if site_inst_type in self.slice_site_enum:
                    logic_site_name = site_inst.getName()
                    logic_sites_in_net.add(logic_site_name)
                else:
                    # IO site - IO site | IO site - Slice site | Clock buffer - Slice site
                    io_site_name = site_inst.getName()
                    io_sites_in_net.add(io_site_name)

            total_site_num = len(io_sites_in_net) + len(logic_sites_in_net)

            if total_site_num >= 2:
                # print(f'INFO: net {net_name}, io sites: {io_sites_in_net}, logic sites: {logic_sites_in_net}')
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
            # else:
            #     print(f"Warning: net {net_name}, skip.")

            if len(logic_sites_in_net) >= 2:
                sites_net_num += 1
                sites_net_list.append(logic_sites_in_net)

        self.net_to_slice_sites_tensor = torch.zeros(sites_net_num, len(self.optimizable_insts), dtype=torch.bool)

        for net_idx, sites in enumerate(sites_net_list):
            for site_name in sites:
                instance_idx = self.get_site_inst_id_by_name(site_name)
                self.net_to_slice_sites_tensor[net_idx, instance_idx] = True

        print(f"INFO: process {len(self.nets)} network, total {len(self.site_to_site_connectivity)} site to site routes, \n \
            {len(self.io_to_site_connectivity)} io to site routes, \n \
            {len(self.net_to_sites)} inter-tile routes.")

    def init_placement(self, dcp_file = '', edf_file='', dcp_output=''):

        design = Design.readCheckpoint(dcp_file)
        # routing report
        # design.unrouteDesign()

        # timing_driven = False
        # if timing_driven:
        #     routed_design = RWRoute.routeDesignFullTimingDriven(design)
        # else:
        #     routed_design = RWRoute.routeDesignFullNonTimingDriven(design)

        device = design.getDevice()

        self.available_slices_ml = list(device.getAllCompatibleSites(SiteTypeEnum.SLICEL))
        self.available_slices_ml.extend(list(device.getAllCompatibleSites(SiteTypeEnum.SLICEM)))

        self.cells = design.getCells()

        self.estimate_hpwl(design)

        print(f"INFO: nets number: {len(self.nets)}, total hpwl: {self.total_hpwl:.2f}, without io: {self.total_hpwl_no_io:.2f}")
        
        self.get_optimizable_insts(design)      # SLICEL
        self.get_fixed_insts(design)            # HPIOB, BITSLICE_COMPONENT_RX_TX
        self.get_clock_insts(design)            # BUFGCE
        self.init_place_areas(design)
        self._init_io_area()
        self._init_clock_buffer_area()
        self.get_available_target_sites(device)
        self.map_site_to_id()
        self.record_site_connectivity()
        self.random_initial_placement(design)

    def estimate_hpwl(self, design):
        self.total_hpwl = 0.0
        self.net_hpwl.clear()
        self.net_bbox.clear()

        self.total_hpwl_no_io = 0.0
        self.net_hpwl_no_io.clear()
        self.net_bbox_no_io.clear()
        
        self.nets = design.getNets()
        
        for net in self.nets:
            net_name = net.getName()
            hpwl, bbox = self._calculate_net_hpwl_rapidwright(net, True)
            self.net_hpwl[net_name] = hpwl
            self.net_bbox[net_name] = bbox
            self.total_hpwl += hpwl

        for net in self.nets:
            net_name = net.getName()
            hpwl, bbox = self._calculate_net_hpwl_rapidwright(net, False)
            self.net_hpwl_no_io[net_name] = hpwl
            self.net_bbox_no_io[net_name] = bbox
            self.total_hpwl_no_io += hpwl
        
        return self.total_hpwl, self.total_hpwl_no_io

    def _calculate_net_hpwl_rapidwright(self, net, include_io=True):

        # if net.isClockNet():
        #     print(f'{net.getName()} is a clock net')
        #     for pin in net.getPins():
        #         print(f'  pin: {pin.getName()}, site: {pin.getSiteInst().getName()}, type: {pin.getSiteInst().getSiteTypeEnum()}')

        if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
            return 0.0, {}
        
        pins = net.getPins()
        if len(pins) < 2:
            return 0.0, {}
        
        coordinates = []
        for pin in pins:

            if not include_io and pin.getSiteInst().getSiteTypeEnum() in self.io_site_enum:
                # print("pin: ", pin.getName(), " type: ", pin.getSiteInst().getSiteTypeEnum())
                continue
    
            tile = pin.getTile()
            col = tile.getColumn()
            row = tile.getRow()
            # print(f"pin: {pin.getName()}, site: {pin.getSiteInst().getName()}, site x: {pin.getSiteInst().getInstanceX()}, site y: {pin.getSiteInst().getInstanceY()} tile: {tile.getName()}, col: {col}, row: {row}")
            coordinates.append((col, row))

        if len(coordinates) < 2:
            return 0.0, {}

        return self._calculate_hpwl_from_coordinates(coordinates)
    
    def _calculate_hpwl_from_coordinates(self, coordinates):
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

    def estimate_solver_hpwl(self, coords, io_coords, include_io=True):
        self.total_hpwl = 0.0
        self.net_hpwl.clear()
        self.net_bbox.clear()
        
        instance_coords = {}
        for instance_id in range(len(coords)):
            site_name = self.get_site_inst_name_by_id(instance_id)
            instance_coords[site_name] = coords[instance_id]

        if include_io:
            for instance_id in range(len(io_coords)):
                site_name = self.get_site_inst_name_by_id(instance_id + self.num_optimizable_insts)
                instance_coords[site_name] = io_coords[instance_id]
        
        skipped = 0
        for net_name, connected_sites in self.net_to_sites.items():
            hpwl, bbox = self._calculate_net_hpwl_from_instance_coords(net_name, connected_sites, instance_coords)

            if (hpwl == 0.0):
                skipped += 1
                continue

            self.net_hpwl[net_name] = hpwl
            self.net_bbox[net_name] = bbox
            self.total_hpwl += hpwl
        
        print(f"INFO: total {len(self.net_to_sites)} nets, skipped {skipped} nets.")
        
        return self.total_hpwl
    
    def _calculate_net_hpwl_from_instance_coords(self, net_name, connected_sites, instance_coords, include_io=True):
        coordinates = []
        # print(f'INFO: connected sites: {connected_sites}')
        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])
            else:
                # print(f"Warning: site {site_name} not found in instance coordinates for net {net_name}")
                pass
        
        if len(coordinates) < 2:
            return 0.0, {}
        
        return self._calculate_hpwl_from_coordinates(coordinates)
    
    # def _calculate_net_hpwl_from_instance_coords(self, net_name, connected_sites, instance_coords):
