import sys
sys.path.append('.')
import numpy as np
import random
import io
import sys
import torch
import rapidwright
from contextlib import contextmanager
from fem_placer.net import NetManager
from fem_placer.grid import Grid
from fem_placer.config import *
from fem_placer.logger import INFO, WARNING, ERROR

from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import Device, Site, SiteTypeEnum
from com.xilinx.rapidwright.rwroute import RWRoute



class FpgaPlacer:

    def __init__(self, 
                 place_orientation = PlaceType.CENTERED, 
                 grid_type = GridType.SQUARE,
                 utilization_factor = 0.3,
                 debug = True,
                 device = 'cpu'):
        
        self.total_insts_num = 0
        self.opti_insts_num = 0
        self.fixed_insts_num = 0
        self.other_insts_num = 0
        self.avail_sites_num = 0

        self.optimizable_insts = []
        self.fixed_insts = []
        self.clock_insts = []
        self.available_sites = []

        self.cells = []
        
        self.grids = {
            'logic': Grid(name='logic', device=device),
            'io': Grid(name='io', device=device),
            'clock': Grid(name='clock', device=device)
        }

        self.unfixed_placements = {}
        self.fixed_placements = {}
        self.optimizable_site_inst_to_id = {} 
        self.available_site_to_id = {}    
        self.optimizable_id_to_site_inst = {}
        self.available_id_to_site = {}   
        
        self.logic_ids = None
        self.io_ids = None

        self.fixed_site_to_id = {}    
        self.fixed_id_to_site = {}
        self.place_orientation = place_orientation
        self.grid_type = grid_type
        self.utilization_factor = utilization_factor
        self.debug = debug
        self.net_manager = NetManager(self.get_site_inst_id_by_name, 
                                      self.get_site_inst_name_by_id,
                                      self.map_coords_to_instance,
                                      debug=self.debug,
                                      device=device)
        
        
        self.logic_site_coords = None
        self.site_coords_all = None
        self.device = device
        self.constraint_alpha = 0
        self.constraint_beta = 0
        pass
    
    def with_io(self):
        if self.place_orientation == PlaceType.IO:
            return True
        return False
    
    def set_alpha(self, alpha):
        self.constraint_alpha = alpha
        
    def set_beta(self, beta):
        self.constraint_beta = beta
    
    def get_grid(self, grid_name) -> Grid:
        return self.grids[grid_name]
    
    @contextmanager
    def suppress_rapidwright_output(self):
        original_py_out = sys.stdout
        original_py_err = sys.stderr
        
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            from java.lang import System
            from java.io import PrintStream
            from java.io import ByteArrayOutputStream
            
            original_java_out = System.out
            original_java_err = System.err
            
            baos = ByteArrayOutputStream()
            null_stream = PrintStream(baos)
            
            System.setOut(null_stream)
            System.setErr(null_stream)
            
            yield
            
        finally:
            sys.stdout = original_py_out
            sys.stderr = original_py_err
            
            try:
                System.setOut(original_java_out)
                System.setErr(original_java_err)
            except NameError:
                pass
        
    def get_site_inst_id_by_name(self, site_name):
        if site_name in self.optimizable_site_inst_to_id:
            return self.optimizable_site_inst_to_id.get(site_name)
        elif site_name in self.fixed_site_to_id:
            return self.fixed_site_to_id.get(site_name) + self.opti_insts_num
        else:
            WARNING(f"Cannot find site_inst id for site_name: {site_name}")
            return None
        
    def get_site_inst_name_by_id(self, id):
        if id in self.optimizable_id_to_site_inst:
            return self.optimizable_id_to_site_inst.get(id)
        elif (id - self.opti_insts_num) in self.fixed_id_to_site:
            return self.fixed_id_to_site.get(id - self.opti_insts_num)
        else:
            WARNING(f"Cannot find site_inst name for id: {id}")
            return None

    def get_site_id_by_name(self, site_name):
        return self.available_site_to_id.get(site_name)
    
    def get_site_name_by_id(self, id):
        return self.available_id_to_site.get(id)

    def get_optimizable_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()

            if site_type in SLICE_SITE_ENUM:
                self.optimizable_insts.append(site)
    
    def get_fixed_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()
            
            if site_type in IO_SITE_ENUM:
                self.fixed_insts.append(site)

    def get_other_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()
            
            if site_type in OTHER_SITE_ENUM:
                # self.fixed_insts.append(site)
                continue

            elif site not in self.optimizable_insts and site not in self.fixed_insts:
                WARNING(f'Cannot orient site type, site: {site.getName()}, type: {site.getSiteTypeEnum()}')

    def get_available_target_sites(self, device):
        
        for site in device.getAllSites():
            site_x = site.getInstanceX()
            site_y = site.getInstanceY()

            # print(f'site: {site.getName()}, type: {site.getSiteTypeEnum()}, x: {site_x}, y: {site_y}')
            
            if (self.grids['logic'].start_x <= site_x <= self.grids['logic'].end_x and 
                self.grids['logic'].start_y <= site_y <= self.grids['logic'].end_y ):
                
                site_type = site.getSiteTypeEnum()

                if site_type in SLICE_SITE_ENUM:
                    self.available_sites.append(site)
        
        if len(self.available_sites) < len(self.optimizable_insts):
            WARNING(f"Available sites({len(self.available_sites)}) less than optimizable sites({len(self.optimizable_insts)})")
            # TODO add sites here
            
        self.avail_sites_num = len(self.available_sites)
        
    def get_ids(self):
        return self.logic_ids, self.io_ids

    def _map_site_to_id(self):

        for idx, site_inst in enumerate(self.optimizable_insts): 
            site_name = site_inst.getName()
            self.optimizable_site_inst_to_id[site_name] = idx
            self.optimizable_id_to_site_inst[idx] = site_name
        
        self.logic_ids = torch.arange(self.opti_insts_num, device=self.device)

        for idx, site_inst in enumerate(self.fixed_insts): 
            site_name = site_inst.getName()
            self.fixed_site_to_id[site_name] = idx
            self.fixed_id_to_site[idx] = site_name
            
        self.io_ids = torch.arange(self.opti_insts_num, self.opti_insts_num + self.fixed_insts_num, device=self.device)
        
        for idx, site in enumerate(self.available_sites):
            site_name = site.getName()
            self.available_site_to_id[site_name] = idx
            self.available_id_to_site[idx] = site_name

        with open('result/optimizable_site_mapping_debug.tsv', 'w') as f:
            f.write("Type\tSiteInst_Name\tID\n")
            for idx in range(len(self.optimizable_insts)):
                site_name = self.optimizable_id_to_site_inst[idx]
                f.write(f"Optimizable\t{site_name}\t{idx}\n")
            f.write(f"TOTAL\t{len(self.optimizable_insts)}\n\n")
            
        with open('result/fixed_site_mapping_debug.tsv', 'w') as f:
            f.write("Type\tSiteInst_Name\tID\n")
            for idx in range(len(self.fixed_insts)):
                site_name = self.fixed_id_to_site[idx]
                f.write(f"Fixed\t{site_name}\t{idx}\n")
            f.write(f"TOTAL\t{len(self.fixed_insts)}\n\n")

    def _init_place_areas(self, design):
        self.total_insts_num = len(design.getSiteInsts())
        self.opti_insts_num = len(self.optimizable_insts)
        self.fixed_insts_num = len(self.fixed_insts)
        self.other_insts_num = self.total_insts_num - (self.opti_insts_num + self.fixed_insts_num)
        
        self.constraint_alpha = self.opti_insts_num / 2

        INFO(f"Sites stat: {self.opti_insts_num} slice sites, {self.fixed_insts_num} fixed sites, {self.other_insts_num} other sites, total {self.total_insts_num} sites.")
        
        if self.grid_type == GridType.SQUARE:
            area_length = int(np.ceil(np.sqrt(self.opti_insts_num / self.utilization_factor)))
            area_height = area_length

            # area_length = 8
            # area_height = 100
        
        elif self.grid_type == GridType.RECTAN:
            # Rectangular grid: consider logic depth from net analysis
            base_area = self.opti_insts_num / self.utilization_factor
            logic_depth = self.net_manager.logic_depth
            
            # Adjust aspect ratio based on logic depth:
            # - logic_depth > 1.0: deeper logic -> longer length, shorter width
            # - logic_depth < 1.0: shallower logic -> shorter length, wider width
            # Use square root of logic depth for aspect ratio adjustment
            aspect_ratio = np.sqrt(logic_depth)
            
            # Solve: length * width = base_area, length / width = aspect_ratio
            # => length = sqrt(base_area * aspect_ratio), width = sqrt(base_area / aspect_ratio)
            area_length = int(np.ceil(np.sqrt(base_area * aspect_ratio)))
            area_height = int(np.ceil(np.sqrt(base_area / aspect_ratio)))
            
            INFO(f"Using RECT grid: logic_depth_factor={logic_depth:.3f}, aspect_ratio={aspect_ratio:.3f}")
        
        else:
            # Default fallback
            area_length = int(np.ceil(np.sqrt(self.opti_insts_num / self.utilization_factor)))
            area_height = 0
        
        start_x = 0
        end_x = start_x + area_length
        start_y = 0 - area_height // 2
        end_y = start_y + area_height
        
        logic_grid = self.grids['logic']
        logic_grid.start_x = start_x
        logic_grid.start_y = start_y
        logic_grid.area_length = area_length
        logic_grid.area_width = area_height
        logic_grid.__post_init__()
        
        utilization = self.opti_insts_num / (area_length * area_height)
        
        INFO(f"Estimate area {area_length} x {area_height} , start=({start_x}, {start_y}), end=({end_x}, {end_y}), utilization {utilization:.3f}")

    def _init_io_area(self):
        num_pins = len(self.fixed_insts)
        
        io_length = 1
        io_width = num_pins + num_pins // 15
        
        io_start_x = self.grids['logic'].start_x - io_length       
        bbox_center_y = self.grids['logic'].center_y
        io_start_y = bbox_center_y - io_width // 2

        io_grid = self.grids['io']
        io_grid.start_x = io_start_x
        io_grid.start_y = io_start_y
        io_grid.area_length = io_length
        io_grid.area_width = io_width
        io_grid.__post_init__()
        
        INFO(f"Left IO area - position: ({io_start_x}, {io_start_y}) to ({io_grid.end_x}, {io_grid.end_y})")

    def _init_clock_buffer_area(self):
        num_clk_buf = len(self.clock_insts)
        clock_length = 1
        clock_height = num_clk_buf
        
        clock_start_x = self.grids['logic'].start_x
        bbox_center_y = self.grids['logic'].center_y
        clock_start_y = bbox_center_y - clock_height // 2
        
        clock_grid = self.grids['clock']
        clock_grid.start_x = clock_start_x
        clock_grid.start_y = clock_start_y
        clock_grid.area_length = clock_length
        clock_grid.area_width = clock_height
        clock_grid.__post_init__()
        
        INFO(f"Clock buffer area - position: ({clock_start_x}, {clock_start_y}) to ({clock_grid.end_x}, {clock_grid.end_y})")

    def random_initial_placement(self, design):
    
        design.unplaceDesign()
        
        INFO(f"Optimizable instances num: {self.opti_insts_num}, available sites num: {self.avail_sites_num}")

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
                        'target_y': target_site.getInstanceY()
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
                    ERROR(f"Site {site.getName()}({site.getSiteTypeEnum().name}) is not compatible {target_site.getName()}({target_site.getSiteTypeEnum().name})")
            else:
                self.unfixed_placements[site.getName()] = {
                    'target_site': None,
                    'source_site': None,
                    'target_x': -1,
                    'target_y': -1,
                    'bel': None,
                }
                ERROR(f"No more site for {site.getName()}")

        # Fixed sites
        for fixed_site in self.fixed_insts:
            self.fixed_placements[fixed_site.getName()] = {
                'target_site': None,
                'target_site': fixed_site,
                'target_x': fixed_site.getInstanceX(),
                'target_y': fixed_site.getInstanceY(),
                'bel': None,
            }

        INFO(f"Initial placement {placed_count} for total {len(self.cells)} sites done, {len(self.fixed_placements)} remains.")

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
  
    def init_placement(self, dcp_file = '', edf_file='', dcp_output=''):
        # Design.setAutoGenerateReadableEdif (false)
        with self.suppress_rapidwright_output():
            design = Design.readCheckpoint(dcp_file)
        INFO(f"Reading DCP: {dcp_file} success")
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
        _, vivado_hpwl = self.net_manager.analyze_design_hpwl(design)
        self.get_optimizable_insts(design) 
        self.get_fixed_insts(design)         
        self.get_other_insts(design)      
        self._init_place_areas(design)
        self._init_io_area()
        self._init_clock_buffer_area()
        self.get_available_target_sites(device)
        self._map_site_to_id()
        site_net_num, total_net_num = self.net_manager.analyze_nets(self.opti_insts_num, self.avail_sites_num, self.fixed_insts_num)
        self.random_initial_placement(design)
        
        self._get_place_area_coords()
        self._get_io_area_coords()
        self._get_combined_coords()

        return vivado_hpwl, self.opti_insts_num, site_net_num, total_net_num

    def map_coords_to_instance(self, coords, io_coords=None, include_io=False):
        instance_coords = {}
        
        for instance_id in range(len(coords)):
            site_name = self.get_site_inst_name_by_id(instance_id)
            instance_coords[site_name] = coords[instance_id]

        if include_io:
            for instance_id in range(len(io_coords)):
                site_name = self.get_site_inst_name_by_id(instance_id + self.opti_insts_num)
                instance_coords[site_name] = io_coords[instance_id]
        
        return instance_coords
        
    def _get_place_area_coords(self):
        place_length = self.grids['logic'].area_length
        place_width = self.grids['logic'].area_width
        
        self.logic_site_coords = self.grids['logic'].to_real_coords_tensor(torch.cartesian_prod(
            torch.arange(place_length, dtype=torch.float32, device=self.device),
            torch.arange(place_width, dtype=torch.float32, device=self.device)
        ))

    # def get_site_coords_all(num_locations, area_width):
    #     indices = torch.arange(num_locations, dtype=torch.float32, device='cuda')
    #     x_coords = indices % area_width
    #     y_coords = indices // area_width
    #     return torch.stack([x_coords, y_coords], dim=1)

    def _get_io_area_coords(self):
        place_width = self.grids['io'].area_width 

        self.io_site_coords = self.grids['io'].to_real_coords_tensor(torch.cartesian_prod(
            torch.tensor([0], dtype=torch.float32, device=self.device),
            torch.arange(place_width, dtype=torch.float32, device=self.device) 
        ))
        
    def _get_combined_coords(self):
        logic_coords = self.logic_site_coords.clone()
        adjusted_io_coords = self.io_site_coords.clone()
        
        place_height = self.grids['logic'].area_width
        io_height = self.fixed_insts_num 
    
        logic_center_y = (place_height - 1) / 2.0
        io_center_y = (io_height - 1) / 2.0 
        
        y_offset = logic_center_y - io_center_y

        adjusted_io_coords[:, 0] = -1.0 
        adjusted_io_coords[:, 1] += y_offset
        
        self.site_coords_all = torch.cat([logic_coords, adjusted_io_coords], dim=0)
        
    def place(self, coords, io_coords=None, include_io=False):
        # TODO replace the legalized logic into FPGA
        min_x = min(site.getInstanceX() for site in self.available_slices_ml)
        max_x = max(site.getInstanceX() for site in self.available_slices_ml)
        min_y = min(site.getInstanceY() for site in self.available_slices_ml)
        max_y = max(site.getInstanceY() for site in self.available_slices_ml)

        device_center_x = (min_x + max_x) // 2
        device_center_y = (min_y + max_y) // 2