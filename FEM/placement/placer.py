import sys
sys.path.append('.')
from utils import *
import random
import io
import sys
from contextlib import contextmanager
import rapidwright
from .net import NetManager
from .config import *
from .logger import INFO, WARNING, ERROR

from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import Device, Site, SiteTypeEnum
from com.xilinx.rapidwright.rwroute import RWRoute
from com.xilinx.rapidwright.util import MessageGenerator

class FpgaPlacer:

    def __init__(self, utilization_factor = 1):
        
        self.name = "Placer"
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
        self.optimizable_site_inst_to_id = {} 
        self.available_site_to_id = {}    
        self.optimizable_id_to_site_inst = {}
        self.available_id_to_site = {}   

        self.fixed_site_to_id = {}    
        self.fixed_id_to_site = {}
        self.place_orientation = PlaceType.IO
        self.debug = True
        self.net_manager = NetManager(self.get_site_inst_id_by_name, 
                                      self.get_site_inst_name_by_id,
                                      self.map_coords_to_instance,
                                      debug=self.debug)
        pass
    
    @contextmanager
    def suppress_rapidwright_output(self):
        """抑制RapidWright的输出"""
        # 保存原始的Python输出
        original_py_out = sys.stdout
        original_py_err = sys.stderr
        
        # 重定向Python输出到空设备
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # 在with块内导入Java类
            from java.lang import System
            from java.io import PrintStream
            from java.io import ByteArrayOutputStream
            
            # 保存原始的Java输出
            original_java_out = System.out
            original_java_err = System.err
            
            # 创建不输出的PrintStream
            baos = ByteArrayOutputStream()
            null_stream = PrintStream(baos)
            
            System.setOut(null_stream)
            System.setErr(null_stream)
            
            yield
            
        finally:
            # 恢复Python输出
            sys.stdout = original_py_out
            sys.stderr = original_py_err
            
            # 恢复Java输出
            try:
                System.setOut(original_java_out)
                System.setErr(original_java_err)
            except NameError:
                # 如果在try块中导入失败，System可能未定义
                pass
        
    def get_site_inst_id_by_name(self, site_name):
        if site_name in self.optimizable_site_inst_to_id:
            return self.optimizable_site_inst_to_id.get(site_name)
        elif site_name in self.fixed_site_to_id:
            return self.fixed_site_to_id.get(site_name) + self.num_optimizable_insts
        else:
            WARNING(f"Cannot find site_inst id for site_name: {site_name}")
            return None
        
    def get_site_inst_name_by_id(self, id):
        if id in self.optimizable_id_to_site_inst:
            return self.optimizable_id_to_site_inst.get(id)
        elif (id - self.num_optimizable_insts) in self.fixed_id_to_site:
            return self.fixed_id_to_site.get(id - self.num_optimizable_insts)
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

    def get_clock_insts(self, design):
        
        for site in design.getSiteInsts():
            site_type = site.getSiteTypeEnum()
            
            if site_type in CLOCK_SITE_ENUM:
                # self.fixed_insts.append(site)
                continue

            elif site not in self.optimizable_insts and site not in self.fixed_insts:
                WARNING(f'Cannot orient site type, site: {site.getName()}, type: {site.getSiteTypeEnum()}')

    def get_available_target_sites(self, device):
        
        for site in device.getAllSites():
            site_x = site.getInstanceX()
            site_y = site.getInstanceY()

            # print(f'site: {site.getName()}, type: {site.getSiteTypeEnum()}, x: {site_x}, y: {site_y}')
            
            if (self.bbox['start_x'] <= site_x <= self.bbox['end_x'] and 
                self.bbox['start_y'] <= site_y <= self.bbox['end_y'] ):
                
                site_type = site.getSiteTypeEnum()

                if site_type in SLICE_SITE_ENUM:
                    self.available_sites.append(site)
        
        if len(self.available_sites) < len(self.optimizable_insts):
            WARNING(f"Available sites({len(self.available_sites)}) less than optimizable sites({len(self.optimizable_insts)})")
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

    def init_place_areas(self, design):
        
        total_site_insts = len(design.getSiteInsts())
        other_sites = total_site_insts - (len(self.optimizable_insts) + len(self.fixed_insts))
        INFO(f"Estimate sites number: {len(self.optimizable_insts)} slice sites, {len(self.fixed_insts)} fixed sites, {other_sites} other sites, total {total_site_insts} sites.")
        side_length = int(np.ceil(np.sqrt(total_site_insts / self.utilization_factor)))
        side_width = 0
        
        min_x = min(site.getInstanceX() for site in self.available_slices_ml)
        max_x = max(site.getInstanceX() for site in self.available_slices_ml)
        min_y = min(site.getInstanceY() for site in self.available_slices_ml)
        max_y = max(site.getInstanceY() for site in self.available_slices_ml)

        device_center_x = (min_x + max_x) // 2
        device_center_y = (min_y + max_y) // 2
        
        start_x = device_center_x - side_length // 2
        end_x = start_x + side_length
        start_y = device_center_y - side_length // 2
        end_y = start_y + side_length
        
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
        INFO(f"Estimate area {side_length} x {side_length} , start=({start_x},{start_y}), end=({end_x},{end_y})")

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
        
        INFO(f"Left IO area - position: ({io_start_x}, {io_start_y}) to ({io_end_x}, {io_end_y})")

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
        
        INFO(f"Clock buffer area - position: ({clock_start_x}, {clock_start_y}) to ({clock_end_x}, {clock_end_y})")

    def random_initial_placement(self, design):
    
        design.unplaceDesign()

        self.num_of_instance = len(self.optimizable_insts)
        self.num_of_sites = len(self.available_sites)
        
        INFO(f"Optimizable instances num: {self.num_of_instance}, available sites num: {self.num_of_sites}")

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
        INFO(f"Reading DCP: {dcp_file}")
        # with self.suppress_rapidwright_output():
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
        _, hpwl_vivado = self.net_manager.analyze_design_hpwl(design)
        self.get_optimizable_insts(design) 
        self.get_fixed_insts(design)         
        self.get_clock_insts(design)      
        self.init_place_areas(design)
        self._init_io_area()
        self._init_clock_buffer_area()
        self.get_available_target_sites(device)
        self.map_site_to_id()
        self.net_manager.analyze_nets(len(self.optimizable_insts), len(self.available_sites), len(self.fixed_insts))
        self.random_initial_placement(design)
        
        return hpwl_vivado

    def map_coords_to_instance(self, coords, io_coords=None, include_io=False):
        instance_coords = {}
        
        for instance_id in range(len(coords)):
            site_name = self.get_site_inst_name_by_id(instance_id)
            instance_coords[site_name] = coords[instance_id]

        if include_io:
            for instance_id in range(len(io_coords)):
                site_name = self.get_site_inst_name_by_id(instance_id + self.num_optimizable_insts)
                instance_coords[site_name] = io_coords[instance_id]
        
        return instance_coords
