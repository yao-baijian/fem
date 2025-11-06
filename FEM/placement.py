import math
import sys
sys.path.append('.')
from FEM import FEM
from utils import *
import torch
import random
import rapidwright

from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import Device, Site, SiteTypeEnum


class RapidWrightWrapper:

    def __init__(self, utilization_factor = 0.6):
        self.utilization_factor = utilization_factor
        self.available_slices_ml = []  # SLICEL and SLICEM
        self.tiles = []
        self.cells = []
        pass

    def print_edfi_info(self, design, device):
        available_sites = []

    def extract_edfi_info(self, design, device):
        available_sites = []

    def print_design_info(self, design, device):
        
        available_sites = []

        for site in device.getAllSites():
            available_sites.append(site)

        print(f"design name: {design.getName()}")
        print(f"cell number: {len(design.getCells())}")
        print(f"net number: {len(design.getNets())}")
        print(f"module number: {len(design.getModules())}")
        print(f"total site tavailable: {len(available_sites)}")

    def extract_design_info(self, design, device):
        design_info = {
            'modules': [],
            'cells': [],
            'nets': [],
            'available_sites': []
        }
        
        for cell in design.getCells():
            cell_info = {
                'name': str(cell.getName()),
                'type': str(cell.getBEL().getBELType().toString()) if cell.getBEL() else 'UNKNOWN',
                'placed': cell.isPlaced()
            }
            design_info['cells'].append(cell_info)

        for module in design.getModules():
            module_info = {
                'name': str(module.getName()),
                'type': str(module.getBEL().getBELType().toString()) if module.getBEL() else 'UNKNOWN',
                'placed': module.isPlaced()
            }
            design_info['module'].append(module_info)
        
        for net in design.getNets():
            net_info = {
                'name': str(net.getName()),
                'pin_count': net.getPins().size()
            }
            design_info['nets'].append(net_info)
        
        site_count = 0
        for site in device.getAllSites():
            if site_count < 1000:
                site_info = {
                    'name': str(site.getName()),
                    'x': site.getInstanceX(),
                    'y': site.getInstanceY(),
                    'type': str(site.getSiteTypeEnum().toString())
                }
                design_info['available_sites'].append(site_info)
                site_count += 1
        
        return design_info

    def apply_optimized_placement(design, placements):
        design.unplaceDesign()
        
        for cell_name, placement in placements.items():
            cell = design.getCell(cell_name)
            if cell:
                x, y, site_type = placement
                site_name = f"SLICE_X{x}Y{y}"
                site = design.getDevice().getSite(site_name)
                if site and site.isAvailable():
                    cell.place(site)


    def estimate_place_areas(self, design, device):

        lut_cells = [cell for cell in self.cells if "LUT" in str(cell.getType())]
        ff_cells = [cell for cell in self.cells if "FF" in str(cell.getType())]
        dsp_cells = [cell for cell in self.cells if "DSP" in str(cell.getType())]
        bram_cells = [cell for cell in self.cells if "BRAM" in str(cell.getType())]
        iobuf_cells = [cell for cell in self.cells if "IOBUF" in str(cell.getType())]
        ibuf_cells = [cell for cell in self.cells if "INBUF" in str(cell.getType()) or "IBUF" in str(cell.getType())]
        obuf_cells = [cell for cell in self.cells if "OBUF" in str(cell.getType())]
        port_cells = [cell for cell in self.cells if "PORT" in str(cell.getType())]
        locked_cells = [cell for cell in self.cells if "LOCKED" in str(cell.getType())] 
        other_cells = [cell for cell in self.cells if cell not in lut_cells + ff_cells + dsp_cells + bram_cells + iobuf_cells + ibuf_cells + obuf_cells + port_cells + locked_cells]
                                                            
        lut_count = len(lut_cells)
        ff_count = len(ff_cells)
        dsp_count = len(dsp_cells)
        bram_count = len(bram_cells)
        iobuf_count = len(iobuf_cells)
        ibuf_count = len(ibuf_cells)
        obuf_count = len(obuf_cells)
        port_count = len(port_cells)
        locked_count = len(locked_cells)
        other_count = len(other_cells)

        print(f"cell: luts={lut_count}, ffs={ff_count}, dsps={dsp_count}, brams={bram_count}, iobufs={iobuf_count}, ibufs={ibuf_count}, obufs={obuf_count}, ports={port_count}, locked={locked_count}, other={other_count}. avail sites: {len(self.available_slices_ml)} ")
        required_slices = int((lut_count + ff_count) / 4 / self.utilization_factor)
        
        min_x = min(site.getInstanceX() for site in self.available_slices_ml)
        max_x = max(site.getInstanceX() for site in self.available_slices_ml)
        min_y = min(site.getInstanceY() for site in self.available_slices_ml)
        max_y = max(site.getInstanceY() for site in self.available_slices_ml)

        print(f"available slice area: x[{min_x}, {max_x}], y[{min_y}, {max_y}]")
        
        total_width = max_x - min_x + 1
        total_height = max_y - min_y + 1
    
        aspect_ratio = total_width / total_height
        required_area = required_slices * 2
        
        area_width = int(math.sqrt(required_area * aspect_ratio))
        area_height = int(required_area / area_width)
        
        area_width = min(area_width, total_width)
        area_height = min(area_height, total_height)

        print(f"estimated place area size: width={area_width}, height={area_height}, required_slices={required_slices}")
        
        # middle placement
        start_x = min_x + (total_width - area_width) // 2
        start_y = min_y + (total_height - area_height) // 2
        
        area_info = {
            'design': {
                'design': design,
                'device': device
            },
            'bounding_box': {
                'start_x': start_x,
                'start_y': start_y,
                'end_x': start_x + area_width - 1,
                'end_y': start_y + area_height - 1,
                'width': area_width,
                'height': area_height
            },
            'resource_info': {
                'total_cells': len(self.cells),
                'lut_count': lut_count,
                'ff_count': ff_count,
                'other_count': other_count,
                'available_slices_ml': len(self.available_slices_ml),
                'required_slices': required_slices,
                'utilization': (lut_count + ff_count) / len(self.available_slices_ml) if self.available_slices_ml else 0
            },
            'device_info': {
                'device_name': device.getName(),
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            },
            'cell_lists': {
                'lut_cells': lut_cells,
                'ff_cells': ff_cells,
                'dsp_cells': dsp_cells,
                'bram_cells': bram_cells,
                'iobuf_cells': iobuf_cells,
                'ibuf_cells': ibuf_cells,
                'obuf_cells': obuf_cells,
                'port_cells': port_cells,
                'locked_cells': locked_cells,
                'other_cells': other_cells
            }
        }
        
        print(f"estimate place area: ({start_x}, {start_y}) 到 ({start_x + area_width - 1}, {start_y + area_height - 1})")
        return area_info

    def random_initial_placement(self, area_info, design, device):
        unfixed_placements = {}
        fixed_placements = {}
        bbox = area_info['bounding_box']

        # available_sites = []
        # tiles = device.getTiles()
        # rows = len(tiles)
        # cols = len(tiles[0])

        # for row in range(rows):
        #     for col in range(cols):
        #         if row >= bbox['start_y'] and row <= bbox['end_y'] and \
        #            col >= bbox['start_x'] and col <= bbox['end_x']:
        #             tiles_sites = tiles[row][col].getSites()
        #             available_sites.append(tiles_sites)

        available_slices = []
        for slice in self.available_slices_ml:
            if (bbox['start_x'] <= slice.getInstanceX() <= bbox['end_x'] and 
                bbox['start_y'] <= slice.getInstanceY() <= bbox['end_y']):
                available_slices.append(slice)
        
        print(f"available slices: {len(available_slices)}")

        if len(available_slices) < len(self.cells):
            available_slices = list(device.getAllCompatibleSites(SiteTypeEnum.SLICEL))
            available_slices.extend(list(device.getAllCompatibleSites(SiteTypeEnum.SLICEM)))
            # available_sites = [site for site in available_sites if site.isSiteAvailable()]
        
        random.shuffle(available_slices)

        lut_ff_cells = area_info['cell_lists']['lut_cells'] + area_info['cell_lists']['ff_cells']

        unfixed_placed_count = 0
        for i, cell in enumerate(lut_ff_cells):
            if i < len(available_slices):
                site = available_slices[i]
                
                if self.is_cell_compatible_with_site(cell, site):
                    unfixed_placements[cell.getName()] = {
                        'site': site.getName(),
                        'x': site.getInstanceX(),
                        'y': site.getInstanceY(),
                        'site_type': site.getSiteTypeEnum().name,
                        'bel': self.suggest_bel_for_cell(cell, site),
                        'cell_type': 'LUT/FF'
                    }
                    unfixed_placed_count += 1
                else:
                    unfixed_placements[cell.getName()] = {
                        'site': None,
                        'x': -1,
                        'y': -1,
                        'site_type': 'INCOMPATIBLE',
                        'bel': None,
                        'cell_type': 'LUT/FF'
                    }
                    print(f"ERROR: Cell {cell.getName()} is not compatible with site {site.getName()}")
            else:
                unfixed_placements[cell.getName()] = {
                    'site': None,
                    'x': -1,
                    'y': -1,
                    'site_type': 'NO_AVAILABLE_SITE',
                    'bel': None,
                    'cell_type': 'LUT/FF'
                }
                print(f"ERROR: No available site for cell {cell.getName()}")
        
        other_cells =   area_info['cell_lists']['dsp_cells'] + \
                        area_info['cell_lists']['bram_cells'] + \
                        area_info['cell_lists']['iobuf_cells'] + \
                        area_info['cell_lists']['ibuf_cells'] + \
                        area_info['cell_lists']['obuf_cells'] + \
                        area_info['cell_lists']['port_cells'] + \
                        area_info['cell_lists']['locked_cells'] + \
                        area_info['cell_lists']['other_cells']
        
        for cell in other_cells:
            fixed_placements[cell.getName()] = {
                'site': 'FIXED_OR_DEFAULT',
                'x': -1,
                'y': -1,
                'site_type': 'OTHER',
                'bel': None,
                'cell_type': 'OTHER'
            }
        
        print(f"initial placement: {unfixed_placed_count}/{len(lut_ff_cells)} for total {len(self.cells)} sites done, {len(fixed_placements)} remains.")
        return unfixed_placements, fixed_placements 

    def is_cell_compatible_with_site(self, cell, site):
        if cell.getBEL() is None:
            return True 
        
        bel_type = cell.getBEL().getBELType()
        site_type = site.getSiteTypeEnum()
        
        if "LUT" in str(bel_type):
            return site_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]
        elif "FF" in str(bel_type):
            return site_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]
        elif "DSP" in str(bel_type):
            return "DSP" in str(site_type)
        elif "BRAM" in str(bel_type):
            return "BRAM" in str(site_type)
        
        return True 

    def suggest_bel_for_cell(self, cell, site):
        """
        为单元建议合适的BEL位置
        """
        if cell.getBEL() is None:
            return "A6LUT"  # 默认LUT位置
        
        bel_type = cell.getBEL().getBELType()
        
        if "LUT" in str(bel_type):
            # 在站点内分配不同的LUT位置
            lut_positions = ["A6LUT", "B6LUT", "C6LUT", "D6LUT"]
            return random.choice(lut_positions)
        elif "FF" in str(bel_type):
            # 在站点内分配不同的FF位置
            ff_positions = ["AFF", "BFF", "CFF", "DFF"]
            return random.choice(ff_positions)
        
        return None

    def apply_placement_to_design(self, design, placements):

        print("应用布局到设计...")
        
        placed_count = 0
        for cell_name, placement in placements.items():
            if placement['site'] is not None:
                cell = design.getCell(cell_name)
                if cell:
                    site = design.getDevice().getSite(placement['site'])
                    if site and site.isSiteAvailable():
                        try:
                            cell.place(site)
                            placed_count += 1
                        except Exception as e:
                            print(f"布局单元 {cell_name} 失败: {e}")
        
        print(f"成功应用 {placed_count} 个单元的布局")
        return placed_count


    def find_available_site(self, device, x, y, site_type):
        site = device.getSite(f"SLICE_X{x}Y{y}")
        if site and site.isAvailable() and site.getSiteTypeEnum().name == site_type:
            return site
        
        for site in device.getAllSites():
            if (site.getInstanceX() == x and 
                site.getInstanceY() == y and 
                site.isAvailable() and
                site.getSiteTypeEnum().name == site_type):
                return site
        
        for site in device.getAllSites():
            if (site.isAvailable() and 
                site.getSiteTypeEnum().name == site_type):
                return site
                
        return None

    def fem_place(self, design, device, netlist, output_wrl='', replace = False):

        num_trials = 20
        num_steps = 1000
        dev = 'cpu'

        case_type = 'fpga_placement'
        instance = ''

        design.unplaceDesign()
        initial_areas = self.estimate_place_areas(design, device)
        placements = self.random_initial_placement(initial_areas, design, device)

        case_placements = FEM.from_file(case_type, instance, index_start=1)
        case_placements.set_up_solver(num_trials, num_steps, dev=dev, q=2, manual_grad= False)
        config, result = case_placements.solve()
        optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
        best_config = config[optimal_inds[0]]

        placements = decode_placement(best_config)

        # for cell_name, placement_info in placements.items():
        #     cell = design.getCell(cell_name)
        #     if cell is None:
        #         print(f"警告: 未找到单元 {cell_name}")
        #         continue
                
        #     if isinstance(placement_info, tuple) and len(placement_info) >= 2:
        #         if len(placement_info) == 2:
        #             x, y = placement_info
        #             site_type = "SLICEL"  # 默认类型
        #         else:
        #             x, y, site_type = placement_info
                
        #         # 查找合适的站点
        #         target_site = find_available_site(device, x, y, site_type)
        #         if target_site:
        #             try:
        #                 cell.place(target_site)
        #                 placed_cells += 1
        #             except Exception as e:
        #                 print(f"布局单元 {cell_name} 失败: {e}")
        #         else:
        #             print(f"未找到合适站点: ({x}, {y}, {site_type})")
        
        # use rapidwright to evaluate placement
        # evaluation = evaluate_placement(design)
        # return evaluation
  
    def optimize_fpga_placement(self, dcp_file = '', edf_file='', dcp_output=''):

        design = Design.readCheckpoint(dcp_file)
        device = design.getDevice()
        # edfi_info = extract_edfi_info(design, device)

        self.available_slices_ml = list(device.getAllCompatibleSites(SiteTypeEnum.SLICEL))
        self.available_slices_ml.extend(list(device.getAllCompatibleSites(SiteTypeEnum.SLICEM)))

        self.cells = design.getCells()

        edfi_netlist = None
        optimized_placements = self.fem_place(design, device, edfi_netlist)
        # apply_optimized_placement(design, optimized_placements)
        design.writeCheckpoint(dcp_output)

# with cell and net: ./vivado/output_dir/post_impl.dcp
# only logic netlist: ./vivado/output_dir/post_synth.dcp

my_placement = RapidWrightWrapper()
my_placement.optimize_fpga_placement('./vivado/output_dir/post_impl.dcp', 'optimized_placement.dcp')


# def evaluate_placement(design):
#     """
#     评估布局质量
#     """
#     evaluation = {}
    
#     try:
#         # 计算布局统计
#         placed_cells = 0
#         total_cells = 0
        
#         for cell in design.getCells():
#             total_cells += 1
#             if cell.isPlaced():
#                 placed_cells += 1
        
#         evaluation['placed_cells'] = placed_cells
#         evaluation['total_cells'] = total_cells
#         evaluation['placement_ratio'] = placed_cells / total_cells if total_cells > 0 else 0
        
#         # 估算线长（简化版）
#         estimated_wirelength = estimate_wirelength(design)
#         evaluation['estimated_wirelength'] = estimated_wirelength
        
#         # 时序分析（需要更复杂的实现）
#         timing_score = estimate_timing(design)
#         evaluation['timing_score'] = timing_score
        
#         print(f"布局评估完成:")
#         print(f"  - 布局单元: {placed_cells}/{total_cells} ({evaluation['placement_ratio']:.1%})")
#         print(f"  - 估计线长: {estimated_wirelength}")
#         print(f"  - 时序评分: {timing_score}")
        
#     except Exception as e:
#         print(f"布局评估错误: {e}")
#     return evaluation

# def save_placement_result(design, output_path, evaluation):
#     """
#     保存布局结果
#     """
#     try:
#         # 保存为DCP文件
#         dcp_file = output_path.replace('.wrl', '.dcp')
#         design.writeCheckpoint(dcp_file)
#         print(f"布局结果保存为: {dcp_file}")
        
#         # 保存评估报告
#         report_file = output_path.replace('.wrl', '_report.txt')
#         with open(report_file, 'w') as f:
#             f.write("FPGA布局评估报告\n")
#             f.write("================\n")
#             f.write(f"设计名称: {design.getName()}\n")
#             f.write(f"布局单元: {evaluation['placed_cells']}/{evaluation['total_cells']}\n")
#             f.write(f"布局比例: {evaluation['placement_ratio']:.1%}\n")
#             f.write(f"估计线长: {evaluation['estimated_wirelength']}\n")
#             f.write(f"时序评分: {evaluation['timing_score']}\n")
        
#         print(f"评估报告保存为: {report_file}")
        
#     except Exception as e:
#         print(f"保存结果失败: {e}")