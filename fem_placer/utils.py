"""
Utility functions for FPGA placement
"""
import torch


def parse_fpga_design(fpga_wrapper):
    """
    Parse FPGA design from wrapper and create coupling matrix

    Args:
        fpga_wrapper: FpgaPlacer instance containing design information

    Returns:
        num_inst: number of instances to place
        num_site: number of available sites
        J: coupling matrix (connectivity between instances)
        J_extend: extended coupling matrix including IO connections
    """
    n = len(fpga_wrapper.optimizable_insts)
    m = len(fpga_wrapper.available_sites)
    J = torch.zeros((n, n))

    # Build coupling matrix from site-to-site connectivity
    for source_site, connections in fpga_wrapper.site_to_site_connectivity.items():
        source_id = fpga_wrapper.get_site_inst_id_by_name(source_site)
        if source_id is not None:
            for target_site, connection_count in connections.items():
                target_id = fpga_wrapper.get_site_inst_id_by_name(target_site)
                if target_id is not None:
                    J[source_id, target_id] = connection_count
                else:
                    print(f'ERROR: cannot find site target id {target_id}')
        else:
            print(f'ERROR: cannot find site source id {source_site}')

    k = len(fpga_wrapper.fixed_insts)
    J_extend = torch.zeros((n + k, n + k))

    # Add IO connections to extended matrix
    for source_site, connections in fpga_wrapper.io_to_site_connectivity.items():
        source_id = fpga_wrapper.get_site_inst_id_by_name(source_site)
        if source_id is not None:
            for target_site, connection_count in connections.items():
                target_id = fpga_wrapper.get_site_inst_id_by_name(target_site)
                if target_id is not None:
                    J_extend[source_id, target_id] = connection_count
                else:
                    print(f'ERROR: cannot find site target id {target_id}')
        else:
            print(f'ERROR: cannot find site source id {source_site}')

    print(f'INFO: site matrix {n} x {n}, io site matrix {n + k} x {n + k}')

    return n, m, J, J_extend
