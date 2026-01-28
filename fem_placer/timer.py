"""
Timing Analysis for FPGA Placement.

This module provides timing-aware placement optimization including:
- Timing path extraction and analysis
- Timing-weighted HPWL calculation
- Congestion estimation and awareness
- Timing closure analysis
"""

import torch
import numpy as np


class Timer:
    """
    Timer class for timing-aware placement optimization.

    This class handles timing analysis, including path delay calculation,
    timing-weighted objectives, and congestion estimation.
    """

    def __init__(self):
        """Initialize the Timer."""
        self.timing_library = {}
        self.cell_delays = {}
        self.net_delays = {}
        self.timing_paths = []
        self.site_to_index = {}
        self.optimizable_sites = []
        self.available_target_sites = []

    def setup_timing_analysis(self, design, timing_library):
        """
        Set up timing analysis from a design.

        Args:
            design: RapidWright design object
            timing_library: Dictionary mapping cell types to timing parameters
        """
        self.timing_library = timing_library
        self.cell_delays = self.extract_cell_delays(design)
        self.net_delays = self.extract_net_delays(design)
        self.timing_paths = self.extract_timing_paths(design)

    def extract_cell_delays(self, design):
        """
        Extract cell delays from a design.

        Args:
            design: RapidWright design object

        Returns:
            cell_delays: Dictionary mapping cell names to delay info
        """
        cell_delays = {}
        for cell in design.getCells():
            cell_type = cell.getType()
            if cell_type in self.timing_library:
                cell_delays[cell.getName()] = self.timing_library[cell_type]
            else:
                # Default delays
                cell_delays[cell.getName()] = {
                    'min_delay': 0.1,
                    'max_delay': 0.2,
                    'setup_time': 0.05,
                    'hold_time': 0.02
                }
        return cell_delays

    def extract_net_delays(self, design):
        """
        Extract net delays from a design.

        Args:
            design: RapidWright design object

        Returns:
            net_delays: Dictionary mapping net names to delay info
        """
        net_delays = {}
        for net in design.getNets():
            net_length = self.estimate_net_length(net)
            net_delays[net.getName()] = {
                'unit_delay': 0.01,
                'estimated_delay': net_length * 0.01
            }
        return net_delays

    def estimate_net_length(self, net):
        """
        Estimate the length of a net.

        Args:
            net: RapidWright net object

        Returns:
            length: Estimated net length
        """
        # Simple estimation based on pin count
        return len(net.getPins()) * 10  # Basic estimation

    def extract_timing_paths(self, design):
        """
        Extract timing paths from a design.

        Args:
            design: RapidWright design object

        Returns:
            timing_paths: List of timing path dictionaries
        """
        timing_paths = []

        for cell in design.getCells():
            if 'FD' in cell.getType():  # Flip-flop
                timing_paths.append({
                    'start_cell': cell.getName(),
                    'end_cell': self.find_connected_flipflop(cell),
                    'required_time': 10.0,
                    'criticality': 1.0
                })

        return timing_paths

    def find_connected_flipflop(self, cell):
        """
        Find a flip-flop connected to the given cell.

        Args:
            cell: RapidWright cell object

        Returns:
            cell_name: Name of connected flip-flop or None
        """
        # Placeholder - in real implementation would trace connections
        return None

    def get_expected_placements_from_index(self, p, area_width):
        """
        Get expected placement coordinates from probability distribution.

        Args:
            p: Probability distribution [batch_size, num_instances, num_sites]
            area_width: Width of the placement area

        Returns:
            expected_coords: Expected coordinates [batch_size, num_instances, 2]
        """
        batch_size, num_instances, num_sites = p.shape
        device = p.device

        # Create coordinate matrix
        indices = torch.arange(num_sites, dtype=torch.float32, device=device)
        x_coords = indices % area_width
        y_coords = indices // area_width
        site_coords = torch.stack([x_coords, y_coords], dim=1)

        # Calculate expected coordinates
        expected_coords = torch.matmul(p, site_coords)
        return expected_coords

    def calculate_timing_based_hpwl(self, J_extended, p, area_width, timing_criticality):
        """
        Calculate timing-weighted HPWL.

        Args:
            J_extended: Extended coupling matrix
            p: Probability distribution
            area_width: Width of the placement area
            timing_criticality: Weight for timing criticality

        Returns:
            weighted_wirelength: Timing-weighted total wirelength
        """
        batch_size = p.shape[0]
        expected_coords = self.get_expected_placements_from_index(p, area_width)

        J = torch.tensor(J_extended, dtype=torch.float32, device=p.device)

        # Apply timing criticality weights
        timing_weights = self.calculate_timing_weights(J_extended, timing_criticality)
        weighted_J = J * timing_weights

        # Calculate weighted HPWL
        coords_i = expected_coords.unsqueeze(2)
        coords_j = expected_coords.unsqueeze(1)
        manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)

        weighted_dist = manhattan_dist * weighted_J.unsqueeze(0)
        triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
        weighted_dist_triu = weighted_dist[:, triu_mask]

        total_wirelength = torch.sum(weighted_dist_triu, dim=1)

        return torch.mean(total_wirelength) / batch_size

    def calculate_timing_weights(self, J_extended, timing_criticality):
        """
        Calculate timing criticality weights.

        Args:
            J_extended: Extended coupling matrix
            timing_criticality: Weight factor for timing

        Returns:
            timing_weights: Timing weight matrix
        """
        num_instances = J_extended.shape[0]
        timing_weights = np.ones_like(J_extended)

        for path in self.timing_paths:
            start_idx = self.site_to_index.get(path['start_cell'], -1)
            end_idx = self.site_to_index.get(path['end_cell'], -1)

            if start_idx != -1 and end_idx != -1:
                criticality = path['criticality'] * timing_criticality
                timing_weights[start_idx, end_idx] += criticality
                timing_weights[end_idx, start_idx] += criticality

        return torch.tensor(timing_weights, dtype=torch.float32)

    def calculate_path_delays(self, p, area_width):
        """
        Calculate timing path delays.

        Args:
            p: Probability distribution
            area_width: Width of the placement area

        Returns:
            path_delays: List of path delay dictionaries
        """
        batch_size = p.shape[0]
        expected_coords = self.get_expected_placements_from_index(p, area_width)

        path_delays = []

        for path in self.timing_paths:
            start_idx = self.site_to_index.get(path['start_cell'], -1)
            end_idx = self.site_to_index.get(path['end_cell'], -1)

            if start_idx != -1 and end_idx != -1:
                start_coords = expected_coords[:, start_idx, :]
                end_coords = expected_coords[:, end_idx, :]
                path_length = torch.sum(torch.abs(start_coords - end_coords), dim=1)

                cell_delay = (self.cell_delays.get(path['start_cell'], {}).get('max_delay', 0.1) +
                             self.cell_delays.get(path['end_cell'], {}).get('max_delay', 0.1))

                net_key = path['start_cell'] + '_to_' + str(path['end_cell'])
                unit_delay = self.net_delays.get(net_key, {}).get('unit_delay', 0.01)
                wire_delay = path_length * unit_delay

                total_delay = cell_delay + wire_delay
                path_delays.append({
                    'delay': total_delay,
                    'required_time': path['required_time'],
                    'slack': path['required_time'] - total_delay,
                    'criticality': path['criticality']
                })

        return path_delays

    def calculate_timing_violation_loss(self, p, area_width):
        """
        Calculate timing violation loss.

        Args:
            p: Probability distribution
            area_width: Width of the placement area

        Returns:
            violation_loss: Average timing violation loss
        """
        path_delays = self.calculate_path_delays(p, area_width)

        if not path_delays:
            return 0.0

        total_violation = 0.0
        critical_path_count = 0

        for path_info in path_delays:
            slack = path_info['slack']
            if isinstance(slack, torch.Tensor):
                slack_val = slack.mean().item()
            else:
                slack_val = slack

            if slack_val < 0:
                violation = -slack_val * path_info['criticality']
                total_violation += violation
                critical_path_count += 1

        return total_violation / len(path_delays) if path_delays else 0.0

    def calculate_congestion_aware_hpwl(self, J_extended, p, area_width, congestion_map=None):
        """
        Calculate congestion-aware HPWL.

        Args:
            J_extended: Extended coupling matrix
            p: Probability distribution
            area_width: Width of the placement area
            congestion_map: Pre-computed congestion map (optional)

        Returns:
            weighted_wirelength: Congestion-weighted wirelength
        """
        if congestion_map is None:
            congestion_map = self.estimate_congestion(p, area_width)

        batch_size = p.shape[0]
        expected_coords = self.get_expected_placements_from_index(p, area_width)

        J = torch.tensor(J_extended, dtype=torch.float32, device=p.device)
        num_instances = J.shape[0]

        coords_i = expected_coords.unsqueeze(2)
        coords_j = expected_coords.unsqueeze(1)
        manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)

        congestion_weights = self.calculate_congestion_weights(expected_coords, congestion_map)

        # Element-wise multiply: manhattan_dist [batch, inst, inst], congestion_weights [batch, inst, inst], J [inst, inst]
        weighted_dist = manhattan_dist * congestion_weights * J.unsqueeze(0)

        triu_mask = torch.triu(torch.ones(num_instances, num_instances, device=p.device), diagonal=1).bool()
        weighted_dist_triu = weighted_dist[:, triu_mask]

        total_wirelength = torch.sum(weighted_dist_triu, dim=1)

        return torch.mean(total_wirelength) / batch_size

    def estimate_congestion(self, p, area_width):
        """
        Estimate placement congestion.

        Args:
            p: Probability distribution
            area_width: Width of the placement area

        Returns:
            congestion_map: Congestion map [batch_size, area_width, area_width]
        """
        batch_size = p.shape[0]

        site_usage = torch.sum(p, dim=1)

        congestion_map = torch.zeros(batch_size, area_width, area_width, device=p.device)

        for b in range(batch_size):
            for loc_idx in range(site_usage.shape[1]):
                x = loc_idx % area_width
                y = loc_idx // area_width
                if x < area_width and y < area_width:
                    congestion_map[b, y, x] = site_usage[b, loc_idx]

        return congestion_map

    def calculate_congestion_weights(self, expected_coords, congestion_map):
        """
        Calculate congestion weights.

        Args:
            expected_coords: Expected placement coordinates
            congestion_map: Congestion map

        Returns:
            congestion_weights: Congestion weight matrix
        """
        batch_size, num_instances, _ = expected_coords.shape
        congestion_weights = torch.ones(batch_size, num_instances, num_instances,
                                        device=expected_coords.device)

        for b in range(batch_size):
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    coord_i = expected_coords[b, i].long()
                    coord_j = expected_coords[b, j].long()

                    path_congestion = self.calculate_path_congestion(
                        coord_i, coord_j, congestion_map[b]
                    )

                    congestion_weights[b, i, j] = 1.0 + path_congestion
                    congestion_weights[b, j, i] = 1.0 + path_congestion

        return congestion_weights

    def calculate_path_congestion(self, start, end, congestion_map):
        """
        Calculate average congestion along a path.

        Args:
            start: Start coordinates [2]
            end: End coordinates [2]
            congestion_map: Congestion map [height, width]

        Returns:
            avg_congestion: Average congestion value (float)
        """
        # Convert to Python ints if tensors
        start_x = int(start[0].item()) if hasattr(start[0], 'item') else int(start[0])
        start_y = int(start[1].item()) if hasattr(start[1], 'item') else int(start[1])
        end_x = int(end[0].item()) if hasattr(end[0], 'item') else int(end[0])
        end_y = int(end[1].item()) if hasattr(end[1], 'item') else int(end[1])

        x_path = range(min(start_x, end_x), max(start_x, end_x) + 1)
        y_path = range(min(start_y, end_y), max(start_y, end_y) + 1)

        total_congestion = 0.0
        point_count = 0

        for x in x_path:
            for y in y_path:
                if x < congestion_map.shape[1] and y < congestion_map.shape[0]:
                    val = congestion_map[y, x]
                    total_congestion += val.item() if hasattr(val, 'item') else float(val)
                    point_count += 1

        return total_congestion / point_count if point_count > 0 else 0.0

    def comprehensive_energy_function(self, J_extended, p, area_width, weights=None):
        """
        Calculate comprehensive energy function with all constraints.

        Args:
            J_extended: Extended coupling matrix
            p: Probability distribution
            area_width: Width of the placement area
            weights: Weight dictionary for different components

        Returns:
            total_energy: Total energy value
            losses: Dictionary of individual loss components
        """
        if weights is None:
            weights = {
                'hpwl': 1.0,
                'timing': 5.0,
                'congestion': 2.0,
                'site_constraint': 10.0,
                'type_constraint': 5.0
            }

        total_energy = 0.0

        # Basic HPWL
        hpwl_loss = self.calculate_timing_based_hpwl(J_extended, p, area_width, 0.0)
        total_energy += weights['hpwl'] * hpwl_loss

        # Timing constraint
        timing_loss = self.calculate_timing_violation_loss(p, area_width)
        total_energy += weights['timing'] * timing_loss

        # Congestion-aware HPWL
        congestion_loss = self.calculate_congestion_aware_hpwl(J_extended, p, area_width)
        total_energy += weights['congestion'] * congestion_loss

        losses = {
            'hpwl_loss': hpwl_loss.item() if hasattr(hpwl_loss, 'item') else hpwl_loss,
            'timing_loss': timing_loss if isinstance(timing_loss, float) else timing_loss.item(),
            'congestion_loss': congestion_loss.item() if hasattr(congestion_loss, 'item') else congestion_loss,
        }

        return total_energy, losses

    def optimize_with_all_constraints(self, J_extended, area_width, num_iterations=1000):
        """
        Optimize placement considering all constraints.

        Args:
            J_extended: Extended coupling matrix
            area_width: Width of the placement area
            num_iterations: Number of optimization iterations

        Returns:
            best_p: Best probability distribution found
            best_energy: Best energy value
        """
        num_instances = len(self.optimizable_sites)
        num_locations = len(self.available_target_sites)

        h = torch.randn(1, num_instances, num_locations, requires_grad=True)
        optimizer = torch.optim.Adam([h], lr=0.005)

        best_energy = float('inf')
        best_p = None

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            p = torch.softmax(h, dim=2)

            total_energy, losses = self.comprehensive_energy_function(J_extended, p, area_width)

            total_energy.backward()
            optimizer.step()

            if total_energy.item() < best_energy:
                best_energy = total_energy.item()
                best_p = p.detach().clone()

            if iteration % 100 == 0:
                print(f"Iter {iteration}: "
                      f"Total={total_energy.item():.3f}, "
                      f"HPWL={losses['hpwl_loss']:.3f}, "
                      f"Timing={losses['timing_loss']:.3f}, "
                      f"Congestion={losses['congestion_loss']:.3f}")

        return best_p, best_energy

    def analyze_timing_closure(self, p, area_width):
        """
        Analyze timing closure.

        Args:
            p: Probability distribution
            area_width: Width of the placement area

        Returns:
            timing_info: Dictionary with timing closure information
        """
        path_delays = self.calculate_path_delays(p, area_width)

        timing_info = {
            'total_paths': len(path_delays),
            'violating_paths': 0,
            'worst_slack': float('inf'),
            'total_slack': 0.0
        }

        for path in path_delays:
            slack = path['slack']
            if isinstance(slack, torch.Tensor):
                slack_val = slack.mean().item()
            else:
                slack_val = slack

            if slack_val < 0:
                timing_info['violating_paths'] += 1
            timing_info['worst_slack'] = min(timing_info['worst_slack'], slack_val)
            timing_info['total_slack'] += slack_val

        timing_info['avg_slack'] = (timing_info['total_slack'] / len(path_delays)
                                    if path_delays else 0)

        return timing_info
