"""
NetManager class for FPGA placement.
Handles network connectivity and HPWL (Half-Perimeter Wirelength) calculations.

This implementation matches the master branch approach using bounding box HPWL.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
# Logger functions not currently used but kept for potential debugging


class NetManager:
    """
    Manages network connectivity and HPWL calculations for FPGA placement.

    Uses bounding box HPWL: For each net, HPWL = (max_x - min_x) + (max_y - min_y)
    """

    def __init__(self, placer: Any, device: str = 'cpu'):
        """
        Initialize NetManager with a placer instance.

        Args:
            placer: FpgaPlacer instance containing network and site information
            device: Torch device for computations
        """
        self.placer = placer
        self.device = device

        # Net to sites mapping: {net_name: [site_names]}
        self._net_to_sites: Dict[str, List[str]] = {}

        # HPWL tracking
        self.net_hpwl: Dict[str, float] = {}
        self.net_bbox: Dict[str, Dict] = {}
        self.net_hpwl_no_io: Dict[str, float] = {}
        self.net_bbox_no_io: Dict[str, Dict] = {}
        self.total_hpwl: float = 0.0
        self.total_hpwl_no_io: float = 0.0

        # Coupling matrix for optimization (different from HPWL calculation)
        self._coupling_matrix: Optional[torch.Tensor] = None
        self._net_tensor: Optional[torch.Tensor] = None

        # Instance counts
        self._num_logic_instances: int = 0
        self._num_io_instances: int = 0

    def initialize_from_placer(self) -> None:
        """
        Initialize network data from the placer instance.
        Should be called after placer has loaded the design.
        """
        if hasattr(self.placer, 'net_to_sites'):
            self._net_to_sites = self.placer.net_to_sites

        if hasattr(self.placer, 'net_to_slice_sites_tensor'):
            self._net_tensor = self.placer.net_to_slice_sites_tensor
            if self._net_tensor is not None:
                self._net_tensor = self._net_tensor.to(self.device)

        if hasattr(self.placer, 'num_optimizable_insts'):
            self._num_logic_instances = self.placer.num_optimizable_insts

        if hasattr(self.placer, 'fixed_insts'):
            self._num_io_instances = len(self.placer.fixed_insts)

    def set_coupling_matrix(self, J: torch.Tensor) -> None:
        """
        Set the coupling matrix directly.

        Args:
            J: Coupling matrix [num_instances, num_instances]
        """
        self._coupling_matrix = J.to(self.device) if J.device != self.device else J

    @property
    def net_tensor(self) -> Optional[torch.Tensor]:
        """Get the network connectivity tensor."""
        return self._net_tensor

    @net_tensor.setter
    def net_tensor(self, value: torch.Tensor) -> None:
        """Set the network connectivity tensor."""
        self._net_tensor = value

    @property
    def coupling_matrix(self) -> Optional[torch.Tensor]:
        """Get the coupling matrix."""
        return self._coupling_matrix

    @property
    def insts_matrix(self) -> Optional[torch.Tensor]:
        """Alias for coupling_matrix for backward compatibility."""
        return self._coupling_matrix

    def _clear(self) -> None:
        """Clear HPWL tracking data."""
        self.total_hpwl = 0.0
        self.net_hpwl.clear()
        self.net_bbox.clear()
        self.total_hpwl_no_io = 0.0
        self.net_hpwl_no_io.clear()
        self.net_bbox_no_io.clear()

    def _map_coords_to_instance(self, coords: torch.Tensor,
                                io_coords: Optional[torch.Tensor],
                                include_io: bool) -> Dict[str, Tuple[float, float]]:
        """
        Map coordinates to instance names (same as master branch).

        Args:
            coords: Logic instance coordinates [num_logic, 2]
            io_coords: IO instance coordinates [num_io, 2]
            include_io: Whether to include IO instances

        Returns:
            Dictionary {site_name: (x, y)}
        """
        instance_coords = {}

        # Map logic instances
        for instance_id in range(len(coords)):
            site_name = self.placer.get_site_inst_name_by_id(instance_id)
            if site_name:
                if torch.is_tensor(coords[instance_id]):
                    x = coords[instance_id][0].item()
                    y = coords[instance_id][1].item()
                else:
                    x = coords[instance_id][0]
                    y = coords[instance_id][1]
                instance_coords[site_name] = (x, y)

        # Map IO instances
        if include_io and io_coords is not None:
            for instance_id in range(len(io_coords)):
                site_name = self.placer.get_site_inst_name_by_id(instance_id + self._num_logic_instances)
                if site_name:
                    if torch.is_tensor(io_coords[instance_id]):
                        x = io_coords[instance_id][0].item()
                        y = io_coords[instance_id][1].item()
                    else:
                        x = io_coords[instance_id][0]
                        y = io_coords[instance_id][1]
                    instance_coords[site_name] = (x, y)

        return instance_coords

    def _compute_hpwl_from_coordinates(self, coordinates: List[Tuple[float, float]]) -> Tuple[float, Dict]:
        """
        Compute bounding box HPWL from a list of coordinates.

        Args:
            coordinates: List of (x, y) tuples

        Returns:
            Tuple of (hpwl, bbox_dict)
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

    def _compute_net_hpwl(self, net_name: str, connected_sites: List[str],
                         instance_coords: Dict[str, Tuple[float, float]],
                         include_io: bool = False) -> Tuple[float, Dict]:
        """
        Compute HPWL for a single net using bounding box approach.

        Args:
            net_name: Name of the net
            connected_sites: List of site names connected by this net
            instance_coords: Dictionary mapping site names to coordinates
            include_io: Whether this is an IO-inclusive calculation

        Returns:
            Tuple of (hpwl, bbox_dict)
        """
        coordinates = []

        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])

        if len(coordinates) < 2:
            return 0.0, {}

        hpwl, bbox = self._compute_hpwl_from_coordinates(coordinates)

        # Store results
        if include_io:
            self.net_hpwl[net_name] = hpwl
            self.net_bbox[net_name] = bbox
            self.total_hpwl += hpwl
        else:
            self.net_hpwl_no_io[net_name] = hpwl
            self.net_bbox_no_io[net_name] = bbox
            self.total_hpwl_no_io += hpwl

        return hpwl, bbox

    def analyze_solver_hpwl(self, coords: torch.Tensor,
                           io_coords: Optional[torch.Tensor] = None,
                           include_io: bool = False) -> Dict[str, float]:
        """
        Analyze HPWL for a given placement (same as master branch).

        Uses bounding box HPWL: For each net, HPWL = (max_x - min_x) + (max_y - min_y)

        Args:
            coords: Logic instance coordinates [num_logic, 2]
            io_coords: IO instance coordinates [num_io, 2] (optional)
            include_io: Whether to include IO in HPWL calculation

        Returns:
            Dictionary with 'hpwl' and 'hpwl_no_io' values
        """
        self._clear()

        # Map coordinates to site names
        instance_coords = self._map_coords_to_instance(coords, io_coords, include_io)

        # Calculate HPWL for each net using bounding box approach
        for net_name, connected_sites in self._net_to_sites.items():
            self._compute_net_hpwl(net_name, connected_sites, instance_coords, include_io=include_io)

        return {
            'hpwl': self.total_hpwl,
            'hpwl_no_io': self.total_hpwl_no_io
        }

    def get_single_instance_net_hpwl(self, instance_id: int,
                                     logic_coords: torch.Tensor,
                                     io_coords: Optional[torch.Tensor],
                                     include_io: bool) -> float:
        """
        Calculate HPWL contribution for a single instance.

        Args:
            instance_id: Instance identifier
            logic_coords: Logic instance coordinates [num_logic, 2]
            io_coords: IO instance coordinates [num_io, 2]
            include_io: Whether to include IO in calculation

        Returns:
            Total HPWL for nets connected to this instance
        """
        # Get the instance name
        instance_name = self.placer.get_site_inst_name_by_id(instance_id)
        if not instance_name:
            return 0.0

        # Map coordinates to site names
        instance_coords = self._map_coords_to_instance(logic_coords, io_coords, include_io)

        total_hpwl = 0.0

        # Find all nets connected to this instance and calculate their HPWL
        for net_name, connected_sites in self._net_to_sites.items():
            if instance_name in connected_sites:
                coordinates = []
                for site_name in connected_sites:
                    if site_name in instance_coords:
                        coordinates.append(instance_coords[site_name])

                if len(coordinates) >= 2:
                    hpwl, _ = self._compute_hpwl_from_coordinates(coordinates)
                    total_hpwl += hpwl

        return total_hpwl

    def get_instance_connectivity(self, instance_id: int) -> int:
        """
        Get the connectivity (number of connections) for an instance.

        Args:
            instance_id: Instance identifier

        Returns:
            Number of connections for this instance
        """
        if self._net_tensor is not None:
            if instance_id < self._net_tensor.shape[1]:
                return int(self._net_tensor[:, instance_id].sum().item())

        if self._coupling_matrix is not None:
            if instance_id < self._coupling_matrix.shape[0]:
                return int((self._coupling_matrix[instance_id] > 0).sum().item())

        return 0

    def get_connected_instances(self, instance_id: int) -> List[int]:
        """
        Get all instances connected to the given instance through nets.

        Args:
            instance_id: Instance identifier

        Returns:
            List of connected instance IDs
        """
        if self._coupling_matrix is not None:
            if instance_id < self._coupling_matrix.shape[0]:
                connected = torch.nonzero(self._coupling_matrix[instance_id], as_tuple=True)[0]
                return connected.tolist()

        instance_name = self.placer.get_site_inst_name_by_id(instance_id)
        if not instance_name:
            return []

        connected = set()

        for net_name, connected_sites in self._net_to_sites.items():
            if instance_name in connected_sites:
                for site_name in connected_sites:
                    other_id = self.placer.get_site_inst_id_by_name(site_name)
                    if other_id is not None and other_id != instance_id:
                        connected.add(other_id)

        return list(connected)
