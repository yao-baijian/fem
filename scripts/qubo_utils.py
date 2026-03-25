import torch


def reconstruct_logic_site_coords(n_sites: int) -> torch.Tensor:
    """Reconstruct a logic site coordinate grid from number of sites.

    Assumes a near-square grid; falls back to a 1D layout if shapes
    do not match exactly. This mirrors the coordinate reconstruction
    used in the *_no_vivado scripts.
    """
    grid_side = int(n_sites ** 0.5)
    if grid_side <= 0:
        return torch.zeros((n_sites, 2), dtype=torch.float32)

    logic_site_coords = torch.cartesian_prod(
        torch.arange(grid_side, dtype=torch.float32),
        torch.arange(max(n_sites // grid_side, 1), dtype=torch.float32),
    )

    if logic_site_coords.shape[0] != n_sites:
        logic_site_coords = torch.zeros((n_sites, 2), dtype=torch.float32)
        logic_site_coords[:, 0] = torch.arange(n_sites, dtype=torch.float32)

    return logic_site_coords
