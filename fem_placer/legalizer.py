import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .grid import Grid
from .placer import FpgaPlacer
from .logger import INFO, WARNING, ERROR

class Legalizer:

    def __init__(self,
                 placer,
                 device = 'cpu',
                 overlap_solver: str = 'greedy',
                 hungarian_distance_weight: float = 0.1,
                 hungarian_max_empty_sites: Optional[int] = None,
                 enable_importance_based_swapping: bool = False,
                 fast_first_improvement: bool = True,
                 hpwl_mode: str = 'cache'):
        self.placer: FpgaPlacer = placer
        self.device = device
        self.hpwl_mode = hpwl_mode  # 'cache' or 'hpwl'

        # Build grids from whatever regions the placer has
        self.grids: Dict[str, Grid] = {}
        for r in placer.regions:
            try:
                self.grids[r] = placer.get_grid(r)
            except Exception:
                WARNING(f"Legalizer: no grid for region '{r}', skipping")

        self.overlap_solver = overlap_solver
        self.hungarian_distance_weight = hungarian_distance_weight
        self.hungarian_max_empty_sites = hungarian_max_empty_sites
        self.enable_importance_based_swapping = enable_importance_based_swapping
        self.fast_first_improvement = fast_first_improvement

        # ---- Flat NumPy HPWL cache (bypasses NetManager entirely) ----
        self._pos = None            # np.ndarray [N, 2]  — mutable positions
        self._inst_to_nets = None   # List[List[int]]    — nets per instance
        self._net_to_insts = None   # List[List[int]]    — instances per net
        self._net_tensor = None     # np.ndarray [num_nets, num_insts] bool (hpwl_mode only)
        self._num_logic = 0         # first N ids are logic

    def legalize_placement(self, region_coords: Dict[str, torch.Tensor],
                           region_ids: Dict[str, torch.Tensor]):
        """
        Legalize placements for all regions.

        Args:
            region_coords: ``{region_name: coords_tensor [N, 2]}``
            region_ids: ``{region_name: ids_tensor [N]}``

        Returns:
            ``(legalized_coords, total_overlap, hpwl_before, hpwl_after)``
            where ``legalized_coords`` is a dict ``{region_name: tensor [N, 2]}``.
        """
        regions = [r for r in region_coords if r in self.grids and r in region_ids]

        # ---- Build the flat NumPy HPWL cache (bypasses NetManager) ----
        self._build_hpwl_cache(region_coords, region_ids)

        # ---- Stage 1: overlap resolution ----
        INFO(f"Stage 1: solve overlap")
        total_moved = 0
        for r in regions:
            self._load_coords_to_grids(self.grids[r], region_coords[r], region_ids[r])
            moved = self._resolve_grid_overlaps(self.grids[r], region_coords)
            total_moved += moved

        # Build legalized coords dict (from grid, not cache — grids are
        # the ground truth after overlap resolution).
        legalized = {}
        for r in regions:
            n = region_coords[r].shape[0]
            legalized[r] = self.grids[r].to_coords_tensor(n)

        # HPWL before / after (via cache)
        hpwl_before = self._cache_total_hpwl()
        self._sync_cache_from_legalized(legalized, region_ids)
        hpwl_after = self._cache_total_hpwl()
        INFO(f"Hpwl {hpwl_before:.2f} -> {hpwl_after:.2f}, moved {total_moved} instances")

        # ---- Stage 2: global optimization (cache-driven) ----
        INFO(f"Stage 2: global optimization")
        optimized = self._global_optimization(legalized, region_ids, iteration=3)
        hpwl_opt = self._cache_total_hpwl()
        INFO(f"Optimized Hpwl {hpwl_opt:.2f}, improve {hpwl_opt - hpwl_after:.2f}")

        # ---- Sync cache back to tensors ----
        optimized = self._sync_cache_to_region_coords(optimized, region_ids)

        return optimized, total_moved, hpwl_before, hpwl_opt

    def _load_coords_to_grids(self, grid: Grid, coords: torch.Tensor, ids: torch.Tensor):
        grid.clear_all()
        grid.from_coords_tensor(coords, ids)
        INFO(f"Loaded {len(ids)} instance to grid")

    def _resolve_grid_overlaps(self, grid: Grid, region_coords: Dict[str, torch.Tensor]) -> int:
        if self.overlap_solver == 'hungarian':
            return self._resolve_grid_overlaps_hungarian(grid)

        moved_count = 0
        conflict_groups = self._collect_conflict_groups(grid)
        sorted_conflicts = sorted(conflict_groups.items(), key=lambda x: len(x[1]), reverse=True)

        for conflict_pos, conflict_instances in sorted_conflicts:
            if len(conflict_instances) <= 1:
                continue
            success, num_moved = self._resolve_conflict_in_grid(
                grid, conflict_pos, conflict_instances
            )
            if success:
                moved_count += num_moved

        remaining_conflicts = self._check_remaining_overlaps(grid)
        if remaining_conflicts > 0:
            WARNING(f'{remaining_conflicts} conflicts are not resolved')
        return moved_count

    def _collect_conflict_groups(self, grid: Grid) -> Dict[Tuple[int, int], List[int]]:
        conflict_groups: Dict[Tuple[int, int], List[int]] = {}
        for instance_id, poz in grid.instance_positions.items():
            pos_tuple = tuple(poz)
            if pos_tuple in conflict_groups:
                conflict_groups[pos_tuple].append(instance_id)
            else:
                conflict_groups[pos_tuple] = [instance_id]
        return {pos: insts for pos, insts in conflict_groups.items() if len(insts) > 1}

    def _resolve_grid_overlaps_hungarian(self, grid):
        conflict_groups = self._collect_conflict_groups(grid)
        if not conflict_groups:
            return 0
        conflict_instances = []
        for instances in conflict_groups.values():
            if len(instances) > 1:
                conflict_instances.extend(instances[1:])
        if not conflict_instances:
            return 0
        empty_positions = list(grid._empty_positions)
        if not empty_positions:
            ERROR(f"Grid '{grid.name}' has no empty place")
            return 0
        if self.hungarian_max_empty_sites is not None and len(empty_positions) > self.hungarian_max_empty_sites:
            cx = sum(pos[0] for pos in conflict_groups) / max(1, len(conflict_groups))
            cy = sum(pos[1] for pos in conflict_groups) / max(1, len(conflict_groups))
            empty_positions = sorted(empty_positions, key=lambda p: abs(p[0]-cx)+abs(p[1]-cy))[:self.hungarian_max_empty_sites]
        if len(empty_positions) < len(conflict_instances):
            WARNING(f"Grid '{grid.name}' empty sites ({len(empty_positions)}) < conflict instances ({len(conflict_instances)})")
        candidate_xy = [(x, y) for x, y in empty_positions]
        n_inst, n_cand = len(conflict_instances), len(candidate_xy)
        if n_inst == 0 or n_cand == 0:
            return 0
        cost_matrix = np.zeros((n_inst, n_cand), dtype=np.float32)
        for i, inst_id in enumerate(conflict_instances):
            cp = grid.get_instance_position(inst_id)
            if cp is None:
                cost_matrix[i, :] = 1e6
                continue
            # _cache_delta_move_batch returns [new - old for each candidate]
            cand = self._cache_delta_move_batch(inst_id, candidate_xy)
            pen = np.array([abs(cx-cp[0])+abs(cy-cp[1]) for cx,cy in candidate_xy], dtype=np.float32) * self.hungarian_distance_weight
            cost_matrix[i, :] = np.asarray(cand, dtype=np.float32) + pen
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        moved = 0
        for ri, ci in zip(row_ind, col_ind):
            inst_id = conflict_instances[ri]
            tx, ty = candidate_xy[ci]
            cp = grid.get_instance_position(inst_id)
            if cp is None or (cp[0]==tx and cp[1]==ty):
                continue
            ok, _, _ = grid.move_instance(inst_id, tx, ty, swap_allowed=False)
            if ok:
                self._cache_apply_move(inst_id, tx, ty)
                moved += 1
        rem = self._check_remaining_overlaps(grid)
        if rem > 0:
            WARNING(f'{rem} conflicts not resolved after Hungarian stage')
        return moved

    def _check_remaining_overlaps(self, grid: Grid) -> int:
        pos_cnt = {}
        for _, pos in grid.instance_positions.items():
            t = tuple(pos)
            pos_cnt[t] = pos_cnt.get(t, 0) + 1
        rem = [(p, c) for p, c in pos_cnt.items() if c > 1]
        if rem:
            INFO(f" remain overlapped: {rem}")
        return len(rem)

    def _resolve_conflict_in_grid(self, grid, conflict_pos, conflict_instances):
        cx, cy = conflict_pos
        needed = len(conflict_instances) + 1
        empty = grid.find_empty_positions_nearby(cx, cy, needed)
        if len(empty) < needed - 1:
            ERROR(f"Grid '{grid.name}' has no empty place")
            return False, 0
        empty.insert(0, (cx, cy, 0))
        m, n_max = len(conflict_instances), min(len(empty), len(conflict_instances) + 3)
        cand_pos = empty[:n_max]
        cand_xy = [(x, y) for x, y, _ in cand_pos]
        cost = torch.zeros((m, n_max), device=self.device)
        for i, inst_id in enumerate(conflict_instances):
            # delta = HPWL_after - HPWL_before for each candidate
            deltas = self._cache_delta_move_batch(inst_id, cand_xy)
            for j in range(n_max):
                _, _, dist = cand_pos[j]
                cost[i, j] = float(deltas[j]) + dist * 0.1
        asgn = self._greedy_assignment(cost)
        moved = 0
        for i, j in enumerate(asgn):
            if j < 0:
                continue
            inst_id = conflict_instances[i]
            tx, ty, _ = empty[j]
            cp = grid.get_instance_position(inst_id)
            if cp and (cp[0]!=tx or cp[1]!=ty):
                ok, _, _ = grid.move_instance(inst_id, tx, ty, swap_allowed=True)
                if ok:
                    self._cache_apply_move(inst_id, tx, ty)
                    moved += 1
        return True, moved

    def _greedy_assignment(self, cost_matrix):
        m, n = cost_matrix.shape
        assigned_positions = set()
        assignment = [-1] * m
        for i in range(m):
            best = float('inf')
            best_j = -1
            for j in range(n):
                if j not in assigned_positions and cost_matrix[i, j] < best:
                    best = cost_matrix[i, j]
                    best_j = j
            if best_j != -1:
                assignment[i] = best_j
                assigned_positions.add(best_j)
        return assignment

    # ==================================================================
    #  Flat NumPy HPWL cache — bypasses NetManager entirely
    # ==================================================================

    def _build_hpwl_cache(self,
                          region_coords: Dict[str, torch.Tensor],
                          region_ids: Dict[str, torch.Tensor]):
        """Extract lightweight NumPy copies of all data needed for
        delta-HPWL computation, so the tight legalizer loops never touch
        ``NetManager`` or PyTorch tensors.

        Populates ``self._pos``, ``self._inst_to_nets``,
        ``self._net_to_insts``.
        """
        nm = self.placer.net_manager

        # ---- net name → integer id ----
        net_names = list(nm.net_to_sites.keys())
        net_name_to_id = {name: i for i, name in enumerate(net_names)}
        num_nets = len(net_names)

        # ---- total instance count (max id + 1) ----
        max_id = 0
        for ids in region_ids.values():
            t = ids if not torch.is_tensor(ids) else ids.cpu()
            if len(t):
                max_id = max(max_id, int(t.max().item()))
        num_insts = max_id + 1

        # ---- initial positions  [num_insts, 2] ----
        pos = np.full((num_insts, 2), np.nan, dtype=np.float64)
        for r in region_coords:
            coords_np = region_coords[r].cpu().numpy()
            ids_np = region_ids[r].cpu().numpy()
            for inst_id, (x, y) in zip(ids_np, coords_np):
                inst_id = int(inst_id)
                if inst_id < num_insts:
                    pos[inst_id] = [float(x), float(y)]

        # ---- instance → nets  &  net → instances ----
        inst_to_nets: List[List[int]] = [[] for _ in range(num_insts)]
        net_to_insts: List[List[int]] = [[] for _ in range(num_nets)]

        for net_name, sites in nm.net_to_sites.items():
            net_id = net_name_to_id.get(net_name)
            if net_id is None:
                continue
            for site in sites:
                inst_id = nm.get_site_inst_id_by_name_func(site)
                if inst_id is not None and 0 <= inst_id < num_insts:
                    inst_to_nets[inst_id].append(net_id)
                    net_to_insts[net_id].append(inst_id)

        # Purge nets with < 2 instances (they contribute zero HPWL)
        for net_id in range(num_nets):
            if len(net_to_insts[net_id]) < 2:
                net_to_insts[net_id] = []
                for inst_list in inst_to_nets:
                    try:
                        inst_list.remove(net_id)
                    except ValueError:
                        pass

        self._pos = pos
        self._inst_to_nets = inst_to_nets
        self._net_to_insts = net_to_insts
        self._num_logic = region_coords.get('logic', torch.empty(0)).shape[0]

        # hpwl_mode: store reference to NumPy net_tensor for connectivity queries
        if self.hpwl_mode == 'hpwl':
            self._net_tensor = nm.net_tensor
        else:
            self._net_tensor = None

    # ------------------------------------------------------------------
    #  Pure-Python / NumPy delta helpers (no PyTorch, no NetManager)
    # ------------------------------------------------------------------

    def _cache_net_hpwl(self, net_id: int) -> float:
        """HPWL of a single net from the current cache positions."""
        insts = self._net_to_insts[net_id]
        n = len(insts)
        if n < 2:
            return 0.0
        xs = self._pos[insts, 0]   # NumPy fancy indexing
        ys = self._pos[insts, 1]
        return float((xs.max() - xs.min()) + (ys.max() - ys.min()))

    def _cache_delta_move(self, inst_id: int,
                          new_x: float, new_y: float) -> float:
        """Delta HPWL = HPWL_after − HPWL_before for moving *inst_id*
        to ``(new_x, new_y)``.  Negative means improvement."""
        if self.hpwl_mode == 'hpwl':
            return self._hpwl_delta_move(inst_id, new_x, new_y)

        delta = 0.0

        for net_id in self._inst_to_nets[inst_id]:
            insts = self._net_to_insts[net_id]
            if len(insts) < 2:
                continue

            # Old HPWL
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())

            # New HPWL — replace inst_id's coords
            idx = insts.index(inst_id)
            xs_new = xs.copy()
            ys_new = ys.copy()
            xs_new[idx] = new_x
            ys_new[idx] = new_y
            new_hpwl = (xs_new.max() - xs_new.min()) + (ys_new.max() - ys_new.min())

            delta += new_hpwl - old_hpwl

        return delta

    def _cache_delta_move_batch(self, inst_id: int,
                                candidates: List[Tuple[float, float]]) -> np.ndarray:
        """Delta HPWL for *inst_id* at each candidate position.
        Returns ``np.ndarray [len(candidates)]``."""
        if self.hpwl_mode == 'hpwl':
            return self._hpwl_delta_move_batch(inst_id, candidates)

        n_cand = len(candidates)
        if n_cand == 0:
            return np.array([], dtype=np.float64)

        deltas = np.zeros(n_cand, dtype=np.float64)

        for net_id in self._inst_to_nets[inst_id]:
            insts = self._net_to_insts[net_id]
            if len(insts) < 2:
                continue

            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())

            idx = insts.index(inst_id)
            # Vectorised over candidates
            for c, (nx, ny) in enumerate(candidates):
                xc, yc = xs.copy(), ys.copy()
                xc[idx] = nx
                yc[idx] = ny
                new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
                deltas[c] += new_hpwl - old_hpwl

        return deltas

    def _cache_delta_swap(self, a: int, b: int) -> float:
        """Delta HPWL for swapping instances *a* and *b*.
        Correctly handles nets that contain both instances."""
        if self.hpwl_mode == 'hpwl':
            return self._hpwl_delta_swap(a, b)

        ax, ay = self._pos[a]
        bx, by = self._pos[b]

        nets_a = set(self._inst_to_nets[a])
        nets_b = set(self._inst_to_nets[b])

        only_a = nets_a - nets_b
        only_b = nets_b - nets_a
        both   = nets_a & nets_b

        delta = 0.0

        # ---- nets only on a: a moves to b's position ----
        for net_id in only_a:
            insts = self._net_to_insts[net_id]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            idx = insts.index(a)
            xc, yc = xs.copy(), ys.copy()
            xc[idx] = bx
            yc[idx] = by
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl

        # ---- nets only on b: b moves to a's position ----
        for net_id in only_b:
            insts = self._net_to_insts[net_id]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            idx = insts.index(b)
            xc, yc = xs.copy(), ys.copy()
            xc[idx] = ax
            yc[idx] = ay
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl

        # ---- nets containing both: simultaneous swap ----
        for net_id in both:
            insts = self._net_to_insts[net_id]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())

            xc, yc = xs.copy(), ys.copy()
            idx_a = insts.index(a)
            idx_b = insts.index(b)
            xc[idx_a], yc[idx_a] = bx, by
            xc[idx_b], yc[idx_b] = ax, ay
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl

        return delta

    # ------------------------------------------------------------------
    #  Cache mutation helpers (call after every accepted move / swap)
    # ------------------------------------------------------------------

    def _cache_apply_move(self, inst_id: int, nx: float, ny: float):
        self._pos[inst_id] = [nx, ny]

    def _cache_apply_swap(self, a: int, b: int):
        pa = self._pos[a].copy()
        pb = self._pos[b].copy()
        self._pos[a] = pb
        self._pos[b] = pa

    # ------------------------------------------------------------------
    #  hpwl_mode helpers — use net_tensor (NumPy bool matrix) for
    #  connectivity lookup instead of the cache's list-of-lists.
    #  Mimics the original NetManager query style.
    # ------------------------------------------------------------------

    def _hpwl_delta_move(self, inst_id: int,
                         new_x: float, new_y: float) -> float:
        delta = 0.0
        nt = self._net_tensor
        if nt is None:
            return delta
        # All nets connected to this instance
        net_ids = np.where(nt[:, inst_id])[0]
        for net_id in net_ids:
            insts = np.where(nt[net_id])[0]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            idx = int(np.where(insts == inst_id)[0][0])
            xc, yc = xs.copy(), ys.copy()
            xc[idx] = new_x
            yc[idx] = new_y
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl
        return delta

    def _hpwl_delta_move_batch(self, inst_id: int,
                                candidates: List[Tuple[float, float]]) -> np.ndarray:
        n_cand = len(candidates)
        if n_cand == 0:
            return np.array([], dtype=np.float64)
        deltas = np.zeros(n_cand, dtype=np.float64)
        nt = self._net_tensor
        if nt is None:
            return deltas
        net_ids = np.where(nt[:, inst_id])[0]
        for net_id in net_ids:
            insts = np.where(nt[net_id])[0]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            idx = int(np.where(insts == inst_id)[0][0])
            for c, (nx, ny) in enumerate(candidates):
                xc, yc = xs.copy(), ys.copy()
                xc[idx] = nx
                yc[idx] = ny
                new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
                deltas[c] += new_hpwl - old_hpwl
        return deltas

    def _hpwl_delta_swap(self, a: int, b: int) -> float:
        nt = self._net_tensor
        if nt is None:
            return 0.0
        ax, ay = self._pos[a]
        bx, by = self._pos[b]
        nets_a = set(np.where(nt[:, a])[0].tolist())
        nets_b = set(np.where(nt[:, b])[0].tolist())
        only_a = nets_a - nets_b
        only_b = nets_b - nets_a
        both   = nets_a & nets_b
        delta = 0.0
        for net_id in only_a:
            insts = np.where(nt[net_id])[0]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            idx = int(np.where(insts == a)[0][0])
            xc, yc = xs.copy(), ys.copy()
            xc[idx] = bx; yc[idx] = by
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl
        for net_id in only_b:
            insts = np.where(nt[net_id])[0]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            idx = int(np.where(insts == b)[0][0])
            xc, yc = xs.copy(), ys.copy()
            xc[idx] = ax; yc[idx] = ay
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl
        for net_id in both:
            insts = np.where(nt[net_id])[0]
            if len(insts) < 2:
                continue
            xs = self._pos[insts, 0]
            ys = self._pos[insts, 1]
            old_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            xc, yc = xs.copy(), ys.copy()
            idx_a = int(np.where(insts == a)[0][0])
            idx_b = int(np.where(insts == b)[0][0])
            xc[idx_a] = bx; yc[idx_a] = by
            xc[idx_b] = ax; yc[idx_b] = ay
            new_hpwl = (xc.max() - xc.min()) + (yc.max() - yc.min())
            delta += new_hpwl - old_hpwl
        return delta

    # ------------------------------------------------------------------
    #  Sync back to tensors once legalization is done
    # ------------------------------------------------------------------

    def _sync_cache_to_region_coords(
            self,
            region_coords: Dict[str, torch.Tensor],
            region_ids: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Overwrite tensors with final cache positions and return the
        updated dict."""
        for r in region_coords:
            ids_np = region_ids[r].cpu().numpy()
            out = torch.zeros_like(region_coords[r])
            for i, inst_id in enumerate(ids_np):
                inst_id = int(inst_id)
                out[i, 0] = float(self._pos[inst_id, 0])
                out[i, 1] = float(self._pos[inst_id, 1])
            region_coords[r] = out
        return region_coords

    def _cache_total_hpwl(self) -> float:
        """Total HPWL over all nets from the current cache positions."""
        total = 0.0
        for net_id in range(len(self._net_to_insts)):
            total += self._cache_net_hpwl(net_id)
        return total

    def _sync_cache_from_legalized(
            self,
            legalized: Dict[str, torch.Tensor],
            region_ids: Dict[str, torch.Tensor]):
        """Update the cache positions from the legalized grid tensors
        (used after Stage 1 overlap resolution)."""
        for r in legalized:
            coords_np = legalized[r].cpu().numpy()
            ids_np = region_ids[r].cpu().numpy()
            for inst_id, (x, y) in zip(ids_np, coords_np):
                inst_id = int(inst_id)
                if inst_id < self._pos.shape[0]:
                    self._pos[inst_id] = [float(x), float(y)]

    def _global_optimization(self, legalized: Dict[str, torch.Tensor],
                             region_ids: Dict[str, torch.Tensor],
                             iteration: int = 3) -> Dict[str, torch.Tensor]:
        for _ in range(iteration):
            improved = False
            for r in legalized:
                if r in self.grids:
                    ok = self._optimize_grid_instances(
                        self.grids[r], legalized[r].shape[0]
                    )
                    if ok:
                        improved = True
            if not improved:
                break
        optimized = {}
        for r in legalized:
            if r in self.grids:
                optimized[r] = self.grids[r].to_coords_tensor(legalized[r].shape[0])
        return optimized

    def _optimize_grid_instances(self, grid: Grid, num_instances: int) -> bool:
        improved = False
        if self.enable_importance_based_swapping:
            critical_instances = self._select_critical_instances_for_grid(
                grid, num_instances
            )
            for instance_id in critical_instances:
                success, improvement = self._optimize_instance_in_grid_importance_aware(
                    grid, instance_id
                )
                if success and improvement > 0:
                    improved = True
        else:
            # 快速贪心优化（速度快）
            for instance_id in list(grid.instance_positions.keys())[:num_instances]:
                success, improvement = self._optimize_instance_in_grid_fast(
                    grid, instance_id
                )
                if success and improvement > 0:
                    improved = True
                    if self.fast_first_improvement:
                        break

        return improved

    def _compute_instance_connectivity(self, instance_id: int) -> float:
        """Return the net-degree (connectivity) score for *instance_id*.

        Uses ``self._net_tensor`` (hpwl_mode) or falls back to
        ``NetManager.net_tensor`` (both NumPy)."""
        nt = self._net_tensor
        if nt is None:
            nt = self.placer.net_manager.net_tensor
        if nt is None or instance_id >= nt.shape[1]:
            return 0.0
        return float(nt[:, instance_id].sum())

    def _select_critical_instances_for_grid(self, grid: Grid, num_instances: int) -> List[int]:
        """为指定网格选择关键实例 (top 20% by connectivity)"""
        # 如果网格中没有实例，返回空列表
        if not grid.instance_positions:
            return []

        # 根据连接度选择关键实例
        instances_in_grid = list(grid.instance_positions.keys())
        connectivity_scores = []

        for inst_id in instances_in_grid:
            connectivity = self._compute_instance_connectivity(inst_id)
            connectivity_scores.append((connectivity, inst_id))

        if not connectivity_scores:
            return []

        # 按连接度降序排序
        connectivity_scores.sort(reverse=True)
        
        # 选择top 20%作为关键实例
        top_k = max(1, len(connectivity_scores) // 5)
        top_k = min(top_k, len(connectivity_scores))
        
        return [inst_id for _, inst_id in connectivity_scores[:top_k]]

    def _optimize_instance_in_grid_fast(self, grid: Grid, instance_id: int) -> Tuple[bool, float]:
        """快速贪心优化: 仅考虑邻域空位, 不做交换 (使用NumPy cache)"""
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0

        cx, cy = current_pos
        search_radius = 2 if grid.name == 'logic' else 1

        # 搜索邻域并收集空位候选
        candidate_xy: List[Tuple[int, int]] = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = cx + dx, cy + dy
                if not grid.is_within_bounds(new_x, new_y):
                    continue
                if grid.is_position_empty(new_x, new_y):
                    candidate_xy.append((new_x, new_y))

        if not candidate_xy:
            return False, 0.0

        # Batch compute deltas via cache (pure NumPy, no NetManager)
        deltas = self._cache_delta_move_batch(instance_id, candidate_xy)
        best_idx = int(np.argmin(deltas))
        best_delta = deltas[best_idx]

        if best_delta < 0:  # negative = improvement
            tx, ty = candidate_xy[best_idx]
            improvement = -best_delta
            success, _, _ = grid.move_instance(instance_id, tx, ty, swap_allowed=False)
            if success:
                self._cache_apply_move(instance_id, tx, ty)
                return success, improvement

        return False, 0.0

    def _optimize_instance_in_grid_importance_aware(self, grid: Grid,
                                                    instance_id: int) -> Tuple[bool, float]:
        """Importance-aware optimization using NumPy cache (no NetManager).

        Strategy:
        1. Prefer empty slots.
        2. Consider swaps with lower-connectivity instances.
        3. Only execute when total HPWL improves.
        """
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0

        cx, cy = current_pos
        best_pos = current_pos
        best_improvement = 0.0
        best_swap_candidate = None

        # Connectivity for pruning
        instance_connectivity = self._compute_instance_connectivity(instance_id)
        search_radius = 3 if grid.name == 'logic' else 1

        # Collect neighbours: empty positions and lower-connectivity occupied
        empty_positions: List[Tuple[int, int]] = []
        occupied_positions: List[Tuple[int, int, int]] = []  # (x, y, swap_id)

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if abs(dx) + abs(dy) > search_radius:
                    continue

                nx, ny = cx + dx, cy + dy
                if not grid.is_within_bounds(nx, ny):
                    continue

                occupants = grid.get_position_occupants(nx, ny)
                if not occupants:
                    empty_positions.append((nx, ny))
                elif occupants[0] != instance_id:
                    swap_conn = self._compute_instance_connectivity(occupants[0])
                    if swap_conn < instance_connectivity:
                        occupied_positions.append((nx, ny, occupants[0]))

        # ---- Evaluate empty slots (batch) ----
        if empty_positions:
            deltas = self._cache_delta_move_batch(instance_id, empty_positions)
            for (nx, ny), d in zip(empty_positions, deltas):
                imp = -d  # delta = new - old, so -delta = improvement
                if imp > best_improvement:
                    best_improvement = imp
                    best_pos = (nx, ny)
                    best_swap_candidate = None

        # ---- Evaluate swap candidates (batch via _cache_delta_swap) ----
        for nx, ny, swap_id in occupied_positions:
            d = self._cache_delta_swap(instance_id, swap_id)  # delta = after - before
            imp = -d
            if imp > best_improvement:
                best_improvement = imp
                best_pos = (nx, ny)
                best_swap_candidate = swap_id

        # ---- Apply best move / swap ----
        if best_improvement > 0 and best_pos != current_pos:
            tx, ty = best_pos
            if best_swap_candidate is not None:
                success, _, _ = grid.move_instance(instance_id, tx, ty, swap_allowed=True)
                if success:
                    self._cache_apply_swap(instance_id, best_swap_candidate)
                    return True, best_improvement
            else:
                success, _, _ = grid.move_instance(instance_id, tx, ty, swap_allowed=False)
                if success:
                    self._cache_apply_move(instance_id, tx, ty)
                    return True, best_improvement

        return False, 0.0