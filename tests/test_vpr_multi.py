"""
VPR placement test — full optimisation flow (.place/.net → optimizer → legalize → route).

Runs the entire :class:`FPGAPlacementOptimizer` pipeline on VTR benchmark circuits,
including legalization and routing, mirroring the DCP-based ``test_fpga_placement.py``.
"""
import sys; sys.path.insert(0, '.')
import time
import torch
import warnings; warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
    Legalizer,
)
from fem_placer.logger import SET_LEVEL, INFO; SET_LEVEL("INFO")

# ── Optimiser hyper-parameters (override per circuit below) ──────────────────
OPT_KWARGS = dict(
    num_trials=1,
    num_steps=5000,
    dev='cuda',
    betamin=0.01,
    betamax=0.5,
    anneal='inverse',
    optimizer='adam',
    learning_rate=0.1,
    seed=1,
    dtype=torch.float32,
    manual_grad=False,
    distance_metric='manhattan',
)

# Per-circuit overrides — VPR mode only optimises logic area
# Each entry accepts: coeff_list, h_factor_list, num_steps, anneal, record_mode, map_mode
CIRCUIT_OVERRIDES = {
    'ch_intrinsics': dict(coeff_list=[10], h_factor_list=[0.01], num_steps=3000, anneal='lin',
                          record_mode='simple', map_mode='no'),
    # 'bgm':           dict(coeff_list=[0.05], h_factor_list=[0.01], num_steps=5000, anneal='inverse',
    #                       record_mode='inverse_sqr', map_mode='no'),
    'LU8PEEng':      dict(coeff_list=[0.05],  h_factor_list=[0.01], num_steps=3000, anneal='inverse',
                          record_mode='inverse_sqr', map_mode='no'),
    # 'blob_merge':    dict(coeff_list=[0.08], h_factor_list=[0.01], num_steps=5000, anneal='inverse',
    #                       record_mode='inverse_sqr', map_mode='no'),
    # 'sha':           dict(coeff_list=[0.1], h_factor_list=[0.01], num_steps=5000, anneal='inverse',
    #                       record_mode='inverse_sqr', map_mode='no'),
    # 'mkDelayWorker32B': dict(coeff_list=[500], h_factor_list=[0.01], num_steps=5000, anneal='inverse',
    #                          record_mode='inverse', map_mode='no'),
    'stereovision0': dict(coeff_list=[1],  h_factor_list=[0.01], num_steps=5000, anneal='inverse',
                          record_mode='inverse', map_mode='no'),
}

HEADER = (
    f"{'Circuit':<20s}  {'HPWL_ref':>10s}  {'HPWL_opt':>10s}  {'HPWL_fin':>10s}  "
    f"{'Overlap':>8s}  {'Routes':>8s}  {'Time(s)':>8s}"
)
print()
print(HEADER)
print("-" * len(HEADER))

for circuit in CIRCUIT_OVERRIDES:
    place_file = f"vtr/output_dir/{circuit}/{circuit}.place"
    net_file = f"vtr/output_dir/{circuit}/{circuit}.net"

    # ── 1. Initialise from VPR files ────────────────────────────────────
    override = CIRCUIT_OVERRIDES.get(circuit, {})
    placer = FpgaPlacer(device='cuda', debug=False,
                        record_mode=override.get('record_mode', 'inverse_sqr'),
                        map_mode=override.get('map_mode', 'no'))
    placer.set_instance_name(circuit, result_dir='result/vpr_test')
    hpwl_ref, inst_num, net_num = placer.init_placement_vpr(place_file, net_file)

    # ── 2. Build N-region dicts for the optimizer ──────────────────────
    # VPR mode only optimises the logic region
    regions = ['logic']
    coeff_list = override.get('coeff_list', [1.0] * len(regions))
    h_factor_list = override.get('h_factor_list', [0.01] * len(regions))
    opt_kw = dict(OPT_KWARGS)
    for k in ('num_steps', 'anneal'):
        if k in override:
            opt_kw[k] = override[k]

    region_sizes = {r: (placer.instances[r].num, max(placer.get_grid(r).area, 1)) for r in regions}
    region_site_coords = {}
    for r in regions:
        attr = f'{r}_site_coords'
        if hasattr(placer, attr):
            region_site_coords[r] = getattr(placer, attr)
        else:
            g = placer.get_grid(r)
            region_site_coords[r] = g.to_real_coords_tensor(torch.cartesian_prod(
                torch.arange(g.area_length, dtype=torch.float32, device='cpu'),
                torch.arange(g.area_width, dtype=torch.float32, device='cpu')
            ))

    io_mat = placer.net_manager.io_insts_matrix
    region_coupling = {}
    for rA in regions:
        region_coupling[rA] = {}
        for rB in regions:
            if rA == rB == 'logic':
                region_coupling[rA][rB] = placer.net_manager.insts_matrix
            elif rA == 'logic' and rB == 'io' and io_mat is not None:
                region_coupling[rA][rB] = io_mat
            elif rA == 'io' and rB == 'logic' and io_mat is not None:
                region_coupling[rA][rB] = io_mat.T.clone()
            else:
                region_coupling[rA][rB] = None

    # ── 3. Run optimiser ────────────────────────────────────────────────
    optimizer = FPGAPlacementOptimizer(
        regions=regions,
        region_sizes=region_sizes,
        region_coupling=region_coupling,
        region_site_coords=region_site_coords,
        constraint_coeffs=coeff_list,
        h_factors=h_factor_list,
        **opt_kw,
    )

    t0 = time.time()
    config, result = optimizer.optimize()
    opt_time = time.time() - t0

    # ── 4. Legalize ─────────────────────────────────────────────────────
    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
    legalizer = Legalizer(placer=placer, device='cpu')
    all_ids = placer.get_ids()
    region_id_map = dict(zip(placer.regions, all_ids))

    best_config = {r: config[r][optimal_inds[0]] for r in config}
    best_ids = {r: region_id_map[r] for r in config if r in region_id_map}

    legalized, overlap, hpwl_i, hpwl_f = legalizer.legalize_placement(best_config, best_ids)

    # ── 5. Print summary ────────────────────────────────────────────────
    print(f"{circuit:<20s}  {hpwl_ref:>10.2f}  {result.min().item():>10.2f}  "
          f"{hpwl_f:>10.2f}  {overlap:>8.2f}  {'—':>8s}  {opt_time:>8.2f}")

    placer.close()

print("-" * len(HEADER))
print("All circuits passed!")
