"""
Standalone plotting script for mode-anneal benchmark results.

Reads `mode_anneal_best_results.csv` and generates a two-panel grouped bar chart:
  - (a): HPWL
  - (b): Runtime (seconds)
"""

import argparse
import csv
import os
from typing import Any, Dict, List, Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot mode-anneal benchmark results from CSV')
    parser.add_argument(
        '--csv',
        default='./result/final_best_results.csv',
        help='Path to best-results CSV file',
    )
    parser.add_argument(
        '--out',
        default='./result/mode_anneal_best_grouped_bars.png',
        help='Output image path',
    )
    parser.add_argument(
        '--font-path',
        default='/usr/share/fonts/opentype/linux-libertine/LinLibertine_RI.otf',
        help='Path to font file (optional)',
    )
    parser.add_argument(
        '--bar-width',
        type=float,
        default=0.12,
        help='Fixed bar width for grouped bars',
    )
    return parser.parse_args()


def load_best_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    instances: List[str] = []
    modes: List[str] = []
    by_instance: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    vivado_hpwl_by_instance: Dict[str, float] = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row.get('instance', '').strip()
            mode = row.get('mode', '').strip()
            best_anneal = row.get('best_anneal', '').strip()

            if not instance or not mode:
                continue

            if instance not in by_instance:
                by_instance[instance] = {}
                instances.append(instance)
            if mode not in modes:
                modes.append(mode)

            if best_anneal == 'N/A':
                by_instance[instance][mode] = None
                vivado_hpwl_str = str(row.get('vivado_hpwl', '')).strip()
                if vivado_hpwl_str and instance not in vivado_hpwl_by_instance:
                    try:
                        vivado_hpwl_by_instance[instance] = float(vivado_hpwl_str)
                    except ValueError:
                        pass
                continue

            try:
                hpwl_final = float(row.get('hpwl_final', 'nan'))
                runtime_s = float(row.get('runtime_s', 'nan'))
            except ValueError:
                by_instance[instance][mode] = None
                continue

            by_instance[instance][mode] = {
                'hpwl_final': hpwl_final,
                'runtime_s': runtime_s,
                'best_anneal': best_anneal,
            }

            vivado_hpwl_str = str(row.get('vivado_hpwl', '')).strip()
            if vivado_hpwl_str and instance not in vivado_hpwl_by_instance:
                try:
                    vivado_hpwl_by_instance[instance] = float(vivado_hpwl_str)
                except ValueError:
                    pass

    return instances, modes, by_instance, vivado_hpwl_by_instance


def plot_best_grouped_bars(instances: List[str],
                           mode_names: List[str],
                           best_by_instance: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
                           vivado_hpwl_by_instance: Dict[str, float],
                           vivado_runtime_by_instance: Dict[str, float],
                           output_path: str,
                           font_path: str,
                           bar_width: float):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.rcParams['font.size'] = 17
    if font_path:
        try:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
        except Exception:
            pass

    x = np.arange(len(instances), dtype=float)
    preferred_mode_order = ['simple', 'inverse-sqr', 'inverse-fanout']
    ordered_modes = [mode for mode in preferred_mode_order if mode in mode_names]
    ordered_modes.extend([mode for mode in mode_names if mode not in ordered_modes])

    hpwl_color_set = ["#F8C48C", "#4BA9D9", "#C198D6", "#D97571"]
    time_color_set = ['#DCC8F0', '#BDF9F9', '#FFE5C2', '#FFD0CE']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=150)

    total_group_bars = len(ordered_modes) + 1  # +1 for Vivado
    center_shift = (total_group_bars - 1) / 2.0

    for idx, mode_name in enumerate(ordered_modes):
        shift = (idx - center_shift) * bar_width
        hpwl_vals = []
        time_vals = []

        for instance in instances:
            row = best_by_instance.get(instance, {}).get(mode_name)
            hpwl_vals.append(np.nan if row is None else row['hpwl_final'])
            time_vals.append(np.nan if row is None else row['runtime_s'])

        hpwl_color = hpwl_color_set[idx] if idx < len(hpwl_color_set) else hpwl_color_set[-1]
        time_color = time_color_set[idx] if idx < len(time_color_set) else time_color_set[-1]
        ax1.bar(
            x + shift,
            hpwl_vals,
            width=bar_width,
            label=mode_name,
            color=hpwl_color,
            alpha=0.85,
            edgecolor='white',
            linewidth=1.5,
            capstyle='round',
            joinstyle='round',
        )
        ax2.bar(
            x + shift,
            time_vals,
            width=bar_width,
            label=mode_name,
            color=time_color,
            alpha=0.85,
            edgecolor='white',
            linewidth=1.5,
            capstyle='round',
            joinstyle='round',
        )

    vivado_shift = (len(ordered_modes) - center_shift) * bar_width
    vivado_x = x + vivado_shift
    vivado_vals = [vivado_hpwl_by_instance.get(instance, np.nan) for instance in instances]
    ax1.bar(
        vivado_x,
        vivado_vals,
        width=bar_width,
        color='lightgray',
        edgecolor='white',
        linewidth=1.5,
        hatch='*',
        label='vivado',
        alpha=0.85,
        zorder=0,
    )

    vivado_runtime_vals = [vivado_runtime_by_instance.get(instance, np.nan) for instance in instances]
    ax2.bar(
        vivado_x,
        vivado_runtime_vals,
        width=bar_width,
        color='lightgray',
        edgecolor='white',
        linewidth=1.5,
        hatch='*',
        label='vivado',
        alpha=0.85,
        zorder=4,
    )

    instance_labels = []
    for instance in instances:
        display_name = instance.replace('_boundary', '')
        best_row = None
        for mode_name in ordered_modes:
            row = best_by_instance.get(instance, {}).get(mode_name)
            if row is None:
                continue
            if best_row is None:
                best_row = row
                continue
            if row['hpwl_final'] < best_row['hpwl_final']:
                best_row = row
            elif row['hpwl_final'] == best_row['hpwl_final'] and row['runtime_s'] < best_row['runtime_s']:
                best_row = row

        if best_row is not None:
            best_anneal = str(best_row.get('best_anneal', '')).strip()
            if best_anneal and best_anneal != 'N/A':
                display_name = f"{display_name} ({best_anneal})"

        instance_labels.append(display_name)

    ax1.set_title('(a)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('HPWL', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xlim(-0.5, len(instances) - 0.5)
    ax1.set_xticklabels(instance_labels, rotation=30, ha='right', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('gray')
    ax1.spines['bottom'].set_linewidth(0.8)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.tick_params(axis='y', which='both', length=0, colors='gray')

    ax2.set_title('(b)', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylabel('Seconds', fontsize=14, fontweight='bold')
    # ax2.set_xlabel('Instance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xlim(-0.5, len(instances) - 0.5)
    ax2.set_xticklabels(instance_labels, rotation=30, ha='right', fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['bottom'].set_linewidth(0.8)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.tick_params(axis='y', which='both', length=0, colors='gray')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles,
        labels,
        loc='upper left',
        frameon=True,
        fontsize=12,
        framealpha=0.95,
        edgecolor='black',
        fancybox=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    instances, modes, best_by_instance, vivado_hpwl_by_instance = load_best_csv(args.csv)
    vivado_runtime_by_instance: Dict[str, float] = {}

    if not instances:
        raise RuntimeError(f'No valid rows found in CSV: {args.csv}')

    plot_best_grouped_bars(
        instances=instances,
        mode_names=modes,
        best_by_instance=best_by_instance,
        vivado_hpwl_by_instance=vivado_hpwl_by_instance,
        vivado_runtime_by_instance=vivado_runtime_by_instance,
        output_path=args.out,
        font_path=args.font_path,
        bar_width=args.bar_width,
    )

    print(f'Plot saved: {args.out}')


if __name__ == '__main__':
    main()