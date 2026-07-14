#!/usr/bin/env python3
"""
合并 rebuttal_result_draw_v1 和 plot_mode_anneal_results 的图表。
生成：
  - combined_figure1.png：四个子图 (a,b,c,d) 宽度比 2:1:1
  - combined_figure2.png：两个子图 (b,d)（同原 rebuttal 的 plot_bd）
如果 CSV 文件不存在，test 部分留空。
"""

import os
import csv
import sys
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ============================================================
# 1. 硬编码的 rebuttal 数据（与原文一致）
# ============================================================

vpr_short = ['bgm', 'LU8', 'blob', 'sha', 'mkDelay', 'stereo']
vpr_hpwl = [25104, 199096, 47895, 15194, 82570, 53979]
mofem_vpr = [23979, 173251, 46089, 12991, 57566, 52161]

timing_short = ['s15850', 's38417', 'ch_int', 'bgm', 'blob', 'sha', 'mkDelay']
vivado_wns = [6.186, 5.364, 8.905, 3.513, 4.091, 4.354, 3.411]
vivado_fmax = [262, 215, 300, 154, 111, 177, 151]
mofem_wns   = [4.399, 5.178, 7.649, 3.071, 4.585, 4.012, 5.423]
mofem_fmax  = [197, 207, 300, 143, 185, 167, 218]

legal_inst = vpr_short
pre_hpwl  = [105947, 121375, 41538, 6201, 21391, 32103]
post_hpwl = [105944, 120489, 40417, 6116, 21317, 32019]
reduction_pct = [(p - q) / p * 100 for p, q in zip(pre_hpwl, post_hpwl)]

runtime_short = vpr_short
prep_r = [1.02, 0.77, 0.84, 0.84, 0.90, 0.71]
gpu_r  = [32.42, 241.49, 12.58, 8.39, 8.40, 22.22]
infer_r= [1.68, 3.08, 0.05, 0.02, 0.03, 0.08]
legal_r= [1.45, 1.28, 1.03, 0.10, 1.36, 0.82]

colors = ["#C99FF4", "#98CCF7", "#FDCD8E", "#83F78F", "#7863F3", "#F67059"]
ylabel_size = 10
tick_size = 9
legend_size = 10
title_size = 13
xtick_size = 9

def setup_font():
    try:
        lb_path = fm.findfont('Libertinus Sans')
        if lb_path:
            prop = fm.FontProperties(fname=lb_path)
            plt.rcParams['font.family'] = prop.get_name()
    except:
        pass
    plt.rcParams['font.size'] = 11

def apply_style(ax, xlabel='', ylabel=''):
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0, colors='gray')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight='bold')

# ============================================================
# 2. 加载 test_annealing_comparison 数据（从 CSV）
#    如果 CSV 缺失，返回空数据，后续绘图留空。
# ============================================================

def load_best_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return [], [], {}, {}

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
                vivado_str = str(row.get('vivado_hpwl', '')).strip()
                if vivado_str and instance not in vivado_hpwl_by_instance:
                    try:
                        vivado_hpwl_by_instance[instance] = float(vivado_str)
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

            vivado_str = str(row.get('vivado_hpwl', '')).strip()
            if vivado_str and instance not in vivado_hpwl_by_instance:
                try:
                    vivado_hpwl_by_instance[instance] = float(vivado_str)
                except ValueError:
                    pass

    return instances, modes, by_instance, vivado_hpwl_by_instance

# ============================================================
# 3. 绘制各子图的函数（test 部分会检查数据是否为空）
# ============================================================

def draw_test_hpwl(ax, instances, modes, by_instance, vivado_hpwl):
    if not instances:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('HPWL', fontsize=ylabel_size, fontweight='bold')
        apply_style(ax)
        return

    x = np.arange(len(instances))
    preferred_mode_order = ['simple', 'inverse-sqr', 'inverse-fanout']
    ordered_modes = [m for m in preferred_mode_order if m in modes]
    ordered_modes.extend([m for m in modes if m not in ordered_modes])

    bar_width = 0.12
    total_group_bars = len(ordered_modes) + 1
    center_shift = (total_group_bars - 1) / 2.0
    hpwl_color_set = ["#F8C48C", "#4BA9D9", "#C198D6", "#D97571"]

    for idx, mode_name in enumerate(ordered_modes):
        shift = (idx - center_shift) * bar_width
        hpwl_vals = []
        for inst in instances:
            row = by_instance.get(inst, {}).get(mode_name)
            hpwl_vals.append(np.nan if row is None else row['hpwl_final'])
        color = hpwl_color_set[idx % len(hpwl_color_set)]
        ax.bar(x + shift, hpwl_vals, width=bar_width,
               label=mode_name, color=color, alpha=0.85,
               edgecolor='white', linewidth=1.5)

    vivado_shift = (len(ordered_modes) - center_shift) * bar_width
    vivado_x = x + vivado_shift
    vivado_vals = [vivado_hpwl.get(inst, np.nan) for inst in instances]
    ax.bar(vivado_x, vivado_vals, width=bar_width,
           label='Vivado', color='lightgray', alpha=0.85,
           edgecolor='white', linewidth=1.5, hatch='*', zorder=0)

    labels = []
    for inst in instances:
        display = inst.replace('_boundary', '')
        best_row = None
        for mode_name in ordered_modes:
            row = by_instance.get(inst, {}).get(mode_name)
            if row is None:
                continue
            if best_row is None or row['hpwl_final'] < best_row['hpwl_final']:
                best_row = row
        if best_row and best_row.get('best_anneal') and best_row['best_anneal'] != 'N/A':
            display = f"{display} ({best_row['best_anneal']})"
        labels.append(display)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('HPWL', fontsize=ylabel_size, fontweight='bold')
    apply_style(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.legend(loc='upper left', fontsize=legend_size, framealpha=0.95,
              edgecolor='black', fancybox=False, ncol=2)
    ax.set_xlim(-0.5, len(instances)-0.5)

def draw_test_runtime(ax, instances, modes, by_instance):
    if not instances:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Seconds', fontsize=ylabel_size, fontweight='bold')
        apply_style(ax)
        return

    x = np.arange(len(instances))
    preferred_mode_order = ['simple', 'inverse-sqr', 'inverse-fanout']
    ordered_modes = [m for m in preferred_mode_order if m in modes]
    ordered_modes.extend([m for m in modes if m not in ordered_modes])

    bar_width = 0.12
    total_group_bars = len(ordered_modes)
    center_shift = (total_group_bars - 1) / 2.0
    time_color_set = ['#DCC8F0', '#BDF9F9', '#FFE5C2', '#FFD0CE']

    for idx, mode_name in enumerate(ordered_modes):
        shift = (idx - center_shift) * bar_width
        time_vals = []
        for inst in instances:
            row = by_instance.get(inst, {}).get(mode_name)
            time_vals.append(np.nan if row is None else row['runtime_s'])
        color = time_color_set[idx % len(time_color_set)]
        ax.bar(x + shift, time_vals, width=bar_width,
               label=mode_name, color=color, alpha=0.85,
               edgecolor='white', linewidth=1.5)

    labels = [inst.replace('_boundary', '') for inst in instances]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('Seconds', fontsize=ylabel_size, fontweight='bold')
    apply_style(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.legend(loc='upper left', fontsize=legend_size, framealpha=0.95,
              edgecolor='black', fancybox=False, ncol=2)
    ax.set_xlim(-0.5, len(instances)-0.5)

def draw_rebuttal_vpr(ax):
    x = np.arange(len(vpr_short))
    width = 0.3
    ax.bar(x - width/2, vpr_hpwl, width, label='VPR',
           color=colors[5], alpha=0.85, edgecolor='white', linewidth=1.5, hatch='.')
    ax.bar(x + width/2, mofem_vpr, width, label='moFEM',
           color=colors[1], alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(vpr_short, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('HPWL', fontsize=ylabel_size, fontweight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(4,4))
    apply_style(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.set_ylim(0, max(vpr_hpwl) * 1.7)

    ax2 = ax.twinx()
    improvement = [(vpr - mofem) / vpr * 100 for vpr, mofem in zip(vpr_hpwl, mofem_vpr)]
    ax2.plot(x, improvement, 'o--', color=colors[2], label='Improvement (%)',
             linewidth=1, markersize=8)
    ax2.set_ylabel('Improvement (%)', fontsize=ylabel_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax2.set_ylim(-15, max(improvement) * 1.4)
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['bottom'].set_linewidth(0.8)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right',
              fontsize=legend_size, framealpha=0.95, edgecolor='black', fancybox=False, ncol=2)
    ax.margins(y=0.15)
    ax2.margins(y=0.15)

def draw_rebuttal_legal(ax):
    x = np.arange(len(legal_inst))
    width = 0.3
    ax.bar(x - width/2, pre_hpwl, width, label='Pre-legal',
           color=colors[0], alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.bar(x + width/2, post_hpwl, width, label='Post-legal',
           color=colors[1], alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(legal_inst, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('HPWL', fontsize=ylabel_size, fontweight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(4,4))
    apply_style(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.set_ylim(0, max(pre_hpwl) * 1.9)

    ax2 = ax.twinx()
    ax2.plot(x, reduction_pct, 'o--', color=colors[2], label='Reduction (%)',
             linewidth=1, markersize=8)
    ax2.set_ylabel('Reduction (%)', fontsize=ylabel_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax2.set_ylim(-5, max(reduction_pct) * 1.7)
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['bottom'].set_linewidth(0.8)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right',
              fontsize=legend_size, framealpha=0.95, edgecolor='black', fancybox=False, ncol=2)
    ax.margins(y=0.15)
    ax2.margins(y=0.15)

def draw_rebuttal_timing(ax):
    x = np.arange(len(timing_short))
    ax.plot(x, vivado_wns, 'o--', color=colors[3], label='Vivado WNS',
            linewidth=1, markersize=8)
    ax.plot(x, mofem_wns, 's--', color=colors[0], label='moFEM WNS',
            linewidth=1, markersize=8)
    ax.set_ylabel('WNS (ns)', fontsize=ylabel_size, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=colors[0], labelsize=tick_size)
    ax.set_ylim(-8, 10)

    ax2 = ax.twinx()
    width = 0.18
    ax2.bar(x - width/2, vivado_fmax, width, label='Vivado Fmax',
            color=colors[2], alpha=0.85, edgecolor='white', linewidth=1.5, hatch='*')
    ax2.bar(x + width/2, mofem_fmax, width, label='moFEM Fmax',
            color=colors[1], alpha=0.85, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Fmax (MHz)', fontsize=ylabel_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=tick_size)
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['bottom'].set_linewidth(0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(timing_short, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc='upper right',
              fontsize=legend_size, framealpha=0.95, edgecolor='black', fancybox=False, ncol=2)
    apply_style(ax, ylabel='')
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax2.grid(False)
    ax.margins(y=0.4)
    ax2.margins(y=0.4)

def draw_rebuttal_runtime_breakdown(ax):
    x = np.arange(len(runtime_short))
    width = 0.6
    bottom_vals = np.zeros(len(runtime_short))
    labels_d = ['Prep', 'GPU iter', 'Infer', 'Legal']
    data_d = [prep_r, gpu_r, infer_r, legal_r]

    for i, (label, data) in enumerate(zip(labels_d, data_d)):
        ax.bar(x, data, width, bottom=bottom_vals, label=label,
               color=colors[i], alpha=0.85, edgecolor='white', linewidth=1.5)
        bottom_vals += np.array(data)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(runtime_short, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('Seconds (log)', fontsize=ylabel_size, fontweight='bold')
    apply_style(ax)
    ax.legend(loc='upper right', fontsize=legend_size, framealpha=0.95,
              edgecolor='black', fancybox=False, ncol=2)
    ax.margins(y=0.15)

# ============================================================
# 4. 生成两张合并图
# ============================================================

def plot_figure1(csv_path='./result/final_best_results.csv', output='combined_figure1.png'):
    setup_font()
    instances, modes, by_instance, vivado_hpwl = load_best_csv(csv_path)

    # 新布局：两行，第一行占满，第二行三等分
    fig = plt.figure(figsize=(16, 10), dpi=150)   # 高度适当增加
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])  # 第一行略高，突出主图

    ax_a = fig.add_subplot(gs[0, :])   # 第一行占满3列
    ax_b = fig.add_subplot(gs[1, 0])   # 第二行左
    ax_c = fig.add_subplot(gs[1, 1])   # 第二行中
    ax_d = fig.add_subplot(gs[1, 2])   # 第二行右

    # 绘制四个子图（内容不变）
    draw_test_hpwl(ax_a, instances, modes, by_instance, vivado_hpwl)
    draw_test_runtime(ax_b, instances, modes, by_instance)
    draw_rebuttal_vpr(ax_c)
    draw_rebuttal_legal(ax_d)

    # 添加标签 (a), (b), (c), (d)
    ax_a.set_title('(a)', fontsize=title_size, fontweight='bold', y=-0.2)
    ax_b.set_title('(b)', fontsize=title_size, fontweight='bold', y=-0.2)
    ax_c.set_title('(c)', fontsize=title_size, fontweight='bold', y=-0.2)
    ax_d.set_title('(d)', fontsize=title_size, fontweight='bold', y=-0.2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, hspace=0.3)  # 行间距适当调整
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {output}')

def plot_figure2(output='combined_figure2.png'):
    setup_font()
    fig = plt.figure(figsize=(18, 3.5), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.2)
    ax_timing = fig.add_subplot(gs[0])
    ax_runtime = fig.add_subplot(gs[1])

    draw_rebuttal_timing(ax_timing)
    draw_rebuttal_runtime_breakdown(ax_runtime)

    ax_timing.set_title('(b)', fontsize=title_size, fontweight='bold', y=-0.2, loc='left')
    ax_runtime.set_title('(d)', fontsize=title_size, fontweight='bold', y=-0.2, loc='left')

    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {output}')

# ============================================================
# 5. 主程序
# ============================================================

if __name__ == '__main__':
    plot_figure1()
    plot_figure2()