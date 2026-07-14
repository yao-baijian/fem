"""
Rebuttal figure: two separate long figures
Figure 1: (a) VPR vs moFEM (removed ch_intrinsics) and (c) Legalization reduction %
Figure 2: (b) Timing dual-axis bars and (d) Runtime compact grouped bars (log y)

Each figure size: (18, 4) with two subplots side by side.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# fm._rebuild()
import numpy as np

# ------------------------------------------------------------
# Data (from rebuttal)
# ------------------------------------------------------------

# (a) VPR vs moFEM – remove ch_intrinsics (index 0)
vpr_inst = ['bgm', 'LU8PEEng', 'blob_merge', 'sha', 'mkDelayWorker32B', 'stereovision0']
vpr_short = ['bgm', 'LU8', 'blob', 'sha', 'mkDelay', 'stereo']
vpr_hpwl = [25104, 199096, 47895, 15194, 82570, 53979]
mofem_vpr = [23979, 173251, 46089, 12991, 57566, 52161]

# (b) Timing – unchanged (7 instances)
timing_short = ['s15850', 's38417', 'ch_int', 'bgm', 'blob', 'sha', 'mkDelay']
vivado_wns = [6.186, 5.364, 8.905, 3.513, 4.091, 4.354, 3.411]
vivado_fmax = [262, 215, 300, 154, 111, 177, 151]
mofem_wns   = [4.399, 5.178, 7.649, 3.071, 4.585, 4.012, 5.423]
mofem_fmax  = [197, 207, 300, 143, 185, 167, 218]

# (c) Legalization – 6 instances, same as vpr_short
legal_inst = vpr_short
pre_hpwl  = [105947, 121375, 41538, 6201, 21391, 32103]
post_hpwl = [105944, 120489, 40417, 6116, 21317, 32019]
# Calculate reduction percentage: (pre - post) / pre * 100
reduction_pct = [(p - q) / p * 100 for p, q in zip(pre_hpwl, post_hpwl)]

# (d) Runtime breakdown – same instances as (a), remove ch_int (index 0)
runtime_short = vpr_short
prep_r = [1.02, 0.77, 0.84, 0.84, 0.90, 0.71]
gpu_r  = [32.42, 241.49, 12.58, 8.39, 8.40, 22.22]
infer_r= [1.68, 3.08, 0.05, 0.02, 0.03, 0.08]
legal_r= [1.45, 1.28, 1.03, 0.10, 1.36, 0.82]

# ------------------------------------------------------------
# Style settings
# ------------------------------------------------------------
colors = ["#C99FF4", "#98CCF7", "#FDCD8E", "#83F78F", "#7863F3", "#F67059"]
ylabel_size = 10
tick_size = 12
legend_size = 10
title_size = 13
xtick_size = 9

def setup_font():    
    lb_path = fm.findfont('Libertinus Sans')
    if lb_path:
        print(f"INFO: Found Linux Libertine font at {lb_path}")
        prop = fm.FontProperties(fname=lb_path)
        plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.size'] = 14

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

# ------------------------------------------------------------
# Figure 1: (a) + (b)
# ------------------------------------------------------------
def plot_ac(output='rebuttal_fig_ac.png'):
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=300, constrained_layout=True)

    # --- Panel (a) VPR vs moFEM with improvement % line ---
    ax = axes[0]
    x = np.arange(len(vpr_short))
    width = 0.3

    # 柱状图：HPWL
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

    # 右 y 轴：improvement 百分比
    ax2 = ax.twinx()
    improvement = [(vpr - mofem) / vpr * 100 for vpr, mofem in zip(vpr_hpwl, mofem_vpr)]
    ax2.plot(x, improvement, 'o--', color=colors[2], label='Improvement (%)', linewidth=1, markersize=8)
    ax2.set_ylabel('Improvement (%)', fontsize=ylabel_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    ax2.set_ylim(-15, max(improvement) * 1.4)
    
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['bottom'].set_linewidth(0.8)

    # 合并图例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right',
              fontsize=legend_size, framealpha=0.95, edgecolor='black', fancybox=False, ncol=2)

    ax.set_title('(a)', fontsize=title_size, fontweight='bold', y=-0.25)
    ax.margins(y=0.15)
    ax2.margins(y=0.15)

    # --- Panel (c) Legalization reduction % (single line) ---
    ax = axes[1]
    x = np.arange(len(legal_inst))
    width = 0.3

    # 主 y 轴：pre 和 post HPWL 柱状图
    ax.bar(x - width/2, pre_hpwl, width, label='Pre-legal',
           color=colors[0], alpha=0.85, edgecolor='white', linewidth=1.5)  # 复用 colors[0]
    ax.bar(x + width/2, post_hpwl, width, label='Post-legal',
           color=colors[1], alpha=0.85, edgecolor='white', linewidth=1.5)  # 复用 colors[1]
    ax.set_xticks(x)
    ax.set_xticklabels(legal_inst, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('HPWL', fontsize=ylabel_size, fontweight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(4,4))
    apply_style(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    ax.set_ylim(0, max(pre_hpwl) * 1.9)

    # 右 y 轴：reduction 百分比
    ax2 = ax.twinx()
    ax2.plot(x, reduction_pct, 'o--', color=colors[2], label='Reduction (%)', linewidth=1, markersize=8)
    ax2.set_ylabel('Reduction (%)', fontsize=ylabel_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    # 设置 reduction 折线的 y 轴范围（可调整起始值，例如从 0 开始）
    ax2.set_ylim(-5, max(reduction_pct) * 1.7)

    # 移除 ax2 的边框
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['bottom'].set_linewidth(0.8)

    # 合并图例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right',
              fontsize=legend_size, framealpha=0.95, edgecolor='black', fancybox=False, ncol=2)

    ax.set_title('(b)', fontsize=title_size, fontweight='bold', y=-0.25)  # 注意：原为 (b)，这里保持为 (b) 还是 (c) 取决于您的需求
    ax.margins(y=0.15)
    ax2.margins(y=0.15)

    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {output}')

# ------------------------------------------------------------
# Figure 2: (b) + (d)
# ------------------------------------------------------------
def plot_bd(output='rebuttal_fig_bd.png'):
    fig = plt.figure(figsize=(18, 3.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.2)
    ax2 = fig.add_subplot(gs[0])   # 图 b (宽)
    ax = fig.add_subplot(gs[1])    # 图 d (窄)

    setup_font()

    # --- Panel (b) Timing: WNS as lines (y from -2), Fmax as bars ---
    def get_valid_indices(data):
        return [i for i, v in enumerate(data) if v is not None and v != 0]

    # 左轴 WNS 折线
    x_vals = np.arange(len(timing_short))
    ax2.plot(x_vals, vivado_wns, 'o--', color=colors[3], label='Vivado WNS', linewidth=1, markersize=8)
    ax2.plot(x_vals, mofem_wns, 's--', color=colors[0], label='moFEM WNS', linewidth=1, markersize=8)
    ax2.set_ylabel('WNS (ns)', fontsize=ylabel_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax2.set_ylim(-8, 10)  # 起始为 -2，上限根据数据调整（最大约9，留点余量）

    # 右轴 Fmax 柱状图
    ax2b = ax2.twinx()
    # 这里需要知道哪些数据有效（Fmax 可能为 None），但数据中已用 300 填充，无需过滤
    width = 0.18
    x = np.arange(len(timing_short))
    ax2b.bar(x - width/2, vivado_fmax, width, label='Vivado Fmax', color=colors[2], alpha=0.85, edgecolor='white', linewidth=1.5, hatch='*')
    ax2b.bar(x + width/2, mofem_fmax, width, label='moFEM Fmax', color=colors[1], alpha=0.85, edgecolor='white', linewidth=1.5)
    ax2b.set_ylabel('Fmax (MHz)', fontsize=ylabel_size, fontweight='bold')
    ax2b.tick_params(axis='y', labelcolor='gray', labelsize=tick_size)

    # 隐藏 ax2b 边框
    for spine in ['top', 'right', 'left']:
        ax2b.spines[spine].set_visible(False)
    ax2b.spines['bottom'].set_color('gray')
    ax2b.spines['bottom'].set_linewidth(0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(timing_short, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    # 合并图例
    handles, labels = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles + handles2, labels + labels2, loc='upper right',
               fontsize=legend_size, framealpha=0.95, edgecolor='black', fancybox=False, ncol=2)
    apply_style(ax2, ylabel='')
    
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax2.set_title('(a)', fontsize=title_size, fontweight='bold', y=-0.2)
    ax2b.grid(False)
    ax2.margins(y=0.4)
    ax2b.margins(y=0.4)

    # --- Panel (d) 堆积柱状图 (log y) ---
    x = np.arange(len(runtime_short))
    width = 0.5  # 柱子宽度

    # 堆积数据
    bottom_vals = np.zeros(len(runtime_short))
    bars = []
    labels_d = ['Prep', 'GPU iter', 'Infer', 'Legal']
    data_d = [prep_r, gpu_r, infer_r, legal_r]
    
    for i, (label, data) in enumerate(zip(labels_d, data_d)):
        bar = ax.bar(x, data, width, bottom=bottom_vals, label=label,
                     color=colors[i], alpha=0.85, edgecolor='white', linewidth=1.5)
        bars.append(bar)
        bottom_vals += np.array(data)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(runtime_short, rotation=30, ha='right', fontsize=xtick_size, fontweight='bold')
    ax.set_ylabel('Seconds (log)', fontsize=ylabel_size, fontweight='bold')
    # ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    ax.legend(loc='upper right', fontsize=legend_size, framealpha=0.95,
              edgecolor='black', fancybox=False, ncol=2)
    ax.set_title('(b)', fontsize=title_size, fontweight='bold', y=-0.2)
    ax.margins(y=0.15)

    apply_style(ax)
    
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {output}')

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
if __name__ == '__main__':
    plot_ac()
    plot_bd()