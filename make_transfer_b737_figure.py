# -*- coding: utf-8 -*-
"""迁移学习实验（A32x → B737-800 / B737-700）的主图 Fig. 5 与补充图。

画法参照 Wang et al. 2024 Nat. Commun. 15:4332 的 Fig. 4b 与 Fig. 5：小提琴图，
每个小提琴含 10 组抽样的点，红线为均值，黑线为均值 ± 标准差。

主图 fig_transfer_b737（2 × 2）
  a, b  测试 RMSE：不用目标数据（源模型直接用，5 个源模型各一个点）、从头训练与微调
        （机型特征编码器、融合层与输出层），后两组按目标机型训练航班数由浅到深
        （同 Wang Fig. 5 按电池数分组）。a 为 B737-800，b 为 B737-700。
        不用目标数据时误差远高于其余各组，纵轴断开：下段为其余各组的范围，上段容纳
        零样本的高值，且包含其均值。
  c, d  从 PINN 微调与从无物理约束消融模型微调的测试 RMSE（同一批航班与种子），
        横轴为训练航班数（同 Wang Fig. 4b 按方法着色）。

补充图 fig_transfer_b737_scope
  微调范围的消融：只微调机型特征编码器、再加输出层、再加融合层（即主图中的微调）。

配色沿用 make_fleet_wide_figure.py 与 make_sampling_figure.py：蓝 #2A78D6 为带物理约束，
橙 #EB6834 为不带物理约束；航班数用同一蓝色由浅到深。
数据取自 train_transfer.py 的 all_runs.csv，不重新推理。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from make_sampling_figure import setup_style

RESULT_DIR = Path('results/transfer')
OUT_DIR = Path('results/figures')
TARGETS = ('B737-800', 'B737-700')
N_DRAWS = 10
N_SOURCE = 5                      # 源模型个数：零样本每个机型 5 个值
ZERO_SHOT = ('S0', '#BDBDBD', 'Without\ntarget data')

# 主图：从头训练对比微调
METHODS = [
    ('S1', 'Train from scratch'),
    ('L3', 'Fine-tuning'),
]
# 补充图：微调范围的消融
SCOPES = [
    ('L1', 'Aircraft-characteristic\nencoder'),
    ('L2', 'Aircraft-characteristic encoder\n+ output layer'),
    ('L3', 'Aircraft-characteristic encoder\n+ fusion and output layers'),
]
# 训练航班数由少到多，蓝色由浅到深；"全部"始终取最深一档
FLIGHT_RAMP = ['#B7D3F6', '#86B6EF', '#5598E7', '#2A78D6', '#1C5CAB', '#104281']
WITH_PHYSICS = ('L3', '#2A78D6', 'Fine-tuned from PINN')
WITHOUT_PHYSICS = ('L3_ablation', '#EB6834', 'Fine-tuned from model without physics')

OUTLINE = '#3A3A3A'
MEAN_COLOUR = '#E3241B'


def load(target: str):
    runs = pd.read_csv(RESULT_DIR / target / 'all_runs.csv')
    pool = int((pd.read_csv(RESULT_DIR / target / 'split_manifest.csv').split == 'pool').sum())
    runs['n_flights'] = runs.k.where(runs.k > 0, pool)
    levels = sorted(runs.loc[runs.arm == 'S1', 'n_flights'].astype(int).unique())
    for name in {m for m, _ in METHODS + SCOPES} | {WITHOUT_PHYSICS[0]}:
        counts = runs[runs.arm == name].groupby('n_flights').size()
        if set(counts.index) != set(levels) or (counts != N_DRAWS).any():
            raise RuntimeError(f'{target} {name} 的抽样不完整：{counts.to_dict()}')
    if (runs.arm == ZERO_SHOT[0]).sum() != N_SOURCE:
        raise RuntimeError(f'{target} 零样本应有 {N_SOURCE} 个值')
    return runs, levels


def violin(ax, x, values, colour, width):
    """Wang et al. 2024 的画法：小提琴、沿中线的点、红色均值线、黑色均值 ± 标准差线。"""
    body = ax.violinplot([values], positions=[x], widths=width,
                         showmeans=False, showmedians=False, showextrema=False)['bodies'][0]
    body.set_facecolor(colour)
    body.set_edgecolor(OUTLINE)
    body.set_linewidth(0.7)
    body.set_alpha(1.0)
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    ax.vlines(x, mean - sd, mean + sd, color='black', lw=1.2, zorder=3)
    ax.scatter(np.full(len(values), x), values, s=3, color='#262626', zorder=4, linewidths=0)
    ax.hlines(mean, x - width * 0.22, x + width * 0.22, color=MEAN_COLOUR, lw=1.1, zorder=5)


def ramp_for(levels):
    """各档航班数对应的颜色：前几档依次取浅色，"全部"取最深一档。"""
    colours = FLIGHT_RAMP[:len(levels) - 1] + [FLIGHT_RAMP[-1]]
    return dict(zip(levels, colours))


def panel_methods(ax, runs, levels, target, methods):
    colours = ramp_for(levels)
    step, width = 0.8, 0.68
    group_width = step * len(levels)
    centres = []
    for g, (arm, _) in enumerate(methods):
        start = g * (group_width + 1.0)
        for i, n in enumerate(levels):
            values = runs.loc[(runs.arm == arm) & (runs.n_flights == n), 'test_RMSE'].to_numpy()
            violin(ax, start + i * step, values, colours[n], width)
        centres.append(start + step * (len(levels) - 1) / 2)
    ax.set_xticks(centres)
    ax.set_xticklabels([label for _, label in methods])
    ax.set_xlim(-0.7, centres[-1] + group_width / 2 + 0.3)
    ax.set_ylabel('Test RMSE (kg s$^{-1}$)')
    ax.set_title(target, fontsize=9)
    handles = [Patch(facecolor=colours[n], edgecolor=OUTLINE, linewidth=0.7,
                     label=f'{n} flight{"s" if n > 1 else ""}' + (' (all)' if n == levels[-1] else ''))
               for n in levels]
    handles += [plt.Line2D([], [], color=MEAN_COLOUR, lw=1.1, label='Mean'),
                plt.Line2D([], [], color='black', lw=1.2, label='Mean ± s.d.')]
    ax.legend(handles=handles, frameon=False, loc='upper right', ncol=2,
              handlelength=1.6, columnspacing=1.2)


def panel_methods_broken(top, bottom, runs, levels, target, methods):
    """a, b：零样本 + 从头训练 + 微调，纵轴断开。"""
    colours = ramp_for(levels)
    step, width = 0.8, 0.68
    group_width = step * len(levels)
    zero = runs.loc[runs.arm == ZERO_SHOT[0], 'test_RMSE'].to_numpy()
    for ax in (top, bottom):
        violin(ax, 0.0, zero, ZERO_SHOT[1], width)
    centres = [0.0]
    others = []
    for g, (arm, _) in enumerate(methods):
        start = 1.6 + g * (group_width + 1.0)
        for i, n in enumerate(levels):
            values = runs.loc[(runs.arm == arm) & (runs.n_flights == n), 'test_RMSE'].to_numpy()
            others.append(values)
            for ax in (top, bottom):
                violin(ax, start + i * step, values, colours[n], width)
        centres.append(start + step * (len(levels) - 1) / 2)

    # 下段：其余各组的范围，上方留出图例的位置
    rest = np.concatenate(others)
    span = rest.max() - rest.min()
    low_lo, low_hi = rest.min() - 0.08 * span, rest.max() + 0.55 * span
    mean = float(zero.mean())
    if mean < low_hi + 0.5 * span:
        # 均值只略高于下段时，下段延长到包含均值，避免均值线落在断口处
        low_hi = max(low_hi, mean + 0.12 * span)
    # 上段：零样本中高于下段的点（及在上段时的均值）
    above = zero[zero > low_hi]
    up_lo = (above.min() if above.size else zero.max()) - 0.08 * np.ptp(zero)
    if mean > low_hi:
        up_lo = min(up_lo, mean - 0.08 * np.ptp(zero))
    up_hi = zero.max() + 0.10 * np.ptp(zero)
    bottom.set_ylim(low_lo, low_hi)
    top.set_ylim(max(up_lo, low_hi), up_hi)

    # 断轴：隐藏相接的边框，画斜线标记
    top.spines['bottom'].set_visible(False)
    bottom.spines['top'].set_visible(False)
    top.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False)
    bottom.tick_params(axis='x', which='both', top=False)
    d = 0.012
    for ax, y in ((top, 0.0), (bottom, 1.0)):
        kw = dict(transform=ax.transAxes, color='black', lw=0.8, clip_on=False)
        ratio = ax.get_position().height
        dy = d / ratio * 0.18
        for x in (0.0, 1.0):
            ax.plot((x - d, x + d), (y - dy, y + dy), **kw)

    bottom.set_xticks(centres)
    bottom.set_xticklabels([ZERO_SHOT[2]] + [label for _, label in methods])
    bottom.set_xlim(-0.7, centres[-1] + group_width / 2 + 0.3)
    top.set_title(target, fontsize=9)
    handles = [Patch(facecolor=colours[n], edgecolor=OUTLINE, linewidth=0.7,
                     label=f'{n} flight{"s" if n > 1 else ""}' + (' (all)' if n == levels[-1] else ''))
               for n in levels]
    handles += [plt.Line2D([], [], color=MEAN_COLOUR, lw=1.1, label='Mean'),
                plt.Line2D([], [], color='black', lw=1.2, label='Mean ± s.d.')]
    bottom.legend(handles=handles, frameon=False, loc='upper right', ncol=2,
                  handlelength=1.6, columnspacing=1.2)


def panel_physics(ax, runs, levels, target):
    offset, width = 0.21, 0.38
    for i, n in enumerate(levels):
        for (arm, colour, _), dx in ((WITH_PHYSICS, -offset), (WITHOUT_PHYSICS, offset)):
            values = runs.loc[(runs.arm == arm) & (runs.n_flights == n), 'test_RMSE'].to_numpy()
            violin(ax, i + dx, values, colour, width)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(n) for n in levels[:-1]] + [f'{levels[-1]} (all)'])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_xlabel('Target-type training flights')
    ax.set_ylabel('Test RMSE (kg s$^{-1}$)')
    ax.set_title(target, fontsize=9)
    bottom, top = ax.get_ylim()
    ax.set_ylim(top=top + 0.08 * (top - bottom))
    handles = [Patch(facecolor=c, edgecolor=OUTLINE, linewidth=0.7, label=l)
               for _, c, l in (WITH_PHYSICS, WITHOUT_PHYSICS)]
    handles += [plt.Line2D([], [], color=MEAN_COLOUR, lw=1.1, label='Mean'),
                plt.Line2D([], [], color='black', lw=1.2, label='Mean ± s.d.')]
    ax.legend(handles=handles, frameon=False, loc='upper right', handlelength=1.6)


def finish(fig, axes, out_stem: Path, label_axes=None) -> None:
    fig.canvas.draw()
    label_axes = list(label_axes) if label_axes is not None else list(axes)
    for ax, ref, tag in zip(axes, label_axes, 'abcd'):
        left = ref.yaxis.label.get_window_extent().x0
        xpos = ax.transAxes.inverted().transform((left, 0))[0]
        ax.text(xpos, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight='bold')
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    plt.close(fig)
    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')


def draw(out_stem: Path) -> pd.DataFrame:
    """主图：a, b 从头训练对比微调；c, d 从 PINN 微调对比从无物理约束模型微调。"""
    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(2, 2, left=0.085, right=0.99, top=0.95, bottom=0.085,
                            wspace=0.24, hspace=0.36)
    source, letters, refs = [], [None] * 4, [None] * 4
    for col, target in enumerate(TARGETS):
        runs, levels = load(target)
        sub = grid[0, col].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.06)
        top = fig.add_subplot(sub[0])
        bottom = fig.add_subplot(sub[1], sharex=top)
        panel_methods_broken(top, bottom, runs, levels, target, METHODS)
        lower = fig.add_subplot(grid[1, col])
        panel_physics(lower, runs, levels, target)
        # y 轴标签放在两段之间的正中
        bottom.set_ylabel('Test RMSE (kg s$^{-1}$)')
        t, b = top.get_position(), bottom.get_position()
        bottom.yaxis.set_label_coords(-0.105, (t.y1 - b.y0) / 2 / b.height)
        letters[col], refs[col] = top, bottom
        letters[2 + col], refs[2 + col] = lower, lower
        keep = runs.arm.isin([m for m, _ in METHODS] + [WITHOUT_PHYSICS[0], ZERO_SHOT[0]])
        source.append(runs.loc[keep, ['arm', 'n_flights', 'draw', 'test_RMSE']].assign(target=target))
    finish(fig, letters, out_stem, label_axes=refs)
    data = pd.concat(source, ignore_index=True)
    data.loc[data.arm == ZERO_SHOT[0], 'n_flights'] = 0     # 零样本没有目标航班
    return data


def draw_scope(out_stem: Path) -> pd.DataFrame:
    """补充图：微调范围的消融。"""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    source = []
    for ax, target in zip(axes, TARGETS):
        runs, levels = load(target)
        panel_methods(ax, runs, levels, target, SCOPES)
        keep = runs.arm.isin([m for m, _ in SCOPES])
        source.append(runs.loc[keep, ['arm', 'n_flights', 'draw', 'test_RMSE']].assign(target=target))
    fig.tight_layout(h_pad=1.6)
    finish(fig, axes, out_stem)
    return pd.concat(source, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='迁移学习实验主图')
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    setup_style()
    for stem, func in (('fig_transfer_b737', draw), ('fig_transfer_b737_scope', draw_scope)):
        source = func(args.out_dir / stem)
        path = args.out_dir / f'{stem}_source_data.csv'
        source.to_csv(path, index=False)
        print(f'作图数据（逐抽样）已保存 {path}')


if __name__ == '__main__':
    main()
