# -*- coding: utf-8 -*-
"""Supplementary Fig. 4：留出测试航班上的物理一致性。

数据取自 physical_consistency_eval.py 的输出（results/physical_consistency），不重新计算。
画法沿用 make_transfer_b737_figure.py：小提琴、沿中线的点、红色均值线、黑色均值 ± 标准差线；
每个小提琴含 5 个点，即每个配置的 5 个训练模型各一个值。

  a  900 s 窗口内预测油耗积分与重量下降量相对差的中位数；虚线为记录油耗本身的闭合误差
  b  门控 𝒢L 内对航迹角响应方向错误（s_γ < 0）的记录比例
  c  门控 𝒢L ∩ 𝒢m 内对重量响应方向错误（s_m < 0）的记录比例
  d  相位过渡点上两侧子网络预测值相对跳变的中位数（MLP 无相位输入，不参与）

配色：带物理约束蓝 #2A78D6、不带物理约束橙 #EB6834，与 Fig. 4、Fig. 5 一致；
MLP 用中性灰，避免与 Fig. 3 中 MLP 的橙色在本图里和“不带物理约束”混淆。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import mlab
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

from make_sampling_figure import setup_style
from make_transfer_b737_figure import MEAN_COLOUR, OUTLINE

IN_DIR = Path('results/physical_consistency')
OUT_DIR = Path('results/figures')
OUT_STEM = 'supp_fig4_physical_consistency'

CONFIGS = [
    ('full', '#2A78D6', 'Full model'),
    ('no_physics', '#EB6834', 'No physics losses'),
    ('mlp', '#A6A6A6', 'MLP'),
]
PANELS = [
    ('closure_median_abs_pct', 'Mass conservation', 'Closure error (%)', ('full', 'no_physics', 'mlp')),
    ('wrong_sign_gamma_pct', 'Response to flight-path angle', 'Wrong-signed records (%)',
     ('full', 'no_physics', 'mlp')),
    ('wrong_sign_mass_pct', 'Response to aircraft mass', 'Wrong-signed records (%)',
     ('full', 'no_physics', 'mlp')),
    ('jump_median_pct', 'Phase continuity', 'Jump at phase transition (%)', ('full', 'no_physics')),
]


def to_axis(ax, values):
    """数据坐标 → 纵轴刻度坐标（线性轴为恒等，symlog 轴为对数变换）。"""
    return ax.yaxis.get_transform().transform(np.asarray(values, float).reshape(-1, 1)).ravel()


def from_axis(ax, values):
    return ax.yaxis.get_transform().inverted().transform(
        np.asarray(values, float).reshape(-1, 1)).ravel()


def violin(ax, x, values, colour, width=0.6):
    """与 ax.violinplot 相同的核密度（Scott 带宽、截于极值），但在纵轴刻度坐标中估计，
    使对数轴上的形状不失真。"""
    values = np.asarray(values, dtype=float)
    t = to_axis(ax, values)
    grid = np.linspace(t.min(), t.max(), 200)
    density = mlab.GaussianKDE(t, 'scott').evaluate(grid)
    half = 0.5 * width * density / density.max()
    ax.fill(np.concatenate([x - half, (x + half)[::-1]]),
            from_axis(ax, np.concatenate([grid, grid[::-1]])),
            facecolor=colour, edgecolor=OUTLINE, linewidth=0.7, zorder=2)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    ax.vlines(x, mean - sd, mean + sd, color='black', lw=1.2, zorder=3)
    ax.scatter(np.full(len(values), x), values, s=4, color='#262626', zorder=4, linewidths=0)
    ax.hlines(mean, x - width * 0.22, x + width * 0.22, color=MEAN_COLOUR, lw=1.1, zorder=5)
    return mean, values.max()


def main() -> None:
    setup_style()
    runs = pd.read_csv(IN_DIR / 'per_seed.csv')
    recorded = pd.read_csv(IN_DIR / 'recorded_closure_windows.csv')
    recorded_test = 100 * float(np.median(np.abs(
        recorded.loc[recorded.partition == 'test', 'relative_difference'])))
    counts = runs.groupby('config').size()
    if (counts != 5).any():
        raise RuntimeError(f'每个配置应有 5 个模型：{counts.to_dict()}')
    colours = {c: col for c, col, _ in CONFIGS}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for ax, (column, title, ylabel, shown), tag in zip(axes.flat, PANELS, 'abcd'):
        closure = column == 'closure_median_abs_pct'
        if not closure:
            # 完整模型在 0.003–0.2%，其余模型在 3–30%，相差约两个数量级，用对数纵轴；
            # 0.01% 以下取线性段，使恰为 0 的种子（13 万条记录无一错误）也能画出
            ax.set_yscale('symlog', linthresh=0.01, linscale=0.6)
        # 闭合误差面板纵轴不从 0 起：三个模型与记录值都集中在 4–5.5%，否则下半幅空白
        bottom = (np.floor((min(recorded_test, runs[column].min()) - 0.1) * 2) / 2
                  if closure else 0.0)
        t_bottom = to_axis(ax, [bottom])[0]
        t_top = to_axis(ax, [runs.loc[runs.config.isin(shown), column].max()])[0]
        for x, config in enumerate(shown):
            values = runs.loc[runs.config == config, column].to_numpy()
            mean, vmax = violin(ax, x, values, colours[config])
            label_y = from_axis(ax, [to_axis(ax, [vmax])[0] + 0.03 * (t_top - t_bottom)])[0]
            ax.text(x, label_y, f'{mean:.2f}' if mean < 1 else f'{mean:.1f}',
                    ha='center', va='bottom', fontsize=6.5, color='#262626')
        if closure:
            ax.axhline(recorded_test, color='#2B2B2B', lw=0.9, ls='--', zorder=1)
        ax.set_ylim(bottom, from_axis(ax, [t_top + 0.15 * (t_top - t_bottom)])[0])
        if not closure:
            ticks = [0, 0.01, 0.1, 1, 10, 100]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels([f'{v:g}' for v in ticks])
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([dict((c, l) for c, _, l in CONFIGS)[c].replace(' losses', '\nlosses')
                            for c in shown])
        ax.set_xlim(-0.6, len(shown) - 0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.text(-0.16, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight='bold')

    handles = [Patch(facecolor=c, edgecolor=OUTLINE, linewidth=0.7, label=l) for _, c, l in CONFIGS]
    handles += [plt.Line2D([], [], color=MEAN_COLOUR, lw=1.1, label='Mean'),
                plt.Line2D([], [], color='black', lw=1.2, label='Mean ± s.d.'),
                plt.Line2D([], [], color='#2B2B2B', lw=0.9, ls='--',
                           label='Closure error of recorded fuel flow')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.6, w_pad=2.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'{OUT_STEM}.{ext}')
    runs.to_csv(OUT_DIR / f'{OUT_STEM}_source.csv', index=False)
    print(f'recorded closure (test windows) {recorded_test:.2f}%')
    print(f'saved {OUT_DIR / OUT_STEM}.pdf/.png')


if __name__ == '__main__':
    main()
