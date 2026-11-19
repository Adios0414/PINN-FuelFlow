# -*- coding: utf-8 -*-
"""迁移学习实验（A32x → B737-800 / B737-700）的训练效率图。

比较从头训练（S1）与迁移（L3：从 PINN 出发，微调机型分支、融合层与输出层）。
画法与 make_transfer_b737_figure.py 相同（参照 Wang et al. 2024 Nat. Commun. Fig. 5）：
小提琴图，每个小提琴含 10 组抽样的点。

  a, b  训练用时（s，对数纵轴）。小提琴与均值线在对数尺度上计算：红线为几何均值，
        黑线为几何均值乘除几何标准差。
  c, d  达到最佳验证误差的轮次（早停选中的轮次）。红线为均值，黑线为均值 ± 标准差。

训练用时在 6 个进程共用一块 GPU 的条件下测得，轮次不受硬件负载影响。
不含源模型的训练时间（源模型为主实验已训练好的 A32x 模型，可供各目标机型复用）。
数据取自各运行单元的 result.json，不重新训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from make_sampling_figure import setup_style
from make_transfer_b737_figure import MEAN_COLOUR, OUTLINE, RESULT_DIR, TARGETS

OUT_DIR = Path('results/figures')
N_DRAWS = 10
ARMS = [
    ('S1', '#1BAF7A', 'Train from scratch'),
    ('L3', '#2A78D6', 'Fine-tuning'),
]


def load(target: str) -> pd.DataFrame:
    pool = int((pd.read_csv(RESULT_DIR / target / 'split_manifest.csv').split == 'pool').sum())
    rows = []
    for arm, *_ in ARMS:
        for path in sorted((RESULT_DIR / target / 'runs' / arm).glob('k*_d*/result.json')):
            config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
            result = json.loads(path.read_text(encoding='utf-8'))
            rows.append({'arm': arm, 'k': config['k'], 'draw': config['draw'],
                         'train_seconds': result['train_seconds'],
                         'best_epoch': result['best_epoch']})
    runs = pd.DataFrame(rows)
    runs['n_flights'] = runs.k.where(runs.k > 0, pool)
    counts = runs.groupby(['arm', 'n_flights']).size()
    if (counts != N_DRAWS).any():
        raise RuntimeError(f'{target} 的抽样不完整：{counts[counts != N_DRAWS].to_dict()}')
    return runs


def violin(ax, x, values, colour, width, log=False):
    """小提琴、沿中线的点、红色均值线、黑色离散范围线。log=True 时在 log10 尺度上计算。"""
    shown = np.log10(values) if log else np.asarray(values, dtype=float)
    body = ax.violinplot([shown], positions=[x], widths=width, showmeans=False,
                         showmedians=False, showextrema=False)['bodies'][0]
    body.set_facecolor(colour)
    body.set_edgecolor(OUTLINE)
    body.set_linewidth(0.7)
    body.set_alpha(1.0)
    centre, spread = float(np.mean(shown)), float(np.std(shown, ddof=1))
    ax.vlines(x, centre - spread, centre + spread, color='black', lw=1.2, zorder=3)
    ax.scatter(np.full(len(shown), x), shown, s=3, color='#262626', zorder=4, linewidths=0)
    ax.hlines(centre, x - width * 0.22, x + width * 0.22, color=MEAN_COLOUR, lw=1.1, zorder=5)
    return shown.max()


def panel(ax, runs, column, ylabel, target, log):
    levels = sorted(runs.n_flights.unique())
    offset, width = 0.21, 0.38
    for i, n in enumerate(levels):
        for (arm, colour, _), dx in zip(ARMS, (-offset, offset)):
            block = runs[(runs.arm == arm) & (runs.n_flights == n)]
            violin(ax, i + dx, block[column].to_numpy(), colour, width, log)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(n) for n in levels[:-1]] + [f'{levels[-1]} (all)'])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_xlabel('Target-type training flights')
    ax.set_ylabel(ylabel)
    ax.set_title(target, fontsize=9)
    if log:
        # 纵轴范围按数据定，刻度直接标秒数
        values = np.log10(runs[column].to_numpy(dtype=float))
        low, high = values.min() - 0.12, values.max() + 0.12
        candidates = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        ticks = [v for v in candidates if low <= np.log10(v) <= high]
        ax.set_yticks(np.log10(ticks))
        ax.set_yticklabels([str(v) for v in ticks])
        ax.set_ylim(low, high)
    bottom, top = ax.get_ylim()
    ax.set_ylim(top=top + 0.10 * (top - bottom))


def draw(out_stem: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    source = []
    for col, target in enumerate(TARGETS):
        runs = load(target)
        panel(axes[0, col], runs, 'train_seconds', 'Training time (s)', target, log=True)
        panel(axes[1, col], runs, 'best_epoch', 'Epochs to best validation error',
              target, log=False)
        source.append(runs.assign(target=target))

    handles = [Patch(facecolor=c, edgecolor=OUTLINE, linewidth=0.7, label=l) for _, c, l in ARMS]
    handles += [plt.Line2D([], [], color=MEAN_COLOUR, lw=1.1, label='Mean'),
                plt.Line2D([], [], color='black', lw=1.2, label='Mean ± s.d.')]
    fig.legend(handles=handles, frameon=False, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.6)

    fig.tight_layout(w_pad=2.0, h_pad=1.4, rect=(0, 0.05, 1, 1))
    fig.canvas.draw()
    for ax, tag in zip(axes.flat, 'abcd'):
        left = ax.yaxis.label.get_window_extent().x0
        xpos = ax.transAxes.inverted().transform((left, 0))[0]
        ax.text(xpos, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight='bold')

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    plt.close(fig)
    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')
    return pd.concat(source, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='迁移学习实验训练效率图')
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    setup_style()
    source = draw(args.out_dir / 'fig_transfer_b737_efficiency')
    path = args.out_dir / 'fig_transfer_b737_efficiency_source_data.csv'
    source.to_csv(path, index=False)
    print(f'作图数据（逐抽样）已保存 {path}')
    summary = source.groupby(['target', 'arm', 'n_flights'])[['train_seconds', 'best_epoch']].mean()
    print(summary.round(1).unstack('arm').to_string())


if __name__ == '__main__':
    main()
