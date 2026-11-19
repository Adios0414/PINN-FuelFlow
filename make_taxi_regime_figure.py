# -*- coding: utf-8 -*-
"""taxi 与六个保留阶段的信息结构差异

三个面板串成一条论证：
  a 七个飞行状态变量在 taxi 中的有效变化。以六个保留阶段的标准差为基准，
    MACH、FPA、TAS 恰好为零——退化的正是决定气动受力的那几个，而 GWC 照常变化。
  b taxi 油耗按运转发动机数分组，单发约为双发的一半；该变量不在模型输入之内。
  c 同一组变量的解释度。空中 0.78，taxi 只有 0.07；补上运转发动机数后升到 0.42。

结论是 taxi 需要另一组地面运行变量，不是 taxi 没有规律。
面板 c 的 R² 仅作诊断，线性形式的局限由面板 a 兜底：方差为零的变量，
任何容量的模型都提取不出信息。
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

DATA = Path('data/qar_a32x.csv')
OUT_DIR = Path('results/figures')
FEATURES = ['ALT_QNH', 'MACH', 'FPA', 'TAS', 'SAT', 'IVV', 'GWC']
LABELS = ['ALT', 'MACH', 'FPA', 'TAS', 'SAT', 'IVV', 'GWC']

TAXI = '#E64B35'
TAXI_LIGHT = '#F2A08C'
AIR = '#4DBBD5'
INK = '#2b2b2b'
MUTED = '#8a8a8a'
# 运转发动机数是有序量，用同一红色系由浅到深
ENGINE_STEPS = ['#F7C7B8', '#F08E73', '#E64B35']


def setup_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': False, 'ytick.right': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def load() -> pd.DataFrame:
    frame = pd.read_csv(DATA, usecols=FEATURES + [
        'FF1C', 'FF2C', 'phase_name', 'eng1_running', 'eng2_running'])
    return frame.assign(FF=frame.FF1C + frame.FF2C,
                        n_engines=frame.eng1_running + frame.eng2_running,
                        ground=frame.phase_name.eq('taxi'))


def explained(frame: pd.DataFrame, extra: Optional[str] = None) -> float:
    """诊断用线性解释度，非模型性能。"""
    columns = FEATURES + ([extra] if extra else [])
    x = frame[columns].to_numpy(float)
    spread = x.std(axis=0)
    x = np.c_[(x - x.mean(axis=0)) / np.where(spread < 1e-9, 1.0, spread),
              np.ones(len(x))]
    y = frame['FF'].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return 1.0 - ((y - x @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def panel_a(ax, data: pd.DataFrame) -> None:
    """只画 taxi。六个阶段按定义恒为 1，用一条参考线表示，不再重复画成柱子。"""
    taxi, air = data[data.ground], data[~data.ground]
    ratio = [taxi[c].std() / air[c].std() for c in FEATURES]
    x = np.arange(len(FEATURES))
    ax.axhline(1.0, color=MUTED, lw=0.8, ls='--', zorder=2)
    ax.text(-0.5, 1.035, 'Six retained phases', ha='left', va='bottom',
            fontsize=6.5, color=MUTED)
    ax.bar(x, ratio, width=0.55, color=TAXI, zorder=3)
    for i, value in enumerate(ratio):
        ax.text(i, value + 0.035, '0' if value < 1e-9 else f'{value:.2f}',
                ha='center', va='bottom', fontsize=6.5, color=INK, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha='right')
    ax.set_xlim(-0.65, len(FEATURES) - 0.35)
    ax.set_ylim(0, 1.42)
    ax.set_ylabel('Relative variability during taxi')
    ax.set_title('Flight-state variability', pad=6)
    print('  panel a:', {n: round(v, 3) for n, v in zip(LABELS, ratio)})


def panel_b(ax, data: pd.DataFrame) -> None:
    """只比较单发与双发。两台都未标记运转的 301 条是启停时标志滞后，不入图，
    在图注中声明。"""
    taxi = data[data.ground]
    counts = (1, 2)
    groups = [taxi[taxi.n_engines == n].FF.to_numpy() for n in counts]
    parts = ax.boxplot(groups, widths=0.45, showfliers=False, patch_artist=True,
                       medianprops=dict(color='white', lw=1.2),
                       whiskerprops=dict(color=INK, lw=0.8),
                       capprops=dict(color=INK, lw=0.8),
                       boxprops=dict(lw=0.0))
    for patch, colour in zip(parts['boxes'], ENGINE_STEPS[1:]):
        patch.set_facecolor(colour)
    for i, block in enumerate(groups, start=1):
        ax.text(i, 0.395, f'n = {len(block):,}', ha='center', va='top',
                fontsize=6.5, color=INK)
        ax.text(i + 0.28, np.median(block), f'{np.median(block):.3f}',
                ha='left', va='center', fontsize=6.5, color=INK, zorder=5)
    ax.set_xticklabels([str(n) for n in counts])
    ax.set_xlim(0.45, 2.75)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel('Operating engines during taxi')
    ax.set_ylabel('Total fuel flow (kg s$^{-1}$)')
    ax.set_title('Ground-operation state', pad=6)
    excluded = int((taxi.n_engines == 0).sum())
    print(f'  panel b medians: {[round(float(np.median(g)), 4) for g in groups]}'
          f'   excluded 0-engine records: {excluded}')


def panel_c(ax, data: pd.DataFrame) -> None:
    taxi, air = data[data.ground], data[~data.ground]
    values = [explained(taxi), explained(taxi, 'n_engines'), explained(air)]
    labels = ['Taxi\nseven state\nvariables',
              'Taxi\n+ operating\nengine count',
              'Six phases\nseven state\nvariables']
    colours = [TAXI_LIGHT, TAXI, AIR]
    x = np.arange(3)
    ax.bar(x, values, width=0.55, color=colours, zorder=3)
    for i, value in enumerate(values):
        ax.text(i, value + 0.022, f'{value:.3f}', ha='center', va='bottom',
                fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Fuel-flow variance explained ($R^2$)')
    ax.set_title('Diagnostic explanatory power', pad=6)
    print('  panel c:', [round(v, 3) for v in values])


def draw(data: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    panel_a(axes[0], data)
    panel_b(axes[1], data)
    panel_c(axes[2], data)
    fig.tight_layout(w_pad=2.0)
    fig.canvas.draw()
    for letter, ax in zip('abc', axes):
        left = ax.yaxis.label.get_window_extent().x0
        ax.text(ax.transAxes.inverted().transform((left, 0))[0], 1.11, letter,
                transform=ax.transAxes, fontsize=10, fontweight='bold')
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    plt.close(fig)
    print(f'saved {out_stem.with_suffix(".pdf")} / {out_stem.with_suffix(".png")}')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='taxi 与六个保留阶段的信息结构差异')
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    setup_style()
    draw(load(), args.out_dir / 'fig_taxi_regime')


if __name__ == '__main__':
    main()
