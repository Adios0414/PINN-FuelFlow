# -*- coding: utf-8 -*-
"""物理损失权重坐标搜索结果图。

三个子图分别展示质量守恒、力学响应和相位连续权重的单因素扫描结果。
每次只改变一个权重，其余两个保持固定基线值；纵轴为 A320-NEO 调参集上
FF_Total 的 R²。搜索只使用 seed 1，因此不绘制误差条或区间。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter

METRICS_CSV = Path('results/weight_search/search_metrics.csv')
OUT_DIR = Path('results/figures')

BASELINE = {'w_mass': 0.10, 'w_force': 0.10, 'w_phase': 0.02}
BASELINE_CONFIG = 'baseline'
SELECTED_CONFIG = 'force_10'

# 每个元组：面板标题、横轴标签、颜色、[(权重值, 配置名)]。
SCANS: List[Tuple[str, str, str, float, List[Tuple[float, str]]]] = [
    (
        'Mass conservation',
        r'Weight $\lambda_{\mathrm{mass}}$',
        '#3C5488',
        0.10,
        [(0.01, 'mass_0.01'), (0.10, 'baseline'), (1.00, 'mass_1')],
    ),
    (
        'Force response',
        r'Weight $\lambda_{\mathrm{force}}$',
        '#E64B35',
        10.0,
        [(0.10, 'baseline'), (1.00, 'force_1'), (10.0, 'force_10')],
    ),
    (
        'Phase continuity',
        r'Weight $\lambda_{\mathrm{phase}}$',
        '#00A087',
        0.02,
        [(0.002, 'phase_0.002'), (0.02, 'baseline'), (0.20, 'phase_0.2')],
    ),
]

EXPECTED_CONFIGS = {
    'baseline', 'mass_0.01', 'mass_1', 'force_1', 'force_10',
    'phase_0.002', 'phase_0.2',
}


def setup_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.2,
        'ytick.major.size': 3.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': False,
        'ytick.right': False,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'未找到搜索结果 {path}')
    metrics = pd.read_csv(path)
    required = {
        'seed', 'partition', 'aircraft_type', 'target', 'config',
        'w_mass', 'w_force', 'w_phase', 'R2',
    }
    missing_columns = sorted(required - set(metrics.columns))
    if missing_columns:
        raise RuntimeError(f'搜索结果缺少列 {missing_columns}')

    result = metrics[
        (metrics['seed'] == 1)
        & (metrics['partition'] == 'tuning_type')
        & (metrics['aircraft_type'] == 'all')
        & (metrics['target'] == 'FF_Total')
    ].copy()

    counts = result.groupby('config').size()
    missing_configs = sorted(EXPECTED_CONFIGS - set(counts.index))
    duplicate_configs = sorted(counts[counts != 1].index.tolist())
    unexpected_configs = sorted(set(counts.index) - EXPECTED_CONFIGS)
    if missing_configs or duplicate_configs or unexpected_configs:
        raise RuntimeError(
            '搜索配置不完整：'
            f'缺少 {missing_configs}，非单行 {duplicate_configs}，'
            f'未预期 {unexpected_configs}'
        )
    if not result['R2'].notna().all():
        raise RuntimeError('搜索结果包含缺失的 R²')

    # 核对结果文件中的权重，防止配置名与实际参数不一致。
    expected_weights: Dict[str, Dict[str, float]] = {
        'baseline': dict(BASELINE),
        'mass_0.01': {**BASELINE, 'w_mass': 0.01},
        'mass_1': {**BASELINE, 'w_mass': 1.00},
        'force_1': {**BASELINE, 'w_force': 1.00},
        'force_10': {**BASELINE, 'w_force': 10.0},
        'phase_0.002': {**BASELINE, 'w_phase': 0.002},
        'phase_0.2': {**BASELINE, 'w_phase': 0.20},
    }
    indexed = result.set_index('config')
    for config, expected in expected_weights.items():
        for column, value in expected.items():
            actual = float(indexed.loc[config, column])
            if abs(actual - value) > 1e-12:
                raise RuntimeError(
                    f'{config} 的 {column}={actual:g}，预期 {value:g}'
                )
    return result


def weight_label(value: float) -> str:
    return f'{value:g}'


def draw(results: pd.DataFrame, out_stem: Path) -> None:
    score = results.set_index('config')['R2'].astype(float).to_dict()
    baseline_score = score[BASELINE_CONFIG]
    selected_score = score[SELECTED_CONFIG]

    all_scores = [score[config] for config in EXPECTED_CONFIGS]
    y_min = min(all_scores) - 0.012
    y_max = max(all_scores) + 0.018

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.50), sharey=True)
    panel_letters = ('a', 'b', 'c')

    for ax, letter, (title, xlabel, colour, selected_value, candidates) in zip(
            axes, panel_letters, SCANS):
        x_values = [value for value, _ in candidates]
        configs = [config for _, config in candidates]
        y_values = [score[config] for config in configs]

        ax.axhline(
            baseline_score, color='#B8B8B8', lw=0.7,
            ls=(0, (3, 2)), zorder=1,
        )
        ax.plot(
            x_values, y_values, color=colour, lw=1.05,
            alpha=0.82, zorder=2,
        )

        for x_value, y_value, config in zip(x_values, y_values, configs):
            is_selected = abs(x_value - selected_value) < 1e-12
            if is_selected:
                ax.scatter(
                    [x_value], [y_value], s=58, marker='*',
                    facecolor=colour, edgecolor='#202020', linewidth=0.55,
                    zorder=5,
                )
            elif config == BASELINE_CONFIG:
                ax.scatter(
                    [x_value], [y_value], s=30, marker='o',
                    facecolor='white', edgecolor='#303030', linewidth=0.8,
                    zorder=4,
                )
            else:
                ax.scatter(
                    [x_value], [y_value], s=28, marker='o',
                    facecolor=colour, edgecolor='white', linewidth=0.55,
                    zorder=4,
                )

            offset = 0.0040
            vertical_alignment = 'bottom'
            if config == BASELINE_CONFIG and not is_selected:
                offset = -0.0045
                vertical_alignment = 'top'
            ax.text(
                x_value, y_value + offset, f'{y_value:.3f}',
                ha='center', va=vertical_alignment, fontsize=6.8,
                color='#303030', zorder=6,
            )

        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(x_values))
        ax.set_xticklabels([weight_label(value) for value in x_values])
        ax.set_xlim(x_values[0] / 1.8, x_values[-1] * 1.8)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=6, fontweight='normal')
        ax.text(
            -0.13, 1.05, letter, transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', va='bottom', ha='left',
        )

    axes[0].set_ylabel(r'A320-NEO tuning $R^2$ for FF$_{\mathrm{Total}}$')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.subplots_adjust(
        left=0.095, right=0.985, top=0.85, bottom=0.22, wspace=0.20,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    plt.close(fig)

    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')
    print(
        '选中配置 '
        f'w_mass={BASELINE["w_mass"]:g}, '
        'w_force=10, '
        f'w_phase={BASELINE["w_phase"]:g}; '
        f'tuning R2={selected_score:.6f}'
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='物理损失权重搜索结果图')
    parser.add_argument('--metrics_csv', type=Path, default=METRICS_CSV)
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    parser.add_argument('--name', default='fig_loss_weight_search')
    args = parser.parse_args(argv)

    setup_style()
    results = load_results(args.metrics_csv)
    draw(results, args.out_dir / args.name)


if __name__ == '__main__':
    main()
