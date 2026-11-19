# -*- coding: utf-8 -*-
"""六个机型的油耗剖面（数据节用图，Fig. 2）。

一张图展示六个机型的平均总燃油流量。六个飞行阶段由横轴刻度线分隔，
不给阶段上底色，避免底色与曲线抢视觉。

时间轴的处理：各航段时长不同，且多数航班存在阶段重访（巡航中的阶梯爬升等），
直接按绝对时间或按总时长归一都无法让阶段对齐。这里按阶段归一：

  同一阶段的全部记录按时间拼接，再等间隔映射到该阶段的固定区间。

每个机型显示一条逐点平均线，半透明带为该机型航班剖面的第 10–90 百分位范围。
纵轴六个机型共用，可以直接比较油耗水平。

只描述数据，不涉及模型：不划分训练与测试，不排除任何航班，剔除滑行记录。

数据：data/qar_a32x.csv（四种 A32x，94 个航班）
      data/qar_b737-800.csv（B737-800，44 个航班）
      data/qar_b737-700.csv（B737-700，15 个航班）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from af_data import DATA_CSV, PHASE_NAMES

B737_CSVS = (Path('data/qar_b737-800.csv'), Path('data/qar_b737-700.csv'))
OUT_DIR = Path('results/figures')

# 复用 make_fleet_wide_figure.py 的 NPG 风格分类色。颜色在本图中只代表机型。
AIRCRAFT_COLOUR: Dict[str, str] = {
    'A320':     '#3C5488',
    'A320-NEO': '#4DBBD5',
    'A321':     '#00A087',
    'A321-NEO': '#E64B35',
    'B737-800': '#8491B4',
    'B737-700': '#7E6148',
}
AIRCRAFT_ORDER = list(AIRCRAFT_COLOUR)
PHASE_LABEL = {'takeoff': 'Take-off', 'climb': 'Climb', 'cruise': 'Cruise',
               'descent': 'Descent', 'approach': 'Approach', 'landing': 'Landing'}
PHASE_RULE = '#C9C9C9'
MIN_PHASE_WIDTH = 0.03
GRID_POINTS = 700
QUANTILES: Tuple[float, float] = (0.10, 0.90)


def setup_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.2,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': False, 'ytick.right': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def load(data_csv: Path, extra_csvs: Sequence[Path],
         types: Sequence[str]) -> pd.DataFrame:
    data = pd.concat([pd.read_csv(data_csv)] + [pd.read_csv(p) for p in extra_csvs],
                     ignore_index=True)
    data = data[data['phase_name'] != 'taxi'].reset_index(drop=True)
    data['FF'] = data['FF1C'] + data['FF2C']
    unexpected = sorted(set(data['aircraft_type']) - set(AIRCRAFT_ORDER))
    if unexpected:
        raise RuntimeError(f'数据中出现未预期的机型 {unexpected}')
    data = data[data['aircraft_type'].isin(types)].reset_index(drop=True)
    if data.empty:
        raise RuntimeError(f'所选机型 {list(types)} 没有数据')
    return data[['aircraft_type', 'flight_id', 'time', 'phase_name', 'FF']]


def phase_edges(data: pd.DataFrame) -> np.ndarray:
    """按全部航班的中位时长占比切分横轴，起飞与着陆设下限保证可见。"""
    share = data.groupby(['flight_id', 'phase_name']).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).reindex(columns=PHASE_NAMES,
                                                         fill_value=0.0)
    widths = np.maximum(share.median().to_numpy(), MIN_PHASE_WIDTH)
    widths = widths / widths.sum()
    return np.concatenate([[0.0], np.cumsum(widths)])


def flight_profile(block: pd.DataFrame, edges: np.ndarray,
                   grid: np.ndarray) -> np.ndarray:
    """把一个航班映射到统一的阶段归一横轴，再落到公共网格上取均值。"""
    positions: List[np.ndarray] = []
    values: List[np.ndarray] = []
    for index, phase in enumerate(PHASE_NAMES):
        # 同一阶段的记录按时间拼接，阶段重访时也只占它自己的区间
        part = block[block['phase_name'] == phase].sort_values('time')
        if part.empty:
            continue
        count = len(part)
        inner = (np.arange(count) + 0.5) / count
        positions.append(edges[index] + inner * (edges[index + 1] - edges[index]))
        values.append(part['FF'].to_numpy())
    if not positions:
        raise RuntimeError('航班没有任何非滑行记录')
    position = np.concatenate(positions)
    value = np.concatenate(values)

    # 按网格分箱取均值，避免逐秒尖峰在抽样时产生假的锯齿。
    bins = np.clip(np.searchsorted(grid, position, side='right') - 1,
                   0, len(grid) - 1)
    total = np.bincount(bins, weights=value, minlength=len(grid))
    count = np.bincount(bins, minlength=len(grid))
    profile = np.divide(total, count, out=np.full(len(grid), np.nan),
                        where=count > 0)
    # 空箱按相邻值线性补齐，保证一条连续曲线。
    filled = np.isfinite(profile)
    if not filled.all():
        profile = np.interp(grid, grid[filled], profile[filled])
    return profile


def build_profiles(data: pd.DataFrame, edges: np.ndarray, grid: np.ndarray
                   ) -> Dict[str, List[np.ndarray]]:
    present = [n for n in AIRCRAFT_ORDER if n in set(data['aircraft_type'])]
    profiles: Dict[str, List[np.ndarray]] = {name: [] for name in present}
    for (aircraft, _flight), block in data.groupby(['aircraft_type', 'flight_id']):
        profiles[aircraft].append(flight_profile(block, edges, grid))
    return profiles


def draw(data: pd.DataFrame, out_stem: Path) -> Dict[str, List[np.ndarray]]:
    edges = phase_edges(data)
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    profiles = build_profiles(data, edges, grid)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    interval_upper = []
    for name in profiles:
        low, high = np.quantile(np.vstack(profiles[name]), QUANTILES, axis=0)
        interval_upper.append(float(high.max()))

    # 不给阶段上底色：阶段边界只用淡虚线交代。
    for edge in edges[1:-1]:
        ax.axvline(edge, color=PHASE_RULE, lw=0.6, ls=(0, (4, 3)),
                   alpha=0.55, zorder=1)

    # 每个机型只画逐点均值和 10–90 百分位阴影，不画单航班轨迹。
    for name in profiles:
        values = np.vstack(profiles[name])
        low, high = np.quantile(values, QUANTILES, axis=0)
        ax.fill_between(grid, low, high, color=AIRCRAFT_COLOUR[name],
                        alpha=0.10, linewidth=0.0, zorder=2)
        mean = np.mean(values, axis=0)
        ax.plot(grid, mean, color=AIRCRAFT_COLOUR[name], lw=0.85, alpha=0.78,
                zorder=3, solid_joinstyle='round')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(interval_upper) * 1.05)
    # 主刻度打在阶段边界只作分隔，阶段名放在次刻度上、位于各区间中点。
    ax.set_xticks(edges)
    ax.set_xticklabels([])
    ax.set_xticks((edges[:-1] + edges[1:]) / 2.0, minor=True)
    ax.set_xticklabels([PHASE_LABEL[phase] for phase in PHASE_NAMES],
                       minor=True, rotation=28, ha='right', fontsize=7)
    ax.tick_params(axis='x', which='minor', length=0.0)
    ax.set_xlabel('Flight phase (progress normalized within each phase)')
    ax.set_ylabel('Mean total fuel flow (kg s$^{-1}$)')

    aircraft_handles = [
        Line2D([], [], color=AIRCRAFT_COLOUR[name], lw=0.85, alpha=0.78,
               label=f'{name} ({len(profiles[name])} flights)')
        for name in profiles
    ]
    ax.legend(handles=aircraft_handles, loc='upper center', frameon=False,
              ncol=3, handlelength=1.8, borderaxespad=0.8,
              columnspacing=1.8, labelspacing=0.35)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    plt.close(fig)
    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')
    return profiles


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='六个机型的油耗剖面图')
    parser.add_argument('--data_csv', type=Path, default=DATA_CSV)
    parser.add_argument('--extra_csvs', type=Path, nargs='+', default=list(B737_CSVS))
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    parser.add_argument('--types', nargs='+', default=AIRCRAFT_ORDER,
                        choices=AIRCRAFT_ORDER, help='要画的机型')
    parser.add_argument('--name', default='fig_fuel_flow_distribution',
                        help='输出文件名主干')
    args = parser.parse_args(argv)
    setup_style()
    data = load(args.data_csv, args.extra_csvs, args.types)
    print(f'机型 {args.types}')
    print(f'航班 {data["flight_id"].nunique()} 个   非滑行记录 {len(data):,} 条')
    profiles = draw(data, args.out_dir / args.name)
    print()
    print('各机型均值剖面的巡航段水平（kg/s）')
    edges = phase_edges(data)
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    inside = (grid >= edges[2]) & (grid < edges[3])
    for name in profiles:
        mean = np.mean(np.vstack(profiles[name]), axis=0)
        print(f'  {name:<10} {len(profiles[name]):>3} 个航班   '
              f'巡航均值 {mean[inside].mean():.3f}')


if __name__ == '__main__':
    main()
