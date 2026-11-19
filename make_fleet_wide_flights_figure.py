# -*- coding: utf-8 -*-
"""Fleet-wide：全部 19 个测试航班的时程对比

版式与 make_zero_shot_flights_figure.py 一致：每个航班一格，黑线实测、红线预测，
格内标该航班的 R² 与 MAPE。4 × 5 共二十格，十九格放航班，剩下一格放图例。

与零样本那张图的区别：fleet-wide 的测试集含四种机型（A320 4 班、A320-NEO 9 班、
A321 4 班、A321-NEO 2 班），因此航班先按机型分组、组内按起飞时刻排序，格标题写明
机型与组内序号。这些机型的其余航班都参与了训练，本图展示的是同分布下的预测表现，
不是外推。

预测在本脚本内重新做一次前向，用的是 fleet-wide 完整模型的同一个 checkpoint 与
同一份划分，结果与该实验的汇总数值一致。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from af_data import (
    STUDY_AIRCRAFT_TYPES, FlightRecordDataset, load_flight_table,
    split_fleet_wide,
)
from af_train import predict, restore_model

PINN_DIR = Path('results/fleet_wide/all_flights_full')
OUT_DIR = Path('results/figures')
OBSERVED = '#2b2b2b'
PREDICTED = '#E64B35'
SPLIT_SEED = 42


def setup_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 8,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': False, 'ytick.right': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def takeoff_time(flight_id: str) -> int:
    """公开数据的航班号以 _D 序号结尾，即起飞先后（原始起飞时刻已匿名化）。"""
    match = re.search(r'_D(\d+)$', str(flight_id))
    return int(match.group(1)) if match else 0


def build_records(seed: int, device: torch.device) -> pd.DataFrame:
    data = load_flight_table()
    _, _, test_frame, manifest = split_fleet_wide(
        data, 0.7, 0.1, 0.2, SPLIT_SEED, verbose=False)
    dataset = FlightRecordDataset(test_frame, 'test').to(device)

    path = PINN_DIR / f'seed_{seed}' / 'best_model.pt'
    if not path.exists():
        raise FileNotFoundError(f'缺少 PINN 检查点 {path}')
    model, _ = restore_model(path, manifest, SPLIT_SEED, device)
    predictions, targets = predict(model, dataset, device, 8192)

    type_ids = dataset.aircraft_type_ids.detach().cpu().numpy()
    return pd.DataFrame({
        'flight_id': [dataset.flight_keys[i] for i in
                      dataset.flight_ids.detach().cpu().numpy()],
        'aircraft_type': [STUDY_AIRCRAFT_TYPES[i] for i in type_ids],
        'time': dataset.time.detach().cpu().numpy(),
        'observed': targets.sum(axis=1),
        'predicted': predictions.sum(axis=1),
    })


def scores(block: pd.DataFrame) -> Tuple[float, float]:
    residual = block.observed - block.predicted
    r2 = 1.0 - float((residual ** 2).sum()
                     / ((block.observed - block.observed.mean()) ** 2).sum())
    mape = float((residual.abs() / block.observed).mean() * 100)
    return r2, mape


def ordered_flights(records: pd.DataFrame) -> List[Tuple[str, str, int]]:
    """先按机型分组，组内按起飞时刻排序；返回 (航班号, 机型, 组内序号)。"""
    ordered: List[Tuple[str, str, int]] = []
    lookup = (records.groupby('flight_id')['aircraft_type'].first())
    for aircraft in STUDY_AIRCRAFT_TYPES:
        flights = sorted((str(f) for f, t in lookup.items() if t == aircraft),
                         key=takeoff_time)
        ordered += [(flight, aircraft, index + 1)
                    for index, flight in enumerate(flights)]
    return ordered


def draw(records: pd.DataFrame, out_stem: Path, n_cols: int = 4) -> None:
    flights = ordered_flights(records)
    n_rows = int(np.ceil((len(flights) + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.05 * n_cols, 1.62 * n_rows))
    flat = axes.ravel()

    for index, (flight, aircraft, within) in enumerate(flights):
        ax = flat[index]
        block = records[records.flight_id.astype(str) == flight]
        minutes = (block.time - block.time.min()) / 60.0
        ax.plot(minutes, block.observed, color=OBSERVED, lw=0.6,
                label='Observed', zorder=2)
        ax.plot(minutes, block.predicted, color=PREDICTED, lw=0.6, alpha=0.85,
                label='Predicted (PINN)', zorder=3)
        r2, mape = scores(block)
        ax.text(0.97, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.2)
        ax.text(0.97, 0.79, f'MAPE = {mape:.1f}%', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.2)
        ax.set_xlim(0, minutes.max())
        ax.set_ylim(0, 1.20 * block.observed.max())
        ax.set_title(f'{aircraft}  {within}', pad=4)
        if index // n_cols == n_rows - 1 or index + n_cols >= len(flights):
            ax.set_xlabel('Time since take-off (min)')
        if index % n_cols == 0:
            ax.set_ylabel('Total fuel flow (kg s$^{-1}$)')

    # 余下的格子放图例，不留空白框
    handles, labels = flat[0].get_legend_handles_labels()
    for ax in flat[len(flights):]:
        ax.axis('off')
    flat[len(flights)].legend(handles, labels, loc='center', frameon=False,
                              handlelength=2.0)

    fig.tight_layout(w_pad=1.0, h_pad=0.9)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    plt.close(fig)
    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')

    for flight, aircraft, within in flights:
        block = records[records.flight_id.astype(str) == flight]
        r2, mape = scores(block)
        print(f'  {aircraft:<9} {within:>2}  {str(flight)[-14:]}  '
              f'{len(block):>7,} 条  R2 {r2:.4f}  MAPE {mape:.1f}%')
    r2, mape = scores(records)
    print(f'  全部 {len(flights)} 班合计  {len(records):,} 条  '
          f'R2 {r2:.4f}  MAPE {mape:.1f}%')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='fleet-wide 逐航班时程图')
    parser.add_argument('--seed', type=int, default=42,
                        help='用哪个模型种子的预测，默认 42')
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    device = torch.device('cuda' if args.device != 'cpu'
                          and torch.cuda.is_available() else 'cpu')
    setup_style()
    draw(build_records(args.seed, device), args.out_dir / 'fig_fleet_wide_flights')


if __name__ == '__main__':
    main()
