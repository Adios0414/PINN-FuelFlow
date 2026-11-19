# -*- coding: utf-8 -*-
"""fleet-wide 航班采样实验

训练航班从每机型 1 个逐步增加到全部，完整模型与去掉物理约束的对照模型各跑一遍，
每格三个种子。验证集与测试集全程不变，只有训练子集在缩，因此各格结果可直接比较。

用于回答"数据量与物理约束的关系"：数据充足时两者应当接近，数据越少差距应当越大。

已完成的格子默认跳过（完整模型的全样本此前已单独跑过，不会重复）。
每格实际调用 train_fleet_wide，训练、评估、诊断的口径与主实验完全一致。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import train_fleet_wide
from af_train import log

DEFAULT_OUT_DIR = Path('results/fleet_wide')
SUMMARY_DIR = Path('results/flight_sampling')


def cell_tag(flights_per_type: int, draw: int, no_physics: bool) -> str:
    """直接问 train_fleet_wide 要目录名，避免命名规则出现第二份。"""
    argv = ['--flights_per_type', str(flights_per_type), '--draw', str(draw)]
    if no_physics:
        argv.append('--no_physics')
    return train_fleet_wide.auto_tag(train_fleet_wide.build_parser().parse_args(argv))


def is_done(run_dir: Path, seeds: Sequence[int]) -> bool:
    path = run_dir / 'per_seed_metrics.csv'
    if not path.exists():
        return False
    try:
        done = set(pd.read_csv(path)['seed'].unique())
    except Exception:
        return False
    return set(seeds).issubset(done)


def run_cell(flights_per_type: int, draw: int, no_physics: bool,
             seeds: Sequence[int], out_dir: Path, device: str,
             epochs: int) -> None:
    argv = ['--flights_per_type', str(flights_per_type),
            '--draw', str(draw),
            '--seeds', *[str(s) for s in seeds],
            '--epochs', str(epochs),
            '--out_dir', str(out_dir),
            '--device', device]
    if no_physics:
        argv.append('--no_physics')
    train_fleet_wide.main(argv)


def collect(out_dir: Path, levels: Sequence[int], draw: int,
            seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, diagnostic_rows = [], []
    for flights_per_type in levels:
        for no_physics in (False, True):
            run_dir = out_dir / cell_tag(flights_per_type, draw, no_physics)
            metric_path = run_dir / 'per_seed_metrics.csv'
            diagnostic_path = run_dir / 'per_seed_diagnostics.csv'
            if not metric_path.exists():
                continue
            metrics = pd.read_csv(metric_path)
            diagnostics = pd.read_csv(diagnostic_path)
            for frame in (metrics, diagnostics):
                frame['flights_per_type'] = flights_per_type
                frame['condition'] = 'no_physics' if no_physics else 'full'
            metric_rows.append(metrics[metrics.seed.isin(seeds)])
            diagnostic_rows.append(diagnostics[diagnostics.seed.isin(seeds)])
    if not metric_rows:
        raise RuntimeError('没有找到任何已完成的格子')
    return (pd.concat(metric_rows, ignore_index=True),
            pd.concat(diagnostic_rows, ignore_index=True))


def report(metrics: pd.DataFrame, diagnostics: pd.DataFrame,
           summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_dir / 'sampling_metrics.csv', index=False)
    diagnostics.to_csv(summary_dir / 'sampling_diagnostics.csv', index=False)

    test = metrics[(metrics.partition == 'test')
                   & (metrics.aircraft_type == 'all')
                   & (metrics.target == 'FF_Total')]
    table = test.groupby(['flights_per_type', 'n_train_flights', 'condition'],
                         as_index=False)[['R2', 'RMSE', 'MAPE']].mean()
    table.to_csv(summary_dir / 'sampling_summary.csv', index=False)

    log('\n' + '=' * 84)
    log('测试集 FF_Total，按训练数据量（三种子均值）')
    log('=' * 84)
    log(f'{"每机型":>6} {"航班":>5}   {"完整 R2":>9} {"对照 R2":>9} {"差":>8}   '
        f'{"完整 MAPE":>10} {"对照 MAPE":>10} {"差":>8}')
    for level in sorted(table.flights_per_type.unique(),
                        key=lambda v: (v == 0, v)):
        block = table[table.flights_per_type == level].set_index('condition')
        if {'full', 'no_physics'} - set(block.index):
            continue
        full, control = block.loc['full'], block.loc['no_physics']
        name = '全部' if level == 0 else str(level)
        log(f'{name:>6} {int(full.n_train_flights):>5}   '
            f'{full.R2:>9.4f} {control.R2:>9.4f} {full.R2 - control.R2:>+8.4f}   '
            f'{full.MAPE:>9.2f}% {control.MAPE:>9.2f}% '
            f'{full.MAPE - control.MAPE:>+7.2f}%')

    columns = ['mass_accounting_ratio_median', 'fpa_violation_rate',
               'gwc_violation_rate_valid_gate', 'ff1_transition_gap']
    physics = diagnostics[diagnostics.partition == 'test'].groupby(
        ['flights_per_type', 'condition'], as_index=False)[columns].mean()
    physics.to_csv(summary_dir / 'sampling_physics.csv', index=False)
    log('\n测试集上的物理诊断')
    log(f'{"每机型":>6} {"条件":>11}  {"油耗账":>8} {"航迹角违反":>11} '
        f'{"重量违反":>10} {"相位跳变":>10}')
    for level in sorted(physics.flights_per_type.unique(),
                        key=lambda v: (v == 0, v)):
        for condition in ('full', 'no_physics'):
            row = physics[(physics.flights_per_type == level)
                          & (physics.condition == condition)]
            if row.empty:
                continue
            r = row.iloc[0]
            name = '全部' if level == 0 else str(level)
            log(f'{name:>6} {condition:>11}  {r.mass_accounting_ratio_median:>8.3f} '
                f'{r.fpa_violation_rate * 100:>10.2f}% '
                f'{r.gwc_violation_rate_valid_gate * 100:>9.2f}% '
                f'{r.ff1_transition_gap * 100:>9.2f}%')
    log(f'\n汇总已写入 {summary_dir}')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='fleet-wide 航班采样实验')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 8, 0],
                        help='每机型训练航班数，0 表示全部')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--draw', type=int, default=1, help='限量时的抽样组号')
    parser.add_argument('--conditions', nargs='+', default=['full', 'no_physics'],
                        choices=['full', 'no_physics'])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--summary_dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--force', action='store_true', help='已完成的格子也重跑')
    parser.add_argument('--collect_only', action='store_true',
                        help='不训练，只汇总已有结果')
    args = parser.parse_args(argv)

    cells: List[Tuple[int, bool]] = [
        (level, condition == 'no_physics')
        for level in args.levels for condition in args.conditions]

    log('fleet-wide 航班采样实验')
    log(f'档位 {args.levels}（0 表示全部）   条件 {args.conditions}   '
        f'种子 {args.seeds}')
    pending = []
    for level, no_physics in cells:
        run_dir = args.out_dir / cell_tag(level, args.draw, no_physics)
        done = is_done(run_dir, args.seeds) and not args.force
        log(f'  {cell_tag(level, args.draw, no_physics):<32} '
            + ('已完成，跳过' if done else '待跑'))
        if not done:
            pending.append((level, no_physics))

    if not args.collect_only and pending:
        log(f'\n共需训练 {len(pending)} 格 × {len(args.seeds)} 个种子')
        for index, (level, no_physics) in enumerate(pending, 1):
            started = time.perf_counter()
            log('\n' + '#' * 88)
            log(f'# 第 {index}/{len(pending)} 格   '
                f'{cell_tag(level, args.draw, no_physics)}')
            log('#' * 88)
            run_cell(level, args.draw, no_physics, args.seeds,
                     args.out_dir, args.device, args.epochs)
            log(f'本格耗时 {(time.perf_counter() - started) / 60:.1f} 分钟')

    metrics, diagnostics = collect(args.out_dir, args.levels, args.draw, args.seeds)
    report(metrics, diagnostics, args.summary_dir)


if __name__ == '__main__':
    main()
