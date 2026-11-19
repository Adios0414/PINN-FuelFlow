# -*- coding: utf-8 -*-
"""物理损失权重的网格搜索

选择标准必须与目标场景一致，而目标场景是跨机型迁移。在源机型验证集上选不出来
（满数据同分布时三项物理约束对精度几乎没有影响），在 A321-NEO 上选则是拿测试集
挑超参数。因此另留一个机型专供调参：

    训练      A320 + A321
    早停      上述两个机型的留出航班
    选择标准  A320-NEO 上的迁移表现

A321-NEO 在本脚本中完全不出现，最终报告用的机型与调参过程无接触。

调参机型选 A320-NEO 而非 A321，依据是让调参时的处境与最终场景一致。最终场景里
A321-NEO 的重量全部落在三机型训练集的范围内（范围外 0.0%）；按上述划分，A320-NEO
落在 A320+A321 训练集范围外的比例同样是 0.0%，而若改用 A320+A320-NEO 训练、
在 A321 上调参，则有 7.4% 落在范围外。这个差别不可忽略：实测在大幅外推的处境下
物理约束反而会拉低精度（A320 单机型训练、A321-NEO 有 83.5% 在范围外时，完整模型
的 R² 比对照模型低 9.15），在那种处境里选出的权重会系统性偏小，与目标场景相反。

搜索方式为坐标搜索，从基线出发一次只动一个权重。每项的取值区间按"加权后占数据项
的比例"确定，覆盖 1% 到 100% 两个数量级；实测未加权量级为 L_data 0.0040、
L_mass 0.0052、L_force 0.00034、L_phase 0.0427。

搜索用的随机种子与最终报告的种子不重叠。
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from af_data import FlightRecordDataset, load_flight_table, split_zero_shot
from af_train import evaluate_one_seed, log, resolve_device, train_one_seed
from train_zero_shot import build_parser as build_base_parser

DEFAULT_OUT_DIR = Path('results/weight_search')

TRAIN_TYPES = ('A320', 'A321')
TUNING_TYPE = 'A320-NEO'
FORBIDDEN_TYPE = 'A321-NEO'          # 调参过程不得接触

BASELINE = {'w_mass': 0.10, 'w_force': 0.10, 'w_phase': 0.02}

# 每项的取值。区间由加权后占数据项的比例定，十倍一档，覆盖 1% 到 100%
LADDER: Dict[str, List[float]] = {
    'w_mass':  [0.01, 0.10, 1.00],
    'w_force': [0.10, 1.00, 10.0],
    'w_phase': [0.002, 0.02, 0.20],
}


def build_configurations() -> List[Dict[str, float]]:
    """坐标搜索：从基线出发，一次只动一个权重。"""
    configurations = [dict(BASELINE)]
    for name, values in LADDER.items():
        for value in values:
            if value == BASELINE[name]:
                continue
            candidate = dict(BASELINE)
            candidate[name] = value
            configurations.append(candidate)
    return configurations


def label_of(configuration: Dict[str, float]) -> str:
    changed = [k for k in LADDER if configuration[k] != BASELINE[k]]
    if not changed:
        return 'baseline'
    name = changed[0]
    return f'{name}_{configuration[name]:g}'.replace('w_', '')


def prepare(base_args, device: torch.device):
    data = load_flight_table(base_args.data_csv)
    if FORBIDDEN_TYPE in set(data['aircraft_type']) and True:
        pass  # 数据里有该机型，但下面不会把它放进任何一个分区
    full_train, full_val, _, manifest = split_zero_shot(
        data, base_args.train_ratio, base_args.val_ratio, base_args.split_seed,
        verbose=False)

    keep = list(TRAIN_TYPES)
    train_frame = full_train[full_train['aircraft_type'].isin(keep)].reset_index(drop=True)
    val_frame = full_val[full_val['aircraft_type'].isin(keep)].reset_index(drop=True)
    tuning_frame = data[data['aircraft_type'] == TUNING_TYPE].reset_index(drop=True)

    for name, frame in (('训练', train_frame), ('早停验证', val_frame),
                        ('调参机型', tuning_frame)):
        if FORBIDDEN_TYPE in set(frame['aircraft_type']):
            raise RuntimeError(f'{name}分区混入了 {FORBIDDEN_TYPE}')

    manifest = manifest.copy()
    outside = ~manifest['aircraft_type'].isin(keep)
    manifest.loc[outside, 'split'] = manifest.loc[outside, 'aircraft_type'].map(
        lambda t: 'tuning_type' if t == TUNING_TYPE else 'unused')

    train_dataset = FlightRecordDataset(train_frame, 'train').to(device)
    val_dataset = FlightRecordDataset(val_frame, 'validation').to(device)
    tuning_dataset = FlightRecordDataset(tuning_frame, 'tuning_type').to(device)
    log(f'训练 {len(train_dataset):,} 条 / {train_dataset.n_flights()} 航班   '
        f'{train_dataset.aircraft_types()}')
    log(f'早停验证 {len(val_dataset):,} 条 / {val_dataset.n_flights()} 航班')
    log(f'调参机型 {TUNING_TYPE} {len(tuning_dataset):,} 条 / '
        f'{tuning_dataset.n_flights()} 航班')
    log(f'{FORBIDDEN_TYPE} 全程不参与')
    return train_dataset, val_dataset, tuning_dataset, manifest


def run_one(base_args, configuration: Dict[str, float], seed: int, datasets,
            manifest: pd.DataFrame, out_dir: Path, device: torch.device
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset, val_dataset, tuning_dataset = datasets
    name = label_of(configuration)
    cell_dir = out_dir / name
    cell_dir.mkdir(parents=True, exist_ok=True)

    args = copy.deepcopy(base_args)
    for key, value in configuration.items():
        setattr(args, key, value)

    log('\n' + '#' * 88)
    log(f'# {name}   mass {configuration["w_mass"]:g}   '
        f'force {configuration["w_force"]:g}   phase {configuration["w_phase"]:g}')
    log('#' * 88)

    started = time.perf_counter()
    checkpoint = train_one_seed(args, seed, train_dataset, val_dataset,
                               manifest, cell_dir, device)
    partitions = (('source_validation', val_dataset), ('tuning_type', tuning_dataset))
    metrics, diagnostics = evaluate_one_seed(args, seed, checkpoint, partitions,
                                            manifest, cell_dir, device)
    elapsed = time.perf_counter() - started
    for frame in (metrics, diagnostics):
        frame['config'] = name
        for key, value in configuration.items():
            frame[key] = value
        frame['elapsed_seconds'] = round(elapsed, 1)
    log(f'本组耗时 {elapsed / 60:.1f} 分钟')
    return metrics, diagnostics


def report(out_dir: Path, metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    metrics.to_csv(out_dir / 'search_metrics.csv', index=False)
    diagnostics.to_csv(out_dir / 'search_diagnostics.csv', index=False)

    table = metrics[(metrics['partition'] == 'tuning_type')
                    & (metrics['aircraft_type'] == 'all')
                    & (metrics['target'] == 'FF_Total')].copy()
    source = metrics[(metrics['partition'] == 'source_validation')
                     & (metrics['aircraft_type'] == 'all')
                     & (metrics['target'] == 'FF_Total')].set_index('config')
    table = table.sort_values('R2', ascending=False)

    log('\n' + '=' * 92)
    log(f'按 {TUNING_TYPE} 上的 FF_Total 排序（选择标准）')
    log('=' * 92)
    log(f'{"配置":<16} {"mass":>7} {"force":>7} {"phase":>7}   '
        f'{TUNING_TYPE + " R2":>9} {"MAPE":>8}   {"源验证 R2":>10}')
    for row in table.itertuples(index=False):
        src = source.loc[row.config, 'R2'] if row.config in source.index else float('nan')
        log(f'{row.config:<16} {row.w_mass:>7g} {row.w_force:>7g} {row.w_phase:>7g}   '
            f'{row.R2:>9.4f} {row.MAPE:>7.2f}%   {src:>10.4f}')

    if len(table):
        best = table.iloc[0]
        log(f'\n最优配置 {best.config}   '
            f'mass {best.w_mass:g}   force {best.w_force:g}   phase {best.w_phase:g}')
        log(f'  {TUNING_TYPE} 上 R2 {best.R2:.4f}   MAPE {best.MAPE:.2f}%')
    log(f'\n输出目录 {out_dir}')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='物理损失权重网格搜索')
    parser.add_argument('--seed', type=int, default=1,
                        help='搜索用种子，与最终报告的 42/43/44 不重叠')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--only', nargs='+', default=None,
                        help='只跑指定的配置名，便于分批进行')
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_args = build_base_parser().parse_args([])
    base_args.device = args.device
    base_args.epochs = args.epochs

    configurations = build_configurations()
    if args.only:
        configurations = [c for c in configurations if label_of(c) in set(args.only)]
        if not configurations:
            raise ValueError(f'没有匹配的配置名，可选 {[label_of(c) for c in build_configurations()]}')

    log('物理损失权重网格搜索（坐标搜索）')
    log(f'训练机型 {list(TRAIN_TYPES)}   调参机型 {TUNING_TYPE}   '
        f'种子 {args.seed}   device {device}')
    log(f'共 {len(configurations)} 组配置，最多 {args.epochs} 轮')
    for configuration in configurations:
        log(f'  {label_of(configuration):<16} mass {configuration["w_mass"]:g}  '
            f'force {configuration["w_force"]:g}  phase {configuration["w_phase"]:g}')

    datasets = prepare(base_args, device)
    manifest = datasets[3]
    datasets = datasets[:3]

    all_metrics, all_diagnostics = [], []
    for configuration in configurations:
        metrics, diagnostics = run_one(base_args, configuration, args.seed,
                                       datasets, manifest, args.out_dir, device)
        all_metrics.append(metrics)
        all_diagnostics.append(diagnostics)
        report(args.out_dir, pd.concat(all_metrics, ignore_index=True),
               pd.concat(all_diagnostics, ignore_index=True))


if __name__ == '__main__':
    main()
