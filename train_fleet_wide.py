# -*- coding: utf-8 -*-
"""fleet-wide 训练入口

四个机型全部参与训练，按机型分层、以完整航班为单位切 train / validation / test。
一套固定超参数，一个种子，训练一个模型。

`--flights_per_type` 用于限量：每个机型只取若干个训练航班，验证集与测试集不变。
不给该参数就是全量。两种情况共用同一份划分，结果可直接比较。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from af_data import (
    STUDY_AIRCRAFT_TYPES, FlightRecordDataset, load_flight_table, split_fleet_wide,
)
from af_model import PHASE_MODES
from af_train import (
    aggregate, evaluate_one_seed, log, resolve_device, reusable_seed_result,
    train_one_seed,
)
from train_zero_shot import build_parser as build_base_parser

DEFAULT_OUT_DIR = Path('results/fleet_wide')


def subsample_train_flights(train_frame: pd.DataFrame, flights_per_type: int,
                            draw_seed: int, verbose: bool = True
                            ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """每个机型只保留若干个训练航班；为 0 时取全部。"""
    chosen: Dict[str, List[str]] = {}
    for aircraft_type in STUDY_AIRCRAFT_TYPES:
        flights = np.array(
            sorted(train_frame.loc[train_frame['aircraft_type'] == aircraft_type,
                                   'flight_id'].astype(str).unique(), key=str),
            dtype=object)
        if flights_per_type == 0:
            chosen[aircraft_type] = sorted(flights.tolist(), key=str)
            continue
        if len(flights) < flights_per_type:
            raise ValueError(
                f'{aircraft_type} 只有 {len(flights)} 个训练航班，'
                f'取不出 {flights_per_type} 个')
        # 不能用 hash(str)：Python 每个进程给字符串哈希加不同的盐，同样的参数
        # 换个进程就会抽到另一批航班。改用机型在固定词表里的下标。
        rng = np.random.default_rng([draw_seed,
                                     STUDY_AIRCRAFT_TYPES.index(aircraft_type)])
        picked = flights[rng.permutation(len(flights))][:flights_per_type]
        chosen[aircraft_type] = sorted(picked.tolist(), key=str)

    keep = set().union(*chosen.values())
    subset = train_frame[
        train_frame['flight_id'].astype(str).isin(keep)].reset_index(drop=True)
    if verbose:
        for aircraft_type, picked in chosen.items():
            n = int((subset['aircraft_type'] == aircraft_type).sum())
            print(f'  {aircraft_type:<9} 训练航班 {len(picked):>3} 个   {n:>8,} 条',
                  flush=True)
    return subset, chosen


def mark_unused(manifest: pd.DataFrame, chosen: Dict[str, List[str]]) -> pd.DataFrame:
    manifest = manifest.copy()
    picked = set().union(*chosen.values())
    unused = ((manifest['split'] == 'train')
              & ~manifest['flight_id'].astype(str).isin(picked))
    manifest.loc[unused, 'split'] = 'unused'
    return manifest.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='fleet-wide 训练')
    parser.add_argument('--flights_per_type', type=int, default=0,
                        help='每个机型取几个训练航班，0 表示全部')
    parser.add_argument('--draw', type=int, default=1, help='限量时的抽样组号')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--tag', default=None, help='不给则按数据量自动命名')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no_physics', action='store_true',
                        help='三项物理权重置零，得到结构相同的对照模型')
    parser.add_argument('--no_aircraft_features', action='store_true')
    parser.add_argument('--phase_mode', choices=PHASE_MODES, default='per_phase')
    return parser


def auto_tag(args) -> str:
    """输出目录名。驱动脚本一律调用这里，不要另抄一份命名规则。"""
    suffix = ('_no_physics' if args.no_physics
              else '_no_aircraft' if args.no_aircraft_features
              else '_no_phase' if args.phase_mode == 'none'
              else '_shared_phase' if args.phase_mode == 'shared'
              else '_full')
    stem = ('all_flights' if args.flights_per_type == 0
            else f'{args.flights_per_type}per_type_draw{args.draw}')
    return stem + suffix


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    # 训练超参数沿用 zero-shot 入口的同一套默认值
    base = build_base_parser().parse_args([])
    base.device = args.device
    base.seeds = list(args.seeds)
    base.split_seed = args.split_seed
    base.train_ratio = args.train_ratio
    base.val_ratio = args.val_ratio
    base.epochs = args.epochs
    base.no_physics = args.no_physics
    base.no_aircraft_features = args.no_aircraft_features
    base.phase_mode = args.phase_mode

    tag = args.tag or auto_tag(args)
    run_dir = args.out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    log('fleet-wide 训练')
    log(f'四机型 {list(STUDY_AIRCRAFT_TYPES)}   种子 {args.seeds}   device {device}')
    log(f'划分 {args.train_ratio:.1f}/{args.val_ratio:.1f}/{args.test_ratio:.1f}   '
        f'划分种子 {args.split_seed}')
    log(f'物理约束 {"关闭" if args.no_physics else "开启"}   '
        f'训练航班 {"全部" if args.flights_per_type == 0 else f"每机型 {args.flights_per_type} 个"}')

    data = load_flight_table(base.data_csv)
    log(f'\n读取数据 {base.data_csv}   {len(data):,} 条')
    train_frame, val_frame, test_frame, manifest = split_fleet_wide(
        data, args.train_ratio, args.val_ratio, args.test_ratio, args.split_seed)

    if args.flights_per_type:
        log('\n限量取训练航班')
        train_frame, chosen = subsample_train_flights(
            train_frame, args.flights_per_type, 1000 * args.flights_per_type + args.draw)
        manifest = mark_unused(manifest, chosen)

    train_dataset = FlightRecordDataset(train_frame, 'train').to(device)
    val_dataset = FlightRecordDataset(val_frame, 'validation').to(device)
    test_dataset = FlightRecordDataset(test_frame, 'test').to(device)
    log('')
    for dataset in (train_dataset, val_dataset, test_dataset):
        log(f'  {dataset.name:<12} {len(dataset):>8,} 条   '
            f'{dataset.n_flights():>3} flights   {dataset.aircraft_types()}')

    manifest_path = run_dir / 'fleet_wide_split_manifest.csv'
    # 只有当目录里已存的划分与本次算出的完全一致时，才允许复用已跑完的种子
    manifest_ok = False
    if manifest_path.exists():
        manifest_ok = (pd.read_csv(manifest_path).to_csv(index=False)
                       == manifest.to_csv(index=False))
        if not manifest_ok:
            log('已有划分清单与本次不一致，所有种子重跑')
    manifest.to_csv(manifest_path, index=False)
    log(f'划分清单 {manifest_path}')

    partitions = (('validation', val_dataset), ('test', test_dataset))
    all_metrics, all_diagnostics = [], []
    for seed in args.seeds:
        cached = reusable_seed_result(run_dir / f'seed_{seed}', base, manifest_ok)
        if cached is not None:
            log(f'种子 {seed} 已有完整结果且设置一致，跳过训练')
            metrics, diagnostics = cached
        else:
            checkpoint = train_one_seed(base, seed, train_dataset, val_dataset,
                                        manifest, run_dir, device)
            metrics, diagnostics = evaluate_one_seed(base, seed, checkpoint, partitions,
                                                     manifest, run_dir, device)
        for frame in (metrics, diagnostics):
            frame['flights_per_type'] = args.flights_per_type
            frame['n_train_flights'] = train_dataset.n_flights()
            frame['n_train_records'] = len(train_dataset)
            frame['condition'] = 'no_physics' if args.no_physics else 'full'
        all_metrics.append(metrics)
        all_diagnostics.append(diagnostics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    aggregate(run_dir, metrics, diagnostics)

    log(f'\n测试集逐机型 FF_Total（{len(args.seeds)} 个种子均值）')
    shown = (metrics[(metrics['partition'] == 'test')
                     & (metrics['target'] == 'FF_Total')]
             .groupby('aircraft_type', as_index=False)[['n_records', 'R2', 'RMSE', 'MAPE']]
             .mean())
    for row in shown.itertuples(index=False):
        log(f'  {row.aircraft_type:<10} {row.n_records:>9,.0f} 条   R2 {row.R2:.4f}   '
            f'RMSE {row.RMSE:.6f}   MAPE {row.MAPE:.2f}%')
    log(f'\n输出目录 {run_dir}')


if __name__ == '__main__':
    main()
