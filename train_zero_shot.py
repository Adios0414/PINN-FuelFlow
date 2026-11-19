# -*- coding: utf-8 -*-
"""逐一留出机型的零样本训练入口。

本轮三个源机型按完整航班切训练与验证，留出机型的全部航班只用于最终评估。
留出机型不参与输入标准化、机型特征标准化、取数、物理梯度、早停、调度与选模。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch

from af_data import (
    DATA_CSV, STUDY_AIRCRAFT_TYPES, WITHHELD_AIRCRAFT_TYPE,
    AIRCRAFT_TYPE_TO_ID, FlightRecordDataset, load_flight_table, split_zero_shot,
)
from af_model import PHASE_MODES
from af_train import (
    aggregate, evaluate_one_seed, log, manifest_digest, resolve_device,
    reusable_seed_result, train_one_seed,
)

DEFAULT_OUT_DIR = Path('results/zero_shot')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='逐一留出机型的零样本实验')
    p.add_argument('--data_csv', type=Path, default=DATA_CSV)
    p.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        '--withheld_type', '--withheld_aircraft_type',
        dest='withheld_aircraft_type', choices=STUDY_AIRCRAFT_TYPES,
        default=WITHHELD_AIRCRAFT_TYPE,
        help='整机型留出作 zero-shot 测试；默认沿用 A321-NEO',
    )
    p.add_argument('--tag', default=None, help='不给则按条件自动命名')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.2)

    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--eval_batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--scheduler_factor', type=float, default=0.5)
    p.add_argument('--scheduler_patience', type=int, default=10)
    p.add_argument('--early_stopping_patience', type=int, default=25)
    p.add_argument('--min_delta', type=float, default=0.0)
    p.add_argument('--gradient_clip', type=float, default=1.0)
    p.add_argument('--trajectory_window', type=int, default=32)

    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--aircraft_feature_dim', type=int, default=16)
    p.add_argument('--no_aircraft_features', action='store_true',
                   help='关闭机型特征一路，用于检验它对精度的贡献')
    p.add_argument('--phase_mode', choices=PHASE_MODES, default='per_phase',
                   help='per_phase 六个子网按相位硬路由；'
                        'shared 换成一个共用子网，形状不变，只去掉按相位取权重；'
                        'none 整块去掉，共享编码器直接进融合层')

    # 由 search_loss_weights.py 的网格搜索选定（调参机型 A320-NEO，种子 1）
    p.add_argument('--w_mass', type=float, default=0.10)
    p.add_argument('--w_force', type=float, default=10.0)
    p.add_argument('--w_phase', type=float, default=0.02)
    p.add_argument('--mass_huber_delta', type=float, default=0.10)
    p.add_argument('--window_seconds', type=int, default=900)
    p.add_argument('--mass_windows_per_step', type=int, default=2)
    p.add_argument('--transitions_per_step', type=int, default=64)
    p.add_argument('--oswald_efficiency', type=float, default=0.80)
    p.add_argument('--max_lift_coefficient', type=float, default=2.5)
    p.add_argument('--no_physics', action='store_true',
                   help='三项物理权重同时置零，得到结构相同的对照模型')
    p.add_argument('--flip_force', action='store_true',
                   help='把力项的方向反过来，要求越陡越省油、越重越省油。'
                        '惩罚项、权重、作用范围一律不变，只有方向是错的，'
                        '用于区分提升来自物理内容还是来自多了一个惩罚项')

    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke_window_seconds', type=int, default=180)
    p.add_argument('--max_batches_per_epoch', type=int, default=0)
    return p


def check_args(args: argparse.Namespace) -> None:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('seed 列表不得为空或重复')
    if min(args.epochs, args.batch_size, args.eval_batch_size) <= 0:
        raise ValueError('epochs 与批大小必须为正')
    if args.lr <= 0:
        raise ValueError('学习率必须为正')
    if not 0 < args.scheduler_factor < 1:
        raise ValueError('scheduler_factor 必须在 0 与 1 之间')
    if args.early_stopping_patience <= args.scheduler_patience:
        raise ValueError('early_stopping_patience 必须大于 scheduler_patience')
    if args.gradient_clip <= 0:
        raise ValueError('gradient_clip 必须为正')
    if min(args.w_mass, args.w_force, args.w_phase) < 0:
        raise ValueError('物理项权重不得为负')
    if args.flip_force and (args.no_physics or args.w_force == 0):
        raise ValueError('flip_force 需要力项生效，不能与 no_physics 或 w_force=0 同用')
    if min(args.window_seconds, args.mass_windows_per_step,
           args.transitions_per_step) <= 0:
        raise ValueError('取数参数必须为正')
    if args.max_batches_per_epoch < 0:
        raise ValueError('max_batches_per_epoch 不得为负')


def check_no_leakage(
    train_dataset: FlightRecordDataset,
    val_dataset: FlightRecordDataset,
    test_dataset: FlightRecordDataset,
    withheld_aircraft_type: str,
    train_aircraft_types: Sequence[str],
) -> None:
    held_id = AIRCRAFT_TYPE_TO_ID[withheld_aircraft_type]
    source_ids = {AIRCRAFT_TYPE_TO_ID[name] for name in train_aircraft_types}
    train_ids = set(train_dataset.aircraft_type_ids.detach().cpu().tolist())
    val_ids = set(val_dataset.aircraft_type_ids.detach().cpu().tolist())
    test_ids = set(test_dataset.aircraft_type_ids.detach().cpu().tolist())
    if train_ids != source_ids:
        raise RuntimeError('训练张量机型与三个源机型不一致')
    if val_ids != source_ids:
        raise RuntimeError('验证张量机型与三个源机型不一致')
    if test_ids != {held_id}:
        raise RuntimeError('测试张量必须只含留出机型')


def auto_tag(args) -> str:
    """输出目录名。驱动脚本一律调用这里，不要另抄一份命名规则。"""
    if args.no_physics:
        return 'no_physics'
    if args.flip_force:
        return 'flip_force'
    if args.no_aircraft_features:
        return 'no_aircraft'
    if args.phase_mode == 'none':
        return 'no_phase'
    if args.phase_mode == 'shared':
        return 'shared_phase'
    return 'full'


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    check_args(args)
    train_aircraft_types = tuple(
        name for name in STUDY_AIRCRAFT_TYPES
        if name != args.withheld_aircraft_type
    )
    # 手动运行非默认目标时也使用独立目录，避免误写旧 A321-NEO 结果。
    if (args.withheld_aircraft_type != WITHHELD_AIRCRAFT_TYPE
            and args.out_dir.resolve() == DEFAULT_OUT_DIR.resolve()):
        args.out_dir = (DEFAULT_OUT_DIR / 'by_withheld_type'
                        / args.withheld_aircraft_type)
    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.epochs = min(args.epochs, 2)

    device = resolve_device(args.device)
    if args.tag is None:
        args.tag = auto_tag(args)
    run_dir = args.out_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    log(f'逐一留出机型的零样本实验   四机型 {list(STUDY_AIRCRAFT_TYPES)}')
    log(f'训练机型 {list(train_aircraft_types)}   '
        f'留出机型 {args.withheld_aircraft_type}')
    log(f'device {device}   seeds {args.seeds}')
    if args.smoke:
        log('smoke 模式   最多 2 轮，每轮最多 3 批')

    data = load_flight_table(args.data_csv)
    log(f'读取数据 {args.data_csv}   {len(data):,} 条')
    train_frame, val_frame, test_frame, manifest = split_zero_shot(
        data, args.train_ratio, args.val_ratio, args.split_seed,
        withheld_aircraft_type=args.withheld_aircraft_type)

    train_dataset = FlightRecordDataset(train_frame, 'train')
    val_dataset = FlightRecordDataset(val_frame, 'validation')
    test_dataset = FlightRecordDataset(test_frame, 'zero_shot_test')
    check_no_leakage(
        train_dataset, val_dataset, test_dataset,
        args.withheld_aircraft_type, train_aircraft_types,
    )
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.to(device)

    for dataset in (train_dataset, val_dataset, test_dataset):
        log(f'  {dataset.name:<15} {len(dataset):>8,} 条   '
            f'{dataset.n_flights():>3} flights   剔除滑行 {dataset.n_dropped:>7,} 条   '
            f'{dataset.aircraft_types()}')

    manifest_path = run_dir / 'zero_shot_split_manifest.csv'
    manifest_ok = manifest_path.exists() and (
        pd.read_csv(manifest_path).to_csv(index=False) == manifest.to_csv(index=False))
    if manifest_path.exists() and not manifest_ok:
        log('已有划分清单与本次不一致，所有种子重跑')
    manifest.to_csv(manifest_path, index=False)
    log(f'划分清单 {manifest_path}')

    partitions = (('source_validation', val_dataset), ('zero_shot_test', test_dataset))
    all_metrics, all_diagnostics = [], []
    expected_manifest_sha256 = manifest_digest(manifest)

    for seed in args.seeds:
        cached = reusable_seed_result(
            run_dir / f'seed_{seed}', args, manifest_ok,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_train_aircraft_types=train_aircraft_types,
        )
        if cached is not None:
            log(f'种子 {seed} 已有完整结果且设置一致，跳过训练')
            metrics, diagnostics = cached
        else:
            checkpoint_path = train_one_seed(
                args, seed, train_dataset, val_dataset, manifest, run_dir, device,
                aircraft_feature_train_types=train_aircraft_types,
            )
            metrics, diagnostics = evaluate_one_seed(args, seed, checkpoint_path,
                                                     partitions, manifest, run_dir, device)
        all_metrics.append(metrics)
        all_diagnostics.append(diagnostics)

    aggregate(run_dir, pd.concat(all_metrics, ignore_index=True),
              pd.concat(all_diagnostics, ignore_index=True))
    log(f'输出目录 {run_dir}')


if __name__ == '__main__':
    main()
