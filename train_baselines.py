# -*- coding: utf-8 -*-
"""基线模型

MLP、RBF、随机森林三个基线，与 PINN 共用同一份数据划分，回答的是"不用这套结构、
常规做法能到什么水平"。隔离物理约束靠的是消融实验（同结构、权重置零），不是这里。

与旧版基线相比改了三处：

  1. 机型。旧版训练机型写的是含 A350-900 的四个，那是数据纠错前的版本；现在源机型
     是 A320、A320-NEO、A321 三个。
  2. 输入。旧版只有七个飞行状态量（且含 time），相位仅用于剔除滑行，机型信息完全
     没有。现在是 7 个飞行状态 + 4 维机型静态特征，共 11 维。机型特征必须给，
     否则基线对留出机型无从下手，零样本这一栏没法比。相位不给：按相位分开处理
     是本文的结构，喂一份相位独热等于把这个模块的简化版送给基线。
  3. 划分。直接调用 af_data 的划分函数，与 PINN 用同一批训练、验证、测试航班。
  4. 输出头。旧版是 sigmoid·1.5，与 PINN 的 Softplus 不同，且在低油流区压梯度。
     现已统一为 Softplus，两边唯一的差别只剩主干结构。

主干结构沿用 comparative_models_MLP_RBF.py，未作改动。

只跑其中一个模型时（`--models mlp`），其余模型已有的结果会保留，不会被覆盖。
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from af_data import (
    AIRCRAFT_FEATURE_NAMES, AIRCRAFT_FEATURE_TABLE, FEATURE_NAMES,
    STUDY_AIRCRAFT_TYPES, TRAIN_AIRCRAFT_TYPES, WITHHELD_AIRCRAFT_TYPE,
    FlightRecordDataset, load_flight_table, split_fleet_wide, split_zero_shot,
)
from af_train import evaluate_outputs, log, resolve_device, set_seed

DEFAULT_OUT_DIR = Path('results/baselines')


# ── 模型：沿用旧基线脚本的定义 ──────────────────────────────────

class FuelFlowHead(nn.Module):
    """输出头与 PINN 一致：Softplus 保正，不设上界。

    旧版用的是 sigmoid(x)·1.5。那个上界从未被触及（两个模型在留出机型上的预测
    最大值都只有 1.05），但 sigmoid 在低油流区把梯度压得很扁——输出 0.05 时链式
    因子只有 0.048，输出 0.75 时是 0.375，差了近八倍，基线因此在低油流区被结构性
    削弱。换成 Softplus 之后，基线与 PINN 唯一的差别就只剩主干结构。
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x):
        return F.softplus(self.linear(x)) + 1e-6


class MLPFuelFlow(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = FuelFlowHead(hidden_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class RBFLayer(nn.Module):
    def __init__(self, in_features: int, num_centers: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.log_gamma = nn.Parameter(torch.zeros(num_centers))

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=2)
        return torch.exp(-torch.exp(self.log_gamma).unsqueeze(0) * dist2)


class RBFFuelFlow(nn.Module):
    def __init__(self, input_dim: int, num_centers: int = 50, hidden_dim: int = 64):
        super().__init__()
        self.rbf = RBFLayer(input_dim, num_centers)
        self.fc = nn.Linear(num_centers, hidden_dim)
        self.head = FuelFlowHead(hidden_dim)

    def forward(self, x):
        return self.head(torch.relu(self.fc(self.rbf(x))))


# ── 输入：七个飞行状态 + 相位独热 + 机型静态特征 ────────────────

def build_inputs(dataset: FlightRecordDataset) -> torch.Tensor:
    """七个飞行状态 + 四维机型静态特征，共 11 维。"""
    table = torch.tensor(
        [AIRCRAFT_FEATURE_TABLE[name] for name in STUDY_AIRCRAFT_TYPES],
        dtype=torch.float32)
    aircraft = table[dataset.aircraft_type_ids.cpu()]
    return torch.cat([dataset.features.cpu(), aircraft], dim=1)


def input_names() -> List[str]:
    return list(FEATURE_NAMES) + list(AIRCRAFT_FEATURE_NAMES)


def standardize(train_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    return mean, scale


# ── 训练 ────────────────────────────────────────────────────────

def train_torch_model(model: nn.Module, train_x, train_y, val_x, val_y,
                      args, device: torch.device, name: str) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience)
    best_state, best_val, best_epoch, stalled = None, float('inf'), 0, 0
    n = len(train_x)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(n, device=device)
        total, steps = 0.0, 0
        for start in range(0, n, args.batch_size):
            index = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(train_x[index]), train_y[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(loss.detach()); steps += 1

        model.eval()
        with torch.no_grad():
            squared, count = 0.0, 0
            for start in range(0, len(val_x), args.eval_batch_size):
                stop = start + args.eval_batch_size
                squared += F.mse_loss(model(val_x[start:stop]), val_y[start:stop],
                                      reduction='sum').item()
                count += val_y[start:stop].numel()
            val_mse = squared / count
        scheduler.step(val_mse)

        improved = val_mse < best_val
        if improved:
            best_val, best_epoch, stalled = val_mse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stalled += 1
        log(f'  {name} epoch {epoch:03d}   train {total / steps:.6f}   '
            f'val {val_mse:.6f}   lr {optimizer.param_groups[0]["lr"]:.2e}   '
            f'{time.perf_counter() - started:.1f}s{"   saved" if improved else ""}')
        if stalled >= args.early_stopping_patience:
            log(f'  {name} early stopping   best epoch {best_epoch}   val {best_val:.6f}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def predict_torch(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(out)


def run_random_forest(train_x, train_y, partitions, args, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    x = train_x.cpu().numpy()
    y = train_y.cpu().numpy()
    if args.rf_max_samples and len(x) > args.rf_max_samples:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(x), args.rf_max_samples, replace=False)
        x, y = x[pick], y[pick]
        log(f'  随机森林训练样本下采样至 {len(x):,} 条')
    forest = RandomForestRegressor(
        n_estimators=args.rf_trees, max_depth=args.rf_max_depth,
        n_jobs=-1, random_state=seed)
    forest.fit(x, y)
    predictions = {name: forest.predict(px.cpu().numpy())
                   for name, (px, _) in partitions.items()}
    return forest, predictions


# ── 主流程 ──────────────────────────────────────────────────────

def save_metrics(run_dir: Path, rows: List[pd.DataFrame],
                 models: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    """把本次跑出来的行并回已有结果。

    同模型同种子的旧行被替换，其余模型的行原样保留，这样单独重跑 MLP 不会把
    RBF 与随机森林的结果抹掉。
    """
    metrics = pd.concat(rows, ignore_index=True)
    path = run_dir / 'baseline_metrics.csv'
    if path.exists():
        old = pd.read_csv(path)
        stale = old['model'].isin(models) & old['seed'].isin(seeds)
        metrics = pd.concat([old[~stale], metrics], ignore_index=True)
    metrics = metrics.sort_values(
        ['model', 'seed', 'partition', 'aircraft_type', 'target'], kind='stable')
    metrics.to_csv(path, index=False)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='基线模型')
    p.add_argument('--split', choices=['zero_shot', 'fleet_wide'], default='zero_shot')
    p.add_argument('--models', nargs='+', default=['mlp', 'rbf', 'rf'],
                   choices=['mlp', 'rbf', 'rf'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--train_ratio', type=float, default=None)
    p.add_argument('--val_ratio', type=float, default=None)
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--eval_batch_size', type=int, default=8192)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--scheduler_factor', type=float, default=0.5)
    p.add_argument('--scheduler_patience', type=int, default=10)
    p.add_argument('--early_stopping_patience', type=int, default=25)
    p.add_argument('--gradient_clip', type=float, default=1.0)
    p.add_argument('--mlp_hidden', type=int, default=128)
    p.add_argument('--rbf_centers', type=int, default=50)
    p.add_argument('--rbf_hidden', type=int, default=64)
    p.add_argument('--rf_trees', type=int, default=100)
    p.add_argument('--rf_max_depth', type=int, default=None)
    p.add_argument('--rf_max_samples', type=int, default=200000)
    p.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return p


def prepare(args, device: torch.device):
    data = load_flight_table()
    if args.split == 'zero_shot':
        train_ratio = args.train_ratio if args.train_ratio is not None else 0.8
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.2
        train_frame, val_frame, test_frame, manifest = split_zero_shot(
            data, train_ratio, val_ratio, args.split_seed)
        test_name = 'zero_shot_test'
        log(f'训练机型 {list(TRAIN_AIRCRAFT_TYPES)}   留出 {WITHHELD_AIRCRAFT_TYPE}')
    else:
        train_ratio = args.train_ratio if args.train_ratio is not None else 0.7
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.1
        train_frame, val_frame, test_frame, manifest = split_fleet_wide(
            data, train_ratio, val_ratio, args.test_ratio, args.split_seed)
        test_name = 'test'
        log(f'四机型全训 {list(STUDY_AIRCRAFT_TYPES)}')

    datasets = {
        'train': FlightRecordDataset(train_frame, 'train'),
        'validation': FlightRecordDataset(val_frame, 'validation'),
        test_name: FlightRecordDataset(test_frame, test_name),
    }
    for name, dataset in datasets.items():
        log(f'  {name:<16} {len(dataset):>8,} 条   {dataset.n_flights():>3} flights   '
            f'{dataset.aircraft_types()}')

    tensors = {}
    for name, dataset in datasets.items():
        tensors[name] = (build_inputs(dataset), dataset.targets.cpu())
    mean, scale = standardize(tensors['train'][0])
    for name, (x, y) in tensors.items():
        tensors[name] = (((x - mean) / scale).to(device), y.to(device))
    log(f'输入 {len(input_names())} 维: {input_names()}')
    return tensors, datasets, manifest, test_name


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    run_dir = args.out_dir / args.split
    run_dir.mkdir(parents=True, exist_ok=True)

    log(f'基线模型   划分 {args.split}   模型 {args.models}   '
        f'种子 {args.seeds}   device {device}')
    tensors, datasets, manifest, test_name = prepare(args, device)
    manifest.to_csv(run_dir / 'split_manifest.csv', index=False)

    input_dim = tensors['train'][0].shape[1]
    rows: List[pd.DataFrame] = []

    for seed in args.seeds:
        for model_name in args.models:
            set_seed(seed)
            log('\n' + '=' * 86)
            log(f'{model_name.upper()}   seed {seed}')
            log('=' * 86)
            started = time.perf_counter()

            model_dir = run_dir / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            if model_name == 'rf':
                partitions = {n: tensors[n] for n in ('validation', test_name)}
                model, predictions = run_random_forest(*tensors['train'], partitions,
                                                       args, seed)
                with open(model_dir / f'rf_seed{seed}.pkl', 'wb') as handle:
                    pickle.dump(model, handle)
            else:
                model = (MLPFuelFlow(input_dim, args.mlp_hidden) if model_name == 'mlp'
                         else RBFFuelFlow(input_dim, args.rbf_centers, args.rbf_hidden))
                log(f'  可训练参数 {sum(p.numel() for p in model.parameters()):,}')
                model = train_torch_model(model, *tensors['train'],
                                          *tensors['validation'], args, device, model_name)
                torch.save({'model_state_dict': model.state_dict(),
                            'model_name': model_name, 'input_dim': input_dim,
                            'input_names': input_names(), 'seed': seed,
                            'split': args.split,
                            'hidden': args.mlp_hidden if model_name == 'mlp'
                                      else args.rbf_hidden,
                            'centers': args.rbf_centers if model_name == 'rbf' else None},
                           model_dir / f'{model_name}_seed{seed}.pt')
                predictions = {n: predict_torch(model, tensors[n][0], args.eval_batch_size)
                               for n in ('validation', test_name)}

            elapsed = time.perf_counter() - started
            prediction_dir = run_dir / 'predictions'
            prediction_dir.mkdir(parents=True, exist_ok=True)
            for partition, pred in predictions.items():
                np.save(prediction_dir / f'{model_name}_seed{seed}_{partition}.npy',
                        np.asarray(pred, dtype=np.float32))
            for partition, pred in predictions.items():
                target = tensors[partition][1].cpu().numpy()
                types = datasets[partition].aircraft_type_ids.cpu().numpy()
                frame = evaluate_outputs(pred, target, types)
                frame.insert(0, 'partition', partition)
                frame.insert(0, 'model', model_name)
                frame.insert(0, 'seed', seed)
                frame['elapsed_seconds'] = round(elapsed, 1)
                rows.append(frame)

            shown = rows[-1]
            shown = shown[(shown['aircraft_type'] == 'all')]
            for row in shown.itertuples(index=False):
                log(f'  {test_name} {row.target:<9} R2 {row.R2:.4f}   '
                    f'RMSE {row.RMSE:.6f}   MAPE {row.MAPE:.2f}%')

            metrics = save_metrics(run_dir, rows, args.models, args.seeds)

    metrics = save_metrics(run_dir, rows, args.models, args.seeds)
    mean_table = metrics.groupby(['model', 'partition', 'aircraft_type', 'target'],
                                 as_index=False)[['R2', 'RMSE', 'MAE', 'MAPE']].mean()
    mean_table.to_csv(run_dir / f'baseline_mean_{len(args.seeds)}seed.csv', index=False)

    log('\n' + '=' * 86)
    log(f'{len(args.seeds)} 个种子均值   {test_name}   FF_Total')
    log('=' * 86)
    shown = mean_table[(mean_table['partition'] == test_name)
                       & (mean_table['target'] == 'FF_Total')]
    for row in shown.sort_values(['aircraft_type', 'model']).itertuples(index=False):
        log(f'  {row.model:<5} {row.aircraft_type:<10} R2 {row.R2:.4f}   '
            f'RMSE {row.RMSE:.6f}   MAPE {row.MAPE:.2f}%')
    with open(run_dir / 'baseline_config.json', 'w', encoding='utf-8') as handle:
        json.dump({'split': args.split, 'input_names': input_names(),
                   'train_types': list(TRAIN_AIRCRAFT_TYPES) if args.split == 'zero_shot'
                   else list(STUDY_AIRCRAFT_TYPES),
                   'withheld_type': WITHHELD_AIRCRAFT_TYPE if args.split == 'zero_shot' else None,
                   'seeds': args.seeds, 'split_seed': args.split_seed},
                  handle, ensure_ascii=False, indent=1)
    log(f'\n输出目录 {run_dir}')


if __name__ == '__main__':
    main()
