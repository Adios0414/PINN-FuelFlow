# -*- coding: utf-8 -*-
"""训练循环、checkpoint、评估与物理诊断

诊断在训练结束后计算，不参与选模，也不改动预测。
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from af_data import (
    AIRCRAFT_FEATURE_NAMES, FEATURE_INDEX, FEATURE_NAMES, PHASE_NAMES,
    STUDY_AIRCRAFT_TYPES, TRAIN_AIRCRAFT_TYPES,
    CyclicBatches, FlightRecordDataset, TrajectoryWindowSampler,
    aircraft_feature_matrix, aircraft_feature_provenance, aircraft_feature_stats,
    build_mass_windows, build_transition_pairs,
    collate_mass_windows, collate_transitions,
)
from af_losses import LawBasedLoss
from af_model import FuelFlowNet

CHECKPOINT_FORMAT_VERSION = 1

# numpy 2.0 起 trapz 更名为 trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def log(message: str = '') -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求 CUDA，但当前 PyTorch 未检测到可用 GPU')
    return torch.device(requested)


def manifest_digest(manifest: pd.DataFrame) -> str:
    stable = manifest.sort_values(['aircraft_type', 'flight_id', 'split']).to_csv(
        index=False, lineterminator='\n')
    return hashlib.sha256(stable.encode('utf-8')).hexdigest()


def to_device(batch: Sequence[torch.Tensor], device: torch.device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


# ── checkpoint ──────────────────────────────────────────────────

def model_config(args) -> Dict[str, object]:
    return {
        'input_dim': len(FEATURE_NAMES),
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'aircraft_feature_dim': args.aircraft_feature_dim,
        'use_aircraft_features': not args.no_aircraft_features,
        'phase_mode': getattr(args, 'phase_mode', 'per_phase'),
    }


def loss_weights(args) -> Dict[str, float]:
    if args.no_physics:
        return {'w_mass': 0.0, 'w_force': 0.0, 'w_phase': 0.0}
    # 不按相位路由时两个分支是同一个网络（none 则根本没有这一块），
    # 相位项恒为零，直接关掉省一次前向
    per_phase = getattr(args, 'phase_mode', 'per_phase') == 'per_phase'
    return {'w_mass': float(args.w_mass), 'w_force': float(args.w_force),
            'w_phase': float(args.w_phase) if per_phase else 0.0}


def reusable_seed_result(
    seed_dir: Path,
    args,
    manifest_ok: bool,
    expected_manifest_sha256: Optional[str] = None,
    expected_train_aircraft_types: Optional[Sequence[str]] = None,
):
    """已有结果与本次设置逐项一致时返回 (metrics, diagnostics)，否则返回 None。

    比对划分清单、checkpoint 自身的划分摘要与源机型、模型结构配置、三个物理
    权重及力项方向，任一不同都重训，避免中断恢复时混入旧设置的种子。
    """
    metrics_path = seed_dir / 'metrics.csv'
    diagnostics_path = seed_dir / 'diagnostics.csv'
    checkpoint_path = seed_dir / 'best_model.pt'
    if not (manifest_ok and metrics_path.exists()
            and diagnostics_path.exists() and checkpoint_path.exists()):
        return None
    try:
        saved = torch.load(checkpoint_path, map_location='cpu',
                           weights_only=False)['metadata']
    except Exception:
        return None
    if (expected_manifest_sha256 is not None
            and saved.get('manifest_sha256') != expected_manifest_sha256):
        return None
    if (expected_train_aircraft_types is not None
            and saved.get('train_aircraft_types')
            != list(expected_train_aircraft_types)):
        return None
    # 旧 checkpoint 没有相位模式字段，那时只有六个子网硬路由一种情况
    saved_config = dict(saved.get('model_config') or {})
    saved_config.setdefault('phase_mode', 'per_phase')
    if saved_config != model_config(args):
        return None
    training = saved.get('training_config', {})
    expected = dict(loss_weights(args))
    expected['force_sign'] = -1.0 if getattr(args, 'flip_force', False) else 1.0
    # force_sign 与 --flip_force 是一起加的，没有这个字段的 checkpoint 必然是正方向
    training = dict(training)
    training.setdefault('force_sign', 1.0)
    for key, value in expected.items():
        if key not in training or abs(float(training[key]) - value) > 1e-12:
            return None
    return pd.read_csv(metrics_path), pd.read_csv(diagnostics_path)


def build_metadata(args, seed: int, model: FuelFlowNet, manifest: pd.DataFrame,
                   window_seconds: int, best_epoch: int, best_val_mse: float,
                   train_types: Optional[Sequence[str]] = None) -> Dict[str, object]:
    return {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'feature_names': list(FEATURE_NAMES),
        'phase_names': list(PHASE_NAMES),
        'aircraft_feature_names': list(AIRCRAFT_FEATURE_NAMES),
        'aircraft_feature_table': aircraft_feature_provenance(),
        'aircraft_types': list(STUDY_AIRCRAFT_TYPES),
        'study_aircraft_types': list(STUDY_AIRCRAFT_TYPES),
        'train_aircraft_types': list(train_types) if train_types
                                else list(TRAIN_AIRCRAFT_TYPES),
        'aircraft_feature_mean': model.aircraft_feature_mean.detach().cpu().tolist(),
        'aircraft_feature_scale': model.aircraft_feature_scale.detach().cpu().tolist(),
        'input_mean': model.x_mean.detach().cpu().tolist(),
        'input_scale': model.x_scale.detach().cpu().tolist(),
        'split_seed': args.split_seed,
        'model_seed': seed,
        'manifest_sha256': manifest_digest(manifest),
        'best_epoch': best_epoch,
        'best_validation_data_mse': best_val_mse,
        'model_config': model_config(args),
        'training_config': {
            'lr': args.lr, 'batch_size': args.batch_size, 'max_epochs': args.epochs,
            'scheduler_factor': args.scheduler_factor,
            'scheduler_patience': args.scheduler_patience,
            'early_stopping_patience': args.early_stopping_patience,
            'gradient_clip': args.gradient_clip,
            'trajectory_window': args.trajectory_window,
            'train_ratio': args.train_ratio, 'val_ratio': args.val_ratio,
            'mass_window_seconds': window_seconds,
            'mass_windows_per_step': args.mass_windows_per_step,
            'transitions_per_step': args.transitions_per_step,
            **loss_weights(args),
            'mass_huber_delta': args.mass_huber_delta,
            'oswald_efficiency': args.oswald_efficiency,
            'max_lift_coefficient': args.max_lift_coefficient,
            'force_sign': -1.0 if getattr(args, 'flip_force', False) else 1.0,
        },
    }


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_model(path: Path, manifest: pd.DataFrame, split_seed: int,
                  device: torch.device) -> Tuple[FuelFlowNet, Mapping[str, object]]:
    checkpoint = load_checkpoint(path, device)
    metadata = checkpoint.get('metadata')
    if not isinstance(metadata, Mapping):
        raise RuntimeError('checkpoint 缺少 metadata')

    expected = {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'feature_names': list(FEATURE_NAMES),
        'phase_names': list(PHASE_NAMES),
        'aircraft_feature_names': list(AIRCRAFT_FEATURE_NAMES),
        'aircraft_feature_table': aircraft_feature_provenance(),
        'aircraft_types': list(STUDY_AIRCRAFT_TYPES),
        'split_seed': split_seed,
        'manifest_sha256': manifest_digest(manifest),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'checkpoint 与当前设置不一致，字段 {key}')

    config = metadata['model_config']
    model = FuelFlowNet(
        aircraft_feature_table=aircraft_feature_matrix(),
        aircraft_feature_mean=torch.tensor(metadata['aircraft_feature_mean']),
        aircraft_feature_scale=torch.tensor(metadata['aircraft_feature_scale']),
        input_dim=int(config['input_dim']),
        hidden_dim=int(config['hidden_dim']),
        num_layers=int(config['num_layers']),
        aircraft_feature_dim=int(config['aircraft_feature_dim']),
        use_aircraft_features=bool(config['use_aircraft_features']),
        phase_mode=str(config.get('phase_mode', 'per_phase')),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), metadata


# ── 训练 ────────────────────────────────────────────────────────

def validate_data_mse(model: FuelFlowNet, dataset: FlightRecordDataset,
                      batch_size: int, max_batches: int = 0) -> float:
    model.eval()
    squared, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataset.iter_batches(batch_size)):
            if max_batches and i >= max_batches:
                break
            features, targets, phases, aircraft = batch
            predicted = model(features, phases, aircraft)
            squared += F.mse_loss(predicted, targets, reduction='sum').item()
            count += targets.numel()
    if count == 0:
        raise RuntimeError('验证集没有可评估的输出')
    return squared / count


def train_one_seed(
    args,
    seed: int,
    train_dataset: FlightRecordDataset,
    val_dataset: FlightRecordDataset,
    manifest: pd.DataFrame,
    run_dir: Path,
    device: torch.device,
    aircraft_feature_train_types: Optional[Sequence[str]] = None,
) -> Path:
    set_seed(seed)
    log('\n' + '=' * 86)
    log(f'seed {seed}   device {device}')
    log('=' * 86)

    input_mean, input_scale = train_dataset.input_stats()
    if aircraft_feature_train_types is None:
        feature_train_types = TRAIN_AIRCRAFT_TYPES
    else:
        feature_train_types = tuple(aircraft_feature_train_types)
        dataset_types = tuple(train_dataset.aircraft_types())
        if feature_train_types != dataset_types:
            raise ValueError(
                f'机型特征归一化源机型 {list(feature_train_types)} '
                f'与训练张量机型 {list(dataset_types)} 不一致')
    aircraft_mean, aircraft_scale = aircraft_feature_stats(feature_train_types)
    model = FuelFlowNet(
        aircraft_feature_table=aircraft_feature_matrix(),
        aircraft_feature_mean=aircraft_mean,
        aircraft_feature_scale=aircraft_scale,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        aircraft_feature_dim=args.aircraft_feature_dim,
        use_aircraft_features=not args.no_aircraft_features,
        phase_mode=args.phase_mode,
    )
    model.set_input_normalization(input_mean, input_scale)
    model.to(device)

    weights = loss_weights(args)
    w_mass, w_force, w_phase = (weights['w_mass'], weights['w_force'],
                                weights['w_phase'])
    loss_fn = LawBasedLoss(
        x_scale=input_scale, w_mass=w_mass, w_force=w_force, w_phase=w_phase,
        mass_huber_delta=args.mass_huber_delta,
        oswald_efficiency=args.oswald_efficiency,
        max_lift_coefficient=args.max_lift_coefficient,
        force_sign=-1.0 if getattr(args, 'flip_force', False) else 1.0,
    ).to(device)

    sampler = TrajectoryWindowSampler(train_dataset, batch_size=args.batch_size,
                                      window_size=args.trajectory_window, seed=seed)

    window_seconds = min(args.window_seconds, args.smoke_window_seconds) if args.smoke \
        else args.window_seconds
    mass_windows = build_mass_windows(train_dataset, window_seconds=window_seconds)
    rows, left, right = build_transition_pairs(train_dataset)
    transitions = list(zip(rows.tolist(), left.tolist(), right.tolist()))
    log(f'physics sampling   mass windows {len(mass_windows):,}   '
        f'observed transitions {len(transitions):,}')
    if w_mass > 0 and not mass_windows:
        raise RuntimeError('启用质量守恒，但训练分区切不出合格窗口')
    if w_phase > 0 and not transitions:
        raise RuntimeError('启用相位连续，但训练分区没有真实过渡点')

    mass_batches = CyclicBatches(mass_windows, args.mass_windows_per_step, seed + 1000)
    transition_batches = CyclicBatches(transitions, args.transitions_per_step, seed + 2000)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience)

    seed_dir = run_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / 'best_model.pt'
    history: List[Dict[str, float]] = []
    best_val, best_epoch, stalled = math.inf, 0, 0
    max_epochs = min(args.epochs, 2) if args.smoke else args.epochs
    max_batches = args.max_batches_per_epoch or (3 if args.smoke else 0)

    log(f'trainable parameters   {sum(p.numel() for p in model.parameters()):,}')
    log(f'aircraft feature block   {"off" if args.no_aircraft_features else "on"}')
    log(f'phase block   {args.phase_mode}')
    log(f'loss weights   data 1.00   mass {w_mass:.3f}   force {w_force:.3f}   '
        f'phase {w_phase:.3f}')
    if getattr(args, 'flip_force', False):
        log('force direction   FLIPPED   力项要求与物理相反的方向，仅用于归因对照')

    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        keys = ('total', 'data', 'mass', 'force', 'phase',
                'force_gamma', 'force_weight', 'force_gate_fraction')
        running = torch.zeros(len(keys), device=device)
        steps = 0

        for index, row_index in enumerate(sampler):
            if max_batches and index >= max_batches:
                break
            features, targets, phases, aircraft = train_dataset.batch(row_index)
            mass_batch = (collate_mass_windows(train_dataset, mass_batches.next())
                          if w_mass > 0 else None)
            transition_batch = (collate_transitions(train_dataset, transition_batches.next())
                                if w_phase > 0 else None)

            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model, features, targets, phases, aircraft,
                             mass_batch=mass_batch, transition_batch=transition_batch)
            if not bool(torch.isfinite(losses['total'])):
                raise FloatingPointError(f'epoch {epoch} 第 {index + 1} 批损失非有限')
            losses['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f'epoch {epoch} 第 {index + 1} 批梯度非有限')
            optimizer.step()

            # 八项合成一个张量累加，每批只搬运一次而不是八次
            running += torch.stack([losses[k].detach() for k in keys])
            steps += 1

        if steps == 0:
            raise RuntimeError('本轮没有执行任何优化步')
        means = dict(zip(keys, (running / steps).cpu().tolist()))
        val_mse = validate_data_mse(model, val_dataset, args.eval_batch_size,
                                    max_batches=3 if args.smoke else 0)
        scheduler.step(val_mse)
        lr = float(optimizer.param_groups[0]['lr'])
        elapsed = time.perf_counter() - started

        improved = val_mse < best_val - args.min_delta
        if improved:
            best_val, best_epoch, stalled = val_mse, epoch, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'metadata': build_metadata(
                            args, seed, model, manifest, window_seconds,
                            best_epoch, best_val, train_dataset.aircraft_types())},
                       checkpoint_path)
        else:
            stalled += 1

        history.append({'epoch': epoch, 'train_total': means['total'],
                        'train_data': means['data'], 'train_mass': means['mass'],
                        'train_force': means['force'], 'train_phase': means['phase'],
                        'train_force_gamma': means['force_gamma'],
                        'train_force_weight': means['force_weight'],
                        'force_gate_fraction': means['force_gate_fraction'],
                        'validation_data_mse': val_mse, 'learning_rate': lr,
                        'elapsed_seconds': elapsed})
        pd.DataFrame(history).to_csv(seed_dir / 'training_history.csv', index=False)
        log(f'epoch {epoch:03d}/{max_epochs:03d}   total {means["total"]:.6f}   '
            f'data {means["data"]:.6f}   mass {means["mass"]:.6f}   '
            f'force {means["force"]:.6f}   phase {means["phase"]:.6f}   '
            f'val_data {val_mse:.6f}   lr {lr:.2e}   {elapsed:.1f}s'
            f'{"   saved" if improved else ""}')

        if stalled >= args.early_stopping_patience:
            log(f'early stopping   best epoch {best_epoch}   val_data {best_val:.6f}')
            break

    if not checkpoint_path.exists():
        raise RuntimeError('训练结束后没有生成 checkpoint')
    return checkpoint_path


# ── 评估 ────────────────────────────────────────────────────────

def predict(model: FuelFlowNet, dataset: FlightRecordDataset, device: torch.device,
            batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataset.iter_batches(batch_size):
            features, batch_targets, phases, aircraft = batch
            predictions.append(model(features, phases, aircraft).cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - actual
    variation = float(np.sum((actual - actual.mean()) ** 2))
    return {
        'R2': (1.0 - float(np.sum(residual ** 2)) / variation) if variation > 0 else float('nan'),
        'RMSE': float(np.sqrt(np.mean(residual ** 2))),
        'MAE': float(np.mean(np.abs(residual))),
        'MAPE': float(np.mean(np.abs(residual) / np.maximum(np.abs(actual), 1e-6)) * 100.0),
    }


def evaluate_outputs(predictions: np.ndarray, targets: np.ndarray,
                     aircraft_type_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
    """整体指标，以及（给出机型时）逐机型的分解。"""
    def block(pred: np.ndarray, true: np.ndarray, label: str) -> List[Dict[str, object]]:
        out = []
        for name, actual, predicted in (
            ('FF1', true[:, 0], pred[:, 0]),
            ('FF2', true[:, 1], pred[:, 1]),
            ('FF_Total', true.sum(axis=1), pred.sum(axis=1)),
        ):
            row: Dict[str, object] = {'aircraft_type': label, 'target': name,
                                      'n_records': int(len(true))}
            row.update(regression_metrics(actual, predicted))
            out.append(row)
        return out

    rows = block(predictions, targets, 'all')
    if aircraft_type_ids is not None:
        for type_id in np.unique(aircraft_type_ids):
            mask = aircraft_type_ids == type_id
            rows += block(predictions[mask], targets[mask],
                          STUDY_AIRCRAFT_TYPES[int(type_id)])
    return pd.DataFrame(rows)


# ── 物理诊断 ────────────────────────────────────────────────────

def mass_accounting(predictions: np.ndarray, dataset: FlightRecordDataset) -> pd.DataFrame:
    total = predictions.sum(axis=1)
    times = dataset.time.detach().cpu().numpy().astype(np.float64)
    flights = dataset.flight_ids.detach().cpu().numpy()
    mass = dataset.features[:, FEATURE_INDEX['GWC']].detach().cpu().numpy().astype(np.float64)
    rows = []

    for flight_id in np.unique(flights):
        index = np.where(flights == flight_id)[0]
        index = index[np.argsort(times[index])]
        gaps = np.diff(times[index])
        breaks = np.where((gaps <= 0) | (gaps > 2.0))[0]
        burn, drop, seconds = 0.0, 0.0, 0.0
        for segment in np.split(index, breaks + 1):
            if len(segment) < 2:
                continue
            segment_times = times[segment]
            burn += float(_trapezoid(total[segment], segment_times))
            delta = float(mass[segment[0]] - mass[segment[-1]])
            drop += max(delta, 0.0)
            seconds += float(segment_times[-1] - segment_times[0])
        rows.append({'flight': str(dataset.flight_keys[flight_id]),
                     'predicted_burn_kg': burn, 'observed_mass_drop_kg': drop,
                     'accounting_ratio': burn / drop if drop > 0 else float('nan'),
                     'integrated_seconds': seconds})
    return pd.DataFrame(rows)


def force_response_rates(model: FuelFlowNet, dataset: FlightRecordDataset,
                         loss_fn: LawBasedLoss, device: torch.device,
                         batch_size: int, max_batches: int = 0) -> Dict[str, float]:
    gamma_bad = gamma_n = weight_bad = weight_valid = weight_n = 0
    model.eval()

    for index, batch in enumerate(dataset.iter_batches(batch_size)):
        if max_batches and index >= max_batches:
            break
        features, _, phases, aircraft = batch
        x = features.detach().clone().requires_grad_(True)
        total = model(x, phases, aircraft).sum(dim=1)
        grad, = torch.autograd.grad(total.sum(), x)
        fpa = x[:, FEATURE_INDEX['FPA']]
        tas = x[:, FEATURE_INDEX['TAS']]
        d_gamma = (grad[:, FEATURE_INDEX['FPA']]
                   + tas * torch.cos(fpa) * grad[:, FEATURE_INDEX['IVV']])
        applies, holds = loss_fn.gates(x.detach(), aircraft)
        weight_gate = applies & holds

        gamma_bad += int(((d_gamma < 0) & applies).sum())
        gamma_n += int(applies.sum())
        weight_bad += int(((grad[:, FEATURE_INDEX['GWC']] < 0) & weight_gate).sum())
        weight_valid += int(weight_gate.sum())
        weight_n += len(weight_gate)

    return {
        'fpa_violation_rate': gamma_bad / gamma_n if gamma_n else float('nan'),
        'gwc_violation_rate_valid_gate': weight_bad / weight_valid if weight_valid else float('nan'),
        'gwc_valid_gate_fraction': weight_valid / weight_n if weight_n else float('nan'),
        'force_samples': gamma_n,
    }


def transition_gaps(model: FuelFlowNet, dataset: FlightRecordDataset,
                    device: torch.device, batch_size: int) -> Dict[str, float]:
    rows, left, right = build_transition_pairs(dataset)
    if len(rows) == 0:
        return {'transition_count': 0, 'ff1_transition_gap': float('nan'),
                'ff2_transition_gap': float('nan')}

    gaps = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            index = torch.as_tensor(rows[start:stop], dtype=torch.long,
                                    device=dataset.features.device)
            features = dataset.features[index]
            aircraft = dataset.aircraft_type_ids[index]
            phase_left = torch.as_tensor(left[start:stop], dtype=torch.long, device=device)
            phase_right = torch.as_tensor(right[start:stop], dtype=torch.long, device=device)
            a = model(features, phase_left, aircraft)
            b = model(features, phase_right, aircraft)
            gaps.append(((a - b).abs() / a.clamp_min(1e-6)).cpu().numpy())
    gaps = np.concatenate(gaps)
    return {'transition_count': int(len(rows)),
            'ff1_transition_gap': float(gaps[:, 0].mean()),
            'ff2_transition_gap': float(gaps[:, 1].mean())}


def evaluate_one_seed(args, seed: int, checkpoint_path: Path,
                      partitions: Sequence[Tuple[str, FlightRecordDataset]],
                      manifest: pd.DataFrame, run_dir: Path, device: torch.device
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model, metadata = restore_model(checkpoint_path, manifest, args.split_seed, device)
    config = metadata['training_config']
    diagnostic_loss = LawBasedLoss(
        x_scale=torch.tensor(metadata['input_scale'], dtype=torch.float32),
        w_mass=0.0, w_force=0.0, w_phase=0.0,
        mass_huber_delta=float(config['mass_huber_delta']),
        oswald_efficiency=float(config['oswald_efficiency']),
        max_lift_coefficient=float(config['max_lift_coefficient']),
    ).to(device)

    seed_dir = run_dir / f'seed_{seed}'
    metric_frames, diagnostic_rows = [], []

    for partition, dataset in partitions:
        predictions, targets = predict(model, dataset, device, args.eval_batch_size)
        metrics = evaluate_outputs(
            predictions, targets,
            dataset.aircraft_type_ids.detach().cpu().numpy())
        metrics.insert(0, 'partition', partition)
        metrics.insert(0, 'seed', seed)
        metric_frames.append(metrics)

        accounting = mass_accounting(predictions, dataset)
        accounting.insert(0, 'partition', partition)
        accounting.to_csv(seed_dir / f'{partition}_mass_accounting.csv', index=False)
        ratios = accounting['accounting_ratio'].replace([np.inf, -np.inf], np.nan).dropna()

        diagnostic_rows.append({
            'seed': seed, 'partition': partition,
            'mass_accounting_ratio_mean': float(ratios.mean()) if len(ratios) else float('nan'),
            'mass_accounting_ratio_median': float(ratios.median()) if len(ratios) else float('nan'),
            **force_response_rates(model, dataset, diagnostic_loss, device,
                                   args.eval_batch_size,
                                   max_batches=3 if args.smoke else 0),
            **transition_gaps(model, dataset, device, args.eval_batch_size),
        })

    metric_frame = pd.concat(metric_frames, ignore_index=True)
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    metric_frame.to_csv(seed_dir / 'metrics.csv', index=False)
    diagnostic_frame.to_csv(seed_dir / 'diagnostics.csv', index=False)

    overall = metric_frame[metric_frame['aircraft_type'] == 'all']
    for partition, _ in partitions:
        log(f'\nseed {seed}   {partition}')
        for row in overall[overall['partition'] == partition].itertuples(index=False):
            log(f'  {row.target:<9} R2 {row.R2:.4f}   RMSE {row.RMSE:.6f}   '
                f'MAE {row.MAE:.6f}   MAPE {row.MAPE:.2f}%')
    return metric_frame, diagnostic_frame


def aggregate(run_dir: Path, metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    metrics.to_csv(run_dir / 'per_seed_metrics.csv', index=False)
    diagnostics.to_csv(run_dir / 'per_seed_diagnostics.csv', index=False)
    n_seeds = int(metrics['seed'].nunique())

    metric_mean = metrics.groupby(['partition', 'aircraft_type', 'target'],
                                  as_index=False)[['R2', 'RMSE', 'MAE', 'MAPE']].mean()
    metric_mean.to_csv(run_dir / f'mean_metrics_{n_seeds}seed.csv', index=False)

    # 只对数值列求均值：调用方可能附加了机型、条件等文本列
    numeric = [c for c in diagnostics.columns
               if c not in ('seed', 'partition')
               and pd.api.types.is_numeric_dtype(diagnostics[c])]
    diagnostics.groupby('partition', as_index=False)[numeric].mean().to_csv(
        run_dir / f'mean_diagnostics_{n_seeds}seed.csv', index=False)

    # 分区名由调用方决定（零样本是 source_validation/zero_shot_test，
    # fleet-wide 是 validation/test），这里按实际出现的分区逐个汇报
    for partition in metric_mean['partition'].unique():
        block = metric_mean[metric_mean['partition'] == partition]
        log(f'\n{n_seeds}-seed mean   {partition}')
        for row in block[block['aircraft_type'] == 'all'].itertuples(index=False):
            log(f'  {row.target:<9} R2 {row.R2:.4f}   RMSE {row.RMSE:.6f}   '
                f'MAE {row.MAE:.6f}   MAPE {row.MAPE:.2f}%')
        by_type = block[(block['target'] == 'FF_Total')
                        & (block['aircraft_type'] != 'all')]
        for row in by_type.itertuples(index=False):
            log(f'    {row.aircraft_type:<10} FF_Total   R2 {row.R2:.4f}   '
                f'RMSE {row.RMSE:.6f}   MAPE {row.MAPE:.2f}%')
