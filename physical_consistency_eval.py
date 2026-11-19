# -*- coding: utf-8 -*-
"""留出测试航班上的物理一致性评估（Supplementary Note 2 “Physical-consistency evaluation”）

对 fleet-wide 全数据实验的三组模型（完整模型、去物理损失模型、MLP，各 5 个种子）计算：

1. 质量闭合  900 s 窗口内预测总油耗积分与 GWC 下降量的相对差 |∫FF dt − ΔGWC| / ΔGWC，
             窗口的构造与训练时完全相同（build_mass_windows）。记录油耗按同一方法给出数据本身
             的闭合误差，并在训练航班上复算，用于核对 SI 中的 3.9%。
2. 力方向    门控 𝒢L 内 s_γ < 0 的记录比例，𝒢L ∩ 𝒢m 内 s_m < 0 的记录比例。方向导数与门控
             直接取自训练所用的 LawBasedLoss，s_γ 沿 FPA–IVV 几何一致的方向求导。
3. 相位连续  测试航班全部真实相位过渡点上，同一状态经前后两个相位子网络的相对跳变
             |FF(p−) − FF(p+)| / FF(p−)，两台发动机分别计。MLP 没有相位输入，不参与。

只做推理，不训练任何模型。

    python physical_consistency_eval.py            # 全部 15 个模型
    python physical_consistency_eval.py --smoke    # 每组只跑一个种子，快速检查
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from af_data import (AIRCRAFT_FEATURE_TABLE, FEATURE_INDEX, STUDY_AIRCRAFT_TYPES,
                     FlightRecordDataset, build_mass_windows, build_transition_pairs,
                     load_flight_table, split_fleet_wide)
from af_losses import LawBasedLoss
from af_train import restore_model
from train_baselines import MLPFuelFlow, standardize

RESULTS = Path('results')
RUNS = {
    'full': RESULTS / 'fleet_wide' / 'all_flights_full',
    'no_physics': RESULTS / 'fleet_wide' / 'all_flights_no_physics',
}
MLP_DIR = RESULTS / 'baselines' / 'fleet_wide' / 'models'
SEEDS = (42, 43, 44, 45, 46)
OUT_DIR = RESULTS / 'physical_consistency'
WINDOW_SECONDS = 900
BATCH = 8192


# ── 模型包装：统一成 f(features, phases, aircraft_ids) → (N, 2) ──────────

class MLPWrapper(torch.nn.Module):
    """把 MLP 包成与 PINN 相同的调用方式，标准化放进前向，便于对原始输入求导。"""

    def __init__(self, model: MLPFuelFlow, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.model = model
        table = torch.tensor([AIRCRAFT_FEATURE_TABLE[n] for n in STUDY_AIRCRAFT_TYPES],
                             dtype=torch.float32)
        self.register_buffer('table', table)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)

    def forward(self, features, phases, aircraft_ids):
        x = torch.cat([features, self.table[aircraft_ids.long()]], dim=1)
        return self.model((x - self.mean) / self.scale)


def load_mlp(seed: int, mean: torch.Tensor, scale: torch.Tensor,
             device: torch.device) -> MLPWrapper:
    checkpoint = torch.load(MLP_DIR / f'mlp_seed{seed}.pt', map_location=device,
                            weights_only=False)
    model = MLPFuelFlow(int(checkpoint['input_dim']), int(checkpoint['hidden']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return MLPWrapper(model, mean, scale).to(device).eval()


# ── 三项指标 ────────────────────────────────────────────────────

def predict_total(model, dataset: FlightRecordDataset) -> np.ndarray:
    out = []
    with torch.no_grad():
        for features, _, phases, aircraft in dataset.iter_batches(BATCH):
            out.append(model(features, phases, aircraft).sum(dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def window_closure(total_flow: np.ndarray, dataset: FlightRecordDataset,
                   windows: List[np.ndarray]) -> np.ndarray:
    """每个窗口的相对差 (∫FF dt − ΔGWC) / ΔGWC，梯形积分，与训练损失一致。"""
    times = dataset.time.cpu().numpy().astype(np.float64)
    mass = dataset.features[:, FEATURE_INDEX['GWC']].cpu().numpy().astype(np.float64)
    rel = np.empty(len(windows))
    for i, w in enumerate(windows):
        burn = float(np.sum(0.5 * (total_flow[w][1:] + total_flow[w][:-1]) * np.diff(times[w])))
        drop = mass[w[0]] - mass[w[-1]]
        rel[i] = (burn - drop) / drop
    return rel


def wrong_sign_rates(model, dataset: FlightRecordDataset,
                     loss_fn: LawBasedLoss) -> Dict[str, float]:
    gamma_bad = gamma_n = mass_bad = mass_n = 0
    for features, _, phases, aircraft in dataset.iter_batches(BATCH):
        x = features.detach().clone().requires_grad_(True)
        total = model(x, phases, aircraft).sum(dim=1)
        grad, = torch.autograd.grad(total.sum(), x)
        fpa = x[:, FEATURE_INDEX['FPA']]
        tas = x[:, FEATURE_INDEX['TAS']]
        # s_γ 与 s_m 只差正的比例因子（训练标准差 / 预测油耗），符号与原始导数一致
        s_gamma = grad[:, FEATURE_INDEX['FPA']] + tas * torch.cos(fpa) * grad[:, FEATURE_INDEX['IVV']]
        s_mass = grad[:, FEATURE_INDEX['GWC']]
        applies, holds = loss_fn.gates(x.detach(), aircraft)
        both = applies & holds
        gamma_bad += int(((s_gamma < 0) & applies).sum())
        gamma_n += int(applies.sum())
        mass_bad += int(((s_mass < 0) & both).sum())
        mass_n += int(both.sum())
    return {'wrong_sign_gamma_pct': 100.0 * gamma_bad / gamma_n,
            'wrong_sign_mass_pct': 100.0 * mass_bad / mass_n,
            'gate_L_records': gamma_n, 'gate_L_and_m_records': mass_n}


def phase_jumps(model, dataset: FlightRecordDataset) -> np.ndarray:
    """每个过渡点、每台发动机的相对跳变，形状 (n_transitions, 2)。"""
    rows, left, right = build_transition_pairs(dataset)
    device = dataset.features.device
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    with torch.no_grad():
        features = dataset.features[index]
        aircraft = dataset.aircraft_type_ids[index]
        a = model(features, torch.as_tensor(left, device=device), aircraft)
        b = model(features, torch.as_tensor(right, device=device), aircraft)
        return ((a - b).abs() / a.clamp_min(1e-6)).cpu().numpy()


# ── 主流程 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--smoke', action='store_true', help='每组只评估第一个种子')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_flight_table()
    train_frame, _, test_frame, manifest = split_fleet_wide(data, 0.7, 0.1, 0.2, 42)
    for run_dir in RUNS.values():
        saved = pd.read_csv(run_dir / 'fleet_wide_split_manifest.csv')
        if saved.to_csv(index=False) != manifest.to_csv(index=False):
            raise RuntimeError(f'{run_dir} 的划分与本次不一致')
    train = FlightRecordDataset(train_frame, 'train').to(device)
    test = FlightRecordDataset(test_frame, 'test').to(device)
    print(f'train {len(train):,} records / {train.n_flights()} flights   '
          f'test {len(test):,} records / {test.n_flights()} flights')

    # 记录数据本身的闭合误差：训练航班（SI 中 3.9% 的出处）与测试航班
    train_windows = build_mass_windows(train, WINDOW_SECONDS)
    test_windows = build_mass_windows(test, WINDOW_SECONDS)
    recorded = {}
    for name, ds, wins in (('train', train, train_windows), ('test', test, test_windows)):
        observed = ds.targets.sum(dim=1).cpu().numpy().astype(np.float64)
        rel = window_closure(observed, ds, wins)
        recorded[name] = rel
        print(f'recorded closure  {name:<5} {len(wins):>5} windows   '
              f'median |rel| {100 * np.median(np.abs(rel)):.2f}%   '
              f'median signed {100 * np.median(rel):+.2f}%')
    pd.DataFrame({'partition': ['train'] * len(recorded['train']) + ['test'] * len(recorded['test']),
                  'relative_difference': np.concatenate([recorded['train'], recorded['test']])}
                 ).to_csv(OUT_DIR / 'recorded_closure_windows.csv', index=False)

    # MLP 的标准化统计量按训练时的方法在训练分区上重算
    table = torch.tensor([AIRCRAFT_FEATURE_TABLE[n] for n in STUDY_AIRCRAFT_TYPES],
                         dtype=torch.float32)
    train_inputs = torch.cat([train.features.cpu(), table[train.aircraft_type_ids.cpu()]], dim=1)
    mlp_mean, mlp_scale = standardize(train_inputs)

    per_seed, window_rows, jump_rows = [], [], []
    for config in ('full', 'no_physics', 'mlp'):
        for seed in seeds:
            if config == 'mlp':
                model = load_mlp(seed, mlp_mean.to(device), mlp_scale.to(device), device)
                x_scale = train.input_stats()[1]
            else:
                model, metadata = restore_model(RUNS[config] / f'seed_{seed}' / 'best_model.pt',
                                                manifest, 42, device)
                x_scale = torch.tensor(metadata['input_scale'], dtype=torch.float32)
            loss_fn = LawBasedLoss(x_scale=x_scale, w_mass=0, w_force=0, w_phase=0).to(device)

            closure = window_closure(predict_total(model, test), test, test_windows)
            row = {'config': config, 'seed': seed, 'n_windows': len(test_windows),
                   'closure_median_abs_pct': 100 * float(np.median(np.abs(closure))),
                   'closure_median_signed_pct': 100 * float(np.median(closure))}
            row.update(wrong_sign_rates(model, test, loss_fn))
            if config != 'mlp':
                jumps = phase_jumps(model, test)
                row.update({'n_transitions': len(jumps),
                            'jump_median_pct': 100 * float(np.median(jumps)),
                            'jump_mean_pct': 100 * float(np.mean(jumps))})
                jump_rows.append(pd.DataFrame({'config': config, 'seed': seed,
                                               'ff1': jumps[:, 0], 'ff2': jumps[:, 1]}))
            window_rows.append(pd.DataFrame({'config': config, 'seed': seed,
                                             'relative_difference': closure}))
            per_seed.append(row)
            print(f"{config:<10} seed {seed}   closure {row['closure_median_abs_pct']:.2f}%   "
                  f"wrong γ {row['wrong_sign_gamma_pct']:.3f}%   "
                  f"wrong m {row['wrong_sign_mass_pct']:.3f}%   "
                  + (f"jump {row['jump_median_pct']:.3f}%" if config != 'mlp' else ''))

    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(OUT_DIR / 'per_seed.csv', index=False)
    pd.concat(window_rows).to_csv(OUT_DIR / 'test_closure_windows.csv', index=False)
    pd.concat(jump_rows).to_csv(OUT_DIR / 'test_phase_jumps.csv', index=False)

    numeric = [c for c in per_seed.columns if c not in ('config', 'seed')]
    summary = per_seed.groupby('config', sort=False)[numeric].agg(['mean', 'std'])
    summary.to_csv(OUT_DIR / 'summary.csv')
    print('\nmean over seeds')
    print(per_seed.groupby('config', sort=False)[
        ['closure_median_abs_pct', 'wrong_sign_gamma_pct', 'wrong_sign_mass_pct',
         'jump_median_pct']].mean().round(3).to_string())
    print(f'\noutput {OUT_DIR}')


if __name__ == '__main__':
    main()
