# -*- coding: utf-8 -*-
"""迁移学习实验：A32x → B737-800 / B737-700

设计见 修改说明/52_迁移学习实验设计.md。三个阶段依次运行：

    python train_transfer.py split        生成按机号划分的清单
    python train_transfer.py run          正式实验（可用 --shard i --n_shards n 分进程）
    python train_transfer.py summarize    汇总、配对比较

实验组分两条流程，两条流程之间只差有无物理约束，其余（航班、种子、冻结范围、
训练设置）完全相同：

  物理约束流程：源模型为 fleet-wide 完整模型，微调时带三项物理损失
    S0          源模型直接预测，不更新
    S1          随机初始化，全部参数，只用目标航班训练
    L1          源模型初始化，只更新机型分支
    L2          L1 + 输出层
    L3          L2 + 融合层
    C           L1 + 编码器输入层（对照：检验机型差异是否在上层）

  无物理约束流程：源模型为消融实验的无物理约束模型（all_flights_no_physics），
  微调时三项物理权重也为零
    S0_ablation、S1_nophys、L1_ablation、L2_ablation、L3_ablation、C_ablation
    （S1_nophys 不用源模型，名称沿用已完成的结果）

训练设置（学习率、批大小、轮数、学习率衰减、早停、梯度裁剪、物理权重）与主实验
完全相同。与主实验训练流程（af_train.train_one_seed）的差别只有三处：
    1. 初始化：源模型或随机。
    2. 可训练参数：按组别冻结其余参数，训练后逐元素核对冻结参数未变。
    3. 输入标准化：所有组别使用源模型的飞行状态与机型特征标准化参数，
       使各组只在初始化和可训练参数上不同。

测试航班只用于最终评价，不参与早停或任何其他决定。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from af_data import (
    AIRCRAFT_FEATURE_TABLE, PHASE_LOOKUP, PHASE_NAMES,
    STUDY_AIRCRAFT_TYPES, TARGET_COLUMNS, FEATURE_NAMES,
    CyclicBatches, FlightRecordDataset, TrajectoryWindowSampler,
    build_mass_windows, build_transition_pairs,
    collate_mass_windows, collate_transitions,
)
from af_losses import LawBasedLoss
from af_model import FuelFlowNet
from af_train import (
    _trapezoid, evaluate_outputs, log, predict, regression_metrics,
    resolve_device, set_seed, validate_data_mse,
)
from train_zero_shot import build_parser as build_base_parser

# ── 源模型与目标数据 ────────────────────────────────────────────

# 两个源模型：完整模型与消融实验中的无物理约束模型，结构、数据、划分、种子都相同
SOURCE_DIRS = {
    'full': Path('results/fleet_wide/all_flights_full'),
    'no_physics': Path('results/fleet_wide/all_flights_no_physics'),
}
SOURCE_SEEDS = (42, 43, 44, 45, 46)
SOURCE_PHYSICS = {'w_mass': 0.10, 'w_force': 10.0, 'w_phase': 0.02}

OUT_DIR = Path('results/transfer')

TARGET_DATA = {
    'B737-800': Path('data/qar_b737-800.csv'),
    'B737-700': Path('data/qar_b737-700.csv'),
}
# 取自 OpenAP。巡航速度按主流程口径，由巡航马赫数在 11 000 m 标准大气下换算，
# 音速 295.07 m/s（A32x 为 M0.78 → 230 m/s）
TARGET_FEATURES: Dict[str, Tuple[float, float, float, float]] = {
    'B737-800': (124.6, 34.32, 79000.0, 0.789 * 295.07),
    'B737-700': (124.6, 34.32, 70000.0, 0.780 * 295.07),
}
ALL_TYPES = tuple(STUDY_AIRCRAFT_TYPES) + tuple(TARGET_FEATURES)
TYPE_TO_ID = {name: i for i, name in enumerate(ALL_TYPES)}

# ── 划分与抽样 ──────────────────────────────────────────────────

SPLIT_SEED = 42
TEST_TAILS = {'B737-800': 6, 'B737-700': 3}
VALIDATION_TAILS = 1
K_VALUES = {'B737-800': (1, 2, 4, 8, 16, 0), 'B737-700': (1, 2, 4, 0)}   # 0 表示微调池全部
N_DRAWS = 10


# ── 实验组 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Arm:
    name: str
    init: str                      # 'source' 或 'scratch'
    trainable: Tuple[str, ...]     # 可训练参数名前缀；('*',) 表示全部，() 表示不训练
    physics: bool
    n_params: int                  # 期望的可训练参数量，构造后核对
    source: str = 'full'           # 源模型，SOURCE_DIRS 的键


def _arm(name, init, trainable, n_params, physics=True, source='full'):
    return Arm(name, init, tuple(trainable), physics, n_params, source)


_LEVELS = {
    'L1': (('aircraft_encoder.',), 80),
    'L2': (('aircraft_encoder.', 'output_layer.'), 210),
    'L3': (('aircraft_encoder.', 'output_layer.', 'fusion.'), 5394),
    'C': (('aircraft_encoder.', 'shared_encoder.0.'), 592),
}

ARMS: Dict[str, Arm] = {a.name: a for a in (
    # 物理约束流程
    _arm('S0', 'source', (), 0),
    _arm('S1', 'scratch', ('*',), 64338),
    *[_arm(name, 'source', trainable, n) for name, (trainable, n) in _LEVELS.items()],
    # 无物理约束流程
    _arm('S0_ablation', 'source', (), 0, physics=False, source='no_physics'),
    _arm('S1_nophys', 'scratch', ('*',), 64338, physics=False),
    *[_arm(f'{name}_ablation', 'source', trainable, n, physics=False, source='no_physics')
      for name, (trainable, n) in _LEVELS.items()],
)}
# 两条流程中一一对应的组：(物理约束流程, 无物理约束流程)
PIPELINE_PAIRS = [('S0', 'S0_ablation'), ('S1', 'S1_nophys')] + [
    (name, f'{name}_ablation') for name in _LEVELS]


# ── 数据 ────────────────────────────────────────────────────────

def feature_table() -> torch.Tensor:
    """四个源机型加两个目标机型的静态特征表，行序同 ALL_TYPES。"""
    rows = [AIRCRAFT_FEATURE_TABLE[name] for name in STUDY_AIRCRAFT_TYPES]
    rows += [TARGET_FEATURES[name] for name in TARGET_FEATURES]
    return torch.tensor(rows, dtype=torch.float32)


def load_target(target: str) -> pd.DataFrame:
    data = pd.read_csv(TARGET_DATA[target]).reset_index(drop=True)
    required = {'time', 'flight_id', 'aircraft_type', 'phase_name',
                *TARGET_COLUMNS, *FEATURE_NAMES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'{TARGET_DATA[target]} 缺少列 {missing}')
    if set(data['aircraft_type'].astype(str)) != {target}:
        raise ValueError(f'{TARGET_DATA[target]} 的机型不全是 {target}')
    unknown = sorted(set(data['phase_name'].astype(str)) - set(PHASE_LOOKUP))
    if unknown:
        raise ValueError(f'phase_name 出现未定义取值 {unknown}')
    data['registration'] = data['flight_id'].astype(str).str.extract(r'(B-[0-9A-Z]{4})')[0]
    if data['registration'].isna().any():
        raise ValueError('有航班无法从 flight_id 解析出机号')
    return data


def make_split(target: str, data: pd.DataFrame) -> pd.DataFrame:
    """按机号整架划分：测试、验证、微调池。"""
    flights = (data.groupby('flight_id', as_index=False)
               .agg(registration=('registration', 'first'), n_records=('time', 'size')))
    tails = np.array(sorted(flights['registration'].unique()), dtype=object)
    rng = np.random.default_rng([SPLIT_SEED, TYPE_TO_ID[target]])
    tails = tails[rng.permutation(len(tails))]
    n_test = TEST_TAILS[target]
    test = set(tails[:n_test])
    validation = set(tails[n_test:n_test + VALIDATION_TAILS])
    flights['split'] = np.where(flights['registration'].isin(test), 'test',
                                np.where(flights['registration'].isin(validation),
                                         'validation', 'pool'))
    flights.insert(0, 'aircraft_type', target)
    flights = flights.sort_values(['split', 'registration', 'flight_id']).reset_index(drop=True)

    for name in ('test', 'validation', 'pool'):
        if not (flights['split'] == name).any():
            raise RuntimeError(f'{target} 的 {name} 分区为空')
    if flights.groupby('registration')['split'].nunique().max() > 1:
        raise RuntimeError('同一架飞机出现在多个分区')
    return flights


def split_path(target: str) -> Path:
    return OUT_DIR / target / 'split_manifest.csv'


def load_split(target: str) -> pd.DataFrame:
    path = split_path(target)
    if not path.exists():
        raise FileNotFoundError(f'未找到 {path}，请先运行 split 阶段')
    return pd.read_csv(path)


def manifest_sha(manifest: pd.DataFrame) -> str:
    text = manifest.to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def pool_order(target: str, manifest: pd.DataFrame, draw: int) -> List[str]:
    """第 draw 组抽样的微调池排列。取前 k 个，各 k 之间嵌套。"""
    pool = sorted(manifest.loc[manifest['split'] == 'pool', 'flight_id'].astype(str))
    rng = np.random.default_rng([SPLIT_SEED, TYPE_TO_ID[target], 1000 + draw])
    return [pool[i] for i in rng.permutation(len(pool))]


def chosen_flights(target: str, manifest: pd.DataFrame, k: int, draw: int) -> List[str]:
    order = pool_order(target, manifest, draw)
    if k > len(order):
        raise ValueError(f'{target} 微调池只有 {len(order)} 个航班，取不出 {k} 个')
    return order if k == 0 else order[:k]


def dataset_for(data: pd.DataFrame, flights: Sequence[str], name: str,
                device: torch.device) -> FlightRecordDataset:
    frame = data[data['flight_id'].astype(str).isin(set(flights))].reset_index(drop=True)
    if frame['flight_id'].nunique() != len(set(flights)):
        raise RuntimeError(f'{name} 中有航班在数据里找不到')
    return FlightRecordDataset(frame, name, type_to_id=TYPE_TO_ID).to(device)


# ── 模型 ────────────────────────────────────────────────────────

def load_source(seed: int, base, source: str = 'full') -> Dict[str, torch.Tensor]:
    """读取源模型，核对其训练机型、结构与物理权重。

    source='full' 为 fleet-wide 完整模型，三项物理权重须为主实验设定；
    source='no_physics' 为消融实验的无物理约束模型，三项物理权重须全为零。
    """
    path = SOURCE_DIRS[source] / f'seed_{seed}' / 'best_model.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    meta = checkpoint['metadata']
    if meta.get('train_aircraft_types') != list(STUDY_AIRCRAFT_TYPES):
        raise RuntimeError(f'{path} 的训练机型不是四个 A32x')
    config = dict(meta.get('model_config') or {})
    config.setdefault('phase_mode', 'per_phase')
    expected = {'input_dim': len(FEATURE_NAMES), 'hidden_dim': base.hidden_dim,
                'num_layers': base.num_layers,
                'aircraft_feature_dim': base.aircraft_feature_dim,
                'use_aircraft_features': True, 'phase_mode': 'per_phase'}
    if config != expected:
        raise RuntimeError(f'{path} 的结构 {config} 与本实验 {expected} 不一致')
    training = meta.get('training_config', {})
    for key, value in SOURCE_PHYSICS.items():
        expected_weight = value if source == 'full' else 0.0
        if abs(float(training.get(key, math.nan)) - expected_weight) > 1e-12:
            raise RuntimeError(f'{path} 的 {key}={training.get(key)}，'
                               f'与源模型类型 {source} 不符')
    state = checkpoint['model_state_dict']
    source_table = state['aircraft_feature_table']
    if not torch.equal(source_table, feature_table()[:len(STUDY_AIRCRAFT_TYPES)]):
        raise RuntimeError(f'{path} 的机型特征表与当前代码不一致')
    return state


def build_model(arm: Arm, source_state: Dict[str, torch.Tensor], base, seed: int,
                device: torch.device) -> Tuple[FuelFlowNet, List[str]]:
    set_seed(seed)
    model = FuelFlowNet(
        aircraft_feature_table=feature_table(),
        aircraft_feature_mean=source_state['aircraft_feature_mean'],
        aircraft_feature_scale=source_state['aircraft_feature_scale'],
        hidden_dim=base.hidden_dim, num_layers=base.num_layers,
        aircraft_feature_dim=base.aircraft_feature_dim, phase_mode='per_phase')
    if arm.init == 'source':
        state = {k: v for k, v in source_state.items() if k != 'aircraft_feature_table'}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or set(missing) != {'aircraft_feature_table'}:
            raise RuntimeError(f'载入源模型异常：missing={missing} unexpected={unexpected}')
    else:
        model.set_input_normalization(source_state['x_mean'], source_state['x_scale'])
    for key in ('x_mean', 'x_scale', 'aircraft_feature_mean', 'aircraft_feature_scale'):
        if not torch.equal(getattr(model, key), source_state[key]):
            raise RuntimeError(f'{key} 与源模型不一致')

    trainable = []
    for name, parameter in model.named_parameters():
        on = arm.trainable == ('*',) or any(name.startswith(p) for p in arm.trainable)
        parameter.requires_grad_(on)
        if on:
            trainable.append(name)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != arm.n_params:
        raise RuntimeError(f'{arm.name} 可训练参数 {n_trainable}，应为 {arm.n_params}')
    return model.to(device), trainable


# ── 训练 ────────────────────────────────────────────────────────

def physics_weights(arm: Arm) -> Dict[str, float]:
    return dict(SOURCE_PHYSICS) if arm.physics else {k: 0.0 for k in SOURCE_PHYSICS}


def fit(model: FuelFlowNet, arm: Arm, train_ds: FlightRecordDataset,
        val_ds: FlightRecordDataset, base, seed: int, device: torch.device,
        smoke: bool = False) -> Dict[str, object]:
    """训练循环与选模规则与 af_train.train_one_seed 相同，只更新可训练参数。"""
    weights = physics_weights(arm)
    loss_fn = LawBasedLoss(
        x_scale=model.x_scale.detach().cpu(), **weights,
        mass_huber_delta=base.mass_huber_delta,
        oswald_efficiency=base.oswald_efficiency,
        max_lift_coefficient=base.max_lift_coefficient,
        aircraft_feature_table=feature_table(),
    ).to(device)

    set_seed(seed)
    sampler = TrajectoryWindowSampler(train_ds, batch_size=base.batch_size,
                                      window_size=base.trajectory_window, seed=seed)
    window_seconds = base.smoke_window_seconds if smoke else base.window_seconds
    mass_windows = build_mass_windows(train_ds, window_seconds=window_seconds)
    rows, left, right = build_transition_pairs(train_ds)
    transitions = list(zip(rows.tolist(), left.tolist(), right.tolist()))
    if weights['w_mass'] > 0 and not mass_windows:
        raise RuntimeError('启用质量守恒，但训练航班切不出合格窗口')
    if weights['w_phase'] > 0 and not transitions:
        raise RuntimeError('启用相位连续，但训练航班没有真实过渡点')
    mass_batches = CyclicBatches(mass_windows, base.mass_windows_per_step, seed + 1000)
    transition_batches = CyclicBatches(transitions, base.transitions_per_step, seed + 2000)

    parameters = [p for p in model.parameters() if p.requires_grad]
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(parameters, lr=base.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=base.scheduler_factor, patience=base.scheduler_patience)

    best_val, best_state, best_epoch, stalled = math.inf, None, 0, 0
    history = []
    max_epochs = 2 if smoke else base.epochs
    max_batches = 3 if smoke else 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        sums = {'total': 0.0, 'data': 0.0, 'mass': 0.0, 'force': 0.0, 'phase': 0.0}
        steps = 0
        for index, row_index in enumerate(sampler):
            if max_batches and index >= max_batches:
                break
            features, targets, phases, aircraft = train_ds.batch(row_index)
            mass_batch = (collate_mass_windows(train_ds, mass_batches.next())
                          if weights['w_mass'] > 0 else None)
            transition_batch = (collate_transitions(train_ds, transition_batches.next())
                                if weights['w_phase'] > 0 else None)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model, features, targets, phases, aircraft,
                             mass_batch=mass_batch, transition_batch=transition_batch)
            if not bool(torch.isfinite(losses['total'])):
                raise FloatingPointError(f'epoch {epoch} 损失非有限')
            losses['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, base.gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f'epoch {epoch} 梯度非有限')
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach())
            steps += 1

        val_mse = validate_data_mse(model, val_ds, base.eval_batch_size)
        scheduler.step(val_mse)
        history.append({'epoch': epoch, **{f'train_{k}': v / max(steps, 1) for k, v in sums.items()},
                        'validation_data_mse': val_mse,
                        'learning_rate': float(optimizer.param_groups[0]['lr'])})
        if val_mse < best_val - base.min_delta:
            best_val, best_epoch, stalled = val_mse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stalled += 1
        if stalled >= base.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError('训练结束后没有得到可用的模型')
    model.load_state_dict(best_state)
    for name, parameter in model.named_parameters():
        if name in frozen and not torch.equal(parameter.detach(), frozen[name]):
            raise RuntimeError(f'冻结参数 {name} 在训练中被改动')

    return {'best_validation_data_mse': best_val, 'best_epoch': best_epoch,
            'epochs_run': len(history), 'train_seconds': time.perf_counter() - started,
            'n_mass_windows': len(mass_windows), 'n_transitions': len(transitions),
            'history': pd.DataFrame(history), 'state': best_state}


# ── 评价 ────────────────────────────────────────────────────────

def flight_burn(predictions: np.ndarray, targets: np.ndarray,
                dataset: FlightRecordDataset) -> pd.DataFrame:
    """逐航班积分油耗：预测与实测燃油流量分别在时间连续段上积分。"""
    times = dataset.time.detach().cpu().numpy().astype(np.float64)
    flights = dataset.flight_ids.detach().cpu().numpy()
    predicted, observed = predictions.sum(axis=1), targets.sum(axis=1)
    rows = []
    for flight_id in np.unique(flights):
        index = np.where(flights == flight_id)[0]
        index = index[np.argsort(times[index])]
        breaks = np.where((np.diff(times[index]) <= 0) | (np.diff(times[index]) > 2.0))[0]
        burn_pred = burn_obs = 0.0
        for segment in np.split(index, breaks + 1):
            if len(segment) < 2:
                continue
            burn_pred += float(_trapezoid(predicted[segment], times[segment]))
            burn_obs += float(_trapezoid(observed[segment], times[segment]))
        rows.append({'flight_id': str(dataset.flight_keys[flight_id]),
                     'observed_burn_kg': burn_obs, 'predicted_burn_kg': burn_pred,
                     'burn_error_pct': 100.0 * (burn_pred - burn_obs) / burn_obs
                     if burn_obs > 0 else math.nan})
    return pd.DataFrame(rows)


def evaluate(model: FuelFlowNet, test_ds: FlightRecordDataset, base,
             device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, targets = predict(model, test_ds, device, base.eval_batch_size)
    overall = evaluate_outputs(predictions, targets)
    phases = test_ds.phases.detach().cpu().numpy()
    phase_rows = []
    for phase_index in np.unique(phases):
        mask = phases == phase_index
        row = {'phase': PHASE_NAMES[int(phase_index)], 'n_records': int(mask.sum())}
        row.update(regression_metrics(targets[mask].sum(axis=1), predictions[mask].sum(axis=1)))
        phase_rows.append(row)
    return overall, pd.DataFrame(phase_rows), flight_burn(predictions, targets, test_ds)


# ── 运行单元 ────────────────────────────────────────────────────

def base_args(smoke: bool):
    base = build_base_parser().parse_args([])
    base.smoke = smoke
    return base


def run_config(target, arm, k, draw, manifest, train_flights, val_flights, base, smoke):
    config = {
        'target': target, 'arm': arm.name, 'init': arm.init, 'trainable': list(arm.trainable),
        'n_trainable': arm.n_params, 'physics': physics_weights(arm), 'k': k,
        'draw': draw, 'lr': base.lr if arm.trainable else None,
        'source_seed': SOURCE_SEEDS[draw % len(SOURCE_SEEDS)],
        'seed': 42 + draw, 'train_flights': list(train_flights),
        'validation_flights': list(val_flights), 'manifest_sha256': manifest_sha(manifest),
        'epochs': base.epochs, 'early_stopping_patience': base.early_stopping_patience,
        'smoke': smoke,
    }
    # 只有无物理约束源模型才记录此项，已完成的完整模型各组配置保持不变，可直接复用
    if arm.source != 'full':
        config['source_model'] = str(SOURCE_DIRS[arm.source])
    return config


def same_config(path: Path, config: Dict[str, object]) -> bool:
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding='utf-8')) == json.loads(
        json.dumps(config, ensure_ascii=False))


class TargetContext:
    """一个目标机型的数据、划分与数据集缓存。"""

    def __init__(self, target: str, device: torch.device):
        self.target = target
        self.device = device
        self.data = load_target(target)
        self.manifest = load_split(target)
        current = make_split(target, self.data)
        if manifest_sha(current) != manifest_sha(self.manifest):
            raise RuntimeError(f'{split_path(target)} 与按当前数据重新划分的结果不一致')
        flights = lambda s: sorted(self.manifest.loc[self.manifest['split'] == s,
                                                     'flight_id'].astype(str))
        self.val_flights = flights('validation')
        self.test_flights = flights('test')
        self._val = None
        self._test = None

    @property
    def val_ds(self):
        if self._val is None:
            self._val = dataset_for(self.data, self.val_flights, 'target_validation', self.device)
        return self._val

    @property
    def test_ds(self):
        if self._test is None:
            self._test = dataset_for(self.data, self.test_flights, 'target_test', self.device)
        return self._test


def run_one(ctx: TargetContext, arm: Arm, k: int, draw: int, base, out_dir: Path,
            smoke: bool, save_model: bool) -> Dict[str, object]:
    train_flights = [] if not arm.trainable else chosen_flights(ctx.target, ctx.manifest, k, draw)
    leaked = set(train_flights) & (set(ctx.val_flights) | set(ctx.test_flights))
    if leaked:
        raise RuntimeError(f'训练航班与验证/测试航班重叠：{sorted(leaked)}')
    config = run_config(ctx.target, arm, k, draw, ctx.manifest, train_flights,
                        ctx.val_flights, base, smoke)
    result_path = out_dir / 'result.json'
    if same_config(out_dir / 'config.json', config) and result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    out_dir.mkdir(parents=True, exist_ok=True)

    source_state = load_source(config['source_seed'], base, arm.source)
    model, trainable = build_model(arm, source_state, base, config['seed'], ctx.device)
    result: Dict[str, object] = {'trainable_parameters': trainable}

    if arm.trainable:
        train_ds = dataset_for(ctx.data, train_flights, 'target_train', ctx.device)
        fitted = fit(model, arm, train_ds, ctx.val_ds, base, config['seed'], ctx.device, smoke)
        fitted['history'].to_csv(out_dir / 'history.csv', index=False)
        if save_model:
            torch.save({'model_state_dict': fitted['state'], 'config': config},
                       out_dir / 'best_model.pt')
        result.update({key: fitted[key] for key in (
            'best_validation_data_mse', 'best_epoch', 'epochs_run', 'train_seconds',
            'n_mass_windows', 'n_transitions')})
        result.update({'n_train_flights': len(train_flights), 'n_train_records': len(train_ds)})
    else:
        result.update({'n_train_flights': 0, 'n_train_records': 0})

    overall, phases, burns = evaluate(model, ctx.test_ds, base, ctx.device)
    overall.to_csv(out_dir / 'metrics.csv', index=False)
    phases.to_csv(out_dir / 'phase_metrics.csv', index=False)
    burns.to_csv(out_dir / 'flight_burn.csv', index=False)
    total = overall[overall['target'] == 'FF_Total'].iloc[0]
    result.update({'test_R2': float(total['R2']), 'test_RMSE': float(total['RMSE']),
                   'test_MAE': float(total['MAE']), 'test_MAPE': float(total['MAPE']),
                   'test_burn_abs_error_pct': float(burns['burn_error_pct'].abs().mean())})

    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


# ── 阶段 ────────────────────────────────────────────────────────

def stage_split(targets: Sequence[str], force: bool) -> None:
    for target in targets:
        manifest = make_split(target, load_target(target))
        n_pool = int((manifest['split'] == 'pool').sum())
        too_big = [k for k in K_VALUES[target] if k > n_pool]
        if too_big:
            raise RuntimeError(f'{target} 微调池 {n_pool} 个航班，放不下 k={too_big}')
        path = split_path(target)
        if path.exists() and not force:
            if manifest_sha(pd.read_csv(path)) != manifest_sha(manifest):
                raise RuntimeError(f'{path} 已存在且与本次划分不同；确认要覆盖请加 --force')
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        summary = manifest.groupby('split').agg(tails=('registration', 'nunique'),
                                                flights=('flight_id', 'size'))
        log(f'\n{target} 划分清单 {path}')
        log(summary.to_string())


def stage_run(targets, arms, base, device, smoke, shard, n_shards, save_model) -> None:
    units = []
    for target in targets:
        for arm_name in arms:
            if ARMS[arm_name].trainable:
                units += [(target, arm_name, k, d)
                          for k in K_VALUES[target] for d in range(N_DRAWS)]
            else:
                # S0 不训练，结果只取决于源模型：每个源模型种子各评价一次，不重复计数
                units += [(target, arm_name, 0, d) for d in range(len(SOURCE_SEEDS))]

    contexts = {t: TargetContext(t, device) for t in targets}
    for i, (target, arm_name, k, draw) in enumerate(units):
        if i % n_shards != shard:
            continue
        out_dir = OUT_DIR / target / 'runs' / arm_name / f'k{k:02d}_d{draw}'
        started = time.perf_counter()
        result = run_one(contexts[target], ARMS[arm_name], k, draw, base, out_dir,
                         smoke=smoke, save_model=save_model)
        log(f'[run {i + 1}/{len(units)}] {target} {arm_name:<10} k {k:>2} draw {draw}   '
            f'test RMSE {result["test_RMSE"]:.5f}  MAPE {result["test_MAPE"]:.2f}%   '
            f'{time.perf_counter() - started:.0f}s')


def stage_summarize(targets) -> None:
    from scipy import stats

    for target in targets:
        rows = []
        for path in sorted((OUT_DIR / target / 'runs').glob('*/k*_d*/result.json')):
            config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
            result = json.loads(path.read_text(encoding='utf-8'))
            rows.append({'arm': config['arm'], 'k': config['k'], 'draw': config['draw'],
                         'n_train_flights': result['n_train_flights'],
                         **{key: result[key] for key in result if key.startswith('test_')}})
        if not rows:
            log(f'{target} 没有结果')
            continue
        runs = pd.DataFrame(rows)
        # 只汇总当前设计中的组；已从设计中删除的组，其结果文件保留在磁盘上但不参与汇总
        runs = runs[runs['arm'].isin(ARMS)].reset_index(drop=True)
        runs.to_csv(OUT_DIR / target / 'all_runs.csv', index=False)

        # S0 不依赖 k：复制到每个 k，便于在同一张表里比较
        untrained = [name for name, arm in ARMS.items() if not arm.trainable]
        s0 = runs[runs['arm'].isin(untrained)]
        runs = pd.concat([runs[~runs['arm'].isin(untrained)]]
                         + [s0.assign(k=k) for k in K_VALUES[target]], ignore_index=True)
        # k = 0 表示全部微调池，换成实际航班数
        pool_size = int((load_split(target)['split'] == 'pool').sum())
        runs['n_flights'] = runs['k'].where(runs['k'] > 0, pool_size)

        metrics = ['test_RMSE', 'test_MAPE', 'test_R2', 'test_burn_abs_error_pct']
        grouped = runs.groupby(['arm', 'k'])
        summary = grouped[metrics].mean().add_suffix('_mean')
        n = grouped.size().rename('n_draws')
        t_value = pd.Series(stats.t.ppf(0.975, n - 1), index=n.index)
        half = grouped[metrics].sem().mul(t_value, axis=0).add_suffix('_ci95')
        summary = pd.concat([grouped['n_flights'].first(), n, summary, half],
                            axis=1).reset_index()
        summary.to_csv(OUT_DIR / target / 'summary.csv', index=False)

        # 配对比较：同一 k、同一组抽样
        #   迁移对比从头训练（两条流程各自比较）；输入端对比输出端；
        #   物理约束流程对比无物理约束流程（同一组别）
        comparisons = [(a, 'S1') for a in _LEVELS]
        comparisons += [(f'{a}_ablation', 'S1_nophys') for a in _LEVELS]
        comparisons += [('L2', 'C')]
        comparisons += PIPELINE_PAIRS
        paired = []
        wide = runs.pivot_table(index=['k', 'draw'], columns='arm', values='test_RMSE')
        for left, right in comparisons:
            if left not in wide or right not in wide:
                continue
            for k, block in wide[[left, right]].dropna().groupby(level='k'):
                diff = block[left] - block[right]
                p = (stats.wilcoxon(block[left], block[right]).pvalue
                     if len(diff) >= 5 and (diff != 0).any() else math.nan)
                paired.append({'left': left, 'right': right, 'k': k, 'n_pairs': len(diff),
                               'mean_rmse_left': block[left].mean(),
                               'mean_rmse_right': block[right].mean(),
                               'mean_diff': diff.mean(),
                               'relative_change_pct': 100 * diff.mean() / block[right].mean(),
                               'wilcoxon_p': p})
        pd.DataFrame(paired).to_csv(OUT_DIR / target / 'paired_comparisons.csv', index=False)

        # 等效从头训练航班数：迁移组用 k 个航班达到的 RMSE 均值，同一流程的从头训练组
        # 最少需要多少个航班才能达到。达不到时留空
        scratch_of = {name: ('S1' if arm.source == 'full' and arm.physics else 'S1_nophys')
                      for name, arm in ARMS.items() if arm.init == 'source' and arm.trainable}
        equivalent = []
        for arm_name, scratch in scratch_of.items():
            ref = summary[summary['arm'] == scratch].sort_values('n_flights')
            for row in summary[summary['arm'] == arm_name].itertuples(index=False):
                reached = ref.loc[ref['test_RMSE_mean'] <= row.test_RMSE_mean, 'n_flights']
                equivalent.append({'arm': arm_name, 'scratch_arm': scratch,
                                   'n_flights': row.n_flights,
                                   'rmse_mean': row.test_RMSE_mean,
                                   'equivalent_scratch_flights':
                                       int(reached.min()) if len(reached) else None})
        pd.DataFrame(equivalent).astype({'equivalent_scratch_flights': 'Int64'}).to_csv(
            OUT_DIR / target / 'equivalent_scratch_flights.csv', index=False)

        log(f'\n{target}   测试集 FF_Total RMSE（{N_DRAWS} 组抽样均值 ± 95% CI）')
        table = summary.pivot(index='arm', columns='k', values='test_RMSE_mean')
        log(table.round(5).to_string())


# ── 入口 ────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='迁移学习实验：A32x → B737')
    parser.add_argument('stage', choices=['split', 'run', 'summarize'])
    parser.add_argument('--targets', nargs='+', default=list(TARGET_DATA),
                        choices=list(TARGET_DATA))
    parser.add_argument('--arms', nargs='+', default=None, choices=list(ARMS))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n_shards', type=int, default=1)
    parser.add_argument('--no_save_model', action='store_true')
    parser.add_argument('--force', action='store_true', help='split 阶段覆盖已有清单')
    parser.add_argument('--smoke', action='store_true',
                        help='2 轮、每轮 3 批，只检查流程；结果写到 results/transfer_smoke')
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.n_shards:
        raise ValueError('--shard 必须在 [0, n_shards) 内')

    global OUT_DIR
    if args.smoke:
        OUT_DIR = Path('results/transfer_smoke')
        if args.stage != 'split' and not split_path(args.targets[0]).exists():
            stage_split(args.targets, force=True)

    base = base_args(args.smoke)
    device = resolve_device(args.device)
    if args.stage == 'split':
        stage_split(args.targets, args.force)
    elif args.stage == 'run':
        stage_run(args.targets, args.arms or list(ARMS), base, device, args.smoke,
                  args.shard, args.n_shards, not args.no_save_model)
    else:
        stage_summarize(args.targets)


if __name__ == '__main__':
    main()
