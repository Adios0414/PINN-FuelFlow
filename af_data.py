# -*- coding: utf-8 -*-
"""数据、机型静态特征与三路取数

本模块不依赖项目中的其他模型文件，四机型词表、相位索引、输入顺序均在此定义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, Dataset

# ── 输入与标签 ──────────────────────────────────────────────────
FEATURE_NAMES = ['ALT_QNH', 'MACH', 'FPA', 'TAS', 'SAT', 'IVV', 'GWC']
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
TARGET_COLUMNS = ['FF1C', 'FF2C']

# ── 相位 ────────────────────────────────────────────────────────
PHASE_NAMES = ['takeoff', 'climb', 'cruise', 'descent', 'approach', 'landing']
PHASE_TO_INDEX = {name: i for i, name in enumerate(PHASE_NAMES)}
DROP_PHASE = -1
PHASE_LOOKUP = dict(PHASE_TO_INDEX)
PHASE_LOOKUP['taxi'] = DROP_PHASE

# ── 机型 ────────────────────────────────────────────────────────
STUDY_AIRCRAFT_TYPES = ('A320', 'A320-NEO', 'A321', 'A321-NEO')
TRAIN_AIRCRAFT_TYPES = ('A320', 'A320-NEO', 'A321')
WITHHELD_AIRCRAFT_TYPE = 'A321-NEO'
AIRCRAFT_TYPE_TO_ID = {name: i for i, name in enumerate(STUDY_AIRCRAFT_TYPES)}

# 机型静态特征。翼面积、翼展、最大起飞重量取自 OpenAP（Sun, Hoekstra &
# Ellerbroek, Aerospace 7(8):104, 2020）逐机型数据；其中最大起飞重量与空客
# Aircraft Characteristics 文件各机型最高重量变型一致。巡航速度由 OpenAP 的
# M0.78 @ 11 000 m 按标准大气换算为 230 m/s。
AIRCRAFT_FEATURE_NAMES = ['wing_area', 'wing_span', 'max_weight', 'cruise_speed']
AIRCRAFT_FEATURE_TABLE: Dict[str, Tuple[float, float, float, float]] = {
    'A320':     (124.0, 35.8, 78000.0, 230.0),
    'A320-NEO': (124.0, 35.8, 79000.0, 230.0),
    'A321':     (128.0, 35.8, 93500.0, 230.0),
    'A321-NEO': (128.0, 35.8, 97000.0, 230.0),
}

DATA_CSV = Path('data/qar_a32x.csv')


def aircraft_feature_matrix() -> torch.Tensor:
    """按机型 ID 排序的 4x4 静态特征表，不参与梯度更新。"""
    return torch.tensor(
        [AIRCRAFT_FEATURE_TABLE[name] for name in STUDY_AIRCRAFT_TYPES],
        dtype=torch.float32,
    )


def aircraft_feature_stats(
    train_types: Sequence[str] = TRAIN_AIRCRAFT_TYPES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """机型特征的标准化参数，每个训练机型只计一次，与记录条数无关。"""
    if not train_types:
        raise ValueError('至少需要一个训练机型')
    rows = torch.tensor(
        [AIRCRAFT_FEATURE_TABLE[name] for name in train_types],
        dtype=torch.float32,
    )
    mean = rows.mean(dim=0)
    scale = rows.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    return mean, scale


def aircraft_feature_provenance() -> Dict[str, Dict[str, float]]:
    return {
        name: dict(zip(AIRCRAFT_FEATURE_NAMES, AIRCRAFT_FEATURE_TABLE[name]))
        for name in STUDY_AIRCRAFT_TYPES
    }


# ── 读表与校验 ──────────────────────────────────────────────────

def load_flight_table(data_csv: Path = DATA_CSV) -> pd.DataFrame:
    if not Path(data_csv).exists():
        raise FileNotFoundError(f'未找到数据文件 {data_csv}')
    data = pd.read_csv(data_csv).reset_index(drop=True)

    required = {'time', 'flight_id', 'aircraft_type', 'phase_name',
                *TARGET_COLUMNS, *FEATURE_NAMES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'数据缺少必要列 {missing}')
    if data['flight_id'].isna().any() or data['aircraft_type'].isna().any():
        raise ValueError('flight_id 或 aircraft_type 存在缺失')

    observed = set(data['aircraft_type'].astype(str).unique())
    if observed != set(STUDY_AIRCRAFT_TYPES):
        raise ValueError(
            f'数据机型集合必须为 {list(STUDY_AIRCRAFT_TYPES)}，实际为 {sorted(observed)}')

    unknown_phase = sorted(set(data['phase_name'].astype(str)) - set(PHASE_LOOKUP))
    if unknown_phase:
        raise ValueError(f'phase_name 出现未定义取值 {unknown_phase}')
    return data


class FlightRecordDataset(Dataset):
    """一个数据分区的记录集合。地面滑行在构造时剔除。"""

    def __init__(self, frame: pd.DataFrame, name: str = '',
                 type_to_id: Optional[Dict[str, int]] = None):
        """type_to_id 不给时用四机型词表；迁移实验会传入含目标机型的扩展词表。"""
        type_to_id = AIRCRAFT_TYPE_TO_ID if type_to_id is None else type_to_id
        if len(frame) == 0:
            raise ValueError(f'数据分区 {name} 为空')
        frame = frame.reset_index(drop=True)
        self.name = name
        self.feature_names = list(FEATURE_NAMES)

        phases = np.array(
            [PHASE_LOOKUP[str(p)] for p in frame['phase_name']], dtype=np.int64)
        keep = phases >= 0
        if not keep.any():
            raise ValueError(f'数据分区 {name} 剔除滑行后为空')
        self.n_dropped = int((~keep).sum())

        frame = frame.loc[keep].reset_index(drop=True)
        self.phases = torch.tensor(phases[keep], dtype=torch.long)
        self.features = torch.tensor(
            frame[FEATURE_NAMES].to_numpy(dtype=np.float64), dtype=torch.float32)
        self.targets = torch.tensor(
            frame[TARGET_COLUMNS].to_numpy(dtype=np.float64), dtype=torch.float32)
        self.time = torch.tensor(
            frame['time'].to_numpy(dtype=np.float64), dtype=torch.float32)
        self.aircraft_type_ids = torch.tensor(
            [type_to_id[str(t)] for t in frame['aircraft_type']],
            dtype=torch.long)
        codes, self.flight_keys = pd.factorize(frame['flight_id'].astype(str))
        self.flight_ids = torch.tensor(codes, dtype=torch.long)

        if not torch.isfinite(self.features).all():
            raise ValueError(f'数据分区 {name} 的输入含非有限值')
        if not torch.isfinite(self.targets).all():
            raise ValueError(f'数据分区 {name} 的标签含非有限值')

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return (self.features[index], self.targets[index],
                self.phases[index], self.aircraft_type_ids[index])

    def to(self, device: torch.device) -> 'FlightRecordDataset':
        """把参与前向的张量常驻目标设备，取批时不再逐批拷贝。"""
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.phases = self.phases.to(device)
        self.aircraft_type_ids = self.aircraft_type_ids.to(device)
        self.time = self.time.to(device)
        self.device = device
        return self

    def batch(self, index) -> Tuple[torch.Tensor, ...]:
        if not torch.is_tensor(index):
            index = torch.as_tensor(index, dtype=torch.long)
        index = index.to(self.features.device)
        return (self.features[index], self.targets[index],
                self.phases[index], self.aircraft_type_ids[index])

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            yield (self.features[start:stop], self.targets[start:stop],
                   self.phases[start:stop], self.aircraft_type_ids[start:stop])

    def n_flights(self) -> int:
        return len(self.flight_keys)

    def aircraft_types(self) -> List[str]:
        ids = torch.unique(self.aircraft_type_ids).tolist()
        return [STUDY_AIRCRAFT_TYPES[i] for i in ids]

    def input_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.features.mean(dim=0)
        scale = self.features.std(dim=0, unbiased=False)
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        return mean, scale


# ── 按完整航班划分 ──────────────────────────────────────────────

def _allocate_two_way(n_flights: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    if n_flights < 2:
        raise ValueError('每个训练机型至少需要两个完整航班')
    ratios = np.array([train_ratio, val_ratio], dtype=float)
    if (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError('train_ratio 与 val_ratio 必须为正且和为 1.0')

    raw = ratios * n_flights
    counts = np.floor(raw).astype(int)
    counts[counts == 0] = 1
    while counts.sum() > n_flights:
        donors = np.where(counts > 1)[0]
        if len(donors) == 0:
            raise RuntimeError('无法形成非空训练集与验证集')
        counts[donors[np.argmax(counts[donors] - raw[donors])]] -= 1
    while counts.sum() < n_flights:
        counts[int(np.argmax(raw - counts))] += 1
    return int(counts[0]), int(counts[1])


def _allocate_three_way(n_flights: int, ratios: np.ndarray) -> Tuple[int, int, int]:
    """把 n_flights 个航班按比例分到三份，每份非空。"""
    active = ratios > 0
    if n_flights < int(active.sum()):
        raise ValueError(f'航班数 {n_flights} 不足以分出 {int(active.sum())} 个非空分区')

    raw = ratios * n_flights
    counts = np.floor(raw).astype(int)
    counts[active & (counts == 0)] = 1
    while counts.sum() > n_flights:
        donors = np.where(counts > active.astype(int))[0]
        if len(donors) == 0:
            raise RuntimeError('无法在保证各分区非空的前提下分配航班')
        counts[donors[np.argmax(counts[donors] - raw[donors])]] -= 1
    while counts.sum() < n_flights:
        counts[int(np.argmax(raw - counts))] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def split_fleet_wide(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """四个机型全部参与训练，按机型分层、以完整航班为单位切三份。"""
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError('split_fleet_wide 要求传入的表索引为 0 起连续编号')
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError('三个比例之和必须为 1.0')

    rng = np.random.default_rng(split_seed)
    masks = {name: np.zeros(len(data), dtype=bool)
             for name in ('train', 'validation', 'test')}
    rows: List[Dict[str, object]] = []

    for aircraft_type in STUDY_AIRCRAFT_TYPES:
        type_frame = data[data['aircraft_type'] == aircraft_type]
        flights = np.array(
            sorted(type_frame['flight_id'].astype(str).unique(), key=str), dtype=object)
        flights = flights[rng.permutation(len(flights))]
        n_train, n_val, _ = _allocate_three_way(len(flights), ratios)
        chosen = {'train': set(flights[:n_train]),
                  'validation': set(flights[n_train:n_train + n_val]),
                  'test': set(flights[n_train + n_val:])}

        type_ids = type_frame['flight_id'].astype(str)
        for name, flight_set in chosen.items():
            if flight_set:
                masks[name][type_frame.index[type_ids.isin(flight_set).to_numpy()]] = True
            for flight_id in sorted(flight_set, key=str):
                rows.append({'aircraft_type': aircraft_type, 'flight_id': flight_id,
                             'split': name, 'split_seed': split_seed,
                             'train_ratio': train_ratio, 'val_ratio': val_ratio,
                             'test_ratio': test_ratio})
        if verbose:
            print(f'  {aircraft_type:<9} train {len(chosen["train"]):>3}  '
                  f'validation {len(chosen["validation"]):>3}  '
                  f'test {len(chosen["test"]):>3} flights', flush=True)

    overlap = (masks['train'] & masks['validation']) | (masks['train'] & masks['test'])         | (masks['validation'] & masks['test'])
    if overlap.any():
        raise RuntimeError('同一条记录进入了多个分区')
    if not (masks['train'] | masks['validation'] | masks['test']).all():
        raise RuntimeError('存在未分配的记录')

    frames = tuple(data.loc[masks[name]].reset_index(drop=True)
                   for name in ('train', 'validation', 'test'))
    manifest = pd.DataFrame(rows)
    if manifest[['aircraft_type', 'flight_id']].astype(str).duplicated().any():
        raise RuntimeError('划分清单中同一航班出现多次')
    return frames + (manifest,)


def split_zero_shot(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    verbose: bool = True,
    withheld_aircraft_type: str = WITHHELD_AIRCRAFT_TYPE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """三个源机型按完整航班切训练与验证，第四个机型整体作零样本测试。"""
    if withheld_aircraft_type not in STUDY_AIRCRAFT_TYPES:
        raise ValueError(
            f'留出机型必须为 {list(STUDY_AIRCRAFT_TYPES)} 之一，'
            f'实际为 {withheld_aircraft_type!r}')
    # 下面用行号直接写 mask，要求 data 的索引是从 0 连续编号的
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError('split_zero_shot 要求传入的表索引为 0 起连续编号')

    train_aircraft_types = tuple(
        name for name in STUDY_AIRCRAFT_TYPES
        if name != withheld_aircraft_type
    )
    observed = set(data['aircraft_type'].astype(str).unique())
    if observed != set(STUDY_AIRCRAFT_TYPES):
        raise ValueError(
            f'数据机型集合必须为 {list(STUDY_AIRCRAFT_TYPES)}，'
            f'实际为 {sorted(observed)}')

    source = data[data['aircraft_type'].isin(train_aircraft_types)]
    held = data[data['aircraft_type'] == withheld_aircraft_type].reset_index(drop=True)
    rng = np.random.default_rng(split_seed)
    train_mask = np.zeros(len(data), dtype=bool)
    val_mask = np.zeros(len(data), dtype=bool)
    rows: List[Dict[str, object]] = []

    # 四个机型始终按同一顺序消耗随机数。这样某机型在不同留出轮次中作为
    # 源机型时使用相同的训练/验证航班；A321-NEO 默认轮次的旧划分不变。
    shuffled_flights: Dict[str, np.ndarray] = {}
    for aircraft_type in STUDY_AIRCRAFT_TYPES:
        type_frame = data[data['aircraft_type'] == aircraft_type]
        flights = np.array(
            sorted(type_frame['flight_id'].astype(str).unique(), key=str), dtype=object)
        shuffled_flights[aircraft_type] = flights[rng.permutation(len(flights))]

    for aircraft_type in train_aircraft_types:
        type_frame = source[source['aircraft_type'] == aircraft_type]
        flights = shuffled_flights[aircraft_type]
        n_train, n_val = _allocate_two_way(len(flights), train_ratio, val_ratio)
        chosen = {'train': set(flights[:n_train]),
                  'validation': set(flights[n_train:n_train + n_val])}

        type_ids = type_frame['flight_id'].astype(str)
        train_mask[type_frame.index[type_ids.isin(chosen['train']).to_numpy()]] = True
        val_mask[type_frame.index[type_ids.isin(chosen['validation']).to_numpy()]] = True

        for split_name, flight_set in chosen.items():
            for flight_id in sorted(flight_set, key=str):
                rows.append({'aircraft_type': aircraft_type, 'flight_id': flight_id,
                             'split': split_name, 'split_seed': split_seed,
                             'train_ratio': train_ratio, 'val_ratio': val_ratio})
        if verbose:
            print(f'  {aircraft_type:<9} train {n_train:>3} flights   '
                  f'validation {n_val:>3} flights', flush=True)

    for flight_id in sorted(held['flight_id'].astype(str).unique(), key=str):
        rows.append({'aircraft_type': withheld_aircraft_type, 'flight_id': flight_id,
                     'split': 'zero_shot_test', 'split_seed': split_seed,
                     'train_ratio': train_ratio, 'val_ratio': val_ratio})

    train_frame = data.loc[train_mask].reset_index(drop=True)
    val_frame = data.loc[val_mask].reset_index(drop=True)
    manifest = pd.DataFrame(rows)
    check_split(
        data, train_frame, val_frame, held, manifest,
        withheld_aircraft_type=withheld_aircraft_type,
        train_aircraft_types=train_aircraft_types,
    )
    return train_frame, val_frame, held, manifest


def check_split(
    data: pd.DataFrame,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    withheld_aircraft_type: str = WITHHELD_AIRCRAFT_TYPE,
    train_aircraft_types: Optional[Sequence[str]] = None,
) -> None:
    """校验 zero-shot 各分区只含允许的完整航班，且清单与数据一致。"""
    if withheld_aircraft_type not in STUDY_AIRCRAFT_TYPES:
        raise ValueError(f'未知留出机型 {withheld_aircraft_type!r}')
    expected_train_types = tuple(
        name for name in STUDY_AIRCRAFT_TYPES
        if name != withheld_aircraft_type
    )
    if train_aircraft_types is None:
        train_aircraft_types = expected_train_types
    else:
        train_aircraft_types = tuple(train_aircraft_types)
    if train_aircraft_types != expected_train_types:
        raise ValueError(
            f'源机型必须按既定顺序为 {list(expected_train_types)}，'
            f'实际为 {list(train_aircraft_types)}')

    source_types = set(train_aircraft_types)
    if set(train_frame['aircraft_type'].astype(str)) != source_types:
        raise RuntimeError('训练集机型集合与三个源机型不一致')
    if set(val_frame['aircraft_type'].astype(str)) != source_types:
        raise RuntimeError('验证集机型集合与三个源机型不一致')
    if set(test_frame['aircraft_type'].astype(str)) != {withheld_aircraft_type}:
        raise RuntimeError('零样本测试集必须只含留出机型')

    frames = {
        'train': train_frame,
        'validation': val_frame,
        'zero_shot_test': test_frame,
    }
    flights = {
        name: set(zip(frame['aircraft_type'].astype(str),
                      frame['flight_id'].astype(str)))
        for name, frame in frames.items()
    }
    names = list(flights)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = flights[left] & flights[right]
            if shared:
                raise RuntimeError(f'{left} 与 {right} 共用航班 {sorted(shared)[:3]}')

    if manifest[['aircraft_type', 'flight_id']].astype(str).duplicated().any():
        raise RuntimeError('划分清单中同一航班出现多次')
    manifest_pairs = {
        split_name: set(zip(
            manifest.loc[manifest['split'] == split_name, 'aircraft_type'].astype(str),
            manifest.loc[manifest['split'] == split_name, 'flight_id'].astype(str),
        ))
        for split_name in frames
    }
    for split_name in frames:
        if manifest_pairs[split_name] != flights[split_name]:
            raise RuntimeError(f'{split_name} 的划分清单与实际航班不一致')

    held_manifest = manifest[manifest['aircraft_type'] == withheld_aircraft_type]
    if set(held_manifest['split']) != {'zero_shot_test'}:
        raise RuntimeError('留出机型的清单标签不是 zero_shot_test')
    expected = set(data.loc[data['aircraft_type'] == withheld_aircraft_type,
                            'flight_id'].astype(str))
    if set(held_manifest['flight_id'].astype(str)) != expected:
        raise RuntimeError('划分清单未覆盖全部留出机型航班')
    if len(train_frame) + len(val_frame) + len(test_frame) != len(data):
        raise RuntimeError('存在未分配或重复分配的记录')


# ── 数据路 ──────────────────────────────────────────────────────

class TrajectoryWindowSampler(BatchSampler):
    """把同一航班内连续的若干秒拼成批，避免整批样本来自互不相干的时刻。"""

    def __init__(self, dataset: FlightRecordDataset, batch_size: int,
                 window_size: int = 32, drop_last: bool = False, seed: int = 42):
        self.batch_size = int(batch_size)
        self.window_size = max(2, min(int(window_size), self.batch_size))
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.windows = self._build(dataset)
        if not self.windows:
            raise ValueError('训练集中没有长度至少为 2 的连续片段')

    def _build(self, dataset: FlightRecordDataset) -> List[np.ndarray]:
        times = dataset.time.detach().cpu().numpy()
        flights = dataset.flight_ids.detach().cpu().numpy()
        windows: List[np.ndarray] = []
        for flight_id in np.unique(flights):
            rows = np.where(flights == flight_id)[0]
            if len(rows) < 2:
                continue
            rows = rows[np.argsort(times[rows])]
            for start in range(0, len(rows), self.window_size):
                window = rows[start:start + self.window_size]
                if len(window) >= 2:
                    windows.append(window)
        return windows

    def __iter__(self):
        batch: List[int] = []
        for index in self.rng.permutation(len(self.windows)):
            window = self.windows[index]
            if batch and len(batch) + len(window) > self.batch_size:
                yield batch
                batch = []
            batch.extend(window[:self.batch_size].tolist())
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(w) for w in self.windows)
        if self.drop_last:
            return total // self.batch_size
        return int(np.ceil(total / self.batch_size))


# ── 质量路与相位路 ──────────────────────────────────────────────

def build_mass_windows(dataset: FlightRecordDataset,
                       window_seconds: int = 900) -> List[np.ndarray]:
    """同一航段内时间连续、重量确有下降的窗口。

    唯一的设定是窗口长度。窗口每半个窗口长起一个，窗口内所有记录点都参与积分。
    """
    if window_seconds <= 0:
        raise ValueError('窗口长度必须为正')
    times = dataset.time.detach().cpu().numpy().astype(np.float64)
    flights = dataset.flight_ids.detach().cpu().numpy()
    mass = dataset.features[:, FEATURE_INDEX['GWC']].detach().cpu().numpy().astype(np.float64)
    stride = max(1, window_seconds // 2)
    windows: List[np.ndarray] = []

    for flight_id in np.unique(flights):
        rows = np.where(flights == flight_id)[0]
        rows = rows[np.argsort(times[rows])]
        gaps = np.diff(times[rows])
        breaks = np.where((gaps <= 0) | (gaps > 2.0))[0]

        for segment in np.split(rows, breaks + 1):
            if len(segment) < 2:
                continue
            seg_times = times[segment]
            next_start = seg_times[0]
            for start in range(len(segment) - 1):
                t0 = seg_times[start]
                if t0 + 1e-9 < next_start:
                    continue
                end = int(np.searchsorted(seg_times, t0 + window_seconds, side='left'))
                if end >= len(segment):
                    break
                window = segment[start:end + 1]
                if mass[window[0]] - mass[window[-1]] > 0:
                    windows.append(window.astype(np.int64))
                next_start = t0 + stride
    return windows


def build_transition_pairs(dataset: FlightRecordDataset
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """真实观测到的相邻相位过渡点。"""
    times = dataset.time.detach().cpu().numpy()
    flights = dataset.flight_ids.detach().cpu().numpy()
    phases = dataset.phases.detach().cpu().numpy()
    order = np.lexsort((times, flights))
    f, t, p = flights[order], times[order], phases[order]
    step = t[1:] - t[:-1]
    adjacent = (f[1:] == f[:-1]) & (step > 0) & (step <= 2.0) & (p[1:] != p[:-1])
    rows = order[:-1][adjacent]
    return (rows.astype(np.int64), p[:-1][adjacent].astype(np.int64),
            p[1:][adjacent].astype(np.int64))


class CyclicBatches:
    """把有限个窗口或过渡点循环打乱后按固定大小取出。"""

    def __init__(self, items: Sequence, batch_size: int, seed: int):
        self.items = list(items)
        self.batch_size = max(1, int(batch_size))
        self.rng = np.random.default_rng(seed)
        self._order: List[int] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self.items)

    def next(self) -> List:
        if not self.items:
            return []
        if self._cursor >= len(self._order):
            self._order = list(self.rng.permutation(len(self.items)))
            self._cursor = 0
        picked = self._order[self._cursor:self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return [self.items[i] for i in picked]


def collate_mass_windows(dataset: FlightRecordDataset, windows: Sequence[np.ndarray]):
    """窗口点数不等时按最短对齐，首末两点始终保留。"""
    if not windows:
        raise ValueError('质量窗口批为空')
    length = min(len(w) for w in windows)
    if length < 2:
        raise ValueError('质量窗口至少需要两个时间点')

    aligned = []
    for window in windows:
        window = np.asarray(window, dtype=np.int64)
        if len(window) > length:
            pick = np.linspace(0, len(window) - 1, length).round().astype(int)
            pick[0], pick[-1] = 0, len(window) - 1
            window = window[pick]
        aligned.append(window)

    stacked = np.stack(aligned)
    flat = torch.as_tensor(stacked.reshape(-1), dtype=torch.long).to(dataset.features.device)
    n, k = stacked.shape
    return (dataset.features[flat].reshape(n, k, -1),
            dataset.phases[flat].reshape(n, k),
            dataset.aircraft_type_ids[flat].reshape(n, k),
            dataset.time[flat].reshape(n, k))


def collate_transitions(dataset: FlightRecordDataset, rows: Sequence[Tuple[int, int, int]]):
    if not rows:
        raise ValueError('相位过渡批为空')
    device = dataset.features.device
    index = torch.as_tensor([r[0] for r in rows], dtype=torch.long, device=device)
    return (dataset.features[index],
            dataset.aircraft_type_ids[index],
            torch.as_tensor([r[1] for r in rows], dtype=torch.long, device=device),
            torch.as_tensor([r[2] for r in rows], dtype=torch.long, device=device))
