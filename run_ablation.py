# -*- coding: utf-8 -*-
"""模块消融实验

在完整模型上逐个去掉一块，看留出集上掉多少：

  no_phase      相位子网整块去掉，共享编码器的输出直接进融合层。
  shared_phase  六个分相位子网换成一个共用子网。形状 64→64→32→64、深度、残差相加
                都不变，去掉的只是"按相位取不同权重"这一步。
  no_aircraft   关掉机型静态特征那一路，融合层退化成只吃状态表示。
  no_physics    三个物理项权重同时置零，结构与完整模型逐参数相同。

两种相位消融的区别：shared_phase 保留了那一块的容量与深度，只去掉分相位；no_phase
连容量一起去掉。两个一起看才能分清掉的是"分相位"还是"少了一块网络"。

相位标签在模型里只从这一路进来，所以 no_phase 与 shared_phase 两种情况下模型都不
知道自己处在哪个飞行阶段，相位连续项恒为零，训练入口会自动把它的权重置零；诊断表
里这两格的相位跳变必然是 0.00%，是结构使然。

fleet-wide 的 no_physics 由航班采样实验的 all_flights_no_physics 那一格产出，设置
完全相同，这里不重复训练，只在汇总时取用。两个划分的完整模型同样只取用不重训。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import train_fleet_wide
import train_zero_shot
from af_train import log

SUMMARY_DIR = Path('results/ablation')

# 条件 → (命令行开关, 中文名)
CONDITIONS: Dict[str, Tuple[List[str], str]] = {
    'full':         ([], '完整模型'),
    'no_phase':     (['--phase_mode', 'none'], '去掉相位子网'),
    'shared_phase': (['--phase_mode', 'shared'], '相位子网不分相位'),
    'no_aircraft':  (['--no_aircraft_features'], '去掉机型特征'),
    'no_physics':   (['--no_physics'], '去掉物理损失'),
}

# 划分 → (入口模块, 评估分区)
SPLITS: Dict[str, Tuple[object, str]] = {
    'fleet_wide': (train_fleet_wide, 'test'),
    'zero_shot':  (train_zero_shot, 'zero_shot_test'),
}

# 本脚本负责训练的格子。full 两个划分都已有；fleet-wide 的 no_physics 由采样实验产出
TRAIN_CONDITIONS: Dict[str, List[str]] = {
    'fleet_wide': ['no_phase', 'shared_phase', 'no_aircraft'],
    'zero_shot':  ['no_phase', 'shared_phase', 'no_aircraft', 'no_physics'],
}

DIAGNOSTIC_COLUMNS = ['mass_accounting_ratio_median', 'fpa_violation_rate',
                      'gwc_violation_rate_valid_gate', 'ff1_transition_gap']


def run_dir_of(split: str, condition: str) -> Path:
    """目录名由入口自己算，驱动这边不另抄一份规则。"""
    entry = SPLITS[split][0]
    args = entry.build_parser().parse_args(CONDITIONS[condition][0])
    return args.out_dir / entry.auto_tag(args)


def is_done(run_dir: Path, seeds: Sequence[int]) -> bool:
    path = run_dir / 'per_seed_metrics.csv'
    if not path.exists():
        return False
    try:
        done = set(pd.read_csv(path)['seed'].unique())
    except Exception:
        return False
    return set(seeds).issubset(done)


def run_cell(split: str, condition: str, seeds: Sequence[int],
             epochs: int, device: str) -> None:
    argv = ['--seeds', *[str(s) for s in seeds], '--epochs', str(epochs),
            '--device', device, *CONDITIONS[condition][0]]
    SPLITS[split][0].main(argv)


def collect(splits: Sequence[str], seeds: Sequence[int]
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, diagnostic_rows = [], []
    for split in splits:
        for condition in CONDITIONS:
            run_dir = run_dir_of(split, condition)
            metric_path = run_dir / 'per_seed_metrics.csv'
            diagnostic_path = run_dir / 'per_seed_diagnostics.csv'
            if not (metric_path.exists() and diagnostic_path.exists()):
                continue
            metrics = pd.read_csv(metric_path)
            diagnostics = pd.read_csv(diagnostic_path)
            for frame in (metrics, diagnostics):
                frame['split'] = split
                frame['condition'] = condition
            metric_rows.append(metrics[metrics.seed.isin(seeds)])
            diagnostic_rows.append(diagnostics[diagnostics.seed.isin(seeds)])
    if not metric_rows:
        raise RuntimeError('没有找到任何已完成的格子')
    return (pd.concat(metric_rows, ignore_index=True),
            pd.concat(diagnostic_rows, ignore_index=True))


def report(metrics: pd.DataFrame, diagnostics: pd.DataFrame,
           splits: Sequence[str], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_dir / 'ablation_metrics.csv', index=False)
    diagnostics.to_csv(summary_dir / 'ablation_diagnostics.csv', index=False)

    summary_rows = []
    for split in splits:
        partition = SPLITS[split][1]
        block = metrics[(metrics['split'] == split)
                        & (metrics.partition == partition)
                        & (metrics.aircraft_type == 'all')
                        & (metrics.target == 'FF_Total')]
        if block.empty:
            continue
        table = block.groupby('condition').agg(
            n_seeds=('seed', 'nunique'), R2=('R2', 'mean'), R2_sd=('R2', 'std'),
            RMSE=('RMSE', 'mean'), MAPE=('MAPE', 'mean'))
        reference = table.loc['full'] if 'full' in table.index else None

        log('\n' + '=' * 80)
        log(f'{split}   分区 {partition}   FF_Total')
        log('=' * 80)
        log(f'{"条件":<18} {"种子":>4} {"R2":>8} {"标准差":>8} {"ΔR2":>9} '
            f'{"MAPE":>8} {"ΔMAPE":>9}')
        for condition in CONDITIONS:
            if condition not in table.index:
                continue
            row = table.loc[condition]
            d_r2 = '' if reference is None else f'{row.R2 - reference.R2:+.4f}'
            d_mape = ('' if reference is None
                      else f'{row.MAPE - reference.MAPE:+.2f}%')
            log(f'{CONDITIONS[condition][1]:<16} {int(row.n_seeds):>5} '
                f'{row.R2:>8.4f} {row.R2_sd:>8.4f} {d_r2:>9} '
                f'{row.MAPE:>7.2f}% {d_mape:>9}')
            summary_rows.append({
                'split': split, 'partition': partition, 'condition': condition,
                'n_seeds': int(row.n_seeds), 'R2': row.R2, 'R2_sd': row.R2_sd,
                'RMSE': row.RMSE, 'MAPE': row.MAPE,
                'delta_R2': None if reference is None else row.R2 - reference.R2,
                'delta_MAPE': None if reference is None
                              else row.MAPE - reference.MAPE,
            })

        physics = diagnostics[(diagnostics['split'] == split)
                              & (diagnostics.partition == partition)]
        if physics.empty:
            continue
        physics = physics.groupby('condition')[DIAGNOSTIC_COLUMNS].mean()
        log(f'\n{"条件":<18} {"油耗账":>8} {"航迹角违反":>11} {"重量违反":>10} '
            f'{"相位跳变":>10}')
        for condition in CONDITIONS:
            if condition not in physics.index:
                continue
            r = physics.loc[condition]
            log(f'{CONDITIONS[condition][1]:<16} '
                f'{r.mass_accounting_ratio_median:>10.3f} '
                f'{r.fpa_violation_rate * 100:>10.2f}% '
                f'{r.gwc_violation_rate_valid_gate * 100:>9.2f}% '
                f'{r.ff1_transition_gap * 100:>9.2f}%')

    pd.DataFrame(summary_rows).to_csv(summary_dir / 'ablation_summary.csv',
                                      index=False)
    log(f'\n汇总已写入 {summary_dir}')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='模块消融实验')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS),
                        choices=list(SPLITS))
    parser.add_argument('--conditions', nargs='+', default=None,
                        choices=[c for c in CONDITIONS if c != 'full'],
                        help='不给则按划分取默认：fleet-wide 三格，zero-shot 四格')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--summary_dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--force', action='store_true', help='已完成的格子也重跑')
    parser.add_argument('--collect_only', action='store_true',
                        help='不训练，只汇总已有结果')
    args = parser.parse_args(argv)

    log('模块消融实验')
    log(f'划分 {args.splits}   种子 {args.seeds}')
    pending: List[Tuple[str, str]] = []
    for split in args.splits:
        for condition in (args.conditions or TRAIN_CONDITIONS[split]):
            run_dir = run_dir_of(split, condition)
            done = is_done(run_dir, args.seeds) and not args.force
            log(f'  {split:<11} {condition:<13} {run_dir}   '
                + ('已完成，跳过' if done else '待跑'))
            if not done:
                pending.append((split, condition))

    if not args.collect_only and pending:
        log(f'\n共需训练 {len(pending)} 格 × {len(args.seeds)} 个种子')
        for index, (split, condition) in enumerate(pending, 1):
            started = time.perf_counter()
            log('\n' + '#' * 88)
            log(f'# 第 {index}/{len(pending)} 格   {split}   {condition}')
            log('#' * 88)
            run_cell(split, condition, args.seeds, args.epochs, args.device)
            log(f'本格耗时 {(time.perf_counter() - started) / 60:.1f} 分钟')

    metrics, diagnostics = collect(args.splits, args.seeds)
    report(metrics, diagnostics, args.splits, args.summary_dir)


if __name__ == '__main__':
    main()
