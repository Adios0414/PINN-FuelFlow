# -*- coding: utf-8 -*-
"""航班采样实验的结果图与对比表。

纵轴为测试误差，横轴为训练航班总数。每个点为种子 42–46 的算术均值，误差棒为
± 标准差（样本标准差，ddof=1）；两个条件在横向错开少许以免误差棒重叠。两个条件
在每一档使用相同的训练航班和相同的随机种子，因此可以直接比较。
另输出 SI 表（table_flight_sampling_sd）：各档两个模型 RMSE 与 MAPE 的均值 ± 标准差。

  a  测试集 RMSE
  b  测试集 MAPE（论文小数口径）

数据取自 run_flight_sampling.py 汇总得到的 CSV，不重新推理。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_DIR = Path('results/flight_sampling')
OUT_DIR = Path('results/figures')

CONDITIONS = [
    ('full', '#2A78D6', 'o', '-', 'With physics'),
    ('no_physics', '#EB6834', 's', (0, (4, 2)), 'Without physics'),
]
PANELS = [
    ('RMSE', 'Test RMSE (kg s$^{-1}$)', 'a', 1.0),
    ('MAPE', 'Test MAPE', 'b', 0.01),
]
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_LEVELS = (4, 8, 16, 32, 66)


def setup_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
        'ytick.minor.size': 1.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        # 只在左轴和下轴打刻度，且只留标注出来的主刻度；上轴右轴保留边框不打刻度
        'xtick.top': False, 'ytick.right': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def load(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / 'sampling_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f'未找到采样实验结果 {path}')
    metrics = pd.read_csv(path)
    required = {'seed', 'partition', 'aircraft_type', 'target', 'condition',
                'n_train_flights', 'n_train_records', 'RMSE', 'MAPE'}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise RuntimeError(f'采样实验结果缺少列 {missing}')

    metrics = metrics[(metrics.partition == 'test')
                      & (metrics.aircraft_type == 'all')
                      & (metrics.target == 'FF_Total')].copy()
    if metrics.empty:
        raise RuntimeError('采样实验没有 test/all/FF_Total 指标')

    key = ['n_train_flights', 'condition', 'seed']
    duplicates = metrics[metrics.duplicated(key, keep=False)]
    if not duplicates.empty:
        raise RuntimeError(f'采样实验存在重复 level/condition/seed：{duplicates[key].to_dict("records")}')

    observed_levels = tuple(sorted(int(v) for v in metrics.n_train_flights.unique()))
    observed_conditions = set(metrics.condition.unique())
    expected_conditions = {condition for condition, *_ in CONDITIONS}
    if observed_levels != EXPECTED_LEVELS:
        raise RuntimeError(f'训练航班档位为 {observed_levels}，预期 {EXPECTED_LEVELS}')
    if observed_conditions != expected_conditions:
        raise RuntimeError(
            f'条件为 {sorted(observed_conditions)}，预期 {sorted(expected_conditions)}')

    expected_seeds = set(EXPECTED_SEEDS)
    errors = []
    for flights in EXPECTED_LEVELS:
        for condition in expected_conditions:
            seeds = set(metrics.loc[
                (metrics.n_train_flights == flights)
                & (metrics.condition == condition), 'seed'].astype(int))
            if seeds != expected_seeds:
                errors.append({
                    'flights': flights,
                    'condition': condition,
                    'missing': sorted(expected_seeds - seeds),
                    'unexpected': sorted(seeds - expected_seeds),
                })
    if errors:
        raise RuntimeError(f'采样实验种子未完整覆盖 42–46：{errors}')
    return metrics


def draw(metrics: pd.DataFrame, out_stem: Path) -> None:
    levels = list(EXPECTED_LEVELS)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.2, 2.8))

    for ax, (column, ylabel, tag, scale) in zip(axes, PANELS):
        for (condition, colour, marker, linestyle, label), dodge in zip(CONDITIONS, (-0.06, 0.06)):
            means, sds = [], []
            for flights in levels:
                values = metrics.loc[
                    (metrics.n_train_flights == flights)
                    & (metrics.condition == condition), column
                ].to_numpy(dtype=float)
                means.append(float(values.mean()) * scale)
                sds.append(float(values.std(ddof=1)) * scale)
            positions = np.arange(len(levels)) + dodge
            ax.errorbar(
                positions, means, yerr=sds, fmt='none', ecolor=colour,
                elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=2,
            )
            ax.plot(
                positions, means, color=colour, marker=marker,
                linestyle=linestyle, markersize=4.6, lw=1.2,
                markeredgecolor='white', markeredgewidth=0.55,
                zorder=3, label=label,
            )
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(v) for v in levels])
        ax.set_xlim(-0.35, len(levels) - 0.65)
        ax.set_xlabel('Training flights (total)')
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc='upper right')

    fig.tight_layout(w_pad=1.8)
    # 版面定下来之后再放 a/b，左边缘对齐各自 y 轴标签的左边缘。
    fig.canvas.draw()
    for ax, (_, _, tag, _) in zip(axes, PANELS):
        left = ax.yaxis.label.get_window_extent().x0
        x = ax.transAxes.inverted().transform((left, 0))[0]
        ax.text(x, 1.04, tag, transform=ax.transAxes,
                fontsize=10, fontweight='bold')
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix))
    # Source Data：误差棒由各种子的值算出，逐种子给出（MAPE 写成小数，与图一致）
    source = metrics[['n_train_flights', 'n_train_records', 'condition', 'seed', 'RMSE', 'MAPE']].copy()
    source['MAPE'] = source['MAPE'] * 0.01
    source.sort_values(['n_train_flights', 'condition', 'seed']).to_csv(
        out_stem.parent / f'{out_stem.name}_source_data.csv', index=False)
    plt.close(fig)
    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')


def write_table(metrics: pd.DataFrame, out_stem: Path) -> None:
    """原始数值表：每一档两个条件的 MAPE 与 RMSE，五个种子均值。"""
    labels = [label for *_, label in CONDITIONS]
    rows = []
    for flights in sorted(metrics.n_train_flights.unique()):
        block = metrics[metrics.n_train_flights == flights]
        row = {'Training flights': int(flights),
               'Training records': int(block.n_train_records.iloc[0])}
        for condition, *_, label in CONDITIONS:
            part = block[block.condition == condition]
            row[f'{label} MAPE'] = part.MAPE.mean()
            row[f'{label} RMSE'] = part.RMSE.mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_stem.with_suffix('.csv'), index=False)

    head = ('| Training flights | Records | '
            + ' | '.join(f'{lab} {m}' for lab in labels for m in ('MAPE', 'RMSE'))
            + ' |')
    lines = [head, '|' + '---|' * (2 + 2 * len(labels))]
    latex = []
    for record in table.to_dict('records'):
        mape = [record[f'{lab} MAPE'] for lab in labels]
        rmse = [record[f'{lab} RMSE'] for lab in labels]
        cells = []
        for index, _ in enumerate(labels):
            # MAPE 与 RMSE 一样写成小数
            text = f'{mape[index] / 100:.4f}'
            cells.append(f'**{text}**' if mape[index] == min(mape) else text)
            text = f'{rmse[index]:.4f}'
            cells.append(f'**{text}**' if rmse[index] == min(rmse) else text)
        lines.append(f'| {record["Training flights"]} | '
                     f'{record["Training records"]:,} | ' + ' | '.join(cells) + ' |')
        latex.append(f'{record["Training flights"]} & '
                     f'{record["Training records"]:,} & '
                     + ' & '.join(c.replace('**', r'\textbf{', 1).replace('**', '}', 1)
                                  if '**' in c else c for c in cells) + r' \\')

    markdown = '\n'.join(lines)
    with open(out_stem.with_suffix('.md'), 'w', encoding='utf-8') as handle:
        handle.write(markdown + '\n')
    header = (r'\begin{tabular}{rr' + 'rr' * len(labels) + '}\n\\hline\n'
              + 'Training flights & Records & '
              + ' & '.join(f'{lab} MAPE & RMSE' for lab in labels)
              + r' \\' + '\n\\hline\n')
    with open(out_stem.with_suffix('.tex'), 'w', encoding='utf-8') as handle:
        handle.write(header + '\n'.join(latex) + '\n\\hline\n\\end{tabular}\n')
    print(f'表已保存 {out_stem.with_suffix(".csv")} / .md / .tex')
    print()
    print(markdown)


def write_sd_table(metrics: pd.DataFrame, out_stem: Path) -> None:
    """SI 表：每一档两个条件 RMSE 与 MAPE 的均值 ± 标准差（五个种子，ddof=1）。
    MAPE 与正文表 2、表 3 一样写成小数。"""
    rows, lines = [], []
    head = ['Training flights', 'Training records']
    for metric in ('RMSE (kg s⁻¹)', 'MAPE'):
        head += [f'{metric}, {label.lower()}' for *_, label in CONDITIONS]
    lines.append('| ' + ' | '.join(head) + ' |')
    lines.append('|' + '---|' * len(head))
    for flights in EXPECTED_LEVELS:
        block = metrics[metrics.n_train_flights == flights]
        row = {'Training flights': flights, 'Training records': int(block.n_train_records.iloc[0])}
        cells = [str(flights), f'{row["Training records"]:,}']
        for metric, scale in (('RMSE', 1.0), ('MAPE', 0.01)):
            for condition, *_, label in CONDITIONS:
                values = block.loc[block.condition == condition, metric].to_numpy(float) * scale
                row[f'{label} {metric} mean'] = values.mean()
                row[f'{label} {metric} sd'] = values.std(ddof=1)
                cells.append(f'{values.mean():.4f} ± {values.std(ddof=1):.4f}')
        rows.append(row)
        lines.append('| ' + ' | '.join(cells) + ' |')
    pd.DataFrame(rows).to_csv(out_stem.with_suffix('.csv'), index=False)
    caption = (
        '**Supplementary Table 6 | Test errors of the models with and without physics losses '
        'at different numbers of training flights.** Errors are for total fuel flow (FF1 + FF2) '
        'on the pooled held-out test set. The training flights correspond to 1, 2, 4 and 8 flights '
        'per aircraft type and the full training set; the validation and test sets were identical '
        'across conditions. Values are means ± s.d. over five random seeds. RMSE is given in '
        'kg s⁻¹, and MAPE as a fraction.')
    with open(out_stem.with_suffix('.md'), 'w', encoding='utf-8') as handle:
        handle.write(caption + '\n\n' + '\n'.join(lines) + '\n')
    print(f'SI 表已保存 {out_stem.with_suffix(".csv")} / .md')
    print('\n'.join(lines))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='航班采样实验结果图与表')
    parser.add_argument('--summary_dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    setup_style()
    metrics = load(args.summary_dir)
    draw(metrics, args.out_dir / 'fig_flight_sampling')
    write_table(metrics, args.out_dir / 'table_flight_sampling')
    write_sd_table(metrics, args.out_dir / 'table_flight_sampling_sd')


if __name__ == '__main__':
    main()
