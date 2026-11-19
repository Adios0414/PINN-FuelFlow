# -*- coding: utf-8 -*-
"""Fleet-wide 结果图与论文表格。

主图为 3 × 3：
    第一行：PINN 对 FF1、FF2 和总燃油流量的预测值与观测值；
    第二行：四个模型在六个飞行阶段上的 RMSE；
    第三行：四个模型在四种飞机类型上的 RMSE。

散点使用一个代表性随机种子的预测值，格内 R²、柱图和表格均使用所选随机种子的
算术均值。所有展示只给均值，不绘制误差条或区间。主表比较三个输出上的总体
R²、RMSE 和 MAPE；MAPE 在论文表中写成小数。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from af_data import (
    PHASE_NAMES, STUDY_AIRCRAFT_TYPES, FlightRecordDataset,
    load_flight_table, split_fleet_wide,
)
from af_train import predict, restore_model

PINN_DIR = Path('results/fleet_wide/all_flights_full')
BASELINE_DIR = Path('results/baselines/fleet_wide')
OUT_DIR = Path('results/figures')

MODELS = [('PINN', 'PINN'), ('mlp', 'MLP'), ('rbf', 'RBF'), ('rf', 'RF')]
TARGETS = [('FF1', 'Engine 1 fuel flow'), ('FF2', 'Engine 2 fuel flow'),
           ('FF_Total', 'Total fuel flow')]
MODEL_LABEL = dict(MODELS)
TARGET_LABEL = dict(TARGETS)

# 六个阶段沿用现有 NPG 风格，并以点形作第二重编码。
PHASE_STYLE: Dict[str, Tuple[str, str]] = {
    'takeoff':  ('#E64B35', 'D'),
    'climb':    ('#3C5488', '^'),
    'cruise':   ('#00A087', 's'),
    'descent':  ('#F39B7F', 'v'),
    'approach': ('#4DBBD5', 'o'),
    'landing':  ('#8491B4', 'P'),
}
PHASE_LABEL = {'takeoff': 'Take-off', 'climb': 'Climb', 'cruise': 'Cruise',
               'descent': 'Descent', 'approach': 'Approach', 'landing': 'Landing'}

# 通过颜色可辨识度校验的相邻分类色；纹理再提供一重模型身份编码。
MODEL_STYLE: Dict[str, Tuple[str, str]] = {
    'PINN': ('#2A78D6', ''),
    'mlp':  ('#EB6834', '///'),
    'rbf':  ('#1BAF7A', '\\\\'),
    'rf':   ('#EDA100', 'xx'),
}
REFERENCE_LINE = '#2B2B2B'
BAR_EDGE = '#303030'


def setup_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.2,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': False, 'ytick.right': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.facecolor': 'white',
        'axes.axisbelow': True, 'savefig.bbox': 'tight',
        'hatch.linewidth': 0.55,
    })


def get_pinn_predictions(dataset: FlightRecordDataset, seed: int,
                         manifest: pd.DataFrame, device: torch.device) -> np.ndarray:
    path = PINN_DIR / f'seed_{seed}' / 'best_model.pt'
    if not path.exists():
        raise FileNotFoundError(f'缺少 PINN 检查点 {path}')
    model, _ = restore_model(path, manifest, 42, device)
    predictions, _ = predict(model, dataset, device, 8192)
    return predictions


def load_baseline_predictions(model_key: str, seed: int, partition: str) -> np.ndarray:
    path = BASELINE_DIR / 'predictions' / f'{model_key}_seed{seed}_{partition}.npy'
    if not path.exists():
        raise FileNotFoundError(f'缺少基线预测值 {path}，请先运行相应基线实验')
    return np.load(path)


def target_columns(predictions: np.ndarray, targets: np.ndarray,
                   name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == 'FF1':
        return targets[:, 0], predictions[:, 0]
    if name == 'FF2':
        return targets[:, 1], predictions[:, 1]
    return targets.sum(axis=1), predictions.sum(axis=1)


def build_mean_metrics(seeds: Sequence[int]) -> pd.DataFrame:
    """读取逐种子指标，并只对明确选择的完整种子取均值。"""
    pinn = pd.read_csv(PINN_DIR / 'per_seed_metrics.csv')
    pinn = pinn[pinn.partition == 'test'].assign(model='PINN')
    base = pd.read_csv(BASELINE_DIR / 'baseline_metrics.csv')
    base = base[base.partition == 'test']
    columns = ['model', 'seed', 'aircraft_type', 'target', 'n_records',
               'R2', 'RMSE', 'MAE', 'MAPE']
    joined = pd.concat([pinn[columns], base[columns]], ignore_index=True)
    joined = joined[joined.seed.isin(seeds)].copy()

    expected = {(model, int(seed), target)
                for model, _ in MODELS for seed in seeds for target, _ in TARGETS}
    observed = set(joined.loc[joined.aircraft_type == 'all',
                              ['model', 'seed', 'target']].itertuples(index=False, name=None))
    missing = sorted(expected - observed, key=str)
    if missing:
        raise RuntimeError(f'总体指标缺少模型/种子/输出组合：{missing}')

    return joined.groupby(['aircraft_type', 'target', 'model'], as_index=False)[
        ['n_records', 'R2', 'RMSE', 'MAE', 'MAPE']].mean()


def mean_scores(mean_table: pd.DataFrame
                ) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """格内标注用的 R² 与 RMSE，均为所选种子的均值。"""
    block = mean_table[mean_table.aircraft_type == 'all']
    return {(row.model, row.target): (float(row.R2), float(row.RMSE))
            for row in block.itertuples(index=False)}


def load_all_predictions(dataset: FlightRecordDataset, seeds: Sequence[int],
                         manifest: pd.DataFrame, device: torch.device,
                         n_targets: int) -> Dict[int, Dict[str, np.ndarray]]:
    """载入各模型逐种子测试集预测，不重新训练模型。"""
    all_predictions: Dict[int, Dict[str, np.ndarray]] = {}
    for seed in seeds:
        print(f'载入 seed {seed} 的测试集预测', flush=True)
        predictions = {'PINN': get_pinn_predictions(dataset, int(seed), manifest, device)}
        for key, _ in MODELS[1:]:
            predictions[key] = load_baseline_predictions(key, int(seed), 'test')
        for key, values in predictions.items():
            if values.shape != (len(dataset), n_targets):
                raise RuntimeError(
                    f'{key} seed {seed} 的预测形状 {values.shape}，'
                    f'应为 {(len(dataset), n_targets)}')
            if not np.isfinite(values).all():
                raise RuntimeError(f'{key} seed {seed} 的预测包含非有限值')
        all_predictions[int(seed)] = predictions
    return all_predictions


def compute_rmse_breakdowns(
        predictions_by_seed: Dict[int, Dict[str, np.ndarray]],
        seeds: Sequence[int], targets: np.ndarray, phases: np.ndarray,
        aircraft_types: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """由逐记录预测计算分阶段和分机型 RMSE，再在种子间取均值。"""
    phase_rows: List[Dict[str, object]] = []
    aircraft_rows: List[Dict[str, object]] = []

    for seed in seeds:
        for model_key, _ in MODELS:
            prediction = predictions_by_seed[int(seed)][model_key]
            for target_key, _ in TARGETS:
                true_values, predicted_values = target_columns(prediction, targets, target_key)
                squared_error = np.square(predicted_values - true_values)

                for phase_id, phase_name in enumerate(PHASE_NAMES):
                    mask = phases == phase_id
                    phase_rows.append({
                        'seed': int(seed), 'target': target_key, 'phase': phase_name,
                        'model': model_key, 'n_records': int(mask.sum()),
                        'RMSE': float(np.sqrt(squared_error[mask].mean())),
                    })

                for aircraft_type in STUDY_AIRCRAFT_TYPES:
                    mask = aircraft_types == aircraft_type
                    aircraft_rows.append({
                        'seed': int(seed), 'target': target_key,
                        'aircraft_type': aircraft_type, 'model': model_key,
                        'n_records': int(mask.sum()),
                        'RMSE': float(np.sqrt(squared_error[mask].mean())),
                    })

    phase = pd.DataFrame(phase_rows).groupby(
        ['target', 'phase', 'model'], as_index=False).agg(
            n_records=('n_records', 'first'), RMSE=('RMSE', 'mean'))
    aircraft = pd.DataFrame(aircraft_rows).groupby(
        ['target', 'aircraft_type', 'model'], as_index=False).agg(
            n_records=('n_records', 'first'), RMSE=('RMSE', 'mean'))
    return phase, aircraft


def scatter_chunks(phases: np.ndarray, per_phase_points: int
                   ) -> Tuple[List[Tuple[int, np.ndarray]], Dict[int, int]]:
    """交错各阶段的散点绘制顺序，避免后绘制阶段系统性遮挡前面阶段。"""
    rng = np.random.default_rng(0)
    chunks: List[Tuple[int, np.ndarray]] = []
    counts: Dict[int, int] = {}
    for phase_id in range(len(PHASE_NAMES)):
        rows = np.where(phases == phase_id)[0]
        if len(rows) == 0:
            continue
        if per_phase_points and len(rows) > per_phase_points:
            rows = rng.choice(rows, per_phase_points, replace=False)
        counts[phase_id] = len(rows)
        rng.shuffle(rows)
        n_chunks = max(1, min(40, len(rows) // 200))
        chunks.extend((phase_id, part) for part in np.array_split(rows, n_chunks))
    rng.shuffle(chunks)
    return chunks, counts


def draw_grouped_bars(ax: plt.Axes, table: pd.DataFrame, target: str,
                      category_column: str, category_order: Sequence[str],
                      category_labels: Sequence[str], rotate: float = 0.0) -> None:
    positions = np.arange(len(category_order), dtype=float)
    width = 0.18
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * width
    highest = 0.0

    for model_index, (model_key, _) in enumerate(MODELS):
        block = table[(table.target == target) & (table.model == model_key)].set_index(
            category_column)
        missing = [value for value in category_order if value not in block.index]
        if missing:
            raise RuntimeError(f'{target}/{model_key} 缺少 {category_column}: {missing}')
        values = block.loc[list(category_order), 'RMSE'].to_numpy(dtype=float)
        highest = max(highest, float(values.max()))
        colour, hatch = MODEL_STYLE[model_key]
        ax.bar(positions + offsets[model_index], values, width=width,
               color=colour, edgecolor=BAR_EDGE, linewidth=0.35,
               hatch=hatch, zorder=2)

    ax.set_xticks(positions)
    ax.set_xticklabels(category_labels, rotation=rotate,
                       ha='right' if rotate else 'center', rotation_mode='anchor')
    ax.set_xlim(-0.55, len(category_order) - 0.45)
    ax.set_ylim(0, highest * 1.16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))


def draw_figure(representative_predictions: np.ndarray, targets: np.ndarray,
                phases: np.ndarray,
                score_lookup: Dict[Tuple[str, str], Tuple[float, float]],
                phase_rmse: pd.DataFrame, aircraft_rmse: pd.DataFrame,
                out_stem: Path, point_size: float, alpha: float,
                per_phase_points: int) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 8.8))
    chunks, counts = scatter_chunks(phases, per_phase_points)
    print('散点中的阶段记录数：' + '；'.join(
        f'{PHASE_LABEL[PHASE_NAMES[k]]} {v:,}' for k, v in counts.items()))

    # 第一行：代表性种子的 PINN 逐记录预测；R² 是所选种子的均值。
    for col, (target_key, target_label) in enumerate(TARGETS):
        ax = axes[0, col]
        true_all, predicted_all = target_columns(
            representative_predictions, targets, target_key)
        for phase_id, rows in chunks:
            colour, marker = PHASE_STYLE[PHASE_NAMES[phase_id]]
            ax.scatter(true_all[rows], predicted_all[rows], s=point_size,
                       c=colour, marker=marker, alpha=alpha, linewidths=0,
                       rasterized=True)
        high = float(max(true_all.max(), predicted_all.max())) * 1.04
        ax.plot([0, high], [0, high], ls='--', lw=1.0,
                color=REFERENCE_LINE, zorder=5)
        ax.set_xlim(0, high)
        ax.set_ylim(0, high)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        # 位数与主表一致，避免同一数值在图与表上因舍入位数不同而看着对不上
        r2_value, rmse_value = score_lookup[('PINN', target_key)]
        ax.text(0.045, 0.95,
                f'$R^2$ = {r2_value:.4f}\nRMSE = {rmse_value:.4f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
                linespacing=1.5, color='#0B0B0B',
                bbox={'facecolor': 'white', 'edgecolor': 'none',
                      'alpha': 0.78, 'pad': 1.0})
        ax.set_title(target_label, pad=8, fontweight='normal')
        ax.set_xlabel('Observed (kg s$^{-1}$)')
        if col == 0:
            ax.set_ylabel('Predicted (kg s$^{-1}$)')

    # 第二行：同一阶段内比较四个模型；第三行：同一机型内比较四个模型。
    phase_labels = [PHASE_LABEL[name] for name in PHASE_NAMES]
    aircraft_labels = [name.replace('-NEO', '\nNEO') for name in STUDY_AIRCRAFT_TYPES]
    lower_panel_targets = ('Engine 1', 'Engine 2', 'Total')
    for col, ((target_key, _), panel_target) in enumerate(
            zip(TARGETS, lower_panel_targets)):
        draw_grouped_bars(axes[1, col], phase_rmse, target_key, 'phase',
                          PHASE_NAMES, phase_labels, rotate=34)
        draw_grouped_bars(axes[2, col], aircraft_rmse, target_key, 'aircraft_type',
                          STUDY_AIRCRAFT_TYPES, aircraft_labels)
        axes[1, col].set_title(f'Phase-specific RMSE ({panel_target})',
                               pad=7, fontweight='normal')
        axes[2, col].set_title(f'Aircraft-specific RMSE ({panel_target})',
                               pad=7, fontweight='normal')
        if col == 0:
            axes[1, col].set_ylabel('RMSE (kg s$^{-1}$)')
            axes[2, col].set_ylabel('RMSE (kg s$^{-1}$)')

    # 三行只保留面板字母；每个坐标轴已有完整、独立的标题。
    fig.subplots_adjust(left=0.080, right=0.992, top=0.910, bottom=0.110,
                        wspace=0.30, hspace=0.43)
    for row, letter in enumerate(('a', 'b', 'c')):
        axes[row, 0].text(-0.30, 1.035, letter,
                          transform=axes[row, 0].transAxes,
                          ha='left', va='bottom', fontsize=9,
                          fontweight='bold', clip_on=False)

    phase_handles = [
        Line2D([], [], marker=PHASE_STYLE[name][1], linestyle='none',
               markerfacecolor=PHASE_STYLE[name][0],
               markeredgecolor=PHASE_STYLE[name][0], markersize=5,
               label=PHASE_LABEL[name])
        for name in PHASE_NAMES
    ]
    phase_handles.append(Line2D([], [], ls='--', lw=1.0,
                                color=REFERENCE_LINE, label='1:1 line'))
    model_handles = [
        Patch(facecolor=MODEL_STYLE[key][0], edgecolor=BAR_EDGE,
              linewidth=0.35, hatch=MODEL_STYLE[key][1], label=label)
        for key, label in MODELS
    ]
    # 每一行使用自己的图例，并紧接在该行图的下方。
    phase_legend_y = (
        axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2 - 0.010
    phase_legend = fig.legend(
        handles=phase_handles, loc='center', ncol=7, frameon=False,
        bbox_to_anchor=(0.5, phase_legend_y),
        handletextpad=0.35, columnspacing=1.05)
    fig.add_artist(phase_legend)

    # 第二、三行共用一个模型图例，只在整幅图最下方保留一次。
    fig.legend(handles=model_handles, loc='center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.045),
               handletextpad=0.45, columnspacing=1.6)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(out_stem.with_suffix(suffix), facecolor='white')
    plt.close(fig)
    print(f'图已保存 {out_stem.with_suffix(".pdf")} 与 {out_stem.with_suffix(".png")}')


def _bold_if_best(value: float, values: Sequence[float], fmt: str,
                  higher_better: bool, latex: bool = False) -> str:
    best = max(values) if higher_better else min(values)
    text = format(value, fmt)
    if not np.isclose(value, best):
        return text
    return rf'\textbf{{{text}}}' if latex else f'**{text}**'


def build_main_table(mean_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    block = mean_table[mean_table.aircraft_type == 'all']
    for target_key, target_label in TARGETS:
        target_block = block[block.target == target_key].set_index('model')
        for model_key, model_label in MODELS:
            if model_key not in target_block.index:
                raise RuntimeError(f'主表缺少 {target_key}/{model_key}')
            row = target_block.loc[model_key]
            rows.append({
                'Target': target_label,
                'Model': model_label,
                'R²': float(row.R2),
                'RMSE': float(row.RMSE),
                # 原始指标文件以百分数储存；论文表按要求改为小数。
                'MAPE': float(row.MAPE) / 100.0,
            })
    return pd.DataFrame(rows)


def write_main_table(mean_table: pd.DataFrame, out_stem: Path,
                     seeds: Sequence[int]) -> pd.DataFrame:
    table = build_main_table(mean_table)
    n_values = mean_table.loc[mean_table.aircraft_type == 'all', 'n_records'].round().astype(int)
    if n_values.nunique() != 1:
        raise RuntimeError(f'主表总体测试记录数不一致：{sorted(n_values.unique())}')
    n_records = int(n_values.iloc[0])
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_stem.with_suffix('.csv'), index=False, float_format='%.8f')

    title = f'Fleet-wide model comparison on the held-out test set (n = {n_records:,})'
    lines = [f'**{title}.**', '',
             '| Target | Model | R² | RMSE (kg s⁻¹) | MAPE |',
             '|---|---|---:|---:|---:|']
    latex_rows: List[str] = []
    for target_label in [label for _, label in TARGETS]:
        block = table[table.Target == target_label]
        r2_values = block['R²'].tolist()
        rmse_values = block.RMSE.tolist()
        mape_values = block.MAPE.tolist()
        for row in block.itertuples(index=False):
            md_cells = [
                _bold_if_best(row[2], r2_values, '.4f', True),
                _bold_if_best(row.RMSE, rmse_values, '.4f', False),
                _bold_if_best(row.MAPE, mape_values, '.4f', False),
            ]
            lines.append(f'| {row.Target} | {row.Model} | ' + ' | '.join(md_cells) + ' |')
            tex_cells = [
                _bold_if_best(row[2], r2_values, '.4f', True, latex=True),
                _bold_if_best(row.RMSE, rmse_values, '.4f', False, latex=True),
                _bold_if_best(row.MAPE, mape_values, '.4f', False, latex=True),
            ]
            latex_rows.append(f'{row.Target} & {row.Model} & '
                              + ' & '.join(tex_cells) + r' \\')
    seed_text = ', '.join(str(int(seed)) for seed in seeds)
    lines += ['', f'Values are means across seeds {seed_text}. MAPE is reported as a decimal.']
    markdown = '\n'.join(lines) + '\n'
    out_stem.with_suffix('.md').write_text(markdown, encoding='utf-8')

    latex = (
        '\\begin{table}[t]\n\\centering\n'
        f'\\caption{{{title}. Values are means across seeds {seed_text}; '
        'MAPE is reported as a decimal.}\n'
        '\\begin{tabular}{llrrr}\n\\toprule\n'
        'Target & Model & $R^2$ & RMSE (kg s$^{-1}$) & MAPE \\\\\n'
        '\\midrule\n' + '\n'.join(latex_rows)
        + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    )
    out_stem.with_suffix('.tex').write_text(latex, encoding='utf-8')
    print(f'主表已保存 {out_stem.with_suffix(".csv")} / .md / .tex')
    return table


def write_rmse_table(table: pd.DataFrame, group_column: str,
                     group_order: Sequence[str], group_labels: Sequence[str],
                     out_stem: Path, title: str, seeds: Sequence[int]) -> None:
    """把图中分组 RMSE 均值写成紧凑补充表，不加入任何区间。"""
    records: List[Dict[str, object]] = []
    for target_key, target_label in TARGETS:
        for group_key, group_label in zip(group_order, group_labels):
            block = table[(table.target == target_key)
                          & (table[group_column] == group_key)].set_index('model')
            if block.empty:
                raise RuntimeError(f'补充表缺少 {target_key}/{group_key}')
            record: Dict[str, object] = {
                'Target': target_label,
                'Group': group_label,
                'n': int(block.n_records.iloc[0]),
            }
            for model_key, model_label in MODELS:
                record[model_label] = float(block.loc[model_key, 'RMSE'])
            records.append(record)
    wide = pd.DataFrame(records)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_stem.with_suffix('.csv'), index=False, float_format='%.8f')

    lines = [f'**{title}.**', '',
             '| Target | Group | n | PINN | MLP | RBF | RF |',
             '|---|---|---:|---:|---:|---:|---:|']
    latex_rows: List[str] = []
    for row in wide.itertuples(index=False):
        values = [row.PINN, row.MLP, row.RBF, row.RF]
        md_values = [_bold_if_best(value, values, '.4f', False) for value in values]
        tex_values = [_bold_if_best(value, values, '.4f', False, latex=True)
                      for value in values]
        lines.append(f'| {row.Target} | {row.Group} | {row.n:,} | '
                     + ' | '.join(md_values) + ' |')
        latex_rows.append(f'{row.Target} & {row.Group} & {row.n:,} & '
                          + ' & '.join(tex_values) + r' \\')
    seed_text = ', '.join(str(int(seed)) for seed in seeds)
    lines += ['', f'RMSE values are means across seeds {seed_text} and are in kg s⁻¹.']
    out_stem.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    latex = (
        '\\begin{table}[t]\n\\centering\n'
        f'\\caption{{{title}. RMSE values are means across seeds {seed_text} '
        'and are in kg s$^{-1}$.}\n'
        '\\begin{tabular}{llrrrrr}\n\\toprule\n'
        'Target & Group & $n$ & PINN & MLP & RBF & RF \\\\\n'
        '\\midrule\n' + '\n'.join(latex_rows)
        + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    )
    out_stem.with_suffix('.tex').write_text(latex, encoding='utf-8')
    print(f'补充表已保存 {out_stem.with_suffix(".csv")} / .md / .tex')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='生成 fleet-wide 3×3 结果图与均值表')
    parser.add_argument('--seed', type=int, default=42, help='第一行散点使用的代表性种子')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                        help='柱图和表格取均值的种子')
    parser.add_argument('--point_size', type=float, default=1.4)
    parser.add_argument('--alpha', type=float, default=0.30)
    parser.add_argument('--per_phase_points', type=int, default=0,
                        help='每个阶段最多绘制的散点数；0 表示全部绘制')
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)

    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError('--seeds 至少需要一个种子')
    load_seeds = list(dict.fromkeys([int(args.seed), *seeds]))

    setup_style()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('明确指定了 CUDA，但当前 PyTorch 无法使用 CUDA')
    device = torch.device('cuda' if args.device != 'cpu'
                          and torch.cuda.is_available() else 'cpu')

    data = load_flight_table()
    _, _, test_frame, manifest = split_fleet_wide(
        data, 0.7, 0.1, 0.2, 42, verbose=False)
    dataset = FlightRecordDataset(test_frame, 'test').to(device)
    targets = dataset.targets.detach().cpu().numpy()
    phases = dataset.phases.detach().cpu().numpy()
    type_ids = dataset.aircraft_type_ids.detach().cpu().numpy()
    aircraft_types = np.asarray([STUDY_AIRCRAFT_TYPES[index] for index in type_ids])
    print(f'测试集 {len(dataset):,} 条记录 / {dataset.n_flights()} 个航班；设备 {device}')

    predictions = load_all_predictions(
        dataset, load_seeds, manifest, device, targets.shape[1])
    mean_table = build_mean_metrics(seeds)
    phase_rmse, aircraft_rmse = compute_rmse_breakdowns(
        predictions, seeds, targets, phases, aircraft_types)

    draw_figure(
        predictions[int(args.seed)]['PINN'], targets, phases,
        mean_scores(mean_table),
        phase_rmse, aircraft_rmse, args.out_dir / 'fig_fleet_wide',
        args.point_size, args.alpha, args.per_phase_points)
    write_main_table(mean_table, args.out_dir / 'table_fleet_wide', seeds)
    write_rmse_table(
        phase_rmse, 'phase', PHASE_NAMES,
        [PHASE_LABEL[name] for name in PHASE_NAMES],
        args.out_dir / 'table_fleet_wide_phase_rmse',
        'Phase-specific fleet-wide RMSE on the held-out test set', seeds)
    write_rmse_table(
        aircraft_rmse, 'aircraft_type', STUDY_AIRCRAFT_TYPES,
        STUDY_AIRCRAFT_TYPES,
        args.out_dir / 'table_fleet_wide_aircraft_rmse',
        'Aircraft-specific fleet-wide RMSE on the held-out test set', seeds)


if __name__ == '__main__':
    main()
