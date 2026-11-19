# -*- coding: utf-8 -*-
"""模块消融的对比表

消融只有几个标量，画图不如列表，所以只出表：CSV、Markdown、LaTeX 三份。

R²、RMSE、MAPE 三个都按小数写，与参考文献的表一致。ΔRMSE 是相对完整模型的相对
变化，是比较量不是误差量，所以用百分数。

数据取自 run_ablation.py 汇总出来的 CSV，不重新推理。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from run_ablation import SPLITS

OUT_DIR = Path('results/figures')
SUMMARY_DIR = Path('results/ablation')

# 条件 → 英文名，顺序即表中行序
ROWS: List[Tuple[str, str]] = [
    ('full', 'Full model'),
    ('no_phase', 'No phase subnetwork'),
    ('shared_phase', 'Shared subnetwork, no phase routing'),
    ('no_aircraft', 'No aircraft features'),
    ('no_physics', 'No physics losses'),
]
SPLIT_LABEL = {'fleet_wide': 'Fleet-wide', 'zero_shot': 'Zero-shot (A321-NEO)'}


def build(summary_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    metrics = pd.read_csv(summary_dir / 'ablation_metrics.csv')
    rows = []
    for split in splits:
        partition = SPLITS[split][1]
        block = metrics[(metrics['split'] == split)
                        & (metrics.partition == partition)
                        & (metrics.aircraft_type == 'all')
                        & (metrics.target == 'FF_Total')]
        if block.empty:
            continue
        reference = block[block.condition == 'full'].RMSE.mean()
        for condition, label in ROWS:
            part = block[block.condition == condition]
            if part.empty:
                continue
            rows.append({
                'Split': SPLIT_LABEL[split], 'Condition': label,
                'Seeds': part.seed.nunique(),
                'R2': part.R2.mean(), 'R2_sd': part.R2.std(ddof=1),
                'RMSE': part.RMSE.mean(), 'RMSE_sd': part.RMSE.std(ddof=1),
                'MAPE': part.MAPE.mean(), 'MAPE_sd': part.MAPE.std(ddof=1),
                'RMSE_change_pct': (part.RMSE.mean() - reference) / reference * 100,
            })
    if not rows:
        raise RuntimeError('没有找到任何消融结果，先跑 run_ablation.py')
    return pd.DataFrame(rows)


def write(table: pd.DataFrame, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_stem.with_suffix('.csv'), index=False)

    columns = ['Condition', 'R²', 'RMSE', 'MAPE', 'ΔRMSE']
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    latex = []
    for split in table.Split.unique():
        block = table[table.Split == split]
        lines.append(f'| **{split}** | | | | |')
        latex.append(r'\multicolumn{5}{l}{\textit{' + split + r'}} \\')
        for record in block.to_dict('records'):
            change = ('—' if record['Condition'] == 'Full model'
                      else f'{record["RMSE_change_pct"]:+.1f}%')
            cells = [record['Condition'], f'{record["R2"]:.4f}',
                     f'{record["RMSE"]:.4f}', f'{record["MAPE"] / 100:.4f}', change]
            lines.append('| ' + ' | '.join(cells) + ' |')
            latex.append(' & '.join(c.replace('%', r'\%') for c in cells) + r' \\')

    markdown = '\n'.join(lines)
    with open(out_stem.with_suffix('.md'), 'w', encoding='utf-8') as handle:
        handle.write(markdown + '\n')
    header = ('\\begin{tabular}{lrrrr}\n\\hline\n'
              'Condition & $R^2$ & RMSE & MAPE & $\\Delta$RMSE \\\\\n'
              '\\hline\n')
    with open(out_stem.with_suffix('.tex'), 'w', encoding='utf-8') as handle:
        handle.write(header + '\n'.join(latex) + '\n\\hline\n\\end{tabular}\n')
    print(f'表已保存 {out_stem.with_suffix(".csv")} / .md / .tex')
    print()
    print(markdown)
    print()
    worst = table.loc[table.R2_sd.idxmax()]
    # 种子数按实际结果报，不写死
    n_seeds = int(table.Seeds.max())
    print(f'{n_seeds} 个种子间 R² 波动最大的一行：{worst.Split} / {worst.Condition}，'
          f'标准差 {worst.R2_sd:.4f}')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='模块消融对比表')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS),
                        choices=list(SPLITS))
    parser.add_argument('--summary_dir', type=Path, default=SUMMARY_DIR)
    parser.add_argument('--out_dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    write(build(args.summary_dir, args.splits), args.out_dir / 'table_ablation')


if __name__ == '__main__':
    main()
