# -*- coding: utf-8 -*-
"""投稿用 Source Data 文件（Nature Communications）。

格式参照 Wang et al. 2024 NC 的源数据（41467_2024_48779_MOESM3_ESM.xlsx）：
每个图（面板结构不同时按面板）一个工作表，第一行为列名，其下为逐点长表，不加标题与格式。
数值取自各画图脚本使用的同一批结果文件，只做推理不做训练：

  Figure 3b / 3c           四个模型 × 五个种子的分阶段 / 分机型 RMSE
  Figure 4                 训练量实验的逐种子 RMSE、MAPE
  Figure 5                 B737 微调实验的每个点
  Table 2 / Table 3        主表、消融表的逐种子值
  Supplementary Figure 1   损失权重坐标搜索
  Supplementary Figure 2a/2c  taxi 诊断中的变异比与 R²
  Supplementary Figure 4   物理一致性的逐模型值
  Supplementary Figure 5/6 微调范围、训练时间与收敛轮数
  Supplementary Table 6    与 Figure 4 同一组逐种子值

QAR 数据不公开，逐记录的油耗不放入：Figure 2（油耗剖面）、Figure 3a（测试集逐记录预测值与
观测值）、Supplementary Figure 2b（滑行油耗）与 Supplementary Figure 3（逐秒油耗曲线）不给源数据。

MAPE 一律写成小数，与正文表格一致；燃油流量单位 kg s⁻¹。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import make_fleet_wide_figure as fleet
import make_taxi_regime_figure as taxi
from af_data import PHASE_NAMES, STUDY_AIRCRAFT_TYPES, FlightRecordDataset, load_flight_table, split_fleet_wide

FIG_DIR = Path('results/figures')
OUT = Path('results/source_data/Source_Data.xlsx')
SEEDS = (42, 43, 44, 45, 46)
FF = '(kg s-1)'
PHASE = {'takeoff': 'Take-off', 'climb': 'Climb', 'cruise': 'Cruise',
         'descent': 'Descent', 'approach': 'Approach', 'landing': 'Landing'}
MODEL = {'PINN': 'PINN', 'mlp': 'MLP', 'rbf': 'RBF', 'rf': 'RF'}
TARGET = {'FF1': 'Engine 1 fuel flow', 'FF2': 'Engine 2 fuel flow', 'FF_Total': 'Total fuel flow'}
CONDITION = {'full': 'Full model', 'no_phase': 'No phase subnetwork',
             'shared_phase': 'Shared subnetwork', 'no_aircraft': 'No aircraft characteristics',
             'no_physics': 'No physics losses'}


# ── 正文 ────────────────────────────────────────────────────────

def figure_3(device: torch.device) -> Dict[str, pd.DataFrame]:
    data = load_flight_table()
    _, _, test_frame, manifest = split_fleet_wide(data, 0.7, 0.1, 0.2, 42, verbose=False)
    dataset = FlightRecordDataset(test_frame, 'test').to(device)
    targets = dataset.targets.detach().cpu().numpy()
    phases = dataset.phases.detach().cpu().numpy()
    types = np.asarray([STUDY_AIRCRAFT_TYPES[i]
                        for i in dataset.aircraft_type_ids.detach().cpu().numpy()])
    predictions = fleet.load_all_predictions(dataset, SEEDS, manifest, device, targets.shape[1])

    rows_b: List[dict] = []
    rows_c: List[dict] = []
    for seed in SEEDS:
        for key, label in MODEL.items():
            for target_key, target_label in TARGET.items():
                observed, predicted = fleet.target_columns(predictions[seed][key], targets, target_key)
                squared = np.square(predicted - observed)
                for i, phase in enumerate(PHASE_NAMES):
                    mask = phases == i
                    rows_b.append({'Output': target_label, 'Flight phase': PHASE[phase],
                                   'Model': label, 'Seed': seed, 'Test records': int(mask.sum()),
                                   f'RMSE {FF}': float(np.sqrt(squared[mask].mean()))})
                for aircraft in STUDY_AIRCRAFT_TYPES:
                    mask = types == aircraft
                    rows_c.append({'Output': target_label, 'Aircraft type': aircraft,
                                   'Model': label, 'Seed': seed, 'Test records': int(mask.sum()),
                                   f'RMSE {FF}': float(np.sqrt(squared[mask].mean()))})
    return {'Figure 3b': pd.DataFrame(rows_b), 'Figure 3c': pd.DataFrame(rows_c)}


def figure_4() -> pd.DataFrame:
    runs = pd.read_csv(FIG_DIR / 'fig_flight_sampling_source_data.csv')
    return pd.DataFrame({
        'Training flights (total)': runs.n_train_flights,
        'Training records': runs.n_train_records,
        'Model': runs.condition.map({'full': 'With physics', 'no_physics': 'Without physics'}),
        'Seed': runs.seed,
        f'Test RMSE, total fuel flow {FF}': runs.RMSE,
        'Test MAPE, total fuel flow': runs.MAPE,
    })


def figure_5() -> pd.DataFrame:
    runs = pd.read_csv(FIG_DIR / 'fig_transfer_b737_source_data.csv')
    method = {'S0': 'Without target data (source model)', 'S1': 'Train from scratch',
              'L3': 'Fine-tuning', 'L3_ablation': 'Fine-tuned from model without physics'}
    parts = []
    # a, b：不用目标数据、从零训练、微调；c, d：从 PINN 与从无物理模型出发的微调
    for panels, arms in ((('a', 'b'), ('S0', 'S1', 'L3')), (('c', 'd'), ('L3', 'L3_ablation'))):
        block = runs[runs.arm.isin(arms)].copy()
        block['Panel'] = np.where(block.target == 'B737-800', panels[0], panels[1])
        if panels[0] == 'c':
            block['Method'] = block.arm.map({'L3': 'Fine-tuned from PINN',
                                             'L3_ablation': method['L3_ablation']})
            block = block[block.n_flights.isin(runs.loc[runs.arm == 'L3_ablation', 'n_flights'])]
        else:
            block['Method'] = block.arm.map(method)
        parts.append(block)
    runs = pd.concat(parts, ignore_index=True)
    runs['Draw'] = np.where(runs.arm == 'S0', np.nan, runs.draw + 1)
    runs['Source model'] = np.where(runs.arm == 'S0', runs.draw + 1, np.nan)
    return pd.DataFrame({
        'Panel': runs.Panel, 'Target type': runs.target, 'Method': runs.Method,
        'Target-type training flights': runs.n_flights,
        'Random draw of training flights': runs.Draw, 'Source model (seed index)': runs['Source model'],
        f'Test RMSE, total fuel flow {FF}': runs.test_RMSE,
    }).sort_values(['Panel', 'Method', 'Target-type training flights',
                    'Random draw of training flights']).reset_index(drop=True)


def table_2() -> pd.DataFrame:
    pinn = pd.read_csv(fleet.PINN_DIR / 'per_seed_metrics.csv').assign(model='PINN')
    base = pd.read_csv(fleet.BASELINE_DIR / 'baseline_metrics.csv')
    rows = pd.concat([pinn, base], ignore_index=True)
    rows = rows[(rows.partition == 'test') & (rows.aircraft_type == 'all') & rows.seed.isin(SEEDS)]
    rows = rows.assign(order=rows.target.map(list(TARGET).index) * 10
                       + rows.model.map(list(MODEL).index)).sort_values(['order', 'seed'])
    return pd.DataFrame({
        'Output': rows.target.map(TARGET), 'Model': rows.model.map(MODEL), 'Seed': rows.seed,
        'Test records': rows.n_records, 'R2': rows.R2, f'RMSE {FF}': rows.RMSE,
        'MAPE': rows.MAPE / 100,
    }).reset_index(drop=True)


def table_3() -> pd.DataFrame:
    rows = pd.read_csv('results/ablation/ablation_metrics.csv')
    rows = rows[(rows.split == 'fleet_wide') & (rows.partition == 'test')
                & (rows.aircraft_type == 'all') & (rows.target == 'FF_Total')]
    rows = rows.assign(order=rows.condition.map(list(CONDITION).index)).sort_values(['order', 'seed'])
    return pd.DataFrame({
        'Condition': rows.condition.map(CONDITION), 'Seed': rows.seed, 'R2': rows.R2,
        f'RMSE, total fuel flow {FF}': rows.RMSE, 'MAPE, total fuel flow': rows.MAPE / 100,
    }).reset_index(drop=True)


# ── 补充材料 ────────────────────────────────────────────────────

def supplementary_figure_1() -> pd.DataFrame:
    import make_loss_weight_search_figure as search
    results = search.load_results(search.METRICS_CSV).set_index('config')
    rows = []
    for (title, _, _, _, scan), panel in zip(search.SCANS, 'abc'):
        for value, config in scan:
            row = results.loc[config]
            rows.append({'Panel': panel, 'Varied weight': title, 'Configuration': config,
                         'lambda_mass': row.w_mass, 'lambda_force': row.w_force,
                         'lambda_phase': row.w_phase,
                         'R2, total fuel flow': row.R2, f'RMSE, total fuel flow {FF}': row.RMSE,
                         'MAPE, total fuel flow': row.MAPE / 100})
    return pd.DataFrame(rows)


def supplementary_figure_2() -> Dict[str, pd.DataFrame]:
    data = taxi.load()
    ground, air = data[data.ground], data[~data.ground]
    panel_a = pd.DataFrame({
        'Flight-state variable': taxi.LABELS,
        'Standard deviation during taxi': [ground[c].std() for c in taxi.FEATURES],
        'Standard deviation in six retained phases': [air[c].std() for c in taxi.FEATURES],
    })
    panel_a['Relative variability during taxi'] = (
        panel_a['Standard deviation during taxi'] / panel_a['Standard deviation in six retained phases'])
    panel_c = pd.DataFrame({
        'Records': ['Taxi', 'Taxi', 'Six retained phases'],
        'Regressors': ['Seven state variables', 'Seven state variables + operating engine count',
                       'Seven state variables'],
        'R2 (diagnostic linear regression)': [taxi.explained(ground),
                                              taxi.explained(ground, 'n_engines'),
                                              taxi.explained(air)],
    })
    return {'Supplementary Figure 2a': panel_a, 'Supplementary Figure 2c': panel_c}


def supplementary_figure_4() -> pd.DataFrame:
    runs = pd.read_csv(FIG_DIR / 'supp_fig4_physical_consistency_source.csv')
    return pd.DataFrame({
        'Model': runs.config.map({'full': 'Full model', 'no_physics': 'No physics losses', 'mlp': 'MLP'}),
        'Seed': runs.seed,
        'a Mass-closure windows': runs.n_windows,
        'a Median absolute closure error (%)': runs.closure_median_abs_pct,
        'b Wrong-signed response to flight-path angle (% of gated records)': runs.wrong_sign_gamma_pct,
        'c Wrong-signed response to aircraft mass (% of gated records)': runs.wrong_sign_mass_pct,
        'b Gated records': runs.gate_L_records,
        'c Gated records': runs.gate_L_and_m_records,
        'd Phase transitions': runs.n_transitions,
        'd Median relative jump at phase transitions (%)': runs.jump_median_pct,
    })


def supplementary_figure_5() -> pd.DataFrame:
    runs = pd.read_csv(FIG_DIR / 'fig_transfer_b737_scope_source_data.csv')
    scope = {'L1': 'Aircraft-characteristic encoder',
             'L2': 'Aircraft-characteristic encoder + output layer',
             'L3': 'Aircraft-characteristic encoder + fusion and output layers'}
    return pd.DataFrame({
        'Panel': np.where(runs.target == 'B737-800', 'a', 'b'), 'Target type': runs.target,
        'Fine-tuned layers': runs.arm.map(scope), 'Target-type training flights': runs.n_flights,
        'Random draw of training flights': runs.draw + 1,
        f'Test RMSE, total fuel flow {FF}': runs.test_RMSE,
    }).sort_values(['Panel', 'Fine-tuned layers', 'Target-type training flights',
                    'Random draw of training flights']).reset_index(drop=True)


def supplementary_figure_6() -> pd.DataFrame:
    runs = pd.read_csv(FIG_DIR / 'fig_transfer_b737_efficiency_source_data.csv')
    is_800 = runs.target == 'B737-800'
    return pd.DataFrame({
        'Panel (training time / best epoch)': np.where(is_800, 'a / c', 'b / d'),
        'Target type': runs.target,
        'Method': runs.arm.map({'S1': 'Train from scratch', 'L3': 'Fine-tuning'}),
        'Target-type training flights': runs.n_flights,
        'Random draw of training flights': runs.draw + 1,
        'Training time (s)': runs.train_seconds,
        'Epoch with lowest validation error': runs.best_epoch,
    }).sort_values(['Target type', 'Method', 'Target-type training flights',
                    'Random draw of training flights'], ascending=[False, True, True, True]
                   ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='生成投稿用 Source Data 文件')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')

    sheets: Dict[str, pd.DataFrame] = dict(figure_3(device))
    sheets['Figure 4'] = figure_4()
    sheets['Figure 5'] = figure_5()
    sheets['Table 2'] = table_2()
    sheets['Table 3'] = table_3()
    sheets['Supplementary Figure 1'] = supplementary_figure_1()
    sheets.update(supplementary_figure_2())
    sheets['Supplementary Figure 4'] = supplementary_figure_4()
    sheets['Supplementary Figure 5'] = supplementary_figure_5()
    sheets['Supplementary Figure 6'] = supplementary_figure_6()
    sheets['Supplementary Table 6'] = figure_4()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            print(f'{name:<26} {len(frame):>8,} rows × {frame.shape[1]} columns')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
