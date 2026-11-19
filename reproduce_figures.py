# -*- coding: utf-8 -*-
"""Reproduce the figures and tables from the provided results.

Uses the trained models and result files in results/; no model is trained.
Run from this folder:  python reproduce_figures.py

The QAR data are not distributed (see README). Without the data files in data/, the
figures and tables that are drawn from the result files alone are reproduced and the
others are skipped. With the data files in data/, everything is reproduced, including
the Source Data file. Outputs are written to results/figures/ and results/source_data/.
"""

import subprocess
import sys
import time
from pathlib import Path

DATA_FILES = [Path('data/qar_a32x.csv'), Path('data/qar_b737-800.csv'), Path('data/qar_b737-700.csv')]

# (command, output, needs the QAR data)
STEPS = [
    (['physical_consistency_eval.py'], 'Supplementary Fig. 4 data (inference on the 19 test flights)', True),
    (['make_fuel_flow_distribution_figure.py'], 'Fig. 2', True),
    (['make_fleet_wide_figure.py'], 'Fig. 3 and Table 2', True),
    (['make_ablation_table.py', '--splits', 'fleet_wide'], 'Table 3', False),
    (['make_sampling_figure.py'], 'Fig. 4 and Supplementary Table 6', False),
    (['make_transfer_b737_figure.py'], 'Fig. 5 and Supplementary Fig. 5', False),
    (['make_loss_weight_search_figure.py'], 'Supplementary Fig. 1', False),
    (['make_taxi_regime_figure.py'], 'Supplementary Fig. 2', True),
    (['make_fleet_wide_flights_figure.py'], 'Supplementary Fig. 3', True),
    (['make_physical_consistency_figure.py'], 'Supplementary Fig. 4', False),
    (['make_transfer_b737_efficiency_figure.py'], 'Supplementary Fig. 6', False),
    (['build_source_data.py'], 'Source Data file', True),
]


def main() -> None:
    has_data = all(path.exists() for path in DATA_FILES)
    if not has_data:
        print('QAR data files not found in data/: only the outputs drawn from the result '
              'files are reproduced (see README, Section 6).')
    skipped = []
    for command, label, needs_data in STEPS:
        if needs_data and not has_data:
            skipped.append(label)
            continue
        print(f'\n=== {label}: python {" ".join(command)}', flush=True)
        start = time.time()
        subprocess.run([sys.executable] + command, check=True)
        print(f'=== done in {time.time() - start:.0f} s', flush=True)
    print('\nOutputs are in results/figures/' + ('' if not has_data else ' and results/source_data/') + '.')
    if skipped:
        print('Skipped because they require the QAR data: ' + '; '.join(skipped) + '.')


if __name__ == '__main__':
    main()
