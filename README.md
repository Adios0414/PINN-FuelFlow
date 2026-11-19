# PINN4FuelFlow
This code is for our paper: *Physics-informed learning of aircraft fuel flow across aircraft types from few flights* (submitted to *Nature Communications*).


# 1. System requirements
python version: 3.10.9

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  2.3.0   |
|  scikit-learn  |  1.2.1   |
|     numpy      |  1.26.4  |
|     pandas     |  1.5.3   |
|     scipy      |  1.10.0  |
|   matplotlib   |  3.7.0   |
|    openpyxl    |  3.0.10  |

The code was tested on Windows 10 with an NVIDIA GeForce RTX 4090 GPU.


# 2. Installation guide
If you are not familiar with Python and Pytorch framework,
you can install Anaconda first and use Anaconda to quickly configure the environment.
## 2.1 Create environment
```
conda create -n new_environment python=3.10.9
```

## 2.2 Activate environment
```
conda activate new_environment
```

## 2.3 Install dependencies
```
conda install pytorch=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn=1.2.1 numpy=1.26.4 pandas=1.5.3 scipy=1.10.0 matplotlib=3.7.0 openpyxl=3.0.10
```


# 3. Demo
All programs are run from this folder. The QAR data should be placed in the `data` folder first (see Section 4).
1. Run `train_fleet_wide.py` to train our model on the A32x fleet. The program saves the results in `results/fleet_wide/all_flights_full`.
   ```
   python train_fleet_wide.py --seeds 42 43 44 45 46
   ```
2. Run `train_fleet_wide.py` with `--no_physics` to train the same model without the physics losses. The results are saved in `results/fleet_wide/all_flights_no_physics`.
   ```
   python train_fleet_wide.py --seeds 42 43 44 45 46 --no_physics
   ```
3. Run `train_baselines.py` to train the MLP, RBF and RF baselines. The results are saved in `results/baselines`.
   ```
   python train_baselines.py --split fleet_wide --models mlp rbf rf --seeds 42 43 44 45 46
   ```
4. Run the other experiments. Steps 1 and 2 must be completed first.
   ```
   python run_ablation.py --splits fleet_wide --seeds 42 43 44 45 46
   python run_flight_sampling.py --levels 1 2 4 8 0 --conditions full no_physics --seeds 42 43 44 45 46
   python search_loss_weights.py
   python train_transfer.py split --force
   python train_transfer.py run
   python train_transfer.py summarize
   python physical_consistency_eval.py
   ```
   These correspond to the component ablation (Table 3), the number of training flights (Fig. 4), the selection of the physics-loss weights (Supplementary Fig. 1), the fine-tuning to the B737-800 and B737-700 (Fig. 5) and the physical-consistency evaluation (Supplementary Fig. 4).
5. Run `reproduce_figures.py` to draw all figures and tables of our manuscript. The figures and tables are saved in `results/figures`.
   ```
   python reproduce_figures.py
   ```

We also provide the trained models and the results of our training,
which are saved in the `results` folder.
These results correspond exactly to the data in our manuscript.
Without the QAR data, `reproduce_figures.py` draws the figures and tables that only use these results
(Table 3, Figs. 4 and 5, Supplementary Figs. 1, 4, 5 and 6, and Supplementary Table 6).


# 4. Additional information
The QAR data of 153 flights operated by China Eastern Airlines are proprietary and cannot be shared publicly.
Aggregated, anonymized datasets are available from the corresponding author to qualified researchers upon reasonable request, subject to a data-use agreement.
The programs read the data from `data/qar_a32x.csv`, `data/qar_b737-800.csv` and `data/qar_b737-700.csv`.

The source data of the figures and tables in our manuscript are provided in `data/Source_Data.xlsx`.

The code for **preprocessing** the raw QAR data is provided in the `preprocessing` folder.

The random-forest models (1.7 GB each), the models of the individual fine-tuning runs and the models of the weight search are not included because of their size. Their predictions and results are included in the `results` folder.
