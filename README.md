# Machine-learning-prediction-of-binary-formation-in-three-body-gravitational-encounters

This repository contains the code used for the machine learning analysis presented in the paper:

> **Machine-learning-prediction-of-binary-formation-in-three-body-gravitational-encounters**  
> Ahmad Farhani Asl

The code is released under the **Apache License 2.0**. 
Please find the simulation data used in this work at: https://zenodo.org/records/21458387  and the trained model at: https://zenodo.org/records/21462557

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `gen-data.py` | Generates initial conditions for three-body scattering experiments, integrates each system using REBOUND's IAS15 integrator, computes encounter features (masses, velocities, impact parameters, angles), and saves the dataset to CSV files. |
| `train-v2.py` | Loads the generated data, cleans it (filters unphysical binaries and high energy errors), builds 33 physically motivated features, balances the dataset, trains an XGBoost classifier with GPU acceleration, evaluates performance using multiple metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, ECE), and saves the trained model along with all evaluation results. |

---

## 🔧 Requirements

The code requires Python 3.8 or newer. Dependencies are listed below:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `rebound` (for N-body integration)
- `tqdm` (for progress bars)

You can install all dependencies using:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn rebound tqdm
```

---

## 🚀 Usage

### 1. Generate Data

Run the data generator to produce a synthetic dataset of three-body encounters:

```bash
python gen-data.py
```

**Configuration** (edit variables at the top of `gen-data.py`):

| Variable | Description | Default |
|----------|-------------|---------|
| `n_packs` | Number of data packs to generate | 1 |
| `n_sim_per_pack` | Number of simulations per pack | 10,000 |
| `mode` | `1` = test run, `2` = production | 1 |
| `mass` | Mass range in M☉ | `[0.08, 150]` |
| `velocity` | Velocity range in au/yr | `[0.01, 100.0]` |
| `enc_r` | Encounter radius range in au | `[0.01, 10000]` |
| `H` | Hardness range (PE/KE) | `[0.0, 1.0]` |

The script will:
- Generate random initial conditions for three-body encounters.
- Integrate each system using REBOUND's IAS15 integrator.
- Compute encounter features (masses, velocities, impact parameters, angles).
- Save the dataset as `data/dataset_XXX.csv`.

**Output location:** `data/dataset_XXX.csv`

### 2. Train the Model

Once the data is generated, train the XGBoost classifier:

```bash
python train-v2.py
```

**Configuration** (edit variables at the top of `train-v2.py`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_PATH` | Path pattern for input CSV files | `"./data/split/dataset_*.csv"` |
| `TEST_SIZE` | Fraction of data for testing | 0.2 |
| `CV_FOLDS` | Number of cross-validation folds | 10 |
| `ENERGY_ERR_CUT` | Maximum energy error for valid simulations | 1e-8 |

The script will:
- Load and clean the data (filter unphysical binaries and high energy errors).
- Build 33 physically motivated features.
- Balance the dataset (downsample the majority class).
- Train an XGBoost model with GPU acceleration (falls back to CPU if unavailable).
- Evaluate performance on the test set.

---

## 📊 Outputs

After running `train-v2.py`, the following files will be produced in the `results/` directory:

| File | Description |
|------|-------------|
| `xgboost_model.json` | Trained XGBoost model (JSON format, recommended). |
| `xgboost_model.ubj` | Trained XGBoost model (binary UBJ format for faster loading). |
| `test_metrics.csv` | Summary of evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, ECE). |
| `cv_results.csv` | Cross-validation PR-AUC mean and standard deviation. |
| `confusion_matrix.csv` | Confusion matrix (TN, FP, FN, TP). |
| `feature_importance.csv` | Feature importance scores. |
| `calibration_data.csv` | Data for calibration curve plotting. |
| `prediction_probabilities.csv` | True labels, predicted labels, and probabilities. |
| `r_enc_performance.csv` | Performance breakdown by encounter radius. |
| `hardness_performance.csv` | Performance breakdown by hardness. |
| `baseline_comparison.csv` | Physical baseline comparison results. |

---

## 📈 Feature Set

The model uses **33 features** derived from the initial conditions of each three-body encounter, including:

- **Mass features**: log masses, mass fractions, mass entropy, hierarchy, asymmetry
- **Velocity features**: velocities, escape velocity, velocity ratios, hierarchy, asymmetry
- **Impact parameter features**: impact parameters and ratios
- **Energy features**: hardness (PE/KE)
- **Angular features**: angular momentum ratio, directional cosines of velocity angles
- **Virial features**: virial radius cut-off

All features are log-transformed where appropriate to improve model performance.

---

## 📦 Model Availability

https://zenodo.org/records/21462557

---

## 📝 Documentation

Full documentation, including feature descriptions, hyperparameter tuning, and performance analysis, is provided in the accompanying paper; see https://arxiv.org/abs/2607.16776

---

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or collaboration, please contact: a.farhaniasl@gmail.com

---

## 📖 Citation

If you use this code in your research, please cite the associated paper:

```bibtex
@misc{asl2026machinelearningpredictionbinary,
      title={Machine learning prediction of binary formation in three-body gravitational encounters}, 
      author={Ahmad Farhani Asl},
      year={2026},
      eprint={2607.16776},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA},
      url={https://arxiv.org/abs/2607.16776}, 
}
```

Also, please acknowledge the use of this code and the following software libraries:

```bibtex
@article{pedregosa2011scikit,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}

@article{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  journal={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016}
}

@article{reinhardt2020rebound,
  title={REBOUND: A high-performance N-body code for collisional dynamics},
  author={Rein, Hanno and Liu, Shang-Fei},
  journal={Astronomy \& Astrophysics},
  volume={537},
  pages={A128},
  year={2012}
}
```
