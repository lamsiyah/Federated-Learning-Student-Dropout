# Privacy-Preserving Federated Learning for Student Dropout Prediction

## Overview
This repository contains the implementation and experimental results for our research on **Privacy-Preserving Federated Learning for Student Dropout Prediction**, integrating post-hoc Explainable AI (XAI) methods to enhance model transparency. We compare centralized deep learning baselines with federated approaches (FedAvg and FedProx) and apply LIME, SHAP, and Integrated Gradients for interpretability.

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Centralized Baseline](#centralized-baseline)
  - [Federated Learning](#federated-learning)
  - [Explainable AI Methods](#explainable-ai-methods)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Features
- **Centralized Learning Baseline**: Feed-forward neural network with weighted cross-entropy.
- **Federated Learning Methods**: Implementations of FedAvg and FedProx to preserve data privacy across multiple clients.
- **Explainable AI**: Post-hoc interpretation using LIME, SHAP (Gradient SHAP), and Integrated Gradients.
- **Datasets**: Experiments on OULAD, KDD Cup 2015, and UIC student dropout datasets.
- **Performance Metrics**: Accuracy, F1-score, and AUC comparisons across methods.

## Repository Structure
```
├── data/                     # Raw and processed datasets
│   ├── OULAD/                # Open University Learning Analytics Dataset
│   ├── KDDCup2015/           # KDD Cup 2015 MOOC data
│   └── UIC/                  # UIC student dropout dataset
├── src/                      # Source code
│   ├── centralized/          # Centralized training scripts
│   ├── federated/            # Federated learning implementations (FedAvg, FedProx)
│   └── xai/                  # Explainability modules (LIME, SHAP, IG)
├── notebooks/                # Jupyter notebooks for analysis and visualizations
├── results/                  # Trained models, logs, and evaluation outputs
├── figures/                  # Pipeline diagrams and XAI plots
├── requirements.txt          # Python dependencies
├── setup.py                  # Package install script
└── README.md                 # This file
```

## Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- captum
- pandas, numpy
- matplotlib
- seaborn (optional for plots)
- tqdm

Install via:
```bash
pip install -r requirements.txt
```

## Data Preparation
1. **Download** the datasets:
   - OULAD: [Kuzilek et al. (2017)](https://analyse.kmi.open.ac.uk/open_dataset)
   - KDD Cup 2015: [Feng et al. (2019)](http://www.kdd.org/kdd-cup/view/kdd-cup-2015)
   - UCI: (https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
2. **Place** raw CSV files under `data/<dataset_name>/raw/`.
3. **Preprocess** and split:
   ```bash
   python src/data_preprocessing.py --dataset OULAD
   python src/data_preprocessing.py --dataset KDDCup2015
   python src/data_preprocessing.py --dataset UIC
   ```
Processed features and train/test splits will be saved to `data/<dataset_name>/processed/`.

## Usage
### Centralized Baseline
Train and evaluate the feed-forward baseline:
```bash
python src/centralized/train.py \
  --data_dir data/OULAD/processed \
  --output_dir results/centralized/OULAD \
  --epochs 50 --lr 0.02 --batch_size 64
```
Metrics (accuracy, F1, AUC) are logged to `results/centralized/<dataset>/metrics.json`.

### Federated Learning
Run FedAvg:
```bash
python src/federated/fedavg.py \
  --data_dir data/OULAD/processed \
  --clients 100 --rounds 50 \
  --local_epochs 2 --lr 0.02 --batch_size 64
```
Run FedProx (with μ=0.01):
```bash
python src/federated/fedprox.py \
  --data_dir data/OULAD/processed \
  --clients 100 --rounds 50 --mu 0.01 \
  --local_epochs 2 --lr 0.02 --batch_size 64
```
Results are saved under `results/federated/<method>/<dataset>/`.

### Explainable AI Methods
Generate XAI attributions for a trained model:
```bash
python src/xai/explain.py \
  --model_path results/federated/fedprox/OULAD/model.pt \
  --method shap --dataset OULAD --sample_id 10
```
Output plots saved to `figures/xai/<method>/<dataset>/sample_<id>.png`.

## Results
- **Centralized vs Federated**: FL matches or slightly outperforms centralized baselines in accuracy and F1 while preserving data privacy.
- **FedProx vs FedAvg**: FedProx shows marginal improvements in convergence stability and predictive accuracy.
- **XAI Insights**: Important features (e.g., assessment scores, VLE engagement) identified via SHAP, LIME, and IG.

Detailed tables and figures are available in the [paper](./paper/PrivacyFL_DropoutPrediction.pdf) and `results/` directory.

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{lamsiyah2025privacy,
  title={Privacy-Preserving Federated Learning for Student Dropout Prediction: Enhancing Model Transparency with Explainable AI},
  author={Lamsiyah, S. and El Mahdaouy, A. and Nourbakhsh, A. and Schommer, C.},
  booktitle={Proceedings of ...},
  year={2025}
}
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

