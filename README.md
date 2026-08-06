# Privacy-Preserving Federated Learning for Student Dropout Prediction

Reproducibility code for:

> Salima Lamsiyah, Aria Nourbakhsh, Samir El-Amrany, and Christoph Schommer. **Privacy-Preserving Federated Learning for Student Dropout Prediction: Enhancing Model Transparency with Explainable AI.** *Artificial Intelligence in Education*, pp. 324-332, Springer Nature Switzerland, 2025. [https://doi.org/10.1007/978-3-031-98465-5_41](https://doi.org/10.1007/978-3-031-98465-5_41)

The study compares a centralized feed-forward neural network with Federated Averaging (FedAvg) and Federated Proximal (FedProx), then explains trained models with LIME, Integrated Gradients, and Gradient SHAP. Experiments cover the UCI student-dropout dataset, the Open University Learning Analytics Dataset (OULAD), and KDD Cup 2015.

## Reproduction notebooks

| Notebook | Dataset and acquisition | Prediction task |
| --- | --- | --- |
| [`01_uci_student_dropout.ipynb`](notebooks/01_uci_student_dropout.ipynb) | [UCI dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success); downloaded automatically from the official UCI archive | Graduate vs. Dropout after removing Enrolled |
| [`02_oulad.ipynb`](notebooks/02_oulad.ipynb) | [Official OULAD page](https://analyse.kmi.open.ac.uk/open_dataset) and [UCI mirror](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset); downloaded automatically | Pass/Fail, Fail/Distinction, Pass/Distinction, and Pass/Withdrawn |
| [`03_kdd_cup_2015.ipynb`](notebooks/03_kdd_cup_2015.ipynb) | [Official competition](https://www.biendata.xyz/competition/kddcup2015/), [ACM KDD description](https://www.kdd.org/kdd2015/calls.html), and [Kaggle mirror](https://www.kaggle.com/datasets/sst2023/kdd-cup-2015) | MOOC dropout vs. completion |

Each notebook contains dataset acquisition instructions, raw-data checks, feature engineering, leakage-safe preprocessing, centralized training, FedAvg, FedProx, evaluation, comparison with the paper, and the three XAI methods. Raw datasets and trained models are deliberately not committed.

## Quick start

```bash
git clone https://github.com/lamsiyah/Federated-Learning-Student-Dropout.git
cd Federated-Learning-Student-Dropout
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

By default, notebooks use a small smoke-test configuration so that the complete pipeline can be checked quickly. For the paper configuration, launch Jupyter with:

```bash
AIED_PAPER_MODE=1 jupyter lab
```

Full paper-mode runs are computationally expensive: they repeat centralized, FedAvg, and FedProx training ten times; OULAD evaluates four binary tasks; and OULAD/KDD use 100 simulated clients. A CUDA device is used automatically when available. The notebooks write generated data under `data/` and metrics under `results/`; both are ignored by Git.

### KDD Cup 2015 data

The competition data are not redistributed. Download and unzip the Kaggle mirror, then place or retain these files anywhere below `data/kddcup2015/raw/`:

```text
date.csv
enrollment_train.csv
enrollment_test.csv
log_train.csv
log_test.csv
truth_train.csv
truth_test.csv
```

The KDD notebook searches recursively, so the mirror's `train/` and `test/` folders do not have to be flattened. Alternatively, set `AIED_KDD_KAGGLE_DOWNLOAD=1` to let `kagglehub` obtain the public mirror. Kaggle authentication or acceptance of the dataset terms may be required.

## Experimental configuration

The paper-mode configuration follows the published long and short manuscripts:

| Setting | Value |
| --- | --- |
| Model | Fully connected network: input - 30 - 10 - 2, ReLU activations |
| Loss | Inverse-frequency weighted cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.02 |
| Batch size | 64 |
| Train/test split | Stratified 80/20 |
| Centralized epochs | 100 |
| FL communication rounds | 50 |
| Local epochs per round | 2 |
| Clients | 100 for OULAD/KDD; 10 for UCI |
| FedProx coefficient | μ = 0.01 |
| Repetitions | 10, using split seeds 0-9 |

The simulated setting is horizontal cross-silo FL: clients have disjoint rows and a common feature space. FedAvg aggregates client parameters in proportion to client sample counts. FedProx uses the same aggregation and adds `(μ/2) ||w_local - w_global||²` to each local objective.

## Published results

These are the mean values reported in the paper. OULAD uses weighted F1, KDD uses binary F1 for the dropout class, and UCI uses macro F1. A dash means that AUC was not reported.

| Dataset/task | Central Acc/F1/AUC | FedAvg Acc/F1/AUC | FedProx Acc/F1/AUC |
| --- | --- | --- | --- |
| OULAD Pass/Fail | .827 / .822 / - | .811 / .813 / - | .830 / .827 / - |
| OULAD Fail/Distinction | .849 / .848 / - | .852 / .855 / - | .862 / .861 / - |
| OULAD Pass/Distinction | .803 / .716 / - | .716 / .730 / - | .804 / .756 / - |
| OULAD Pass/Withdrawn | .903 / .924 / - | .892 / .893 / - | .892 / .892 / - |
| KDD Cup 2015 | .872 / .922 / .875 | .876 / .925 / .879 | .887 / .932 / .887 |
| UCI | .899 / .865 / .890 | .904 / .896 / .941 | .914 / .906 / .951 |

Exact reruns can vary slightly across PyTorch versions and hardware even with fixed seeds.

## Reproducibility notes and corrections

The research archive and the cited AIED 2024 baseline were audited while preparing these notebooks. The publication-ready version makes the following points explicit:

- The original OULAD experiment began from precomputed feature CSVs. The complete raw OULAD feature-engineering pipeline is now included, based on the cited [Federated-Learning-Analytics](https://github.com/MaxvanHaastrecht/Federated-Learning-Analytics) implementation.
- Some archived cells used 20 rounds/3 local epochs for UCI and μ=0.1 for OULAD/KDD, although the paper reports 50 rounds/2 local epochs and μ=0.01. Paper mode uses the published settings.
- Some archived cells computed class weights but later replaced the weighted loss with ordinary cross-entropy. These notebooks consistently use inverse-frequency weighted cross-entropy.
- Scaling in the archived notebooks was fitted before the train/test split. These notebooks fit `StandardScaler` on training rows only and transform the test rows afterward, avoiding test-set leakage. This correction can cause small differences from the reported values.
- Client partitions, model initialization, data-loader shuffling, and train/test splits now receive explicit seeds.

## Data, privacy, and responsible use

- [UCI dataset 697](https://doi.org/10.24432/C5MC89) and [OULAD](https://doi.org/10.24432/C5KK69) are available under CC BY 4.0; cite the dataset creators when using them.
- KDD Cup 2015 data remain subject to the terms of the selected host. This repository does not redistribute them.
- Federated learning avoids centralizing raw client records, but FedAvg/FedProx alone do not provide formal differential-privacy or secure-aggregation guarantees.
- Predictions should support, not replace, educator judgment. Inspect subgroup performance and avoid punitive or fully automated decisions about students.

## Citation

```bibtex
@inproceedings{lamsiyah2025privacy,
  author    = {Salima Lamsiyah and Aria Nourbakhsh and Samir El-Amrany and Christoph Schommer},
  title     = {Privacy-Preserving Federated Learning for Student Dropout Prediction: Enhancing Model Transparency with Explainable AI},
  booktitle = {Artificial Intelligence in Education},
  pages     = {324--332},
  publisher = {Springer Nature Switzerland},
  year      = {2025},
  doi       = {10.1007/978-3-031-98465-5_41}
}
```

Machine-readable metadata are also provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgments and license

The preprocessing and FedAvg structure were adapted from Max van Haastrecht, Matthieu Brinkhuis, and Marco Spruit's MIT-licensed [Federated-Learning-Analytics](https://github.com/MaxvanHaastrecht/Federated-Learning-Analytics) code. See [`LICENSE`](LICENSE) for the MIT terms and retained attribution.
