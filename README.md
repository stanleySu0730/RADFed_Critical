# RADFed Critical-E experiment supplement

This repository contains the minimal PyTorch 2.13 / Python 3.12 implementation used for the paper's Critical-E experiments. It includes the training algorithm, theoretical schedule, model definitions, client-data loader, and one consolidated manifest for MNIST, CIFAR-10, COVFEAT-L, and Shakespeare Transformer under both datacenter and edge timing profiles.

It intentionally excludes unit tests, diagnostics, plotting code, generated figures, old TensorFlow/Ray code, recovery utilities, Nomad-specific jobs, and previous experimental variants.

## Environment

Create a Python 3.12 environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-pytorch.txt
```

The dependency file pins NumPy, Pillow, PyTorch, and torchvision. See [DEPENDENCIES.md](DEPENDENCIES.md) for the version table, CUDA and driver requirements, installation checks, and runtime inputs.

## Required data

Dataset files are not committed because the extracted partitions total approximately 520 MiB. Download and extract the original RADFed client partitions so the following directories exist:

```text
data/
├── 100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233/
├── cifar10/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366/
├── covertype/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122/
└── shakespeare/143_client_data_seed245/
```

Documented source archives:

- MNIST: https://www.dropbox.com/s/0k327mg7ssrycqi/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233.tar.gz?dl=1
- CIFAR-10: https://www.dropbox.com/s/rzuvemautwlx8pj/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366.tar.gz?dl=1
- COVFEAT: https://www.dropbox.com/s/dfy32fuc8cuqcm4/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122.tar.gz?dl=1
- Shakespeare: https://www.dropbox.com/s/4m8ihsl18kopfad/143_client_data_seed245.tar.gz?dl=1

Checksums of the local archives used to prepare the recorded experiments:

| Archive | SHA-256 |
| --- | --- |
| `mnist_paper.tar.gz` | `7a6f8181e51782c89c40598344e57d6d0823c3f2b721ff6ef4cb09de6ffa222d` |
| `paper-cifar10-partition.tar.gz` | `7a4c1c1aef78c29749f4fddc137d30f3e0d447c502103a5b228e9de40833dbf8` |
| `covfeat.tar.gz` | `1f8ede704cb89bfc9919087c881719e1d18be0a3648b2ab43c76c2c8f52813af` |
| `shakespeare.tar.gz` | `3b9b22d8816049eae03aeade81f1389030219390c5ef40e2c14bf73b0b833427` |

Every extracted dataset must contain `measures_<client-id>`, `labels_<client-id>`, and the 15 `fold0`--`fold4` training, validation, and test client-list files.

## MobileNetV2 checkpoint

The required CIFAR-10 checkpoint is included at:

```text
mobilenet_checkpoints/mobilenet_v2-7ebf99e0.pth
```

Its SHA-256 is `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`.

## Run all paper experiments

From the repository root:

```powershell
.\.venv\Scripts\python.exe run_paper_experiments.py `
  --config experiments\paper_experiments.json `
  --resume
```

The manifest runs the two timing profiles for all four benchmarks. It performs per-fold training-only smoothness calibration, validation-only fixed-E grid search, validation-only epsilon selection where specified, and final evaluation with seeds 1, 2, and 3 over five client folds.

Use repeated `--dataset` options to run only selected entries. For example:

```powershell
.\.venv\Scripts\python.exe run_paper_experiments.py `
  --config experiments\paper_experiments.json `
  --dataset mnist_datacenter `
  --dataset mnist_edge `
  --resume
```

Available dataset entries are:

- `mnist_datacenter`
- `mnist_edge`
- `cifar10_datacenter`
- `cifar10_edge`
- `covfeat_l_datacenter`
- `covfeat_l_edge`
- `shakespeare_transformer_datacenter`
- `shakespeare_transformer_edge`

## Outputs

Outputs are written to `results/paper_experiments/`. The runner records the resolved manifest, calibration and selection tables, per-run configurations, current-model validation metrics, profile statistics, final test metrics, method summaries, and paired comparisons. It does not generate graphs or apply a running-best transformation.

## Reproducibility notes

- Standard SGD without momentum is the only optimizer.
- Each local update draws a fresh random mini-batch.
- Calibration seeds are 1001, 1002, and 1003.
- Validation-selection seeds are 101, 102, and 103.
- Final evaluation seeds are 1, 2, and 3.
- Test clients are not evaluated during calibration or hyperparameter selection.
- Datacenter uses `(t_comm, t_comp) = (0.01, 0.10)` seconds.
- Edge uses `(t_comm, t_comp) = (0.50, 0.05)` seconds.
- Timing constants enter the analytical schedule but do not create artificial delays.

## Provenance and licensing

This implementation extends the code associated with *Aggregation Delayed Federated Learning*. Before publishing this repository, add a license compatible with the upstream repository and preserve all required upstream attribution. No license was present in the source working directory from which this minimal supplement was assembled, so no license has been invented here.

