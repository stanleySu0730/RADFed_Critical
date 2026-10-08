# RADFed Critical experiment code

Start with `main.py` to run the paper experiments. Each experiment follows the same sequence: load a saved client fold, calibrate smoothness, select the fixed budget and Critical E settings on validation clients, then evaluate with separate final seeds.

## Read the pipeline

| Step | File | What it does |
| --- | --- | --- |
| Choose an experiment | `main.py` | Selects the dataset and timing profile, checks inputs, and records each subprocess's elapsed time. |
| Split the clients | `radfed_torch/data_splits.py` | Reads the released training, validation, and test client lists for the selected fold. |
| Prepare examples | `radfed_torch/data_pipeline.py` | Loads client arrays, encodes examples, applies normalization, and supplies minibatches. |
| Build the model | `radfed_torch/model_pipeline.py` | Defines the networks, initializes weights, and provides model-state averaging. |
| Train one run | `radfed_torch/run_pipeline.py` | Connects the arguments, data, model, and trainer. `train.py` is its command-line entry point. |
| Update and average | `radfed_torch/training_pipeline.py` | Performs local SGD, client redistribution, averaging, gradient profiling, and evaluation. |
| Select and evaluate | `radfed_torch/experiment_pipeline.py` | Runs calibration, fixed-budget search, tolerance search, and final evaluation in that order. |

`schedule.py` constructs client paths, `theory.py` computes critical budgets and feasible integer schedules, and `metrics.py` computes evaluation scores. `experiment_utils.py` contains shared configuration checks and selection summaries; `experiment_support.py` launches trials and checks whether saved results can be reused.

The `radfed_torch/image/` directory contains the image versions of the data, model, training, and experiment pipelines used for the recorded MNIST and CIFAR-10 results. It shares the split loader, character encoding, scheduling, budget formulas, metrics, and common selection helpers with the rest of the package. The image variant retains truncated-normal FFN initialization and raw learning-rate updates; the standard variant retains Kaiming initialization and nonincreasing learning-rate updates.

## Setup

Use Python 3.12. From the repository root on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.1 Pillow==12.3.0 torch==2.13.0+cu126 torchvision==0.28.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
```

GPU execution requires a driver compatible with the CUDA 12.6 wheels. The configurations use `device: auto`, which also supports CPU execution. On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` in place of the Windows interpreter path.

## Data and client splits

Download and extract the original RADFed client partitions into the paths below. The data are not committed because the extracted partitions total approximately 520 MiB.

```text
data/
  100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233/
  cifar10/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366/
  covertype/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122/
  shakespeare/143_client_data_seed245/
```

Download sources:

- [MNIST partitions](https://www.dropbox.com/s/0k327mg7ssrycqi/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233.tar.gz?dl=1)
- [CIFAR-10 partitions](https://www.dropbox.com/s/rzuvemautwlx8pj/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366.tar.gz?dl=1)
- [Covertype partitions](https://www.dropbox.com/s/dfy32fuc8cuqcm4/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122.tar.gz?dl=1)
- [Shakespeare partitions](https://www.dropbox.com/s/4m8ihsl18kopfad/143_client_data_seed245.tar.gz?dl=1)

Each directory contains `measures_<client-id>` and `labels_<client-id>`, plus these three lists for every fold from 0 to 4:

```text
fold0_tr_client_ids.lst
fold0_val_client_ids.lst
fold0_te_client_ids.lst
```

The loader uses the supplied client partitions and fold lists; it does not generate a new random data split. Training clients supply SGD updates and smoothness calibration. Validation clients select budgets and tolerances. Only the final evaluation runs evaluate test clients. For global tabular normalization, the mean and standard deviation are computed from training clients and applied to every split. Local normalization uses each client's feature statistics.

CIFAR-10 uses the included checkpoint `mobilenet_checkpoints/mobilenet_v2-7ebf99e0.pth` (SHA-256: `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`).

## Run the paper experiments

Check a configuration without training:

```powershell
.\.venv\Scripts\python.exe main.py --dataset mnist --validate-only
```

Run either timing profile or both:

```powershell
.\.venv\Scripts\python.exe main.py --dataset mnist --resume
.\.venv\Scripts\python.exe main.py --dataset cifar10 --resume
.\.venv\Scripts\python.exe main.py --dataset covertype --resume
.\.venv\Scripts\python.exe main.py --dataset shakespeare --resume
.\.venv\Scripts\python.exe main.py --dataset cifar10 --profile edge --resume
```

`--profile` accepts `all` (the default), `datacenter`, or `edge`. The settings are in the corresponding files under `experiments/`: MNIST and Covertype use FFNs, CIFAR-10 uses MobileNetV2, and Shakespeare uses a four-layer, four-head Transformer.

CIFAR-10 defaults to the original pilot tolerance in `experiments/cifar10.json`. Its predefined validation-search configuration is also available:

```powershell
.\.venv\Scripts\python.exe main.py --dataset cifar10 --cifar10-selection validation --resume
```

That choice uses `experiments/cifar10_validation.json`, including the declared tolerance candidates `[50, 75, 100, 150, 200, 300]`. Its newly measured selection cost belongs to that configuration; it does not reconstruct the unlogged historical pilot cost.

Outputs are written to `results/<dataset>/`, or `results/cifar10_validation/` for the CIFAR validation search. Use `--output-root` to choose a different directory. Resume reuses compatible completed trials and retains separate records for failed and retried attempts. The reorganized source has new provenance hashes, so existing outputs from the old layout require a separate output directory.

## Results and timing

The runner saves per-round metrics, final JSON results, model checkpoints, and CSV summaries of calibration, selection, and paired final comparisons. Every launched trial has an `execution.json` record and a process log under its `attempts/` directory. Whole-process timing includes imports, loading, profiling, evaluation, saving, and infeasible tuning attempts.

Datacenter and edge constants enter the scheduling formula without simulated network delays. Client updates run sequentially; the paper's recorded experiments used one GeForce GTX 1080. Measured execution time and modeled scheduling time are recorded separately. This repository exports numerical results without generating figures.

## Attribution

The partitions, original RADFed method, and character encoding helper come from *Aggregation Delayed Federated Learning* ([paper](https://arxiv.org/abs/2108.07433)). The bundled MobileNetV2 checkpoint comes from the official PyTorch model distribution; see the [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE). The datasets and dependencies retain their respective terms; no repository-wide license is asserted.
