# RADFed Critical experiment supplement

This repository provides the implementations and declared experiment protocols for *Analytically Grounded Robust Aggregation-Delayed Federated Learning*. The named launcher chooses the implementation that corresponds to each recorded experiment.

| Protocol | Implementation and behavior |
| --- | --- |
| `mnist-archive` | September 1 archived trainer; truncated-normal FFN initialization; raw candidate learning rates; validation-selected tolerance. |
| `cifar10-archive` | Original CIFAR-10 protocol, retained for historical comparisons. It does not provide a measured historical adaptive tuning cost. |
| `cifar10-rerun` | Prospective October 4 protocol on the unchanged archived trainer; validation candidates `[50,75,100,150,200,300]`; raw candidate learning rates; every attempt timed and logged. |
| `covertype` | Maintained trainer; Kaiming hidden weights; validation-selected tolerance; nonincreasing learning rates. |
| `shakespeare` | Maintained four-layer, four-head Transformer; validation-selected tolerance; nonincreasing learning rates. |

The maintained FFN default remains Kaiming. `experiments/paper_experiments.json` retains the modern eight-entry configuration, but its image settings are not the implementation that generated the archived manuscript results. Use the named protocols below for those results. The archived image sources are immutable and checksum-verified before execution; see [PROTOCOLS.md](PROTOCOLS.md).

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
â”œâ”€â”€ 100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_search0.002_piter5e5_biter5e5_5folds_seed233/
â”œâ”€â”€ cifar10/100_client_data_dirichlet_noniid_classes_random_qp_alpha1_beta0.1_0.1-0.2opt_loss_piter5e5_biter5e5_seed2366/
â”œâ”€â”€ covertype/100_client_data_dirichlet_noniid_cat_features_classes_random_qp_alpha1_lambda0.1_theta0.1_5folds_seed1122/
â””â”€â”€ shakespeare/143_client_data_seed245/
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

## Validate and run

Run these commands from the repository root. Validation checks the manifest, archived source hashes and selection/final seed separation without launching training:

```powershell
.\.venv\Scripts\python.exe reproduce_paper.py --protocol cifar10-rerun --validate-only
```

To reproduce the prospective CIFAR-10 experiment:

```powershell
.\.venv\Scripts\python.exe reproduce_paper.py --protocol cifar10-rerun --resume
.\.venv\Scripts\python.exe audit_cifar10_rerun.py --root results\cifar10-rerun
```

The declared matrix has **420 training trials plus 30 shared calibration trials**: two regimes, five folds, three seeds, six fixed budgets, six tolerance candidates and two selected final methods. All three tuning seeds must be feasible before a tolerance is eligible. Selection uses final-round validation loss, with the declared tie breakers. Final seeds alone evaluate test clients. The audit replays selection independently and computes paired fold intervals and full procedure time.

Use the other named protocols in the same launcher:

```powershell
.\.venv\Scripts\python.exe reproduce_paper.py --protocol mnist-archive --resume
.\.venv\Scripts\python.exe reproduce_paper.py --protocol covertype --resume
.\.venv\Scripts\python.exe reproduce_paper.py --protocol shakespeare --resume
```

To run only one regime, append `--dataset cifar10_datacenter` or `--dataset cifar10_edge` to the CIFAR launcher command. A completed regime can be audited with the matching `--dataset` option; that audit does not certify the complete two-regime matrix. `--output-root` selects a separate output directory. Linux uses `.venv/bin/python` in place of the PowerShell interpreter path.

## Outputs and timing

The launcher writes outputs under `results/<protocol>/`, which is excluded from Git. Every launched subprocess has a durable `execution.json`, a separate `attempts/<number>/process.log`, UTC timestamps, exit status and whole-process elapsed time. Result markers are hashed when written. `runtime_provenance.json` records package versions, GPU information and training/selection source hashes. The archived trainer additionally writes its original round-clock metrics; the two timer scopes are reported separately.

Resume accepts the same manifest, implementation and runtime, and refuses to invent timings for previously unlogged results. Attempt records remain separate when a failed trial is retried. The CIFAR audit stops on missing records, failed attempts, test leakage, ineligible or incorrect selections, modified results or undeclared trials. Failed process attempts require separate diagnosis before costs are reported. A local output lock prevents two launchers from using the same directory; after an interrupted launcher, remove its `.run.lock` only after confirming that process has stopped.

Full procedure time is shared training-client calibration plus **every** tuning attempt plus three selected final runs per fold. Include calibration once in each method's hypothetical pipeline, and once overall for the combined experiment. Newly measured rerun costs are not historical pilot measurements and must not be appended to old final-run costs. The audit reports minutes saved and percentage savings from the ratio of mean procedure totals; paired final-run percentages are a separate statistic.

## Reproducibility notes

- Standard SGD without momentum; five updates per client visit.
- One cohort per outer round, with cyclic distinct-client paths and delayed averaging.
- Calibration seeds `1001,1002,1003` use training clients only.
- Tuning seeds `101,102,103` use validation clients; final seeds `1,2,3` alone evaluate test clients.
- Datacenter uses `(t_comm,t_comp)=(0.01,0.10)` seconds; edge uses `(0.50,0.05)` seconds.
- These constants enter the scheduling formula and do not introduce simulated communication delays. Recorded experiments run clients sequentially on one GeForce GTX 1080.
- The budget optimum is conditional on the stationarity surrogate described in the paper. Empirical profiling quantities are scheduling inputs, not uniform certificates of the theorem's assumptions.

Run the focused protocol/logging checks with `python -m unittest discover -s tests`. `python tests/smoke_archived.py` additionally runs a small CPU example on temporary synthetic clients and verifies logged infeasibility, test isolation and resume. These checks do not replace a complete GPU experiment rerun.

## Attribution and asset terms

The partitions and original RADFed method come from *Aggregation Delayed Federated Learning* ([paper](https://arxiv.org/abs/2108.07433)). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the bundled helper and pretrained checkpoint provenance. No blanket license for this supplement or the original data partitions is asserted; their respective asset terms remain applicable.
