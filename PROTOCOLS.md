# Protocol and source provenance

The launcher separates two implementations. It does not change their optimization, data loading, gradient profiling, routing or selection mathematics.

## Archived image experiments

`protocols/archived_images_20260901/` contains the training code extracted from the September 1 experiment archive. Its archive SHA-256 is `9d0ed10f01306f83e690b219689fb89b1292bf40845f121bfddbba463064fc9f`. Each included file was checked against the October 4 rerun artifact inventory. The file hashes are stored in `source_hashes.json`; the launcher refuses modified files. Git attributes preserve their exact bytes on Windows and Linux.

Both fixed and adaptive methods apply the raw candidate learning rate after gradient profiling, including increases. The archived MNIST network uses truncated-normal initialization with standard deviation 0.02 and zero biases. MobileNetV2 uses the bundled pretrained weights and the archived training behavior. The helper `language_utils.py` and loss-plot module are retained because the archived loader and runner import them. They have no cluster or credential dependencies.

`mnist_archived_20260901.json` and `cifar10_archived_20260901.json` retain the corresponding entries from the saved historical manifests, with only the output root and protocol label changed. The MNIST search retains the original feasibility-expansion metadata; the archived runner evaluates the declared candidate list. All recorded MNIST folds had eligible candidates within that list. The historical CIFAR manifest retains its original tolerance setting solely to reproduce the original final-run protocol. It is separate from the new validation-selected rerun.

## Prospective CIFAR-10 rerun

`experiments/cifar10_rerun_20261004.json` is byte-identical to the declared server manifest, SHA-256 `766396807b6f62bad8bccb2f2b17fedd8bf70466b81389112dcce06e73f2c872`. The launcher overrides only the output location. It runs the same archived implementation used on the server.

For each of five folds and both timing regimes:

- Three training-only calibration seeds initialize smoothness with their maximum positive secant estimate.
- Six fixed visit counts, each with three validation tuning seeds, select the lowest mean final validation loss, breaking ties toward smaller total steps.
- Six tolerance candidates `[50,75,100,150,200,300]`, each with those three tuning seeds, are attempted. All seeds must be feasible. Selection minimizes mean final validation loss, then mean client SGD steps, then prefers the larger tolerance.
- Two locked methods each run three separate final seeds. These runs alone evaluate test clients.

This gives 30 calibration and 420 training attempts. There is no automatic search expansion in this prospective protocol. If a fold has no eligible tolerance, report the declared search's limitation before defining a separate validation-only expansion. Do not use test results or final seeds to expand or select candidates.

`audit_cifar10_rerun.py` checks every expected identity, configuration, execution record, log, calibration estimate, result and round clock. It recomputes selection from tuning records and checks final configurations against those choices. It independently reports paired fold-level effects and measured complete procedure totals. It can audit one completed regime without certifying the other regime.

## Covertype and Shakespeare

`experiments/covertype_shakespeare.json` retains the four relevant entries of the maintained manifest. These entries match the saved Covertype and Transformer experiment manifests. The maintained trainer uses nonincreasing rates for both methods; Covertype has Kaiming hidden-layer initialization and Shakespeare uses the 824,144-parameter causal Transformer. The maintained defaults and `paper_experiments.json` remain available for new experiments.

## Verification scope

The supplement's checks validate source integrity, protocol dispatch, logging, resumption restrictions, selection replay and test isolation. Small CPU training checks exercise the archived trainer. They do not remeasure the full GPU experiments, establish missing historical costs, or prove the stationarity premise for the adaptive implementation.
