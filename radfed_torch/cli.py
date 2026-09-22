from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .data import FederatedData
from .theory import NoFeasibleCriticalSchedule
from .trainer import RADFedTrainer, TrainingConfig


def _positive_widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden sizes must be comma-separated integers") from error
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python 3.12 / PyTorch runner for corrected RADFed theory."
    )
    parser.add_argument("--client-data-path", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=None)
    parser.add_argument(
        "--split-prefix",
        default="fold",
        help="prefix for fold client-list files; default uses fold0_tr_client_ids.lst",
    )
    parser.add_argument(
        "--method",
        choices=["corrected-fixed", "critical", "critical-r1", "critical-frozen", "original-radfed", "fedavg"],
        default="corrected-fixed",
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        choices=["ffn", "lr", "lstm", "transformer", "mobilenetv2", "resnet18"],
        default="ffn",
    )
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-size", type=int, default=512)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument(
        "--ffn-hidden-sizes",
        type=_positive_widths,
        default=None,
        help="comma-separated FFN widths; omitted preserves the two-layer hidden-size model",
    )
    parser.add_argument(
        "--ffn-initialization",
        choices=["legacy", "kaiming"],
        default="kaiming",
        help="Kaiming ReLU hidden layers and Xavier output (default); legacy reproduces std=0.02",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--participation", type=float, default=0.1)
    parser.add_argument(
        "--r-local-steps",
        type=int,
        default=5,
        help=(
            "fixed client SGD steps R; the critical method computes "
            "E_crit(R) from Equation (66)"
        ),
    )
    parser.add_argument("--s-client-visits", type=int, default=5)
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        help="paper-baseline epochs; valid only for original-radfed and fedavg",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="optimizer rate for original-radfed and fedavg; corrected methods use eta=1/(sqrt(6)EL)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=["sgd"],
        default="sgd",
        help="standard SGD is required by the RADFed update and convergence analysis",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "f1", "auc", "loss"],
        default="loss",
        help="secondary validation score; validation loss is always recorded",
    )
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument(
        "--training-eval-frequency",
        type=int,
        default=0,
        help=(
            "evaluate the aggregated model on all training clients every N outer "
            "rounds and at the final round; 0 disables this diagnostic"
        ),
    )
    parser.add_argument("--profile-frequency", type=int, default=5)
    parser.add_argument("--profile-batches", type=int, default=1)
    parser.add_argument("--profile-diagnostics", action="store_true",
                        help="measure fixed-budget runs without changing E, L or learning rate")
    parser.add_argument("--skip-validation-evaluation", dest="evaluate_validation",
                        action="store_false", help="training-only calibration; also disable test evaluation")
    parser.add_argument(
        "--profile-clients",
        type=int,
        default=0,
        help="fixed profiling cohort size; 0 uses the N active trajectories",
    )
    parser.add_argument("--target-epsilon", type=float, default=None)
    parser.add_argument(
        "--allow-infeasible-critical",
        action="store_true",
        help=(
            "record a validation-only critical-schedule infeasibility and exit "
            "successfully; intended only for epsilon search"
        ),
    )
    parser.add_argument("--l-smooth", type=float, default=1.0)
    parser.add_argument(
        "--l-smooth-update",
        choices=["fixed", "raw", "ema"],
        default="ema",
        help=(
            "fixed keeps L constant; raw uses the observed secant ratio directly; "
            "ema retains the legacy smoothed update"
        ),
    )
    parser.add_argument(
        "--calibrate-l-only",
        action="store_true",
        help="estimate L from a discarded training-client secant probe and exit",
    )
    parser.add_argument(
        "--l-calibration-relative-probe-distance",
        type=float,
        default=1e-3,
        help="probe displacement relative to max(||w_0||,1) during L calibration",
    )
    parser.add_argument("--t-comm", type=float, default=0.0)
    parser.add_argument("--t-comp", type=float, default=0.01)
    parser.add_argument("--t-outer", type=float, default=0.0)
    parser.add_argument(
        "--fixed-timing-costs",
        action="store_true",
        help=(
            "keep configured t_comm and t_comp fixed in schedule selection; "
            "do not replace t_comp with an online wall-time estimate"
        ),
    )
    parser.add_argument("--max-total-steps", type=int, default=250)
    parser.add_argument("--max-local-steps", type=int, default=25)
    parser.add_argument("--normalization", choices=["none", "local", "global"], default="none")
    parser.add_argument("--normalized-features", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="out/pytorch_run")
    parser.add_argument("--pretrained-mobilenet", action="store_true")
    parser.add_argument("--mobilenet-weights-path", default=None)
    parser.add_argument(
        "--skip-model-checkpoint",
        dest="save_model",
        action="store_false",
        help=(
            "do not save current_model.pt; used for validation-only tuning "
            "runs to avoid retaining hundreds of large model checkpoints"
        ),
    )
    parser.add_argument(
        "--shuffle-client-data",
        action="store_true",
        help="shuffle examples each local epoch; paper artifact commands leave this disabled",
    )
    parser.add_argument(
        "--skip-test-evaluation",
        dest="evaluate_test",
        action="store_false",
        help="do not load or evaluate test clients (used by validation-only tuning runs)",
    )
    parser.set_defaults(evaluate_test=True, save_model=True, evaluate_validation=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_infeasible_critical and args.evaluate_test:
        raise ValueError(
            "--allow-infeasible-critical requires --skip-test-evaluation"
        )
    if args.allow_infeasible_critical and not args.method.startswith("critical"):
        raise ValueError(
            "--allow-infeasible-critical is valid only for a critical method"
        )
    data = FederatedData.load(
        directory=args.client_data_path,
        fold=args.fold,
        inner_fold=args.inner_fold,
        model_name=args.model_name,
        normalization=args.normalization,
        normalized_features=args.normalized_features,
        sequence_length=args.sequence_length,
        split_prefix=args.split_prefix,
    )
    config_values = vars(args).copy()
    for name in [
        "client_data_path",
        "fold",
        "inner_fold",
        "normalization",
        "normalized_features",
        "allow_infeasible_critical",
        "calibrate_l_only",
        "l_calibration_relative_probe_distance",
    ]:
        config_values.pop(name)
    config = TrainingConfig(**config_values)
    trainer = RADFedTrainer(data, config)
    if args.calibrate_l_only:
        if args.evaluate_test:
            raise ValueError("--calibrate-l-only requires --skip-test-evaluation")
        result = trainer.estimate_l_smooth_secant(
            args.l_calibration_relative_probe_distance
        )
        result.update(
            {
                "fold": int(args.fold),
                "seed": int(args.seed),
                "client_data_path": str(args.client_data_path),
                "model_name": str(args.model_name),
                "split_prefix": str(args.split_prefix),
            }
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "l_smooth_calibration.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "config.json").write_text(
            json.dumps(config_values, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    try:
        result = trainer.train()
    except NoFeasibleCriticalSchedule as error:
        if not args.allow_infeasible_critical:
            raise
        output_dir = Path(args.output_dir)
        metrics_path = output_dir / "round_metrics.csv"
        completed_round = 0
        client_sgd_steps = 0
        if metrics_path.is_file():
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                completed_round = int(rows[-1].get("outer_round") or 0)
                client_sgd_steps = int(
                    float(rows[-1].get("cumulative_client_sgd_steps") or 0)
                )
        result = {
            "status": "infeasible",
            "method": args.method,
            "target_epsilon": float(args.target_epsilon),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "current_outer_round": completed_round,
            "client_sgd_steps": client_sgd_steps,
            "test_evaluated": False,
            "current_validation_loss": None,
            "current_validation_score": None,
        }
        if not math.isfinite(result["target_epsilon"]):
            raise ValueError("target_epsilon must be finite") from error
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "infeasible_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
