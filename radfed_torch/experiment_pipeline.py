"""Paper experiment pipeline: calibrate -> select fixed/critical budgets -> final evaluation -> CSV summaries."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .experiment_utils import (
    _dataset_manifests,
    _write_csv,
    _grid_search_summaries,
    _epsilon_token,
    _critical_epsilon_summaries,
    _l_smooth_calibration_summaries,
    _summaries,
    _paired_tests,
)
from .experiment_support import (
    _validate_manifest,
    _fold_has_fully_feasible_epsilon,
    _run_l_smooth_calibration,
    _run_training,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON experiment manifest")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="run only this dataset entry; may be repeated",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="override output_root (useful for Nomad allocation storage)",
    )
    parser.add_argument("--resume", action="store_true", help="reuse completed results.json files")
    parser.add_argument(
        "--rerun-critical-final-dataset",
        action="append",
        default=[],
        help=(
            "dataset whose final Critical-E runs must be rerun even when a "
            "compatible result exists; may be repeated"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    manifest = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    output_root = Path(
        args.output_root or manifest.get("output_root", "out/paper_protocol")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    train_script = Path(__file__).resolve().parents[1] / "train.py"
    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    grid_summaries: list[dict[str, Any]] = []
    grid_selections: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    epsilon_summaries: list[dict[str, Any]] = []
    epsilon_selections: list[dict[str, Any]] = []
    epsilon_expansion_rows: list[dict[str, Any]] = []
    epsilon_expected_runs: dict[str, int] = {}
    l_calibration_rows: list[dict[str, Any]] = []
    l_calibration_summaries: list[dict[str, Any]] = []
    l_calibration_selections: list[dict[str, Any]] = []
    datasets = _dataset_manifests(manifest)
    if args.dataset:
        requested = set(args.dataset)
        available = {name for name, _ in datasets}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError("unknown dataset filter(s): " + ", ".join(unknown))
        datasets = [(name, dataset) for name, dataset in datasets if name in requested]
    multi_dataset = "datasets" in manifest
    for dataset_name, dataset in datasets:
        dataset_root = output_root / dataset_name if multi_dataset else output_root
        selected_s_by_fold: dict[int, int] = {}
        selected_epsilon_by_fold: dict[int, float] = {}
        selected_l_by_fold: dict[int, float] = {}
        # 1. Calibrate smoothness using only training clients.
        l_calibration = dataset.get("l_smooth_calibration")
        if isinstance(l_calibration, dict) and l_calibration.get("enabled", True):
            calibration_seeds = [
                int(value) for value in l_calibration["calibration_seeds"]
            ]
            relative_probe_distance = float(
                l_calibration.get("relative_probe_distance", 1e-3)
            )
            calibration_common = {
                **dataset["common"],
                "client_data_path": dataset["client_data_path"],
            }
            for fold in dataset["folds"]:
                for calibration_seed in calibration_seeds:
                    run_dir = (
                        dataset_root
                        / "l_smooth_calibration"
                        / f"fold_{fold}"
                        / f"seed_{calibration_seed}"
                    )
                    label = (
                        f"{dataset_name}/L-calibration fold={fold} "
                        f"seed={calibration_seed}"
                    )
                    result = _run_l_smooth_calibration(
                        train_script=train_script,
                        fold=int(fold),
                        seed=calibration_seed,
                        run_dir=run_dir,
                        merged=calibration_common,
                        relative_probe_distance=relative_probe_distance,
                        resume=args.resume,
                        label=label,
                    )
                    l_calibration_rows.append(
                        {
                            "dataset": dataset_name,
                            "fold": int(fold),
                            "calibration_seed": calibration_seed,
                            "estimator": "gradient_secant",
                            "relative_probe_distance": relative_probe_distance,
                            "profile_clients": int(result["profile_clients"]),
                            "gradient_difference_norm": float(
                                result["gradient_difference_norm"]
                            ),
                            "parameter_difference_norm": float(
                                result["parameter_difference_norm"]
                            ),
                            "l_raw": float(result["l_raw"]),
                            "validation_clients_used": False,
                            "test_clients_used": False,
                            "run_dir": str(run_dir),
                        }
                    )
                    _write_csv(
                        output_root / "l_smooth_calibration_results.csv",
                        l_calibration_rows,
                    )
            l_calibration_summaries, l_calibration_selections = (
                _l_smooth_calibration_summaries(l_calibration_rows)
            )
            _write_csv(
                output_root / "l_smooth_calibration_summary.csv",
                l_calibration_summaries,
            )
            _write_csv(
                output_root / "l_smooth_calibration_selection.csv",
                l_calibration_selections,
            )
            (output_root / "l_smooth_calibration_selection.json").write_text(
                json.dumps(l_calibration_selections, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            selected_l_by_fold = {
                int(row["fold"]): float(row["selected_l_smooth"])
                for row in l_calibration_selections
                if row["dataset"] == dataset_name
            }
            missing_folds = sorted(set(dataset["folds"]) - set(selected_l_by_fold))
            if missing_folds:
                raise ValueError(
                    f"dataset {dataset_name}: no L calibration for folds {missing_folds}"
                )
        # 2. Search fixed budgets using validation loss.
        grid = dataset.get("grid_search")
        if isinstance(grid, dict) and grid.get("enabled", True):
            fixed_overrides = dataset["methods"]["corrected-fixed"]
            tuning_seeds = [int(value) for value in grid["tuning_seeds"]]
            candidates = sorted(int(value) for value in grid["s_candidates"])
            for fold in dataset["folds"]:
                for s_visits in candidates:
                    merged = {
                        **dataset["common"],
                        **fixed_overrides,
                        "client_data_path": dataset["client_data_path"],
                        "s_client_visits": s_visits,
                    }
                    if selected_l_by_fold:
                        merged["l_smooth"] = selected_l_by_fold[int(fold)]
                    r_steps = int(merged.get("r_local_steps", 5))
                    for tuning_seed in tuning_seeds:
                        run_dir = (
                            dataset_root
                            / "grid_search"
                            / "corrected-fixed"
                            / f"fold_{fold}"
                            / f"r_{r_steps}_s_{s_visits}"
                            / f"seed_{tuning_seed}"
                        )
                        label = (
                            f"{dataset_name}/grid-search fold={fold} "
                            f"R={r_steps} S={s_visits} seed={tuning_seed}"
                        )
                        result = _run_training(
                            train_script=train_script,
                            method="corrected-fixed",
                            fold=int(fold),
                            seed=tuning_seed,
                            run_dir=run_dir,
                            merged=merged,
                            resume=args.resume,
                            evaluate_test=False,
                            label=label,
                        )
                        if result.get("test_evaluated") is not False:
                            raise ValueError(
                                f"grid-search run evaluated test clients: {run_dir}"
                            )
                        if int(result["current_outer_round"]) != int(merged["rounds"]):
                            raise ValueError(
                                f"grid-search run did not reach its final round: {run_dir}"
                            )
                        grid_rows.append(
                            {
                                "dataset": dataset_name,
                                "fold": int(fold),
                                "tuning_seed": tuning_seed,
                                "r_local_steps": r_steps,
                                "s_client_visits": s_visits,
                                "e_total_steps": r_steps * s_visits,
                                "initial_l_smooth": float(merged["l_smooth"]),
                                "selection_round": int(result["current_outer_round"]),
                                "current_validation_score": float(
                                    result["current_validation_score"]
                                ),
                                "current_validation_loss": float(
                                    result["current_validation_loss"]
                                ),
                                "test_evaluated": False,
                                "run_dir": str(run_dir),
                            }
                        )
                        _write_csv(output_root / "grid_search_results.csv", grid_rows)

            grid_summaries, grid_selections = _grid_search_summaries(grid_rows)
            _write_csv(output_root / "grid_search_summary.csv", grid_summaries)
            _write_csv(output_root / "grid_search_selection.csv", grid_selections)
            (output_root / "grid_search_selection.json").write_text(
                json.dumps(grid_selections, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            selected_s_by_fold = {
                int(row["fold"]): int(row["selected_s_client_visits"])
                for row in grid_selections
                if row["dataset"] == dataset_name
            }
            missing_folds = sorted(set(dataset["folds"]) - set(selected_s_by_fold))
            if missing_folds:
                raise ValueError(
                    f"dataset {dataset_name}: no grid-search selection for folds {missing_folds}"
                )

        # 3. Select the critical-budget tolerance on validation clients.
        epsilon_search = dataset.get("critical_epsilon_search")
        if isinstance(epsilon_search, dict) and epsilon_search.get("enabled", True):
            critical_overrides = dataset["methods"]["critical"]
            tuning_seeds = [
                int(value) for value in epsilon_search["tuning_seeds"]
            ]
            epsilon_expected_runs[dataset_name] = len(tuning_seeds)
            candidates = sorted(
                float(value) for value in epsilon_search["epsilon_candidates"]
            )
            automatic_expansion = bool(
                epsilon_search.get("automatic_expansion", False)
            )
            expansion_factor = float(epsilon_search.get("expansion_factor", 2.0))
            max_expansions = int(epsilon_search.get("max_expansions", 6))
            for fold in dataset["folds"]:
                pending_candidates = list(candidates)
                evaluated_candidates = list(candidates)
                expansion_index = 0
                while True:
                    for epsilon in pending_candidates:
                        merged = {
                            **dataset["common"],
                            **critical_overrides,
                            "client_data_path": dataset["client_data_path"],
                            "target_epsilon": epsilon,
                        }
                        if selected_l_by_fold:
                            merged["l_smooth"] = selected_l_by_fold[int(fold)]
                        for tuning_seed in tuning_seeds:
                            run_dir = (
                                dataset_root
                                / "critical_epsilon_search"
                                / "critical"
                                / f"fold_{fold}"
                                / f"epsilon_{_epsilon_token(epsilon)}"
                                / f"seed_{tuning_seed}"
                            )
                            label = (
                                f"{dataset_name}/epsilon-search fold={fold} "
                                f"epsilon={epsilon:.12g} seed={tuning_seed}"
                            )
                            result = _run_training(
                                train_script=train_script,
                                method="critical",
                                fold=int(fold),
                                seed=tuning_seed,
                                run_dir=run_dir,
                                merged=merged,
                                resume=args.resume,
                                evaluate_test=False,
                                label=label,
                                allow_infeasible=True,
                            )
                            feasible = result.get("status") != "infeasible"
                            if result.get("test_evaluated") is not False:
                                raise ValueError(
                                    f"epsilon-search run evaluated test clients: {run_dir}"
                                )
                            if feasible and int(result["current_outer_round"]) != int(
                                merged["rounds"]
                            ):
                                raise ValueError(
                                    f"epsilon-search run did not reach its final round: {run_dir}"
                                )
                            epsilon_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "fold": int(fold),
                                    "tuning_seed": tuning_seed,
                                    "target_epsilon": epsilon,
                                    "initial_l_smooth": float(merged["l_smooth"]),
                                    "status": "feasible" if feasible else "infeasible",
                                    "selection_round": (
                                        int(result["current_outer_round"])
                                        if feasible
                                        else None
                                    ),
                                    "current_validation_score": (
                                        float(result["current_validation_score"])
                                        if feasible
                                        else None
                                    ),
                                    "current_validation_loss": (
                                        float(result["current_validation_loss"])
                                        if feasible
                                        else None
                                    ),
                                    "client_sgd_steps": int(
                                        result.get("client_sgd_steps", 0)
                                    ),
                                    "failure_message": (
                                        None if feasible else result["failure_message"]
                                    ),
                                    "test_evaluated": False,
                                    "run_dir": str(run_dir),
                                }
                            )
                            _write_csv(
                                output_root / "critical_epsilon_search_results.csv",
                                epsilon_rows,
                            )
                    if _fold_has_fully_feasible_epsilon(
                        epsilon_rows,
                        dataset=dataset_name,
                        fold=int(fold),
                        expected_tuning_runs=len(tuning_seeds),
                    ):
                        break
                    if not automatic_expansion:
                        break
                    if expansion_index >= max_expansions:
                        raise ValueError(
                            f"dataset {dataset_name} fold {fold}: automatic epsilon "
                            f"expansion exhausted {max_expansions} levels without a "
                            "fully feasible candidate"
                        )
                    previous_maximum = max(evaluated_candidates)
                    expanded_epsilon = previous_maximum * expansion_factor
                    if not math.isfinite(expanded_epsilon):
                        raise ValueError(
                            f"dataset {dataset_name} fold {fold}: epsilon expansion overflow"
                        )
                    expansion_index += 1
                    evaluated_candidates.append(expanded_epsilon)
                    pending_candidates = [expanded_epsilon]
                    epsilon_expansion_rows.append(
                        {
                            "dataset": dataset_name,
                            "fold": int(fold),
                            "expansion_index": expansion_index,
                            "trigger": "no_candidate_feasible_for_all_tuning_seeds",
                            "expansion_factor": expansion_factor,
                            "previous_maximum_epsilon": previous_maximum,
                            "expanded_epsilon": expanded_epsilon,
                            "validation_loss_used_to_trigger_expansion": False,
                            "test_clients_used": False,
                        }
                    )
                    _write_csv(
                        output_root / "critical_epsilon_expansion.csv",
                        epsilon_expansion_rows,
                    )
                    (output_root / "critical_epsilon_expansion.json").write_text(
                        json.dumps(epsilon_expansion_rows, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )

            if automatic_expansion:
                (output_root / "critical_epsilon_expansion.json").write_text(
                    json.dumps(epsilon_expansion_rows, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            epsilon_summaries, epsilon_selections = _critical_epsilon_summaries(
                epsilon_rows, epsilon_expected_runs
            )
            for selection in epsilon_selections:
                selection_dataset = dict(datasets)[str(selection["dataset"])]
                selection_search = selection_dataset.get(
                    "critical_epsilon_search", {}
                )
                expanded = [
                    float(row["expanded_epsilon"])
                    for row in epsilon_expansion_rows
                    if row["dataset"] == selection["dataset"]
                    and int(row["fold"]) == int(selection["fold"])
                ]
                selection["automatic_expansion_enabled"] = bool(
                    selection_search.get("automatic_expansion", False)
                )
                selection["expansion_trigger"] = selection_search.get(
                    "expansion_trigger", "feasibility_only"
                )
                selection["expanded_candidates"] = expanded
                selection["expansion_count"] = len(expanded)
            _write_csv(
                output_root / "critical_epsilon_search_summary.csv",
                epsilon_summaries,
            )
            _write_csv(
                output_root / "critical_epsilon_selection.csv",
                epsilon_selections,
            )
            (output_root / "critical_epsilon_selection.json").write_text(
                json.dumps(epsilon_selections, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            selected_epsilon_by_fold = {
                int(row["fold"]): float(row["selected_target_epsilon"])
                for row in epsilon_selections
                if row["dataset"] == dataset_name
            }
            missing_folds = sorted(
                set(dataset["folds"]) - set(selected_epsilon_by_fold)
            )
            if missing_folds:
                raise ValueError(
                    f"dataset {dataset_name}: no epsilon selection for folds {missing_folds}"
                )

        # 4. Lock fold-specific choices and evaluate the separate final seeds.
        for method, overrides in dataset["methods"].items():
            for fold in dataset["folds"]:
                merged = {
                    **dataset["common"],
                    **overrides,
                    "client_data_path": dataset["client_data_path"],
                }
                if selected_l_by_fold:
                    merged["l_smooth"] = selected_l_by_fold[int(fold)]
                if method == "corrected-fixed" and selected_s_by_fold:
                    merged["s_client_visits"] = selected_s_by_fold[int(fold)]
                if method == "critical" and selected_epsilon_by_fold:
                    merged["target_epsilon"] = selected_epsilon_by_fold[int(fold)]
                for seed in dataset["seeds"]:
                    run_dir = dataset_root / method / f"fold_{fold}" / f"seed_{seed}"
                    label = f"{dataset_name}/{method} fold={fold} seed={seed}"
                    result = _run_training(
                        train_script=train_script,
                        method=method,
                        fold=int(fold),
                        seed=int(seed),
                        run_dir=run_dir,
                        merged=merged,
                        resume=args.resume,
                        evaluate_test=True,
                        label=label,
                        force_rerun=(
                            method == "critical"
                            and dataset_name in args.rerun_critical_final_dataset
                        ),
                    )
                    score = float(result["test_score"])
                    loss = float(result["test_loss"])
                    if not math.isfinite(score) or not math.isfinite(loss):
                        raise ValueError(f"non-finite result in {run_dir / 'results.json'}")
                    r_steps = int(merged.get("r_local_steps", 5))
                    # Adaptive schedules have no single configured S. Report
                    # their actual final-aggregation S; round_metrics.csv keeps
                    # the complete schedule history.
                    s_visits = int(
                        result["current_visit_in_outer"]
                        if method.startswith("critical")
                        else merged.get("s_client_visits", 1)
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "fold": int(fold),
                            "seed": int(seed),
                            "test_score": score,
                            "test_loss": loss,
                            "r_local_steps": r_steps,
                            "s_client_visits": s_visits,
                            "e_total_steps": r_steps * s_visits,
                            "initial_l_smooth": float(merged["l_smooth"]),
                            "grid_search_selected": (
                                method == "corrected-fixed" and bool(selected_s_by_fold)
                            ),
                            "target_epsilon": (
                                float(merged["target_epsilon"])
                                if method.startswith("critical")
                                else None
                            ),
                            "epsilon_search_selected": (
                                method == "critical" and bool(selected_epsilon_by_fold)
                            ),
                            "current_round": int(result["current_round"]),
                            "current_outer_round": int(
                                result.get("current_outer_round", result["current_round"])
                            ),
                            "current_visit_in_outer": int(
                                result.get("current_visit_in_outer", 1)
                            ),
                            "local_training_stages": int(
                                result.get("local_training_stages", 0)
                            ),
                            "aggregation_count": int(
                                result.get("aggregation_count", 0)
                            ),
                            "client_sgd_steps": int(
                                result.get("client_sgd_steps", 0)
                            ),
                            "current_validation_score": float(
                                result["current_validation_score"]
                            ),
                            "current_validation_loss": float(
                                result["current_validation_loss"]
                            ),
                            "training_evaluated": bool(
                                result.get("training_evaluated", False)
                            ),
                            "current_training_score": (
                                float(result["current_training_score"])
                                if result.get("current_training_score") is not None
                                else None
                            ),
                            "current_training_loss": (
                                float(result["current_training_loss"])
                                if result.get("current_training_loss") is not None
                                else None
                            ),
                            "current_learning_rate": (
                                float(result["current_learning_rate"])
                                if result.get("current_learning_rate") is not None
                                else None
                            ),
                            "learning_rate_nonincreasing": (
                                bool(result["learning_rate_nonincreasing"])
                                if result.get("learning_rate_nonincreasing") is not None
                                else None
                            ),
                            "learning_rate_increase_blocks": (
                                int(result["learning_rate_increase_blocks"])
                                if result.get("learning_rate_increase_blocks") is not None
                                else None
                            ),
                            "evaluation_checkpoint": "current_final",
                            "run_dir": str(run_dir),
                        }
                    )
                    _write_csv(output_root / "paired_results.csv", rows)

    summaries = _summaries(rows)
    _write_csv(output_root / "summary.csv", summaries)
    _write_csv(output_root / "paired_tests.csv", _paired_tests(rows))
    (output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)
