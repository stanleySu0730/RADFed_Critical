"""Run full paper protocols or explicitly marked bounded pilot manifests."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

RESERVED = {"method", "fold", "seed", "output_dir"}
ALLOWED_METHODS = {
    "fedavg",
    "original-radfed",
    "corrected-fixed",
    "critical",
    "critical-r1",
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _arguments(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        if name in RESERVED or value is None or value is False:
            continue
        flag = _flag(name)
        if value is True:
            result.append(flag)
        elif isinstance(value, list):
            for item in value:
                result.extend([flag, str(item)])
        else:
            result.extend([flag, str(value)])
    return result


def _dataset_manifests(
    manifest: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Return fully resolved dataset manifests.

    The original single-dataset schema remains supported. In the multi-dataset
    schema, ``folds``, ``seeds``, and ``common`` may be supplied as top-level
    defaults and overridden for one dataset. Method dictionaries are kept
    dataset-specific so that paper settings cannot accidentally leak between
    MNIST and CIFAR-10.
    """

    datasets = manifest.get("datasets")
    if datasets is None:
        name = str(manifest.get("dataset_name", "dataset"))
        return [(name, dict(manifest))]
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("manifest datasets must be a nonempty object")

    resolved: list[tuple[str, dict[str, Any]]] = []
    for raw_name, raw_dataset in datasets.items():
        name = str(raw_name)
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"invalid dataset name: {name!r}")
        if not isinstance(raw_dataset, dict):
            raise ValueError(f"dataset {name} must be an object")
        dataset = dict(raw_dataset)
        for key in ("folds", "seeds"):
            if key not in dataset and key in manifest:
                dataset[key] = manifest[key]
        dataset["common"] = {
            **manifest.get("common", {}),
            **dataset.get("common", {}),
        }
        if "grid_search" not in dataset and "grid_search" in manifest:
            dataset["grid_search"] = dict(manifest["grid_search"])
        if (
            "critical_epsilon_search" not in dataset
            and "critical_epsilon_search" in manifest
        ):
            dataset["critical_epsilon_search"] = dict(
                manifest["critical_epsilon_search"]
            )
        if (
            "l_smooth_calibration" not in dataset
            and "l_smooth_calibration" in manifest
        ):
            dataset["l_smooth_calibration"] = dict(
                manifest["l_smooth_calibration"]
            )
        resolved.append((name, dataset))
    return resolved


def _validate_grid_search(name: str, manifest: dict[str, Any]) -> None:
    grid = manifest.get("grid_search")
    if grid is None:
        return
    if not isinstance(grid, dict):
        raise ValueError(f"dataset {name}: grid_search must be an object")
    if not grid.get("enabled", True):
        return
    if grid.get("method", "corrected-fixed") != "corrected-fixed":
        raise ValueError(f"dataset {name}: grid_search method must be corrected-fixed")
    if grid.get("selection_scope", "per_fold") != "per_fold":
        raise ValueError(f"dataset {name}: grid_search selection_scope must be per_fold")
    if grid.get("selection_metric", "current_validation_loss") != "current_validation_loss":
        raise ValueError(
            f"dataset {name}: grid_search must select current_validation_loss"
        )
    if grid.get("tie_breaker", "smaller_e") != "smaller_e":
        raise ValueError(f"dataset {name}: grid_search tie_breaker must be smaller_e")
    candidates = grid.get("s_candidates")
    tuning_seeds = grid.get("tuning_seeds")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(value, int) or value <= 0 for value in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError(
            f"dataset {name}: grid_search s_candidates must be unique positive integers"
        )
    if (
        not isinstance(tuning_seeds, list)
        or not tuning_seeds
        or any(not isinstance(value, int) for value in tuning_seeds)
        or len(set(tuning_seeds)) != len(tuning_seeds)
    ):
        raise ValueError(
            f"dataset {name}: grid_search tuning_seeds must be unique integers"
        )
    if set(tuning_seeds) & set(manifest["seeds"]):
        raise ValueError(
            f"dataset {name}: grid-search and final-evaluation seeds must be disjoint"
        )
    fixed = manifest["methods"].get("corrected-fixed")
    if fixed is None:
        raise ValueError(f"dataset {name}: grid_search requires corrected-fixed")
    if "s_client_visits" in fixed:
        raise ValueError(
            f"dataset {name}: remove corrected-fixed s_client_visits when grid_search is enabled"
        )
    merged_fixed = {**manifest["common"], **fixed}
    r_steps = int(merged_fixed.get("r_local_steps", 5))
    max_total_steps = int(merged_fixed.get("max_total_steps", 250))
    if any(r_steps * int(s_visits) > max_total_steps for s_visits in candidates):
        raise ValueError(
            f"dataset {name}: a grid-search candidate exceeds max_total_steps"
        )


def _validate_critical_epsilon_search(
    name: str, manifest: dict[str, Any]
) -> None:
    search = manifest.get("critical_epsilon_search")
    if search is None:
        return
    if not isinstance(search, dict):
        raise ValueError(
            f"dataset {name}: critical_epsilon_search must be an object"
        )
    if not search.get("enabled", True):
        return
    if search.get("method", "critical") != "critical":
        raise ValueError(
            f"dataset {name}: critical_epsilon_search method must be critical"
        )
    if search.get("selection_scope", "per_fold") != "per_fold":
        raise ValueError(
            f"dataset {name}: critical_epsilon_search selection_scope must be per_fold"
        )
    if search.get("selection_metric", "current_validation_loss") != (
        "current_validation_loss"
    ):
        raise ValueError(
            f"dataset {name}: critical_epsilon_search must select current_validation_loss"
        )
    expected_tie_breaker = "smaller_mean_client_sgd_steps_then_larger_epsilon"
    if search.get("tie_breaker", expected_tie_breaker) != expected_tie_breaker:
        raise ValueError(
            f"dataset {name}: critical_epsilon_search tie_breaker must be "
            f"{expected_tie_breaker}"
        )
    if search.get("require_all_tuning_seeds_feasible", True) is not True:
        raise ValueError(
            f"dataset {name}: every tuning seed must be feasible before selection"
        )
    candidates = search.get("epsilon_candidates")
    tuning_seeds = search.get("tuning_seeds")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in candidates
        )
        or len({float(value) for value in candidates}) != len(candidates)
    ):
        raise ValueError(
            f"dataset {name}: epsilon_candidates must be unique positive finite numbers"
        )
    if (
        not isinstance(tuning_seeds, list)
        or not tuning_seeds
        or any(not isinstance(value, int) or isinstance(value, bool) for value in tuning_seeds)
        or len(set(tuning_seeds)) != len(tuning_seeds)
    ):
        raise ValueError(
            f"dataset {name}: epsilon tuning seeds must be unique integers"
        )
    if set(tuning_seeds) & set(manifest["seeds"]):
        raise ValueError(
            f"dataset {name}: epsilon-tuning and final-evaluation seeds must be disjoint"
        )
    automatic_expansion = search.get("automatic_expansion", False)
    if not isinstance(automatic_expansion, bool):
        raise ValueError(
            f"dataset {name}: automatic_expansion must be true or false"
        )
    if automatic_expansion:
        if search.get("expansion_trigger", "feasibility_only") != "feasibility_only":
            raise ValueError(
                f"dataset {name}: expansion_trigger must be feasibility_only"
            )
        factor = search.get("expansion_factor", 2.0)
        if (
            not isinstance(factor, (int, float))
            or isinstance(factor, bool)
            or not math.isfinite(float(factor))
            or float(factor) <= 1.0
        ):
            raise ValueError(
                f"dataset {name}: expansion_factor must be finite and greater than one"
            )
        max_expansions = search.get("max_expansions", 6)
        if (
            not isinstance(max_expansions, int)
            or isinstance(max_expansions, bool)
            or max_expansions <= 0
        ):
            raise ValueError(
                f"dataset {name}: max_expansions must be a positive integer"
            )
    if "critical" not in manifest["methods"]:
        raise ValueError(
            f"dataset {name}: critical_epsilon_search requires the critical method"
        )


def _validate_l_smooth_calibration(name: str, manifest: dict[str, Any]) -> None:
    calibration = manifest.get("l_smooth_calibration")
    if calibration is None:
        return
    if not isinstance(calibration, dict):
        raise ValueError(f"dataset {name}: l_smooth_calibration must be an object")
    if not calibration.get("enabled", True):
        return
    if calibration.get("estimator", "gradient_secant") != "gradient_secant":
        raise ValueError(
            f"dataset {name}: l_smooth_calibration estimator must be gradient_secant"
        )
    if calibration.get("selection_scope", "per_fold") != "per_fold":
        raise ValueError(
            f"dataset {name}: l_smooth_calibration selection_scope must be per_fold"
        )
    if calibration.get("aggregation", "maximum") != "maximum":
        raise ValueError(
            f"dataset {name}: l_smooth_calibration aggregation must be maximum"
        )
    seeds = calibration.get("calibration_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(value, int) or isinstance(value, bool) for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            f"dataset {name}: calibration_seeds must be unique integers"
        )
    reserved_seeds = set(manifest["seeds"])
    grid = manifest.get("grid_search")
    if isinstance(grid, dict) and grid.get("enabled", True):
        reserved_seeds.update(grid["tuning_seeds"])
    epsilon = manifest.get("critical_epsilon_search")
    if isinstance(epsilon, dict) and epsilon.get("enabled", True):
        reserved_seeds.update(epsilon["tuning_seeds"])
    if set(seeds) & reserved_seeds:
        raise ValueError(
            f"dataset {name}: L-calibration seeds must be disjoint from tuning and final seeds"
        )
    distance = calibration.get("relative_probe_distance", 1e-3)
    if (
        not isinstance(distance, (int, float))
        or isinstance(distance, bool)
        or not math.isfinite(float(distance))
        or float(distance) <= 0
    ):
        raise ValueError(
            f"dataset {name}: relative_probe_distance must be positive and finite"
        )
    if "l_smooth" in manifest["common"]:
        raise ValueError(
            f"dataset {name}: remove hardcoded common.l_smooth when calibration is enabled"
        )
    for method, overrides in manifest["methods"].items():
        if "l_smooth" in overrides:
            raise ValueError(
                f"dataset {name}: remove hardcoded {method}.l_smooth when calibration is enabled"
            )


def _validate_dataset_manifest(
    name: str, manifest: dict[str, Any], *, pilot_mode: bool = False
) -> None:
    required = {"client_data_path", "folds", "seeds", "common", "methods"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"dataset {name} is missing: {', '.join(missing)}")
    folds = manifest["folds"]
    seeds = manifest["seeds"]
    if not isinstance(folds, list) or not isinstance(seeds, list) or not folds or not seeds:
        raise ValueError(f"dataset {name}: folds and seeds must be nonempty lists")
    if not pilot_mode and (len(folds) != 5 or len(seeds) != 3):
        raise ValueError(
            f"dataset {name}: paper protocol requires exactly five folds and three seeds"
        )
    if len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"dataset {name}: fold and seed identifiers must be unique")
    if not isinstance(manifest["methods"], dict) or not manifest["methods"]:
        raise ValueError(f"dataset {name}: methods must be a nonempty object")
    for method, overrides in manifest["methods"].items():
        if method not in ALLOWED_METHODS:
            raise ValueError(f"dataset {name}: unsupported method {method}")
        if not isinstance(overrides, dict):
            raise ValueError(f"dataset {name}: method {method} must be an object")
        merged = {**manifest["common"], **overrides}
        if method.startswith("critical") and merged.get("target_epsilon") is None:
            raise ValueError(
                f"dataset {name}: {method} requires a validation-selected target_epsilon"
            )
        if method.startswith("critical") and float(merged["target_epsilon"]) <= 0:
            raise ValueError(
                f"dataset {name}: {method} target_epsilon must be positive"
            )
        if method not in {"original-radfed", "fedavg"} and merged.get("local_epochs"):
            raise ValueError(f"dataset {name}: local_epochs cannot be used by {method}")
        if merged.get("optimizer", "sgd") != "sgd":
            raise ValueError(f"dataset {name}: {method} must use standard SGD")
    _validate_grid_search(name, manifest)
    _validate_critical_epsilon_search(name, manifest)
    _validate_l_smooth_calibration(name, manifest)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    pilot_mode = manifest.get("pilot_mode", False)
    if not isinstance(pilot_mode, bool):
        raise ValueError("pilot_mode must be true or false")
    for name, dataset in _dataset_manifests(manifest):
        _validate_dataset_manifest(name, dataset, pilot_mode=pilot_mode)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _grid_search_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize tuning runs and select one fixed schedule per outer fold."""

    grouped: dict[tuple[str, int, int, int], list[float]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["fold"]),
            int(row["r_local_steps"]),
            int(row["s_client_visits"]),
        )
        grouped.setdefault(key, []).append(float(row["current_validation_loss"]))

    summaries: list[dict[str, Any]] = []
    for (dataset, fold, r_steps, s_visits), losses in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": dataset,
                "fold": fold,
                "r_local_steps": r_steps,
                "s_client_visits": s_visits,
                "e_total_steps": r_steps * s_visits,
                "tuning_runs": len(losses),
                "current_validation_loss_mean": statistics.fmean(losses),
                "current_validation_loss_sample_std": (
                    statistics.stdev(losses) if len(losses) > 1 else 0.0
                ),
                "selected": False,
            }
        )

    selections: list[dict[str, Any]] = []
    fold_keys = sorted({(str(row["dataset"]), int(row["fold"])) for row in summaries})
    for dataset, fold in fold_keys:
        candidates = [
            row
            for row in summaries
            if row["dataset"] == dataset and int(row["fold"]) == fold
        ]
        selected = min(
            candidates,
            key=lambda row: (
                float(row["current_validation_loss_mean"]),
                int(row["e_total_steps"]),
                int(row["s_client_visits"]),
            ),
        )
        selected["selected"] = True
        selections.append(
            {
                "dataset": dataset,
                "fold": fold,
                "selection_metric": "current_validation_loss",
                "selection_round": "final",
                "tie_breaker": "smaller_e",
                "r_local_steps": int(selected["r_local_steps"]),
                "selected_s_client_visits": int(selected["s_client_visits"]),
                "selected_e_total_steps": int(selected["e_total_steps"]),
                "tuning_runs": int(selected["tuning_runs"]),
                "current_validation_loss_mean": float(
                    selected["current_validation_loss_mean"]
                ),
                "current_validation_loss_sample_std": float(
                    selected["current_validation_loss_sample_std"]
                ),
            }
        )
    return summaries, selections


def _epsilon_token(value: float) -> str:
    """Return a stable, path-safe representation of an epsilon candidate."""

    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _critical_epsilon_summaries(
    rows: list[dict[str, Any]],
    expected_tuning_runs: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize validation-only epsilon trials and select one value per fold."""

    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["fold"]),
            float(row["target_epsilon"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, fold, epsilon), group_rows in sorted(grouped.items()):
        expected = int(expected_tuning_runs[dataset])
        feasible = [row for row in group_rows if row["status"] == "feasible"]
        losses = [float(row["current_validation_loss"]) for row in feasible]
        steps = [float(row["client_sgd_steps"]) for row in feasible]
        all_feasible = len(group_rows) == expected and len(feasible) == expected
        summaries.append(
            {
                "dataset": dataset,
                "fold": fold,
                "target_epsilon": epsilon,
                "expected_tuning_runs": expected,
                "observed_tuning_runs": len(group_rows),
                "feasible_tuning_runs": len(feasible),
                "all_tuning_seeds_feasible": all_feasible,
                "current_validation_loss_mean": (
                    statistics.fmean(losses) if losses else None
                ),
                "current_validation_loss_sample_std": (
                    statistics.stdev(losses) if len(losses) > 1 else 0.0
                ) if losses else None,
                "mean_client_sgd_steps": (
                    statistics.fmean(steps) if steps else None
                ),
                "selected": False,
            }
        )

    selections: list[dict[str, Any]] = []
    fold_keys = sorted(
        {(str(row["dataset"]), int(row["fold"])) for row in summaries}
    )
    for dataset, fold in fold_keys:
        candidates = [
            row
            for row in summaries
            if row["dataset"] == dataset
            and int(row["fold"]) == fold
            and bool(row["all_tuning_seeds_feasible"])
            and row["current_validation_loss_mean"] is not None
        ]
        if not candidates:
            raise ValueError(
                f"dataset {dataset} fold {fold}: no epsilon candidate was feasible "
                "for every validation tuning seed; expand epsilon_candidates"
            )
        selected = min(
            candidates,
            key=lambda row: (
                float(row["current_validation_loss_mean"]),
                float(row["mean_client_sgd_steps"]),
                -float(row["target_epsilon"]),
            ),
        )
        selected["selected"] = True
        selections.append(
            {
                "dataset": dataset,
                "fold": fold,
                "selection_metric": "current_validation_loss",
                "selection_round": "final",
                "tie_breaker": (
                    "smaller_mean_client_sgd_steps_then_larger_epsilon"
                ),
                "selected_target_epsilon": float(selected["target_epsilon"]),
                "tuning_runs": int(selected["feasible_tuning_runs"]),
                "current_validation_loss_mean": float(
                    selected["current_validation_loss_mean"]
                ),
                "current_validation_loss_sample_std": float(
                    selected["current_validation_loss_sample_std"]
                ),
                "mean_client_sgd_steps": float(
                    selected["mean_client_sgd_steps"]
                ),
                "all_tuning_seeds_feasible": True,
                "test_clients_used_for_selection": False,
            }
        )
    return summaries, selections


def _fold_has_fully_feasible_epsilon(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    fold: int,
    expected_tuning_runs: int,
) -> bool:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["dataset"]) == dataset and int(row["fold"]) == fold:
            grouped.setdefault(float(row["target_epsilon"]), []).append(row)
    return any(
        len(candidate_rows) == expected_tuning_runs
        and all(row["status"] == "feasible" for row in candidate_rows)
        for candidate_rows in grouped.values()
    )


def _l_smooth_calibration_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["dataset"]), int(row["fold"])), []
        ).append(row)
    summaries: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for (dataset, fold), group_rows in sorted(grouped.items()):
        estimates = [float(row["l_raw"]) for row in group_rows]
        if not estimates or any(not math.isfinite(value) or value <= 0 for value in estimates):
            raise ValueError(
                f"dataset {dataset} fold {fold}: invalid L-calibration estimates"
            )
        selected = max(estimates)
        summaries.append(
            {
                "dataset": dataset,
                "fold": fold,
                "estimator": "gradient_secant",
                "aggregation": "maximum",
                "calibration_runs": len(estimates),
                "l_raw_min": min(estimates),
                "l_raw_mean": statistics.fmean(estimates),
                "l_raw_sample_std": (
                    statistics.stdev(estimates) if len(estimates) > 1 else 0.0
                ),
                "l_raw_max": selected,
                "selected_l_smooth": selected,
                "validation_clients_used": False,
                "test_clients_used": False,
            }
        )
        selections.append(
            {
                "dataset": dataset,
                "fold": fold,
                "estimator": "gradient_secant",
                "aggregation": "maximum_across_calibration_seeds",
                "selected_l_smooth": selected,
                "calibration_runs": len(estimates),
                "validation_clients_used": False,
                "test_clients_used": False,
            }
        )
    return summaries, selections


def _run_l_smooth_calibration(
    *,
    train_script: Path,
    fold: int,
    seed: int,
    run_dir: Path,
    merged: dict[str, Any],
    relative_probe_distance: float,
    resume: bool,
    label: str,
) -> dict[str, Any]:
    result_path = run_dir / "l_smooth_calibration.json"
    compatible = False
    if resume and result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        compatible = (
            int(previous.get("fold", -1)) == fold
            and int(previous.get("seed", -1)) == seed
            and previous.get("estimator") == "gradient_secant"
            and math.isclose(
                float(previous.get("relative_probe_distance", -1.0)),
                relative_probe_distance,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and str(previous.get("client_data_path"))
            == str(merged["client_data_path"])
            and str(previous.get("model_name"))
            == str(merged.get("model_name", "ffn"))
            and str(previous.get("split_prefix", "fold"))
            == str(merged.get("split_prefix", "fold"))
        )
    if compatible:
        print(f"[{label}] reusing completed L calibration", flush=True)
    else:
        result_path.unlink(missing_ok=True)
        calibration_values = {
            **merged,
            "l_smooth": 1.0,
            "l_smooth_update": "fixed",
            "s_client_visits": 1,
        }
        command = [
            sys.executable,
            str(train_script),
            "--method",
            "corrected-fixed",
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--output-dir",
            str(run_dir),
            *_arguments(calibration_values),
            "--calibrate-l-only",
            "--l-calibration-relative-probe-distance",
            str(relative_probe_distance),
            "--skip-test-evaluation",
            "--skip-model-checkpoint",
        ]
        print(f"[{label}]", flush=True)
        subprocess.run(command, check=True)
    if not result_path.is_file():
        raise FileNotFoundError(f"L calibration produced no result: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    l_raw = float(result["l_raw"])
    if not math.isfinite(l_raw) or l_raw <= 0:
        raise ValueError(f"invalid L calibration in {result_path}")
    if result.get("validation_clients_used") is not False:
        raise ValueError(f"L calibration used validation clients: {result_path}")
    if result.get("test_clients_used") is not False:
        raise ValueError(f"L calibration used test clients: {result_path}")
    return result


def _result_is_compatible(
    result_path: Path,
    *,
    expected: dict[str, Any],
    evaluate_test: bool,
) -> bool:
    if not result_path.is_file():
        return False
    config_path = result_path.with_name("config.json")
    if not config_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("evaluation_checkpoint") != "current_final":
        return False
    if "current_validation_loss" not in result:
        return False
    if bool(result.get("test_evaluated")) != evaluate_test:
        return False
    expected_config = {
        **expected,
        "evaluate_test": evaluate_test,
    }
    if str(config.get("split_prefix", "fold")) != str(
        expected_config.get("split_prefix", "fold")
    ):
        return False
    expected_training_frequency = int(
        expected_config.get("training_eval_frequency", 0)
    )
    if int(config.get("training_eval_frequency", 0)) != expected_training_frequency:
        return False
    if expected_training_frequency > 0 and (
        result.get("training_evaluated") is not True
        or result.get("current_training_loss") is None
    ):
        return False
    return all(
        name not in config or config[name] == value
        for name, value in expected_config.items()
    )


def _infeasible_is_compatible(
    result_path: Path,
    *,
    expected: dict[str, Any],
) -> bool:
    if not result_path.is_file():
        return False
    config_path = result_path.with_name("config.json")
    if not config_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("status") != "infeasible" or result.get("test_evaluated") is not False:
        return False
    expected_config = {**expected, "evaluate_test": False}
    if str(config.get("split_prefix", "fold")) != str(
        expected_config.get("split_prefix", "fold")
    ):
        return False
    return all(
        name not in config or config[name] == value
        for name, value in expected_config.items()
    )


def _run_training(
    *,
    train_script: Path,
    method: str,
    fold: int,
    seed: int,
    run_dir: Path,
    merged: dict[str, Any],
    resume: bool,
    evaluate_test: bool,
    label: str,
    allow_infeasible: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    execution_values = dict(merged)
    if not evaluate_test:
        # Training-loss traces are diagnostics for the held-out final runs.
        # They are not needed for validation-only hyperparameter selection and
        # would substantially increase the tuning matrix's runtime.
        execution_values["training_eval_frequency"] = 0
    result_path = run_dir / "results.json"
    infeasible_path = run_dir / "infeasible_result.json"
    expected = {**execution_values, "method": method, "seed": seed}
    reuse_result = resume and not force_rerun and _result_is_compatible(
        result_path,
        expected=expected,
        evaluate_test=evaluate_test,
    )
    reuse_infeasible = (
        resume
        and not force_rerun
        and allow_infeasible
        and _infeasible_is_compatible(infeasible_path, expected=expected)
    )
    if reuse_result:
        print(f"[{label}] reusing completed result", flush=True)
    elif reuse_infeasible:
        print(f"[{label}] reusing recorded infeasible candidate", flush=True)
    else:
        if resume and (result_path.is_file() or infeasible_path.is_file()):
            print(f"[{label}] rerunning stale incompatible result", flush=True)
        result_path.unlink(missing_ok=True)
        infeasible_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(train_script),
            "--method",
            method,
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--output-dir",
            str(run_dir),
            *_arguments(execution_values),
        ]
        if not evaluate_test:
            command.append("--skip-test-evaluation")
            command.append("--skip-model-checkpoint")
        if allow_infeasible:
            command.append("--allow-infeasible-critical")
        print(f"[{label}]", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            # A CUDA/PyTorch interpreter can occasionally abort during process
            # teardown after the CLI has already written a complete
            # validation-only infeasibility marker. Accept only that narrow,
            # fully validated case; every other nonzero exit remains fatal.
            recorded_infeasible = (
                allow_infeasible
                and _infeasible_is_compatible(
                    infeasible_path,
                    expected=expected,
                )
            )
            if not recorded_infeasible:
                completed.check_returncode()
            print(
                f"[{label}] subprocess exited with {completed.returncode} "
                "after writing a compatible infeasibility marker; continuing",
                flush=True,
            )
    completed_path = result_path if result_path.is_file() else infeasible_path
    if not completed_path.is_file():
        raise FileNotFoundError(f"training produced no result marker: {run_dir}")
    result = json.loads(completed_path.read_text(encoding="utf-8"))
    if result.get("status") == "infeasible":
        if not allow_infeasible or evaluate_test:
            raise ValueError(f"unexpected infeasible final result in {completed_path}")
        return result
    if not math.isfinite(float(result["current_validation_loss"])):
        raise ValueError(f"non-finite validation loss in {result_path}")
    return result


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset", "dataset")) for row in rows})
    for dataset in datasets:
        dataset_rows = [
            row for row in rows if str(row.get("dataset", "dataset")) == dataset
        ]
        for method in sorted({str(row["method"]) for row in dataset_rows}):
            selected = [row for row in dataset_rows if row["method"] == method]
            scores = [float(row["test_score"]) for row in selected]
            losses = [float(row["test_loss"]) for row in selected]
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "runs": len(selected),
                    "test_score_mean": statistics.fmean(scores),
                    "test_score_sample_std": (
                        statistics.stdev(scores) if len(scores) > 1 else 0.0
                    ),
                    "test_loss_mean": statistics.fmean(losses),
                    "test_loss_sample_std": (
                        statistics.stdev(losses) if len(losses) > 1 else 0.0
                    ),
                }
            )
    return summaries


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _paired_signed_rank(
    first: list[float], second: list[float]
) -> tuple[int, float, float, float]:
    if len(first) != len(second):
        raise ValueError("paired samples must have equal length")
    differences = [left - right for left, right in zip(first, second) if left != right]
    if not differences:
        return 0, 0.0, 1.0, 0.0
    ranks = _average_ranks([abs(value) for value in differences])
    observed = sum(rank for rank, difference in zip(ranks, differences) if difference > 0)
    possible = [
        sum(rank for rank, positive in zip(ranks, signs) if positive)
        for signs in itertools.product((False, True), repeat=len(ranks))
    ]
    tolerance = 1e-12
    lower = sum(value <= observed + tolerance for value in possible) / len(possible)
    upper = sum(value >= observed - tolerance for value in possible) / len(possible)
    p_value = min(1.0, 2.0 * min(lower, upper))
    return len(differences), observed, p_value, statistics.fmean(differences)


def _paired_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exact paired tests on test loss (lower is better)."""

    results: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset", "dataset")) for row in rows})
    for dataset in datasets:
        dataset_rows = [
            row for row in rows if str(row.get("dataset", "dataset")) == dataset
        ]
        methods = sorted({str(row["method"]) for row in dataset_rows})
        lookup = {
            (str(row["method"]), int(row["fold"]), int(row["seed"])): float(
                row["test_loss"]
            )
            for row in dataset_rows
        }
        keys = sorted(
            {(int(row["fold"]), int(row["seed"])) for row in dataset_rows}
        )
        for first_method, second_method in itertools.combinations(methods, 2):
            first = [lookup[(first_method, *key)] for key in keys]
            second = [lookup[(second_method, *key)] for key in keys]
            n_nonzero, statistic, p_value, mean_difference = _paired_signed_rank(
                first, second
            )
            results.append(
                {
                    "dataset": dataset,
                    "measure": "test_loss",
                    "lower_is_better": True,
                    "method_a": first_method,
                    "method_b": second_method,
                    "paired_runs": len(keys),
                    "nonzero_differences": n_nonzero,
                    "wilcoxon_w_plus": statistic,
                    "two_sided_exact_p": p_value,
                    "mean_loss_difference_a_minus_b": mean_difference,
                }
            )
    return results


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

    train_script = Path(__file__).resolve().with_name("train_pytorch.py")
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


if __name__ == "__main__":
    main()
