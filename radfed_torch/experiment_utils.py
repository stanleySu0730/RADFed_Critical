"""Shared experiment helpers: configuration checks, validation selection, and numerical summaries."""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
import statistics
from typing import Any

RESERVED = {"method", "fold", "seed", "output_dir"}
ALLOWED_METHODS = {"fedavg", "original-radfed", "corrected-fixed", "critical", "critical-r1"}

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
