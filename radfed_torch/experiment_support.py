"""Variant-specific experiment checks and trial execution; shared selectors are in experiment_utils."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from .experiment_utils import ALLOWED_METHODS, _arguments, _dataset_manifests, _validate_grid_search, _validate_l_smooth_calibration

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
