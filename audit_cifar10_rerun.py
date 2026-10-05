"""Audit the declared CIFAR-10 matrix, selection and timing without altering results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from reproduce_paper import ARCHIVE, PROJECT, read_json, sha256

MANIFEST = PROJECT / "experiments" / "cifar10_rerun_20261004.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value, *, positive=False) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and (not positive or value > 0)


def option(command: list[str], name: str) -> str:
    flag = "--" + name.replace("_", "-")
    require(flag in command and command.index(flag) + 1 < len(command), f"missing {flag} in execution record")
    return command[command.index(flag) + 1]


def check_trial(root: Path, path: Path, dataset: dict, fold: int, seed: int,
                phase: str, method: str, **settings) -> tuple[dict, float, float]:
    execution = read_json(path / "execution.json")
    require(execution.get("status") == "completed" and execution.get("returncode") == 0, f"unsuccessful process: {path}")
    require(execution.get("phase") == phase, f"incorrect phase: {path}")
    require(execution.get("run_dir") == path.relative_to(root).as_posix(), f"incorrect run identity: {path}")
    require(finite(execution.get("elapsed_seconds"), positive=True), f"missing measured time: {path}")
    command = execution["command"]
    require(int(option(command, "fold")) == fold and int(option(command, "seed")) == seed, f"fold/seed mismatch: {path}")
    require(option(command, "client_data_path") == dataset["client_data_path"], f"data mismatch: {path}")
    require(option(command, "normalization") == dataset["common"]["normalization"], f"normalization mismatch: {path}")
    config = read_json(path / "config.json")
    require(config.get("seed") == seed and config.get("method") == method, f"config identity mismatch: {path}")
    for key, value in {**dataset["common"], **settings}.items():
        if key == "normalization":
            continue  # This is a loader argument, checked in the command instead.
        if phase == "calibration" and key == "l_smooth_update":
            value = "fixed"
        require(config.get(key) == value, f"config mismatch for {key}: {path}")
    final = phase == "final_evaluation"
    require(config.get("evaluate_test") is final, f"test isolation violated: {path}")
    require(("--skip-test-evaluation" in command) is (not final), f"test flag mismatch: {path}")
    log = path / "process.log"
    whole_seconds = execution["elapsed_seconds"]
    if "attempt" in execution:
        attempt_dirs = sorted((path / "attempts").glob("[0-9]*"))
        require(bool(attempt_dirs), f"missing attempt records: {path}")
        whole_seconds = 0.0
        for attempt_dir in attempt_dirs:
            attempt = read_json(attempt_dir / "execution.json")
            require(finite(attempt.get("elapsed_seconds"), positive=True), f"untimed attempt: {attempt_dir}")
            require(attempt.get("status") == "completed" and attempt.get("returncode") == 0,
                    f"failed attempt needs separate diagnosis/accounting: {attempt_dir}")
            require((attempt_dir / "process.log").is_file(), f"missing attempt log: {attempt_dir}")
            whole_seconds += attempt["elapsed_seconds"]
        log = path / "attempts" / f"{execution['attempt']:04d}" / "process.log"
        require(read_json(log.with_name("execution.json")) == execution, f"latest attempt mismatch: {path}")
    require(log.is_file(), f"missing process log: {path}")
    if phase == "calibration":
        marker = path / "l_smooth_calibration.json"
        result = read_json(marker)
        require(result.get("fold") == fold and result.get("seed") == seed, f"calibration identity mismatch: {path}")
        require(result.get("validation_clients_used") is False and result.get("test_clients_used") is False,
                f"calibration leakage: {path}")
        require(finite(result.get("l_raw"), positive=True), f"invalid calibration: {path}")
        round_seconds = 0.0
    else:
        feasible = path / "results.json"
        infeasible = path / "infeasible_result.json"
        require(feasible.is_file() != infeasible.is_file(), f"ambiguous or absent result marker: {path}")
        marker = feasible if feasible.is_file() else infeasible
        result = read_json(marker)
        require(result.get("test_evaluated") is final, f"result test leakage: {path}")
        if infeasible.is_file():
            require(phase == "tolerance_tuning" and result.get("status") == "infeasible", f"unexpected infeasibility: {path}")
        else:
            require(result.get("current_outer_round") == dataset["common"]["rounds"], f"incomplete training: {path}")
            require(finite(result.get("current_validation_loss")), f"nonfinite validation loss: {path}")
            require(finite(result.get("client_sgd_steps"), positive=True), f"missing SGD count: {path}")
            if final:
                require(finite(result.get("test_loss")), f"nonfinite test loss: {path}")
        rows = []
        if (path / "round_metrics.csv").exists():
            with (path / "round_metrics.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        round_seconds = float(rows[-1]["wall_total_seconds"]) if rows else 0.0
        if feasible.is_file():
            require(len(rows) == dataset["common"]["rounds"] and finite(round_seconds, positive=True), f"missing round clock: {path}")
    if execution.get("result_sha256") is not None:
        require(execution["result_sha256"] == sha256(marker), f"result changed after timing: {path}")
    return result, whole_seconds, round_seconds


def interval(fold_values: list[float]) -> dict:
    mean = statistics.fmean(fold_values)
    half_width = 2.7764451051977987 * statistics.stdev(fold_values) / math.sqrt(5)
    return {"mean": mean, "fold_level_95_percent_interval": [mean - half_width, mean + half_width]}


def audit(root: Path, names: list[str] | None = None) -> dict:
    root = root.resolve()
    manifest = read_json(MANIFEST)
    require(read_json(root / "prospective_manifest.json") == manifest, "results use a different prospective manifest")
    provenance = read_json(root / "runtime_provenance.json")
    require(provenance.get("manifest_sha256", provenance.get("config_sha256")) == sha256(MANIFEST), "manifest checksum mismatch")
    expected_hashes = read_json(ARCHIVE / "source_hashes.json")["files"]
    for name, expected in expected_hashes.items():
        # Earlier server provenance did not hash these two support modules.
        if name in {"language_utils.py", "plot_pytorch_losses.py"} and name not in provenance["source_files"]:
            continue
        require(provenance["source_files"].get(name) == expected, f"incorrect archived source: {name}")
    selected_names = names or list(manifest["datasets"])
    require(bool(selected_names) and len(set(selected_names)) == len(selected_names)
            and set(selected_names) <= manifest["datasets"].keys(), "invalid dataset filter")
    grid_saved = {(r["dataset"], int(r["fold"])): r for r in read_json(root / "grid_search_selection.json")}
    epsilon_saved = {(r["dataset"], int(r["fold"])): r for r in read_json(root / "critical_epsilon_selection.json")}
    counts = {p: 0 for p in ["calibration", "fixed_budget_tuning", "tolerance_tuning", "final_evaluation"]}
    checked_paths, reports = set(), {}
    for name in selected_names:
        d = manifest["datasets"][name]
        base = root / name
        cost_rows, effects = [], {k: [] for k in ["validation_loss", "test_loss", "sgd_percent", "round_clock_percent", "whole_process_percent"]}
        for fold in d["folds"]:
            totals = {p: 0.0 for p in ["calibration", "fixed_budget_tuning", "tolerance_tuning", "fixed_final", "adaptive_final"]}
            calibrations, grid_candidates, epsilon_candidates = [], [], []

            def trial(path, phase, seed, method, **settings):
                result, seconds, round_seconds = check_trial(root, path, d, fold, seed, phase, method, **settings)
                counts[phase] += 1
                checked_paths.add(path)
                return result, seconds, round_seconds

            for seed in d["l_smooth_calibration"]["calibration_seeds"]:
                result, seconds, _ = trial(base / "l_smooth_calibration" / f"fold_{fold}" / f"seed_{seed}", "calibration", seed, "corrected-fixed")
                calibrations.append(result["l_raw"])
                totals["calibration"] += seconds
            initial_l = max(calibrations)
            for s in d["grid_search"]["s_candidates"]:
                losses = []
                for seed in d["grid_search"]["tuning_seeds"]:
                    result, seconds, _ = trial(base / "grid_search" / "corrected-fixed" / f"fold_{fold}" / f"r_5_s_{s}" / f"seed_{seed}",
                                               "fixed_budget_tuning", seed, "corrected-fixed", r_local_steps=5, s_client_visits=s, l_smooth=initial_l)
                    losses.append(result["current_validation_loss"])
                    totals["fixed_budget_tuning"] += seconds
                grid_candidates.append((statistics.fmean(losses), s))
            chosen_s = min(grid_candidates)[1]
            require(grid_saved[(name, fold)]["selected_s_client_visits"] == chosen_s, f"grid selection differs: {name}, fold {fold}")
            for epsilon in d["critical_epsilon_search"]["epsilon_candidates"]:
                token = format(float(epsilon), ".12g").replace("-", "m").replace(".", "p")
                results = []
                for seed in d["critical_epsilon_search"]["tuning_seeds"]:
                    result, seconds, _ = trial(base / "critical_epsilon_search" / "critical" / f"fold_{fold}" / f"epsilon_{token}" / f"seed_{seed}",
                                               "tolerance_tuning", seed, "critical", **{**d["methods"]["critical"], "target_epsilon": epsilon, "l_smooth": initial_l})
                    results.append(result)
                    totals["tolerance_tuning"] += seconds
                if all(r.get("status") != "infeasible" for r in results):
                    epsilon_candidates.append((statistics.fmean(r["current_validation_loss"] for r in results),
                                               statistics.fmean(r["client_sgd_steps"] for r in results), -float(epsilon)))
            require(bool(epsilon_candidates), f"no eligible tolerance: {name}, fold {fold}; report this before any separately versioned expansion")
            chosen_epsilon = -min(epsilon_candidates)[2]
            require(epsilon_saved[(name, fold)]["selected_target_epsilon"] == chosen_epsilon, f"epsilon selection differs: {name}, fold {fold}")
            paired = {k: [] for k in effects}
            for seed in d["seeds"]:
                fixed, fixed_seconds, fixed_round = trial(base / "corrected-fixed" / f"fold_{fold}" / f"seed_{seed}", "final_evaluation", seed,
                    "corrected-fixed", r_local_steps=5, s_client_visits=chosen_s, l_smooth=initial_l)
                adaptive, adaptive_seconds, adaptive_round = trial(base / "critical" / f"fold_{fold}" / f"seed_{seed}", "final_evaluation", seed,
                    "critical", **{**d["methods"]["critical"], "target_epsilon": chosen_epsilon, "l_smooth": initial_l})
                totals["fixed_final"] += fixed_seconds
                totals["adaptive_final"] += adaptive_seconds
                for key, field in [("validation_loss", "current_validation_loss"), ("test_loss", "test_loss")]:
                    paired[key].append(adaptive[field] - fixed[field])
                paired["sgd_percent"].append(100 * (adaptive["client_sgd_steps"] / fixed["client_sgd_steps"] - 1))
                paired["round_clock_percent"].append(100 * (adaptive_round / fixed_round - 1))
                paired["whole_process_percent"].append(100 * (adaptive_seconds / fixed_seconds - 1))
            for key in effects:
                effects[key].append(statistics.fmean(paired[key]))
            cost_rows.append(totals)
        mean_minutes = {key: statistics.fmean(r[key] for r in cost_rows) / 60 for key in cost_rows[0]}
        fixed_total = mean_minutes["calibration"] + mean_minutes["fixed_budget_tuning"] + mean_minutes["fixed_final"]
        adaptive_total = mean_minutes["calibration"] + mean_minutes["tolerance_tuning"] + mean_minutes["adaptive_final"]
        unexpected = {p.parent for p in base.rglob("execution.json") if "attempts" not in p.relative_to(base).parts} - checked_paths
        require(not unexpected, f"unexpected trials outside the declared matrix: {name}")
        reports[name] = {"whole_process_minutes_per_fold": mean_minutes, "fixed_procedure_minutes": fixed_total,
                         "adaptive_procedure_minutes": adaptive_total, "saved_minutes": fixed_total - adaptive_total,
                         "saved_percent": 100 * (1 - adaptive_total / fixed_total),
                         "paired_effects": {key: interval(values) for key, values in effects.items()}}
    return {"status": "passed", "datasets": selected_names, "complete_two_regime_matrix": len(selected_names) == 2,
            "expected_records_checked": counts, "total_records": sum(counts.values()), "results": reports,
            "timing_note": "Shared calibration is charged once in each hypothetical method pipeline. It was measured once; do not double-charge the combined experiment. Whole-process and round-clock times have separate scopes. No new cost is a historical pilot measurement."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", action="append", help="audit one completed regime without certifying the full matrix")
    args = parser.parse_args()
    report = audit(Path(args.root), args.dataset)
    destination = Path(args.root) / ("independent_audit.json" if report["complete_two_regime_matrix"] else "independent_partial_audit.json")
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
