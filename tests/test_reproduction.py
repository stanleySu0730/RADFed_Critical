"""Check protocol integrity and reject incomplete or contaminated experiment records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_cifar10_rerun as audit_module
import reproduce_paper as launch


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class LoggingTests(unittest.TestCase):
    def test_failed_attempt_and_successful_retry_are_both_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "dataset" / "grid_search" / "trial"
            command = ["python", "train.py", "--output-dir", str(run)]

            def fake(command, *args, **kwargs):
                kwargs["stdout"].write("training log\n")
                return subprocess.CompletedProcess(command, 1)

            with self.assertRaises(subprocess.CalledProcessError):
                launch.documented_run(command, output_root=root, original_run=fake, check=True)
            first = launch.read_json(run / "attempts/0001/execution.json")
            self.assertEqual(first["status"], "failed")
            self.assertGreater(first["elapsed_seconds"], 0)

            def succeeds(command, *args, **kwargs):
                dump(run / "results.json", {"status": "feasible"})
                return subprocess.CompletedProcess(command, 0)

            launch.documented_run(command, output_root=root, original_run=succeeds, check=True)
            latest = launch.read_json(run / "execution.json")
            self.assertEqual(latest["attempt"], 2)
            self.assertEqual(latest["result_sha256"], launch.sha256(run / "results.json"))
            self.assertEqual(launch.read_json(run / "attempts/0001/execution.json"), first)
            self.assertTrue((run / "attempts/0002/process.log").exists())

    def test_archived_source_and_prospective_manifest_are_immutable(self):
        launch.verify_archive()
        self.assertEqual(launch.sha256(audit_module.MANIFEST),
                         "766396807b6f62bad8bccb2f2b17fedd8bf70466b81389112dcce06e73f2c872")
        manifest = launch.read_json(audit_module.MANIFEST)
        for dataset in manifest["datasets"].values():
            self.assertEqual(dataset["critical_epsilon_search"]["epsilon_candidates"], [50, 75, 100, 150, 200, 300])
            self.assertFalse(dataset["critical_epsilon_search"].get("automatic_expansion", False))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        manifest = launch.read_json(audit_module.MANIFEST)
        dump(self.root / "prospective_manifest.json", manifest)
        dump(self.root / "runtime_provenance.json", {
            "manifest_sha256": launch.sha256(audit_module.MANIFEST),
            "source_files": launch.read_json(launch.ARCHIVE / "source_hashes.json")["files"],
        })
        selections, epsilons = [], []
        for name, dataset in manifest["datasets"].items():
            for fold in dataset["folds"]:
                base = self.root / name
                for seed in dataset["l_smooth_calibration"]["calibration_seeds"]:
                    self.trial(base / "l_smooth_calibration" / f"fold_{fold}" / f"seed_{seed}", dataset, fold, seed,
                               "calibration", "corrected-fixed")
                for s in dataset["grid_search"]["s_candidates"]:
                    for seed in dataset["grid_search"]["tuning_seeds"]:
                        self.trial(base / "grid_search/corrected-fixed" / f"fold_{fold}/r_5_s_{s}/seed_{seed}",
                                   dataset, fold, seed, "fixed_budget_tuning", "corrected-fixed",
                                   s_client_visits=s, r_local_steps=5, l_smooth=1.0, loss=abs(s - 2) + 1)
                for epsilon in dataset["critical_epsilon_search"]["epsilon_candidates"]:
                    for seed in dataset["critical_epsilon_search"]["tuning_seeds"]:
                        self.trial(base / "critical_epsilon_search/critical" / f"fold_{fold}/epsilon_{epsilon}/seed_{seed}",
                                   dataset, fold, seed, "tolerance_tuning", "critical",
                                   **{**dataset["methods"]["critical"], "target_epsilon": epsilon, "l_smooth": 1.0},
                                   loss=0.0 if epsilon == 50 else abs(epsilon - 100) + 1,
                                   infeasible=epsilon == 50 and seed == 103)
                for method in dataset["methods"]:
                    for seed in dataset["seeds"]:
                        settings = {"r_local_steps": 5, "s_client_visits": 2} if method == "corrected-fixed" else {
                            **dataset["methods"]["critical"], "target_epsilon": 100}
                        self.trial(base / method / f"fold_{fold}/seed_{seed}", dataset, fold, seed,
                                   "final_evaluation", method, **settings, l_smooth=1.0)
                selections.append({"dataset": name, "fold": fold, "selected_s_client_visits": 2})
                epsilons.append({"dataset": name, "fold": fold, "selected_target_epsilon": 100})
        dump(self.root / "grid_search_selection.json", selections)
        dump(self.root / "critical_epsilon_selection.json", epsilons)

    def trial(self, path, dataset, fold, seed, phase, method, *, loss=1.0, infeasible=False, **settings):
        final = phase == "final_evaluation"
        config = {**dataset["common"], **settings, "method": method, "seed": seed, "evaluate_test": final}
        if phase == "calibration":
            config["l_smooth_update"] = "fixed"
        dump(path / "config.json", config)
        command = ["python", "train.py", "--fold", str(fold), "--seed", str(seed), "--client-data-path",
                   dataset["client_data_path"], "--normalization", dataset["common"]["normalization"]]
        if not final:
            command.append("--skip-test-evaluation")
        dump(path / "execution.json", {"status": "completed", "returncode": 0, "phase": phase,
                                      "run_dir": path.relative_to(self.root).as_posix(), "command": command,
                                      "elapsed_seconds": 1.0})
        (path / "process.log").write_text("synthetic fixture log\n")
        if phase == "calibration":
            dump(path / "l_smooth_calibration.json", {"fold": fold, "seed": seed, "l_raw": 1.0,
                 "validation_clients_used": False, "test_clients_used": False})
        elif infeasible:
            dump(path / "infeasible_result.json", {"status": "infeasible", "test_evaluated": False})
        else:
            dump(path / "results.json", {"current_outer_round": 30, "current_validation_loss": loss,
                                        "client_sgd_steps": 900, "test_evaluated": final, "test_loss": 1.0})
            with (path / "round_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["wall_total_seconds"])
                writer.writeheader()
                writer.writerows({"wall_total_seconds": (i + 1) / 100} for i in range(30))

    def test_complete_matrix_counts_shared_calibration_and_rejects_ineligible_best_loss(self):
        report = audit_module.audit(self.root)
        self.assertEqual(report["total_records"], 450)
        self.assertEqual(report["expected_records_checked"]["calibration"], 30)
        for result in report["results"].values():
            self.assertAlmostEqual(result["fixed_procedure_minutes"], 0.4)
            self.assertAlmostEqual(result["adaptive_procedure_minutes"], 0.4)

    def test_missing_infeasible_execution_record_is_rejected(self):
        path = self.root / "cifar10_datacenter/critical_epsilon_search/critical/fold_0/epsilon_50/seed_103/execution.json"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            audit_module.audit(self.root)

    def test_test_leakage_in_tuning_is_rejected(self):
        path = self.root / "cifar10_datacenter/grid_search/corrected-fixed/fold_0/r_5_s_1/seed_101/results.json"
        value = launch.read_json(path)
        value["test_evaluated"] = True
        dump(path, value)
        with self.assertRaisesRegex(ValueError, "test leakage"):
            audit_module.audit(self.root)

    def test_selecting_one_infeasible_seed_is_rejected(self):
        path = self.root / "critical_epsilon_selection.json"
        rows = launch.read_json(path)
        rows[0]["selected_target_epsilon"] = 50
        dump(path, rows)
        with self.assertRaisesRegex(ValueError, "epsilon selection differs"):
            audit_module.audit(self.root)

    def test_partial_audit_does_not_certify_full_matrix(self):
        report = audit_module.audit(self.root, ["cifar10_datacenter"])
        self.assertFalse(report["complete_two_regime_matrix"])
        self.assertEqual(report["total_records"], 225)


if __name__ == "__main__":
    unittest.main()
