"""Exercise the archived trainer and logged launcher on temporary synthetic clients."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce_paper as launch


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        data = root / "data"
        data.mkdir()
        rng = np.random.default_rng(11)
        for client in range(8):
            x = rng.normal(size=(12, 4))
            y = (x[:, 0] + x[:, 1] > 0).astype(int)
            np.savetxt(data / f"measures_{client}", x, delimiter=",")
            np.savetxt(data / f"labels_{client}", y, delimiter=",", fmt="%d")
        for split, clients in [("tr", range(6)), ("val", [6]), ("te", [7])]:
            np.savetxt(data / f"fold0_{split}_client_ids.lst", list(clients), delimiter=",", fmt="%d")
        reference = launch.read_json(launch.PROJECT / "experiments/mnist_archived_20260901.json")
        dataset = reference["datasets"]["mnist_datacenter"]
        dataset["client_data_path"] = str(data)
        dataset["folds"], dataset["seeds"] = [0], [1]
        dataset["common"].update(device="cpu", hidden_size=8, num_classes=2, batch_size=2,
                                  rounds=2, profile_frequency=1, profile_clients=3, profile_batches=1,
                                  eval_batch_size=128, participation=0.5, normalization="none")
        dataset["grid_search"].update(s_candidates=[1, 2], tuning_seeds=[101])
        dataset["critical_epsilon_search"].update(epsilon_candidates=[1e-20, 1000], tuning_seeds=[101], automatic_expansion=False)
        dataset["l_smooth_calibration"]["calibration_seeds"] = [1001]
        (root / "experiments").mkdir()
        (root / "experiments/smoke.json").write_text(json.dumps({"pilot_mode": True, "datasets": {"synthetic": dataset}}))
        output = root / "results"
        try:
            with patch.object(launch, "PROJECT", root), patch.dict(launch.PROTOCOLS, {"smoke": (launch.ARCHIVE, "smoke.json", None)}):
                for iteration in range(2):
                    with patch.object(sys, "argv", ["reproduce_paper.py", "--protocol", "smoke", "--output-root", str(output), "--resume"]):
                        launch.main()
                    records = list(output.rglob("execution.json"))
                    assert len(records) == 14, "resume repeated a trial or lost its durable record"
                cases = [p for p in output.rglob("execution.json") if "attempts" not in p.parts]
                assert len(cases) == 7
                markers = list(output.rglob("infeasible_result.json"))
                assert len(markers) == 1
                for path in output.rglob("results.json"):
                    result = launch.read_json(path)
                    final = "grid_search" not in path.parts and "critical_epsilon_search" not in path.parts
                    assert result["test_evaluated"] is final
                print("CPU smoke passed: 7 attempts, infeasible candidate logged, test isolation and resume verified.")
        finally:
            os.chdir(original_directory)


if __name__ == "__main__":
    main()
