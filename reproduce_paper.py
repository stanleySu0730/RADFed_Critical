"""Run a named paper protocol with a separate implementation and durable timings."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parent
ARCHIVE = PROJECT / "protocols" / "archived_images_20260901"
PROTOCOLS = {
    "mnist-archive": (ARCHIVE, "mnist_archived_20260901.json", None),
    "cifar10-archive": (ARCHIVE, "cifar10_archived_20260901.json", None),
    "cifar10-rerun": (ARCHIVE, "cifar10_rerun_20261004.json", None),
    "covertype": (PROJECT, "covertype_shakespeare.json", "covfeat_l_"),
    "shakespeare": (PROJECT, "covertype_shakespeare.json", "shakespeare_transformer_"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_archive() -> None:
    for name, expected in read_json(ARCHIVE / "source_hashes.json")["files"].items():
        if sha256(ARCHIVE / name) != expected:
            raise ValueError(f"archived source was modified: {name}")


def source_hashes(source: Path) -> dict[str, str]:
    paths = [source / "train_pytorch.py", source / "run_paper_experiments.py",
             source / "language_utils.py", *sorted((source / "radfed_torch").glob("*.py"))]
    if (source / "plot_pytorch_losses.py").exists():
        paths.append(source / "plot_pytorch_losses.py")
    return {p.relative_to(source).as_posix(): sha256(p) for p in paths}


def load_runner(source: Path):
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("paper_protocol_runner", source / "run_paper_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime() -> dict:
    import numpy
    import PIL
    import torch
    import torchvision
    return {
        "python": sys.version, "numpy": numpy.__version__, "Pillow": PIL.__version__,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }


def append_event(root: Path, record: dict) -> None:
    with (root / "execution_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def documented_run(command, *args, output_root: Path, original_run, **kwargs):
    if "--output-dir" not in command:
        return original_run(command, *args, **kwargs)
    run_dir = Path(command[command.index("--output-dir") + 1]).resolve()
    relative = run_dir.relative_to(output_root).as_posix()
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = len(list(attempts.glob("[0-9]*"))) + 1
    attempt_dir = attempts / f"{attempt:04d}"
    attempt_dir.mkdir()
    record = {
        "run_dir": relative, "attempt": attempt,
        "command": [str(value) for value in command], "started_at_utc": now(), "status": "running",
        "phase": ("calibration" if "--calibrate-l-only" in command else
                  "fixed_budget_tuning" if "/grid_search/" in relative else
                  "tolerance_tuning" if "/critical_epsilon_search/" in relative else "final_evaluation"),
        "elapsed_time_scope": "whole subprocess including imports, loading, profiling, evaluation and saving",
    }

    def save():
        write_json(attempt_dir / "execution.json", record)
        write_json(run_dir / "execution.json", record)
        append_event(output_root, record)

    save()
    check = kwargs.pop("check", False)
    started = time.perf_counter()
    try:
        with (attempt_dir / "process.log").open("w", encoding="utf-8") as log:
            completed = original_run(command, *args, stdout=log, stderr=subprocess.STDOUT, check=False, **kwargs)
        record.update(ended_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                      returncode=completed.returncode,
                      status="completed" if completed.returncode == 0 else "failed")
        for filename, status in [("infeasible_result.json", "infeasible"),
                                 ("results.json", "feasible"), ("l_smooth_calibration.json", "calibrated")]:
            marker = run_dir / filename
            if marker.is_file():
                record.update(scientific_status=status, result_marker=filename, result_sha256=sha256(marker))
                break
    except BaseException as error:
        record.update(ended_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                      status="failed", failure_type=type(error).__name__, failure_message=str(error))
        save()
        raise
    save()
    print(f"Recorded {relative}: {record['status']}, {record['elapsed_seconds']:.2f}s", flush=True)
    if check and completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--dataset", action="append", default=[], help="one or more entries within this protocol")
    parser.add_argument("--output-root", help="defaults to results/<protocol>; never reuse historical archives")
    parser.add_argument("--resume", action="store_true", help="reuse compatible, logged completed trials")
    parser.add_argument("--validate-only", action="store_true", help="check source hashes, manifest and seed separation without training")
    args = parser.parse_args()
    source, manifest_name, prefix = PROTOCOLS[args.protocol]
    if source == ARCHIVE:
        verify_archive()
    config = PROJECT / "experiments" / manifest_name
    manifest = read_json(config)
    runner = load_runner(source)
    runner._validate_manifest(manifest)
    available = {n for n in manifest["datasets"] if prefix is None or n.startswith(prefix)}
    selected = set(args.dataset) if args.dataset else available
    if not selected or not selected <= available:
        raise ValueError(f"choose dataset entries from {sorted(available)}")
    for name in selected:
        dataset = manifest["datasets"][name]
        groups = [set(dataset["seeds"])]
        for key, seed_key in [("grid_search", "tuning_seeds"), ("critical_epsilon_search", "tuning_seeds"),
                              ("l_smooth_calibration", "calibration_seeds")]:
            if key in dataset:
                group = set(dataset[key][seed_key])
                if group & groups[0]:
                    raise ValueError(f"selection and final seeds overlap: {name}")
                if key == "l_smooth_calibration" and any(group & other for other in groups):
                    raise ValueError(f"calibration seeds overlap: {name}")
                groups.append(group)
    if args.validate_only:
        print(json.dumps({"status": "validated", "protocol": args.protocol,
                          "datasets": sorted(selected), "implementation": source.relative_to(PROJECT).as_posix(),
                          "manifest_sha256": sha256(config)}, indent=2))
        return
    os.chdir(PROJECT)
    for name in selected:
        dataset = manifest["datasets"][name]
        data = Path(dataset["client_data_path"])
        for fold in dataset["folds"]:
            for split in ["tr", "val", "te"]:
                if not (data / f"fold{fold}_{split}_client_ids.lst").is_file():
                    raise FileNotFoundError(f"missing fold input: {data}/fold{fold}_{split}_client_ids.lst")
        checkpoint = dataset["common"].get("mobilenet_weights_path")
        if checkpoint and sha256(Path(checkpoint)) != "7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f":
            raise ValueError("MobileNetV2 checkpoint checksum differs")
    root = Path(args.output_root or f"results/{args.protocol}").resolve()
    if root == PROJECT or root in PROJECT.parents or root == ARCHIVE or ARCHIVE in root.parents:
        raise ValueError("output root must not overwrite the source tree")
    root.mkdir(parents=True, exist_ok=True)
    provenance = {"protocol": args.protocol, "manifest_sha256": sha256(config), "source_files": source_hashes(source)}
    previous_path = root / "runtime_provenance.json"
    if previous_path.exists():
        previous = read_json(previous_path)
        if not args.resume:
            raise ValueError("output exists; choose a new output root or --resume")
        if any(previous.get(k) != v for k, v in provenance.items()):
            raise ValueError("resume would mix different protocols or training/selection sources")
        for marker in [*root.rglob("results.json"), *root.rglob("infeasible_result.json"),
                       *root.rglob("l_smooth_calibration.json")]:
            if not marker.with_name("execution.json").exists():
                raise ValueError("cannot add new measured costs to unlogged runs; choose a new output root")
    elif any(root.iterdir()):
        raise ValueError("output root contains untracked results; choose an empty output root")
    lock = root / ".run.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(handle)
    original_run, original_argv = subprocess.run, sys.argv
    started = time.perf_counter()
    state = {"status": "running", "started_at_utc": now(), "datasets": sorted(selected)}
    try:
        environment = runtime()
        if previous_path.exists() and read_json(previous_path).get("runtime") != environment:
            raise ValueError("runtime changed; use a separate output root for comparable timings")
        if not previous_path.exists():
            write_json(previous_path, {**provenance, "runtime": environment, "started_at_utc": now()})
            (root / "prospective_manifest.json").write_bytes(config.read_bytes())
        write_json(root / "job_state.json", state)
        sys.argv = [str(source / "run_paper_experiments.py"), "--config", str(config), "--output-root", str(root)]
        for name in sorted(selected):
            sys.argv.extend(["--dataset", name])
        if args.resume:
            sys.argv.append("--resume")
        subprocess.run = lambda command, *a, **kw: documented_run(command, *a, output_root=root, original_run=original_run, **kw)
        runner.main()
        state["status"] = "training_complete_pending_audit"
    except BaseException as error:
        state.update(status="failed", failure_type=type(error).__name__, failure_message=str(error))
        raise
    finally:
        subprocess.run, sys.argv = original_run, original_argv
        state.update(ended_at_utc=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(root / "job_state.json", state)
        lock.unlink()


if __name__ == "__main__":
    main()
