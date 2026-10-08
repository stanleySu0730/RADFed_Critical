"""Run one training or calibration job; main.py runs the full paper matrix."""
from __future__ import annotations

import argparse
import importlib
import sys


def main() -> None:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--implementation", choices=["standard", "image"], default="standard")
    args, remaining = selector.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    module = "radfed_torch.image.run_pipeline" if args.implementation == "image" else "radfed_torch.run_pipeline"
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
