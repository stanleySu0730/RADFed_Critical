# Supplement verification: 5 October 2026

- All 12 archived image implementation/support files match the experiment artifact's SHA-256 inventory. The prospective CIFAR manifest is byte-identical to the server's declared manifest.
- All five named protocols pass manifest, source and seed-separation validation. Covertype and Shakespeare entries match their saved experiment manifests. Maintained training sources and all four explicit Kaiming FFN entries are unchanged.
- Seven focused checks pass: complete 450-record matrix accounting, missing infeasible records, selection eligibility, test leakage, partial-audit scope, immutable protocol files, and preservation of failed/successful attempt records.
- A CPU smoke run executes seven archived attempts: calibration, fixed-budget tuning, tolerance tuning including an infeasible candidate, and selected final evaluations. A second invocation reuses all seven completed attempts. Test clients appear only in final results.
- The independent CIFAR audit passes all 225 completed datacenter records, replays validation selection and recovers the previously verified whole-process procedure totals. This is a completed-regime audit, not certification of the full two-regime rerun.
- All Python sources parse. Text/code files were scanned for identifying local paths, private server addresses and credentials. Generated results and runtime logs are excluded from Git.

The local smoke/check environment uses Python 3.12, NumPy 2.5.1, Pillow 12.3.0, torch 2.13.0+cpu and torchvision 0.28.0+cpu. The recorded server experiment uses CUDA builds on a GeForce GTX 1080, as documented in the runtime provenance. This supplement update does not rerun the full GPU matrix or supply missing historical pilot measurements.
