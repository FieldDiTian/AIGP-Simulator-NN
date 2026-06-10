# Logs Directory

This directory is intentionally kept out of normal Git tracking because raw
FlightSim logs, processed datasets, coverage reports, and pipeline outputs can
be several gigabytes.

Tracked files here only preserve the directory layout:

```text
logs/raw/             raw jsonl collection logs
logs/processed/       train/val/test datasets and normalization stats
logs/coverage/        coverage and model-error reports
logs/catalog/         categorized raw-log manifests
logs/pipeline_runs/   pipeline stdout/stderr/status files
logs/rejected/        invalid rejected raw logs
logs/ui_probe/        UI probing artifacts
logs/ui_screenshots/  launcher failure/ready screenshots
```

For sharing selected real training data, use Git LFS. This repository tracks
these large artifact patterns through `.gitattributes`:

```text
logs/raw/*.jsonl
logs/raw/**/*.jsonl
logs/processed/**/*.npz
checkpoints/**/*.pt
checkpoints/**/*.pth
checkpoints/**/*.ckpt
```

Because generated data is ignored by default, intentionally add selected files
with `git add -f ...`, then confirm with `git lfs status` before committing.
