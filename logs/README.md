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

For sharing real training data, use an artifact store, release asset, Git LFS,
or a separate data sync location instead of normal Git commits.
