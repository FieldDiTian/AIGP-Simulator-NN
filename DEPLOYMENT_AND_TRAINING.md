# FlightSim Dynamics Deployment and Training Guide

This guide is for a fresh clone. It covers Git LFS data, Python/CUDA setup,
evaluation, training, dataset rebuilds, and new FlightSim data collection.

## 1. Verified Remote Contents

Remote repository:

```text
git@github.com:FieldDiTian/AI-GP-Simulator-v1.0.3364.git
```

Uploaded training stack:

```text
dynamics/                 collection, dataset, training, evaluation, models
logs/raw/                 raw MAVLink jsonl logs, Git LFS
logs/processed/           train/val/test npz datasets, Git LFS
logs/catalog/             run/profile/category manifests
logs/coverage/            coverage and high-error reports
checkpoints/              trained model weights, Git LFS
configs/train_mlp.yaml    current rollout-tuned MLP config
configs/train_gru.yaml    reproducible GRU config
requirements.txt          non-Torch Python dependencies
```

Large files are tracked by `.gitattributes`:

```text
logs/raw/*.jsonl
logs/raw/**/*.jsonl
logs/processed/**/*.npz
checkpoints/**/*.pt
checkpoints/**/*.pth
checkpoints/**/*.ckpt
```

## 2. System Requirements

Training from uploaded datasets can run without FlightSim. New data collection
requires Windows and the simulator executable.

Verified local stack:

```text
OS: Windows 11
Python: 3.14.5
GPU: NVIDIA GeForce RTX 4090 Laptop GPU
Torch: 2.9.0+cu129
CUDA runtime reported by torch: 12.9
Git LFS: 3.7.1
```

Minimum practical setup:

```text
Windows 10/11 for FlightSim collection
NVIDIA GPU and recent NVIDIA driver for CUDA training
Python 3.14.x
Git
Git LFS
PowerShell
```

FlightSim itself is not committed:

```text
AIGP_3364/FlightSim.exe
```

`AIGP_3364/` is intentionally ignored by Git. A new user must install or copy
the simulator locally only if they need to collect new data.

## 3. Clone and Pull LFS Data

Install Git LFS once:

```powershell
git lfs install
git lfs version
```

Clone and download uploaded training artifacts:

```powershell
git clone git@github.com:FieldDiTian/AI-GP-Simulator-v1.0.3364.git
cd "AI-GP-Simulator-v1.0.3364"
git lfs pull
```

Faster clone with explicit data pull:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone git@github.com:FieldDiTian/AI-GP-Simulator-v1.0.3364.git
cd "AI-GP-Simulator-v1.0.3364"
git lfs pull --include "logs/raw/**,logs/processed/**,checkpoints/**"
```

Verify that LFS data is present:

```powershell
git lfs ls-files
Test-Path logs\processed\full_plus_higherror_closedloop9_20260609\train.npz
Test-Path checkpoints\full_plus_higherror_closedloop9_rolloutloss10_20260609\best_val_model.pt
```

## 4. Python and CUDA Setup

Create a virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CUDA PyTorch explicitly, then install the rest:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

If the CUDA 12.9 wheel is not available on the target machine, install the
matching PyTorch CUDA wheel recommended by the PyTorch installer page, then keep
the remaining commands unchanged.

Verify CUDA:

```powershell
python -c "import torch, numpy, pymavlink, cv2, matplotlib; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected for CUDA training:

```text
torch.cuda.is_available() -> True
```

## 5. Current Dataset and Models

Recommended closed-loop dataset:

```text
logs/processed/full_plus_higherror_closedloop9_20260609/
```

It uses the rollout-safe state:

```text
state = [v_body(3), omega_body(3), gravity_body(3)]
state_dim = 9
action_dim = 4
K = 10
input_dim = 11 * (9 + 4) = 143
target_dim = 12
```

Current best uploaded MLP:

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```

Configured GRU:

```text
checkpoints/full_plus_higherror_closedloop9_gru_rollout10_20260609/best_val_model.pt
```

The GRU is reproducible, but the current best model is still the rollout-tuned
MLP.

## 6. Verify Evaluation After Clone

One-step evaluation:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_one_step.py `
  --checkpoint checkpoints\full_plus_higherror_closedloop9_rolloutloss10_20260609\best_val_model.pt `
  --data logs\processed\full_plus_higherror_closedloop9_20260609\test.npz `
  --device cuda
```

Bounded raw-log rollout check:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\full_plus_higherror_closedloop9_rolloutloss10_20260609\best_val_model.pt `
  --raw-input logs\raw `
  --horizons-s 1 3 5 `
  --max-rollouts-per-horizon 100 `
  --rollout-stride 600 `
  --position-integration delta `
  --device cuda
```

## 7. Train the Current MLP

Full current MLP style:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_plus_higherror_closedloop9_20260609 `
  --checkpoint-dir checkpoints\full_plus_higherror_closedloop9_rolloutloss10_retrain `
  --config-dir configs `
  --model-type mlp `
  --hidden-dim 512 `
  --num-layers 5 `
  --batch-size 2048 `
  --epochs 20 `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --patience 6 `
  --device cuda `
  --rollout-loss-weight 0.2 `
  --rollout-steps 10 `
  --rollout-batch-size 512
```

Quick CUDA training smoke test:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\smoke_4profiles_cuda `
  --checkpoint-dir checkpoints\deploy_smoke_verify `
  --config-dir configs `
  --model-type mlp `
  --hidden-dim 64 `
  --num-layers 2 `
  --batch-size 256 `
  --epochs 1 `
  --patience 1 `
  --device cuda
```

## 8. Train the Configured GRU

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_plus_higherror_closedloop9_20260609 `
  --checkpoint-dir checkpoints\full_plus_higherror_closedloop9_gru_retrain `
  --config-dir configs `
  --model-type gru `
  --hidden-dim 384 `
  --num-layers 3 `
  --dropout 0.05 `
  --epochs 24 `
  --batch-size 2048 `
  --patience 6 `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --device cuda `
  --rollout-loss-weight 0.2 `
  --rollout-steps 10 `
  --rollout-batch-size 512
```

## 9. Rebuild Dataset From Uploaded Raw Logs

The current rollout-friendly dataset excludes actuator and IMU because the
model does not predict those future exogenous features during closed-loop
rollout:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\build_dataset.py `
  --input logs\raw `
  --output-dir logs\processed\full_current_noact_noimu `
  --k 10 `
  --hz 60 `
  --max-rate 1.0 `
  --no-actuator `
  --no-imu
```

Cleaning is reset-aware:

```text
collision/crash row and following rows are dropped only inside the current run_id/reset_counter segment
post-reset rows become a new clean segment and can be used
sustained frozen-position suffixes are dropped
```

## 10. Collect New Data

Copy or install the simulator locally:

```text
AIGP_3364/FlightSim.exe
```

UI-auto launcher:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode ui-auto `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 3 `
  --restart-delay-s 2 `
  --save-screenshots
```

UI sequence:

```text
PRESS ANY BUTTON -> SUBMIT -> AVAILABLE -> RACE
```

The launcher waits for the flyable HUD, not just MAVLink heartbeat.

Broad collection pipeline:

```powershell
$run = "new_sweep_$(Get-Date -Format yyyyMMdd_HHmmss)"

.\.venv\Scripts\python.exe dynamics\pipeline\run_dynamics_pipeline.py `
  --profiles `
    low_speed_hover_attitude_perturb `
    low_speed_feedback_perturb `
    thrust_step `
    thrust_sine_sweep `
    high_omega_thrust_response `
    error_yaw_thrust_grid `
    high_roll_pitch_response `
    error_roll_angle_reversal `
    high_speed_yaw_roll_correction `
    error_vertical_velocity_mix `
    coupled_rpyt_combo `
  --duration-s 120 `
  --run-prefix $run `
  --dataset-name $run `
  --coverage-name $run `
  --skip-train `
  --relaunch-per-profile `
  --close-after-profile `
  --profile-attempts 2 `
  --launch-attempts 3 `
  --launch-timeout-s 90 `
  --heartbeat-timeout-s 30 `
  --telemetry-timeout-s 30 `
  --coverage-min-samples-per-bin 50 `
  --save-ui-screenshots
```

## 11. Publish New Data

Generated logs and checkpoints are ignored by default. Publish selected
artifacts explicitly:

```powershell
git add -f logs\raw\my_run.jsonl
git add -f logs\processed\my_dataset\train.npz
git add -f logs\processed\my_dataset\val.npz
git add -f logs\processed\my_dataset\test.npz
git add -f logs\processed\my_dataset\dataset_summary.json
git add -f logs\processed\my_dataset\normalization_stats.json
git add -f checkpoints\my_run\best_val_model.pt
git add -f checkpoints\my_run\normalization_stats.json
git add -f checkpoints\my_run\training_curve.csv
git lfs status
git commit -m "Add selected FlightSim training data"
git push origin main
```

Do not publish `logs/ui_screenshots/`, `logs/pipeline_runs/`, or temporary
debug outputs unless they are needed to reproduce a specific issue.
