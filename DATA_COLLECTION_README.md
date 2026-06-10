# FlightSim Dynamics Data Collection README

This document describes how to collect dynamics data from the AI-GP FlightSim
simulator, how the raw data is turned into supervised learning samples, what
the body-centric wrapper does, and how the current MLP/GRU dynamics models are
trained.

## Repository Layout

```text
AIGP_3364/FlightSim.exe                         Flight simulator executable
PyAIPilotExample/                               MAVLink control, telemetry, logging
dynamics/collection/collect_identification_data.py
dynamics/collection/validate_dynamics_log.py
dynamics/collection/launch_flightsim.py
dynamics/pipeline/run_dynamics_pipeline.py
dynamics/dataset/build_dataset.py
dynamics/dataset/catalog_raw_logs.py
dynamics/models/flightsim_wrapper.py
dynamics/models/mlp_dynamics.py
dynamics/training/train_mlp_dynamics.py
dynamics/evaluation/
logs/raw/                                       Raw jsonl collection logs
logs/processed/                                 Train/val/test datasets
logs/coverage/                                  Coverage reports
checkpoints/                                    Trained model checkpoints
configs/train_mlp.yaml                          Last saved MLP-style training config
configs/train_gru.yaml                          Reproducible GRU training config
```

Run all commands from the repository root.

```powershell
cd "C:\Workspace\AI-GP Simulator v1.0.3364"
```

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe --version
```

## Git LFS Deployment for Training Data

This repository is configured to use Git LFS for large training artifacts.
The tracked LFS patterns are stored in `.gitattributes`:

```text
logs/raw/*.jsonl
logs/raw/**/*.jsonl
logs/processed/**/*.npz
checkpoints/**/*.pt
checkpoints/**/*.pth
checkpoints/**/*.ckpt
```

Install and initialize Git LFS once on each machine:

```powershell
git lfs install
git lfs version
```

After cloning the repository, download LFS files with:

```powershell
git lfs pull
```

For a faster clone without downloading large files immediately:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone git@github.com:FieldDiTian/AI-GP-Simulator-v1.0.3364.git
cd AI-GP-Simulator-v1.0.3364
git lfs pull --include "logs/raw/**,logs/processed/**,checkpoints/**"
```

Generated logs and checkpoints are still ignored by `.gitignore` by default.
This is intentional: it prevents accidentally pushing every local run. When you
want to publish a selected dataset or checkpoint through LFS, add it explicitly
with `-f`:

```powershell
# Raw collection logs
git add -f logs\raw\my_run_001.jsonl

# Processed training arrays
git add -f logs\processed\my_dataset\train.npz
git add -f logs\processed\my_dataset\val.npz
git add -f logs\processed\my_dataset\test.npz

# Model weights
git add -f checkpoints\my_model\best_val_model.pt

# Small metadata can also be published when useful
git add -f logs\processed\my_dataset\normalization_stats.json
git add -f logs\processed\my_dataset\dataset_summary.json
git add -f checkpoints\my_model\normalization_stats.json
git add -f checkpoints\my_model\training_curve.csv
```

Before committing, verify that large files are tracked by LFS:

```powershell
git lfs status
git lfs ls-files
```

Then commit and push normally:

```powershell
git commit -m "Add selected FlightSim training data"
git push origin main
```

Do not publish `logs/pipeline_runs`, `logs/ui_screenshots`, or temporary
debug outputs unless they are needed for reproducing a specific issue. GitHub
LFS has storage and bandwidth quotas, so prefer publishing curated runs,
processed datasets, and final checkpoints instead of every exploratory run.

## What Is Collected

The collector connects to FlightSim through MAVLink on `127.0.0.1:14550`,
arms the vehicle, sends `SET_ATTITUDE_TARGET` commands, and writes one jsonl
file per run under `logs/raw/`.

Each raw log starts with an optional metadata row, followed by `sample_type:
"dynamics"` rows. A dynamics row contains:

```text
action                 commanded roll/pitch/yaw rates and thrust
odometry               position, quaternion, linear velocity, body angular rate
local_position_ned     NED velocity when available
imu                    high-rate IMU acceleration
attitude               attitude telemetry
actuator_output        four simulator actuator/motor outputs
collision              collision counter and last collision event
reset_counter          simulator reset segment id
```

`actuator_output.actuator[0:4]` is the simulator's normalized four-motor output.
It is useful as a motor-speed proxy for coverage and supervised one-step
models, but it is not calibrated physical RPM unless a separate RPM calibration
is added.

## Collection Profiles

Profiles are implemented in:

```text
dynamics/collection/collect_identification_data.py
```

Common profiles:

```text
hover_perturb
forward_flight
left_turn
right_turn
climb_descend
roll_step
pitch_step
yaw_step
thrust_step
roll_sine_sweep
pitch_sine_sweep
yaw_sine_sweep
thrust_sine_sweep
smooth_random
race_like
high_roll_pitch_response
coupled_rpyt_combo
high_speed_yaw_roll_correction
low_speed_hover_attitude_perturb
low_speed_feedback_perturb
boundary_approach_recover
climb_descend_turn
high_omega_thrust_response
error_attitude_corner_response
error_pitch_brake_velocity
error_roll_angle_reversal
error_yaw_thrust_grid
error_vertical_velocity_mix
```

For attitude, speed, and motor-output coverage, use a mix of low-speed,
high-speed, thrust, high-angular-rate, coupled-control, and error-targeted
profiles.

## Single-Profile Collection

Use this when FlightSim is already open and in the flyable HUD:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\collect_identification_data.py `
  --profile forward_flight `
  --duration-s 60 `
  --heartbeat-timeout-s 45 `
  --telemetry-timeout-s 45 `
  --log-dir logs\raw `
  --run-id forward_flight_manual_001
```

The collector waits for required telemetry, arms the drone, runs a short
flight-response probe, then starts logging. If the simulator is still in a menu
or not accepting control commands, collection fails before writing normal
training data.

Validate every raw log before training:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\validate_dynamics_log.py `
  logs\raw\forward_flight_manual_001.jsonl `
  --json
```

Do not train on logs that fail validation.

## Recommended Batch Collection Pipeline

The end-to-end orchestrator is:

```text
dynamics/pipeline/run_dynamics_pipeline.py
```

It runs:

```text
launch FlightSim
collect each profile
validate each raw log
move invalid logs to logs/rejected/
build a processed dataset from the valid logs
optionally train a model
optionally evaluate one-step and rollout error
write a coverage report
```

Collection itself still writes raw MAVLink jsonl. The wrapper is not used during
raw collection; it starts at dataset building, evaluation, and coverage.

Example broad sweep for attitude, speed, and motor-output coverage:

```powershell
$run = "att_speed_motor_sweep_$(Get-Date -Format yyyyMMdd_HHmmss)"

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

Use `--skip-train` when the goal is only to collect and inspect data. Remove
`--skip-train` and provide training hyperparameters when you want the pipeline
to train immediately after collection.

The most recent broad sweep in this workspace wrote:

```text
logs/processed/att_speed_motor_sweep_20260609_232058/
logs/processed/full_plus_att_speed_motor_sweep_20260609_232058/
logs/coverage/full_plus_att_speed_motor_sweep_20260609_232058/
```

## Catalog Raw Logs

Create a categorized manifest for all raw logs:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\catalog_raw_logs.py `
  --input logs\raw `
  --catalog-name all_runs_current
```

The catalog is written to:

```text
logs/catalog/<catalog_name>/
  all_runs.json
  summary.json
  categories/<profile_category>.json
  profiles/<profile>.json
```

Use the catalog to build focused datasets, such as only low-speed profiles,
only high-error profiles, or a full dataset after a new top-up collection.

## Build a Dataset From Raw Logs

Dataset construction is implemented in:

```text
dynamics/dataset/build_dataset.py
dynamics/models/flightsim_wrapper.py
```

Build from all raw logs with actuator and IMU included:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\build_dataset.py `
  --input logs\raw `
  --output-dir logs\processed\full_current `
  --k 10 `
  --hz 60 `
  --max-rate 1.0
```

Build the current closed-loop MLP dataset shape, excluding actuator and IMU:

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

The dataset builder:

```text
1. reads raw jsonl rows
2. keeps only rows with required action and telemetry fields
3. groups by run_id and reset_counter
4. drops the collision suffix inside each reset segment
5. drops sustained frozen-position suffixes
6. resamples each segment to a fixed frequency, usually 60 Hz
7. calls FlightSimBodyCentricWrapper
8. builds history windows
9. splits by run into train, validation, and test sets
10. saves train.npz, val.npz, test.npz, normalization_stats.json, dataset_summary.json
```

## What the Wrapper Does

`FlightSimBodyCentricWrapper` converts raw simulator telemetry into a compact
body-centric learning problem. This keeps the model independent of absolute
world position and map location.

For each log row, the wrapper builds a state:

```text
v_body        NED velocity rotated into the body frame
omega_body    body angular rate from odometry
gravity_body  gravity vector expressed in the body frame
imu_acc       optional IMU acceleration
actuator[4]   optional normalized motor/actuator outputs
```

It also normalizes actions:

```text
u = [
  roll_rate_cmd / max_rate,
  pitch_rate_cmd / max_rate,
  yaw_rate_cmd / max_rate,
  thrust_cmd
]
```

For each adjacent frame pair, the wrapper creates the 12-dimensional target:

```text
y = [
  delta_p_body[3],
  delta_rotvec_body[3],
  v_body_next[3],
  omega_body_next[3]
]
```

The model input is a history stack:

```text
[x_{t-K}, u_{t-K}, ..., x_t, u_t] -> y_t
```

With the default `K=10`, there are 11 time steps in each input sample.

Default full-state shape:

```text
state_dim = 16   v_body 3 + omega 3 + gravity 3 + imu 3 + actuator 4
action_dim = 4
input_dim = 11 * (16 + 4) = 220
output_dim = 12
```

Current closed-loop MLP shape:

```text
state_dim = 9    v_body 3 + omega 3 + gravity 3
action_dim = 4
input_dim = 11 * (9 + 4) = 143
output_dim = 12
```

The closed-loop model excludes actuator and IMU because the first MLP does not
predict future actuator feedback or future IMU acceleration during rollout.
Raw logs still keep actuator output, so actuator-aware one-step models can be
trained later.

## Current MLP Training

Training is implemented in:

```text
dynamics/training/train_mlp_dynamics.py
dynamics/models/mlp_dynamics.py
```

The default model type is `mlp`. The MLP is a feed-forward network:

```text
Linear(input_dim, hidden_dim)
SiLU
repeated hidden Linear + SiLU layers
Linear(hidden_dim, 12)
```

The current best rollout-tuned MLP checkpoint uses:

```text
checkpoint: checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
model_type: mlp
input_dim: 143
output_dim: 12
hidden_dim: 512
num_layers: 5
k: 10
state_dim: 9
action_dim: 4
use_actuator: false
use_imu: false
```

It was trained as a closed-loop state model using normalized train-set
statistics from the processed dataset. Inputs and targets are normalized before
loss computation, and the checkpoint stores:

```text
model_state_dict
model_config
normalization_stats
```

The trainer uses AdamW, MSE loss, validation-based early stopping, and writes:

```text
best_val_model.pt
training_curve.csv
normalization_stats.json
```

## Train the Current MLP Style

First build the no-actuator/no-IMU dataset:

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

Then train a baseline MLP:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_current_noact_noimu `
  --checkpoint-dir checkpoints\full_current_mlp_baseline `
  --config-dir configs `
  --model-type mlp `
  --hidden-dim 512 `
  --num-layers 5 `
  --batch-size 2048 `
  --epochs 40 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 6 `
  --device cuda
```

For rollout tuning, initialize from the baseline or train directly with rollout
loss:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_current_noact_noimu `
  --checkpoint-dir checkpoints\full_current_mlp_rolloutloss10 `
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

`--rollout-loss-weight` enables closed-loop training. During rollout loss, the
model predicts the next target, updates the self-contained state
`[v_body, omega_body, gravity_body]`, shifts the history window, inserts the
next logged action, and repeats for the requested number of rollout steps.

The recommended rollout-tuned MLP style is:

```text
model_type: mlp
input_dim: 143
output_dim: 12
hidden_dim: 512
num_layers: 5
k: 10
state_dim: 9
action_dim: 4
use_actuator: False
use_imu: False
learning_rate: 0.0005
batch_size: 2048
epochs: 20
weight_decay: 0.0001
early_stopping_patience: 6
rollout_loss_weight: 0.2
rollout_steps: 10
rollout_batch_size: 512
```

## Train the Configured GRU Style

GRU support is implemented in `dynamics/models/mlp_dynamics.py` and the
reproducible config is saved at `configs/train_gru.yaml`.

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_plus_higherror_closedloop9_20260609 `
  --checkpoint-dir checkpoints\full_plus_higherror_closedloop9_gru_rollout10_20260609 `
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

Current GRU result:

```text
one-step delta_p RMSE: 0.09880 m
raw rollout 1s/3s/5s position RMSE: 3.25 / 19.65 / 37.43 m
```

The GRU configuration is reproducible, but it is not the current best model.
The current best checkpoint remains:

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```

## Train an Actuator-Aware One-Step MLP

If the goal is to use motor/actuator output as an input feature, build without
`--no-actuator` and train the default 16-state model:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\build_dataset.py `
  --input logs\raw `
  --output-dir logs\processed\full_current_with_actuator `
  --k 10 `
  --hz 60 `
  --max-rate 1.0

.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_current_with_actuator `
  --checkpoint-dir checkpoints\full_current_with_actuator_mlp `
  --config-dir configs `
  --model-type mlp `
  --hidden-dim 512 `
  --num-layers 5 `
  --batch-size 1024 `
  --epochs 100 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 10 `
  --device cuda
```

This is appropriate for one-step prediction experiments. For long closed-loop
rollouts, prefer the no-actuator/no-IMU state unless the model is extended to
predict future actuator and IMU features.

## Evaluate and Analyze Coverage

One-step evaluation from a processed test split:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_one_step.py `
  --checkpoint checkpoints\full_current_mlp_rolloutloss30\best_val_model.pt `
  --data logs\processed\full_current_noact_noimu\test.npz `
  --device cuda
```

Raw-log rollout evaluation through the wrapper:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\full_current_mlp_rolloutloss30\best_val_model.pt `
  --raw-input logs\raw `
  --horizons-s 1 3 5 `
  --device cuda `
  --no-actuator `
  --no-imu
```

Coverage without a checkpoint:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\analyze_coverage.py `
  --input logs\raw `
  --output-dir logs\coverage\full_current `
  --k 10 `
  --hz 60 `
  --max-rate 1.0 `
  --min-samples-per-bin 50
```

Coverage with model error bins:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\analyze_coverage.py `
  --input logs\raw `
  --output-dir logs\coverage\full_current_with_model `
  --k 10 `
  --hz 60 `
  --max-rate 1.0 `
  --min-samples-per-bin 50 `
  --checkpoint checkpoints\full_current_mlp_rolloutloss30\best_val_model.pt `
  --device cuda `
  --no-actuator `
  --no-imu
```

Use coverage reports to choose the next collection profiles. Low-count bins
need more data; high-error bins need targeted data in that state/action region.

## Practical Notes

Use `--relaunch-per-profile --close-after-profile` for long batch collection so
each profile starts from a clean simulator state.

Use longer durations, usually 60 to 120 seconds per profile, for coverage runs.
Short smoke runs are only for checking that the pipeline works.

Keep invalid or collision-dominated logs out of training. The dataset builder
can drop collision suffixes, but a bad run still wastes storage and can skew
coverage.

When changing `--k`, `--hz`, `--no-actuator`, or `--no-imu`, rebuild the dataset
before training because the model input dimension and normalization statistics
change.

When using a checkpoint for initialization, the dataset input dimension and
model type must match the checkpoint. For example, a 143-dimensional no-actuator
MLP checkpoint cannot initialize a 220-dimensional actuator-aware MLP.
