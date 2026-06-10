# FlightSim Black-Box Dynamics Data Chain

This document explains the dynamics data chain currently used in this project:

```text
Collect FlightSim black-box inputs and outputs
-> build body-centric training samples
-> train the first MLP dynamics surrogate with CUDA
-> run one-step / rollout smoke evaluation
```

The first model is trained to learn:

```text
past K body-frame states + past K control inputs
-> next-step body-centric motion target
```

Model form:

```text
f_theta(x_{t-K:t}^B, u_{t-K:t}) -> y_t
```

The control input is fixed to the four MAVLink `SET_ATTITUDE_TARGET` fields:

```python
u_t = [
    roll_rate_cmd,
    pitch_rate_cmd,
    yaw_rate_cmd,
    thrust_cmd,
]
```

Images, gates, track status, race status, and collision are not model inputs.
Collision/reset information is recorded only to filter contaminated data.
The filter is reset-aware: the collision/crash row and the following rows are
dropped only inside the current `run_id/reset_counter` segment. If FlightSim
automatically resets and `reset_counter` changes, the post-reset rows are
treated as a new clean segment and may be used for training. The same cleaning
stage also drops sustained frozen-position suffixes, because these usually mean
that the aircraft is stuck on a boundary while controls are still being sent.

## 1. Overall Data Flow

```text
FlightSim.exe
  -> UI automatically enters the flight HUD
  -> MAVLink heartbeat / telemetry ready
  -> collect_identification_data.py generates a control profile
  -> controller.py sends SET_ATTITUDE_TARGET
  -> mavlink_rx.py receives FlightSim telemetry
  -> logger.py writes raw jsonl
  -> validate_dynamics_log.py checks whether the aircraft really moved
  -> flightsim_wrapper.py packs rows into a unified body-centric representation
  -> build_dataset.py converts rows into a training dataset
  -> train_mlp_dynamics.py trains an MLP with CUDA
  -> eval_one_step.py / eval_rollout.py / analyze_coverage.py evaluate and find undercovered regions
```

Important: `telemetry ready` does not mean the simulator is already in the
flyable state. The launcher now checks both:

```text
1. the flight HUD is visible
2. FlightSim telemetry is being received
```

The flight HUD check looks for the in-race screen, including `FLIGHT MODE ACRO`
and the speed HUD.

## 2. What Is Collected

Raw logs are jsonl files:

```text
logs/raw/*.jsonl
```

Each row is a sample near one control cycle. The core fields are:

```python
{
    "sample_type": "dynamics",
    "run_id": "...",
    "t_wall_ns": ...,
    "t_boot_ms": ...,
    "dt": ...,

    "action": {
        "control_mode": "SET_ATTITUDE_TARGET",
        "roll_rate_cmd": ...,
        "pitch_rate_cmd": ...,
        "yaw_rate_cmd": ...,
        "thrust_cmd": ...
    },

    "odometry": {
        "time_usec": ...,
        "frame_id": ...,
        "child_frame_id": ...,
        "p_ned": [x, y, z],
        "q_wxyz": [qw, qx, qy, qz],
        "v": [vx, vy, vz],
        "omega": [rollspeed, pitchspeed, yawspeed],
        "reset_counter": ...
    },

    "local_position_ned": {
        "p_ned": [x, y, z],
        "v_ned": [vx, vy, vz]
    },

    "imu": {
        "acc": [xacc, yacc, zacc],
        "gyro": [xgyro, ygyro, zgyro]
    },

    "attitude": {
        "roll": ...,
        "pitch": ...,
        "yaw": ...,
        "rollspeed": ...,
        "pitchspeed": ...,
        "yawspeed": ...
    },

    "actuator_output": {
        "actuator": [...]
    },

    "collision": {
        "collision_count": ...,
        "last_collision": ...
    }
}
```

In this log:

```text
action is the black-box input
telemetry is the black-box output / observed state
```

So each raw log records:

```text
the actual control u_t sent to FlightSim
+
the state x_t returned by FlightSim
```

## 3. Directory Layout

Dynamics-specific code is grouped under:

```text
dynamics/
  collection/
    launch_flightsim.py
    collect_identification_data.py
    validate_dynamics_log.py
  dataset/
    build_dataset.py
  training/
    train_mlp_dynamics.py
  evaluation/
    eval_one_step.py
    eval_rollout.py
    plot_rollout.py
    analyze_coverage.py
  models/
    dynamics_math.py
    flightsim_wrapper.py
    mlp_dynamics.py
    neural_sim.py
```

The repository root keeps project entry documents, the original competition
template, logs, datasets, and checkpoints.

Raw logs:

```text
logs/raw/
```

Invalid or rejected logs:

```text
logs/rejected/
```

Processed datasets:

```text
logs/processed/<dataset_name>/
  train.npz
  val.npz
  test.npz
  normalization_stats.json
  dataset_summary.json
```

Model checkpoints:

```text
checkpoints/<run_name>/
  best_val_model.pt
  normalization_stats.json
  training_curve.csv
```

Current CUDA smoke-test dataset:

```text
logs/processed/smoke_4profiles_cuda/
```

Current CUDA smoke-test checkpoint:

```text
checkpoints/smoke_4profiles_cuda/best_val_model.pt
```

The four valid smoke-test raw logs are:

```text
logs/raw/smoke_forward_flight_20260604_225556.jsonl
logs/raw/smoke_left_turn_20260604_232857.jsonl
logs/raw/smoke_right_turn_20260604_233044.jsonl
logs/raw/smoke_climb_descend_20260604_233227.jsonl
```

A previous log where the simulator was not in flight and the aircraft did not
move has been moved to:

```text
logs/rejected/smoke_forward_flight_20260604_223202.not_in_flight.jsonl
```

## 4. Starting FlightSim and Entering the Simulation

Automatic UI mode:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode ui-auto `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 2 `
  --restart-delay-s 2 `
  --save-screenshots
```

Automatic click sequence:

```text
PRESS ANY BUTTON -> SUBMIT -> AVAILABLE -> RACE
```

`ui-auto` currently uses:

```text
pywin32 to foreground the simulator window
Win32 SendInput / pydirectinput / pyautogui to send game input
pyautogui screenshots to detect the HUD
```

If one UI attempt times out, `--attempts` makes the launcher close FlightSim,
wait `--restart-delay-s`, restart the `.exe`, and run the UI sequence again.

If UI automation is unstable on a machine, manually enter the flight HUD and run:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode manual-ready `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 2 `
  --restart-delay-s 2
```

## 5. Data Collection

Collection script:

```text
dynamics/collection/collect_identification_data.py
```

Supported profiles:

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

Targeted coverage-fill profiles correspond to:

```text
large roll / large pitch attitude response
combined roll + pitch + yaw + thrust commands
yaw / roll correction during high-speed forward flight
low-speed / hover attitude perturbations
low-speed perturbations with telemetry-based braking
normal dynamics near a boundary before collision
climb / descent while turning
thrust response at high angular rate
```

`error_*` profiles are second-pass coverage profiles derived from coverage/error
reports, not ordinary repeated action profiles:

```text
error_attitude_corner_response: gravity_body corner bins, i.e. large roll/pitch attitudes
error_pitch_brake_velocity: high v_body_x and pitch-rate error regions
error_roll_angle_reversal: reverse roll-rate commands at extreme roll angles
error_yaw_thrust_grid: extreme yaw-rate and thrust combinations
error_vertical_velocity_mix: vertical velocity mixed with pitch/thrust
```

Each raw jsonl metadata row includes:

```text
profile
profile_category
coverage_goal
```

Use these fields or the catalog manifests for later cleaning and top-up
collection. Do not infer a run's purpose only from its filename.

Collect one forward-flight run:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\collect_identification_data.py `
  --profile forward_flight `
  --duration-s 8 `
  --heartbeat-timeout-s 45 `
  --telemetry-timeout-s 45 `
  --log-dir logs\raw `
  --run-id smoke_forward_flight_test
```

Before normal samples are written to raw jsonl, the collector runs a short
flight-response probe. It sends a small rate/thrust command and requires
measurable position or velocity change. If the simulator is still in a menu, if
only heartbeat/telemetry exists, or if FlightSim is not accepting
`SET_ATTITUDE_TARGET`, collection fails before writing training samples.

For the special `low_speed_hover_attitude_perturb` profile, once UI automation
is known to be stable, `--skip-flight-response-probe` can be used so the probe
does not give the aircraft initial speed before the official log starts.

Validate motion after collection:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\validate_dynamics_log.py `
  logs\raw\smoke_forward_flight_test.jsonl `
  --json
```

If the result contains:

```json
{
  "valid": false
}
```

do not add that log to training.

### 5.0 Raw Log Catalog

Generate a raw-log catalog:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\catalog_raw_logs.py `
  --input logs\raw `
  --catalog-name all_runs_current
```

Output:

```text
logs/catalog/<catalog_name>/
  all_runs.json
  summary.json
  categories/<profile_category>.json
  profiles/<profile>.json
```

The catalog stores paths and metadata only; it does not copy raw jsonl files.
Use it to filter data by:

```text
baseline_*
targeted_*
high_error_*
single_axis_*
low_speed
```

### 5.1 One-Command Automation Pipeline

The orchestrator is:

```text
dynamics/pipeline/run_dynamics_pipeline.py
```

It connects the following chain:

```text
FlightSim UI ready
-> collect profiles
-> validate_dynamics_log.py checks whether the aircraft actually moved
-> invalid logs are moved to logs/rejected
-> build_dataset.py builds the dataset
-> train_mlp_dynamics.py trains with CUDA
-> eval_one_step.py compares raw logs through the wrapper
-> eval_rollout.py compares closed-loop rollout through the wrapper
-> analyze_coverage.py reports coverage and model error
```

Collection still writes only raw MAVLink jsonl. The wrapper is intentionally
not used during collection because raw logs are the original ground truth and
can be reprocessed later. The wrapper is used automatically from the
dataset/evaluation/coverage stages onward.

Dry-run the command chain:

```powershell
.\.venv\Scripts\python.exe dynamics\pipeline\run_dynamics_pipeline.py `
  --skip-launch `
  --profiles forward_flight `
  --duration-s 1 `
  --run-prefix dryrun_test `
  --dry-run
```

Run a UI-auto smoke pipeline:

```powershell
.\.venv\Scripts\python.exe dynamics\pipeline\run_dynamics_pipeline.py `
  --launch-mode ui-auto `
  --launch-attempts 2 `
  --profile-attempts 2 `
  --restart-delay-s 2 `
  --relaunch-per-profile `
  --close-after-profile `
  --profiles forward_flight left_turn right_turn climb_descend `
  --duration-s 8 `
  --run-prefix smoke_pipeline `
  --hidden-dim 64 `
  --num-layers 2 `
  --batch-size 256 `
  --epochs 2 `
  --patience 2 `
  --device cuda `
  --save-ui-screenshots
```

For formal multi-profile coverage collection, use:

```text
--relaunch-per-profile
```

Then each profile launches FlightSim, enters the flight HUD, collects,
validates, and closes the simulator independently, avoiding contamination from
the previous run's state or collision.

If FlightSim is already manually in the flight HUD, skip launch/UI:

```powershell
.\.venv\Scripts\python.exe dynamics\pipeline\run_dynamics_pipeline.py `
  --skip-launch `
  --profiles forward_flight left_turn right_turn climb_descend `
  --duration-s 8 `
  --run-prefix manual_ready_pipeline `
  --device cuda
```

## 6. Building the Training Dataset

Dataset builder:

```text
dynamics/dataset/build_dataset.py
dynamics/models/flightsim_wrapper.py
```

The standard coordinate conversion and target definition live in:

```python
FlightSimBodyCentricWrapper
```

It packs raw FlightSim logs into:

```text
BodyCentricFrame:
  p_ned
  q_wxyz
  x_body = [v_body, omega_body, gravity_body, imu_acc, actuator]
  u = [roll_rate_cmd_norm, pitch_rate_cmd_norm, yaw_rate_cmd_norm, thrust_cmd]
  collision_count
  reset_counter

BodyCentricTarget:
  y = [delta_p_body, delta_rotvec_body, v_body_next, omega_body_next]
```

The purpose is:

```text
keep raw MAVLink logs as original truth
generate training datasets from the wrapper
compare MLP predictions against wrapper targets in one-step evaluation
use the same body-centric definition for coverage analysis
change state definitions in one place by editing the wrapper
```

The wrapper converts raw jsonl into:

```text
(x_{t-K:t}^B, u_{t-K:t}) -> y_t
```

Dataset processing steps:

```text
1. read raw jsonl
2. remove rows missing action / telemetry
3. split by run_id and reset_counter
4. inside each reset segment, drop the suffix starting at the collision row
5. inside each reset segment, drop sustained frozen-position boundary-stuck suffixes
6. keep new post-reset segments for training
7. resample to a fixed frequency, default 60 Hz
8. call FlightSimBodyCentricWrapper
9. normalize quaternions
10. convert NED velocity to body velocity
11. compute gravity_body
12. compute delta_p_body
13. compute delta_rotvec_body
14. build K=10 history windows
15. save train/val/test and mean/std statistics
```

Build the current four-profile smoke dataset:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\build_dataset.py `
  --input `
    logs\raw\smoke_forward_flight_20260604_225556.jsonl `
    logs\raw\smoke_left_turn_20260604_232857.jsonl `
    logs\raw\smoke_right_turn_20260604_233044.jsonl `
    logs\raw\smoke_climb_descend_20260604_233227.jsonl `
  --output-dir logs\processed\smoke_4profiles_cuda `
  --k 10
```

Common options:

```text
--input        input raw jsonl files; accepts multiple paths
--output-dir   output dataset directory
--k            history length, default 10
--hz           resampling frequency, default 60
--max-rate     rate-command normalization scale
--no-actuator  do not use actuator_output; state changes from 16D to 12D
--stuck-position-epsilon-m  displacement threshold for frozen-position detection, default 1e-4 m
--stuck-min-duration-s      duration before dropping a frozen-position suffix, default 0.5 s
```

If `--k` or `--no-actuator` changes, rebuild the dataset because the MLP input
dimension changes.

## 7. Training Input

Default per-step state is 16-dimensional:

```python
x_t = [
    v_body_x, v_body_y, v_body_z,
    omega_body_x, omega_body_y, omega_body_z,
    gravity_body_x, gravity_body_y, gravity_body_z,
    imu_acc_x, imu_acc_y, imu_acc_z,
    actuator_0, actuator_1, actuator_2, actuator_3,
]
```

Control input is 4-dimensional:

```python
u_t = [
    roll_rate_cmd,
    pitch_rate_cmd,
    yaw_rate_cmd,
    thrust_cmd,
]
```

With `K=10`, one training sample contains 11 time steps:

```text
[x_{t-10}, u_{t-10}, ..., x_t, u_t]
```

With actuator:

```text
state_dim = 16
action_dim = 4
input_dim = 11 * (16 + 4) = 220
```

Without actuator:

```text
state_dim = 12
action_dim = 4
input_dim = 11 * (12 + 4) = 176
```

## 8. Training Output

The model predicts a 12-dimensional target:

```python
y_t = [
    delta_p_body_x,
    delta_p_body_y,
    delta_p_body_z,

    delta_rotvec_body_x,
    delta_rotvec_body_y,
    delta_rotvec_body_z,

    v_body_next_x,
    v_body_next_y,
    v_body_next_z,

    omega_body_next_x,
    omega_body_next_y,
    omega_body_next_z,
]
```

In other words, the model does not directly predict absolute NED position or
absolute quaternion. It predicts:

```text
relative displacement in body coordinates
relative rotation in body coordinates
next body velocity
next body angular velocity
```

## 9. Starting CUDA Training

Training script:

```text
dynamics/training/train_mlp_dynamics.py
```

Model definition:

```text
dynamics/models/mlp_dynamics.py
```

CUDA smoke training:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\smoke_4profiles_cuda `
  --checkpoint-dir checkpoints\smoke_4profiles_cuda `
  --config-dir configs `
  --hidden-dim 64 `
  --num-layers 2 `
  --batch-size 256 `
  --epochs 2 `
  --patience 2 `
  --device cuda
```

Larger first-pass training:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\<dataset_name> `
  --checkpoint-dir checkpoints\<run_name> `
  --config-dir configs `
  --hidden-dim 512 `
  --num-layers 5 `
  --batch-size 1024 `
  --epochs 100 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 10 `
  --device cuda
```

## 10. Changing MLP Hyperparameters

If someone says `NLP` here, read it as the `MLP` dynamics network.

Training hyperparameters are command-line options:

```text
--hidden-dim      MLP hidden width
--num-layers      number of MLP layers
--batch-size      batch size
--epochs          maximum number of epochs
--lr              learning rate
--weight-decay    AdamW weight decay
--patience        early stopping patience
--device          cuda or cpu
```

Example: 3 layers, hidden size 256, 20 epochs:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\smoke_4profiles_cuda `
  --checkpoint-dir checkpoints\smoke_mlp_256x3 `
  --config-dir configs `
  --hidden-dim 256 `
  --num-layers 3 `
  --batch-size 512 `
  --epochs 20 `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --patience 5 `
  --device cuda
```

Dataset-related hyperparameters are changed in `build_dataset.py`:

```text
--k
--hz
--max-rate
--no-actuator
```

These change the training samples themselves, so rebuild the dataset before
training.

## 11. Evaluation

One-step evaluation:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_one_step.py `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --data logs\processed\smoke_4profiles_cuda\test.npz `
  --device cuda
```

Raw-jsonl one-step evaluation through the wrapper:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_one_step.py `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --raw-input `
    logs\raw\smoke_forward_flight_20260604_225556.jsonl `
    logs\raw\smoke_left_turn_20260604_232857.jsonl `
    logs\raw\smoke_right_turn_20260604_233044.jsonl `
    logs\raw\smoke_climb_descend_20260604_233227.jsonl `
  --device cuda
```

That path is:

```text
raw FlightSim log
-> FlightSimBodyCentricWrapper
-> y_sim
-> MLP(x_history)
-> y_hat
-> error = y_hat - y_sim
```

Rollout evaluation:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\<run_name>\best_val_model.pt `
  --data logs\processed\<dataset_name>\test.npz `
  --device cuda
```

Raw-log closed-loop rollout through `FlightSimBodyCentricWrapper` and
`NeuralDroneDynamics`:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --raw-input `
    logs\raw\smoke_forward_flight_20260604_225556.jsonl `
    logs\raw\smoke_left_turn_20260604_232857.jsonl `
    logs\raw\smoke_right_turn_20260604_233044.jsonl `
    logs\raw\smoke_climb_descend_20260604_233227.jsonl `
  --horizons-s 1 3 5 `
  --device cuda
```

That path is:

```text
raw FlightSim log
-> wrapper gets true body-centric history and true action sequence
-> NeuralDroneDynamics resets from the true initial state
-> MLP closed-loop step(action)
-> compare against the FlightSim reference trajectory in position, velocity, attitude, and angular velocity
```

Plot rollout:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\plot_rollout.py `
  --rollout-json <rollout_result.json> `
  --output <plot_path.png>
```

Coverage and model-error analysis:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\analyze_coverage.py `
  --input `
    logs\raw\smoke_forward_flight_20260604_225556.jsonl `
    logs\raw\smoke_left_turn_20260604_232857.jsonl `
    logs\raw\smoke_right_turn_20260604_233044.jsonl `
    logs\raw\smoke_climb_descend_20260604_233227.jsonl `
  --output-dir logs\coverage\smoke_4profiles_cuda `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --device cuda `
  --k 10 `
  --min-samples-per-bin 20
```

Output:

```text
logs/coverage/<name>/coverage_report.json
logs/coverage/<name>/undercovered_bins.json
```

The first version does not bin the full high-dimensional joint space. It uses
low-dimensional views:

```text
gravity_body_x vs gravity_body_y
v_body_x vs v_body_z
omega_body_x vs omega_body_y
roll_rate_cmd vs pitch_rate_cmd
yaw_rate_cmd vs thrust_cmd
v_body_x vs pitch_rate_cmd
roll angle vs roll_rate_cmd
pitch angle vs thrust_cmd
```

Each bin records:

```text
sample_count
valid_sample_count
collision_filtered_count
mean_action
target_std
model_error_mean
```

Undercoverage is not only low sample count; high model error also matters:

```text
valid_sample_count < min_samples_per_bin
or model_error_mean is clearly high
```

Without `--checkpoint`, the script only analyzes data coverage. With a
checkpoint, it also maps one-step MLP error onto the same bins.

## 12. Current Smoke-Test Results

Four valid collections:

```text
forward_flight: 457 complete rows, displacement 30.175 m, max speed 34.101 m/s
left_turn:      459 complete rows, displacement 163.545 m, max speed 28.524 m/s
right_turn:     459 complete rows, displacement 163.265 m, max speed 28.526 m/s
climb_descend:  458 complete rows, displacement 168.602 m, max speed 28.811 m/s
```

Dataset summary:

```json
{
  "raw_rows": 1833,
  "segments": 4,
  "samples": 1850,
  "train_samples": 1387,
  "val_samples": 0,
  "test_samples": 463,
  "dropped_windows": 0
}
```

CUDA smoke training:

```text
epoch=001 train_loss=0.806750 val_loss=0.535139
epoch=002 train_loss=0.446449 val_loss=0.343431
```

One-step smoke evaluation:

```json
{
  "samples": 463,
  "delta_p_body_rmse_m": 0.2616,
  "delta_rotvec_body_rmse_rad": 0.00997,
  "v_body_next_rmse_mps": 15.03,
  "omega_body_next_rmse_radps": 0.532
}
```

These results only prove that the pipeline runs. They do not prove that the
model is good. Real training needs longer runs, broader attitude/velocity/
angular-rate coverage, and train/val/test split by run.

## 13. Current High-Error Fill Results

Top-up run prefix:

```text
high_error_fill_20260605_154651
```

Five new raw-log groups:

```text
high_error_attitude
high_error_velocity_pitch
high_error_roll_angle
high_error_yaw_thrust
high_error_vertical_velocity
```

Categorized catalog:

```text
logs/catalog/all_with_high_error_fill_20260605_154651/
```

Full rebuilt dataset:

```text
logs/processed/full_plus_higherror_stuckclean_20260605_154651/
```

Dataset summary:

```json
{
  "raw_rows": 296090,
  "segments": 70,
  "samples": 220158,
  "train_samples": 158610,
  "val_samples": 20357,
  "test_samples": 41191,
  "collision_filtered_rows": 81597,
  "stuck_position_filtered_rows": 573
}
```

CUDA baseline checkpoint:

```text
checkpoints/full_plus_higherror_baseline30_20260605_154651/best_val_model.pt
```

One-step test:

```json
{
  "delta_p_body_rmse_m": 0.0988,
  "delta_rotvec_body_rmse_rad": 0.00503,
  "v_body_next_rmse_mps": 0.612,
  "omega_body_next_rmse_radps": 0.0547
}
```

This means one-step position-delta error is just below 0.1 m, but rollout is
still unreliable:

```text
1s rollout position RMSE: 9.59 m
3s rollout position RMSE: 56.93 m
5s rollout position RMSE: 108.73 m
```

Therefore the next priority is not simply more ordinary flight time. Focus on:

```text
high speed + large vertical velocity
extreme roll attitude + opposite roll-rate commands
extreme yaw-rate + low/high thrust
large pitch attitude + thrust limits
```

Inspect these regions through:

```text
logs/coverage/full_plus_higherror_stuckclean_20260605_154651_with_baseline30_model/
```

## 14. Rollout Tuning Results

The first rollout failure was not simply caused by insufficient data. The model
input included exogenous quantities that it cannot update during rollout:

```text
imu_acc
actuator_output
```

In one-step training these values come from real logs, so the metrics look good.
During rollout, however, the model can only reuse stale values, making the
history increasingly false.

The current recommended closed-loop dataset removes those exogenous quantities
and keeps only:

```text
x_body = [
  v_body_x, v_body_y, v_body_z,
  omega_body_x, omega_body_y, omega_body_z,
  gravity_body_x, gravity_body_y, gravity_body_z
]
```

Build command:

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\build_dataset.py `
  --input logs\raw `
  --output-dir logs\processed\full_plus_higherror_closedloop9_20260609 `
  --k 10 `
  --hz 60 `
  --max-rate 1.0 `
  --no-actuator `
  --no-imu
```

Training adds multi-step rollout loss so the model rolls forward 10 steps using
its own predicted state:

```powershell
.\.venv\Scripts\python.exe dynamics\training\train_mlp_dynamics.py `
  --data-dir logs\processed\full_plus_higherror_closedloop9_20260609 `
  --checkpoint-dir checkpoints\full_plus_higherror_closedloop9_rolloutloss10_20260609 `
  --hidden-dim 512 `
  --num-layers 5 `
  --epochs 20 `
  --batch-size 2048 `
  --lr 5e-4 `
  --patience 6 `
  --device cuda `
  --rollout-loss-weight 0.2 `
  --rollout-steps 10 `
  --rollout-batch-size 512
```

Current recommended checkpoint:

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```

One-step test:

```json
{
  "delta_p_body_rmse_m": 0.09836,
  "delta_rotvec_body_rmse_rad": 0.00500,
  "v_body_next_rmse_mps": 0.342,
  "omega_body_next_rmse_radps": 0.0419
}
```

Formal raw-log rollout, sampled across runs with a large stride:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\full_plus_higherror_closedloop9_rolloutloss10_20260609\best_val_model.pt `
  --raw-input logs\raw `
  --horizons-s 1 3 5 `
  --max-rollouts-per-horizon 1000 `
  --rollout-stride 600 `
  --position-integration delta `
  --device cuda
```

Same evaluation protocol, old 16D model vs new closed-loop9 + rollout loss:

```text
old 1s/3s/5s position RMSE: 4.62 / 30.07 / 57.29 m
new 1s/3s/5s position RMSE: 2.16 / 14.09 / 26.95 m
```

Rollout improved substantially, but it is still not good enough to replace
FlightSim. Next steps:

```text
trace high-rollout-error windows back to their profiles
add low-speed and mid-speed stable segments instead of only high-speed extremes
try GRU/LSTM or explicit delay state
use longer multi-step loss carefully with small weight; strong 30-step loss already showed overfitting/regression
```

## 15. GRU Configuration

GRU support is configured in:

```text
dynamics/models/mlp_dynamics.py
configs/train_gru.yaml
```

GRU and MLP use the same closed-loop9 dataset:

```text
logs/processed/full_plus_higherror_closedloop9_20260609/
```

Training command:

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

Current GRU checkpoint:

```text
checkpoints/full_plus_higherror_closedloop9_gru_rollout10_20260609/best_val_model.pt
```

Measured result:

```text
one-step delta_p RMSE: 0.09880 m
raw rollout 1s/3s/5s position RMSE: 3.25 / 19.65 / 37.43 m
```

Conclusion: GRU training and evaluation are reproducible, but the current GRU
does not beat closed-loop9 MLP + 10-step rollout loss. The current best remains:

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```
