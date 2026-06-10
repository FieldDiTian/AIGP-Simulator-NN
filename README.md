## AI Grand Prix (AI-GP) Development Kit
Conceived by Anduril founder Palmer Luckey and partnered with the Drone Champions League (DCL), Neros Technologies, and JobsOhio, AI-GP is a premier autonomous drone racing competition.
This global challenge invites elite engineers and teams of up to 8 people to design, build, and deploy autonomy software capable of piloting high-speed racing drones through professional-grade courses—with absolutely zero human intervention.
For complete competition details and updates, visit the official website at www.theaigrandprix.com.

## 🏆 Competition Highlights

* The Stakes: Compete for a share of a $500,000 prize pool and career opportunities at Anduril.
* The Hardware: Complete competitive parity. All teams utilize identical racing drones built by Neros Technologies incorporating DCL's AI vector module.
* The Mission: Program the ultimate AI pilot to conquer dynamic, real-world flight conditions using onboard vision sensing—no GPS or absolute coordinate data will be provided.

------------------------------
## 📁 Repository Contents
This package contains the foundational tools required to develop, test, and qualify your autonomous flight software.
## 1. AIGP_X.zip (The Simulator)
This archive contains the official AI-GP flight simulator environment for Windows.

* Setup: Extract the ZIP archive to your local directory.
* Execution: Launch the simulator by running FlightSim.exe from the unzipped root folder.
* Authentication: Access the virtual qualifier within the simulator by logging in with your official simulator account credentials.

## 2. PyAIPilotExample.zip (The Code Template)
This archive provides a starter template to help you interface with the simulator and write your autonomous flight algorithms.

* Environment: Tested and verified on Python 3.14.2.
* Setup:
1. Unzip the archive.
   2. Install the required dependencies:
   
   pip install -r requirements.txt
   
   * Execution: Run the primary script to connect to the simulator:

python main.py


------------------------------
## 💻 System Requirements
The simulator environment has been successfully tested on Windows 11 with a GeForce RTX 3070. For stable performance, your system should meet or exceed the following hardware specifications:

| Requirement | Minimum Specification |
|---|---|
| OS | 64-bit Windows 10 / 11 |
| Processor | Intel Core i7 4770k (or AMD equivalent) |
| Memory | 8 GB RAM |
| Graphics | NVIDIA GeForce GTX 970 |
| Network | Broadband Internet connection |
| Storage | 12 GB available space |

------------------------------
## 📅 Timeline & Structure

* Virtual Qualifier Round 1: Simple, high-contrast, desaturated gate environment to test core flight logic.
* Virtual Qualifier Round 2: High-fidelity, visually complex 3D-scanned environments.
* Physical Qualifier (September 2026): Top teams advance to a live, indoor testing phase in Southern California.
* The Finals (November 2026): The premier AI Grand Prix live event in Ohio.

------------------------------
## ℹ️ Technical Specification & More Information
Can be found here:

https://www.theaigrandprix.com/previousupdates/

------------------------------
## FlightSim Dynamics Data Pipeline

Deployment and training quickstart:

```text
DEPLOYMENT_AND_TRAINING.md
```

Detailed dynamics data-chain documentation:

```text
DYNAMICS_DATA_CHAIN_README.md
```

Competition and simulator notes:

```text
COMPETITION_ANALYSIS.md
ENGINE_INFO.md
```

This section documents the current black-box dynamics data chain used to train a
first-pass neural dynamics surrogate for FlightSim.

The goal is:

```text
past body-centric states + past rate-control commands
-> MLP dynamics model
-> next-step body-centric motion target
```

The model learns the response of FlightSim under MAVLink `SET_ATTITUDE_TARGET`
rate-control commands:

```python
u_t = [
    roll_rate_cmd,
    pitch_rate_cmd,
    yaw_rate_cmd,
    thrust_cmd,
]
```

It does not use images, gates, track status, or collision as model inputs.
Collision/reset information is recorded only so that invalid samples can be
filtered out before training. The filter is reset-aware: a collision/crash
drops the collision row and the following rows only inside the current
`run_id/reset_counter` segment. If FlightSim automatically resets and
`reset_counter` changes, the post-reset rows are treated as a new clean segment
and can be used for training. The same cleaning stage also drops sustained
frozen-position suffixes, because those usually mean the aircraft is stuck on a
boundary while controls are still being sent.

### Data Flow

```text
FlightSim.exe
  -> UI auto/manual ready
  -> MAVLink connection
  -> collect_identification_data.py
  -> controller.py sends SET_ATTITUDE_TARGET
  -> mavlink_rx.py receives telemetry
  -> logger.py writes raw jsonl
  -> flightsim_wrapper.py creates body-centric frames/targets
  -> build_dataset.py builds 60 Hz windows
  -> train_mlp_dynamics.py trains MLP
  -> eval_one_step.py / eval_rollout.py / analyze_coverage.py evaluate
```

### What Is Collected

Each raw jsonl row records the command that was actually sent to FlightSim and
the latest telemetry received from FlightSim:

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
        "p_ned": [x, y, z],
        "q_wxyz": [qw, qx, qy, qz],
        "v": [vx, vy, vz],
        "omega": [rollspeed, pitchspeed, yawspeed],
        "frame_id": ...,
        "child_frame_id": ...,
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

The raw logs are the direct black-box input/output record:

```text
actual sent action u_t + FlightSim telemetry x_t
```

### Where Data Lives

Dynamics-specific code is grouped under:

```text
dynamics/
  collection/
  dataset/
  training/
  evaluation/
  models/
```

The central body-centric conversion lives in:

```text
dynamics/models/flightsim_wrapper.py
```

`build_dataset.py`, one-step evaluation, rollout evaluation, and coverage
analysis should all use this wrapper so that FlightSim targets and MLP
predictions are compared in the same coordinate/state definition.

Raw logs:

```text
logs/raw/*.jsonl
```

Rejected/invalid logs:

```text
logs/rejected/*.jsonl
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

Checkpoints:

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

The four smoke-test raw logs are:

```text
logs/raw/smoke_forward_flight_20260604_225556.jsonl
logs/raw/smoke_left_turn_20260604_232857.jsonl
logs/raw/smoke_right_turn_20260604_233044.jsonl
logs/raw/smoke_climb_descend_20260604_233227.jsonl
```

### FlightSim UI Ready

The simulator must be inside the actual flight HUD before data collection.
`MAVLink telemetry` alone is not enough, because menus can also emit telemetry.

Use:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode ui-auto `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 2 `
  --restart-delay-s 2 `
  --save-screenshots
```

The `ui-auto` sequence is:

```text
PRESS ANY BUTTON -> SUBMIT -> AVAILABLE -> RACE
```

The launcher only returns ready after it detects the flight HUD and receives
telemetry. The HUD check looks for the in-race screen, including `FLIGHT MODE
ACRO` and the speed HUD.

If an attempt times out, `--attempts` makes the launcher kill FlightSim, wait
`--restart-delay-s`, restart the `.exe`, and retry the UI sequence.

If full UI automation is unstable on a machine, use manual mode:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode manual-ready `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 2 `
  --restart-delay-s 2
```

Then manually click through the UI to the flight HUD before collection.

### Collect Identification Data

Collection profiles are implemented in:

```text
dynamics/collection/collect_identification_data.py
```

Available examples:

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

The targeted coverage-fill profiles are:

```text
large roll/pitch attitude response
coupled roll + pitch + yaw + thrust commands
high-speed yaw/roll correction
low-speed/hover attitude perturbation
low-speed perturbation with telemetry-based braking
near-boundary approach/recover before collision
climb/descend while turning
high angular-rate thrust response
```

The `error_*` profiles are second-pass coverage profiles derived from
coverage/error reports. They intentionally focus on bins where either sample
count is low or the trained MLP has high one-step error:

```text
error_attitude_corner_response: gravity_body corner bins
error_pitch_brake_velocity: high-speed v_body_x vs pitch-rate bins
error_roll_angle_reversal: extreme roll angle vs roll-rate bins
error_yaw_thrust_grid: extreme yaw-rate and thrust bins
error_vertical_velocity_mix: vertical velocity mixed with pitch/thrust bins
```

Each raw log metadata row includes:

```text
profile
profile_category
coverage_goal
```

Use those fields to keep later cleaning and supplementation organized.

Example collection command:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\collect_identification_data.py `
  --profile forward_flight `
  --duration-s 8 `
  --heartbeat-timeout-s 45 `
  --telemetry-timeout-s 45 `
  --log-dir logs\raw `
  --run-id smoke_forward_flight_test
```

Before opening the raw jsonl for normal samples, the collector now performs a
short flight-response probe. It sends a small rate/thrust command and requires
measurable motion. If the simulator is still in a menu, not accepting
`SET_ATTITUDE_TARGET`, or otherwise not truly flyable, collection fails before
writing training data.

For the special low-speed/hover perturbation profile, you may use
`--skip-flight-response-probe` after UI automation is known to be working. This
avoids giving the aircraft an initial speed before logging starts.

After collecting, validate that the aircraft actually moved:

```powershell
.\.venv\Scripts\python.exe dynamics\collection\validate_dynamics_log.py `
  logs\raw\smoke_forward_flight_test.jsonl `
  --json
```

Do not train on logs that fail motion validation.

### Catalog Raw Logs

Create a categorized manifest without copying the raw logs:

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

Use these manifests when building focused datasets, for example only
`high_error_*` runs, only low-speed runs, or only validated baseline runs.

### One-Command Smoke Pipeline

The end-to-end orchestrator is:

```text
dynamics/pipeline/run_dynamics_pipeline.py
```

It runs:

```text
FlightSim UI ready
-> collect profiles
-> validate moving logs
-> reject invalid logs
-> build_dataset.py
-> train_mlp_dynamics.py
-> raw-log one-step eval
-> raw-log rollout eval
-> analyze_coverage.py
```

Collection still writes raw MAVLink jsonl only. The wrapper is used
automatically from dataset/evaluation/coverage onward.

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

For longer identification runs, increase `--duration-s`, add the step/sine/random
profiles, use `--relaunch-per-profile` so each profile starts from a fresh
simulator state, and use larger training hyperparameters.

### Build Dataset

Dataset construction is implemented in:

```text
dynamics/dataset/build_dataset.py
dynamics/models/flightsim_wrapper.py
```

It converts raw jsonl into supervised training samples:

```text
(x_{t-K:t}^B, u_{t-K:t}) -> y_t
```

Default settings:

```text
frequency: 60 Hz
history K: 10
action_dim: 4
state_dim: 16 with actuator output
input_dim: 11 * (16 + 4) = 220
output_dim: 12
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

Important dataset builder options:

```text
--input        one or more raw jsonl files
--output-dir   output dataset directory
--k            history length
--hz           resampling frequency
--max-rate     action normalization scale for rate commands
--no-actuator  use 12-dim state instead of 16-dim state
--stuck-position-epsilon-m  position-change threshold for stuck detection
--stuck-min-duration-s      sustained frozen-position duration before dropping
```

The dataset builder:

```text
1. reads raw jsonl
2. removes rows missing action/telemetry
3. splits by run_id/reset_counter
4. drops collision suffixes inside each reset segment only
5. drops sustained frozen-position suffixes inside each reset segment only
6. resamples to fixed frequency
7. calls FlightSimBodyCentricWrapper
8. normalizes quaternions
9. converts NED velocity to body velocity
10. computes gravity_body
11. computes delta_p_body and delta_rotvec_body
12. builds history windows
13. saves train/val/test npz and mean/std stats
```

### Training Input

Each time step state is:

```python
x_t = [
    v_body_x, v_body_y, v_body_z,
    omega_body_x, omega_body_y, omega_body_z,
    gravity_body_x, gravity_body_y, gravity_body_z,
    imu_acc_x, imu_acc_y, imu_acc_z,
    actuator_0, actuator_1, actuator_2, actuator_3,
]
```

Each action is:

```python
u_t = [
    roll_rate_cmd,
    pitch_rate_cmd,
    yaw_rate_cmd,
    thrust_cmd,
]
```

With `K=10`, the MLP input is:

```text
[x_{t-10}, u_{t-10}, ..., x_t, u_t]
```

### Training Output

The target is 12-dimensional:

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

The model predicts relative body-centric motion, not absolute NED position.

### Train MLP With CUDA

Training is implemented in:

```text
dynamics/training/train_mlp_dynamics.py
dynamics/models/mlp_dynamics.py
```

CUDA smoke-test command:

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

Fuller first-pass training can use larger settings:

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

### MLP Hyperparameters

MLP/training hyperparameters are changed from the command line:

```text
--hidden-dim      hidden layer width
--num-layers      number of MLP linear layers
--batch-size      batch size
--epochs          max training epochs
--lr              learning rate
--weight-decay    AdamW weight decay
--patience        early stopping patience
--device          cuda or cpu
```

Dataset/model-shape hyperparameters are changed during dataset build:

```text
--k            history length
--hz           training sample frequency
--max-rate     rate-command normalization scale
--no-actuator  remove actuator output from state
```

If `--k` or `--no-actuator` changes, rebuild the dataset before training,
because the MLP input dimension changes.

### Evaluate

One-step evaluation:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_one_step.py `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --data logs\processed\smoke_4profiles_cuda\test.npz `
  --device cuda
```

Direct raw-log one-step evaluation through the same wrapper:

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

Rollout evaluation:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\<run_name>\best_val_model.pt `
  --data logs\processed\<dataset_name>\test.npz `
  --device cuda
```

Direct raw-log rollout through `FlightSimBodyCentricWrapper` and
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

Plot rollout:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\plot_rollout.py `
  --rollout-json <rollout_result.json> `
  --output <plot_path.png>
```

Coverage analysis:

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\analyze_coverage.py `
  --input logs\raw\smoke_forward_flight_20260604_225556.jsonl `
          logs\raw\smoke_left_turn_20260604_232857.jsonl `
          logs\raw\smoke_right_turn_20260604_233044.jsonl `
          logs\raw\smoke_climb_descend_20260604_233227.jsonl `
  --output-dir logs\coverage\smoke_4profiles_cuda `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --device cuda `
  --k 10 `
  --min-samples-per-bin 20
```

Coverage outputs:

```text
logs/coverage/<name>/coverage_report.json
logs/coverage/<name>/undercovered_bins.json
```

Coverage is analyzed on low-dimensional body-centric views such as
`gravity_body_x vs gravity_body_y`, `v_body_x vs v_body_z`,
`omega_body_x vs omega_body_y`, command pairs, and attitude-command pairs.
If a checkpoint is supplied, the same bins also include one-step MLP error.

### Current Smoke-Test Result

The current smoke dataset summary is:

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

The CUDA smoke training command completed with:

```text
epoch=001 train_loss=0.806750 val_loss=0.535139
epoch=002 train_loss=0.446449 val_loss=0.343431
```

This is only a smoke test. The dataset is too small for model quality claims.
For real training, collect longer runs and split train/val/test by run, not by
random rows.

### Current High-Error Fill Result

Latest high-error fill prefix:

```text
high_error_fill_20260605_154651
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

The one-step position-delta error is just under 0.1 m, but rollout is still not
usable as a dynamics replacement. Inspect the remaining high-error regions in:

```text
logs/coverage/full_plus_higherror_stuckclean_20260605_154651_with_baseline30_model/
```

### Current Rollout-Tuned Model

The first rollout fix is to train a closed-loop self-contained state, excluding
features that the model does not predict during rollout:

```text
state = [v_body, omega_body, gravity_body]
```

Dataset:

```text
logs/processed/full_plus_higherror_closedloop9_20260609/
```

Checkpoint:

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```

Training uses `--no-actuator --no-imu` for dataset build and
`--rollout-loss-weight 0.2 --rollout-steps 10` for training.

One-step test:

```json
{
  "delta_p_body_rmse_m": 0.09836,
  "delta_rotvec_body_rmse_rad": 0.00500,
  "v_body_next_rmse_mps": 0.342,
  "omega_body_next_rmse_radps": 0.0419
}
```

Raw-log rollout with `--rollout-stride 600` improved from:

```text
old 16-dim model: 1s/3s/5s position RMSE = 4.62 / 30.07 / 57.29 m
new rollout model: 1s/3s/5s position RMSE = 2.16 / 14.09 / 26.95 m
```

This is a substantial improvement, but still not sufficient as a high-quality
FlightSim replacement.

### GRU Dynamics Configuration

GRU support is configured in:

```text
dynamics/models/mlp_dynamics.py
configs/train_gru.yaml
```

Train the configured GRU on the current closed-loop dataset:

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

Current GRU test result:

```text
one-step delta_p RMSE: 0.09880 m
raw rollout 1s/3s/5s position RMSE: 3.25 / 19.65 / 37.43 m
```

The configured GRU is reproducible, but the current best model is still the
closed-loop9 MLP with 10-step rollout loss.
