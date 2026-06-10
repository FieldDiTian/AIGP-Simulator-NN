# FlightSim 黑盒动力学数据链说明

本文档解释当前项目里的动力学数据链：

```text
采集 FlightSim 黑盒输入输出
-> 构建 body-centric 训练样本
-> 用 CUDA 训练第一版 MLP 动力学替代模型
-> 做 one-step / rollout 冒烟评估
```

当前第一版模型目标是学习：

```text
过去 K 步机体系状态 + 过去 K 步控制输入
-> 下一步机体系运动 target
```

模型形式：

```text
f_theta(x_{t-K:t}^B, u_{t-K:t}) -> y_t
```

控制输入固定为 MAVLink `SET_ATTITUDE_TARGET` 的四个字段：

```python
u_t = [
    roll_rate_cmd,
    pitch_rate_cmd,
    yaw_rate_cmd,
    thrust_cmd,
]
```

不把图像、gate、track、race status、collision 作为模型输入。collision/reset 只用于过滤污染数据。
过滤规则是 reset-aware 的：碰撞/crash 当行以及其后的数据只在当前
`run_id/reset_counter` segment 内丢弃；如果 FlightSim 自动 reset，
`reset_counter` 变化后的数据会作为新的干净 segment 继续采纳。
同一清洗阶段也会丢弃持续位置不变的后缀，因为这通常表示飞机卡在边界上，
控制仍在发送但 FlightSim 的位置已经冻结。

## 1. 总体数据流

```text
FlightSim.exe
  -> UI 自动进入飞行 HUD
  -> MAVLink heartbeat / telemetry ready
  -> collect_identification_data.py 产生控制 profile
  -> controller.py 发送 SET_ATTITUDE_TARGET
  -> mavlink_rx.py 接收 FlightSim telemetry
  -> logger.py 写 raw jsonl
  -> validate_dynamics_log.py 检查飞机是否真的运动
  -> flightsim_wrapper.py 打包成统一 body-centric 表示
  -> build_dataset.py 转成训练 dataset
  -> train_mlp_dynamics.py 用 CUDA 训练 MLP
  -> eval_one_step.py / eval_rollout.py / analyze_coverage.py 评估和查欠覆盖
```

重点：`telemetry ready` 不等于已经进入仿真。现在 launcher 会同时检查：

```text
1. 已经进入飞行 HUD
2. 收到 FlightSim telemetry
```

飞行 HUD 判据包含右上角 `FLIGHT MODE ACRO` 和底部速度 HUD。

## 2. 采集到什么数据

raw 日志是 jsonl 格式：

```text
logs/raw/*.jsonl
```

每一行是一帧控制周期附近的数据，核心字段如下：

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

其中：

```text
action 是黑盒输入
telemetry 是黑盒输出/状态观测
```

所以 raw 日志记录的是：

```text
实际发给 FlightSim 的控制 u_t
+
FlightSim 返回的状态 x_t
```

## 3. 目录位置

动力学相关代码现在集中在专门目录：

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

根目录只保留项目入口文档、原始比赛模板、日志、数据集和 checkpoint。

原始日志：

```text
logs/raw/
```

被判定无效或不应训练的数据：

```text
logs/rejected/
```

处理后的训练集：

```text
logs/processed/<dataset_name>/
  train.npz
  val.npz
  test.npz
  normalization_stats.json
  dataset_summary.json
```

模型 checkpoint：

```text
checkpoints/<run_name>/
  best_val_model.pt
  normalization_stats.json
  training_curve.csv
```

当前已经跑通的 CUDA 冒烟数据集：

```text
logs/processed/smoke_4profiles_cuda/
```

当前 CUDA 冒烟 checkpoint：

```text
checkpoints/smoke_4profiles_cuda/best_val_model.pt
```

当前四组有效冒烟 raw 日志：

```text
logs/raw/smoke_forward_flight_20260604_225556.jsonl
logs/raw/smoke_left_turn_20260604_232857.jsonl
logs/raw/smoke_right_turn_20260604_233044.jsonl
logs/raw/smoke_climb_descend_20260604_233227.jsonl
```

一份曾经没进入仿真、飞机没动的日志已经移走：

```text
logs/rejected/smoke_forward_flight_20260604_223202.not_in_flight.jsonl
```

## 4. 如何启动 FlightSim 并进入仿真

自动 UI 模式：

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode ui-auto `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 2 `
  --restart-delay-s 2 `
  --save-screenshots
```

自动点击顺序：

```text
PRESS ANY BUTTON -> SUBMIT -> AVAILABLE -> RACE
```

现在 `ui-auto` 使用：

```text
pywin32 前台化窗口
Win32 SendInput / pydirectinput / pyautogui 多路发送游戏输入
pyautogui 截图检测 HUD
```

如果一次 UI 进入超时，`--attempts` 会让脚本关闭 FlightSim、等待
`--restart-delay-s`，重新启动 `.exe` 并再次执行 UI 序列。

如果 UI 自动化不稳定，可以手动进入飞行 HUD 后运行：

```powershell
.\.venv\Scripts\python.exe dynamics\collection\launch_flightsim.py `
  --mode manual-ready `
  --ready-signal telemetry `
  --timeout-s 150 `
  --attempts 2 `
  --restart-delay-s 2
```

## 5. 如何采集数据

采集脚本：

```text
dynamics/collection/collect_identification_data.py
```

当前支持的 profile：

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

定向补覆盖 profile 对应：

```text
大 roll / 大 pitch 姿态下的控制响应
roll + pitch + yaw + thrust 同时变化的组合
高速前飞时的 yaw / roll 修正
低速 / 悬停附近的姿态扰动
带遥测速度反馈制动的低速姿态扰动
接近边界但未碰撞前的正常动力学
上升 / 下降同时转弯
高角速度下的 thrust 响应
```

`error_*` 是第二轮按覆盖/误差报告补采的 profile，不是普通动作重复采样：

```text
error_attitude_corner_response: gravity_body 角落 bin，也就是大 roll/pitch 姿态区
error_pitch_brake_velocity: 高速 v_body_x 和 pitch-rate 组合误差区
error_roll_angle_reversal: 极端 roll angle 下 roll-rate 反向控制区
error_yaw_thrust_grid: 极端 yaw-rate 和 thrust 组合区
error_vertical_velocity_mix: 垂直速度与 pitch/thrust 混合区
```

每个 raw jsonl 的 metadata 行都会写入：

```text
profile
profile_category
coverage_goal
```

后续清洗、重建 dataset、继续补采时，不要只按文件名猜用途；优先用这些字段或 catalog 里的分类清单。

采集一组前飞数据：

```powershell
.\.venv\Scripts\python.exe dynamics\collection\collect_identification_data.py `
  --profile forward_flight `
  --duration-s 8 `
  --heartbeat-timeout-s 45 `
  --telemetry-timeout-s 45 `
  --log-dir logs\raw `
  --run-id smoke_forward_flight_test
```

采集脚本默认会在正式写入 raw jsonl 前做一次飞机响应探测：发送一小段
rate/thrust 控制，并要求位置或速度发生可测变化。如果只是进了菜单、只有
heartbeat/telemetry、或者飞机没有接受 `SET_ATTITUDE_TARGET`，采集会直接失败，
不会写训练样本。

对于 `low_speed_hover_attitude_perturb` 这种专门补低速/悬停附近的数据，
在确认 UI 自动进入 HUD 已经稳定后，可以加 `--skip-flight-response-probe`。
这样正式日志不会先被响应探针赋予一段初速度。

采集后必须验证飞机真的动了：

```powershell
.\.venv\Scripts\python.exe dynamics\collection\validate_dynamics_log.py `
  logs\raw\smoke_forward_flight_test.jsonl `
  --json
```

如果返回：

```json
{
  "valid": false
}
```

则不要把这份日志加入训练。

## 5.0 raw 日志分类目录

生成 raw log 分类目录：

```powershell
.\.venv\Scripts\python.exe dynamics\dataset\catalog_raw_logs.py `
  --input logs\raw `
  --catalog-name all_runs_current
```

输出目录：

```text
logs/catalog/<catalog_name>/
  all_runs.json
  summary.json
  categories/<profile_category>.json
  profiles/<profile>.json
```

这个 catalog 只保存路径和 metadata，不复制 raw jsonl。后续可以直接按：

```text
baseline_*
targeted_*
high_error_*
single_axis_*
low_speed
```

筛选数据，便于清洗和补充。

## 5.1 一键自动化管线

现在有一个总控脚本：

```text
dynamics/pipeline/run_dynamics_pipeline.py
```

它串起来的是：

```text
FlightSim UI ready
-> collect profiles
-> validate_dynamics_log.py 检查飞机是否真的动
-> 无效日志移入 logs/rejected
-> build_dataset.py 构建 dataset
-> train_mlp_dynamics.py 用 CUDA 训练
-> eval_one_step.py 直接从 raw log 走 wrapper 对比
-> eval_rollout.py 直接从 raw log 走 wrapper 闭环对比
-> analyze_coverage.py 统计覆盖和模型误差
```

注意：采集阶段仍然只写 raw MAVLink jsonl，不在采集时调用 wrapper。这是故意的，因为 raw log 是原始真值，后续可以反复重处理。wrapper 从 dataset/evaluation/coverage 阶段自动使用。

先 dry-run 看命令链：

```powershell
.\.venv\Scripts\python.exe dynamics\pipeline\run_dynamics_pipeline.py `
  --skip-launch `
  --profiles forward_flight `
  --duration-s 1 `
  --run-prefix dryrun_test `
  --dry-run
```

跑一次 UI-auto 烟测管线：

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

正式做多 profile 覆盖采集时，建议加：

```text
--relaunch-per-profile
```

这样每个 profile 都会独立启动 FlightSim、进入飞行 HUD、采集、validate，然后关闭仿真，避免上一段飞行状态或碰撞污染下一段。

如果 FlightSim 已经手动进入飞行 HUD，可以跳过启动/UI：

```powershell
.\.venv\Scripts\python.exe dynamics\pipeline\run_dynamics_pipeline.py `
  --skip-launch `
  --profiles forward_flight left_turn right_turn climb_descend `
  --duration-s 8 `
  --run-prefix manual_ready_pipeline `
  --device cuda
```

## 6. 如何构建训练数据集

dataset builder：

```text
dynamics/dataset/build_dataset.py
dynamics/models/flightsim_wrapper.py
```

现在坐标变换和训练 target 的标准定义不直接散落在 dataset/eval 脚本里，而是集中在：

```python
FlightSimBodyCentricWrapper
```

它负责把 raw FlightSim log 包装成：

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

这样做的目的：

```text
raw MAVLink log 作为原始真值保留
训练 dataset 由 wrapper 生成
one-step 评估用 wrapper target 对比 MLP 输出
coverage 分析也用同一套 body-centric 定义
以后改状态定义，只需要先改 wrapper
```

它把 raw jsonl 转成：

```text
(x_{t-K:t}^B, u_{t-K:t}) -> y_t
```

处理步骤：

```text
1. 读取 raw jsonl
2. 删除缺 action / telemetry 的行
3. 按 run_id 和 reset_counter 切 segment
4. 在每个 reset segment 内，从 collision 当行开始丢弃后缀
5. 在每个 reset segment 内，丢弃持续位置不变的卡边界后缀
6. reset 后的新 segment 继续保留并可用于训练
7. 重采样到固定频率，默认 60 Hz
8. 调用 FlightSimBodyCentricWrapper
9. 归一化 quaternion
10. NED velocity 转 body velocity
11. 计算 gravity_body
12. 计算 delta_p_body
13. 计算 delta_rotvec_body
14. 构造 K=10 history window
15. 保存 train/val/test 和 mean/std
```

构建当前四组冒烟数据集：

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

常用参数：

```text
--input        输入 raw jsonl，可传多个
--output-dir   输出 dataset 目录
--k            history 长度，默认 10
--hz           重采样频率，默认 60
--max-rate     rate command 归一化尺度
--no-actuator  不使用 actuator_output，状态从 16 维变 12 维
--stuck-position-epsilon-m  判定位置冻结的位移阈值，默认 1e-4 m
--stuck-min-duration-s      位置持续冻结多久后丢弃后缀，默认 0.5 s
```

如果改了 `--k` 或 `--no-actuator`，必须重新 build dataset，因为 MLP 输入维度会变。

## 7. 训练输入是什么

每个时间步的状态默认是 16 维：

```python
x_t = [
    v_body_x, v_body_y, v_body_z,
    omega_body_x, omega_body_y, omega_body_z,
    gravity_body_x, gravity_body_y, gravity_body_z,
    imu_acc_x, imu_acc_y, imu_acc_z,
    actuator_0, actuator_1, actuator_2, actuator_3,
]
```

控制输入是 4 维：

```python
u_t = [
    roll_rate_cmd,
    pitch_rate_cmd,
    yaw_rate_cmd,
    thrust_cmd,
]
```

当 `K=10` 时，一个训练样本输入是 11 个时间步拼接：

```text
[x_{t-10}, u_{t-10}, ..., x_t, u_t]
```

如果使用 actuator：

```text
state_dim = 16
action_dim = 4
input_dim = 11 * (16 + 4) = 220
```

如果不用 actuator：

```text
state_dim = 12
action_dim = 4
input_dim = 11 * (12 + 4) = 176
```

## 8. 训练输出是什么

模型输出 12 维 target：

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

也就是说，模型不直接预测绝对 NED 位置和绝对四元数，而是预测：

```text
body 坐标系里的相对位移
body 坐标系里的相对旋转
下一步 body velocity
下一步 body angular velocity
```

## 9. 如何启动 CUDA 训练

训练脚本：

```text
dynamics/training/train_mlp_dynamics.py
```

模型定义：

```text
dynamics/models/mlp_dynamics.py
```

CUDA 冒烟训练命令：

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

正式一点的第一版训练可以用：

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

## 10. 如何更改 MLP 超参数

你说的 `NLP` 这里应理解为 `MLP` 动力学网络。

训练超参数通过命令行改：

```text
--hidden-dim      MLP hidden width
--num-layers      MLP 层数
--batch-size      batch size
--epochs          最大训练 epoch
--lr              learning rate
--weight-decay    AdamW weight decay
--patience        early stopping patience
--device          cuda 或 cpu
```

示例：改成 3 层、hidden 256、训练 20 epoch：

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

dataset 相关超参数通过 `build_dataset.py` 改：

```text
--k
--hz
--max-rate
--no-actuator
```

这类参数会改变训练样本本身，所以要先重新 build dataset，再重新 train。

## 11. 如何评估

one-step 评估：

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_one_step.py `
  --checkpoint checkpoints\smoke_4profiles_cuda\best_val_model.pt `
  --data logs\processed\smoke_4profiles_cuda\test.npz `
  --device cuda
```

也可以直接从 raw jsonl 走 wrapper 生成 `x/y` 后对比 MLP：

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

这条路径对应：

```text
raw FlightSim log
-> FlightSimBodyCentricWrapper
-> y_sim
-> MLP(x_history)
-> y_hat
-> error = y_hat - y_sim
```

rollout 评估：

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\eval_rollout.py `
  --checkpoint checkpoints\<run_name>\best_val_model.pt `
  --data logs\processed\<dataset_name>\test.npz `
  --device cuda
```

也可以直接从 raw jsonl 走 wrapper + `NeuralDroneDynamics` 做闭环 rollout：

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

这条路径对应：

```text
raw FlightSim log
-> wrapper 得到真实 body-centric history 和真实 action sequence
-> NeuralDroneDynamics 从真实初始状态 reset
-> MLP 闭环 step(action)
-> 和 FlightSim reference 轨迹比较位置、速度、姿态、角速度误差
```

画 rollout：

```powershell
.\.venv\Scripts\python.exe dynamics\evaluation\plot_rollout.py `
  --rollout-json <rollout_result.json> `
  --output <plot_path.png>
```

覆盖度和模型误差分析：

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

输出位置：

```text
logs/coverage/<name>/coverage_report.json
logs/coverage/<name>/undercovered_bins.json
```

第一版不在完整高维空间里硬分箱，而是看这些低维视图：

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

每个 bin 统计：

```text
sample_count
valid_sample_count
collision_filtered_count
mean_action
target_std
model_error_mean
```

欠覆盖不只看样本数，也看模型误差：

```text
valid_sample_count < min_samples_per_bin
或者该 bin 的 model_error_mean 明显偏高
```

如果不传 `--checkpoint`，脚本只做数据覆盖分析；传入 checkpoint 后，会额外把 MLP one-step error 映射到同一套 bin 上。

## 12. 当前冒烟测试结果

四组有效采集：

```text
forward_flight: 457 complete rows, displacement 30.175 m, max speed 34.101 m/s
left_turn:      459 complete rows, displacement 163.545 m, max speed 28.524 m/s
right_turn:     459 complete rows, displacement 163.265 m, max speed 28.526 m/s
climb_descend:  458 complete rows, displacement 168.602 m, max speed 28.811 m/s
```

dataset summary：

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

CUDA 冒烟训练：

```text
epoch=001 train_loss=0.806750 val_loss=0.535139
epoch=002 train_loss=0.446449 val_loss=0.343431
```

one-step 冒烟评估：

```json
{
  "samples": 463,
  "delta_p_body_rmse_m": 0.2616,
  "delta_rotvec_body_rmse_rad": 0.00997,
  "v_body_next_rmse_mps": 15.03,
  "omega_body_next_rmse_radps": 0.532
}
```

注意：这些结果只说明链路跑通，不说明模型已经好用。正式训练需要更长数据、更多姿态/速度/角速度覆盖，并且 train/val/test 要按 run 划分。

## 13. 当前高误差补采结果

本轮补采 run 前缀：

```text
high_error_fill_20260605_154651
```

新增 5 类 raw 日志：

```text
high_error_attitude
high_error_velocity_pitch
high_error_roll_angle
high_error_yaw_thrust
high_error_vertical_velocity
```

分类 catalog：

```text
logs/catalog/all_with_high_error_fill_20260605_154651/
```

全量重建后的 dataset：

```text
logs/processed/full_plus_higherror_stuckclean_20260605_154651/
```

dataset summary：

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

CUDA baseline checkpoint：

```text
checkpoints/full_plus_higherror_baseline30_20260605_154651/best_val_model.pt
```

one-step test：

```json
{
  "delta_p_body_rmse_m": 0.0988,
  "delta_rotvec_body_rmse_rad": 0.00503,
  "v_body_next_rmse_mps": 0.612,
  "omega_body_next_rmse_radps": 0.0547
}
```

这个结果说明 one-step 位置增量误差刚压到 0.1 m 以下，但 rollout 仍然不可靠：

```text
1s rollout position RMSE: 9.59 m
3s rollout position RMSE: 56.93 m
5s rollout position RMSE: 108.73 m
```

所以后续目标不是只继续堆普通飞行时长，而是优先处理：

```text
高速 + 大垂直速度
极端 roll 姿态 + 反向 roll-rate 命令
极端 yaw-rate + 低/高 thrust
大 pitch 姿态 + thrust 边界
```

这些区域可以通过 coverage 报告查看：

```text
logs/coverage/full_plus_higherror_stuckclean_20260605_154651_with_baseline30_model/
```

## 14. Rollout 调优结果

第一轮 rollout 差的主要原因不是单纯数据少，而是模型输入包含了 rollout 时无法自我更新的外生量：

```text
imu_acc
actuator_output
```

one-step 训练时这些来自真实日志，所以指标会好看；rollout 时模型只能沿用旧值，历史输入会越来越假。

当前推荐的闭环自洽 dataset 去掉了这两个外生量，只保留：

```text
x_body = [
  v_body_x, v_body_y, v_body_z,
  omega_body_x, omega_body_y, omega_body_z,
  gravity_body_x, gravity_body_y, gravity_body_z
]
```

构建命令：

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

训练时增加 multi-step rollout loss，让模型用自己的预测状态继续滚动 10 步：

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

当前推荐 checkpoint：

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```

one-step test：

```json
{
  "delta_p_body_rmse_m": 0.09836,
  "delta_rotvec_body_rmse_rad": 0.00500,
  "v_body_next_rmse_mps": 0.342,
  "omega_body_next_rmse_radps": 0.0419
}
```

用 raw logs 做正式 rollout，并用较大 stride 跨 run 采样：

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

同一口径下，旧 16 维模型 vs 新 closed-loop9 + rollout loss：

```text
old 1s/3s/5s position RMSE: 4.62 / 30.07 / 57.29 m
new 1s/3s/5s position RMSE: 2.16 / 14.09 / 26.95 m
```

这说明 rollout 已经明显改善，但仍未达到可替代 FlightSim 的水平。下一步重点应是：

```text
按 rollout 高误差片段反查 profile
补低速/中速稳定段，而不是只补高速极端段
尝试 GRU/LSTM 或显式延迟状态
用更长 multi-step loss，但需要小权重，30 步强权重已经出现过拟合/退化
```

## 15. GRU 配置

GRU 支持已经配置在：

```text
dynamics/models/mlp_dynamics.py
configs/train_gru.yaml
```

GRU 和 MLP 使用同一个 closed-loop9 dataset：

```text
logs/processed/full_plus_higherror_closedloop9_20260609/
```

训练命令：

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

当前 GRU checkpoint：

```text
checkpoints/full_plus_higherror_closedloop9_gru_rollout10_20260609/best_val_model.pt
```

实测结果：

```text
one-step delta_p RMSE: 0.09880 m
raw rollout 1s/3s/5s position RMSE: 3.25 / 19.65 / 37.43 m
```

结论：GRU 已经可复现训练和评估，但当前没有超过 closed-loop9 MLP + 10-step rollout loss。因此当前最佳仍然是：

```text
checkpoints/full_plus_higherror_closedloop9_rolloutloss10_20260609/best_val_model.pt
```
