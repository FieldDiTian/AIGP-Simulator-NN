# ROS 2 Humble Rosbag Migration

This project keeps `logs/raw/*.jsonl` as the source of truth and adds a ROS 2
Humble bag layer for replay, debugging, simulator integration, and dataset
reconstruction.

The implementation lives in:

```text
ros2_ws/src/aigp_bag_tools/
```

## Environment

Use Ubuntu 22.04 with ROS 2 Humble. Windows remains useful for collecting data
from the closed FlightSim executable, but the bag tools expect a sourced ROS
environment:

```bash
sudo apt update
sudo apt install \
  ros-humble-rosbag2-py \
  ros-humble-tf2-msgs \
  ros-humble-mavros-msgs \
  ros-humble-diagnostic-msgs

cd ros2_ws
colcon build --packages-select aigp_bag_tools
source install/setup.bash
```

## Convert

Run from the repository root:

```bash
ros2 run aigp_bag_tools jsonl_to_rosbag \
  --input logs/raw/smoke_forward_flight_20260604_225556.jsonl \
  --output-root logs/rosbags \
  --overwrite
```

Directories and catalog JSON files are also accepted:

```bash
ros2 run aigp_bag_tools jsonl_to_rosbag \
  --input logs/catalog/all_with_att_speed_motor_sweep_20260609_232058/all_runs.json \
  --output-root logs/rosbags
```

By default, bag storage timestamps are the JSONL control-row timestamps so each
row remains a synchronized snapshot. Individual message `header.stamp` values
use the original source timestamp when the JSON section has `time_wall_ns`.

## Coordinate Rules

FlightSim/MAVLink source data is NED world + FRD body:

```text
NED: north, east, down
FRD: forward, right, down
```

ROS-facing topics use ENU world + FLU body:

```text
ENU: east, north, up
FLU: forward, left, up
```

Conversions:

```text
[x, y, z]_enu = [y, x, -z]_ned
[x, y, z]_flu = [x, -y, -z]_frd
R_enu_flu = T_ned_to_enu * R_ned_frd * T_flu_to_frd
```

## Topic Layout

| Topic | Type | Data |
|---|---|---|
| `/aigp/raw/odom_ned` | `nav_msgs/msg/Odometry` | raw `odometry.p_ned`, `q_wxyz`, body FRD velocity and angular velocity |
| `/odom` | `nav_msgs/msg/Odometry` | ENU pose, FLU body velocity, FLU body angular velocity |
| `/tf` | `tf2_msgs/msg/TFMessage` | `odom -> base_link` |
| `/aigp/state/pose` | `geometry_msgs/msg/PoseStamped` | split pose from `/odom` |
| `/aigp/state/twist` | `geometry_msgs/msg/TwistStamped` | split twist from `/odom` |
| `/aigp/raw/local_position_ned` | `nav_msgs/msg/Odometry` | raw `local_position_ned.p_ned` and `v_ned` |
| `/imu/data` | `sensor_msgs/msg/Imu` | FLU `imu.gyro` and `imu.acc` |
| `/aigp/raw/imu_frd` | `sensor_msgs/msg/Imu` | raw FRD IMU |
| `/aigp/state/attitude_rpy` | `geometry_msgs/msg/Vector3Stamped` | RPY derived from converted ROS quaternion |
| `/aigp/raw/attitude_rpy_frd` | `geometry_msgs/msg/Vector3Stamped` | raw `attitude.roll/pitch/yaw` |
| `/aigp/state/angular_velocity` | `geometry_msgs/msg/Vector3Stamped` | FLU angular velocity |
| `/aigp/state/linear_acceleration` | `geometry_msgs/msg/Vector3Stamped` | FLU acceleration |
| `/aigp/state/gravity_body` | `geometry_msgs/msg/Vector3Stamped` | gravity in `base_link` FLU |
| `/aigp/control/attitude_target` | `mavros_msgs/msg/AttitudeTarget` | original body-rate/thrust SET_ATTITUDE_TARGET, frame `base_link_frd` |
| `/aigp/control/body_rates_cmd` | `geometry_msgs/msg/Vector3Stamped` | control body rates converted to FLU |
| `/aigp/control/thrust_cmd` | `std_msgs/msg/Float32` | normalized thrust command |
| `/aigp/actuator_output` | `mavros_msgs/msg/ActuatorControl` | `actuator_output.actuator[0:4]` in `controls[0:4]` |
| `/aigp/collision` | `diagnostic_msgs/msg/DiagnosticArray` | collision count, reset counter, last collision JSON |
| `/aigp/run/metadata` | `diagnostic_msgs/msg/DiagnosticArray` | run profile, category, coverage goal, covariance policy |
| `/aigp/run/reset_counter` | `std_msgs/msg/Int32` | reset segment id |
| `/aigp/debug/sample_info` | `diagnostic_msgs/msg/DiagnosticArray` | row index, original timestamps, frame ids, dt |
| `/aigp/debug/raw_json` | `std_msgs/msg/String` | optional raw JSON line with `--include-raw-json` |

## Field Mapping

- `odometry.p_ned` -> `/aigp/raw/odom_ned.pose.pose.position` and converted `/odom.pose.pose.position`
- `odometry.q_wxyz` -> raw `/aigp/raw/odom_ned.pose.pose.orientation` and converted `/odom.pose.pose.orientation`
- `local_position_ned.v_ned` when present, otherwise `odometry.v` -> body velocity in `/odom.twist.twist.linear`
- `odometry.omega` -> `/odom.twist.twist.angular`
- `imu.gyro` and `imu.acc` -> `/imu/data`
- `attitude.roll/pitch/yaw` -> `/aigp/raw/attitude_rpy_frd`
- `action.roll_rate_cmd/pitch_rate_cmd/yaw_rate_cmd/thrust_cmd` -> `/aigp/control/attitude_target`
- `actuator_output.actuator[0:4]` -> `/aigp/actuator_output.controls[0:4]`
- `collision`, `metadata`, `dt`, `t_boot_ms`, `time_usec`, and `reset_counter` -> diagnostic/std_msgs topics

## Verify

```bash
ros2 run aigp_bag_tools verify_bag \
  --bag logs/rosbags/smoke_forward_flight_20260604_225556 \
  --jsonl logs/raw/smoke_forward_flight_20260604_225556.jsonl
```

## Rebuild Dataset From Bag

```bash
ros2 run aigp_bag_tools rosbag_to_dataset \
  --bag logs/rosbags/smoke_forward_flight_20260604_225556 \
  --output-dir logs/processed/from_bag_smoke \
  --k 10 --hz 60 --max-rate 1.0 --no-actuator --no-imu \
  --compare-jsonl logs/raw/smoke_forward_flight_20260604_225556.jsonl
```

The rebuild path reconstructs the current wrapper-compatible row format from:

```text
/odom
/imu/data
/aigp/control/attitude_target
/aigp/actuator_output
/aigp/debug/sample_info
```

Then it reuses the existing dataset builder logic for segmentation, collision
filtering, stuck-position filtering, resampling, and train/val/test splitting.
