# AI-GP ROS 2 Bag Tools

This package converts `logs/raw/*.jsonl` FlightSim black-box dynamics logs into
ROS 2 Humble rosbags. JSONL remains the original truth. The bag is a replay and
simulation interface built from standard ROS messages.

## Build

Install ROS 2 Humble on Ubuntu 22.04, then install MAVROS messages:

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

Run tools from the repository root, or pass `--repo-root`.

## Convert JSONL To Rosbag

```bash
ros2 run aigp_bag_tools jsonl_to_rosbag \
  --input logs/raw/smoke_forward_flight_20260604_225556.jsonl \
  --output-root logs/rosbags \
  --overwrite
```

The default writes one bag per JSONL file at `logs/rosbags/<run_id>/`.

## Standard Topics

| Topic | Type | Frame / meaning |
|---|---|---|
| `/aigp/raw/odom_ned` | `nav_msgs/msg/Odometry` | FlightSim NED/FRD truth, `map_ned -> base_link_frd` |
| `/odom` | `nav_msgs/msg/Odometry` | ROS ENU/FLU state, `odom -> base_link` |
| `/tf` | `tf2_msgs/msg/TFMessage` | `odom -> base_link` transform |
| `/aigp/state/pose` | `geometry_msgs/msg/PoseStamped` | pose split from `/odom` |
| `/aigp/state/twist` | `geometry_msgs/msg/TwistStamped` | body-frame FLU velocity and angular velocity |
| `/aigp/raw/local_position_ned` | `nav_msgs/msg/Odometry` | raw NED local position and NED velocity for log comparison |
| `/imu/data` | `sensor_msgs/msg/Imu` | ROS FLU IMU |
| `/aigp/raw/imu_frd` | `sensor_msgs/msg/Imu` | raw FRD IMU |
| `/aigp/state/attitude_rpy` | `geometry_msgs/msg/Vector3Stamped` | ROS roll/pitch/yaw derived from `/odom` |
| `/aigp/raw/attitude_rpy_frd` | `geometry_msgs/msg/Vector3Stamped` | raw MAVLink roll/pitch/yaw |
| `/aigp/state/angular_velocity` | `geometry_msgs/msg/Vector3Stamped` | ROS FLU angular velocity |
| `/aigp/state/linear_acceleration` | `geometry_msgs/msg/Vector3Stamped` | ROS FLU linear acceleration |
| `/aigp/state/gravity_body` | `geometry_msgs/msg/Vector3Stamped` | gravity vector in ROS `base_link` FLU |
| `/aigp/control/attitude_target` | `mavros_msgs/msg/AttitudeTarget` | original SET_ATTITUDE_TARGET body rates and thrust, `base_link_frd` |
| `/aigp/control/body_rates_cmd` | `geometry_msgs/msg/Vector3Stamped` | command body rates converted to ROS FLU |
| `/aigp/control/thrust_cmd` | `std_msgs/msg/Float32` | normalized thrust command |
| `/aigp/actuator_output` | `mavros_msgs/msg/ActuatorControl` | four normalized actuator/motor outputs in `controls[0:4]` |
| `/aigp/collision` | `diagnostic_msgs/msg/DiagnosticArray` | collision count, reset segment, last collision JSON |
| `/aigp/run/metadata` | `diagnostic_msgs/msg/DiagnosticArray` | run profile/category/coverage metadata |
| `/aigp/run/reset_counter` | `std_msgs/msg/Int32` | reset segment id |
| `/aigp/debug/sample_info` | `diagnostic_msgs/msg/DiagnosticArray` | row ids, original timestamps, frame ids, dt |
| `/aigp/debug/raw_json` | `std_msgs/msg/String` | optional raw JSON line, off by default |

## Coordinate Conversion

- NED to ENU: `[x, y, z]_enu = [y, x, -z]_ned`
- FRD to FLU: `[x, y, z]_flu = [x, -y, -z]_frd`
- Quaternion conversion uses rotation matrices:
  `R_enu_flu = T_ned_to_enu * R_ned_frd * T_flu_to_frd`

Bag storage timestamps default to the JSONL row timestamp so all row snapshot
topics stay syncable. Message header stamps use each source's own timestamp
when present. If a source timestamp predates the first row, it is clamped to
zero and the original absolute time remains in `/aigp/debug/sample_info`.

## Verify And Rebuild Dataset

```bash
ros2 run aigp_bag_tools verify_bag \
  --bag logs/rosbags/smoke_forward_flight_20260604_225556 \
  --jsonl logs/raw/smoke_forward_flight_20260604_225556.jsonl

ros2 run aigp_bag_tools rosbag_to_dataset \
  --bag logs/rosbags/smoke_forward_flight_20260604_225556 \
  --output-dir logs/processed/from_bag_smoke \
  --k 10 --hz 60 --max-rate 1.0 --no-actuator --no-imu
```
