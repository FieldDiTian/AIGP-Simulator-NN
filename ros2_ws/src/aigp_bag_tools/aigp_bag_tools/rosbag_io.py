"""ROS message builders and rosbag2 read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aigp_bag_tools.conversions import (
    action_from_row,
    actuator4_from_row,
    collision_count,
    flu_to_frd,
    frd_to_flu,
    gravity_body_flu_from_q,
    ned_to_enu,
    q_enu_flu_to_q_ned_frd,
    q_ned_frd_to_q_enu_flu,
    q_wxyz_to_ros_xyzw,
    q_wxyz_to_rotmat,
    rel_time_ns,
    ros_xyzw_to_q_wxyz,
    row_reset_counter,
    rpy_from_rotmat,
    source_wall_time_ns,
    stamp_from_ns,
    velocity_ned_from_row,
)


TOPIC_TYPES = {
    "/aigp/raw/odom_ned": "nav_msgs/msg/Odometry",
    "/odom": "nav_msgs/msg/Odometry",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/aigp/state/pose": "geometry_msgs/msg/PoseStamped",
    "/aigp/state/twist": "geometry_msgs/msg/TwistStamped",
    "/aigp/raw/local_position_ned": "nav_msgs/msg/Odometry",
    "/imu/data": "sensor_msgs/msg/Imu",
    "/aigp/raw/imu_frd": "sensor_msgs/msg/Imu",
    "/aigp/state/attitude_rpy": "geometry_msgs/msg/Vector3Stamped",
    "/aigp/raw/attitude_rpy_frd": "geometry_msgs/msg/Vector3Stamped",
    "/aigp/state/angular_velocity": "geometry_msgs/msg/Vector3Stamped",
    "/aigp/state/linear_acceleration": "geometry_msgs/msg/Vector3Stamped",
    "/aigp/state/gravity_body": "geometry_msgs/msg/Vector3Stamped",
    "/aigp/control/attitude_target": "mavros_msgs/msg/AttitudeTarget",
    "/aigp/control/body_rates_cmd": "geometry_msgs/msg/Vector3Stamped",
    "/aigp/control/thrust_cmd": "std_msgs/msg/Float32",
    "/aigp/actuator_output": "mavros_msgs/msg/ActuatorControl",
    "/aigp/collision": "diagnostic_msgs/msg/DiagnosticArray",
    "/aigp/run/metadata": "diagnostic_msgs/msg/DiagnosticArray",
    "/aigp/run/reset_counter": "std_msgs/msg/Int32",
    "/aigp/debug/sample_info": "diagnostic_msgs/msg/DiagnosticArray",
    "/aigp/debug/raw_json": "std_msgs/msg/String",
}


DEFAULT_TOPICS = [topic for topic in TOPIC_TYPES if topic != "/aigp/debug/raw_json"]


class RosImports:
    """Lazy holder for ROS imports.

    Keeping these imports lazy lets pure conversion tests run on non-ROS
    machines, while the CLI gives a useful error only when ROS functionality is
    actually requested.
    """

    def __init__(self):
        from builtin_interfaces.msg import Time
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped, Vector3Stamped
        from mavros_msgs.msg import ActuatorControl, AttitudeTarget
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message, serialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Float32, Header, Int32, String
        from tf2_msgs.msg import TFMessage

        self.Time = Time
        self.Header = Header
        self.DiagnosticArray = DiagnosticArray
        self.DiagnosticStatus = DiagnosticStatus
        self.KeyValue = KeyValue
        self.PoseStamped = PoseStamped
        self.TransformStamped = TransformStamped
        self.TwistStamped = TwistStamped
        self.Vector3Stamped = Vector3Stamped
        self.ActuatorControl = ActuatorControl
        self.AttitudeTarget = AttitudeTarget
        self.Odometry = Odometry
        self.Imu = Imu
        self.Float32 = Float32
        self.Int32 = Int32
        self.String = String
        self.TFMessage = TFMessage
        self.ConverterOptions = ConverterOptions
        self.SequentialReader = SequentialReader
        self.SequentialWriter = SequentialWriter
        self.StorageOptions = StorageOptions
        self.TopicMetadata = TopicMetadata
        self.deserialize_message = deserialize_message
        self.serialize_message = serialize_message
        self.get_message = get_message


def import_ros() -> RosImports:
    try:
        return RosImports()
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are not available. Run this inside a sourced "
            "ROS 2 Humble environment with rosbag2_py and mavros_msgs installed."
        ) from exc


def make_header(ros: RosImports, stamp_ns: int, frame_id: str):
    header = ros.Header()
    sec, nanosec = stamp_from_ns(stamp_ns)
    header.stamp = ros.Time(sec=int(sec), nanosec=int(nanosec))
    header.frame_id = frame_id
    return header


def set_vector3(target, values):
    target.x = float(values[0])
    target.y = float(values[1])
    target.z = float(values[2])


def set_quaternion(target, q_wxyz):
    x, y, z, w = q_wxyz_to_ros_xyzw(q_wxyz)
    target.x = x
    target.y = y
    target.z = z
    target.w = w


def diagnostic_array(ros: RosImports, stamp_ns: int, frame_id: str, name: str, values: dict, message="OK"):
    array = ros.DiagnosticArray()
    array.header = make_header(ros, stamp_ns, frame_id)
    status = ros.DiagnosticStatus()
    status.level = ros.DiagnosticStatus.OK
    status.name = name
    status.message = message
    status.hardware_id = str(values.get("run_id", "aigp_flightsim"))
    for key, value in values.items():
        kv = ros.KeyValue()
        kv.key = str(key)
        if isinstance(value, (dict, list)):
            kv.value = json.dumps(value, ensure_ascii=True, sort_keys=True)
        elif value is None:
            kv.value = ""
        else:
            kv.value = str(value)
        status.values.append(kv)
    array.status.append(status)
    return array


def diagnostic_values(array_msg) -> dict:
    values = {}
    for status in getattr(array_msg, "status", []):
        for kv in getattr(status, "values", []):
            values[kv.key] = kv.value
    return values


def create_writer(output_dir: Path, storage_id="sqlite3"):
    ros = import_ros()
    writer = ros.SequentialWriter()
    writer.open(
        ros.StorageOptions(uri=str(output_dir), storage_id=storage_id),
        ros.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return ros, writer


def create_topic(writer, ros: RosImports, topic: str, msg_type: str):
    try:
        metadata = ros.TopicMetadata(
            name=topic,
            type=msg_type,
            serialization_format="cdr",
            offered_qos_profiles="",
        )
    except TypeError:
        metadata = ros.TopicMetadata(name=topic, type=msg_type, serialization_format="cdr")
    writer.create_topic(metadata)


def write_message(writer, ros: RosImports, topic: str, msg, stamp_ns: int):
    writer.write(topic, ros.serialize_message(msg), int(max(0, stamp_ns)))


def register_default_topics(writer, ros: RosImports, include_raw_json=False):
    topics = list(DEFAULT_TOPICS)
    if include_raw_json:
        topics.append("/aigp/debug/raw_json")
    for topic in topics:
        create_topic(writer, ros, topic, TOPIC_TYPES[topic])


def build_row_messages(
    ros: RosImports,
    row: dict,
    origin_wall_ns: int,
    *,
    line_index: int,
    raw_line: str | None = None,
    storage_time_mode="row",
    include_raw_json=False,
) -> list[tuple[str, object, int]]:
    row_stamp_ns = rel_time_ns(row.get("t_wall_ns"), origin_wall_ns)
    run_id = str(row.get("run_id", "unknown_run"))
    odom = row.get("odometry") or {}
    action = row.get("action") or {}
    attitude = row.get("attitude") or {}
    imu = row.get("imu") or {}
    local_position = row.get("local_position_ned") or {}
    actuator_output = row.get("actuator_output") or {}

    messages: list[tuple[str, object, int]] = []

    def topic_stamp(section=None):
        return rel_time_ns(source_wall_time_ns(row, section), origin_wall_ns)

    def storage_stamp(section=None):
        if storage_time_mode == "source":
            return topic_stamp(section)
        return row_stamp_ns

    q_ned_frd = np.asarray(odom.get("q_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    R_ned_frd = q_wxyz_to_rotmat(q_ned_frd)
    q_enu_flu = q_ned_frd_to_q_enu_flu(q_ned_frd)
    p_ned = np.asarray(odom.get("p_ned", [0.0, 0.0, 0.0]), dtype=np.float64)
    p_enu = ned_to_enu(p_ned)
    v_ned = velocity_ned_from_row(row)
    v_body_frd = R_ned_frd.T @ v_ned
    v_body_flu = frd_to_flu(v_body_frd)
    omega_frd = np.asarray(odom.get("omega", attitude_body_rates(attitude, imu)), dtype=np.float64)
    omega_flu = frd_to_flu(omega_frd)
    imu_acc_frd = np.asarray(imu.get("acc", [0.0, 0.0, 0.0]), dtype=np.float64)
    imu_gyro_frd = np.asarray(imu.get("gyro", omega_frd), dtype=np.float64)
    imu_acc_flu = frd_to_flu(imu_acc_frd)
    imu_gyro_flu = frd_to_flu(imu_gyro_frd)

    if odom:
        raw_odom = ros.Odometry()
        raw_odom.header = make_header(ros, topic_stamp("odometry"), "map_ned")
        raw_odom.child_frame_id = "base_link_frd"
        set_vector3(raw_odom.pose.pose.position, p_ned)
        set_quaternion(raw_odom.pose.pose.orientation, q_ned_frd)
        set_vector3(raw_odom.twist.twist.linear, v_body_frd)
        set_vector3(raw_odom.twist.twist.angular, omega_frd)
        messages.append(("/aigp/raw/odom_ned", raw_odom, storage_stamp("odometry")))

        odom_msg = ros.Odometry()
        odom_msg.header = make_header(ros, topic_stamp("odometry"), "odom")
        odom_msg.child_frame_id = "base_link"
        set_vector3(odom_msg.pose.pose.position, p_enu)
        set_quaternion(odom_msg.pose.pose.orientation, q_enu_flu)
        set_vector3(odom_msg.twist.twist.linear, v_body_flu)
        set_vector3(odom_msg.twist.twist.angular, omega_flu)
        messages.append(("/odom", odom_msg, storage_stamp("odometry")))

        pose = ros.PoseStamped()
        pose.header = odom_msg.header
        pose.pose = odom_msg.pose.pose
        messages.append(("/aigp/state/pose", pose, storage_stamp("odometry")))

        twist = ros.TwistStamped()
        twist.header = make_header(ros, topic_stamp("odometry"), "base_link")
        twist.twist = odom_msg.twist.twist
        messages.append(("/aigp/state/twist", twist, storage_stamp("odometry")))

        tf = ros.TransformStamped()
        tf.header = odom_msg.header
        tf.child_frame_id = odom_msg.child_frame_id
        tf.transform.translation.x = odom_msg.pose.pose.position.x
        tf.transform.translation.y = odom_msg.pose.pose.position.y
        tf.transform.translation.z = odom_msg.pose.pose.position.z
        tf.transform.rotation = odom_msg.pose.pose.orientation
        tf_msg = ros.TFMessage()
        tf_msg.transforms.append(tf)
        messages.append(("/tf", tf_msg, storage_stamp("odometry")))

        rpy = ros.Vector3Stamped()
        rpy.header = make_header(ros, topic_stamp("odometry"), "base_link")
        set_vector3(rpy.vector, rpy_from_rotmat(q_wxyz_to_rotmat(q_enu_flu)))
        messages.append(("/aigp/state/attitude_rpy", rpy, storage_stamp("odometry")))

        gravity = ros.Vector3Stamped()
        gravity.header = make_header(ros, topic_stamp("odometry"), "base_link")
        set_vector3(gravity.vector, gravity_body_flu_from_q(q_ned_frd))
        messages.append(("/aigp/state/gravity_body", gravity, storage_stamp("odometry")))

    if local_position:
        local = ros.Odometry()
        local.header = make_header(ros, topic_stamp("local_position_ned"), "map_ned")
        local.child_frame_id = "map_ned"
        set_vector3(local.pose.pose.position, local_position.get("p_ned", p_ned))
        set_vector3(local.twist.twist.linear, local_position.get("v_ned", v_ned))
        messages.append(("/aigp/raw/local_position_ned", local, storage_stamp("local_position_ned")))

    if imu:
        imu_msg = ros.Imu()
        imu_msg.header = make_header(ros, topic_stamp("imu"), "base_link")
        set_vector3(imu_msg.angular_velocity, imu_gyro_flu)
        set_vector3(imu_msg.linear_acceleration, imu_acc_flu)
        imu_msg.orientation_covariance[0] = -1.0
        imu_msg.angular_velocity_covariance[0] = -1.0
        imu_msg.linear_acceleration_covariance[0] = -1.0
        messages.append(("/imu/data", imu_msg, storage_stamp("imu")))

        raw_imu = ros.Imu()
        raw_imu.header = make_header(ros, topic_stamp("imu"), "base_link_frd")
        set_vector3(raw_imu.angular_velocity, imu_gyro_frd)
        set_vector3(raw_imu.linear_acceleration, imu_acc_frd)
        raw_imu.orientation_covariance[0] = -1.0
        raw_imu.angular_velocity_covariance[0] = -1.0
        raw_imu.linear_acceleration_covariance[0] = -1.0
        messages.append(("/aigp/raw/imu_frd", raw_imu, storage_stamp("imu")))

        acc = ros.Vector3Stamped()
        acc.header = make_header(ros, topic_stamp("imu"), "base_link")
        set_vector3(acc.vector, imu_acc_flu)
        messages.append(("/aigp/state/linear_acceleration", acc, storage_stamp("imu")))

    if attitude:
        raw_rpy = ros.Vector3Stamped()
        raw_rpy.header = make_header(ros, topic_stamp("attitude"), "base_link_frd")
        set_vector3(raw_rpy.vector, [attitude.get("roll", 0.0), attitude.get("pitch", 0.0), attitude.get("yaw", 0.0)])
        messages.append(("/aigp/raw/attitude_rpy_frd", raw_rpy, storage_stamp("attitude")))

        omega_debug = ros.Vector3Stamped()
        omega_debug.header = make_header(ros, topic_stamp("attitude"), "base_link")
        set_vector3(omega_debug.vector, frd_to_flu(attitude_body_rates(attitude, imu)))
        messages.append(("/aigp/state/angular_velocity", omega_debug, storage_stamp("attitude")))

    if action:
        action_raw, _action_norm = action_from_row(row)
        target = ros.AttitudeTarget()
        target.header = make_header(ros, topic_stamp("action"), "base_link_frd")
        target.type_mask = int(getattr(ros.AttitudeTarget, "IGNORE_ATTITUDE", 128))
        target.orientation.w = 1.0
        set_vector3(target.body_rate, action_raw[:3])
        target.thrust = float(action_raw[3])
        messages.append(("/aigp/control/attitude_target", target, storage_stamp("action")))

        body_rates = ros.Vector3Stamped()
        body_rates.header = make_header(ros, topic_stamp("action"), "base_link")
        set_vector3(body_rates.vector, frd_to_flu(action_raw[:3]))
        messages.append(("/aigp/control/body_rates_cmd", body_rates, storage_stamp("action")))

        thrust = ros.Float32()
        thrust.data = float(action_raw[3])
        messages.append(("/aigp/control/thrust_cmd", thrust, storage_stamp("action")))

    if actuator_output:
        actuator = ros.ActuatorControl()
        actuator.header = make_header(ros, topic_stamp("actuator_output"), "base_link_frd")
        actuator.group_mix = 0
        controls = [float(v) for v in actuator4_from_row(row)] + [0.0, 0.0, 0.0, 0.0]
        actuator.controls = controls[:8]
        messages.append(("/aigp/actuator_output", actuator, storage_stamp("actuator_output")))

    reset = ros.Int32()
    reset.data = row_reset_counter(row)
    messages.append(("/aigp/run/reset_counter", reset, row_stamp_ns))

    collision_values = {
        "run_id": run_id,
        "reset_counter": row_reset_counter(row),
        "count": collision_count(row),
        "last": (row.get("collision") or {}).get("last"),
    }
    messages.append(("/aigp/collision", diagnostic_array(ros, row_stamp_ns, "base_link", "aigp_collision", collision_values), row_stamp_ns))

    sample_values = build_sample_info_values(row, line_index, row_stamp_ns)
    messages.append(("/aigp/debug/sample_info", diagnostic_array(ros, row_stamp_ns, "base_link", "aigp_sample_info", sample_values), row_stamp_ns))

    if include_raw_json and raw_line is not None:
        raw_msg = ros.String()
        raw_msg.data = raw_line.rstrip("\n")
        messages.append(("/aigp/debug/raw_json", raw_msg, row_stamp_ns))

    return messages


def attitude_body_rates(attitude: dict, imu: dict | None = None):
    if attitude:
        return np.asarray(
            [
                float(attitude.get("rollspeed", 0.0)),
                float(attitude.get("pitchspeed", 0.0)),
                float(attitude.get("yawspeed", 0.0)),
            ],
            dtype=np.float64,
        )
    if imu and imu.get("gyro") is not None:
        return np.asarray(imu["gyro"], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def build_sample_info_values(row: dict, line_index: int, row_stamp_ns: int) -> dict:
    odom = row.get("odometry") or {}
    local = row.get("local_position_ned") or {}
    imu = row.get("imu") or {}
    attitude = row.get("attitude") or {}
    actuator = row.get("actuator_output") or {}
    action = row.get("action") or {}
    return {
        "line_index": line_index,
        "sample_type": row.get("sample_type"),
        "run_id": row.get("run_id"),
        "relative_t_ns": row_stamp_ns,
        "t_wall_ns": row.get("t_wall_ns"),
        "t_boot_ms": row.get("t_boot_ms"),
        "dt": row.get("dt"),
        "reset_counter": row_reset_counter(row),
        "odometry_time_wall_ns": odom.get("time_wall_ns"),
        "odometry_time_usec": odom.get("time_usec"),
        "odometry_frame_id": odom.get("frame_id"),
        "odometry_child_frame_id": odom.get("child_frame_id"),
        "local_position_time_wall_ns": local.get("time_wall_ns"),
        "local_position_time_boot_ms": local.get("time_boot_ms"),
        "imu_time_wall_ns": imu.get("time_wall_ns"),
        "imu_time_usec": imu.get("time_usec"),
        "attitude_time_wall_ns": attitude.get("time_wall_ns"),
        "attitude_time_boot_ms": attitude.get("time_boot_ms"),
        "actuator_time_wall_ns": actuator.get("time_wall_ns"),
        "actuator_time_usec": actuator.get("time_usec"),
        "action_t_send_wall_ns": action.get("t_send_wall_ns"),
        "action_time_boot_ms": action.get("time_boot_ms"),
    }


def build_metadata_message(ros: RosImports, metadata_row: dict | None, run_id: str, origin_wall_ns: int):
    metadata = (metadata_row or {}).get("metadata") or {}
    values = {
        "run_id": run_id,
        "profile": metadata.get("profile"),
        "profile_category": metadata.get("profile_category"),
        "category": metadata.get("category") or metadata.get("profile_category"),
        "coverage_goal": metadata.get("coverage_goal"),
        "first_t_wall_ns": origin_wall_ns,
        "covariance_policy": "simulator ground truth; odom covariance zero; imu covariance unknown marker -1",
        "raw_json": metadata_row,
    }
    return diagnostic_array(ros, 0, "map_ned", "aigp_run_metadata", values)


def read_bag_messages(bag_dir: Path, storage_id="sqlite3"):
    ros = import_ros()
    reader = ros.SequentialReader()
    reader.open(
        ros.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        ros.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    msg_classes = {topic: ros.get_message(type_name) for topic, type_name in topic_types.items()}
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        yield topic, ros.deserialize_message(data, msg_classes[topic]), int(timestamp), topic_types[topic]


def odom_msg_to_row_parts(odom_msg):
    p_enu = np.asarray(
        [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
        ],
        dtype=np.float64,
    )
    q_enu_flu = ros_xyzw_to_q_wxyz(
        [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ]
    )
    q_ned_frd = q_enu_flu_to_q_ned_frd(q_enu_flu)
    R_ned_frd = q_wxyz_to_rotmat(q_ned_frd)
    v_body_flu = np.asarray(
        [
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z,
        ],
        dtype=np.float64,
    )
    omega_flu = np.asarray(
        [
            odom_msg.twist.twist.angular.x,
            odom_msg.twist.twist.angular.y,
            odom_msg.twist.twist.angular.z,
        ],
        dtype=np.float64,
    )
    v_body_frd = flu_to_frd(v_body_flu)
    v_ned = R_ned_frd @ v_body_frd
    return {
        "p_ned": [float(p_enu[1]), float(p_enu[0]), float(-p_enu[2])],
        "q_wxyz": [float(v) for v in q_ned_frd],
        "v_ned": [float(v) for v in v_ned],
        "omega": [float(v) for v in flu_to_frd(omega_flu)],
    }


def stamp_to_float_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
