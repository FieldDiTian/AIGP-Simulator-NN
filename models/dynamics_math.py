import math

import numpy as np


GRAVITY_NED = np.array([0.0, 0.0, 9.81], dtype=np.float64)


def normalize_quat(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q


def quat_to_rotmat(q_wxyz):
    w, x, y, z = normalize_quat(q_wxyz)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def rotmat_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return normalize_quat([w, x, y, z])


def rotmat_log(R):
    R = np.asarray(R, dtype=np.float64)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.array([
            0.5 * (R[2, 1] - R[1, 2]),
            0.5 * (R[0, 2] - R[2, 0]),
            0.5 * (R[1, 0] - R[0, 1]),
        ], dtype=np.float64)
    scale = theta / (2.0 * math.sin(theta))
    return scale * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)


def rotvec_to_rotmat(rotvec):
    rv = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rv)
    if theta < 1e-8:
        K = skew(rv)
        return np.eye(3) + K
    axis = rv / theta
    K = skew(axis)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def skew(v):
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def build_body_state(row, use_actuator=True):
    odom = row["odometry"]
    imu = row["imu"]
    q = normalize_quat(odom["q_wxyz"])
    R_body_to_ned = quat_to_rotmat(q)
    v_ned = velocity_ned_from_row(row)
    v_body = R_body_to_ned.T @ v_ned
    gravity_body = R_body_to_ned.T @ GRAVITY_NED
    omega_body = np.asarray(odom["omega"], dtype=np.float64)
    imu_acc = np.asarray(imu["acc"], dtype=np.float64)
    parts = [v_body, omega_body, gravity_body, imu_acc]
    if use_actuator:
        actuator = actuator4_from_row(row)
        parts.append(actuator)
    return np.concatenate(parts).astype(np.float32)


def velocity_ned_from_row(row):
    local_pos = row.get("local_position_ned")
    if local_pos is not None and local_pos.get("v_ned") is not None:
        return np.asarray(local_pos["v_ned"], dtype=np.float64)
    return np.asarray(row["odometry"]["v"], dtype=np.float64)


def actuator4_from_row(row):
    actuator = row.get("actuator_output")
    if not actuator or actuator.get("actuator") is None:
        return np.zeros(4, dtype=np.float64)
    values = list(actuator["actuator"])
    values = (values + [0.0, 0.0, 0.0, 0.0])[:4]
    return np.asarray(values, dtype=np.float64)


def action_from_row(row, max_rate=1.0):
    action = row["action"]
    return np.asarray([
        float(action["roll_rate_cmd"]) / max_rate,
        float(action["pitch_rate_cmd"]) / max_rate,
        float(action["yaw_rate_cmd"]) / max_rate,
        float(action["thrust_cmd"]),
    ], dtype=np.float32)


def target_from_rows(row, next_row):
    odom = row["odometry"]
    next_odom = next_row["odometry"]
    R = quat_to_rotmat(odom["q_wxyz"])
    R_next = quat_to_rotmat(next_odom["q_wxyz"])
    p = np.asarray(odom["p_ned"], dtype=np.float64)
    p_next = np.asarray(next_odom["p_ned"], dtype=np.float64)
    delta_p_body = R.T @ (p_next - p)
    delta_rot_body = R.T @ R_next
    delta_rotvec_body = rotmat_log(delta_rot_body)
    next_v_body = R_next.T @ velocity_ned_from_row(next_row)
    next_omega_body = np.asarray(next_odom["omega"], dtype=np.float64)
    return np.concatenate([
        delta_p_body,
        delta_rotvec_body,
        next_v_body,
        next_omega_body,
    ]).astype(np.float32)
