"""Pure conversion helpers shared by the ROS bag tools.

The source simulator follows MAVLink-style local NED and aircraft FRD frames.
ROS consumers usually expect ENU and FLU.  Keep this module ROS-free so it can
be tested on machines that do not have ROS 2 installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


GRAVITY_NED = np.array([0.0, 0.0, 9.81], dtype=np.float64)
T_NED_TO_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
T_ENU_TO_NED = T_NED_TO_ENU
T_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
T_FLU_TO_FRD = T_FRD_TO_FLU


def as_vec3(value, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        padded = np.zeros(3, dtype=np.float64)
        padded[: arr.size] = arr
        return padded
    return arr[:3].astype(np.float64)


def as_vec4(value, default=(0.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    padded = np.zeros(4, dtype=np.float64)
    padded[: min(4, arr.size)] = arr[:4]
    return padded


def normalize_quat_wxyz(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q


def q_wxyz_to_rotmat(q_wxyz) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_q_wxyz(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
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
            s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif idx == 1:
            s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s
    return normalize_quat_wxyz([w, x, y, z])


def q_wxyz_to_ros_xyzw(q_wxyz):
    q = normalize_quat_wxyz(q_wxyz)
    return float(q[1]), float(q[2]), float(q[3]), float(q[0])


def ros_xyzw_to_q_wxyz(q_xyzw):
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return normalize_quat_wxyz([w, x, y, z])


def ned_to_enu(v_ned) -> np.ndarray:
    return T_NED_TO_ENU @ as_vec3(v_ned)


def enu_to_ned(v_enu) -> np.ndarray:
    return T_ENU_TO_NED @ as_vec3(v_enu)


def frd_to_flu(v_frd) -> np.ndarray:
    return T_FRD_TO_FLU @ as_vec3(v_frd)


def flu_to_frd(v_flu) -> np.ndarray:
    return T_FLU_TO_FRD @ as_vec3(v_flu)


def R_ned_frd_to_R_enu_flu(R_ned_frd) -> np.ndarray:
    return T_NED_TO_ENU @ np.asarray(R_ned_frd, dtype=np.float64).reshape(3, 3) @ T_FLU_TO_FRD


def R_enu_flu_to_R_ned_frd(R_enu_flu) -> np.ndarray:
    return T_ENU_TO_NED @ np.asarray(R_enu_flu, dtype=np.float64).reshape(3, 3) @ T_FRD_TO_FLU


def q_ned_frd_to_q_enu_flu(q_wxyz) -> np.ndarray:
    return rotmat_to_q_wxyz(R_ned_frd_to_R_enu_flu(q_wxyz_to_rotmat(q_wxyz)))


def q_enu_flu_to_q_ned_frd(q_wxyz) -> np.ndarray:
    return rotmat_to_q_wxyz(R_enu_flu_to_R_ned_frd(q_wxyz_to_rotmat(q_wxyz)))


def rpy_from_rotmat(R) -> np.ndarray:
    """Return roll, pitch, yaw from a body-to-world rotation matrix."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def velocity_ned_from_row(row) -> np.ndarray:
    local_position = row.get("local_position_ned") or {}
    if local_position.get("v_ned") is not None:
        return as_vec3(local_position["v_ned"])
    return as_vec3((row.get("odometry") or {}).get("v"))


def row_reset_counter(row) -> int:
    reset_counter = row.get("reset_counter")
    if reset_counter is None:
        reset_counter = (row.get("odometry") or {}).get("reset_counter", 0)
    return int(reset_counter or 0)


def collision_count(row) -> int:
    collision = row.get("collision") or {}
    return int(collision.get("count") or collision.get("collision_count") or 0)


def actuator4_from_row(row) -> np.ndarray:
    actuator = row.get("actuator_output") or {}
    return as_vec4(actuator.get("actuator"))


def action_from_row(row, max_rate=1.0) -> tuple[np.ndarray, np.ndarray]:
    action = row.get("action") or {}
    raw = np.asarray(
        [
            float(action.get("roll_rate_cmd", 0.0)),
            float(action.get("pitch_rate_cmd", 0.0)),
            float(action.get("yaw_rate_cmd", 0.0)),
            float(action.get("thrust_cmd", 0.0)),
        ],
        dtype=np.float32,
    )
    norm = raw.copy()
    if max_rate:
        norm[:3] = norm[:3] / float(max_rate)
    return raw, norm


def gravity_body_frd_from_q(q_wxyz) -> np.ndarray:
    return q_wxyz_to_rotmat(q_wxyz).T @ GRAVITY_NED


def gravity_body_flu_from_q(q_wxyz) -> np.ndarray:
    return frd_to_flu(gravity_body_frd_from_q(q_wxyz))


def rel_time_ns(source_time_ns, origin_time_ns) -> int:
    if source_time_ns is None:
        source_time_ns = origin_time_ns
    return max(0, int(source_time_ns) - int(origin_time_ns))


def stamp_from_ns(stamp_ns: int) -> tuple[int, int]:
    stamp_ns = max(0, int(stamp_ns))
    return stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000


def source_wall_time_ns(row, section=None):
    if section:
        value = (row.get(section) or {}).get("time_wall_ns")
        if value is not None:
            return int(value)
    if section == "action":
        value = (row.get("action") or {}).get("t_send_wall_ns")
        if value is not None:
            return int(value)
    value = row.get("t_wall_ns")
    return int(value) if value is not None else None


def find_repo_root(start=None) -> Path | None:
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    for candidate in list(candidates):
        candidates.extend(candidate.parents)
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "dynamics" / "models" / "flightsim_wrapper.py").exists():
            return candidate
    return None


def expand_input_paths(items: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
            continue
        if path.suffix.lower() == ".json":
            paths.extend(_paths_from_catalog_json(path))
            continue
        paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _paths_from_catalog_json(path: Path) -> list[Path]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict) and "path" in value:
        return [Path(value["path"])]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "path" in item:
                out.append(Path(item["path"]))
            elif isinstance(item, str):
                out.append(Path(item))
        return out
    return []
