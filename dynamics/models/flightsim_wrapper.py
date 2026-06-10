from dataclasses import dataclass

import numpy as np

from dynamics.models.dynamics_math import (
    action_from_row,
    build_body_state,
    normalize_quat,
    quat_to_rotmat,
    rotmat_log,
    velocity_ned_from_row,
)


TARGET_SLICES = {
    "delta_p_body": slice(0, 3),
    "delta_rotvec_body": slice(3, 6),
    "v_body_next": slice(6, 9),
    "omega_body_next": slice(9, 12),
}


@dataclass(frozen=True)
class BodyCentricFrame:
    t_s: float
    t_wall_ns: int
    run_id: str
    reset_counter: int
    p_ned: np.ndarray
    q_wxyz: np.ndarray
    v_ned: np.ndarray
    omega_body: np.ndarray
    x_body: np.ndarray
    u: np.ndarray
    u_raw: np.ndarray
    collision_count: int
    row: dict

    def to_dict(self):
        return {
            "t_s": float(self.t_s),
            "t_wall_ns": int(self.t_wall_ns),
            "run_id": self.run_id,
            "reset_counter": int(self.reset_counter),
            "p_ned": self.p_ned.tolist(),
            "q_wxyz": self.q_wxyz.tolist(),
            "v_ned": self.v_ned.tolist(),
            "omega_body": self.omega_body.tolist(),
            "x_body": self.x_body.tolist(),
            "u": self.u.tolist(),
            "u_raw": self.u_raw.tolist(),
            "collision_count": int(self.collision_count),
        }


@dataclass(frozen=True)
class BodyCentricTarget:
    y: np.ndarray
    frame_t: BodyCentricFrame
    frame_next: BodyCentricFrame

    def to_dict(self):
        return {
            "y": self.y.tolist(),
            "t_s": float(self.frame_t.t_s),
            "t_next_s": float(self.frame_next.t_s),
        }


class FlightSimBodyCentricWrapper:
    def __init__(self, max_rate=1.0, use_actuator=True, use_imu=True):
        self.max_rate = float(max_rate)
        self.use_actuator = bool(use_actuator)
        self.use_imu = bool(use_imu)
        self.state_dim = 9 + (3 if self.use_imu else 0) + (4 if self.use_actuator else 0)
        self.action_dim = 4
        self.output_dim = 12

    def required_fields_present(self, row):
        return required_fields_present(row)

    def collision_count(self, row):
        return collision_count(row)

    def frame_from_log_row(self, row):
        odom = row["odometry"]
        action = row["action"]
        t_wall_ns = int(row["t_wall_ns"])
        t_s = float(row.get("_resampled_time_s", t_wall_ns / 1e9))
        q_wxyz = normalize_quat(odom["q_wxyz"])
        v_ned = velocity_ned_from_row(row)
        omega_body = np.asarray(odom["omega"], dtype=np.float64)
        x_body = build_body_state(
            row,
            use_actuator=self.use_actuator,
            use_imu=self.use_imu,
        )
        u = action_from_row(row, max_rate=self.max_rate)
        u_raw = np.asarray([
            float(action["roll_rate_cmd"]),
            float(action["pitch_rate_cmd"]),
            float(action["yaw_rate_cmd"]),
            float(action["thrust_cmd"]),
        ], dtype=np.float32)
        reset_counter = row.get("reset_counter")
        if reset_counter is None:
            reset_counter = odom.get("reset_counter", 0)
        return BodyCentricFrame(
            t_s=t_s,
            t_wall_ns=t_wall_ns,
            run_id=str(row.get("run_id", "unknown_run")),
            reset_counter=int(reset_counter or 0),
            p_ned=np.asarray(odom["p_ned"], dtype=np.float64),
            q_wxyz=q_wxyz,
            v_ned=np.asarray(v_ned, dtype=np.float64),
            omega_body=omega_body,
            x_body=x_body,
            u=u,
            u_raw=u_raw,
            collision_count=collision_count(row),
            row=row,
        )

    def target_from_frames(self, frame_t, frame_next):
        R = quat_to_rotmat(frame_t.q_wxyz)
        R_next = quat_to_rotmat(frame_next.q_wxyz)
        delta_p_body = R.T @ (frame_next.p_ned - frame_t.p_ned)
        delta_rot_body = R.T @ R_next
        delta_rotvec_body = rotmat_log(delta_rot_body)
        next_v_body = R_next.T @ frame_next.v_ned
        y = np.concatenate([
            delta_p_body,
            delta_rotvec_body,
            next_v_body,
            frame_next.omega_body,
        ]).astype(np.float32)
        return BodyCentricTarget(y=y, frame_t=frame_t, frame_next=frame_next)

    def target_from_log_rows(self, row, next_row):
        return self.target_from_frames(
            self.frame_from_log_row(row),
            self.frame_from_log_row(next_row),
        )

    def model_input_from_history(self, frames):
        return np.concatenate([
            np.concatenate([frame.x_body, frame.u]).astype(np.float32)
            for frame in frames
        ]).astype(np.float32)

    def compare_prediction(self, pred_y, target_y):
        pred = np.asarray(pred_y, dtype=np.float64)
        target = np.asarray(target_y, dtype=np.float64)
        error = pred - target
        return {
            "delta_p_body_error_m": float(np.linalg.norm(error[TARGET_SLICES["delta_p_body"]])),
            "delta_rotvec_body_error_rad": float(np.linalg.norm(error[TARGET_SLICES["delta_rotvec_body"]])),
            "v_body_next_error_mps": float(np.linalg.norm(error[TARGET_SLICES["v_body_next"]])),
            "omega_body_next_error_radps": float(np.linalg.norm(error[TARGET_SLICES["omega_body_next"]])),
            "target_error_norm": float(np.linalg.norm(error)),
        }

    @staticmethod
    def prediction_metrics(pred_y, target_y):
        pred = np.asarray(pred_y, dtype=np.float64)
        target = np.asarray(target_y, dtype=np.float64)
        return {
            "samples": int(len(pred)),
            "delta_p_body_rmse_m": rmse(pred[:, TARGET_SLICES["delta_p_body"]], target[:, TARGET_SLICES["delta_p_body"]]),
            "delta_rotvec_body_rmse_rad": rmse(pred[:, TARGET_SLICES["delta_rotvec_body"]], target[:, TARGET_SLICES["delta_rotvec_body"]]),
            "v_body_next_rmse_mps": rmse(pred[:, TARGET_SLICES["v_body_next"]], target[:, TARGET_SLICES["v_body_next"]]),
            "omega_body_next_rmse_radps": rmse(pred[:, TARGET_SLICES["omega_body_next"]], target[:, TARGET_SLICES["omega_body_next"]]),
        }


def required_fields_present(row):
    return (
        row.get("action") is not None
        and row.get("odometry") is not None
        and row.get("imu") is not None
        and row.get("attitude") is not None
        and row["odometry"].get("p_ned") is not None
        and row["odometry"].get("q_wxyz") is not None
        and row["odometry"].get("omega") is not None
        and (
            row.get("local_position_ned") is not None
            or row["odometry"].get("v") is not None
        )
    )


def collision_count(row):
    collision = row.get("collision") or {}
    return int(collision.get("count") or collision.get("collision_count") or 0)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
